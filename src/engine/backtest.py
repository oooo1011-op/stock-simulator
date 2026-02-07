"""
回测引擎模块
基于Alpha因子的批量回测和筛选
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import uuid
import numpy as np
import json

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import BacktestConfig, FactorConfig
from src.models.alpha_factors import AlphaCalculator
from src.database.postgres import (
    save_backtest_result, load_backtest_results, 
    save_alpha_results, load_alpha_results
)
import loguru
logger = loguru.logger


@dataclass
class BacktestResult:
    """回测结果"""
    factor_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    daily_returns: List[float]
    portfolio_values: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'factor_name': self.factor_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'annual_return': self.annual_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'total_trades': self.total_trades,
            'daily_returns': self.daily_returns,
            'portfolio_values': self.portfolio_values,
        }



def _convert_numpy(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = None,
        fee_rate: float = None,
        slippage: float = None,
        num_positions: int = None,
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            fee_rate: 手续费率
            slippage: 滑点
            num_positions: 持仓数量
        """
        self.initial_capital = initial_capital or BacktestConfig.INITIAL_CAPITAL
        self.fee_rate = fee_rate or BacktestConfig.TRANSACTION_FEE
        self.slippage = slippage or BacktestConfig.SLIPPAGE
        self.num_positions = num_positions or BacktestConfig.MIN_POSITIONS
        
        self.alpha_calculator = AlphaCalculator(use_cache=True)
        logger.info(f"BacktestEngine initialized: capital={self.initial_capital}, "
                   f"fee={self.fee_rate}, slippage={self.slippage}, positions={self.num_positions}")
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        factor_name: str,
        start_date: str = None,
        end_date: str = None,
    ) -> BacktestResult:
        """
        运行单个因子的回测
        
        Args:
            data: 价格数据
            factor_name: 因子名称 (如 'alpha1')
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            BacktestResult
        """
        logger.info(f"Running backtest for {factor_name}")
        
        # 过滤日期
        if start_date:
            data = data[data.index.get_level_values('date') >= start_date]
        if end_date:
            data = data[data.index.get_level_values('date') <= end_date]
        
        if data.empty:
            logger.warning(f"No data for {factor_name}")
            return None
        
        # 获取日期范围
        dates = sorted(data.index.get_level_values('date').unique())
        if not dates:
            logger.warning(f"No dates for {factor_name}")
            return None
        
        start = str(dates[0])[:10]
        end = str(dates[-1])[:10]
        
        # 计算因子值
        try:
            alpha_values = self.alpha_calculator.calculate_single_alpha(data, factor_name)
        except Exception as e:
            logger.error(f"Failed to calculate {factor_name}: {e}")
            return None
        
        if alpha_values is None or alpha_values.empty:
            logger.warning(f"Empty alpha values for {factor_name}")
            return None
        
        # 逐日回测
        portfolio_value = self.initial_capital
        portfolio_values = []
        daily_returns = []
        trades = 0
        prev_positions = set()
        
        for date in dates:
            # 获取当日的因子值
            try:
                daily_alpha = alpha_values.loc[date].dropna()
            except KeyError:
                daily_alpha = pd.Series(dtype=float)
            
            if daily_alpha.empty:
                # 无因子值，保持仓位
                pass
            else:
                # 选择因子值最高的股票
                top_stocks = daily_alpha.nlargest(self.num_positions).index.tolist()
                
                # 计算换手
                if prev_positions:
                    sells = prev_positions - set(top_stocks)
                    buys = set(top_stocks) - prev_positions
                    trades += len(sells) + len(buys)
                else:
                    trades += len(top_stocks)
                
                prev_positions = set(top_stocks)
            
            # 计算当日收益率（使用市场平均收益作为代理）
            daily_return = self._calculate_daily_return(data, date)
            daily_returns.append(daily_return)
            
            # 应用手续费和滑点
            adjusted_return = daily_return - self.fee_rate - self.slippage
            
            # 更新组合价值
            portfolio_value = portfolio_value * (1 + adjusted_return)
            portfolio_values.append(portfolio_value)
        
        # 计算指标
        metrics = self._calculate_metrics(
            portfolio_values, daily_returns, start, end
        )
        
        result = BacktestResult(
            factor_name=factor_name,
            start_date=start,
            end_date=end,
            initial_capital=self.initial_capital,
            final_capital=portfolio_value,
            **metrics,
        )
        
        logger.info(f"{factor_name}: return={metrics['annual_return']:.2%}, "
                   f"sharpe={metrics['sharpe_ratio']:.2f}, "
                   f"max_dd={metrics['max_drawdown']:.2%}")
        
        return result
    
    def _calculate_daily_return(self, data: pd.DataFrame, date: str) -> float:
        """计算市场日收益率（限制范围）"""
        try:
            day_data = data.loc[date]
            if isinstance(day_data, pd.DataFrame):
                returns = day_data['close'].pct_change().dropna()
                if not returns.empty:
                    ret = returns.mean()
                    # 限制收益率范围 -10% ~ +10%
                    return max(-0.10, min(0.10, ret))
            return 0
        except (KeyError, TypeError):
            pass
        return 0
    
    def _calculate_metrics(
        self,
        portfolio_values: List[float],
        daily_returns: List[float],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """计算回测指标"""
        # 年化收益率
        n_days = len(portfolio_values)
        if n_days == 0:
            return {
                'annual_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'daily_returns': [],
                'portfolio_values': [],
            }
        
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1 if portfolio_values[0] > 0 else 0
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # 夏普比率
        daily_returns_arr = np.array(daily_returns)
        mean_return = np.mean(daily_returns_arr)
        std_return = np.std(daily_returns_arr, ddof=1)
        
        if std_return > 0:
            sharpe_ratio = np.sqrt(252) * mean_return / std_return
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        # 胜率
        wins = sum(1 for r in daily_returns if r > 0)
        win_rate = wins / len(daily_returns) if daily_returns else 0
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': 0,
            'daily_returns': daily_returns,
            'portfolio_values': portfolio_values,
        }
    
    def run_all_factors(
        self,
        data: pd.DataFrame,
        factor_names: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        parallel: bool = True,
    ) -> List[BacktestResult]:
        """
        运行所有因子的回测
        
        Args:
            data: 价格数据
            factor_names: 因子列表（默认所有）
            start_date: 开始日期
            end_date: 结束日期
            parallel: 是否并行计算
        
        Returns:
            回测结果列表
        """
        # 获取因子列表
        if factor_names is None:
            factor_names = [f'alpha{i}' for i in range(1, 48)]
            factor_names += [f'alpha{i}' for i in range(49, 58)]
            factor_names += ['alpha60', 'alpha61', 'alpha62', 'alpha71']
            factor_names += [f'alpha{i}' for i in range(83, 87)]
            factor_names += ['alpha88', 'alpha92', 'alpha95', 'alpha101']
        
        logger.info(f"Running backtest for {len(factor_names)} factors")
        
        results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.run_backtest, data, name, start_date, end_date): name
                    for name in factor_names
                }
                
                for future in futures:
                    name = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"Error in {name}: {e}")
        else:
            for name in factor_names:
                result = self.run_backtest(data, name, start_date, end_date)
                if result:
                    results.append(result)
        
        logger.info(f"Completed {len(results)} factor backtests")
        return results
    
    def filter_factors(
        self,
        results: List[BacktestResult],
        min_return: float = None,
        min_sharpe: float = None,
        max_dd: float = None,
        min_win_rate: float = None,
        top_n: int = None,
    ) -> List[BacktestResult]:
        """
        筛选表现好的因子
        
        Args:
            results: 回测结果列表
            min_return: 最小年化收益率
            min_sharpe: 最小夏普比率
            max_dd: 最大回撤
            min_win_rate: 最小胜率
            top_n: 返回前N个
        
        Returns:
            筛选后的结果列表
        """
        min_return = min_return or FactorConfig.MIN_ANNUAL_RETURN
        min_sharpe = min_sharpe or FactorConfig.MIN_SHARPE_RATIO
        max_dd = max_dd or FactorConfig.MAX_DRAWDOWN
        min_win_rate = min_win_rate or FactorConfig.MIN_WIN_RATE
        
        filtered = []
        for r in results:
            if (r.annual_return >= min_return and
                r.sharpe_ratio >= min_sharpe and
                r.max_drawdown <= max_dd and
                r.win_rate >= min_win_rate):
                filtered.append(r)
        
        # 按夏普比率排序
        filtered.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        
        if top_n:
            filtered = filtered[:top_n]
        
        logger.info(f"Filtered {len(filtered)} factors from {len(results)}")
        return filtered
    
    def save_results(
        self,
        results: List[BacktestResult],
        backtest_id: str = None,
    ) -> str:
        """
        保存回测结果到数据库
        
        Args:
            results: 回测结果列表
            backtest_id: 回测ID
        
        Returns:
            回测ID
        """
        backtest_id = backtest_id or f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 汇总指标
        metrics_df = pd.DataFrame([{
            'factor_name': r.factor_name,
            'annual_return': float(r.annual_return),
            'sharpe_ratio': float(r.sharpe_ratio),
            'max_drawdown': float(r.max_drawdown),
            'win_rate': float(r.win_rate),
            'total_trades': int(r.total_trades),
        } for r in results])
        
        best = results[0] if results else None
        
        # 序列化结果 - 确保numpy类型被转换
        factor_details = []
        for r in results:
            d = r.to_dict()
            d = _convert_numpy(d)
            factor_details.append(d)
        
        save_backtest_result(backtest_id, {
            'factor_list': [r.factor_name for r in results],
            'start_date': best.start_date if best else None,
            'end_date': best.end_date if best else None,
            'initial_capital': float(self.initial_capital),
            'final_capital': float(best.final_capital) if best else 0,
            'annual_return': float(best.annual_return) if best else 0,
            'sharpe_ratio': float(best.sharpe_ratio) if best else 0,
            'max_drawdown': float(best.max_drawdown) if best else 0,
            'win_rate': float(best.win_rate) if best else 0,
            'total_trades': int(sum(r.total_trades for r in results)),
            'results_json': {
                'metrics': _convert_numpy(metrics_df.to_dict(orient='records')),
                'factor_details': factor_details,
            }
        })
        
        logger.info(f"Saved backtest results: {backtest_id}")
        return backtest_id


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 示例用法
    from src.data.akshare_fetcher import AStockDataFetcher
    
    fetcher = AStockDataFetcher()
    engine = BacktestEngine()
    
    # 获取少量股票数据测试
    symbols = ['000001', '000002', '000003']
    data = fetcher.get_bulk_daily_data(symbols, "20240101", "20240601")
    
    # 合并数据
    all_data = []
    for symbol, df in data.items():
        all_data.append(df)
    
    if all_data:
        combined = pd.concat(all_data)
        combined = combined.set_index(['date', 'symbol'])
        
        # 运行回测
        results = engine.run_all_factors(combined, ['alpha1', 'alpha2', 'alpha3'])
        
        for r in results:
            print(f"{r.factor_name}: return={r.annual_return:.2%}, sharpe={r.sharpe_ratio:.2f}")
