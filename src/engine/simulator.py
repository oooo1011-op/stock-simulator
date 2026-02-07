"""
模拟盘引擎
实时调仓、持仓管理、信号生成
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import SimulatorConfig
from src.models.alpha_factors import AlphaCalculator
from src.database.postgres import save_simulator_portfolio, load_latest_portfolio
from src.database.redis_cache import get_cache
import loguru
logger = loguru.logger


@dataclass
class Position:
    """持仓"""
    stock_code: str
    shares: int
    avg_price: float
    position_value: float
    signal: float
    weight: float


@dataclass
class Trade:
    """交易"""
    date: str
    stock_code: str
    action: str  # 'buy' or 'sell'
    shares: int
    price: float
    amount: float
    fee: float


class SimulatorEngine:
    """模拟盘引擎"""
    
    def __init__(
        self,
        initial_capital: float = None,
        fee_rate: float = None,
        slippage: float = None,
        num_positions: int = None,
    ):
        """初始化"""
        self.initial_capital = initial_capital or SimulatorConfig.INITIAL_CAPITAL
        self.fee_rate = fee_rate or SimulatorConfig.TRANSACTION_FEE
        self.slippage = slippage or SimulatorConfig.SLIPPAGE
        self.num_positions = num_positions or SimulatorConfig.MIN_POSITIONS
        
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[Dict] = []
        
        self.alpha_calculator = AlphaCalculator(use_cache=True)
        self.cache = get_cache()
        
        logger.info(f"SimulatorEngine initialized: capital={self.initial_capital}, "
                   f"positions={self.num_positions}")
    
    def generate_signals(
        self, 
        data: pd.DataFrame, 
        factor_name: str,
        date: str = None
    ) -> pd.Series:
        """
        生成交易信号
        
        Args:
            data: 价格数据
            factor_name: 因子名称
            date: 特定日期
        
        Returns:
            当日信号Series
        """
        # 计算因子值
        alpha_values = self.alpha_calculator.calculate_single_alpha(data, factor_name)
        
        if date:
            try:
                return alpha_values.loc[date]
            except KeyError:
                return pd.Series(dtype=float)
        
        return alpha_values
    
    def select_stocks(
        self, 
        signals: pd.Series, 
        top_n: int = None
    ) -> List[Tuple[str, float]]:
        """
        选股
        
        Args:
            signals: 因子信号
            top_n: 选取数量
        
        Returns:
            [(stock_code, signal), ...]
        """
        top_n = top_n or self.num_positions
        
        # 剔除停牌/无数据的股票
        valid_signals = signals.dropna()
        
        # 选择信号最强的
        top = valid_signals.nlargest(top_n)
        
        return [(str(idx), float(val)) for idx, val in top.items()]
    
    def calculate_weights(
        self, 
        selected: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """
        计算权重（等权）
        
        Args:
            selected: 选中的股票
        
        Returns:
            {stock_code: weight}
        """
        n = len(selected)
        if n == 0:
            return {}
        
        weight = 1.0 / n
        return {code: weight for code, _ in selected}
    
    def rebalance(
        self,
        date: str,
        selected: List[Tuple[str, float]],
        prices: Dict[str, float],
    ) -> List[Trade]:
        """
        调仓
        
        Args:
            date: 日期
            selected: 选中的股票列表
            prices: 当日收盘价
        
        Returns:
            交易列表
        """
        trades = []
        weights = self.calculate_weights(selected)
        total_value = self.cash + sum(
            pos.position_value for pos in self.positions.values()
        )
        
        # 目标持仓
        target_shares = {}
        for code, weight in weights.items():
            target_value = total_value * weight
            if code in prices:
                target_shares[code] = int(target_value / prices[code])
        
        # 卖出不在目标列表的股票
        for code, pos in list(self.positions.items()):
            if code not in target_shares:
                if pos.shares > 0:
                    sell_price = prices.get(code, pos.avg_price) * (1 - self.slippage)
                    sell_amount = pos.shares * sell_price
                    fee = sell_amount * self.fee_rate
                    
                    self.cash += sell_amount - fee
                    self.trades.append(Trade(
                        date=date, stock_code=code, action='sell',
                        shares=pos.shares, price=sell_price, amount=sell_amount, fee=fee
                    ))
                    del self.positions[code]
        
        # 买入/增持目标股票
        for code, target in target_shares.items():
            current = self.positions.get(code)
            current_shares = current.shares if current else 0
            diff = target - current_shares
            
            if diff > 0:
                buy_price = prices.get(code, total_value / len(weights) / target) * (1 + self.slippage)
                buy_amount = diff * buy_price
                fee = buy_amount * self.fee_rate
                
                if self.cash >= buy_amount + fee:
                    self.cash -= buy_amount + fee
                    avg_price = (current_shares * current.avg_price + buy_amount) / (current_shares + diff) if current_shares > 0 else buy_price / diff
                    
                    self.positions[code] = Position(
                        stock_code=code,
                        shares=diff,
                        avg_price=avg_price,
                        position_value=buy_amount,
                        signal=next((s for c, s in selected if c == code), 0),
                        weight=weights[code]
                    )
                    
                    self.trades.append(Trade(
                        date=date, stock_code=code, action='buy',
                        shares=diff, price=buy_price, amount=buy_amount, fee=fee
                    ))
        
        logger.info(f"Rebalance {date}: {len(trades)} trades")
        return trades
    
    def run_daily(
        self,
        date: str,
        data: pd.DataFrame,
        factor_name: str = 'alpha1',
    ) -> Dict[str, Any]:
        """
        每日运行
        
        Args:
            date: 日期
            data: 价格数据
            factor_name: 因子名称
        
        Returns:
            当日结果
        """
        # 更新持仓市值
        for pos in self.positions.values():
            if pos.stock_code in data.index.get_level_values('symbol'):
                try:
                    price = data.loc[(date, pos.stock_code), 'close']
                    pos.position_value = pos.shares * price
                except:
                    pass
        
        # 生成信号
        signals = self.generate_signals(data, factor_name, date)
        
        # 选股
        selected = self.select_stocks(signals)
        
        # 获取价格
        prices = {}
        for code in selected:
            try:
                prices[code[0]] = data.loc[(date, code[0]), 'close']
            except:
                pass
        
        # 调仓
        trades = self.rebalance(date, selected, prices)
        
        # 计算当日净值
        total_value = self.cash + sum(p.position_value for p in self.positions.values())
        
        result = {
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'positions_count': len(self.positions),
            'trades': len(trades),
            'selected': selected[:5],  # 前5只
        }
        
        self.portfolio_values.append(result)
        
        return result
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """获取组合摘要"""
        if not self.portfolio_values:
            return {}
        
        latest = self.portfolio_values[-1]
        peak = max(v['total_value'] for v in self.portfolio_values)
        drawdown = (peak - latest['total_value']) / peak if peak > 0 else 0
        
        return {
            'date': latest['date'],
            'total_value': latest['total_value'],
            'cash': latest['cash'],
            'positions': len(self.positions),
            'total_return': (latest['total_value'] / self.initial_capital - 1) if self.initial_capital > 0 else 0,
            'drawdown': drawdown,
            'trades_count': len(self.trades),
        }
    
    def save_portfolio(self, date: str):
        """保存当日持仓"""
        portfolio = []
        for pos in self.positions.values():
            portfolio.append({
                'stock_code': pos.stock_code,
                'shares': pos.shares,
                'avg_price': pos.avg_price,
                'position_value': pos.position_value,
                'signal': pos.signal,
            })
        
        save_simulator_portfolio(date, portfolio)
        logger.info(f"Saved portfolio for {date}")
    
    def get_position_details(self) -> List[Dict]:
        """获取持仓详情"""
        details = []
        for pos in self.positions.values():
            details.append({
                'stock_code': pos.stock_code,
                'shares': pos.shares,
                'avg_price': pos.avg_price,
                'position_value': pos.position_value,
                'weight': pos.weight,
                'signal': pos.signal,
            })
        return sorted(details, key=lambda x: x['position_value'], reverse=True)


if __name__ == "__main__":
    # 示例
    engine = SimulatorEngine(initial_capital=100000)
    print("SimulatorEngine initialized")
