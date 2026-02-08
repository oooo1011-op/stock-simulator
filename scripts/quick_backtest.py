#!/usr/bin/env python3
"""
快速回测脚本 - 使用数据库中的真实数据
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from src.config import DatabaseConfig
from src.models.alpha_factors import AlphaCalculator
import loguru

logger = loguru.logger


def load_data_from_db():
    """Load data from PostgreSQL"""
    logger.info("📊 从数据库加载数据...")
    engine = create_engine(DatabaseConfig.get_postgres_uri())
    
    with engine.connect() as conn:
        df = pd.read_sql('SELECT * FROM daily_prices ORDER BY stock_code, date', conn)
    
    logger.info(f"✅ 加载完成: {len(df):,}条记录, {df['stock_code'].nunique()}只股票")
    
    # 准备数据格式
    df = df.rename(columns={'stock_code': 'symbol'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date', 'symbol'])
    
    # 计算必要字段
    df['returns'] = df.groupby('symbol')['close'].pct_change()
    # vwap = amount(千元) * 1000 / volume(手) * 100 = 元/股
    df['vwap'] = df['amount'] * 1000 / (df['volume'] * 100).replace(0, np.nan)
    df['adv20'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
    df['cap'] = df['close'] * df['volume'] * 100  # 市值近似
    
    return df


def backtest_factor(df, factor_name, factor_series, initial_capital=100000):
    """
    简单回测单个因子
    
    策略: 每月初选择因子值最高的10只股票，等权重持仓
    """
    # 重置索引以便处理
    data = df.copy()
    data['factor'] = factor_series
    data = data.reset_index()
    
    # 按月调仓
    data['year_month'] = data['date'].dt.to_period('M')
    months = data['year_month'].unique()
    
    if len(months) < 12:
        return None
    
    capital = initial_capital
    positions = {}
    equity_curve = []
    
    for month in months[12:]:  # 跳过前12个月预热期
        month_data = data[data['year_month'] == month]
        
        # 每月第一个交易日调仓
        first_day = month_data.groupby('symbol').first().reset_index()
        first_day = first_day.dropna(subset=['factor', 'close'])
        
        if len(first_day) < 10:
            continue
        
        # 选择因子值最高的10只
        selected = first_day.nlargest(10, 'factor')
        
        # 清仓
        capital += sum(positions.values())
        positions = {}
        
        # 等权重买入
        invest_per_stock = capital * 0.09  # 每只股票9%仓位，留10%现金
        
        for _, row in selected.iterrows():
            price = row['close']
            if price > 0 and invest_per_stock > 0:
                shares = int(invest_per_stock / price / 100) * 100  # 整手
                if shares > 0:
                    cost = shares * price
                    positions[row['symbol']] = cost
                    capital -= cost
        
        # 月底计算净值
        last_day = month_data.groupby('symbol').last().reset_index()
        portfolio_value = capital
        
        for symbol, cost in positions.items():
            last_price = last_day[last_day['symbol'] == symbol]['close']
            if not last_price.empty and last_price.iloc[0] > 0:
                shares = int(cost / last_price.iloc[0] / 100) * 100
                portfolio_value += shares * last_price.iloc[0]
        
        equity_curve.append({
            'date': month_data['date'].max(),
            'value': portfolio_value
        })
    
    if not equity_curve:
        return None
    
    # 计算绩效指标
    eq_df = pd.DataFrame(equity_curve)
    
    total_return = (eq_df['value'].iloc[-1] / initial_capital - 1) * 100
    years = len(eq_df) / 12
    annual_return = total_return / years if years > 0 else 0
    
    # 最大回撤
    cummax = eq_df['value'].cummax()
    drawdown = (cummax - eq_df['value']) / cummax
    max_drawdown = drawdown.max() * 100
    
    # 夏普比率（简化）
    monthly_returns = eq_df['value'].pct_change().dropna()
    if monthly_returns.std() > 0:
        sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
    else:
        sharpe = 0
    
    return {
        'factor': factor_name,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe,
        'final_value': eq_df['value'].iloc[-1]
    }


def main():
    logger.info("="*60)
    logger.info("🚀 Alpha因子快速回测")
    logger.info("="*60)
    
    # 加载数据
    df = load_data_from_db()
    
    # 计算Alpha因子
    logger.info("\n📊 计算Alpha因子...")
    calc = AlphaCalculator()
    
    # 测试几个代表性因子
    test_factors = ['alpha1', 'alpha2', 'alpha5', 'alpha10', 'alpha20']
    results = []
    
    for factor_name in test_factors:
        try:
            logger.info(f"🔄 回测 {factor_name}...")
            factor_method = getattr(calc, factor_name, None)
            
            if factor_method:
                factor_values = factor_method(df)
                result = backtest_factor(df, factor_name, factor_values)
                
                if result:
                    results.append(result)
                    logger.info(f"  ✅ 年化收益: {result['annual_return']:.2f}%, "
                              f"夏普: {result['sharpe_ratio']:.2f}, "
                              f"最大回撤: {result['max_drawdown']:.2f}%")
                else:
                    logger.info(f"  ⚠️ 回测失败")
            else:
                logger.info(f"  ⚠️ 因子方法不存在")
                
        except Exception as e:
            logger.warning(f"  ⚠️ {factor_name} 错误: {e}")
    
    # 输出汇总
    if results:
        logger.info("\n" + "="*60)
        logger.info("📊 回测结果汇总")
        logger.info("="*60)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        for _, r in results_df.iterrows():
            logger.info(f"{r['factor']:<12} 年化: {r['annual_return']:>7.2f}%  "
                       f"夏普: {r['sharpe_ratio']:>5.2f}  回撤: {r['max_drawdown']:>6.2f}%")
        
        best = results_df.iloc[0]
        logger.info(f"\n🏆 最优因子: {best['factor']}")
        logger.info(f"   年化收益: {best['annual_return']:.2f}%")
        logger.info(f"   夏普比率: {best['sharpe_ratio']:.2f}")
        logger.info(f"   最大回撤: {best['max_drawdown']:.2f}%")
    else:
        logger.warning("❌ 没有有效的回测结果")


if __name__ == '__main__':
    main()
