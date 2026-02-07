#!/usr/bin/env python3
"""
回测脚本（改进版）
- 使用更合理的模拟数据
- 修正收益率计算
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.backtest import BacktestEngine
from src.database.postgres import init_db
from src.config import FactorConfig
import pandas as pd
import numpy as np
from datetime import datetime

def create_realistic_data():
    """创建更合理的模拟数据"""
    print("创建模拟数据（合理范围）...")
    np.random.seed(42)  # 可重复
    
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='B')
    symbols = [f'{str(i).zfill(6)}' for i in range(1, 51)]
    
    data = []
    for date in dates:
        for symbol in symbols:
            # 随机游走价格
            base_price = np.random.uniform(10, 100)
            daily_return = np.random.normal(0.0003, 0.02)  # 日均0.03%，日波动2%
            close = base_price * (1 + daily_return)
            
            data.append({
                'date': date,
                'symbol': symbol,
                'open': close * np.random.uniform(0.999, 1.001),
                'high': close * np.random.uniform(1.0, 1.015),
                'low': close * np.random.uniform(0.985, 1.0),
                'close': close,
                'volume': np.random.uniform(1000000, 50000000),
                'amount': close * np.random.uniform(1000000, 50000000) * 100,
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['date', 'symbol'])
    print(f"数据: {len(df)} 条, {len(dates)} 天 x {len(symbols)} 股票")
    return df

def main():
    print("="*50)
    print("Alpha因子回测（改进版）")
    print("="*50)
    
    # 初始化数据库
    print("\n[1/4] 初始化数据库...")
    init_db()
    
    # 准备数据
    print("\n[2/4] 准备数据...")
    data = create_realistic_data()
    
    # 因子列表
    factors = [f'alpha{i}' for i in range(1, 11)] + [f'alpha{i}' for i in [49, 50, 51, 60, 83, 84]]
    
    print(f"\n[3/4] 运行回测 ({len(factors)} 个因子)...")
    
    engine = BacktestEngine(
        initial_capital=100000,
        fee_rate=0.0005,
        slippage=0.001,
        num_positions=5,
    )
    
    results = engine.run_all_factors(data, factors, parallel=True)
    
    if not results:
        print("错误: 无回测结果")
        return
    
    # 筛选
    filtered = engine.filter_factors(
        results,
        min_return=FactorConfig.MIN_ANNUAL_RETURN,
        min_sharpe=FactorConfig.MIN_SHARPE_RATIO,
        max_dd=FactorConfig.MAX_DRAWDOWN,
        min_win_rate=FactorConfig.MIN_WIN_RATE,
    )
    
    # 保存
    backtest_id = engine.save_results(results)
    
    # 结果
    print("\n" + "="*50)
    print("回测结果")
    print("="*50)
    print(f"总因子: {len(results)}")
    print(f"符合筛选: {len(filtered)}")
    
    if filtered:
        print("\n推荐因子:")
        for r in filtered[:10]:
            print(f"  {r.factor_name}: "
                  f"年化={r.annual_return:.2%}, "
                  f"夏普={r.sharpe_ratio:.2f}, "
                  f"回撤={r.max_drawdown:.2%}")
    
    print(f"\n回测ID: {backtest_id}")

if __name__ == "__main__":
    main()
