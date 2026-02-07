#!/usr/bin/env python3
"""
回测脚本（使用模拟数据演示）
网络恢复后修改USE_REAL_DATA=True使用真实数据
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine.backtest import BacktestEngine
from src.database.postgres import init_db
from src.config import BacktestConfig, FactorConfig
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

USE_REAL_DATA = False  # 改为True使用真实数据

def create_mock_data():
    """创建模拟数据用于测试"""
    print("创建模拟数据...")
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='B')  # 工作日
    symbols = [f'{str(i).zfill(6)}' for i in range(1, 51)]  # 50只股票
    
    data = []
    for date in dates:
        for symbol in symbols:
            base_price = np.random.uniform(5, 50)
            data.append({
                'date': date,
                'symbol': symbol,
                'open': base_price * np.random.uniform(0.99, 1.01),
                'high': base_price * np.random.uniform(1.0, 1.03),
                'low': base_price * np.random.uniform(0.97, 1.0),
                'close': base_price * np.random.uniform(0.98, 1.02),
                'volume': np.random.uniform(1000000, 50000000),
                'amount': np.random.uniform(10000000, 500000000),
            })
    
    df = pd.DataFrame(data)
    df = df.set_index(['date', 'symbol'])
    print(f"模拟数据: {len(df)} 条")
    return df

def main():
    print("="*50)
    print("Alpha因子回测（演示版）")
    print("="*50)
    
    # 初始化数据库
    print("\n[1/4] 初始化数据库...")
    init_db()
    
    # 准备数据
    if USE_REAL_DATA:
        print("\n[2/4] 从数据库加载真实数据...")
        from src.database.postgres import load_daily_data
        symbols = [f'{str(i).zfill(6)}' for i in range(1, 51)]
        all_data = []
        for symbol in symbols:
            try:
                df = load_daily_data(symbol, '20240101', '20241231')
                if not df.empty:
                    all_data.append(df)
            except:
                pass
        data = pd.concat(all_data) if all_data else None
    else:
        data = create_mock_data()
    
    if data is None or data.empty:
        print("错误: 无数据")
        return
    
    # 定义因子列表
    factors = (
        [f'alpha{i}' for i in range(1, 21)] +  # 先用前20个因子测试
        [f'alpha{i}' for i in range(49, 58)][:3] +
        ['alpha60', 'alpha71'] +
        [f'alpha{i}' for i in range(83, 87)]
    )
    
    print(f"\n[3/4] 运行回测 ({len(factors)} 个因子)...")
    
    engine = BacktestEngine(
        initial_capital=BacktestConfig.INITIAL_CAPITAL,
        fee_rate=BacktestConfig.TRANSACTION_FEE,
        slippage=BacktestConfig.SLIPPAGE,
        num_positions=BacktestConfig.MIN_POSITIONS,
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
        print("\n推荐因子 (TOP 10):")
        for r in filtered[:10]:
            print(f"  {r.factor_name}: "
                  f"年化={r.annual_return:.2%}, "
                  f"夏普={r.sharpe_ratio:.2f}, "
                  f"回撤={r.max_drawdown:.2%}")
    
    print(f"\n回测ID: {backtest_id}")
    print("\n使用真实数据请修改 USE_REAL_DATA=True")

if __name__ == "__main__":
    main()
