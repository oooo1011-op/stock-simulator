#!/usr/bin/env python3
"""
运行因子回测
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.postgres import load_daily_data, init_db
from src.engine.backtest import BacktestEngine
from src.config import BacktestConfig

def main():
    print("="*50)
    print("因子回测")
    print("="*50)
    
    # 初始化数据库
    print("\n[1/4] 初始化数据库...")
    init_db()
    
    # 获取股票列表（取前50只作为样本）
    print("\n[2/4] 获取股票列表...")
    from src.data.akshare_fetcher import AStockDataFetcher
    fetcher = AStockDataFetcher()
    stocks = fetcher.get_stock_list()
    symbols = stocks['symbol'].tolist()[:50]  # 先用50只测试
    print(f"使用 {len(symbols)} 只股票进行回测")
    
    # 加载数据
    print("\n[3/4] 加载数据...")
    start_date = BacktestConfig.BACKTEST_START_DATE
    end_date = BacktestConfig.BACKTEST_END_DATE
    
    all_data = []
    for symbol in symbols:
        try:
            df = load_daily_data(symbol, start_date, end_date)
            if not df.empty:
                all_data.append(df)
                print(f"  {symbol}: {len(df)} 条")
        except Exception as e:
            print(f"  {symbol}: 加载失败 - {e}")
    
    if not all_data:
        print("错误: 没有加载到任何数据")
        return
    
    import pandas as pd
    combined = pd.concat(all_data)
    print(f"总数据量: {len(combined)} 条")
    
    # 定义因子列表
    factor_names = (
        [f'alpha{i}' for i in range(1, 48)] +
        [f'alpha{i}' for i in range(49, 58)] +
        ['alpha60', 'alpha61', 'alpha62', 'alpha71'] +
        [f'alpha{i}' for i in range(83, 87)] +
        ['alpha88', 'alpha92', 'alpha95', 'alpha101']
    )
    
    print(f"\n[4/4] 运行回测 ({len(factor_names)} 个因子)...")
    
    engine = BacktestEngine(
        initial_capital=BacktestConfig.INITIAL_CAPITAL,
        fee_rate=BacktestConfig.TRANSACTION_FEE,
        slippage=BacktestConfig.SLIPPAGE,
        num_positions=BacktestConfig.MIN_POSITIONS,
    )
    
    # 运行回测
    results = engine.run_all_factors(
        combined,
        factor_names=factor_names,
        parallel=True,
    )
    
    if not results:
        print("错误: 没有回测结果")
        return
    
    # 筛选因子
    filtered = engine.filter_factors(
        results,
        min_return=BacktestConfig.MIN_ANNUAL_RETURN,
        min_sharpe=BacktestConfig.MIN_SHARPE_RATIO,
        max_dd=BacktestConfig.MAX_DRAWDOWN,
        min_win_rate=BacktestConfig.MIN_WIN_RATE,
    )
    
    # 保存结果
    backtest_id = engine.save_results(results)
    print(f"\n回测ID: {backtest_id}")
    
    # 展示结果
    print("\n" + "="*50)
    print("回测结果摘要")
    print("="*50)
    print(f"总因子数: {len(results)}")
    print(f"筛选后: {len(filtered)}")
    
    if filtered:
        print("\n推荐因子 (TOP 10):")
        for r in filtered[:10]:
            print(f"  {r.factor_name}: "
                  f"年化={r.annual_return:.2%}, "
                  f"夏普={r.sharpe_ratio:.2f}, "
                  f"回撤={r.max_drawdown:.2%}")
    else:
        print("\n没有因子符合筛选标准，降低阈值重试")

if __name__ == "__main__":
    main()
