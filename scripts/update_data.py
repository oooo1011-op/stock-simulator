#!/usr/bin/env python3
"""
全量更新A股数据（2008-2025）
注意单位：
- volume: akshare返回"手"，转为"股"（×100）
- amount: akshare返回"万元"，转为"元"（×10000）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.akshare_fetcher import AStockDataFetcher
from src.database.postgres import save_daily_data, init_db
from src.config import BacktestConfig

def main():
    print("="*50)
    print("A股数据全量更新")
    print("="*50)
    
    # 初始化数据库
    print("\n[1/4] 初始化数据库...")
    init_db()
    
    # 获取股票列表
    print("\n[2/4] 获取A股股票列表...")
    fetcher = AStockDataFetcher()
    stocks = fetcher.get_stock_list()
    symbols = stocks['symbol'].tolist()
    print(f"共 {len(symbols)} 只股票")
    
    # 数据参数
    start_date = BacktestConfig.BACKTEST_START_DATE
    end_date = BacktestConfig.BACKTEST_END_DATE
    print(f"\n[3/4] 更新数据范围: {start_date} ~ {end_date}")
    
    # 批量更新
    print("\n[4/4] 开始更新数据...")
    total = len(symbols)
    success = 0
    failed = []
    
    for i, symbol in enumerate(symbols):
        try:
            df = fetcher.get_daily_data(symbol, start_date, end_date, adjust="qfq")
            
            if df.empty:
                continue
            
            # 修正单位
            # volume: 手 -> 股 (×100)
            if 'volume' in df.columns:
                df['volume'] = df['volume'] * 100
            
            # amount: 万元 -> 元 (×10000)  
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * 10000
            
            # 保存
            save_daily_data(symbol, df)
            success += 1
            
            if (i + 1) % 100 == 0:
                print(f"进度: {i+1}/{total} ({success}成功, {len(failed)}失败)")
                
        except Exception as e:
            failed.append((symbol, str(e)))
            if len(failed) <= 10:
                print(f"  警告: {symbol} - {e}")
    
    print(f"\n完成！成功: {success}, 失败: {len(failed)}")

if __name__ == "__main__":
    main()
