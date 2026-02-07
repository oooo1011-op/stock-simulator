#!/usr/bin/env python3
"""
使用baostock更新A股数据
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.baostock_fetcher import BaoStockFetcher
from src.database.postgres import save_daily_data, init_db
from src.config import BacktestConfig

def main():
    print("="*50)
    print("A股数据更新 (baostock)")
    print("="*50)
    
    # 初始化数据库
    print("\n[1/3] 初始化数据库...")
    init_db()
    
    # 获取股票列表
    print("\n[2/3] 获取股票列表...")
    fetcher = BaoStockFetcher()
    stocks = fetcher.get_stock_list()
    # 过滤沪深A股
    symbols = stocks[stocks['code'].str.contains('sh\\.|sz\\.')]['code'].tolist()
    symbols = [s.split('.')[-1] for s in symbols[:100]]  # 先取100只
    print(f"共 {len(symbols)} 只股票")
    
    # 数据范围
    start_date = "20080101"
    end_date = "20251231"
    print(f"\n[3/3] 更新数据: {start_date} ~ {end_date}")
    
    success = 0
    for i, symbol in enumerate(symbols):
        try:
            df = fetcher.get_daily_data(symbol, start_date, end_date)
            if not df.empty:
                save_daily_data(symbol, df)
                success += 1
            if (i + 1) % 10 == 0:
                print(f"进度: {i+1}/{len(symbols)}")
        except Exception as e:
            print(f"  警告: {symbol} - {e}")
    
    print(f"\n完成! 成功: {success}/{len(symbols)}")

if __name__ == "__main__":
    main()
