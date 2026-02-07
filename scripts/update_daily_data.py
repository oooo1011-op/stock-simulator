#!/usr/bin/env python3
"""
每日增量数据更新脚本
更新从数据库最新日期到昨天的数据

Usage:
    python update_daily_data.py          # 更新到昨天
    python update_daily_data.py --date 20240101  # 更新到指定日期
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime, timedelta
import argparse

from sqlalchemy import create_engine, text, func
from src.config import DatabaseConfig
from src.data.tushare_fetcher import TushareDataFetcher
from src.data.akshare_fetcher import AStockDataFetcher
from src.data.baostock_fetcher import BaostockDataFetcher

import loguru
logger = loguru.logger


class DailyDataUpdater:
    """Daily incremental data updater"""
    
    def __init__(self):
        self.engine = create_engine(DatabaseConfig.get_postgres_uri())
        self._init_fetchers()
    
    def _init_fetchers(self):
        """Initialize data fetchers"""
        self.fetchers = {}
        
        try:
            self.fetchers['tushare'] = TushareDataFetcher()
            logger.info("✅ Tushare ready")
        except Exception as e:
            logger.warning(f"⚠️ Tushare: {e}")
        
        try:
            self.fetchers['akshare'] = AStockDataFetcher()
            logger.info("✅ Akshare ready")
        except Exception as e:
            logger.warning(f"⚠️ Akshare: {e}")
        
        try:
            self.fetchers['baostock'] = BaostockDataFetcher()
            logger.info("✅ Baostock ready")
        except Exception as e:
            logger.warning(f"⚠️ Baostock: {e}")
    
    def get_latest_date_in_db(self) -> str:
        """Get the latest date in database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT MAX(date) FROM daily_prices"))
                latest_date = result.scalar()
                
                if latest_date:
                    # If it's today or yesterday, we might already have latest data
                    return latest_date.strftime('%Y%m%d')
                else:
                    return '20080101'  # Default start if no data
                    
        except Exception as e:
            logger.error(f"Failed to get latest date: {e}")
            return '20080101'
    
    def get_stocks_in_db(self) -> list:
        """Get list of stocks already in database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT DISTINCT stock_code FROM daily_prices"))
                stocks = [row[0] for row in result]
                return stocks
        except Exception as e:
            logger.error(f"Failed to get stock list: {e}")
            return []
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get stock data with fallback"""
        symbol_clean = symbol.split('.')[0] if '.' in symbol else symbol
        
        for source_name, fetcher in self.fetchers.items():
            try:
                df = None
                
                if source_name == 'tushare':
                    prefix = symbol_clean[:3]
                    suffix = '.SZ' if prefix in ['000', '001', '002'] else '.SH'
                    ts_code = f"{symbol_clean}{suffix}"
                    df = fetcher.get_daily_data(ts_code, start_date, end_date, adj='qfq')
                    
                elif source_name == 'akshare':
                    df = fetcher.get_daily_data(symbol_clean, start_date, end_date)
                    
                elif source_name == 'baostock':
                    prefix = 'sh' if symbol_clean[:3] in ['600', '601', '603'] else 'sz'
                    bs_code = f"{prefix}.{symbol_clean}"
                    df = fetcher.get_daily_data(bs_code, start_date, end_date)
                
                if df is not None and not df.empty:
                    return self._standardize_columns(df, symbol_clean, source_name)
                    
            except Exception as e:
                continue
        
        return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        """Standardize column names"""
        df = df.copy()
        
        # Column mapping
        column_mapping = {
            'tushare': {
                'ts_code': 'stock_code',
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'pre_close': 'pre_close',
                'change': 'change',
                'pct_chg': 'pct_change',
                'vol': 'volume',
                'amount': 'amount'
            },
            'akshare': {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change'
            },
            'baostock': {
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'preclose': 'pre_close',
                'volume': 'volume',
                'amount': 'amount',
                'pctChg': 'pct_change'
            }
        }
        
        mapping = column_mapping.get(source, {})
        df = df.rename(columns=mapping)
        
        # Ensure date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'trade_date' in df.columns:
            df['date'] = pd.to_datetime(df['trade_date'])
        
        # Add stock_code
        if 'stock_code' not in df.columns:
            df['stock_code'] = symbol
        else:
            df['stock_code'] = df['stock_code'].astype(str).str.split('.').str[0]
        
        # Convert to numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Unit conversion
        if source == 'akshare':
            if 'volume' in df.columns:
                df['volume'] = df['volume'] / 100  # 股 -> 手
            if 'amount' in df.columns:
                df['amount'] = df['amount'] / 1000  # 元 -> 千元
        
        # Select standard columns
        standard_cols = ['stock_code', 'date', 'open', 'high', 'low', 'close', 
                        'pre_close', 'change', 'pct_change', 'volume', 'amount']
        available_cols = [c for c in standard_cols if c in df.columns]
        
        return df[available_cols]
    
    def save_to_database(self, df: pd.DataFrame) -> int:
        """Save data to database"""
        if df.empty:
            return 0
        
        try:
            rows_inserted = 0
            
            with self.engine.connect() as conn:
                for _, row in df.iterrows():
                    stmt = text("""
                        INSERT INTO daily_prices 
                        (stock_code, date, open, high, low, close, pre_close, change, pct_change, volume, amount)
                        VALUES (:stock_code, :date, :open, :high, :low, :close, :pre_close, :change, :pct_change, :volume, :amount)
                        ON CONFLICT (stock_code, date) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            pre_close = EXCLUDED.pre_close,
                            change = EXCLUDED.change,
                            pct_change = EXCLUDED.pct_change,
                            volume = EXCLUDED.volume,
                            amount = EXCLUDED.amount,
                            created_at = CURRENT_TIMESTAMP
                    """)
                    
                    params = {
                        'stock_code': row.get('stock_code'),
                        'date': row.get('date'),
                        'open': float(row.get('open')) if pd.notna(row.get('open')) else None,
                        'high': float(row.get('high')) if pd.notna(row.get('high')) else None,
                        'low': float(row.get('low')) if pd.notna(row.get('low')) else None,
                        'close': float(row.get('close')) if pd.notna(row.get('close')) else None,
                        'pre_close': float(row.get('pre_close')) if pd.notna(row.get('pre_close')) else None,
                        'change': float(row.get('change')) if pd.notna(row.get('change')) else None,
                        'pct_change': float(row.get('pct_change')) if pd.notna(row.get('pct_change')) else None,
                        'volume': float(row.get('volume')) if pd.notna(row.get('volume')) else None,
                        'amount': float(row.get('amount')) if pd.notna(row.get('amount')) else None
                    }
                    
                    conn.execute(stmt, params)
                    rows_inserted += 1
                
                conn.commit()
            
            return rows_inserted
            
        except Exception as e:
            logger.error(f"Database error: {e}")
            return 0
    
    def update(self, end_date: str = None):
        """
        Update data from latest in DB to end_date.
        
        Args:
            end_date: End date (YYYYMMDD), default yesterday
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        # Get latest date in database
        latest_date = self.get_latest_date_in_db()
        
        # Calculate start date (latest + 1 day)
        latest_dt = datetime.strptime(latest_date, '%Y%m%d')
        start_dt = latest_dt + timedelta(days=1)
        start_date = start_dt.strftime('%Y%m%d')
        
        # Check if we need to update
        if start_date > end_date:
            logger.info("✅ Database is already up to date!")
            logger.info(f"   Latest date in DB: {latest_date}")
            logger.info(f"   Target end date: {end_date}")
            return
        
        logger.info("="*60)
        logger.info("📊 每日增量数据更新")
        logger.info("="*60)
        logger.info(f"更新范围: {start_date} ~ {end_date}")
        
        # Get stocks in database
        stocks = self.get_stocks_in_db()
        if not stocks:
            logger.error("❌ No stocks found in database. Run import_data_to_db.py first.")
            return
        
        logger.info(f"股票数量: {len(stocks)}")
        
        # Update data
        success = 0
        failed = 0
        total_records = 0
        
        logger.info("\n🚀 开始更新...")
        
        for i, symbol in enumerate(stocks):
            try:
                df = self.get_stock_data(symbol, start_date, end_date)
                
                if not df.empty:
                    rows = self.save_to_database(df)
                    total_records += rows
                    success += 1
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"进度: {i+1}/{len(stocks)} | 成功: {success} | 新增记录: {total_records}")
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                logger.warning(f"更新失败 {symbol}: {e}")
        
        logger.info("\n" + "="*60)
        logger.info("📊 更新完成!")
        logger.info("="*60)
        logger.info(f"成功: {success}")
        logger.info(f"失败: {failed}")
        logger.info(f"新增记录: {total_records:,}")


def main():
    parser = argparse.ArgumentParser(description='Daily incremental data update')
    parser.add_argument('--date', default=None, help='End date (YYYYMMDD), default yesterday')
    
    args = parser.parse_args()
    
    updater = DailyDataUpdater()
    updater.update(end_date=args.date)


if __name__ == '__main__':
    main()
