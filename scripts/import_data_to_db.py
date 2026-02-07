#!/usr/bin/env python3
"""
一次性数据导入脚本
导入2008-01-01到昨天（交易日）的主板股票数据到PostgreSQL

数据源优先级: Tushare -> Akshare -> Baostock
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import argparse

from sqlalchemy import create_engine, text
from src.config import DatabaseConfig, DataConfig, TushareConfig
from src.database.postgres import init_db, get_db_session, daily_prices_table, metadata

import loguru
logger = loguru.logger


class DataImporter:
    """Data importer with multi-source fallback"""
    
    def __init__(self):
        self.engine = create_engine(DatabaseConfig.get_postgres_uri())
        self._init_fetchers()
        
        # Calculate yesterday's date
        self.yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        # Statistics
        self.stats = {
            'total_stocks': 0,
            'success_stocks': 0,
            'failed_stocks': 0,
            'total_records': 0,
            'tushare_requests': 0,
            'akshare_requests': 0,
            'baostock_requests': 0
        }
    
    def _init_fetchers(self):
        """Initialize data fetchers with fallback"""
        self.fetchers = {}
        
        # Tushare (primary)
        try:
            from src.data.tushare_fetcher import TushareDataFetcher
            self.fetchers['tushare'] = TushareDataFetcher()
            logger.info("✅ Tushare initialized (primary)")
        except Exception as e:
            logger.warning(f"⚠️ Tushare failed: {e}")
        
        # Akshare (backup 1)
        try:
            from src.data.akshare_fetcher import AStockDataFetcher
            self.fetchers['akshare'] = AStockDataFetcher()
            logger.info("✅ Akshare initialized (backup 1)")
        except Exception as e:
            logger.warning(f"⚠️ Akshare failed: {e}")
        
        # Baostock (backup 2)
        try:
            from src.data.baostock_fetcher import BaostockDataFetcher
            self.fetchers['baostock'] = BaostockDataFetcher()
            logger.info("✅ Baostock initialized (backup 2)")
        except Exception as e:
            logger.warning(f"⚠️ Baostock failed: {e}")
        
        if not self.fetchers:
            raise RuntimeError("❌ No data source available!")
    
    def get_stock_list(self) -> pd.DataFrame:
        """Get mainboard stock list"""
        logger.info("📋 Getting mainboard stock list...")
        
        for source_name, fetcher in self.fetchers.items():
            try:
                if source_name == 'tushare':
                    df = fetcher.get_stock_list(market='mainboard')
                else:
                    df = fetcher.get_stock_list()
                    # Filter mainboard
                    if not df.empty and 'symbol' in df.columns:
                        df['prefix'] = df['symbol'].astype(str).str[:3]
                        df = df[df['prefix'].isin(['600', '601', '603', '000', '001', '002'])]
                
                if not df.empty:
                    logger.info(f"✅ Got {len(df)} stocks from {source_name}")
                    return df
            except Exception as e:
                logger.warning(f"{source_name} failed: {e}")
        
        return pd.DataFrame()
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get stock data with fallback.
        Returns DataFrame with standardized columns.
        """
        # Normalize symbol
        symbol_clean = symbol.split('.')[0] if '.' in symbol else symbol
        
        # Try each source
        for source_name, fetcher in self.fetchers.items():
            try:
                df = None
                
                if source_name == 'tushare':
                    # Tushare needs exchange suffix
                    prefix = symbol_clean[:3]
                    suffix = '.SZ' if prefix in ['000', '001', '002'] else '.SH'
                    ts_code = f"{symbol_clean}{suffix}"
                    df = fetcher.get_daily_data(ts_code, start_date, end_date, adj='qfq')
                    self.stats['tushare_requests'] += 1
                    
                elif source_name == 'akshare':
                    df = fetcher.get_daily_data(symbol_clean, start_date, end_date)
                    self.stats['akshare_requests'] += 1
                    
                elif source_name == 'baostock':
                    prefix = 'sh' if symbol_clean[:3] in ['600', '601', '603'] else 'sz'
                    bs_code = f"{prefix}.{symbol_clean}"
                    df = fetcher.get_daily_data(bs_code, start_date, end_date)
                    self.stats['baostock_requests'] += 1
                
                if df is not None and not df.empty and len(df) >= 60:
                    # Standardize columns
                    df = self._standardize_columns(df, symbol_clean, source_name)
                    return df
                    
            except Exception as e:
                logger.debug(f"{source_name} failed for {symbol}: {e}")
                continue
        
        return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
        """Standardize column names and units"""
        df = df.copy()
        
        # Map source-specific columns to standard
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
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            },
            'baostock': {
                'date': 'date',
                'code': 'stock_code',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'preclose': 'pre_close',
                'volume': 'volume',
                'amount': 'amount',
                'turn': 'turnover',
                'pctChg': 'pct_change'
            }
        }
        
        mapping = column_mapping.get(source, {})
        df = df.rename(columns=mapping)
        
        # Ensure date column
        if 'date' not in df.columns:
            if 'trade_date' in df.columns:
                df['date'] = df['trade_date']
        
        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Add stock_code if missing
        if 'stock_code' not in df.columns:
            df['stock_code'] = symbol
        else:
            # Clean stock_code (remove suffix)
            df['stock_code'] = df['stock_code'].astype(str).str.split('.').str[0]
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pre_close', 'change', 'pct_change']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Unit conversions
        # Tushare: vol=手, amount=千元
        # Akshare: volume=股, amount=元
        # Standardize to: volume=手, amount=千元
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
        """Save DataFrame to database using UPSERT"""
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
    
    def import_data(self, start_date: str = "20080101", end_date: str = None, limit: int = None):
        """
        Import all stock data to database.
        
        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD), default yesterday
            limit: Limit number of stocks (for testing)
        """
        if end_date is None:
            end_date = self.yesterday
        
        logger.info("="*60)
        logger.info("📊 A股主板数据导入")
        logger.info("="*60)
        logger.info(f"时间范围: {start_date} ~ {end_date}")
        
        # Initialize database
        logger.info("\n🔧 Initializing database...")
        init_db()
        
        # Get stock list
        stocks_df = self.get_stock_list()
        if stocks_df.empty:
            logger.error("❌ Failed to get stock list")
            return
        
        # Prepare stock list
        if 'symbol' in stocks_df.columns:
            stocks = stocks_df['symbol'].astype(str).tolist()
        elif 'ts_code' in stocks_df.columns:
            stocks = stocks_df['ts_code'].astype(str).str.split('.').str[0].tolist()
        else:
            logger.error("❌ No valid stock code column found")
            return
        
        # Remove duplicates
        stocks = list(set(stocks))
        
        if limit:
            stocks = stocks[:limit]
        
        self.stats['total_stocks'] = len(stocks)
        logger.info(f"\n📈 Total stocks to import: {len(stocks)}")
        
        # Import data
        logger.info("\n🚀 Starting import...")
        start_time = time.time()
        
        for i, symbol in enumerate(stocks):
            try:
                # Get data
                df = self.get_stock_data(symbol, start_date, end_date)
                
                if not df.empty:
                    # Save to database
                    rows = self.save_to_database(df)
                    self.stats['total_records'] += rows
                    self.stats['success_stocks'] += 1
                    
                    if (i + 1) % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed * 60
                        logger.info(f"Progress: {i+1}/{len(stocks)} | "
                                  f"Success: {self.stats['success_stocks']} | "
                                  f"Records: {self.stats['total_records']:,} | "
                                  f"Rate: {rate:.0f} stocks/min")
                else:
                    self.stats['failed_stocks'] += 1
                    logger.debug(f"No data for {symbol}")
                
            except Exception as e:
                self.stats['failed_stocks'] += 1
                logger.warning(f"Failed to import {symbol}: {e}")
                continue
        
        # Final report
        elapsed = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("📊 Import Complete!")
        logger.info("="*60)
        logger.info(f"Total stocks: {self.stats['total_stocks']}")
        logger.info(f"Success: {self.stats['success_stocks']}")
        logger.info(f"Failed: {self.stats['failed_stocks']}")
        logger.info(f"Total records: {self.stats['total_records']:,}")
        logger.info(f"Time elapsed: {elapsed/60:.1f} minutes")
        logger.info(f"Average rate: {self.stats['total_stocks']/elapsed*60:.0f} stocks/min")
        logger.info(f"\nAPI requests:")
        logger.info(f"  Tushare: {self.stats['tushare_requests']}")
        logger.info(f"  Akshare: {self.stats['akshare_requests']}")
        logger.info(f"  Baostock: {self.stats['baostock_requests']}")
        
        # Save report
        self._save_report(start_date, end_date, elapsed)
    
    def _save_report(self, start_date: str, end_date: str, elapsed: float):
        """Save import report to file"""
        report_file = f"logs/import_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        os.makedirs('logs', exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("A股主板数据导入报告\n")
            f.write("="*60 + "\n")
            f.write(f"导入时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"时间范围: {start_date} ~ {end_date}\n")
            f.write(f"\n统计:\n")
            f.write(f"  总股票数: {self.stats['total_stocks']}\n")
            f.write(f"  成功: {self.stats['success_stocks']}\n")
            f.write(f"  失败: {self.stats['failed_stocks']}\n")
            f.write(f"  总记录数: {self.stats['total_records']:,}\n")
            f.write(f"  耗时: {elapsed/60:.1f} 分钟\n")
            f.write(f"\nAPI调用:\n")
            f.write(f"  Tushare: {self.stats['tushare_requests']}\n")
            f.write(f"  Akshare: {self.stats['akshare_requests']}\n")
            f.write(f"  Baostock: {self.stats['baostock_requests']}\n")
        
        logger.info(f"\n📄 Report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Import A-share data to database')
    parser.add_argument('--start', default='20080101', help='Start date (YYYYMMDD)')
    parser.add_argument('--end', default=None, help='End date (YYYYMMDD), default yesterday')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of stocks (for testing)')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start, '%Y%m%d')
        if args.end:
            datetime.strptime(args.end, '%Y%m%d')
    except ValueError:
        print("❌ Invalid date format. Use YYYYMMDD")
        return
    
    # Run import
    importer = DataImporter()
    importer.import_data(start_date=args.start, end_date=args.end, limit=args.limit)


if __name__ == '__main__':
    main()
