"""
A-share data fetcher using akshare.
Fetches daily OHLCV data from 2008-01-01 to present.
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import time
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DataConfig

import loguru
logger = loguru.logger


class AStockDataFetcher:
    """A-share data fetcher using akshare"""
    
    def __init__(self, request_delay: float = None):
        """
        Initialize fetcher.
        
        Args:
            request_delay: Delay between API requests in seconds
        """
        self.request_delay = request_delay or DataConfig.REQUEST_DELAY
        self._stock_list_cache = None
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        time.sleep(self.request_delay)
    
    def get_stock_list(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get list of all A-share stocks.
        
        Returns:
            DataFrame with columns: [symbol, name, market, area, industry, list_date]
        """
        if self._stock_list_cache is not None and not force_refresh:
            return self._stock_list_cache
        
        try:
            # Use akshare East Money data
            df = ak.stock_zh_a_spot_em()
            
            # Standardize columns
            df.columns = df.columns.str.lower()
            
            # Select and rename relevant columns
            columns_map = {
                '代码': 'symbol',
                '名称': 'name',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '最高': 'high',
                '最低': 'low',
                '今开': 'open',
                '昨收': 'pre_close',
                '换手率': 'turnover',
                '市盈率-动态': 'pe',
                '市净率': 'pb',
                '总市值': 'total_mv',
                '流通市值': 'circ_mv'
            }
            
            # Keep only existing columns
            available_cols = {k: v for k, v in columns_map.items() if k in df.columns}
            df = df.rename(columns=available_cols)
            
            # Add prefix to symbol for consistency
            df['symbol'] = df['symbol'].astype(str).str.zfill(6)
            
            # Filter out ST stocks, etc.
            # df = df[~df['name'].str.contains('ST', na=False)]
            
            self._stock_list_cache = df
            logger.info(f"获取到 {len(df)} 只A股股票")
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise
    
    def get_daily_data(
        self, 
        symbol: str, 
        start_date: str = "20080101", 
        end_date: str = None,
        adjust: str = "qfq"  # qfq: 前复权, hf: 后复权, None: 不复权
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data for a single stock.
        
        Args:
            symbol: Stock code (e.g., '000001', '600000')
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format (default: today)
            adjust: Price adjustment type ('qfq', 'hf', or None)
        
        Returns:
            DataFrame with columns:
            - date: Trading date (YYYY-MM-DD)
            - open: Opening price
            - high: Highest price
            - low: Lowest price
            - close: Closing price
            - volume: Trading volume (shares)
            - amount: Trading amount (yuan)
            - change_pct: Daily change percentage
            - change: Daily change amount
            - turnover: Turnover rate
            - amplitude: Amplitude
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        # Determine market
        if symbol.startswith('6'):
            market = "sh"
        elif symbol.startswith('0') or symbol.startswith('3'):
            market = "sz"
        else:
            raise ValueError(f"Invalid stock code: {symbol}")
        
        try:
            # Fetch data from akshare
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            
            if df.empty:
                logger.warning(f"No data for {symbol} from {start_date} to {end_date}")
                return df
            
            # Standardize column names
            df.columns = df.columns.str.lower()
            
            # Rename columns to standard format
            columns_map = {
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '换手率': 'turnover'
            }
            
            df = df.rename(columns=columns_map)
            
            # Convert date format
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Convert numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                          'change_pct', 'change', 'turnover', 'amplitude']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Adjust volume unit (akshare returns 手, convert to 股)
            if 'volume' in df.columns:
                df['volume'] = df['volume'] * 100
            
            # Adjust amount unit (akshare returns 万元, convert to 元)
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * 10000
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.debug(f"获取 {symbol} 数据: {len(df)} 条, {df['date'].min()} ~ {df['date'].max()}")
            
            self._rate_limit()
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {symbol} 日线数据失败: {e}")
            raise
    
    def get_bulk_daily_data(
        self,
        symbols: List[str],
        start_date: str = "20080101",
        end_date: str = None,
        progress: bool = True,
        on_progress: callable = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch daily data for multiple stocks.
        
        Args:
            symbols: List of stock codes
            start_date: Start date
            end_date: End date
            progress: Whether to show progress
            on_progress: Optional callback function(progress_percent, current, total
        
        Returns:
            Dict mapping stock code to DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        results = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                df = self.get_daily_data(symbol, start_date, end_date)
                results[symbol] = df
                
                if progress and (i + 1) % 10 == 0:
                    pct = (i + 1) / total * 100
                    logger.info(f"进度: {i+1}/{total} ({pct:.1f}%)")
                    if on_progress:
                        on_progress(pct, i + 1, total)
                        
            except Exception as e:
                logger.warning(f"获取 {symbol} 数据失败: {e}")
                results[symbol] = pd.DataFrame()
                continue
        
        logger.info(f"批量获取完成: {len(results)} 只股票, 成功 {sum(1 for v in results.values() if not v.empty)} 只")
        
        return results
    
    def get_trading_calender(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get trading calendar.
        
        Args:
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
        
        Returns:
            DataFrame with trading dates
        """
        try:
            cal = ak.tool_trade_date_hist_sina()
            
            # Filter by date range
            cal = cal[
                (cal['trade_date'] >= start_date) & 
                (cal['trade_date'] <= end_date)
            ]
            
            cal = cal.sort_values('trade_date').reset_index(drop=True)
            
            logger.info(f"交易日历: {len(cal)} 个交易日")
            
            return cal
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            raise
    
    def get_market_indices(self) -> Dict[str, pd.DataFrame]:
        """
        Get major market indices.
        
        Returns:
            Dict mapping index code to DataFrame
        """
        indices = {
            'sh000001': '上证指数',
            'sz399001': '深证成指',
            'sz399006': '创业板指',
            'sh000300': '沪深300',
            'sh000016': '上证50',
        }
        
        result = {}
        
        for code, name in indices.items():
            try:
                df = ak.stock_zh_index_daily(symbol=code)
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                df['symbol'] = code
                df['name'] = name
                result[code] = df
                logger.debug(f"获取 {name} ({code}) 数据")
                
                self._rate_limit()
                
            except Exception as e:
                logger.warning(f"获取 {name} 数据失败: {e}")
                continue
        
        return result
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed stock information.
        
        Args:
            symbol: Stock code
        
        Returns:
            Dict with stock details
        """
        try:
            info = ak.stock_info_a_code_name_map()
            
            if symbol in info:
                return {
                    'symbol': symbol,
                    'name': info[symbol],
                    'listed': True
                }
            else:
                return {
                    'symbol': symbol,
                    'name': None,
                    'listed': False
                }
                
        except Exception as e:
            logger.error(f"获取 {symbol} 股票信息失败: {e}")
            raise


class DataFetcher:
    """Unified data fetcher interface"""
    
    def __init__(self, source: str = "akshare"):
        """
        Initialize data fetcher.
        
        Args:
            source: Data source ('akshare', 'baostock')
        """
        self.source = source
        
        if source == "akshare":
            self.fetcher = AStockDataFetcher()
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def get_daily(
        self, 
        symbol: str, 
        start_date: str = "20080101",
        end_date: str = None
    ) -> pd.DataFrame:
        """Get daily data"""
        return self.fetcher.get_daily_data(symbol, start_date, end_date)
    
    def get_bulk(
        self,
        symbols: List[str],
        start_date: str = "20080101",
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Get bulk daily data"""
        return self.fetcher.get_bulk_daily_data(symbols, start_date, end_date)
    
    def get_trading_dates(
        self, 
        start_date: str, 
        end_date: str
    ) -> List[str]:
        """Get list of trading dates"""
        cal = self.fetcher.get_trading_calender(start_date, end_date)
        return cal['trade_date'].tolist()
    
    def get_universe(self, listed_only: bool = True) -> List[str]:
        """Get list of stock symbols"""
        df = self.fetcher.get_stock_list()
        if listed_only:
            # Filter by list date (before now)
            now = datetime.now().strftime('%Y%m%d')
            df = df[df.get('list_date', now) <= now]
        return df['symbol'].tolist()


if __name__ == "__main__":
    # Test data fetcher
    fetcher = AStockDataFetcher()
    
    # Test single stock
    print("测试获取单只股票数据...")
    df = fetcher.get_daily_data("000001", "20240101", "20240201")
    print(f"000001: {len(df)} 条记录")
    print(df.head())
    
    # Test stock list
    print("\n测试获取股票列表...")
    stocks = fetcher.get_stock_list()
    print(f"总股票数: {len(stocks)}")
    
    # Test trading calendar
    print("\n测试获取交易日历...")
    cal = fetcher.get_trading_calender("20240101", "20240201")
    print(f"2024年1-2月交易日: {len(cal)} 天")
