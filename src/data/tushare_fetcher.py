"""
A-share data fetcher using Tushare Pro API.
Primary data source with akshare and baostock as fallbacks.
"""
import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DatabaseConfig, DataConfig

import loguru
logger = loguru.logger


class TushareDataFetcher:
    """
    Tushare Pro data fetcher with rate limiting.
    
    Rate limits for 2000 points:
    - 200 requests per minute
    - 100,000 requests per day per API
    """
    
    def __init__(self, token: str = None, request_delay: float = None):
        """
        Initialize Tushare fetcher.
        
        Args:
            token: Tushare API token (default from env)
            request_delay: Delay between requests in seconds (default 0.34s for 180 req/min)
        """
        self.token = token or os.getenv('TUSHARE_APIKEY') or os.getenv('TUSHARE_TOKEN')
        if not self.token:
            raise ValueError("Tushare token not found. Set TUSHARE_APIKEY in .env")
        
        # Initialize pro API
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
        # Rate limiting: 180 req/min for safety (leaving 10% margin)
        self.request_delay = request_delay or 0.34  # 60/180 = 0.33s
        self.request_count = 0
        self.last_request_time = None
        
        self._stock_list_cache = None
        logger.info("TushareDataFetcher initialized (rate limit: 180 req/min)")
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Log progress every 100 requests
        if self.request_count % 100 == 0:
            logger.info(f"Tushare API requests: {self.request_count}")
    
    def get_stock_list(self, market: str = 'mainboard', force_refresh: bool = False) -> pd.DataFrame:
        """
        Get stock list.
        
        Args:
            market: 'mainboard' for A-share mainboard, 'all' for all stocks
            force_refresh: Force refresh cache
        
        Returns:
            DataFrame with columns: [ts_code, symbol, name, area, industry, list_date, market]
        """
        if self._stock_list_cache is not None and not force_refresh:
            return self._stock_list_cache
        
        try:
            self._rate_limit()
            
            # Get all stocks
            df = self.pro.stock_basic(
                exchange='', 
                list_status='L',  # Listed
                fields='ts_code,symbol,name,area,industry,list_date,market,exchange'
            )
            
            if df is None or df.empty:
                logger.error("Failed to get stock list from Tushare")
                return pd.DataFrame()
            
            # Filter mainboard stocks
            if market == 'mainboard':
                # Mainboard: Shanghai 600/601/603, Shenzhen 000/001/002
                def is_mainboard(ts_code):
                    code = ts_code.split('.')[0]
                    prefix = code[:3]
                    return prefix in ['600', '601', '603', '000', '001', '002']
                
                df = df[df['ts_code'].apply(is_mainboard)]
            
            logger.info(f"Got {len(df)} {market} stocks from Tushare")
            self._stock_list_cache = df
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock list: {e}")
            return pd.DataFrame()
    
    def get_daily_data(
        self, 
        ts_code: str, 
        start_date: str = "20080101", 
        end_date: str = None,
        adj: str = "qfq"  # qfq: 前复权, hfq: 后复权, None: 不复权
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data for a single stock.
        
        Args:
            ts_code: Stock code with exchange suffix (e.g., '000001.SZ')
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD), default yesterday
            adj: Adjustment type
        
        Returns:
            DataFrame with columns: [ts_code, trade_date, open, high, low, close, 
                                     pre_close, change, pct_chg, vol, amount]
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        try:
            self._rate_limit()
            
            # Get daily data
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Get adjustment factor if needed
            if adj:
                self._rate_limit()
                adj_df = self.pro.adj_factor(ts_code=ts_code)
                if adj_df is not None and not adj_df.empty:
                    df = df.merge(adj_df[['trade_date', 'adj_factor']], on='trade_date', how='left')
                    
                    # Apply adjustment
                    if adj == 'qfq':  # 前复权
                        latest_adj = adj_df['adj_factor'].iloc[0]
                        for col in ['open', 'high', 'low', 'close', 'pre_close']:
                            df[col] = df[col] * df['adj_factor'] / latest_adj
                    elif adj == 'hfq':  # 后复权
                        for col in ['open', 'high', 'low', 'close', 'pre_close']:
                            df[col] = df[col] * df['adj_factor']
            
            # Sort by date
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            
            return df
            
        except Exception as e:
            logger.warning(f"Error getting daily data for {ts_code}: {e}")
            return pd.DataFrame()
    
    def get_daily_basic(
        self, 
        ts_code: str = None, 
        start_date: str = "20080101", 
        end_date: str = None,
        trade_date: str = None
    ) -> pd.DataFrame:
        """
        Get daily basic indicators (PE, PB, turnover, etc.).
        
        Args:
            ts_code: Stock code (optional, if None returns all stocks for trade_date)
            start_date: Start date
            end_date: End date
            trade_date: Specific trade date (gets all stocks for that day)
        
        Returns:
            DataFrame with columns: [ts_code, trade_date, close, turnover_rate, 
                                     turnover_rate_f, volume_ratio, pe, pe_ttm, pb, 
                                     ps, ps_ttm, dv_ratio, total_share, float_share, 
                                     total_mv, circ_mv]
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        try:
            self._rate_limit()
            
            params = {
                'start_date': start_date,
                'end_date': end_date
            }
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date
            
            df = self.pro.daily_basic(**params)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values(['ts_code', 'trade_date'])
            
            return df
            
        except Exception as e:
            logger.warning(f"Error getting daily basic: {e}")
            return pd.DataFrame()
    
    def get_trade_calendar(self, start_date: str = "20080101", end_date: str = None) -> pd.DataFrame:
        """
        Get trade calendar.
        
        Returns:
            DataFrame with columns: [exchange, cal_date, is_open, pretrade_date]
        """
        if end_date is None:
            end_date = (datetime.now() + timedelta(days=365)).strftime('%Y%m%d')
        
        try:
            self._rate_limit()
            
            df = self.pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting trade calendar: {e}")
            return pd.DataFrame()
    
    def batch_get_daily_data(
        self,
        ts_codes: List[str],
        start_date: str = "20080101",
        end_date: str = None,
        progress_interval: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch get daily data for multiple stocks.
        
        Args:
            ts_codes: List of stock codes
            start_date: Start date
            end_date: End date
            progress_interval: Log progress every N stocks
        
        Returns:
            Dict mapping ts_code to DataFrame
        """
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        results = {}
        total = len(ts_codes)
        success = 0
        failed = 0
        
        logger.info(f"Batch downloading {total} stocks from {start_date} to {end_date}")
        
        for i, ts_code in enumerate(ts_codes):
            df = self.get_daily_data(ts_code, start_date, end_date)
            
            if not df.empty:
                results[ts_code] = df
                success += 1
            else:
                failed += 1
            
            if (i + 1) % progress_interval == 0:
                logger.info(f"Progress: {i+1}/{total}, Success: {success}, Failed: {failed}")
        
        logger.info(f"Batch download complete: Success={success}, Failed={failed}")
        return results
    
    def get_request_stats(self) -> Dict:
        """Get request statistics"""
        return {
            'total_requests': self.request_count,
            'rate_limit': '180 req/min',
            'request_delay': self.request_delay
        }


class MultiSourceDataFetcher:
    """
    Multi-source data fetcher with automatic fallback.
    Priority: Tushare -> Akshare -> Baostock
    """
    
    def __init__(self):
        self.fetchers = {}
        
        # Try Tushare first
        try:
            self.fetchers['tushare'] = TushareDataFetcher()
            logger.info("Tushare initialized as primary source")
        except Exception as e:
            logger.warning(f"Tushare initialization failed: {e}")
        
        # Fallback to akshare
        try:
            from .akshare_fetcher import AStockDataFetcher
            self.fetchers['akshare'] = AStockDataFetcher()
            logger.info("Akshare initialized as backup source")
        except Exception as e:
            logger.warning(f"Akshare initialization failed: {e}")
        
        # Fallback to baostock
        try:
            from .baostock_fetcher import BaostockDataFetcher
            self.fetchers['baostock'] = BaostockDataFetcher()
            logger.info("Baostock initialized as backup source")
        except Exception as e:
            logger.warning(f"Baostock initialization failed: {e}")
        
        if not self.fetchers:
            raise RuntimeError("No data source available")
    
    def get_stock_list(self, market: str = 'mainboard') -> pd.DataFrame:
        """Get stock list from available sources"""
        for source_name, fetcher in self.fetchers.items():
            try:
                df = fetcher.get_stock_list(market=market)
                if not df.empty:
                    logger.info(f"Got stock list from {source_name}")
                    return df
            except Exception as e:
                logger.warning(f"{source_name} failed: {e}")
        
        return pd.DataFrame()
    
    def get_daily_data(self, symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """Get daily data with fallback"""
        # Ensure symbol has exchange suffix for tushare
        if '.' not in symbol:
            # Default to SZ for 000/001/002, SH for 600/601/603
            prefix = symbol[:3]
            if prefix in ['000', '001', '002']:
                ts_code = f"{symbol}.SZ"
            else:
                ts_code = f"{symbol}.SH"
        else:
            ts_code = symbol
        
        for source_name, fetcher in self.fetchers.items():
            try:
                if source_name == 'tushare':
                    df = fetcher.get_daily_data(ts_code, start_date, end_date)
                else:
                    df = fetcher.get_daily_data(symbol, start_date, end_date)
                
                if not df.empty:
                    logger.debug(f"Got data from {source_name}")
                    return df
            except Exception as e:
                logger.warning(f"{source_name} failed for {symbol}: {e}")
        
        return pd.DataFrame()
