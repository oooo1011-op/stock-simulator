"""
A股数据获取模块
使用akshare获取2008年至今的历史数据
"""
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional, Dict, List
import time

class AStockDataFetcher:
    """A股数据获取器"""
    
    def __init__(self):
        self.cache = {}
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        try:
            df = ak.stock_zh_a_spot_em()
            logger.info(f"获取到 {len(df)} 只A股股票")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise
    
    def get_daily_data(
        self, 
        symbol: str, 
        start_date: str = "20080101", 
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取单只股票日线数据
        
        Args:
            symbol: 股票代码，如 '000001'
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD，默认到今天
        
        Returns:
            包含以下字段的DataFrame:
            - date: 日期
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - amount: 成交额
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # akshare的股票代码格式
            if symbol.startswith('6'):
                market = "sh"
            else:
                market = "sz"
            
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            # 重命名列为统一格式
            df.columns = df.columns.str.lower()
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
            
            # 选择需要的列
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
            
            # 确保volume是数值类型（单位：手 -> 股）
            if 'volume' in df.columns:
                df['volume'] = df['volume'] * 100
            
            # 确保amount是数值类型（单位：万元 -> 元）
            if 'amount' in df.columns:
                df['amount'] = df['amount'] * 10000
            
            # 确保价格是数值类型
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 缓存
            self.cache[cache_key] = df
            
            logger.info(f"获取 {symbol} 从 {start_date} 到 {end_date} 的数据，共 {len(df)} 条")
            
            # 避免请求过快
            time.sleep(0.1)
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {symbol} 日线数据失败: {e}")
            raise
    
    def get_bulk_daily_data(
        self, 
        symbols: List[str], 
        start_date: str = "20080101",
        end_date: str = None,
        progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取多只股票日线数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            progress: 是否显示进度
        
        Returns:
            dict: {symbol: DataFrame}
        """
        results = {}
        total = len(symbols)
        
        for i, symbol in enumerate(symbols):
            try:
                df = self.get_daily_data(symbol, start_date, end_date)
                results[symbol] = df
                
                if progress and (i + 1) % 10 == 0:
                    logger.info(f"进度: {i+1}/{total}")
                    
            except Exception as e:
                logger.warning(f"获取 {symbol} 数据失败: {e}")
                continue
        
        return results
    
    def get_market_data(
        self, 
        start_date: str = "20080101", 
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取市场指数数据
        
        Returns:
            上证指数、深证成指、创业板指等
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        try:
            # 上证指数
            sh_index = ak.stock_zh_index_daily(symbol="sh000001")
            sh_index['date'] = pd.to_datetime(sh_index['date']).dt.strftime('%Y-%m-%d')
            
            # 深圳成指
            sz_index = ak.stock_zh_index_daily(symbol="sz399001")
            sz_index['date'] = pd.to_datetime(sz_index['date']).dt.strftime('%Y-%m-%d')
            
            logger.info(f"获取市场指数数据成功")
            
            return {
                'sh000001': sh_index,
                'sz399001': sz_index
            }
            
        except Exception as e:
            logger.error(f"获取市场指数数据失败: {e}")
            raise
    
    def get_trading_calender(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            交易日历DataFrame
        """
        try:
            cal = ak.tool_trade_date_hist_sina()
            cal = cal[
                (cal['trade_date'] >= start_date) & 
                (cal['trade_date'] <= end_date)
            ]
            return cal
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            raise


if __name__ == "__main__":
    fetcher = AStockDataFetcher()
    
    # 测试获取单只股票数据
    df = fetcher.get_daily_data("000001", "20240101", "20240201")
    print(df.head())
