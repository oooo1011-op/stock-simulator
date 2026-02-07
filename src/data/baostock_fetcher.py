"""
baostock 数据获取器
备选数据源，当akshare不可用时使用
"""
import baostock as bs
import pandas as pd
from loguru import logger

# 日期格式映射
DATE_FORMAT_MAP = {
    "20240101": "2024-01-01",
    "20240630": "2024-06-30",
}

# adjustflag映射
ADJUST_MAP = {
    "qfq": "2",  # 前复权
    "hfq": "1",  # 后复权
    "": "3",     # 不复权
}


def convert_date(date_str: str) -> str:
    """转换日期格式 20240101 -> 2024-01-01"""
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str


class BaostockDataFetcher:
    def __init__(self):
        self.lg = None
        
    def _login(self):
        """登录baostock"""
        if self.lg is None:
            self.lg = bs.login()
            if self.lg.error_code != '0':
                raise ConnectionError(f"Baostock login failed: {self.lg.error_msg}")
            logger.info("Baostock login success")
    
    def _logout(self):
        """登出"""
        if self.lg:
            bs.logout()
            self.lg = None
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表"""
        self._login()
        
        # 常见股票代码列表（从行业数据获取）
        common_stocks = []
        
        # 获取金融行业
        rs = bs.query_stock_industry()
        while (rs.error_code == '0') and rs.next():
            row = rs.get_row_data()
            if row[1].startswith('sh.') or row[1].startswith('sz.'):
                code = row[1].split('.')[-1]
                if code not in common_stocks:
                    common_stocks.append(code)
        
        df = pd.DataFrame({
            'symbol': common_stocks,
            'name': [f"stock_{c}" for c in common_stocks]
        })
        
        logger.info(f"获取 {len(df)} 只股票")
        return df
    
    def get_daily_data(self, symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            symbol: 股票代码 (如 '600000', 'sh.600000', 'sz.000001')
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            adjust: 复权类型 'qfq'前复权 'hfq'后复权 ''不复权
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
        """
        self._login()
        
        # 处理各种股票代码格式
        if '.' in symbol:
            # 已经是 sh.xxxxxx 或 sz.xxxxxx 格式
            bs_symbol = symbol
        else:
            # 纯数字代码，需要添加前缀
            if symbol.startswith('6'):
                bs_symbol = f"sh.{symbol}"
            else:
                bs_symbol = f"sz.{symbol}"
        
        # 转换日期格式
        bs_start = convert_date(start_date)
        bs_end = convert_date(end_date)
        adj_flag = ADJUST_MAP.get(adjust, "2")
        
        rs = bs.query_history_k_data_plus(
            bs_symbol,
            "date,open,high,low,close,volume,amount",
            start_date=bs_start,
            end_date=bs_end,
            frequency="d",
            adjustflag=adj_flag
        )
        
        if rs.error_code != '0':
            logger.warning(f"{symbol}: baostock error - {rs.error_msg}")
            return pd.DataFrame()
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 数据清洗
        df = df[df['close'] != '']  # 剔除空值
        
        # 转换类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_cols)
        
        # 计算change和pct_change
        df['pre_close'] = df['close'].shift(1)
        df['change'] = df['close'] - df['pre_close']
        df['pct_change'] = df['change'] / df['pre_close'] * 100
        
        # 修正单位
        # baostock volume 是 股，转为手
        # baostock amount 是 元，转为千元
        df['amount'] = df['amount'] / 1000  # 元 -> 千元
        
        # 保持date为列，不设置索引
        df['date'] = pd.to_datetime(df['date'])
        df['symbol'] = symbol.split('.')[-1]  # 去掉sh./sz.前缀
        
        logger.debug(f"{symbol}: {len(df)} 条数据")
        return df
    
    def __del__(self):
        self._logout()


if __name__ == "__main__":
    fetcher = BaostockDataFetcher()
    
    # 测试
    df = fetcher.get_daily_data("600000", "20240101", "20240630")
    print(f"\n浦发银行: {len(df)} 条数据")
    if not df.empty:
        print(df.head())
    
    fetcher._logout()
