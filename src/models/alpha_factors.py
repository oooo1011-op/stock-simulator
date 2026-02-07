"""
Alpha因子计算模块
基于Kakushadze的101个Alpha公式

保留的因子: #1-47, #49-57, #60-62, #71, #83-86, #88, #92, #95, #101
剔除: 含indneutralize的因子
"""
import pandas as pd
import numpy as np
from typing import Union

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loguru
logger = loguru.logger

# ============================================================================
# 基础辅助函数
# ============================================================================

def rank(x: Union[pd.Series, pd.DataFrame], axis: int = 0) -> Union[pd.Series, pd.DataFrame]:
    """截面排序（百分比排名）"""
    if isinstance(x, pd.DataFrame):
        return x.groupby(level='date').rank(pct=True) if axis == 0 else x.groupby(level='symbol').rank(pct=True)
    return x

def delay(x: Union[pd.Series, pd.DataFrame], d: int = 1) -> Union[pd.Series, pd.DataFrame]:
    """滞后操作"""
    return x.shift(d)

def delta(x: Union[pd.Series, pd.DataFrame], d: int = 1) -> Union[pd.Series, pd.DataFrame]:
    """差分操作"""
    return x - x.shift(d)

def correlation(x: Union[pd.Series, pd.DataFrame], y: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """滚动相关性"""
    return x.rolling(window=d).corr(y) if isinstance(x, pd.Series) else x.rolling(window=d).corr(y)

def covariance(x: Union[pd.Series, pd.DataFrame], y: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """滚动协方差"""
    return x.rolling(window=d).cov(y)

def sum_rolling(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """滚动求和"""
    return x.rolling(window=d).sum()

def product(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """滚动乘积"""
    return x.rolling(window=d).apply(lambda arr: np.prod(arr), raw=True)

def stddev(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """滚动标准差"""
    return x.rolling(window=d).std()

def ts_rank(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """时序排名"""
    def _rank(arr):
        if np.all(np.isnan(arr)): return np.nan
        valid = arr[~np.isnan(arr)]
        return (np.argsort(np.argsort(valid))[-1] + 1) / len(valid) if len(valid) > 0 else np.nan
    return x.rolling(window=d).apply(_rank, raw=False)

def ts_min(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """时序最小值"""
    return x.rolling(window=d).min()

def ts_max(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """时序最大值"""
    return x.rolling(window=d).max()

def ts_argmin(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """时序最小值位置"""
    def _argmin(arr):
        if np.all(np.isnan(arr)): return np.nan
        return np.nanargmin(arr) + 1
    return x.rolling(window=d).apply(_argmin, raw=False)

def ts_argmax(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """时序最大值位置"""
    def _argmax(arr):
        if np.all(np.isnan(arr)): return np.nan
        return np.nanargmax(arr) + 1
    return x.rolling(window=d).apply(_argmax, raw=False)

def scale(x: Union[pd.Series, pd.DataFrame], a: float = 1) -> Union[pd.Series, pd.DataFrame]:
    """归一化"""
    if isinstance(x, pd.DataFrame):
        return x.mul(a / x.abs().sum(), axis=0)
    s = x.abs().sum()
    return x * (a / s) if s > 0 and not np.isnan(s) else x

def decay_linear(x: Union[pd.Series, pd.DataFrame], d: int = 10) -> Union[pd.Series, pd.DataFrame]:
    """线性衰减加权"""
    weights = np.arange(1, d + 1) / np.sum(np.arange(1, d + 1))
    return x.rolling(window=d).apply(lambda arr: np.nansum(arr * weights[::-1]), raw=True)

def signedpower(x: Union[pd.Series, pd.DataFrame], a: float = 2) -> Union[pd.Series, pd.DataFrame]:
    """符号幂运算"""
    return np.sign(x) * np.abs(x) ** a

def sign(x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """符号函数"""
    return np.sign(x)

def abs(x: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """绝对值"""
    return np.abs(x)


# ============================================================================
# Alpha因子计算器
# ============================================================================

class AlphaCalculator:
    """Alpha因子计算器"""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache = {}
        logger.info("AlphaCalculator initialized")
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备数据"""
        df = df.copy()
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        
        df['returns'] = df.groupby('symbol')['close'].pct_change()
        df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
        df['adv20'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20).mean())
        df['cap'] = df['close'] * df['volume']
        return df.set_index(['date', 'symbol'])
    
    def calculate_all_alphas(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有Alpha因子"""
        df = self._prepare_data(df)
        alphas = {}
        methods = [m for m in dir(self) if m.startswith('alpha') and callable(getattr(self, m))]
        
        for method_name in methods:
            try:
                method = getattr(self, method_name)
                alphas[method_name] = method(df)
            except Exception as e:
                logger.warning(f"Failed {method_name}: {e}")
                alphas[method_name] = pd.Series(np.nan, index=df.index)
        
        result = pd.DataFrame(alphas)
        logger.info(f"Calculated {len(alphas)} alphas")
        return result
    
    def calculate_single_alpha(self, df: pd.DataFrame, alpha_name: str) -> pd.Series:
        """计算单个Alpha"""
        df = self._prepare_data(df)
        name = alpha_name.lower().replace('#', '').replace('alpha', '')
        method = getattr(self, f"alpha{name}", None)
        if method is None:
            raise ValueError(f"Unknown alpha: {alpha_name}")
        return method(df)

    # =========================================================================
    # Alpha因子实现
    # =========================================================================
    
    def alpha1(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
        close, returns = df['close'], df['returns']
        std_ret = stddev(returns, 20)
        x = pd.Series(np.where(returns < 0, std_ret, close), index=df.index)
        return rank(ts_argmax(signedpower(x, 2), 5)) - 0.5
    
    def alpha2(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
        return -1 * correlation(rank(delta(np.log(df['volume']), 2)), 
                               rank((df['close'] - df['open']) / df['open']), 6)
    
    def alpha3(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))"""
        return -1 * correlation(rank(df['open']), rank(df['volume']), 10)
    
    def alpha4(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#4: (-1 * Ts_Rank(rank(low), 9))"""
        return -1 * ts_rank(rank(df['low']), 9)
    
    def alpha5(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
        vwap, open_, close = df['vwap'], df['open'], df['close']
        return rank(open_ - sum_rolling(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))
    
    def alpha6(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#6: (-1 * correlation(open, volume, 10))"""
        return -1 * correlation(df['open'], df['volume'], 10)
    
    def alpha7(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"""
        close, volume, adv20 = df['close'], df['volume'], df['adv20']
        delta7 = delta(close, 7)
        cond = adv20 < volume
        return pd.Series(np.where(cond, -1 * ts_rank(abs(delta7), 60) * np.sign(delta7), -1), index=df.index)
    
    def alpha8(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
        open_, returns = df['open'], df['returns']
        prod = sum_rolling(open_, 5) * sum_rolling(returns, 5)
        return -1 * rank(prod - delay(prod, 10))
    
    def alpha9(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#9: 三元条件表达式"""
        close = df['close']
        delta1 = delta(close, 1)
        tsmin, tsmax = ts_min(delta1, 5), ts_max(delta1, 5)
        return pd.Series(np.where(tsmin > 0, delta1, np.where(tsmax < 0, delta1, -1 * delta1)), index=df.index)
    
    def alpha10(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#10: rank(...)"""
        close = df['close']
        delta1 = delta(close, 1)
        tsmin, tsmax = ts_min(delta1, 4), ts_max(delta1, 4)
        inner = np.where(tsmin > 0, delta1, np.where(tsmax < 0, delta1, -1 * delta1))
        return rank(pd.Series(inner, index=df.index))
    
    def alpha11(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
        vwap, close, volume = df['vwap'], df['close'], df['volume']
        diff = vwap - close
        return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(delta(volume, 3))
    
    def alpha12(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
        return np.sign(delta(df['volume'], 1)) * (-1 * delta(df['close'], 1))
    
    def alpha13(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))"""
        return -1 * rank(covariance(rank(df['close']), rank(df['volume']), 5))
    
    def alpha14(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
        returns = df['returns']
        return -1 * rank(delta(returns, 3)) * correlation(df['open'], df['volume'], 10)
    
    def alpha15(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
        return -1 * sum_rolling(rank(correlation(rank(df['high']), rank(df['volume']), 3)), 3)
    
    def alpha16(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))"""
        return -1 * rank(covariance(rank(df['high']), rank(df['volume']), 5))
    
    def alpha17(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
        close, volume, adv20 = df['close'], df['volume'], df['adv20']
        return (-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))
    
    def alpha18(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
        close, open_ = df['close'], df['open']
        diff = close - open_
        return -1 * rank((stddev(abs(diff), 5) + diff) + correlation(close, open_, 10))
    
    def alpha19(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
        close, returns = df['close'], df['returns']
        sign_part = -1 * np.sign((close - delay(close, 7)) + delta(close, 7))
        rank_part = 1 + rank(1 + sum_rolling(returns, 250))
        return sign_part * rank_part
    
    def alpha20(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#20: 三个rank相乘"""
        open_, high, close, low = df['open'], df['high'], df['close'], df['low']
        return (-1 * rank(open_ - delay(high, 1))) * rank(open_ - delay(close, 1)) * rank(open_ - delay(low, 1))
    
    def alpha21(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#21: 复杂条件"""
        close, volume, adv20 = df['close'], df['volume'], df['adv20']
        sum8, sum2 = sum_rolling(close, 8) / 8, sum_rolling(close, 2) / 2
        std8 = stddev(close, 8)
        vol_ratio = volume / adv20
        cond1 = sum8 + std8 < sum2
        cond2 = sum2 < sum8 - std8
        cond3 = (vol_ratio > 1) | (vol_ratio == 1)
        return pd.Series(np.where(cond1, -1, np.where(cond2, 1, np.where(cond3, 1, -1))), index=df.index)
    
    def alpha22(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
        high, volume, close = df['high'], df['volume'], df['close']
        corr = correlation(high, volume, 5)
        return -1 * delta(corr, 5) * rank(stddev(close, 20))
    
    def alpha23(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
        high = df['high']
        return pd.Series(np.where(sum_rolling(high, 20) / 20 < high, -1 * delta(high, 2), 0), index=df.index)
    
    def alpha24(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#24: 条件表达式"""
        close = df['close']
        sum100 = sum_rolling(close, 100) / 100
        ratio = delta(sum100, 100) / delay(close, 100)
        cond = (ratio < 0.05) | (ratio == 0.05)
        return pd.Series(np.where(cond, -1 * (close - ts_min(close, 100)), -1 * delta(close, 3)), index=df.index)
    
    def alpha25(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#25: rank((((-1 * returns) * adv20) * vwap) * (high - close))"""
        returns, adv20, vwap, high, close = df['returns'], df['adv20'], df['vwap'], df['high'], df['close']
        return rank(((-1 * returns) * adv20) * vwap * (high - close))
    
    def alpha26(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
        volume, high = df['volume'], df['high']
        corr = correlation(ts_rank(volume, 5), ts_rank(high, 5), 5)
        return -1 * ts_max(corr, 3)
    
    def alpha27(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#27: ((0.5 < rank(...)) ? (-1 * 1) : 1)"""
        volume, vwap = df['volume'], df['vwap']
        avg = sum_rolling(correlation(rank(volume), rank(vwap), 6), 2) / 2.0
        return pd.Series(np.where(rank(avg) > 0.5, -1, 1), index=df.index)
    
    def alpha28(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
        adv20, low, high, close = df['adv20'], df['low'], df['high'], df['close']
        return scale((correlation(adv20, low, 5) + (high + low) / 2) - close)
    
    def alpha29(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#29: 复杂公式"""
        close, returns = df['close'], df['returns']
        rank3 = product(rank(rank(scale(np.log(ts_min(rank(rank(delta(close - 1, 5))), 2))))), 1)
        return ts_min(rank3, 5) + ts_rank(delay(-1 * returns, 6), 5)
    
    def alpha30(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#30: (((1.0 - rank(...)) * sum(volume, 5)) / sum(volume, 20))"""
        close, volume = df['close'], df['volume']
        signs = np.sign(close - delay(close, 1)) + np.sign(delay(close, 1) - delay(close, 2)) + np.sign(delay(close, 2) - delay(close, 3))
        return (1.0 - rank(signs)) * sum_rolling(volume, 5) / sum_rolling(volume, 20)

    def alpha31(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#31: 复杂组合"""
        close, adv20, low = df['close'], df['adv20'], df['low']
        delta_close = delta(close, 10)
        rank1 = rank(delta_close)
        rank2 = rank(rank1)
        decay = decay_linear(-1 * rank2, 10)
        rank3 = rank(decay)
        rank4 = rank(rank3)
        rank_delta = rank(-1 * delta(close, 3))
        corr = correlation(adv20, low, 12)
        scale_corr = scale(corr)
        return rank4 + rank_delta + np.sign(scale_corr)
    
    def alpha32(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
        close, vwap = df['close'], df['vwap']
        scale1 = scale(sum_rolling(close, 7) / 7 - close)
        scale2 = 20 * scale(correlation(vwap, delay(close, 5), 230))
        return scale1 + scale2
    
    def alpha33(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#33: rank((-1 * ((1 - (open / close))^1)))"""
        open_, close = df['open'], df['close']
        return -1 * rank(1 - open_ / close)
    
    def alpha34(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
        returns, close = df['returns'], df['close']
        return (1 - rank(stddev(returns, 2) / stddev(returns, 5))) + (1 - rank(delta(close, 1)))
    
    def alpha35(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
        volume, close, high, low, returns = df['volume'], df['close'], df['high'], df['low'], df['returns']
        return (ts_rank(volume, 32) * (1 - ts_rank((close + high) - low, 16)) * (1 - ts_rank(returns, 32)))
    
    def alpha36(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#36: 复杂组合"""
        close, open_, volume, returns, vwap, adv20 = df['close'], df['open'], df['volume'], df['returns'], df['vwap'], df['adv20']
        term1 = 2.21 * rank(correlation(close - open_, delay(volume, 1), 15))
        term2 = 0.7 * rank(open_ - close)
        term3 = 0.73 * rank(ts_rank(delay(-1 * returns, 6), 5))
        term4 = rank(abs(correlation(vwap, adv20, 6)))
        term5 = 0.6 * rank((((sum_rolling(close, 200) / 200) - open_) * (close - open_)))
        return term1 + term2 + term3 + term4 + term5
    
    def alpha37(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
        open_, close = df['open'], df['close']
        return rank(correlation(delay(open_ - close, 1), close, 200)) + rank(open_ - close)
    
    def alpha38(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"""
        close, open_ = df['close'], df['open']
        return -1 * rank(ts_rank(close, 10)) * rank(close / open_)
    
    def alpha39(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
        close, volume, adv20, returns = df['close'], df['volume'], df['adv20'], df['returns']
        decay = decay_linear(volume / adv20, 9)
        return -1 * rank(delta(close, 7) * (1 - rank(decay))) * (1 + rank(sum_rolling(returns, 250)))
    
    def alpha40(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))"""
        high, volume = df['high'], df['volume']
        return -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)
    
    def alpha41(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#41: (((high * low)^0.5) - vwap)"""
        high, low, vwap = df['high'], df['low'], df['vwap']
        return np.sqrt(high * low) - vwap
    
    def alpha42(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#42: (rank((vwap - close)) / rank((vwap + close)))"""
        vwap, close = df['vwap'], df['close']
        return rank(vwap - close) / rank(vwap + close)
    
    def alpha43(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
        volume, adv20, close = df['volume'], df['adv20'], df['close']
        return ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)
    
    def alpha44(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#44: (-1 * correlation(high, rank(volume), 5))"""
        high, volume = df['high'], df['volume']
        return -1 * correlation(high, rank(volume), 5)
    
    def alpha45(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
        close, volume = df['close'], df['volume']
        sum_close_5 = sum_rolling(close, 5)
        sum_close_20 = sum_rolling(close, 20)
        delay_sum = delay(sum_close_5, 5)
        return -1 * (rank(sum_rolling(delay_sum, 20) / 20) * correlation(close, volume, 2) * 
                    rank(correlation(sum_close_5, sum_close_20, 2)))
    
    def alpha46(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#46: 条件表达式"""
        close = df['close']
        d20, d10 = delay(close, 20), delay(close, 10)
        slope = (d20 - d10) / 10 - (d10 - close) / 10
        cond1 = slope > 0.25
        cond2 = slope < 0
        return pd.Series(np.where(cond1, -1, np.where(cond2, 1, -1 * delta(close, 1))), index=df.index)
    
    def alpha47(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#47: 复杂公式"""
        close, volume, high, vwap, adv20 = df['close'], df['volume'], df['high'], df['vwap'], df['adv20']
        return ((rank(rank(1 / close)) * volume / adv20) * 
                (high * rank(high - close) / (sum_rolling(high, 5) / 5)) - 
                rank(vwap - delay(vwap, 5)))
    
    # Skip alpha48 (has indneutralize)
    
    def alpha49(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#49: 条件表达式"""
        close = df['close']
        d20, d10 = delay(close, 20), delay(close, 10)
        slope = (d20 - d10) / 10 - (d10 - close) / 10
        cond = slope < -0.1
        return pd.Series(np.where(cond, 1, -1 * delta(close, 1)), index=df.index)
    
    def alpha50(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
        volume, vwap = df['volume'], df['vwap']
        corr = correlation(rank(volume), rank(vwap), 5)
        return -1 * ts_max(rank(corr), 5)
    
    def alpha51(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#51: 条件表达式"""
        close = df['close']
        d20, d10 = delay(close, 20), delay(close, 10)
        slope = (d20 - d10) / 10 - (d10 - close) / 10
        cond = slope < -0.05
        return pd.Series(np.where(cond, 1, -1 * delta(close, 1)), index=df.index)
    
    def alpha52(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
        low, close, returns, volume = df['low'], df['close'], df['returns'], df['volume']
        min_low = ts_min(low, 5)
        return ((-1 * min_low + delay(min_low, 5)) * 
                rank((sum_rolling(returns, 240) - sum_rolling(returns, 20)) / 220) * 
                ts_rank(volume, 5))
    
    def alpha53(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
        close, low, high = df['close'], df['low'], df['high']
        inner = ((close - low) - (high - close)) / (close - low)
        return -1 * delta(inner, 9)
    
    def alpha54(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
        low, close, high, open_ = df['low'], df['close'], df['high'], df['open']
        return -1 * ((low - close) * (open_ ** 5)) / ((low - high) * (close ** 5))
    
    def alpha55(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
        close, low, high, volume = df['close'], df['low'], df['high'], df['volume']
        normalized = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))
        return -1 * correlation(rank(normalized), rank(volume), 6)
    
    def alpha56(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""
        returns, cap = df['returns'], df['cap']
        return 0 - (1 * (rank(sum_rolling(returns, 10) / sum_rolling(sum_rolling(returns, 2), 3)) * rank(returns * cap)))
    
    def alpha57(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
        close, vwap = df['close'], df['vwap']
        decay = decay_linear(rank(ts_argmax(close, 30)), 2)
        return 0 - ((close - vwap) / decay)

    # Skip alpha58-59 (have indneutralize)
    
    def alpha60(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10)))))"""
        close, low, high, volume = df['close'], df['low'], df['high'], df['volume']
        inner = ((close - low) - (high - close)) / (high - low) * volume
        return 0 - (2 * scale(rank(inner)) - scale(rank(ts_argmax(close, 10))))
    
    def alpha61(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))"""
        vwap, adv180 = df['vwap'], df['adv20']  # Use adv20 as proxy
        return (rank(vwap - ts_min(vwap, 16)) < rank(correlation(vwap, adv180, 18))).astype(int) * 2 - 1
    
    def alpha62(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#62: 复杂条件"""
        vwap, open_, high, low, adv20 = df['vwap'], df['open'], df['high'], df['low'], df['adv20']
        return ((rank(correlation(vwap, sum_rolling(adv20, 22), 10)) < 
                rank((rank(open_) + rank(open_) < rank((high + low) / 2) + rank(high)))).astype(int) * 2 - 1)
    
    # Skip alpha63-70 (have indneutralize)
    
    def alpha71(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#71: max(Ts_Rank(...), Ts_Rank(...))"""
        close, adv180, vwap, low, open_ = df['close'], df['adv20'], df['vwap'], df['low'], df['open']
        term1 = ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180, 12), 18), 4), 15)
        inner = ((low + open_) - (vwap + vwap)) ** 2
        term2 = ts_rank(decay_linear(rank(inner), 16), 4)
        return pd.Series(np.maximum(term1, term2), index=df.index)
    
    # Skip alpha72-82 (have indneutralize)
    
    def alpha83(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close))"""
        high, low, close, volume, vwap = df['high'], df['low'], df['close'], df['volume'], df['vwap']
        range_ = (high - low) / (sum_rolling(close, 5) / 5)
        return (rank(delay(range_, 2)) * rank(rank(volume))) / (range_ / (vwap - close))
    
    def alpha84(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))"""
        vwap, close = df['vwap'], df['close']
        return signedpower(ts_rank(vwap - ts_max(vwap, 15), 20), delta(close, 5))
    
    def alpha85(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))"""
        high, close, adv30, volume = df['high'], df['close'], df['adv20'], df['volume']  # Use adv20
        combined = high * 0.876703 + close * 0.123297
        term1 = rank(correlation(combined, adv30, 10))
        term2 = correlation(ts_rank((high + low) / 2, 4), ts_rank(volume, 10), 7)
        return signedpower(term1, term2)
    
    def alpha86(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#86: 条件"""
        close, open_, vwap, adv20 = df['close'], df['open'], df['vwap'], df['adv20']
        cond = ts_rank(correlation(close, sum_rolling(adv20, 15), 6), 20) < rank((open_ + close) - (vwap + open_))
        return cond.astype(int) * 2 - 1
    
    # Skip alpha87 (has indneutralize)
    
    def alpha88(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#88: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))"""
        high, low, vwap, adv40 = df['high'], df['low'], df['vwap'], df['adv20']
        term1 = rank(decay_linear((high + low) / 2 + high - (vwap + high), 20))
        term2 = rank(decay_linear(correlation((high + low) / 2, adv40, 3), 6))
        return pd.Series(np.minimum(term1, term2), index=df.index)
    
    # Skip alpha89-94, 96-100 (have indneutralize)
    
    def alpha92(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))"""
        high, low, close, open_, adv30 = df['high'], df['low'], df['close'], df['open'], df['adv20']
        cond1 = ((high + low) / 2 + close) < (low + open_)
        term1 = ts_rank(decay_linear(cond1, 15), 19)
        term2 = ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7), 7)
        return pd.Series(np.minimum(term1, term2), index=df.index)
    
    def alpha95(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))"""
        open_, high, low, adv40 = df['open'], df['high'], df['low'], df['adv20']
        rank1 = rank(open_ - ts_min(open_, 12))
        combined = (high + low) / 2
        corr = correlation(sum_rolling(combined, 19), sum_rolling(adv40, 19), 13)
        rank2 = ts_rank(rank(corr) ** 5, 12)
        return (rank1 < rank2).astype(int) * 2 - 1
    
    # Skip alpha96-100 (have indneutralize)
    
    def alpha101(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#101: ((close - open) / ((high - low) + .001))"""
        close, open_, high, low = df['close'], df['open'], df['high'], df['low']
        return (close - open_) / ((high - low) + 0.001)

    def alpha64(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404)), 3.69741))) * -1"""
        open_, low, high, vwap, adv20 = df['open'], df['low'], df['high'], df['vwap'], df['adv20']
        combined1 = open_ * 0.178404 + low * 0.821596
        combined2 = ((high + low) / 2) * 0.178404 + vwap * 0.821596
        cond = rank(correlation(sum_rolling(combined1, 13), sum_rolling(adv20, 13), 17)) < rank(delta(combined2, 4))
        return cond.astype(int) * 2 - 1
    
    def alpha65(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1"""
        open_, vwap, adv20 = df['open'], df['vwap'], df['adv20']
        combined = open_ * 0.00817205 + vwap * 0.991828
        cond = rank(correlation(combined, sum_rolling(adv20, 9), 6)) < rank(open_ - ts_min(open_, 14))
        return cond.astype(int) * 2 - 1
    
    def alpha66(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1"""
        low, vwap, open_, high = df['low'], df['vwap'], df['open'], df['high']
        decay1 = decay_linear(delta(vwap, 4), 7)
        inner = ((low * 0.96633 + low * 0.03367) - vwap) / (open_ - (high + low) / 2)
        decay2 = ts_rank(decay_linear(inner, 11), 7)
        return (rank(decay1) + decay2) * -1
    
    def alpha68(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1"""
        high, low, close, adv20 = df['high'], df['low'], df['close'], df['adv20']
        combined = close * 0.518371 + low * 0.481629
        cond = ts_rank(correlation(rank(high), rank(adv20), 9), 14) < rank(delta(combined, 1))
        return cond.astype(int) * 2 - 1
    
    def alpha72(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011))"""
        high, low, vwap, volume, adv20 = df['high'], df['low'], df['vwap'], df['volume'], df['adv20']
        mid = (high + low) / 2
        num = rank(decay_linear(correlation(mid, adv20, 9), 10))
        denom = rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))
        return num / denom
    
    def alpha73(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1"""
        vwap, open_, low = df['vwap'], df['open'], df['low']
        delta_vwap = delta(vwap, 5)
        term1 = rank(decay_linear(delta_vwap, 3))
        combined = open_ * 0.147155 + low * 0.852845
        delta_combined = delta(combined, 2)
        term2 = ts_rank(decay_linear(-1 * delta_combined / combined, 3), 17)
        return (pd.Series(np.maximum(term1, term2), index=df.index) * -1)
    
    def alpha75(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))"""
        vwap, volume, low, adv20 = df['vwap'], df['volume'], df['low'], df['adv20']
        cond = rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv20), 12))
        return cond.astype(int) * 2 - 1
    
    def alpha76(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1"""
        vwap, low, adv20 = df['vwap'], df['low'], df['adv20']
        term1 = rank(decay_linear(delta(vwap, 1), 12))
        inner = correlation(low, adv20, 8)
        term2 = ts_rank(decay_linear(ts_rank(inner, 20), 17), 19)
        return pd.Series(np.maximum(term1, term2), index=df.index) * -1
    
    def alpha77(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))"""
        high, low, vwap, adv20 = df['high'], df['low'], df['vwap'], df['adv20']
        mid = (high + low) / 2
        inner1 = mid + high - (vwap + high)
        term1 = rank(decay_linear(inner1, 20))
        term2 = rank(decay_linear(correlation(mid, adv20, 3), 6))
        return pd.Series(np.minimum(term1, term2), index=df.index)
    
    def alpha78(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))"""
        low, vwap, adv20, volume = df['low'], df['vwap'], df['adv20'], df['volume']
        combined = low * 0.352233 + vwap * 0.647767
        term1 = rank(correlation(sum_rolling(combined, 20), sum_rolling(adv20, 20), 7))
        term2 = rank(correlation(rank(vwap), rank(volume), 6))
        return signedpower(term1, term2)
    
    def alpha81(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1"""
        vwap, volume, adv20 = df['vwap'], df['volume'], df['adv20']
        inner = correlation(vwap, sum_rolling(adv20, 50), 8)
        prod = product(rank(rank(inner) ** 4), 15)
        term1 = rank(np.log(prod))
        term2 = rank(correlation(rank(vwap), rank(volume), 5))
        cond = term1 < term2
        return cond.astype(int) * 2 - 1
    
    def alpha82(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1"""
        open_, volume = df['open'], df['volume']
        term1 = rank(decay_linear(delta(open_, 1), 15))
        inner = volume * 0.634196 + open_ * 0.365804
        term2 = ts_rank(decay_linear(correlation(volume, inner, 17), 7), 13)
        return pd.Series(np.minimum(term1, term2), index=df.index) * -1
    
    def alpha89(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))"""
        low, adv20, vwap = df['low'], df['adv20'], df['vwap']
        combined = low * 0.967285 + low * 0.032715
        term1 = ts_rank(decay_linear(correlation(combined, adv20, 7), 6), 4)
        term2 = ts_rank(decay_linear(delta(vwap, 3), 10), 15)
        return term1 - term2
    
    def alpha94(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1"""
        vwap, adv20 = df['vwap'], df['adv20']
        term1 = rank(vwap - ts_min(vwap, 12))
        term2 = ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv20, 4), 18), 3)
        return signedpower(term1, term2) * -1
    
    def alpha96(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1"""
        vwap, volume, close, adv20 = df['vwap'], df['volume'], df['close'], df['adv20']
        term1 = ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 4), 4))
        inner = ts_argmax(correlation(ts_rank(close, 7), ts_rank(adv20, 4), 4), 13)
        term2 = ts_rank(decay_linear(inner, 14), 13)
        return pd.Series(np.maximum(term1, term2), index=df.index) * -1
    
    def alpha99(self, df: pd.DataFrame) -> pd.Series:
        """Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1"""
        high, low, volume, adv20 = df['high'], df['low'], df['volume'], df['adv20']
        mid = (high + low) / 2
        cond = rank(correlation(sum_rolling(mid, 20), sum_rolling(adv20, 20), 9)) < rank(correlation(low, volume, 6))
        return cond.astype(int) * 2 - 1
