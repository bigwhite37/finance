"""
Alpha因子计算模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AlphaFactors:
    """Alpha因子计算器"""

    def __init__(self, config: Dict):
        self.config = config

    def calculate_factors(self,
                         price_data: pd.DataFrame,
                         volume_data: Optional[pd.DataFrame] = None,
                         factors: List[str] = None) -> pd.DataFrame:
        """
        计算Alpha因子

        Args:
            price_data: 价格数据 (DataFrame with DatetimeIndex)
            volume_data: 成交量数据 (DataFrame with DatetimeIndex)
            factors: 指定计算的因子列表

        Returns:
            Alpha因子数据 (DataFrame with DatetimeIndex)
        """
        factor_results = {}

        if factors is None:
            factors = ['return_20d', 'return_60d', 'price_momentum', 'rsi_14d', 'ma_ratio_20d']

        for factor_name in factors:
            factor_func = getattr(self, f'calculate_{factor_name}')
            factor_data = factor_func(price_data, volume_data)
            factor_results[factor_name] = factor_data

        return pd.concat(factor_results, axis=1)

    def calculate_return_5d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算5日收益率"""
        return price_data.pct_change(periods=5).fillna(0)

    def calculate_return_20d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算20日收益率"""
        return price_data.pct_change(periods=20).fillna(0)

    def calculate_return_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算60日收益率"""
        return price_data.pct_change(periods=60).fillna(0)

    def calculate_price_momentum(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算价格动量因子 (12个月收益率，排除最近1个月)"""
        return price_data.pct_change(periods=232).fillna(0)

    def calculate_rsi_14d(self, price_data: pd.DataFrame, volume_data=None, window: int = 14) -> pd.DataFrame:
        """计算RSI相对强弱指标"""
        delta = price_data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_ma_ratio_10d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算10日均线比率"""
        ma_10 = price_data.rolling(window=10).mean()
        return (price_data / ma_10 - 1).fillna(0)

    def calculate_ma_ratio_20d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算20日均线比率"""
        ma_20 = price_data.rolling(window=20).mean()
        return (price_data / ma_20 - 1).fillna(0)

    def calculate_price_reversal(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算短期价格反转因子"""
        return -price_data.pct_change(periods=5).fillna(0)

    def calculate_volume_price_trend(self,
                                   price_data: pd.DataFrame,
                                   volume_data: pd.DataFrame) -> pd.DataFrame:
        """计算量价趋势因子"""
        if volume_data is None:
            return pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
        price_change = price_data.pct_change()
        volume_change = volume_data.pct_change()
        return price_change.rolling(window=20).corr(volume_change).fillna(0)

    def calculate_bollinger_position(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算布林带位置因子"""
        window = 20
        sma = price_data.rolling(window=window).mean()
        std = price_data.rolling(window=window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        position = (price_data - lower_band) / (upper_band - lower_band)
        return position.clip(0, 1).fillna(0.5)

    def calculate_williams_r(self, price_data: pd.DataFrame, volume_data=None, window: int = 14) -> pd.DataFrame:
        """计算威廉指标"""
        highest = price_data.rolling(window=window).max()
        lowest = price_data.rolling(window=window).min()
        williams_r = (highest - price_data) / (highest - lowest) * -100
        return williams_r.fillna(-50)

    def calculate_stochastic_k(self, price_data: pd.DataFrame, volume_data=None, window: int = 14) -> pd.DataFrame:
        """计算随机指标K值"""
        highest = price_data.rolling(window=window).max()
        lowest = price_data.rolling(window=window).min()
        k_value = (price_data - lowest) / (highest - lowest) * 100
        return k_value.fillna(50)

    def calculate_momentum_quality(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算动量质量因子"""
        mom_5d = price_data.pct_change(periods=5)
        mom_20d = price_data.pct_change(periods=20)
        mom_60d = price_data.pct_change(periods=60)
        momentum_consistency = ((mom_5d > 0).astype(int) +
                                (mom_20d > 0).astype(int) +
                                (mom_60d > 0).astype(int)) / 3
        momentum_strength = (abs(mom_5d) + abs(mom_20d) + abs(mom_60d)) / 3
        return (momentum_consistency * momentum_strength).fillna(0)
