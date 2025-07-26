"""
风险因子计算模块
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RiskFactors:
    """风险因子计算器"""

    def __init__(self, config: Dict):
        self.config = config

    def calculate_factors(self,
                         price_data: pd.DataFrame,
                         volume_data: Optional[pd.DataFrame] = None,
                         factors: List[str] = None) -> pd.DataFrame:
        """
        计算风险因子

        Args:
            price_data: 价格数据 (DataFrame with DatetimeIndex)
            volume_data: 成交量数据 (DataFrame with DatetimeIndex)
            factors: 指定计算的因子列表

        Returns:
            风险因子数据 (DataFrame with DatetimeIndex)
        """
        factor_results = {}

        if factors is None:
            factors = ['volatility_60d', 'volume_ratio', 'turnover_rate', 'beta_60d']

        for factor_name in factors:
            if hasattr(self, f'calculate_{factor_name}'):
                factor_func = getattr(self, f'calculate_{factor_name}')
                factor_data = factor_func(price_data, volume_data)
                factor_results[factor_name] = factor_data
            else:
                logger.warning(f"风险因子计算方法不存在: calculate_{factor_name}")

        if factor_results:
            return pd.concat(factor_results, axis=1)
        else:
            # 返回空DataFrame，保持与输入数据相同的索引
            return pd.DataFrame(index=price_data.index)

    def calculate_volatility_20d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算20日波动率"""
        returns = price_data.pct_change()
        return returns.rolling(window=20).std().fillna(0) * np.sqrt(252)

    def calculate_volatility_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算60日波动率"""
        returns = price_data.pct_change()
        return returns.rolling(window=60).std().fillna(0) * np.sqrt(252)

    def calculate_volume_ratio(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量比率"""
        if volume_data is None:
            return pd.DataFrame(1.0, index=price_data.index, columns=price_data.columns)
        volume_ma = volume_data.rolling(window=20).mean()
        return (volume_data / volume_ma).fillna(1.0)

    def calculate_turnover_rate(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
        """计算换手率"""
        if volume_data is None:
            return pd.DataFrame(0.1, index=price_data.index, columns=price_data.columns)
        volume_mean = volume_data.rolling(window=20).mean()
        volume_std = volume_data.rolling(window=20).std()
        return (volume_std / volume_mean).fillna(0.1)

    def calculate_beta_60d(self, price_data: pd.DataFrame, volume_data=None,
                          benchmark_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """计算60日Beta"""
        returns = price_data.pct_change()
        market_returns = returns.mean(axis=1)
        rolling_cov = returns.rolling(window=60).cov(market_returns)
        rolling_var = market_returns.rolling(window=60).var()
        return (rolling_cov.T / rolling_var).T.fillna(1.0)

    def calculate_skewness_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算60日收益率偏度"""
        returns = price_data.pct_change()
        return returns.rolling(window=60).skew().fillna(0)

    def calculate_kurtosis_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算60日收益率峰度"""
        returns = price_data.pct_change()
        return returns.rolling(window=60).kurt().fillna(0)

    def calculate_max_drawdown_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算60日最大回撤"""
        cumulative_returns = (1 + price_data.pct_change()).cumprod()
        rolling_max = cumulative_returns.rolling(window=60, min_periods=1).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.rolling(window=60).min().fillna(0)

    def calculate_var_95(self, price_data: pd.DataFrame, volume_data=None, window: int = 60) -> pd.DataFrame:
        """计算95% VaR"""
        returns = price_data.pct_change()
        return returns.rolling(window=window).quantile(0.05).fillna(0)

    def calculate_cvar_95(self, price_data: pd.DataFrame, volume_data=None, window: int = 60) -> pd.DataFrame:
        """计算95% CVaR条件风险价值"""
        returns = price_data.pct_change()
        var_95 = returns.rolling(window=window).quantile(0.05)
        return returns[returns <= var_95].rolling(window=window).mean().fillna(0)

    def calculate_downside_deviation(self, price_data: pd.DataFrame, volume_data=None,
                                   target_return: float = 0, window: int = 60) -> pd.DataFrame:
        """计算下行偏差"""
        returns = price_data.pct_change()
        downside_returns = returns[returns < target_return]
        return downside_returns.rolling(window=window).std().fillna(0) * np.sqrt(252)

    def calculate_upside_deviation(self, price_data: pd.DataFrame, volume_data=None,
                                 target_return: float = 0, window: int = 60) -> pd.DataFrame:
        """计算上行偏差"""
        returns = price_data.pct_change()
        upside_returns = returns[returns > target_return]
        return upside_returns.rolling(window=window).std().fillna(0) * np.sqrt(252)

    def calculate_volatility_regime(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算波动率状态因子"""
        returns = price_data.pct_change()
        short_vol = returns.rolling(window=20).std()
        long_vol = returns.rolling(window=60).std()
        return (short_vol / long_vol).fillna(1.0)
