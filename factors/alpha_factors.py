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
                         roe_data: Optional[pd.DataFrame] = None,
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
            if hasattr(self, f'calculate_{factor_name}'):
                factor_func = getattr(self, f'calculate_{factor_name}')
                # 检查函数参数，智能传递参数
                import inspect
                sig = inspect.signature(factor_func)
                param_names = list(sig.parameters.keys())[1:]  # 去掉 self 参数
                
                # 根据参数名决定传递哪些参数
                kwargs = {}
                if 'price_data' in param_names:
                    kwargs['price_data'] = price_data
                if 'volume_data' in param_names and volume_data is not None:
                    kwargs['volume_data'] = volume_data
                if 'roe_data' in param_names and roe_data is not None:
                    kwargs['roe_data'] = roe_data
                
                # 按位置传递参数（确保兼容性）
                if len(param_names) >= 3:
                    factor_data = factor_func(price_data, volume_data, roe_data)
                elif len(param_names) >= 2:
                    factor_data = factor_func(price_data, volume_data)
                else:
                    factor_data = factor_func(price_data)
                    
                factor_results[factor_name] = factor_data
            else:
                logger.warning(f"Alpha因子计算方法不存在: calculate_{factor_name}")

        if factor_results:
            return pd.concat(factor_results, axis=1)
        else:
            # 返回空DataFrame，保持与输入数据相同的索引
            return pd.DataFrame(index=price_data.index)

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

    def calculate_momentum_20d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算20日动量因子"""
        return price_data.pct_change(periods=20).fillna(0)

    def calculate_momentum_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算60日动量因子"""
        return price_data.pct_change(periods=60).fillna(0)

    def calculate_ma_ratio_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算60日均线比率"""
        ma_60 = price_data.rolling(window=60).mean()
        return (price_data / ma_60 - 1).fillna(0)

    def calculate_price_volume_correlation(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
        """计算价格成交量相关性因子"""
        if volume_data is None:
            return pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
        price_change = price_data.pct_change()
        volume_change = volume_data.pct_change()
        return price_change.rolling(window=20).corr(volume_change).fillna(0)

    def calculate_mean_reversion_5d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算5日均值回归因子"""
        returns = price_data.pct_change()
        # 计算5日累计收益率的负值作为均值回归信号
        cumulative_5d = (1 + returns).rolling(window=5).apply(lambda x: x.prod()) - 1
        return -cumulative_5d.fillna(0)

    def calculate_trend_strength(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算趋势强度因子"""
        # 使用线性回归斜率衡量趋势强度
        def calc_trend_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope / np.mean(y) if np.mean(y) != 0 else 0
            except:
                return 0
        
        trend_strength = price_data.rolling(window=20).apply(calc_trend_slope)
        return trend_strength.fillna(0)

    def calculate_volume_momentum(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量动量因子"""
        if volume_data is None:
            return pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
        
        # 计算成交量的动量（20日成交量变化率）
        volume_ma_short = volume_data.rolling(window=5).mean()
        volume_ma_long = volume_data.rolling(window=20).mean()
        volume_momentum = (volume_ma_short / volume_ma_long - 1).fillna(0)
        
        return volume_momentum

    def calculate_sharpe_ratio_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算60日夏普比率因子"""
        returns = price_data.pct_change()
        mean_return_60d = returns.rolling(window=60).mean() * 252
        std_dev_60d = returns.rolling(window=60).std() * np.sqrt(252)
        sharpe_ratio = mean_return_60d / std_dev_60d
        return sharpe_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)

    def calculate_composite_alpha(self, price_data: pd.DataFrame, volume_data=None) -> pd.DataFrame:
        """计算复合Alpha因子"""
        momentum_60d = self.calculate_momentum_60d(price_data)
        price_reversal = self.calculate_price_reversal(price_data)
        
        # 简单的线性组合
        composite_factor = 0.7 * momentum_60d + 0.3 * price_reversal
        return composite_factor.fillna(0)

    def calculate_roe_factor(self, price_data: pd.DataFrame, volume_data=None, roe_data: pd.DataFrame=None) -> pd.DataFrame:
        """计算ROE因子"""
        if roe_data is None:
            return pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
        return roe_data.fillna(0)
