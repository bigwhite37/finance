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
            price_data: 价格数据
            volume_data: 成交量数据
            factors: 指定计算的因子列表
            
        Returns:
            Alpha因子数据
        """
        factor_results = {}
        
        if factors is None:
            factors = ['return_20d', 'return_60d', 'price_momentum', 'rsi_14d', 'ma_ratio_20d']
        
        for factor_name in factors:
            factor_func = getattr(self, f'calculate_{factor_name}')
            factor_data = factor_func(price_data, volume_data)
            factor_results[factor_name] = factor_data
        
        return pd.DataFrame(factor_results)
    
    def calculate_return_20d(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算20日收益率"""
        return price_data.pct_change(periods=20).iloc[-1]
    
    def calculate_return_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算60日收益率"""
        return price_data.pct_change(periods=60).iloc[-1]
    
    def calculate_price_momentum(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算价格动量因子"""
        # 使用过去12个月收益率，排除最近1个月
        if len(price_data) < 252:
            window = len(price_data) - 20 if len(price_data) > 20 else len(price_data)
        else:
            window = 232  # 12个月 - 1个月
            
        long_return = price_data.pct_change(periods=window)
        return long_return.iloc[-1]
    
    def calculate_rsi_14d(self, price_data: pd.DataFrame, volume_data=None, window: int = 14) -> pd.Series:
        """计算RSI相对强弱指标"""
        def rsi_single(prices):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        
        return price_data.apply(rsi_single)
    
    def calculate_ma_ratio_20d(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算20日均线比率"""
        ma_20 = price_data.rolling(window=20).mean()
        current_price = price_data.iloc[-1]
        return current_price / ma_20.iloc[-1] - 1
    
    def calculate_price_reversal(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算短期价格反转因子"""
        # 使用过去5日收益率的负值作为反转因子
        short_return = price_data.pct_change(periods=5).iloc[-1]
        return -short_return
    
    def calculate_volume_price_trend(self, 
                                   price_data: pd.DataFrame, 
                                   volume_data: pd.DataFrame) -> pd.Series:
        """计算量价趋势因子"""
        if volume_data is None:
            return pd.Series(index=price_data.columns, dtype=float)
        
        # 价格变化
        price_change = price_data.pct_change()
        
        # 成交量变化
        volume_change = volume_data.pct_change()
        
        # 计算量价相关性
        correlation = price_change.rolling(window=20).corr(volume_change.rolling(window=20))
        
        return correlation.iloc[-1] if not correlation.empty else pd.Series(index=price_data.columns, dtype=float)
    
    def calculate_bollinger_position(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算布林带位置因子"""
        window = 20
        std_window = 20
        
        # 计算布林带
        sma = price_data.rolling(window=window).mean()
        std = price_data.rolling(window=std_window).std()
        
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        # 计算当前价格在布林带中的位置
        current_price = price_data.iloc[-1]
        upper = upper_band.iloc[-1]
        lower = lower_band.iloc[-1]
        
        position = (current_price - lower) / (upper - lower)
        return position.clip(0, 1)  # 限制在0-1范围内
    
    def calculate_williams_r(self, price_data: pd.DataFrame, volume_data=None, window: int = 14) -> pd.Series:
        """计算威廉指标"""
        if len(price_data) < window:
            return pd.Series(index=price_data.columns, dtype=float)
        
        # 假设price_data是收盘价，简化计算
        highest = price_data.rolling(window=window).max()
        lowest = price_data.rolling(window=window).min()
        current_price = price_data.iloc[-1]
        
        williams_r = (highest.iloc[-1] - current_price) / (highest.iloc[-1] - lowest.iloc[-1]) * -100
        return williams_r
    
    def calculate_stochastic_k(self, price_data: pd.DataFrame, volume_data=None, window: int = 14) -> pd.Series:
        """计算随机指标K值"""
        if len(price_data) < window:
            return pd.Series(index=price_data.columns, dtype=float)
        
        highest = price_data.rolling(window=window).max()
        lowest = price_data.rolling(window=window).min()
        current_price = price_data.iloc[-1]
        
        k_value = (current_price - lowest.iloc[-1]) / (highest.iloc[-1] - lowest.iloc[-1]) * 100
        return k_value
    
    def calculate_momentum_quality(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算动量质量因子"""
        # 计算不同周期的动量
        mom_5d = price_data.pct_change(periods=5)
        mom_20d = price_data.pct_change(periods=20)
        mom_60d = price_data.pct_change(periods=60)
        
        # 动量一致性得分
        momentum_consistency = (
            (mom_5d.iloc[-1] > 0).astype(int) +
            (mom_20d.iloc[-1] > 0).astype(int) +
            (mom_60d.iloc[-1] > 0).astype(int)
        ) / 3
        
        # 结合动量强度
        momentum_strength = (abs(mom_5d.iloc[-1]) + abs(mom_20d.iloc[-1]) + abs(mom_60d.iloc[-1])) / 3
        
        return momentum_consistency * momentum_strength