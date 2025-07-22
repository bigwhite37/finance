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
            price_data: 价格数据
            volume_data: 成交量数据
            factors: 指定计算的因子列表
            
        Returns:
            风险因子数据
        """
        factor_results = {}
        
        if factors is None:
            factors = ['volatility_60d', 'volume_ratio', 'turnover_rate', 'beta_60d']
        
        for factor_name in factors:
            factor_func = getattr(self, f'calculate_{factor_name}')
            factor_data = factor_func(price_data, volume_data)
            factor_results[factor_name] = factor_data
        
        return pd.DataFrame(factor_results)
    
    def calculate_volatility_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算60日波动率"""
        returns = price_data.pct_change()
        volatility = returns.rolling(window=60).std() * np.sqrt(252)
        return volatility.iloc[-1]
    
    def calculate_volume_ratio(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.Series:
        """计算成交量比率"""
        volume_ma = volume_data.rolling(window=20).mean()
        current_volume = volume_data.iloc[-1]
        return current_volume / volume_ma.iloc[-1]
    
    def calculate_turnover_rate(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> pd.Series:
        """计算换手率"""
        # 简化计算：成交量相对于20日均值的比率
        volume_std = volume_data.rolling(window=20).std()
        volume_mean = volume_data.rolling(window=20).mean()
        turnover = volume_std.iloc[-1] / volume_mean.iloc[-1]
        return turnover
    
    def calculate_beta_60d(self, price_data: pd.DataFrame, volume_data=None, 
                          benchmark_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """计算60日Beta"""
        returns = price_data.pct_change()
        
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change()
            
            def calc_beta(stock_returns):
                covariance = stock_returns.rolling(window=60).cov(benchmark_returns.iloc[:, 0])
                variance = benchmark_returns.iloc[:, 0].rolling(window=60).var()
                return (covariance / variance).iloc[-1]
            
            return returns.apply(calc_beta)
        else:
            # 使用等权重市场作为基准
            market_returns = returns.mean(axis=1)
            
            def calc_beta_simple(stock_returns):
                covariance = stock_returns.rolling(window=60).cov(market_returns)
                variance = market_returns.rolling(window=60).var()
                return (covariance / variance).iloc[-1]
            
            return returns.apply(calc_beta_simple)
    
    def calculate_skewness_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算60日收益率偏度"""
        returns = price_data.pct_change()
        skewness = returns.rolling(window=60).skew()
        return skewness.iloc[-1]
    
    def calculate_kurtosis_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算60日收益率峰度"""
        returns = price_data.pct_change()
        kurtosis = returns.rolling(window=60).kurt()
        return kurtosis.iloc[-1]
    
    def calculate_max_drawdown_60d(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算60日最大回撤"""
        cumulative_returns = (1 + price_data.pct_change()).cumprod()
        rolling_max = cumulative_returns.rolling(window=60, min_periods=1).max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(window=60).min()
        return max_drawdown.iloc[-1]
    
    def calculate_var_95(self, price_data: pd.DataFrame, volume_data=None, window: int = 60) -> pd.Series:
        """计算95% VaR"""
        returns = price_data.pct_change()
        var_95 = returns.rolling(window=window).quantile(0.05)
        return var_95.iloc[-1]
    
    def calculate_cvar_95(self, price_data: pd.DataFrame, volume_data=None, window: int = 60) -> pd.Series:
        """计算95% CVaR条件风险价值"""
        returns = price_data.pct_change()
        
        def calc_cvar(series):
            var_95 = series.quantile(0.05)
            cvar = series[series <= var_95].mean()
            return cvar
        
        cvar = returns.rolling(window=window).apply(calc_cvar)
        return cvar.iloc[-1]
    
    def calculate_downside_deviation(self, price_data: pd.DataFrame, volume_data=None, 
                                   target_return: float = 0, window: int = 60) -> pd.Series:
        """计算下行偏差"""
        returns = price_data.pct_change()
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.rolling(window=window).std()
        return downside_deviation.iloc[-1]
    
    def calculate_upside_deviation(self, price_data: pd.DataFrame, volume_data=None,
                                 target_return: float = 0, window: int = 60) -> pd.Series:
        """计算上行偏差"""
        returns = price_data.pct_change()
        excess_returns = returns - target_return
        upside_returns = excess_returns[excess_returns > 0]
        upside_deviation = upside_returns.rolling(window=window).std()
        return upside_deviation.iloc[-1]
    
    def calculate_volatility_regime(self, price_data: pd.DataFrame, volume_data=None) -> pd.Series:
        """计算波动率状态因子"""
        returns = price_data.pct_change()
        
        # 短期波动率(20日)
        short_vol = returns.rolling(window=20).std()
        # 长期波动率(60日) 
        long_vol = returns.rolling(window=60).std()
        
        # 波动率比率
        vol_ratio = short_vol.iloc[-1] / long_vol.iloc[-1]
        return vol_ratio