"""
高级Alpha因子库
针对8%年化收益目标设计的高质量预测因子
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedAlphaFactors:
    """高级Alpha因子计算器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.factor_cache = {}
        
        # 因子计算参数
        self.short_window = config.get('short_window', 5)
        self.medium_window = config.get('medium_window', 20)
        self.long_window = config.get('long_window', 60)
        
        # 因子列表 - 针对中国A股市场优化
        self.alpha_factors = [
            # 动量类因子
            'momentum_reversal_5d', 'momentum_reversal_20d', 'momentum_trend_strength',
            'price_acceleration', 'relative_strength_index', 'momentum_quality',
            
            # 价值类因子  
            'price_book_momentum', 'earnings_momentum', 'value_momentum',
            'price_efficiency', 'fundamental_strength',
            
            # 技术类因子
            'technical_alpha', 'volume_price_correlation', 'volatility_adjusted_momentum',
            'breakout_strength', 'support_resistance', 'trend_consistency',
            
            # 市场微观结构因子
            'order_flow_imbalance', 'market_impact', 'liquidity_premium',
            'information_ratio', 'price_discovery_efficiency',
            
            # 组合类因子
            'sector_relative_strength', 'style_momentum', 'cross_asset_momentum',
            'market_regime_alpha', 'volatility_risk_premium'
        ]
        
        logger.info(f"初始化高级Alpha因子库，包含 {len(self.alpha_factors)} 个因子")
    
    def calculate_all_factors(self, price_data: pd.DataFrame, 
                            volume_data: Optional[pd.DataFrame] = None,
                            fundamental_data: Optional[pd.DataFrame] = None,
                            factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算所有Alpha因子
        
        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            fundamental_data: 基本面数据
            factors: 指定计算的因子列表
            
        Returns:
            因子数据DataFrame
        """
        if factors is None:
            factors = self.alpha_factors
        
        factor_results = {}
        
        for factor_name in factors:
            try:
                if hasattr(self, f'calculate_{factor_name}'):
                    factor_func = getattr(self, f'calculate_{factor_name}')
                    factor_data = factor_func(price_data, volume_data, fundamental_data)
                    
                    # 数据质量检查
                    if self._validate_factor_data(factor_data, factor_name):
                        factor_results[factor_name] = factor_data
                    else:
                        logger.warning(f"因子 {factor_name} 数据质量检查失败")
                        
                else:
                    logger.warning(f"因子计算方法不存在: calculate_{factor_name}")
                    
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 时出错: {e}")
        
        if factor_results:
            return pd.concat(factor_results, axis=1)
        else:
            return pd.DataFrame(index=price_data.index, columns=price_data.columns)
    
    def _validate_factor_data(self, factor_data: pd.DataFrame, factor_name: str) -> bool:
        """验证因子数据质量"""
        if factor_data is None or factor_data.empty:
            return False
        
        # 检查NaN比例
        nan_ratio = factor_data.isnull().sum().sum() / (factor_data.shape[0] * factor_data.shape[1])
        if nan_ratio > 0.8:
            logger.warning(f"因子 {factor_name} 的NaN比例过高: {nan_ratio:.2%}")
            return False
        
        # 检查无穷值
        if np.isinf(factor_data.values).any():
            logger.warning(f"因子 {factor_name} 包含无穷值")
            return False
        
        # 检查数值范围合理性
        factor_std = factor_data.std().mean()
        if factor_std == 0 or factor_std > 1000:
            logger.warning(f"因子 {factor_name} 方差异常: {factor_std}")
            return False
        
        return True
    
    # =================== 动量类因子 ===================
    
    def calculate_momentum_reversal_5d(self, price_data: pd.DataFrame, 
                                     volume_data: Optional[pd.DataFrame] = None,
                                     fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """5日动量反转因子"""
        # 短期收益率
        ret_1d = price_data.pct_change(1)
        ret_5d = price_data.pct_change(5)
        
        # 动量反转信号：当前强势但短期可能反转
        momentum_strength = ret_5d.rolling(window=5).mean()
        recent_momentum = ret_1d.rolling(window=3).mean()
        
        # 反转信号 = 中期动量强度 * (1 - 短期动量持续性)
        reversal_factor = momentum_strength * (1 - recent_momentum.abs())
        
        return self._standardize_factor(reversal_factor)
    
    def calculate_momentum_reversal_20d(self, price_data: pd.DataFrame,
                                      volume_data: Optional[pd.DataFrame] = None,
                                      fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """20日动量反转因子"""
        ret_20d = price_data.pct_change(20)
        ret_5d = price_data.pct_change(5)
        
        # 基于回归残差的反转信号
        def calculate_reversal(series):
            if len(series) < 25:
                return pd.Series(0, index=series.index)
            
            reversal_signals = []
            for i in range(20, len(series)):
                # 使用线性回归预测
                y = series.iloc[i-20:i].values
                x = np.arange(len(y))
                
                if len(y) >= 10:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    predicted = slope * len(y) + intercept
                    actual = series.iloc[i]
                    
                    # 实际值与预测值的偏差作为反转信号
                    reversal_signal = (actual - predicted) / (std_err + 1e-8)
                    reversal_signals.append(reversal_signal)
                else:
                    reversal_signals.append(0)
            
            # 补齐前面的值
            full_signals = [0] * 20 + reversal_signals
            return pd.Series(full_signals, index=series.index)
        
        reversal_factor = price_data.apply(calculate_reversal)
        return self._standardize_factor(reversal_factor)
    
    def calculate_momentum_trend_strength(self, price_data: pd.DataFrame,
                                        volume_data: Optional[pd.DataFrame] = None,
                                        fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """趋势强度因子"""
        # 使用多个时间窗口的价格趋势
        windows = [5, 10, 20, 40]
        trend_signals = []
        
        for window in windows:
            # 计算趋势斜率
            def trend_slope(series):
                if len(series) < window + 5:
                    return pd.Series(0, index=series.index)
                
                slopes = []
                for i in range(window, len(series)):
                    y = series.iloc[i-window:i].values
                    x = np.arange(len(y))
                    slope, _, r_value, _, _ = stats.linregress(x, y)
                    
                    # 趋势强度 = 斜率 * R方值
                    trend_strength = slope * (r_value ** 2) if not np.isnan(r_value) else 0
                    slopes.append(trend_strength)
                
                full_slopes = [0] * window + slopes
                return pd.Series(full_slopes, index=series.index)
            
            trend_signal = price_data.apply(trend_slope)
            trend_signals.append(trend_signal)
        
        # 加权平均不同窗口的趋势信号
        weights = [0.4, 0.3, 0.2, 0.1]  # 偏重短期趋势
        combined_trend = sum(w * signal for w, signal in zip(weights, trend_signals))
        
        return self._standardize_factor(combined_trend)
    
    def calculate_price_acceleration(self, price_data: pd.DataFrame,
                                   volume_data: Optional[pd.DataFrame] = None,
                                   fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """价格加速度因子"""
        # 一阶导数（速度）和二阶导数（加速度）
        returns = price_data.pct_change()
        velocity = returns.rolling(window=5).mean()
        acceleration = velocity.diff()
        
        # 结合成交量的价格加速度
        if volume_data is not None:
            volume_acceleration = volume_data.pct_change().rolling(window=5).mean().diff()
            # 价量背离作为反向信号
            price_volume_divergence = acceleration * volume_acceleration
            acceleration = acceleration + 0.3 * price_volume_divergence
        
        return self._standardize_factor(acceleration)
    
    def calculate_relative_strength_index(self, price_data: pd.DataFrame,
                                        volume_data: Optional[pd.DataFrame] = None,
                                        fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """相对强弱指数（改进版RSI）"""
        def enhanced_rsi(series, window=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # 标准化到[-1, 1]区间，中值为50
            normalized_rsi = (rsi - 50) / 50
            
            return normalized_rsi
        
        rsi_factor = price_data.apply(lambda x: enhanced_rsi(x))
        return self._standardize_factor(rsi_factor)
    
    def calculate_momentum_quality(self, price_data: pd.DataFrame,
                                 volume_data: Optional[pd.DataFrame] = None,
                                 fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """动量质量因子"""
        # 计算多期动量
        ret_5d = price_data.pct_change(5)
        ret_20d = price_data.pct_change(20)
        ret_60d = price_data.pct_change(60)
        
        # 动量一致性
        momentum_consistency = np.sign(ret_5d) * np.sign(ret_20d) * np.sign(ret_60d)
        
        # 动量强度
        momentum_strength = (abs(ret_5d) + abs(ret_20d) + abs(ret_60d)) / 3
        
        # 质量得分 = 一致性 * 强度
        quality_score = momentum_consistency * momentum_strength
        
        return self._standardize_factor(quality_score)
    
    # =================== 价值类因子 ===================
    
    def calculate_price_book_momentum(self, price_data: pd.DataFrame,
                                    volume_data: Optional[pd.DataFrame] = None,
                                    fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """价格账面价值动量因子"""
        if fundamental_data is None:
            # 使用价格代理
            book_proxy = price_data.rolling(window=252).mean()  # 年均价格作为代理
        else:
            book_proxy = fundamental_data
        
        # 市净率的变化趋势
        pb_ratio = price_data / (book_proxy + 1e-8)
        pb_momentum = pb_ratio.pct_change(20).rolling(window=10).mean()
        
        # 反向价值因子：低PB + 上升趋势
        value_momentum = -pb_ratio.rank(axis=1, pct=True) + pb_momentum
        
        return self._standardize_factor(value_momentum)
    
    def calculate_earnings_momentum(self, price_data: pd.DataFrame,
                                  volume_data: Optional[pd.DataFrame] = None,
                                  fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """盈利动量因子"""
        # 使用价格变化作为盈利代理
        earnings_proxy = price_data.rolling(window=60).mean()
        earnings_growth = earnings_proxy.pct_change(60)
        
        # 盈利动量 = 增长率的趋势
        earnings_momentum = earnings_growth.rolling(window=20).apply(
            lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 10 else 0
        )
        
        return self._standardize_factor(earnings_momentum)
    
    # =================== 技术类因子 ===================
    
    def calculate_technical_alpha(self, price_data: pd.DataFrame,
                                volume_data: Optional[pd.DataFrame] = None,
                                fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """技术Alpha因子"""
        # 多重技术指标组合
        # 1. 布林带位置
        sma_20 = price_data.rolling(window=20).mean()
        std_20 = price_data.rolling(window=20).std()
        bollinger_position = (price_data - sma_20) / (2 * std_20)
        
        # 2. 价格相对位置
        price_position = (price_data - price_data.rolling(window=40).min()) / (
            price_data.rolling(window=40).max() - price_data.rolling(window=40).min() + 1e-8
        )
        
        # 3. 移动平均收敛背离(MACD简化版)
        ema_12 = price_data.ewm(span=12).mean()
        ema_26 = price_data.ewm(span=26).mean()
        macd = (ema_12 - ema_26) / price_data
        
        # 综合技术信号
        technical_alpha = 0.4 * bollinger_position + 0.3 * price_position + 0.3 * macd
        
        return self._standardize_factor(technical_alpha)
    
    def calculate_volume_price_correlation(self, price_data: pd.DataFrame,
                                         volume_data: Optional[pd.DataFrame] = None,
                                         fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """价量相关性因子"""
        if volume_data is None:
            return pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
        
        price_change = price_data.pct_change()
        volume_change = volume_data.pct_change()
        
        # 滚动相关性
        def rolling_correlation(window=20):
            correlation = pd.DataFrame(index=price_data.index, columns=price_data.columns)
            
            for col in price_data.columns:
                if col in volume_data.columns:
                    correlation[col] = price_change[col].rolling(window=window).corr(volume_change[col])
            
            return correlation.fillna(0)
        
        # 多期相关性
        corr_short = rolling_correlation(10)
        corr_medium = rolling_correlation(20)
        
        # 组合信号
        combined_correlation = 0.6 * corr_short + 0.4 * corr_medium
        
        return self._standardize_factor(combined_correlation)
    
    # =================== 市场微观结构因子 ===================
    
    def calculate_order_flow_imbalance(self, price_data: pd.DataFrame,
                                     volume_data: Optional[pd.DataFrame] = None,
                                     fundamental_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """订单流不平衡因子（简化版）"""
        if volume_data is None:
            return pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
        
        # 基于价格变化和成交量的订单流估计
        price_change = price_data.pct_change()
        volume_normalized = volume_data / volume_data.rolling(window=20).mean()
        
        # 订单流不平衡估计
        buy_pressure = (price_change > 0).astype(float) * volume_normalized
        sell_pressure = (price_change < 0).astype(float) * volume_normalized
        
        order_flow_imbalance = (buy_pressure - sell_pressure).rolling(window=10).mean()
        
        return self._standardize_factor(order_flow_imbalance)
    
    # =================== 辅助方法 ===================
    
    def _standardize_factor(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """因子标准化"""
        # 时间序列标准化（每个时间点横截面标准化）
        standardized = factor_data.apply(
            lambda x: (x - x.mean()) / (x.std() + 1e-8), axis=1
        )
        
        # 限制极值
        standardized = standardized.clip(-3, 3)
        
        # 填充NaN
        standardized = standardized.fillna(0)
        
        return standardized
    
    def get_factor_statistics(self, factor_data: pd.DataFrame) -> Dict:
        """获取因子统计信息"""
        stats_dict = {}
        
        for factor_name in factor_data.columns:
            factor_series = factor_data[factor_name].dropna()
            
            if len(factor_series) > 0:
                stats_dict[factor_name] = {
                    'mean': factor_series.mean(),
                    'std': factor_series.std(),
                    'skew': factor_series.skew(),
                    'kurt': factor_series.kurtosis(),
                    'non_null_ratio': len(factor_series) / len(factor_data),
                    'ic_mean': 0.0,  # 需要收益数据才能计算
                    'ic_std': 0.0
                }
        
        return stats_dict
    
    def calculate_ic_analysis(self, factor_data: pd.DataFrame, 
                            future_returns: pd.DataFrame,
                            periods: List[int] = [1, 5, 10]) -> Dict:
        """计算因子IC分析"""
        ic_results = {}
        
        for period in periods:
            forward_returns = future_returns.shift(-period)
            
            ic_values = {}
            for factor_name in factor_data.columns:
                ic_series = []
                
                for date in factor_data.index:
                    if date in forward_returns.index:
                        factor_cross_section = factor_data.loc[date].dropna()
                        return_cross_section = forward_returns.loc[date].dropna()
                        
                        # 找到共同的股票
                        common_stocks = factor_cross_section.index.intersection(return_cross_section.index)
                        
                        if len(common_stocks) > 10:
                            factor_values = factor_cross_section[common_stocks]
                            return_values = return_cross_section[common_stocks]
                            
                            correlation = factor_values.corr(return_values)
                            if not np.isnan(correlation):
                                ic_series.append(correlation)
                
                if ic_series:
                    ic_values[factor_name] = {
                        'ic_mean': np.mean(ic_series),
                        'ic_std': np.std(ic_series),
                        'ic_ir': np.mean(ic_series) / (np.std(ic_series) + 1e-8),
                        'ic_positive_ratio': np.mean([ic > 0 for ic in ic_series])
                    }
            
            ic_results[f'{period}d'] = ic_values
        
        return ic_results