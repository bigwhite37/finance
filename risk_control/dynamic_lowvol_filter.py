"""
动态低波筛选器核心模块

实现动态、预测式、市场状态感知的低波股票筛选系统，
替代传统的静态阈值方法。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# 导入性能优化模块
from .performance_optimizer import (
    PerformanceOptimizer, CacheConfig, ParallelConfig,
    AdvancedCacheManager, ParallelProcessingManager,
    VectorizedComputeOptimizer, MemoryMonitor,
    performance_monitor
)


# ============================================================================
# 异常类定义
# ============================================================================

class FilterException(Exception):
    """筛选器基础异常"""
    pass


class DataQualityException(FilterException):
    """数据质量异常 - 数据不符合质量要求时抛出"""
    pass


class ModelFittingException(FilterException):
    """模型拟合异常 - GARCH或其他模型拟合失败时抛出"""
    pass


class RegimeDetectionException(FilterException):
    """状态检测异常 - 市场状态检测失败时抛出"""
    pass


class InsufficientDataException(FilterException):
    """数据不足异常 - 历史数据不足以进行计算时抛出"""
    pass


class ConfigurationException(FilterException):
    """配置异常 - 配置参数不正确时抛出"""
    pass


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class FilterInputData:
    """筛选器输入数据结构"""
    price_data: pd.DataFrame      # 价格数据 (日期 x 股票)
    volume_data: pd.DataFrame     # 成交量数据
    factor_data: pd.DataFrame     # 因子数据
    market_data: pd.DataFrame     # 市场指数数据
    current_date: pd.Timestamp    # 当前交易日期


@dataclass
class FilterOutputData:
    """筛选器输出数据结构"""
    tradable_mask: np.ndarray     # 可交易股票掩码
    current_regime: str           # 当前市场状态
    regime_signal: float          # 状态信号 (-1, 0, 1)
    adaptive_target_vol: float    # 自适应目标波动率
    filter_statistics: Dict       # 筛选统计信息


@dataclass
class DynamicLowVolConfig:
    """动态低波筛选器配置数据类"""
    
    # 滚动分位筛选配置
    rolling_windows: List[int] = field(default_factory=lambda: [20, 60])
    percentile_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "低": 0.4, "中": 0.3, "高": 0.2
    })
    
    # GARCH预测配置
    garch_window: int = 250
    forecast_horizon: int = 5
    enable_ml_predictor: bool = False
    
    # IVOL约束配置
    ivol_bad_threshold: float = 0.3
    ivol_good_threshold: float = 0.6
    
    # 市场状态检测配置
    regime_detection_window: int = 60
    regime_model_type: str = "HMM"  # HMM, MS-GARCH
    
    # 性能优化配置
    enable_caching: bool = True
    cache_expiry_days: int = 1
    parallel_processing: bool = True
    
    def __post_init__(self):
        """配置参数验证"""
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的合理性"""
        # 验证滚动窗口
        if not all(isinstance(w, int) and w > 0 for w in self.rolling_windows):
            raise ConfigurationException("滚动窗口必须为正整数")
        
        # 验证分位数阈值
        for regime, threshold in self.percentile_thresholds.items():
            if not (0 < threshold < 1):
                raise ConfigurationException(f"分位数阈值必须在(0,1)范围内，当前{regime}状态阈值为{threshold}")
        
        # 验证GARCH配置
        if not isinstance(self.garch_window, int) or self.garch_window < 100:
            raise ConfigurationException("GARCH窗口长度必须至少为100")
        
        if not isinstance(self.forecast_horizon, int) or self.forecast_horizon < 1:
            raise ConfigurationException("预测期限必须为正整数")
        
        # 验证IVOL阈值
        if not (0 < self.ivol_bad_threshold < 1):
            raise ConfigurationException(f"IVOL坏波动阈值必须在(0,1)范围内，当前为{self.ivol_bad_threshold}")
        
        if not (0 < self.ivol_good_threshold < 1):
            raise ConfigurationException(f"IVOL好波动阈值必须在(0,1)范围内，当前为{self.ivol_good_threshold}")
        
        # 验证状态检测配置
        if not isinstance(self.regime_detection_window, int) or self.regime_detection_window < 20:
            raise ConfigurationException("状态检测窗口长度必须至少为20")
        
        if self.regime_model_type not in ["HMM", "MS-GARCH"]:
            raise ConfigurationException(f"不支持的状态检测模型类型: {self.regime_model_type}")
        
        # 验证缓存配置
        if not isinstance(self.cache_expiry_days, int) or self.cache_expiry_days < 0:
            raise ConfigurationException("缓存过期天数必须为非负整数")


# ============================================================================
# 数据预处理模块
# ============================================================================

class DataPreprocessor:
    """数据预处理模块
    
    负责数据清洗、缺失值处理、异常值检测、收益率计算和滚动窗口数据准备
    """
    
    def __init__(self, config: DynamicLowVolConfig):
        """初始化数据预处理器
        
        Args:
            config: 筛选器配置
        """
        self.config = config
        self.missing_threshold = 0.1  # 缺失值比例阈值
        self.outlier_threshold = 5.0  # 异常值标准差倍数阈值
    
    def preprocess_price_data(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """预处理价格数据
        
        Args:
            price_data: 原始价格数据 (日期 x 股票)
            
        Returns:
            清洗后的价格数据
            
        Raises:
            DataQualityException: 数据质量不符合要求
            InsufficientDataException: 数据长度不足
        """
        if price_data.empty:
            raise DataQualityException("价格数据为空")
        
        # 检查数据长度
        min_required_length = max(self.config.rolling_windows) + self.config.garch_window
        if len(price_data) < min_required_length:
            raise InsufficientDataException(
                f"价格数据长度{len(price_data)}不足，需要至少{min_required_length}个观测值"
            )
        
        # 检查缺失值比例
        missing_ratio = price_data.isna().sum() / len(price_data)
        problematic_stocks = missing_ratio[missing_ratio > self.missing_threshold]
        if len(problematic_stocks) > 0:
            raise DataQualityException(
                f"以下股票缺失值比例超过{self.missing_threshold:.1%}: "
                f"{problematic_stocks.to_dict()}"
            )
        
        # 数据标准化处理
        cleaned_data = price_data.copy()
        
        # 前向填充缺失值
        cleaned_data = cleaned_data.ffill()
        
        # 检查是否还有缺失值
        if cleaned_data.isna().any().any():
            # 后向填充剩余缺失值
            cleaned_data = cleaned_data.bfill()
        
        # 异常值检测和处理
        cleaned_data = self._handle_outliers(cleaned_data)
        
        return cleaned_data
    
    def calculate_returns(self, price_data: pd.DataFrame, 
                         return_type: str = 'simple') -> pd.DataFrame:
        """计算收益率
        
        Args:
            price_data: 价格数据 (日期 x 股票)
            return_type: 收益率类型 ('simple' 或 'log')
            
        Returns:
            收益率数据
            
        Raises:
            DataQualityException: 数据质量问题
            ConfigurationException: 不支持的收益率类型
        """
        if price_data.empty:
            raise DataQualityException("价格数据为空，无法计算收益率")
        
        if return_type not in ['simple', 'log']:
            raise ConfigurationException(f"不支持的收益率类型: {return_type}")
        
        # 检查价格数据是否包含非正值
        if (price_data <= 0).any().any():
            raise DataQualityException("价格数据包含非正值，无法计算对数收益率")
        
        if return_type == 'simple':
            returns = price_data.pct_change()
        else:  # log returns
            returns = np.log(price_data / price_data.shift(1))
        
        # 移除第一行（NaN值）
        returns = returns.iloc[1:]
        
        # 检查收益率异常值
        returns = self._validate_returns(returns)
        
        return returns
    
    def prepare_rolling_windows(self, data: pd.DataFrame, 
                              windows: List[int]) -> Dict[int, pd.DataFrame]:
        """准备滚动窗口数据
        
        Args:
            data: 输入数据 (日期 x 股票)
            windows: 滚动窗口长度列表
            
        Returns:
            不同窗口长度的滚动数据字典
            
        Raises:
            InsufficientDataException: 数据长度不足
            DataQualityException: 数据质量问题
        """
        if data.empty:
            raise DataQualityException("输入数据为空")
        
        max_window = max(windows)
        if len(data) < max_window:
            raise InsufficientDataException(
                f"数据长度{len(data)}不足，最大窗口长度为{max_window}"
            )
        
        rolling_data = {}
        
        for window in windows:
            # 确保有足够的数据进行滚动计算
            if len(data) < window:
                raise InsufficientDataException(
                    f"数据长度{len(data)}不足以计算{window}期滚动窗口"
                )
            
            # 创建滚动窗口视图
            rolling_data[window] = data.copy()
        
        return rolling_data
    
    def validate_data_quality(self, data, 
                            data_name: str = "数据") -> None:
        """验证数据质量
        
        Args:
            data: 待验证的数据
            data_name: 数据名称（用于错误信息）
            
        Raises:
            DataQualityException: 数据质量不符合要求
        """
        # 检查数据类型
        if not isinstance(data, pd.DataFrame):
            raise DataQualityException(f"{data_name}必须为DataFrame类型")
        
        if data.empty:
            raise DataQualityException(f"{data_name}为空")
        
        # 检查索引类型
        if not isinstance(data.index, (pd.DatetimeIndex, pd.RangeIndex)):
            raise DataQualityException(f"{data_name}索引必须为DatetimeIndex或RangeIndex类型")
        
        # 检查缺失值比例
        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        missing_ratio = missing_cells / total_cells
        
        if missing_ratio > self.missing_threshold:
            raise DataQualityException(
                f"{data_name}整体缺失值比例{missing_ratio:.2%}超过阈值{self.missing_threshold:.1%}"
            )
        
        # 检查数值类型列是否包含无穷值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            inf_count = np.isinf(data[numeric_cols]).sum().sum()
            if inf_count > 0:
                raise DataQualityException(f"{data_name}包含{inf_count}个无穷值")
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理异常值
        
        Args:
            data: 输入数据
            
        Returns:
            处理后的数据
        """
        cleaned_data = data.copy()
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) == 0:
                continue
            
            # 计算Z-score
            mean_val = series.mean()
            std_val = series.std()
            
            if std_val == 0:
                continue
            
            z_scores = np.abs((series - mean_val) / std_val)
            
            # 标记异常值
            outliers = z_scores > self.outlier_threshold
            
            if outliers.sum() > 0:
                # 使用中位数替换异常值
                median_val = series.median()
                cleaned_data.loc[outliers, column] = median_val
        
        return cleaned_data
    
    def _validate_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """验证收益率数据质量
        
        Args:
            returns: 收益率数据
            
        Returns:
            验证后的收益率数据
            
        Raises:
            DataQualityException: 收益率数据异常
        """
        # 检查极端收益率
        extreme_threshold = 0.5  # 50%的单日收益率阈值
        extreme_returns_count = (np.abs(returns) > extreme_threshold).sum().sum()
        total_observations = len(returns) * len(returns.columns)
        extreme_ratio_threshold = 0.01  # 1%的观测值阈值
        
        if extreme_returns_count > total_observations * extreme_ratio_threshold:
            raise DataQualityException(
                f"极端收益率（绝对值>{extreme_threshold:.0%}）数量过多: {extreme_returns_count}个，"
                f"占比{extreme_returns_count/total_observations:.2%}，超过{extreme_ratio_threshold:.1%}阈值"
            )
        
        # 检查收益率的统计特征
        for column in returns.columns:
            series = returns[column].dropna()
            if len(series) == 0:
                continue
            
            # 检查方差是否为0（价格不变）
            if series.var() == 0:
                raise DataQualityException(f"股票{column}收益率方差为0，价格可能未变化")
            
            # 检查是否存在过多的零收益率
            zero_returns_ratio = (series == 0).sum() / len(series)
            if zero_returns_ratio > 0.5:
                raise DataQualityException(
                    f"股票{column}零收益率比例{zero_returns_ratio:.1%}过高，可能存在停牌"
                )
        
        return returns


# ============================================================================
# 滚动分位筛选层
# ============================================================================

class RollingPercentileFilter:
    """滚动分位筛选层
    
    基于滚动窗口计算股票波动率的市场分位数排名，
    实现动态阈值调整以"跟随市场呼吸"。
    """
    
    def __init__(self, config: DynamicLowVolConfig):
        """初始化滚动分位筛选器
        
        Args:
            config: 筛选器配置
        """
        self.config = config
        self.rolling_windows = config.rolling_windows
        self.percentile_thresholds = config.percentile_thresholds
        
        # 缓存机制
        self._volatility_cache = {} if config.enable_caching else None
        self._percentile_cache = {} if config.enable_caching else None
    
    def apply_percentile_filter(self, 
                              returns: pd.DataFrame,
                              current_date: pd.Timestamp,
                              window: int = 20,
                              percentile_threshold: float = 0.3,
                              by_industry: bool = False,
                              industry_mapping: Optional[Dict[str, str]] = None) -> np.ndarray:
        """应用滚动分位筛选
        
        Args:
            returns: 收益率数据 (日期 x 股票)
            current_date: 当前日期
            window: 滚动窗口长度，默认20日
            percentile_threshold: 分位数阈值，默认0.3 (30%)
            by_industry: 是否按行业分组计算分位数
            industry_mapping: 股票到行业的映射字典
            
        Returns:
            可交易股票掩码数组 (True表示可交易)
            
        Raises:
            InsufficientDataException: 数据长度不足
            DataQualityException: 数据质量问题
            ConfigurationException: 配置参数错误
        """
        # 参数验证
        if not isinstance(returns, pd.DataFrame):
            raise DataQualityException("收益率数据必须为DataFrame类型")
        
        if returns.empty:
            raise DataQualityException("收益率数据为空")
        
        if window <= 0:
            raise ConfigurationException(f"滚动窗口长度必须为正数，当前为{window}")
        
        if not (0 < percentile_threshold < 1):
            raise ConfigurationException(f"分位数阈值必须在(0,1)范围内，当前为{percentile_threshold}")
        
        if len(returns) < window:
            raise InsufficientDataException(
                f"数据长度{len(returns)}不足以计算{window}期滚动窗口"
            )
        
        # 计算滚动波动率
        rolling_volatility = self._calculate_rolling_volatility(
            returns, window, current_date
        )
        
        # 计算分位数排名
        percentile_ranks = self._calculate_percentile_ranks(
            rolling_volatility, by_industry, industry_mapping
        )
        
        # 应用阈值筛选
        tradable_mask = percentile_ranks <= percentile_threshold
        
        # 验证结果
        self._validate_filter_result(tradable_mask, returns.columns)
        
        return tradable_mask.values
    
    def calculate_dynamic_threshold(self, 
                                  market_volatility: pd.Series,
                                  base_threshold: float = 0.3,
                                  sensitivity: float = 0.5) -> float:
        """计算动态分位数阈值，实现"跟随市场呼吸"
        
        Args:
            market_volatility: 市场整体波动率时间序列
            base_threshold: 基础分位数阈值
            sensitivity: 对市场波动的敏感度 (0-1)
            
        Returns:
            调整后的分位数阈值
            
        Raises:
            DataQualityException: 市场波动率数据问题
            ConfigurationException: 参数配置错误
        """
        if market_volatility.empty:
            raise DataQualityException("市场波动率数据为空")
        
        if not (0 < base_threshold < 1):
            raise ConfigurationException(f"基础阈值必须在(0,1)范围内，当前为{base_threshold}")
        
        if not (0 <= sensitivity <= 1):
            raise ConfigurationException(f"敏感度必须在[0,1]范围内，当前为{sensitivity}")
        
        # 计算市场波动率的历史分位数
        current_vol = market_volatility.iloc[-1]
        historical_vol = market_volatility.iloc[:-1]
        
        if len(historical_vol) == 0:
            return base_threshold
        
        # 计算当前波动率在历史分布中的分位数
        vol_percentile = (historical_vol < current_vol).mean()
        
        # 动态调整阈值
        # 当市场波动率较高时，收紧阈值（选择更少的股票）
        # 当市场波动率较低时，放宽阈值（选择更多的股票）
        adjustment_factor = (0.5 - vol_percentile) * sensitivity
        dynamic_threshold = base_threshold + adjustment_factor
        
        # 确保阈值在合理范围内
        dynamic_threshold = np.clip(dynamic_threshold, 0.1, 0.5)
        
        return dynamic_threshold
    
    def get_multi_window_filter(self, 
                               returns: pd.DataFrame,
                               current_date: pd.Timestamp,
                               windows: Optional[List[int]] = None,
                               combination_method: str = 'intersection') -> np.ndarray:
        """多窗口组合筛选
        
        Args:
            returns: 收益率数据
            current_date: 当前日期
            windows: 滚动窗口长度列表，默认使用配置中的窗口
            combination_method: 组合方法 ('intersection', 'union', 'weighted')
            
        Returns:
            组合筛选结果掩码
            
        Raises:
            ConfigurationException: 不支持的组合方法
        """
        if windows is None:
            windows = self.rolling_windows
        
        if combination_method not in ['intersection', 'union', 'weighted']:
            raise ConfigurationException(f"不支持的组合方法: {combination_method}")
        
        # 计算各窗口的筛选结果
        filter_results = []
        for window in windows:
            mask = self.apply_percentile_filter(
                returns, current_date, window=window
            )
            filter_results.append(mask)
        
        # 组合筛选结果
        if combination_method == 'intersection':
            # 交集：所有窗口都通过筛选
            combined_mask = np.all(filter_results, axis=0)
        elif combination_method == 'union':
            # 并集：任一窗口通过筛选
            combined_mask = np.any(filter_results, axis=0)
        else:  # weighted
            # 加权组合：短窗口权重更高
            weights = np.array([1.0 / w for w in windows])
            weights = weights / weights.sum()
            
            weighted_scores = np.zeros(len(returns.columns))
            for i, mask in enumerate(filter_results):
                weighted_scores += mask.astype(float) * weights[i]
            
            # 使用0.5作为阈值
            combined_mask = weighted_scores >= 0.5
        
        return combined_mask
    
    def _calculate_rolling_volatility(self, 
                                    returns: pd.DataFrame,
                                    window: int,
                                    current_date: pd.Timestamp) -> pd.Series:
        """计算滚动波动率
        
        Args:
            returns: 收益率数据
            window: 滚动窗口长度
            current_date: 当前日期
            
        Returns:
            当前日期的滚动波动率
        """
        # 检查缓存
        cache_key = (current_date, window) if self._volatility_cache is not None else None
        if cache_key and cache_key in self._volatility_cache:
            return self._volatility_cache[cache_key]
        
        # 计算滚动标准差并年化
        rolling_std = returns.rolling(window=window, min_periods=window).std()
        rolling_volatility = rolling_std * np.sqrt(252)  # 年化波动率
        
        # 获取当前日期的波动率
        try:
            current_volatility = rolling_volatility.loc[current_date]
        except KeyError:
            # 如果当前日期不存在，使用最近的可用日期
            available_dates = rolling_volatility.index
            if len(available_dates) == 0:
                raise DataQualityException("没有可用的波动率数据")
            
            # 找到最接近当前日期的数据
            closest_date = available_dates[available_dates <= current_date][-1]
            current_volatility = rolling_volatility.loc[closest_date]
        
        # 缓存结果
        if cache_key:
            self._volatility_cache[cache_key] = current_volatility
        
        return current_volatility
    
    def _calculate_percentile_ranks(self, 
                                  volatility: pd.Series,
                                  by_industry: bool = False,
                                  industry_mapping: Optional[Dict[str, str]] = None) -> pd.Series:
        """计算波动率分位数排名
        
        Args:
            volatility: 波动率数据
            by_industry: 是否按行业分组
            industry_mapping: 行业映射字典
            
        Returns:
            分位数排名 (0-1)
        """
        # 移除缺失值
        valid_volatility = volatility.dropna()
        
        if len(valid_volatility) == 0:
            raise DataQualityException("没有有效的波动率数据")
        
        if by_industry and industry_mapping:
            # 按行业分组计算分位数
            percentile_ranks = pd.Series(index=volatility.index, dtype=float)
            
            for stock in valid_volatility.index:
                if stock not in industry_mapping:
                    # 如果股票没有行业信息，使用全市场排名
                    rank = (valid_volatility < valid_volatility[stock]).mean()
                else:
                    # 获取同行业股票
                    industry = industry_mapping[stock]
                    industry_stocks = [s for s, ind in industry_mapping.items() 
                                     if ind == industry and s in valid_volatility.index]
                    
                    if len(industry_stocks) <= 1:
                        # 如果行业内股票太少，使用全市场排名
                        rank = (valid_volatility < valid_volatility[stock]).mean()
                    else:
                        # 计算行业内排名
                        industry_volatility = valid_volatility[industry_stocks]
                        rank = (industry_volatility < valid_volatility[stock]).mean()
                
                percentile_ranks[stock] = rank
        else:
            # 全市场分位数排名
            percentile_ranks = valid_volatility.rank(pct=True)
        
        # 对于缺失值，设置为1.0（最高风险，不可交易）
        percentile_ranks = percentile_ranks.reindex(volatility.index, fill_value=1.0)
        
        return percentile_ranks
    
    def _validate_filter_result(self, 
                              tradable_mask: pd.Series,
                              stock_universe: pd.Index) -> None:
        """验证筛选结果的合理性
        
        Args:
            tradable_mask: 可交易掩码
            stock_universe: 股票池
            
        Raises:
            DataQualityException: 筛选结果异常
        """
        if len(tradable_mask) != len(stock_universe):
            raise DataQualityException(
                f"筛选结果长度{len(tradable_mask)}与股票池长度{len(stock_universe)}不匹配"
            )
        
        # 检查筛选比例是否合理
        selection_ratio = tradable_mask.sum() / len(tradable_mask)
        
        if selection_ratio == 0:
            raise DataQualityException("筛选结果为空，没有股票通过筛选")
        
        if selection_ratio > 0.8:
            raise DataQualityException(
                f"筛选比例{selection_ratio:.1%}过高，可能存在参数配置问题"
            )
        
        # 检查数据类型
        if not tradable_mask.dtype == bool:
            raise DataQualityException("筛选结果必须为布尔类型")


# ============================================================================
# IVOL约束筛选器
# ============================================================================

class IVOLConstraintFilter:
    """IVOL约束筛选器
    
    使用五因子回归模型分解特异性波动，区分"好波动"和"坏波动"，
    实现IVOL双重约束筛选功能。
    """
    
    def __init__(self, config: DynamicLowVolConfig):
        """初始化IVOL约束筛选器
        
        Args:
            config: 筛选器配置
        """
        self.config = config
        self.ivol_bad_threshold = config.ivol_bad_threshold
        self.ivol_good_threshold = config.ivol_good_threshold
        
        # 缓存机制
        self._ivol_cache = {} if config.enable_caching else None
        self._factor_cache = {} if config.enable_caching else None
        
        # 导入回归分析库
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            self.LinearRegression = LinearRegression
            self.StandardScaler = StandardScaler
        except ImportError:
            raise ConfigurationException(
                "需要安装scikit-learn库来使用回归模型: pip install scikit-learn"
            )
    
    def apply_ivol_constraint(self, 
                            returns: pd.DataFrame,
                            factor_data: pd.DataFrame,
                            current_date: pd.Timestamp,
                            market_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """应用IVOL双重约束筛选
        
        Args:
            returns: 股票收益率数据 (日期 x 股票)
            factor_data: 因子数据 (日期 x 因子)
            current_date: 当前日期
            market_data: 市场指数数据，用于构建市场因子
            
        Returns:
            通过IVOL约束的股票掩码 (True表示通过筛选)
            
        Raises:
            InsufficientDataException: 数据长度不足
            DataQualityException: 数据质量问题
            ModelFittingException: 回归模型拟合失败
        """
        # 数据质量检查
        self._validate_input_data(returns, factor_data, current_date)
        
        # 构建五因子数据
        five_factors = self._construct_five_factors(
            returns, factor_data, market_data, current_date
        )
        
        # 分解IVOL
        ivol_good, ivol_bad = self.decompose_ivol(returns, five_factors)
        
        # 计算分位数排名
        good_percentiles = self._calculate_ivol_percentiles(ivol_good)
        bad_percentiles = self._calculate_ivol_percentiles(ivol_bad)
        
        # 应用双重约束
        constraint_mask = (
            (bad_percentiles <= self.ivol_bad_threshold) &
            (good_percentiles <= self.ivol_good_threshold)
        )
        
        # 验证筛选结果
        self._validate_constraint_result(constraint_mask, returns.columns)
        
        return constraint_mask.values
    
    def decompose_ivol(self, 
                      returns: pd.DataFrame,
                      five_factors: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """使用五因子回归分解特异性波动
        
        Args:
            returns: 股票收益率数据 (日期 x 股票)
            five_factors: 五因子数据 (日期 x 因子)
            
        Returns:
            (好波动序列, 坏波动序列)
            
        Raises:
            ModelFittingException: 回归模型拟合失败
            DataQualityException: 数据质量问题
        """
        if returns.empty or five_factors.empty:
            raise DataQualityException("收益率数据或因子数据为空")
        
        # 确保数据对齐
        common_dates = returns.index.intersection(five_factors.index)
        if len(common_dates) < 60:  # 至少需要60个观测值
            raise InsufficientDataException(
                f"对齐后的数据长度{len(common_dates)}不足，需要至少60个观测值"
            )
        
        aligned_returns = returns.loc[common_dates]
        aligned_factors = five_factors.loc[common_dates]
        
        ivol_good = {}
        ivol_bad = {}
        
        # 标准化因子数据
        scaler = self.StandardScaler()
        scaled_factors = scaler.fit_transform(aligned_factors)
        scaled_factors_df = pd.DataFrame(
            scaled_factors, 
            index=aligned_factors.index, 
            columns=aligned_factors.columns
        )
        
        for stock in aligned_returns.columns:
            try:
                stock_returns = aligned_returns[stock].dropna()
                
                # 确保有足够的观测值
                if len(stock_returns) < 60:
                    ivol_good[stock] = np.nan
                    ivol_bad[stock] = np.nan
                    continue
                
                # 对齐股票收益率和因子数据
                common_idx = stock_returns.index.intersection(scaled_factors_df.index)
                if len(common_idx) < 60:
                    ivol_good[stock] = np.nan
                    ivol_bad[stock] = np.nan
                    continue
                
                y = stock_returns.loc[common_idx].values
                X = scaled_factors_df.loc[common_idx].values
                
                # 五因子回归
                model = self.LinearRegression()
                model.fit(X, y)
                
                # 计算残差
                residuals = y - model.predict(X)
                residuals_series = pd.Series(residuals, index=common_idx)
                
                # 分解好坏波动
                good_vol, bad_vol = self._decompose_residual_volatility(residuals_series)
                
                ivol_good[stock] = good_vol
                ivol_bad[stock] = bad_vol
                
            except Exception as e:
                # 回归失败时设为NaN
                ivol_good[stock] = np.nan
                ivol_bad[stock] = np.nan
        
        # 转换为Series并处理缺失值
        ivol_good_series = pd.Series(ivol_good, name='ivol_good')
        ivol_bad_series = pd.Series(ivol_bad, name='ivol_bad')
        
        # 对于缺失值，使用历史波动率作为替代
        ivol_good_series = self._fill_missing_ivol(ivol_good_series, returns, 'good')
        ivol_bad_series = self._fill_missing_ivol(ivol_bad_series, returns, 'bad')
        
        return ivol_good_series, ivol_bad_series
    
    def get_ivol_statistics(self, 
                           returns: pd.DataFrame,
                           factor_data: pd.DataFrame,
                           current_date: pd.Timestamp) -> Dict:
        """获取IVOL统计信息
        
        Args:
            returns: 股票收益率数据
            factor_data: 因子数据
            current_date: 当前日期
            
        Returns:
            IVOL统计信息字典
        """
        try:
            five_factors = self._construct_five_factors(returns, factor_data, None, current_date)
            ivol_good, ivol_bad = self.decompose_ivol(returns, five_factors)
            
            statistics = {
                'ivol_good_mean': ivol_good.mean(),
                'ivol_good_std': ivol_good.std(),
                'ivol_good_median': ivol_good.median(),
                'ivol_bad_mean': ivol_bad.mean(),
                'ivol_bad_std': ivol_bad.std(),
                'ivol_bad_median': ivol_bad.median(),
                'good_bad_correlation': ivol_good.corr(ivol_bad),
                'valid_stocks_count': (~(ivol_good.isna() | ivol_bad.isna())).sum(),
                'total_stocks_count': len(returns.columns)
            }
            
            return statistics
            
        except Exception as e:
            return {
                'error': str(e),
                'ivol_good_mean': np.nan,
                'ivol_good_std': np.nan,
                'ivol_good_median': np.nan,
                'ivol_bad_mean': np.nan,
                'ivol_bad_std': np.nan,
                'ivol_bad_median': np.nan,
                'good_bad_correlation': np.nan,
                'valid_stocks_count': 0,
                'total_stocks_count': len(returns.columns) if not returns.empty else 0
            }
    
    def _construct_five_factors(self, 
                              returns: pd.DataFrame,
                              factor_data: pd.DataFrame,
                              market_data: Optional[pd.DataFrame],
                              current_date: pd.Timestamp) -> pd.DataFrame:
        """构建五因子数据
        
        Args:
            returns: 股票收益率数据
            factor_data: 原始因子数据
            market_data: 市场数据
            current_date: 当前日期
            
        Returns:
            五因子数据框 (市场、规模、价值、盈利、投资)
        """
        # 检查缓存
        cache_key = (current_date, 'five_factors') if self._factor_cache else None
        if cache_key and cache_key in self._factor_cache:
            return self._factor_cache[cache_key]
        
        # 构建市场因子 (Market)
        if market_data is not None and not market_data.empty:
            market_factor = market_data.pct_change().iloc[:, 0]
        else:
            # 使用等权重市场收益率作为市场因子
            market_factor = returns.mean(axis=1)
        
        # 构建规模因子 (Size) - 使用市值代理
        # 这里简化处理，使用价格水平作为规模代理
        size_factor = self._construct_size_factor(returns, factor_data)
        
        # 构建价值因子 (Value) - 使用账面市值比代理
        value_factor = self._construct_value_factor(returns, factor_data)
        
        # 构建盈利因子 (Profitability) - 使用ROE代理
        profitability_factor = self._construct_profitability_factor(returns, factor_data)
        
        # 构建投资因子 (Investment) - 使用资产增长率代理
        investment_factor = self._construct_investment_factor(returns, factor_data)
        
        # 组合五因子
        five_factors = pd.DataFrame({
            'Market': market_factor,
            'Size': size_factor,
            'Value': value_factor,
            'Profitability': profitability_factor,
            'Investment': investment_factor
        })
        
        # 移除缺失值
        five_factors = five_factors.dropna()
        
        # 缓存结果
        if cache_key:
            self._factor_cache[cache_key] = five_factors
        
        return five_factors
    
    def _construct_size_factor(self, 
                             returns: pd.DataFrame,
                             factor_data: pd.DataFrame) -> pd.Series:
        """构建规模因子 (SMB - Small Minus Big)
        
        简化实现：使用股票价格水平的倒数作为规模代理
        """
        # 计算累积价格水平（价格指数）
        price_levels = (1 + returns).cumprod()
        
        # 使用最新价格水平的倒数作为规模因子
        size_proxy = 1 / price_levels.iloc[-1]
        
        # 构建多空组合：小市值 - 大市值
        size_median = size_proxy.median()
        small_stocks = size_proxy[size_proxy > size_median]
        big_stocks = size_proxy[size_proxy <= size_median]
        
        # 计算SMB因子时间序列
        smb_series = []
        for date in returns.index:
            small_return = returns.loc[date, small_stocks.index].mean()
            big_return = returns.loc[date, big_stocks.index].mean()
            smb_series.append(small_return - big_return)
        
        return pd.Series(smb_series, index=returns.index, name='Size')
    
    def _construct_value_factor(self, 
                              returns: pd.DataFrame,
                              factor_data: pd.DataFrame) -> pd.Series:
        """构建价值因子 (HML - High Minus Low)
        
        简化实现：使用价格动量的倒数作为价值代理
        """
        # 使用长期收益率的倒数作为价值代理
        long_term_returns = returns.rolling(window=252, min_periods=60).mean()
        value_proxy = -long_term_returns.iloc[-1]  # 负号表示低收益率=高价值
        
        # 构建多空组合：高价值 - 低价值
        value_median = value_proxy.median()
        high_value_stocks = value_proxy[value_proxy > value_median]
        low_value_stocks = value_proxy[value_proxy <= value_median]
        
        # 计算HML因子时间序列
        hml_series = []
        for date in returns.index:
            high_return = returns.loc[date, high_value_stocks.index].mean()
            low_return = returns.loc[date, low_value_stocks.index].mean()
            hml_series.append(high_return - low_return)
        
        return pd.Series(hml_series, index=returns.index, name='Value')
    
    def _construct_profitability_factor(self, 
                                      returns: pd.DataFrame,
                                      factor_data: pd.DataFrame) -> pd.Series:
        """构建盈利因子 (RMW - Robust Minus Weak)
        
        简化实现：使用短期收益率稳定性作为盈利能力代理
        """
        # 使用收益率的夏普比率作为盈利能力代理
        rolling_mean = returns.rolling(window=60, min_periods=20).mean()
        rolling_std = returns.rolling(window=60, min_periods=20).std()
        sharpe_proxy = (rolling_mean / rolling_std).iloc[-1]
        
        # 构建多空组合：高盈利 - 低盈利
        profit_median = sharpe_proxy.median()
        robust_stocks = sharpe_proxy[sharpe_proxy > profit_median]
        weak_stocks = sharpe_proxy[sharpe_proxy <= profit_median]
        
        # 计算RMW因子时间序列
        rmw_series = []
        for date in returns.index:
            robust_return = returns.loc[date, robust_stocks.index].mean()
            weak_return = returns.loc[date, weak_stocks.index].mean()
            rmw_series.append(robust_return - weak_return)
        
        return pd.Series(rmw_series, index=returns.index, name='Profitability')
    
    def _construct_investment_factor(self, 
                                   returns: pd.DataFrame,
                                   factor_data: pd.DataFrame) -> pd.Series:
        """构建投资因子 (CMA - Conservative Minus Aggressive)
        
        简化实现：使用收益率波动性作为投资风格代理
        """
        # 使用波动率作为投资激进程度代理
        volatility_proxy = returns.rolling(window=60, min_periods=20).std().iloc[-1]
        
        # 构建多空组合：保守投资 - 激进投资
        vol_median = volatility_proxy.median()
        conservative_stocks = volatility_proxy[volatility_proxy <= vol_median]
        aggressive_stocks = volatility_proxy[volatility_proxy > vol_median]
        
        # 计算CMA因子时间序列
        cma_series = []
        for date in returns.index:
            conservative_return = returns.loc[date, conservative_stocks.index].mean()
            aggressive_return = returns.loc[date, aggressive_stocks.index].mean()
            cma_series.append(conservative_return - aggressive_return)
        
        return pd.Series(cma_series, index=returns.index, name='Investment')
    
    def _decompose_residual_volatility(self, residuals: pd.Series) -> Tuple[float, float]:
        """分解残差波动为好波动和坏波动
        
        Args:
            residuals: 回归残差序列
            
        Returns:
            (好波动, 坏波动)
        """
        # 分离正负残差
        positive_residuals = residuals[residuals > 0]
        negative_residuals = residuals[residuals < 0]
        
        # 计算好波动（正残差的标准差）
        if len(positive_residuals) > 1:
            good_volatility = positive_residuals.std()
        else:
            good_volatility = 0.0
        
        # 计算坏波动（负残差的标准差）
        if len(negative_residuals) > 1:
            bad_volatility = abs(negative_residuals.std())
        else:
            bad_volatility = 0.0
        
        # 年化处理
        good_volatility *= np.sqrt(252)
        bad_volatility *= np.sqrt(252)
        
        return good_volatility, bad_volatility
    
    def _calculate_ivol_percentiles(self, ivol_series: pd.Series) -> pd.Series:
        """计算IVOL分位数排名
        
        Args:
            ivol_series: IVOL数据序列
            
        Returns:
            分位数排名序列 (0-1)
        """
        # 移除缺失值
        valid_ivol = ivol_series.dropna()
        
        if len(valid_ivol) == 0:
            return pd.Series(1.0, index=ivol_series.index)
        
        # 计算分位数排名
        percentiles = valid_ivol.rank(pct=True)
        
        # 对缺失值设置为1.0（最高风险）
        percentiles = percentiles.reindex(ivol_series.index, fill_value=1.0)
        
        return percentiles
    
    def _fill_missing_ivol(self, 
                          ivol_series: pd.Series,
                          returns: pd.DataFrame,
                          ivol_type: str) -> pd.Series:
        """填充缺失的IVOL值
        
        Args:
            ivol_series: IVOL序列
            returns: 收益率数据
            ivol_type: IVOL类型 ('good' 或 'bad')
            
        Returns:
            填充后的IVOL序列
        """
        filled_series = ivol_series.copy()
        missing_mask = ivol_series.isna()
        
        if missing_mask.sum() == 0:
            return filled_series
        
        # 对于缺失值，使用历史波动率的一半作为替代
        for stock in ivol_series.index[missing_mask]:
            if stock in returns.columns:
                stock_returns = returns[stock].dropna()
                if len(stock_returns) > 0:
                    historical_vol = stock_returns.std() * np.sqrt(252)
                    # 好波动使用较小值，坏波动使用较大值
                    if ivol_type == 'good':
                        filled_series[stock] = historical_vol * 0.3
                    else:  # bad
                        filled_series[stock] = historical_vol * 0.7
                else:
                    # 如果完全没有数据，设置为中等水平
                    filled_series[stock] = 0.2 if ivol_type == 'good' else 0.3
            else:
                filled_series[stock] = 0.2 if ivol_type == 'good' else 0.3
        
        return filled_series
    
    def _validate_input_data(self, 
                           returns: pd.DataFrame,
                           factor_data: pd.DataFrame,
                           current_date: pd.Timestamp) -> None:
        """验证输入数据质量
        
        Args:
            returns: 收益率数据
            factor_data: 因子数据
            current_date: 当前日期
            
        Raises:
            DataQualityException: 数据质量问题
            InsufficientDataException: 数据长度不足
        """
        if returns.empty:
            raise DataQualityException("收益率数据为空")
        
        if factor_data.empty:
            raise DataQualityException("因子数据为空")
        
        if len(returns) < 60:
            raise InsufficientDataException(
                f"收益率数据长度{len(returns)}不足，需要至少60个观测值"
            )
        
        # 检查数据对齐
        common_dates = returns.index.intersection(factor_data.index)
        if len(common_dates) < 60:
            raise InsufficientDataException(
                f"收益率和因子数据对齐后长度{len(common_dates)}不足，需要至少60个观测值"
            )
        
        # 检查当前日期是否在数据范围内
        if current_date not in returns.index:
            if current_date < returns.index.min() or current_date > returns.index.max():
                raise DataQualityException(
                    f"当前日期{current_date}超出收益率数据范围"
                    f"[{returns.index.min()}, {returns.index.max()}]"
                )
    
    def _validate_constraint_result(self, 
                                  constraint_mask: pd.Series,
                                  stock_universe: pd.Index) -> None:
        """验证约束筛选结果
        
        Args:
            constraint_mask: 约束筛选掩码
            stock_universe: 股票池
            
        Raises:
            DataQualityException: 筛选结果异常
        """
        if len(constraint_mask) != len(stock_universe):
            raise DataQualityException(
                f"约束筛选结果长度{len(constraint_mask)}与股票池长度{len(stock_universe)}不匹配"
            )
        
        # 检查筛选比例
        selection_ratio = constraint_mask.sum() / len(constraint_mask)
        
        if selection_ratio == 0:
            raise DataQualityException("IVOL约束筛选结果为空，没有股票通过筛选")
        
        if selection_ratio > 0.9:
            raise DataQualityException(
                f"IVOL约束筛选比例{selection_ratio:.1%}过高，可能存在参数配置问题"
            )
        
        # 检查数据类型
        if not constraint_mask.dtype == bool:
            raise DataQualityException("约束筛选结果必须为布尔类型")


# ============================================================================
# GARCH波动率预测器
# ============================================================================

class GARCHVolatilityPredictor:
    """GARCH波动率预测器
    
    使用GARCH(1,1)+t分布模型预测股票未来波动率，
    支持预测结果缓存机制避免重复计算。
    """
    
    def __init__(self, config: DynamicLowVolConfig):
        """初始化GARCH预测器
        
        Args:
            config: 筛选器配置
        """
        self.config = config
        self.garch_window = config.garch_window
        self.forecast_horizon = config.forecast_horizon
        self.enable_caching = config.enable_caching
        
        # 预测结果缓存
        self._prediction_cache = {} if self.enable_caching else None
        self._model_cache = {} if self.enable_caching else None
        
        # 导入GARCH相关库
        try:
            from arch import arch_model
            self.arch_model = arch_model
        except ImportError:
            raise ConfigurationException(
                "需要安装arch库来使用GARCH模型: pip install arch"
            )
    
    def predict_volatility(self, 
                          returns: pd.Series,
                          stock_code: str,
                          current_date: pd.Timestamp,
                          horizon: Optional[int] = None) -> float:
        """预测单只股票的未来波动率
        
        Args:
            returns: 股票收益率时间序列
            stock_code: 股票代码
            current_date: 当前日期
            horizon: 预测期限，默认使用配置中的forecast_horizon
            
        Returns:
            预测的年化波动率
            
        Raises:
            InsufficientDataException: 数据长度不足
            DataQualityException: 数据质量问题
            ModelFittingException: GARCH模型拟合失败
        """
        if horizon is None:
            horizon = self.forecast_horizon
        
        # 数据质量检查
        self._validate_input_data(returns, stock_code, horizon)
        
        # 检查缓存
        cache_key = (stock_code, current_date, horizon) if self.enable_caching else None
        if cache_key and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        # 准备GARCH拟合数据
        garch_data = self._prepare_garch_data(returns, current_date)
        
        # 拟合GARCH模型
        fitted_model = self._fit_garch_model(garch_data, stock_code)
        
        # 预测波动率
        predicted_vol = self._forecast_volatility(fitted_model, horizon)
        
        # 缓存结果
        if cache_key:
            self._prediction_cache[cache_key] = predicted_vol
        
        return predicted_vol
    
    def predict_batch_volatility(self, 
                               returns_df: pd.DataFrame,
                               current_date: pd.Timestamp,
                               horizon: Optional[int] = None) -> pd.Series:
        """批量预测多只股票的波动率
        
        Args:
            returns_df: 收益率数据框 (日期 x 股票)
            current_date: 当前日期
            horizon: 预测期限
            
        Returns:
            各股票的预测波动率序列
            
        Raises:
            DataQualityException: 数据质量问题
        """
        if returns_df.empty:
            raise DataQualityException("收益率数据为空")
        
        predicted_vols = {}
        
        for stock_code in returns_df.columns:
            try:
                stock_returns = returns_df[stock_code].dropna()
                if len(stock_returns) >= self.garch_window:
                    vol = self.predict_volatility(
                        stock_returns, stock_code, current_date, horizon
                    )
                    predicted_vols[stock_code] = vol
                else:
                    # 数据不足时使用历史波动率作为预测
                    historical_vol = stock_returns.std() * np.sqrt(252)
                    predicted_vols[stock_code] = historical_vol
            except (ModelFittingException, InsufficientDataException):
                # 模型拟合失败时使用历史波动率
                stock_returns = returns_df[stock_code].dropna()
                if len(stock_returns) > 0:
                    historical_vol = stock_returns.std() * np.sqrt(252)
                    predicted_vols[stock_code] = historical_vol
                else:
                    predicted_vols[stock_code] = np.nan
        
        return pd.Series(predicted_vols)
    
    def get_model_diagnostics(self, 
                            returns: pd.Series,
                            stock_code: str) -> Dict:
        """获取GARCH模型诊断信息
        
        Args:
            returns: 股票收益率
            stock_code: 股票代码
            
        Returns:
            模型诊断信息字典
        """
        try:
            garch_data = returns.dropna().iloc[-self.garch_window:]
            fitted_model = self._fit_garch_model(garch_data, stock_code)
            
            diagnostics = {
                'converged': fitted_model.converged,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'log_likelihood': fitted_model.loglikelihood,
                'num_observations': fitted_model.nobs,
                'alpha': fitted_model.params.get('alpha[1]', np.nan),
                'beta': fitted_model.params.get('beta[1]', np.nan),
                'omega': fitted_model.params.get('omega', np.nan)
            }
            
            return diagnostics
            
        except Exception as e:
            return {
                'converged': False,
                'error': str(e),
                'aic': np.nan,
                'bic': np.nan,
                'log_likelihood': np.nan,
                'num_observations': len(returns),
                'alpha': np.nan,
                'beta': np.nan,
                'omega': np.nan
            }
    
    def clear_cache(self, older_than_days: Optional[int] = None):
        """清理缓存
        
        Args:
            older_than_days: 清理多少天前的缓存，None表示清理全部
        """
        if not self.enable_caching:
            return
        
        if older_than_days is None:
            # 清理全部缓存
            if self._prediction_cache:
                self._prediction_cache.clear()
            if self._model_cache:
                self._model_cache.clear()
        else:
            # 清理过期缓存
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=older_than_days)
            
            # 清理预测缓存
            if self._prediction_cache:
                expired_keys = [
                    key for key in self._prediction_cache.keys()
                    if len(key) >= 2 and isinstance(key[1], pd.Timestamp) and key[1] < cutoff_date
                ]
                for key in expired_keys:
                    del self._prediction_cache[key]
            
            # 清理模型缓存
            if self._model_cache:
                expired_keys = [
                    key for key in self._model_cache.keys()
                    if len(key) >= 2 and isinstance(key[1], pd.Timestamp) and key[1] < cutoff_date
                ]
                for key in expired_keys:
                    del self._model_cache[key]
    
    def _validate_input_data(self, 
                           returns: pd.Series,
                           stock_code: str,
                           horizon: int) -> None:
        """验证输入数据
        
        Args:
            returns: 收益率数据
            stock_code: 股票代码
            horizon: 预测期限
            
        Raises:
            DataQualityException: 数据质量问题
            InsufficientDataException: 数据不足
            ConfigurationException: 配置错误
        """
        if not isinstance(returns, pd.Series):
            raise DataQualityException("收益率数据必须为Series类型")
        
        if returns.empty:
            raise DataQualityException(f"股票{stock_code}收益率数据为空")
        
        if not isinstance(stock_code, str) or not stock_code:
            raise DataQualityException("股票代码必须为非空字符串")
        
        if not isinstance(horizon, int) or horizon <= 0:
            raise ConfigurationException(f"预测期限必须为正整数，当前为{horizon}")
        
        # 检查数据长度
        valid_returns = returns.dropna()
        if len(valid_returns) < self.garch_window:
            raise InsufficientDataException(
                f"股票{stock_code}有效数据长度{len(valid_returns)}不足，"
                f"需要至少{self.garch_window}个观测值"
            )
        
        # 检查缺失值比例
        missing_ratio = returns.isna().sum() / len(returns)
        if missing_ratio > 0.2:
            raise DataQualityException(
                f"股票{stock_code}缺失值比例{missing_ratio:.1%}过高"
            )
        
        # 检查收益率方差
        if valid_returns.var() == 0 or np.isclose(valid_returns.var(), 0, atol=1e-10):
            raise DataQualityException(f"股票{stock_code}收益率方差为0，无法拟合GARCH模型")
        
        # 检查极端值
        extreme_threshold = 0.5  # 50%的单日收益率
        extreme_count = (np.abs(valid_returns) > extreme_threshold).sum()
        if extreme_count > len(valid_returns) * 0.05:  # 超过5%的观测值
            raise DataQualityException(
                f"股票{stock_code}极端收益率过多：{extreme_count}个，"
                f"占比{extreme_count/len(valid_returns):.1%}"
            )
    
    def _prepare_garch_data(self, 
                          returns: pd.Series,
                          current_date: pd.Timestamp) -> pd.Series:
        """准备GARCH拟合数据
        
        Args:
            returns: 原始收益率数据
            current_date: 当前日期
            
        Returns:
            用于GARCH拟合的数据
        """
        # 获取截止到当前日期的数据
        if current_date in returns.index:
            data_end_idx = returns.index.get_loc(current_date)
            available_data = returns.iloc[:data_end_idx + 1]
        else:
            # 如果当前日期不在索引中，使用最近的可用数据
            available_dates = returns.index[returns.index <= current_date]
            if len(available_dates) == 0:
                raise InsufficientDataException(f"没有截止到{current_date}的可用数据")
            available_data = returns.loc[:available_dates[-1]]
        
        # 取最近的garch_window个观测值
        garch_data = available_data.dropna().iloc[-self.garch_window:]
        
        # 数据预处理：移除极端异常值
        garch_data = self._winsorize_returns(garch_data)
        
        return garch_data
    
    def _fit_garch_model(self, 
                        returns: pd.Series,
                        stock_code: str) -> object:
        """拟合GARCH(1,1)+t分布模型
        
        Args:
            returns: 收益率数据
            stock_code: 股票代码
            
        Returns:
            拟合后的GARCH模型
            
        Raises:
            ModelFittingException: 模型拟合失败
        """
        try:
            # 检查模型缓存
            cache_key = (stock_code, returns.index[-1]) if self.enable_caching else None
            if cache_key and cache_key in self._model_cache:
                return self._model_cache[cache_key]
            
            # 创建GARCH(1,1)模型，使用t分布
            model = self.arch_model(
                returns * 100,  # 转换为百分比以提高数值稳定性
                vol='GARCH',
                p=1,  # GARCH项数
                q=1,  # ARCH项数
                dist='t'  # 使用t分布
            )
            
            # 拟合模型
            fitted_model = model.fit(
                disp='off',  # 不显示拟合过程
                show_warning=False,  # 不显示警告
                options={'maxiter': 1000}  # 最大迭代次数
            )
            
            # 检查模型收敛性
            if not fitted_model.converged:
                raise ModelFittingException(f"股票{stock_code}的GARCH模型未收敛")
            
            # 检查参数合理性
            self._validate_garch_parameters(fitted_model, stock_code)
            
            # 缓存模型
            if cache_key:
                self._model_cache[cache_key] = fitted_model
            
            return fitted_model
            
        except Exception as e:
            if isinstance(e, ModelFittingException):
                raise
            else:
                raise ModelFittingException(f"股票{stock_code}的GARCH模型拟合失败: {str(e)}")
    
    def _forecast_volatility(self, 
                           fitted_model: object,
                           horizon: int) -> float:
        """使用拟合的GARCH模型预测波动率
        
        Args:
            fitted_model: 拟合后的GARCH模型
            horizon: 预测期限
            
        Returns:
            预测的年化波动率
            
        Raises:
            ModelFittingException: 预测失败
        """
        try:
            # 进行波动率预测
            forecast = fitted_model.forecast(horizon=horizon, method='simulation')
            
            # 获取预测的方差
            forecast_variance = forecast.variance.iloc[-1, :horizon].mean()
            
            # 转换为年化波动率
            # 注意：由于输入时乘以了100，这里需要除以100
            predicted_vol = np.sqrt(forecast_variance) / 100 * np.sqrt(252)
            
            # 验证预测结果
            if not np.isfinite(predicted_vol) or predicted_vol <= 0:
                raise ModelFittingException(f"GARCH预测结果异常: {predicted_vol}")
            
            # 合理性检查：年化波动率应该在合理范围内
            if predicted_vol > 2.0:  # 200%年化波动率
                raise ModelFittingException(f"预测波动率{predicted_vol:.1%}过高，可能存在模型问题")
            
            if predicted_vol < 0.01:  # 1%年化波动率
                raise ModelFittingException(f"预测波动率{predicted_vol:.1%}过低，可能存在模型问题")
            
            return predicted_vol
            
        except Exception as e:
            if isinstance(e, ModelFittingException):
                raise
            else:
                raise ModelFittingException(f"GARCH波动率预测失败: {str(e)}")
    
    def _validate_garch_parameters(self, 
                                 fitted_model: object,
                                 stock_code: str) -> None:
        """验证GARCH模型参数的合理性
        
        Args:
            fitted_model: 拟合后的模型
            stock_code: 股票代码
            
        Raises:
            ModelFittingException: 参数不合理
        """
        params = fitted_model.params
        
        # 检查关键参数是否存在
        required_params = ['omega', 'alpha[1]', 'beta[1]']
        for param in required_params:
            if param not in params:
                raise ModelFittingException(f"股票{stock_code}的GARCH模型缺少参数{param}")
        
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        
        # 检查参数符号
        if omega <= 0:
            raise ModelFittingException(f"股票{stock_code}的omega参数{omega}必须为正")
        
        if alpha < 0:
            raise ModelFittingException(f"股票{stock_code}的alpha参数{alpha}不能为负")
        
        if beta < 0:
            raise ModelFittingException(f"股票{stock_code}的beta参数{beta}不能为负")
        
        # 检查平稳性条件
        if alpha + beta >= 1:
            raise ModelFittingException(
                f"股票{stock_code}的GARCH模型不满足平稳性条件: alpha({alpha}) + beta({beta}) = {alpha + beta} >= 1"
            )
        
        # 检查参数合理性
        if alpha > 0.5:
            raise ModelFittingException(f"股票{stock_code}的alpha参数{alpha}过大，可能存在过拟合")
        
        if beta > 0.99:
            raise ModelFittingException(f"股票{stock_code}的beta参数{beta}过大，接近单位根")
    
    def _winsorize_returns(self, 
                         returns: pd.Series,
                         lower_percentile: float = 0.01,
                         upper_percentile: float = 0.99) -> pd.Series:
        """对收益率进行缩尾处理，移除极端异常值
        
        Args:
            returns: 原始收益率
            lower_percentile: 下分位数
            upper_percentile: 上分位数
            
        Returns:
            处理后的收益率
        """
        lower_bound = returns.quantile(lower_percentile)
        upper_bound = returns.quantile(upper_percentile)
        
        winsorized_returns = returns.clip(lower=lower_bound, upper=upper_bound)
        
        return winsorized_returns


# ============================================================================
# 市场状态检测器
# ============================================================================

class MarketRegimeDetector:
    """市场状态检测器
    
    使用HMM模型识别市场波动状态（低/中/高），
    为动态阈值调整提供市场状态信号。
    """
    
    def __init__(self, config: DynamicLowVolConfig):
        """初始化市场状态检测器
        
        Args:
            config: 筛选器配置
            
        Raises:
            ConfigurationException: 配置参数错误或缺少依赖库
        """
        self.config = config
        self.detection_window = config.regime_detection_window
        self.model_type = config.regime_model_type
        
        # 状态映射
        self.regime_mapping = {0: "低", 1: "中", 2: "高"}
        self.reverse_mapping = {"低": 0, "中": 1, "高": 2}
        
        # 缓存机制
        self._regime_cache = {} if config.enable_caching else None
        self._model_cache = {} if config.enable_caching else None
        
        # 导入HMM库
        try:
            from hmmlearn import hmm
            self.GaussianHMM = hmm.GaussianHMM
        except ImportError:
            raise ConfigurationException(
                "需要安装hmmlearn库来使用HMM模型: pip install hmmlearn"
            )
        
        # 导入其他必需库
        try:
            from sklearn.preprocessing import StandardScaler
            self.StandardScaler = StandardScaler
        except ImportError:
            raise ConfigurationException(
                "需要安装scikit-learn库: pip install scikit-learn"
            )
    
    def detect_regime(self, 
                     market_returns: pd.Series,
                     current_date: pd.Timestamp) -> str:
        """检测当前市场波动状态
        
        Args:
            market_returns: 市场收益率时间序列
            current_date: 当前日期
            
        Returns:
            市场状态字符串 ("低", "中", "高")
            
        Raises:
            InsufficientDataException: 数据长度不足
            DataQualityException: 数据质量问题
            RegimeDetectionException: 状态检测失败
        """
        # 数据质量检查
        self._validate_input_data(market_returns, current_date)
        
        # 检查缓存
        cache_key = (current_date, len(market_returns)) if self._regime_cache is not None else None
        if cache_key and cache_key in self._regime_cache:
            return self._regime_cache[cache_key]
        
        # 准备检测数据
        detection_data = self._prepare_detection_data(market_returns, current_date)
        
        # 拟合HMM模型
        hmm_model = self._fit_hmm_model(detection_data)
        
        # 预测当前状态
        current_state = self._predict_current_state(hmm_model, detection_data)
        
        # 验证状态结果
        regime = self._validate_and_map_state(current_state)
        
        # 缓存结果
        if cache_key:
            self._regime_cache[cache_key] = regime
        
        return regime
    
    def get_regime_probabilities(self, 
                               market_returns: pd.Series,
                               current_date: pd.Timestamp) -> Dict[str, float]:
        """获取各状态的概率分布
        
        Args:
            market_returns: 市场收益率时间序列
            current_date: 当前日期
            
        Returns:
            各状态概率字典 {"低": prob1, "中": prob2, "高": prob3}
            
        Raises:
            InsufficientDataException: 数据长度不足
            RegimeDetectionException: 状态检测失败
        """
        # 数据质量检查
        self._validate_input_data(market_returns, current_date)
        
        # 准备检测数据
        detection_data = self._prepare_detection_data(market_returns, current_date)
        
        # 拟合HMM模型
        hmm_model = self._fit_hmm_model(detection_data)
        
        # 获取状态概率
        state_probs = hmm_model.predict_proba(detection_data.values.reshape(-1, 1))
        
        # 获取最后一个时点的概率
        current_probs = state_probs[-1]
        
        # 映射到状态名称
        prob_dict = {}
        for state_idx, prob in enumerate(current_probs):
            regime_name = self.regime_mapping[state_idx]
            prob_dict[regime_name] = float(prob)
        
        return prob_dict
    
    def get_regime_statistics(self, 
                            market_returns: pd.Series,
                            current_date: pd.Timestamp) -> Dict:
        """获取状态检测统计信息
        
        Args:
            market_returns: 市场收益率时间序列
            current_date: 当前日期
            
        Returns:
            统计信息字典
        """
        # 准备数据
        detection_data = self._prepare_detection_data(market_returns, current_date)
        
        # 拟合模型
        hmm_model = self._fit_hmm_model(detection_data)
        
        # 预测所有状态
        states = hmm_model.predict(detection_data.values.reshape(-1, 1))
        
        # 计算统计信息
        statistics = {
            'current_regime': self.regime_mapping[states[-1]],
            'regime_distribution': {},
            'transition_matrix': hmm_model.transmat_.tolist(),
            'means': hmm_model.means_.flatten().tolist(),
            'covariances': hmm_model.covars_.flatten().tolist(),
            'model_score': hmm_model.score(detection_data.values.reshape(-1, 1)),
            'n_iter': hmm_model.n_iter_,
            'converged': hmm_model.monitor_.converged
        }
        
        # 计算状态分布
        for state_idx in range(3):
            regime_name = self.regime_mapping[state_idx]
            count = np.sum(states == state_idx)
            statistics['regime_distribution'][regime_name] = count / len(states)
        
        return statistics
    
    def _validate_input_data(self, 
                           market_returns: pd.Series,
                           current_date: pd.Timestamp) -> None:
        """验证输入数据质量
        
        Args:
            market_returns: 市场收益率数据
            current_date: 当前日期
            
        Raises:
            DataQualityException: 数据质量问题
            InsufficientDataException: 数据长度不足
        """
        if not isinstance(market_returns, pd.Series):
            raise DataQualityException("市场收益率数据必须为Series类型")
        
        if market_returns.empty:
            raise DataQualityException("市场收益率数据为空")
        
        if len(market_returns) < self.detection_window:
            raise InsufficientDataException(
                f"市场数据长度{len(market_returns)}不足，需要至少{self.detection_window}个观测值"
            )
        
        # 检查缺失值比例
        missing_ratio = market_returns.isna().sum() / len(market_returns)
        if missing_ratio > 0.1:
            raise DataQualityException(
                f"市场收益率缺失值比例{missing_ratio:.2%}过高，超过10%阈值"
            )
        
        # 检查数据方差
        valid_returns = market_returns.dropna()
        if len(valid_returns) == 0:
            raise DataQualityException("没有有效的市场收益率数据")
        
        if np.isclose(valid_returns.var(), 0, atol=1e-10):
            raise DataQualityException("市场收益率方差为0，无法进行状态检测")
        
        # 检查极端值比例
        extreme_threshold = 0.2  # 20%的单日收益率阈值
        extreme_count = (np.abs(valid_returns) > extreme_threshold).sum()
        extreme_ratio = extreme_count / len(valid_returns)
        
        if extreme_ratio > 0.05:  # 5%的极端值阈值
            raise DataQualityException(
                f"极端收益率（绝对值>{extreme_threshold:.0%}）比例{extreme_ratio:.2%}过高"
            )
        
        # 检查当前日期是否在数据范围内
        if current_date not in market_returns.index:
            # 检查是否有接近的日期
            available_dates = market_returns.index
            if isinstance(available_dates, pd.DatetimeIndex):
                closest_dates = available_dates[available_dates <= current_date]
                if len(closest_dates) == 0:
                    raise DataQualityException(f"当前日期{current_date}早于所有可用数据")
    
    def _prepare_detection_data(self, 
                              market_returns: pd.Series,
                              current_date: pd.Timestamp) -> pd.Series:
        """准备状态检测数据
        
        Args:
            market_returns: 市场收益率数据
            current_date: 当前日期
            
        Returns:
            用于状态检测的数据
        """
        # 获取截止到当前日期的数据
        available_dates = market_returns.index[market_returns.index <= current_date]
        if len(available_dates) == 0:
            raise DataQualityException("没有可用的历史数据")
        
        # 取最近的检测窗口长度的数据
        end_idx = len(available_dates) - 1
        start_idx = max(0, end_idx - self.detection_window + 1)
        
        selected_dates = available_dates[start_idx:end_idx + 1]
        detection_data = market_returns.loc[selected_dates]
        
        # 前向填充缺失值
        detection_data = detection_data.ffill()
        
        # 如果仍有缺失值，后向填充
        if detection_data.isna().any():
            detection_data = detection_data.bfill()
        
        # 计算滚动波动率作为特征
        # 使用5日滚动波动率作为HMM的观测变量
        rolling_vol = detection_data.rolling(window=5, min_periods=1).std()
        rolling_vol = rolling_vol * np.sqrt(252)  # 年化
        
        # 处理可能的NaN值
        if rolling_vol.isna().any():
            rolling_vol = rolling_vol.fillna(rolling_vol.mean())
        
        # 标准化处理
        scaler = self.StandardScaler()
        scaled_vol = scaler.fit_transform(rolling_vol.values.reshape(-1, 1)).flatten()
        
        # 确保没有NaN值
        scaled_vol = np.nan_to_num(scaled_vol, nan=0.0)
        
        return pd.Series(scaled_vol, index=detection_data.index)
    
    def _fit_hmm_model(self, detection_data: pd.Series):
        """拟合HMM模型
        
        Args:
            detection_data: 检测数据
            
        Returns:
            拟合好的HMM模型
            
        Raises:
            RegimeDetectionException: 模型拟合失败
        """
        # 检查模型缓存
        cache_key = (len(detection_data), detection_data.iloc[-1]) if self._model_cache is not None else None
        if cache_key and cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        try:
            # 创建HMM模型
            model = self.GaussianHMM(
                n_components=3,  # 3个状态：低、中、高
                covariance_type="full",
                n_iter=100,
                tol=1e-4,
                random_state=42
            )
            
            # 准备训练数据
            X = detection_data.values.reshape(-1, 1)
            
            # 拟合模型
            model.fit(X)
            
            # 检查收敛性
            if not hasattr(model, 'monitor_') or not model.monitor_.converged:
                raise RegimeDetectionException("HMM模型未收敛")
            
            # 验证模型参数合理性
            self._validate_hmm_parameters(model)
            
            # 缓存模型
            if cache_key:
                self._model_cache[cache_key] = model
            
            return model
            
        except Exception as e:
            if isinstance(e, RegimeDetectionException):
                raise
            else:
                raise RegimeDetectionException(f"HMM模型拟合失败: {str(e)}")
    
    def _validate_hmm_parameters(self, model) -> None:
        """验证HMM模型参数的合理性
        
        Args:
            model: 拟合好的HMM模型
            
        Raises:
            RegimeDetectionException: 模型参数不合理
        """
        # 检查转移矩阵
        if not np.allclose(model.transmat_.sum(axis=1), 1.0, rtol=1e-3):
            raise RegimeDetectionException("转移矩阵行和不为1")
        
        if np.any(model.transmat_ < 0):
            raise RegimeDetectionException("转移矩阵包含负值")
        
        # 检查初始状态概率
        if not np.allclose(model.startprob_.sum(), 1.0, rtol=1e-3):
            raise RegimeDetectionException("初始状态概率和不为1")
        
        if np.any(model.startprob_ < 0):
            raise RegimeDetectionException("初始状态概率包含负值")
        
        # 检查均值参数
        means = model.means_.flatten()
        if np.any(~np.isfinite(means)):
            raise RegimeDetectionException("模型均值参数包含无穷值或NaN")
        
        # 检查协方差参数
        covars = model.covars_.flatten()
        if np.any(covars <= 0):
            raise RegimeDetectionException("协方差参数必须为正")
        
        if np.any(~np.isfinite(covars)):
            raise RegimeDetectionException("协方差参数包含无穷值或NaN")
        
        # 检查状态区分度
        # 均值应该有明显差异
        sorted_means = np.sort(means)
        min_diff = np.min(np.diff(sorted_means))
        if min_diff < 0.1:
            raise RegimeDetectionException(f"状态均值差异{min_diff:.3f}过小，状态区分度不足")
    
    def _predict_current_state(self, model, detection_data: pd.Series) -> int:
        """预测当前状态
        
        Args:
            model: 拟合好的HMM模型
            detection_data: 检测数据
            
        Returns:
            当前状态索引 (0, 1, 2)
            
        Raises:
            RegimeDetectionException: 状态预测失败
        """
        try:
            # 预测状态序列
            X = detection_data.values.reshape(-1, 1)
            states = model.predict(X)
            
            # 获取当前状态
            current_state = states[-1]
            
            # 验证状态有效性
            if current_state not in [0, 1, 2]:
                raise RegimeDetectionException(f"检测到无效状态: {current_state}")
            
            return int(current_state)
            
        except Exception as e:
            if isinstance(e, RegimeDetectionException):
                raise
            else:
                raise RegimeDetectionException(f"状态预测失败: {str(e)}")
    
    def _validate_and_map_state(self, state_idx: int) -> str:
        """验证并映射状态
        
        Args:
            state_idx: 状态索引
            
        Returns:
            状态名称
            
        Raises:
            RegimeDetectionException: 状态无效
        """
        if state_idx not in self.regime_mapping:
            raise RegimeDetectionException(f"检测到无效状态索引: {state_idx}")
        
        regime = self.regime_mapping[state_idx]
        
        # 验证状态名称
        if regime not in ["低", "中", "高"]:
            raise RegimeDetectionException(f"映射到无效状态名称: {regime}")
        
        return regime
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> None:
        """清理缓存
        
        Args:
            older_than_days: 清理多少天前的缓存，None表示清理全部
        """
        if self._regime_cache is None or self._model_cache is None:
            return
        
        if older_than_days is None:
            # 清理全部缓存
            self._regime_cache.clear()
            self._model_cache.clear()
        else:
            # 清理过期缓存
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=older_than_days)
            
            # 清理状态缓存
            expired_keys = [
                key for key in self._regime_cache.keys()
                if isinstance(key, tuple) and len(key) >= 1 and 
                isinstance(key[0], pd.Timestamp) and key[0] < cutoff_date
            ]
            for key in expired_keys:
                del self._regime_cache[key]
            
            # 清理模型缓存
            expired_keys = [
                key for key in self._model_cache.keys()
                if isinstance(key, tuple) and len(key) >= 1 and 
                isinstance(key[0], pd.Timestamp) and key[0] < cutoff_date
            ]
            for key in expired_keys:
                del self._model_cache[key]


# ============================================================================
# 状态感知阈值调节器
# ============================================================================

class RegimeAwareThresholdAdjuster:
    """状态感知阈值调节器
    
    根据市场状态动态调整筛选阈值，实现不同波动状态下的参数配置逻辑。
    在高波动状态下收紧阈值，在低波动状态下放宽阈值。
    """
    
    def __init__(self, config: DynamicLowVolConfig):
        """初始化状态感知阈值调节器
        
        Args:
            config: 筛选器配置
            
        Raises:
            ConfigurationException: 配置参数错误
        """
        self.config = config
        
        # 默认阈值配置
        self.default_thresholds = {
            "percentile_cut": config.percentile_thresholds.get("中", 0.3),
            "target_vol": 0.40,
            "ivol_bad_threshold": config.ivol_bad_threshold,
            "ivol_good_threshold": config.ivol_good_threshold,
            "garch_confidence": 0.95
        }
        
        # 状态特定阈值配置
        self.regime_thresholds = {
            "高": {
                "percentile_cut": config.percentile_thresholds.get("高", 0.2),
                "target_vol": 0.35,
                "ivol_bad_threshold": config.ivol_bad_threshold * 0.8,  # 更严格
                "ivol_good_threshold": config.ivol_good_threshold * 0.9,  # 更严格
                "garch_confidence": 0.99  # 更高置信度
            },
            "中": {
                "percentile_cut": config.percentile_thresholds.get("中", 0.3),
                "target_vol": 0.40,
                "ivol_bad_threshold": config.ivol_bad_threshold,
                "ivol_good_threshold": config.ivol_good_threshold,
                "garch_confidence": 0.95
            },
            "低": {
                "percentile_cut": config.percentile_thresholds.get("低", 0.4),
                "target_vol": 0.45,
                "ivol_bad_threshold": config.ivol_bad_threshold * 1.2,  # 更宽松
                "ivol_good_threshold": config.ivol_good_threshold * 1.1,  # 更宽松
                "garch_confidence": 0.90  # 较低置信度
            }
        }
        
        # 验证配置
        self._validate_threshold_config()
        
        # 阈值调整历史记录
        self.adjustment_history = []
        
        # 平滑参数
        self.smoothing_factor = 0.3  # 用于平滑阈值调整
        self.previous_thresholds = None
    
    def adjust_thresholds(self, 
                         current_regime: str,
                         market_volatility: Optional[float] = None,
                         regime_confidence: Optional[float] = None) -> Dict[str, float]:
        """根据市场状态动态调整筛选阈值
        
        Args:
            current_regime: 当前市场状态 ("低", "中", "高")
            market_volatility: 当前市场波动率，用于微调
            regime_confidence: 状态检测置信度 (0-1)
            
        Returns:
            调整后的阈值配置字典
            
        Raises:
            ConfigurationException: 状态参数无效
            DataQualityException: 输入数据异常
        """
        # 验证输入参数
        self._validate_adjustment_inputs(current_regime, market_volatility, regime_confidence)
        
        # 获取基础阈值
        base_thresholds = self._get_base_thresholds(current_regime)
        
        # 根据市场波动率微调
        if market_volatility is not None:
            base_thresholds = self._apply_volatility_adjustment(
                base_thresholds, market_volatility, current_regime
            )
        
        # 根据状态置信度调整
        if regime_confidence is not None:
            base_thresholds = self._apply_confidence_adjustment(
                base_thresholds, regime_confidence, current_regime
            )
        
        # 应用平滑处理
        smoothed_thresholds = self._apply_smoothing(base_thresholds)
        
        # 验证调整结果
        self._validate_adjusted_thresholds(smoothed_thresholds)
        
        # 记录调整历史
        self._record_adjustment(current_regime, smoothed_thresholds, 
                              market_volatility, regime_confidence)
        
        # 更新前一次阈值
        self.previous_thresholds = smoothed_thresholds.copy()
        
        return smoothed_thresholds
    
    def get_regime_specific_config(self, regime: str) -> Dict[str, float]:
        """获取特定状态下的完整配置
        
        Args:
            regime: 市场状态 ("低", "中", "高")
            
        Returns:
            该状态下的完整阈值配置
            
        Raises:
            ConfigurationException: 状态参数无效
        """
        if regime not in self.regime_thresholds:
            raise ConfigurationException(f"不支持的市场状态: {regime}")
        
        return self.regime_thresholds[regime].copy()
    
    def calculate_adaptive_percentile_threshold(self, 
                                              current_regime: str,
                                              market_stress_level: float = 0.0) -> float:
        """计算自适应分位数阈值
        
        Args:
            current_regime: 当前市场状态
            market_stress_level: 市场压力水平 (-1到1，负值表示低压力，正值表示高压力)
            
        Returns:
            自适应分位数阈值
            
        Raises:
            ConfigurationException: 参数无效
        """
        if current_regime not in self.regime_thresholds:
            raise ConfigurationException(f"不支持的市场状态: {current_regime}")
        
        if not (-1 <= market_stress_level <= 1):
            raise ConfigurationException(f"市场压力水平必须在[-1,1]范围内，当前为{market_stress_level}")
        
        # 获取基础阈值
        base_threshold = self.regime_thresholds[current_regime]["percentile_cut"]
        
        # 根据压力水平调整
        # 高压力时收紧阈值，低压力时放宽阈值
        stress_adjustment = -market_stress_level * 0.1  # 最大调整10%
        adaptive_threshold = base_threshold + stress_adjustment
        
        # 确保阈值在合理范围内
        adaptive_threshold = np.clip(adaptive_threshold, 0.1, 0.5)
        
        return adaptive_threshold
    
    def get_threshold_adjustment_statistics(self) -> Dict:
        """获取阈值调整统计信息
        
        Returns:
            调整统计信息字典
        """
        if not self.adjustment_history:
            return {
                'total_adjustments': 0,
                'regime_distribution': {},
                'average_thresholds': {},
                'threshold_volatility': {}
            }
        
        # 统计调整次数
        total_adjustments = len(self.adjustment_history)
        
        # 统计状态分布
        regimes = [record['regime'] for record in self.adjustment_history]
        regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
        regime_distribution = {
            regime: count / total_adjustments 
            for regime, count in regime_counts.items()
        }
        
        # 计算平均阈值
        threshold_keys = ['percentile_cut', 'target_vol', 'ivol_bad_threshold', 'ivol_good_threshold']
        average_thresholds = {}
        threshold_volatility = {}
        
        for key in threshold_keys:
            values = [record['thresholds'][key] for record in self.adjustment_history]
            average_thresholds[key] = np.mean(values)
            threshold_volatility[key] = np.std(values)
        
        return {
            'total_adjustments': total_adjustments,
            'regime_distribution': regime_distribution,
            'average_thresholds': average_thresholds,
            'threshold_volatility': threshold_volatility,
            'latest_regime': regimes[-1] if regimes else None,
            'latest_thresholds': self.adjustment_history[-1]['thresholds'] if self.adjustment_history else {}
        }
    
    def reset_adjustment_history(self) -> None:
        """重置调整历史记录"""
        self.adjustment_history.clear()
        self.previous_thresholds = None
    
    def _validate_threshold_config(self) -> None:
        """验证阈值配置的合理性
        
        Raises:
            ConfigurationException: 配置参数不合理
        """
        for regime, thresholds in self.regime_thresholds.items():
            # 验证分位数阈值
            if not (0 < thresholds["percentile_cut"] < 1):
                raise ConfigurationException(
                    f"{regime}状态的分位数阈值{thresholds['percentile_cut']}必须在(0,1)范围内"
                )
            
            # 验证目标波动率
            if not (0 < thresholds["target_vol"] < 2):
                raise ConfigurationException(
                    f"{regime}状态的目标波动率{thresholds['target_vol']}必须在(0,2)范围内"
                )
            
            # 验证IVOL阈值
            if not (0 < thresholds["ivol_bad_threshold"] < 1):
                raise ConfigurationException(
                    f"{regime}状态的IVOL坏波动阈值{thresholds['ivol_bad_threshold']}必须在(0,1)范围内"
                )
            
            if not (0 < thresholds["ivol_good_threshold"] < 1):
                raise ConfigurationException(
                    f"{regime}状态的IVOL好波动阈值{thresholds['ivol_good_threshold']}必须在(0,1)范围内"
                )
            
            # 验证置信度
            if not (0 < thresholds["garch_confidence"] < 1):
                raise ConfigurationException(
                    f"{regime}状态的GARCH置信度{thresholds['garch_confidence']}必须在(0,1)范围内"
                )
        
        # 验证状态间阈值的合理性
        high_cut = self.regime_thresholds["高"]["percentile_cut"]
        mid_cut = self.regime_thresholds["中"]["percentile_cut"]
        low_cut = self.regime_thresholds["低"]["percentile_cut"]
        
        if not (high_cut <= mid_cut <= low_cut):
            raise ConfigurationException(
                f"分位数阈值应满足：高波动({high_cut}) <= 中波动({mid_cut}) <= 低波动({low_cut})"
            )
    
    def _validate_adjustment_inputs(self, 
                                  current_regime: str,
                                  market_volatility: Optional[float],
                                  regime_confidence: Optional[float]) -> None:
        """验证调整输入参数
        
        Args:
            current_regime: 当前市场状态
            market_volatility: 市场波动率
            regime_confidence: 状态置信度
            
        Raises:
            ConfigurationException: 参数无效
            DataQualityException: 数据异常
        """
        if current_regime not in self.regime_thresholds:
            raise ConfigurationException(f"不支持的市场状态: {current_regime}")
        
        if market_volatility is not None:
            if not isinstance(market_volatility, (int, float)):
                raise DataQualityException("市场波动率必须为数值类型")
            
            if market_volatility < 0:
                raise DataQualityException(f"市场波动率不能为负数: {market_volatility}")
            
            if market_volatility > 2.0:  # 200%的年化波动率
                raise DataQualityException(f"市场波动率{market_volatility:.2%}异常过高")
        
        if regime_confidence is not None:
            if not isinstance(regime_confidence, (int, float)):
                raise DataQualityException("状态置信度必须为数值类型")
            
            if not (0 <= regime_confidence <= 1):
                raise DataQualityException(f"状态置信度必须在[0,1]范围内: {regime_confidence}")
    
    def _get_base_thresholds(self, current_regime: str) -> Dict[str, float]:
        """获取基础阈值配置
        
        Args:
            current_regime: 当前市场状态
            
        Returns:
            基础阈值配置
        """
        return self.regime_thresholds[current_regime].copy()
    
    def _apply_volatility_adjustment(self, 
                                   base_thresholds: Dict[str, float],
                                   market_volatility: float,
                                   current_regime: str) -> Dict[str, float]:
        """根据市场波动率微调阈值
        
        Args:
            base_thresholds: 基础阈值配置
            market_volatility: 当前市场波动率
            current_regime: 当前市场状态
            
        Returns:
            调整后的阈值配置
        """
        adjusted_thresholds = base_thresholds.copy()
        
        # 定义各状态的正常波动率范围
        normal_volatility_ranges = {
            "低": (0.10, 0.25),   # 10%-25%
            "中": (0.20, 0.40),   # 20%-40%
            "高": (0.35, 0.70)    # 35%-70%
        }
        
        normal_min, normal_max = normal_volatility_ranges[current_regime]
        normal_mid = (normal_min + normal_max) / 2
        
        # 计算波动率偏离度
        if market_volatility < normal_min:
            # 波动率低于正常范围，放宽阈值
            deviation = (normal_min - market_volatility) / normal_mid
            adjustment_factor = 1 + deviation * 0.2  # 最大放宽20%
        elif market_volatility > normal_max:
            # 波动率高于正常范围，收紧阈值
            deviation = (market_volatility - normal_max) / normal_mid
            adjustment_factor = 1 - deviation * 0.2  # 最大收紧20%
        else:
            # 波动率在正常范围内，微调
            deviation = (market_volatility - normal_mid) / normal_mid
            adjustment_factor = 1 - deviation * 0.1  # 最大调整10%
        
        # 确保调整因子在合理范围内
        adjustment_factor = np.clip(adjustment_factor, 0.7, 1.3)
        
        # 应用调整
        adjusted_thresholds["percentile_cut"] *= adjustment_factor
        adjusted_thresholds["ivol_bad_threshold"] *= adjustment_factor
        
        # 确保调整后的阈值在合理范围内
        adjusted_thresholds["percentile_cut"] = np.clip(
            adjusted_thresholds["percentile_cut"], 0.1, 0.5
        )
        adjusted_thresholds["ivol_bad_threshold"] = np.clip(
            adjusted_thresholds["ivol_bad_threshold"], 0.1, 0.8
        )
        
        return adjusted_thresholds
    
    def _apply_confidence_adjustment(self, 
                                   base_thresholds: Dict[str, float],
                                   regime_confidence: float,
                                   current_regime: str) -> Dict[str, float]:
        """根据状态检测置信度调整阈值
        
        Args:
            base_thresholds: 基础阈值配置
            regime_confidence: 状态检测置信度
            current_regime: 当前市场状态
            
        Returns:
            调整后的阈值配置
        """
        adjusted_thresholds = base_thresholds.copy()
        
        # 当置信度较低时，向中性状态的阈值靠拢
        if regime_confidence < 0.7:
            # 获取中性状态（"中"）的阈值
            neutral_thresholds = self.regime_thresholds["中"]
            
            # 计算向中性状态靠拢的程度
            confidence_factor = regime_confidence / 0.7  # 归一化到[0,1]
            blend_factor = 1 - confidence_factor  # 置信度越低，越向中性靠拢
            
            # 混合当前状态和中性状态的阈值
            for key in ["percentile_cut", "ivol_bad_threshold", "ivol_good_threshold"]:
                current_value = adjusted_thresholds[key]
                neutral_value = neutral_thresholds[key]
                adjusted_thresholds[key] = (
                    current_value * confidence_factor + 
                    neutral_value * blend_factor
                )
        
        return adjusted_thresholds
    
    def _apply_smoothing(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """应用平滑处理，避免阈值剧烈变化
        
        Args:
            base_thresholds: 基础阈值配置
            
        Returns:
            平滑后的阈值配置
        """
        if self.previous_thresholds is None:
            return base_thresholds
        
        smoothed_thresholds = {}
        
        for key, current_value in base_thresholds.items():
            if key in self.previous_thresholds:
                previous_value = self.previous_thresholds[key]
                # 指数移动平均平滑
                smoothed_value = (
                    self.smoothing_factor * current_value + 
                    (1 - self.smoothing_factor) * previous_value
                )
                smoothed_thresholds[key] = smoothed_value
            else:
                smoothed_thresholds[key] = current_value
        
        return smoothed_thresholds
    
    def _validate_adjusted_thresholds(self, thresholds: Dict[str, float]) -> None:
        """验证调整后阈值的合理性
        
        Args:
            thresholds: 调整后的阈值配置
            
        Raises:
            ConfigurationException: 阈值不合理
        """
        # 验证分位数阈值
        if not (0.05 <= thresholds["percentile_cut"] <= 0.6):
            raise ConfigurationException(
                f"调整后的分位数阈值{thresholds['percentile_cut']:.3f}超出合理范围[0.05, 0.6]"
            )
        
        # 验证目标波动率
        if not (0.2 <= thresholds["target_vol"] <= 0.8):
            raise ConfigurationException(
                f"调整后的目标波动率{thresholds['target_vol']:.3f}超出合理范围[0.2, 0.8]"
            )
        
        # 验证IVOL阈值
        if not (0.05 <= thresholds["ivol_bad_threshold"] <= 0.9):
            raise ConfigurationException(
                f"调整后的IVOL坏波动阈值{thresholds['ivol_bad_threshold']:.3f}超出合理范围[0.05, 0.9]"
            )
        
        if not (0.1 <= thresholds["ivol_good_threshold"] <= 0.95):
            raise ConfigurationException(
                f"调整后的IVOL好波动阈值{thresholds['ivol_good_threshold']:.3f}超出合理范围[0.1, 0.95]"
            )
        
        # 验证GARCH置信度
        if not (0.8 <= thresholds["garch_confidence"] <= 0.999):
            raise ConfigurationException(
                f"调整后的GARCH置信度{thresholds['garch_confidence']:.3f}超出合理范围[0.8, 0.999]"
            )
    
    def _record_adjustment(self, 
                         regime: str,
                         thresholds: Dict[str, float],
                         market_volatility: Optional[float],
                         regime_confidence: Optional[float]) -> None:
        """记录阈值调整历史
        
        Args:
            regime: 市场状态
            thresholds: 调整后的阈值
            market_volatility: 市场波动率
            regime_confidence: 状态置信度
        """
        record = {
            'timestamp': pd.Timestamp.now(),
            'regime': regime,
            'thresholds': thresholds.copy(),
            'market_volatility': market_volatility,
            'regime_confidence': regime_confidence
        }
        
        self.adjustment_history.append(record)
        
        # 限制历史记录长度，避免内存占用过多
        max_history_length = 1000
        if len(self.adjustment_history) > max_history_length:
            self.adjustment_history = self.adjustment_history[-max_history_length:]


# ============================================================================
# 主控制器DynamicLowVolFilter
# ============================================================================

class DynamicLowVolFilter:
    """动态低波筛选器主控制器
    
    统一管理四层筛选流水线：
    1. 滚动分位筛选层
    2. GARCH预测筛选层  
    3. IVOL双重约束层
    4. 市场状态感知层
    
    协调各个组件，提供统一的筛选接口，与交易环境和风险控制器集成。
    """
    
    def __init__(self, config: Dict, data_manager):
        """初始化筛选器
        
        Args:
            config: 筛选器配置字典
            data_manager: 数据管理器实例
            
        Raises:
            ConfigurationException: 配置参数错误
            DataQualityException: 数据管理器无效
        """
        # 验证输入参数
        if not isinstance(config, dict):
            raise ConfigurationException("配置参数必须为字典类型")
        
        if data_manager is None:
            raise DataQualityException("数据管理器不能为空")
        
        # 创建配置对象
        self.config = DynamicLowVolConfig(**config) if config else DynamicLowVolConfig()
        self.data_manager = data_manager
        
        # 初始化性能优化器
        cache_config = CacheConfig(
            enable_disk_cache=self.config.enable_caching,
            enable_memory_cache=self.config.enable_caching,
            cache_expiry_hours=self.config.cache_expiry_days * 24,
            cache_directory="./cache/dynamic_lowvol"
        )
        
        parallel_config = ParallelConfig(
            enable_parallel=self.config.parallel_processing,
            max_workers=None,  # 使用CPU核心数
            use_process_pool=False,  # 使用线程池以避免序列化开销
            chunk_size=10
        )
        
        self.performance_optimizer = PerformanceOptimizer(
            cache_config=cache_config,
            parallel_config=parallel_config,
            enable_memory_monitoring=True
        )
        
        # 初始化各个组件
        self.data_preprocessor = DataPreprocessor(self.config)
        self.rolling_filter = RollingPercentileFilter(self.config)
        self.garch_predictor = GARCHVolatilityPredictor(self.config)
        self.ivol_filter = IVOLConstraintFilter(self.config)
        self.regime_detector = MarketRegimeDetector(self.config)
        self.threshold_adjuster = RegimeAwareThresholdAdjuster(self.config)
        
        # 状态跟踪
        self._current_regime = "中"  # 默认中等波动状态
        self._current_regime_confidence = 0.5
        self._current_market_volatility = 0.3
        self._last_update_date = None
        self._current_tradable_mask = None
        self._current_stock_universe = None
        
        # 统计信息
        self._filter_statistics = {
            'total_updates': 0,
            'regime_history': [],
            'filter_pass_rates': {
                'rolling_percentile': [],
                'garch_prediction': [],
                'ivol_constraint': [],
                'final_combined': []
            },
            'performance_metrics': {
                'avg_update_time': 0.0,
                'cache_hit_rate': 0.0,
                'error_count': 0,
                'memory_usage_mb': 0.0,
                'parallel_efficiency': 0.0
            }
        }
    
    @performance_monitor()
    def update_tradable_mask(self, date: pd.Timestamp) -> np.ndarray:
        """更新可交易股票掩码
        
        协调四层筛选流水线，生成最终的可交易股票掩码。
        使用性能优化技术包括缓存、并行处理和向量化计算。
        
        Args:
            date: 当前交易日期
            
        Returns:
            可交易股票掩码数组 (True表示可交易)
            
        Raises:
            DataQualityException: 数据质量问题
            InsufficientDataException: 数据长度不足
            ModelFittingException: 模型拟合失败
            RegimeDetectionException: 状态检测失败
        """
        import time
        start_time = time.time()
        
        try:
            # 检查内存状态
            if self.performance_optimizer.memory_monitor:
                memory_status = self.performance_optimizer.memory_monitor.check_memory_status()
                if memory_status['status'] == 'critical':
                    # 清理缓存以释放内存
                    self.performance_optimizer.cache_manager.clear_cache(memory_only=True)
            
            # 获取必要的数据
            input_data = self._prepare_input_data(date)
            
            # 数据预处理（使用向量化优化）
            processed_data = self._preprocess_data_optimized(input_data)
            
            # 检测市场状态
            current_regime = self._detect_market_regime(processed_data, date)
            
            # 调整筛选阈值
            adjusted_thresholds = self._adjust_thresholds(current_regime, processed_data)
            
            # 执行四层筛选流水线（使用性能优化）
            tradable_mask = self._execute_filter_pipeline_optimized(
                processed_data, date, adjusted_thresholds
            )
            
            # 更新内部状态
            self._update_internal_state(date, current_regime, tradable_mask, processed_data)
            
            # 更新统计信息（包含性能指标）
            self._update_statistics_optimized(start_time, tradable_mask, processed_data)
            
            return tradable_mask
            
        except Exception as e:
            self._filter_statistics['performance_metrics']['error_count'] += 1
            # 根据异常处理策略，直接抛出异常
            raise e
    
    def get_current_regime(self) -> str:
        """获取当前市场波动状态
        
        Returns:
            市场状态字符串 ("低", "中", "高")
        """
        return self._current_regime
    
    def get_adaptive_target_volatility(self) -> float:
        """获取自适应目标波动率
        
        根据当前市场状态和波动水平，计算自适应的目标波动率。
        
        Returns:
            自适应目标波动率 (年化)
        """
        # 基础目标波动率映射
        base_target_volatility = {
            "低": 0.45,   # 低波动状态：可以接受更高的目标波动率
            "中": 0.40,   # 中等波动状态：标准目标波动率
            "高": 0.35    # 高波动状态：降低目标波动率以控制风险
        }
        
        base_vol = base_target_volatility.get(self._current_regime, 0.40)
        
        # 根据市场波动率进行微调
        if self._current_market_volatility is not None:
            # 当市场波动率高于正常水平时，进一步降低目标波动率
            if self._current_market_volatility > 0.5:
                adjustment_factor = 0.9  # 降低10%
            elif self._current_market_volatility > 0.4:
                adjustment_factor = 0.95  # 降低5%
            elif self._current_market_volatility < 0.2:
                adjustment_factor = 1.05  # 提高5%
            else:
                adjustment_factor = 1.0
            
            base_vol *= adjustment_factor
        
        # 根据状态检测置信度进行调整
        if self._current_regime_confidence < 0.7:
            # 置信度较低时，向中性状态靠拢
            neutral_vol = base_target_volatility["中"]
            confidence_weight = self._current_regime_confidence
            base_vol = confidence_weight * base_vol + (1 - confidence_weight) * neutral_vol
        
        # 确保目标波动率在合理范围内
        return np.clip(base_vol, 0.25, 0.60)
    
    def get_filter_statistics(self) -> Dict:
        """获取筛选统计信息
        
        Returns:
            筛选统计信息字典，包含：
            - 当前状态信息
            - 筛选通过率统计
            - 性能指标
            - 历史记录摘要
        """
        current_stats = self._filter_statistics.copy()
        
        # 添加当前状态信息
        current_stats['current_state'] = {
            'regime': self._current_regime,
            'regime_confidence': self._current_regime_confidence,
            'market_volatility': self._current_market_volatility,
            'last_update_date': self._last_update_date.isoformat() if self._last_update_date else None,
            'adaptive_target_volatility': self.get_adaptive_target_volatility(),
            'tradable_stocks_count': int(self._current_tradable_mask.sum()) if self._current_tradable_mask is not None else 0,
            'total_stocks_count': len(self._current_tradable_mask) if self._current_tradable_mask is not None else 0
        }
        
        # 计算筛选通过率统计
        if current_stats['filter_pass_rates']['final_combined']:
            current_stats['filter_summary'] = {
                'avg_pass_rate': np.mean(current_stats['filter_pass_rates']['final_combined']),
                'min_pass_rate': np.min(current_stats['filter_pass_rates']['final_combined']),
                'max_pass_rate': np.max(current_stats['filter_pass_rates']['final_combined']),
                'pass_rate_std': np.std(current_stats['filter_pass_rates']['final_combined'])
            }
        else:
            current_stats['filter_summary'] = {
                'avg_pass_rate': 0.0,
                'min_pass_rate': 0.0,
                'max_pass_rate': 0.0,
                'pass_rate_std': 0.0
            }
        
        # 状态分布统计
        if current_stats['regime_history']:
            regime_counts = {}
            for regime in current_stats['regime_history']:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            total_count = len(current_stats['regime_history'])
            current_stats['regime_distribution'] = {
                regime: count / total_count 
                for regime, count in regime_counts.items()
            }
        else:
            current_stats['regime_distribution'] = {}
        
        return current_stats
    
    def _prepare_input_data(self, date: pd.Timestamp) -> FilterInputData:
        """准备输入数据
        
        Args:
            date: 当前日期
            
        Returns:
            筛选器输入数据结构
            
        Raises:
            DataQualityException: 数据获取失败
        """
        try:
            # 从数据管理器获取数据
            # 这里假设data_manager有相应的方法
            price_data = self.data_manager.get_price_data(
                end_date=date, 
                lookback_days=max(self.config.rolling_windows) + self.config.garch_window + 50
            )
            
            volume_data = self.data_manager.get_volume_data(
                end_date=date,
                lookback_days=max(self.config.rolling_windows) + 50
            )
            
            factor_data = self.data_manager.get_factor_data(
                end_date=date,
                lookback_days=self.config.garch_window + 50
            )
            
            market_data = self.data_manager.get_market_data(
                end_date=date,
                lookback_days=self.config.regime_detection_window + 50
            )
            
            return FilterInputData(
                price_data=price_data,
                volume_data=volume_data,
                factor_data=factor_data,
                market_data=market_data,
                current_date=date
            )
            
        except Exception as e:
            raise DataQualityException(f"数据获取失败: {str(e)}")
    
    def _preprocess_data(self, input_data: FilterInputData) -> Dict:
        """预处理数据
        
        Args:
            input_data: 原始输入数据
            
        Returns:
            预处理后的数据字典
        """
        # 预处理价格数据
        cleaned_price_data = self.data_preprocessor.preprocess_price_data(
            input_data.price_data
        )
        
        # 计算收益率
        returns_data = self.data_preprocessor.calculate_returns(
            cleaned_price_data, return_type='simple'
        )
        
        # 准备滚动窗口数据
        rolling_windows_data = self.data_preprocessor.prepare_rolling_windows(
            returns_data, self.config.rolling_windows
        )
        
        # 验证数据质量
        self.data_preprocessor.validate_data_quality(returns_data, "收益率数据")
        self.data_preprocessor.validate_data_quality(input_data.factor_data, "因子数据")
        self.data_preprocessor.validate_data_quality(input_data.market_data, "市场数据")
        
        return {
            'price_data': cleaned_price_data,
            'returns_data': returns_data,
            'rolling_windows_data': rolling_windows_data,
            'factor_data': input_data.factor_data,
            'market_data': input_data.market_data,
            'volume_data': input_data.volume_data
        }
    
    def _detect_market_regime(self, processed_data: Dict, date: pd.Timestamp) -> str:
        """检测市场状态
        
        Args:
            processed_data: 预处理后的数据
            date: 当前日期
            
        Returns:
            市场状态字符串
        """
        # 从市场数据中提取收益率
        if 'returns' in processed_data['market_data'].columns:
            market_returns = processed_data['market_data']['returns']
        else:
            # 如果没有直接的收益率列，从价格计算
            market_prices = processed_data['market_data'].iloc[:, 0]  # 假设第一列是价格
            market_returns = market_prices.pct_change().dropna()
        
        # 检测市场状态
        current_regime = self.regime_detector.detect_regime(market_returns, date)
        
        # 获取状态概率分布以计算置信度
        try:
            regime_probabilities = self.regime_detector.get_regime_probabilities(
                market_returns, date
            )
            self._current_regime_confidence = regime_probabilities.get(current_regime, 0.5)
        except:
            self._current_regime_confidence = 0.5
        
        # 计算当前市场波动率
        recent_returns = market_returns.tail(20)  # 最近20天
        if len(recent_returns) > 0:
            self._current_market_volatility = recent_returns.std() * np.sqrt(252)
        else:
            self._current_market_volatility = 0.3
        
        return current_regime
    
    def _adjust_thresholds(self, current_regime: str, processed_data: Dict) -> Dict:
        """调整筛选阈值
        
        Args:
            current_regime: 当前市场状态
            processed_data: 预处理后的数据
            
        Returns:
            调整后的阈值字典
        """
        return self.threshold_adjuster.adjust_thresholds(
            current_regime,
            market_volatility=self._current_market_volatility,
            regime_confidence=self._current_regime_confidence
        )
    
    def _execute_filter_pipeline(self, processed_data: Dict, date: pd.Timestamp, 
                                thresholds: Dict) -> np.ndarray:
        """执行四层筛选流水线
        
        Args:
            processed_data: 预处理后的数据
            date: 当前日期
            thresholds: 调整后的阈值
            
        Returns:
            最终的可交易股票掩码
        """
        returns_data = processed_data['returns_data']
        factor_data = processed_data['factor_data']
        
        # 第一层：滚动分位筛选
        percentile_mask = self.rolling_filter.apply_percentile_filter(
            returns_data,
            current_date=date,
            window=self.config.rolling_windows[0],  # 使用第一个窗口
            percentile_threshold=thresholds['percentile_cut']
        )
        
        # 第二层：GARCH预测筛选
        garch_mask = self.garch_predictor.apply_garch_filter(
            returns_data,
            current_date=date,
            target_volatility=thresholds.get('target_vol', 0.4)
        )
        
        # 第三层：IVOL约束筛选
        ivol_mask = self.ivol_filter.apply_ivol_constraint(
            returns_data,
            factor_data,
            current_date=date,
            market_data=processed_data['market_data']
        )
        
        # 组合筛选结果（取交集）
        combined_mask = percentile_mask & garch_mask & ivol_mask
        
        # 记录各层筛选通过率
        total_stocks = len(returns_data.columns)
        self._filter_statistics['filter_pass_rates']['rolling_percentile'].append(
            percentile_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['garch_prediction'].append(
            garch_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['ivol_constraint'].append(
            ivol_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['final_combined'].append(
            combined_mask.sum() / total_stocks
        )
        
        # 限制历史记录长度
        max_history = 100
        for key in self._filter_statistics['filter_pass_rates']:
            if len(self._filter_statistics['filter_pass_rates'][key]) > max_history:
                self._filter_statistics['filter_pass_rates'][key] = \
                    self._filter_statistics['filter_pass_rates'][key][-max_history:]
        
        return combined_mask
    
    def _update_internal_state(self, date: pd.Timestamp, regime: str, 
                              tradable_mask: np.ndarray, processed_data: Dict):
        """更新内部状态
        
        Args:
            date: 当前日期
            regime: 检测到的市场状态
            tradable_mask: 可交易股票掩码
            processed_data: 预处理后的数据
        """
        self._last_update_date = date
        self._current_regime = regime
        self._current_tradable_mask = tradable_mask
        self._current_stock_universe = processed_data['returns_data'].columns.tolist()
        
        # 更新状态历史
        self._filter_statistics['regime_history'].append(regime)
        
        # 限制历史记录长度
        max_history = 200
        if len(self._filter_statistics['regime_history']) > max_history:
            self._filter_statistics['regime_history'] = \
                self._filter_statistics['regime_history'][-max_history:]
    
    def _update_statistics(self, start_time: float, tradable_mask: np.ndarray, 
                          processed_data: Dict):
        """更新统计信息
        
        Args:
            start_time: 开始时间戳
            tradable_mask: 可交易股票掩码
            processed_data: 预处理后的数据
        """
        import time
        
        # 更新性能指标
        update_time = time.time() - start_time
        self._filter_statistics['total_updates'] += 1
        
        # 计算平均更新时间
        current_avg = self._filter_statistics['performance_metrics']['avg_update_time']
        total_updates = self._filter_statistics['total_updates']
        new_avg = (current_avg * (total_updates - 1) + update_time) / total_updates
        self._filter_statistics['performance_metrics']['avg_update_time'] = new_avg
    
    def reset_statistics(self):
        """重置统计信息"""
        self._filter_statistics = {
            'total_updates': 0,
            'regime_history': [],
            'filter_pass_rates': {
                'rolling_percentile': [],
                'garch_prediction': [],
                'ivol_constraint': [],
                'final_combined': []
            },
            'performance_metrics': {
                'avg_update_time': 0.0,
                'cache_hit_rate': 0.0,
                'error_count': 0
            }
        }
    
    def get_current_tradable_stocks(self) -> List[str]:
        """获取当前可交易股票列表
        
        Returns:
            可交易股票代码列表
        """
        if self._current_tradable_mask is None or self._current_stock_universe is None:
            return []
        
        return [
            stock for i, stock in enumerate(self._current_stock_universe)
            if i < len(self._current_tradable_mask) and self._current_tradable_mask[i]
        ]
    
    def get_regime_transition_probability(self) -> Dict[str, float]:
        """获取状态转换概率
        
        Returns:
            状态转换概率字典
        """
        if len(self._filter_statistics['regime_history']) < 2:
            return {"低": 0.33, "中": 0.34, "高": 0.33}
        
        # 计算状态转换统计
        transitions = {}
        history = self._filter_statistics['regime_history']
        
        for i in range(len(history) - 1):
            current_state = history[i]
            next_state = history[i + 1]
            
            if current_state not in transitions:
                transitions[current_state] = {}
            
            if next_state not in transitions[current_state]:
                transitions[current_state][next_state] = 0
            
            transitions[current_state][next_state] += 1
        
        # 计算当前状态的转换概率
        current_regime = self._current_regime
        if current_regime in transitions:
            total_transitions = sum(transitions[current_regime].values())
            return {
                state: count / total_transitions
                for state, count in transitions[current_regime].items()
            }
        else:
            return {"低": 0.33, "中": 0.34, "高": 0.33}  
  
    # ============================================================================
    # 性能优化方法
    # ============================================================================
    
    def _preprocess_data_optimized(self, input_data: FilterInputData) -> Dict:
        """优化的数据预处理方法
        
        使用向量化计算和并行处理优化数据预处理性能。
        
        Args:
            input_data: 原始输入数据
            
        Returns:
            预处理后的数据字典
        """
        # 预处理价格数据
        cleaned_price_data = self.data_preprocessor.preprocess_price_data(
            input_data.price_data
        )
        
        # 使用向量化计算收益率
        returns_data = self.data_preprocessor.calculate_returns(
            cleaned_price_data, return_type='simple'
        )
        
        # 使用向量化方法计算多窗口滚动波动率
        rolling_volatilities = self.performance_optimizer.vectorized_optimizer.vectorized_rolling_volatility(
            returns_data, self.config.rolling_windows
        )
        
        # 准备滚动窗口数据
        rolling_windows_data = self.data_preprocessor.prepare_rolling_windows(
            returns_data, self.config.rolling_windows
        )
        
        # 验证数据质量
        self.data_preprocessor.validate_data_quality(returns_data, "收益率数据")
        self.data_preprocessor.validate_data_quality(input_data.factor_data, "因子数据")
        self.data_preprocessor.validate_data_quality(input_data.market_data, "市场数据")
        
        return {
            'price_data': cleaned_price_data,
            'returns_data': returns_data,
            'rolling_windows_data': rolling_windows_data,
            'rolling_volatilities': rolling_volatilities,
            'factor_data': input_data.factor_data,
            'market_data': input_data.market_data,
            'volume_data': input_data.volume_data
        }
    
    def _execute_filter_pipeline_optimized(self, processed_data: Dict, date: pd.Timestamp, 
                                         thresholds: Dict) -> np.ndarray:
        """优化的四层筛选流水线执行
        
        使用并行处理和缓存优化筛选性能。
        
        Args:
            processed_data: 预处理后的数据
            date: 当前日期
            thresholds: 调整后的阈值
            
        Returns:
            最终的可交易股票掩码
        """
        returns_data = processed_data['returns_data']
        factor_data = processed_data['factor_data']
        
        # 使用并行处理管理器执行筛选
        with self.performance_optimizer.parallel_manager as parallel_mgr:
            
            # 定义筛选任务
            filter_tasks = [
                ('percentile', self._apply_percentile_filter_task, 
                 (returns_data, date, thresholds)),
                ('garch', self._apply_garch_filter_task, 
                 (returns_data, date, thresholds)),
                ('ivol', self._apply_ivol_filter_task, 
                 (returns_data, factor_data, date, processed_data['market_data']))
            ]
            
            # 并行执行筛选任务
            if self.config.parallel_processing and len(filter_tasks) > 1:
                filter_results = parallel_mgr.parallel_map(
                    self._execute_filter_task, filter_tasks
                )
                
                # 提取结果
                percentile_mask = filter_results[0]
                garch_mask = filter_results[1]
                ivol_mask = filter_results[2]
            else:
                # 串行执行
                percentile_mask = self._execute_filter_task(filter_tasks[0])
                garch_mask = self._execute_filter_task(filter_tasks[1])
                ivol_mask = self._execute_filter_task(filter_tasks[2])
        
        # 组合筛选结果（取交集）
        combined_mask = percentile_mask & garch_mask & ivol_mask
        
        # 记录各层筛选通过率
        total_stocks = len(returns_data.columns)
        self._filter_statistics['filter_pass_rates']['rolling_percentile'].append(
            percentile_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['garch_prediction'].append(
            garch_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['ivol_constraint'].append(
            ivol_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['final_combined'].append(
            combined_mask.sum() / total_stocks
        )
        
        # 限制历史记录长度
        max_history = 100
        for key in self._filter_statistics['filter_pass_rates']:
            if len(self._filter_statistics['filter_pass_rates'][key]) > max_history:
                self._filter_statistics['filter_pass_rates'][key] = \
                    self._filter_statistics['filter_pass_rates'][key][-max_history:]
        
        return combined_mask
    
    def _execute_filter_task(self, task_info: Tuple) -> np.ndarray:
        """执行单个筛选任务
        
        Args:
            task_info: 任务信息元组 (任务名, 任务函数, 任务参数)
            
        Returns:
            筛选结果掩码
        """
        task_name, task_func, task_args = task_info
        
        try:
            return task_func(*task_args)
        except Exception as e:
            # 记录错误并重新抛出，以便上层调用者能感知到失败
            import logging
            logging.error(f"筛选任务 {task_name} 执行失败: {e}")
            raise
    
    def _apply_percentile_filter_task(self, returns: pd.DataFrame, 
                                    date: pd.Timestamp, 
                                    thresholds: Dict) -> np.ndarray:
        """分位数筛选任务"""
        return self.rolling_filter.apply_percentile_filter(
            returns,
            current_date=date,
            window=self.config.rolling_windows[0],
            percentile_threshold=thresholds['percentile_cut']
        )
    
    def _apply_garch_filter_task(self, returns: pd.DataFrame, 
                               date: pd.Timestamp, 
                               thresholds: Dict) -> np.ndarray:
        """GARCH筛选任务"""
        return self.garch_predictor.apply_garch_filter(
            returns,
            current_date=date,
            target_volatility=thresholds.get('target_vol', 0.4)
        )
    
    def _apply_ivol_filter_task(self, returns: pd.DataFrame, 
                              factor_data: pd.DataFrame,
                              date: pd.Timestamp, 
                              market_data: pd.DataFrame) -> np.ndarray:
        """IVOL筛选任务"""
        return self.ivol_filter.apply_ivol_constraint(
            returns,
            factor_data,
            current_date=date,
            market_data=market_data
        )
    
    def _update_statistics_optimized(self, start_time: float, 
                                   tradable_mask: np.ndarray, 
                                   processed_data: Dict):
        """优化的统计信息更新
        
        包含性能指标和内存使用监控。
        
        Args:
            start_time: 开始时间戳
            tradable_mask: 可交易股票掩码
            processed_data: 预处理后的数据
        """
        import time
        
        # 更新基础性能指标
        update_time = time.time() - start_time
        self._filter_statistics['total_updates'] += 1
        
        # 计算平均更新时间
        current_avg = self._filter_statistics['performance_metrics']['avg_update_time']
        total_updates = self._filter_statistics['total_updates']
        new_avg = (current_avg * (total_updates - 1) + update_time) / total_updates
        self._filter_statistics['performance_metrics']['avg_update_time'] = new_avg
        
        # 更新缓存命中率
        cache_stats = self.performance_optimizer.cache_manager.get_cache_stats()
        self._filter_statistics['performance_metrics']['cache_hit_rate'] = cache_stats['hit_rate']
        
        # 更新内存使用情况
        if self.performance_optimizer.memory_monitor:
            memory_info = self.performance_optimizer.memory_monitor.get_current_memory_usage()
            self._filter_statistics['performance_metrics']['memory_usage_mb'] = memory_info['current_memory_mb']
        
        # 更新并行处理效率
        parallel_stats = self.performance_optimizer.parallel_manager.get_performance_stats()
        self._filter_statistics['performance_metrics']['parallel_efficiency'] = parallel_stats.get('parallel_efficiency', 0.0)
    
    def optimize_performance(self) -> Dict[str, Any]:
        """执行性能优化
        
        Returns:
            优化结果报告
        """
        return self.performance_optimizer.optimize_system_performance()
    
    def get_performance_report(self) -> Dict:
        """获取性能报告
        
        Returns:
            详细的性能报告
        """
        base_report = self.performance_optimizer.get_comprehensive_performance_report()
        
        # 添加筛选器特定的性能指标
        base_report['filter_performance'] = {
            'total_updates': self._filter_statistics['total_updates'],
            'avg_update_time': self._filter_statistics['performance_metrics']['avg_update_time'],
            'error_count': self._filter_statistics['performance_metrics']['error_count'],
            'cache_hit_rate': self._filter_statistics['performance_metrics']['cache_hit_rate'],
            'memory_usage_mb': self._filter_statistics['performance_metrics'].get('memory_usage_mb', 0.0),
            'parallel_efficiency': self._filter_statistics['performance_metrics'].get('parallel_efficiency', 0.0)
        }
        
        return base_report
    
    def clear_performance_cache(self):
        """清理性能缓存"""
        self.performance_optimizer.cache_manager.clear_cache()
        
        # 清理组件缓存
        if hasattr(self.rolling_filter, '_volatility_cache') and self.rolling_filter._volatility_cache:
            self.rolling_filter._volatility_cache.clear()
        
        if hasattr(self.garch_predictor, '_prediction_cache') and self.garch_predictor._prediction_cache:
            self.garch_predictor._prediction_cache.clear()
        
        if hasattr(self.ivol_filter, '_ivol_cache') and self.ivol_filter._ivol_cache:
            self.ivol_filter._ivol_cache.clear()
    
    def get_memory_optimization_suggestions(self) -> List[str]:
        """获取内存优化建议
        
        Returns:
            内存优化建议列表
        """
        suggestions = []
        
        if self.performance_optimizer.memory_monitor:
            suggestions.extend(
                self.performance_optimizer.memory_monitor.get_memory_optimization_suggestions()
            )
        
        # 添加筛选器特定的建议
        if self._filter_statistics['total_updates'] > 1000:
            suggestions.append("考虑定期重置统计信息以减少内存占用")
        
        cache_stats = self.performance_optimizer.cache_manager.get_cache_stats()
        if cache_stats['memory_cache_size'] > 500:
            suggestions.append("内存缓存项目过多，考虑减少缓存大小或增加清理频率")
        
        return suggestions
    
    def benchmark_performance(self, test_data: Dict, iterations: int = 10) -> Dict:
        """性能基准测试
        
        Args:
            test_data: 测试数据
            iterations: 测试迭代次数
            
        Returns:
            性能基准测试结果
        """
        import time
        
        # 准备测试数据
        test_dates = pd.date_range(start='2023-01-01', periods=iterations, freq='D')
        
        # 测试串行执行性能
        serial_times = []
        original_parallel_setting = self.config.parallel_processing
        self.config.parallel_processing = False
        
        for date in test_dates:
            start_time = time.time()
            try:
                self.update_tradable_mask(date)
                serial_times.append(time.time() - start_time)
            except:
                serial_times.append(float('inf'))
        
        # 测试并行执行性能
        parallel_times = []
        self.config.parallel_processing = True
        
        for date in test_dates:
            start_time = time.time()
            try:
                self.update_tradable_mask(date)
                parallel_times.append(time.time() - start_time)
            except:
                parallel_times.append(float('inf'))
        
        # 恢复原始设置
        self.config.parallel_processing = original_parallel_setting
        
        # 计算性能指标
        avg_serial_time = np.mean([t for t in serial_times if t != float('inf')])
        avg_parallel_time = np.mean([t for t in parallel_times if t != float('inf')])
        
        speedup_ratio = avg_serial_time / avg_parallel_time if avg_parallel_time > 0 else 1.0
        
        return {
            'avg_serial_time': avg_serial_time,
            'avg_parallel_time': avg_parallel_time,
            'speedup_ratio': speedup_ratio,
            'serial_times': serial_times,
            'parallel_times': parallel_times,
            'iterations': iterations,
            'performance_improvement': f"{(speedup_ratio - 1) * 100:.1f}%"
        }