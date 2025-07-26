"""
动态低波筛选器数据结构定义

定义了筛选器模块中使用的核心数据结构，包括输入输出数据格式和配置类。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .exceptions import ConfigurationException


@dataclass
class FilterInputData:
    """筛选器输入数据结构
    
    封装了筛选器所需的所有输入数据，确保数据格式的一致性和完整性。
    
    Attributes:
        price_data: 价格数据 (日期 x 股票)，包含股票的历史价格信息
        volume_data: 成交量数据 (日期 x 股票)，用于流动性分析
        factor_data: 因子数据 (日期 x 因子)，包含风险因子和风格因子
        market_data: 市场指数数据 (日期 x 指数)，用于市场状态分析
        current_date: 当前交易日期，筛选的基准日期
    """
    price_data: pd.DataFrame      # 价格数据 (日期 x 股票)
    volume_data: pd.DataFrame     # 成交量数据
    factor_data: pd.DataFrame     # 因子数据
    market_data: pd.DataFrame     # 市场指数数据
    current_date: pd.Timestamp    # 当前交易日期


@dataclass
class FilterOutputData:
    """筛选器输出数据结构
    
    封装了筛选器的输出结果，提供完整的筛选信息和统计数据。
    
    Attributes:
        tradable_mask: 可交易股票掩码，True表示通过筛选的股票
        current_regime: 当前市场状态标识 ("低波动", "中波动", "高波动")
        regime_signal: 状态信号强度 (-1到1之间，-1表示强烈看空，1表示强烈看多)
        adaptive_target_vol: 自适应目标波动率，根据市场状态动态调整
        filter_statistics: 筛选统计信息，包含各种筛选指标和中间结果
    """
    tradable_mask: np.ndarray     # 可交易股票掩码
    current_regime: str           # 当前市场状态
    regime_signal: float          # 状态信号 (-1, 0, 1)
    adaptive_target_vol: float    # 自适应目标波动率
    filter_statistics: Dict       # 筛选统计信息


@dataclass
class DynamicLowVolConfig:
    """动态低波筛选器配置数据类
    
    包含筛选器运行所需的所有配置参数，支持参数验证和默认值设置。
    """
    
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
    
    # 数据质量配置
    max_missing_ratio: float = 0.1
    outlier_threshold: float = 5.0
    
    def __post_init__(self):
        """配置参数验证
        
        在对象创建后自动调用，验证所有配置参数的合理性。
        
        Raises:
            ConfigurationException: 配置参数不合理时抛出
        """
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的合理性
        
        检查所有配置参数是否在有效范围内，确保配置的一致性和合理性。
        
        Raises:
            ConfigurationException: 参数验证失败时抛出
        """
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