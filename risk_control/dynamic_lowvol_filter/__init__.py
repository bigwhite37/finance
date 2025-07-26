"""
动态低波筛选器模块

实现动态、预测式、市场状态感知的低波股票筛选系统，
替代传统的静态阈值方法。

本模块提供以下主要功能：
- 数据预处理和质量验证
- 滚动分位筛选
- IVOL约束筛选
- GARCH波动率预测
- 市场状态检测
- 状态感知阈值调整

主要类：
- DynamicLowVolFilter: 主控制器
- DataPreprocessor: 数据预处理器
- RollingPercentileFilter: 滚动分位筛选器
- IVOLConstraintFilter: IVOL约束筛选器
- GARCHVolatilityPredictor: GARCH波动率预测器
- MarketRegimeDetector: 市场状态检测器
- RegimeAwareThresholdAdjuster: 状态感知阈值调节器

数据结构：
- FilterInputData: 筛选器输入数据
- FilterOutputData: 筛选器输出数据
- DynamicLowVolConfig: 配置数据类

异常类：
- FilterException: 基础异常
- DataQualityException: 数据质量异常
- ModelFittingException: 模型拟合异常
- RegimeDetectionException: 状态检测异常
- InsufficientDataException: 数据不足异常
- ConfigurationException: 配置异常
"""

# 为了保持向后兼容性，暂时保留原有的导入方式
# 在后续版本中将添加弃用警告

__version__ = "2.0.0"  # 重构后版本
__author__ = "Risk Control Team"

# 重构状态标识
REFACTORING_COMPLETED = True

# 导入已完成重构的模块
from .core import DynamicLowVolFilter
from .data_preprocessor import DataPreprocessor
from .filters import RollingPercentileFilter, IVOLConstraintFilter
from .predictors import GARCHVolatilityPredictor
from .regime import MarketRegimeDetector, RegimeAwareThresholdAdjuster
from .data_structures import FilterInputData, FilterOutputData, DynamicLowVolConfig
from .exceptions import (
    FilterException, DataQualityException, ModelFittingException,
    RegimeDetectionException, InsufficientDataException, ConfigurationException
)

__all__ = [
    'DynamicLowVolFilter',
    'DataPreprocessor', 
    'RollingPercentileFilter',
    'IVOLConstraintFilter',
    'GARCHVolatilityPredictor',
    'MarketRegimeDetector',
    'RegimeAwareThresholdAdjuster',
    'FilterInputData',
    'FilterOutputData', 
    'DynamicLowVolConfig',
    'FilterException',
    'DataQualityException',
    'ModelFittingException',
    'RegimeDetectionException',
    'InsufficientDataException',
    'ConfigurationException'
]