"""
市场状态检测模块

包含市场状态检测和阈值调整的实现：
- MarketRegimeDetector: 市场状态检测器
- RegimeAwareThresholdAdjuster: 状态感知阈值调节器

这些组件用于检测市场状态变化并相应调整筛选阈值。
"""

from .detector import MarketRegimeDetector
from .threshold_adjuster import RegimeAwareThresholdAdjuster

__all__ = [
    'MarketRegimeDetector',
    'RegimeAwareThresholdAdjuster'
]