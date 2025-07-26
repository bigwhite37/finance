"""
预测器模块

包含各种波动率预测器的实现：
- GARCHVolatilityPredictor: GARCH波动率预测器

这些预测器用于预测未来的波动率，为筛选决策提供前瞻性信息。
"""

from .garch_volatility import GARCHVolatilityPredictor

__all__ = [
    'GARCHVolatilityPredictor'
]