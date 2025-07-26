"""
动态低波筛选器性能优化模块

提供缓存管理、并行处理、向量化计算和内存监控功能，
用于优化动态低波筛选器的性能表现。

主要组件:
- PerformanceOptimizer: 主性能优化器
- UnifiedCacheManager: 统一缓存管理
- ParallelProcessingManager: 并行处理管理
- VectorizedOptimizer: 向量化计算优化
- MemoryMonitor: 内存使用监控

版本: 2.0.0 (重构版本)
"""

# 导入主要的性能优化类
from .core import PerformanceOptimizer

# 导入各个子模块的主要类
from .cache import (
    CacheConfig,
    MemoryCacheManager,
    DiskCacheManager,
    UnifiedCacheManager,
    AdvancedCacheManager  # 向后兼容
)

from .parallel import (
    ParallelConfig,
    ParallelProcessingManager
)

from .compute import (
    VectorizedOptimizer,
    VectorizedComputeOptimizer  # 向后兼容
)

from .monitoring import (
    MemoryMonitor,
    performance_monitor,
    memory_profiler,
    execution_timer
)

# 导入数据结构和配置
from .data_structures import (
    PerformanceMetrics,
    MonitoringConfig,
    OptimizationResult
)

# 版本信息
__version__ = "2.0.0"
__refactored__ = True

# 公开的API
__all__ = [
    # 主要类
    'PerformanceOptimizer',
    
    # 缓存相关
    'CacheConfig',
    'MemoryCacheManager',
    'DiskCacheManager',
    'UnifiedCacheManager',
    'AdvancedCacheManager',  # 向后兼容
    
    # 并行处理相关
    'ParallelConfig',
    'ParallelProcessingManager',
    
    # 计算优化相关
    'VectorizedOptimizer',
    'VectorizedComputeOptimizer',  # 向后兼容
    
    # 监控相关
    'MemoryMonitor',
    'performance_monitor',
    'memory_profiler',
    'execution_timer',
    
    # 数据结构
    'PerformanceMetrics',
    'MonitoringConfig',
    'OptimizationResult',
    
    # 版本信息
    '__version__',
    '__refactored__'
]

# 为了向后兼容，保持原有的导入方式
# 这样原来的 from risk_control.performance_optimizer import PerformanceOptimizer 仍然有效