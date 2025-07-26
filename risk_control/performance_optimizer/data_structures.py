"""
性能优化器数据结构模块

定义性能优化器使用的各种数据结构和配置类。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_efficiency: float = 0.0
    vectorization_speedup: float = 0.0
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'cache_hit_rate': self.cache_hit_rate,
            'parallel_efficiency': self.parallel_efficiency,
            'vectorization_speedup': self.vectorization_speedup,
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate': self.successful_operations / self.total_operations if self.total_operations > 0 else 0.0
        }


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_memory_monitoring: bool = True
    memory_warning_threshold_mb: float = 1000.0
    memory_critical_threshold_mb: float = 2000.0
    performance_log_threshold_seconds: float = 1.0
    memory_log_threshold_mb: float = 100.0
    history_size: int = 1000
    
    def validate(self) -> bool:
        """验证配置参数"""
        if self.memory_warning_threshold_mb >= self.memory_critical_threshold_mb:
            raise ValueError("内存警告阈值必须小于临界阈值")
        
        if self.memory_warning_threshold_mb <= 0 or self.memory_critical_threshold_mb <= 0:
            raise ValueError("内存阈值必须大于0")
        
        if self.performance_log_threshold_seconds <= 0:
            raise ValueError("性能日志阈值必须大于0")
        
        if self.memory_log_threshold_mb <= 0:
            raise ValueError("内存日志阈值必须大于0")
        
        if self.history_size <= 0:
            raise ValueError("历史记录大小必须大于0")
        
        return True


@dataclass
class OptimizationResult:
    """优化结果数据结构"""
    success: bool = False
    actions_taken: list = field(default_factory=list)
    performance_improvement: float = 0.0
    memory_saved_mb: float = 0.0
    execution_time_saved_seconds: float = 0.0
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'success': self.success,
            'actions_taken': self.actions_taken,
            'performance_improvement': self.performance_improvement,
            'memory_saved_mb': self.memory_saved_mb,
            'execution_time_saved_seconds': self.execution_time_saved_seconds,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }