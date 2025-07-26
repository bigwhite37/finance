"""
性能优化器核心模块

统一管理缓存、并行处理、向量化计算和内存监控。
"""

import time
import logging
import gc
from typing import Dict, Any, Optional, List
import threading

from .cache.cache_config import CacheConfig
from .cache.unified_cache import UnifiedCacheManager
from .parallel.config import ParallelConfig
from .parallel.processing_manager import ParallelProcessingManager
from .compute.vectorized_optimizer import VectorizedOptimizer
from .monitoring.memory_monitor import MemoryMonitor
from .monitoring.performance_decorator import performance_monitor
from .data_structures import PerformanceMetrics, MonitoringConfig, OptimizationResult


class PerformanceOptimizer:
    """性能优化管理器
    
    统一管理缓存、并行处理、向量化计算和内存监控。
    """
    
    def __init__(self, 
                 cache_config: Optional[CacheConfig] = None,
                 parallel_config: Optional[ParallelConfig] = None,
                 monitoring_config: Optional[MonitoringConfig] = None):
        """初始化性能优化管理器
        
        Args:
            cache_config: 缓存配置
            parallel_config: 并行处理配置
            monitoring_config: 监控配置
        """
        self._lock = threading.RLock()
        
        # 初始化配置
        self.cache_config = cache_config or CacheConfig.create_default_config()
        self.parallel_config = parallel_config or ParallelConfig.create_default_config()
        self.monitoring_config = monitoring_config or MonitoringConfig()
        
        # 验证配置
        self.cache_config.validate()
        self.parallel_config.validate()
        self.monitoring_config.validate()
        
        # 初始化组件
        self.cache_manager = UnifiedCacheManager(self.cache_config)
        self.parallel_manager = ParallelProcessingManager(self.parallel_config)
        self.vectorized_optimizer = VectorizedOptimizer()
        
        # 初始化内存监控
        if self.monitoring_config.enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor(
                warning_threshold_mb=self.monitoring_config.memory_warning_threshold_mb,
                critical_threshold_mb=self.monitoring_config.memory_critical_threshold_mb,
                history_size=self.monitoring_config.history_size
            )
        else:
            self.memory_monitor = None
        
        # 性能统计
        self._performance_history: List[PerformanceMetrics] = []
        self._optimization_history: List[OptimizationResult] = []
        
        logging.info("性能优化器初始化完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.parallel_manager.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.parallel_manager.__exit__(exc_type, exc_val, exc_tb)
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """获取综合性能报告
        
        Returns:
            综合性能报告
        """
        with self._lock:
            report = {
                'timestamp': time.time(),
                'cache_stats': self.cache_manager.get_comprehensive_stats(),
                'parallel_stats': self.parallel_manager.get_performance_stats(),
                'vectorized_stats': self.vectorized_optimizer.get_performance_stats(),
                'configuration': {
                    'cache_config': self.cache_config.__dict__,
                    'parallel_config': self.parallel_config.__dict__,
                    'monitoring_config': self.monitoring_config.__dict__
                }
            }
            
            if self.memory_monitor:
                report['memory_status'] = self.memory_monitor.check_memory_status()
                report['memory_suggestions'] = self.memory_monitor.get_memory_optimization_suggestions()
                report['memory_stats'] = self.memory_monitor.get_monitoring_stats()
            
            # 添加历史统计
            if self._performance_history:
                recent_metrics = self._performance_history[-10:]  # 最近10次的指标
                report['recent_performance'] = {
                    'average_execution_time': sum(m.execution_time for m in recent_metrics) / len(recent_metrics),
                    'average_memory_usage': sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics),
                    'average_cache_hit_rate': sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
                    'total_operations': sum(m.total_operations for m in recent_metrics),
                    'success_rate': sum(m.successful_operations for m in recent_metrics) / sum(m.total_operations for m in recent_metrics) if sum(m.total_operations for m in recent_metrics) > 0 else 0.0
                }
            
            return report
    
    def optimize_system_performance(self) -> OptimizationResult:
        """优化系统性能
        
        Returns:
            优化结果
        """
        with self._lock:
            start_time = time.time()
            optimization_actions = []
            memory_saved = 0.0
            
            try:
                # 检查内存状态并优化
                if self.memory_monitor:
                    memory_status = self.memory_monitor.check_memory_status()
                    current_memory = memory_status['memory_info']['current_memory_mb']
                    
                    if memory_status['status'] in ['warning', 'critical']:
                        # 清理缓存
                        cache_cleanup_result = self.cache_manager.clear()
                        if cache_cleanup_result['memory_cleared'] or cache_cleanup_result['disk_cleared']:
                            optimization_actions.append("清理缓存")
                        
                        # 清理过期缓存项
                        expired_cleanup = self.cache_manager.cleanup_expired()
                        if expired_cleanup['memory_cleaned'] > 0 or expired_cleanup['disk_cleaned'] > 0:
                            optimization_actions.append(f"清理过期缓存项: 内存{expired_cleanup['memory_cleaned']}项, 磁盘{expired_cleanup['disk_cleaned']}项")
                        
                        if memory_status['status'] == 'critical':
                            # 强制垃圾回收
                            gc.collect()
                            optimization_actions.append("执行垃圾回收")
                        
                        # 重新检查内存使用
                        new_memory_status = self.memory_monitor.check_memory_status()
                        new_memory = new_memory_status['memory_info']['current_memory_mb']
                        memory_saved = current_memory - new_memory
                
                # 检查缓存效率
                cache_stats = self.cache_manager.get_comprehensive_stats()
                overall_hit_rate = cache_stats['overall']['hit_rate']
                
                if overall_hit_rate < 0.3:  # 命中率低于30%
                    optimization_actions.append("缓存命中率较低，建议调整缓存策略")
                    
                    # 尝试优化缓存
                    cache_optimization = self.cache_manager.optimize_cache()
                    if cache_optimization['actions_taken']:
                        optimization_actions.extend(cache_optimization['actions_taken'])
                
                # 检查并行处理效率
                parallel_stats = self.parallel_manager.get_performance_stats()
                if parallel_stats['parallel_efficiency'] < 1.5 and parallel_stats['total_tasks'] > 100:
                    optimization_actions.append("并行处理效率较低，建议检查任务分配策略")
                
                # 重置统计信息（如果需要）
                if len(self._performance_history) > 1000:
                    self._performance_history = self._performance_history[-500:]
                    optimization_actions.append("清理性能历史记录")
                
                execution_time = time.time() - start_time
                
                result = OptimizationResult(
                    success=True,
                    actions_taken=optimization_actions,
                    memory_saved_mb=memory_saved,
                    execution_time_saved_seconds=0.0,  # 这个需要通过对比来计算
                    performance_improvement=len(optimization_actions) / 10.0  # 简单的改进指标
                )
                
                self._optimization_history.append(result)
                
                if optimization_actions:
                    logging.info(f"系统性能优化完成，执行了{len(optimization_actions)}个优化操作")
                
                return result
                
            except Exception as e:
                logging.error(f"系统性能优化失败: {e}")
                result = OptimizationResult(
                    success=False,
                    error_message=str(e)
                )
                self._optimization_history.append(result)
                return result
    
    def record_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """记录性能指标
        
        Args:
            metrics: 性能指标
        """
        with self._lock:
            self._performance_history.append(metrics)
            
            # 限制历史记录长度
            if len(self._performance_history) > 1000:
                self._performance_history = self._performance_history[-500:]
    
    def get_optimization_history(self, last_n: Optional[int] = None) -> List[OptimizationResult]:
        """获取优化历史记录
        
        Args:
            last_n: 获取最近N次优化记录，None表示获取全部
            
        Returns:
            优化历史记录列表
        """
        with self._lock:
            if last_n is None:
                return self._optimization_history.copy()
            else:
                return self._optimization_history[-last_n:]
    
    def reset_performance_data(self) -> None:
        """重置性能数据"""
        with self._lock:
            self._performance_history.clear()
            self._optimization_history.clear()
            
            if self.memory_monitor:
                self.memory_monitor.reset_monitoring_data()
            
            self.parallel_manager.reset_stats()
            self.vectorized_optimizer.reset_stats()
            
            logging.info("性能数据已重置")
    
    def update_configuration(self, 
                           cache_config: Optional[CacheConfig] = None,
                           parallel_config: Optional[ParallelConfig] = None,
                           monitoring_config: Optional[MonitoringConfig] = None) -> bool:
        """更新配置
        
        Args:
            cache_config: 新的缓存配置
            parallel_config: 新的并行处理配置
            monitoring_config: 新的监控配置
            
        Returns:
            是否成功更新
        """
        with self._lock:
            try:
                # 验证新配置
                if cache_config:
                    cache_config.validate()
                    self.cache_config = cache_config
                    # 注意：实际应用中可能需要重新初始化缓存管理器
                
                if parallel_config:
                    parallel_config.validate()
                    self.parallel_config = parallel_config
                    # 注意：实际应用中可能需要重新初始化并行管理器
                
                if monitoring_config:
                    monitoring_config.validate()
                    self.monitoring_config = monitoring_config
                    
                    # 更新内存监控器配置
                    if self.memory_monitor and monitoring_config.enable_memory_monitoring:
                        self.memory_monitor.set_thresholds(
                            monitoring_config.memory_warning_threshold_mb,
                            monitoring_config.memory_critical_threshold_mb
                        )
                
                logging.info("配置更新成功")
                return True
                
            except Exception as e:
                logging.error(f"配置更新失败: {e}")
                return False
    
    def get_performance_decorator(self):
        """获取性能监控装饰器
        
        Returns:
            配置好的性能监控装饰器
        """
        return performance_monitor(
            cache_manager=self.cache_manager,
            enable_memory_monitoring=self.monitoring_config.enable_memory_monitoring,
            log_threshold_seconds=self.monitoring_config.performance_log_threshold_seconds,
            memory_threshold_mb=self.monitoring_config.memory_log_threshold_mb
        )
    
    def export_performance_report(self, include_history: bool = True) -> Dict[str, Any]:
        """导出完整的性能报告
        
        Args:
            include_history: 是否包含历史数据
            
        Returns:
            完整的性能报告
        """
        with self._lock:
            report = self.get_comprehensive_performance_report()
            
            if include_history:
                report['performance_history'] = [m.to_dict() for m in self._performance_history]
                report['optimization_history'] = [r.to_dict() for r in self._optimization_history]
            
            # 添加内存详细报告
            if self.memory_monitor:
                report['detailed_memory_report'] = self.memory_monitor.export_memory_report()
            
            return report


# 为了向后兼容，保留原有的导入方式
__all__ = ['PerformanceOptimizer']