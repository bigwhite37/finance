"""
性能监控装饰器模块

提供函数性能监控装饰器，用于自动监控函数执行时间和内存使用。
"""

import time
import logging
from functools import wraps
from typing import Optional, Callable, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # 创建占位符类
    class psutil:
        class Process:
            def memory_info(self):
                return type('obj', (object,), {'rss': 0, 'vms': 0})()

try:
    from ..cache.unified_cache import UnifiedCacheManager
except ImportError:
    UnifiedCacheManager = None


def performance_monitor(cache_manager: Optional[UnifiedCacheManager] = None,
                       enable_memory_monitoring: bool = True,
                       log_threshold_seconds: float = 1.0,
                       memory_threshold_mb: float = 100.0):
    """性能监控装饰器
    
    Args:
        cache_manager: 缓存管理器
        enable_memory_monitoring: 是否启用内存监控
        log_threshold_seconds: 记录日志的执行时间阈值
        memory_threshold_mb: 记录日志的内存增长阈值
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            start_memory = 0.0
            
            # 获取开始时的内存使用
            memory_monitoring_enabled = enable_memory_monitoring
            if memory_monitoring_enabled:
                try:
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024
                except Exception:
                    memory_monitoring_enabled = False
            
            try:
                # 检查缓存
                cache_key = None
                if cache_manager:
                    # 生成缓存键
                    cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
                    cached_result = cache_manager.get(cache_key)
                    if cached_result is not None:
                        return cached_result
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 存储到缓存
                if cache_manager and cache_key:
                    cache_manager.put(cache_key, result)
                
                return result
                
            finally:
                # 记录性能指标
                execution_time = time.time() - start_time
                
                if memory_monitoring_enabled:
                    try:
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / 1024 / 1024
                        memory_delta = end_memory - start_memory
                        
                        if memory_delta > memory_threshold_mb:
                            logging.warning(
                                f"函数 {func.__name__} 内存使用增长 {memory_delta:.1f}MB，"
                                f"执行时间 {execution_time:.3f}s"
                            )
                    except Exception:
                        pass
                
                if execution_time > log_threshold_seconds:
                    logging.info(f"函数 {func.__name__} 执行时间: {execution_time:.3f}s")
        
        return wrapper
    return decorator


def memory_profiler(threshold_mb: float = 50.0):
    """内存分析装饰器
    
    Args:
        threshold_mb: 内存增长阈值(MB)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024
                
                result = func(*args, **kwargs)
                
                end_memory = process.memory_info().rss / 1024 / 1024
                memory_delta = end_memory - start_memory
                
                if abs(memory_delta) > threshold_mb:
                    logging.info(
                        f"函数 {func.__name__} 内存变化: {memory_delta:+.1f}MB "
                        f"(开始: {start_memory:.1f}MB, 结束: {end_memory:.1f}MB)"
                    )
                
                return result
                
            except Exception as e:
                logging.warning(f"内存分析失败: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def execution_timer(log_level: str = 'INFO'):
    """执行时间计时装饰器
    
    Args:
        log_level: 日志级别
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                
                log_func = getattr(logging, log_level.lower(), logging.info)
                log_func(f"函数 {func.__name__} 执行时间: {execution_time:.3f}s")
        
        return wrapper
    return decorator