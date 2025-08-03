"""
内存优化器

该模块提供内存优化功能，包括：
- 内存使用监控
- 内存泄漏检测
- 缓存管理
- 大数据集处理优化
"""

import gc
import psutil
import weakref
import time
import threading
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Callable, Iterator, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class MemoryCache:
    """内存缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        初始化内存缓存
        
        Args:
            max_size: 最大缓存项目数
            ttl: 生存时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return default
            
            # 检查是否过期
            if time.time() - self._timestamps[key] > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                self._misses += 1
                return default
            
            # 移到末尾（LRU策略）
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
    
    def set(self, key: str, value: Any):
        """设置缓存值"""
        with self._lock:
            # 如果达到最大大小，删除最旧的项目
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key, _ = self._cache.popitem(last=False)
                del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'ttl': self.ttl
            }


class MemoryOptimizer:
    """内存优化器"""
    
    def __init__(self):
        """初始化内存优化器"""
        self.process = psutil.Process()
        self.memory_baseline = self.get_memory_usage()
        self._memory_pool = {}
        self._pool_lock = threading.Lock()
        
    def get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化DataFrame内存使用"""
        memory_before = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        optimized_df = df.copy()
        
        # 优化数值类型
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # 优化字符串类型
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].dtype == 'object':
                try:
                    optimized_df[col] = optimized_df[col].astype('category')
                except Exception:
                    pass
        
        memory_after = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
        
        logger.info(f"DataFrame内存优化: {memory_before:.2f}MB -> {memory_after:.2f}MB "
                   f"(节省 {(memory_before - memory_after)/memory_before*100:.1f}%)")
        
        return optimized_df
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """强制垃圾回收"""
        memory_before = self.get_memory_usage()
        
        collected = gc.collect()
        
        memory_after = self.get_memory_usage()
        memory_freed = memory_before - memory_after
        
        return {
            'objects_collected': collected,
            'memory_freed_mb': memory_freed,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after
        }
    
    def process_in_chunks(self, data: np.ndarray, chunk_size: int) -> Iterator[np.ndarray]:
        """分块处理大型数据集"""
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    def initialize_memory_pool(self, pool_size: int):
        """初始化内存池"""
        with self._pool_lock:
            self._memory_pool['available'] = [np.zeros(pool_size, dtype=np.float64)]
            self._memory_pool['in_use'] = []
    
    def get_array_from_pool(self, size: int) -> np.ndarray:
        """从内存池获取数组"""
        with self._pool_lock:
            if 'available' not in self._memory_pool:
                # 如果没有初始化内存池，直接分配
                return np.zeros(size, dtype=np.float64)
            
            # 查找合适大小的数组
            for i, arr in enumerate(self._memory_pool['available']):
                if len(arr) >= size:
                    # 移到使用中列表
                    array = self._memory_pool['available'].pop(i)
                    self._memory_pool['in_use'].append(array)
                    return array[:size]
            
            # 如果没有找到合适的，分配新数组
            new_array = np.zeros(size, dtype=np.float64)
            self._memory_pool['in_use'].append(new_array)
            return new_array
    
    def return_array_to_pool(self, array: np.ndarray):
        """归还数组到内存池"""
        with self._pool_lock:
            if 'in_use' in self._memory_pool and array in self._memory_pool['in_use']:
                self._memory_pool['in_use'].remove(array)
                self._memory_pool['available'].append(array)
    
    def memory_efficient_matrix_multiply(self, a: np.ndarray, b: np.ndarray, 
                                       chunk_size: int = 1000) -> np.ndarray:
        """内存高效的矩阵乘法"""
        if a.shape[1] != b.shape[0]:
            raise ValueError("矩阵维度不匹配")
        
        result = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
        
        # 分块计算以减少内存使用
        for i in range(0, a.shape[0], chunk_size):
            end_i = min(i + chunk_size, a.shape[0])
            for j in range(0, b.shape[1], chunk_size):
                end_j = min(j + chunk_size, b.shape[1])
                
                # 计算子块
                result[i:end_i, j:end_j] = np.dot(a[i:end_i, :], b[:, j:end_j])
        
        return result


class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        """初始化内存分析器"""
        self.process = psutil.Process()
    
    def get_current_memory(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def profile_function(self, func: Callable, *args, **kwargs) -> tuple:
        """
        分析函数内存使用
        
        Returns:
            tuple: (函数结果, 内存分析结果)
        """
        start_memory = self.get_current_memory()
        start_time = time.time()
        
        peak_memory = start_memory
        
        # 在后台监控内存使用
        def monitor_memory():
            nonlocal peak_memory
            while getattr(monitor_memory, 'running', True):
                current_memory = self.get_current_memory()
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.01)  # 10ms采样
        
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.daemon = True
        monitor_memory.running = True
        monitor_thread.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            monitor_memory.running = False
            monitor_thread.join(timeout=1)
        
        end_time = time.time()
        end_memory = self.get_current_memory()
        
        profile = {
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': end_memory - start_memory,
            'execution_time': end_time - start_time
        }
        
        return result, profile
    
    @contextmanager
    def profile_context(self):
        """内存分析上下文管理器"""
        start_memory = self.get_current_memory()
        peak_memory = start_memory
        
        profile = {
            'start_memory': start_memory,
            'peak_memory': peak_memory
        }
        
        try:
            yield profile
        finally:
            end_memory = self.get_current_memory()
            profile.update({
                'end_memory': end_memory,
                'peak_memory': max(profile['peak_memory'], end_memory),
                'memory_delta': end_memory - start_memory
            })
    
    def detect_memory_leak(self, initial_memory: float, final_memory: float, 
                          threshold_mb: float = 50) -> bool:
        """检测内存泄漏"""
        memory_growth = final_memory - initial_memory
        return memory_growth > threshold_mb
    
    def analyze_memory_hotspots(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """分析内存热点"""
        result, profile = self.profile_function(func, *args, **kwargs)
        
        # 简化的热点分析
        hotspots = {
            'total_allocations': 1,  # 简化实现
            'peak_usage_mb': profile['peak_memory_mb'],
            'allocation_pattern': 'sequential' if profile['memory_growth_mb'] > 0 else 'stable'
        }
        
        return hotspots


class MemoryEfficientPortfolioData:
    """内存高效的投资组合数据结构"""
    
    def __init__(self):
        """初始化内存高效投资组合数据"""
        self.weights_matrix = None
        self.returns_matrix = None
        self.metadata_list = []
        self.portfolio_count = 0
    
    def add_portfolio(self, portfolio_id: int, weights: np.ndarray, 
                     returns: np.ndarray, metadata: Dict[str, Any]):
        """添加投资组合"""
        if self.weights_matrix is None:
            # 初始化矩阵
            self.weights_matrix = weights.reshape(1, -1)
            self.returns_matrix = returns.reshape(1, -1)
        else:
            # 追加数据
            self.weights_matrix = np.vstack([self.weights_matrix, weights])
            self.returns_matrix = np.vstack([self.returns_matrix, returns])
        
        metadata['portfolio_id'] = portfolio_id
        self.metadata_list.append(metadata)
        self.portfolio_count += 1
    
    def get_portfolio(self, portfolio_id: int) -> Dict[str, Any]:
        """获取投资组合数据"""
        if portfolio_id >= self.portfolio_count:
            raise IndexError(f"投资组合ID {portfolio_id} 超出范围")
        
        return {
            'weights': self.weights_matrix[portfolio_id],
            'returns': self.returns_matrix[portfolio_id],
            'metadata': self.metadata_list[portfolio_id]
        }
    
    def get_portfolio_count(self) -> int:
        """获取投资组合数量"""
        return self.portfolio_count
    
    def get_memory_usage(self) -> float:
        """获取内存使用量（字节）"""
        memory = 0
        if self.weights_matrix is not None:
            memory += self.weights_matrix.nbytes
        if self.returns_matrix is not None:
            memory += self.returns_matrix.nbytes
        return memory


class SparseMatrixOptimizer:
    """稀疏矩阵优化器"""
    
    def optimize_sparse_matrix(self, matrix: np.ndarray, threshold: float = 1e-10) -> sp.csr_matrix:
        """优化稀疏矩阵"""
        # 将小于阈值的值设为0
        matrix_copy = matrix.copy()
        matrix_copy[np.abs(matrix_copy) < threshold] = 0
        
        # 转换为稀疏矩阵
        sparse_matrix = sp.csr_matrix(matrix_copy)
        
        return sparse_matrix
    
    def get_memory_usage(self, sparse_matrix: sp.csr_matrix) -> int:
        """获取稀疏矩阵内存使用量"""
        return (sparse_matrix.data.nbytes + 
                sparse_matrix.indices.nbytes + 
                sparse_matrix.indptr.nbytes)
    
    def to_dense(self, sparse_matrix: sp.csr_matrix) -> np.ndarray:
        """转换回密集矩阵"""
        return sparse_matrix.toarray()