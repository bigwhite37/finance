"""
动态低波筛选器性能优化模块

实现GARCH模型结果缓存、并行处理、向量化计算优化和内存监控功能。
"""

import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple, Callable, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import pickle
import hashlib
import os
from functools import wraps
import logging


# ============================================================================
# 性能监控数据结构
# ============================================================================

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


@dataclass
class CacheConfig:
    """缓存配置"""
    enable_disk_cache: bool = True
    enable_memory_cache: bool = True
    max_memory_cache_size: int = 1000  # 最大内存缓存条目数
    max_disk_cache_size_mb: int = 500  # 最大磁盘缓存大小(MB)
    cache_expiry_hours: int = 24  # 缓存过期时间(小时)
    cache_directory: str = "./cache/dynamic_lowvol"
    compression_level: int = 6  # 压缩级别(0-9)


@dataclass
class ParallelConfig:
    """并行处理配置"""
    enable_parallel: bool = True
    max_workers: Optional[int] = None  # None表示使用CPU核心数
    use_process_pool: bool = False  # True使用进程池，False使用线程池
    chunk_size: int = 10  # 批处理大小
    timeout_seconds: int = 300  # 超时时间


# ============================================================================
# 高级缓存管理器
# ============================================================================

class AdvancedCacheManager:
    """高级缓存管理器
    
    支持内存缓存、磁盘缓存、LRU淘汰策略和缓存压缩。
    """
    
    def __init__(self, config: CacheConfig):
        """初始化缓存管理器
        
        Args:
            config: 缓存配置
        """
        self.config = config
        self._memory_cache = {} if config.enable_memory_cache else None
        self._cache_access_times = {} if config.enable_memory_cache else None
        self._cache_lock = threading.RLock()
        
        # 创建磁盘缓存目录
        if config.enable_disk_cache:
            os.makedirs(config.cache_directory, exist_ok=True)
        
        # 性能统计
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'disk_hits': 0,
            'memory_hits': 0,
            'evictions': 0,
            'disk_writes': 0,
            'disk_reads': 0
        }
    
    def get(self, key: str, default=None) -> Any:
        """获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值或默认值
        """
        with self._cache_lock:
            # 首先检查内存缓存
            if self._memory_cache and key in self._memory_cache:
                self._cache_stats['hits'] += 1
                self._cache_stats['memory_hits'] += 1
                self._cache_access_times[key] = time.time()
                return self._memory_cache[key]
            
            # 检查磁盘缓存
            if self.config.enable_disk_cache:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    self._cache_stats['hits'] += 1
                    self._cache_stats['disk_hits'] += 1
                    
                    # 将磁盘缓存加载到内存缓存
                    if self._memory_cache is not None:
                        self._put_to_memory(key, disk_value)
                    
                    return disk_value
            
            # 缓存未命中
            self._cache_stats['misses'] += 1
            return default
    
    def put(self, key: str, value: Any) -> None:
        """存储缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._cache_lock:
            # 存储到内存缓存
            if self._memory_cache is not None:
                self._put_to_memory(key, value)
            
            # 存储到磁盘缓存
            if self.config.enable_disk_cache:
                self._put_to_disk(key, value)
    
    def _put_to_memory(self, key: str, value: Any) -> None:
        """存储到内存缓存"""
        # 如果key不存在且缓存已满，需要淘汰LRU项
        if key not in self._memory_cache and len(self._memory_cache) >= self.config.max_memory_cache_size:
            self._evict_lru_memory()
        
        self._memory_cache[key] = value
        self._cache_access_times[key] = time.time()
    
    def _put_to_disk(self, key: str, value: Any) -> None:
        """存储到磁盘缓存"""
        cache_file = self._get_cache_file_path(key)
        
        # 检查磁盘缓存大小限制
        self._check_disk_cache_size()
        
        # 序列化并压缩数据
        serialized_data = pickle.dumps(value)
        
        if self.config.compression_level > 0:
            import gzip
            with gzip.open(cache_file, 'wb', compresslevel=self.config.compression_level) as f:
                f.write(serialized_data)
        else:
            with open(cache_file, 'wb') as f:
                f.write(serialized_data)
        
        self._cache_stats['disk_writes'] += 1
    
    def _get_from_disk(self, key: str) -> Any:
        """从磁盘缓存获取值"""
        try:
            cache_file = self._get_cache_file_path(key)
            
            if not os.path.exists(cache_file):
                return None
            
            # 检查缓存是否过期
            if self._is_cache_expired(cache_file):
                os.remove(cache_file)
                return None
            
            # 读取并反序列化数据
            if self.config.compression_level > 0:
                import gzip
                with gzip.open(cache_file, 'rb') as f:
                    serialized_data = f.read()
            else:
                with open(cache_file, 'rb') as f:
                    serialized_data = f.read()
            
            value = pickle.loads(serialized_data)
            self._cache_stats['disk_reads'] += 1
            
            return value
            
        except Exception as e:
            logging.warning(f"磁盘缓存读取失败: {e}")
            return None
    
    def _get_cache_file_path(self, key: str) -> str:
        """获取缓存文件路径"""
        # 使用MD5哈希作为文件名
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.config.cache_directory, f"{key_hash}.cache")
    
    def _is_cache_expired(self, cache_file: str) -> bool:
        """检查缓存是否过期"""
        try:
            file_time = os.path.getmtime(cache_file)
            current_time = time.time()
            expiry_seconds = self.config.cache_expiry_hours * 3600
            
            return (current_time - file_time) > expiry_seconds
        except:
            return True
    
    def _evict_lru_memory(self) -> None:
        """淘汰最近最少使用的内存缓存项"""
        if not self._cache_access_times:
            return
        
        # 找到最久未访问的键
        lru_key = min(self._cache_access_times.keys(), 
                     key=lambda k: self._cache_access_times[k])
        
        # 删除LRU项
        del self._memory_cache[lru_key]
        del self._cache_access_times[lru_key]
        self._cache_stats['evictions'] += 1
    
    def _check_disk_cache_size(self) -> None:
        """检查并清理磁盘缓存大小"""
        try:
            cache_dir = self.config.cache_directory
            if not os.path.exists(cache_dir):
                return
            
            # 计算缓存目录大小
            total_size = 0
            cache_files = []
            
            for filename in os.listdir(cache_dir):
                if filename.endswith('.cache'):
                    filepath = os.path.join(cache_dir, filename)
                    if os.path.isfile(filepath):
                        file_size = os.path.getsize(filepath)
                        file_time = os.path.getmtime(filepath)
                        total_size += file_size
                        cache_files.append((filepath, file_size, file_time))
            
            # 如果超过大小限制，删除最旧的文件
            max_size_bytes = self.config.max_disk_cache_size_mb * 1024 * 1024
            
            if total_size > max_size_bytes:
                # 按修改时间排序，删除最旧的文件
                cache_files.sort(key=lambda x: x[2])  # 按时间排序
                
                for filepath, file_size, _ in cache_files:
                    if total_size <= max_size_bytes:
                        break
                    
                    try:
                        os.remove(filepath)
                        total_size -= file_size
                    except:
                        pass
        
        except Exception as e:
            logging.warning(f"磁盘缓存大小检查失败: {e}")
    
    def clear_cache(self, memory_only: bool = False) -> None:
        """清理缓存
        
        Args:
            memory_only: 是否只清理内存缓存
        """
        with self._cache_lock:
            # 清理内存缓存
            if self._memory_cache:
                self._memory_cache.clear()
            if self._cache_access_times:
                self._cache_access_times.clear()
            
            # 清理磁盘缓存
            if not memory_only and self.config.enable_disk_cache:
                try:
                    cache_dir = self.config.cache_directory
                    if os.path.exists(cache_dir):
                        for filename in os.listdir(cache_dir):
                            if filename.endswith('.cache'):
                                filepath = os.path.join(cache_dir, filename)
                                os.remove(filepath)
                except Exception as e:
                    logging.warning(f"磁盘缓存清理失败: {e}")
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        with self._cache_lock:
            total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
            hit_rate = self._cache_stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'memory_cache_size': len(self._memory_cache) if self._memory_cache else 0,
                'disk_cache_files': self._count_disk_cache_files(),
                **self._cache_stats
            }
    
    def _count_disk_cache_files(self) -> int:
        """统计磁盘缓存文件数量"""
        try:
            cache_dir = self.config.cache_directory
            if not os.path.exists(cache_dir):
                return 0
            
            return len([f for f in os.listdir(cache_dir) if f.endswith('.cache')])
        except OSError:
            return 0


# ============================================================================
# 并行处理管理器
# ============================================================================

class ParallelProcessingManager:
    """并行处理管理器
    
    支持线程池和进程池并行处理，自动负载均衡和错误处理。
    """
    
    def __init__(self, config: ParallelConfig):
        """初始化并行处理管理器
        
        Args:
            config: 并行处理配置
        """
        self.config = config
        self._executor = None
        self._performance_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'parallel_execution_time': 0.0
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        if self.config.enable_parallel:
            max_workers = self.config.max_workers or cpu_count()
            
            if self.config.use_process_pool:
                self._executor = ProcessPoolExecutor(max_workers=max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    chunk_size: Optional[int] = None) -> List[Any]:
        """并行映射函数
        
        Args:
            func: 要并行执行的函数
            items: 输入项列表
            chunk_size: 批处理大小
            
        Returns:
            结果列表
        """
        if not self.config.enable_parallel or not items:
            # 串行执行
            return [func(item) for item in items]
        
        chunk_size = chunk_size or self.config.chunk_size
        start_time = time.time()
        
        try:
            # 分批处理
            chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
            
            # 提交任务
            futures = []
            for chunk in chunks:
                future = self._executor.submit(self._process_chunk, func, chunk)
                futures.append(future)
            
            # 收集结果
            results = []
            completed_tasks = 0
            failed_tasks = 0
            
            for future in as_completed(futures, timeout=self.config.timeout_seconds):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    completed_tasks += len(chunk_results)
                except Exception as e:
                    logging.error(f"并行任务执行失败: {e}")
                    failed_tasks += len(chunks[futures.index(future)])
            
            # 更新统计信息
            execution_time = time.time() - start_time
            self._performance_stats['total_tasks'] += len(items)
            self._performance_stats['completed_tasks'] += completed_tasks
            self._performance_stats['failed_tasks'] += failed_tasks
            self._performance_stats['parallel_execution_time'] += execution_time
            
            return results
            
        except Exception as e:
            logging.error(f"并行处理框架失败: {e}")
            # 向上抛出异常，而不是回退到串行执行
            raise
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """处理数据块"""
        results = []
        for item in chunk:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                logging.warning(f"处理项目失败: {e}")
                # 可以选择跳过失败的项目或使用默认值
                results.append(None)
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        total_tasks = self._performance_stats['total_tasks']
        if total_tasks == 0:
            return self._performance_stats
        
        success_rate = self._performance_stats['completed_tasks'] / total_tasks
        
        # 计算并行效率（如果有串行对比数据）
        parallel_time = self._performance_stats['parallel_execution_time']
        total_time = self._performance_stats['total_execution_time']
        
        efficiency = 0.0
        if total_time > 0 and parallel_time > 0:
            efficiency = total_time / parallel_time
        
        return {
            **self._performance_stats,
            'success_rate': success_rate,
            'parallel_efficiency': efficiency
        }


# ============================================================================
# 向量化计算优化器
# ============================================================================

class VectorizedComputeOptimizer:
    """向量化计算优化器
    
    提供高性能的向量化计算函数，替代循环操作。
    """
    
    @staticmethod
    def vectorized_rolling_volatility(returns: pd.DataFrame, 
                                    windows: List[int]) -> Dict[int, pd.DataFrame]:
        """向量化滚动波动率计算
        
        Args:
            returns: 收益率数据 (日期 x 股票)
            windows: 滚动窗口长度列表
            
        Returns:
            不同窗口的滚动波动率字典
        """
        results = {}
        
        # 使用pandas的向量化操作
        for window in windows:
            # 计算滚动标准差并年化
            rolling_std = returns.rolling(window=window, min_periods=window//2).std()
            results[window] = rolling_std * np.sqrt(252)
        
        return results
    
    @staticmethod
    def vectorized_percentile_ranks(data: pd.DataFrame, 
                                  axis: int = 1) -> pd.DataFrame:
        """向量化分位数排名计算
        
        Args:
            data: 输入数据
            axis: 计算轴 (0=按列, 1=按行)
            
        Returns:
            分位数排名数据
        """
        # 使用pandas的rank函数进行向量化排名
        return data.rank(axis=axis, pct=True, method='min')
    
    @staticmethod
    def vectorized_factor_regression(returns: pd.DataFrame,
                                   factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """向量化因子回归计算
        
        Args:
            returns: 股票收益率数据 (日期 x 股票)
            factors: 因子数据 (日期 x 因子)
            
        Returns:
            (回归系数, 残差)
        """
        # 确保数据对齐
        common_dates = returns.index.intersection(factors.index)
        aligned_returns = returns.loc[common_dates]
        aligned_factors = factors.loc[common_dates]
        
        # 使用numpy的向量化线性代数运算
        X = aligned_factors.values
        Y = aligned_returns.values
        
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # 批量计算回归系数: beta = (X'X)^(-1)X'Y
        try:
            XTX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            XTY = X_with_intercept.T @ Y
            betas = XTX_inv @ XTY
            
            # 计算残差
            Y_pred = X_with_intercept @ betas
            residuals = Y - Y_pred
            
            # 转换回DataFrame格式
            beta_df = pd.DataFrame(
                betas[1:].T,  # 排除截距项
                index=aligned_returns.columns,
                columns=aligned_factors.columns
            )
            
            residuals_df = pd.DataFrame(
                residuals,
                index=aligned_returns.index,
                columns=aligned_returns.columns
            )
            
            return beta_df, residuals_df
            
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，返回空结果
            beta_df = pd.DataFrame(
                np.nan,
                index=aligned_returns.columns,
                columns=aligned_factors.columns
            )
            residuals_df = pd.DataFrame(
                np.nan,
                index=aligned_returns.index,
                columns=aligned_returns.columns
            )
            
            return beta_df, residuals_df
    
    @staticmethod
    def vectorized_volatility_decomposition(residuals: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """向量化波动率分解
        
        Args:
            residuals: 残差数据 (日期 x 股票)
            
        Returns:
            (好波动序列, 坏波动序列)
        """
        # 分离正负残差
        positive_residuals = residuals.where(residuals > 0, np.nan)
        negative_residuals = residuals.where(residuals < 0, np.nan)
        
        # 向量化计算标准差
        good_volatility = positive_residuals.std() * np.sqrt(252)
        bad_volatility = negative_residuals.std().abs() * np.sqrt(252)
        
        # 填充缺失值
        good_volatility = good_volatility.fillna(0.0)
        bad_volatility = bad_volatility.fillna(0.0)
        
        return good_volatility, bad_volatility
    
    @staticmethod
    def vectorized_garch_batch_prediction(returns_dict: Dict[str, pd.Series],
                                        horizon: int = 5) -> Dict[str, float]:
        """批量GARCH预测（简化版本）
        
        Args:
            returns_dict: 股票收益率字典 {股票代码: 收益率序列}
            horizon: 预测期限
            
        Returns:
            预测波动率字典 {股票代码: 预测波动率}
        """
        predictions = {}
        
        for stock_code, returns_series in returns_dict.items():
            try:
                # 简化的GARCH预测：使用EWMA模型
                # 这里使用指数加权移动平均作为GARCH的快速近似
                lambda_param = 0.94  # RiskMetrics标准参数
                
                # 计算平方收益率
                squared_returns = returns_series ** 2
                
                # EWMA波动率预测
                ewma_var = squared_returns.ewm(alpha=1-lambda_param, adjust=False).mean()
                predicted_vol = np.sqrt(ewma_var.iloc[-1] * 252)  # 年化
                
                predictions[stock_code] = predicted_vol
                
            except Exception:
                # 预测失败时使用历史波动率
                predictions[stock_code] = returns_series.std() * np.sqrt(252)
        
        return predictions


# ============================================================================
# 内存监控器
# ============================================================================

class MemoryMonitor:
    """内存使用监控器
    
    监控内存使用情况，提供内存优化建议。
    """
    
    def __init__(self, warning_threshold_mb: float = 1000.0,
                 critical_threshold_mb: float = 2000.0):
        """初始化内存监控器
        
        Args:
            warning_threshold_mb: 警告阈值(MB)
            critical_threshold_mb: 临界阈值(MB)
        """
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self._memory_history = []
        self._peak_memory = 0.0
        
    def get_current_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况
        
        Returns:
            内存使用信息字典
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        current_memory_mb = memory_info.rss / 1024 / 1024
        virtual_memory_mb = memory_info.vms / 1024 / 1024
        
        # 更新峰值内存
        self._peak_memory = max(self._peak_memory, current_memory_mb)
        
        # 记录内存历史
        self._memory_history.append({
            'timestamp': time.time(),
            'memory_mb': current_memory_mb
        })
        
        # 限制历史记录长度
        if len(self._memory_history) > 1000:
            self._memory_history = self._memory_history[-500:]
        
        return {
            'current_memory_mb': current_memory_mb,
            'virtual_memory_mb': virtual_memory_mb,
            'peak_memory_mb': self._peak_memory,
            'memory_percent': process.memory_percent(),
            'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_status(self) -> Dict[str, Any]:
        """检查内存状态
        
        Returns:
            内存状态信息
        """
        memory_info = self.get_current_memory_usage()
        current_memory = memory_info['current_memory_mb']
        
        if current_memory >= self.critical_threshold:
            status = 'critical'
            message = f"内存使用达到临界水平: {current_memory:.1f}MB"
            recommendations = [
                "立即清理缓存",
                "减少并行处理线程数",
                "使用数据分块处理",
                "考虑重启应用程序"
            ]
        elif current_memory >= self.warning_threshold:
            status = 'warning'
            message = f"内存使用较高: {current_memory:.1f}MB"
            recommendations = [
                "清理不必要的缓存",
                "优化数据结构",
                "减少内存中的数据量"
            ]
        else:
            status = 'normal'
            message = f"内存使用正常: {current_memory:.1f}MB"
            recommendations = []
        
        return {
            'status': status,
            'message': message,
            'recommendations': recommendations,
            'memory_info': memory_info,
            'memory_trend': self._calculate_memory_trend()
        }
    
    def _calculate_memory_trend(self) -> str:
        """计算内存使用趋势"""
        if len(self._memory_history) < 10:
            return 'insufficient_data'
        
        recent_memory = [item['memory_mb'] for item in self._memory_history[-10:]]
        
        # 计算线性趋势
        x = np.arange(len(recent_memory))
        slope = np.polyfit(x, recent_memory, 1)[0]
        
        if slope > 5:  # 每次测量增长超过5MB
            return 'increasing'
        elif slope < -5:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_memory_optimization_suggestions(self) -> List[str]:
        """获取内存优化建议"""
        memory_info = self.get_current_memory_usage()
        suggestions = []
        
        if memory_info['current_memory_mb'] > 500:
            suggestions.append("考虑使用数据分块处理大型数据集")
        
        if memory_info['memory_percent'] > 80:
            suggestions.append("系统内存使用率过高，考虑增加物理内存")
        
        if self._peak_memory > memory_info['current_memory_mb'] * 2:
            suggestions.append("存在内存峰值，考虑优化算法以减少内存峰值使用")
        
        trend = self._calculate_memory_trend()
        if trend == 'increasing':
            suggestions.append("内存使用呈上升趋势，可能存在内存泄漏")
        
        return suggestions


# ============================================================================
# 性能装饰器
# ============================================================================

def performance_monitor(cache_manager: Optional[AdvancedCacheManager] = None,
                       enable_memory_monitoring: bool = True):
    """性能监控装饰器
    
    Args:
        cache_manager: 缓存管理器
        enable_memory_monitoring: 是否启用内存监控
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = 0.0
            
            if enable_memory_monitoring:
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024
            
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
                
                if enable_memory_monitoring:
                    end_memory = process.memory_info().rss / 1024 / 1024
                    memory_delta = end_memory - start_memory
                    
                    if memory_delta > 100:  # 内存增长超过100MB
                        logging.warning(
                            f"函数 {func.__name__} 内存使用增长 {memory_delta:.1f}MB，"
                            f"执行时间 {execution_time:.3f}s"
                        )
                
                if execution_time > 1.0:  # 执行时间超过1秒
                    logging.info(f"函数 {func.__name__} 执行时间: {execution_time:.3f}s")
        
        return wrapper
    return decorator


# ============================================================================
# 性能优化管理器
# ============================================================================

class PerformanceOptimizer:
    """性能优化管理器
    
    统一管理缓存、并行处理、向量化计算和内存监控。
    """
    
    def __init__(self, 
                 cache_config: Optional[CacheConfig] = None,
                 parallel_config: Optional[ParallelConfig] = None,
                 enable_memory_monitoring: bool = True):
        """初始化性能优化管理器
        
        Args:
            cache_config: 缓存配置
            parallel_config: 并行处理配置
            enable_memory_monitoring: 是否启用内存监控
        """
        # 初始化组件
        self.cache_manager = AdvancedCacheManager(
            cache_config or CacheConfig()
        )
        
        self.parallel_manager = ParallelProcessingManager(
            parallel_config or ParallelConfig()
        )
        
        self.vectorized_optimizer = VectorizedComputeOptimizer()
        
        if enable_memory_monitoring:
            self.memory_monitor = MemoryMonitor()
        else:
            self.memory_monitor = None
        
        # 性能统计
        self._performance_history = []
    
    def __enter__(self):
        """上下文管理器入口"""
        self.parallel_manager.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.parallel_manager.__exit__(exc_type, exc_val, exc_tb)
    
    def get_comprehensive_performance_report(self) -> Dict:
        """获取综合性能报告
        
        Returns:
            综合性能报告
        """
        report = {
            'timestamp': time.time(),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'parallel_stats': self.parallel_manager.get_performance_stats(),
        }
        
        if self.memory_monitor:
            report['memory_status'] = self.memory_monitor.check_memory_status()
            report['memory_suggestions'] = self.memory_monitor.get_memory_optimization_suggestions()
        
        return report
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """优化系统性能
        
        Returns:
            优化结果报告
        """
        optimization_actions = []
        
        # 检查内存状态并优化
        if self.memory_monitor:
            memory_status = self.memory_monitor.check_memory_status()
            
            if memory_status['status'] in ['warning', 'critical']:
                # 清理缓存
                self.cache_manager.clear_cache(memory_only=True)
                optimization_actions.append("清理内存缓存")
                
                if memory_status['status'] == 'critical':
                    # 强制垃圾回收
                    import gc
                    gc.collect()
                    optimization_actions.append("执行垃圾回收")
        
        # 检查缓存效率
        cache_stats = self.cache_manager.get_cache_stats()
        if cache_stats['hit_rate'] < 0.3:  # 命中率低于30%
            optimization_actions.append("缓存命中率较低，建议调整缓存策略")
        
        return {
            'optimization_actions': optimization_actions,
            'performance_report': self.get_comprehensive_performance_report()
        }