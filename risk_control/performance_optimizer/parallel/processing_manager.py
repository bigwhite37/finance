"""
并行处理管理器模块

提供多线程和多进程并行处理功能，支持自动负载均衡和错误处理。
"""

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple, Callable, Any
import traceback

from .config import ParallelConfig


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
        self._lock = threading.RLock()
        
        # 性能统计
        self._performance_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0,
            'parallel_execution_time': 0.0,
            'serial_execution_time': 0.0,
            'chunks_processed': 0,
            'average_chunk_time': 0.0
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        with self._lock:
            if self.config.enable_parallel and self._executor is None:
                max_workers = self.config.max_workers or cpu_count()
                
                try:
                    if self.config.use_process_pool:
                        self._executor = ProcessPoolExecutor(max_workers=max_workers)
                        logging.info(f"启动进程池，工作进程数: {max_workers}")
                    else:
                        self._executor = ThreadPoolExecutor(max_workers=max_workers)
                        logging.info(f"启动线程池，工作线程数: {max_workers}")
                except Exception as e:
                    logging.warning(f"创建执行器失败: {e}，将使用串行执行")
                    self._executor = None
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        with self._lock:
            if self._executor:
                try:
                    self._executor.shutdown(wait=True)
                    logging.info("并行执行器已关闭")
                except Exception as e:
                    logging.warning(f"关闭执行器时出错: {e}")
                finally:
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
        if not items:
            return []
        
        chunk_size = chunk_size or self.config.chunk_size
        start_time = time.time()
        
        # 如果不启用并行或没有执行器，使用串行执行
        if not self.config.enable_parallel or self._executor is None:
            return self._serial_execute(func, items, start_time)
        
        try:
            return self._parallel_execute(func, items, chunk_size, start_time)
        except Exception as e:
            logging.error(f"并行处理失败，回退到串行执行: {e}")
            logging.debug(f"错误详情: {traceback.format_exc()}")
            return self._serial_execute(func, items, start_time)
    
    def parallel_map_with_progress(self, func: Callable, items: List[Any],
                                 progress_callback: Optional[Callable[[int, int], None]] = None,
                                 chunk_size: Optional[int] = None) -> List[Any]:
        """带进度回调的并行映射函数
        
        Args:
            func: 要并行执行的函数
            items: 输入项列表
            progress_callback: 进度回调函数 (completed, total)
            chunk_size: 批处理大小
            
        Returns:
            结果列表
        """
        if not items:
            return []
        
        chunk_size = chunk_size or self.config.chunk_size
        start_time = time.time()
        
        if not self.config.enable_parallel or self._executor is None:
            return self._serial_execute_with_progress(func, items, progress_callback, start_time)
        
        try:
            return self._parallel_execute_with_progress(func, items, chunk_size, progress_callback, start_time)
        except Exception as e:
            logging.error(f"并行处理失败，回退到串行执行: {e}")
            return self._serial_execute_with_progress(func, items, progress_callback, start_time)
    
    def _parallel_execute(self, func: Callable, items: List[Any], 
                         chunk_size: int, start_time: float) -> List[Any]:
        """并行执行函数"""
        # 分批处理
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # 提交任务
        futures = []
        for chunk_idx, chunk in enumerate(chunks):
            future = self._executor.submit(self._process_chunk, func, chunk, chunk_idx)
            futures.append(future)
        
        # 收集结果
        results = []
        completed_tasks = 0
        failed_tasks = 0
        chunk_times = []
        
        for future in as_completed(futures, timeout=self.config.timeout_seconds):
            try:
                chunk_start_time = time.time()
                chunk_results, chunk_errors = future.result()
                chunk_time = time.time() - chunk_start_time
                chunk_times.append(chunk_time)
                
                results.extend(chunk_results)
                completed_tasks += len(chunk_results) - chunk_errors
                failed_tasks += chunk_errors
                
            except Exception as e:
                logging.error(f"并行任务执行失败: {e}")
                # 尝试找到对应的chunk大小
                chunk_idx = futures.index(future)
                if chunk_idx < len(chunks):
                    failed_tasks += len(chunks[chunk_idx])
        
        # 更新统计信息
        execution_time = time.time() - start_time
        with self._lock:
            self._performance_stats['total_tasks'] += len(items)
            self._performance_stats['completed_tasks'] += completed_tasks
            self._performance_stats['failed_tasks'] += failed_tasks
            self._performance_stats['parallel_execution_time'] += execution_time
            self._performance_stats['chunks_processed'] += len(chunks)
            
            if chunk_times:
                avg_chunk_time = sum(chunk_times) / len(chunk_times)
                self._performance_stats['average_chunk_time'] = avg_chunk_time
        
        return results
    
    def _parallel_execute_with_progress(self, func: Callable, items: List[Any],
                                      chunk_size: int, progress_callback: Optional[Callable],
                                      start_time: float) -> List[Any]:
        """带进度回调的并行执行函数"""
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        futures = []
        for chunk_idx, chunk in enumerate(chunks):
            future = self._executor.submit(self._process_chunk, func, chunk, chunk_idx)
            futures.append(future)
        
        results = []
        completed_tasks = 0
        failed_tasks = 0
        
        for future in as_completed(futures, timeout=self.config.timeout_seconds):
            try:
                chunk_results, chunk_errors = future.result()
                results.extend(chunk_results)
                completed_tasks += len(chunk_results) - chunk_errors
                failed_tasks += chunk_errors
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(completed_tasks, len(items))
                    
            except Exception as e:
                logging.error(f"并行任务执行失败: {e}")
                chunk_idx = futures.index(future)
                if chunk_idx < len(chunks):
                    failed_tasks += len(chunks[chunk_idx])
                
                if progress_callback:
                    progress_callback(completed_tasks, len(items))
        
        # 更新统计信息
        execution_time = time.time() - start_time
        with self._lock:
            self._performance_stats['total_tasks'] += len(items)
            self._performance_stats['completed_tasks'] += completed_tasks
            self._performance_stats['failed_tasks'] += failed_tasks
            self._performance_stats['parallel_execution_time'] += execution_time
            self._performance_stats['chunks_processed'] += len(chunks)
        
        return results
    
    def _serial_execute(self, func: Callable, items: List[Any], start_time: float) -> List[Any]:
        """串行执行函数"""
        results = []
        completed_tasks = 0
        failed_tasks = 0
        
        for item in items:
            try:
                result = func(item)
                results.append(result)
                completed_tasks += 1
            except Exception as e:
                logging.warning(f"处理项目失败: {e}")
                results.append(None)
                failed_tasks += 1
        
        # 更新统计信息
        execution_time = time.time() - start_time
        with self._lock:
            self._performance_stats['total_tasks'] += len(items)
            self._performance_stats['completed_tasks'] += completed_tasks
            self._performance_stats['failed_tasks'] += failed_tasks
            self._performance_stats['serial_execution_time'] += execution_time
        
        return results
    
    def _serial_execute_with_progress(self, func: Callable, items: List[Any],
                                    progress_callback: Optional[Callable],
                                    start_time: float) -> List[Any]:
        """带进度回调的串行执行函数"""
        results = []
        completed_tasks = 0
        failed_tasks = 0
        
        for i, item in enumerate(items):
            try:
                result = func(item)
                results.append(result)
                completed_tasks += 1
            except Exception as e:
                logging.warning(f"处理项目失败: {e}")
                results.append(None)
                failed_tasks += 1
            
            # 调用进度回调
            if progress_callback and (i + 1) % 10 == 0:  # 每10个项目回调一次
                progress_callback(completed_tasks, len(items))
        
        # 最终进度回调
        if progress_callback:
            progress_callback(completed_tasks, len(items))
        
        # 更新统计信息
        execution_time = time.time() - start_time
        with self._lock:
            self._performance_stats['total_tasks'] += len(items)
            self._performance_stats['completed_tasks'] += completed_tasks
            self._performance_stats['failed_tasks'] += failed_tasks
            self._performance_stats['serial_execution_time'] += execution_time
        
        return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any], chunk_idx: int) -> Tuple[List[Any], int]:
        """处理数据块
        
        Args:
            func: 处理函数
            chunk: 数据块
            chunk_idx: 块索引
            
        Returns:
            (结果列表, 错误数量)
        """
        results = []
        error_count = 0
        
        for item in chunk:
            try:
                result = func(item)
                results.append(result)
            except Exception as e:
                logging.warning(f"处理块{chunk_idx}中的项目失败: {e}")
                results.append(None)
                error_count += 1
        
        return results, error_count
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        with self._lock:
            total_tasks = self._performance_stats['total_tasks']
            if total_tasks == 0:
                return self._performance_stats
            
            success_rate = self._performance_stats['completed_tasks'] / total_tasks
            
            # 计算并行效率
            parallel_time = self._performance_stats['parallel_execution_time']
            serial_time = self._performance_stats['serial_execution_time']
            total_time = self._performance_stats['total_execution_time']
            
            efficiency = 0.0
            if parallel_time > 0 and serial_time > 0:
                efficiency = serial_time / parallel_time
            elif total_time > 0 and parallel_time > 0:
                efficiency = total_time / parallel_time
            
            return {
                **self._performance_stats,
                'success_rate': success_rate,
                'parallel_efficiency': efficiency,
                'average_tasks_per_second': total_tasks / (parallel_time + serial_time) if (parallel_time + serial_time) > 0 else 0.0
            }
    
    def reset_stats(self) -> None:
        """重置性能统计信息"""
        with self._lock:
            self._performance_stats = {
                'total_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'total_execution_time': 0.0,
                'parallel_execution_time': 0.0,
                'serial_execution_time': 0.0,
                'chunks_processed': 0,
                'average_chunk_time': 0.0
            }
    
    def is_parallel_enabled(self) -> bool:
        """检查是否启用并行处理"""
        return self.config.enable_parallel and self._executor is not None
    
    def get_worker_count(self) -> int:
        """获取工作线程/进程数量"""
        if self._executor is None:
            return 1
        
        return self.config.max_workers or cpu_count()