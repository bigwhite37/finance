"""
异步优化器

该模块提供异步处理优化功能，包括：
- 异步函数执行优化
- 并发任务管理
- 异步批处理
- 异步流水线处理
"""

import asyncio
import time
import concurrent.futures
from typing import Dict, List, Any, Callable, Optional, Union, Coroutine
import logging
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)


class AsyncOptimizer:
    """异步优化器"""
    
    def __init__(self, 
                 max_workers: int = 4, 
                 max_concurrent_tasks: int = 10,
                 timeout: float = 300.0):
        """
        初始化异步优化器
        
        Args:
            max_workers: 最大工作线程数
            max_concurrent_tasks: 最大并发任务数
            timeout: 默认超时时间（秒）
        """
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.timeout = timeout
        
        # 创建线程池执行器
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # 任务跟踪
        self.active_tasks = set()
        self.completed_tasks = []
        
        logger.info(f"异步优化器初始化完成，最大工作线程: {max_workers}, 最大并发任务: {max_concurrent_tasks}")
    
    async def execute_async_function(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行异步函数
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        async with self.semaphore:
            try:
                if asyncio.iscoroutinefunction(func):
                    # 如果是协程函数，直接await
                    result = await func(*args, **kwargs)
                else:
                    # 如果是同步函数，在线程池中执行
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(self.executor, func, *args, **kwargs)
                
                return result
                
            except Exception as e:
                logger.error(f"异步函数执行失败: {e}")
                raise
    
    async def submit_task(self, func: Callable, *args, **kwargs) -> asyncio.Task:
        """
        提交异步任务
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            asyncio.Task: 异步任务对象
        """
        task = asyncio.create_task(self.execute_async_function(func, *args, **kwargs))
        self.active_tasks.add(task)
        
        # 任务完成时的清理
        def task_done_callback(completed_task):
            self.active_tasks.discard(completed_task)
            self.completed_tasks.append(completed_task)
        
        task.add_done_callback(task_done_callback)
        return task
    
    async def process_batches_async(self, 
                                  func: Callable, 
                                  data_batches: List[Any],
                                  batch_size: int = None) -> List[Any]:
        """
        异步批处理
        
        Args:
            func: 处理函数
            data_batches: 数据批次列表
            batch_size: 批次大小（用于进一步分组）
            
        Returns:
            处理结果列表
        """
        if batch_size and len(data_batches) > batch_size:
            # 如果需要进一步分组
            grouped_batches = [
                data_batches[i:i + batch_size] 
                for i in range(0, len(data_batches), batch_size)
            ]
        else:
            grouped_batches = [data_batches]
        
        all_results = []
        
        for group in grouped_batches:
            # 为每个组创建并发任务
            tasks = [self.submit_task(func, batch) for batch in group]
            
            # 等待当前组的所有任务完成
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            for i, result in enumerate(group_results):
                if isinstance(result, Exception):
                    logger.error(f"批处理任务 {i} 失败: {result}")
                    all_results.append(None)
                else:
                    all_results.append(result)
        
        return all_results
    
    async def execute_pipeline(self, 
                             stages: List[Callable], 
                             initial_data: Any) -> Any:
        """
        执行异步流水线
        
        Args:
            stages: 流水线阶段函数列表
            initial_data: 初始数据
            
        Returns:
            流水线最终结果
        """
        current_data = initial_data
        
        for i, stage in enumerate(stages):
            try:
                logger.debug(f"执行流水线阶段 {i+1}/{len(stages)}")
                current_data = await self.execute_async_function(stage, current_data)
            except Exception as e:
                logger.error(f"流水线阶段 {i+1} 执行失败: {e}")
                raise
        
        return current_data
    
    async def parallel_map(self, 
                          func: Callable, 
                          items: List[Any],
                          max_concurrent: int = None) -> List[Any]:
        """
        并行映射函数
        
        Args:
            func: 映射函数
            items: 要处理的项目列表
            max_concurrent: 最大并发数
            
        Returns:
            处理结果列表
        """
        if max_concurrent is None:
            max_concurrent = self.max_concurrent_tasks
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_func(item):
            async with semaphore:
                return await self.execute_async_function(func, item)
        
        tasks = [bounded_func(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def wait_for_completion(self, timeout: float = None) -> Dict[str, Any]:
        """
        等待所有活动任务完成
        
        Args:
            timeout: 超时时间
            
        Returns:
            完成状态信息
        """
        if not self.active_tasks:
            return {
                'status': 'no_active_tasks',
                'completed_count': len(self.completed_tasks)
            }
        
        timeout = timeout or self.timeout
        
        try:
            # 等待所有活动任务完成
            await asyncio.wait_for(
                asyncio.gather(*self.active_tasks, return_exceptions=True),
                timeout=timeout
            )
            
            return {
                'status': 'all_completed',
                'completed_count': len(self.completed_tasks),
                'timeout': False
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"等待任务完成超时: {timeout}秒")
            return {
                'status': 'timeout',
                'active_count': len(self.active_tasks),
                'completed_count': len(self.completed_tasks),
                'timeout': True
            }
    
    async def cancel_all_tasks(self):
        """取消所有活动任务"""
        if not self.active_tasks:
            return
        
        logger.info(f"取消 {len(self.active_tasks)} 个活动任务")
        
        for task in self.active_tasks.copy():
            task.cancel()
        
        # 等待任务取消
        await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        self.active_tasks.clear()
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_workers': self.max_workers,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'available_permits': self.semaphore._value
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cancel_all_tasks()
        self.executor.shutdown(wait=True)


# 为了兼容测试，从event_driven_processor导入EventDrivenProcessor
try:
    from .event_driven_processor import EventDrivenProcessor
except ImportError:
    # 如果导入失败，提供一个简化的实现
    class EventDrivenProcessor:
        def __init__(self):
            self.subscribers = {}
        
        def subscribe(self, event_type: str, handler: Callable):
            if event_type not in self.subscribers:
                self.subscribers[event_type] = set()
            self.subscribers[event_type].add(handler)
        
        def unsubscribe(self, event_type: str, handler: Callable):
            if event_type in self.subscribers:
                self.subscribers[event_type].discard(handler)
        
        async def emit(self, event_type: str, event_data: Any):
            if event_type in self.subscribers:
                tasks = []
                for handler in self.subscribers[event_type]:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event_data))
                    else:
                        tasks.append(asyncio.create_task(asyncio.to_thread(handler, event_data)))
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        
        async def emit_with_priority(self, event_type: str, event_data: Any, priority: int = 5):
            await self.emit(event_type, event_data)