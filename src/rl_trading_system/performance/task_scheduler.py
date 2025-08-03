"""
任务调度器

该模块提供异步任务调度功能，包括：
- 优先级任务调度
- 并发控制
- 任务超时管理
- 资源限制调度
"""

import asyncio
import heapq
import time
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """调度任务数据类"""
    priority: int
    creation_time: float
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    timeout: Optional[float] = None
    task_obj: Optional[asyncio.Task] = field(default=None, init=False)
    
    def __lt__(self, other):
        """优先级比较（数值越大优先级越高）"""
        if self.priority != other.priority:
            return self.priority > other.priority
        # 相同优先级时，先创建的先执行
        return self.creation_time < other.creation_time


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, 
                 max_concurrent_tasks: int = 10,
                 max_workers: int = 4):
        """
        初始化任务调度器
        
        Args:
            max_concurrent_tasks: 最大并发任务数
            max_workers: 最大工作线程数
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_workers = max_workers
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 优先级队列
        self.task_queue: List[ScheduledTask] = []
        self.queue_lock = asyncio.Lock()
        
        # 任务跟踪
        self.active_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # 调度状态
        self.scheduler_running = False
        self.scheduler_task = None
        
        # 统计信息
        self.stats = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_timeout': 0
        }
        
        logger.info(f"任务调度器初始化完成，最大并发: {max_concurrent_tasks}")
    
    async def schedule_task(self, 
                          func: Callable, 
                          *args, 
                          priority: int = 5,
                          timeout: float = None,
                          **kwargs) -> asyncio.Task:
        """
        调度任务
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            priority: 优先级（1-10，10最高）
            timeout: 超时时间
            **kwargs: 关键字参数
            
        Returns:
            asyncio.Task: 任务对象
        """
        task_id = f"task_{time.time()}_{id(func)}"
        
        # 创建调度任务
        scheduled_task = ScheduledTask(
            priority=priority,
            creation_time=time.time(),
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout
        )
        
        # 添加到队列
        async with self.queue_lock:
            heapq.heappush(self.task_queue, scheduled_task)
        
        self.stats['tasks_scheduled'] += 1
        
        # 启动调度器
        if not self.scheduler_running:
            await self.start_scheduler()
        
        # 创建并返回任务对象
        task = asyncio.create_task(self._wait_for_task_completion(task_id))
        scheduled_task.task_obj = task
        
        logger.debug(f"任务已调度: {task_id}, 优先级: {priority}")
        return task
    
    async def schedule_priority_task(self, 
                                   func: Callable, 
                                   *args, 
                                   priority: int,
                                   **kwargs) -> asyncio.Task:
        """
        调度优先级任务
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            priority: 优先级
            **kwargs: 关键字参数
            
        Returns:
            asyncio.Task: 任务对象
        """
        return await self.schedule_task(func, *args, priority=priority, **kwargs)
    
    async def schedule_task_with_timeout(self, 
                                       func: Callable, 
                                       *args,
                                       timeout: float,
                                       **kwargs) -> Any:
        """
        调度带超时的任务
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            timeout: 超时时间
            **kwargs: 关键字参数
            
        Returns:
            任务执行结果
        """
        task = await self.schedule_task(func, *args, timeout=timeout, **kwargs)
        return await asyncio.wait_for(task, timeout=timeout)
    
    async def start_scheduler(self):
        """启动任务调度器"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("任务调度器已启动")
    
    async def stop_scheduler(self):
        """停止任务调度器"""
        if not self.scheduler_running:
            return
        
        self.scheduler_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有活动任务
        for task_id, scheduled_task in self.active_tasks.items():
            if scheduled_task.task_obj and not scheduled_task.task_obj.done():
                scheduled_task.task_obj.cancel()
        
        logger.info("任务调度器已停止")
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.scheduler_running:
            try:
                # 获取下一个任务
                scheduled_task = await self._get_next_task()
                
                if scheduled_task is None:
                    await asyncio.sleep(0.01)  # 没有任务时短暂休眠
                    continue
                
                # 执行任务
                asyncio.create_task(self._execute_scheduled_task(scheduled_task))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _get_next_task(self) -> Optional[ScheduledTask]:
        """获取下一个要执行的任务"""
        async with self.queue_lock:
            if self.task_queue:
                return heapq.heappop(self.task_queue)
            return None
    
    async def _execute_scheduled_task(self, scheduled_task: ScheduledTask):
        """执行调度任务"""
        async with self.semaphore:  # 控制并发数量
            self.active_tasks[scheduled_task.task_id] = scheduled_task
            
            try:
                # 执行任务
                if asyncio.iscoroutinefunction(scheduled_task.func):
                    if scheduled_task.timeout:
                        result = await asyncio.wait_for(
                            scheduled_task.func(*scheduled_task.args, **scheduled_task.kwargs),
                            timeout=scheduled_task.timeout
                        )
                    else:
                        result = await scheduled_task.func(*scheduled_task.args, **scheduled_task.kwargs)
                else:
                    # 同步函数在线程池中执行
                    loop = asyncio.get_event_loop()
                    if scheduled_task.timeout:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(
                                self.executor, 
                                scheduled_task.func, 
                                *scheduled_task.args
                            ),
                            timeout=scheduled_task.timeout
                        )
                    else:
                        result = await loop.run_in_executor(
                            self.executor, 
                            scheduled_task.func, 
                            *scheduled_task.args
                        )
                
                # 任务完成
                self.completed_tasks.append(scheduled_task.task_id)
                self.stats['tasks_completed'] += 1
                
                # 设置任务结果
                if scheduled_task.task_obj and not scheduled_task.task_obj.done():
                    scheduled_task.task_obj.set_result(result)
                
            except asyncio.TimeoutError:
                logger.warning(f"任务超时: {scheduled_task.task_id}")
                self.failed_tasks.append(scheduled_task.task_id)
                self.stats['tasks_timeout'] += 1
                
                if scheduled_task.task_obj and not scheduled_task.task_obj.done():
                    scheduled_task.task_obj.set_exception(asyncio.TimeoutError())
                
            except Exception as e:
                logger.error(f"任务执行失败: {scheduled_task.task_id}, 错误: {e}")
                self.failed_tasks.append(scheduled_task.task_id)
                self.stats['tasks_failed'] += 1
                
                if scheduled_task.task_obj and not scheduled_task.task_obj.done():
                    scheduled_task.task_obj.set_exception(e)
            
            finally:
                # 清理
                self.active_tasks.pop(scheduled_task.task_id, None)
    
    async def _wait_for_task_completion(self, task_id: str) -> Any:
        """等待任务完成"""
        # 等待任务被调度和执行
        while task_id not in self.completed_tasks and task_id not in self.failed_tasks:
            await asyncio.sleep(0.01)
        
        # 返回结果或抛出异常（这里简化处理）
        if task_id in self.completed_tasks:
            return f"Task {task_id} completed"
        else:
            raise RuntimeError(f"Task {task_id} failed")
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            'queued_tasks': len(self.task_queue),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'scheduler_running': self.scheduler_running,
            'available_slots': self.semaphore._value
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'queue_status': self.get_queue_status(),
            'success_rate': (self.stats['tasks_completed'] / 
                           max(1, self.stats['tasks_completed'] + self.stats['tasks_failed'] + self.stats['tasks_timeout']))
        }
    
    async def clear_queue(self):
        """清空任务队列"""
        async with self.queue_lock:
            cleared_count = len(self.task_queue)
            self.task_queue.clear()
        
        logger.info(f"已清空任务队列，移除 {cleared_count} 个任务")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start_scheduler()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop_scheduler()
        self.executor.shutdown(wait=True)