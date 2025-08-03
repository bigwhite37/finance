"""
事件驱动处理器

该模块提供事件驱动的异步处理功能，包括：
- 事件订阅和发布
- 异步事件处理
- 事件优先级管理
- 事件过滤和路由
"""

import asyncio
import heapq
import time
from typing import Dict, List, Callable, Any, Optional, Set
from collections import defaultdict
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PriorityEvent:
    """优先级事件数据类"""
    priority: int
    timestamp: float
    event_type: str
    event_data: Any
    event_id: str = field(default_factory=lambda: str(time.time()))
    
    def __lt__(self, other):
        """优先级比较（数值越大优先级越高）"""
        return self.priority > other.priority


class EventDrivenProcessor:
    """事件驱动处理器"""
    
    def __init__(self, max_queue_size: int = 1000):
        """
        初始化事件驱动处理器
        
        Args:
            max_queue_size: 最大队列大小
        """
        self.max_queue_size = max_queue_size
        
        # 事件订阅者
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        
        # 优先级队列
        self.priority_queue: List[PriorityEvent] = []
        self.queue_lock = asyncio.Lock()
        
        # 处理状态
        self.processing = False
        self.processor_task = None
        
        # 统计信息
        self.stats = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'active_subscribers': 0
        }
        
        logger.info("事件驱动处理器初始化完成")
    
    def subscribe(self, event_type: str, handler: Callable):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理器
        """
        self.subscribers[event_type].add(handler)
        self.stats['active_subscribers'] = sum(len(handlers) for handlers in self.subscribers.values())
        
        logger.debug(f"订阅事件: {event_type}, 处理器: {handler.__name__}")
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """
        取消订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理器
        """
        if event_type in self.subscribers:
            self.subscribers[event_type].discard(handler)
            if not self.subscribers[event_type]:
                del self.subscribers[event_type]
        
        self.stats['active_subscribers'] = sum(len(handlers) for handlers in self.subscribers.values())
        
        logger.debug(f"取消订阅事件: {event_type}, 处理器: {handler.__name__}")
    
    async def emit(self, event_type: str, event_data: Any):
        """
        发送事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        await self.emit_with_priority(event_type, event_data, priority=5)  # 默认中等优先级
    
    async def emit_with_priority(self, 
                               event_type: str, 
                               event_data: Any, 
                               priority: int = 5):
        """
        发送带优先级的事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
            priority: 优先级（1-10，10最高）
        """
        if event_type not in self.subscribers:
            logger.debug(f"没有订阅者处理事件: {event_type}")
            return
        
        # 创建优先级事件
        priority_event = PriorityEvent(
            priority=priority,
            timestamp=time.time(),
            event_type=event_type,
            event_data=event_data
        )
        
        async with self.queue_lock:
            if len(self.priority_queue) >= self.max_queue_size:
                # 队列满时，移除最低优先级的事件
                self.priority_queue.sort()
                self.priority_queue.pop()
                logger.warning("事件队列已满，移除最低优先级事件")
            
            heapq.heappush(self.priority_queue, priority_event)
        
        self.stats['events_published'] += 1
        
        # 如果处理器未运行，启动它
        if not self.processing:
            await self.start_processing()
        
        logger.debug(f"事件已发送: {event_type}, 优先级: {priority}")
    
    async def start_processing(self):
        """启动事件处理"""
        if self.processing:
            return
        
        self.processing = True
        self.processor_task = asyncio.create_task(self._process_events())
        
        logger.info("事件处理已启动")
    
    async def stop_processing(self):
        """停止事件处理"""
        if not self.processing:
            return
        
        self.processing = False
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("事件处理已停止")
    
    async def _process_events(self):
        """事件处理循环"""
        while self.processing:
            try:
                # 获取下一个事件
                event = await self._get_next_event()
                if event is None:
                    await asyncio.sleep(0.01)  # 没有事件时短暂休眠
                    continue
                
                # 处理事件
                await self._handle_event(event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"事件处理循环错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _get_next_event(self) -> Optional[PriorityEvent]:
        """获取下一个要处理的事件"""
        async with self.queue_lock:
            if self.priority_queue:
                return heapq.heappop(self.priority_queue)
            return None
    
    async def _handle_event(self, event: PriorityEvent):
        """处理单个事件"""
        handlers = self.subscribers.get(event.event_type, set())
        
        if not handlers:
            return
        
        # 并发执行所有处理器
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(self._safe_call_handler(handler, event.event_data))
            tasks.append(task)
        
        if tasks:
            # 等待所有处理器完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 统计处理结果
            for result in results:
                if isinstance(result, Exception):
                    self.stats['events_failed'] += 1
                    logger.error(f"事件处理器失败: {result}")
                else:
                    self.stats['events_processed'] += 1
    
    async def _safe_call_handler(self, handler: Callable, event_data: Any):
        """安全调用事件处理器"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event_data)
            else:
                # 同步处理器在线程池中执行
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler, event_data)
        except Exception as e:
            logger.error(f"事件处理器 {handler.__name__} 执行失败: {e}")
            raise
    
    async def flush_events(self, timeout: float = 10.0):
        """刷新所有待处理事件"""
        start_time = time.time()
        
        while self.priority_queue and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        if self.priority_queue:
            logger.warning(f"刷新超时，仍有 {len(self.priority_queue)} 个事件待处理")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'queue_size': len(self.priority_queue),
            'subscriber_count': len(self.subscribers),
            'processing': self.processing
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        if not self.priority_queue:
            return {
                'size': 0,
                'max_size': self.max_queue_size,
                'usage_percent': 0.0
            }
        
        return {
            'size': len(self.priority_queue),
            'max_size': self.max_queue_size,
            'usage_percent': len(self.priority_queue) / self.max_queue_size * 100,
            'highest_priority': max(event.priority for event in self.priority_queue),
            'lowest_priority': min(event.priority for event in self.priority_queue)
        }
    
    async def clear_queue(self):
        """清空事件队列"""
        async with self.queue_lock:
            cleared_count = len(self.priority_queue)
            self.priority_queue.clear()
        
        logger.info(f"已清空事件队列，移除 {cleared_count} 个事件")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start_processing()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop_processing()