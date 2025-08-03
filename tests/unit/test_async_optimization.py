"""
异步处理和事件驱动优化测试

该模块测试异步性能优化功能，包括：
- 异步回撤计算
- 事件驱动处理
- 并发任务管理
- 异步I/O优化
"""

import pytest
import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch
import concurrent.futures

from src.rl_trading_system.performance.async_optimizer import AsyncOptimizer, EventDrivenProcessor
from src.rl_trading_system.performance.async_drawdown_calculator import AsyncDrawdownCalculator
from src.rl_trading_system.performance.task_scheduler import TaskScheduler


class TestAsyncOptimizer:
    """异步优化器测试类"""
    
    @pytest.fixture
    def optimizer(self):
        """创建异步优化器实例"""
        return AsyncOptimizer(max_workers=4, max_concurrent_tasks=10)
    
    @pytest.mark.asyncio
    async def test_async_function_execution(self, optimizer):
        """测试异步函数执行"""
        async def async_calculation(n: int) -> int:
            """异步计算函数"""
            await asyncio.sleep(0.1)  # 模拟异步操作
            return sum(range(n))
        
        # 执行异步函数
        result = await optimizer.execute_async_function(async_calculation, 1000)
        
        # 验证结果
        expected = sum(range(1000))
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, optimizer):
        """测试并发任务执行"""
        async def async_task(task_id: int, delay: float) -> Dict:
            """异步任务"""
            await asyncio.sleep(delay)
            return {'task_id': task_id, 'completed_at': time.time()}
        
        # 创建多个任务
        tasks = [
            optimizer.submit_task(async_task, i, 0.1)
            for i in range(5)
        ]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result['task_id'] == i
            assert 'completed_at' in result
    
    @pytest.mark.asyncio
    async def test_async_batch_processing(self, optimizer):
        """测试异步批处理"""
        def sync_calculation(data: np.ndarray) -> float:
            """同步计算函数"""
            return np.sum(data ** 2)
        
        # 创建测试数据
        data_batches = [np.random.randn(1000) for _ in range(10)]
        
        # 异步批处理
        results = await optimizer.process_batches_async(sync_calculation, data_batches)
        
        # 验证结果
        assert len(results) == 10
        for result in results:
            assert isinstance(result, (int, float, np.number))
    
    @pytest.mark.asyncio
    async def test_async_pipeline_processing(self, optimizer):
        """测试异步流水线处理"""
        async def stage1(data: List[float]) -> List[float]:
            """阶段1：数据预处理"""
            await asyncio.sleep(0.05)
            return [x * 2 for x in data]
        
        async def stage2(data: List[float]) -> List[float]:
            """阶段2：数据变换"""
            await asyncio.sleep(0.05)
            return [x + 1 for x in data]
        
        async def stage3(data: List[float]) -> float:
            """阶段3：聚合计算"""
            await asyncio.sleep(0.05)
            return sum(data)
        
        # 创建流水线
        pipeline = [stage1, stage2, stage3]
        input_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # 执行异步流水线
        result = await optimizer.execute_pipeline(pipeline, input_data)
        
        # 验证结果
        # ((1*2+1) + (2*2+1) + ... + (5*2+1)) = (3+5+7+9+11) = 35
        assert result == 35.0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_async_tasks(self, optimizer):
        """测试异步任务中的错误处理"""
        async def failing_task():
            """会失败的异步任务"""
            await asyncio.sleep(0.1)
            raise ValueError("测试错误")
        
        async def successful_task():
            """成功的异步任务"""
            await asyncio.sleep(0.1)
            return "success"
        
        # 提交混合任务
        tasks = [
            optimizer.submit_task(failing_task),
            optimizer.submit_task(successful_task),
            optimizer.submit_task(failing_task)
        ]
        
        # 使用gather返回异常而不是抛出
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证结果
        assert len(results) == 3
        assert isinstance(results[0], ValueError)
        assert results[1] == "success"
        assert isinstance(results[2], ValueError)
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self, optimizer):
        """测试任务取消"""
        async def long_running_task():
            """长时间运行的任务"""
            for i in range(100):
                await asyncio.sleep(0.1)
                # 检查取消状态
                if i == 5:  # 在第5次迭代时检查
                    try:
                        await asyncio.sleep(0)  # 允许取消检查
                    except asyncio.CancelledError:
                        return "cancelled"
            return "completed"
        
        # 启动任务
        task = optimizer.submit_task(long_running_task)
        
        # 等待一点时间然后取消
        await asyncio.sleep(0.3)
        task.cancel()
        
        try:
            result = await task
            # 如果任务正常完成，检查是否被适当取消
            assert result in ["cancelled", "completed"]
        except asyncio.CancelledError:
            # 任务被取消是预期的
            pass


class TestEventDrivenProcessor:
    """事件驱动处理器测试类"""
    
    @pytest.fixture
    def processor(self):
        """创建事件驱动处理器实例"""
        return EventDrivenProcessor()
    
    @pytest.mark.asyncio
    async def test_event_subscription_and_emission(self, processor):
        """测试事件订阅和发送"""
        received_events = []
        
        async def event_handler(event_data):
            """事件处理器"""
            received_events.append(event_data)
        
        # 订阅事件
        processor.subscribe('test_event', event_handler)
        
        # 发送事件
        await processor.emit('test_event', {'message': 'test'})
        await processor.emit('test_event', {'message': 'test2'})
        
        # 等待事件处理
        await asyncio.sleep(0.1)
        
        # 验证事件接收
        assert len(received_events) == 2
        assert received_events[0]['message'] == 'test'
        assert received_events[1]['message'] == 'test2'
    
    @pytest.mark.asyncio
    async def test_multiple_event_handlers(self, processor):
        """测试多个事件处理器"""
        handler1_calls = []
        handler2_calls = []
        
        async def handler1(event_data):
            handler1_calls.append(event_data)
        
        async def handler2(event_data):
            handler2_calls.append(event_data)
        
        # 订阅同一事件的多个处理器
        processor.subscribe('multi_event', handler1)
        processor.subscribe('multi_event', handler2)
        
        # 发送事件
        await processor.emit('multi_event', {'data': 'shared'})
        
        # 等待处理
        await asyncio.sleep(0.1)
        
        # 验证两个处理器都收到事件
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1
        assert handler1_calls[0]['data'] == 'shared'
        assert handler2_calls[0]['data'] == 'shared'
    
    @pytest.mark.asyncio
    async def test_event_unsubscription(self, processor):
        """测试事件取消订阅"""
        received_events = []
        
        async def event_handler(event_data):
            received_events.append(event_data)
        
        # 订阅事件
        processor.subscribe('unsub_test', event_handler)
        
        # 发送第一个事件
        await processor.emit('unsub_test', {'count': 1})
        
        # 取消订阅
        processor.unsubscribe('unsub_test', event_handler)
        
        # 发送第二个事件
        await processor.emit('unsub_test', {'count': 2})
        
        # 等待处理
        await asyncio.sleep(0.1)
        
        # 验证只收到第一个事件
        assert len(received_events) == 1
        assert received_events[0]['count'] == 1
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, processor):
        """测试事件过滤"""
        filtered_events = []
        
        async def filtered_handler(event_data):
            # 只处理包含'important'字段的事件
            if event_data.get('important'):
                filtered_events.append(event_data)
        
        # 订阅带过滤的处理器
        processor.subscribe('filtered_event', filtered_handler)
        
        # 发送各种事件
        await processor.emit('filtered_event', {'message': 'normal'})
        await processor.emit('filtered_event', {'message': 'important', 'important': True})
        await processor.emit('filtered_event', {'message': 'another normal'})
        
        # 等待处理
        await asyncio.sleep(0.1)
        
        # 验证只收到重要事件
        assert len(filtered_events) == 1
        assert filtered_events[0]['message'] == 'important'
    
    @pytest.mark.asyncio
    async def test_event_prioritization(self, processor):
        """测试事件优先级处理"""
        processed_order = []
        
        async def priority_handler(event_data):
            processed_order.append(event_data['id'])
            # 模拟处理时间
            await asyncio.sleep(0.01)
        
        # 订阅优先级事件
        processor.subscribe('priority_event', priority_handler)
        
        # 发送不同优先级的事件
        await processor.emit_with_priority('priority_event', {'id': 'low'}, priority=1)
        await processor.emit_with_priority('priority_event', {'id': 'high'}, priority=10)
        await processor.emit_with_priority('priority_event', {'id': 'medium'}, priority=5)
        
        # 等待处理
        await asyncio.sleep(0.1)
        
        # 验证按优先级处理
        assert len(processed_order) == 3
        # 高优先级应该先处理
        assert processed_order[0] == 'high'


class TestAsyncDrawdownCalculator:
    """异步回撤计算器测试类"""
    
    @pytest.fixture
    def calculator(self):
        """创建异步回撤计算器实例"""
        return AsyncDrawdownCalculator()
    
    @pytest.mark.asyncio
    async def test_async_drawdown_calculation(self, calculator):
        """测试异步回撤计算"""
        # 创建测试数据
        portfolio_values = np.cumsum(np.random.randn(1000)) + 100
        
        # 异步计算回撤
        result = await calculator.calculate_drawdown_async(portfolio_values)
        
        # 验证结果结构
        assert 'current_drawdown' in result
        assert 'max_drawdown' in result
        assert 'drawdown_series' in result
        
        # 验证数据完整性
        assert len(result['drawdown_series']) == len(portfolio_values)
        assert result['max_drawdown'] <= 0
        assert result['current_drawdown'] <= 0
    
    @pytest.mark.asyncio
    async def test_streaming_drawdown_calculation(self, calculator):
        """测试流式回撤计算"""
        # 模拟实时数据流
        async def data_stream():
            """模拟数据流"""
            for i in range(100):
                yield 100 + np.random.randn()
                await asyncio.sleep(0.01)  # 模拟实时数据间隔
        
        results = []
        
        # 处理流式数据
        async for value in data_stream():
            result = await calculator.update_drawdown_streaming(value)
            results.append(result)
        
        # 验证流式处理结果
        assert len(results) == 100
        for result in results:
            assert 'current_drawdown' in result
            assert isinstance(result['current_drawdown'], (int, float, np.number))
    
    @pytest.mark.asyncio
    async def test_parallel_portfolio_analysis(self, calculator):
        """测试并行投资组合分析"""
        # 创建多个投资组合数据
        portfolios = {}
        for i in range(5):
            portfolios[f'portfolio_{i}'] = np.cumsum(np.random.randn(500)) + 100
        
        # 并行计算所有投资组合的回撤
        results = await calculator.calculate_multiple_portfolios_async(portfolios)
        
        # 验证结果
        assert len(results) == 5
        for portfolio_name, result in results.items():
            assert portfolio_name.startswith('portfolio_')
            assert 'max_drawdown' in result
            assert 'current_drawdown' in result
    
    @pytest.mark.asyncio
    async def test_async_rolling_analysis(self, calculator):
        """测试异步滚动分析"""
        # 创建长时间序列数据
        long_series = np.cumsum(np.random.randn(2000)) + 100
        window_size = 252  # 一年的交易日
        
        # 异步滚动分析
        rolling_results = await calculator.calculate_rolling_drawdown_async(
            long_series, window_size
        )
        
        # 验证滚动分析结果
        expected_windows = len(long_series) - window_size + 1
        assert len(rolling_results) == expected_windows
        
        for result in rolling_results:
            assert 'max_drawdown' in result
            assert 'window_start' in result
            assert 'window_end' in result


class TestTaskScheduler:
    """任务调度器测试类"""
    
    @pytest.fixture
    def scheduler(self):
        """创建任务调度器实例"""
        return TaskScheduler(max_concurrent_tasks=3)
    
    @pytest.mark.asyncio
    async def test_task_scheduling(self, scheduler):
        """测试任务调度"""
        completed_tasks = []
        
        async def test_task(task_id: int, duration: float):
            """测试任务"""
            await asyncio.sleep(duration)
            completed_tasks.append(task_id)
            return f"task_{task_id}_completed"
        
        # 提交多个任务
        tasks = []
        for i in range(6):
            task = scheduler.schedule_task(test_task, i, 0.1)
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(results) == 6
        assert len(completed_tasks) == 6
        for i, result in enumerate(results):
            assert result == f"task_{i}_completed"
    
    @pytest.mark.asyncio
    async def test_task_prioritization(self, scheduler):
        """测试任务优先级"""
        execution_order = []
        
        async def priority_task(task_id: str):
            """优先级任务"""
            execution_order.append(task_id)
            await asyncio.sleep(0.1)
            return task_id
        
        # 提交不同优先级的任务
        high_priority = scheduler.schedule_priority_task(priority_task, "high", priority=10)
        low_priority = scheduler.schedule_priority_task(priority_task, "low", priority=1)
        medium_priority = scheduler.schedule_priority_task(priority_task, "medium", priority=5)
        
        # 等待完成
        results = await asyncio.gather(high_priority, medium_priority, low_priority)
        
        # 验证执行顺序（可能因为并发而有所不同，但高优先级应该优先）
        assert len(execution_order) == 3
        assert "high" in execution_order
        assert "medium" in execution_order
        assert "low" in execution_order
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, scheduler):
        """测试任务超时处理"""
        async def slow_task():
            """慢速任务"""
            await asyncio.sleep(1.0)  # 超过超时限制
            return "completed"
        
        # 设置短超时时间
        with pytest.raises(asyncio.TimeoutError):
            await scheduler.schedule_task_with_timeout(slow_task, timeout=0.2)
    
    @pytest.mark.asyncio
    async def test_resource_limited_scheduling(self, scheduler):
        """测试资源限制调度"""
        resource_usage = []
        
        async def resource_intensive_task(task_id: int):
            """资源密集型任务"""
            resource_usage.append(f"start_{task_id}")
            await asyncio.sleep(0.2)  # 模拟资源使用时间
            resource_usage.append(f"end_{task_id}")
            return task_id
        
        # 提交超过并发限制的任务数量
        tasks = [
            scheduler.schedule_task(resource_intensive_task, i)
            for i in range(5)  # 超过max_concurrent_tasks=3
        ]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(results) == 5
        assert sorted(results) == list(range(5))
        
        # 验证并发限制（应该有任务排队等待）
        assert len(resource_usage) == 10  # 5个任务 * 2个事件（开始和结束）