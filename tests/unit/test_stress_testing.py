"""
实时性能压力测试

该模块提供系统压力测试功能，包括：
- 负载压力测试
- 并发压力测试
- 内存压力测试
- 实时性能监控压力测试
"""

import pytest
import asyncio
import time
import numpy as np
import pandas as pd
import psutil
import threading
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
from unittest.mock import Mock, patch

from src.rl_trading_system.performance.stress_tester import StressTester, LoadGenerator
from src.rl_trading_system.performance.performance_monitor import PerformanceMonitor
from src.rl_trading_system.performance.vectorized_calculator import VectorizedCalculator
from src.rl_trading_system.performance.async_optimizer import AsyncOptimizer


class TestStressTester:
    """压力测试器测试类"""
    
    @pytest.fixture
    def stress_tester(self):
        """创建压力测试器实例"""
        return StressTester(max_workers=4, max_concurrent_tests=10)
    
    def test_cpu_stress_test(self, stress_tester):
        """测试CPU压力测试"""
        def cpu_intensive_task(iterations: int) -> float:
            """CPU密集型任务"""
            result = 0.0
            for i in range(iterations):
                result += i ** 2 * np.sin(i) * np.cos(i)
            return result
        
        # 执行CPU压力测试
        stress_result = stress_tester.run_cpu_stress_test(
            task_func=cpu_intensive_task,
            task_args=(100000,),
            duration_seconds=5,
            concurrent_tasks=4
        )
        
        # 验证压力测试结果
        assert stress_result['test_type'] == 'cpu_stress'
        assert stress_result['duration'] >= 4  # 允许一些误差
        assert stress_result['total_tasks'] > 0
        assert stress_result['completed_tasks'] > 0
        assert 'peak_cpu_usage' in stress_result
        assert 'avg_cpu_usage' in stress_result
        assert 'task_throughput' in stress_result
    
    def test_memory_stress_test(self, stress_tester):
        """测试内存压力测试"""
        def memory_intensive_task(array_size: int) -> int:
            """内存密集型任务"""
            # 创建大型数组
            large_array = np.random.randn(array_size, array_size)
            # 执行一些计算以使用内存
            processed = np.dot(large_array, large_array.T)
            return processed.shape[0]
        
        # 执行内存压力测试
        stress_result = stress_tester.run_memory_stress_test(
            task_func=memory_intensive_task,
            task_args=(500,),  # 500x500 数组
            duration_seconds=3,
            concurrent_tasks=2
        )
        
        # 验证压力测试结果
        assert stress_result['test_type'] == 'memory_stress'
        assert stress_result['duration'] >= 2
        assert stress_result['total_tasks'] > 0
        assert 'peak_memory_usage_mb' in stress_result
        assert 'memory_growth_mb' in stress_result
        assert stress_result['peak_memory_usage_mb'] > 0
    
    def test_io_stress_test(self, stress_tester):
        """测试I/O压力测试"""
        def io_intensive_task(data_size: int) -> int:
            """I/O密集型任务"""
            # 创建DataFrame进行I/O操作
            data = np.random.randn(data_size, 10)
            df = pd.DataFrame(data)
            
            # 模拟I/O操作
            processed = df.rolling(window=5).mean()
            aggregated = processed.groupby(processed.index % 10).sum()
            
            return len(aggregated)
        
        # 执行I/O压力测试
        stress_result = stress_tester.run_io_stress_test(
            task_func=io_intensive_task,
            task_args=(1000,),
            duration_seconds=3,
            concurrent_tasks=3
        )
        
        # 验证压力测试结果
        assert stress_result['test_type'] == 'io_stress'
        assert stress_result['duration'] >= 2
        assert stress_result['total_tasks'] > 0
        assert 'avg_task_duration' in stress_result
    
    def test_mixed_workload_stress_test(self, stress_tester):
        """测试混合工作负载压力测试"""
        def mixed_task(task_type: str, size: int) -> Dict[str, Any]:
            """混合任务"""
            if task_type == 'cpu':
                result = sum(i ** 2 for i in range(size))
                return {'type': 'cpu', 'result': result}
            elif task_type == 'memory':
                arr = np.random.randn(size, size // 10)
                return {'type': 'memory', 'size': arr.nbytes}
            elif task_type == 'io':
                df = pd.DataFrame(np.random.randn(size, 5))
                processed = df.describe()
                return {'type': 'io', 'rows': len(processed)}
            else:
                return {'type': 'unknown', 'result': None}
        
        # 创建混合工作负载
        workloads = [
            (mixed_task, ('cpu', 10000)),
            (mixed_task, ('memory', 100)),
            (mixed_task, ('io', 500)),
        ]
        
        # 执行混合压力测试
        stress_result = stress_tester.run_mixed_stress_test(
            workloads=workloads,
            duration_seconds=4,
            concurrent_tasks=3
        )
        
        # 验证结果
        assert stress_result['test_type'] == 'mixed_stress'
        assert stress_result['total_tasks'] > 0
        assert 'workload_distribution' in stress_result
        assert len(stress_result['workload_distribution']) == 3
    
    def test_scalability_stress_test(self, stress_tester):
        """测试可扩展性压力测试"""
        def scalable_task(load_factor: int) -> Dict[str, Any]:
            """可扩展任务"""
            start_time = time.time()
            
            # 根据负载因子调整计算量
            result = np.sum(np.random.randn(load_factor * 100))
            
            end_time = time.time()
            return {
                'load_factor': load_factor,
                'execution_time': end_time - start_time,
                'result': result
            }
        
        # 不同负载级别
        load_factors = [1, 2, 4, 8]
        
        # 执行可扩展性测试
        stress_result = stress_tester.run_scalability_test(
            task_func=scalable_task,
            load_factors=load_factors,
            iterations_per_load=3
        )
        
        # 验证结果
        assert stress_result['test_type'] == 'scalability_stress'
        assert len(stress_result['load_results']) == len(load_factors)
        
        # 验证性能随负载的变化
        for load_result in stress_result['load_results']:
            assert 'load_factor' in load_result
            assert 'avg_execution_time' in load_result
            assert 'throughput' in load_result
    
    def test_endurance_stress_test(self, stress_tester):
        """测试耐久性压力测试"""
        def endurance_task() -> Dict[str, Any]:
            """耐久性任务"""
            # 模拟常规操作
            data = np.random.randn(100, 100)
            result = np.linalg.det(data)
            return {'determinant': result, 'timestamp': time.time()}
        
        # 执行耐久性测试（短时间版本）
        stress_result = stress_tester.run_endurance_test(
            task_func=endurance_task,
            duration_seconds=3,
            task_interval=0.1
        )
        
        # 验证结果
        assert stress_result['test_type'] == 'endurance_stress'
        assert stress_result['total_tasks'] > 10  # 应该执行多个任务
        assert 'memory_stability' in stress_result
        assert 'performance_degradation' in stress_result


class TestLoadGenerator:
    """负载生成器测试类"""
    
    @pytest.fixture
    def load_generator(self):
        """创建负载生成器实例"""
        return LoadGenerator()
    
    def test_constant_load_generation(self, load_generator):
        """测试恒定负载生成"""
        task_results = []
        
        def simple_task(task_id: int) -> int:
            """简单任务"""
            task_results.append(task_id)
            time.sleep(0.01)
            return task_id
        
        # 生成恒定负载
        load_result = load_generator.generate_constant_load(
            task_func=simple_task,
            tasks_per_second=50,
            duration_seconds=2
        )
        
        # 验证负载生成结果
        assert load_result['load_type'] == 'constant'
        assert load_result['total_tasks'] > 80  # 约100个任务，允许一些误差
        assert load_result['duration'] >= 1.8
        assert len(task_results) == load_result['total_tasks']
    
    def test_burst_load_generation(self, load_generator):
        """测试突发负载生成"""
        task_results = []
        
        def burst_task(burst_id: int, task_id: int) -> tuple:
            """突发任务"""
            result = (burst_id, task_id)
            task_results.append(result)
            return result
        
        # 生成突发负载
        load_result = load_generator.generate_burst_load(
            task_func=burst_task,
            tasks_per_burst=10,
            burst_interval=0.5,
            num_bursts=3
        )
        
        # 验证突发负载结果
        assert load_result['load_type'] == 'burst'
        assert load_result['num_bursts'] == 3
        assert load_result['tasks_per_burst'] == 10
        assert load_result['total_tasks'] == 30
        assert len(task_results) == 30
    
    def test_ramp_load_generation(self, load_generator):
        """测试斜坡负载生成"""
        task_execution_times = []
        
        def ramp_task(task_id: int) -> int:
            """斜坡任务"""
            task_execution_times.append(time.time())
            time.sleep(0.01)
            return task_id
        
        # 生成斜坡负载
        load_result = load_generator.generate_ramp_load(
            task_func=ramp_task,
            initial_rate=10,
            final_rate=50,
            duration_seconds=2
        )
        
        # 验证斜坡负载结果
        assert load_result['load_type'] == 'ramp'
        assert load_result['initial_rate'] == 10
        assert load_result['final_rate'] == 50
        assert load_result['total_tasks'] > 20  # 应该有递增的任务数
        
        # 验证任务执行时间间隔递减（频率递增）
        if len(task_execution_times) > 5:
            early_intervals = np.diff(task_execution_times[:5])
            late_intervals = np.diff(task_execution_times[-5:])
            avg_early = np.mean(early_intervals)
            avg_late = np.mean(late_intervals)
            assert avg_early > avg_late  # 后期间隔应该更小（频率更高）
    
    def test_random_load_generation(self, load_generator):
        """测试随机负载生成"""
        task_results = []
        
        def random_task(task_id: int) -> int:
            """随机任务"""
            task_results.append(task_id)
            # 随机执行时间
            time.sleep(np.random.uniform(0.001, 0.02))
            return task_id
        
        # 生成随机负载
        load_result = load_generator.generate_random_load(
            task_func=random_task,
            avg_tasks_per_second=30,
            duration_seconds=2,
            randomness_factor=0.5
        )
        
        # 验证随机负载结果
        assert load_result['load_type'] == 'random'
        assert load_result['avg_tasks_per_second'] == 30
        assert load_result['total_tasks'] > 30  # 应该接近60个任务
        assert len(task_results) == load_result['total_tasks']


class TestRealTimePerformanceStress:
    """实时性能压力测试类"""
    
    @pytest.fixture
    def performance_monitor(self):
        """创建性能监控器"""
        return PerformanceMonitor(sampling_interval=0.1)
    
    @pytest.fixture
    def vectorized_calculator(self):
        """创建向量化计算器"""
        return VectorizedCalculator(enable_parallel=True)
    
    @pytest.fixture
    def async_optimizer(self):
        """创建异步优化器"""
        return AsyncOptimizer(max_workers=4, max_concurrent_tasks=8)
    
    def test_real_time_drawdown_calculation_stress(self, vectorized_calculator):
        """测试实时回撤计算压力"""
        # 生成大量测试数据
        num_portfolios = 20
        portfolio_length = 5000
        
        portfolios = {}
        for i in range(num_portfolios):
            # 生成随机走势的投资组合
            returns = np.random.randn(portfolio_length) * 0.02
            portfolio_values = np.cumprod(1 + returns) * 100
            portfolios[f'portfolio_{i}'] = portfolio_values
        
        # 执行压力测试
        start_time = time.time()
        results = {}
        
        for name, values in portfolios.items():
            result = vectorized_calculator.calculate_vectorized_drawdown(values)
            results[name] = result
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证性能
        assert len(results) == num_portfolios
        assert total_time < 10.0  # 应该在10秒内完成
        
        # 验证结果正确性
        for result in results.values():
            assert 'max_drawdown' in result
            assert 'current_drawdown' in result
            assert result['max_drawdown'] <= 0
        
        # 计算吞吐量
        throughput = (num_portfolios * portfolio_length) / total_time
        print(f"回撤计算吞吐量: {throughput:.0f} 数据点/秒")
        
        assert throughput > 50000  # 至少5万数据点/秒
    
    @pytest.mark.asyncio
    async def test_async_performance_stress(self, async_optimizer):
        """测试异步性能压力"""
        def compute_intensive_task(size: int) -> float:
            """计算密集型任务"""
            matrix = np.random.randn(size, size)
            eigenvalues = np.linalg.eigvals(matrix)
            return np.sum(np.real(eigenvalues))
        
        # 创建大量异步任务
        num_tasks = 50
        matrix_size = 100
        
        start_time = time.time()
        
        # 提交所有任务
        tasks = []
        for i in range(num_tasks):
            task = await async_optimizer.submit_task(compute_intensive_task, matrix_size)
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证结果
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= num_tasks * 0.9  # 至少90%成功
        
        # 验证性能
        assert total_time < 30.0  # 应该在30秒内完成
        throughput = num_tasks / total_time
        print(f"异步任务吞吐量: {throughput:.1f} 任务/秒")
    
    def test_concurrent_monitoring_stress(self, performance_monitor):
        """测试并发监控压力"""
        def monitored_operation():
            """被监控的操作"""
            # 模拟各种操作
            cpu_work = sum(i ** 2 for i in range(10000))
            memory_work = np.random.randn(1000, 100)
            io_work = pd.DataFrame(memory_work).describe()
            return cpu_work + memory_work.sum() + io_work.sum().sum()
        
        # 启动监控
        performance_monitor.start_monitoring()
        
        # 并发执行监控操作
        num_threads = 8
        operations_per_thread = 5
        
        def worker_thread():
            """工作线程"""
            results = []
            for _ in range(operations_per_thread):
                result = monitored_operation()
                results.append(result)
                time.sleep(0.1)  # 间隔执行
            return results
        
        # 启动多个工作线程
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread) for _ in range(num_threads)]
            thread_results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        
        # 停止监控
        performance_monitor.stop_monitoring()
        
        # 获取监控数据
        metrics = performance_monitor.get_metrics()
        
        # 验证监控效果
        assert len(metrics) > 0
        assert len(thread_results) == num_threads
        
        # 验证监控数据完整性
        total_operations = num_threads * operations_per_thread
        execution_time = end_time - start_time
        
        print(f"并发操作数: {total_operations}")
        print(f"执行时间: {execution_time:.2f}秒")
        print(f"监控样本数: {len(metrics)}")
        
        assert execution_time < 10.0  # 应该及时完成
        assert len(metrics) >= execution_time * 5  # 至少每200ms一个样本
    
    def test_memory_pressure_stress(self):
        """测试内存压力测试"""
        def memory_pressure_task():
            """内存压力任务"""
            # 分配大量内存
            large_arrays = []
            for i in range(10):
                arr = np.random.randn(1000, 1000)  # 约8MB每个
                large_arrays.append(arr)
            
            # 执行一些计算
            total = 0
            for arr in large_arrays:
                total += np.sum(arr ** 2)
            
            return total, len(large_arrays)
        
        # 监控内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行内存压力任务
        start_time = time.time()
        
        # 并发执行多个内存压力任务
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(memory_pressure_task) for _ in range(8)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 验证结果
        assert len(results) == 8
        execution_time = end_time - start_time
        memory_usage = final_memory - initial_memory
        
        print(f"内存压力测试执行时间: {execution_time:.2f}秒")
        print(f"峰值内存使用: {memory_usage:.1f}MB")
        
        # 验证性能要求
        assert execution_time < 15.0  # 应该在15秒内完成
        
        # 验证内存管理（在GC后内存应该有所释放）
        time.sleep(1)  # 等待GC完成
        gc.collect()
        post_gc_memory = process.memory_info().rss / 1024 / 1024
        memory_recovered = final_memory - post_gc_memory
        
        print(f"GC后回收内存: {memory_recovered:.1f}MB")
    
    def test_system_resource_limits_stress(self):
        """测试系统资源限制压力测试"""
        def resource_intensive_task(task_id: int) -> Dict[str, Any]:
            """资源密集型任务"""
            start_time = time.time()
            
            # CPU密集型部分
            cpu_result = sum(i ** 2 for i in range(50000))
            
            # 内存密集型部分
            memory_data = np.random.randn(500, 500)
            memory_result = np.linalg.det(memory_data)
            
            # I/O密集型部分
            df = pd.DataFrame(memory_data)
            io_result = df.rolling(10).mean().sum().sum()
            
            end_time = time.time()
            
            return {
                'task_id': task_id,
                'execution_time': end_time - start_time,
                'cpu_result': cpu_result,
                'memory_result': memory_result,
                'io_result': io_result
            }
        
        # 获取系统资源信息
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"系统资源: {cpu_count} CPU核心, {memory_gb:.1f}GB 内存")
        
        # 执行资源密集型压力测试
        num_tasks = cpu_count * 4  # 超过CPU核心数的任务
        
        start_time = time.time()
        system_start = psutil.cpu_percent(), psutil.virtual_memory().percent
        
        with ThreadPoolExecutor(max_workers=cpu_count * 2) as executor:
            futures = [
                executor.submit(resource_intensive_task, i) 
                for i in range(num_tasks)
            ]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        system_end = psutil.cpu_percent(), psutil.virtual_memory().percent
        
        # 分析结果
        total_time = end_time - start_time
        avg_task_time = np.mean([r['execution_time'] for r in results])
        task_throughput = num_tasks / total_time
        
        print(f"压力测试结果:")
        print(f"  总执行时间: {total_time:.2f}秒")
        print(f"  平均任务时间: {avg_task_time:.3f}秒")
        print(f"  任务吞吐量: {task_throughput:.1f} 任务/秒")
        print(f"  系统CPU使用: {system_start[0]:.1f}% -> {system_end[0]:.1f}%")
        print(f"  系统内存使用: {system_start[1]:.1f}% -> {system_end[1]:.1f}%")
        
        # 验证压力测试效果
        assert len(results) == num_tasks
        assert total_time < 60.0  # 应该在1分钟内完成
        assert task_throughput > 1.0  # 至少每秒1个任务