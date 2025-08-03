"""
性能基准测试和瓶颈分析测试

该模块测试性能分析功能，包括：
- 函数性能基准测试
- 系统瓶颈分析
- 性能回归检测
- 性能报告生成
"""

import pytest
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Callable
from unittest.mock import Mock, patch
import psutil

from src.rl_trading_system.performance.benchmark import PerformanceBenchmark, BenchmarkResult
from src.rl_trading_system.performance.bottleneck_analyzer import BottleneckAnalyzer, BottleneckAnalysis
from src.rl_trading_system.performance.performance_monitor import PerformanceMonitor


class TestPerformanceBenchmark:
    """性能基准测试类"""
    
    @pytest.fixture
    def benchmark(self):
        """创建性能基准测试器实例"""
        return PerformanceBenchmark(warmup_iterations=2, benchmark_iterations=5)
    
    def test_function_benchmarking(self, benchmark):
        """测试函数基准测试"""
        def test_function(n: int) -> int:
            """测试函数：计算平方和"""
            return sum(i ** 2 for i in range(n))
        
        # 基准测试
        result = benchmark.benchmark_function(test_function, 1000)
        
        # 验证结果结构
        assert isinstance(result, BenchmarkResult)
        assert result.function_name == 'test_function'
        assert result.success is True
        assert result.execution_time > 0
        assert result.iterations == 5
        assert result.throughput > 0
        
        # 验证内存和CPU指标
        assert isinstance(result.memory_usage_mb, (int, float))
        assert isinstance(result.cpu_percent, (int, float))
    
    def test_numpy_function_benchmarking(self, benchmark):
        """测试NumPy函数基准测试"""
        def numpy_function(size: int) -> np.ndarray:
            """NumPy测试函数"""
            arr = np.random.randn(size, size)
            return np.dot(arr, arr.T)
        
        result = benchmark.benchmark_function(numpy_function, 100)
        
        assert result.success is True
        assert result.execution_time > 0
        assert result.function_name == 'numpy_function'
    
    def test_error_handling_in_benchmark(self, benchmark):
        """测试基准测试中的错误处理"""
        def failing_function():
            """会失败的函数"""
            raise ValueError("测试错误")
        
        result = benchmark.benchmark_function(failing_function)
        
        assert result.success is False
        assert result.error_message == "测试错误"
        assert result.execution_time == 0.0
        assert result.throughput == 0.0
    
    def test_memory_intensive_function_benchmark(self, benchmark):
        """测试内存密集型函数基准测试"""
        def memory_intensive_function():
            """内存密集型函数"""
            large_arrays = []
            for i in range(10):
                large_arrays.append(np.random.randn(1000, 1000))
            return len(large_arrays)
        
        result = benchmark.benchmark_function(memory_intensive_function)
        
        assert result.success is True
        assert result.memory_usage_mb > 0  # 应该有明显的内存使用
    
    def test_function_comparison(self, benchmark):
        """测试函数性能对比"""
        def python_sum(n: int) -> int:
            """Python原生求和"""
            return sum(range(n))
        
        def numpy_sum(n: int) -> int:
            """NumPy求和"""
            return np.sum(np.arange(n))
        
        functions = [python_sum, numpy_sum]
        results = benchmark.compare_functions(functions, 10000)
        
        # 验证对比结果
        assert len(results) == 2
        assert 'python_sum' in results
        assert 'numpy_sum' in results
        
        # 验证所有结果都成功
        for func_name, result in results.items():
            assert result.success is True
            assert result.execution_time > 0
    
    def test_benchmark_with_different_parameters(self, benchmark):
        """测试不同参数下的基准测试"""
        def parametric_function(size: int, multiplier: float = 1.0) -> float:
            """参数化函数"""
            arr = np.random.randn(size) * multiplier
            return np.mean(arr)
        
        # 测试不同参数
        result1 = benchmark.benchmark_function(parametric_function, 1000)
        result2 = benchmark.benchmark_function(parametric_function, 1000, multiplier=2.0)
        
        assert result1.success is True
        assert result2.success is True
        
        # 两次调用应该有相似的性能特征
        assert abs(result1.execution_time - result2.execution_time) < result1.execution_time
    
    def test_benchmark_reproducibility(self, benchmark):
        """测试基准测试的重现性"""
        def deterministic_function(n: int) -> int:
            """确定性函数"""
            np.random.seed(42)
            arr = np.random.randn(n)
            return int(np.sum(arr))
        
        # 多次运行基准测试
        results = []
        for _ in range(3):
            result = benchmark.benchmark_function(deterministic_function, 1000)
            results.append(result)
        
        # 验证结果的一致性（允许一些变化）
        execution_times = [r.execution_time for r in results]
        avg_time = np.mean(execution_times)
        
        for exec_time in execution_times:
            # 执行时间变化应该在合理范围内（50%以内）
            assert abs(exec_time - avg_time) / avg_time < 0.5


class TestBottleneckAnalyzer:
    """瓶颈分析器测试类"""
    
    @pytest.fixture
    def analyzer(self):
        """创建瓶颈分析器实例"""
        return BottleneckAnalyzer()
    
    def test_system_bottleneck_analysis(self, analyzer):
        """测试系统瓶颈分析"""
        analysis = analyzer.analyze_system_bottlenecks()
        
        # 验证分析结果结构
        assert isinstance(analysis, BottleneckAnalysis)
        assert hasattr(analysis, 'cpu_usage')
        assert hasattr(analysis, 'memory_usage')
        assert hasattr(analysis, 'disk_io')
        assert hasattr(analysis, 'network_io')
        assert hasattr(analysis, 'recommendations')
        
        # 验证数值范围
        assert 0 <= analysis.cpu_usage <= 100
        assert 0 <= analysis.memory_usage <= 100
        assert isinstance(analysis.recommendations, list)
    
    def test_function_profiling(self, analyzer):
        """测试函数性能分析"""
        def complex_function():
            """复杂函数用于性能分析"""
            # 计算密集操作
            result = 0
            for i in range(10000):
                result += i ** 2
            
            # 内存操作
            large_list = list(range(1000))
            large_array = np.array(large_list)
            
            # 数学运算
            matrix = np.random.randn(100, 100)
            eigenvals = np.linalg.eigvals(matrix)
            
            return result + np.sum(eigenvals)
        
        profile_result = analyzer.profile_function(complex_function)
        
        # 验证分析结果
        assert 'result' in profile_result
        assert 'profile_stats' in profile_result
        assert isinstance(profile_result['profile_stats'], str)
        assert len(profile_result['profile_stats']) > 0
    
    def test_io_bottleneck_detection(self, analyzer):
        """测试I/O瓶颈检测"""
        def io_intensive_function():
            """I/O密集型函数"""
            # 模拟文件I/O
            data = np.random.randn(10000)
            df = pd.DataFrame(data)
            
            # 模拟数据处理
            processed = df.apply(lambda x: x ** 2)
            return processed.sum().sum()
        
        # 分析I/O操作
        with patch('psutil.disk_io_counters') as mock_disk_io:
            mock_disk_io.return_value = Mock(read_bytes=1000000, write_bytes=500000)
            
            analysis = analyzer.analyze_system_bottlenecks()
            
            # 验证I/O指标
            assert 'read_bytes' in analysis.disk_io
            assert 'write_bytes' in analysis.disk_io
            assert analysis.disk_io['read_bytes'] > 0
            assert analysis.disk_io['write_bytes'] > 0
    
    def test_memory_bottleneck_detection(self, analyzer):
        """测试内存瓶颈检测"""
        # 模拟高内存使用
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(percent=85)  # 85%内存使用率
            
            analysis = analyzer.analyze_system_bottlenecks()
            
            # 验证内存瓶颈检测
            assert analysis.memory_usage == 85
            
            # 验证推荐建议
            memory_recommendations = [r for r in analysis.recommendations 
                                    if '内存' in r]
            assert len(memory_recommendations) > 0
    
    def test_cpu_bottleneck_detection(self, analyzer):
        """测试CPU瓶颈检测"""
        # 模拟高CPU使用
        with patch('psutil.cpu_percent') as mock_cpu:
            mock_cpu.return_value = 90  # 90% CPU使用率
            
            analysis = analyzer.analyze_system_bottlenecks()
            
            # 验证CPU瓶颈检测
            assert analysis.cpu_usage == 90
            
            # 验证推荐建议
            cpu_recommendations = [r for r in analysis.recommendations 
                                 if 'CPU' in r]
            assert len(cpu_recommendations) > 0
    
    def test_concurrent_analysis(self, analyzer):
        """测试并发分析"""
        def concurrent_function():
            """并发测试函数"""
            import threading
            import time
            
            results = []
            
            def worker():
                for i in range(1000):
                    results.append(i ** 2)
                time.sleep(0.01)
            
            threads = []
            for _ in range(4):
                t = threading.Thread(target=worker)
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            return len(results)
        
        profile_result = analyzer.profile_function(concurrent_function)
        
        # 验证并发函数分析
        assert profile_result['result'] > 0
        assert 'threading' in profile_result['profile_stats'] or 'thread' in profile_result['profile_stats'].lower()


class TestPerformanceMonitor:
    """性能监控器测试类"""
    
    @pytest.fixture
    def monitor(self):
        """创建性能监控器实例"""
        return PerformanceMonitor(sampling_interval=0.1)
    
    def test_real_time_monitoring(self, monitor):
        """测试实时性能监控"""
        # 启动监控
        monitor.start_monitoring()
        
        # 执行一些操作
        def monitored_operation():
            time.sleep(0.2)
            arr = np.random.randn(1000, 1000)
            return np.sum(arr)
        
        result = monitored_operation()
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 获取监控数据
        metrics = monitor.get_metrics()
        
        # 验证监控结果
        assert len(metrics) > 0
        assert 'timestamp' in metrics[0]
        assert 'cpu_percent' in metrics[0]
        assert 'memory_mb' in metrics[0]
        
        # 验证数据合理性
        for metric in metrics:
            assert 0 <= metric['cpu_percent'] <= 100
            assert metric['memory_mb'] > 0
    
    def test_performance_alerts(self, monitor):
        """测试性能告警"""
        # 设置告警阈值
        monitor.set_alert_thresholds(cpu_threshold=80, memory_threshold=80)
        
        # 模拟高CPU使用的函数
        def cpu_intensive_function():
            # 计算密集型操作
            for i in range(100000):
                _ = i ** 3
        
        # 启动监控
        monitor.start_monitoring()
        
        # 执行CPU密集型操作
        cpu_intensive_function()
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 检查告警
        alerts = monitor.get_alerts()
        
        # 验证告警机制
        assert isinstance(alerts, list)
        # 可能有CPU告警，取决于系统负载
    
    def test_performance_trend_analysis(self, monitor):
        """测试性能趋势分析"""
        # 收集一段时间的性能数据
        monitor.start_monitoring()
        
        # 模拟不同负载的操作
        operations = [
            lambda: time.sleep(0.1),  # 轻负载
            lambda: np.random.randn(500, 500).sum(),  # 中等负载
            lambda: np.linalg.inv(np.random.randn(200, 200) + np.eye(200)),  # 高负载
        ]
        
        for op in operations:
            op()
            time.sleep(0.1)
        
        monitor.stop_monitoring()
        
        # 分析性能趋势
        trend_analysis = monitor.analyze_performance_trends()
        
        # 验证趋势分析结果
        assert 'avg_cpu' in trend_analysis
        assert 'avg_memory' in trend_analysis
        assert 'cpu_trend' in trend_analysis
        assert 'memory_trend' in trend_analysis
        
        # 验证数值合理性
        assert 0 <= trend_analysis['avg_cpu'] <= 100
        assert trend_analysis['avg_memory'] > 0
    
    def test_performance_comparison(self, monitor):
        """测试性能对比分析"""
        # 测试两个不同的函数
        def function_a():
            """函数A：使用Python原生操作"""
            result = []
            for i in range(10000):
                result.append(i ** 2)
            return sum(result)
        
        def function_b():
            """函数B：使用NumPy操作"""
            arr = np.arange(10000)
            return np.sum(arr ** 2)
        
        # 比较两个函数的性能
        comparison = monitor.compare_functions([function_a, function_b])
        
        # 验证对比结果
        assert len(comparison) == 2
        assert 'function_a' in comparison
        assert 'function_b' in comparison
        
        # 验证每个函数的指标
        for func_name, metrics in comparison.items():
            assert 'execution_time' in metrics
            assert 'cpu_usage' in metrics
            assert 'memory_usage' in metrics
            assert metrics['execution_time'] > 0
    
    def test_resource_utilization_analysis(self, monitor):
        """测试资源利用率分析"""
        # 启动资源监控
        monitor.start_resource_monitoring()
        
        # 执行各种类型的操作
        # CPU密集型
        np.linalg.eigvals(np.random.randn(300, 300))
        
        # 内存密集型
        large_data = [np.random.randn(1000) for _ in range(100)]
        
        # I/O模拟（创建DataFrame）
        df = pd.DataFrame(np.random.randn(10000, 10))
        processed_df = df.apply(lambda x: x.rolling(10).mean())
        
        # 停止监控
        utilization = monitor.stop_resource_monitoring()
        
        # 验证资源利用率分析
        assert 'cpu_utilization' in utilization
        assert 'memory_utilization' in utilization
        assert 'peak_cpu' in utilization
        assert 'peak_memory' in utilization
        
        # 验证数值合理性
        assert 0 <= utilization['cpu_utilization'] <= 100
        assert utilization['memory_utilization'] > 0
        assert utilization['peak_cpu'] >= utilization['cpu_utilization']
        assert utilization['peak_memory'] >= utilization['memory_utilization']