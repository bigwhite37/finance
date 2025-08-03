"""
性能基准测试模块

该模块提供性能基准测试功能，包括：
- 函数执行时间测试
- 内存使用量测试
- 吞吐量测试
- 性能回归检测
"""

import time
import gc
import psutil
import functools
import statistics
import threading
from typing import Dict, Any, Callable, List, Optional, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    function_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    iterations: int
    throughput: float
    success: bool
    error_message: Optional[str] = None
    
    # 详细统计信息
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    std_time: Optional[float] = None
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class PerformanceProfile:
    """性能分析结果"""
    function_name: str
    total_time: float
    internal_time: float
    calls: int
    per_call_time: float
    cumulative_time: float
    filename: str
    line_number: int


class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, 
                 warmup_iterations: int = 3, 
                 benchmark_iterations: int = 10,
                 collect_detailed_stats: bool = True):
        """
        初始化性能基准测试器
        
        Args:
            warmup_iterations: 预热迭代次数
            benchmark_iterations: 基准测试迭代次数
            collect_detailed_stats: 是否收集详细统计信息
        """
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.collect_detailed_stats = collect_detailed_stats
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 历史基准数据存储
        self.historical_results = {}
        
    def benchmark_function(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """
        对函数进行基准测试
        
        Args:
            func: 要测试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            BenchmarkResult: 测试结果
        """
        function_name = func.__name__
        
        try:
            # 预热
            for _ in range(self.warmup_iterations):
                func(*args, **kwargs)
            
            # 清理垃圾回收
            gc.collect()
            
            # 记录详细的执行时间
            execution_times = []
            memory_measurements = []
            cpu_measurements = []
            
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            # 基准测试
            for i in range(self.benchmark_iterations):
                # 测量CPU使用率（在执行前）
                cpu_before = self.process.cpu_percent()
                
                # 执行函数并测量时间
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                execution_times.append(execution_time)
                
                # 测量内存使用
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
                
                # 测量CPU使用率（在执行后）
                cpu_after = self.process.cpu_percent()
                cpu_measurements.append(cpu_after - cpu_before)
            
            end_memory = self.process.memory_info().rss / 1024 / 1024
            
            # 计算统计指标
            avg_execution_time = statistics.mean(execution_times)
            memory_usage = end_memory - start_memory
            avg_cpu = statistics.mean(cpu_measurements) if cpu_measurements else 0
            throughput = self.benchmark_iterations / sum(execution_times)
            
            # 详细统计
            detailed_stats = {}
            if self.collect_detailed_stats and len(execution_times) > 1:
                detailed_stats = {
                    'min_time': min(execution_times),
                    'max_time': max(execution_times),
                    'std_time': statistics.stdev(execution_times),
                    'percentile_95': np.percentile(execution_times, 95),
                    'percentile_99': np.percentile(execution_times, 99)
                }
            
            result = BenchmarkResult(
                function_name=function_name,
                execution_time=avg_execution_time,
                memory_usage_mb=memory_usage,
                cpu_percent=avg_cpu,
                iterations=self.benchmark_iterations,
                throughput=throughput,
                success=True,
                **detailed_stats
            )
            
            # 存储历史结果
            self._store_historical_result(function_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"基准测试失败 {function_name}: {e}")
            return BenchmarkResult(
                function_name=function_name,
                execution_time=0.0,
                memory_usage_mb=0.0,
                cpu_percent=0.0,
                iterations=0,
                throughput=0.0,
                success=False,
                error_message=str(e)
            )
    
    def compare_functions(self, functions: List[Callable], *args, **kwargs) -> Dict[str, BenchmarkResult]:
        """
        比较多个函数的性能
        
        Args:
            functions: 函数列表
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            Dict[str, BenchmarkResult]: 测试结果字典
        """
        results = {}
        
        for func in functions:
            result = self.benchmark_function(func, *args, **kwargs)
            results[func.__name__] = result
        
        return results
    
    def benchmark_with_scaling(self, 
                              func: Callable, 
                              scale_params: List[tuple],
                              param_names: List[str] = None) -> pd.DataFrame:
        """
        在不同规模参数下进行基准测试
        
        Args:
            func: 要测试的函数
            scale_params: 不同规模的参数列表
            param_names: 参数名称列表
            
        Returns:
            pd.DataFrame: 包含不同规模下性能数据的DataFrame
        """
        results = []
        
        for i, params in enumerate(scale_params):
            if isinstance(params, (list, tuple)):
                result = self.benchmark_function(func, *params)
            else:
                result = self.benchmark_function(func, params)
            
            result_dict = result.to_dict()
            
            # 添加参数信息
            if param_names and len(param_names) == len(params):
                for j, param_name in enumerate(param_names):
                    result_dict[param_name] = params[j]
            else:
                result_dict['scale_index'] = i
                if isinstance(params, (list, tuple)):
                    for j, param in enumerate(params):
                        result_dict[f'param_{j}'] = param
                else:
                    result_dict['param'] = params
            
            results.append(result_dict)
        
        return pd.DataFrame(results)
    
    def detect_performance_regression(self, 
                                    function_name: str, 
                                    current_result: BenchmarkResult,
                                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        检测性能回归
        
        Args:
            function_name: 函数名
            current_result: 当前测试结果
            threshold: 回归阈值（比例）
            
        Returns:
            Dict: 回归检测结果
        """
        if function_name not in self.historical_results:
            return {
                'regression_detected': False,
                'reason': '无历史数据进行对比'
            }
        
        historical = self.historical_results[function_name]
        
        # 计算性能变化
        time_change = (current_result.execution_time - historical['execution_time']) / historical['execution_time']
        memory_change = (current_result.memory_usage_mb - historical['memory_usage_mb']) / max(historical['memory_usage_mb'], 0.1)
        
        regression_detected = False
        issues = []
        
        if time_change > threshold:
            regression_detected = True
            issues.append(f"执行时间增加 {time_change:.1%}")
        
        if memory_change > threshold:
            regression_detected = True
            issues.append(f"内存使用增加 {memory_change:.1%}")
        
        return {
            'regression_detected': regression_detected,
            'issues': issues,
            'time_change': time_change,
            'memory_change': memory_change,
            'current_time': current_result.execution_time,
            'baseline_time': historical['execution_time']
        }
    
    def _store_historical_result(self, function_name: str, result: BenchmarkResult):
        """存储历史基准结果"""
        self.historical_results[function_name] = {
            'execution_time': result.execution_time,
            'memory_usage_mb': result.memory_usage_mb,
            'cpu_percent': result.cpu_percent,
            'throughput': result.throughput,
            'timestamp': time.time()
        }
    
    def save_baseline(self, filepath: Union[str, Path]):
        """保存基准数据到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.historical_results, f, indent=2)
        
        logger.info(f"基准数据已保存到 {filepath}")
    
    def load_baseline(self, filepath: Union[str, Path]):
        """从文件加载基准数据"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"基准数据文件不存在: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            self.historical_results = json.load(f)
        
        logger.info(f"基准数据已从 {filepath} 加载")
    
    def generate_performance_report(self, results: Dict[str, BenchmarkResult]) -> str:
        """生成性能报告"""
        report = "# 性能基准测试报告\n\n"
        report += f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"测试迭代次数: {self.benchmark_iterations}\n"
        report += f"预热迭代次数: {self.warmup_iterations}\n\n"
        
        # 创建性能对比表
        report += "## 性能对比\n\n"
        report += "| 函数名 | 执行时间(ms) | 内存使用(MB) | CPU(%) | 吞吐量(/s) | 状态 |\n"
        report += "|--------|-------------|-------------|--------|-----------|------|\n"
        
        for func_name, result in results.items():
            status = "✅" if result.success else "❌"
            report += f"| {func_name} | {result.execution_time*1000:.3f} | {result.memory_usage_mb:.2f} | {result.cpu_percent:.1f} | {result.throughput:.1f} | {status} |\n"
        
        # 详细统计信息
        if self.collect_detailed_stats:
            report += "\n## 详细统计信息\n\n"
            for func_name, result in results.items():
                if result.success and result.std_time is not None:
                    report += f"### {func_name}\n"
                    report += f"- 最小时间: {result.min_time*1000:.3f}ms\n"
                    report += f"- 最大时间: {result.max_time*1000:.3f}ms\n"
                    report += f"- 标准差: {result.std_time*1000:.3f}ms\n"
                    report += f"- 95%分位数: {result.percentile_95*1000:.3f}ms\n"
                    report += f"- 99%分位数: {result.percentile_99*1000:.3f}ms\n\n"
        
        return report
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import platform
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        }