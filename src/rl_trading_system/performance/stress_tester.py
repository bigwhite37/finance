"""
压力测试器

该模块提供系统压力测试功能，包括：
- CPU压力测试
- 内存压力测试
- I/O压力测试
- 混合工作负载测试
- 可扩展性测试
- 耐久性测试
"""

import time
import threading
import psutil
import numpy as np
import statistics
from typing import Dict, List, Any, Callable, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import gc

logger = logging.getLogger(__name__)


class LoadGenerator:
    """负载生成器"""
    
    def __init__(self):
        """初始化负载生成器"""
        self.active_tasks = []
        logger.info("负载生成器初始化完成")
    
    def generate_constant_load(self, 
                             task_func: Callable,
                             tasks_per_second: int,
                             duration_seconds: float) -> Dict[str, Any]:
        """
        生成恒定负载
        
        Args:
            task_func: 任务函数
            tasks_per_second: 每秒任务数
            duration_seconds: 持续时间
            
        Returns:
            Dict: 负载生成结果
        """
        start_time = time.time()
        task_interval = 1.0 / tasks_per_second
        completed_tasks = 0
        
        with ThreadPoolExecutor(max_workers=min(tasks_per_second, 20)) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                # 提交新任务
                if len(futures) < tasks_per_second:
                    future = executor.submit(task_func, completed_tasks)
                    futures.append(future)
                
                # 收集完成的任务
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    futures.remove(future)
                    completed_tasks += 1
                
                time.sleep(min(task_interval, 0.1))
            
            # 等待剩余任务完成
            for future in futures:
                try:
                    future.result(timeout=5)
                    completed_tasks += 1
                except Exception:
                    pass
        
        total_duration = time.time() - start_time
        
        return {
            'load_type': 'constant',
            'total_tasks': completed_tasks,
            'duration': total_duration,
            'target_rate': tasks_per_second,
            'actual_rate': completed_tasks / total_duration if total_duration > 0 else 0
        }
    
    def generate_burst_load(self,
                          task_func: Callable,
                          tasks_per_burst: int,
                          burst_interval: float,
                          num_bursts: int) -> Dict[str, Any]:
        """
        生成突发负载
        
        Args:
            task_func: 任务函数
            tasks_per_burst: 每个突发的任务数
            burst_interval: 突发间隔
            num_bursts: 突发次数
            
        Returns:
            Dict: 负载生成结果
        """
        start_time = time.time()
        total_completed = 0
        
        with ThreadPoolExecutor(max_workers=tasks_per_burst * 2) as executor:
            for burst_id in range(num_bursts):
                # 提交一批任务
                futures = []
                for task_id in range(tasks_per_burst):
                    future = executor.submit(task_func, burst_id, task_id)
                    futures.append(future)
                
                # 等待当前批次完成
                for future in futures:
                    try:
                        future.result(timeout=10)
                        total_completed += 1
                    except Exception:
                        pass
                
                # 等待间隔
                if burst_id < num_bursts - 1:
                    time.sleep(burst_interval)
        
        total_duration = time.time() - start_time
        
        return {
            'load_type': 'burst',
            'num_bursts': num_bursts,
            'tasks_per_burst': tasks_per_burst,
            'total_tasks': total_completed,
            'duration': total_duration,
            'burst_interval': burst_interval
        }
    
    def generate_ramp_load(self,
                         task_func: Callable,
                         initial_rate: int,
                         final_rate: int,
                         duration_seconds: float) -> Dict[str, Any]:
        """
        生成斜坡负载
        
        Args:
            task_func: 任务函数
            initial_rate: 初始速率
            final_rate: 最终速率
            duration_seconds: 持续时间
            
        Returns:
            Dict: 负载生成结果
        """
        start_time = time.time()
        completed_tasks = 0
        rate_increment = (final_rate - initial_rate) / duration_seconds
        
        with ThreadPoolExecutor(max_workers=max(final_rate, 20)) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                elapsed = time.time() - start_time
                current_rate = initial_rate + rate_increment * elapsed
                target_interval = 1.0 / max(current_rate, 1)
                
                # 提交新任务
                if len(futures) < int(current_rate * 2):
                    future = executor.submit(task_func, completed_tasks)
                    futures.append(future)
                
                # 收集完成的任务
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    futures.remove(future)
                    completed_tasks += 1
                
                time.sleep(min(target_interval, 0.1))
            
            # 等待剩余任务完成
            for future in futures:
                try:
                    future.result(timeout=5)
                    completed_tasks += 1
                except Exception:
                    pass
        
        total_duration = time.time() - start_time
        
        return {
            'load_type': 'ramp',
            'initial_rate': initial_rate,
            'final_rate': final_rate,
            'total_tasks': completed_tasks,
            'duration': total_duration
        }
    
    def generate_random_load(self,
                           task_func: Callable,
                           avg_tasks_per_second: int,
                           duration_seconds: float,
                           randomness_factor: float = 0.5) -> Dict[str, Any]:
        """
        生成随机负载
        
        Args:
            task_func: 任务函数
            avg_tasks_per_second: 平均每秒任务数
            duration_seconds: 持续时间
            randomness_factor: 随机因子 (0-1)
            
        Returns:
            Dict: 负载生成结果
        """
        start_time = time.time()
        completed_tasks = 0
        
        with ThreadPoolExecutor(max_workers=avg_tasks_per_second * 2) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                # 计算随机间隔
                base_interval = 1.0 / avg_tasks_per_second
                random_factor = 1.0 + randomness_factor * (np.random.random() - 0.5) * 2
                interval = base_interval * random_factor
                
                # 提交新任务
                if len(futures) < avg_tasks_per_second * 2:
                    future = executor.submit(task_func, completed_tasks)
                    futures.append(future)
                
                # 收集完成的任务
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    futures.remove(future)
                    completed_tasks += 1
                
                time.sleep(max(interval, 0.001))
            
            # 等待剩余任务完成
            for future in futures:
                try:
                    future.result(timeout=5)
                    completed_tasks += 1
                except Exception:
                    pass
        
        total_duration = time.time() - start_time
        
        return {
            'load_type': 'random',
            'avg_tasks_per_second': avg_tasks_per_second,
            'total_tasks': completed_tasks,
            'duration': total_duration,
            'randomness_factor': randomness_factor
        }


@dataclass
class StressTestResult:
    """压力测试结果数据类"""
    test_type: str
    duration: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    peak_cpu_usage: float
    avg_cpu_usage: float
    peak_memory_usage_mb: float
    avg_memory_usage_mb: float
    memory_growth_mb: float
    task_throughput: float
    avg_task_duration: float
    error_rate: float
    additional_metrics: Dict[str, Any]


class StressTester:
    """压力测试器"""
    
    def __init__(self, max_workers: int = 8, max_concurrent_tests: int = 16):
        """
        初始化压力测试器
        
        Args:
            max_workers: 最大工作线程数
            max_concurrent_tests: 最大并发测试数
        """
        self.max_workers = max_workers
        self.max_concurrent_tests = max_concurrent_tests
        self.process = psutil.Process()
        
        # 测试状态跟踪
        self.active_tests = set()
        self.test_results = []
        
        logger.info(f"压力测试器初始化完成，最大工作线程: {max_workers}")
    
    def run_cpu_stress_test(self, 
                           task_func: Callable,
                           task_args: tuple = (),
                           duration_seconds: float = 10,
                           concurrent_tasks: int = None) -> Dict[str, Any]:
        """
        运行CPU压力测试
        
        Args:
            task_func: CPU密集型任务函数
            task_args: 任务参数
            duration_seconds: 测试持续时间
            concurrent_tasks: 并发任务数
            
        Returns:
            Dict: 压力测试结果
        """
        if concurrent_tasks is None:
            concurrent_tasks = psutil.cpu_count()
        
        logger.info(f"开始CPU压力测试，持续时间: {duration_seconds}秒，并发数: {concurrent_tasks}")
        
        # 监控指标
        cpu_measurements = []
        memory_measurements = []
        task_durations = []
        completed_tasks = 0
        failed_tasks = 0
        
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 监控线程
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        def monitor_resources():
            """资源监控线程"""
            while monitoring_active.is_set():
                cpu_measurements.append(psutil.cpu_percent())
                memory_measurements.append(self.process.memory_info().rss / 1024 / 1024)
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
        
        # 执行压力测试
        with ThreadPoolExecutor(max_workers=concurrent_tasks) as executor:
            futures = []
            
            # 持续提交任务直到时间到达
            while time.time() - start_time < duration_seconds:
                if len(futures) < concurrent_tasks:
                    future = executor.submit(self._execute_timed_task, task_func, task_args)
                    futures.append(future)
                
                # 收集完成的任务
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    futures.remove(future)
                    try:
                        result, task_duration = future.result()
                        task_durations.append(task_duration)
                        completed_tasks += 1
                    except Exception as e:
                        logger.error(f"任务执行失败: {e}")
                        failed_tasks += 1
                
                time.sleep(0.01)  # 避免过度占用CPU
            
            # 等待剩余任务完成
            for future in as_completed(futures, timeout=10):
                try:
                    result, task_duration = future.result()
                    task_durations.append(task_duration)
                    completed_tasks += 1
                except Exception as e:
                    logger.error(f"任务执行失败: {e}")
                    failed_tasks += 1
        
        # 停止监控
        monitoring_active.clear()
        monitor_thread.join(timeout=1)
        
        end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 计算结果
        total_duration = end_time - start_time
        total_tasks = completed_tasks + failed_tasks
        
        result = {
            'test_type': 'cpu_stress',
            'duration': total_duration,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'peak_cpu_usage': max(cpu_measurements) if cpu_measurements else 0,
            'avg_cpu_usage': statistics.mean(cpu_measurements) if cpu_measurements else 0,
            'peak_memory_usage_mb': max(memory_measurements) if memory_measurements else initial_memory,
            'avg_memory_usage_mb': statistics.mean(memory_measurements) if memory_measurements else initial_memory,
            'memory_growth_mb': final_memory - initial_memory,
            'task_throughput': completed_tasks / total_duration if total_duration > 0 else 0,
            'avg_task_duration': statistics.mean(task_durations) if task_durations else 0,
            'error_rate': failed_tasks / total_tasks if total_tasks > 0 else 0,
            'concurrent_tasks': concurrent_tasks
        }
        
        logger.info(f"CPU压力测试完成，吞吐量: {result['task_throughput']:.1f} 任务/秒")
        return result
    
    def run_memory_stress_test(self,
                              task_func: Callable,
                              task_args: tuple = (),
                              duration_seconds: float = 10,
                              concurrent_tasks: int = 4) -> Dict[str, Any]:
        """
        运行内存压力测试
        
        Args:
            task_func: 内存密集型任务函数
            task_args: 任务参数
            duration_seconds: 测试持续时间
            concurrent_tasks: 并发任务数
            
        Returns:
            Dict: 压力测试结果
        """
        logger.info(f"开始内存压力测试，持续时间: {duration_seconds}秒")
        
        # 强制垃圾回收以获得准确的基线
        gc.collect()
        
        memory_measurements = []
        task_durations = []
        completed_tasks = 0
        failed_tasks = 0
        
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 内存监控
        monitoring_active = threading.Event()
        monitoring_active.set()
        
        def monitor_memory():
            """内存监控线程"""
            while monitoring_active.is_set():
                memory_measurements.append(self.process.memory_info().rss / 1024 / 1024)
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        # 执行内存压力测试
        with ThreadPoolExecutor(max_workers=concurrent_tasks) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                if len(futures) < concurrent_tasks:
                    future = executor.submit(self._execute_timed_task, task_func, task_args)
                    futures.append(future)
                
                # 收集完成的任务
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    futures.remove(future)
                    try:
                        result, task_duration = future.result()
                        task_durations.append(task_duration)
                        completed_tasks += 1
                    except Exception as e:
                        logger.error(f"内存任务执行失败: {e}")
                        failed_tasks += 1
                
                time.sleep(0.05)
            
            # 等待剩余任务完成
            for future in as_completed(futures, timeout=15):
                try:
                    result, task_duration = future.result()
                    task_durations.append(task_duration)
                    completed_tasks += 1
                except Exception as e:
                    failed_tasks += 1
        
        # 停止监控
        monitoring_active.clear()
        monitor_thread.join(timeout=1)
        
        end_time = time.time()
        
        # 再次强制垃圾回收
        gc.collect()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 计算结果
        total_duration = end_time - start_time
        total_tasks = completed_tasks + failed_tasks
        
        result = {
            'test_type': 'memory_stress',
            'duration': total_duration,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'peak_memory_usage_mb': max(memory_measurements) if memory_measurements else initial_memory,
            'avg_memory_usage_mb': statistics.mean(memory_measurements) if memory_measurements else initial_memory,
            'memory_growth_mb': final_memory - initial_memory,
            'task_throughput': completed_tasks / total_duration if total_duration > 0 else 0,
            'avg_task_duration': statistics.mean(task_durations) if task_durations else 0,
            'error_rate': failed_tasks / total_tasks if total_tasks > 0 else 0,
            'concurrent_tasks': concurrent_tasks
        }
        
        logger.info(f"内存压力测试完成，峰值内存: {result['peak_memory_usage_mb']:.1f}MB")
        return result
    
    def run_io_stress_test(self,
                          task_func: Callable,
                          task_args: tuple = (),
                          duration_seconds: float = 10,
                          concurrent_tasks: int = 6) -> Dict[str, Any]:
        """
        运行I/O压力测试
        
        Args:
            task_func: I/O密集型任务函数
            task_args: 任务参数
            duration_seconds: 测试持续时间
            concurrent_tasks: 并发任务数
            
        Returns:
            Dict: 压力测试结果
        """
        logger.info(f"开始I/O压力测试，持续时间: {duration_seconds}秒")
        
        task_durations = []
        completed_tasks = 0
        failed_tasks = 0
        
        start_time = time.time()
        
        # 执行I/O压力测试
        with ThreadPoolExecutor(max_workers=concurrent_tasks) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                if len(futures) < concurrent_tasks:
                    future = executor.submit(self._execute_timed_task, task_func, task_args)
                    futures.append(future)
                
                # 收集完成的任务
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    futures.remove(future)
                    try:
                        result, task_duration = future.result()
                        task_durations.append(task_duration)
                        completed_tasks += 1
                    except Exception as e:
                        logger.error(f"I/O任务执行失败: {e}")
                        failed_tasks += 1
                
                time.sleep(0.02)
            
            # 等待剩余任务完成
            for future in as_completed(futures, timeout=20):
                try:
                    result, task_duration = future.result()
                    task_durations.append(task_duration)
                    completed_tasks += 1
                except Exception as e:
                    failed_tasks += 1
        
        end_time = time.time()
        total_duration = end_time - start_time
        total_tasks = completed_tasks + failed_tasks
        
        result = {
            'test_type': 'io_stress',
            'duration': total_duration,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'task_throughput': completed_tasks / total_duration if total_duration > 0 else 0,
            'avg_task_duration': statistics.mean(task_durations) if task_durations else 0,
            'error_rate': failed_tasks / total_tasks if total_tasks > 0 else 0,
            'concurrent_tasks': concurrent_tasks
        }
        
        logger.info(f"I/O压力测试完成，平均任务时间: {result['avg_task_duration']:.3f}秒")
        return result
    
    def run_mixed_stress_test(self,
                             workloads: List[Tuple[Callable, tuple]],
                             duration_seconds: float = 15,
                             concurrent_tasks: int = 8) -> Dict[str, Any]:
        """
        运行混合工作负载压力测试
        
        Args:
            workloads: 工作负载列表 [(函数, 参数), ...]
            duration_seconds: 测试持续时间
            concurrent_tasks: 并发任务数
            
        Returns:
            Dict: 压力测试结果
        """
        logger.info(f"开始混合工作负载压力测试，{len(workloads)}种负载类型")
        
        task_results = []
        workload_counts = {i: 0 for i in range(len(workloads))}
        completed_tasks = 0
        failed_tasks = 0
        
        start_time = time.time()
        
        # 执行混合压力测试
        with ThreadPoolExecutor(max_workers=concurrent_tasks) as executor:
            futures = []
            workload_index = 0
            
            while time.time() - start_time < duration_seconds:
                if len(futures) < concurrent_tasks:
                    # 轮询选择工作负载
                    func, args = workloads[workload_index]
                    future = executor.submit(self._execute_timed_task, func, args)
                    future.workload_id = workload_index
                    futures.append(future)
                    
                    workload_index = (workload_index + 1) % len(workloads)
                
                # 收集完成的任务
                completed_futures = [f for f in futures if f.done()]
                for future in completed_futures:
                    futures.remove(future)
                    try:
                        result, task_duration = future.result()
                        task_results.append({
                            'workload_id': future.workload_id,
                            'duration': task_duration,
                            'result': result
                        })
                        workload_counts[future.workload_id] += 1
                        completed_tasks += 1
                    except Exception as e:
                        logger.error(f"混合任务执行失败: {e}")
                        failed_tasks += 1
                
                time.sleep(0.01)
            
            # 等待剩余任务完成
            for future in as_completed(futures, timeout=25):
                try:
                    result, task_duration = future.result()
                    task_results.append({
                        'workload_id': future.workload_id,
                        'duration': task_duration,
                        'result': result
                    })
                    workload_counts[future.workload_id] += 1
                    completed_tasks += 1
                except Exception as e:
                    failed_tasks += 1
        
        end_time = time.time()
        total_duration = end_time - start_time
        total_tasks = completed_tasks + failed_tasks
        
        # 分析工作负载分布
        workload_distribution = {}
        for i, count in workload_counts.items():
            workload_distribution[f'workload_{i}'] = {
                'count': count,
                'percentage': count / completed_tasks * 100 if completed_tasks > 0 else 0
            }
        
        result = {
            'test_type': 'mixed_stress',
            'duration': total_duration,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'task_throughput': completed_tasks / total_duration if total_duration > 0 else 0,
            'error_rate': failed_tasks / total_tasks if total_tasks > 0 else 0,
            'workload_distribution': workload_distribution,
            'concurrent_tasks': concurrent_tasks
        }
        
        logger.info(f"混合压力测试完成，总吞吐量: {result['task_throughput']:.1f} 任务/秒")
        return result
    
    def run_scalability_test(self,
                            task_func: Callable,
                            load_factors: List[int],
                            iterations_per_load: int = 5) -> Dict[str, Any]:
        """
        运行可扩展性测试
        
        Args:
            task_func: 测试任务函数（接受load_factor参数）
            load_factors: 不同的负载因子
            iterations_per_load: 每个负载级别的迭代次数
            
        Returns:
            Dict: 可扩展性测试结果
        """
        logger.info(f"开始可扩展性测试，负载因子: {load_factors}")
        
        load_results = []
        
        for load_factor in load_factors:
            logger.info(f"测试负载因子: {load_factor}")
            
            execution_times = []
            throughputs = []
            
            for iteration in range(iterations_per_load):
                start_time = time.time()
                
                try:
                    result = task_func(load_factor)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    execution_times.append(execution_time)
                    throughputs.append(1 / execution_time if execution_time > 0 else 0)
                    
                except Exception as e:
                    logger.error(f"可扩展性测试失败，负载因子 {load_factor}: {e}")
                    continue
            
            if execution_times:
                load_result = {
                    'load_factor': load_factor,
                    'iterations': len(execution_times),
                    'avg_execution_time': statistics.mean(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times),
                    'std_execution_time': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                    'throughput': statistics.mean(throughputs),
                    'throughput_std': statistics.stdev(throughputs) if len(throughputs) > 1 else 0
                }
                load_results.append(load_result)
        
        # 分析可扩展性
        scalability_analysis = self._analyze_scalability(load_results)
        
        result = {
            'test_type': 'scalability_stress',
            'load_factors': load_factors,
            'iterations_per_load': iterations_per_load,
            'load_results': load_results,
            'scalability_analysis': scalability_analysis
        }
        
        logger.info("可扩展性测试完成")
        return result
    
    def run_endurance_test(self,
                          task_func: Callable,
                          duration_seconds: float = 30,
                          task_interval: float = 0.1) -> Dict[str, Any]:
        """
        运行耐久性测试
        
        Args:
            task_func: 测试任务函数
            duration_seconds: 测试持续时间
            task_interval: 任务间隔时间
            
        Returns:
            Dict: 耐久性测试结果
        """
        logger.info(f"开始耐久性测试，持续时间: {duration_seconds}秒")
        
        execution_times = []
        memory_measurements = []
        task_results = []
        completed_tasks = 0
        failed_tasks = 0
        
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 执行耐久性测试
        while time.time() - start_time < duration_seconds:
            task_start = time.time()
            
            try:
                result = task_func()
                task_end = time.time()
                
                execution_time = task_end - task_start
                execution_times.append(execution_time)
                task_results.append(result)
                completed_tasks += 1
                
                # 记录内存使用
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory)
                
            except Exception as e:
                logger.error(f"耐久性测试任务失败: {e}")
                failed_tasks += 1
            
            # 等待间隔
            elapsed = time.time() - task_start
            if elapsed < task_interval:
                time.sleep(task_interval - elapsed)
        
        end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 分析性能退化
        performance_degradation = self._analyze_performance_degradation(execution_times)
        memory_stability = self._analyze_memory_stability(memory_measurements, initial_memory)
        
        total_duration = end_time - start_time
        total_tasks = completed_tasks + failed_tasks
        
        result = {
            'test_type': 'endurance_stress',
            'duration': total_duration,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'task_interval': task_interval,
            'avg_execution_time': statistics.mean(execution_times) if execution_times else 0,
            'execution_time_std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'memory_growth_mb': final_memory - initial_memory,
            'memory_stability': memory_stability,
            'performance_degradation': performance_degradation,
            'error_rate': failed_tasks / total_tasks if total_tasks > 0 else 0
        }
        
        logger.info(f"耐久性测试完成，性能退化: {performance_degradation:.2%}")
        return result
    
    def _execute_timed_task(self, func: Callable, args: tuple) -> Tuple[Any, float]:
        """执行计时任务"""
        start_time = time.time()
        result = func(*args)
        end_time = time.time()
        return result, end_time - start_time
    
    def _analyze_scalability(self, load_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析可扩展性"""
        if len(load_results) < 2:
            return {'analysis': 'insufficient_data'}
        
        # 计算性能缩放比例
        base_load = load_results[0]
        scaling_ratios = []
        
        for result in load_results[1:]:
            load_ratio = result['load_factor'] / base_load['load_factor']
            time_ratio = result['avg_execution_time'] / base_load['avg_execution_time']
            scaling_ratio = time_ratio / load_ratio
            scaling_ratios.append(scaling_ratio)
        
        avg_scaling_ratio = statistics.mean(scaling_ratios)
        
        # 判断可扩展性类型
        if avg_scaling_ratio < 0.8:
            scalability_type = 'super_linear'
        elif avg_scaling_ratio <= 1.2:
            scalability_type = 'linear'
        elif avg_scaling_ratio <= 2.0:
            scalability_type = 'sub_linear'
        else:
            scalability_type = 'poor'
        
        return {
            'avg_scaling_ratio': avg_scaling_ratio,
            'scalability_type': scalability_type,
            'scaling_ratios': scaling_ratios
        }
    
    def _analyze_performance_degradation(self, execution_times: List[float]) -> float:
        """分析性能退化"""
        if len(execution_times) < 10:
            return 0.0
        
        # 比较前10%和后10%的性能
        early_portion = int(len(execution_times) * 0.1)
        late_portion = int(len(execution_times) * 0.1)
        
        early_avg = statistics.mean(execution_times[:early_portion])
        late_avg = statistics.mean(execution_times[-late_portion:])
        
        degradation = (late_avg - early_avg) / early_avg if early_avg > 0 else 0
        return degradation
    
    def _analyze_memory_stability(self, memory_measurements: List[float], initial_memory: float) -> Dict[str, Any]:
        """分析内存稳定性"""
        if not memory_measurements:
            return {'stability': 'no_data'}
        
        memory_growth = memory_measurements[-1] - initial_memory
        memory_volatility = statistics.stdev(memory_measurements) if len(memory_measurements) > 1 else 0
        max_memory = max(memory_measurements)
        
        # 判断内存稳定性
        if memory_growth < 10 and memory_volatility < 5:
            stability = 'stable'
        elif memory_growth < 50 and memory_volatility < 20:
            stability = 'moderate'
        else:
            stability = 'unstable'
        
        return {
            'stability': stability,
            'memory_growth_mb': memory_growth,
            'memory_volatility_mb': memory_volatility,
            'peak_memory_mb': max_memory
        }