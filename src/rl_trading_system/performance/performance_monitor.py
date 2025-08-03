"""
性能监控器

该模块提供实时性能监控功能，包括：
- 实时系统资源监控
- 函数性能监控
- 性能告警
- 性能趋势分析
"""

import time
import threading
import psutil
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, asdict
from collections import deque
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_io_sent: int
    network_io_recv: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class FunctionPerformance:
    """函数性能数据类"""
    function_name: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    start_time: float
    end_time: float


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, sampling_interval: float = 1.0, max_samples: int = 1000):
        """
        初始化性能监控器
        
        Args:
            sampling_interval: 采样间隔（秒）
            max_samples: 最大样本数
        """
        self.sampling_interval = sampling_interval
        self.max_samples = max_samples
        self.process = psutil.Process()
        
        # 监控数据存储
        self.metrics = deque(maxlen=max_samples)
        self.function_metrics = {}
        self.alerts = []
        
        # 监控控制
        self.monitoring = False
        self.monitor_thread = None
        
        # 告警阈值
        self.cpu_threshold = 80.0
        self.memory_threshold = 80.0
        
        # 资源监控状态
        self.resource_monitoring = False
        self.resource_start_time = None
        self.resource_metrics = []
    
    def start_monitoring(self):
        """启动实时监控"""
        if self.monitoring:
            logger.warning("监控已经在运行")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        if not self.monitoring:
            logger.warning("监控未在运行")
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                # 收集系统指标
                metric = self._collect_system_metrics()
                self.metrics.append(metric)
                
                # 检查告警
                self._check_alerts(metric)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(self.sampling_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetric:
        """收集系统性能指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent()
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        # 磁盘I/O
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes if disk_io else 0
        disk_write = disk_io.write_bytes if disk_io else 0
        
        # 网络I/O
        network_io = psutil.net_io_counters()
        net_sent = network_io.bytes_sent if network_io else 0
        net_recv = network_io.bytes_recv if network_io else 0
        
        return PerformanceMetric(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory.percent,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_io_sent=net_sent,
            network_io_recv=net_recv
        )
    
    def _check_alerts(self, metric: PerformanceMetric):
        """检查性能告警"""
        alerts = []
        
        if metric.cpu_percent > self.cpu_threshold:
            alerts.append({
                'type': 'cpu_high',
                'message': f'CPU使用率过高: {metric.cpu_percent:.1f}%',
                'timestamp': metric.timestamp,
                'value': metric.cpu_percent,
                'threshold': self.cpu_threshold
            })
        
        if metric.memory_percent > self.memory_threshold:
            alerts.append({
                'type': 'memory_high',
                'message': f'内存使用率过高: {metric.memory_percent:.1f}%',
                'timestamp': metric.timestamp,
                'value': metric.memory_percent,
                'threshold': self.memory_threshold
            })
        
        # 添加到告警列表
        self.alerts.extend(alerts)
        
        # 限制告警数量
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def set_alert_thresholds(self, cpu_threshold: float = None, memory_threshold: float = None):
        """设置告警阈值"""
        if cpu_threshold is not None:
            self.cpu_threshold = cpu_threshold
        if memory_threshold is not None:
            self.memory_threshold = memory_threshold
        
        logger.info(f"告警阈值已更新: CPU={self.cpu_threshold}%, Memory={self.memory_threshold}%")
    
    def get_metrics(self) -> List[Dict[str, Any]]:
        """获取监控指标"""
        return [metric.to_dict() for metric in self.metrics]
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警信息"""
        return self.alerts.copy()
    
    def clear_alerts(self):
        """清空告警"""
        self.alerts.clear()
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(self.metrics) < 2:
            return {'error': '数据不足，无法分析趋势'}
        
        # 提取数据
        cpu_values = [m.cpu_percent for m in self.metrics]
        memory_values = [m.memory_mb for m in self.metrics]
        timestamps = [m.timestamp for m in self.metrics]
        
        # 计算统计指标
        analysis = {
            'data_points': len(self.metrics),
            'time_span': timestamps[-1] - timestamps[0],
            'avg_cpu': statistics.mean(cpu_values),
            'avg_memory': statistics.mean(memory_values),
            'max_cpu': max(cpu_values),
            'max_memory': max(memory_values),
            'min_cpu': min(cpu_values),
            'min_memory': min(memory_values)
        }
        
        # 趋势分析（简化的线性趋势）
        if len(cpu_values) > 1:
            cpu_trend = self._calculate_trend(cpu_values)
            memory_trend = self._calculate_trend(memory_values)
            
            analysis.update({
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'cpu_volatility': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0,
                'memory_volatility': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            })
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return 'unknown'
        
        # 简单的趋势计算：比较前半段和后半段的平均值
        mid = len(values) // 2
        first_half_avg = statistics.mean(values[:mid])
        second_half_avg = statistics.mean(values[mid:])
        
        change_ratio = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
        
        if change_ratio > 0.1:
            return 'increasing'
        elif change_ratio < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def compare_functions(self, functions: List[Callable]) -> Dict[str, Dict[str, Any]]:
        """比较多个函数的性能"""
        results = {}
        
        for func in functions:
            # 监控函数执行
            start_time = time.time()
            start_cpu = psutil.cpu_percent()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            try:
                # 执行函数
                func()
                
                end_time = time.time()
                end_cpu = psutil.cpu_percent()
                end_memory = self.process.memory_info().rss / 1024 / 1024
                
                results[func.__name__] = {
                    'execution_time': end_time - start_time,
                    'cpu_usage': end_cpu - start_cpu,
                    'memory_usage': end_memory - start_memory,
                    'success': True
                }
                
            except Exception as e:
                results[func.__name__] = {
                    'execution_time': 0,
                    'cpu_usage': 0,
                    'memory_usage': 0,
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def start_resource_monitoring(self):
        """启动资源监控"""
        self.resource_monitoring = True
        self.resource_start_time = time.time()
        self.resource_metrics = []
        
        # 记录初始状态
        initial_metric = self._collect_system_metrics()
        self.resource_metrics.append(initial_metric)
        
        logger.info("资源监控已启动")
    
    def stop_resource_monitoring(self) -> Dict[str, Any]:
        """停止资源监控并返回分析结果"""
        if not self.resource_monitoring:
            return {'error': '资源监控未启动'}
        
        # 记录最终状态
        final_metric = self._collect_system_metrics()
        self.resource_metrics.append(final_metric)
        
        self.resource_monitoring = False
        
        # 分析资源利用率
        if len(self.resource_metrics) >= 2:
            cpu_values = [m.cpu_percent for m in self.resource_metrics]
            memory_values = [m.memory_mb for m in self.resource_metrics]
            
            analysis = {
                'duration': time.time() - self.resource_start_time,
                'cpu_utilization': statistics.mean(cpu_values),
                'memory_utilization': statistics.mean(memory_values),
                'peak_cpu': max(cpu_values),
                'peak_memory': max(memory_values),
                'min_cpu': min(cpu_values),
                'min_memory': min(memory_values),
                'samples_count': len(self.resource_metrics)
            }
        else:
            analysis = {'error': '数据不足'}
        
        logger.info("资源监控已停止")
        return analysis
    
    def get_current_status(self) -> Dict[str, Any]:
        """获取当前监控状态"""
        current_metric = self._collect_system_metrics()
        
        return {
            'monitoring_active': self.monitoring,
            'current_metrics': current_metric.to_dict(),
            'total_samples': len(self.metrics),
            'recent_alerts': len([a for a in self.alerts if time.time() - a['timestamp'] < 300]),  # 5分钟内的告警
            'cpu_threshold': self.cpu_threshold,
            'memory_threshold': self.memory_threshold
        }