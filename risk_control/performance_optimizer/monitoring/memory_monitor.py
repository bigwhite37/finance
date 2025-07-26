"""
内存监控模块

提供内存使用监控、趋势分析和优化建议功能。
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # 创建占位符类
    class psutil:
        class Process:
            def memory_info(self):
                return type('obj', (object,), {'rss': 0, 'vms': 0})()
            def memory_percent(self):
                return 0.0
        @staticmethod
        def virtual_memory():
            return type('obj', (object,), {'available': 0, 'total': 0, 'percent': 0.0})()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class np:
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0
        @staticmethod
        def polyfit(x, y, deg):
            return [0] * (deg + 1)
        @staticmethod
        def arange(n):
            return list(range(n))


class MemoryMonitor:
    """内存使用监控器
    
    监控内存使用情况，提供内存优化建议和趋势分析。
    """
    
    def __init__(self, warning_threshold_mb: float = 1000.0,
                 critical_threshold_mb: float = 2000.0,
                 history_size: int = 1000):
        """初始化内存监控器
        
        Args:
            warning_threshold_mb: 警告阈值(MB)
            critical_threshold_mb: 临界阈值(MB)
            history_size: 历史记录最大长度
        """
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.history_size = history_size
        
        self._memory_history = []
        self._peak_memory = 0.0
        self._lock = threading.RLock()
        
        # 监控统计
        self._monitoring_stats = {
            'total_checks': 0,
            'warning_count': 0,
            'critical_count': 0,
            'normal_count': 0,
            'peak_memory_mb': 0.0,
            'average_memory_mb': 0.0
        }
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况
        
        Returns:
            内存使用信息字典
        """
        with self._lock:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()
                
                current_memory_mb = memory_info.rss / 1024 / 1024
                virtual_memory_mb = memory_info.vms / 1024 / 1024
                
                # 更新峰值内存
                self._peak_memory = max(self._peak_memory, current_memory_mb)
                self._monitoring_stats['peak_memory_mb'] = self._peak_memory
                
                # 记录内存历史
                self._memory_history.append({
                    'timestamp': time.time(),
                    'memory_mb': current_memory_mb,
                    'virtual_memory_mb': virtual_memory_mb,
                    'memory_percent': process.memory_percent()
                })
                
                # 限制历史记录长度
                if len(self._memory_history) > self.history_size:
                    self._memory_history = self._memory_history[-self.history_size//2:]
                
                # 更新平均内存使用
                if self._memory_history:
                    avg_memory = sum(item['memory_mb'] for item in self._memory_history) / len(self._memory_history)
                    self._monitoring_stats['average_memory_mb'] = avg_memory
                
                return {
                    'current_memory_mb': current_memory_mb,
                    'virtual_memory_mb': virtual_memory_mb,
                    'peak_memory_mb': self._peak_memory,
                    'memory_percent': process.memory_percent(),
                    'available_memory_mb': system_memory.available / 1024 / 1024,
                    'total_system_memory_mb': system_memory.total / 1024 / 1024,
                    'system_memory_percent': system_memory.percent
                }
                
            except Exception as e:
                logging.error(f"获取内存使用情况失败: {e}")
                return {
                    'current_memory_mb': 0.0,
                    'virtual_memory_mb': 0.0,
                    'peak_memory_mb': self._peak_memory,
                    'memory_percent': 0.0,
                    'available_memory_mb': 0.0,
                    'total_system_memory_mb': 0.0,
                    'system_memory_percent': 0.0
                }
    
    def check_memory_status(self) -> Dict[str, Any]:
        """检查内存状态
        
        Returns:
            内存状态信息
        """
        with self._lock:
            self._monitoring_stats['total_checks'] += 1
            
            memory_info = self.get_current_memory_usage()
            current_memory = memory_info['current_memory_mb']
            
            if current_memory >= self.critical_threshold:
                status = 'critical'
                message = f"内存使用达到临界水平: {current_memory:.1f}MB"
                recommendations = [
                    "立即清理缓存",
                    "减少并行处理线程数",
                    "使用数据分块处理",
                    "考虑重启应用程序",
                    "检查是否存在内存泄漏"
                ]
                self._monitoring_stats['critical_count'] += 1
                
            elif current_memory >= self.warning_threshold:
                status = 'warning'
                message = f"内存使用较高: {current_memory:.1f}MB"
                recommendations = [
                    "清理不必要的缓存",
                    "优化数据结构",
                    "减少内存中的数据量",
                    "考虑使用更高效的算法"
                ]
                self._monitoring_stats['warning_count'] += 1
                
            else:
                status = 'normal'
                message = f"内存使用正常: {current_memory:.1f}MB"
                recommendations = []
                self._monitoring_stats['normal_count'] += 1
            
            return {
                'status': status,
                'message': message,
                'recommendations': recommendations,
                'memory_info': memory_info,
                'memory_trend': self._calculate_memory_trend(),
                'memory_growth_rate': self._calculate_memory_growth_rate()
            }
    
    def get_memory_history(self, last_n_minutes: Optional[int] = None) -> List[Dict]:
        """获取内存使用历史
        
        Args:
            last_n_minutes: 获取最近N分钟的历史，None表示获取全部
            
        Returns:
            内存使用历史列表
        """
        with self._lock:
            if not self._memory_history:
                return []
            
            if last_n_minutes is None:
                return self._memory_history.copy()
            
            cutoff_time = time.time() - (last_n_minutes * 60)
            return [item for item in self._memory_history if item['timestamp'] >= cutoff_time]
    
    def _calculate_memory_trend(self) -> str:
        """计算内存使用趋势"""
        if len(self._memory_history) < 10:
            return 'insufficient_data'
        
        recent_memory = [item['memory_mb'] for item in self._memory_history[-10:]]
        
        try:
            # 计算线性趋势
            x = np.arange(len(recent_memory))
            slope = np.polyfit(x, recent_memory, 1)[0]
            
            if slope > 5:  # 每次测量增长超过5MB
                return 'increasing'
            elif slope < -5:
                return 'decreasing'
            else:
                return 'stable'
        except Exception:
            return 'unknown'
    
    def _calculate_memory_growth_rate(self) -> float:
        """计算内存增长率 (MB/分钟)"""
        if len(self._memory_history) < 2:
            return 0.0
        
        try:
            # 使用最近的数据点计算增长率
            recent_data = self._memory_history[-min(20, len(self._memory_history)):]
            
            if len(recent_data) < 2:
                return 0.0
            
            time_diff = recent_data[-1]['timestamp'] - recent_data[0]['timestamp']
            memory_diff = recent_data[-1]['memory_mb'] - recent_data[0]['memory_mb']
            
            if time_diff > 0:
                # 转换为每分钟的增长率
                growth_rate = (memory_diff / time_diff) * 60
                return growth_rate
            
            return 0.0
        except Exception:
            return 0.0
    
    def get_memory_optimization_suggestions(self) -> List[str]:
        """获取内存优化建议"""
        with self._lock:
            memory_info = self.get_current_memory_usage()
            suggestions = []
            
            current_memory = memory_info['current_memory_mb']
            system_memory_percent = memory_info['system_memory_percent']
            
            # 基于当前内存使用的建议
            if current_memory > 500:
                suggestions.append("考虑使用数据分块处理大型数据集")
            
            if system_memory_percent > 80:
                suggestions.append("系统内存使用率过高，考虑增加物理内存")
            
            if self._peak_memory > current_memory * 2:
                suggestions.append("存在内存峰值，考虑优化算法以减少内存峰值使用")
            
            # 基于趋势的建议
            trend = self._calculate_memory_trend()
            if trend == 'increasing':
                growth_rate = self._calculate_memory_growth_rate()
                if growth_rate > 10:  # 每分钟增长超过10MB
                    suggestions.append("内存使用快速增长，可能存在内存泄漏")
                else:
                    suggestions.append("内存使用呈上升趋势，建议监控")
            
            # 基于历史数据的建议
            if len(self._memory_history) > 100:
                recent_avg = np.mean([item['memory_mb'] for item in self._memory_history[-50:]])
                historical_avg = np.mean([item['memory_mb'] for item in self._memory_history[:-50]])
                
                if recent_avg > historical_avg * 1.5:
                    suggestions.append("最近内存使用显著增加，建议检查代码变更")
            
            return suggestions
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """获取监控统计信息
        
        Returns:
            监控统计信息字典
        """
        with self._lock:
            total_checks = self._monitoring_stats['total_checks']
            
            return {
                **self._monitoring_stats,
                'warning_rate': self._monitoring_stats['warning_count'] / total_checks if total_checks > 0 else 0.0,
                'critical_rate': self._monitoring_stats['critical_count'] / total_checks if total_checks > 0 else 0.0,
                'normal_rate': self._monitoring_stats['normal_count'] / total_checks if total_checks > 0 else 0.0,
                'history_length': len(self._memory_history),
                'monitoring_duration_hours': (time.time() - self._memory_history[0]['timestamp']) / 3600 if self._memory_history else 0.0
            }
    
    def reset_monitoring_data(self) -> None:
        """重置监控数据"""
        with self._lock:
            self._memory_history.clear()
            self._peak_memory = 0.0
            self._monitoring_stats = {
                'total_checks': 0,
                'warning_count': 0,
                'critical_count': 0,
                'normal_count': 0,
                'peak_memory_mb': 0.0,
                'average_memory_mb': 0.0
            }
    
    def set_thresholds(self, warning_threshold_mb: float, critical_threshold_mb: float) -> None:
        """设置内存阈值
        
        Args:
            warning_threshold_mb: 警告阈值(MB)
            critical_threshold_mb: 临界阈值(MB)
        """
        with self._lock:
            if warning_threshold_mb >= critical_threshold_mb:
                raise ValueError("警告阈值必须小于临界阈值")
            
            if warning_threshold_mb <= 0 or critical_threshold_mb <= 0:
                raise ValueError("阈值必须大于0")
            
            self.warning_threshold = warning_threshold_mb
            self.critical_threshold = critical_threshold_mb
            
            logging.info(f"内存阈值已更新: 警告={warning_threshold_mb}MB, 临界={critical_threshold_mb}MB")
    
    def export_memory_report(self) -> Dict[str, Any]:
        """导出内存使用报告
        
        Returns:
            详细的内存使用报告
        """
        with self._lock:
            current_status = self.check_memory_status()
            monitoring_stats = self.get_monitoring_stats()
            optimization_suggestions = self.get_memory_optimization_suggestions()
            
            return {
                'report_timestamp': time.time(),
                'current_status': current_status,
                'monitoring_statistics': monitoring_stats,
                'optimization_suggestions': optimization_suggestions,
                'configuration': {
                    'warning_threshold_mb': self.warning_threshold,
                    'critical_threshold_mb': self.critical_threshold,
                    'history_size': self.history_size
                },
                'memory_trend_analysis': {
                    'trend': self._calculate_memory_trend(),
                    'growth_rate_mb_per_minute': self._calculate_memory_growth_rate(),
                    'peak_memory_mb': self._peak_memory
                }
            }