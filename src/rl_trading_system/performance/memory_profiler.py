"""
内存分析器

该模块提供详细的内存分析功能，包括：
- 函数级内存分析
- 内存泄漏检测
- 内存热点分析
- 内存使用模式分析
"""

import gc
import psutil
import time
import threading
from typing import Dict, Any, Callable, Optional, List
from contextlib import contextmanager
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        """初始化内存分析器"""
        self.process = psutil.Process()
        self.baseline_memory = self.get_current_memory()
    
    def get_current_memory(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def profile_function(self, func: Callable, *args, **kwargs) -> tuple:
        """
        分析函数内存使用
        
        Returns:
            tuple: (函数结果, 内存分析结果)
        """
        start_memory = self.get_current_memory()
        start_time = time.time()
        
        peak_memory = start_memory
        memory_samples = []
        
        # 在后台监控内存使用
        def monitor_memory():
            nonlocal peak_memory
            while getattr(monitor_memory, 'running', True):
                current_memory = self.get_current_memory()
                peak_memory = max(peak_memory, current_memory)
                memory_samples.append((time.time() - start_time, current_memory))
                time.sleep(0.01)  # 10ms采样
        
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.daemon = True
        monitor_memory.running = True
        monitor_thread.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            monitor_memory.running = False
            monitor_thread.join(timeout=1)
        
        end_time = time.time()
        end_memory = self.get_current_memory()
        
        profile = {
            'start_memory_mb': start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': end_memory - start_memory,
            'execution_time': end_time - start_time,
            'memory_samples': memory_samples
        }
        
        return result, profile
    
    @contextmanager
    def profile_context(self):
        """内存分析上下文管理器"""
        start_memory = self.get_current_memory()
        peak_memory = start_memory
        
        profile = {
            'start_memory': start_memory,
            'peak_memory': peak_memory
        }
        
        try:
            yield profile
        finally:
            end_memory = self.get_current_memory()
            profile.update({
                'end_memory': end_memory,
                'peak_memory': max(profile['peak_memory'], end_memory),
                'memory_delta': end_memory - start_memory
            })
    
    def detect_memory_leak(self, initial_memory: float, final_memory: float, 
                          threshold_mb: float = 50) -> bool:
        """检测内存泄漏"""
        memory_growth = final_memory - initial_memory
        return memory_growth > threshold_mb
    
    def analyze_memory_hotspots(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """分析内存热点"""
        result, profile = self.profile_function(func, *args, **kwargs)
        
        # 分析内存使用模式
        memory_samples = profile.get('memory_samples', [])
        
        if len(memory_samples) > 1:
            # 计算内存增长率
            times = [sample[0] for sample in memory_samples]
            memories = [sample[1] for sample in memory_samples]
            
            # 简单的增长率分析
            if len(memories) > 2:
                growth_rate = (memories[-1] - memories[0]) / (times[-1] - times[0]) if times[-1] > times[0] else 0
                volatility = np.std(memories)
            else:
                growth_rate = 0
                volatility = 0
        else:
            growth_rate = 0
            volatility = 0
        
        hotspots = {
            'total_allocations': 1,  # 简化实现
            'peak_usage_mb': profile['peak_memory_mb'],
            'allocation_pattern': self._classify_allocation_pattern(growth_rate, volatility),
            'memory_efficiency': self._calculate_memory_efficiency(profile),
            'growth_rate_mb_per_sec': growth_rate
        }
        
        return hotspots
    
    def _classify_allocation_pattern(self, growth_rate: float, volatility: float) -> str:
        """分类内存分配模式"""
        if abs(growth_rate) < 0.1:
            return 'stable'
        elif growth_rate > 1.0:
            return 'rapid_growth'
        elif growth_rate > 0.1:
            return 'gradual_growth'
        elif volatility > 10:
            return 'volatile'
        else:
            return 'sequential'
    
    def _calculate_memory_efficiency(self, profile: Dict[str, Any]) -> float:
        """计算内存效率"""
        if profile['peak_memory_mb'] > 0:
            return profile['memory_growth_mb'] / profile['peak_memory_mb']
        return 1.0