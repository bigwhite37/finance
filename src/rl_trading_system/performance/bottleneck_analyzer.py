"""
瓶颈分析器

该模块提供系统瓶颈分析功能，包括：
- CPU瓶颈分析
- 内存瓶颈分析
- I/O瓶颈分析
- 函数调用分析
"""

import time
import psutil
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BottleneckAnalysis:
    """瓶颈分析结果"""
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    top_functions: List[Dict[str, Any]]
    recommendations: List[str]


class BottleneckAnalyzer:
    """瓶颈分析器"""
    
    def __init__(self):
        """初始化瓶颈分析器"""
        self.process = psutil.Process()
    
    def analyze_system_bottlenecks(self) -> BottleneckAnalysis:
        """分析系统瓶颈"""
        # 收集系统指标
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # 分析建议
        recommendations = []
        
        if cpu_usage > 80:
            recommendations.append("CPU使用率过高，考虑优化算法或增加并行处理")
        
        if memory.percent > 80:
            recommendations.append("内存使用率过高，考虑优化内存使用或增加内存")
        
        return BottleneckAnalysis(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_io={'read_bytes': disk_io.read_bytes, 'write_bytes': disk_io.write_bytes} if disk_io else {},
            network_io={'bytes_sent': network_io.bytes_sent, 'bytes_recv': network_io.bytes_recv} if network_io else {},
            top_functions=[],
            recommendations=recommendations
        )
    
    def profile_function(self, func, *args, **kwargs) -> Dict[str, Any]:
        """分析函数性能"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
        
        # 获取统计信息
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        return {
            'result': result,
            'profile_stats': s.getvalue()
        }