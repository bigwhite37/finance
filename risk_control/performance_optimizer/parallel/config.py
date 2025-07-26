"""
并行处理配置模块

定义并行处理的各种配置选项
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelConfig:
    """并行处理配置"""
    enable_parallel: bool = True
    max_workers: Optional[int] = None  # None表示使用CPU核心数
    use_process_pool: bool = False  # True使用进程池，False使用线程池
    chunk_size: int = 10  # 批处理大小
    timeout_seconds: int = 300  # 超时时间
    
    def validate(self) -> bool:
        """验证配置参数"""
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValueError("最大工作线程数必须大于0")
        
        if self.chunk_size <= 0:
            raise ValueError("批处理大小必须大于0")
        
        if self.timeout_seconds <= 0:
            raise ValueError("超时时间必须大于0")
        
        return True
    
    @classmethod
    def create_default_config(cls) -> 'ParallelConfig':
        """创建默认配置"""
        return cls()
    
    @classmethod
    def create_high_performance_config(cls) -> 'ParallelConfig':
        """创建高性能配置"""
        return cls(
            enable_parallel=True,
            max_workers=None,  # 使用所有CPU核心
            use_process_pool=True,  # 使用进程池获得更好的并行性
            chunk_size=5,  # 较小的批处理大小
            timeout_seconds=600
        )
    
    @classmethod
    def create_conservative_config(cls) -> 'ParallelConfig':
        """创建保守配置"""
        import multiprocessing
        return cls(
            enable_parallel=True,
            max_workers=max(1, multiprocessing.cpu_count() // 2),  # 使用一半CPU核心
            use_process_pool=False,  # 使用线程池，开销较小
            chunk_size=20,  # 较大的批处理大小
            timeout_seconds=180
        )
    
    @classmethod
    def create_single_thread_config(cls) -> 'ParallelConfig':
        """创建单线程配置（用于调试）"""
        return cls(
            enable_parallel=False,
            max_workers=1,
            use_process_pool=False,
            chunk_size=1,
            timeout_seconds=60
        )