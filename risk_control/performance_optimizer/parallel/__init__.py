"""
并行处理模块

提供多线程和多进程并行处理功能
"""

from .config import ParallelConfig
from .processing_manager import ParallelProcessingManager

__all__ = [
    'ParallelConfig',
    'ParallelProcessingManager'
]