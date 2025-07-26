"""
Memory optimization utilities for O2O RL training.
Provides memory-efficient data structures and monitoring tools.
"""

import gc
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import threading
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None


class MemoryMonitor:
    """Real-time memory usage monitor"""
    
    def __init__(self, check_interval: float = 1.0, alert_threshold: float = 0.85):
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.monitoring = False
        self.stats_history = deque(maxlen=1000)
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                stats = self.get_memory_stats()
                self.stats_history.append(stats)
                
                # Check for memory alerts
                if stats.memory_percent > self.alert_threshold:
                    logger.warning(f"High memory usage: {stats.memory_percent:.1f}%")
                    
                if stats.gpu_memory_used and stats.gpu_memory_total:
                    gpu_percent = stats.gpu_memory_used / stats.gpu_memory_total
                    if gpu_percent > self.alert_threshold:
                        logger.warning(f"High GPU memory usage: {gpu_percent:.1f}%")
                        
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        memory = psutil.virtual_memory()
        
        gpu_memory_used = None
        gpu_memory_total = None
        
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            except Exception:
                pass
                
        return MemoryStats(
            total_memory=memory.total / 1024**3,  # GB
            available_memory=memory.available / 1024**3,  # GB
            used_memory=memory.used / 1024**3,  # GB
            memory_percent=memory.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total
        )
        
    def get_peak_memory_usage(self) -> Optional[MemoryStats]:
        """Get peak memory usage from history"""
        if not self.stats_history:
            return None
            
        return max(self.stats_history, key=lambda x: x.memory_percent)
        
    def clear_history(self):
        """Clear monitoring history"""
        self.stats_history.clear()


class MemoryEfficientBuffer:
    """Memory-efficient circular buffer with compression"""
    
    def __init__(self, capacity: int, compress_threshold: int = 1000):
        self.capacity = capacity
        self.compress_threshold = compress_threshold
        self.buffer = deque(maxlen=capacity)
        self.compressed_data = []
        self.compression_enabled = True
        
    def add(self, item: Any):
        """Add item to buffer with automatic compression"""
        self.buffer.append(item)
        
        # Compress old data if buffer is getting large
        if (self.compression_enabled and 
            len(self.buffer) > self.compress_threshold):
            self._compress_old_data()
            
    def _compress_old_data(self):
        """Compress older data to save memory"""
        # Move half of the buffer to compressed storage
        compress_count = len(self.buffer) // 2
        
        for _ in range(compress_count):
            if self.buffer:
                item = self.buffer.popleft()
                # Simple compression - convert to bytes if possible
                if hasattr(item, 'numpy'):
                    compressed = np.array(item.numpy(), dtype=np.float16)
                    self.compressed_data.append(compressed)
                else:
                    self.compressed_data.append(item)
                    
        # Keep compressed data size manageable
        if len(self.compressed_data) > self.capacity:
            self.compressed_data = self.compressed_data[-self.capacity//2:]
            
    def get_recent(self, count: int) -> List[Any]:
        """Get most recent items"""
        return list(self.buffer)[-count:] if count <= len(self.buffer) else list(self.buffer)
        
    def __len__(self):
        return len(self.buffer) + len(self.compressed_data)
        
    def clear(self):
        """Clear all data"""
        self.buffer.clear()
        self.compressed_data.clear()


class GradientAccumulator:
    """Gradient accumulation for large batch training with limited memory"""
    
    def __init__(self, model: torch.nn.Module, accumulation_steps: int = 4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.accumulated_steps = 0
        self.accumulated_loss = 0.0
        
    @contextmanager
    def accumulate(self):
        """Context manager for gradient accumulation"""
        try:
            # Scale gradients by accumulation steps
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                yield
        finally:
            self.accumulated_steps += 1
            
    def should_update(self) -> bool:
        """Check if gradients should be updated"""
        return self.accumulated_steps >= self.accumulation_steps
        
    def update_and_reset(self, optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """Update model parameters and reset accumulation"""
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            
        optimizer.zero_grad()
        self.accumulated_steps = 0
        self.accumulated_loss = 0.0
        
    def add_loss(self, loss: torch.Tensor):
        """Add loss to accumulation"""
        self.accumulated_loss += loss.item() / self.accumulation_steps


class MemoryOptimizedTensorDataset:
    """Memory-optimized tensor dataset with lazy loading"""
    
    def __init__(self, data_path: str, chunk_size: int = 1000):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.current_chunk = None
        self.current_chunk_idx = -1
        self.total_samples = 0
        self._load_metadata()
        
    def _load_metadata(self):
        """Load dataset metadata without loading full data"""
        # This would be implemented based on your data format
        # For now, assume we can get sample count from file
        pass
        
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx: int):
        """Get item with lazy loading"""
        chunk_idx = idx // self.chunk_size
        
        # Load chunk if not already loaded
        if chunk_idx != self.current_chunk_idx:
            self._load_chunk(chunk_idx)
            
        local_idx = idx % self.chunk_size
        return self.current_chunk[local_idx]
        
    def _load_chunk(self, chunk_idx: int):
        """Load specific data chunk"""
        # Implementation would depend on your data format
        # This is a placeholder for chunk loading logic
        self.current_chunk_idx = chunk_idx
        # self.current_chunk = load_chunk_from_file(self.data_path, chunk_idx)


def optimize_tensor_memory(tensor: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Optimize tensor memory usage by reducing precision"""
    if tensor.dtype != dtype and dtype in [torch.float16, torch.bfloat16]:
        return tensor.to(dtype)
    return tensor


def clear_gpu_cache():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def force_garbage_collection():
    """Force garbage collection to free memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def memory_profiler(name: str = "operation"):
    """Context manager for profiling memory usage"""
    monitor = MemoryMonitor()
    
    # Get initial memory
    initial_stats = monitor.get_memory_stats()
    logger.info(f"[{name}] Initial memory: {initial_stats.memory_percent:.1f}%")
    
    try:
        yield monitor
    finally:
        # Get final memory
        final_stats = monitor.get_memory_stats()
        memory_diff = final_stats.used_memory - initial_stats.used_memory
        
        logger.info(f"[{name}] Final memory: {final_stats.memory_percent:.1f}%")
        logger.info(f"[{name}] Memory change: {memory_diff:+.2f} GB")
        
        if memory_diff > 0.5:  # Alert if memory increased by more than 500MB
            logger.warning(f"[{name}] Large memory increase detected: {memory_diff:.2f} GB")


class ModelParallelWrapper:
    """Wrapper for model parallelism across multiple GPUs"""
    
    def __init__(self, model: torch.nn.Module, device_ids: List[int] = None):
        self.model = model
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.is_parallel = len(self.device_ids) > 1
        
        if self.is_parallel and torch.cuda.is_available():
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)
            logger.info(f"Model parallelized across GPUs: {self.device_ids}")
        else:
            logger.info("Single GPU or CPU mode")
            
    def forward(self, *args, **kwargs):
        """Forward pass with automatic device handling"""
        return self.model(*args, **kwargs)
        
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()
        
    def state_dict(self):
        """Get model state dict"""
        if self.is_parallel:
            return self.model.module.state_dict()
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        """Load model state dict"""
        if self.is_parallel:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


# Global memory monitor instance
global_memory_monitor = MemoryMonitor()