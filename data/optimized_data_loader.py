"""
Memory-optimized data loader for O2O RL training.
Provides efficient data loading with memory management and parallel processing.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Iterator, Any
from torch.utils.data import Dataset, DataLoader, IterableDataset
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import pickle
import mmap
import h5py
from utils.memory_optimizer import (
    MemoryMonitor, optimize_tensor_memory, memory_profiler,
    clear_gpu_cache, MemoryEfficientBuffer
)

logger = logging.getLogger(__name__)


@dataclass
class DataChunk:
    """Data chunk for efficient loading"""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    metadata: Dict[str, Any]


class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for large-scale data"""
    
    def __init__(self, data_path: str, chunk_size: int = 1000):
        self.data_path = Path(data_path)
        self.chunk_size = chunk_size
        self.memory_map = None
        self.data_info = None
        self._load_metadata()
        
    def _load_metadata(self):
        """Load dataset metadata"""
        metadata_path = self.data_path.with_suffix('.meta')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.data_info = pickle.load(f)
        else:
            # Create metadata if not exists
            self._create_metadata()
            
    def _create_metadata(self):
        """Create metadata for the dataset"""
        # This would analyze the data file and create index
        # Implementation depends on your data format
        self.data_info = {
            'total_samples': 0,
            'chunk_offsets': [],
            'data_shapes': {},
            'data_types': {}
        }
        
    def __len__(self):
        return self.data_info['total_samples']
        
    def __getitem__(self, idx: int):
        """Get item with memory mapping"""
        if self.memory_map is None:
            self.memory_map = self._open_memory_map()
            
        # Calculate chunk and local index
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        # Load data from memory map
        return self._load_from_memory_map(chunk_idx, local_idx)
        
    def _open_memory_map(self):
        """Open memory-mapped file"""
        # Implementation depends on your data format
        # This is a placeholder
        return None
        
    def _load_from_memory_map(self, chunk_idx: int, local_idx: int):
        """Load data from memory map"""
        # Implementation depends on your data format
        # This is a placeholder
        return {}


class StreamingDataset(IterableDataset):
    """Streaming dataset for continuous data loading"""
    
    def __init__(self, data_sources: List[str], batch_size: int = 32):
        self.data_sources = data_sources
        self.batch_size = batch_size
        self.current_source_idx = 0
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through data sources"""
        while True:
            for source in self.data_sources:
                yield from self._load_from_source(source)
                
    def _load_from_source(self, source: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Load data from a single source"""
        # Implementation depends on your data format
        # This is a placeholder
        for i in range(100):  # Placeholder
            yield {
                'states': torch.randn(self.batch_size, 10),
                'actions': torch.randn(self.batch_size, 3),
                'rewards': torch.randn(self.batch_size),
            }


class ParallelDataLoader:
    """Parallel data loader with memory optimization"""
    
    def __init__(self, 
                 dataset: Dataset,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 use_memory_pinning: bool = True,
                 use_shared_memory: bool = True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = min(num_workers, mp.cpu_count())
        self.prefetch_factor = prefetch_factor
        self.use_memory_pinning = use_memory_pinning
        self.use_shared_memory = use_shared_memory
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor(check_interval=5.0)
        
        # Create optimized data loader
        self.dataloader = self._create_dataloader()
        
    def _create_dataloader(self) -> DataLoader:
        """Create optimized PyTorch DataLoader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.use_memory_pinning and torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=self._optimized_collate_fn
        )
        
    def _optimized_collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Memory-optimized collate function"""
        if not batch:
            return {}
            
        # Collect all keys
        keys = batch[0].keys()
        result = {}
        
        for key in keys:
            values = [item[key] for item in batch]
            
            if isinstance(values[0], torch.Tensor):
                # Stack tensors and optimize memory
                stacked = torch.stack(values)
                result[key] = optimize_tensor_memory(stacked)
            elif isinstance(values[0], np.ndarray):
                # Convert numpy arrays to optimized tensors
                stacked = torch.from_numpy(np.stack(values))
                result[key] = optimize_tensor_memory(stacked)
            else:
                # Keep other types as is
                result[key] = values
                
        return result
        
    def __iter__(self):
        """Iterate through batches"""
        self.memory_monitor.start_monitoring()
        try:
            for batch in self.dataloader:
                yield batch
                
                # Periodic memory cleanup
                if hasattr(self, '_batch_count'):
                    self._batch_count += 1
                else:
                    self._batch_count = 1
                    
                if self._batch_count % 50 == 0:
                    clear_gpu_cache()
                    
        finally:
            self.memory_monitor.stop_monitoring()
            
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        stats = self.memory_monitor.get_memory_stats()
        return {
            'memory_percent': stats.memory_percent,
            'used_memory_gb': stats.used_memory,
            'gpu_memory_gb': stats.gpu_memory_used or 0.0
        }


class AdaptiveBatchSampler:
    """Adaptive batch sampler that adjusts batch size based on memory usage"""
    
    def __init__(self, 
                 dataset_size: int,
                 initial_batch_size: int = 32,
                 min_batch_size: int = 8,
                 max_batch_size: int = 128,
                 memory_threshold: float = 0.8):
        
        self.dataset_size = dataset_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.start_monitoring()
        
    def __iter__(self):
        """Generate batch indices with adaptive sizing"""
        indices = list(range(self.dataset_size))
        np.random.shuffle(indices)
        
        i = 0
        while i < len(indices):
            # Check memory usage and adjust batch size
            self._adjust_batch_size()
            
            # Generate batch
            batch_end = min(i + self.current_batch_size, len(indices))
            batch_indices = indices[i:batch_end]
            
            yield batch_indices
            i = batch_end
            
    def _adjust_batch_size(self):
        """Adjust batch size based on memory usage"""
        stats = self.memory_monitor.get_memory_stats()
        
        if stats.memory_percent > self.memory_threshold:
            # Reduce batch size if memory usage is high
            new_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
            if new_size != self.current_batch_size:
                logger.info(f"Reducing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
        elif stats.memory_percent < self.memory_threshold * 0.6:
            # Increase batch size if memory usage is low
            new_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
            if new_size != self.current_batch_size:
                logger.info(f"Increasing batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size


class HDF5Dataset(Dataset):
    """HDF5-based dataset for efficient large-scale data storage"""
    
    def __init__(self, hdf5_path: str, chunk_cache_size: int = 1000):
        self.hdf5_path = hdf5_path
        self.chunk_cache_size = chunk_cache_size
        self.file_handle = None
        self.cache = MemoryEfficientBuffer(chunk_cache_size)
        
        # Open file and get metadata
        with h5py.File(hdf5_path, 'r') as f:
            self.total_samples = f['states'].shape[0]
            self.state_shape = f['states'].shape[1:]
            self.action_shape = f['actions'].shape[1:]
            
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx: int):
        """Get item with caching"""
        # Check cache first
        cached_item = self._get_from_cache(idx)
        if cached_item is not None:
            return cached_item
            
        # Load from file
        if self.file_handle is None:
            self.file_handle = h5py.File(self.hdf5_path, 'r')
            
        item = {
            'state': torch.from_numpy(self.file_handle['states'][idx]).float(),
            'action': torch.from_numpy(self.file_handle['actions'][idx]).float(),
            'reward': torch.tensor(self.file_handle['rewards'][idx]).float(),
            'value': torch.tensor(self.file_handle['values'][idx]).float(),
            'log_prob': torch.tensor(self.file_handle['log_probs'][idx]).float(),
            'done': torch.tensor(self.file_handle['dones'][idx]).bool()
        }
        
        # Add to cache
        self.cache.add((idx, item))
        
        return item
        
    def _get_from_cache(self, idx: int):
        """Get item from cache"""
        recent_items = self.cache.get_recent(self.chunk_cache_size)
        for cached_idx, cached_item in recent_items:
            if cached_idx == idx:
                return cached_item
        return None
        
    def __del__(self):
        """Clean up file handle"""
        if self.file_handle is not None:
            self.file_handle.close()


class DataLoaderFactory:
    """Factory for creating optimized data loaders"""
    
    @staticmethod
    def create_memory_optimized_loader(
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = None,
        use_adaptive_batching: bool = False,
        **kwargs
    ) -> DataLoader:
        """Create memory-optimized data loader"""
        
        if num_workers is None:
            num_workers = min(4, mp.cpu_count())
            
        if use_adaptive_batching:
            # Use adaptive batch sampler
            sampler = AdaptiveBatchSampler(
                len(dataset),
                initial_batch_size=batch_size,
                **kwargs
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=True if num_workers > 0 else False
            )
        else:
            # Use parallel data loader
            return ParallelDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                **kwargs
            )
    
    @staticmethod
    def create_streaming_loader(
        data_sources: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> DataLoader:
        """Create streaming data loader"""
        dataset = StreamingDataset(data_sources, batch_size)
        return DataLoader(
            dataset,
            batch_size=None,  # Batching handled by dataset
            num_workers=0,    # Streaming datasets don't work well with multiprocessing
            **kwargs
        )
    
    @staticmethod
    def create_hdf5_loader(
        hdf5_path: str,
        batch_size: int = 32,
        num_workers: int = None,
        **kwargs
    ) -> DataLoader:
        """Create HDF5-based data loader"""
        dataset = HDF5Dataset(hdf5_path, **kwargs)
        
        if num_workers is None:
            num_workers = min(2, mp.cpu_count())  # HDF5 doesn't scale well with many workers
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
            persistent_workers=True if num_workers > 0 else False
        )


def benchmark_data_loader(loader: DataLoader, num_batches: int = 100) -> Dict[str, float]:
    """Benchmark data loader performance"""
    
    start_time = time.time()
    memory_monitor = MemoryMonitor()
    memory_monitor.start_monitoring()
    
    try:
        batch_times = []
        
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
                
            batch_start = time.time()
            
            # Simulate processing
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        _ = value.mean()
                        
            batch_times.append(time.time() - batch_start)
            
        total_time = time.time() - start_time
        memory_stats = memory_monitor.get_memory_stats()
        
        return {
            'total_time': total_time,
            'avg_batch_time': np.mean(batch_times),
            'batches_per_second': len(batch_times) / total_time,
            'peak_memory_percent': memory_stats.memory_percent,
            'peak_gpu_memory_gb': memory_stats.gpu_memory_used or 0.0
        }
        
    finally:
        memory_monitor.stop_monitoring()