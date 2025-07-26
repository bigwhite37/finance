"""
Integration tests for optimization and performance tuning components.
"""

import pytest
import numpy as np
import torch
from typing import Dict

from trainer.adaptive_hyperparameter_tuner import (
    AdaptiveHyperparameterTuner, PerformanceMetrics
)
from utils.memory_optimizer import MemoryMonitor, GradientAccumulator


def test_memory_monitor_basic():
    """Test basic memory monitoring functionality"""
    monitor = MemoryMonitor(check_interval=0.1)
    
    # Test getting memory stats
    stats = monitor.get_memory_stats()
    assert stats.total_memory > 0
    assert stats.used_memory > 0
    assert 0 <= stats.memory_percent <= 100
    
    # Test monitoring start/stop
    monitor.start_monitoring()
    assert monitor.monitoring is True
    
    monitor.stop_monitoring()
    assert monitor.monitoring is False


def test_gradient_accumulator():
    """Test gradient accumulation functionality"""
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    accumulator = GradientAccumulator(model, accumulation_steps=4)
    
    assert accumulator.accumulation_steps == 4
    assert accumulator.accumulated_steps == 0
    
    # Test accumulation context
    with accumulator.accumulate():
        pass
        
    assert accumulator.accumulated_steps == 1
    assert not accumulator.should_update()
    
    # Accumulate more steps
    for _ in range(3):
        with accumulator.accumulate():
            pass
            
    assert accumulator.should_update()


def test_adaptive_hyperparameter_tuner_integration():
    """Test adaptive hyperparameter tuner integration"""
    config = {
        'initial_lr': 3e-4,
        'min_lr': 1e-6,
        'max_lr': 1e-2,
        'initial_rho': 0.2,
        'min_rho': 0.1,
        'max_rho': 0.9,
        'initial_beta': 1.0,
        'target_kl': 0.01
    }
    
    tuner = AdaptiveHyperparameterTuner(config)
    
    # Test initial state
    assert tuner.step_count == 0
    assert len(tuner.adaptation_history) == 0
    
    # Test update
    metrics = PerformanceMetrics(
        loss=0.5,
        reward=0.6,
        cvar_estimate=-0.02,
        kl_divergence=0.015
    )
    
    updated_params = tuner.update(metrics, kl_divergence=0.015)
    
    # Check that parameters are updated
    assert 'learning_rate' in updated_params
    assert 'sampling_ratio' in updated_params
    assert 'trust_region_beta' in updated_params
    
    # Check bounds (allow some tolerance for adaptive adjustments)
    assert updated_params['learning_rate'] > 0
    assert updated_params['learning_rate'] <= config['max_lr']
    assert config['min_rho'] <= updated_params['sampling_ratio'] <= config['max_rho']
    
    # Check state updates
    assert tuner.step_count == 1
    assert len(tuner.adaptation_history) == 1


def test_performance_optimization_workflow():
    """Test complete performance optimization workflow"""
    # Create memory monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.start_monitoring()
    
    try:
        # Create adaptive tuner
        tuner_config = {
            'initial_lr': 1e-3,
            'initial_rho': 0.3,
            'initial_beta': 1.0,
            'target_kl': 0.01
        }
        
        tuner = AdaptiveHyperparameterTuner(tuner_config)
        
        # Simulate training loop with optimization
        for step in range(10):
            # Simulate performance metrics
            performance = 0.5 + step * 0.02 + np.random.normal(0, 0.01)
            kl_div = 0.01 + np.random.normal(0, 0.005)
            
            metrics = PerformanceMetrics(
                loss=1.0 - performance,
                reward=performance,
                cvar_estimate=-0.02,
                kl_divergence=kl_div
            )
            
            # Update hyperparameters
            updated_params = tuner.update(metrics, kl_div)
            
            # Check memory usage
            memory_stats = memory_monitor.get_memory_stats()
            assert memory_stats.memory_percent > 0
            
            # Verify parameter updates
            assert all(key in updated_params for key in 
                      ['learning_rate', 'sampling_ratio', 'trust_region_beta'])
        
        # Get final statistics
        stats = tuner.get_statistics()
        assert stats['step_count'] == 10
        assert 'learning_rate_stats' in stats
        assert 'sampling_ratio_stats' in stats
        assert 'trust_region_stats' in stats
        
    finally:
        memory_monitor.stop_monitoring()


def test_memory_optimization_with_tensors():
    """Test memory optimization with actual tensors"""
    from utils.memory_optimizer import optimize_tensor_memory, clear_gpu_cache
    
    # Create test tensors
    tensor_fp32 = torch.randn(1000, 1000, dtype=torch.float32)
    
    # Optimize memory
    tensor_fp16 = optimize_tensor_memory(tensor_fp32, dtype=torch.float16)
    
    # Check that optimization worked
    assert tensor_fp16.dtype == torch.float16
    assert tensor_fp16.shape == tensor_fp32.shape
    
    # Test GPU cache clearing (should not raise error)
    clear_gpu_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])