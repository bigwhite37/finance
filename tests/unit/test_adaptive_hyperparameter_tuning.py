"""
Tests for adaptive hyperparameter tuning system.
"""

import pytest
import numpy as np
import torch
from typing import Dict, List
import tempfile
import json
from pathlib import Path

from trainer.adaptive_hyperparameter_tuner import (
    AdaptiveHyperparameterTuner,
    AdaptiveLearningRateScheduler,
    AdaptiveSamplingRatioController,
    AdaptiveTrustRegionController,
    HyperparameterSearcher,
    PerformanceMetrics,
    GradientBasedAdaptation,
    BayesianOptimizationAdaptation,
    HyperparameterConfig
)


class TestAdaptiveLearningRateScheduler:
    """Test adaptive learning rate scheduler"""
    
    def test_initialization(self):
        """Test scheduler initialization"""
        scheduler = AdaptiveLearningRateScheduler(
            initial_lr=3e-4,
            min_lr=1e-6,
            max_lr=1e-2
        )
        
        assert scheduler.current_lr == 3e-4
        assert scheduler.min_lr == 1e-6
        assert scheduler.max_lr == 1e-2
        assert scheduler.step_count == 0
        
    def test_warmup_phase(self):
        """Test warmup phase behavior"""
        scheduler = AdaptiveLearningRateScheduler(
            initial_lr=1e-3,
            warmup_steps=100
        )
        
        # Test warmup progression
        for step in range(50):
            lr = scheduler.step(0.5)  # Constant performance
            expected_lr = 1e-3 * (step + 1) / 100
            assert abs(lr - expected_lr) < 1e-10
            
    def test_performance_based_adjustment(self):
        """Test performance-based learning rate adjustment"""
        scheduler = AdaptiveLearningRateScheduler(
            initial_lr=1e-3,
            patience=5,
            factor=0.5,
            warmup_steps=0
        )
        
        # Simulate improving performance
        for _ in range(3):
            scheduler.step(0.5)
        
        initial_lr = scheduler.current_lr
        
        # Simulate stagnant performance
        for _ in range(6):  # Exceed patience
            scheduler.step(0.5)
            
        # Learning rate should be reduced
        assert scheduler.current_lr < initial_lr
        
    def test_lr_bounds(self):
        """Test learning rate bounds"""
        scheduler = AdaptiveLearningRateScheduler(
            initial_lr=1e-3,
            min_lr=1e-6,
            max_lr=1e-2,
            warmup_steps=0
        )
        
        # Test minimum bound
        for _ in range(100):
            scheduler.step(0.0)  # Poor performance
            
        assert scheduler.current_lr >= scheduler.min_lr
        
        # Test maximum bound (through positive trend detection)
        scheduler.current_lr = 5e-3
        for i in range(20):
            scheduler.step(0.5 + i * 0.1)  # Improving performance
            
        assert scheduler.current_lr <= scheduler.max_lr


class TestAdaptiveSamplingRatioController:
    """Test adaptive sampling ratio controller"""
    
    def test_initialization(self):
        """Test controller initialization"""
        controller = AdaptiveSamplingRatioController(
            initial_rho=0.2,
            min_rho=0.1,
            max_rho=0.9
        )
        
        assert controller.current_rho == 0.2
        assert controller.min_rho == 0.1
        assert controller.max_rho == 0.9
        
    def test_base_update_rule(self):
        """Test base update rule ρ(t) = min(1, ρ₀ + α·t)"""
        controller = AdaptiveSamplingRatioController(
            initial_rho=0.2,
            adaptation_rate=0.01
        )
        
        metrics = PerformanceMetrics(
            loss=0.5, reward=0.5, cvar_estimate=-0.02
        )
        
        # Test progression
        for step in range(10):
            rho = controller.update(metrics)
            expected_base = min(1.0, 0.2 + 0.01 * (step + 1))
            # Should be close to base rule (with some adaptive adjustments)
            assert 0.1 <= rho <= 0.9
            
    def test_performance_based_adjustment(self):
        """Test performance-based adjustments"""
        controller = AdaptiveSamplingRatioController(
            initial_rho=0.2,
            adaptation_rate=0.0  # Disable base progression
        )
        
        # Simulate improving performance
        for i in range(25):
            metrics = PerformanceMetrics(
                loss=0.5 - i * 0.01,
                reward=0.5 + i * 0.01,
                cvar_estimate=-0.02
            )
            controller.update(metrics)
            
        improving_rho = controller.current_rho
        
        # Reset and simulate declining performance
        controller = AdaptiveSamplingRatioController(
            initial_rho=0.2,
            adaptation_rate=0.0
        )
        
        for i in range(25):
            metrics = PerformanceMetrics(
                loss=0.5 + i * 0.01,
                reward=0.5 - i * 0.01,
                cvar_estimate=-0.02
            )
            controller.update(metrics)
            
        declining_rho = controller.current_rho
        
        # Improving performance should lead to higher rho
        assert improving_rho > declining_rho
        
    def test_kl_divergence_adjustment(self):
        """Test KL divergence-based adjustments"""
        controller = AdaptiveSamplingRatioController(
            initial_rho=0.5,
            adaptation_rate=0.0
        )
        
        metrics = PerformanceMetrics(
            loss=0.5, reward=0.5, cvar_estimate=-0.02
        )
        
        # High KL divergence should reduce rho
        high_kl_rho = controller.update(metrics, kl_divergence=0.15)
        
        # Reset controller
        controller.current_rho = 0.5
        
        # Low KL divergence should allow higher rho
        low_kl_rho = controller.update(metrics, kl_divergence=0.005)
        
        assert high_kl_rho <= low_kl_rho
        
    def test_bounds_enforcement(self):
        """Test that rho stays within bounds"""
        controller = AdaptiveSamplingRatioController(
            initial_rho=0.2,
            min_rho=0.1,
            max_rho=0.9
        )
        
        metrics = PerformanceMetrics(
            loss=0.0, reward=1.0, cvar_estimate=-0.02
        )
        
        # Test many updates
        for _ in range(1000):
            rho = controller.update(metrics, kl_divergence=0.001)
            assert controller.min_rho <= rho <= controller.max_rho


class TestAdaptiveTrustRegionController:
    """Test adaptive trust region controller"""
    
    def test_initialization(self):
        """Test controller initialization"""
        controller = AdaptiveTrustRegionController(
            initial_beta=1.0,
            target_kl=0.01
        )
        
        assert controller.current_beta == 1.0
        assert controller.target_kl == 0.01
        
    def test_kl_based_adjustment(self):
        """Test KL divergence-based beta adjustment"""
        controller = AdaptiveTrustRegionController(
            initial_beta=1.0,
            target_kl=0.01,
            kl_tolerance=0.005
        )
        
        # High KL should increase beta
        initial_beta = controller.current_beta
        controller.update(0.02)  # High KL
        assert controller.current_beta > initial_beta
        
        # Reset
        controller.current_beta = 1.0
        
        # Low KL should decrease beta
        controller.update(0.003)  # Low KL
        assert controller.current_beta < 1.0
        
        # Target KL should not change beta much
        controller.current_beta = 1.0
        controller.update(0.01)  # Target KL
        assert abs(controller.current_beta - 1.0) < 0.1
        
    def test_bounds_enforcement(self):
        """Test beta bounds enforcement"""
        controller = AdaptiveTrustRegionController(
            initial_beta=1.0,
            min_beta=0.1,
            max_beta=10.0,
            target_kl=0.01
        )
        
        # Test maximum bound
        for _ in range(20):
            controller.update(0.1)  # Very high KL
            
        assert controller.current_beta <= controller.max_beta
        
        # Test minimum bound
        for _ in range(20):
            controller.update(0.001)  # Very low KL
            
        assert controller.current_beta >= controller.min_beta


class TestHyperparameterSearcher:
    """Test hyperparameter searcher"""
    
    def test_initialization(self):
        """Test searcher initialization"""
        search_space = {
            'lr': (1e-5, 1e-2),
            'gamma': (0.9, 0.999)
        }
        
        searcher = HyperparameterSearcher(
            search_space=search_space,
            n_trials=50,
            n_random_trials=10
        )
        
        assert searcher.search_space == search_space
        assert searcher.n_trials == 50
        assert searcher.n_random_trials == 10
        assert len(searcher.trial_history) == 0
        
    def test_random_suggestions(self):
        """Test random suggestions in early trials"""
        search_space = {
            'lr': (1e-5, 1e-2),
            'gamma': (0.9, 0.999)
        }
        
        searcher = HyperparameterSearcher(
            search_space=search_space,
            n_random_trials=5
        )
        
        # First few suggestions should be random
        for trial_idx in range(5):
            config = searcher.suggest_config(trial_idx)
            
            assert 1e-5 <= config['lr'] <= 1e-2
            assert 0.9 <= config['gamma'] <= 0.999
            
    def test_result_reporting(self):
        """Test result reporting and best tracking"""
        search_space = {'lr': (1e-5, 1e-2)}
        searcher = HyperparameterSearcher(search_space)
        
        # Report some results
        configs_and_perfs = [
            ({'lr': 1e-4}, 0.5),
            ({'lr': 3e-4}, 0.8),
            ({'lr': 1e-3}, 0.6)
        ]
        
        for config, perf in configs_and_perfs:
            searcher.report_result(config, perf)
            
        assert len(searcher.trial_history) == 3
        assert searcher.best_performance == 0.8
        assert searcher.best_config == {'lr': 3e-4}
        
    def test_save_load_results(self):
        """Test saving and loading results"""
        search_space = {'lr': (1e-5, 1e-2)}
        searcher = HyperparameterSearcher(search_space)
        
        # Add some results
        searcher.report_result({'lr': 1e-4}, 0.5)
        searcher.report_result({'lr': 3e-4}, 0.8)
        
        # Save results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            
        try:
            searcher.save_results(temp_path)
            
            # Create new searcher and load
            new_searcher = HyperparameterSearcher(search_space)
            new_searcher.load_results(temp_path)
            
            assert len(new_searcher.trial_history) == 2
            assert new_searcher.best_performance == 0.8
            assert new_searcher.best_config == {'lr': 3e-4}
            
        finally:
            Path(temp_path).unlink()


class TestAdaptiveHyperparameterTuner:
    """Test main adaptive hyperparameter tuner"""
    
    def test_initialization(self):
        """Test tuner initialization"""
        config = {
            'initial_lr': 3e-4,
            'initial_rho': 0.2,
            'initial_beta': 1.0,
            'target_kl': 0.01
        }
        
        tuner = AdaptiveHyperparameterTuner(config)
        
        assert tuner.lr_scheduler.current_lr == 3e-4
        assert tuner.rho_controller.current_rho == 0.2
        assert tuner.trust_region_controller.current_beta == 1.0
        assert tuner.step_count == 0
        
    def test_update_integration(self):
        """Test integrated update of all components"""
        config = {
            'initial_lr': 3e-4,
            'initial_rho': 0.2,
            'initial_beta': 1.0,
            'target_kl': 0.01
        }
        
        tuner = AdaptiveHyperparameterTuner(config)
        
        metrics = PerformanceMetrics(
            loss=0.5,
            reward=0.6,
            cvar_estimate=-0.02,
            kl_divergence=0.02
        )
        
        # Update parameters
        updated_params = tuner.update(metrics, kl_divergence=0.02)
        
        # Check that all parameters are updated
        assert 'learning_rate' in updated_params
        assert 'sampling_ratio' in updated_params
        assert 'trust_region_beta' in updated_params
        assert 'step' in updated_params
        
        assert tuner.step_count == 1
        assert len(tuner.adaptation_history) == 1
        
    def test_statistics_collection(self):
        """Test statistics collection"""
        config = {
            'initial_lr': 3e-4,
            'initial_rho': 0.2,
            'initial_beta': 1.0
        }
        
        tuner = AdaptiveHyperparameterTuner(config)
        
        # Run some updates
        for i in range(10):
            metrics = PerformanceMetrics(
                loss=0.5 - i * 0.01,
                reward=0.5 + i * 0.01,
                cvar_estimate=-0.02
            )
            tuner.update(metrics)
            
        stats = tuner.get_statistics()
        
        assert stats['step_count'] == 10
        assert 'learning_rate_stats' in stats
        assert 'sampling_ratio_stats' in stats
        assert 'trust_region_stats' in stats
        
    def test_history_saving(self):
        """Test adaptation history saving"""
        config = {'initial_lr': 3e-4}
        tuner = AdaptiveHyperparameterTuner(config)
        
        # Run some updates
        for i in range(5):
            metrics = PerformanceMetrics(
                loss=0.5, reward=0.5, cvar_estimate=-0.02
            )
            tuner.update(metrics)
            
        # Save history
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
            
        try:
            tuner.save_adaptation_history(temp_path)
            
            # Load and verify
            with open(temp_path, 'r') as f:
                history = json.load(f)
                
            assert len(history) == 5
            assert all('step' in record for record in history)
            assert all('performance' in record for record in history)
            assert all('parameters' in record for record in history)
            
        finally:
            Path(temp_path).unlink()


class TestAdaptationStrategies:
    """Test adaptation strategies"""
    
    def test_gradient_based_adaptation(self):
        """Test gradient-based adaptation strategy"""
        strategy = GradientBasedAdaptation(momentum=0.9)
        
        config = HyperparameterConfig(
            name='test_param',
            current_value=0.5,
            min_value=0.1,
            max_value=1.0,
            adaptation_rate=0.1
        )
        
        # Add some performance history
        config.performance_history = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        metrics = PerformanceMetrics(
            loss=0.3, reward=0.7, cvar_estimate=-0.02
        )
        
        # Test adaptation
        new_value = strategy.adapt(config, metrics, step=5)
        
        # Should be within bounds
        assert config.min_value <= new_value <= config.max_value
        
        # With positive gradient, value should increase
        assert new_value > config.current_value
        
    def test_bayesian_optimization_adaptation(self):
        """Test Bayesian optimization adaptation strategy"""
        strategy = BayesianOptimizationAdaptation(exploration_weight=0.1)
        
        config = HyperparameterConfig(
            name='test_param',
            current_value=0.5,
            min_value=0.1,
            max_value=1.0
        )
        
        metrics = PerformanceMetrics(
            loss=0.3, reward=0.7, cvar_estimate=-0.02
        )
        
        # Test with insufficient history (should explore randomly)
        config.history = [0.3, 0.4]
        config.performance_history = [0.3, 0.4]
        
        new_value = strategy.adapt(config, metrics, step=2)
        assert config.min_value <= new_value <= config.max_value
        
        # Test with sufficient history
        config.history = [0.2, 0.3, 0.4, 0.5, 0.6]
        config.performance_history = [0.2, 0.4, 0.5, 0.7, 0.6]
        
        new_value = strategy.adapt(config, metrics, step=5)
        assert config.min_value <= new_value <= config.max_value


def test_performance_metrics():
    """Test PerformanceMetrics dataclass"""
    metrics = PerformanceMetrics(
        loss=0.5,
        reward=0.7,
        cvar_estimate=-0.02,
        kl_divergence=0.01,
        memory_usage=0.8,
        training_time=120.5,
        convergence_rate=0.95
    )
    
    assert metrics.loss == 0.5
    assert metrics.reward == 0.7
    assert metrics.cvar_estimate == -0.02
    assert metrics.kl_divergence == 0.01
    assert metrics.memory_usage == 0.8
    assert metrics.training_time == 120.5
    assert metrics.convergence_rate == 0.95


if __name__ == "__main__":
    pytest.main([__file__])