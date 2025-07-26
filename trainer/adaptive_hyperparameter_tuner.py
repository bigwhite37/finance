"""
Adaptive hyperparameter tuning system for O2O RL training.
Automatically adjusts learning rates, sampling ratios, and trust region parameters.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import json
from pathlib import Path
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    adaptation_rate: float = 0.1
    patience: int = 10
    improvement_threshold: float = 0.01
    decay_factor: float = 0.95
    history: List[float] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    last_update_step: int = 0


@dataclass
class PerformanceMetrics:
    """Performance metrics for hyperparameter tuning"""
    loss: float
    reward: float
    cvar_estimate: float
    kl_divergence: float = 0.0
    memory_usage: float = 0.0
    training_time: float = 0.0
    convergence_rate: float = 0.0


class AdaptationStrategy(ABC):
    """Abstract base class for adaptation strategies"""
    
    @abstractmethod
    def adapt(self, 
              config: HyperparameterConfig, 
              metrics: PerformanceMetrics,
              step: int) -> float:
        """Adapt hyperparameter based on performance metrics"""
        pass


class GradientBasedAdaptation(AdaptationStrategy):
    """Gradient-based adaptation strategy"""
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.velocity = {}
        
    def adapt(self, 
              config: HyperparameterConfig, 
              metrics: PerformanceMetrics,
              step: int) -> float:
        
        if len(config.performance_history) < 2:
            return config.current_value
            
        # Calculate performance gradient
        recent_performance = config.performance_history[-5:]
        if len(recent_performance) >= 2:
            gradient = np.gradient(recent_performance)[-1]
        else:
            gradient = 0.0
            
        # Update velocity with momentum
        if config.name not in self.velocity:
            self.velocity[config.name] = 0.0
            
        self.velocity[config.name] = (
            self.momentum * self.velocity[config.name] + 
            config.adaptation_rate * gradient
        )
        
        # Update parameter
        new_value = config.current_value + self.velocity[config.name]
        
        # Clip to bounds
        new_value = np.clip(new_value, config.min_value, config.max_value)
        
        return new_value


class BayesianOptimizationAdaptation(AdaptationStrategy):
    """Bayesian optimization-based adaptation"""
    
    def __init__(self, exploration_weight: float = 0.1):
        self.exploration_weight = exploration_weight
        self.gaussian_process = None
        
    def adapt(self, 
              config: HyperparameterConfig, 
              metrics: PerformanceMetrics,
              step: int) -> float:
        
        # Simplified Bayesian optimization
        # In practice, you'd use a proper GP library like scikit-optimize
        
        if len(config.history) < 3:
            # Random exploration in early stages
            return np.random.uniform(config.min_value, config.max_value)
            
        # Calculate expected improvement
        best_performance = max(config.performance_history)
        
        # Simple acquisition function (Upper Confidence Bound)
        candidates = np.linspace(config.min_value, config.max_value, 20)
        scores = []
        
        for candidate in candidates:
            # Estimate mean and variance (simplified)
            distances = [abs(candidate - h) for h in config.history[-10:]]
            weights = [1.0 / (d + 1e-6) for d in distances]
            weighted_performances = [w * p for w, p in zip(weights, config.performance_history[-10:])]
            
            mean_estimate = sum(weighted_performances) / sum(weights)
            variance_estimate = np.var(config.performance_history[-10:])
            
            # UCB score
            ucb_score = mean_estimate + self.exploration_weight * np.sqrt(variance_estimate)
            scores.append(ucb_score)
            
        best_idx = np.argmax(scores)
        return candidates[best_idx]


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler based on training progress"""
    
    def __init__(self, 
                 initial_lr: float = 3e-4,
                 min_lr: float = 1e-6,
                 max_lr: float = 1e-2,
                 patience: int = 20,
                 factor: float = 0.8,
                 warmup_steps: int = 1000):
        
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        self.warmup_steps = warmup_steps
        
        self.best_performance = float('-inf')
        self.patience_counter = 0
        self.step_count = 0
        self.performance_history = deque(maxlen=100)
        
    def step(self, performance: float) -> float:
        """Update learning rate based on performance"""
        self.step_count += 1
        self.performance_history.append(performance)
        
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            warmup_lr = self.initial_lr * (self.step_count / self.warmup_steps)
            self.current_lr = min(warmup_lr, self.initial_lr)
            return self.current_lr
            
        # Check for improvement
        if performance > self.best_performance + 1e-6:
            self.best_performance = performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        # Reduce learning rate if no improvement
        if self.patience_counter >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.patience_counter = 0
            logger.info(f"Reducing learning rate to {self.current_lr:.2e}")
            
        # Increase learning rate if performance is consistently improving
        elif len(self.performance_history) >= 10:
            recent_trend = np.polyfit(range(10), list(self.performance_history)[-10:], 1)[0]
            if recent_trend > 0.01:  # Strong positive trend
                self.current_lr = min(self.current_lr * 1.1, self.max_lr)
                
        return self.current_lr


class AdaptiveSamplingRatioController:
    """Adaptive controller for O2O sampling ratio ρ(t)"""
    
    def __init__(self,
                 initial_rho: float = 0.2,
                 min_rho: float = 0.1,
                 max_rho: float = 0.9,
                 adaptation_rate: float = 0.01,
                 performance_window: int = 50):
        
        self.initial_rho = initial_rho
        self.current_rho = initial_rho
        self.min_rho = min_rho
        self.max_rho = max_rho
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window
        
        self.performance_history = deque(maxlen=performance_window)
        self.rho_history = deque(maxlen=performance_window)
        self.step_count = 0
        
    def update(self, 
               performance_metrics: PerformanceMetrics,
               kl_divergence: float = 0.0) -> float:
        """Update sampling ratio based on performance and distribution drift"""
        
        self.step_count += 1
        self.performance_history.append(performance_metrics.reward)
        self.rho_history.append(self.current_rho)
        
        # Base update rule: ρ(t) = min(1, ρ₀ + α·t)
        base_rho = min(1.0, self.initial_rho + self.adaptation_rate * self.step_count)
        
        # Adaptive adjustments
        if len(self.performance_history) >= 10:
            # Performance-based adjustment
            recent_performance = np.mean(list(self.performance_history)[-10:])
            older_performance = np.mean(list(self.performance_history)[-20:-10]) if len(self.performance_history) >= 20 else recent_performance
            
            performance_change = recent_performance - older_performance
            
            # If performance is improving, increase online ratio
            if performance_change > 0.01:
                adjustment = 0.05
            # If performance is declining, decrease online ratio
            elif performance_change < -0.01:
                adjustment = -0.05
            else:
                adjustment = 0.0
                
            # KL divergence adjustment
            if kl_divergence > 0.1:  # High distribution drift
                adjustment -= 0.1  # Reduce online ratio
            elif kl_divergence < 0.01:  # Low distribution drift
                adjustment += 0.02  # Can increase online ratio
                
            # Apply adjustment
            self.current_rho = np.clip(
                base_rho + adjustment,
                self.min_rho,
                self.max_rho
            )
        else:
            self.current_rho = base_rho
            
        return self.current_rho
    
    def get_statistics(self) -> Dict[str, float]:
        """Get sampling ratio statistics"""
        if not self.rho_history:
            return {}
            
        return {
            'current_rho': self.current_rho,
            'mean_rho': np.mean(self.rho_history),
            'std_rho': np.std(self.rho_history),
            'min_rho': np.min(self.rho_history),
            'max_rho': np.max(self.rho_history)
        }


class AdaptiveTrustRegionController:
    """Adaptive controller for trust region parameter β"""
    
    def __init__(self,
                 initial_beta: float = 1.0,
                 min_beta: float = 0.1,
                 max_beta: float = 10.0,
                 target_kl: float = 0.01,
                 kl_tolerance: float = 0.005):
        
        self.initial_beta = initial_beta
        self.current_beta = initial_beta
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance
        
        self.kl_history = deque(maxlen=50)
        self.beta_history = deque(maxlen=50)
        
    def update(self, kl_divergence: float) -> float:
        """Update trust region parameter based on KL divergence"""
        
        self.kl_history.append(kl_divergence)
        self.beta_history.append(self.current_beta)
        
        # Adaptive beta update
        if kl_divergence > self.target_kl + self.kl_tolerance:
            # KL too high, increase constraint (increase beta)
            self.current_beta = min(self.current_beta * 1.5, self.max_beta)
            logger.debug(f"Increasing trust region beta to {self.current_beta:.3f} (KL: {kl_divergence:.4f})")
            
        elif kl_divergence < self.target_kl - self.kl_tolerance:
            # KL too low, relax constraint (decrease beta)
            self.current_beta = max(self.current_beta * 0.8, self.min_beta)
            logger.debug(f"Decreasing trust region beta to {self.current_beta:.3f} (KL: {kl_divergence:.4f})")
            
        return self.current_beta
    
    def get_statistics(self) -> Dict[str, float]:
        """Get trust region statistics"""
        if not self.beta_history:
            return {}
            
        return {
            'current_beta': self.current_beta,
            'mean_beta': np.mean(self.beta_history),
            'mean_kl': np.mean(self.kl_history) if self.kl_history else 0.0,
            'target_kl': self.target_kl
        }


class HyperparameterSearcher:
    """Hyperparameter search tool for finding optimal configurations"""
    
    def __init__(self, 
                 search_space: Dict[str, Tuple[float, float]],
                 n_trials: int = 100,
                 n_random_trials: int = 20):
        
        self.search_space = search_space
        self.n_trials = n_trials
        self.n_random_trials = n_random_trials
        
        self.trial_history = []
        self.best_config = None
        self.best_performance = float('-inf')
        
    def suggest_config(self, trial_idx: int) -> Dict[str, float]:
        """Suggest next configuration to try"""
        
        if trial_idx < self.n_random_trials:
            # Random search for initial trials
            config = {}
            for param, (min_val, max_val) in self.search_space.items():
                config[param] = np.random.uniform(min_val, max_val)
            return config
        else:
            # Bayesian optimization for later trials
            return self._bayesian_suggest()
            
    def _bayesian_suggest(self) -> Dict[str, float]:
        """Bayesian optimization suggestion (simplified)"""
        
        if not self.trial_history:
            # Fallback to random
            config = {}
            for param, (min_val, max_val) in self.search_space.items():
                config[param] = np.random.uniform(min_val, max_val)
            return config
            
        # Simple acquisition function based on historical performance
        best_configs = sorted(self.trial_history, key=lambda x: x['performance'], reverse=True)[:5]
        
        # Generate candidates around best configurations
        config = {}
        for param, (min_val, max_val) in self.search_space.items():
            # Get values from best configs
            best_values = [c['config'][param] for c in best_configs]
            
            # Add noise around best values
            if best_values:
                base_value = np.mean(best_values)
                noise_scale = (max_val - min_val) * 0.1
                new_value = base_value + np.random.normal(0, noise_scale)
                config[param] = np.clip(new_value, min_val, max_val)
            else:
                config[param] = np.random.uniform(min_val, max_val)
                
        return config
    
    def report_result(self, config: Dict[str, float], performance: float):
        """Report result of a trial"""
        
        trial_result = {
            'config': config.copy(),
            'performance': performance,
            'timestamp': time.time()
        }
        
        self.trial_history.append(trial_result)
        
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_config = config.copy()
            logger.info(f"New best configuration found: {config}, performance: {performance:.4f}")
            
    def get_best_config(self) -> Optional[Dict[str, float]]:
        """Get best configuration found so far"""
        return self.best_config
    
    def save_results(self, filepath: str):
        """Save search results to file"""
        results = {
            'search_space': self.search_space,
            'trial_history': self.trial_history,
            'best_config': self.best_config,
            'best_performance': self.best_performance
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
    def load_results(self, filepath: str):
        """Load search results from file"""
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        self.search_space = results['search_space']
        self.trial_history = results['trial_history']
        self.best_config = results['best_config']
        self.best_performance = results['best_performance']


class AdaptiveHyperparameterTuner:
    """Main adaptive hyperparameter tuning system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize adaptive controllers
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            initial_lr=config.get('initial_lr', 3e-4),
            min_lr=config.get('min_lr', 1e-6),
            max_lr=config.get('max_lr', 1e-2)
        )
        
        self.rho_controller = AdaptiveSamplingRatioController(
            initial_rho=config.get('initial_rho', 0.2),
            min_rho=config.get('min_rho', 0.1),
            max_rho=config.get('max_rho', 0.9)
        )
        
        self.trust_region_controller = AdaptiveTrustRegionController(
            initial_beta=config.get('initial_beta', 1.0),
            target_kl=config.get('target_kl', 0.01)
        )
        
        # Hyperparameter searcher for global optimization
        if config.get('enable_hyperparameter_search', False):
            search_space = config.get('search_space', {
                'learning_rate': (1e-5, 1e-2),
                'cvar_lambda': (0.1, 5.0),
                'clip_epsilon': (0.1, 0.3),
                'gamma': (0.95, 0.999)
            })
            self.searcher = HyperparameterSearcher(search_space)
        else:
            self.searcher = None
            
        self.step_count = 0
        self.adaptation_history = []
        
    def update(self, 
               performance_metrics: PerformanceMetrics,
               kl_divergence: float = 0.0) -> Dict[str, float]:
        """Update all adaptive hyperparameters"""
        
        self.step_count += 1
        
        # Update learning rate
        new_lr = self.lr_scheduler.step(performance_metrics.reward)
        
        # Update sampling ratio
        new_rho = self.rho_controller.update(performance_metrics, kl_divergence)
        
        # Update trust region parameter
        new_beta = self.trust_region_controller.update(kl_divergence)
        
        # Collect updated parameters
        updated_params = {
            'learning_rate': new_lr,
            'sampling_ratio': new_rho,
            'trust_region_beta': new_beta,
            'step': self.step_count
        }
        
        # Record adaptation history
        adaptation_record = {
            'step': self.step_count,
            'performance': performance_metrics.reward,
            'kl_divergence': kl_divergence,
            'parameters': updated_params.copy()
        }
        self.adaptation_history.append(adaptation_record)
        
        # Log significant changes
        if self.step_count % 100 == 0:
            logger.info(f"Adaptive hyperparameters at step {self.step_count}: "
                       f"LR={new_lr:.2e}, ρ={new_rho:.3f}, β={new_beta:.3f}")
            
        return updated_params
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tuning statistics"""
        stats = {
            'step_count': self.step_count,
            'learning_rate_stats': {
                'current': self.lr_scheduler.current_lr,
                'best_performance': self.lr_scheduler.best_performance
            },
            'sampling_ratio_stats': self.rho_controller.get_statistics(),
            'trust_region_stats': self.trust_region_controller.get_statistics()
        }
        
        if self.searcher:
            stats['search_stats'] = {
                'n_trials': len(self.searcher.trial_history),
                'best_performance': self.searcher.best_performance,
                'best_config': self.searcher.best_config
            }
            
        return stats
    
    def save_adaptation_history(self, filepath: str):
        """Save adaptation history to file"""
        with open(filepath, 'w') as f:
            json.dump(self.adaptation_history, f, indent=2)
            
    def suggest_hyperparameter_config(self) -> Optional[Dict[str, float]]:
        """Suggest hyperparameter configuration for global search"""
        if self.searcher:
            trial_idx = len(self.searcher.trial_history)
            return self.searcher.suggest_config(trial_idx)
        return None
    
    def report_hyperparameter_result(self, config: Dict[str, float], performance: float):
        """Report result of hyperparameter configuration"""
        if self.searcher:
            self.searcher.report_result(config, performance)