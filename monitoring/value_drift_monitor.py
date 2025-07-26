"""
Value Drift Monitor for O2O RL System.

This module implements distribution drift detection for monitoring changes
between offline and online Q-value distributions, Sharpe ratio degradation,
and CVaR threshold breaches.
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    """Data structure for drift detection events."""
    timestamp: pd.Timestamp
    event_type: str  # 'kl_divergence', 'sharpe_drop', 'cvar_breach'
    severity: str    # 'low', 'medium', 'high'
    value: float
    threshold: float
    description: str
    metadata: Dict[str, Any]


class ValueDriftMonitor:
    """
    价值分布漂移监控器
    
    Monitors distribution drift between offline and online Q-values,
    tracks performance degradation through Sharpe ratio monitoring,
    and detects CVaR threshold breaches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Value Drift Monitor.
        
        Args:
            config: Configuration dictionary containing:
                - kl_threshold: KL divergence threshold (default: 0.1)
                - sharpe_drop_threshold: Sharpe ratio drop threshold (default: 0.2)
                - cvar_breach_threshold: CVaR breach threshold (default: -0.02)
                - sharpe_window: Window size for Sharpe ratio calculation (default: 30)
                - cvar_window: Window size for CVaR monitoring (default: 3)
                - min_samples: Minimum samples for distribution comparison (default: 100)
        """
        self.config = config
        
        # Thresholds
        self.kl_threshold = config.get('kl_threshold', 0.1)
        self.sharpe_drop_threshold = config.get('sharpe_drop_threshold', 0.2)
        self.cvar_breach_threshold = config.get('cvar_breach_threshold', -0.02)
        
        # Window sizes
        self.sharpe_window = config.get('sharpe_window', 30)
        self.cvar_window = config.get('cvar_window', 3)
        self.min_samples = config.get('min_samples', 100)
        
        # Data storage
        self.offline_q_values = []
        self.online_q_values = []
        self.sharpe_history = deque(maxlen=self.sharpe_window)
        self.cvar_history = deque(maxlen=self.cvar_window)
        self.returns_history = deque(maxlen=self.sharpe_window)
        
        # Event tracking
        self.drift_events = []
        self.last_kl_divergence = None
        self.baseline_sharpe = None
        
        # Statistics
        self.stats = {
            'total_drift_events': 0,
            'kl_events': 0,
            'sharpe_events': 0,
            'cvar_events': 0,
            'last_update': None
        }
        
        logger.info(f"ValueDriftMonitor initialized with thresholds: "
                   f"KL={self.kl_threshold}, Sharpe={self.sharpe_drop_threshold}, "
                   f"CVaR={self.cvar_breach_threshold}")
    
    def update_offline_values(self, q_values: np.ndarray) -> None:
        """
        Update offline Q-value distribution.
        
        Args:
            q_values: Array of Q-values from offline training
        """
        if not isinstance(q_values, np.ndarray):
            q_values = np.array(q_values)
            
        # Remove invalid values
        valid_mask = np.isfinite(q_values)
        if not np.any(valid_mask):
            logger.warning("All offline Q-values are invalid (NaN/Inf)")
            return
            
        valid_q_values = q_values[valid_mask]
        self.offline_q_values.extend(valid_q_values.flatten())
        
        # Keep only recent values to prevent memory issues
        max_samples = self.config.get('max_offline_samples', 10000)
        if len(self.offline_q_values) > max_samples:
            self.offline_q_values = self.offline_q_values[-max_samples:]
            
        logger.debug(f"Updated offline Q-values: {len(self.offline_q_values)} total samples")
    
    def update_online_values(self, q_values: np.ndarray) -> None:
        """
        Update online Q-value distribution.
        
        Args:
            q_values: Array of Q-values from online learning
        """
        if not isinstance(q_values, np.ndarray):
            q_values = np.array(q_values)
            
        # Remove invalid values
        valid_mask = np.isfinite(q_values)
        if not np.any(valid_mask):
            logger.warning("All online Q-values are invalid (NaN/Inf)")
            return
            
        valid_q_values = q_values[valid_mask]
        self.online_q_values.extend(valid_q_values.flatten())
        
        # Keep only recent values
        max_samples = self.config.get('max_online_samples', 5000)
        if len(self.online_q_values) > max_samples:
            self.online_q_values = self.online_q_values[-max_samples:]
            
        logger.debug(f"Updated online Q-values: {len(self.online_q_values)} total samples")
    
    def update_performance_metrics(self, returns: float, portfolio_value: float) -> None:
        """
        Update performance metrics for Sharpe ratio and CVaR monitoring.
        
        Args:
            returns: Daily returns
            portfolio_value: Current portfolio value
        """
        if not np.isfinite(returns):
            logger.warning(f"Invalid return value: {returns}")
            return
            
        self.returns_history.append(returns)
        
        # Calculate current Sharpe ratio if we have enough data
        if len(self.returns_history) >= 5:  # Minimum for meaningful calculation
            current_sharpe = self._calculate_sharpe_ratio(list(self.returns_history))
            self.sharpe_history.append(current_sharpe)
            
            # Set baseline Sharpe if not set
            if self.baseline_sharpe is None and len(self.sharpe_history) >= 10:
                self.baseline_sharpe = np.mean(list(self.sharpe_history)[:10])
                logger.info(f"Baseline Sharpe ratio set to: {self.baseline_sharpe:.4f}")
        
        # Calculate CVaR (5% worst-case returns)
        if len(self.returns_history) >= 20:  # Need sufficient data for CVaR
            current_cvar = self._calculate_cvar(list(self.returns_history), alpha=0.05)
            self.cvar_history.append(current_cvar)
        
        self.stats['last_update'] = pd.Timestamp.now()
    
    def calculate_kl_divergence(self) -> Optional[float]:
        """
        Calculate KL divergence between offline and online Q-value distributions.
        
        Returns:
            KL divergence value or None if insufficient data
        """
        if (len(self.offline_q_values) < self.min_samples or 
            len(self.online_q_values) < self.min_samples):
            logger.debug("Insufficient samples for KL divergence calculation")
            return None
        
        try:
            # Convert to numpy arrays
            offline_vals = np.array(self.offline_q_values)
            online_vals = np.array(self.online_q_values)
            
            # Create histograms with same bins
            min_val = min(offline_vals.min(), online_vals.min())
            max_val = max(offline_vals.max(), online_vals.max())
            
            # Handle edge case where all values are the same
            if np.isclose(min_val, max_val):
                return 0.0
            
            bins = np.linspace(min_val, max_val, 50)
            
            # Calculate probability distributions
            offline_hist, _ = np.histogram(offline_vals, bins=bins, density=True)
            online_hist, _ = np.histogram(online_vals, bins=bins, density=True)
            
            # Normalize to probabilities
            offline_prob = offline_hist / (offline_hist.sum() + 1e-10)
            online_prob = online_hist / (online_hist.sum() + 1e-10)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            offline_prob = offline_prob + epsilon
            online_prob = online_prob + epsilon
            
            # Calculate KL divergence: KL(online || offline)
            kl_div = np.sum(online_prob * np.log(online_prob / offline_prob))
            
            self.last_kl_divergence = kl_div
            logger.debug(f"KL divergence calculated: {kl_div:.6f}")
            
            return kl_div
            
        except Exception as e:
            logger.error(f"Error calculating KL divergence: {e}")
            return None
    
    def check_drift_conditions(self) -> Tuple[bool, List[DriftEvent]]:
        """
        Check all drift conditions and return detected events.
        
        Returns:
            Tuple of (drift_detected, list_of_events)
        """
        events = []
        drift_detected = False
        
        # Check KL divergence
        kl_div = self.calculate_kl_divergence()
        if kl_div is not None and kl_div > self.kl_threshold:
            event = DriftEvent(
                timestamp=pd.Timestamp.now(),
                event_type='kl_divergence',
                severity='high' if kl_div > 2 * self.kl_threshold else 'medium',
                value=kl_div,
                threshold=self.kl_threshold,
                description=f"KL divergence {kl_div:.4f} exceeds threshold {self.kl_threshold}",
                metadata={'offline_samples': len(self.offline_q_values), 
                         'online_samples': len(self.online_q_values)}
            )
            events.append(event)
            drift_detected = True
            self.stats['kl_events'] += 1
            logger.warning(f"KL divergence drift detected: {kl_div:.4f} > {self.kl_threshold}")
        
        # Check Sharpe ratio drop
        if self.baseline_sharpe is not None and len(self.sharpe_history) >= 5:
            recent_sharpe = np.mean(list(self.sharpe_history)[-5:])  # Last 5 days average
            sharpe_drop = (self.baseline_sharpe - recent_sharpe) / abs(self.baseline_sharpe)
            
            if sharpe_drop > self.sharpe_drop_threshold:
                event = DriftEvent(
                    timestamp=pd.Timestamp.now(),
                    event_type='sharpe_drop',
                    severity='high' if sharpe_drop > 2 * self.sharpe_drop_threshold else 'medium',
                    value=sharpe_drop,
                    threshold=self.sharpe_drop_threshold,
                    description=f"Sharpe ratio dropped {sharpe_drop:.2%} from baseline",
                    metadata={'baseline_sharpe': self.baseline_sharpe,
                             'recent_sharpe': recent_sharpe}
                )
                events.append(event)
                drift_detected = True
                self.stats['sharpe_events'] += 1
                logger.warning(f"Sharpe ratio drift detected: {sharpe_drop:.2%} drop")
        
        # Check CVaR breach (consecutive days)
        if len(self.cvar_history) >= self.cvar_window:
            recent_cvars = list(self.cvar_history)[-self.cvar_window:]
            consecutive_breaches = all(cvar < self.cvar_breach_threshold for cvar in recent_cvars)
            
            if consecutive_breaches:
                avg_cvar = np.mean(recent_cvars)
                event = DriftEvent(
                    timestamp=pd.Timestamp.now(),
                    event_type='cvar_breach',
                    severity='high',
                    value=avg_cvar,
                    threshold=self.cvar_breach_threshold,
                    description=f"CVaR breach for {self.cvar_window} consecutive days",
                    metadata={'recent_cvars': recent_cvars}
                )
                events.append(event)
                drift_detected = True
                self.stats['cvar_events'] += 1
                logger.warning(f"CVaR breach detected: {avg_cvar:.4f} < {self.cvar_breach_threshold}")
        
        # Store events
        self.drift_events.extend(events)
        self.stats['total_drift_events'] += len(events)
        
        return drift_detected, events
    
    def should_trigger_warmup(self) -> bool:
        """
        Determine if warmup retraining should be triggered.
        
        Returns:
            True if warmup should be triggered
        """
        drift_detected, events = self.check_drift_conditions()
        
        if not drift_detected:
            return False
        
        # Trigger warmup if any high severity event or multiple medium events
        high_severity_events = [e for e in events if e.severity == 'high']
        medium_severity_events = [e for e in events if e.severity == 'medium']
        
        should_trigger = (len(high_severity_events) > 0 or 
                         len(medium_severity_events) >= 2)
        
        if should_trigger:
            logger.info(f"Warmup retraining triggered by {len(events)} drift events")
        
        return should_trigger
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get summary of drift detection status.
        
        Returns:
            Dictionary containing drift monitoring statistics
        """
        summary = {
            'stats': self.stats.copy(),
            'current_metrics': {
                'kl_divergence': self.last_kl_divergence,
                'baseline_sharpe': self.baseline_sharpe,
                'current_sharpe': np.mean(list(self.sharpe_history)[-5:]) if len(self.sharpe_history) >= 5 else None,
                'recent_cvar': list(self.cvar_history)[-1] if self.cvar_history else None
            },
            'data_status': {
                'offline_samples': len(self.offline_q_values),
                'online_samples': len(self.online_q_values),
                'sharpe_history_length': len(self.sharpe_history),
                'cvar_history_length': len(self.cvar_history)
            },
            'recent_events': self.drift_events[-10:] if self.drift_events else []
        }
        
        return summary
    
    def reset_monitoring(self) -> None:
        """Reset monitoring state (typically called after retraining)."""
        logger.info("Resetting drift monitoring state")
        
        # Keep some offline data but clear online data
        if len(self.offline_q_values) > 1000:
            self.offline_q_values = self.offline_q_values[-1000:]
        
        self.online_q_values.clear()
        self.sharpe_history.clear()
        self.cvar_history.clear()
        self.returns_history.clear()
        
        # Reset baseline
        self.baseline_sharpe = None
        self.last_kl_divergence = None
        
        # Keep event history but reset counters
        self.stats.update({
            'kl_events': 0,
            'sharpe_events': 0,
            'cvar_events': 0,
            'last_update': pd.Timestamp.now()
        })
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        if np.std(returns_array) == 0:
            return 0.0
            
        # Annualized Sharpe ratio (assuming daily returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        sharpe = (mean_return / std_return) * np.sqrt(252)  # 252 trading days
        
        return sharpe
    
    def _calculate_cvar(self, returns: List[float], alpha: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if len(returns) < 20:  # Need sufficient data
            return 0.0
            
        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, alpha * 100)
        
        # CVaR is the mean of returns below VaR threshold
        tail_returns = returns_array[returns_array <= var_threshold]
        if len(tail_returns) == 0:
            return var_threshold
            
        cvar = np.mean(tail_returns)
        return cvar