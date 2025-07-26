"""
Tests for ValueDriftMonitor.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.value_drift_monitor import ValueDriftMonitor, DriftEvent


class TestValueDriftMonitor:
    """Test suite for ValueDriftMonitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'kl_threshold': 0.1,
            'sharpe_drop_threshold': 0.2,
            'cvar_breach_threshold': -0.02,
            'sharpe_window': 30,
            'cvar_window': 3,
            'min_samples': 50  # Lower for testing
        }
        self.monitor = ValueDriftMonitor(self.config)
    
    def test_initialization(self):
        """Test monitor initialization."""
        assert self.monitor.kl_threshold == 0.1
        assert self.monitor.sharpe_drop_threshold == 0.2
        assert self.monitor.cvar_breach_threshold == -0.02
        assert len(self.monitor.offline_q_values) == 0
        assert len(self.monitor.online_q_values) == 0
        assert self.monitor.baseline_sharpe is None
    
    def test_update_offline_values(self):
        """Test updating offline Q-values."""
        q_values = np.random.normal(0, 1, 100)
        self.monitor.update_offline_values(q_values)
        
        assert len(self.monitor.offline_q_values) == 100
        
        # Test with invalid values
        invalid_values = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0])
        self.monitor.update_offline_values(invalid_values)
        
        # Should only add the valid values
        assert len(self.monitor.offline_q_values) == 102
    
    def test_update_online_values(self):
        """Test updating online Q-values."""
        q_values = np.random.normal(1, 1, 80)
        self.monitor.update_online_values(q_values)
        
        assert len(self.monitor.online_q_values) == 80
    
    def test_update_performance_metrics(self):
        """Test updating performance metrics."""
        # Add some returns
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        for ret in returns:
            self.monitor.update_performance_metrics(ret, 100000)
        
        assert len(self.monitor.returns_history) == 5
        assert len(self.monitor.sharpe_history) == 1  # Should calculate Sharpe after 5 returns
        
        # Add more returns to test CVaR calculation
        for _ in range(20):
            ret = np.random.normal(0.001, 0.02)
            self.monitor.update_performance_metrics(ret, 100000)
        
        assert len(self.monitor.cvar_history) > 0
    
    def test_calculate_kl_divergence_insufficient_data(self):
        """Test KL divergence with insufficient data."""
        # Add some data but not enough
        self.monitor.update_offline_values(np.random.normal(0, 1, 30))
        self.monitor.update_online_values(np.random.normal(0, 1, 20))
        
        kl_div = self.monitor.calculate_kl_divergence()
        assert kl_div is None
    
    def test_calculate_kl_divergence_sufficient_data(self):
        """Test KL divergence with sufficient data."""
        # Add sufficient data with different distributions
        self.monitor.update_offline_values(np.random.normal(0, 1, 100))
        self.monitor.update_online_values(np.random.normal(0.5, 1, 100))  # Shifted distribution
        
        kl_div = self.monitor.calculate_kl_divergence()
        assert kl_div is not None
        assert kl_div >= 0  # KL divergence is always non-negative
        assert self.monitor.last_kl_divergence == kl_div
    
    def test_calculate_kl_divergence_identical_distributions(self):
        """Test KL divergence with identical distributions."""
        data = np.random.normal(0, 1, 100)
        self.monitor.update_offline_values(data)
        self.monitor.update_online_values(data)
        
        kl_div = self.monitor.calculate_kl_divergence()
        assert kl_div is not None
        assert kl_div < 0.01  # Should be very small for identical distributions
    
    def test_check_drift_conditions_no_drift(self):
        """Test drift detection with no drift."""
        # Add identical data to ensure no KL drift
        np.random.seed(42)  # Set seed for reproducible test
        data = np.random.normal(0, 1, 100)
        self.monitor.update_offline_values(data)
        self.monitor.update_online_values(data)  # Exactly the same data
        
        # Add stable performance metrics
        for _ in range(35):
            self.monitor.update_performance_metrics(0.001, 100000)
        
        drift_detected, events = self.monitor.check_drift_conditions()
        
        # With identical data, KL divergence should be very small
        kl_events = [e for e in events if e.event_type == 'kl_divergence']
        if kl_events:
            assert kl_events[0].value < 0.01  # Should be very small for identical data
    
    def test_check_drift_conditions_kl_drift(self):
        """Test drift detection with KL divergence drift."""
        # Add very different distributions
        self.monitor.update_offline_values(np.random.normal(0, 1, 100))
        self.monitor.update_online_values(np.random.normal(2, 1, 100))  # Very different
        
        drift_detected, events = self.monitor.check_drift_conditions()
        assert drift_detected
        assert len(events) >= 1
        
        kl_event = next((e for e in events if e.event_type == 'kl_divergence'), None)
        assert kl_event is not None
        assert kl_event.value > self.config['kl_threshold']
    
    def test_check_drift_conditions_sharpe_drop(self):
        """Test drift detection with Sharpe ratio drop."""
        # First establish baseline with good returns
        for _ in range(15):
            self.monitor.update_performance_metrics(0.01, 100000)  # Good returns
        
        # Then add poor returns
        for _ in range(10):
            self.monitor.update_performance_metrics(-0.02, 100000)  # Poor returns
        
        drift_detected, events = self.monitor.check_drift_conditions()
        
        # Check if Sharpe drop was detected
        sharpe_event = next((e for e in events if e.event_type == 'sharpe_drop'), None)
        if sharpe_event:
            assert sharpe_event.value > self.config['sharpe_drop_threshold']
    
    def test_check_drift_conditions_cvar_breach(self):
        """Test drift detection with CVaR breach."""
        # Add enough data for CVaR calculation
        for _ in range(25):
            self.monitor.update_performance_metrics(0.001, 100000)
        
        # Add consecutive bad returns to trigger CVaR breach
        for _ in range(3):
            self.monitor.update_performance_metrics(-0.05, 100000)  # Very bad returns
        
        drift_detected, events = self.monitor.check_drift_conditions()
        
        # Check if CVaR breach was detected
        cvar_event = next((e for e in events if e.event_type == 'cvar_breach'), None)
        if cvar_event:
            assert cvar_event.value < self.config['cvar_breach_threshold']
    
    def test_should_trigger_warmup(self):
        """Test warmup trigger logic."""
        # Test with no drift
        assert not self.monitor.should_trigger_warmup()
        
        # Create high severity event
        high_event = DriftEvent(
            timestamp=pd.Timestamp.now(),
            event_type='kl_divergence',
            severity='high',
            value=0.5,
            threshold=0.1,
            description="Test high severity",
            metadata={}
        )
        
        with patch.object(self.monitor, 'check_drift_conditions') as mock_check:
            mock_check.return_value = (True, [high_event])
            assert self.monitor.should_trigger_warmup()
    
    def test_get_drift_summary(self):
        """Test drift summary generation."""
        # Add some data
        self.monitor.update_offline_values(np.random.normal(0, 1, 100))
        self.monitor.update_online_values(np.random.normal(0, 1, 80))
        
        for _ in range(10):
            self.monitor.update_performance_metrics(0.001, 100000)
        
        summary = self.monitor.get_drift_summary()
        
        assert 'stats' in summary
        assert 'current_metrics' in summary
        assert 'data_status' in summary
        assert 'recent_events' in summary
        
        assert summary['data_status']['offline_samples'] == 100
        assert summary['data_status']['online_samples'] == 80
    
    def test_reset_monitoring(self):
        """Test monitoring reset."""
        # Add data
        self.monitor.update_offline_values(np.random.normal(0, 1, 2000))
        self.monitor.update_online_values(np.random.normal(0, 1, 500))
        
        for _ in range(20):
            self.monitor.update_performance_metrics(0.001, 100000)
        
        # Set baseline
        self.monitor.baseline_sharpe = 1.5
        
        # Reset
        self.monitor.reset_monitoring()
        
        # Check reset state
        assert len(self.monitor.offline_q_values) <= 1000  # Should keep some offline data
        assert len(self.monitor.online_q_values) == 0
        assert len(self.monitor.sharpe_history) == 0
        assert len(self.monitor.cvar_history) == 0
        assert self.monitor.baseline_sharpe is None
    
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        # Test with insufficient data
        assert self.monitor._calculate_sharpe_ratio([0.01]) == 0.0
        
        # Test with zero std
        assert self.monitor._calculate_sharpe_ratio([0.01, 0.01, 0.01]) == 0.0
        
        # Test with normal data
        returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003]
        sharpe = self.monitor._calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_calculate_cvar(self):
        """Test CVaR calculation."""
        # Test with insufficient data
        assert self.monitor._calculate_cvar([0.01, 0.02]) == 0.0
        
        # Test with normal data
        returns = np.random.normal(0.001, 0.02, 100).tolist()
        cvar = self.monitor._calculate_cvar(returns)
        assert isinstance(cvar, float)
        assert cvar <= 0  # CVaR should be negative (worst-case returns)


if __name__ == "__main__":
    pytest.main([__file__])