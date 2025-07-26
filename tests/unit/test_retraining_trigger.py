"""
Tests for RetrainingTrigger.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.retraining_trigger import RetrainingTrigger, RetrainingDecision, RetrainingEvent
from monitoring.value_drift_monitor import ValueDriftMonitor, DriftEvent


class TestRetrainingTrigger:
    """Test suite for RetrainingTrigger."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.drift_config = {
            'kl_threshold': 0.1,
            'sharpe_drop_threshold': 0.2,
            'cvar_breach_threshold': -0.02,
            'min_samples': 50
        }
        
        self.trigger_config = {
            'evaluation_window': 5,
            'confidence_threshold': 0.7,
            'cooldown_period': 24,  # hours
            'initial_rho': 0.2,
            'emergency_thresholds': {
                'kl_divergence': 1.0,
                'sharpe_drop': 0.5,
                'cvar_breach': -0.05,
                'consecutive_losses': 5
            },
            'event_weights': {
                'kl_divergence': 1.0,
                'sharpe_drop': 0.8,
                'cvar_breach': 1.2
            }
        }
        
        self.drift_monitor = ValueDriftMonitor(self.drift_config)
        self.trigger = RetrainingTrigger(self.trigger_config, self.drift_monitor)
    
    def test_initialization(self):
        """Test trigger initialization."""
        assert self.trigger.confidence_threshold == 0.7
        assert self.trigger.cooldown_period == pd.Timedelta(hours=24)
        assert self.trigger.current_rho == 0.2
        assert self.trigger.consecutive_losses == 0
        assert len(self.trigger.retraining_history) == 0
        assert len(self.trigger.event_log) == 0
    
    def test_set_callbacks(self):
        """Test setting callback functions."""
        retraining_callback = Mock()
        rho_callback = Mock()
        
        self.trigger.set_retraining_callback(retraining_callback)
        self.trigger.set_rho_reset_callback(rho_callback)
        
        assert self.trigger.retraining_callback == retraining_callback
        assert self.trigger.rho_reset_callback == rho_callback
    
    def test_evaluate_drift_conditions_no_drift(self):
        """Test drift evaluation with no drift."""
        # Mock no drift detected
        with patch.object(self.drift_monitor, 'check_drift_conditions') as mock_check:
            mock_check.return_value = (False, [])
            
            decision, confidence, reason = self.trigger.evaluate_drift_conditions()
            
            assert decision == RetrainingDecision.NO_ACTION
            assert confidence == 0.0
            assert "No drift detected" in reason
    
    def test_evaluate_drift_conditions_normal_drift(self):
        """Test drift evaluation with normal drift conditions."""
        # Create mock drift events
        drift_event = DriftEvent(
            timestamp=pd.Timestamp.now(),
            event_type='kl_divergence',
            severity='medium',
            value=0.15,
            threshold=0.1,
            description="Test drift",
            metadata={}
        )
        
        with patch.object(self.drift_monitor, 'check_drift_conditions') as mock_check:
            mock_check.return_value = (True, [drift_event])
            
            decision, confidence, reason = self.trigger.evaluate_drift_conditions()
            
            # Should make some decision based on the drift
            assert decision in [RetrainingDecision.NO_ACTION, RetrainingDecision.WARMUP_RETRAINING]
            assert isinstance(confidence, float)
            assert 0.0 <= confidence <= 1.0
    
    def test_evaluate_drift_conditions_emergency(self):
        """Test drift evaluation with emergency conditions."""
        # Create emergency-level drift event
        emergency_event = DriftEvent(
            timestamp=pd.Timestamp.now(),
            event_type='kl_divergence',
            severity='high',
            value=1.5,  # Above emergency threshold
            threshold=0.1,
            description="Emergency drift",
            metadata={}
        )
        
        with patch.object(self.drift_monitor, 'check_drift_conditions') as mock_check:
            mock_check.return_value = (True, [emergency_event])
            
            decision, confidence, reason = self.trigger.evaluate_drift_conditions()
            
            assert decision == RetrainingDecision.EMERGENCY_STOP
            assert confidence == 1.0
            assert "Emergency conditions detected" in reason
    
    def test_cooldown_period(self):
        """Test cooldown period functionality."""
        # Set last retraining to recent time
        self.trigger.last_retraining = pd.Timestamp.now() - pd.Timedelta(hours=12)
        
        # Create drift event that would normally trigger retraining
        drift_event = DriftEvent(
            timestamp=pd.Timestamp.now(),
            event_type='kl_divergence',
            severity='high',
            value=0.2,
            threshold=0.1,
            description="Test drift",
            metadata={}
        )
        
        with patch.object(self.drift_monitor, 'check_drift_conditions') as mock_check:
            mock_check.return_value = (True, [drift_event])
            
            decision, confidence, reason = self.trigger.evaluate_drift_conditions()
            
            # Should have reduced confidence due to cooldown
            assert "cooldown penalty applied" in reason
            assert confidence < 0.7  # Should be reduced
    
    def test_trigger_retraining_if_needed_no_action(self):
        """Test retraining trigger with no action needed."""
        with patch.object(self.trigger, 'evaluate_drift_conditions') as mock_eval:
            mock_eval.return_value = (RetrainingDecision.NO_ACTION, 0.0, "No drift")
            
            result = self.trigger.trigger_retraining_if_needed()
            
            assert result is None
    
    def test_trigger_retraining_if_needed_warmup(self):
        """Test retraining trigger with warmup needed."""
        # Mock callbacks
        retraining_callback = Mock()
        rho_callback = Mock()
        self.trigger.set_retraining_callback(retraining_callback)
        self.trigger.set_rho_reset_callback(rho_callback)
        
        # Mock drift conditions
        drift_event = DriftEvent(
            timestamp=pd.Timestamp.now(),
            event_type='kl_divergence',
            severity='medium',
            value=0.15,
            threshold=0.1,
            description="Test drift",
            metadata={}
        )
        
        with patch.object(self.trigger, 'evaluate_drift_conditions') as mock_eval:
            mock_eval.return_value = (RetrainingDecision.WARMUP_RETRAINING, 0.8, "High confidence")
            
            with patch.object(self.drift_monitor, 'check_drift_conditions') as mock_check:
                mock_check.return_value = (True, [drift_event])
                
                result = self.trigger.trigger_retraining_if_needed()
                
                assert result is not None
                assert result.decision == RetrainingDecision.WARMUP_RETRAINING
                assert result.confidence == 0.8
                
                # Check callbacks were called
                retraining_callback.assert_called_once()
                rho_callback.assert_called_once()
                
                # Check state updates
                assert len(self.trigger.retraining_history) == 1
                assert self.trigger.last_retraining is not None
    
    def test_reset_sampling_ratio(self):
        """Test sampling ratio reset."""
        # Set initial ratio
        self.trigger.current_rho = 0.8
        
        # Mock callback
        rho_callback = Mock()
        self.trigger.set_rho_reset_callback(rho_callback)
        
        # Reset to default
        new_rho = self.trigger.reset_sampling_ratio()
        
        assert new_rho == 0.2  # Should reset to initial value
        assert self.trigger.current_rho == 0.2
        rho_callback.assert_called_once_with(0.2)
        
        # Check event log
        assert len(self.trigger.event_log) == 1
        assert self.trigger.event_log[0]['event_type'] == 'rho_reset'
        
        # Reset to specific value
        new_rho = self.trigger.reset_sampling_ratio(0.5)
        assert new_rho == 0.5
        assert self.trigger.current_rho == 0.5
    
    def test_update_performance_feedback(self):
        """Test performance feedback updates."""
        # Test with losses
        self.trigger.update_performance_feedback(-0.01, is_loss=True)
        assert self.trigger.consecutive_losses == 1
        
        self.trigger.update_performance_feedback(-0.02, is_loss=True)
        assert self.trigger.consecutive_losses == 2
        
        # Test with gain (should reset consecutive losses)
        self.trigger.update_performance_feedback(0.01, is_loss=False)
        assert self.trigger.consecutive_losses == 0
        
        # Test with automatic loss detection
        self.trigger.update_performance_feedback(-0.01)  # Should infer is_loss=True
        assert self.trigger.consecutive_losses == 1
    
    def test_consecutive_losses_emergency(self):
        """Test emergency trigger due to consecutive losses."""
        # Set up consecutive losses
        for _ in range(5):
            self.trigger.update_performance_feedback(-0.01, is_loss=True)
        
        # Create any drift event to trigger evaluation
        drift_event = DriftEvent(
            timestamp=pd.Timestamp.now(),
            event_type='kl_divergence',
            severity='low',
            value=0.05,
            threshold=0.1,
            description="Minor drift",
            metadata={}
        )
        
        with patch.object(self.drift_monitor, 'check_drift_conditions') as mock_check:
            mock_check.return_value = (True, [drift_event])
            
            decision, confidence, reason = self.trigger.evaluate_drift_conditions()
            
            assert decision == RetrainingDecision.FULL_RETRAINING
            assert confidence == 1.0
    
    def test_composite_score_calculation(self):
        """Test composite score calculation."""
        # Create events with different severities
        events = [
            DriftEvent(
                timestamp=pd.Timestamp.now(),
                event_type='kl_divergence',
                severity='high',
                value=0.2,
                threshold=0.1,
                description="High KL drift",
                metadata={}
            ),
            DriftEvent(
                timestamp=pd.Timestamp.now(),
                event_type='sharpe_drop',
                severity='medium',
                value=0.3,
                threshold=0.2,
                description="Sharpe drop",
                metadata={}
            )
        ]
        
        score = self.trigger._calculate_composite_score(events)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        # Test with empty events
        assert self.trigger._calculate_composite_score([]) == 0.0
    
    def test_get_retraining_history(self):
        """Test retraining history retrieval."""
        # Add some mock history
        event1 = RetrainingEvent(
            timestamp=pd.Timestamp.now(),
            decision=RetrainingDecision.WARMUP_RETRAINING,
            trigger_events=[],
            confidence=0.8,
            metadata={},
            reason="Test"
        )
        
        self.trigger.retraining_history.append(event1)
        
        history = self.trigger.get_retraining_history(limit=5)
        
        assert len(history) == 1
        assert history[0]['decision'] == 'warmup_retraining'
        assert history[0]['confidence'] == 0.8
    
    def test_get_event_log(self):
        """Test event log retrieval."""
        # Add some events to log
        self.trigger.reset_sampling_ratio(0.3)
        
        log = self.trigger.get_event_log(limit=10)
        
        assert len(log) >= 1
        assert log[0]['event_type'] == 'rho_reset'
        assert log[0]['new_value'] == 0.3
    
    def test_emergency_stop_execution(self):
        """Test emergency stop execution."""
        # Mock callbacks
        retraining_callback = Mock()
        rho_callback = Mock()
        self.trigger.set_retraining_callback(retraining_callback)
        self.trigger.set_rho_reset_callback(rho_callback)
        
        # Create emergency event
        event = RetrainingEvent(
            timestamp=pd.Timestamp.now(),
            decision=RetrainingDecision.EMERGENCY_STOP,
            trigger_events=[],
            confidence=1.0,
            metadata={},
            reason="Emergency test"
        )
        
        self.trigger._execute_emergency_stop(event)
        
        # Check conservative rho reset
        assert self.trigger.current_rho == 0.1
        
        # Check consecutive losses reset
        assert self.trigger.consecutive_losses == 0
        
        # Check callbacks called
        retraining_callback.assert_called_once_with(event)
        rho_callback.assert_called_once_with(0.1)
        
        # Check history updated
        assert len(self.trigger.retraining_history) == 1


if __name__ == "__main__":
    pytest.main([__file__])