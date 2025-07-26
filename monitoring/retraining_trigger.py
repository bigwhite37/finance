"""
Automatic Retraining Trigger Mechanism for O2O RL System.

This module implements the logic for automatically triggering warmup retraining
based on drift detection events and comprehensive evaluation algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque
import json

from .value_drift_monitor import ValueDriftMonitor, DriftEvent

logger = logging.getLogger(__name__)


class RetrainingDecision(Enum):
    """Enumeration of retraining decisions."""
    NO_ACTION = "no_action"
    WARMUP_RETRAINING = "warmup_retraining"
    FULL_RETRAINING = "full_retraining"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class RetrainingEvent:
    """Data structure for retraining events."""
    timestamp: pd.Timestamp
    decision: RetrainingDecision
    trigger_events: List[DriftEvent]
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]
    reason: str


class RetrainingTrigger:
    """
    自动重训练触发机制
    
    Implements comprehensive drift condition evaluation and automatic
    warmup retraining trigger logic with event logging and sampling
    ratio reset functionality.
    """
    
    def __init__(self, config: Dict[str, Any], drift_monitor: ValueDriftMonitor):
        """
        Initialize the Retraining Trigger.
        
        Args:
            config: Configuration dictionary containing:
                - evaluation_window: Window for evaluating multiple events (default: 5)
                - confidence_threshold: Minimum confidence for triggering (default: 0.7)
                - cooldown_period: Minimum time between retrainings in hours (default: 24)
                - emergency_thresholds: Thresholds for emergency stop conditions
                - event_weights: Weights for different event types in scoring
            drift_monitor: ValueDriftMonitor instance for drift detection
        """
        self.config = config
        self.drift_monitor = drift_monitor
        
        # Configuration parameters
        self.evaluation_window = config.get('evaluation_window', 5)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.cooldown_period = pd.Timedelta(hours=config.get('cooldown_period', 24))
        
        # Emergency thresholds
        self.emergency_thresholds = config.get('emergency_thresholds', {
            'kl_divergence': 1.0,
            'sharpe_drop': 0.5,
            'cvar_breach': -0.05,
            'consecutive_losses': 5
        })
        
        # Event weights for scoring
        self.event_weights = config.get('event_weights', {
            'kl_divergence': 1.0,
            'sharpe_drop': 0.8,
            'cvar_breach': 1.2
        })
        
        # State tracking
        self.retraining_history = deque(maxlen=100)
        self.last_retraining = None
        self.current_rho = config.get('initial_rho', 0.2)
        self.consecutive_losses = 0
        
        # Event logging
        self.event_log = []
        
        # Callbacks
        self.retraining_callback: Optional[Callable] = None
        self.rho_reset_callback: Optional[Callable] = None
        
        logger.info(f"RetrainingTrigger initialized with confidence_threshold={self.confidence_threshold}, "
                   f"cooldown_period={self.cooldown_period}")
    
    def set_retraining_callback(self, callback: Callable[[RetrainingEvent], None]) -> None:
        """Set callback function to be called when retraining is triggered."""
        self.retraining_callback = callback
    
    def set_rho_reset_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback function to be called when sampling ratio needs reset."""
        self.rho_reset_callback = callback
    
    def evaluate_drift_conditions(self) -> Tuple[RetrainingDecision, float, str]:
        """
        Comprehensive evaluation of drift conditions.
        
        Returns:
            Tuple of (decision, confidence, reason)
        """
        # Get current drift status
        drift_detected, current_events = self.drift_monitor.check_drift_conditions()
        
        if not drift_detected or not current_events:
            return RetrainingDecision.NO_ACTION, 0.0, "No drift detected"
        
        # Check for emergency conditions first
        emergency_decision = self._check_emergency_conditions(current_events)
        if emergency_decision != RetrainingDecision.NO_ACTION:
            return emergency_decision, 1.0, "Emergency conditions detected"
        
        # Calculate composite score based on event severity and types
        composite_score = self._calculate_composite_score(current_events)
        
        # Consider recent event history
        historical_score = self._calculate_historical_score()
        
        # Combine scores
        final_score = 0.7 * composite_score + 0.3 * historical_score
        
        # Check cooldown period
        if self._is_in_cooldown():
            cooldown_penalty = 0.5
            final_score *= cooldown_penalty
            reason_suffix = " (cooldown penalty applied)"
        else:
            reason_suffix = ""
        
        # Make decision based on final score
        if final_score >= self.confidence_threshold:
            decision = RetrainingDecision.WARMUP_RETRAINING
            reason = f"Composite score {final_score:.3f} exceeds threshold{reason_suffix}"
        else:
            decision = RetrainingDecision.NO_ACTION
            reason = f"Composite score {final_score:.3f} below threshold{reason_suffix}"
        
        logger.info(f"Drift evaluation: {decision.value}, score={final_score:.3f}, reason={reason}")
        
        return decision, final_score, reason
    
    def trigger_retraining_if_needed(self) -> Optional[RetrainingEvent]:
        """
        Main method to check conditions and trigger retraining if needed.
        
        Returns:
            RetrainingEvent if retraining was triggered, None otherwise
        """
        decision, confidence, reason = self.evaluate_drift_conditions()
        
        if decision == RetrainingDecision.NO_ACTION:
            return None
        
        # Get current drift events for logging
        _, current_events = self.drift_monitor.check_drift_conditions()
        
        # Create retraining event
        retraining_event = RetrainingEvent(
            timestamp=pd.Timestamp.now(),
            decision=decision,
            trigger_events=current_events,
            confidence=confidence,
            metadata={
                'current_rho': self.current_rho,
                'consecutive_losses': self.consecutive_losses,
                'last_retraining': self.last_retraining,
                'drift_summary': self.drift_monitor.get_drift_summary()
            },
            reason=reason
        )
        
        # Log the event
        self._log_retraining_event(retraining_event)
        
        # Execute retraining actions
        if decision in [RetrainingDecision.WARMUP_RETRAINING, RetrainingDecision.FULL_RETRAINING]:
            self._execute_retraining(retraining_event)
        elif decision == RetrainingDecision.EMERGENCY_STOP:
            self._execute_emergency_stop(retraining_event)
        
        return retraining_event
    
    def reset_sampling_ratio(self, new_rho: Optional[float] = None) -> float:
        """
        Reset sampling ratio ρ in response to drift events.
        
        Args:
            new_rho: New sampling ratio, if None uses configured initial value
            
        Returns:
            New sampling ratio value
        """
        if new_rho is None:
            new_rho = self.config.get('initial_rho', 0.2)
        
        old_rho = self.current_rho
        self.current_rho = new_rho
        
        logger.info(f"Sampling ratio reset: {old_rho:.3f} -> {new_rho:.3f}")
        
        # Call callback if set
        if self.rho_reset_callback:
            try:
                self.rho_reset_callback(new_rho)
            except Exception as e:
                logger.error(f"Error in rho reset callback: {e}")
        
        # Log the reset event
        self.event_log.append({
            'timestamp': pd.Timestamp.now(),
            'event_type': 'rho_reset',
            'old_value': old_rho,
            'new_value': new_rho,
            'reason': 'drift_response'
        })
        
        return new_rho
    
    def update_performance_feedback(self, returns: float, is_loss: bool = None) -> None:
        """
        Update performance feedback for decision making.
        
        Args:
            returns: Recent returns
            is_loss: Whether this represents a loss (if None, inferred from returns)
        """
        if is_loss is None:
            is_loss = returns < 0
        
        if is_loss:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Update drift monitor with performance data
        self.drift_monitor.update_performance_metrics(returns, 100000)  # Dummy portfolio value
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent retraining history.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of retraining events as dictionaries
        """
        recent_events = list(self.retraining_history)[-limit:]
        return [self._retraining_event_to_dict(event) for event in recent_events]
    
    def get_event_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent event log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of log entries
        """
        return self.event_log[-limit:]
    
    def _check_emergency_conditions(self, events: List[DriftEvent]) -> RetrainingDecision:
        """Check for emergency conditions requiring immediate action."""
        for event in events:
            if event.event_type == 'kl_divergence':
                if event.value > self.emergency_thresholds['kl_divergence']:
                    logger.critical(f"Emergency: KL divergence {event.value:.4f} exceeds emergency threshold")
                    return RetrainingDecision.EMERGENCY_STOP
            
            elif event.event_type == 'sharpe_drop':
                if event.value > self.emergency_thresholds['sharpe_drop']:
                    logger.critical(f"Emergency: Sharpe drop {event.value:.2%} exceeds emergency threshold")
                    return RetrainingDecision.FULL_RETRAINING
            
            elif event.event_type == 'cvar_breach':
                if event.value < self.emergency_thresholds['cvar_breach']:
                    logger.critical(f"Emergency: CVaR {event.value:.4f} exceeds emergency threshold")
                    return RetrainingDecision.FULL_RETRAINING
        
        # Check consecutive losses
        if self.consecutive_losses >= self.emergency_thresholds['consecutive_losses']:
            logger.critical(f"Emergency: {self.consecutive_losses} consecutive losses")
            return RetrainingDecision.FULL_RETRAINING
        
        return RetrainingDecision.NO_ACTION
    
    def _calculate_composite_score(self, events: List[DriftEvent]) -> float:
        """Calculate composite score from current events."""
        if not events:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for event in events:
            # Get base weight for event type
            weight = self.event_weights.get(event.event_type, 1.0)
            
            # Adjust weight based on severity
            if event.severity == 'high':
                weight *= 1.5
            elif event.severity == 'low':
                weight *= 0.5
            
            # Calculate normalized score based on threshold exceedance
            if event.event_type == 'kl_divergence':
                score = min(1.0, event.value / event.threshold)
            elif event.event_type == 'sharpe_drop':
                score = min(1.0, event.value / event.threshold)
            elif event.event_type == 'cvar_breach':
                score = min(1.0, abs(event.value / event.threshold))
            else:
                score = 0.5  # Default score for unknown event types
            
            total_score += weight * score
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_historical_score(self) -> float:
        """Calculate score based on recent event history."""
        if not self.retraining_history:
            return 0.0
        
        # Look at recent retraining frequency
        recent_events = [e for e in self.retraining_history 
                        if e.timestamp > pd.Timestamp.now() - pd.Timedelta(days=7)]
        
        if len(recent_events) >= 3:
            return 0.8  # High score if frequent retraining needed
        elif len(recent_events) >= 2:
            return 0.5  # Medium score
        else:
            return 0.2  # Low score
    
    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period after last retraining."""
        if self.last_retraining is None:
            return False
        
        return pd.Timestamp.now() - self.last_retraining < self.cooldown_period
    
    def _execute_retraining(self, event: RetrainingEvent) -> None:
        """Execute retraining actions."""
        logger.info(f"Executing {event.decision.value} with confidence {event.confidence:.3f}")
        
        # Reset sampling ratio
        self.reset_sampling_ratio()
        
        # Reset drift monitoring state
        self.drift_monitor.reset_monitoring()
        
        # Update last retraining timestamp
        self.last_retraining = event.timestamp
        
        # Call retraining callback if set
        if self.retraining_callback:
            try:
                self.retraining_callback(event)
            except Exception as e:
                logger.error(f"Error in retraining callback: {e}")
        
        # Add to history
        self.retraining_history.append(event)
    
    def _execute_emergency_stop(self, event: RetrainingEvent) -> None:
        """Execute emergency stop actions."""
        logger.critical(f"Executing emergency stop: {event.reason}")
        
        # Reset to very conservative sampling ratio
        self.reset_sampling_ratio(0.1)
        
        # Reset consecutive losses
        self.consecutive_losses = 0
        
        # Call retraining callback with emergency flag
        if self.retraining_callback:
            try:
                self.retraining_callback(event)
            except Exception as e:
                logger.error(f"Error in emergency callback: {e}")
        
        # Add to history
        self.retraining_history.append(event)
    
    def _log_retraining_event(self, event: RetrainingEvent) -> None:
        """Log retraining event to event log."""
        log_entry = {
            'timestamp': event.timestamp,
            'event_type': 'retraining_trigger',
            'decision': event.decision.value,
            'confidence': event.confidence,
            'reason': event.reason,
            'trigger_event_count': len(event.trigger_events),
            'trigger_event_types': [e.event_type for e in event.trigger_events]
        }
        
        self.event_log.append(log_entry)
        
        # Keep log size manageable
        if len(self.event_log) > 1000:
            self.event_log = self.event_log[-500:]
    
    def _retraining_event_to_dict(self, event: RetrainingEvent) -> Dict[str, Any]:
        """Convert RetrainingEvent to dictionary for serialization."""
        return {
            'timestamp': event.timestamp.isoformat(),
            'decision': event.decision.value,
            'confidence': event.confidence,
            'reason': event.reason,
            'trigger_events': len(event.trigger_events),
            'metadata': {k: str(v) for k, v in event.metadata.items()}  # Convert to strings for JSON
        }