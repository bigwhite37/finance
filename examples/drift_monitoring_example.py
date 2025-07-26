"""
Example demonstrating the O2O drift monitoring and risk constraint system.

This example shows how to use the ValueDriftMonitor, RetrainingTrigger,
and DynamicRiskConstraint components together for comprehensive
distribution drift detection and adaptive risk management.
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import O2O monitoring components
from monitoring.value_drift_monitor import ValueDriftMonitor
from monitoring.retraining_trigger import RetrainingTrigger, RetrainingDecision
from risk_control.dynamic_risk_constraint import DynamicRiskConstraint, MarketRegime


def simulate_market_data(n_days: int = 100, regime_change_day: int = 50) -> Dict[str, np.ndarray]:
    """Simulate market data with regime change."""
    np.random.seed(42)
    
    # Normal regime (first half)
    normal_returns = np.random.normal(0.001, 0.015, regime_change_day)
    
    # High volatility regime (second half)
    volatile_returns = np.random.normal(-0.002, 0.04, n_days - regime_change_day)
    
    returns = np.concatenate([normal_returns, volatile_returns])
    
    # Generate Q-values (offline vs online)
    offline_q_values = np.random.normal(0.5, 0.2, n_days * 10)
    online_q_values_normal = np.random.normal(0.5, 0.2, regime_change_day * 10)
    online_q_values_shifted = np.random.normal(0.8, 0.3, (n_days - regime_change_day) * 10)
    online_q_values = np.concatenate([online_q_values_normal, online_q_values_shifted])
    
    return {
        'returns': returns,
        'offline_q_values': offline_q_values,
        'online_q_values': online_q_values
    }


def retraining_callback(event):
    """Callback function for retraining events."""
    logger.info(f"🔄 Retraining triggered: {event.decision.value}")
    logger.info(f"   Confidence: {event.confidence:.3f}")
    logger.info(f"   Reason: {event.reason}")
    logger.info(f"   Trigger events: {len(event.trigger_events)}")


def rho_reset_callback(new_rho):
    """Callback function for sampling ratio resets."""
    logger.info(f"📊 Sampling ratio reset to: {new_rho:.3f}")


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting O2O Drift Monitoring Example")
    
    # Configuration
    drift_config = {
        'kl_threshold': 0.1,
        'sharpe_drop_threshold': 0.2,
        'cvar_breach_threshold': -0.02,
        'min_samples': 50
    }
    
    trigger_config = {
        'confidence_threshold': 0.7,
        'cooldown_period': 24,  # hours
        'initial_rho': 0.2,
        'emergency_thresholds': {
            'kl_divergence': 1.0,
            'sharpe_drop': 0.5,
            'cvar_breach': -0.05,
            'consecutive_losses': 5
        }
    }
    
    risk_config = {
        'base_cvar_lambda': 1.0,
        'rho_sensitivity': 2.0,
        'emergency_thresholds': {
            'max_drawdown': -0.15,
            'volatility_spike': 0.05,
            'consecutive_losses': 5
        }
    }
    
    # Initialize components
    logger.info("🔧 Initializing monitoring components...")
    drift_monitor = ValueDriftMonitor(drift_config)
    retraining_trigger = RetrainingTrigger(trigger_config, drift_monitor)
    risk_constraint = DynamicRiskConstraint(risk_config)
    
    # Set up callbacks
    retraining_trigger.set_retraining_callback(retraining_callback)
    retraining_trigger.set_rho_reset_callback(rho_reset_callback)
    
    # Simulate market data
    logger.info("📈 Generating simulated market data...")
    market_data = simulate_market_data(n_days=100, regime_change_day=50)
    
    # Initialize with offline data
    logger.info("💾 Initializing with offline Q-values...")
    drift_monitor.update_offline_values(market_data['offline_q_values'])
    
    # Simulate daily trading loop
    logger.info("🔄 Starting daily trading simulation...")
    current_rho = 0.2
    portfolio_value = 100000
    
    for day in range(len(market_data['returns'])):
        daily_return = market_data['returns'][day]
        portfolio_value *= (1 + daily_return)
        
        # Update monitoring systems
        drift_monitor.update_performance_metrics(daily_return, portfolio_value)
        retraining_trigger.update_performance_feedback(daily_return)
        risk_constraint.update_market_data(daily_return, portfolio_value)
        
        # Add online Q-values (simulate batch)
        start_idx = day * 10
        end_idx = (day + 1) * 10
        daily_online_q = market_data['online_q_values'][start_idx:end_idx]
        drift_monitor.update_online_values(daily_online_q)
        
        # Update sampling ratio (gradually increase online proportion)
        if day > 20:  # Start increasing after warmup
            current_rho = min(0.8, 0.2 + (day - 20) * 0.01)
            retraining_trigger.current_rho = current_rho
            risk_constraint.update_sampling_ratio(current_rho)
        
        # Check for retraining needs
        retraining_event = retraining_trigger.trigger_retraining_if_needed()
        
        # Get current risk parameters
        risk_params = risk_constraint.get_risk_parameters()
        
        # Log important events
        if day % 20 == 0 or retraining_event:
            logger.info(f"\n📅 Day {day + 1}:")
            logger.info(f"   Portfolio Value: ${portfolio_value:,.2f}")
            logger.info(f"   Current ρ: {current_rho:.3f}")
            logger.info(f"   Market Regime: {risk_constraint.current_regime.value}")
            logger.info(f"   CVaR λ: {risk_params.cvar_lambda:.3f}")
            logger.info(f"   Emergency Mode: {risk_constraint.emergency_mode}")
            
            # Check drift status
            drift_detected, events = drift_monitor.check_drift_conditions()
            if drift_detected:
                logger.info(f"   ⚠️  Drift detected: {len(events)} events")
                for event in events:
                    logger.info(f"      - {event.event_type}: {event.value:.4f} (severity: {event.severity})")
        
        # Simulate trading with risk constraints
        if day % 10 == 0:  # Every 10 days, simulate a trading decision
            # Mock position proposal
            proposed_positions = np.random.normal(0, 0.3, 5)
            adjusted_positions = risk_constraint.check_position_limits(proposed_positions)
            
            # Mock loss calculation with CVaR constraint
            base_loss = torch.tensor(0.1)
            cvar_estimate = torch.tensor(0.05)
            enhanced_loss = risk_constraint.apply_enhanced_cvar_constraint(
                base_loss, cvar_estimate, current_rho
            )
            
            if day % 20 == 0:
                logger.info(f"   Trading: Enhanced loss = {enhanced_loss.item():.4f}")
    
    # Final summary
    logger.info("\n📊 Final Summary:")
    
    # Drift monitoring summary
    drift_summary = drift_monitor.get_drift_summary()
    logger.info(f"Total drift events: {drift_summary['stats']['total_drift_events']}")
    logger.info(f"KL events: {drift_summary['stats']['kl_events']}")
    logger.info(f"Sharpe events: {drift_summary['stats']['sharpe_events']}")
    logger.info(f"CVaR events: {drift_summary['stats']['cvar_events']}")
    
    # Retraining history
    retraining_history = retraining_trigger.get_retraining_history()
    logger.info(f"Retraining events: {len(retraining_history)}")
    
    # Risk summary
    risk_summary = risk_constraint.get_risk_summary()
    logger.info(f"Final regime: {risk_summary['current_regime']}")
    logger.info(f"Emergency mode: {risk_summary['emergency_mode']}")
    
    logger.info("✅ Example completed successfully!")


if __name__ == "__main__":
    main()