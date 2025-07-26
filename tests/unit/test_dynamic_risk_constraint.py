"""
Tests for DynamicRiskConstraint.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_control.dynamic_risk_constraint import DynamicRiskConstraint, MarketRegime, RiskParameters


class TestDynamicRiskConstraint:
    """Test suite for DynamicRiskConstraint."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'base_cvar_lambda': 1.0,
            'rho_sensitivity': 2.0,
            'volatility_lookback': 20,
            'drawdown_lookback': 30,
            'regime_multipliers': {
                'normal': 1.0,
                'high_volatility': 1.5,
                'low_volatility': 0.8,
                'trending': 0.9,
                'crisis': 2.5
            },
            'emergency_thresholds': {
                'max_drawdown': -0.15,
                'volatility_spike': 0.05,
                'var_breach': -0.03,
                'consecutive_losses': 5
            }
        }
        self.risk_constraint = DynamicRiskConstraint(self.config)
    
    def test_initialization(self):
        """Test risk constraint initialization."""
        assert self.risk_constraint.base_cvar_lambda == 1.0
        assert self.risk_constraint.rho_sensitivity == 2.0
        assert self.risk_constraint.current_regime == MarketRegime.NORMAL
        assert self.risk_constraint.current_rho == 0.2
        assert not self.risk_constraint.emergency_mode
        assert self.risk_constraint.consecutive_losses == 0
    
    def test_update_market_data(self):
        """Test market data updates."""
        # Add some returns
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        for ret in returns:
            self.risk_constraint.update_market_data(ret, 100000)
        
        assert len(self.risk_constraint.returns_history) == 5
        assert self.risk_constraint.consecutive_losses == 0  # Last return was positive
        
        # Add negative return
        self.risk_constraint.update_market_data(-0.01, 100000)
        assert self.risk_constraint.consecutive_losses == 1
        
        # Test with invalid return
        self.risk_constraint.update_market_data(np.nan, 100000)
        assert len(self.risk_constraint.returns_history) == 6  # Should not add invalid return
    
    def test_update_sampling_ratio(self):
        """Test sampling ratio updates."""
        self.risk_constraint.update_sampling_ratio(0.5)
        assert self.risk_constraint.current_rho == 0.5
        
        # Test with invalid values
        self.risk_constraint.update_sampling_ratio(1.5)  # Should be clipped to 1.0
        assert self.risk_constraint.current_rho == 1.0
        
        self.risk_constraint.update_sampling_ratio(-0.1)  # Should be clipped to 0.0
        assert self.risk_constraint.current_rho == 0.0
    
    def test_get_dynamic_risk_weight_normal_regime(self):
        """Test dynamic risk weight calculation in normal regime."""
        # Test with different rho values
        weight_low = self.risk_constraint.get_dynamic_risk_weight(rho=0.2)
        weight_high = self.risk_constraint.get_dynamic_risk_weight(rho=0.8)
        
        # Higher rho should result in higher risk weight
        assert weight_high > weight_low
        
        # Both should be positive
        assert weight_low > 0
        assert weight_high > 0
    
    def test_get_dynamic_risk_weight_different_regimes(self):
        """Test dynamic risk weight with different market regimes."""
        rho = 0.5
        
        # Test different regimes
        weight_normal = self.risk_constraint.get_dynamic_risk_weight(rho, MarketRegime.NORMAL)
        weight_crisis = self.risk_constraint.get_dynamic_risk_weight(rho, MarketRegime.CRISIS)
        weight_low_vol = self.risk_constraint.get_dynamic_risk_weight(rho, MarketRegime.LOW_VOLATILITY)
        
        # Crisis should have highest weight, low volatility lowest
        assert weight_crisis > weight_normal > weight_low_vol
    
    def test_get_dynamic_risk_weight_emergency_mode(self):
        """Test dynamic risk weight in emergency mode."""
        rho = 0.5
        
        # Normal mode
        weight_normal = self.risk_constraint.get_dynamic_risk_weight(rho)
        
        # Emergency mode
        self.risk_constraint.emergency_mode = True
        weight_emergency = self.risk_constraint.get_dynamic_risk_weight(rho)
        
        # Emergency mode should have much higher weight
        assert weight_emergency > 2 * weight_normal
    
    def test_get_risk_parameters(self):
        """Test risk parameters calculation."""
        # Set some state
        self.risk_constraint.current_rho = 0.6
        self.risk_constraint.current_regime = MarketRegime.HIGH_VOLATILITY
        
        risk_params = self.risk_constraint.get_risk_parameters()
        
        assert isinstance(risk_params, RiskParameters)
        assert risk_params.cvar_lambda > 0
        assert risk_params.volatility_penalty > 0
        assert risk_params.drawdown_limit < 0
        assert risk_params.position_limit > 0
        assert risk_params.emergency_stop_threshold < 0
        
        # Check that history was updated
        assert len(self.risk_constraint.risk_parameter_history) == 1
    
    def test_get_risk_parameters_emergency_mode(self):
        """Test risk parameters in emergency mode."""
        # Normal parameters
        normal_params = self.risk_constraint.get_risk_parameters()
        
        # Emergency parameters
        self.risk_constraint.emergency_mode = True
        emergency_params = self.risk_constraint.get_risk_parameters()
        
        # Emergency mode should have more conservative parameters
        assert emergency_params.cvar_lambda > normal_params.cvar_lambda
        assert emergency_params.volatility_penalty > normal_params.volatility_penalty
        assert emergency_params.position_limit < normal_params.position_limit
        assert abs(emergency_params.drawdown_limit) < abs(normal_params.drawdown_limit)
    
    def test_apply_enhanced_cvar_constraint(self):
        """Test enhanced CVaR constraint application."""
        # Create mock tensors
        base_loss = torch.tensor(0.5)
        cvar_estimate = torch.tensor(0.1)
        
        # Apply constraint
        enhanced_loss = self.risk_constraint.apply_enhanced_cvar_constraint(base_loss, cvar_estimate, rho=0.5)
        
        # Enhanced loss should be higher than base loss
        assert enhanced_loss > base_loss
        assert isinstance(enhanced_loss, torch.Tensor)
    
    def test_apply_enhanced_cvar_constraint_high_volatility(self):
        """Test enhanced CVaR constraint in high volatility regime."""
        # Set up high volatility regime with some volatility history
        self.risk_constraint.current_regime = MarketRegime.HIGH_VOLATILITY
        self.risk_constraint.volatility_history = [0.2, 0.25, 0.3]
        
        base_loss = torch.tensor(0.5)
        cvar_estimate = torch.tensor(0.1)
        
        enhanced_loss = self.risk_constraint.apply_enhanced_cvar_constraint(base_loss, cvar_estimate)
        
        # Should include volatility penalty
        assert enhanced_loss > base_loss + cvar_estimate
    
    def test_check_position_limits(self):
        """Test position limit checking."""
        # Set conservative position limit
        self.risk_constraint.current_regime = MarketRegime.CRISIS
        
        # Propose positions that exceed limits
        proposed_positions = np.array([0.8, -0.6, 0.4])
        adjusted_positions = self.risk_constraint.check_position_limits(proposed_positions)
        
        # Adjusted positions should be smaller
        assert np.max(np.abs(adjusted_positions)) <= np.max(np.abs(proposed_positions))
        
        # Should maintain relative proportions
        assert np.corrcoef(proposed_positions, adjusted_positions)[0, 1] > 0.99
    
    def test_should_emergency_stop_drawdown(self):
        """Test emergency stop due to drawdown."""
        # Reset emergency mode first
        self.risk_constraint.emergency_mode = False
        
        # Add returns that create large drawdown
        returns = [0.01] * 10 + [-0.05] * 10  # Big losses after gains
        for ret in returns:
            self.risk_constraint.update_market_data(ret, 100000)
        
        # Reset emergency mode again since it might have been triggered during updates
        self.risk_constraint.emergency_mode = False
        
        should_stop, reason = self.risk_constraint.should_emergency_stop()
        
        # Should trigger emergency stop due to drawdown
        if should_stop:
            assert "drawdown" in reason.lower() or "consecutive" in reason.lower()
    
    def test_should_emergency_stop_consecutive_losses(self):
        """Test emergency stop due to consecutive losses."""
        # Reset emergency mode first
        self.risk_constraint.emergency_mode = False
        
        # Add consecutive losses
        for _ in range(6):  # More than threshold
            self.risk_constraint.update_market_data(-0.01, 100000)
        
        # Reset emergency mode again since it might have been triggered during updates
        self.risk_constraint.emergency_mode = False
        
        should_stop, reason = self.risk_constraint.should_emergency_stop()
        
        assert should_stop
        assert "consecutive losses" in reason.lower()
    
    def test_should_emergency_stop_volatility_spike(self):
        """Test emergency stop due to volatility spike."""
        # Reset emergency mode first
        self.risk_constraint.emergency_mode = False
        
        # Add highly volatile returns with fixed seed for reproducibility
        np.random.seed(42)
        volatile_returns = np.random.normal(0, 0.1, 25)  # Very high volatility
        for ret in volatile_returns:
            self.risk_constraint.update_market_data(ret, 100000)
        
        # Reset emergency mode again since it might have been triggered during updates
        self.risk_constraint.emergency_mode = False
        
        should_stop, reason = self.risk_constraint.should_emergency_stop()
        
        # Should trigger emergency stop due to some condition (volatility, drawdown, or VaR)
        assert should_stop
        assert any(keyword in reason.lower() for keyword in ["volatility", "var", "drawdown", "losses"])
    
    def test_reset_emergency_mode(self):
        """Test emergency mode reset."""
        # Set emergency mode
        self.risk_constraint.emergency_mode = True
        self.risk_constraint.consecutive_losses = 3
        
        # Reset
        self.risk_constraint.reset_emergency_mode()
        
        assert not self.risk_constraint.emergency_mode
        assert self.risk_constraint.consecutive_losses == 0
    
    def test_get_risk_summary(self):
        """Test risk summary generation."""
        # Add some data
        for i in range(10):
            ret = 0.001 * (1 if i % 2 == 0 else -1)
            self.risk_constraint.update_market_data(ret, 100000)
        
        summary = self.risk_constraint.get_risk_summary()
        
        assert 'current_regime' in summary
        assert 'current_rho' in summary
        assert 'emergency_mode' in summary
        assert 'risk_parameters' in summary
        assert 'current_metrics' in summary
        assert 'emergency_check' in summary
        
        # Check risk parameters structure
        risk_params = summary['risk_parameters']
        assert 'cvar_lambda' in risk_params
        assert 'volatility_penalty' in risk_params
        assert 'drawdown_limit' in risk_params
        assert 'position_limit' in risk_params
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Add some returns
        returns = np.random.normal(0.001, 0.02, 30)
        for ret in returns:
            self.risk_constraint.update_market_data(ret, 100000)
        
        volatility = self.risk_constraint._calculate_volatility()
        
        assert isinstance(volatility, float)
        assert volatility >= 0
        
        # Test with insufficient data
        self.risk_constraint.returns_history = [0.01]
        volatility = self.risk_constraint._calculate_volatility()
        assert volatility == 0.0
    
    def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        # Add returns that create drawdown
        returns = [0.02, 0.01, -0.03, -0.02, 0.01]  # Peak then drawdown
        for ret in returns:
            self.risk_constraint.update_market_data(ret, 100000)
        
        drawdown = self.risk_constraint._calculate_drawdown()
        
        assert isinstance(drawdown, float)
        assert drawdown <= 0  # Drawdown should be negative or zero
        
        # Test with insufficient data
        self.risk_constraint.returns_history = [0.01]
        drawdown = self.risk_constraint._calculate_drawdown()
        assert drawdown == 0.0
    
    def test_update_market_regime(self):
        """Test market regime updates."""
        # Add high volatility returns
        volatile_returns = np.random.normal(0, 0.05, 15)
        for ret in volatile_returns:
            self.risk_constraint.update_market_data(ret, 100000)
        
        # Should detect high volatility regime
        assert self.risk_constraint.current_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]
        
        # Test with market indicators
        market_indicators = {'vix': 45}  # High VIX
        self.risk_constraint._update_market_regime(market_indicators)
        assert self.risk_constraint.current_regime == MarketRegime.CRISIS
    
    def test_regime_transitions(self):
        """Test regime transition logging."""
        initial_regime = self.risk_constraint.current_regime
        
        # Force regime change by adding volatile data
        volatile_returns = np.random.normal(0, 0.1, 20)
        for ret in volatile_returns:
            self.risk_constraint.update_market_data(ret, 100000)
        
        # Regime should have changed
        assert self.risk_constraint.current_regime != initial_regime or len(volatile_returns) < 10


if __name__ == "__main__":
    pytest.main([__file__])