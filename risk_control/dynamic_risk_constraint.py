"""
Dynamic Risk Constraint Adjustment for O2O RL System.

This module implements adaptive risk control that adjusts CVaR weights
and risk parameters based on online sampling ratio and market regime
conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import torch

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    CRISIS = "crisis"


@dataclass
class RiskParameters:
    """Risk parameter configuration."""
    cvar_lambda: float
    volatility_penalty: float
    drawdown_limit: float
    position_limit: float
    emergency_stop_threshold: float


class DynamicRiskConstraint:
    """
    动态风险约束调整器
    
    Implements adaptive risk control with dynamic CVaR weight calculation,
    market regime-aware risk parameter adjustment, and emergency risk
    control for extreme market conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Dynamic Risk Constraint system.
        
        Args:
            config: Configuration dictionary containing:
                - base_cvar_lambda: Base CVaR weight (default: 1.0)
                - rho_sensitivity: Sensitivity to online ratio changes (default: 2.0)
                - regime_multipliers: Risk multipliers for different regimes
                - emergency_thresholds: Thresholds for emergency risk control
                - volatility_lookback: Days for volatility calculation (default: 20)
                - drawdown_lookback: Days for drawdown monitoring (default: 30)
        """
        self.config = config
        
        # Base parameters
        self.base_cvar_lambda = config.get('base_cvar_lambda', 1.0)
        self.rho_sensitivity = config.get('rho_sensitivity', 2.0)
        self.volatility_lookback = config.get('volatility_lookback', 20)
        self.drawdown_lookback = config.get('drawdown_lookback', 30)
        
        # Regime-specific multipliers
        self.regime_multipliers = config.get('regime_multipliers', {
            MarketRegime.NORMAL.value: 1.0,
            MarketRegime.HIGH_VOLATILITY.value: 1.5,
            MarketRegime.LOW_VOLATILITY.value: 0.8,
            MarketRegime.TRENDING.value: 0.9,
            MarketRegime.CRISIS.value: 2.5
        })
        
        # Emergency thresholds
        self.emergency_thresholds = config.get('emergency_thresholds', {
            'max_drawdown': -0.15,
            'volatility_spike': 0.05,
            'var_breach': -0.03,
            'consecutive_losses': 5
        })
        
        # State tracking
        self.current_regime = MarketRegime.NORMAL
        self.current_rho = 0.2
        self.returns_history = []
        self.volatility_history = []
        self.drawdown_history = []
        self.consecutive_losses = 0
        self.emergency_mode = False
        
        # Risk parameter history
        self.risk_parameter_history = []
        
        logger.info(f"DynamicRiskConstraint initialized with base_cvar_lambda={self.base_cvar_lambda}")
    
    def update_market_data(self, returns: float, portfolio_value: float, 
                          market_indicators: Optional[Dict[str, float]] = None) -> None:
        """
        Update market data for risk parameter calculation.
        
        Args:
            returns: Current period returns
            portfolio_value: Current portfolio value
            market_indicators: Optional market indicators for regime detection
        """
        if not np.isfinite(returns):
            logger.warning(f"Invalid returns value: {returns}")
            return
        
        # Update returns history
        self.returns_history.append(returns)
        if len(self.returns_history) > max(self.volatility_lookback, self.drawdown_lookback):
            self.returns_history = self.returns_history[-max(self.volatility_lookback, self.drawdown_lookback):]
        
        # Update consecutive losses
        if returns < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Calculate and update volatility
        if len(self.returns_history) >= 5:
            current_volatility = self._calculate_volatility()
            self.volatility_history.append(current_volatility)
            if len(self.volatility_history) > self.volatility_lookback:
                self.volatility_history = self.volatility_history[-self.volatility_lookback:]
        
        # Calculate and update drawdown
        if len(self.returns_history) >= 2:
            current_drawdown = self._calculate_drawdown()
            self.drawdown_history.append(current_drawdown)
            if len(self.drawdown_history) > self.drawdown_lookback:
                self.drawdown_history = self.drawdown_history[-self.drawdown_lookback:]
        
        # Update market regime
        self._update_market_regime(market_indicators)
        
        # Check emergency conditions
        self._check_emergency_conditions()
    
    def update_sampling_ratio(self, rho: float) -> None:
        """
        Update the current online sampling ratio.
        
        Args:
            rho: Current online sampling ratio (0.0 to 1.0)
        """
        if not 0.0 <= rho <= 1.0:
            logger.warning(f"Invalid rho value: {rho}, clipping to [0, 1]")
            rho = np.clip(rho, 0.0, 1.0)
        
        self.current_rho = rho
        logger.debug(f"Updated sampling ratio to {rho:.3f}")
    
    def get_dynamic_risk_weight(self, rho: Optional[float] = None, 
                               regime: Optional[MarketRegime] = None) -> float:
        """
        Calculate dynamic CVaR weight λ(ρ) based on online ratio and market regime.
        
        Args:
            rho: Online sampling ratio (uses current if None)
            regime: Market regime (uses current if None)
            
        Returns:
            Dynamic CVaR weight
        """
        if rho is None:
            rho = self.current_rho
        if regime is None:
            regime = self.current_regime
        
        # Base calculation: λ(ρ) = λ_base * (1 + α * ρ^β)
        # Higher online ratio -> higher risk weight
        rho_factor = 1 + self.rho_sensitivity * (rho ** 1.5)
        
        # Apply regime multiplier
        regime_multiplier = self.regime_multipliers.get(regime.value, 1.0)
        
        # Calculate final weight
        dynamic_weight = self.base_cvar_lambda * rho_factor * regime_multiplier
        
        # Apply emergency multiplier if in emergency mode
        if self.emergency_mode:
            dynamic_weight *= 3.0  # Significantly increase risk aversion
        
        logger.debug(f"Dynamic CVaR weight: base={self.base_cvar_lambda:.3f}, "
                    f"rho_factor={rho_factor:.3f}, regime_mult={regime_multiplier:.3f}, "
                    f"final={dynamic_weight:.3f}")
        
        return dynamic_weight
    
    def get_risk_parameters(self) -> RiskParameters:
        """
        Get current risk parameters based on market conditions.
        
        Returns:
            RiskParameters object with current settings
        """
        cvar_lambda = self.get_dynamic_risk_weight()
        
        # Base parameters
        base_volatility_penalty = 0.1
        base_drawdown_limit = -0.10
        base_position_limit = 1.0
        base_emergency_threshold = -0.05
        
        # Adjust based on regime
        regime_mult = self.regime_multipliers.get(self.current_regime.value, 1.0)
        
        # Calculate adjusted parameters
        volatility_penalty = base_volatility_penalty * regime_mult
        drawdown_limit = base_drawdown_limit * (2.0 - regime_mult)  # Tighter limits in high-risk regimes
        position_limit = base_position_limit / regime_mult  # Smaller positions in high-risk regimes
        emergency_threshold = base_emergency_threshold * (2.0 - regime_mult)
        
        # Apply emergency adjustments
        if self.emergency_mode:
            volatility_penalty *= 2.0
            drawdown_limit *= 0.5  # Much tighter drawdown limit
            position_limit *= 0.3  # Much smaller positions
            emergency_threshold *= 0.5
        
        risk_params = RiskParameters(
            cvar_lambda=cvar_lambda,
            volatility_penalty=volatility_penalty,
            drawdown_limit=drawdown_limit,
            position_limit=position_limit,
            emergency_stop_threshold=emergency_threshold
        )
        
        # Store in history
        self.risk_parameter_history.append({
            'timestamp': pd.Timestamp.now(),
            'regime': self.current_regime.value,
            'rho': self.current_rho,
            'emergency_mode': self.emergency_mode,
            'parameters': risk_params
        })
        
        # Keep history manageable
        if len(self.risk_parameter_history) > 1000:
            self.risk_parameter_history = self.risk_parameter_history[-500:]
        
        return risk_params
    
    def apply_enhanced_cvar_constraint(self, loss: torch.Tensor, 
                                     cvar_estimate: torch.Tensor, 
                                     rho: Optional[float] = None) -> torch.Tensor:
        """
        Apply enhanced CVaR constraint with dynamic weighting.
        
        Args:
            loss: Base policy loss
            cvar_estimate: CVaR estimate from the agent
            rho: Online sampling ratio (uses current if None)
            
        Returns:
            Enhanced loss with dynamic CVaR constraint
        """
        if rho is None:
            rho = self.current_rho
        
        # Get dynamic risk weight
        dynamic_lambda = self.get_dynamic_risk_weight(rho)
        
        # Apply CVaR constraint: L_total = L_policy + λ(ρ) * CVaR
        enhanced_loss = loss + dynamic_lambda * cvar_estimate
        
        # Add volatility penalty if in high volatility regime
        if self.current_regime == MarketRegime.HIGH_VOLATILITY and len(self.volatility_history) > 0:
            volatility_penalty = self.volatility_history[-1] * 0.1
            enhanced_loss = enhanced_loss + volatility_penalty
        
        # Add emergency penalty if in emergency mode
        if self.emergency_mode:
            emergency_penalty = torch.abs(cvar_estimate) * 2.0
            enhanced_loss = enhanced_loss + emergency_penalty
        
        return enhanced_loss
    
    def check_position_limits(self, proposed_positions: np.ndarray) -> np.ndarray:
        """
        Check and adjust positions based on current risk limits.
        
        Args:
            proposed_positions: Proposed position weights
            
        Returns:
            Adjusted position weights
        """
        risk_params = self.get_risk_parameters()
        
        # Scale positions if they exceed limits
        max_position = np.max(np.abs(proposed_positions))
        if max_position > risk_params.position_limit:
            scale_factor = risk_params.position_limit / max_position
            adjusted_positions = proposed_positions * scale_factor
            logger.info(f"Scaled positions by {scale_factor:.3f} due to position limits")
        else:
            adjusted_positions = proposed_positions.copy()
        
        return adjusted_positions
    
    def should_emergency_stop(self) -> Tuple[bool, str]:
        """
        Check if emergency stop should be triggered.
        
        Returns:
            Tuple of (should_stop, reason)
        """
        if self.emergency_mode:
            return True, "Already in emergency mode"
        
        # Check drawdown
        if len(self.drawdown_history) > 0:
            current_drawdown = self.drawdown_history[-1]
            if current_drawdown < self.emergency_thresholds['max_drawdown']:
                return True, f"Maximum drawdown exceeded: {current_drawdown:.2%}"
        
        # Check volatility spike
        if len(self.volatility_history) >= 2:
            recent_vol = self.volatility_history[-1]
            if recent_vol > self.emergency_thresholds['volatility_spike']:
                return True, f"Volatility spike detected: {recent_vol:.4f}"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.emergency_thresholds['consecutive_losses']:
            return True, f"Too many consecutive losses: {self.consecutive_losses}"
        
        # Check VaR breach (using recent returns)
        if len(self.returns_history) >= 20:
            var_95 = np.percentile(self.returns_history[-20:], 5)
            if var_95 < self.emergency_thresholds['var_breach']:
                return True, f"VaR breach detected: {var_95:.4f}"
        
        return False, ""
    
    def reset_emergency_mode(self) -> None:
        """Reset emergency mode (typically after successful retraining)."""
        if self.emergency_mode:
            logger.info("Resetting emergency mode")
            self.emergency_mode = False
            self.consecutive_losses = 0
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk monitoring summary.
        
        Returns:
            Dictionary containing current risk status
        """
        risk_params = self.get_risk_parameters()
        
        summary = {
            'current_regime': self.current_regime.value,
            'current_rho': self.current_rho,
            'emergency_mode': self.emergency_mode,
            'consecutive_losses': self.consecutive_losses,
            'risk_parameters': {
                'cvar_lambda': risk_params.cvar_lambda,
                'volatility_penalty': risk_params.volatility_penalty,
                'drawdown_limit': risk_params.drawdown_limit,
                'position_limit': risk_params.position_limit,
                'emergency_threshold': risk_params.emergency_stop_threshold
            },
            'current_metrics': {
                'volatility': self.volatility_history[-1] if self.volatility_history else None,
                'drawdown': self.drawdown_history[-1] if self.drawdown_history else None,
                'recent_returns': self.returns_history[-5:] if len(self.returns_history) >= 5 else self.returns_history
            },
            'emergency_check': self.should_emergency_stop()
        }
        
        return summary
    
    def _calculate_volatility(self) -> float:
        """Calculate rolling volatility from returns history."""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns_array = np.array(self.returns_history[-self.volatility_lookback:])
        return np.std(returns_array, ddof=1) * np.sqrt(252)  # Annualized volatility
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from returns history."""
        if len(self.returns_history) < 2:
            return 0.0
        
        # Calculate cumulative returns
        returns_array = np.array(self.returns_history[-self.drawdown_lookback:])
        cumulative_returns = np.cumprod(1 + returns_array)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        return drawdown[-1]  # Return current drawdown
    
    def _update_market_regime(self, market_indicators: Optional[Dict[str, float]] = None) -> None:
        """Update market regime based on current conditions."""
        if len(self.returns_history) < 10:
            return  # Need sufficient data
        
        # Simple regime detection based on volatility and returns
        current_vol = self.volatility_history[-1] if self.volatility_history else 0.0
        recent_returns = np.array(self.returns_history[-10:])
        avg_return = np.mean(recent_returns)
        
        # Regime classification logic
        if current_vol > 0.3:  # High volatility threshold
            new_regime = MarketRegime.CRISIS if avg_return < -0.02 else MarketRegime.HIGH_VOLATILITY
        elif current_vol < 0.1:  # Low volatility threshold
            new_regime = MarketRegime.LOW_VOLATILITY
        elif abs(avg_return) > 0.01:  # Trending threshold
            new_regime = MarketRegime.TRENDING
        else:
            new_regime = MarketRegime.NORMAL
        
        # Use market indicators if provided
        if market_indicators:
            vix = market_indicators.get('vix', 20)
            if vix > 40:
                new_regime = MarketRegime.CRISIS
            elif vix > 25:
                new_regime = MarketRegime.HIGH_VOLATILITY
        
        if new_regime != self.current_regime:
            logger.info(f"Market regime changed: {self.current_regime.value} -> {new_regime.value}")
            self.current_regime = new_regime
    
    def _check_emergency_conditions(self) -> None:
        """Check and update emergency mode status."""
        should_stop, reason = self.should_emergency_stop()
        
        if should_stop and not self.emergency_mode:
            logger.critical(f"Entering emergency mode: {reason}")
            self.emergency_mode = True
        elif not should_stop and self.emergency_mode:
            # Don't automatically exit emergency mode - require explicit reset
            pass