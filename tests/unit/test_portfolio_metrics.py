"""
投资组合指标计算模块的单元测试
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.rl_trading_system.metrics.portfolio_metrics import (
    PortfolioMetricsCalculator,
    PortfolioMetrics,
    AgentBehaviorMetrics,
    RiskControlMetrics
)


class TestPortfolioMetricsCalculator:
    """投资组合指标计算器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.calculator = PortfolioMetricsCalculator()
        
        # 创建测试数据
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.portfolio_values = [1000000 + i * 1000 + np.random.normal(0, 500) for i in range(100)]
        self.benchmark_values = [1000000 + i * 800 + np.random.normal(0, 300) for i in range(100)]
        
        # 确保没有负值
        self.portfolio_values = [max(val, 500000) for val in self.portfolio_values]
        self.benchmark_values = [max(val, 500000) for val in self.benchmark_values]
    
    def test_calculate_sharpe_ratio_success(self):
        """测试夏普比率计算成功"""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        risk_free_rate = 0.03
        
        sharpe = self.calculator.calculate_sharpe_ratio(returns, risk_free_rate)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        assert not np.isinf(sharpe)
    
    def test_calculate_sharpe_ratio_zero_volatility(self):
        """测试零波动率时的夏普比率计算"""
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])  # 无波动
        risk_free_rate = 0.03
        
        sharpe = self.calculator.calculate_sharpe_ratio(returns, risk_free_rate)
        
        # 零波动率时应该返回0或处理为特殊情况
        assert sharpe == 0.0 or np.isinf(sharpe)
    
    def test_calculate_max_drawdown_success(self):
        """测试最大回撤计算成功"""
        values = [1000, 1100, 1050, 900, 950, 1200, 1150]
        
        max_dd = self.calculator.calculate_max_drawdown(values)
        
        # 最大回撤应该是从1100到900，即18.18%
        expected_dd = (1100 - 900) / 1100
        assert abs(max_dd - expected_dd) < 0.001
        assert max_dd >= 0  # 回撤应该是正值
    
    def test_calculate_max_drawdown_no_drawdown(self):
        """测试无回撤情况"""
        values = [1000, 1100, 1200, 1300, 1400]  # 单调递增
        
        max_dd = self.calculator.calculate_max_drawdown(values)
        
        assert max_dd == 0.0
    
    def test_calculate_alpha_beta_success(self):
        """测试Alpha和Beta计算成功"""
        portfolio_returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        benchmark_returns = np.array([0.008, 0.015, -0.005, 0.012, 0.003])
        risk_free_rate = 0.03
        
        alpha, beta = self.calculator.calculate_alpha_beta(
            portfolio_returns, benchmark_returns, risk_free_rate
        )
        
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert not np.isnan(alpha)
        assert not np.isnan(beta)
        assert not np.isinf(alpha)
        assert not np.isinf(beta)
    
    def test_calculate_alpha_beta_zero_benchmark_variance(self):
        """测试基准收益率无变化时的Alpha和Beta计算"""
        portfolio_returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        benchmark_returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])  # 无变化
        risk_free_rate = 0.03
        
        alpha, beta = self.calculator.calculate_alpha_beta(
            portfolio_returns, benchmark_returns, risk_free_rate
        )
        
        # 基准无变化时，Beta应该为0，Alpha应该等于超额收益
        assert beta == 0.0
        assert isinstance(alpha, float)
    
    def test_calculate_annualized_return_success(self):
        """测试年化收益率计算成功"""
        start_value = 1000000
        end_value = 1200000
        days = 252  # 一年
        
        annual_return = self.calculator.calculate_annualized_return(
            start_value, end_value, days
        )
        
        expected_return = (end_value / start_value) - 1
        assert abs(annual_return - expected_return) < 0.001
    
    def test_calculate_annualized_return_multi_year(self):
        """测试多年期年化收益率计算"""
        start_value = 1000000
        end_value = 1440000  # 44%总收益
        days = 504  # 两年
        
        annual_return = self.calculator.calculate_annualized_return(
            start_value, end_value, days
        )
        
        # 两年44%收益，年化收益率应该约为20%
        expected_annual = (end_value / start_value) ** (252 / days) - 1
        assert abs(annual_return - expected_annual) < 0.001
    
    def test_calculate_portfolio_metrics_integration(self):
        """测试投资组合指标计算的集成测试"""
        metrics = self.calculator.calculate_portfolio_metrics(
            portfolio_values=self.portfolio_values,
            benchmark_values=self.benchmark_values,
            dates=self.dates,
            risk_free_rate=0.03
        )
        
        assert isinstance(metrics, PortfolioMetrics)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.alpha, float)
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.annualized_return, float)
        
        # 验证指标的合理性
        assert metrics.max_drawdown >= 0
        assert not np.isnan(metrics.sharpe_ratio)
        assert not np.isnan(metrics.alpha)
        assert not np.isnan(metrics.beta)
        assert not np.isnan(metrics.annualized_return)
    
    def test_calculate_agent_behavior_metrics_success(self):
        """测试智能体行为指标计算成功"""
        # 模拟智能体数据
        entropy_values = [2.5, 2.3, 2.1, 1.9, 1.8]
        position_weights_history = [
            np.array([0.2, 0.3, 0.5]),
            np.array([0.25, 0.25, 0.5]),
            np.array([0.3, 0.2, 0.5]),
            np.array([0.35, 0.15, 0.5]),
            np.array([0.4, 0.1, 0.5])
        ]
        
        metrics = self.calculator.calculate_agent_behavior_metrics(
            entropy_values=entropy_values,
            position_weights_history=position_weights_history
        )
        
        assert isinstance(metrics, AgentBehaviorMetrics)
        assert isinstance(metrics.mean_entropy, float)
        assert isinstance(metrics.entropy_trend, float)
        assert isinstance(metrics.mean_position_concentration, float)
        assert isinstance(metrics.turnover_rate, float)
        
        # 验证指标合理性
        assert metrics.mean_entropy > 0
        assert 0 <= metrics.mean_position_concentration <= 1
        assert metrics.turnover_rate >= 0
    
    def test_calculate_risk_control_metrics_success(self):
        """测试风险控制指标计算成功"""
        # 模拟风险控制数据
        risk_budget_history = [0.1, 0.08, 0.06, 0.05, 0.07]
        risk_usage_history = [0.08, 0.07, 0.05, 0.04, 0.06]
        control_signals = [
            {'type': 'position_adjustment', 'timestamp': datetime.now()},
            {'type': 'stop_loss', 'timestamp': datetime.now()},
            {'type': 'risk_budget_change', 'timestamp': datetime.now()}
        ]
        market_regime_history = ['bull', 'bull', 'bear', 'bear', 'neutral']
        
        metrics = self.calculator.calculate_risk_control_metrics(
            risk_budget_history=risk_budget_history,
            risk_usage_history=risk_usage_history,
            control_signals=control_signals,
            market_regime_history=market_regime_history
        )
        
        assert isinstance(metrics, RiskControlMetrics)
        assert isinstance(metrics.avg_risk_budget_utilization, float)
        assert isinstance(metrics.risk_budget_efficiency, float)
        assert isinstance(metrics.control_signal_frequency, float)
        assert isinstance(metrics.market_regime_stability, float)
        
        # 验证指标合理性
        assert 0 <= metrics.avg_risk_budget_utilization <= 1
        assert metrics.risk_budget_efficiency >= 0
        assert metrics.control_signal_frequency >= 0
        assert 0 <= metrics.market_regime_stability <= 1


class TestPortfolioMetrics:
    """投资组合指标数据类测试"""
    
    def test_portfolio_metrics_creation(self):
        """测试投资组合指标创建"""
        metrics = PortfolioMetrics(
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            alpha=0.08,
            beta=1.2,
            annualized_return=0.12,
            timestamp=datetime.now()
        )
        
        assert metrics.sharpe_ratio == 1.5
        assert metrics.max_drawdown == 0.15
        assert metrics.alpha == 0.08
        assert metrics.beta == 1.2
        assert metrics.annualized_return == 0.12
        assert isinstance(metrics.timestamp, datetime)
    
    def test_agent_behavior_metrics_creation(self):
        """测试智能体行为指标创建"""
        metrics = AgentBehaviorMetrics(
            mean_entropy=2.1,
            entropy_trend=-0.1,
            mean_position_concentration=0.6,
            turnover_rate=0.25,
            timestamp=datetime.now()
        )
        
        assert metrics.mean_entropy == 2.1
        assert metrics.entropy_trend == -0.1
        assert metrics.mean_position_concentration == 0.6
        assert metrics.turnover_rate == 0.25
        assert isinstance(metrics.timestamp, datetime)
    
    def test_risk_control_metrics_creation(self):
        """测试风险控制指标创建"""
        metrics = RiskControlMetrics(
            avg_risk_budget_utilization=0.8,
            risk_budget_efficiency=1.2,
            control_signal_frequency=0.1,
            market_regime_stability=0.7,
            timestamp=datetime.now()
        )
        
        assert metrics.avg_risk_budget_utilization == 0.8
        assert metrics.risk_budget_efficiency == 1.2
        assert metrics.control_signal_frequency == 0.1
        assert metrics.market_regime_stability == 0.7
        assert isinstance(metrics.timestamp, datetime)


if __name__ == '__main__':
    pytest.main([__file__])