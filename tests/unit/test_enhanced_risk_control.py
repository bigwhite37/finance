"""
增强风险控制模块的单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.rl_trading_system.risk_control.enhanced_drawdown_controller import (
    EnhancedDrawdownController
)
from src.rl_trading_system.risk_control.enhanced_adaptive_risk_budget import (
    EnhancedAdaptiveRiskBudget,
    EnhancedAdaptiveRiskBudgetConfig
)
from src.rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig


class TestEnhancedAdaptiveRiskBudget:
    """增强自适应风险预算测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = EnhancedAdaptiveRiskBudgetConfig(
            base_risk_budget=0.1,
            enable_detailed_logging=True,
            log_budget_changes=True,
            log_usage_analysis=True
        )
        self.risk_budget = EnhancedAdaptiveRiskBudget(self.config)
    
    def test_enhanced_config_creation(self):
        """测试增强配置创建"""
        assert self.config.enable_detailed_logging is True
        assert self.config.log_budget_changes is True
        assert self.config.log_usage_analysis is True
        assert self.config.budget_change_threshold == 0.01
    
    @patch('src.rl_trading_system.risk_control.enhanced_adaptive_risk_budget.logger')
    def test_calculate_adaptive_risk_budget_with_logging(self, mock_logger):
        """测试带日志的自适应风险预算计算"""
        # 添加一些历史数据
        from src.rl_trading_system.risk_control.adaptive_risk_budget import PerformanceMetrics, MarketMetrics
        
        perf_metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown=0.1,
            volatility=0.15,
            win_rate=0.6,
            profit_factor=1.8,
            consecutive_losses=2,
            consecutive_wins=5,
            total_return=0.12,
            downside_deviation=0.08,
            sortino_ratio=1.8,
            var_95=0.02,
            expected_shortfall=0.03,
            timestamp=datetime.now()
        )
        
        market_metrics = MarketMetrics(
            market_volatility=0.2,
            market_trend=0.05,
            correlation_with_market=0.7,
            liquidity_score=0.8,
            uncertainty_index=0.3,
            regime_stability=0.7,
            timestamp=datetime.now()
        )
        
        self.risk_budget.update_performance_metrics(perf_metrics)
        self.risk_budget.update_market_metrics(market_metrics)
        
        # 计算风险预算
        budget = self.risk_budget.calculate_adaptive_risk_budget()
        
        # 验证返回值
        assert isinstance(budget, float)
        assert 0 < budget <= 1
        
        # 验证日志被调用
        assert mock_logger.info.called
    
    @patch('src.rl_trading_system.risk_control.enhanced_adaptive_risk_budget.logger')
    def test_log_budget_usage_analysis(self, mock_logger):
        """测试风险预算使用分析日志"""
        # 添加使用历史
        self.risk_budget.risk_budget_history = [0.1, 0.08, 0.06, 0.05, 0.07]
        self.risk_budget.risk_usage_history = [0.08, 0.07, 0.05, 0.04, 0.06]
        
        self.risk_budget._log_budget_usage_analysis()
        
        # 验证日志被调用
        assert mock_logger.info.called
    
    def test_get_detailed_budget_info(self):
        """测试获取详细预算信息"""
        # 添加一些历史数据
        self.risk_budget.risk_budget_history = [0.1, 0.08, 0.06]
        self.risk_budget.risk_usage_history = [0.08, 0.07, 0.05]
        # 添加效率历史以确保efficiency_score存在
        self.risk_budget.efficiency_history = [1.2, 1.1, 1.0]
        
        info = self.risk_budget.get_detailed_budget_info()
        
        assert isinstance(info, dict)
        assert 'current_budget' in info
        assert 'average_utilization' in info
        assert 'budget_trend' in info
        assert 'efficiency_score' in info


class TestEnhancedDrawdownController:
    """增强回撤控制器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.config = DrawdownControlConfig(
            portfolio_stop_loss=0.15,
            base_risk_budget=0.1
        )
        self.controller = EnhancedDrawdownController(self.config)
    
    def test_enhanced_controller_initialization(self):
        """测试增强控制器初始化"""
        assert hasattr(self.controller, 'detailed_logging_enabled')
        assert hasattr(self.controller, 'control_decision_history')
        assert hasattr(self.controller, 'market_state_history')
    
    @patch('src.rl_trading_system.risk_control.enhanced_drawdown_controller.logger')
    def test_log_control_decision_details(self, mock_logger):
        """测试控制决策详细日志"""
        from src.rl_trading_system.risk_control.drawdown_controller import (
            ControlSignal, ControlSignalType, ControlSignalPriority
        )
        
        # 创建测试信号
        signal = ControlSignal(
            signal_type=ControlSignalType.STOP_LOSS,
            priority=ControlSignalPriority.CRITICAL,
            timestamp=datetime.now(),
            source_component='test',
            content={'action': 'reduce_positions', 'risk_reduction_factor': 0.5}
        )
        
        self.controller._log_control_decision_details([signal])
        
        # 验证日志被调用
        assert mock_logger.info.called
    
    @patch('src.rl_trading_system.risk_control.enhanced_drawdown_controller.logger')
    def test_log_market_state_analysis(self, mock_logger):
        """测试市场状态分析日志"""
        from src.rl_trading_system.risk_control.drawdown_controller import MarketState
        
        market_state = MarketState(
            prices={'AAPL': 150.0, 'GOOGL': 2800.0},
            volumes={'AAPL': 1000000, 'GOOGL': 500000},
            timestamp=datetime.now(),
            market_indicators={'volatility': 0.2, 'trend': 0.05}
        )
        
        self.controller._log_market_state_analysis(market_state)
        
        # 验证日志被调用
        assert mock_logger.info.called
    
    @patch('src.rl_trading_system.risk_control.enhanced_drawdown_controller.logger')
    def test_log_risk_budget_details(self, mock_logger):
        """测试风险预算详细日志"""
        self.controller._log_risk_budget_details(
            current_budget=0.08,
            usage_rate=0.75,
            efficiency=1.2
        )
        
        # 验证日志被调用
        assert mock_logger.info.called
    
    def test_get_control_summary(self):
        """测试获取控制摘要"""
        summary = self.controller.get_control_summary()
        
        assert isinstance(summary, dict)
        assert 'total_decisions' in summary
        assert 'decision_types' in summary
        assert 'market_state_changes' in summary
        assert 'risk_budget_info' in summary


if __name__ == '__main__':
    pytest.main([__file__])