"""
增强训练器的单元测试
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.rl_trading_system.training.enhanced_trainer import (
    EnhancedRLTrainer,
    EnhancedTrainingConfig
)
from src.rl_trading_system.training.trainer import TrainingConfig
from src.rl_trading_system.metrics.portfolio_metrics import (
    PortfolioMetrics,
    AgentBehaviorMetrics,
    RiskControlMetrics
)


class TestEnhancedTrainingConfig:
    """增强训练配置测试"""
    
    def test_enhanced_config_creation(self):
        """测试增强配置创建"""
        config = EnhancedTrainingConfig(
            enable_portfolio_metrics=True,
            enable_agent_behavior_metrics=True,
            enable_risk_control_metrics=True,
            metrics_calculation_frequency=10
        )
        
        assert config.enable_portfolio_metrics is True
        assert config.enable_agent_behavior_metrics is True
        assert config.enable_risk_control_metrics is True
        assert config.metrics_calculation_frequency == 10
    
    def test_enhanced_config_validation(self):
        """测试增强配置验证"""
        # 测试无效的指标计算频率
        with pytest.raises(ValueError, match="metrics_calculation_frequency必须为正数"):
            EnhancedTrainingConfig(metrics_calculation_frequency=0)
        
        # 测试无效的基准数据路径
        with pytest.raises(ValueError, match="benchmark_data_path不能为空"):
            EnhancedTrainingConfig(
                enable_portfolio_metrics=True,
                benchmark_data_path=""
            )


class TestEnhancedRLTrainer:
    """增强强化学习训练器测试"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建模拟配置
        self.config = EnhancedTrainingConfig(
            n_episodes=100,
            enable_portfolio_metrics=True,
            enable_agent_behavior_metrics=True,
            enable_risk_control_metrics=True,
            metrics_calculation_frequency=10
        )
        
        # 创建模拟组件
        self.mock_environment = Mock()
        self.mock_environment.reset.return_value = {
            'features': np.random.randn(60, 5, 12),
            'positions': np.zeros(5),
            'market_state': np.random.randn(10)
        }
        self.mock_environment.step.return_value = (
            {
                'features': np.random.randn(60, 5, 12),
                'positions': np.random.randn(5),
                'market_state': np.random.randn(10)
            },
            0.01,  # reward
            False,  # done
            {'portfolio_value': 1010000}  # info
        )
        
        self.mock_agent = Mock()
        self.mock_agent.get_action.return_value = torch.randn(5)
        self.mock_agent.update.return_value = {
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'alpha': 0.2,
            'policy_entropy': 2.1
        }
        self.mock_agent.can_update.return_value = True
        self.mock_agent.add_experience = Mock()
        
        self.mock_data_split = Mock()
        
        # 创建训练器
        self.trainer = EnhancedRLTrainer(
            config=self.config,
            environment=self.mock_environment,
            agent=self.mock_agent,
            data_split=self.mock_data_split
        )
    
    def test_trainer_initialization(self):
        """测试训练器初始化"""
        assert self.trainer.config == self.config
        assert self.trainer.environment == self.mock_environment
        assert self.trainer.agent == self.mock_agent
        assert hasattr(self.trainer, 'metrics_calculator')
        assert hasattr(self.trainer, 'portfolio_values_history')
        assert hasattr(self.trainer, 'benchmark_values_history')
        assert hasattr(self.trainer, 'entropy_history')
        assert hasattr(self.trainer, 'position_weights_history')
    
    def test_calculate_portfolio_metrics_success(self):
        """测试投资组合指标计算成功"""
        # 准备测试数据
        self.trainer.portfolio_values_history = [1000000, 1010000, 1020000, 1015000, 1025000]
        self.trainer.benchmark_values_history = [1000000, 1008000, 1016000, 1012000, 1020000]
        self.trainer.dates_history = [
            datetime(2023, 1, 1),
            datetime(2023, 1, 2),
            datetime(2023, 1, 3),
            datetime(2023, 1, 4),
            datetime(2023, 1, 5)
        ]
        
        metrics = self.trainer._calculate_portfolio_metrics()
        
        assert isinstance(metrics, PortfolioMetrics)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.alpha, float)
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.annualized_return, float)
    
    def test_calculate_portfolio_metrics_insufficient_data(self):
        """测试数据不足时的投资组合指标计算"""
        # 数据不足
        self.trainer.portfolio_values_history = [1000000]
        self.trainer.benchmark_values_history = [1000000]
        
        metrics = self.trainer._calculate_portfolio_metrics()
        
        assert metrics is None
    
    def test_calculate_agent_behavior_metrics_success(self):
        """测试智能体行为指标计算成功"""
        # 准备测试数据
        self.trainer.entropy_history = [2.5, 2.3, 2.1, 1.9, 1.8]
        self.trainer.position_weights_history = [
            np.array([0.2, 0.3, 0.5]),
            np.array([0.25, 0.25, 0.5]),
            np.array([0.3, 0.2, 0.5]),
            np.array([0.35, 0.15, 0.5]),
            np.array([0.4, 0.1, 0.5])
        ]
        
        metrics = self.trainer._calculate_agent_behavior_metrics()
        
        assert isinstance(metrics, AgentBehaviorMetrics)
        assert isinstance(metrics.mean_entropy, float)
        assert isinstance(metrics.entropy_trend, float)
        assert isinstance(metrics.mean_position_concentration, float)
        assert isinstance(metrics.turnover_rate, float)
    
    def test_calculate_agent_behavior_metrics_no_data(self):
        """测试无数据时的智能体行为指标计算"""
        # 无数据
        self.trainer.entropy_history = []
        self.trainer.position_weights_history = []
        
        metrics = self.trainer._calculate_agent_behavior_metrics()
        
        assert metrics is None
    
    def test_calculate_risk_control_metrics_success(self):
        """测试风险控制指标计算成功"""
        # 模拟风险控制器
        mock_drawdown_controller = Mock()
        mock_drawdown_controller.adaptive_risk_budget.risk_budget_history = [0.1, 0.08, 0.06, 0.05, 0.07]
        mock_drawdown_controller.adaptive_risk_budget.risk_usage_history = [0.08, 0.07, 0.05, 0.04, 0.06]
        mock_drawdown_controller.control_signal_queue = [
            {'type': 'position_adjustment', 'timestamp': datetime.now()},
            {'type': 'stop_loss', 'timestamp': datetime.now()}
        ]
        mock_drawdown_controller.market_regime_detector = Mock()
        mock_drawdown_controller.market_regime_detector.regime_history = ['bull', 'bull', 'bear', 'bear', 'neutral']
        
        self.trainer.environment.drawdown_controller = mock_drawdown_controller
        
        metrics = self.trainer._calculate_risk_control_metrics()
        
        assert isinstance(metrics, RiskControlMetrics)
        assert isinstance(metrics.avg_risk_budget_utilization, float)
        assert isinstance(metrics.risk_budget_efficiency, float)
        assert isinstance(metrics.control_signal_frequency, float)
        assert isinstance(metrics.market_regime_stability, float)
    
    def test_calculate_risk_control_metrics_no_controller(self):
        """测试无风险控制器时的指标计算"""
        # 无风险控制器
        self.trainer.environment.drawdown_controller = None
        
        metrics = self.trainer._calculate_risk_control_metrics()
        
        assert metrics is None
    
    def test_log_enhanced_metrics_success(self):
        """测试增强指标日志记录成功"""
        # 准备测试数据
        portfolio_metrics = PortfolioMetrics(
            sharpe_ratio=1.5,
            max_drawdown=0.15,
            alpha=0.08,
            beta=1.2,
            annualized_return=0.12,
            timestamp=datetime.now()
        )
        
        agent_metrics = AgentBehaviorMetrics(
            mean_entropy=2.1,
            entropy_trend=-0.1,
            mean_position_concentration=0.6,
            turnover_rate=0.25,
            timestamp=datetime.now()
        )
        
        risk_metrics = RiskControlMetrics(
            avg_risk_budget_utilization=0.8,
            risk_budget_efficiency=1.2,
            control_signal_frequency=0.1,
            market_regime_stability=0.7,
            timestamp=datetime.now()
        )
        
        # 测试日志记录（不应该抛出异常）
        with patch('src.rl_trading_system.training.enhanced_trainer.logger') as mock_logger:
            self.trainer._log_enhanced_metrics(
                episode=50,
                portfolio_metrics=portfolio_metrics,
                agent_metrics=agent_metrics,
                risk_metrics=risk_metrics
            )
            
            # 验证日志被调用
            assert mock_logger.info.called
    
    def test_update_metrics_histories_success(self):
        """测试指标历史更新成功"""
        # 模拟episode信息
        episode_info = {
            'portfolio_value': 1010000,
            'positions': np.array([0.2, 0.3, 0.5]),
            'benchmark_value': 1008000
        }
        
        # 模拟智能体更新信息
        update_info = {
            'policy_entropy': 2.1
        }
        
        initial_portfolio_len = len(self.trainer.portfolio_values_history)
        initial_positions_len = len(self.trainer.position_weights_history)
        initial_entropy_len = len(self.trainer.entropy_history)
        
        self.trainer._update_metrics_histories(episode_info, update_info)
        
        # 验证历史数据被更新
        assert len(self.trainer.portfolio_values_history) == initial_portfolio_len + 1
        assert len(self.trainer.position_weights_history) == initial_positions_len + 1
        assert len(self.trainer.entropy_history) == initial_entropy_len + 1
        assert self.trainer.portfolio_values_history[-1] == 1010000
        assert np.array_equal(self.trainer.position_weights_history[-1], np.array([0.2, 0.3, 0.5]))
        assert self.trainer.entropy_history[-1] == 2.1
    
    def test_should_calculate_metrics_true(self):
        """测试应该计算指标的情况"""
        # 设置为指标计算频率的倍数
        episode = self.config.metrics_calculation_frequency
        
        should_calculate = self.trainer._should_calculate_metrics(episode)
        
        assert should_calculate is True
    
    def test_should_calculate_metrics_false(self):
        """测试不应该计算指标的情况"""
        # 设置为非指标计算频率的倍数
        episode = self.config.metrics_calculation_frequency - 1
        
        should_calculate = self.trainer._should_calculate_metrics(episode)
        
        assert should_calculate is False
    
    @patch('src.rl_trading_system.training.enhanced_trainer.logger')
    def test_run_episode_with_metrics_calculation(self, mock_logger):
        """测试带指标计算的episode运行"""
        # 设置为应该计算指标的episode
        episode = self.config.metrics_calculation_frequency
        
        # 准备历史数据
        self.trainer.portfolio_values_history = [1000000, 1010000, 1020000]
        self.trainer.benchmark_values_history = [1000000, 1008000, 1016000]
        self.trainer.dates_history = [datetime.now() for _ in range(3)]
        self.trainer.entropy_history = [2.5, 2.3, 2.1]
        self.trainer.position_weights_history = [
            np.array([0.2, 0.3, 0.5]),
            np.array([0.25, 0.25, 0.5]),
            np.array([0.3, 0.2, 0.5])
        ]
        
        # 运行episode
        reward, length = self.trainer._run_episode(episode, training=True)
        
        # 验证返回值
        assert isinstance(reward, float)
        assert isinstance(length, int)
        
        # 验证指标计算相关的日志被调用
        assert mock_logger.info.called


if __name__ == '__main__':
    pytest.main([__file__])