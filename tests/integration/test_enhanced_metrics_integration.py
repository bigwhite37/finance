"""
增强指标系统集成测试
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from datetime import datetime

from src.rl_trading_system.training.enhanced_trainer import (
    EnhancedRLTrainer,
    EnhancedTrainingConfig
)
from src.rl_trading_system.models.sac_agent import SACAgent, SACConfig
from src.rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from src.rl_trading_system.risk_control.enhanced_drawdown_controller import EnhancedDrawdownController
from src.rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig


class TestEnhancedMetricsIntegration:
    """增强指标系统集成测试"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建模拟环境
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
        self.mock_environment.total_value = 1010000
        self.mock_environment.current_positions = np.array([0.2, 0.3, 0.5, 0.0, 0.0])
        
        # 创建增强回撤控制器
        drawdown_config = DrawdownControlConfig(
            portfolio_stop_loss=0.15,
            base_risk_budget=0.1
        )
        self.mock_environment.drawdown_controller = EnhancedDrawdownController(drawdown_config)
        
        # 创建模拟智能体
        self.mock_agent = Mock()
        self.mock_agent.get_action.return_value = torch.randn(5)
        self.mock_agent.update.return_value = {
            'actor_loss': 0.1,
            'critic_loss': 0.2,
            'alpha': 0.2,
            'policy_entropy': 2.1  # 关键：确保返回熵值
        }
        self.mock_agent.can_update.return_value = True
        self.mock_agent.add_experience = Mock()
        
        # 创建增强训练配置
        self.config = EnhancedTrainingConfig(
            n_episodes=20,  # 较少的episode用于测试
            enable_portfolio_metrics=True,
            enable_agent_behavior_metrics=True,
            enable_risk_control_metrics=True,
            metrics_calculation_frequency=5,  # 每5个episode计算一次
            detailed_metrics_logging=True
        )
        
        self.mock_data_split = Mock()
        
        # 创建增强训练器
        self.trainer = EnhancedRLTrainer(
            config=self.config,
            environment=self.mock_environment,
            agent=self.mock_agent,
            data_split=self.mock_data_split
        )
    
    @patch('src.rl_trading_system.training.enhanced_trainer.logger')
    def test_full_training_with_enhanced_metrics(self, mock_logger):
        """测试完整的增强指标训练流程"""
        # 运行几个episode来积累数据
        for episode in range(1, 11):
            reward, length = self.trainer._run_episode(episode, training=True)
            
            # 验证基本返回值
            assert isinstance(reward, float)
            assert isinstance(length, int)
            
            # 模拟episode信息更新
            episode_info = {
                'portfolio_value': 1000000 + episode * 1000,
                'positions': np.random.rand(5),
                'benchmark_value': 1000000 + episode * 800
            }
            update_info = {
                'actor_loss': 0.1,
                'critic_loss': 0.2,
                'policy_entropy': 2.5 - episode * 0.1  # 模拟熵值下降
            }
            
            self.trainer._update_metrics_histories(episode_info, update_info)
        
        # 验证历史数据被正确记录
        assert len(self.trainer.portfolio_values_history) == 10
        assert len(self.trainer.entropy_history) == 10
        assert len(self.trainer.position_weights_history) == 10
        
        # 验证指标计算在第5和第10个episode被触发
        # 通过检查日志调用来验证
        assert mock_logger.info.called
        
        # 验证熵值趋势（应该是下降的）
        entropy_trend = np.polyfit(range(len(self.trainer.entropy_history)), self.trainer.entropy_history, 1)[0]
        assert entropy_trend < 0  # 熵值应该下降
    
    def test_portfolio_metrics_calculation(self):
        """测试投资组合指标计算"""
        # 准备测试数据
        self.trainer.portfolio_values_history = [1000000, 1010000, 1020000, 1015000, 1025000]
        self.trainer.benchmark_values_history = [1000000, 1008000, 1016000, 1012000, 1020000]
        self.trainer.dates_history = [datetime.now() for _ in range(5)]
        
        # 计算指标
        metrics = self.trainer._calculate_portfolio_metrics()
        
        # 验证指标
        assert metrics is not None
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.alpha, float)
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.annualized_return, float)
        
        # 验证指标合理性
        assert metrics.max_drawdown >= 0
        assert not np.isnan(metrics.sharpe_ratio)
        assert not np.isnan(metrics.alpha)
        assert not np.isnan(metrics.beta)
    
    def test_agent_behavior_metrics_calculation(self):
        """测试智能体行为指标计算"""
        # 准备测试数据
        self.trainer.entropy_history = [2.5, 2.3, 2.1, 1.9, 1.8]  # 下降趋势
        self.trainer.position_weights_history = [
            np.array([0.2, 0.3, 0.5]),
            np.array([0.25, 0.25, 0.5]),
            np.array([0.3, 0.2, 0.5]),
            np.array([0.35, 0.15, 0.5]),
            np.array([0.4, 0.1, 0.5])
        ]
        
        # 计算指标
        metrics = self.trainer._calculate_agent_behavior_metrics()
        
        # 验证指标
        assert metrics is not None
        assert isinstance(metrics.mean_entropy, float)
        assert isinstance(metrics.entropy_trend, float)
        assert isinstance(metrics.mean_position_concentration, float)
        assert isinstance(metrics.turnover_rate, float)
        
        # 验证指标合理性
        assert metrics.mean_entropy > 0
        assert metrics.entropy_trend < 0  # 应该是下降趋势
        assert 0 <= metrics.mean_position_concentration <= 1
        assert metrics.turnover_rate >= 0
    
    def test_risk_control_metrics_calculation(self):
        """测试风险控制指标计算"""
        # 模拟风险控制器数据
        risk_budget = self.trainer.environment.drawdown_controller.adaptive_risk_budget
        risk_budget.risk_budget_history = [0.1, 0.08, 0.06, 0.05, 0.07]
        risk_budget.risk_usage_history = [0.08, 0.07, 0.05, 0.04, 0.06]
        
        # 计算指标
        metrics = self.trainer._calculate_risk_control_metrics()
        
        # 验证指标
        assert metrics is not None
        assert isinstance(metrics.avg_risk_budget_utilization, float)
        assert isinstance(metrics.risk_budget_efficiency, float)
        assert isinstance(metrics.control_signal_frequency, float)
        assert isinstance(metrics.market_regime_stability, float)
        
        # 验证指标合理性
        assert 0 <= metrics.avg_risk_budget_utilization <= 1
        assert metrics.risk_budget_efficiency >= 0
        assert metrics.control_signal_frequency >= 0
        assert 0 <= metrics.market_regime_stability <= 1
    
    @patch('src.rl_trading_system.training.enhanced_trainer.logger')
    def test_enhanced_metrics_logging(self, mock_logger):
        """测试增强指标日志记录"""
        # 准备所有类型的指标数据
        self.trainer.portfolio_values_history = [1000000, 1010000, 1020000]
        self.trainer.benchmark_values_history = [1000000, 1008000, 1016000]
        self.trainer.dates_history = [datetime.now() for _ in range(3)]
        self.trainer.entropy_history = [2.5, 2.3, 2.1]
        self.trainer.position_weights_history = [
            np.array([0.2, 0.3, 0.5]),
            np.array([0.25, 0.25, 0.5]),
            np.array([0.3, 0.2, 0.5])
        ]
        
        # 触发指标计算和日志记录
        self.trainer._calculate_and_log_enhanced_metrics(episode_num=10)
        
        # 验证日志被调用
        assert mock_logger.info.called
        
        # 检查日志内容包含关键指标
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        log_content = ' '.join(log_calls)
        
        # 验证包含投资组合指标
        assert '夏普比率' in log_content or 'Sharpe Ratio' in log_content
        assert '最大回撤' in log_content or 'Max Drawdown' in log_content
        assert 'Alpha' in log_content
        assert 'Beta' in log_content
        
        # 验证包含智能体行为指标
        assert '熵值' in log_content or 'Entropy' in log_content
        assert '换手率' in log_content or 'Turnover' in log_content
    
    def test_enhanced_training_stats(self):
        """测试增强训练统计信息"""
        # 添加一些历史数据
        self.trainer.portfolio_values_history = [1000000, 1010000, 1020000]
        self.trainer.entropy_history = [2.5, 2.3, 2.1]
        self.trainer.position_weights_history = [
            np.array([0.2, 0.3, 0.5]),
            np.array([0.25, 0.25, 0.5]),
            np.array([0.3, 0.2, 0.5])
        ]
        
        # 获取增强统计信息
        stats = self.trainer.get_enhanced_training_stats()
        
        # 验证统计信息
        assert isinstance(stats, dict)
        assert 'portfolio_values_count' in stats
        assert 'entropy_values_count' in stats
        assert 'position_weights_count' in stats
        assert 'latest_portfolio_value' in stats
        assert 'latest_entropy' in stats
        
        # 验证数值正确性
        assert stats['portfolio_values_count'] == 3
        assert stats['entropy_values_count'] == 3
        assert stats['position_weights_count'] == 3
        assert stats['latest_portfolio_value'] == 1020000
        assert stats['latest_entropy'] == 2.1
    
    def test_metrics_history_reset(self):
        """测试指标历史重置"""
        # 添加一些历史数据
        self.trainer.portfolio_values_history = [1000000, 1010000]
        self.trainer.entropy_history = [2.5, 2.3]
        self.trainer.position_weights_history = [np.array([0.2, 0.3, 0.5])]
        
        # 验证数据存在
        assert len(self.trainer.portfolio_values_history) > 0
        assert len(self.trainer.entropy_history) > 0
        assert len(self.trainer.position_weights_history) > 0
        
        # 重置历史数据
        self.trainer.reset_enhanced_histories()
        
        # 验证数据被清空
        assert len(self.trainer.portfolio_values_history) == 0
        assert len(self.trainer.entropy_history) == 0
        assert len(self.trainer.position_weights_history) == 0


if __name__ == '__main__':
    pytest.main([__file__])