"""
测试回撤计算修复
验证训练器现在使用投资组合价值而非累积奖励来计算回撤
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from src.rl_trading_system.training.trainer import DrawdownEarlyStopping, RLTrainer, TrainingConfig


class TestDrawdownEarlyStopping:
    """测试回撤早停机制"""
    
    def test_drawdown_calculation_with_portfolio_values(self):
        """测试基于投资组合价值的回撤计算"""
        early_stopping = DrawdownEarlyStopping(max_drawdown=0.15, patience=2)  # 降低阈值以便测试
        
        # 模拟投资组合价值变化：先上升到峰值，然后下跌
        portfolio_values = [100000, 110000, 120000, 115000, 100000, 90000]
        expected_drawdowns = [0.0, 0.0, 0.0, 0.0417, 0.1667, 0.25]  # 相对于120000的峰值
        
        for i, value in enumerate(portfolio_values):
            should_stop = early_stopping.step(value)
            current_drawdown = early_stopping.get_current_drawdown()
            
            # 验证回撤计算正确
            assert abs(current_drawdown - expected_drawdowns[i]) < 0.01, \
                f"Episode {i}: 期望回撤 {expected_drawdowns[i]:.4f}, 实际回撤 {current_drawdown:.4f}"
            
            # 验证早停逻辑：从第4个值开始回撤超过阈值0.15，第6个值应该触发早停
            if i < 5:
                assert not should_stop, f"Episode {i}: 不应该早停"
            elif i == 5:  # 第6个值（索引5）应该触发早停
                assert should_stop, f"Episode {i}: 应该触发早停，回撤={current_drawdown:.4f}, 计数器={early_stopping.counter}"
                break
        
        # 验证最终确实触发了早停
        assert early_stopping.early_stop, "应该触发早停"
    
    def test_no_drawdown_with_monotonic_increase(self):
        """测试单调递增的投资组合价值不会产生回撤"""
        early_stopping = DrawdownEarlyStopping(max_drawdown=0.1, patience=2)
        
        # 模拟单调递增的投资组合价值
        portfolio_values = [100000, 105000, 110000, 115000, 120000]
        
        for value in portfolio_values:
            should_stop = early_stopping.step(value)
            current_drawdown = early_stopping.get_current_drawdown()
            
            # 单调递增应该没有回撤
            assert current_drawdown == 0.0, f"单调递增不应该有回撤，实际回撤: {current_drawdown}"
            assert not should_stop, "单调递增不应该触发早停"
    
    def test_peak_value_updates_correctly(self):
        """测试峰值更新逻辑"""
        early_stopping = DrawdownEarlyStopping(max_drawdown=0.5, patience=5)
        
        # 测试峰值更新
        early_stopping.step(100000)
        assert early_stopping.peak_value == 100000
        
        early_stopping.step(120000)  # 新峰值
        assert early_stopping.peak_value == 120000
        
        early_stopping.step(110000)  # 下跌，峰值不变
        assert early_stopping.peak_value == 120000
        
        early_stopping.step(130000)  # 新峰值
        assert early_stopping.peak_value == 130000


class TestTrainerDrawdownMonitoring:
    """测试训练器的回撤监控"""
    
    def test_monitor_drawdown_uses_portfolio_value(self):
        """测试回撤监控使用投资组合价值而非累积奖励"""
        # 创建模拟环境
        mock_env = Mock()
        mock_env.total_value = 100000
        mock_env._calculate_current_drawdown.return_value = 0.05
        # 不设置get_drawdown_metrics以避免Mock问题
        del mock_env.get_drawdown_metrics
        
        # 创建模拟智能体
        mock_agent = Mock()
        
        # 创建模拟数据分割
        mock_data_split = Mock()
        
        # 创建训练配置
        config = TrainingConfig(
            n_episodes=10,
            enable_drawdown_monitoring=True,
            max_training_drawdown=0.3
        )
        
        # 创建训练器
        trainer = RLTrainer(config, mock_env, mock_agent, mock_data_split)
        
        # 模拟几个episode的回撤监控
        episode_rewards = [200, 150, -50]  # 注意：有正有负的奖励
        portfolio_values = [100000, 105000, 98000]  # 投资组合价值变化
        
        for i, (reward, portfolio_value) in enumerate(zip(episode_rewards, portfolio_values)):
            mock_env.total_value = portfolio_value
            trainer._monitor_drawdown(reward, i + 1)
        
        # 验证回撤指标记录了投资组合价值而非累积奖励
        assert len(trainer.drawdown_metrics) == 3
        
        for i, metric in enumerate(trainer.drawdown_metrics):
            assert 'portfolio_value' in metric
            assert 'episode_reward' in metric
            assert 'training_drawdown' in metric
            assert 'env_drawdown' in metric
            
            # 验证记录的是投资组合价值，不是累积奖励
            assert metric['portfolio_value'] == portfolio_values[i]
            assert metric['episode_reward'] == episode_rewards[i]
    
    def test_drawdown_monitoring_without_portfolio_value(self):
        """测试环境不支持投资组合价值时的处理"""
        # 创建不支持total_value的模拟环境
        mock_env = Mock()
        del mock_env.total_value  # 删除total_value属性
        
        mock_agent = Mock()
        mock_data_split = Mock()
        
        config = TrainingConfig(
            n_episodes=10,
            enable_drawdown_monitoring=True
        )
        
        trainer = RLTrainer(config, mock_env, mock_agent, mock_data_split)
        
        # 应该能够正常处理，不会抛出异常
        trainer._monitor_drawdown(100, 1)
        
        # 应该没有记录任何回撤指标
        assert len(trainer.drawdown_metrics) == 0


def test_cumulative_reward_vs_portfolio_value_difference():
    """
    演示累积奖励和投资组合价值在回撤计算上的差异
    这个测试说明了为什么使用累积奖励计算回撤是错误的
    """
    # 模拟场景：奖励都是正数，但投资组合价值有起伏
    episode_rewards = [200, 180, 220, 190, 210]  # 都是正数
    cumulative_rewards = np.cumsum(episode_rewards)  # [200, 380, 600, 790, 1000] - 单调递增
    
    portfolio_values = [100000, 102000, 105000, 103000, 106000]  # 有起伏
    
    # 使用累积奖励的错误回撤计算
    cumulative_early_stopping = DrawdownEarlyStopping(max_drawdown=0.1, patience=2)
    cumulative_drawdowns = []
    
    for reward in cumulative_rewards:
        cumulative_early_stopping.step(reward)
        cumulative_drawdowns.append(cumulative_early_stopping.get_current_drawdown())
    
    # 使用投资组合价值的正确回撤计算
    portfolio_early_stopping = DrawdownEarlyStopping(max_drawdown=0.1, patience=2)
    portfolio_drawdowns = []
    
    for value in portfolio_values:
        portfolio_early_stopping.step(value)
        portfolio_drawdowns.append(portfolio_early_stopping.get_current_drawdown())
    
    # 验证差异
    print(f"累积奖励: {cumulative_rewards}")
    print(f"累积奖励回撤: {cumulative_drawdowns}")
    print(f"投资组合价值: {portfolio_values}")
    print(f"投资组合回撤: {portfolio_drawdowns}")
    
    # 累积奖励的回撤应该都是0（因为单调递增）
    assert all(d == 0.0 for d in cumulative_drawdowns), "累积奖励回撤应该都是0"
    
    # 投资组合价值的回撤应该有变化
    assert any(d > 0.0 for d in portfolio_drawdowns), "投资组合价值回撤应该有非零值"
    
    # 具体验证：从105000下跌到103000应该有回撤
    expected_drawdown_at_step_4 = (105000 - 103000) / 105000  # ≈ 0.019
    assert abs(portfolio_drawdowns[3] - expected_drawdown_at_step_4) < 0.001


if __name__ == "__main__":
    # 运行演示测试
    test_cumulative_reward_vs_portfolio_value_difference()
    print("回撤计算修复验证通过！")