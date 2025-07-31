"""
测试实际的SAC智能体实现
"""
import pytest
import torch
import numpy as np

from src.rl_trading_system.models.sac_agent import SACAgent, SACConfig
from src.rl_trading_system.models.replay_buffer import Experience


class TestSACAgentImplementation:
    """测试实际SAC智能体实现"""
    
    @pytest.fixture
    def sac_config(self):
        """SAC配置fixture"""
        return SACConfig(
            state_dim=64,
            action_dim=10,
            hidden_dim=128,
            batch_size=32,
            buffer_capacity=1000,
            learning_starts=50
        )
    
    @pytest.fixture
    def sac_agent(self, sac_config):
        """SAC智能体fixture"""
        return SACAgent(sac_config)
    
    def test_agent_initialization(self, sac_agent, sac_config):
        """测试智能体初始化"""
        assert sac_agent.config.state_dim == sac_config.state_dim
        assert sac_agent.config.action_dim == sac_config.action_dim
        assert sac_agent.training_step == 0
        assert sac_agent.total_env_steps == 0
        
        # 检查网络组件
        assert hasattr(sac_agent, 'actor')
        assert hasattr(sac_agent, 'critic')
        assert hasattr(sac_agent, 'replay_buffer')
        
    def test_get_action(self, sac_agent, sac_config):
        """测试动作生成"""
        state = torch.randn(sac_config.state_dim)
        
        # 测试确定性动作
        action_det = sac_agent.get_action(state, deterministic=True)
        assert action_det.shape == (sac_config.action_dim,)
        assert torch.all(action_det >= 0)
        assert torch.allclose(torch.sum(action_det), torch.tensor(1.0), atol=1e-5)
        
        # 测试随机动作
        action_stoch, log_prob = sac_agent.get_action(state, return_log_prob=True)
        assert action_stoch.shape == (sac_config.action_dim,)
        assert log_prob.shape == ()
        assert torch.all(action_stoch >= 0)
        assert torch.allclose(torch.sum(action_stoch), torch.tensor(1.0), atol=1e-5)
        
    def test_add_experience(self, sac_agent, sac_config):
        """测试添加经验"""
        experience = Experience(
            state=torch.randn(sac_config.state_dim),
            action=torch.rand(sac_config.action_dim),
            reward=1.0,
            next_state=torch.randn(sac_config.state_dim),
            done=False
        )
        
        initial_size = sac_agent.replay_buffer.size
        sac_agent.add_experience(experience)
        
        assert sac_agent.replay_buffer.size == initial_size + 1
        assert sac_agent.total_env_steps == 1
        
    def test_can_update(self, sac_agent, sac_config):
        """测试更新条件检查"""
        # 初始状态不能更新
        assert not sac_agent.can_update()
        
        # 添加足够的经验
        for i in range(sac_config.learning_starts + sac_config.batch_size):
            experience = Experience(
                state=torch.randn(sac_config.state_dim),
                action=torch.rand(sac_config.action_dim),
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(sac_config.state_dim),
                done=np.random.choice([True, False])
            )
            sac_agent.add_experience(experience)
        
        # 现在应该可以更新
        assert sac_agent.can_update()
        
    def test_update(self, sac_agent, sac_config):
        """测试网络更新"""
        # 添加足够的经验
        for i in range(sac_config.learning_starts + sac_config.batch_size):
            experience = Experience(
                state=torch.randn(sac_config.state_dim),
                action=torch.rand(sac_config.action_dim),
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(sac_config.state_dim),
                done=np.random.choice([True, False])
            )
            sac_agent.add_experience(experience)
        
        # 执行更新
        losses = sac_agent.update()
        
        # 检查返回的损失
        assert 'critic_loss' in losses
        assert 'actor_loss' in losses
        assert 'alpha_loss' in losses
        assert 'alpha' in losses
        
        # 检查训练步数增加
        assert sac_agent.training_step > 0
        
    def test_training_stats(self, sac_agent, sac_config):
        """测试训练统计"""
        # 添加经验并训练
        for i in range(sac_config.learning_starts + sac_config.batch_size):
            experience = Experience(
                state=torch.randn(sac_config.state_dim),
                action=torch.rand(sac_config.action_dim),
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(sac_config.state_dim),
                done=False
            )
            sac_agent.add_experience(experience)
        
        # 多次更新
        for _ in range(5):
            sac_agent.update()
        
        # 获取统计信息
        stats = sac_agent.get_training_stats()
        
        assert 'training_step' in stats
        assert 'total_env_steps' in stats
        assert 'buffer_size' in stats
        assert stats['training_step'] > 0
        assert stats['total_env_steps'] > 0
        
    def test_eval_train_modes(self, sac_agent):
        """测试评估和训练模式"""
        # 训练模式
        sac_agent.train()
        assert sac_agent.training
        
        # 评估模式
        sac_agent.eval()
        assert not sac_agent.training
        
    def test_temperature_parameter(self, sac_agent):
        """测试温度参数"""
        alpha = sac_agent.alpha
        assert alpha > 0
        assert torch.is_tensor(alpha)
        
        # 温度参数应该可以更新
        initial_alpha = alpha.item()
        
        # 添加经验并训练
        for i in range(100):
            experience = Experience(
                state=torch.randn(sac_agent.config.state_dim),
                action=torch.rand(sac_agent.config.action_dim),
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(sac_agent.config.state_dim),
                done=False
            )
            sac_agent.add_experience(experience)
        
        # 多次更新
        for _ in range(10):
            if sac_agent.can_update():
                sac_agent.update()
        
        # 温度参数可能会变化
        final_alpha = sac_agent.alpha.item()
        # 注意：温度参数可能增加或减少，这里只检查它仍然为正
        assert final_alpha > 0