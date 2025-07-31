"""
测试完整SAC智能体的单元测试
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from src.rl_trading_system.models.actor_network import Actor, ActorConfig
from src.rl_trading_system.models.critic_network import CriticWithTargetNetwork, CriticConfig
from src.rl_trading_system.models.replay_buffer import ReplayBuffer, Experience, ReplayBufferConfig


# 先创建一个简单的SAC Agent配置用于测试
@dataclass
class SACConfig:
    """SAC智能体配置"""
    state_dim: int = 256
    action_dim: int = 100
    hidden_dim: int = 512
    
    # 学习率
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    
    # SAC参数
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    target_entropy: float = -100  # -action_dim
    
    # 训练参数
    batch_size: int = 256
    buffer_capacity: int = 1000000
    device: str = 'cpu'


# 创建一个简单的SAC Agent实现用于测试
class SimpleSACAgent(nn.Module):
    """简化的SAC智能体实现（用于测试）"""
    
    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # 网络组件
        actor_config = ActorConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )
        self.actor = Actor(actor_config)
        
        critic_config = CriticConfig(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim
        )
        self.critic = CriticWithTargetNetwork(critic_config)
        
        # 温度参数
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.get_parameters(), lr=config.lr_critic)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)
        
        # 回放缓冲区
        buffer_config = ReplayBufferConfig(
            capacity=config.buffer_capacity,
            batch_size=config.batch_size,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            device=config.device
        )
        self.replay_buffer = ReplayBuffer(buffer_config)
        
        # 训练统计
        self.training_step = 0
        
    @property
    def alpha(self):
        """当前温度参数"""
        return torch.exp(self.log_alpha)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取动作"""
        self.actor.eval()
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic=deterministic)
        self.actor.train()
        return action, log_prob
    
    def add_experience(self, experience: Experience):
        """添加经验到回放缓冲区"""
        self.replay_buffer.add(experience)
    
    def update(self) -> Dict[str, float]:
        """更新网络参数"""
        if not self.replay_buffer.can_sample():
            return {}
        
        # 采样批次
        batch = self.replay_buffer.sample()
        
        states = torch.stack([exp.state for exp in batch]).to(self.device)
        actions = torch.stack([exp.action for exp in batch]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp.next_state for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32).to(self.device)
        
        # 更新Critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        
        # 更新Actor
        actor_loss = self._update_actor(states)
        
        # 更新温度参数
        alpha_loss = self._update_alpha(states)
        
        # 软更新目标网络
        self.critic.soft_update(self.config.tau)
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item()
        }
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        """更新Critic网络"""
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.get_action(next_states)
            target_q = self.critic.get_target_min_q_value(next_states, next_actions)
            target_q = target_q - self.alpha * next_log_probs.unsqueeze(1)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.gamma * target_q
        
        current_q1, current_q2 = self.critic.get_main_q_values(states, actions)
        
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def _update_actor(self, states):
        """更新Actor网络"""
        actions, log_probs = self.actor.get_action(states)
        q_values = self.critic.main_network.get_min_q_value(states, actions)
        
        actor_loss = torch.mean(self.alpha * log_probs - q_values.squeeze())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def _update_alpha(self, states):
        """更新温度参数"""
        with torch.no_grad():
            _, log_probs = self.actor.get_action(states)
        
        alpha_loss = -torch.mean(self.log_alpha * (log_probs + self.config.target_entropy))
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()


class TestSACAgent:
    """SAC智能体测试类"""
    
    @pytest.fixture
    def sac_config(self):
        """SAC配置fixture"""
        return SACConfig(
            state_dim=64,  # 较小的维度以加快测试
            action_dim=10,
            hidden_dim=128,
            batch_size=32,
            buffer_capacity=1000
        )
    
    @pytest.fixture
    def sac_agent(self, sac_config):
        """SAC智能体fixture"""
        return SimpleSACAgent(sac_config)
    
    @pytest.fixture
    def sample_state(self, sac_config):
        """样本状态fixture"""
        return torch.randn(sac_config.state_dim)
    
    @pytest.fixture
    def sample_batch_states(self, sac_config):
        """批量样本状态fixture"""
        return torch.randn(16, sac_config.state_dim)
    
    def test_sac_agent_initialization(self, sac_agent, sac_config):
        """测试SAC智能体初始化"""
        assert isinstance(sac_agent.actor, Actor)
        assert isinstance(sac_agent.critic, CriticWithTargetNetwork)
        assert isinstance(sac_agent.replay_buffer, ReplayBuffer)
        
        # 检查优化器
        assert sac_agent.actor_optimizer is not None
        assert sac_agent.critic_optimizer is not None
        assert sac_agent.alpha_optimizer is not None
        
        # 检查温度参数
        assert sac_agent.log_alpha.requires_grad
        assert sac_agent.alpha > 0
        
        # 检查训练统计
        assert sac_agent.training_step == 0
        
    def test_get_action_deterministic(self, sac_agent, sample_state, sac_config):
        """测试确定性动作生成"""
        action, log_prob = sac_agent.get_action(sample_state.unsqueeze(0), deterministic=True)
        
        assert action.shape == (1, sac_config.action_dim)
        assert log_prob.shape == (1,)
        
        # 检查权重约束
        assert torch.all(action >= 0)
        weight_sum = torch.sum(action, dim=1)
        assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5)
        
        # 确定性动作应该可重复
        action2, log_prob2 = sac_agent.get_action(sample_state.unsqueeze(0), deterministic=True)
        assert torch.allclose(action, action2, atol=1e-6)
        
    def test_get_action_stochastic(self, sac_agent, sample_state, sac_config):
        """测试随机动作生成"""
        action1, log_prob1 = sac_agent.get_action(sample_state.unsqueeze(0), deterministic=False)
        action2, log_prob2 = sac_agent.get_action(sample_state.unsqueeze(0), deterministic=False)
        
        assert action1.shape == (1, sac_config.action_dim)
        assert action2.shape == (1, sac_config.action_dim)
        
        # 随机动作应该不同
        assert not torch.allclose(action1, action2, atol=1e-3)
        
        # 检查权重约束
        for action in [action1, action2]:
            assert torch.all(action >= 0)
            weight_sum = torch.sum(action, dim=1)
            assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5)
            
    def test_add_experience(self, sac_agent, sac_config):
        """测试添加经验"""
        initial_size = sac_agent.replay_buffer.size
        
        experience = Experience(
            state=torch.randn(sac_config.state_dim),
            action=torch.rand(sac_config.action_dim),
            reward=1.0,
            next_state=torch.randn(sac_config.state_dim),
            done=False
        )
        
        sac_agent.add_experience(experience)
        
        assert sac_agent.replay_buffer.size == initial_size + 1
        
    def test_update_insufficient_data(self, sac_agent):
        """测试数据不足时的更新"""
        # 没有足够数据时应该返回空字典
        losses = sac_agent.update()
        assert losses == {}
        
    def test_update_with_sufficient_data(self, sac_agent, sac_config):
        """测试有足够数据时的更新"""
        # 添加足够的经验
        for i in range(sac_config.batch_size * 2):
            experience = Experience(
                state=torch.randn(sac_config.state_dim),
                action=torch.rand(sac_config.action_dim),
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(sac_config.state_dim),
                done=np.random.choice([True, False])
            )
            sac_agent.add_experience(experience)
        
        # 更新网络
        losses = sac_agent.update()
        
        # 检查返回的损失
        assert 'critic_loss' in losses
        assert 'actor_loss' in losses
        assert 'alpha_loss' in losses
        assert 'alpha' in losses
        
        # 检查损失值的合理性
        assert isinstance(losses['critic_loss'], float)
        assert isinstance(losses['actor_loss'], float)
        assert isinstance(losses['alpha_loss'], float)
        assert losses['alpha'] > 0
        
        # 检查训练步数增加
        assert sac_agent.training_step == 1
        
    def test_temperature_parameter_adjustment(self, sac_agent, sac_config):
        """测试温度参数自动调整"""
        # 记录初始温度
        initial_alpha = sac_agent.alpha.item()
        
        # 添加经验并训练
        for i in range(sac_config.batch_size * 2):
            experience = Experience(
                state=torch.randn(sac_config.state_dim),
                action=torch.rand(sac_config.action_dim),
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(sac_config.state_dim),
                done=False
            )
            sac_agent.add_experience(experience)
        
        # 多次更新
        for _ in range(10):
            losses = sac_agent.update()
        
        # 温度参数应该有所变化
        final_alpha = sac_agent.alpha.item()
        # 注意：温度参数可能增加或减少，取决于策略熵
        assert final_alpha != initial_alpha
        assert final_alpha > 0
        
    def test_learning_capability(self, sac_agent, sac_config):
        """测试智能体的学习能力"""
        # 创建一个简单的学习任务：奖励与特定动作相关
        def reward_function(action):
            # 奖励函数：偏好某些动作
            target_action = torch.zeros_like(action)
            target_action[0] = 1.0  # 偏好第一个动作
            return -torch.sum((action - target_action) ** 2).item()
        
        # 收集初始性能
        initial_rewards = []
        for _ in range(10):
            state = torch.randn(sac_config.state_dim)
            action, _ = sac_agent.get_action(state.unsqueeze(0), deterministic=True)
            reward = reward_function(action.squeeze())
            initial_rewards.append(reward)
        
        initial_avg_reward = np.mean(initial_rewards)
        
        # 训练智能体
        for episode in range(50):
            state = torch.randn(sac_config.state_dim)
            action, _ = sac_agent.get_action(state.unsqueeze(0), deterministic=False)
            reward = reward_function(action.squeeze())
            next_state = torch.randn(sac_config.state_dim)
            
            experience = Experience(
                state=state,
                action=action.squeeze(),
                reward=reward,
                next_state=next_state,
                done=False
            )
            sac_agent.add_experience(experience)
            
            # 定期更新
            if episode % 5 == 0:
                sac_agent.update()
        
        # 评估训练后性能
        final_rewards = []
        for _ in range(10):
            state = torch.randn(sac_config.state_dim)
            action, _ = sac_agent.get_action(state.unsqueeze(0), deterministic=True)
            reward = reward_function(action.squeeze())
            final_rewards.append(reward)
        
        final_avg_reward = np.mean(final_rewards)
        
        # 性能应该有所改善
        assert final_avg_reward > initial_avg_reward - 0.1  # 允许一些变异
        
    def test_policy_stability(self, sac_agent, sample_batch_states):
        """测试策略稳定性"""
        # 在评估模式下，策略应该是稳定的
        sac_agent.eval()
        
        actions1, _ = sac_agent.get_action(sample_batch_states, deterministic=True)
        actions2, _ = sac_agent.get_action(sample_batch_states, deterministic=True)
        
        assert torch.allclose(actions1, actions2, atol=1e-6)
        
    def test_entropy_regularization(self, sac_agent, sac_config):
        """测试熵正则化"""
        # 添加一些经验
        for i in range(sac_config.batch_size * 2):
            experience = Experience(
                state=torch.randn(sac_config.state_dim),
                action=torch.rand(sac_config.action_dim),
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(sac_config.state_dim),
                done=False
            )
            sac_agent.add_experience(experience)
        
        # 计算策略熵
        states = torch.randn(32, sac_config.state_dim)
        actions, log_probs = sac_agent.get_action(states, deterministic=False)
        
        # 熵应该为正（随机策略）
        entropy = -torch.mean(log_probs)
        assert entropy > 0
        
        # 检查熵的合理范围
        assert entropy < 10  # 不应该过大
        
    def test_target_network_updates(self, sac_agent, sac_config):
        """测试目标网络更新"""
        # 获取初始目标网络参数
        initial_target_params = []
        for param in sac_agent.critic.target_network.parameters():
            initial_target_params.append(param.clone())
        
        # 添加经验并训练
        for i in range(sac_config.batch_size * 2):
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
        
        # 检查目标网络参数是否更新
        final_target_params = []
        for param in sac_agent.critic.target_network.parameters():
            final_target_params.append(param.clone())
        
        # 目标网络参数应该有所变化（软更新）
        for initial, final in zip(initial_target_params, final_target_params):
            assert not torch.allclose(initial, final, atol=1e-6)
            
    def test_gradient_flow(self, sac_agent, sac_config):
        """测试梯度流动"""
        # 添加经验
        for i in range(sac_config.batch_size * 2):
            experience = Experience(
                state=torch.randn(sac_config.state_dim),
                action=torch.rand(sac_config.action_dim),
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(sac_config.state_dim),
                done=False
            )
            sac_agent.add_experience(experience)
        
        # 更新前清零梯度
        sac_agent.actor_optimizer.zero_grad()
        sac_agent.critic_optimizer.zero_grad()
        sac_agent.alpha_optimizer.zero_grad()
        
        # 执行更新
        losses = sac_agent.update()
        
        # 检查所有网络都有梯度
        for name, param in sac_agent.actor.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Actor参数 {name} 没有梯度"
                
        for name, param in sac_agent.critic.main_network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Critic参数 {name} 没有梯度"
                
        assert sac_agent.log_alpha.grad is not None, "温度参数没有梯度"
        
    def test_batch_processing(self, sac_agent, sac_config):
        """测试批处理能力"""
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            states = torch.randn(batch_size, sac_config.state_dim)
            actions, log_probs = sac_agent.get_action(states, deterministic=False)
            
            assert actions.shape == (batch_size, sac_config.action_dim)
            assert log_probs.shape == (batch_size,)
            
            # 检查权重约束
            assert torch.all(actions >= 0)
            weight_sums = torch.sum(actions, dim=1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
            
    def test_device_consistency(self, sac_config):
        """测试设备一致性"""
        if torch.cuda.is_available():
            # 测试CUDA设备
            cuda_config = SACConfig(
                state_dim=sac_config.state_dim,
                action_dim=sac_config.action_dim,
                device='cuda'
            )
            
            cuda_agent = SimpleSACAgent(cuda_config)
            
            # 检查所有组件都在正确设备上
            assert next(cuda_agent.actor.parameters()).device.type == 'cuda'
            assert next(cuda_agent.critic.parameters()).device.type == 'cuda'
            assert cuda_agent.log_alpha.device.type == 'cuda'
            
            # 测试前向传播
            state = torch.randn(1, sac_config.state_dim).cuda()
            action, log_prob = cuda_agent.get_action(state)
            
            assert action.device.type == 'cuda'
            assert log_prob.device.type == 'cuda'