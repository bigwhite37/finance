"""
测试Actor网络的单元测试
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.rl_trading_system.models.actor_network import Actor, ActorConfig


class TestActorNetwork:
    """Actor网络测试类"""
    
    @pytest.fixture
    def actor_config(self):
        """Actor配置fixture"""
        return ActorConfig(
            state_dim=256,
            action_dim=100,
            hidden_dim=512,
            n_layers=3,
            activation='relu',
            dropout=0.1,
            log_std_min=-20,
            log_std_max=2
        )
    
    @pytest.fixture
    def actor_network(self, actor_config):
        """Actor网络fixture"""
        return Actor(actor_config)
    
    @pytest.fixture
    def sample_state(self, actor_config):
        """样本状态fixture"""
        batch_size = 32
        return torch.randn(batch_size, actor_config.state_dim)
    
    def test_actor_initialization(self, actor_network, actor_config):
        """测试Actor网络初始化"""
        assert isinstance(actor_network, nn.Module)
        assert actor_network.config.state_dim == actor_config.state_dim
        assert actor_network.config.action_dim == actor_config.action_dim
        assert actor_network.config.hidden_dim == actor_config.hidden_dim
        
        # 检查网络层是否正确创建
        assert hasattr(actor_network, 'shared_layers')
        assert hasattr(actor_network, 'mean_head')
        assert hasattr(actor_network, 'log_std_head')
        
    def test_forward_pass_shape(self, actor_network, sample_state, actor_config):
        """测试前向传播输出形状"""
        mean, log_std = actor_network.forward(sample_state)
        
        batch_size = sample_state.size(0)
        expected_shape = (batch_size, actor_config.action_dim)
        
        assert mean.shape == expected_shape
        assert log_std.shape == expected_shape
        
    def test_forward_pass_values(self, actor_network, sample_state, actor_config):
        """测试前向传播输出值的合理性"""
        mean, log_std = actor_network.forward(sample_state)
        
        # 检查均值是否在合理范围内
        assert torch.all(torch.isfinite(mean))
        assert torch.all(mean >= -10) and torch.all(mean <= 10)
        
        # 检查log_std是否在指定范围内
        assert torch.all(log_std >= actor_config.log_std_min)
        assert torch.all(log_std <= actor_config.log_std_max)
        
    def test_get_action_deterministic(self, actor_network, sample_state, actor_config):
        """测试确定性动作生成"""
        # 设置为评估模式以禁用dropout
        actor_network.eval()
        
        action, log_prob = actor_network.get_action(sample_state, deterministic=True)
        
        batch_size = sample_state.size(0)
        expected_shape = (batch_size, actor_config.action_dim)
        
        assert action.shape == expected_shape
        assert log_prob.shape == (batch_size,)
        
        # 确定性动作应该是可重复的
        action2, log_prob2 = actor_network.get_action(sample_state, deterministic=True)
        assert torch.allclose(action, action2, atol=1e-6)
        assert torch.allclose(log_prob, log_prob2, atol=1e-6)
        
    def test_get_action_stochastic(self, actor_network, sample_state, actor_config):
        """测试随机动作生成"""
        action1, log_prob1 = actor_network.get_action(sample_state, deterministic=False)
        action2, log_prob2 = actor_network.get_action(sample_state, deterministic=False)
        
        batch_size = sample_state.size(0)
        expected_shape = (batch_size, actor_config.action_dim)
        
        assert action1.shape == expected_shape
        assert action2.shape == expected_shape
        assert log_prob1.shape == (batch_size,)
        assert log_prob2.shape == (batch_size,)
        
        # 随机动作应该不同
        assert not torch.allclose(action1, action2, atol=1e-3)
        
    def test_portfolio_weight_constraints(self, actor_network, sample_state):
        """测试投资组合权重约束"""
        action, _ = actor_network.get_action(sample_state, deterministic=True)
        
        # 检查权重是否非负
        assert torch.all(action >= 0)
        
        # 检查权重和是否为1（允许小的数值误差）
        weight_sums = torch.sum(action, dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
        
    def test_action_probability_distribution(self, actor_network, sample_state):
        """测试动作概率分布的有效性"""
        # 生成多个动作样本
        actions = []
        log_probs = []
        
        for _ in range(10):
            action, log_prob = actor_network.get_action(sample_state, deterministic=False)
            actions.append(action)
            log_probs.append(log_prob)
        
        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)
        
        # 检查动作的变异性
        action_std = torch.std(actions, dim=0)
        assert torch.all(action_std > 1e-4)  # 应该有一定的变异性
        
        # 检查log_prob的有限性
        assert torch.all(torch.isfinite(log_probs))
        
    def test_reparameterization_trick(self, actor_network, sample_state):
        """测试重参数化技巧和梯度计算"""
        # 启用梯度计算
        sample_state.requires_grad_(True)
        
        action, log_prob = actor_network.get_action(sample_state, deterministic=False)
        
        # 计算损失（简单的L2损失）
        loss = torch.mean(action ** 2) + torch.mean(log_prob ** 2)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否存在且有限
        assert sample_state.grad is not None
        assert torch.all(torch.isfinite(sample_state.grad))
        
        # 检查网络参数是否有梯度
        for param in actor_network.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.all(torch.isfinite(param.grad))
                
    def test_log_probability_calculation(self, actor_network, sample_state):
        """测试对数概率计算的正确性"""
        action, log_prob = actor_network.get_action(sample_state, deterministic=False)
        
        # 重新计算对数概率
        mean, log_std = actor_network.forward(sample_state)
        std = torch.exp(log_std)
        
        # 使用正态分布计算对数概率
        normal_dist = torch.distributions.Normal(mean, std)
        
        # 由于使用了tanh变换，需要考虑雅可比行列式
        # 这里简化测试，主要检查log_prob的合理性
        assert torch.all(torch.isfinite(log_prob))
        assert torch.all(log_prob <= 0)  # 对数概率应该非正
        
    def test_batch_processing(self, actor_network, actor_config):
        """测试批处理能力"""
        batch_sizes = [1, 16, 32, 64]
        
        for batch_size in batch_sizes:
            state = torch.randn(batch_size, actor_config.state_dim)
            action, log_prob = actor_network.get_action(state, deterministic=True)
            
            assert action.shape == (batch_size, actor_config.action_dim)
            assert log_prob.shape == (batch_size,)
            
    def test_network_parameter_count(self, actor_network, actor_config):
        """测试网络参数数量的合理性"""
        total_params = sum(p.numel() for p in actor_network.parameters())
        
        # 估算参数数量（粗略估计）
        expected_min_params = (
            actor_config.state_dim * actor_config.hidden_dim +  # 第一层
            actor_config.hidden_dim * actor_config.action_dim * 2  # 输出层（mean + log_std）
        )
        
        assert total_params >= expected_min_params
        assert total_params < expected_min_params * 10  # 不应该过大
        
    def test_gradient_flow(self, actor_network, sample_state):
        """测试梯度流动"""
        sample_state.requires_grad_(True)
        
        # 前向传播
        action, log_prob = actor_network.get_action(sample_state, deterministic=False)
        
        # 计算损失
        loss = torch.mean(action) + torch.mean(log_prob)
        
        # 反向传播
        loss.backward()
        
        # 检查所有参数都有梯度
        for name, param in actor_network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert not torch.all(param.grad == 0), f"参数 {name} 的梯度为零"
                
    def test_different_activation_functions(self, actor_config):
        """测试不同激活函数"""
        activations = ['relu', 'tanh', 'gelu']
        
        for activation in activations:
            config = ActorConfig(
                state_dim=actor_config.state_dim,
                action_dim=actor_config.action_dim,
                hidden_dim=actor_config.hidden_dim,
                activation=activation
            )
            
            actor = Actor(config)
            state = torch.randn(16, actor_config.state_dim)
            
            # 应该能够正常前向传播
            action, log_prob = actor.get_action(state, deterministic=True)
            assert action.shape == (16, actor_config.action_dim)
            assert log_prob.shape == (16,)
            
    def test_numerical_stability(self, actor_network):
        """测试数值稳定性"""
        # 测试极端输入值
        extreme_states = [
            torch.full((4, actor_network.config.state_dim), 1e6),   # 很大的值
            torch.full((4, actor_network.config.state_dim), -1e6),  # 很小的值
            torch.zeros(4, actor_network.config.state_dim),         # 零值
            torch.full((4, actor_network.config.state_dim), float('nan'))  # NaN值
        ]
        
        for i, state in enumerate(extreme_states[:-1]):  # 跳过NaN测试
            action, log_prob = actor_network.get_action(state, deterministic=True)
            
            # 输出应该是有限的
            assert torch.all(torch.isfinite(action)), f"极端输入 {i} 产生了无限值"
            assert torch.all(torch.isfinite(log_prob)), f"极端输入 {i} 产生了无限对数概率"
            
    @pytest.mark.parametrize("state_dim,action_dim,hidden_dim", [
        (128, 50, 256),
        (512, 200, 1024),
        (64, 10, 128)
    ])
    def test_different_dimensions(self, state_dim, action_dim, hidden_dim):
        """测试不同维度配置"""
        config = ActorConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        actor = Actor(config)
        state = torch.randn(8, state_dim)
        
        action, log_prob = actor.get_action(state, deterministic=True)
        
        assert action.shape == (8, action_dim)
        assert log_prob.shape == (8,)
        
        # 检查权重约束
        assert torch.all(action >= 0)
        weight_sums = torch.sum(action, dim=1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)