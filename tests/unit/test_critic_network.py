"""
测试Critic网络的单元测试
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.rl_trading_system.models.critic_network import Critic, CriticConfig


class TestCriticNetwork:
    """Critic网络测试类"""
    
    @pytest.fixture
    def critic_config(self):
        """Critic配置fixture"""
        return CriticConfig(
            state_dim=256,
            action_dim=100,
            hidden_dim=512,
            n_layers=3,
            activation='relu',
            dropout=0.1
        )
    
    @pytest.fixture
    def critic_network(self, critic_config):
        """Critic网络fixture"""
        return Critic(critic_config)
    
    @pytest.fixture
    def sample_state(self, critic_config):
        """样本状态fixture"""
        batch_size = 32
        return torch.randn(batch_size, critic_config.state_dim)
    
    @pytest.fixture
    def sample_action(self, critic_config):
        """样本动作fixture"""
        batch_size = 32
        # 生成标准化的投资组合权重
        action = torch.rand(batch_size, critic_config.action_dim)
        action = action / action.sum(dim=1, keepdim=True)
        return action
    
    def test_critic_initialization(self, critic_network, critic_config):
        """测试Critic网络初始化"""
        assert isinstance(critic_network, nn.Module)
        assert critic_network.config.state_dim == critic_config.state_dim
        assert critic_network.config.action_dim == critic_config.action_dim
        assert critic_network.config.hidden_dim == critic_config.hidden_dim
        
        # 检查网络层是否正确创建
        assert hasattr(critic_network, 'state_encoder')
        assert hasattr(critic_network, 'action_encoder')
        assert hasattr(critic_network, 'q_network')
        
    def test_forward_pass_shape(self, critic_network, sample_state, sample_action):
        """测试前向传播输出形状"""
        q_value = critic_network.forward(sample_state, sample_action)
        
        batch_size = sample_state.size(0)
        expected_shape = (batch_size, 1)
        
        assert q_value.shape == expected_shape
        
    def test_forward_pass_values(self, critic_network, sample_state, sample_action):
        """测试前向传播输出值的合理性"""
        q_value = critic_network.forward(sample_state, sample_action)
        
        # 检查Q值是否有限
        assert torch.all(torch.isfinite(q_value))
        
        # Q值应该在合理范围内（不应该过大或过小）
        assert torch.all(q_value > -1000) and torch.all(q_value < 1000)
        
    def test_q_value_estimation_consistency(self, critic_network, sample_state, sample_action):
        """测试Q值估计的一致性"""
        # 相同输入应该产生相同输出
        critic_network.eval()
        
        q_value1 = critic_network.forward(sample_state, sample_action)
        q_value2 = critic_network.forward(sample_state, sample_action)
        
        assert torch.allclose(q_value1, q_value2, atol=1e-6)
        
    def test_different_actions_different_q_values(self, critic_network, sample_state, critic_config):
        """测试不同动作产生不同Q值"""
        critic_network.eval()
        
        # 生成两个极端不同的动作
        batch_size = sample_state.size(0)
        
        # 动作1：所有权重给第一个资产
        action1 = torch.zeros(batch_size, critic_config.action_dim)
        action1[:, 0] = 1.0
        
        # 动作2：均匀分布权重
        action2 = torch.ones(batch_size, critic_config.action_dim) / critic_config.action_dim
        
        q_value1 = critic_network.forward(sample_state, action1)
        q_value2 = critic_network.forward(sample_state, action2)
        
        # 极端不同的动作应该产生不同的Q值
        different_count = torch.sum(torch.abs(q_value1 - q_value2) > 1e-6)
        assert different_count > 0, "极端不同的动作应该产生不同的Q值"
        
        # 检查Q值的变异性
        q_diff = torch.abs(q_value1 - q_value2)
        max_diff = torch.max(q_diff)
        assert max_diff > 1e-6, f"Q值差异过小: {max_diff}"
        
    def test_gradient_flow(self, critic_network, sample_state, sample_action):
        """测试梯度流动"""
        sample_state.requires_grad_(True)
        sample_action.requires_grad_(True)
        
        # 前向传播
        q_value = critic_network.forward(sample_state, sample_action)
        
        # 计算损失
        loss = torch.mean(q_value ** 2)
        
        # 反向传播
        loss.backward()
        
        # 检查输入梯度
        assert sample_state.grad is not None
        assert sample_action.grad is not None
        assert torch.all(torch.isfinite(sample_state.grad))
        assert torch.all(torch.isfinite(sample_action.grad))
        
        # 检查网络参数梯度
        for name, param in critic_network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"参数 {name} 没有梯度"
                assert torch.all(torch.isfinite(param.grad)), f"参数 {name} 的梯度不是有限值"
                
    def test_batch_processing(self, critic_network, critic_config):
        """测试批处理能力"""
        batch_sizes = [1, 16, 32, 64]
        
        for batch_size in batch_sizes:
            state = torch.randn(batch_size, critic_config.state_dim)
            action = torch.rand(batch_size, critic_config.action_dim)
            action = action / action.sum(dim=1, keepdim=True)
            
            q_value = critic_network.forward(state, action)
            
            assert q_value.shape == (batch_size, 1)
            assert torch.all(torch.isfinite(q_value))
            
    def test_network_parameter_count(self, critic_network, critic_config):
        """测试网络参数数量的合理性"""
        total_params = sum(p.numel() for p in critic_network.parameters())
        
        # 估算参数数量（粗略估计）
        expected_min_params = (
            critic_config.state_dim * critic_config.hidden_dim +  # 状态编码器
            critic_config.action_dim * critic_config.hidden_dim +  # 动作编码器
            critic_config.hidden_dim * 1  # 输出层
        )
        
        assert total_params >= expected_min_params
        assert total_params < expected_min_params * 20  # 不应该过大
        
    def test_different_activation_functions(self, critic_config):
        """测试不同激活函数"""
        activations = ['relu', 'tanh', 'gelu']
        
        for activation in activations:
            config = CriticConfig(
                state_dim=critic_config.state_dim,
                action_dim=critic_config.action_dim,
                hidden_dim=critic_config.hidden_dim,
                activation=activation
            )
            
            critic = Critic(config)
            state = torch.randn(16, critic_config.state_dim)
            action = torch.rand(16, critic_config.action_dim)
            action = action / action.sum(dim=1, keepdim=True)
            
            # 应该能够正常前向传播
            q_value = critic.forward(state, action)
            assert q_value.shape == (16, 1)
            assert torch.all(torch.isfinite(q_value))
            
    def test_numerical_stability(self, critic_network, critic_config):
        """测试数值稳定性"""
        # 测试极端输入值
        extreme_states = [
            torch.full((4, critic_config.state_dim), 1e6),   # 很大的值
            torch.full((4, critic_config.state_dim), -1e6),  # 很小的值
            torch.zeros(4, critic_config.state_dim),         # 零值
        ]
        
        extreme_actions = [
            torch.ones(4, critic_config.action_dim) / critic_config.action_dim,  # 均匀分布
            torch.zeros(4, critic_config.action_dim),  # 零权重（需要处理）
        ]
        
        # 处理零权重动作
        extreme_actions[1][:, 0] = 1.0  # 全部权重给第一个资产
        
        for i, state in enumerate(extreme_states):
            for j, action in enumerate(extreme_actions):
                q_value = critic_network.forward(state, action)
                
                # 输出应该是有限的
                assert torch.all(torch.isfinite(q_value)), f"极端输入 state_{i}, action_{j} 产生了无限值"
                
    def test_state_action_interaction(self, critic_network, critic_config):
        """测试状态-动作交互"""
        batch_size = 16
        
        # 固定状态，变化动作
        fixed_state = torch.randn(1, critic_config.state_dim).repeat(batch_size, 1)
        
        # 生成不同的动作
        actions = []
        for i in range(batch_size):
            action = torch.zeros(critic_config.action_dim)
            action[i % critic_config.action_dim] = 1.0  # 单一资产权重为1
            actions.append(action)
        
        actions = torch.stack(actions)
        
        q_values = critic_network.forward(fixed_state, actions)
        
        # 不同动作应该产生不同的Q值
        q_values_unique = torch.unique(q_values.round(decimals=4))
        assert len(q_values_unique) > 1, "相同状态下不同动作应该产生不同Q值"
        
    def test_weight_initialization(self, critic_config):
        """测试权重初始化"""
        critic = Critic(critic_config)
        
        # 检查权重是否在合理范围内
        for name, param in critic.named_parameters():
            if 'weight' in name:
                # 权重应该不全为零
                assert not torch.all(param == 0), f"权重 {name} 全为零"
                
                # 权重应该在合理范围内
                assert torch.all(torch.abs(param) < 10), f"权重 {name} 过大"
                
            elif 'bias' in name:
                # 偏置通常初始化为零或小值
                assert torch.all(torch.abs(param) < 1), f"偏置 {name} 过大"
                
    def test_output_sensitivity(self, critic_network, sample_state, sample_action):
        """测试输出对输入的敏感性"""
        sample_state.requires_grad_(True)
        sample_action.requires_grad_(True)
        
        q_value = critic_network.forward(sample_state, sample_action)
        loss = torch.mean(q_value)
        loss.backward()
        
        # 检查梯度的大小
        state_grad_norm = torch.norm(sample_state.grad)
        action_grad_norm = torch.norm(sample_action.grad)
        
        # 梯度不应该过大或过小
        assert state_grad_norm > 1e-6, "状态梯度过小，可能存在梯度消失"
        assert state_grad_norm < 1e3, "状态梯度过大，可能存在梯度爆炸"
        assert action_grad_norm > 1e-6, "动作梯度过小，可能存在梯度消失"
        assert action_grad_norm < 1e3, "动作梯度过大，可能存在梯度爆炸"
        
    @pytest.mark.parametrize("state_dim,action_dim,hidden_dim", [
        (128, 50, 256),
        (512, 200, 1024),
        (64, 10, 128)
    ])
    def test_different_dimensions(self, state_dim, action_dim, hidden_dim):
        """测试不同维度配置"""
        config = CriticConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        critic = Critic(config)
        state = torch.randn(8, state_dim)
        action = torch.rand(8, action_dim)
        action = action / action.sum(dim=1, keepdim=True)
        
        q_value = critic.forward(state, action)
        
        assert q_value.shape == (8, 1)
        assert torch.all(torch.isfinite(q_value))
        
    def test_training_vs_eval_mode(self, critic_network, sample_state, sample_action):
        """测试训练模式vs评估模式"""
        # 训练模式
        critic_network.train()
        q_value_train1 = critic_network.forward(sample_state, sample_action)
        q_value_train2 = critic_network.forward(sample_state, sample_action)
        
        # 评估模式
        critic_network.eval()
        q_value_eval1 = critic_network.forward(sample_state, sample_action)
        q_value_eval2 = critic_network.forward(sample_state, sample_action)
        
        # 评估模式下应该是确定性的
        assert torch.allclose(q_value_eval1, q_value_eval2, atol=1e-6)
        
        # 如果有dropout，训练模式可能有随机性
        if critic_network.config.dropout > 0:
            # 训练模式可能有轻微差异（由于dropout）
            pass
        else:
            # 没有dropout时，训练模式也应该是确定性的
            assert torch.allclose(q_value_train1, q_value_train2, atol=1e-6)