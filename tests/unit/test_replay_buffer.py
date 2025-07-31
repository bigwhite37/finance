"""
测试经验回放缓冲区的单元测试
"""
import pytest
import torch
import numpy as np
import multiprocessing as mp
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from src.rl_trading_system.models.replay_buffer import (
    ReplayBuffer, 
    PrioritizedReplayBuffer, 
    Experience, 
    ReplayBufferConfig
)


@dataclass
class MockExperience:
    """模拟经验数据"""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    info: Dict[str, Any] = None


class TestReplayBuffer:
    """经验回放缓冲区测试类"""
    
    @pytest.fixture
    def buffer_config(self):
        """缓冲区配置fixture"""
        return ReplayBufferConfig(
            capacity=1000,
            batch_size=32,
            state_dim=256,
            action_dim=100,
            device='cpu'
        )
    
    @pytest.fixture
    def replay_buffer(self, buffer_config):
        """回放缓冲区fixture"""
        return ReplayBuffer(buffer_config)
    
    @pytest.fixture
    def sample_experience(self, buffer_config):
        """样本经验fixture"""
        return Experience(
            state=torch.randn(buffer_config.state_dim),
            action=torch.rand(buffer_config.action_dim),
            reward=np.random.uniform(-1, 1),
            next_state=torch.randn(buffer_config.state_dim),
            done=np.random.choice([True, False]),
            info={'step': 0}
        )
    
    def test_buffer_initialization(self, replay_buffer, buffer_config):
        """测试缓冲区初始化"""
        assert replay_buffer.capacity == buffer_config.capacity
        assert replay_buffer.batch_size == buffer_config.batch_size
        assert replay_buffer.size == 0
        assert replay_buffer.position == 0
        assert len(replay_buffer.buffer) == buffer_config.capacity
        
    def test_add_experience(self, replay_buffer, sample_experience):
        """测试添加经验"""
        initial_size = replay_buffer.size
        replay_buffer.add(sample_experience)
        
        assert replay_buffer.size == initial_size + 1
        assert replay_buffer.position == 1
        
        # 检查存储的经验
        stored_exp = replay_buffer.buffer[0]
        assert torch.allclose(stored_exp.state, sample_experience.state)
        assert torch.allclose(stored_exp.action, sample_experience.action)
        assert stored_exp.reward == sample_experience.reward
        assert stored_exp.done == sample_experience.done
        
    def test_buffer_overflow(self, replay_buffer, buffer_config):
        """测试缓冲区溢出处理"""
        # 填满缓冲区
        for i in range(buffer_config.capacity + 10):
            exp = Experience(
                state=torch.randn(buffer_config.state_dim),
                action=torch.rand(buffer_config.action_dim),
                reward=float(i),
                next_state=torch.randn(buffer_config.state_dim),
                done=False,
                info={'step': i}
            )
            replay_buffer.add(exp)
        
        # 缓冲区大小不应超过容量
        assert replay_buffer.size == buffer_config.capacity
        
        # 位置应该循环
        assert replay_buffer.position == 10
        
        # 检查最新的经验是否正确存储
        latest_exp = replay_buffer.buffer[9]  # position - 1
        assert latest_exp.reward == float(buffer_config.capacity + 9)
        
    def test_sample_batch(self, replay_buffer, buffer_config):
        """测试批量采样"""
        # 添加足够的经验
        experiences = []
        for i in range(buffer_config.batch_size * 2):
            exp = Experience(
                state=torch.randn(buffer_config.state_dim),
                action=torch.rand(buffer_config.action_dim),
                reward=float(i),
                next_state=torch.randn(buffer_config.state_dim),
                done=i % 10 == 0,
                info={'step': i}
            )
            experiences.append(exp)
            replay_buffer.add(exp)
        
        # 采样批次
        batch = replay_buffer.sample()
        
        assert len(batch) == buffer_config.batch_size
        assert all(isinstance(exp, Experience) for exp in batch)
        
        # 检查批次数据的形状
        states = torch.stack([exp.state for exp in batch])
        actions = torch.stack([exp.action for exp in batch])
        
        assert states.shape == (buffer_config.batch_size, buffer_config.state_dim)
        assert actions.shape == (buffer_config.batch_size, buffer_config.action_dim)
        
    def test_sample_insufficient_data(self, replay_buffer, buffer_config):
        """测试数据不足时的采样"""
        # 只添加少量经验
        for i in range(buffer_config.batch_size // 2):
            exp = Experience(
                state=torch.randn(buffer_config.state_dim),
                action=torch.rand(buffer_config.action_dim),
                reward=float(i),
                next_state=torch.randn(buffer_config.state_dim),
                done=False
            )
            replay_buffer.add(exp)
        
        # 应该抛出异常或返回所有可用数据
        with pytest.raises(ValueError):
            replay_buffer.sample()
            
    def test_can_sample(self, replay_buffer, buffer_config):
        """测试是否可以采样"""
        # 初始状态不能采样
        assert not replay_buffer.can_sample()
        
        # 添加足够的经验后可以采样
        for i in range(buffer_config.batch_size):
            exp = Experience(
                state=torch.randn(buffer_config.state_dim),
                action=torch.rand(buffer_config.action_dim),
                reward=float(i),
                next_state=torch.randn(buffer_config.state_dim),
                done=False
            )
            replay_buffer.add(exp)
        
        assert replay_buffer.can_sample()
        
    def test_clear_buffer(self, replay_buffer, buffer_config):
        """测试清空缓冲区"""
        # 添加一些经验
        for i in range(10):
            exp = Experience(
                state=torch.randn(buffer_config.state_dim),
                action=torch.rand(buffer_config.action_dim),
                reward=float(i),
                next_state=torch.randn(buffer_config.state_dim),
                done=False
            )
            replay_buffer.add(exp)
        
        assert replay_buffer.size > 0
        
        # 清空缓冲区
        replay_buffer.clear()
        
        assert replay_buffer.size == 0
        assert replay_buffer.position == 0
        
    def test_get_all_experiences(self, replay_buffer, buffer_config):
        """测试获取所有经验"""
        experiences = []
        for i in range(50):
            exp = Experience(
                state=torch.randn(buffer_config.state_dim),
                action=torch.rand(buffer_config.action_dim),
                reward=float(i),
                next_state=torch.randn(buffer_config.state_dim),
                done=False,
                info={'step': i}
            )
            experiences.append(exp)
            replay_buffer.add(exp)
        
        all_exp = replay_buffer.get_all_experiences()
        
        assert len(all_exp) == 50
        assert all(isinstance(exp, Experience) for exp in all_exp)
        
        # 检查顺序是否正确
        for i, exp in enumerate(all_exp):
            assert exp.info['step'] == i
            
    def test_memory_efficiency(self, buffer_config):
        """测试内存效率"""
        # 创建大容量缓冲区
        large_config = ReplayBufferConfig(
            capacity=10000,
            batch_size=64,
            state_dim=buffer_config.state_dim,
            action_dim=buffer_config.action_dim
        )
        
        large_buffer = ReplayBuffer(large_config)
        
        # 添加经验并检查内存使用
        for i in range(1000):
            exp = Experience(
                state=torch.randn(buffer_config.state_dim),
                action=torch.rand(buffer_config.action_dim),
                reward=float(i),
                next_state=torch.randn(buffer_config.state_dim),
                done=False
            )
            large_buffer.add(exp)
        
        # 缓冲区应该正常工作
        assert large_buffer.size == 1000
        assert large_buffer.can_sample()
        
        batch = large_buffer.sample()
        assert len(batch) == large_config.batch_size


class TestPrioritizedReplayBuffer:
    """优先级回放缓冲区测试类"""
    
    @pytest.fixture
    def priority_config(self):
        """优先级缓冲区配置fixture"""
        return ReplayBufferConfig(
            capacity=1000,
            batch_size=32,
            state_dim=256,
            action_dim=100,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
            epsilon=1e-6,
            device='cpu'
        )
    
    @pytest.fixture
    def priority_buffer(self, priority_config):
        """优先级回放缓冲区fixture"""
        return PrioritizedReplayBuffer(priority_config)
    
    def test_priority_buffer_initialization(self, priority_buffer, priority_config):
        """测试优先级缓冲区初始化"""
        assert priority_buffer.capacity == priority_config.capacity
        assert priority_buffer.alpha == priority_config.alpha
        assert priority_buffer.beta == priority_config.beta
        assert priority_buffer.beta_increment == priority_config.beta_increment
        assert hasattr(priority_buffer, 'priorities')
        assert hasattr(priority_buffer, 'max_priority')
        
    def test_add_with_priority(self, priority_buffer, priority_config):
        """测试带优先级的添加"""
        exp = Experience(
            state=torch.randn(priority_config.state_dim),
            action=torch.rand(priority_config.action_dim),
            reward=1.0,
            next_state=torch.randn(priority_config.state_dim),
            done=False
        )
        
        initial_max_priority = priority_buffer.max_priority
        priority_buffer.add(exp, priority=0.8)
        
        assert priority_buffer.size == 1
        assert priority_buffer.max_priority >= initial_max_priority
        
    def test_priority_sampling(self, priority_buffer, priority_config):
        """测试优先级采样"""
        # 添加不同优先级的经验
        priorities = [0.1, 0.5, 0.9, 0.3, 0.7]
        
        for i, priority in enumerate(priorities * 10):  # 重复以获得足够的样本
            exp = Experience(
                state=torch.randn(priority_config.state_dim),
                action=torch.rand(priority_config.action_dim),
                reward=float(i),
                next_state=torch.randn(priority_config.state_dim),
                done=False,
                info={'priority': priority}
            )
            priority_buffer.add(exp, priority=priority)
        
        # 采样批次
        batch, indices, weights = priority_buffer.sample()
        
        assert len(batch) == priority_config.batch_size
        assert len(indices) == priority_config.batch_size
        assert len(weights) == priority_config.batch_size
        
        # 检查重要性权重
        assert torch.all(weights > 0)
        assert torch.all(torch.isfinite(weights))
        
    def test_update_priorities(self, priority_buffer, priority_config):
        """测试更新优先级"""
        # 添加一些经验
        for i in range(priority_config.batch_size * 2):
            exp = Experience(
                state=torch.randn(priority_config.state_dim),
                action=torch.rand(priority_config.action_dim),
                reward=float(i),
                next_state=torch.randn(priority_config.state_dim),
                done=False
            )
            priority_buffer.add(exp, priority=0.5)
        
        # 采样并更新优先级
        batch, indices, weights = priority_buffer.sample()
        
        new_priorities = torch.rand(len(indices)) + 0.1  # 避免零优先级
        priority_buffer.update_priorities(indices, new_priorities)
        
        # 验证优先级已更新（考虑alpha指数）
        for idx, new_priority in zip(indices, new_priorities):
            stored_priority = priority_buffer.priorities[idx]
            expected_priority = (new_priority.item() ** priority_config.alpha)
            assert abs(stored_priority - expected_priority) < 1e-6
            
    def test_beta_annealing(self, priority_buffer, priority_config):
        """测试beta参数退火"""
        # 先添加足够的经验
        for i in range(priority_config.batch_size * 2):
            exp = Experience(
                state=torch.randn(priority_config.state_dim),
                action=torch.rand(priority_config.action_dim),
                reward=float(i),
                next_state=torch.randn(priority_config.state_dim),
                done=False
            )
            priority_buffer.add(exp, priority=0.5)
        
        initial_beta = priority_buffer.beta
        
        # 多次采样应该增加beta
        for _ in range(100):
            if priority_buffer.can_sample():
                priority_buffer.sample()
        
        assert priority_buffer.beta > initial_beta
        assert priority_buffer.beta <= 1.0
        
    def test_importance_sampling_weights(self, priority_buffer, priority_config):
        """测试重要性采样权重计算"""
        # 添加不同优先级的经验
        high_priority_exp = Experience(
            state=torch.randn(priority_config.state_dim),
            action=torch.rand(priority_config.action_dim),
            reward=1.0,
            next_state=torch.randn(priority_config.state_dim),
            done=False
        )
        
        low_priority_exp = Experience(
            state=torch.randn(priority_config.state_dim),
            action=torch.rand(priority_config.action_dim),
            reward=0.0,
            next_state=torch.randn(priority_config.state_dim),
            done=False
        )
        
        # 添加多个高优先级和低优先级经验
        for _ in range(priority_config.batch_size):
            priority_buffer.add(high_priority_exp, priority=0.9)
            priority_buffer.add(low_priority_exp, priority=0.1)
        
        # 采样多次并检查权重分布
        weight_sums = []
        for _ in range(10):
            batch, indices, weights = priority_buffer.sample()
            weight_sums.append(torch.sum(weights).item())
        
        # 权重应该有合理的分布
        avg_weight_sum = np.mean(weight_sums)
        assert avg_weight_sum > 0
        assert avg_weight_sum < priority_config.batch_size * 10  # 不应该过大


class TestMultiprocessReplayBuffer:
    """多进程回放缓冲区测试类"""
    
    @pytest.fixture
    def mp_config(self):
        """多进程配置fixture"""
        return ReplayBufferConfig(
            capacity=1000,
            batch_size=32,
            state_dim=64,  # 较小的维度以加快测试
            action_dim=10,
            n_workers=2,
            device='cpu'
        )
    
    def test_multiprocess_add(self, mp_config):
        """测试多进程添加经验"""
        # 这个测试需要实际的多进程实现
        # 这里提供测试框架
        pass
    
    def test_concurrent_sampling(self, mp_config):
        """测试并发采样"""
        # 测试多个进程同时采样的情况
        pass
    
    def test_data_consistency(self, mp_config):
        """测试数据一致性"""
        # 测试多进程环境下的数据一致性
        pass


class TestReplayBufferIntegration:
    """回放缓冲区集成测试"""
    
    def test_with_actor_critic(self):
        """测试与Actor-Critic网络的集成"""
        # 创建模拟的Actor和Critic网络
        from src.rl_trading_system.models.actor_network import Actor, ActorConfig
        from src.rl_trading_system.models.critic_network import Critic, CriticConfig
        
        actor_config = ActorConfig(state_dim=64, action_dim=10, hidden_dim=128)
        critic_config = CriticConfig(state_dim=64, action_dim=10, hidden_dim=128)
        buffer_config = ReplayBufferConfig(capacity=1000, batch_size=16, state_dim=64, action_dim=10)
        
        actor = Actor(actor_config)
        critic = Critic(critic_config)
        buffer = ReplayBuffer(buffer_config)
        
        # 生成一些经验
        for i in range(50):
            state = torch.randn(64)
            action, _ = actor.get_action(state.unsqueeze(0), deterministic=True)
            action = action.squeeze(0)
            
            exp = Experience(
                state=state,
                action=action,
                reward=np.random.uniform(-1, 1),
                next_state=torch.randn(64),
                done=np.random.choice([True, False])
            )
            buffer.add(exp)
        
        # 采样并测试网络
        if buffer.can_sample():
            batch = buffer.sample()
            
            states = torch.stack([exp.state for exp in batch])
            actions = torch.stack([exp.action for exp in batch])
            
            # 测试Critic网络
            q_values = critic(states, actions)
            assert q_values.shape == (len(batch), 1)
            
            # 测试Actor网络
            new_actions, log_probs = actor.get_action(states)
            assert new_actions.shape == (len(batch), 10)
            assert log_probs.shape == (len(batch),)
            
    def test_memory_leak_prevention(self):
        """测试内存泄漏预防"""
        config = ReplayBufferConfig(capacity=100, batch_size=16, state_dim=32, action_dim=5)
        buffer = ReplayBuffer(config)
        
        # 大量添加和采样操作
        for epoch in range(10):
            # 添加经验
            for i in range(config.capacity):
                exp = Experience(
                    state=torch.randn(config.state_dim),
                    action=torch.rand(config.action_dim),
                    reward=np.random.uniform(-1, 1),
                    next_state=torch.randn(config.state_dim),
                    done=np.random.choice([True, False])
                )
                buffer.add(exp)
            
            # 多次采样
            for _ in range(20):
                if buffer.can_sample():
                    batch = buffer.sample()
                    del batch  # 显式删除
        
        # 缓冲区应该仍然正常工作
        assert buffer.size == config.capacity
        assert buffer.can_sample()
        
    def test_serialization(self):
        """测试序列化和反序列化"""
        config = ReplayBufferConfig(capacity=100, batch_size=16, state_dim=32, action_dim=5)
        buffer = ReplayBuffer(config)
        
        # 添加一些经验
        original_experiences = []
        for i in range(50):
            exp = Experience(
                state=torch.randn(config.state_dim),
                action=torch.rand(config.action_dim),
                reward=float(i),
                next_state=torch.randn(config.state_dim),
                done=i % 10 == 0,
                info={'step': i}
            )
            original_experiences.append(exp)
            buffer.add(exp)
        
        # 保存状态
        state_dict = buffer.state_dict()
        
        # 创建新缓冲区并加载状态
        new_buffer = ReplayBuffer(config)
        new_buffer.load_state_dict(state_dict)
        
        # 验证数据一致性
        assert new_buffer.size == buffer.size
        assert new_buffer.position == buffer.position
        
        # 验证经验数据
        original_all = buffer.get_all_experiences()
        loaded_all = new_buffer.get_all_experiences()
        
        assert len(original_all) == len(loaded_all)
        
        for orig, loaded in zip(original_all, loaded_all):
            assert torch.allclose(orig.state, loaded.state)
            assert torch.allclose(orig.action, loaded.action)
            assert orig.reward == loaded.reward
            assert orig.done == loaded.done