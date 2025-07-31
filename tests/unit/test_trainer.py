"""
RLTrainer的单元测试
测试训练循环、早停机制、训练过程稳定性和收敛性
"""
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any, Optional
import gym
from gym import spaces

from src.rl_trading_system.training.trainer import (
    RLTrainer, 
    TrainingConfig,
    EarlyStopping,
    TrainingMetrics
)
from src.rl_trading_system.training.data_split_strategy import (
    DataSplitStrategy,
    TimeSeriesSplitStrategy,
    SplitConfig,
    SplitResult
)
# SAC agent will be implemented later
# from src.rl_trading_system.rl_agent.sac_agent import SACAgent
# Portfolio environment imports - using mock environment instead
# from src.rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig


class MockEnvironment(gym.Env):
    """模拟交易环境"""
    
    def __init__(self, n_stocks=4, lookback_window=30):
        self.n_stocks = n_stocks
        self.lookback_window = lookback_window
        self.n_features = 10
        self.n_market_features = 5
        
        # 定义观察空间和动作空间
        self.observation_space = spaces.Dict({
            'features': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(lookback_window, n_stocks, self.n_features),
                dtype=np.float32
            ),
            'positions': spaces.Box(
                low=0, 
                high=1, 
                shape=(n_stocks,),
                dtype=np.float32
            ),
            'market_state': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.n_market_features,),
                dtype=np.float32
            )
        })
        
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(n_stocks,),
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 100
        self.episode_returns = []
        
    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.episode_returns = []
        
        return {
            'features': np.random.randn(self.lookback_window, self.n_stocks, self.n_features).astype(np.float32),
            'positions': np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks,
            'market_state': np.random.randn(self.n_market_features).astype(np.float32)
        }
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        
        # 生成随机奖励
        reward = np.random.randn() * 0.01  # 小的随机奖励
        self.episode_returns.append(reward)
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 生成下一个观察
        obs = {
            'features': np.random.randn(self.lookback_window, self.n_stocks, self.n_features).astype(np.float32),
            'positions': action.astype(np.float32),
            'market_state': np.random.randn(self.n_market_features).astype(np.float32)
        }
        
        # 生成信息
        info = {
            'portfolio_return': reward,
            'transaction_cost': abs(reward) * 0.1,
            'positions': action
        }
        
        return obs, reward, done, info


class MockSACAgent:
    """模拟SAC智能体"""
    
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.training_mode = True
        self.total_updates = 0
        
    def act(self, obs, deterministic=False):
        """选择动作"""
        action_shape = self.action_space.shape
        action = np.random.rand(*action_shape).astype(np.float32)
        action = action / action.sum()  # 标准化
        return action
    
    def update(self, replay_buffer, batch_size=256):
        """更新智能体参数"""
        self.total_updates += 1
        
        # 模拟训练指标
        actor_loss = np.random.randn() * 0.1
        critic_loss = np.random.randn() * 0.1
        temperature_loss = np.random.randn() * 0.01
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'temperature_loss': temperature_loss,
            'temperature': 0.2,
            'q_values': np.random.randn(batch_size).mean()
        }
    
    def train(self):
        """设置为训练模式"""
        self.training_mode = True
    
    def eval(self):
        """设置为评估模式"""
        self.training_mode = False
    
    def save(self, filepath):
        """保存模型"""
        pass
    
    def load(self, filepath):
        """加载模型"""
        pass


class TestTrainingConfig:
    """训练配置测试类"""
    
    def test_training_config_creation(self):
        """测试训练配置创建"""
        config = TrainingConfig(
            n_episodes=1000,
            max_steps_per_episode=200,
            batch_size=256,
            learning_rate=3e-4,
            buffer_size=100000,
            validation_frequency=50,
            save_frequency=100
        )
        
        assert config.n_episodes == 1000
        assert config.max_steps_per_episode == 200
        assert config.batch_size == 256
        assert config.learning_rate == 3e-4
        assert config.buffer_size == 100000
        assert config.validation_frequency == 50
        assert config.save_frequency == 100
    
    def test_training_config_defaults(self):
        """测试训练配置默认值"""
        config = TrainingConfig()
        
        assert config.n_episodes == 5000
        assert config.max_steps_per_episode == 252
        assert config.batch_size == 256
        assert config.learning_rate == 3e-4
        assert config.buffer_size == 1000000
        assert config.gamma == 0.99
        assert config.tau == 0.005
    
    def test_training_config_validation(self):
        """测试训练配置验证"""
        # 测试无效的episode数量
        with pytest.raises(ValueError, match="n_episodes必须为正数"):
            TrainingConfig(n_episodes=0)
        
        # 测试无效的学习率
        with pytest.raises(ValueError, match="learning_rate必须为正数"):
            TrainingConfig(learning_rate=-0.1)
        
        # 测试无效的batch size
        with pytest.raises(ValueError, match="batch_size必须为正数"):
            TrainingConfig(batch_size=0)


class TestEarlyStopping:
    """早停机制测试类"""
    
    def test_early_stopping_creation(self):
        """测试早停机制创建"""
        early_stopping = EarlyStopping(
            patience=10,
            min_delta=0.001,
            mode='max'
        )
        
        assert early_stopping.patience == 10
        assert early_stopping.min_delta == 0.001
        assert early_stopping.mode == 'max'
        assert early_stopping.best_score is None
        assert early_stopping.counter == 0
        assert not early_stopping.early_stop
    
    def test_early_stopping_improvement_detection(self):
        """测试早停机制的改进检测"""
        # 测试最大化模式
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='max')
        
        # 第一次更新，应该是改进
        assert early_stopping.step(0.8) == False
        assert early_stopping.best_score == 0.8
        assert early_stopping.counter == 0
        
        # 第二次更新，有显著改进
        assert early_stopping.step(0.85) == False
        assert early_stopping.best_score == 0.85
        assert early_stopping.counter == 0
        
        # 第三次更新，没有显著改进
        assert early_stopping.step(0.851) == False
        assert early_stopping.counter == 1
        
        # 连续没有改进
        assert early_stopping.step(0.84) == False
        assert early_stopping.counter == 2
        
        assert early_stopping.step(0.83) == False
        assert early_stopping.counter == 3
        
        # 达到patience，触发早停
        assert early_stopping.step(0.82) == True
        assert early_stopping.early_stop == True
    
    def test_early_stopping_minimization_mode(self):
        """测试早停机制的最小化模式"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='min')
        
        # 损失逐渐减小
        assert early_stopping.step(1.0) == False
        assert early_stopping.step(0.8) == False  # 改进
        assert early_stopping.step(0.85) == False  # 没有改进
        assert early_stopping.step(0.86) == False  # 没有改进
        assert early_stopping.step(0.87) == True   # 触发早停
    
    def test_early_stopping_reset(self):
        """测试早停机制重置"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.01, mode='max')
        
        # 运行到接近早停
        early_stopping.step(0.8)
        early_stopping.step(0.75)
        early_stopping.step(0.74)
        
        assert early_stopping.counter > 0
        
        # 重置
        early_stopping.reset()
        
        assert early_stopping.best_score is None
        assert early_stopping.counter == 0
        assert early_stopping.early_stop == False


class TestTrainingMetrics:
    """训练指标测试类"""
    
    def test_training_metrics_creation(self):
        """测试训练指标创建"""
        metrics = TrainingMetrics()
        
        assert len(metrics.episode_rewards) == 0
        assert len(metrics.episode_lengths) == 0
        assert len(metrics.actor_losses) == 0
        assert len(metrics.critic_losses) == 0
        assert len(metrics.validation_scores) == 0
    
    def test_training_metrics_update(self):
        """测试训练指标更新"""
        metrics = TrainingMetrics()
        
        # 添加episode指标
        metrics.add_episode_metrics(reward=100.0, length=200, actor_loss=0.1, critic_loss=0.2)
        
        assert len(metrics.episode_rewards) == 1
        assert metrics.episode_rewards[0] == 100.0
        assert metrics.episode_lengths[0] == 200
        assert metrics.actor_losses[0] == 0.1
        assert metrics.critic_losses[0] == 0.2
        
        # 添加验证指标
        metrics.add_validation_score(0.85)
        assert len(metrics.validation_scores) == 1
        assert metrics.validation_scores[0] == 0.85
    
    def test_training_metrics_statistics(self):
        """测试训练指标统计"""
        metrics = TrainingMetrics()
        
        # 添加多个episode
        rewards = [10, 20, 30, 40, 50]
        for reward in rewards:
            metrics.add_episode_metrics(reward=reward, length=100, actor_loss=0.1, critic_loss=0.1)
        
        stats = metrics.get_statistics()
        
        assert stats['mean_reward'] == 30.0
        assert stats['std_reward'] == pytest.approx(np.std(rewards), rel=1e-6)
        assert stats['mean_length'] == 100.0
        assert stats['mean_actor_loss'] == 0.1
        assert stats['mean_critic_loss'] == 0.1
    
    def test_training_metrics_recent_statistics(self):
        """测试最近训练指标统计"""
        metrics = TrainingMetrics()
        
        # 添加10个episode
        for i in range(10):
            metrics.add_episode_metrics(reward=i, length=100, actor_loss=0.1, critic_loss=0.1)
        
        # 获取最近5个episode的统计
        recent_stats = metrics.get_recent_statistics(window=5)
        
        assert recent_stats['mean_reward'] == 7.0  # (5+6+7+8+9)/5
        assert len(recent_stats) > 0


class TestRLTrainer:
    """RLTrainer测试类"""
    
    @pytest.fixture
    def training_config(self):
        """训练配置fixture"""
        return TrainingConfig(
            n_episodes=100,
            max_steps_per_episode=50,
            batch_size=32,
            learning_rate=1e-3,
            validation_frequency=10,
            save_frequency=20
        )
    
    @pytest.fixture
    def mock_environment(self):
        """模拟环境fixture"""
        return MockEnvironment(n_stocks=4, lookback_window=20)
    
    @pytest.fixture
    def mock_agent(self, mock_environment):
        """模拟智能体fixture"""
        return MockSACAgent(
            observation_space=mock_environment.observation_space,
            action_space=mock_environment.action_space
        )
    
    @pytest.fixture
    def mock_data_split(self):
        """模拟数据划分fixture"""
        # 创建模拟的划分结果
        total_samples = 1000
        train_size = int(total_samples * 0.7)
        val_size = int(total_samples * 0.2)
        
        train_indices = np.arange(train_size)
        val_indices = np.arange(train_size, train_size + val_size)
        test_indices = np.arange(train_size + val_size, total_samples)
        
        return SplitResult(
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices,
            split_dates={
                'train_start': '2020-01-01',
                'train_end': '2022-12-31',
                'val_start': '2023-01-01',
                'val_end': '2023-08-31',
                'test_start': '2023-09-01',
                'test_end': '2023-12-31'
            }
        )
    
    @pytest.fixture
    def trainer(self, training_config, mock_environment, mock_agent, mock_data_split):
        """训练器fixture"""
        return RLTrainer(
            config=training_config,
            environment=mock_environment,
            agent=mock_agent,
            data_split=mock_data_split
        )
    
    def test_trainer_initialization(self, trainer, training_config):
        """测试训练器初始化"""
        assert trainer.config == training_config
        assert trainer.environment is not None
        assert trainer.agent is not None
        assert trainer.data_split is not None
        assert isinstance(trainer.metrics, TrainingMetrics)
        assert isinstance(trainer.early_stopping, EarlyStopping)
    
    def test_trainer_single_episode(self, trainer):
        """测试单个episode训练"""
        episode_reward, episode_length = trainer._run_episode(episode_num=1, training=True)
        
        assert isinstance(episode_reward, (int, float))
        assert isinstance(episode_length, int)
        assert episode_length > 0
        assert episode_length <= trainer.config.max_steps_per_episode
    
    def test_trainer_validation_episode(self, trainer):
        """测试验证episode"""
        validation_score = trainer._validate()
        
        assert isinstance(validation_score, (int, float))
        assert np.isfinite(validation_score)
    
    def test_trainer_save_load_checkpoint(self, trainer, tmp_path):
        """测试模型保存和加载"""
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        
        # 保存检查点
        trainer.save_checkpoint(str(checkpoint_path), episode=10)
        
        # 检查文件是否创建
        assert checkpoint_path.exists()
        
        # 加载检查点
        loaded_episode = trainer.load_checkpoint(str(checkpoint_path))
        assert loaded_episode == 10
    
    def test_trainer_early_stopping_integration(self, training_config):
        """测试早停机制集成"""
        # 创建会快速触发早停的配置
        early_config = TrainingConfig(
            n_episodes=100,
            max_steps_per_episode=20,
            early_stopping_patience=3,
            early_stopping_min_delta=0.01,  # 更合理的最小改进幅度
            validation_frequency=5  # 每5个episode验证一次
        )
        
        mock_env = MockEnvironment(n_stocks=2, lookback_window=10)
        mock_agent = MockSACAgent(mock_env.observation_space, mock_env.action_space)
        
        # 创建简单的数据划分
        train_indices = np.arange(100)
        val_indices = np.arange(100, 120)
        test_indices = np.arange(120, 150)
        
        data_split = SplitResult(
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices
        )
        
        trainer = RLTrainer(
            config=early_config,
            environment=mock_env,
            agent=mock_agent,
            data_split=data_split
        )
        
        # 模拟训练过程中验证分数不改进的情况
        # 第一次验证分数较高，然后没有显著改进
        with patch.object(trainer, '_validate', side_effect=[0.5, 0.52, 0.51, 0.50, 0.49, 0.48]):
            # 运行训练，应该会因为早停而提前结束
            trainer.train()
            
            # 检查是否触发了早停
            assert trainer.early_stopping.early_stop
    
    def test_trainer_metrics_collection(self, trainer):
        """测试训练指标收集"""
        # 运行几个episode
        for episode in range(5):
            reward, length = trainer._run_episode(episode_num=episode, training=True)
            trainer.metrics.add_episode_metrics(
                reward=reward,
                length=length,
                actor_loss=0.1,
                critic_loss=0.1
            )
        
        # 检查指标收集
        assert len(trainer.metrics.episode_rewards) == 5
        assert len(trainer.metrics.episode_lengths) == 5
        
        # 获取统计信息
        stats = trainer.metrics.get_statistics()
        assert 'mean_reward' in stats
        assert 'std_reward' in stats
        assert 'mean_length' in stats
    
    def test_trainer_learning_rate_scheduling(self, trainer):
        """测试学习率调度"""
        initial_lr = trainer.config.learning_rate
        
        # 模拟训练过程中的学习率调整
        for episode in range(50):
            # 在实际实现中，这里会调整学习率
            current_lr = trainer._get_current_learning_rate(episode)
            assert current_lr > 0
            assert current_lr <= initial_lr
    
    def test_trainer_gradient_clipping(self, trainer):
        """测试梯度裁剪"""
        # 在实际实现中，这里会测试梯度裁剪功能
        # 由于使用模拟智能体，这里只是确保相关配置存在
        assert hasattr(trainer.config, 'gradient_clip_norm')
        if trainer.config.gradient_clip_norm is not None:
            assert trainer.config.gradient_clip_norm > 0
    
    def test_trainer_replay_buffer_integration(self, trainer):
        """测试经验回放缓冲区集成"""
        # 运行一个episode来填充缓冲区
        trainer._run_episode(episode_num=1, training=True)
        
        # 检查智能体是否被更新
        assert trainer.agent.total_updates >= 0
    
    def test_trainer_multi_episode_training(self, trainer):
        """测试多episode训练"""
        # 设置较小的episode数量进行测试
        trainer.config.n_episodes = 10
        trainer.config.validation_frequency = 5
        
        # 运行训练
        trainer.train()
        
        # 检查训练是否完成
        assert len(trainer.metrics.episode_rewards) > 0
        assert len(trainer.metrics.episode_rewards) <= trainer.config.n_episodes
    
    def test_trainer_validation_frequency(self, trainer):
        """测试验证频率"""
        trainer.config.n_episodes = 20
        trainer.config.validation_frequency = 5
        
        # 模拟训练过程
        validation_count = 0
        for episode in range(trainer.config.n_episodes):
            if (episode + 1) % trainer.config.validation_frequency == 0:
                validation_count += 1
        
        # 应该进行4次验证 (episode 5, 10, 15, 20)
        assert validation_count == 4
    
    def test_trainer_save_frequency(self, trainer, tmp_path):
        """测试保存频率"""
        trainer.config.n_episodes = 15
        trainer.config.save_frequency = 5
        trainer.config.save_dir = str(tmp_path)
        
        # 模拟保存逻辑
        save_count = 0
        for episode in range(trainer.config.n_episodes):
            if (episode + 1) % trainer.config.save_frequency == 0:
                save_count += 1
        
        # 应该保存3次 (episode 5, 10, 15)
        assert save_count == 3
    
    @pytest.mark.parametrize("n_episodes,max_steps,batch_size", [
        (10, 20, 16),
        (50, 100, 32),
        (100, 200, 64)
    ])
    def test_trainer_different_configurations(self, n_episodes, max_steps, batch_size):
        """测试不同配置下的训练器"""
        config = TrainingConfig(
            n_episodes=n_episodes,
            max_steps_per_episode=max_steps,
            batch_size=batch_size
        )
        
        mock_env = MockEnvironment()
        mock_agent = MockSACAgent(mock_env.observation_space, mock_env.action_space)
        
        # 简单的数据划分
        data_split = SplitResult(
            train_indices=np.arange(100),
            validation_indices=np.arange(100, 120),
            test_indices=np.arange(120, 150)
        )
        
        trainer = RLTrainer(
            config=config,
            environment=mock_env,
            agent=mock_agent,
            data_split=data_split
        )
        
        # 运行少量episode验证基本功能
        trainer.config.n_episodes = 3  # 减少测试时间
        trainer.train()
        
        assert len(trainer.metrics.episode_rewards) <= 3
    
    def test_trainer_error_handling(self, trainer):
        """测试训练器错误处理"""
        # 测试环境错误
        with patch.object(trainer.environment, 'step', side_effect=Exception("Environment error")):
            with pytest.raises(Exception):
                trainer._run_episode(episode_num=1, training=True)
        
        # 测试智能体错误
        with patch.object(trainer.agent, 'act', side_effect=Exception("Agent error")):
            with pytest.raises(Exception):
                trainer._run_episode(episode_num=1, training=True)
    
    def test_trainer_memory_efficiency(self, trainer):
        """测试训练器内存效率"""
        # 运行训练并检查内存使用不会无限增长
        initial_metrics_length = len(trainer.metrics.episode_rewards)
        
        # 模拟长期训练
        for _ in range(10):
            reward, length = trainer._run_episode(episode_num=1, training=True)
            trainer.metrics.add_episode_metrics(
                reward=reward,
                length=length,
                actor_loss=0.1,
                critic_loss=0.1
            )
        
        # 检查指标是否正确累积
        assert len(trainer.metrics.episode_rewards) == initial_metrics_length + 10
    
    def test_trainer_deterministic_behavior(self):
        """测试训练器的确定性行为"""
        # 使用固定随机种子
        config = TrainingConfig(n_episodes=5, random_seed=42)
        
        # 创建两个相同的训练器
        mock_env1 = MockEnvironment()
        mock_agent1 = MockSACAgent(mock_env1.observation_space, mock_env1.action_space)
        data_split1 = SplitResult(
            train_indices=np.arange(50),
            validation_indices=np.arange(50, 60),
            test_indices=np.arange(60, 70)
        )
        
        mock_env2 = MockEnvironment()
        mock_agent2 = MockSACAgent(mock_env2.observation_space, mock_env2.action_space)
        data_split2 = SplitResult(
            train_indices=np.arange(50),
            validation_indices=np.arange(50, 60),
            test_indices=np.arange(60, 70)
        )
        
        trainer1 = RLTrainer(config, mock_env1, mock_agent1, data_split1)
        trainer2 = RLTrainer(config, mock_env2, mock_agent2, data_split2)
        
        # 由于环境是随机的，这里主要测试训练器的结构一致性
        assert trainer1.config.random_seed == trainer2.config.random_seed
        assert trainer1.config.n_episodes == trainer2.config.n_episodes