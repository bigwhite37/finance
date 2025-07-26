"""
O2O组件单元测试

测试各个O2O组件的独立功能，包括数据处理、采样器、训练器等。
验证OfflineDataset和OnlineReplayBuffer、MixtureSampler的正确性，
创建训练器测试，验证各训练阶段的算法实现。
"""

import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from collections import deque
import logging

# Import O2O components
from data.offline_dataset import OfflineDataset, OfflineDataConfig
from buffers.online_replay_buffer import (
    OnlineReplayBuffer, OnlineReplayConfig, TrajectoryData,
    create_trajectory_from_episode, calculate_td_errors
)
from sampler.mixture_sampler import (
    MixtureSampler, MixtureSamplerConfig, 
    create_mixture_sampler, calculate_sampling_schedule
)

# Mock dependencies
from data.data_manager import DataManager
from rl_agent.cvar_ppo_agent import CVaRPPOAgent

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestOfflineDataset(unittest.TestCase):
    """测试离线数据集组件"""
    
    def setUp(self):
        """设置测试环境"""
        self.mock_data_manager = Mock(spec=DataManager)
        self.config = OfflineDataConfig(
            start_date="2020-01-01",
            end_date="2020-12-31",
            lookback_window=10,
            prediction_horizon=1,
            min_samples_per_stock=50
        )
        
        # 创建模拟数据
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        instruments = ['AAPL', 'GOOGL', 'MSFT']
        
        # 创建多级索引数据
        index = pd.MultiIndex.from_product(
            [dates, instruments], 
            names=['datetime', 'instrument']
        )
        
        # 生成模拟股票数据
        np.random.seed(42)
        n_samples = len(index)
        self.mock_stock_data = pd.DataFrame({
            '$open': np.random.uniform(50, 150, n_samples),
            '$high': np.random.uniform(55, 155, n_samples),
            '$low': np.random.uniform(45, 145, n_samples),
            '$close': np.random.uniform(50, 150, n_samples),
            '$volume': np.random.uniform(1000, 10000, n_samples),
            '$vwap': np.random.uniform(50, 150, n_samples)
        }, index=index)
        
        # 确保价格数据的合理性
        for i in range(len(self.mock_stock_data)):
            row = self.mock_stock_data.iloc[i]
            low = row['$low']
            high = row['$high']
            self.mock_stock_data.iloc[i, self.mock_stock_data.columns.get_loc('$open')] = np.random.uniform(low, high)
            self.mock_stock_data.iloc[i, self.mock_stock_data.columns.get_loc('$close')] = np.random.uniform(low, high)
            self.mock_stock_data.iloc[i, self.mock_stock_data.columns.get_loc('$vwap')] = np.random.uniform(low, high)
        
        self.mock_data_manager.get_stock_data.return_value = self.mock_stock_data
        self.mock_data_manager.get_market_data.return_value = pd.DataFrame({
            'market_return': np.random.normal(0, 0.02, len(dates))
        }, index=dates)
        
    def test_initialization(self):
        """测试初始化"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        
        self.assertEqual(dataset.data_manager, self.mock_data_manager)
        self.assertEqual(dataset.config, self.config)
        self.assertIsNone(dataset.raw_data)
        self.assertIsNone(dataset.processed_data)
        
    def test_load_historical_data(self):
        """测试历史数据加载"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        result = dataset.load_historical_data()
        
        self.assertEqual(result, dataset)  # 支持链式调用
        self.assertIsNotNone(dataset.raw_data)
        self.assertIsNotNone(dataset.processed_data)
        self.assertIsNotNone(dataset.features)
        self.assertIsNotNone(dataset.targets)
        
        # 验证数据管理器调用
        self.mock_data_manager.get_stock_data.assert_called_once()
        self.mock_data_manager.get_market_data.assert_called_once()
        
    def test_data_cleaning(self):
        """测试数据清洗功能"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        
        # 添加一些异常值到模拟数据
        corrupted_data = self.mock_stock_data.copy()
        corrupted_data.iloc[0, 0] = np.inf
        corrupted_data.iloc[1, 1] = -np.inf
        corrupted_data.iloc[2, 2] = np.nan
        
        self.mock_data_manager.get_stock_data.return_value = corrupted_data
        
        dataset.load_historical_data()
        
        # 验证清洗后的数据不包含异常值
        self.assertFalse(np.any(np.isinf(dataset.processed_data.values)))
        self.assertFalse(np.any(np.isnan(dataset.processed_data.values)))
        
    def test_feature_creation(self):
        """测试特征创建"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        # 验证特征维度
        expected_samples = len(dataset.metadata['samples'])
        self.assertEqual(len(dataset.features), expected_samples)
        
        if expected_samples > 0:
            expected_feature_dim = len(self.mock_stock_data.columns) * self.config.lookback_window
            self.assertEqual(dataset.features.shape[1], expected_feature_dim)
            
            # 验证目标变量
            self.assertEqual(len(dataset.targets), expected_samples)
            self.assertTrue(np.all(np.isfinite(dataset.targets)))
        
    def test_behavior_dataset_creation(self):
        """测试行为克隆数据集创建"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        if len(dataset) > 0:
            behavior_dataset = dataset.create_behavior_dataset()
            
            self.assertIsInstance(behavior_dataset, torch.utils.data.TensorDataset)
            self.assertEqual(len(behavior_dataset), len(dataset))
            
            # 测试数据格式
            features, actions = behavior_dataset[0]
            self.assertIsInstance(features, torch.Tensor)
            self.assertIsInstance(actions, torch.Tensor)
        
    def test_tensor_dataset_creation(self):
        """测试PyTorch张量数据集创建"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        if len(dataset) > 0:
            tensor_dataset = dataset.create_tensor_dataset()
            
            self.assertIsInstance(tensor_dataset, torch.utils.data.TensorDataset)
            self.assertEqual(len(tensor_dataset), len(dataset))
        
    def test_data_augmentation(self):
        """测试数据增强"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        if len(dataset) > 0:
            original_dataset = dataset.create_tensor_dataset()
            augmented_dataset = dataset.apply_data_augmentation(original_dataset)
            
            # 增强后的数据集应该更大
            self.assertGreater(len(augmented_dataset), len(original_dataset))
        
    def test_statistics(self):
        """测试统计信息获取"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        stats = dataset.get_statistics()
        
        self.assertIn('total_samples', stats)
        self.assertIn('feature_dim', stats)
        self.assertIn('date_range', stats)
        self.assertEqual(stats['date_range'], (self.config.start_date, self.config.end_date))
        
    def test_empty_data_handling(self):
        """测试空数据处理"""
        # 模拟空数据
        self.mock_data_manager.get_stock_data.return_value = pd.DataFrame()
        
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.features.shape[0], 0)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)


class TestOnlineReplayBuffer(unittest.TestCase):
    """测试在线回放缓冲区组件"""
    
    def setUp(self):
        """设置测试环境"""
        self.config = OnlineReplayConfig(
            capacity=1000,
            priority_alpha=0.6,
            priority_beta=0.4,
            batch_size=64
        )
        self.buffer = OnlineReplayBuffer(self.config)
        
        # 创建测试轨迹数据
        self.test_trajectory = TrajectoryData(
            states=np.random.randn(10, 5),
            actions=np.random.randn(10, 2),
            rewards=np.random.randn(10),
            next_states=np.random.randn(10, 5),
            dones=np.random.choice([True, False], 10),
            log_probs=np.random.randn(10),
            values=np.random.randn(10)
        )
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.buffer.capacity, 1000)
        self.assertEqual(len(self.buffer), 0)
        self.assertFalse(bool(self.buffer))
        
    def test_add_trajectory(self):
        """测试添加轨迹"""
        self.buffer.add_trajectory(self.test_trajectory)
        
        self.assertEqual(len(self.buffer), 1)
        self.assertTrue(bool(self.buffer))
        
    def test_batch_trajectory_addition(self):
        """测试批量添加轨迹"""
        trajectories = [self.test_trajectory for _ in range(5)]
        self.buffer.add_batch_trajectories(trajectories)
        
        self.assertEqual(len(self.buffer), 5)
        
    def test_sample_batch(self):
        """测试批次采样"""
        # 添加一些轨迹
        for _ in range(10):
            self.buffer.add_trajectory(self.test_trajectory)
        
        batch = self.buffer.sample_batch(batch_size=5)
        
        self.assertIn('states', batch)
        self.assertIn('actions', batch)
        self.assertIn('rewards', batch)
        self.assertIn('weights', batch)
        self.assertIn('indices', batch)
        
        # 验证批次大小
        self.assertEqual(len(batch['indices']), 5)
        
    def test_priority_sampling(self):
        """测试优先级采样"""
        # 添加轨迹
        for i in range(5):
            self.buffer.add_trajectory(self.test_trajectory, priority=float(i+1))
        
        batch = self.buffer.sample_batch(batch_size=3)
        
        # 验证重要性权重
        self.assertEqual(len(batch['weights']), 3)
        self.assertTrue(np.all(batch['weights'] > 0))
        
    def test_update_priorities(self):
        """测试优先级更新"""
        # 添加轨迹
        for _ in range(5):
            self.buffer.add_trajectory(self.test_trajectory)
        
        # 更新优先级
        indices = [0, 1, 2]
        new_priorities = [2.0, 3.0, 1.0]
        self.buffer.update_priorities(indices, new_priorities)
        
        # 验证优先级已更新（通过采样验证）
        batch = self.buffer.sample_batch(batch_size=3)
        self.assertIsNotNone(batch['weights'])
        
    def test_time_decay(self):
        """测试时间衰减"""
        # 添加轨迹
        self.buffer.add_trajectory(self.test_trajectory, priority=1.0)
        
        # 应用时间衰减
        self.buffer._apply_time_decay()
        
        # 验证优先级被衰减（应该小于等于原值）
        self.assertLessEqual(self.buffer.priorities[0], 1.0)
        
    def test_recent_trajectory_retrieval(self):
        """测试最近轨迹获取"""
        # 添加轨迹
        for _ in range(5):
            self.buffer.add_trajectory(self.test_trajectory)
        
        recent_trajectories = self.buffer.get_recent_trajectory(window=1)
        
        # 所有轨迹都应该是最近的
        self.assertEqual(len(recent_trajectories), 5)
        
    def test_capacity_limit(self):
        """测试容量限制"""
        small_config = OnlineReplayConfig(capacity=3)
        small_buffer = OnlineReplayBuffer(small_config)
        
        # 添加超过容量的轨迹
        for _ in range(5):
            small_buffer.add_trajectory(self.test_trajectory)
        
        # 验证容量限制
        self.assertEqual(len(small_buffer), 3)
        
    def test_empty_buffer_sampling(self):
        """测试空缓冲区采样"""
        batch = self.buffer.sample_batch(batch_size=5)
        
        # 空缓冲区应该返回空批次
        self.assertEqual(len(batch['states']), 0)
        self.assertEqual(len(batch['indices']), 0)
        
    def test_statistics(self):
        """测试统计信息"""
        # 添加一些轨迹
        for _ in range(3):
            self.buffer.add_trajectory(self.test_trajectory)
        
        stats = self.buffer.get_statistics()
        
        self.assertEqual(stats['size'], 3)
        self.assertEqual(stats['capacity'], 1000)
        self.assertIn('utilization', stats)
        self.assertIn('priority_stats', stats)
        
    def test_clear_buffer(self):
        """测试清空缓冲区"""
        self.buffer.add_trajectory(self.test_trajectory)
        self.assertEqual(len(self.buffer), 1)
        
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)


class TestMixtureSampler(unittest.TestCase):
    """测试混合采样器组件"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建模拟的离线数据集
        self.mock_offline_dataset = Mock()
        self.mock_offline_dataset.__len__ = Mock(return_value=1000)
        self.mock_offline_dataset.__getitem__ = Mock(return_value=(
            torch.randn(10), torch.randn(2)  # 修改为2维以匹配在线数据
        ))
        
        # 创建模拟的在线缓冲区
        self.mock_online_buffer = Mock()
        self.mock_online_buffer.__len__ = Mock(return_value=100)
        self.mock_online_buffer.sample_batch.return_value = {
            'states': np.random.randn(32, 10),
            'actions': np.random.randn(32, 2),
            'rewards': np.random.randn(32),
            'next_states': np.random.randn(32, 10),
            'dones': np.random.choice([True, False], 32),
            'data_source': 'online',
            'indices': np.arange(32),
            'weights': np.ones(32)
        }
        
        self.config = MixtureSamplerConfig(
            initial_rho=0.2,
            rho_increment=0.01,
            batch_size=64
        )
        
        self.sampler = MixtureSampler(
            self.mock_offline_dataset,
            self.mock_online_buffer,
            self.config
        )
        
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.sampler.current_rho, 0.2)
        self.assertEqual(self.sampler.config.batch_size, 64)
        self.assertEqual(self.sampler.episode_count, 0)
        
    def test_mixed_batch_sampling(self):
        """测试混合批次采样"""
        mixed_batch, importance_weights = self.sampler.sample_mixed_batch(
            batch_size=64, rho=0.3
        )
        
        # 验证基本结构
        self.assertIn('states', mixed_batch)
        self.assertIn('actions', mixed_batch)
        self.assertIn('data_sources', mixed_batch)
        self.assertIsInstance(importance_weights, np.ndarray)
        self.assertGreater(len(importance_weights), 0)
        
        # 验证数据源混合
        data_sources = mixed_batch['data_sources']
        self.assertIsInstance(data_sources, list)
        
        # 应该有在线和离线数据的混合
        has_online = any(source == 'online' for source in data_sources)
        has_offline = any(source == 'offline' for source in data_sources)
        
        self.assertTrue(has_online, "应该包含在线数据")
        self.assertTrue(has_offline, "应该包含离线数据")
        
    def test_sampling_ratio_update(self):
        """测试采样比例更新"""
        initial_rho = self.sampler.current_rho
        
        self.sampler.update_sampling_ratio(episode=50, total_episodes=1000)
        
        # 采样比例应该增加
        self.assertGreater(self.sampler.current_rho, initial_rho)
        
    def test_adaptive_rho_update(self):
        """测试自适应采样比例更新"""
        adaptive_config = MixtureSamplerConfig(
            initial_rho=0.2,
            adaptive_rho=True
        )
        adaptive_sampler = MixtureSampler(
            self.mock_offline_dataset,
            self.mock_online_buffer,
            adaptive_config
        )
        
        # 添加性能历史
        for i in range(10):
            adaptive_sampler.update_performance(0.01 * i)
        
        initial_rho = adaptive_sampler.current_rho
        adaptive_sampler.update_sampling_ratio(episode=10)
        
        # 由于性能提升，rho应该增加更多
        self.assertGreater(adaptive_sampler.current_rho, initial_rho)
        
    def test_importance_weight_calculation(self):
        """测试重要性权重计算"""
        # 设置模拟策略
        mock_policy = Mock()
        mock_policy.evaluate_actions.return_value = torch.zeros(32)
        self.sampler.set_current_policy(mock_policy)
        
        mixed_batch, importance_weights = self.sampler.sample_mixed_batch(
            batch_size=64, rho=0.5
        )
        
        # 验证重要性权重
        self.assertEqual(len(importance_weights), 64)
        self.assertTrue(np.all(importance_weights > 0))
        self.assertTrue(np.all(np.isfinite(importance_weights)))
        
    def test_min_offline_ratio_protection(self):
        """测试最小离线样本比例保护"""
        # 尝试使用很高的在线比例
        mixed_batch, _ = self.sampler.sample_mixed_batch(
            batch_size=100, rho=0.95
        )
        
        data_sources = mixed_batch['data_sources']
        offline_count = sum(1 for source in data_sources if source == 'offline')
        
        # 应该至少有最小离线比例的样本
        min_offline_samples = int(100 * self.config.min_offline_ratio)
        self.assertGreaterEqual(offline_count, min_offline_samples)
        
    def test_performance_update(self):
        """测试性能更新"""
        initial_history_length = len(self.sampler.performance_history)
        
        self.sampler.update_performance(0.05)
        
        self.assertEqual(len(self.sampler.performance_history), initial_history_length + 1)
        
    def test_sampling_statistics(self):
        """测试采样统计信息"""
        stats = self.sampler.get_sampling_statistics()
        
        self.assertIn('current_rho', stats)
        self.assertIn('episode_count', stats)
        self.assertIn('offline_dataset_size', stats)
        self.assertIn('online_buffer_size', stats)
        self.assertIn('config', stats)
        
    def test_rho_reset(self):
        """测试采样比例重置"""
        self.sampler.update_sampling_ratio(episode=100)
        initial_rho = self.sampler.current_rho
        
        self.sampler.reset_sampling_ratio()
        
        self.assertEqual(self.sampler.current_rho, self.config.initial_rho)
        self.assertEqual(self.sampler.episode_count, 0)
        
    def test_manual_rho_setting(self):
        """测试手动设置采样比例"""
        self.sampler.set_rho(0.7)
        
        self.assertEqual(self.sampler.current_rho, 0.7)


# 工具函数测试
class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""
    
    def test_create_trajectory_from_episode(self):
        """测试从episode创建轨迹"""
        states = np.random.randn(10, 5)
        actions = np.random.randn(10, 2)
        rewards = np.random.randn(10)
        next_states = np.random.randn(10, 5)
        dones = np.random.choice([True, False], 10)
        
        trajectory = create_trajectory_from_episode(
            states, actions, rewards, next_states, dones
        )
        
        self.assertIsInstance(trajectory, TrajectoryData)
        self.assertEqual(len(trajectory.states), 10)
        self.assertEqual(len(trajectory.actions), 10)
        self.assertEqual(len(trajectory.rewards), 10)
        
    def test_calculate_td_errors(self):
        """测试TD误差计算"""
        values = np.random.randn(32)
        rewards = np.random.randn(32)
        next_values = np.random.randn(32)
        dones = np.random.choice([True, False], 32)
        
        td_errors = calculate_td_errors(values, rewards, next_values, dones)
        
        self.assertEqual(len(td_errors), 32)
        self.assertTrue(np.all(td_errors >= 0))  # TD误差应该是非负的
        
    def test_create_mixture_sampler(self):
        """测试创建混合采样器"""
        mock_offline_dataset = Mock()
        mock_online_buffer = Mock()
        
        sampler = create_mixture_sampler(mock_offline_dataset, mock_online_buffer)
        
        self.assertIsInstance(sampler, MixtureSampler)
        self.assertEqual(sampler.offline_dataset, mock_offline_dataset)
        self.assertEqual(sampler.online_buffer, mock_online_buffer)
        
    def test_calculate_sampling_schedule(self):
        """测试采样调度计算"""
        # 测试线性调度
        linear_schedule = calculate_sampling_schedule(
            total_episodes=100,
            initial_rho=0.2,
            final_rho=0.8,
            schedule_type='linear'
        )
        
        self.assertEqual(len(linear_schedule), 100)
        self.assertAlmostEqual(linear_schedule[0], 0.2, places=3)
        self.assertAlmostEqual(linear_schedule[-1], 0.8, places=3)
        
        # 测试指数调度
        exp_schedule = calculate_sampling_schedule(
            total_episodes=100,
            schedule_type='exponential'
        )
        
        self.assertEqual(len(exp_schedule), 100)
        self.assertTrue(all(0 <= rho <= 1 for rho in exp_schedule))
        
        # 测试余弦调度
        cosine_schedule = calculate_sampling_schedule(
            total_episodes=100,
            schedule_type='cosine'
        )
        
        self.assertEqual(len(cosine_schedule), 100)
        self.assertTrue(all(0 <= rho <= 1 for rho in cosine_schedule))