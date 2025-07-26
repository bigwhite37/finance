#!/usr/bin/env python3
"""
测试O2O增强的TradingEnvironment功能
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.trading_environment import TradingEnvironment


class TestO2OTradingEnvironment(unittest.TestCase):
    """测试O2O增强的TradingEnvironment"""

    def setUp(self):
        """设置测试环境"""
        # 创建模拟数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        n_stocks = 10
        n_factors = 20
        
        # 生成随机价格数据
        np.random.seed(42)
        self.price_data = pd.DataFrame(
            np.random.lognormal(0, 0.02, (100, n_stocks)),
            index=dates,
            columns=[f'stock_{i}' for i in range(n_stocks)]
        )
        
        # 生成因子数据
        self.factor_data = pd.DataFrame(
            np.random.normal(0, 1, (100, n_factors)),
            index=dates,
            columns=[f'factor_{i}' for i in range(n_factors)]
        )
        
        # 环境配置
        self.config = {
            'lookback_window': 20,
            'transaction_cost': 0.001,
            'max_position': 0.1,
            'max_leverage': 1.2,
            'trajectory_buffer_size': 1000
        }
        
        # 创建环境
        self.env = TradingEnvironment(self.factor_data, self.price_data, self.config)

    def test_mode_switching(self):
        """测试模式切换功能"""
        # 测试初始模式
        self.assertEqual(self.env.get_mode(), 'offline')
        
        # 测试切换到在线模式
        self.env.set_mode('online')
        self.assertEqual(self.env.get_mode(), 'online')
        
        # 测试切换到离线模式
        self.env.set_mode('offline')
        self.assertEqual(self.env.get_mode(), 'offline')
        
        # 测试无效模式
        with self.assertRaises(ValueError):
            self.env.set_mode('invalid')

    def test_trajectory_collection_online_mode(self):
        """测试在线模式下的轨迹收集"""
        self.env.set_mode('online')
        obs, info = self.env.reset()
        
        # 执行几步
        initial_buffer_size = len(self.env.trajectory_buffer)
        for i in range(5):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # 检查轨迹是否被收集
        self.assertGreater(len(self.env.trajectory_buffer), initial_buffer_size)
        
        # 检查轨迹记录的结构
        if self.env.trajectory_buffer:
            trajectory = self.env.trajectory_buffer[0]
            required_keys = ['timestamp', 'step', 'state', 'action', 'reward', 
                           'next_state', 'done', 'portfolio_value', 'market_regime']
            for key in required_keys:
                self.assertIn(key, trajectory)

    def test_trajectory_collection_offline_mode(self):
        """测试离线模式下不收集轨迹"""
        self.env.set_mode('offline')
        obs, info = self.env.reset()
        
        # 执行几步
        for i in range(5):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # 离线模式下不应该收集轨迹
        self.assertEqual(len(self.env.trajectory_buffer), 0)

    def test_get_recent_trajectory(self):
        """测试获取最近轨迹功能"""
        self.env.set_mode('online')
        obs, info = self.env.reset()
        
        # 执行一些步骤
        for i in range(10):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # 测试获取最近轨迹
        recent_trajectories = self.env.get_recent_trajectory(window=5)
        self.assertLessEqual(len(recent_trajectories), 5)
        self.assertLessEqual(len(recent_trajectories), len(self.env.trajectory_buffer))

    def test_trajectory_statistics(self):
        """测试轨迹统计功能"""
        # 空缓冲区的统计
        stats = self.env.get_trajectory_statistics()
        self.assertEqual(stats['total_trajectories'], 0)
        
        # 有数据的统计
        self.env.set_mode('online')
        obs, info = self.env.reset()
        
        for i in range(5):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        stats = self.env.get_trajectory_statistics()
        self.assertGreater(stats['total_trajectories'], 0)
        self.assertIn('avg_reward', stats)
        self.assertIn('regime_distribution', stats)

    def test_extended_observation_space(self):
        """测试扩展的观测空间"""
        obs, info = self.env.reset()
        
        # 检查观测空间维度是否正确扩展
        expected_dim = (
            self.factor_data.shape[1] +  # 因子数据
            5 +  # 宏观状态
            3 +  # 组合状态
            3 +  # 市场制度信号 (one-hot)
            4 +  # 时间信息
            2    # 筛选器状态
        )
        
        self.assertEqual(obs.shape[0], expected_dim)
        self.assertEqual(self.env.observation_space.shape[0], expected_dim)

    def test_market_regime_info(self):
        """测试市场制度信息获取"""
        regime_info = self.env.get_market_regime_info()
        
        required_keys = ['current_regime', 'regime_confidence', 'filter_active', 
                        'tradable_stocks_count', 'filter_strength']
        for key in required_keys:
            self.assertIn(key, regime_info)
        
        # 检查数据类型
        self.assertIsInstance(regime_info['current_regime'], str)
        self.assertIsInstance(regime_info['filter_active'], bool)
        self.assertIsInstance(regime_info['tradable_stocks_count'], int)

    def test_extended_state_info(self):
        """测试扩展状态信息"""
        extended_info = self.env.get_extended_state_info()
        
        required_keys = ['current_date', 'trading_day_progress', 'market_regime', 
                        'portfolio_state', 'time_features', 'risk_metrics']
        for key in required_keys:
            self.assertIn(key, extended_info)
        
        # 检查时间特征
        time_features = extended_info['time_features']
        self.assertIn('day_of_year', time_features)
        self.assertIn('month', time_features)
        self.assertIn('quarter', time_features)

    def test_info_contains_mode(self):
        """测试info包含模式信息"""
        obs, info = self.env.reset()
        self.assertIn('mode', info)
        self.assertEqual(info['mode'], 'offline')
        
        self.env.set_mode('online')
        obs, info = self.env.reset()
        self.assertEqual(info['mode'], 'online')

    def test_online_mode_extended_info(self):
        """测试在线模式下的扩展信息"""
        self.env.set_mode('online')
        obs, info = self.env.reset()
        
        # 在线模式下应该包含扩展状态信息
        self.assertIn('extended_state', info)
        
        self.env.set_mode('offline')
        obs, info = self.env.reset()
        
        # 离线模式下不包含扩展状态信息
        self.assertNotIn('extended_state', info)

    def test_trajectory_buffer_management(self):
        """测试轨迹缓冲区管理"""
        self.env.set_mode('online')
        
        # 测试缓冲区大小限制
        original_buffer_size = self.config['trajectory_buffer_size']
        self.config['trajectory_buffer_size'] = 5  # 设置小的缓冲区大小
        
        obs, info = self.env.reset()
        
        # 执行超过缓冲区大小的步数
        for i in range(10):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # 检查缓冲区大小是否被限制
        self.assertLessEqual(len(self.env.trajectory_buffer), 5)
        
        # 恢复原始配置
        self.config['trajectory_buffer_size'] = original_buffer_size

    def test_clear_trajectory_buffer(self):
        """测试清空轨迹缓冲区"""
        self.env.set_mode('online')
        obs, info = self.env.reset()
        
        # 收集一些轨迹
        for i in range(3):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # 确保有轨迹数据
        self.assertGreater(len(self.env.trajectory_buffer), 0)
        
        # 清空缓冲区
        self.env.clear_trajectory_buffer()
        self.assertEqual(len(self.env.trajectory_buffer), 0)

    def test_backward_compatibility(self):
        """测试向后兼容性"""
        # 确保原有的环境功能仍然正常工作
        obs, info = self.env.reset()
        
        # 基本的step功能
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 检查基本的info字段
        basic_info_keys = ['current_step', 'portfolio_value', 'max_drawdown', 
                          'portfolio_weights', 'total_return']
        for key in basic_info_keys:
            self.assertIn(key, info)


if __name__ == '__main__':
    unittest.main()