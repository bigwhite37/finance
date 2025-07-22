"""
交易环境测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent import TradingEnvironment


class TestTradingEnvironment(unittest.TestCase):
    """交易环境测试类"""
    
    def setUp(self):
        """测试初始化"""
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_stocks = 5
        n_factors = 3
        
        np.random.seed(42)
        
        # 价格数据
        self.price_data = pd.DataFrame(
            index=dates,
            columns=[f'stock_{i}' for i in range(n_stocks)],
            data=100 + np.cumsum(np.random.normal(0, 0.02, (len(dates), n_stocks)), axis=0)
        )
        
        # 因子数据
        self.factor_data = pd.DataFrame(
            index=dates,
            columns=[f'factor_{i}' for i in range(n_factors)],
            data=np.random.normal(0, 1, (len(dates), n_factors))
        )
        
        # 环境配置
        self.config = {
            'lookback_window': 20,
            'transaction_cost': 0.001,
            'max_position': 0.1,
            'max_leverage': 1.2,
            'lambda1': 2.0,
            'lambda2': 1.0,
            'max_dd_threshold': 0.05
        }
        
        # 创建环境
        self.env = TradingEnvironment(self.factor_data, self.price_data, self.config)
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        # 验证属性设置
        self.assertEqual(self.env.n_stocks, 5)
        self.assertEqual(self.env.n_factors, 3)
        self.assertEqual(self.env.lookback_window, 20)
        self.assertEqual(self.env.transaction_cost, 0.001)
        
        # 验证空间设置
        self.assertEqual(self.env.action_space.shape[0], 5)
        self.assertTrue(self.env.observation_space.shape[0] > 0)
    
    def test_reset_environment(self):
        """测试环境重置"""
        observation, info = self.env.reset()
        
        # 验证重置后状态
        self.assertEqual(self.env.current_step, self.env.lookback_window)
        self.assertTrue(np.array_equal(self.env.portfolio_weights, np.zeros(self.env.n_stocks)))
        self.assertEqual(self.env.cash, 1.0)
        self.assertEqual(self.env.portfolio_value, 1.0)
        
        # 验证返回值
        self.assertEqual(len(observation), self.env.observation_space.shape[0])
        self.assertIsInstance(info, dict)
    
    def test_constrain_action(self):
        """测试动作约束"""
        # 测试单股票仓位限制
        action = np.array([0.15, -0.12, 0.08, 0.05, -0.03])
        constrained = self.env._constrain_action(action)
        
        # 验证单股票仓位限制
        self.assertTrue(np.all(np.abs(constrained) <= self.env.max_position))
        
        # 测试总杠杆限制
        high_leverage_action = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        constrained_leverage = self.env._constrain_action(high_leverage_action)
        
        # 验证总杠杆限制
        total_leverage = np.sum(np.abs(constrained_leverage))
        self.assertLessEqual(total_leverage, self.env.max_leverage + 1e-10)
    
    def test_step_function(self):
        """测试步进函数"""
        # 重置环境
        observation, info = self.env.reset()
        
        # 执行动作
        action = np.array([0.05, -0.03, 0.02, 0.01, -0.01])
        next_obs, reward, terminated, truncated, next_info = self.env.step(action)
        
        # 验证返回值类型
        self.assertEqual(len(next_obs), len(observation))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(next_info, dict)
        
        # 验证状态更新
        self.assertEqual(self.env.current_step, self.env.lookback_window + 1)
        self.assertTrue(np.array_equal(self.env.portfolio_weights, action))
    
    def test_portfolio_return_calculation(self):
        """测试组合收益率计算"""
        # 设置初始状态
        self.env.reset()
        self.env.current_step = 21  # 确保有前一天数据
        self.env.portfolio_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # 计算收益率
        portfolio_return = self.env._calculate_portfolio_return()
        
        # 验证收益率为数值
        self.assertIsInstance(portfolio_return, (int, float))
        self.assertFalse(np.isnan(portfolio_return))
    
    def test_drawdown_calculation(self):
        """测试回撤计算"""
        # 重置环境
        self.env.reset()
        
        # 模拟净值变化
        self.env.portfolio_value = 1.1
        self.env._update_drawdown_stats()
        self.assertEqual(self.env.peak_value, 1.1)
        self.assertEqual(self.env.max_drawdown, 0.0)
        
        # 模拟下跌
        self.env.portfolio_value = 1.05
        self.env._update_drawdown_stats()
        expected_drawdown = (1.1 - 1.05) / 1.1
        self.assertAlmostEqual(self.env.max_drawdown, expected_drawdown, places=6)
    
    def test_reward_calculation(self):
        """测试奖励函数计算"""
        # 重置环境
        self.env.reset()
        
        # 设置测试状态
        self.env.portfolio_returns = [0.01, -0.005, 0.002, -0.01, 0.008]
        self.env.peak_value = 1.0
        self.env.portfolio_value = 0.98
        
        # 计算奖励
        reward = self.env._calculate_reward(0.005, 0.001)
        
        # 验证奖励为数值
        self.assertIsInstance(reward, float)
        self.assertFalse(np.isnan(reward))
    
    def test_termination_conditions(self):
        """测试终止条件"""
        # 重置环境
        self.env.reset()
        
        # 测试破产条件
        self.env.portfolio_value = 0.7
        self.assertTrue(self.env._check_termination())
        
        # 测试极端回撤条件
        self.env.portfolio_value = 1.0
        self.env.max_drawdown = 0.2
        self.assertTrue(self.env._check_termination())
        
        # 测试正常条件
        self.env.portfolio_value = 1.0
        self.env.max_drawdown = 0.02
        self.assertFalse(self.env._check_termination())
    
    def test_get_info(self):
        """测试信息获取"""
        # 重置环境并运行几步
        self.env.reset()
        action = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        self.env.step(action)
        
        # 获取信息
        info = self.env._get_info()
        
        # 验证信息内容
        required_keys = [
            'current_step', 'portfolio_value', 'max_drawdown',
            'portfolio_weights', 'cash', 'total_return',
            'annual_return', 'volatility', 'sharpe_ratio'
        ]
        
        for key in required_keys:
            self.assertIn(key, info)
        
        # 验证数据类型
        self.assertIsInstance(info['current_step'], int)
        self.assertIsInstance(info['portfolio_value'], float)
        self.assertIsInstance(info['portfolio_weights'], np.ndarray)


if __name__ == '__main__':
    unittest.main()