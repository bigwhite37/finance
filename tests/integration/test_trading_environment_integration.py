"""
交易环境与动态低波筛选器集成测试

测试TradingEnvironment与DynamicLowVolFilter的集成功能：
1. 筛选器初始化
2. 可交易掩码更新
3. 动作约束应用
4. 市场状态信号观测
5. 完整交易流程
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.trading_environment import TradingEnvironment


class TestTradingEnvironmentIntegration(unittest.TestCase):
    """交易环境集成测试类"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建模拟的价格数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']
        
        # 价格数据 - 模拟股票价格走势
        np.random.seed(42)
        price_data = pd.DataFrame(
            np.random.lognormal(0, 0.02, (100, 5)).cumprod(axis=0) * 100,
            index=dates,
            columns=stocks
        )
        
        # 因子数据 - 模拟多因子数据
        factor_names = ['momentum', 'value', 'quality', 'size', 'volatility']
        factor_data = pd.DataFrame(
            np.random.randn(100, 5),
            index=dates,
            columns=factor_names
        )
        
        self.price_data = price_data
        self.factor_data = factor_data
        self.stocks = stocks
        self.dates = dates
        
        # 基础配置
        self.base_config = {
            'lookback_window': 20,
            'transaction_cost': 0.001,
            'max_position': 0.2,
            'max_leverage': 1.0,
            'lambda1': 2.0,
            'lambda2': 1.0,
            'max_dd_threshold': 0.05
        }
    
    def test_environment_initialization_without_filter(self):
        """测试不使用筛选器的环境初始化"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        # 验证基本属性
        self.assertEqual(env.n_stocks, 5)
        self.assertEqual(env.n_factors, 5)
        self.assertIsNone(env.lowvol_filter)
        self.assertIsNone(env._current_tradable_mask)
        
        # 验证观测空间维度（因子5 + 宏观5 + 组合3 + 市场制度3 + 时间4 + 筛选器2 = 22）
        self.assertEqual(env.observation_space.shape[0], 22)
        
        # 验证动作空间
        self.assertEqual(env.action_space.shape[0], 5)
    
    @patch('rl_agent.trading_environment.DynamicLowVolFilter')
    def test_environment_initialization_with_filter(self, mock_filter_class):
        """测试使用筛选器的环境初始化"""
        # 模拟筛选器实例
        mock_filter = Mock()
        mock_filter_class.return_value = mock_filter
        
        # 添加筛选器配置
        config_with_filter = self.base_config.copy()
        config_with_filter['dynamic_lowvol'] = {
            'rolling_windows': [20, 60],
            'percentile_thresholds': {'低': 0.4, '中': 0.3, '高': 0.2}
        }
        
        with patch('rl_agent.trading_environment.DynamicLowVolFilter', mock_filter_class):
            with patch('rl_agent.trading_environment.DataManager', Mock()):
                env = TradingEnvironment(
                    factor_data=self.factor_data,
                    price_data=self.price_data,
                    config=config_with_filter
                )
        
        # 验证筛选器被正确初始化
        self.assertIsNotNone(env.lowvol_filter)
        mock_filter_class.assert_called_once()
    
    def test_simplified_data_manager(self):
        """测试简化数据管理器功能"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        data_manager = env._create_data_manager()
        
        # 测试价格数据获取
        price_data = data_manager.get_price_data()
        self.assertEqual(price_data.shape, self.price_data.shape)
        
        # 测试指定日期的数据获取
        end_date = self.dates[50]
        limited_price_data = data_manager.get_price_data(end_date=end_date, lookback_days=20)
        self.assertLessEqual(len(limited_price_data), 21)  # 最多21天数据
        
        # 测试成交量数据获取
        volume_data = data_manager.get_volume_data()
        self.assertEqual(volume_data.shape, self.price_data.shape)
        
        # 测试因子数据获取
        factor_data = data_manager.get_factor_data()
        self.assertEqual(factor_data.shape, self.factor_data.shape)
        
        # 测试市场数据获取
        market_data = data_manager.get_market_data()
        self.assertEqual(len(market_data), len(self.price_data))
        self.assertIn('market_index', market_data.columns)
        self.assertIn('returns', market_data.columns)
    
    def test_observation_with_regime_signal(self):
        """测试包含市场状态信号的观测"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        # 重置环境
        obs, info = env.reset()
        
        # 验证观测维度
        expected_dim = 5 + 5 + 3 + 3 + 4 + 2  # 因子 + 宏观 + 组合 + 市场制度 + 时间 + 筛选器
        self.assertEqual(len(obs), expected_dim)
        
        # 验证市场制度信号默认为中等波动制度 [0, 1, 0]
        regime_signals = obs[13:16]  # 市场制度信号在位置13-15
        expected_regime = np.array([0.0, 1.0, 0.0])  # 默认中等波动制度
        np.testing.assert_array_equal(regime_signals, expected_regime)
        
        # 验证观测数值稳定性
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertTrue(np.all(np.abs(obs) <= 1000))
    
    @patch('rl_agent.trading_environment.DynamicLowVolFilter')
    def test_observation_with_mock_regime_signal(self, mock_filter_class):
        """测试使用模拟筛选器的市场状态信号"""
        # 创建模拟筛选器
        mock_filter = Mock()
        mock_filter.get_current_regime.return_value = "高"
        mock_filter.get_filter_strength.return_value = 0.7  # 确保返回数值
        mock_filter_class.return_value = mock_filter
        
        config_with_filter = self.base_config.copy()
        config_with_filter['dynamic_lowvol'] = {}
        
        with patch('rl_agent.trading_environment.DataManager', Mock()):
            env = TradingEnvironment(
                factor_data=self.factor_data,
                price_data=self.price_data,
                config=config_with_filter
            )
            env.lowvol_filter = mock_filter
        
        # 获取观测
        obs, info = env.reset()
        
        # 验证状态信号为高波动状态(1.0)
        regime_signal = obs[-1]
        self.assertEqual(regime_signal, 1.0)
    
    def test_action_constraint_without_filter(self):
        """测试不使用筛选器的动作约束"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        # 测试正常动作
        action = np.array([0.1, -0.05, 0.15, 0.0, -0.1])
        constrained_action = env._constrain_action(action)
        
        # 验证单股票仓位限制
        self.assertTrue(np.all(np.abs(constrained_action) <= env.max_position))
        
        # 验证总杠杆限制
        total_leverage = np.sum(np.abs(constrained_action))
        self.assertLessEqual(total_leverage, env.max_leverage)
    
    def test_action_constraint_with_tradable_mask(self):
        """测试使用可交易掩码的动作约束"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        # 设置可交易掩码（只有前3只股票可交易）
        env._current_tradable_mask = np.array([True, True, True, False, False])
        
        # 测试动作约束
        action = np.array([0.1, -0.05, 0.15, 0.1, -0.1])
        constrained_action = env._constrain_action(action)
        
        # 验证不可交易股票的权重被设为0
        self.assertEqual(constrained_action[3], 0.0)
        self.assertEqual(constrained_action[4], 0.0)
        
        # 验证可交易股票的权重保持不变（在约束范围内）
        self.assertAlmostEqual(constrained_action[0], 0.1)
        self.assertAlmostEqual(constrained_action[1], -0.05)
        self.assertAlmostEqual(constrained_action[2], 0.15)
    
    @patch('rl_agent.trading_environment.DynamicLowVolFilter')
    def test_step_with_filter_update(self, mock_filter_class):
        """测试包含筛选器更新的交易步骤"""
        # 创建模拟筛选器
        mock_filter = Mock()
        mock_filter.update_tradable_mask.return_value = np.array([True, True, False, True, False])
        mock_filter.get_current_regime.return_value = "低"
        mock_filter.get_filter_strength.return_value = 0.6  # 确保返回数值
        mock_filter_class.return_value = mock_filter
        
        config_with_filter = self.base_config.copy()
        config_with_filter['dynamic_lowvol'] = {}
        
        with patch('rl_agent.trading_environment.DataManager', Mock()):
            env = TradingEnvironment(
                factor_data=self.factor_data,
                price_data=self.price_data,
                config=config_with_filter
            )
            env.lowvol_filter = mock_filter
        
        # 重置环境
        obs, info = env.reset()
        
        # 执行一步
        action = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 验证筛选器的update_tradable_mask被调用
        mock_filter.update_tradable_mask.assert_called_once()
        
        # 验证可交易掩码被正确应用
        expected_mask = np.array([True, True, False, True, False])
        np.testing.assert_array_equal(env._current_tradable_mask, expected_mask)
        
        # 验证不可交易股票的权重为0
        self.assertEqual(env.portfolio_weights[2], 0.0)
        self.assertEqual(env.portfolio_weights[4], 0.0)
    
    def test_complete_trading_episode(self):
        """测试完整的交易回合"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        # 重置环境
        obs, info = env.reset()
        
        # 执行多步交易
        total_reward = 0
        steps = 0
        max_steps = 20
        
        while steps < max_steps:
            # 生成随机动作
            action = np.random.uniform(-0.1, 0.1, size=5)
            
            # 执行步骤
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # 验证观测维度
            self.assertGreaterEqual(len(obs), 14)
            
            # 验证信息字典
            self.assertIn('current_step', info)
            self.assertIn('portfolio_value', info)
            self.assertIn('max_drawdown', info)
            
            if terminated or truncated:
                break
        
        # 验证交易执行了多步
        self.assertGreater(steps, 1)
        
        # 验证组合价值为正数
        self.assertGreater(info['portfolio_value'], 0)
    
    def test_filter_error_handling(self):
        """测试筛选器错误处理"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        # 创建会抛出异常的模拟筛选器
        mock_filter = Mock()
        mock_filter.update_tradable_mask.side_effect = Exception("筛选器错误")
        mock_filter.get_current_regime.side_effect = Exception("状态检测错误")
        mock_filter.get_filter_strength.return_value = 0.5  # 确保返回数值
        
        env.lowvol_filter = mock_filter
        
        # 重置环境
        obs, info = env.reset()
        
        # 执行步骤（应该能正常处理异常）
        action = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 验证环境仍能正常运行
        self.assertGreaterEqual(len(obs), 14)
        self.assertIsNone(env._current_tradable_mask)
        
        # 验证市场制度信号回退到默认值 [0, 1, 0]
        regime_signals = obs[13:16]  # 市场制度信号在位置13-15
        expected_regime = np.array([0.0, 1.0, 0.0])  # 默认中等波动制度
        np.testing.assert_array_equal(regime_signals, expected_regime)
    
    def test_mask_length_mismatch_handling(self):
        """测试掩码长度不匹配的处理"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        # 设置长度不匹配的掩码
        env._current_tradable_mask = np.array([True, True, True])  # 长度为3，但股票数为5
        
        # 测试动作约束
        action = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        constrained_action = env._constrain_action(action)
        
        # 验证动作没有被掩码修改（因为长度不匹配）
        np.testing.assert_array_almost_equal(
            constrained_action, 
            np.clip(action, -env.max_position, env.max_position)
        )
    
    def test_regime_signal_mapping(self):
        """测试市场状态信号映射"""
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.base_config
        )
        
        # 测试不同状态的映射 (one-hot编码)
        test_cases = [
            ("低", np.array([1.0, 0.0, 0.0])),
            ("中", np.array([0.0, 1.0, 0.0])),
            ("高", np.array([0.0, 0.0, 1.0])),
            ("未知", np.array([0.0, 1.0, 0.0]))  # 未知状态应该映射为中性
        ]
        
        for regime, expected_regime in test_cases:
            mock_filter = Mock()
            mock_filter.get_current_regime.return_value = regime
            mock_filter.get_filter_strength.return_value = 0.5  # 确保返回数值
            env.lowvol_filter = mock_filter
            
            obs = env._get_observation()
            # 验证市场制度信号 (one-hot编码)
            regime_signals = obs[13:16]  # 市场制度信号在位置13-15
            
            np.testing.assert_array_equal(regime_signals, expected_regime, 
                           f"状态'{regime}'应该映射为{expected_regime}")


class TestTradingEnvironmentPerformance(unittest.TestCase):
    """交易环境性能测试"""
    
    def setUp(self):
        """设置性能测试数据"""
        # 创建较大的数据集
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        stocks = [f'STOCK_{i:03d}' for i in range(50)]  # 50只股票
        
        np.random.seed(42)
        price_data = pd.DataFrame(
            np.random.lognormal(0, 0.02, (1000, 50)).cumprod(axis=0) * 100,
            index=dates,
            columns=stocks
        )
        
        factor_names = [f'factor_{i}' for i in range(20)]  # 20个因子
        factor_data = pd.DataFrame(
            np.random.randn(1000, 20),
            index=dates,
            columns=factor_names
        )
        
        self.price_data = price_data
        self.factor_data = factor_data
        
        self.config = {
            'lookback_window': 20,
            'transaction_cost': 0.001,
            'max_position': 0.02,  # 降低单股票仓位限制
            'max_leverage': 1.0
        }
    
    def test_environment_initialization_performance(self):
        """测试环境初始化性能"""
        import time
        
        start_time = time.time()
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.config
        )
        initialization_time = time.time() - start_time
        
        # 验证初始化时间合理（应该小于1秒）
        self.assertLess(initialization_time, 1.0)
        
        # 验证环境正确初始化
        self.assertEqual(env.n_stocks, 50)
        self.assertEqual(env.n_factors, 20)
    
    def test_step_performance(self):
        """测试单步执行性能"""
        import time
        
        env = TradingEnvironment(
            factor_data=self.factor_data,
            price_data=self.price_data,
            config=self.config
        )
        
        obs, info = env.reset()
        
        # 测试多步执行时间
        step_times = []
        for _ in range(100):
            action = np.random.uniform(-0.01, 0.01, size=50)
            
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)
            
            if terminated or truncated:
                break
        
        # 验证平均步骤时间合理（应该小于10ms）
        avg_step_time = np.mean(step_times)
        self.assertLess(avg_step_time, 0.01)
        
        print(f"平均步骤执行时间: {avg_step_time*1000:.2f}ms")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)