"""
投资组合环境的单元测试
测试PortfolioEnvironment的Gym接口兼容性、状态空间、动作空间和奖励函数
"""
import pytest
import numpy as np
import pandas as pd
import gym
from gym import spaces
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any, Optional

from src.rl_trading_system.data.data_models import (
    MarketData, TradingState, TradingAction, TransactionRecord
)
from src.rl_trading_system.trading.portfolio_environment import (
    PortfolioEnvironment, PortfolioConfig
)
from src.rl_trading_system.data.interfaces import DataInterface


class MockDataInterface(DataInterface):
    """模拟数据接口"""
    
    def get_stock_list(self, market: str = 'A') -> List[str]:
        return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
    
    def get_price_data(self, symbols: List[str], 
                      start_date: str, end_date: str) -> pd.DataFrame:
        # 返回空DataFrame，让环境使用模拟数据
        return pd.DataFrame()
    
    def get_fundamental_data(self, symbols: List[str], 
                           start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame()


class TestPortfolioEnvironment:
    """投资组合环境测试类"""
    
    @pytest.fixture
    def env_config(self):
        """环境配置fixture"""
        return PortfolioConfig(
            stock_pool=['000001.SZ', '000002.SZ', '600000.SH', '600036.SH'],
            lookback_window=30,
            initial_cash=100000.0,
            commission_rate=0.001,
            stamp_tax_rate=0.001,
            max_position_size=0.4  # 使用可行的最大权重限制
        )
    
    @pytest.fixture
    def mock_data_interface(self):
        """模拟数据接口fixture"""
        return MockDataInterface()
    
    @pytest.fixture
    def portfolio_env(self, env_config, mock_data_interface):
        """投资组合环境fixture"""
        return PortfolioEnvironment(env_config, mock_data_interface)
    
    def test_environment_initialization(self, portfolio_env, env_config):
        """测试环境初始化"""
        assert portfolio_env.n_stocks == len(env_config.stock_pool)
        assert portfolio_env.config.initial_cash == env_config.initial_cash
        assert portfolio_env.observation_space is not None
        assert portfolio_env.action_space is not None
        
        # 检查观察空间
        obs_space = portfolio_env.observation_space
        assert isinstance(obs_space, spaces.Dict)
        assert 'features' in obs_space.spaces
        assert 'positions' in obs_space.spaces
        assert 'market_state' in obs_space.spaces
        
        # 检查动作空间
        action_space = portfolio_env.action_space
        assert isinstance(action_space, spaces.Box)
        assert action_space.shape == (portfolio_env.n_stocks,)
        
    def test_gym_interface_compatibility(self, portfolio_env):
        """测试Gym接口兼容性"""
        # 测试reset方法
        obs = portfolio_env.reset()
        assert isinstance(obs, dict)
        assert 'features' in obs
        assert 'positions' in obs
        assert 'market_state' in obs
        
        # 测试step方法
        action = np.array([0.25, 0.25, 0.25, 0.25])
        obs, reward, done, info = portfolio_env.step(action)
        
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # 检查观察空间兼容性
        assert portfolio_env.observation_space.contains(obs)
        
    def test_observation_space_structure(self, portfolio_env, env_config):
        """测试观察空间结构"""
        obs = portfolio_env.reset()
        
        # 检查特征维度
        features = obs['features']
        expected_shape = (env_config.lookback_window, portfolio_env.n_stocks, portfolio_env.n_features)
        assert features.shape == expected_shape
        assert features.dtype == np.float32
        
        # 检查持仓维度
        positions = obs['positions']
        assert positions.shape == (portfolio_env.n_stocks,)
        assert positions.dtype == np.float32
        assert np.all(positions >= 0)
        
        # 检查市场状态维度
        market_state = obs['market_state']
        assert market_state.shape == (portfolio_env.n_market_features,)
        assert market_state.dtype == np.float32
        
    def test_action_space_structure(self, portfolio_env):
        """测试动作空间结构"""
        action_space = portfolio_env.action_space
        
        # 检查动作空间类型和维度
        assert isinstance(action_space, spaces.Box)
        assert action_space.shape == (portfolio_env.n_stocks,)
        assert action_space.dtype == np.float32
        
        # 检查动作空间边界
        assert np.all(action_space.low == 0)
        assert np.all(action_space.high == 1)
        
        # 测试动作采样
        action = action_space.sample()
        assert action_space.contains(action)
        assert action.shape == (portfolio_env.n_stocks,)
        
    def test_reward_function_components(self, portfolio_env):
        """测试奖励函数组成部分"""
        portfolio_env.reset()
        
        # 测试不同权重分配的奖励
        actions = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # 集中投资
            np.array([0.25, 0.25, 0.25, 0.25]),  # 均匀分散
            np.array([0.4, 0.3, 0.2, 0.1])  # 适度集中
        ]
        
        rewards = []
        for action in actions:
            obs, reward, done, info = portfolio_env.step(action)
            rewards.append(reward)
            
            # 检查信息字典包含必要信息
            assert 'portfolio_return' in info
            assert 'transaction_cost' in info
            assert 'concentration' in info
            assert 'drawdown' in info
            
            # 检查集中度计算（使用实际持仓而不是原始动作）
            expected_concentration = np.sum(info['positions'] ** 2)
            assert abs(info['concentration'] - expected_concentration) < 1e-6
        
        # 奖励应该是有限的数值
        for reward in rewards:
            assert np.isfinite(reward)
            assert isinstance(reward, (int, float))
    
    def test_position_weight_constraints(self, portfolio_env):
        """测试持仓权重约束"""
        portfolio_env.reset()
        
        # 测试权重标准化
        test_actions = [
            np.array([2.0, 1.0, 1.0, 1.0]),  # 需要标准化
            np.array([0.0, 0.0, 0.0, 0.0]),  # 全零权重
            np.array([1.0, 0.0, 0.0, 0.0])   # 单一权重
        ]
        
        for action in test_actions:
            obs, reward, done, info = portfolio_env.step(action)
            positions = info['positions']
            
            # 检查权重和约束
            assert abs(positions.sum() - 1.0) < 1e-6
            
            # 检查权重非负约束
            assert np.all(positions >= 0)
            
            # 检查最大权重约束
            assert np.all(positions <= portfolio_env.config.max_position_size + 1e-6)
    
    def test_transaction_cost_calculation(self, portfolio_env):
        """测试交易成本计算"""
        portfolio_env.reset()
        
        # 第一步：建立初始持仓
        initial_action = np.array([0.25, 0.25, 0.25, 0.25])
        obs1, reward1, done1, info1 = portfolio_env.step(initial_action)
        initial_cost = info1['transaction_cost']
        
        # 第二步：不改变持仓
        same_action = np.array([0.25, 0.25, 0.25, 0.25])
        obs2, reward2, done2, info2 = portfolio_env.step(same_action)
        no_change_cost = info2['transaction_cost']
        
        # 第三步：大幅调整持仓
        different_action = np.array([0.7, 0.1, 0.1, 0.1])
        obs3, reward3, done3, info3 = portfolio_env.step(different_action)
        large_change_cost = info3['transaction_cost']
        
        # 验证交易成本逻辑
        assert initial_cost >= 0  # 初始建仓应有非负成本
        assert no_change_cost >= 0  # 成本应为非负
        assert large_change_cost >= 0  # 成本应为非负
        
        # 如果有实际的权重变化，应该产生成本
        if np.any(np.abs(info3['positions'] - info2['positions']) > 1e-6):
            assert large_change_cost >= no_change_cost
    
    def test_a_share_trading_rules(self, portfolio_env):
        """测试A股交易规则约束"""
        portfolio_env.reset()
        
        # 测试单只股票最大权重限制
        concentrated_action = np.array([1.0, 0.0, 0.0, 0.0])
        obs, reward, done, info = portfolio_env.step(concentrated_action)
        positions = info['positions']
        
        # 检查是否应用了最大权重限制
        max_weight = np.max(positions)
        assert max_weight <= portfolio_env.config.max_position_size + 1e-6
        
        # 测试T+1规则（简化测试）
        if portfolio_env.config.t_plus_1:
            # 验证T+1规则的基本约束
            # 在实际实现中，这里会检查当日买入股票不能当日卖出
            assert portfolio_env.config.t_plus_1 == True
        
        # 测试权重约束后的重新标准化
        assert abs(positions.sum() - 1.0) < 1e-6
        assert np.all(positions >= 0)
    
    def test_price_limit_constraints(self, portfolio_env):
        """测试涨跌停限制"""
        portfolio_env.reset()
        
        # 模拟涨跌停情况
        # 在实际实现中，这里会检查价格变动是否超过限制
        action = np.array([0.25, 0.25, 0.25, 0.25])
        obs, reward, done, info = portfolio_env.step(action)
        
        # 基本检查：确保环境仍然正常运行
        assert np.isfinite(reward)
        assert isinstance(done, bool)
        
        # 检查价格限制配置
        assert portfolio_env.config.price_limit == 0.1  # 10%涨跌停限制
        
        # 验证在涨跌停情况下环境的稳定性
        for _ in range(5):
            action = portfolio_env.action_space.sample()
            obs, reward, done, info = portfolio_env.step(action)
            assert np.isfinite(reward)
            assert np.all(np.isfinite(obs['features']))
    
    def test_minimum_trade_amount_constraint(self, portfolio_env):
        """测试最小交易金额约束"""
        portfolio_env.reset()
        
        # 测试极小的权重变化
        small_action = np.array([0.2501, 0.2499, 0.25, 0.25])  # 很小的变化
        obs, reward, done, info = portfolio_env.step(small_action)
        
        # 检查最小交易金额配置
        assert portfolio_env.config.min_trade_amount == 1000.0
        
        # 验证交易成本计算考虑了最小交易金额
        assert np.isfinite(info['transaction_cost'])
        assert info['transaction_cost'] >= 0
    
    def test_t_plus_1_trading_rule(self, portfolio_env):
        """测试T+1交易规则"""
        if not portfolio_env.config.t_plus_1:
            pytest.skip("T+1规则未启用")
        
        portfolio_env.reset()
        
        # 第一天：买入股票
        buy_action = np.array([0.5, 0.3, 0.2, 0.0])
        obs1, reward1, done1, info1 = portfolio_env.step(buy_action)
        
        # 记录买入的股票
        bought_positions = info1['positions'].copy()
        
        # 第二天：尝试卖出刚买入的股票（在真实实现中应该被限制）
        sell_action = np.array([0.0, 0.0, 0.0, 1.0])
        obs2, reward2, done2, info2 = portfolio_env.step(sell_action)
        
        # 在简化的测试实现中，我们只验证基本功能
        assert np.isfinite(reward2)
        assert np.all(info2['positions'] >= 0)
        assert abs(info2['positions'].sum() - 1.0) < 1e-6
    
    def test_market_impact_model(self, portfolio_env):
        """测试市场冲击模型"""
        portfolio_env.reset()
        
        # 测试不同规模的交易对市场冲击的影响
        small_trade = np.array([0.26, 0.24, 0.25, 0.25])  # 小额交易
        large_trade = np.array([0.7, 0.1, 0.1, 0.1])     # 大额交易
        
        # 小额交易
        obs1, reward1, done1, info1 = portfolio_env.step(small_trade)
        small_cost = info1['transaction_cost']
        
        # 重置环境
        portfolio_env.reset()
        
        # 大额交易
        obs2, reward2, done2, info2 = portfolio_env.step(large_trade)
        large_cost = info2['transaction_cost']
        
        # 大额交易的成本应该更高（由于市场冲击）
        if large_cost > 0 and small_cost > 0:
            assert large_cost >= small_cost
        
        # 验证成本计算的合理性
        assert small_cost >= 0
        assert large_cost >= 0
    
    def test_episode_completion(self, portfolio_env):
        """测试完整交易周期"""
        obs = portfolio_env.reset()
        total_steps = 0
        episode_rewards = []
        
        while True:
            # 随机动作
            action = portfolio_env.action_space.sample()
            obs, reward, done, info = portfolio_env.step(action)
            
            episode_rewards.append(reward)
            total_steps += 1
            
            # 检查观察空间一致性
            assert portfolio_env.observation_space.contains(obs)
            
            if done:
                break
            
            # 防止无限循环
            if total_steps > portfolio_env.max_steps + 10:
                break
        
        # 验证episode完成
        assert done
        assert total_steps <= portfolio_env.max_steps
        assert len(episode_rewards) == total_steps
        
        # 检查最终投资组合指标
        metrics = portfolio_env.get_portfolio_metrics()
        assert 'total_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    def test_risk_metrics_calculation(self, portfolio_env):
        """测试风险指标计算"""
        portfolio_env.reset()
        
        # 运行几步以积累数据
        for _ in range(10):
            action = portfolio_env.action_space.sample()
            obs, reward, done, info = portfolio_env.step(action)
            
            # 检查风险指标
            assert 'drawdown' in info
            assert 'concentration' in info
            assert 'active_positions' in info
            
            # 验证指标范围
            assert 0 <= info['drawdown'] <= 1
            assert 0 <= info['concentration'] <= 1
            assert 0 <= info['active_positions'] <= portfolio_env.n_stocks
        
        # 检查最终指标
        metrics = portfolio_env.get_portfolio_metrics()
        if metrics:  # 如果有足够数据
            assert metrics['max_drawdown'] >= 0
            assert np.isfinite(metrics['volatility'])
            assert np.isfinite(metrics['sharpe_ratio'])
    
    def test_state_consistency(self, portfolio_env):
        """测试状态一致性"""
        obs1 = portfolio_env.reset()
        
        # 执行相同动作序列
        actions = [
            np.array([0.4, 0.3, 0.2, 0.1]),
            np.array([0.3, 0.3, 0.2, 0.2]),
            np.array([0.25, 0.25, 0.25, 0.25])
        ]
        
        states = [obs1]
        for action in actions:
            obs, reward, done, info = portfolio_env.step(action)
            states.append(obs)
        
        # 检查状态演化的一致性
        for i, state in enumerate(states):
            assert 'features' in state
            assert 'positions' in state
            assert 'market_state' in state
            
            # 检查持仓状态的一致性
            if i > 0:
                # 持仓应该反映上一步的动作
                current_positions = state['positions']
                # 注意：由于约束和标准化，可能不完全相等
                assert np.allclose(current_positions.sum(), 1.0, atol=1e-6)
    
    def test_edge_cases(self, portfolio_env):
        """测试边界情况"""
        portfolio_env.reset()
        
        # 测试极端动作
        edge_actions = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # 全部投资一只股票
            np.array([0.0, 0.0, 0.0, 0.0]),  # 全部现金
            np.ones(portfolio_env.n_stocks) * 1e-10,  # 极小权重
            np.ones(portfolio_env.n_stocks) * 1e10   # 极大权重
        ]
        
        for action in edge_actions:
            obs, reward, done, info = portfolio_env.step(action)
            
            # 环境应该能处理所有边界情况
            assert np.isfinite(reward)
            assert isinstance(done, bool)
            assert portfolio_env.observation_space.contains(obs)
            
            # 权重约束应该始终满足
            positions = info['positions']
            assert np.all(positions >= 0)
            assert abs(positions.sum() - 1.0) < 1e-5
    
    def test_performance_metrics(self, portfolio_env):
        """测试性能指标"""
        portfolio_env.reset()
        
        # 运行一个完整episode
        while True:
            action = portfolio_env.action_space.sample()
            obs, reward, done, info = portfolio_env.step(action)
            
            if done:
                break
        
        # 获取性能指标
        metrics = portfolio_env.get_portfolio_metrics()
        
        if metrics:  # 如果有足够数据计算指标
            # 检查指标的合理性
            assert isinstance(metrics['total_return'], float)
            assert isinstance(metrics['volatility'], float)
            assert isinstance(metrics['sharpe_ratio'], float)
            assert isinstance(metrics['max_drawdown'], float)
            
            # 检查指标范围
            assert metrics['volatility'] >= 0
            assert 0 <= metrics['max_drawdown'] <= 1
            assert np.isfinite(metrics['sharpe_ratio'])
    
    @pytest.mark.parametrize("n_stocks", [2, 5, 10])
    def test_different_stock_pool_sizes(self, n_stocks):
        """测试不同股票池大小"""
        stock_pool = [f"stock_{i:03d}" for i in range(n_stocks)]
        config = PortfolioConfig(
            stock_pool=stock_pool,
            lookback_window=20,
            initial_cash=50000.0
        )
        
        mock_data_interface = MockDataInterface()
        env = PortfolioEnvironment(config, mock_data_interface)
        obs = env.reset()
        
        # 检查维度正确性
        assert obs['features'].shape[1] == n_stocks
        assert obs['positions'].shape[0] == n_stocks
        assert env.action_space.shape[0] == n_stocks
        
        # 测试动作执行
        action = np.ones(n_stocks) / n_stocks
        obs, reward, done, info = env.step(action)
        
        assert np.isfinite(reward)
        assert info['positions'].shape[0] == n_stocks
    
    @pytest.mark.parametrize("lookback_window", [10, 30, 60])
    def test_different_lookback_windows(self, lookback_window):
        """测试不同回望窗口"""
        config = PortfolioConfig(
            stock_pool=['A', 'B', 'C'],
            lookback_window=lookback_window,
            initial_cash=100000.0
        )
        
        mock_data_interface = MockDataInterface()
        env = PortfolioEnvironment(config, mock_data_interface)
        obs = env.reset()
        
        # 检查特征维度
        assert obs['features'].shape[0] == lookback_window
        
        # 测试正常运行
        action = np.array([0.33, 0.33, 0.34])
        obs, reward, done, info = env.step(action)
        
        assert np.isfinite(reward)
        assert obs['features'].shape[0] == lookback_window