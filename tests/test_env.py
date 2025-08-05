"""
PortfolioEnv环境测试用例
测试环境的初始化、step、reset等核心功能
"""
import pytest
import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from unittest.mock import Mock, patch

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from env import PortfolioEnv, DrawdownEarlyStoppingCallback


class TestPortfolioEnv:
    """PortfolioEnv测试类"""
    
    def setup_method(self):
        """设置测试数据"""
        # 创建模拟数据
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        stocks = ["000001.SZ", "000002.SZ", "600000.SH"]
        
        data_list = []
        np.random.seed(42)  # 设置随机种子保证结果可重复
        
        for i, date in enumerate(dates):
            for j, stock in enumerate(stocks):
                base_price = 100 + j * 10  # 不同股票不同基准价格
                price_change = np.sin(i * 0.1) * 5 + np.random.normal(0, 2)  # 添加趋势和噪音
                
                data_list.append({
                    "datetime": date,
                    "instrument": stock,
                    "$close": base_price + price_change,
                    "$open": base_price + price_change + np.random.normal(0, 0.5),
                    "$high": base_price + price_change + abs(np.random.normal(0, 1)),
                    "$low": base_price + price_change - abs(np.random.normal(0, 1)),
                    "$volume": np.random.randint(1000, 10000)
                })
        
        self.test_data = pd.DataFrame(data_list)
        self.test_data = self.test_data.set_index(["datetime", "instrument"])
        
        # 创建环境
        self.env = PortfolioEnv(
            data=self.test_data,
            initial_cash=100000,
            lookback_window=10,
            transaction_cost=0.003,
            max_drawdown_threshold=0.15
        )
    
    def test_init(self):
        """测试环境初始化"""
        assert self.env.initial_cash == 100000
        assert self.env.lookback_window == 10
        assert self.env.transaction_cost == 0.003
        assert self.env.num_stocks == 3
        assert len(self.env.time_index) > 300  # 大约一年的交易日
        
        # 检查动作和观察空间
        assert isinstance(self.env.action_space, gym.spaces.Box)
        assert self.env.action_space.shape == (3,)  # 3只股票
        assert isinstance(self.env.observation_space, gym.spaces.Box)
    
    def test_reset(self):
        """测试环境重置"""
        obs, info = self.env.reset(seed=42)
        
        # 检查重置后状态
        assert self.env.current_step == self.env.lookback_window
        assert self.env.total_value == self.env.initial_cash
        assert self.env.cash == self.env.initial_cash
        assert np.allclose(self.env.holdings, 0)
        assert np.allclose(self.env.weights, 0)
        
        # 检查观察
        assert isinstance(obs, np.ndarray)
        assert obs.shape == self.env.observation_space.shape
        
        # 检查信息
        assert isinstance(info, dict)
        assert "total_value" in info
        assert "current_drawdown" in info
    
    def test_step_basic(self):
        """测试基本step操作"""
        obs, info = self.env.reset(seed=42)
        
        # 执行一个动作
        action = np.array([0.4, 0.3, 0.3])  # 权重分配
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 检查返回值
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # 检查环境状态变化
        assert self.env.current_step == self.env.lookback_window + 1
        assert np.allclose(self.env.weights, action, atol=1e-6)
        assert self.env.total_value != self.env.initial_cash  # 价值应该有变化
    
    def test_step_sequence(self):
        """测试连续step操作"""
        obs, info = self.env.reset(seed=42)
        
        total_rewards = 0
        steps = 0
        
        for _ in range(20):  # 执行20步
            action = np.random.dirichlet([1, 1, 1])  # 随机权重分配，和为1
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_rewards += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps > 0
        assert isinstance(total_rewards, (float, np.floating))
        assert self.env.current_step > self.env.lookback_window
    
    def test_action_normalization(self):
        """测试动作归一化"""
        obs, info = self.env.reset(seed=42)
        
        # 测试权重和不为1的情况
        action = np.array([0.6, 0.8, 0.4])  # 和为1.8
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 权重应该被归一化为和为1
        expected_weights = action / action.sum()
        assert np.allclose(self.env.weights, expected_weights, atol=1e-6)
    
    def test_action_clipping(self):
        """测试动作裁剪"""
        obs, info = self.env.reset(seed=42)
        
        # 测试超出范围的动作
        action = np.array([-0.1, 1.5, 0.8])  # 包含负值和大于1的值
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 动作应该被裁剪到[0,1]范围并归一化
        clipped_action = np.clip(action, 0, 1)
        expected_weights = clipped_action / clipped_action.sum()
        assert np.allclose(self.env.weights, expected_weights, atol=1e-6)
    
    def test_transaction_cost_calculation(self):
        """测试交易成本计算"""
        obs, info = self.env.reset(seed=42)
        
        initial_transaction_costs = self.env.transaction_costs
        
        # 执行一个较大的权重变化
        action = np.array([1.0, 0.0, 0.0])  # 全部买入第一只股票
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 交易成本应该增加
        assert self.env.transaction_costs > initial_transaction_costs
    
    def test_drawdown_calculation(self):
        """测试回撤计算"""
        obs, info = self.env.reset(seed=42)
        
        # 执行几步使组合价值发生变化
        for _ in range(10):
            action = np.random.dirichlet([1, 1, 1])
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # 回撤应该在合理范围内
        assert 0 <= self.env.current_drawdown <= 1
        assert len(self.env.drawdown_history) > 0
    
    def test_performance_metrics(self):
        """测试性能指标计算"""
        obs, info = self.env.reset(seed=42)
        
        # 运行一段时间
        for _ in range(50):
            action = np.random.dirichlet([1, 1, 1])
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # 获取性能指标
        performance = self.env.get_portfolio_performance()
        
        assert isinstance(performance, dict)
        assert "total_return" in performance
        assert "annualized_return" in performance
        assert "volatility" in performance
        assert "sharpe_ratio" in performance
        assert "max_drawdown" in performance
    
    def test_termination_conditions(self):
        """测试终止条件"""
        # 创建一个容易触发回撤限制的环境
        env = PortfolioEnv(
            data=self.test_data,
            initial_cash=100000,
            max_drawdown_threshold=0.01,  # 很严格的回撤限制
            lookback_window=5
        )
        
        obs, info = env.reset(seed=42)
        
        terminated = False
        steps = 0
        
        while not terminated and steps < 100:
            # 使用可能导致损失的策略
            action = np.array([1.0, 0.0, 0.0])  # 总是全仓一只股票
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
        
        # 应该在某个时候终止（要么回撤过大，要么到达序列末尾）
        assert terminated or steps == 100
    
    def test_info_dict_completeness(self):
        """测试info字典的完整性"""
        obs, info = self.env.reset(seed=42)
        
        action = np.array([0.4, 0.3, 0.3])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 检查必要的信息字段
        required_fields = [
            "total_value", "cash", "holdings_value", "weights",
            "current_drawdown", "max_drawdown", "transaction_costs",
            "total_return", "sharpe_ratio", "current_step", "num_trades"
        ]
        
        for field in required_fields:
            assert field in info, f"缺少字段: {field}"
    
    def test_render(self):
        """测试渲染功能"""
        obs, info = self.env.reset(seed=42)
        
        # 应该不抛出异常
        self.env.render()
        
        action = np.array([0.4, 0.3, 0.3])
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.env.render()


class TestDrawdownEarlyStoppingCallback:
    """回撤早停回调测试类"""
    
    def test_init(self):
        """测试初始化"""
        callback = DrawdownEarlyStoppingCallback(max_drawdown=0.1, patience=5)
        
        assert callback.max_drawdown == 0.1
        assert callback.patience == 5
        assert callback.violation_count == 0
    
    def test_no_violation(self):
        """测试无违反情况"""
        callback = DrawdownEarlyStoppingCallback(max_drawdown=0.1, patience=3)
        
        # 创建Mock环境
        mock_env = Mock()
        mock_env.current_drawdown = 0.05  # 未超过阈值
        
        result = callback(mock_env)
        
        assert result == False  # 不应该早停
        assert callback.violation_count == 0
    
    def test_violation_not_enough_patience(self):
        """测试违反但未达到耐心值"""
        callback = DrawdownEarlyStoppingCallback(max_drawdown=0.1, patience=3)
        
        mock_env = Mock()
        mock_env.current_drawdown = 0.15  # 超过阈值
        
        # 违反但次数不够
        for i in range(2):
            result = callback(mock_env)
            assert result == False
            assert callback.violation_count == i + 1
    
    def test_violation_trigger_early_stop(self):
        """测试违反并触发早停"""
        callback = DrawdownEarlyStoppingCallback(max_drawdown=0.1, patience=3)
        
        mock_env = Mock()
        mock_env.current_drawdown = 0.15  # 超过阈值
        
        # 连续违反直到触发早停
        for i in range(3):
            result = callback(mock_env)
            if i < 2:
                assert result == False
            else:
                assert result == True  # 应该触发早停
    
    def test_violation_reset(self):
        """测试违反计数重置"""
        callback = DrawdownEarlyStoppingCallback(max_drawdown=0.1, patience=3)
        
        mock_env = Mock()
        
        # 先违反
        mock_env.current_drawdown = 0.15
        callback(mock_env)
        assert callback.violation_count == 1
        
        # 然后恢复正常
        mock_env.current_drawdown = 0.05
        callback(mock_env)
        assert callback.violation_count == 0


class TestPortfolioEnvIntegration:
    """环境集成测试"""
    
    def test_env_with_stable_baselines3_interface(self):
        """测试与Stable-Baselines3接口兼容性"""
        # 创建简单测试数据
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
        stocks = ["A", "B"]
        
        data_list = []
        for date in dates:
            for stock in stocks:
                data_list.append({
                    "datetime": date,
                    "instrument": stock,
                    "$close": 100 + np.random.randn(),
                    "$open": 100 + np.random.randn(),
                    "$high": 102 + abs(np.random.randn()),
                    "$low": 98 - abs(np.random.randn()),
                    "$volume": np.random.randint(1000, 5000)
                })
        
        data = pd.DataFrame(data_list).set_index(["datetime", "instrument"])
        
        env = PortfolioEnv(data=data, initial_cash=10000, lookback_window=5)
        
        # 测试gym接口
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        
        # 测试重置-步骤循环
        obs, info = env.reset()
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 检查返回值类型
            assert obs in env.observation_space
            assert isinstance(reward, (float, np.floating))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            if terminated or truncated:
                break
    
    def test_multiple_reset_consistency(self):
        """测试多次重置的一致性"""
        dates = pd.date_range("2023-01-01", "2023-06-30", freq="D")
        stocks = ["X", "Y", "Z"]
        
        data_list = []
        np.random.seed(123)
        for date in dates:
            for stock in stocks:
                data_list.append({
                    "datetime": date,
                    "instrument": stock,
                    "$close": 50 + np.random.randn() * 5,
                    "$open": 50 + np.random.randn() * 5,
                    "$high": 55 + abs(np.random.randn()),
                    "$low": 45 - abs(np.random.randn()),
                    "$volume": np.random.randint(500, 2000)
                })
        
        data = pd.DataFrame(data_list).set_index(["datetime", "instrument"])
        env = PortfolioEnv(data=data, initial_cash=50000, lookback_window=8)
        
        # 多次重置应该返回相同的初始状态
        obs1, info1 = env.reset(seed=999)
        obs2, info2 = env.reset(seed=999)
        
        assert np.allclose(obs1, obs2)
        assert info1['total_value'] == info2['total_value']
        assert info1['current_drawdown'] == info2['current_drawdown']


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])