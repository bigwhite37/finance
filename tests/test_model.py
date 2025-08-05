"""
模型组件测试用例
测试TimeSeriesTransformer、风险感知奖励等模型组件
"""
import pytest
import os
import sys 
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, patch

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import (
    TimeSeriesTransformer, 
    PositionalEncoding,
    MultiHeadStockAttention,
    RiskAwareRewardWrapper,
    PortfolioMetrics
)


class TestTimeSeriesTransformer:
    """TimeSeriesTransformer测试类"""
    
    def setup_method(self):
        """设置测试参数"""
        self.lookback_window = 20
        self.num_stocks = 5
        self.num_features = 4
        self.batch_size = 16
        
        # 计算观察空间维度
        hist_dim = self.lookbook_window * self.num_stocks * self.num_features
        weight_dim = self.num_stocks  
        state_dim = 3
        obs_dim = hist_dim + weight_dim + state_dim
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.extractor = TimeSeriesTransformer(
            observation_space=self.observation_space,
            lookback_window=self.lookback_window,
            num_stocks=self.num_stocks,
            num_features=self.num_features,
            features_dim=128
        )
    
    def test_init(self):
        """测试初始化"""
        assert self.extractor.lookback_window == self.lookback_window
        assert self.extractor.num_stocks == self.num_stocks
        assert self.extractor.num_features == self.num_features
        assert self.extractor.d_model == 128  # 默认值
        
        # 检查网络层
        assert isinstance(self.extractor.input_projection, nn.Linear)
        assert isinstance(self.extractor.positional_encoding, PositionalEncoding)
        assert isinstance(self.extractor.transformer, nn.TransformerEncoder)
        assert isinstance(self.extractor.stock_attention, MultiHeadStockAttention)
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        # 创建输入
        obs = torch.randn(self.batch_size, self.observation_space.shape[0])
        
        with torch.no_grad():
            output = self.extractor(obs)
        
        # 检查输出形状
        assert output.shape == (self.batch_size, 128)  # features_dim
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_forward_different_batch_sizes(self):
        """测试不同批次大小的前向传播"""
        batch_sizes = [1, 8, 32, 64]
        
        for batch_size in batch_sizes:
            obs = torch.randn(batch_size, self.observation_space.shape[0])
            
            with torch.no_grad():
                output = self.extractor(obs)
            
            assert output.shape == (batch_size, 128)
            assert not torch.isnan(output).any()
    
    def test_gradient_flow(self):
        """测试梯度流"""
        obs = torch.randn(self.batch_size, self.observation_space.shape[0], requires_grad=True)
        
        output = self.extractor(obs)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度存在
        assert obs.grad is not None
        assert not torch.isnan(obs.grad).any()
        
        # 检查模型参数梯度
        for param in self.extractor.parameters():  
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    def test_parameter_count(self):
        """测试参数数量合理性"""
        total_params = sum(p.numel() for p in self.extractor.parameters())
        trainable_params = sum(p.numel() for p in self.extractor.parameters() if p.requires_grad)
        
        # 参数数量应该在合理范围内（避免过大或过小）
        assert 1000 < total_params < 10000000  # 1K到10M参数
        assert trainable_params == total_params  # 所有参数都应该可训练


class TestPositionalEncoding:
    """位置编码测试类"""
    
    def test_init_and_forward(self):
        """测试初始化和前向传播"""
        d_model = 64
        max_len = 100
        
        pe = PositionalEncoding(d_model, max_len)
        
        # 测试输入
        batch_size = 8
        seq_len = 50
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pe(x)
        
        # 检查输出形状
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_positional_encoding_pattern(self):
        """测试位置编码模式"""
        d_model = 4
        pe = PositionalEncoding(d_model, max_len=10)
        
        x = torch.zeros(1, 5, d_model)
        output = pe(x)
        
        # 位置编码应该随位置变化
        pos_0 = output[0, 0, :]
        pos_1 = output[0, 1, :]
        
        assert not torch.allclose(pos_0, pos_1, atol=1e-6)


class TestMultiHeadStockAttention:
    """多头股票注意力测试类"""
    
    def test_forward(self):
        """测试前向传播"""
        d_model = 64
        nhead = 8
        batch_size = 4
        num_stocks = 10
        
        attention = MultiHeadStockAttention(d_model, nhead)
        
        # 输入: [batch_size, num_stocks, d_model]
        stock_features = torch.randn(batch_size, num_stocks, d_model)
        
        output = attention(stock_features)
        
        # 输出: [batch_size, d_model]
        assert output.shape == (batch_size, d_model)
        assert not torch.isnan(output).any()
    
    def test_attention_aggregation(self):
        """测试注意力聚合"""
        d_model = 32
        nhead = 4
        attention = MultiHeadStockAttention(d_model, nhead)
        
        # 创建有明显差异的输入
        batch_size = 2
        num_stocks = 5
        stock_features = torch.randn(batch_size, num_stocks, d_model)
        
        # 设置一个股票特征为很大的值
        stock_features[:, 0, :] = 10.0
        
        output = attention(stock_features)
        
        # 输出不应该全零且应该有变化
        assert not torch.allclose(output, torch.zeros_like(output))
        assert output.shape == (batch_size, d_model)


class TestRiskAwareRewardWrapper:
    """风险感知奖励包装器测试类"""
    
    def test_init(self):
        """测试初始化"""
        wrapper = RiskAwareRewardWrapper(
            base_penalty=2.5,
            volatility_threshold=0.03,
            drawdown_threshold=0.08,
            adaptive_penalty=True
        )
        
        assert wrapper.base_penalty == 2.5
        assert wrapper.volatility_threshold == 0.03
        assert wrapper.drawdown_threshold == 0.08
        assert wrapper.adaptive_penalty == True
    
    def test_calculate_reward_normal_case(self):
        """测试正常情况下的奖励计算"""
        wrapper = RiskAwareRewardWrapper()
        
        reward = wrapper.calculate_reward(
            portfolio_return=0.01,    # 1%收益
            current_drawdown=0.02,    # 2%回撤（低于阈值）
            volatility=0.015,         # 1.5%波动（低于阈值）
            trade_cost=50,
            total_value=1000000
        )
        
        # 奖励应该接近基础收益减去交易成本
        expected_reward = 0.01 - 50/1000000  # 基础收益 - 成本
        assert abs(reward - expected_reward) < 0.001
    
    def test_calculate_reward_high_drawdown(self):
        """测试高回撤情况"""
        wrapper = RiskAwareRewardWrapper(base_penalty=2.0, drawdown_threshold=0.05)
        
        reward = wrapper.calculate_reward(
            portfolio_return=0.01,
            current_drawdown=0.08,    # 高回撤
            volatility=0.01,
            trade_cost=0,
            total_value=1000000
        )
        
        # 奖励应该被回撤惩罚显著降低
        expected_penalty = 2.0 * (0.08 - 0.05)  # 超出阈值的部分
        expected_reward = 0.01 - expected_penalty
        assert abs(reward - expected_reward) < 0.001
    
    def test_calculate_reward_high_volatility(self):
        """测试高波动情况"""
        wrapper = RiskAwareRewardWrapper(volatility_threshold=0.02)
        
        reward = wrapper.calculate_reward(
            portfolio_return=0.01,
            current_drawdown=0.02,   # 低回撤
            volatility=0.035,        # 高波动
            trade_cost=0,
            total_value=1000000
        )
        
        # 奖励应该被波动惩罚
        volatility_penalty = 0.5 * (0.035 - 0.02)
        expected_reward = 0.01 - volatility_penalty
        assert abs(reward - expected_reward) < 0.001
    
    def test_calculate_reward_clipping(self):
        """测试奖励裁剪"""
        wrapper = RiskAwareRewardWrapper(base_penalty=10.0)  # 很大的惩罚
        
        reward = wrapper.calculate_reward(
            portfolio_return=-0.05,   # 负收益
            current_drawdown=0.2,     # 很大回撤
            volatility=0.1,           # 很高波动
            trade_cost=1000,
            total_value=1000000
        )
        
        # 奖励应该被裁剪在[-0.1, 0.1]范围内
        assert -0.1 <= reward <= 0.1
    
    def test_adaptive_penalty(self):
        """测试自适应惩罚"""
        wrapper = RiskAwareRewardWrapper(adaptive_penalty=True, base_penalty=1.0)
        
        # 先计算几个奖励来建立历史
        for drawdown in [0.06, 0.07, 0.08, 0.09, 0.10]:
            wrapper.calculate_reward(0.01, drawdown, 0.01, 0, 1000000)
        
        # 现在计算一个高回撤的奖励
        reward = wrapper.calculate_reward(0.01, 0.12, 0.01, 0, 1000000)
        
        # 由于有历史高回撤，惩罚应该被放大
        assert reward < 0.01 - 1.0 * (0.12 - 0.05)  # 比基础惩罚更严格


class TestPortfolioMetrics:
    """组合指标测试类"""
    
    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        # 正常收益序列
        returns = np.array([0.01, -0.005, 0.015, 0.008, -0.002])
        sharpe = PortfolioMetrics.calculate_sharpe_ratio(returns)
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # 零波动率情况
        zero_vol_returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe_zero = PortfolioMetrics.calculate_sharpe_ratio(zero_vol_returns)
        assert sharpe_zero == 0.0
        
        # 空数组情况
        empty_returns = np.array([])
        sharpe_empty = PortfolioMetrics.calculate_sharpe_ratio(empty_returns)
        assert sharpe_empty == 0.0
    
    def test_calculate_max_drawdown(self):
        """测试最大回撤计算"""
        # 有回撤的净值序列
        values = np.array([100, 110, 105, 120, 115, 125])
        max_dd = PortfolioMetrics.calculate_max_drawdown(values)
        
        assert isinstance(max_dd, float)
        assert 0 <= max_dd <= 1
        assert max_dd > 0  # 应该有回撤
        
        # 单调增长情况
        increasing_values = np.array([100, 110, 120, 130])
        max_dd_increasing = PortfolioMetrics.calculate_max_drawdown(increasing_values)
        assert max_dd_increasing == 0.0
        
        # 空数组情况
        empty_values = np.array([])
        max_dd_empty = PortfolioMetrics.calculate_max_drawdown(empty_values)
        assert max_dd_empty == 0.0
    
    def test_calculate_calmar_ratio(self):
        """测试Calmar比率计算"""
        returns = np.array([0.01, 0.005, -0.008, 0.012, 0.003])
        max_drawdown = 0.05
        
        calmar = PortfolioMetrics.calculate_calmar_ratio(returns, max_drawdown)
        
        assert isinstance(calmar, float)
        assert not np.isnan(calmar)
        
        # 零回撤情况
        calmar_zero_dd = PortfolioMetrics.calculate_calmar_ratio(returns, 0.0)
        assert calmar_zero_dd == np.inf
    
    def test_calculate_sortino_ratio(self):
        """测试Sortino比率计算"""
        returns = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
        sortin = PortfolioMetrics.calculate_sortino_ratio(returns)
        
        assert isinstance(sortino, float)
        
        # 只有正收益的情况
        positive_returns = np.array([0.01, 0.02, 0.015, 0.008])
        sortino_positive = PortfolioMetrics.calculate_sortino_ratio(positive_returns)
        assert sortino_positive == np.inf
    
    def test_calculate_information_ratio(self):
        """测试信息比率计算"""
        returns = np.array([0.01, 0.005, -0.008, 0.012])
        benchmark_returns = np.array([0.008, 0.003, -0.005, 0.01])
        
        info_ratio = PortfolioMetrics.calculate_information_ratio(returns, benchmark_returns)
        
        assert isinstance(info_ratio, float)
        assert not np.isnan(info_ratio)
        
        # 长度不匹配情况
        short_benchmark = np.array([0.01, 0.02])
        info_ratio_mismatch = PortfolioMetrics.calculate_information_ratio(returns, short_benchmark)
        assert info_ratio_mismatch == 0.0
        
        # 零跟踪误差情况
        info_ratio_zero = PortfolioMetrics.calculate_information_ratio(returns, returns)
        assert info_ratio_zero == 0.0


class TestModelIntegration:
    """模型集成测试"""
    
    def test_transformer_with_realistic_data(self):
        """测试Transformer处理真实尺寸数据"""
        lookback_window = 30
        num_stocks = 20
        num_features = 6
        batch_size = 32
        
        # 计算观察空间维度
        hist_dim = lookback_window * num_stocks * num_features
        obs_dim = hist_dim + num_stocks + 3
        
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        extractor = TimeSeriesTransformer(
            observation_space=observation_space,
            lookback_window=lookback_window,
            num_stocks=num_stocks, 
            num_features=num_features,
            d_model=256,
            nhead=8,
            num_layers=4,
            features_dim=512
        )
        
        # 创建真实尺寸的输入
        obs = torch.randn(batch_size, obs_dim)
        
        with torch.no_grad():
            features = extractor(obs)
        
        assert features.shape == (batch_size, 512)
        assert not torch.isnan(features).any()
        assert torch.isfinite(features).all()
    
    def test_reward_wrapper_integration(self):
        """测试奖励包装器集成"""
        wrapper = RiskAwareRewardWrapper(adaptive_penalty=True)
        
        # 模拟一系列交易周期
        portfolio_returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        drawdowns = [0.0, 0.02, 0.01, 0.05, 0.03]
        volatilities = [0.015, 0.02, 0.018, 0.025, 0.019]
        
        rewards = []
        for i in range(len(portfolio_returns)):
            reward = wrapper.calculate_reward(
                portfolio_returns[i],
                drawdowns[i],
                volatilities[i],
                trade_cost=100,
                total_value=1000000
            )
            rewards.append(reward)
        
        # 检查奖励序列合理性
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)
        assert all(-0.1 <= r <= 0.1 for r in rewards)  # 在裁剪范围内


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])