#!/usr/bin/env python3
"""
测试Transformer集成到SAC Agent的功能
按照TDD原则，这些测试应该先失败，然后通过实现功能使其通过
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.rl_trading_system.models.transformer import TransformerConfig, TimeSeriesTransformer
from src.rl_trading_system.models.sac_agent import SACAgent, SACConfig


class TestTransformerIntegration:
    """测试Transformer与SAC Agent的集成"""
    
    def setup_method(self):
        """测试设置"""
        # Transformer配置
        self.transformer_config = TransformerConfig(
            d_model=128,
            n_heads=8,
            n_layers=4,
            d_ff=512,
            dropout=0.1,
            max_seq_len=60,
            n_features=12  # 每只股票的特征数
        )
        
        # SAC配置 - 应该使用transformer的输出维度
        self.sac_config = SACConfig(
            state_dim=128,  # 应该等于transformer的d_model
            action_dim=3,
            hidden_dim=256,
            learning_starts=10,
            batch_size=32,
            use_transformer=True,
            transformer_config=self.transformer_config
        )
        
        # 模拟观察数据 - 原始高维观察
        self.raw_observation = {
            'features': np.random.randn(60, 3, 12),  # (time, stocks, features)
            'positions': np.array([0.3, 0.4, 0.3]),
            'market_state': np.random.randn(10)
        }
        
    def test_sac_agent_should_use_transformer_for_encoding(self):
        """测试SAC Agent应该使用Transformer对观察进行编码"""
        # Red: 这个测试应该失败，因为当前SAC Agent没有集成Transformer
        
        transformer = TimeSeriesTransformer(self.transformer_config)
        sac_agent = SACAgent(self.sac_config)
        
        # SAC Agent应该有一个transformer属性或方法
        assert hasattr(sac_agent, 'transformer'), "SAC Agent应该集成Transformer"
        
        # 或者SAC Agent应该有encode方法来处理高维观察
        assert hasattr(sac_agent, 'encode_observation'), "SAC Agent应该有观察编码方法"
        
    def test_sac_agent_should_handle_high_dim_observations(self):
        """测试SAC Agent应该能处理高维时序观察"""
        # Red: 这个测试应该失败
        
        transformer = TimeSeriesTransformer(self.transformer_config)
        sac_agent = SACAgent(self.sac_config)
        
        # 高维观察应该被编码为低维表示
        encoded_obs = sac_agent.encode_observation(self.raw_observation)
        
        # 编码后的观察应该是1D张量，维度为d_model
        assert isinstance(encoded_obs, torch.Tensor), "编码后的观察应该是torch.Tensor"
        assert encoded_obs.dim() == 1, "编码后的观察应该是1维张量"
        assert encoded_obs.size(0) == self.transformer_config.d_model, \
            f"编码后维度应该是{self.transformer_config.d_model}"
            
    def test_sac_agent_get_action_with_transformer_encoding(self):
        """测试SAC Agent应该使用Transformer编码后的观察来选择动作"""
        # Red: 这个测试应该失败
        
        transformer = TimeSeriesTransformer(self.transformer_config)
        sac_agent = SACAgent(self.sac_config)
        
        # get_action方法应该自动进行Transformer编码
        action, log_prob = sac_agent.get_action(self.raw_observation, return_log_prob=True)
        
        # 确保动作是有效的投资组合权重
        assert isinstance(action, torch.Tensor), "动作应该是torch.Tensor"
        assert action.size(0) == self.sac_config.action_dim, "动作维度应该正确"
        assert torch.allclose(action.sum(), torch.tensor(1.0), atol=1e-6), "投资组合权重应该和为1"
        assert (action >= 0).all(), "投资组合权重应该非负"
        
    def test_transformer_encoding_process(self):
        """测试Transformer编码过程的正确性"""
        # Red: 这个测试应该失败，因为需要实现正确的编码流程
        
        transformer = TimeSeriesTransformer(self.transformer_config)
        
        # 从字典观察中提取特征并重塑为Transformer输入格式
        features = self.raw_observation['features']  # (60, 3, 12)
        
        # 应该重塑为 (60, 36) - 时间步长 x 展平的特征
        transformer_input = features.reshape(60, -1)  # (60, 36)
        
        # 但是transformer期望的输入应该是 (60, 37) 包括所有特征
        # 这里需要正确处理观察到transformer输入的转换
        
        # 这个测试会失败，因为我们需要实现正确的观察处理逻辑
        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            encoded = transformer(torch.FloatTensor(transformer_input))
            
    def test_end_to_end_observation_flow(self):
        """测试端到端的观察处理流程"""
        # Red: 这个测试应该失败，因为完整流程还未实现
        
        transformer = TimeSeriesTransformer(self.transformer_config)
        sac_agent = SACAgent(self.sac_config)
        
        # 完整流程：原始观察 -> 预处理 -> Transformer编码 -> SAC决策
        
        # 1. 预处理观察
        processed_obs = sac_agent.preprocess_observation(self.raw_observation)
        
        # 2. Transformer编码
        encoded_obs = transformer(processed_obs)
        
        # 编码后的观察需要进行降维处理 - 对股票维度平均
        if encoded_obs.dim() == 3:
            encoded_obs = encoded_obs.mean(dim=1)  # [batch_size, d_model]
        
        # 移除批次维度（测试中只有一个样本）
        if encoded_obs.size(0) == 1:
            encoded_obs = encoded_obs.squeeze(0)  # [d_model]
        
        # 3. SAC决策
        action, log_prob = sac_agent.get_action_from_encoded(encoded_obs)
        
        # 验证最终输出
        assert isinstance(action, torch.Tensor)
        assert action.size(0) == self.sac_config.action_dim
        assert torch.allclose(action.sum(), torch.tensor(1.0), atol=1e-6)
        
    def test_transformer_output_dimension_matches_sac_input(self):
        """测试Transformer输出维度与SAC输入维度匹配"""
        # Red: 这个测试应该失败，因为需要确保维度匹配
        
        transformer = TimeSeriesTransformer(self.transformer_config)
        
        # 创建正确格式的输入: [batch_size, seq_len, n_stocks, n_features_per_stock]
        batch_size = 1
        seq_len = self.transformer_config.max_seq_len
        n_stocks = 3
        n_features_per_stock = 12
        
        dummy_input = torch.randn(batch_size, seq_len, n_stocks, n_features_per_stock)
        
        # Transformer输出应该是 (batch_size, n_stocks, d_model)
        output = transformer(dummy_input)
        
        expected_output_shape = (batch_size, n_stocks, self.transformer_config.d_model)
        assert output.size() == expected_output_shape, \
            f"Transformer输出维度应该是 {expected_output_shape}"
            
        # 聚合后的输出应该能传递给SAC的Actor网络
        aggregated_output = output.mean(dim=1)  # 对股票维度进行平均
        assert aggregated_output.size() == (batch_size, self.transformer_config.d_model), \
            f"聚合后输出维度应该是 ({batch_size}, {self.transformer_config.d_model})"
            
        assert aggregated_output.size(1) == self.sac_config.state_dim, \
            "聚合后的Transformer输出维度应该匹配SAC的state_dim"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])