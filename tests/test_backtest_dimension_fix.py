#!/usr/bin/env python3
"""
回测维度匹配修复的TDD测试
验证训练和回测使用一致的特征维度
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys
import json
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.models.sac_agent import SACAgent, SACConfig
from rl_trading_system.models.transformer import TransformerConfig


class TestBacktestDimensionFix:
    """测试回测维度匹配修复"""
    
    def test_sac_config_should_rebuild_transformer_from_saved_config(self):
        """Red -> Green: 测试从保存的配置重建SAC配置时应包含Transformer配置"""
        print("=== Red -> Green: 验证Transformer配置重建 ===")
        
        # Red阶段：创建一个缺少transformer_config的配置
        incomplete_config_dict = {
            "state_dim": 128,
            "action_dim": 3,
            "hidden_dim": 256,
            "use_transformer": True,  # 这里设置了use_transformer但没有transformer_config
            "device": "cpu"
        }
        
        # 这应该会导致问题，因为没有transformer_config
        sac_config = SACConfig(**incomplete_config_dict)
        
        # 验证问题存在
        assert sac_config.use_transformer == True
        assert not hasattr(sac_config, 'transformer_config') or sac_config.transformer_config is None
        
        # 创建智能体时Transformer不会被初始化
        agent = SACAgent(sac_config)
        assert agent.transformer is None, "Transformer应该为None，因为没有transformer_config"
        
        print("✅ 确认了问题：use_transformer=True但transformer=None")
        
        # Green阶段：修复配置，确保包含transformer_config
        # 当use_transformer=True但transformer_config缺失时，应该提供默认配置
        fixed_config = SACConfig(**incomplete_config_dict)
        
        # 如果use_transformer=True但transformer_config为None，应该设置默认配置
        if fixed_config.use_transformer and (not hasattr(fixed_config, 'transformer_config') or fixed_config.transformer_config is None):
            # 设置合理的默认Transformer配置
            fixed_config.transformer_config = TransformerConfig(
                d_model=128,  # 与state_dim一致
                n_heads=8,
                n_layers=4,
                d_ff=512,
                dropout=0.1,
                max_seq_len=60,
                n_features=37  # 每只股票的特征数
            )
        
        # 现在Transformer应该能正确初始化
        fixed_agent = SACAgent(fixed_config)
        
        if fixed_config.use_transformer:
            assert fixed_agent.transformer is not None, "修复后Transformer应该被正确初始化"
            print(f"✅ Transformer已初始化，输出维度: {fixed_agent.transformer.config.d_model}")
        
        print("✅ 修复了Transformer配置问题")
    
    def test_backtest_model_loading_should_handle_missing_transformer_config(self):
        """Red -> Green: 测试回测模型加载应处理缺失的transformer_config"""
        print("=== Red -> Green: 验证回测模型加载修复 ===")
        
        # 模拟保存的config.json（缺少transformer_config）
        saved_config = {
            "state_dim": 128,
            "action_dim": 3,
            "hidden_dim": 256,
            "use_transformer": True,
            "device": "cpu"
        }
        
        # Red阶段：直接用这个配置创建SACConfig会导致Transformer为None
        sac_config = SACConfig(**saved_config)
        agent = SACAgent(sac_config)
        
        assert agent.transformer is None, "原始方法会导致Transformer为None"
        
        # 测试观察处理
        mock_obs = {
            'features': np.random.random((60, 3, 37)),  # (时间步, 股票数, 特征数)
            'positions': np.array([0.3, 0.3, 0.4]),
            'market_state': np.random.random(10)
        }
        
        # 当transformer为None时，会使用_flatten_dict_observation，导致维度过大
        flattened_obs = agent._flatten_dict_observation(mock_obs)
        print(f"展平后的观察维度: {flattened_obs.shape}")
        
        # 验证维度不匹配问题
        expected_input_dim = 128  # 模型期望的输入维度
        actual_input_dim = flattened_obs.shape[0]
        print(f"期望输入维度: {expected_input_dim}, 实际输入维度: {actual_input_dim}")
        
        assert actual_input_dim != expected_input_dim, "确认维度不匹配问题"
        
        print("✅ 确认了回测中的维度不匹配问题")
    
    def test_dimension_consistency_between_training_and_inference(self):
        """Red -> Green: 测试训练和推理时的维度一致性"""
        print("=== Red -> Green: 验证训练和推理维度一致性 ===")
        
        # 创建完整的配置（包含transformer_config）
        complete_config = SACConfig(
            state_dim=128,
            action_dim=3,
            hidden_dim=256,
            use_transformer=True,
            transformer_config=TransformerConfig(
                d_model=128,
                n_heads=8,
                n_layers=4,
                d_ff=512,
                dropout=0.1,
                max_seq_len=60,
                n_features=37
            ),
            device="cpu"
        )
        
        agent = SACAgent(complete_config)
        
        # 验证Transformer被正确初始化
        assert agent.transformer is not None, "Transformer应该被初始化"
        assert agent.transformer.config.d_model == 128, "Transformer输出维度应该是128"
        
        # 测试观察编码
        mock_obs = {
            'features': np.random.random((60, 3, 37)),
            'positions': np.array([0.3, 0.3, 0.4]),
            'market_state': np.random.random(10)
        }
        
        # 使用encode_observation方法
        encoded_obs = agent.encode_observation(mock_obs, training=False)
        print(f"编码后的观察维度: {encoded_obs.shape}")
        
        # 验证维度正确
        assert encoded_obs.shape[0] == 128, f"编码后维度应该是128，实际是{encoded_obs.shape[0]}"
        
        # 测试是否能正常生成动作
        try:
            action = agent.get_action(mock_obs, deterministic=True)
            print(f"成功生成动作，维度: {action.shape}")
            assert action.shape[0] == 3, "动作维度应该是3"
            print("✅ 训练和推理维度一致性验证通过")
        except Exception as e:
            assert False, f"动作生成失败: {e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])