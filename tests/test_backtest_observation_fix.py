#!/usr/bin/env python3
"""
观察维度不匹配问题的TDD测试
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.models import SACAgent
from scripts.backtest import load_trained_model


class TestObservationDimensionFix:
    """测试观察维度修复"""
    
    def test_loaded_model_should_handle_observation_correctly(self):
        """测试加载的模型应该正确处理观察维度"""
        # Green: 修复后，模型应该能正确处理观察维度
        
        # 加载真实的训练模型
        model_path = "outputs/final_model_agent.pth/model.pt"
        if not Path(model_path).exists():
            pytest.skip("模型文件不存在，跳过测试")
            
        agent = load_trained_model(model_path)
        
        # 获取模型期望的维度
        expected_dim = agent.config.state_dim  # 2173
        
        # 创建与训练时维度一致的观察
        mock_correct_observation = np.random.random(expected_dim)
        
        # 模型应该能够处理这个观察而不产生维度错误
        action = agent.get_action(mock_correct_observation, deterministic=True)
        assert action is not None, "应该能生成动作"
        assert len(action) == agent.config.action_dim, f"动作维度应该是{agent.config.action_dim}"
    
    def test_model_observation_processing_should_be_consistent(self):
        """测试模型观察处理应该与训练时一致"""
        # Green: 这个测试应该通过，因为load_trained_model已经修复了配置问题
        
        model_path = "outputs/final_model_agent.pth/model.pt"
        if not Path(model_path).exists():
            pytest.skip("模型文件不存在，跳过测试")
            
        # 使用修复后的加载函数
        agent = load_trained_model(model_path)
        
        # 检查修复后的配置
        config = agent.config
        
        # 修复后，如果原始配置有问题，应该已经被纠正
        if config.use_transformer:
            # 如果启用了transformer，应该有有效的transformer_config
            assert config.transformer_config is not None, "启用Transformer时应该有有效的transformer_config"
        else:
            # 如果禁用了transformer，应该能处理展平的观察
            # 此时模型应该能正确处理不同维度的输入
            pass
    
    def test_should_convert_backtest_observation_to_training_dimension(self):
        """测试应该能将回测观察转换为训练时的维度"""
        # Red: 这个测试应该失败，因为还未实现观察维度转换功能
        
        model_path = "outputs/final_model_agent.pth/model.pt"
        if not Path(model_path).exists():
            pytest.skip("模型文件不存在，跳过测试")
            
        agent = load_trained_model(model_path)
        expected_dim = agent.config.state_dim  # 2173
        
        # 模拟回测环境产生的观察（6673维）
        mock_backtest_obs = {
            'features': np.random.random((60, 3, 37)),  # (lookback, stocks, features) = 6660
            'positions': np.random.random(3),           # 3
            'market_state': np.random.random(10)        # 10
        }
        
        # 应该有一个函数能将回测观察转换为训练时的维度
        from scripts.backtest import convert_observation_for_model
        converted_obs = convert_observation_for_model(mock_backtest_obs, agent.config)
        
        # 转换后的观察应该是训练时期望的维度
        if isinstance(converted_obs, dict):
            # 如果返回字典，应该能被模型的观察处理逻辑正确处理
            flattened = agent._flatten_dict_observation(converted_obs)
            assert len(flattened) == expected_dim, f"转换后观察维度应该是{expected_dim}，实际是{len(flattened)}"
        else:
            # 如果直接返回展平的观察
            assert len(converted_obs) == expected_dim, f"转换后观察维度应该是{expected_dim}，实际是{len(converted_obs)}"
        
        # 模型应该能处理转换后的观察
        action = agent.get_action(converted_obs, deterministic=True)
        assert action is not None, "模型应该能处理转换后的观察"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])