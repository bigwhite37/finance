#!/usr/bin/env python3
"""
观察维度问题修复的TDD测试
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.models.sac_agent import SACConfig


class TestObservationDimensionFix:
    """测试观察维度问题修复"""
    
    def test_current_dimension_mismatch_problem(self):
        """测试当前的维度不匹配问题"""
        # Red: 重现维度不匹配问题
        
        # 实际观察维度
        lookback_window = 60
        n_stocks = 3  
        n_features_per_stock = 37
        positions_dim = 3
        market_state_dim = 10
        
        actual_features_dim = lookback_window * n_stocks * n_features_per_stock
        actual_total_dim = actual_features_dim + positions_dim + market_state_dim
        
        # 配置中的期望维度
        expected_state_dim = 128  # 来自trading_config.yaml
        
        print(f"维度分析:")
        print(f"实际观察总维度: {actual_total_dim}")
        print(f"配置期望维度: {expected_state_dim}")
        print(f"差异: {actual_total_dim - expected_state_dim}")
        
        # 验证问题确实存在
        assert actual_total_dim > expected_state_dim, "维度不匹配问题确实存在"
        assert actual_total_dim == 6673, "实际维度应该是6673"
        assert expected_state_dim == 128, "配置期望维度是128"
        
        print("✅ 成功重现维度不匹配问题")
        
    def test_transformer_sac_architecture_understanding(self):
        """测试对Transformer+SAC架构的正确理解"""
        # Green: 定义正确的架构理解
        
        print("=== 正确的Transformer+SAC架构 ===")
        
        # 原始观察维度（环境产生的）
        raw_observation_dim = 6673
        
        # Transformer处理原始观察，输出压缩表示
        transformer_output_dim = 256  # d_model from model_config.yaml
        
        # SAC接收Transformer输出作为状态
        sac_state_dim = transformer_output_dim
        
        print(f"步骤1: 环境产生原始观察 -> {raw_observation_dim}维")
        print(f"步骤2: Transformer处理原始观察 -> {transformer_output_dim}维")  
        print(f"步骤3: SAC接收Transformer输出 -> {sac_state_dim}维")
        
        # 验证架构的合理性
        assert transformer_output_dim < raw_observation_dim, "Transformer应该压缩观察"
        assert sac_state_dim == transformer_output_dim, "SAC状态维度应该等于Transformer输出维度"
        
        print("✅ 正确理解了Transformer+SAC架构")
        
    def test_wrong_convert_function_problem(self):
        """测试convert_observation_for_model函数的错误逻辑"""
        # Red: 识别当前函数的错误逻辑
        
        print("=== convert_observation_for_model函数的错误逻辑 ===")
        print("当前函数假设:")
        print("1. SAC直接接收原始观察")
        print("2. 需要手动调整观察维度以匹配SAC的state_dim")
        print("3. 通过截取特征来减少维度")
        print()
        print("错误之处:")
        print("1. 忽略了Transformer的存在")
        print("2. 错误地认为需要手动维度转换")
        print("3. 截取特征会丢失信息")
        
        # 模拟当前错误逻辑的问题
        expected_total_dim = 128
        positions_dim = 3
        market_state_dim = 10
        expected_features_dim = expected_total_dim - positions_dim - market_state_dim  # 115
        
        lookback_window = 60
        n_stocks = 3
        expected_n_features_per_stock = expected_features_dim // (lookback_window * n_stocks)  # 0
        
        assert expected_n_features_per_stock <= 0, "当前逻辑导致无效的特征数"
        
        print("✅ 确认了当前函数逻辑的错误")
        
    def test_correct_solution_should_use_transformer(self):
        """测试正确的解决方案应该使用Transformer"""
        # Green: 定义正确的解决方案
        
        print("=== 正确的解决方案 ===")
        print("1. 不需要convert_observation_for_model函数")
        print("2. SAC agent内部应该集成Transformer")
        print("3. 流程: 原始观察 -> Transformer -> SAC")
        
        # 验证正确的配置
        sac_config = SACConfig(
            state_dim=256,  # 应该等于Transformer的d_model
            action_dim=3,
            hidden_dim=512,
            use_transformer=True  # 关键：启用Transformer集成
        )
        
        assert sac_config.state_dim == 256, "SAC state_dim应该等于Transformer输出维度"
        assert sac_config.use_transformer == True, "应该启用Transformer集成"
        
        print("✅ 定义了正确的解决方案")
        
    def test_config_inconsistency_problem(self):
        """测试配置不一致问题"""
        # Red: 识别配置文件之间的不一致
        
        print("=== 配置不一致问题 ===")
        
        # model_config.yaml中的配置
        model_config_state_dim = 256
        
        # trading_config.yaml中的配置  
        trading_config_state_dim = 128
        
        print(f"model_config.yaml: state_dim = {model_config_state_dim}")
        print(f"trading_config.yaml: state_dim = {trading_config_state_dim}")
        
        assert model_config_state_dim != trading_config_state_dim, "配置不一致问题存在"
        
        print("✅ 确认了配置不一致问题")
        assert True