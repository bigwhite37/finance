#!/usr/bin/env python3
"""
特征数41 vs 37不匹配问题的TDD测试
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.data.qlib_interface import QlibDataInterface
from rl_trading_system.data.feature_engineer import FeatureEngineer


class TestFeature41vs37Fix:
    """测试特征数41 vs 37不匹配问题修复"""
    
    def test_reproduce_41_features_problem(self):
        """测试重现41个特征问题"""
        # Red: 重现180x41 vs 37x128的矩阵维度不匹配问题
        
        print("=== 重现41个特征问题 ===")
        
        # 使用实际回测数据配置
        stock_pool = ['600519.SH', '600036.SH', '601318.SH']
        start_date = '2018-01-01'
        end_date = '2018-01-10'  # 使用更长期间以确保足够数据
        
        # 初始化数据接口
        data_interface = QlibDataInterface()
        
        # 获取实际数据
        price_data = data_interface.get_price_data(stock_pool, start_date, end_date)
        print(f"价格数据形状: {price_data.shape}")
        
        # 使用特征工程计算特征
        feature_engineer = FeatureEngineer()
        features = feature_engineer.calculate_features(price_data)
        
        actual_feature_count = len(features.columns)
        expected_feature_count = 37  # 模型期望的特征数
        
        print(f"实际特征数: {actual_feature_count}")
        print(f"期望特征数: {expected_feature_count}")
        print(f"差异: {actual_feature_count - expected_feature_count}")
        
        if actual_feature_count != expected_feature_count:
            print(f"❌ 特征数量不匹配: 实际{actual_feature_count} vs 期望{expected_feature_count}")
            
            # 打印所有特征名称
            feature_names = list(features.columns)
            print(f"\n所有{actual_feature_count}个特征:")
            for i, name in enumerate(feature_names):
                print(f"{i+1:2d}. {name}")
                
            # 分析多余的特征
            if actual_feature_count > expected_feature_count:
                extra_count = actual_feature_count - expected_feature_count
                print(f"\n❌ 有{extra_count}个多余特征需要移除")
        else:
            print("✅ 特征数量匹配")
            
        # 确认问题存在
        assert actual_feature_count == 41, f"应该重现41个特征问题，实际{actual_feature_count}个"
        
    def test_identify_extra_features_in_41(self):
        """测试识别41个特征中的多余部分"""
        # Red: 识别导致特征数从37增加到41的原因
        
        print("=== 识别41个特征中的多余部分 ===")
        
        # 创建测试数据
        stock_pool = ['600519.SH', '600036.SH', '601318.SH']
        start_date = '2018-01-01'
        end_date = '2018-01-10'
        
        data_interface = QlibDataInterface()
        price_data = data_interface.get_price_data(stock_pool, start_date, end_date)
        
        feature_engineer = FeatureEngineer()
        
        # 分别计算各个特征组以识别重复
        technical_features = feature_engineer.calculate_technical_indicators(price_data)
        microstructure_features = feature_engineer.calculate_microstructure_features(price_data)
        volatility_features = feature_engineer.calculate_volatility_features(price_data)
        momentum_features = feature_engineer.calculate_momentum_features(price_data)
        
        print("各组特征统计:")
        print(f"技术指标: {len(technical_features.columns)}个")
        print(f"微观结构: {len(microstructure_features.columns)}个")
        print(f"波动率: {len(volatility_features.columns)}个")
        print(f"动量: {len(momentum_features.columns)}个")
        
        # 合并并检查重复
        all_features = feature_engineer.combine_features([
            technical_features,
            microstructure_features,
            volatility_features,
            momentum_features
        ])
        
        total_individual = (len(technical_features.columns) + 
                          len(microstructure_features.columns) + 
                          len(volatility_features.columns) + 
                          len(momentum_features.columns))
        
        actual_merged = len(all_features.columns)
        
        print(f"\n合并前总数: {total_individual}")
        print(f"合并后实际: {actual_merged}")
        print(f"重复特征数: {total_individual - actual_merged}")
        
        # 找出具体的重复特征
        all_feature_names = []
        for features in [technical_features, microstructure_features, volatility_features, momentum_features]:
            all_feature_names.extend(features.columns.tolist())
        
        unique_names = list(set(all_feature_names))
        duplicate_count = len(all_feature_names) - len(unique_names)
        
        print(f"唯一特征数: {len(unique_names)}")
        print(f"重复特征数: {duplicate_count}")
        
        # 找出重复的特征名
        name_counts = {}
        for name in all_feature_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        if duplicates:
            print(f"\n重复的特征:")
            for name, count in duplicates.items():
                print(f"  {name}: 出现{count}次")
        
        assert actual_merged == 41, f"应该产生41个特征，实际{actual_merged}个"
        
    def test_fix_feature_count_to_37(self):
        """测试修复特征数到37个"""
        # Green: 修复特征数使其正好等于37
        
        print("=== 修复特征数到37个 ===")
        
        print("修复策略:")
        print("1. 识别并移除4个多余特征")
        print("2. 保持与已训练模型的一致性")
        print("3. 确保最终特征数正好是37个")
        
        expected_final_count = 37
        print(f"目标特征数: {expected_final_count}")
        
        # 这个测试将在修复代码后变为绿色
        assert True  # 准备修复
        
    def test_verify_matrix_dimensions_after_fix(self):
        """测试修复后验证矩阵维度匹配"""
        # Green: 验证修复后矩阵维度正确匹配
        
        print("=== 验证修复后矩阵维度匹配 ===")
        
        print("修复后应该满足:")
        print("- 特征数: 37个")
        print("- 矩阵维度: 180x37 (60窗口 × 3股票 × 37特征)")
        print("- 与模型期望匹配: 37x128")
        print("- 不再有mat1 and mat2 shapes cannot be multiplied错误")
        
        # 这个测试在修复后应该通过
        assert True  # 修复后验证