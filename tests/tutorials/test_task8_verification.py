#!/usr/bin/env python3
"""
任务8验证脚本：验证DynamicLowVolFilter主控制器实现

验证以下功能：
1. 主控制器类的基本结构
2. 核心方法的存在和签名
3. 配置处理和组件初始化
4. 基本的状态管理功能
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_main_controller_structure():
    """测试主控制器类结构"""
    print("=" * 60)
    print("测试1: 主控制器类结构")
    print("=" * 60)
    
    try:
        from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter, DynamicLowVolConfig
        
        # 验证类存在
        print("✓ DynamicLowVolFilter类导入成功")
        
        # 验证核心方法存在
        required_methods = [
            'update_tradable_mask',
            'get_current_regime', 
            'get_adaptive_target_volatility',
            'get_filter_statistics'
        ]
        
        for method in required_methods:
            if hasattr(DynamicLowVolFilter, method):
                print(f"✓ 方法 {method} 存在")
            else:
                print(f"✗ 方法 {method} 缺失")
                return False
        
        # 验证辅助方法存在
        helper_methods = [
            'get_current_tradable_stocks',
            'get_regime_transition_probability',
            'reset_statistics'
        ]
        
        for method in helper_methods:
            if hasattr(DynamicLowVolFilter, method):
                print(f"✓ 辅助方法 {method} 存在")
            else:
                print(f"✗ 辅助方法 {method} 缺失")
        
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_configuration_handling():
    """测试配置处理"""
    print("\n" + "=" * 60)
    print("测试2: 配置处理")
    print("=" * 60)
    
    try:
        from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter, DynamicLowVolConfig
        
        # 测试基本配置
        basic_config = {
            'rolling_windows': [20, 60],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            'enable_caching': False  # 禁用缓存避免依赖问题
        }
        
        mock_data_manager = Mock()
        
        # 模拟所有组件以避免依赖问题
        with patch('risk_control.dynamic_lowvol_filter.DataPreprocessor') as mock_preprocessor, \
             patch('risk_control.dynamic_lowvol_filter.RollingPercentileFilter') as mock_rolling, \
             patch('risk_control.dynamic_lowvol_filter.GARCHVolatilityPredictor') as mock_garch, \
             patch('risk_control.dynamic_lowvol_filter.IVOLConstraintFilter') as mock_ivol, \
             patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector') as mock_detector, \
             patch('risk_control.dynamic_lowvol_filter.RegimeAwareThresholdAdjuster') as mock_adjuster:
            
            # 创建筛选器实例
            filter_instance = DynamicLowVolFilter(basic_config, mock_data_manager)
            
            print("✓ 基本配置处理成功")
            
            # 验证配置对象创建
            assert isinstance(filter_instance.config, DynamicLowVolConfig)
            print("✓ 配置对象创建成功")
            
            # 验证数据管理器设置
            assert filter_instance.data_manager == mock_data_manager
            print("✓ 数据管理器设置成功")
            
            # 验证组件初始化调用
            mock_preprocessor.assert_called_once()
            mock_rolling.assert_called_once()
            mock_garch.assert_called_once()
            mock_ivol.assert_called_once()
            mock_detector.assert_called_once()
            mock_adjuster.assert_called_once()
            print("✓ 所有组件初始化调用成功")
            
            return True
            
    except Exception as e:
        print(f"✗ 配置处理失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试3: 基本功能")
    print("=" * 60)
    
    try:
        from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter
        
        config = {
            'rolling_windows': [20],
            'enable_caching': False
        }
        
        mock_data_manager = Mock()
        
        # 模拟所有组件
        with patch('risk_control.dynamic_lowvol_filter.DataPreprocessor'), \
             patch('risk_control.dynamic_lowvol_filter.RollingPercentileFilter'), \
             patch('risk_control.dynamic_lowvol_filter.GARCHVolatilityPredictor'), \
             patch('risk_control.dynamic_lowvol_filter.IVOLConstraintFilter'), \
             patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector'), \
             patch('risk_control.dynamic_lowvol_filter.RegimeAwareThresholdAdjuster'):
            
            filter_instance = DynamicLowVolFilter(config, mock_data_manager)
            
            # 测试get_current_regime
            regime = filter_instance.get_current_regime()
            assert regime in ["低", "中", "高"], f"无效的市场状态: {regime}"
            print(f"✓ get_current_regime返回: {regime}")
            
            # 测试get_adaptive_target_volatility
            target_vol = filter_instance.get_adaptive_target_volatility()
            assert isinstance(target_vol, float), "目标波动率应为浮点数"
            assert 0.25 <= target_vol <= 0.60, f"目标波动率{target_vol}超出合理范围"
            print(f"✓ get_adaptive_target_volatility返回: {target_vol:.3f}")
            
            # 测试get_filter_statistics
            stats = filter_instance.get_filter_statistics()
            assert isinstance(stats, dict), "统计信息应为字典类型"
            
            required_keys = ['current_state', 'total_updates', 'regime_history', 
                           'filter_pass_rates', 'performance_metrics']
            for key in required_keys:
                assert key in stats, f"统计信息缺少键: {key}"
            
            print("✓ get_filter_statistics返回完整统计信息")
            
            # 测试get_current_tradable_stocks
            tradable_stocks = filter_instance.get_current_tradable_stocks()
            assert isinstance(tradable_stocks, list), "可交易股票列表应为列表类型"
            print(f"✓ get_current_tradable_stocks返回: {len(tradable_stocks)}只股票")
            
            # 测试reset_statistics
            filter_instance.reset_statistics()
            stats_after_reset = filter_instance.get_filter_statistics()
            assert stats_after_reset['total_updates'] == 0, "重置后更新次数应为0"
            print("✓ reset_statistics功能正常")
            
            return True
            
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_management():
    """测试状态管理"""
    print("\n" + "=" * 60)
    print("测试4: 状态管理")
    print("=" * 60)
    
    try:
        from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter
        
        config = {'enable_caching': False}
        mock_data_manager = Mock()
        
        with patch('risk_control.dynamic_lowvol_filter.DataPreprocessor'), \
             patch('risk_control.dynamic_lowvol_filter.RollingPercentileFilter'), \
             patch('risk_control.dynamic_lowvol_filter.GARCHVolatilityPredictor'), \
             patch('risk_control.dynamic_lowvol_filter.IVOLConstraintFilter'), \
             patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector'), \
             patch('risk_control.dynamic_lowvol_filter.RegimeAwareThresholdAdjuster'):
            
            filter_instance = DynamicLowVolFilter(config, mock_data_manager)
            
            # 测试初始状态
            assert filter_instance._current_regime == "中", "初始状态应为中等波动"
            assert filter_instance._current_regime_confidence == 0.5, "初始置信度应为0.5"
            assert filter_instance._current_market_volatility == 0.3, "初始市场波动率应为0.3"
            print("✓ 初始状态设置正确")
            
            # 测试状态更新
            filter_instance._current_regime = "高"
            filter_instance._current_regime_confidence = 0.8
            filter_instance._current_market_volatility = 0.5
            
            assert filter_instance.get_current_regime() == "高", "状态更新失败"
            
            # 测试自适应目标波动率随状态变化
            high_vol_target = filter_instance.get_adaptive_target_volatility()
            
            filter_instance._current_regime = "低"
            low_vol_target = filter_instance.get_adaptive_target_volatility()
            
            # 低波动状态的目标波动率应该更高
            assert low_vol_target > high_vol_target, "低波动状态目标波动率应更高"
            print(f"✓ 自适应目标波动率: 高波动状态={high_vol_target:.3f}, 低波动状态={low_vol_target:.3f}")
            
            # 测试状态转换概率
            filter_instance._filter_statistics['regime_history'] = ["高", "高", "中", "低", "中"]
            filter_instance._current_regime = "高"
            
            transition_prob = filter_instance.get_regime_transition_probability()
            assert isinstance(transition_prob, dict), "状态转换概率应为字典"
            assert abs(sum(transition_prob.values()) - 1.0) < 0.01, "概率和应接近1"
            print("✓ 状态转换概率计算正确")
            
            return True
            
    except Exception as e:
        print(f"✗ 状态管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试5: 错误处理")
    print("=" * 60)
    
    try:
        from risk_control.dynamic_lowvol_filter import (
            DynamicLowVolFilter, 
            ConfigurationException, 
            DataQualityException
        )
        
        # 测试无效配置
        try:
            DynamicLowVolFilter("invalid_config", Mock())
            print("✗ 应该抛出ConfigurationException")
            return False
        except ConfigurationException:
            print("✓ 无效配置正确抛出ConfigurationException")
        
        # 测试空数据管理器
        try:
            DynamicLowVolFilter({}, None)
            print("✗ 应该抛出DataQualityException")
            return False
        except DataQualityException:
            print("✓ 空数据管理器正确抛出DataQualityException")
        
        return True
        
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("动态低波筛选器主控制器实现验证")
    print("=" * 60)
    
    tests = [
        test_main_controller_structure,
        test_configuration_handling,
        test_basic_functionality,
        test_state_management,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("测试失败")
        except Exception as e:
            print(f"测试异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("✓ 所有测试通过！主控制器实现正确。")
        return True
    else:
        print("✗ 部分测试失败，需要修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)