#!/usr/bin/env python3
"""
验证任务1的实现：核心异常类和配置数据结构
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

# 导入实现的组件
from risk_control.dynamic_lowvol_filter import (
    FilterException,
    DataQualityException, 
    ModelFittingException,
    RegimeDetectionException,
    InsufficientDataException,
    ConfigurationException,
    DynamicLowVolConfig,
    FilterInputData,
    FilterOutputData,
    DynamicLowVolFilter
)

def test_exception_classes():
    """测试异常类的继承关系"""
    print("测试异常类...")
    
    # 测试基础异常
    try:
        raise FilterException("基础异常测试")
    except FilterException as e:
        print(f"✓ FilterException: {e}")
    
    # 测试派生异常
    exceptions_to_test = [
        (DataQualityException, "数据质量异常"),
        (ModelFittingException, "模型拟合异常"),
        (RegimeDetectionException, "状态检测异常"),
        (InsufficientDataException, "数据不足异常"),
        (ConfigurationException, "配置异常")
    ]
    
    for exc_class, message in exceptions_to_test:
        try:
            raise exc_class(message)
        except FilterException as e:
            print(f"✓ {exc_class.__name__}: {e}")
        except Exception as e:
            print(f"✗ {exc_class.__name__} 不是FilterException的子类")
            return False
    
    return True

def test_config_data_structure():
    """测试配置数据结构"""
    print("\n测试配置数据结构...")
    
    # 测试默认配置
    try:
        config = DynamicLowVolConfig()
        print("✓ 默认配置创建成功")
        print(f"  - 滚动窗口: {config.rolling_windows}")
        print(f"  - 分位数阈值: {config.percentile_thresholds}")
        print(f"  - GARCH窗口: {config.garch_window}")
        print(f"  - 预测期限: {config.forecast_horizon}")
    except Exception as e:
        print(f"✗ 默认配置创建失败: {e}")
        return False
    
    # 测试自定义配置
    try:
        custom_config = DynamicLowVolConfig(
            rolling_windows=[30, 90],
            garch_window=300,
            forecast_horizon=3
        )
        print("✓ 自定义配置创建成功")
    except Exception as e:
        print(f"✗ 自定义配置创建失败: {e}")
        return False
    
    # 测试配置验证 - 无效滚动窗口
    try:
        invalid_config = DynamicLowVolConfig(rolling_windows=[-1, 0])
        print("✗ 应该抛出配置异常但没有")
        return False
    except ConfigurationException as e:
        print(f"✓ 无效滚动窗口正确抛出异常: {e}")
    
    # 测试配置验证 - 无效分位数阈值
    try:
        invalid_config = DynamicLowVolConfig(
            percentile_thresholds={"低": 1.5, "中": 0.3, "高": 0.2}
        )
        print("✗ 应该抛出配置异常但没有")
        return False
    except ConfigurationException as e:
        print(f"✓ 无效分位数阈值正确抛出异常: {e}")
    
    return True

def test_input_output_data_structures():
    """测试输入输出数据结构"""
    print("\n测试输入输出数据结构...")
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    stocks = ['000001', '000002', '000003']
    
    price_data = pd.DataFrame(
        np.random.randn(100, 3) * 0.02 + 100,
        index=dates,
        columns=stocks
    )
    
    volume_data = pd.DataFrame(
        np.random.randint(1000, 10000, (100, 3)),
        index=dates,
        columns=stocks
    )
    
    factor_data = pd.DataFrame(
        np.random.randn(100, 5),
        index=dates,
        columns=['factor1', 'factor2', 'factor3', 'factor4', 'factor5']
    )
    
    market_data = pd.DataFrame(
        np.random.randn(100, 1) * 0.015 + 3000,
        index=dates,
        columns=['market_index']
    )
    
    # 测试FilterInputData
    try:
        input_data = FilterInputData(
            price_data=price_data,
            volume_data=volume_data,
            factor_data=factor_data,
            market_data=market_data,
            current_date=pd.Timestamp('2023-04-10')
        )
        print("✓ FilterInputData创建成功")
        print(f"  - 价格数据形状: {input_data.price_data.shape}")
        print(f"  - 当前日期: {input_data.current_date}")
    except Exception as e:
        print(f"✗ FilterInputData创建失败: {e}")
        return False
    
    # 测试FilterOutputData
    try:
        output_data = FilterOutputData(
            tradable_mask=np.array([True, False, True]),
            current_regime="中",
            regime_signal=0.0,
            adaptive_target_vol=0.4,
            filter_statistics={
                'filtered_count': 2,
                'total_count': 3,
                'filter_ratio': 0.67
            }
        )
        print("✓ FilterOutputData创建成功")
        print(f"  - 可交易掩码: {output_data.tradable_mask}")
        print(f"  - 当前状态: {output_data.current_regime}")
        print(f"  - 状态信号: {output_data.regime_signal}")
    except Exception as e:
        print(f"✗ FilterOutputData创建失败: {e}")
        return False
    
    return True

def test_main_controller_interface():
    """测试主控制器接口"""
    print("\n测试主控制器接口...")
    
    try:
        config = DynamicLowVolConfig()
        filter_controller = DynamicLowVolFilter(config, data_manager=None)
        print("✓ DynamicLowVolFilter创建成功")
        
        # 测试接口方法存在（应该抛出NotImplementedError）
        methods_to_test = [
            'update_tradable_mask',
            'get_current_regime', 
            'get_adaptive_target_volatility',
            'get_filter_statistics'
        ]
        
        for method_name in methods_to_test:
            if hasattr(filter_controller, method_name):
                print(f"✓ 方法 {method_name} 存在")
            else:
                print(f"✗ 方法 {method_name} 不存在")
                return False
        
    except Exception as e:
        print(f"✗ DynamicLowVolFilter创建失败: {e}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("任务1验证：创建核心异常类和配置数据结构")
    print("=" * 60)
    
    tests = [
        test_exception_classes,
        test_config_data_structure,
        test_input_output_data_structures,
        test_main_controller_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_func.__name__} 失败")
        except Exception as e:
            print(f"✗ {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 任务1实现验证通过！")
        return True
    else:
        print("✗ 任务1实现存在问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)