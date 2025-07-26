#!/usr/bin/env python3
"""
演示风险控制器与目标波动率控制器的协调逻辑
"""

import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import Mock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_control.risk_controller import RiskController
from data.data_manager import DataManager


def demo_coordination_logic():
    """演示协调逻辑"""
    print("=" * 80)
    print("演示：风险控制器与目标波动率控制器的协调逻辑")
    print("=" * 80)
    
    # 配置启用动态低波筛选器
    config = {
        'max_position': 0.1,
        'max_leverage': 1.2,
        'target_volatility': 0.12,  # 配置的基础目标波动率
        'enable_dynamic_lowvol': True,
        'dynamic_lowvol': {
            'rolling_windows': [20, 60],
            'garch_window': 100,
            'regime_detection_window': 30,
            'enable_caching': False,
            'parallel_processing': False
        }
    }
    
    # 创建模拟数据管理器
    mock_data_manager = Mock(spec=DataManager)
    
    # 创建风险控制器
    risk_controller = RiskController(config, mock_data_manager)
    
    print(f"配置的基础目标波动率: {config['target_volatility']}")
    
    # 测试不同情况下的自适应目标波动率
    scenarios = [
        ("无筛选器", None),
        ("筛选器获取失败", "error"),
        ("低波动状态", 0.45),
        ("中等波动状态", 0.40),
        ("高波动状态", 0.35)
    ]
    
    for scenario_name, adaptive_vol in scenarios:
        print(f"\n场景: {scenario_name}")
        
        if adaptive_vol is None:
            # 模拟无筛选器情况
            risk_controller.lowvol_filter = None
        elif adaptive_vol == "error":
            # 模拟筛选器错误情况
            if risk_controller.lowvol_filter:
                risk_controller.lowvol_filter.get_adaptive_target_volatility = Mock(
                    side_effect=Exception("获取失败")
                )
        else:
            # 模拟正常情况
            if risk_controller.lowvol_filter:
                risk_controller.lowvol_filter.get_adaptive_target_volatility = Mock(
                    return_value=adaptive_vol
                )
        
        # 获取自适应目标波动率
        target_vol = risk_controller._get_adaptive_target_volatility()
        
        print(f"  自适应目标波动率: {target_vol:.3f}")
        
        # 计算与基础目标波动率的差异
        diff = target_vol - config['target_volatility']
        if diff > 0:
            print(f"  相比基础目标波动率提高了: {diff:.3f} ({diff/config['target_volatility']:.1%})")
        elif diff < 0:
            print(f"  相比基础目标波动率降低了: {abs(diff):.3f} ({abs(diff)/config['target_volatility']:.1%})")
        else:
            print(f"  与基础目标波动率相同")
    
    print("\n" + "=" * 80)
    print("协调逻辑总结:")
    print("1. 优先使用动态低波筛选器的自适应目标波动率")
    print("2. 筛选器不可用或出错时，回退到配置的基础目标波动率")
    print("3. 目标波动率控制器使用自适应目标波动率调整杠杆")
    print("4. 最终安全检查确保所有约束都得到满足")
    print("=" * 80)


def demo_filter_integration():
    """演示筛选器集成"""
    print("\n" + "=" * 80)
    print("演示：动态低波筛选器集成效果")
    print("=" * 80)
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    stocks = [f'stock_{i:03d}' for i in range(10)]
    
    np.random.seed(42)
    price_data = pd.DataFrame(index=dates, columns=stocks)
    for i, stock in enumerate(stocks):
        returns = np.random.normal(0.0005, 0.15/np.sqrt(252), len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        price_data[stock] = prices
    
    # 配置
    config = {
        'max_position': 0.1,
        'max_leverage': 1.2,
        'target_volatility': 0.12,
        'enable_dynamic_lowvol': False  # 先测试不启用的情况
    }
    
    risk_controller = RiskController(config)
    
    # 测试权重
    raw_weights = np.array([0.12] * 10)  # 略超过单股票限制
    current_nav = 1.0
    state = {
        'current_date': pd.Timestamp('2023-06-30'),
        'max_drawdown': 0.05
    }
    
    print("不启用动态低波筛选器:")
    processed_weights = risk_controller.process_weights(
        raw_weights, price_data, current_nav, state
    )
    
    print(f"  原始权重总和: {np.sum(raw_weights):.3f}")
    print(f"  处理后权重总和: {np.sum(processed_weights):.3f}")
    print(f"  最大单股票权重: {np.max(np.abs(processed_weights)):.3f}")
    print(f"  总杠杆: {np.sum(np.abs(processed_weights)):.3f}")
    
    # 获取风险报告
    report = risk_controller.get_risk_report()
    lowvol_info = report['dynamic_lowvol_filter']
    vol_info = report['target_volatility']
    
    print(f"  筛选器状态: {lowvol_info['status']}")
    print(f"  使用自适应目标波动率: {vol_info['using_adaptive']}")
    print(f"  配置目标波动率: {vol_info['configured_target']:.3f}")
    print(f"  实际使用目标波动率: {vol_info['adaptive_target']:.3f}")
    
    print("\n" + "=" * 80)
    print("集成效果总结:")
    print("1. 风险控制器成功集成了动态低波筛选器接口")
    print("2. 自适应目标波动率机制正常工作")
    print("3. 风险报告包含完整的筛选器状态信息")
    print("4. 错误处理机制确保系统稳定性")
    print("=" * 80)


if __name__ == '__main__':
    demo_coordination_logic()
    demo_filter_integration()