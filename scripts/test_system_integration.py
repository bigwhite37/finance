#!/usr/bin/env python3
"""
系统集成测试脚本

测试完整的交易系统集成功能
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_trading_system.system_integration import (
    SystemConfig, TradingSystem, SystemManager, system_manager
)


def test_basic_system_creation():
    """测试基本系统创建"""
    print("=== 测试基本系统创建 ===")
    
    # 创建系统配置
    config = SystemConfig(
        data_source="qlib",
        stock_pool=['000001.SZ', '000002.SZ'],
        lookback_window=30,
        update_frequency="1D",
        initial_cash=100000.0,
        transformer_config={
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'dropout': 0.1
        },
        sac_config={
            'hidden_dim': 128,
            'lr_actor': 0.001,
            'lr_critic': 0.001
        },
        enable_monitoring=False,  # 简化测试
        enable_audit=False,
        enable_risk_control=False,
        log_level="INFO"
    )
    
    # 创建系统
    success = system_manager.create_system("test_system", config)
    if success:
        print("✓ 系统创建成功")
    else:
        print("✗ 系统创建失败")
        return False
    
    # 检查系统状态
    status = system_manager.get_system_status("test_system")
    if status:
        print(f"✓ 系统状态: {status['state']}")
        print(f"✓ 组合价值: {status['portfolio_value']}")
    else:
        print("✗ 无法获取系统状态")
        return False
    
    return True


def test_system_lifecycle():
    """测试系统生命周期"""
    print("\n=== 测试系统生命周期 ===")
    
    # 启动系统
    print("启动系统...")
    success = system_manager.start_system("test_system")
    if success:
        print("✓ 系统启动成功")
    else:
        print("✗ 系统启动失败")
        return False
    
    # 等待系统运行
    print("等待系统运行...")
    time.sleep(5)
    
    # 检查运行状态
    status = system_manager.get_system_status("test_system")
    if status and status['state'] == 'running':
        print("✓ 系统正在运行")
    else:
        print(f"✗ 系统状态异常: {status['state'] if status else 'None'}")
    
    # 停止系统
    print("停止系统...")
    success = system_manager.stop_system("test_system")
    if success:
        print("✓ 系统停止成功")
    else:
        print("✗ 系统停止失败")
        return False
    
    # 清理系统
    system_manager.remove_system("test_system")
    print("✓ 系统清理完成")
    
    return True


def test_multiple_systems():
    """测试多系统管理"""
    print("\n=== 测试多系统管理 ===")
    
    # 创建多个系统
    configs = []
    for i in range(3):
        config = SystemConfig(
            data_source="qlib",
            stock_pool=['000001.SZ', '000002.SZ'],
            initial_cash=100000.0 * (i + 1),
            enable_monitoring=False,
            enable_audit=False,
            enable_risk_control=False,
            log_level="WARNING"  # 减少日志输出
        )
        configs.append(config)
    
    # 创建系统
    system_names = []
    for i, config in enumerate(configs):
        name = f"test_system_{i}"
        success = system_manager.create_system(name, config)
        if success:
            system_names.append(name)
            print(f"✓ 系统 {name} 创建成功")
        else:
            print(f"✗ 系统 {name} 创建失败")
    
    # 列出所有系统
    all_systems = system_manager.list_systems()
    print(f"✓ 当前系统数量: {len(all_systems)}")
    
    # 清理所有测试系统
    for name in system_names:
        system_manager.remove_system(name)
    
    print("✓ 多系统测试完成")
    return True


def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 测试启动不存在的系统
    success = system_manager.start_system("non_existent_system")
    if not success:
        print("✓ 正确处理不存在系统的启动请求")
    else:
        print("✗ 错误处理失败")
        return False
    
    # 测试获取不存在系统的状态
    status = system_manager.get_system_status("non_existent_system")
    if status is None:
        print("✓ 正确处理不存在系统的状态查询")
    else:
        print("✗ 错误处理失败")
        return False
    
    print("✓ 错误处理测试完成")
    return True


def main():
    """主测试函数"""
    print("开始系统集成测试...")
    
    tests = [
        test_basic_system_creation,
        test_system_lifecycle,
        test_multiple_systems,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ 测试异常: {e}")
            failed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")
    
    if failed == 0:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())