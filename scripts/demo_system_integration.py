#!/usr/bin/env python3
"""
系统集成演示脚本

演示完整的交易系统集成功能
"""

import sys
import time
import signal
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_trading_system.system_integration import (
    SystemConfig, TradingSystem, SystemManager, system_manager
)


def signal_handler(signum, frame):
    """信号处理器"""
    print("\n收到停止信号，正在关闭系统...")
    # 停止所有系统
    for name in system_manager.list_systems():
        system_manager.stop_system(name)
        system_manager.remove_system(name)
    print("所有系统已关闭")
    sys.exit(0)


def create_demo_system():
    """创建演示系统"""
    print("=== 创建演示交易系统 ===")
    
    # 创建系统配置
    config = SystemConfig(
        data_source="qlib",
        stock_pool=['000001.SZ', '000002.SZ', '600000.SH'],
        lookback_window=60,
        update_frequency="1D",
        initial_cash=1000000.0,
        
        # 模型配置
        transformer_config={
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'dropout': 0.1
        },
        sac_config={
            'hidden_dim': 256,
            'lr_actor': 0.0003,
            'lr_critic': 0.0003
        },
        
        # 交易配置
        commission_rate=0.001,
        stamp_tax_rate=0.001,
        max_position_size=0.2,
        
        # 系统配置
        enable_monitoring=True,
        enable_audit=True,
        enable_risk_control=True,
        log_level="INFO"
    )
    
    # 创建系统
    success = system_manager.create_system("demo_system", config)
    if success:
        print("✓ 演示系统创建成功")
        return True
    else:
        print("✗ 演示系统创建失败")
        return False


def run_demo_system():
    """运行演示系统"""
    print("\n=== 启动演示系统 ===")
    
    # 启动系统
    success = system_manager.start_system("demo_system")
    if not success:
        print("✗ 系统启动失败")
        return False
    
    print("✓ 系统启动成功")
    print("系统正在运行，监控中...")
    
    # 监控系统运行
    try:
        for i in range(30):  # 运行30秒
            time.sleep(1)
            
            # 每5秒显示一次状态
            if i % 5 == 0:
                status = system_manager.get_system_status("demo_system")
                if status:
                    print(f"[{i:2d}s] 状态: {status['state']}, "
                          f"组合价值: ¥{status['portfolio_value']:,.2f}, "
                          f"总收益: {status['stats']['total_return']:.4f}")
                else:
                    print(f"[{i:2d}s] 无法获取系统状态")
    
    except KeyboardInterrupt:
        print("\n收到中断信号")
    
    return True


def stop_demo_system():
    """停止演示系统"""
    print("\n=== 停止演示系统 ===")
    
    # 停止系统
    success = system_manager.stop_system("demo_system")
    if success:
        print("✓ 系统停止成功")
    else:
        print("✗ 系统停止失败")
    
    # 获取最终状态
    status = system_manager.get_system_status("demo_system")
    if status:
        print("\n=== 最终统计 ===")
        print(f"运行时间: {status['stats'].get('start_time', 'N/A')}")
        print(f"总交易次数: {status['stats']['total_trades']}")
        print(f"成功交易次数: {status['stats']['successful_trades']}")
        print(f"最终组合价值: ¥{status['portfolio_value']:,.2f}")
        print(f"总收益率: {status['stats']['total_return']:.4f}")
        print(f"最大回撤: {status['stats']['max_drawdown']:.4f}")
    
    # 清理系统
    system_manager.remove_system("demo_system")
    print("✓ 系统清理完成")


def create_multiple_systems_demo():
    """创建多系统演示"""
    print("\n=== 多系统演示 ===")
    
    # 创建不同配置的系统
    systems = [
        {
            'name': 'conservative_system',
            'config': SystemConfig(
                stock_pool=['000001.SZ'],
                initial_cash=500000.0,
                max_position_size=0.1,
                enable_monitoring=False,
                enable_audit=False,
                enable_risk_control=True,
                log_level="WARNING"
            )
        },
        {
            'name': 'aggressive_system',
            'config': SystemConfig(
                stock_pool=['000001.SZ', '000002.SZ', '600000.SH'],
                initial_cash=1000000.0,
                max_position_size=0.3,
                enable_monitoring=False,
                enable_audit=False,
                enable_risk_control=False,
                log_level="WARNING"
            )
        }
    ]
    
    # 创建系统
    created_systems = []
    for system_info in systems:
        name = system_info['name']
        config = system_info['config']
        
        success = system_manager.create_system(name, config)
        if success:
            created_systems.append(name)
            print(f"✓ {name} 创建成功")
        else:
            print(f"✗ {name} 创建失败")
    
    # 显示所有系统状态
    print(f"\n当前系统数量: {len(system_manager.list_systems())}")
    for name in created_systems:
        status = system_manager.get_system_status(name)
        if status:
            print(f"- {name}: {status['state']}, "
                  f"初始资金: ¥{status['config']['initial_cash']:,.0f}")
    
    # 清理系统
    for name in created_systems:
        system_manager.remove_system(name)
    
    print("✓ 多系统演示完成")


def main():
    """主函数"""
    print("🚀 强化学习量化交易系统集成演示")
    print("=" * 50)
    
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 1. 创建演示系统
        if not create_demo_system():
            return 1
        
        # 2. 运行演示系统
        if not run_demo_system():
            return 1
        
        # 3. 停止演示系统
        stop_demo_system()
        
        # 4. 多系统演示
        create_multiple_systems_demo()
        
        print("\n🎉 演示完成！")
        print("\n系统集成功能验证:")
        print("✓ 系统创建和初始化")
        print("✓ 系统启动和停止")
        print("✓ 系统状态监控")
        print("✓ 多系统管理")
        print("✓ 错误处理和清理")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())