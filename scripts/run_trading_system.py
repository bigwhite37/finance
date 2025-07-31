#!/usr/bin/env python3
"""
交易系统启动脚本

提供命令行接口来管理和运行交易系统
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_trading_system.system_integration import (
    SystemManager, TradingSystem, SystemConfig, SystemState, system_manager
)


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """从文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}


def create_default_config() -> SystemConfig:
    """创建默认配置"""
    return SystemConfig(
        data_source="qlib",
        stock_pool=['000001.SZ', '000002.SZ', '600000.SH'],
        lookback_window=60,
        update_frequency="1D",
        initial_cash=1000000.0,
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
        enable_monitoring=True,
        enable_audit=True,
        enable_risk_control=True,
        log_level="INFO"
    )


def create_system(args):
    """创建交易系统"""
    print(f"创建交易系统: {args.name}")
    
    # 加载配置
    if args.config:
        config_dict = load_config_from_file(args.config)
        config = SystemConfig(**config_dict)
    else:
        config = create_default_config()
    
    # 覆盖命令行参数
    if args.stock_pool:
        config.stock_pool = args.stock_pool.split(',')
    if args.initial_cash:
        config.initial_cash = args.initial_cash
    if args.model_path:
        config.model_path = args.model_path
    
    # 创建系统
    success = system_manager.create_system(args.name, config)
    if success:
        print(f"✓ 系统 {args.name} 创建成功")
    else:
        print(f"✗ 系统 {args.name} 创建失败")
        sys.exit(1)


def start_system(args):
    """启动交易系统"""
    print(f"启动交易系统: {args.name}")
    
    success = system_manager.start_system(args.name)
    if success:
        print(f"✓ 系统 {args.name} 启动成功")
        
        if args.daemon:
            print("系统在后台运行...")
        else:
            print("系统运行中，按 Ctrl+C 停止...")
            try:
                while True:
                    status = system_manager.get_system_status(args.name)
                    if status and status['state'] == SystemState.RUNNING.value:
                        print(f"状态: {status['state']}, "
                              f"组合价值: {status['portfolio_value']:.2f}, "
                              f"总收益: {status['stats']['total_return']:.4f}")
                    time.sleep(10)
            except KeyboardInterrupt:
                print("\n收到停止信号，正在停止系统...")
                system_manager.stop_system(args.name)
                print("系统已停止")
    else:
        print(f"✗ 系统 {args.name} 启动失败")
        sys.exit(1)


def stop_system(args):
    """停止交易系统"""
    print(f"停止交易系统: {args.name}")
    
    success = system_manager.stop_system(args.name)
    if success:
        print(f"✓ 系统 {args.name} 停止成功")
    else:
        print(f"✗ 系统 {args.name} 停止失败")
        sys.exit(1)


def status_system(args):
    """查看系统状态"""
    if args.name:
        # 查看特定系统状态
        status = system_manager.get_system_status(args.name)
        if status:
            print(f"系统: {args.name}")
            print(f"状态: {status['state']}")
            print(f"组合价值: {status['portfolio_value']:.2f}")
            print(f"当前持仓: {status['current_positions']}")
            print(f"最后更新: {status['last_update_time']}")
            print(f"统计信息:")
            for key, value in status['stats'].items():
                print(f"  {key}: {value}")
        else:
            print(f"系统 {args.name} 不存在")
    else:
        # 查看所有系统状态
        systems = system_manager.list_systems()
        if not systems:
            print("没有运行的系统")
        else:
            print("系统列表:")
            for name in systems:
                status = system_manager.get_system_status(name)
                if status:
                    print(f"  {name}: {status['state']} "
                          f"(价值: {status['portfolio_value']:.2f})")


def list_systems(args):
    """列出所有系统"""
    systems = system_manager.list_systems()
    if not systems:
        print("没有创建的系统")
    else:
        print("系统列表:")
        for name in systems:
            status = system_manager.get_system_status(name)
            if status:
                print(f"  {name}: {status['state']}")


def remove_system(args):
    """移除系统"""
    print(f"移除交易系统: {args.name}")
    
    success = system_manager.remove_system(args.name)
    if success:
        print(f"✓ 系统 {args.name} 移除成功")
    else:
        print(f"✗ 系统 {args.name} 移除失败")
        sys.exit(1)


def run_backtest(args):
    """运行回测"""
    print(f"运行回测: {args.name}")
    
    # 创建回测配置
    config = create_default_config()
    config.update_frequency = "1D"  # 回测使用日频
    
    if args.config:
        config_dict = load_config_from_file(args.config)
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 设置回测参数
    if args.start_date:
        config.start_date = args.start_date
    if args.end_date:
        config.end_date = args.end_date
    if args.initial_cash:
        config.initial_cash = args.initial_cash
    
    # 创建回测系统
    backtest_name = f"{args.name}_backtest"
    success = system_manager.create_system(backtest_name, config)
    
    if success:
        print("开始回测...")
        system_manager.start_system(backtest_name)
        
        # 等待回测完成（简化版本）
        time.sleep(5)
        
        # 获取回测结果
        status = system_manager.get_system_status(backtest_name)
        if status:
            print("回测结果:")
            print(f"  总收益: {status['stats']['total_return']:.4f}")
            print(f"  最大回撤: {status['stats']['max_drawdown']:.4f}")
            print(f"  交易次数: {status['stats']['total_trades']}")
        
        # 清理回测系统
        system_manager.remove_system(backtest_name)
        print("回测完成")
    else:
        print("回测系统创建失败")
        sys.exit(1)


def generate_config(args):
    """生成配置文件模板"""
    config = create_default_config()
    config_dict = config.__dict__.copy()
    
    # 转换为JSON可序列化的格式
    for key, value in config_dict.items():
        if hasattr(value, '__dict__'):
            config_dict[key] = value.__dict__
    
    output_path = args.output or "trading_config.json"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ 配置文件模板已生成: {output_path}")
    except Exception as e:
        print(f"✗ 生成配置文件失败: {e}")
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="交易系统管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 创建系统
    create_parser = subparsers.add_parser('create', help='创建交易系统')
    create_parser.add_argument('name', help='系统名称')
    create_parser.add_argument('--config', help='配置文件路径')
    create_parser.add_argument('--stock-pool', help='股票池，逗号分隔')
    create_parser.add_argument('--initial-cash', type=float, help='初始资金')
    create_parser.add_argument('--model-path', help='模型路径')
    create_parser.set_defaults(func=create_system)
    
    # 启动系统
    start_parser = subparsers.add_parser('start', help='启动交易系统')
    start_parser.add_argument('name', help='系统名称')
    start_parser.add_argument('--daemon', action='store_true', help='后台运行')
    start_parser.set_defaults(func=start_system)
    
    # 停止系统
    stop_parser = subparsers.add_parser('stop', help='停止交易系统')
    stop_parser.add_argument('name', help='系统名称')
    stop_parser.set_defaults(func=stop_system)
    
    # 查看状态
    status_parser = subparsers.add_parser('status', help='查看系统状态')
    status_parser.add_argument('name', nargs='?', help='系统名称（可选）')
    status_parser.set_defaults(func=status_system)
    
    # 列出系统
    list_parser = subparsers.add_parser('list', help='列出所有系统')
    list_parser.set_defaults(func=list_systems)
    
    # 移除系统
    remove_parser = subparsers.add_parser('remove', help='移除交易系统')
    remove_parser.add_argument('name', help='系统名称')
    remove_parser.set_defaults(func=remove_system)
    
    # 运行回测
    backtest_parser = subparsers.add_parser('backtest', help='运行回测')
    backtest_parser.add_argument('name', help='回测名称')
    backtest_parser.add_argument('--config', help='配置文件路径')
    backtest_parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    backtest_parser.add_argument('--initial-cash', type=float, help='初始资金')
    backtest_parser.set_defaults(func=run_backtest)
    
    # 生成配置
    config_parser = subparsers.add_parser('config', help='生成配置文件模板')
    config_parser.add_argument('--output', help='输出文件路径')
    config_parser.set_defaults(func=generate_config)
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 执行命令
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"执行命令时发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()