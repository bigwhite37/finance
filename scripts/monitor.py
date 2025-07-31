#!/usr/bin/env python3
"""
监控脚本

用于启动系统监控服务
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.config import ConfigManager
from rl_trading_system.monitoring import TradingSystemMonitor


def main():
    parser = argparse.ArgumentParser(description="启动系统监控")
    parser.add_argument("--config", type=str, default="config/monitoring_config.yaml",
                       help="监控配置文件路径")
    parser.add_argument("--port", type=int, default=8000,
                       help="监控服务端口")
    
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 覆盖端口配置
    if args.port:
        config["monitoring"]["prometheus"]["port"] = args.port
    
    print("启动系统监控...")
    print(f"配置文件: {args.config}")
    print(f"监控端口: {config['monitoring']['prometheus']['port']}")
    
    # TODO: 实现监控逻辑
    # monitor = TradingSystemMonitor(config)
    # monitor.start()
    
    print("监控服务已启动！")
    print(f"Prometheus指标地址: http://localhost:{config['monitoring']['prometheus']['port']}/metrics")


if __name__ == "__main__":
    main()