#!/usr/bin/env python3
"""
训练脚本

用于训练强化学习交易智能体
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.config import ConfigManager
from rl_trading_system.training import RLTrainer


def main():
    parser = argparse.ArgumentParser(description="训练强化学习交易智能体")
    parser.add_argument("--config", type=str, default="config/model_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--data-config", type=str, default="config/trading_config.yaml",
                       help="交易配置文件路径")
    parser.add_argument("--episodes", type=int, default=None,
                       help="训练轮数")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager()
    model_config = config_manager.load_config(args.config)
    trading_config = config_manager.load_config(args.data_config)
    
    # 覆盖配置参数
    if args.episodes:
        model_config["training"]["n_episodes"] = args.episodes
    
    print("开始训练强化学习交易智能体...")
    print(f"配置文件: {args.config}")
    print(f"训练轮数: {model_config['training']['n_episodes']}")
    print(f"输出目录: {args.output_dir}")
    
    # TODO: 实现训练逻辑
    # trainer = RLTrainer(model_config, trading_config)
    # trainer.train()
    
    print("训练完成！")


if __name__ == "__main__":
    main()