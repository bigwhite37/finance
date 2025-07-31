#!/usr/bin/env python3
"""
评估脚本

用于评估训练好的模型性能
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.config import ConfigManager
from rl_trading_system.backtest import MultiFrequencyBacktest


def main():
    parser = argparse.ArgumentParser(description="评估交易模型性能")
    parser.add_argument("--model-path", type=str, required=True,
                       help="模型文件路径")
    parser.add_argument("--config", type=str, default="config/trading_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--start-date", type=str, default=None,
                       help="回测开始日期")
    parser.add_argument("--end-date", type=str, default=None,
                       help="回测结束日期")
    parser.add_argument("--output-dir", type=str, default="./backtest_results",
                       help="结果输出目录")
    
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 覆盖配置参数
    if args.start_date:
        config["backtest"]["start_date"] = args.start_date
    if args.end_date:
        config["backtest"]["end_date"] = args.end_date
    
    print("开始模型评估...")
    print(f"模型路径: {args.model_path}")
    print(f"回测期间: {config['backtest']['start_date']} - {config['backtest']['end_date']}")
    print(f"输出目录: {args.output_dir}")
    
    # TODO: 实现评估逻辑
    # backtest_engine = MultiFrequencyBacktest(config)
    # results = backtest_engine.run_backtest(model, start_date, end_date)
    
    print("评估完成！")


if __name__ == "__main__":
    main()