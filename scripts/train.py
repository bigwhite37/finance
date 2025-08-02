#!/usr/bin/env python3
"""
训练脚本

用于训练强化学习交易智能体
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.config import ConfigManager
from rl_trading_system.training import RLTrainer, TrainingConfig, create_split_strategy, SplitConfig
from rl_trading_system.data import QlibDataInterface, FeatureEngineer, DataProcessor
from rl_trading_system.models import TimeSeriesTransformer, SACAgent, TransformerConfig, SACConfig
from rl_trading_system.trading import PortfolioEnvironment, PortfolioConfig
from rl_trading_system.utils.terminal_colors import (
    ColorFormatter, print_banner, print_section, print_model_recommendation,
    print_training_stats, print_evaluation_results
)


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """设置日志系统"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def create_training_components(model_config: dict, trading_config: dict, output_dir: str):
    """创建训练所需的组件"""
    logger = logging.getLogger(__name__)
    logger.info("初始化训练组件...")

    # 创建数据组件
    data_interface = QlibDataInterface()
    feature_engineer = FeatureEngineer()
    data_processor = DataProcessor()

    # 加载和处理数据
    logger.info("加载股票数据...")
    # 从nested配置中提取参数
    trading_env = trading_config.get("trading", {}).get("environment", {})
    backtest_config = trading_config.get("backtest", {})
    
    stock_pool = trading_env.get("stock_pool")
    if not stock_pool:  # 如果stock_pool为空列表，使用默认值
        raise RuntimeError(f"empty tock_pool: {stock_pool}")
    
    start_date = backtest_config.get("start_date", "2020-01-01")
    end_date = backtest_config.get("end_date", "2023-12-31")

    market_data = data_interface.get_price_data(
        symbols=stock_pool,
        start_date=start_date,
        end_date=end_date
    )

    # 特征工程
    logger.info("执行特征工程...")
    if market_data.empty:
        raise ValueError(f"无法获取股票数据: symbols={stock_pool}, "
                        f"start_date={start_date}, end_date={end_date}")
        
    # 计算技术指标
    technical_features = feature_engineer.calculate_technical_indicators(market_data)
    
    # 计算其他特征
    microstructure_features = feature_engineer.calculate_microstructure_features(market_data)
    volatility_features = feature_engineer.calculate_volatility_features(market_data)
    momentum_features = feature_engineer.calculate_momentum_features(market_data)
    
    # 合并所有特征
    processed_data = feature_engineer.combine_features([
        market_data,
        technical_features,
        microstructure_features,
        volatility_features,
        momentum_features
    ])
    
    # 标准化特征
    normalized_data = feature_engineer.normalize_features(processed_data)

    # 数据分割
    logger.debug("分割训练/验证数据...")
    split_config = SplitConfig(
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1
    )
    split_strategy = create_split_strategy("time_series", split_config)
    data_split = split_strategy.split(normalized_data)

    # 创建Transformer配置
    transformer_config = TransformerConfig(
        d_model=model_config["model"]["transformer"]["d_model"],
        n_heads=model_config["model"]["transformer"]["n_heads"],
        n_layers=model_config["model"]["transformer"]["n_layers"],
        d_ff=model_config["model"]["transformer"]["d_ff"],
        dropout=model_config["model"]["transformer"]["dropout"],
        max_seq_len=model_config["model"]["transformer"]["max_seq_len"],
        n_features=model_config["model"]["transformer"]["n_features"]
    )

    # 创建SAC配置，集成Transformer
    sac_config = SACConfig(
        state_dim=transformer_config.d_model,  # 使用Transformer输出维度作为SAC状态维度
        action_dim=model_config["model"]["sac"]["action_dim"],
        hidden_dim=model_config["model"]["sac"]["hidden_dim"],
        lr_actor=model_config["model"]["sac"]["lr_actor"],
        lr_critic=model_config["model"]["sac"]["lr_critic"],
        lr_alpha=model_config["model"]["sac"]["lr_alpha"],
        gamma=model_config["model"]["sac"]["gamma"],
        tau=model_config["model"]["sac"]["tau"],
        alpha=model_config["model"]["sac"]["alpha"],
        target_entropy=model_config["model"]["sac"]["target_entropy"],
        buffer_capacity=model_config["model"]["sac"]["buffer_size"],
        batch_size=model_config["model"]["sac"]["batch_size"],
        use_transformer=True,  # 启用Transformer集成
        transformer_config=transformer_config  # 传入Transformer配置
    )

    # 创建模型（SAC智能体内部集成了Transformer）
    logger.debug("初始化SAC智能体（集成Transformer）...")
    sac_agent = SACAgent(sac_config)

    # 创建交易环境
    logger.debug("创建交易环境...")
    portfolio_config = PortfolioConfig(
        stock_pool=stock_pool,
        initial_cash=trading_env.get("initial_cash", 1000000.0),
        commission_rate=trading_env.get("commission_rate", 0.001),
        stamp_tax_rate=trading_env.get("stamp_tax_rate", 0.001),
        max_position_size=trading_env.get("max_position_size", 0.1)
    )

    # Extract training data using indices
    train_data = normalized_data.iloc[data_split.train_indices]
    
    portfolio_env = PortfolioEnvironment(
        config=portfolio_config,
        data_interface=data_interface,
        feature_engineer=feature_engineer,
        start_date=start_date,
        end_date=end_date
    )

    # 创建训练配置
    training_config = TrainingConfig(
        n_episodes=model_config["training"]["n_episodes"],
        save_frequency=model_config["training"].get("eval_freq", 100),
        validation_frequency=model_config["training"].get("eval_freq", 100),
        early_stopping_patience=model_config["training"].get("patience", 50),
        early_stopping_min_delta=model_config["training"].get("min_delta", 0.001),
        save_dir=output_dir
    )

    return portfolio_env, sac_agent, data_split, training_config


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
    parser.add_argument("--resume", type=str, default=None,
                       help="从检查点恢复训练")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="训练设备")
    parser.add_argument("--no-color", action="store_true",
                       help="禁用彩色输出")

    args = parser.parse_args()
    
    # 初始化彩色格式化器
    formatter = ColorFormatter(enable_color=not args.no_color)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger_instance = setup_logging(str(output_dir), args.log_level)

    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 打印标题横幅
    print_banner(
        "🚀 强化学习交易智能体训练",
        f"SAC + Transformer | 设备: {device}",
        formatter
    )

    try:
        # 加载配置
        print_section("📁 加载配置文件", formatter)
        config_manager = ConfigManager()

        # 检查配置文件是否存在
        model_config_path = Path(args.config)
        if not model_config_path.exists():
            print(formatter.error(f"❌ 模型配置文件不存在: {args.config}"))
            raise FileNotFoundError(f"模型配置文件不存在: {args.config}")

        data_config_path = Path(args.data_config)
        if not data_config_path.exists():
            print(formatter.error(f"❌ 数据配置文件不存在: {args.data_config}"))
            raise FileNotFoundError(f"数据配置文件不存在: {args.data_config}")

        model_config = config_manager.load_config(str(model_config_path))
        trading_config = config_manager.load_config(str(data_config_path))

        # 调试：打印配置内容
        logger_instance.debug(f"模型配置内容: {model_config}")
        logger_instance.debug(f"交易配置内容: {trading_config}")

        # 覆盖配置参数
        if args.episodes:
            if "training" not in model_config:
                model_config["training"] = {}
            model_config["training"]["n_episodes"] = args.episodes

        print(f"  {formatter.success('✅ 模型配置文件')}: {formatter.path(args.config)}")
        print(f"  {formatter.success('✅ 交易配置文件')}: {formatter.path(args.data_config)}")
        
        # 安全地获取训练轮数
        n_episodes = model_config.get("model", {}).get("training", {}).get("n_episodes", 100)
        print(f"  {formatter.info('训练轮数')}: {formatter.number(str(n_episodes))}")
        print(f"  {formatter.info('输出目录')}: {formatter.path(args.output_dir)}")
        print(f"  {formatter.info('训练设备')}: {formatter.highlight(device)}")
        print()

        # 创建训练组件
        environment, agent, data_split, training_config = create_training_components(
            model_config, trading_config, str(output_dir)
        )

        # 设置设备
        training_config.device = device

        # 创建训练器
        logger_instance.info("初始化训练器...")
        trainer = RLTrainer(training_config, environment, agent, data_split)

        # 从检查点恢复（如果指定）
        start_episode = 0
        if args.resume:
            if Path(args.resume).exists():
                logger_instance.info(f"从检查点恢复训练: {args.resume}")
                start_episode = trainer.load_checkpoint(args.resume)
            else:
                logger_instance.warning(f"检查点文件不存在: {args.resume}")

        # 开始训练
        print_section("🎯 开始训练", formatter)
        print(f"  {formatter.info('正在训练强化学习交易智能体...')}")
        print()
        
        training_stats = trainer.train()

        # 输出训练统计
        print_training_stats(training_stats, formatter)

        # 运行最终评估
        print_section("📊 最终评估", formatter)
        print(f"  {formatter.info('正在运行模型评估...')}")
        evaluation_stats = trainer.evaluate(n_episodes=20)

        # 输出评估结果
        print_evaluation_results(evaluation_stats, formatter)

        # 显示模型保存信息和使用建议
        model_paths = {
            'final_model': str(output_dir / "final_model_agent.pth"),
        }
        
        # 检查是否有最佳模型
        best_model_path = output_dir / "best_model_agent.pth"
        if best_model_path.exists():
            model_paths['best_model'] = str(best_model_path)
        
        print_model_recommendation(model_paths, formatter)
        
        print(formatter.success(f"🎉 训练完成！所有结果已保存到: {formatter.path(str(output_dir))}"))

    except Exception as e:
        logger_instance.error(f"训练过程中发生错误: {str(e)}")
        logger_instance.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()