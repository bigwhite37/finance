"""
主训练脚本
集成所有组件实现分层RL系统训练
基于O3设计和Claude实现方案
"""
import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
import gym

# 项目模块导入
from src.data_loader import QlibDataLoader
from envs.trading_env import TradingEnvironment
from models.expert_policy import ExpertPopulation
from models.meta_router import MetaRouter
from replay.shared_buffer import SharedReplayBuffer
from trainers.mfpbt_trainer import MFPBTTrainer


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志系统"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    # 设置第三方库日志级别
    logging.getLogger('stable_baselines3').setLevel(logging.WARNING)
    logging.getLogger('gym').setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return config


def create_training_environment(config: Dict[str, Any]) -> TradingEnvironment:
    """创建训练环境"""
    env_config = config['environment']
    
    # 初始化数据加载器
    data_loader = QlibDataLoader()
    data_loader.initialize_qlib()
    
    # 获取股票列表
    instruments = data_loader.get_stock_list(
        market=env_config.get('market', 'csi300'),
        limit=env_config.get('max_stocks', 50)
    )
    
    logging.info(f"选择股票池: {env_config.get('market', 'csi300')}, 股票数量: {len(instruments)}")
    
    # 从数据配置获取时间范围
    data_config = config['data']
    start_date = data_config.get('train_start', data_config.get('start_time', '2020-01-01'))
    end_date = data_config.get('train_end', data_config.get('end_time', '2022-12-31'))
    
    # 创建交易环境
    env = TradingEnvironment(
        instruments=instruments,
        start_date=start_date,
        end_date=end_date,
        initial_capital=env_config.get('initial_capital', 1000000),
        data_loader=data_loader,
        lookback_window=env_config.get('lookback_window', 60),
        max_stocks=env_config.get('max_stocks', 50),
        normalize_observations=env_config.get('normalize_observations', True),
        random_start=env_config.get('random_start', True)
    )
    
    return env


def create_expert_population(config: Dict[str, Any], env_factory) -> ExpertPopulation:
    """创建专家种群"""
    expert_config = config['experts']
    
    expert_population = ExpertPopulation(
        n_experts=expert_config['n_experts'],
        env_factory=env_factory,
        skill_dim=expert_config.get('skill_dim', 10),
        buffer_size=expert_config.get('buffer_size', int(1e6)),
        learning_rate=expert_config.get('learning_rate', 3e-4),
        mi_reward_weight=expert_config.get('mi_reward_weight', 0.5)
    )
    
    logging.info(f"专家种群创建完成，专家数量: {expert_config['n_experts']}")
    
    return expert_population


def create_meta_router(config: Dict[str, Any], experts, base_env) -> MetaRouter:
    """创建元路由器"""
    router_config = config['meta_router']
    
    meta_router = MetaRouter(
        experts=experts,
        base_env=base_env,
        kl_lambda=router_config.get('kl_lambda', 0.1),
        n_steps=router_config.get('n_steps', 1024),
        batch_size=router_config.get('batch_size', 64),
        gamma=router_config.get('gamma', 0.99),
        gae_lambda=router_config.get('gae_lambda', 0.95),
        ent_coef=router_config.get('ent_coef', 0.01),
        learning_rate=router_config.get('learning_rate', 3e-4)
    )
    
    logging.info("元路由器创建完成")
    
    return meta_router


def create_shared_buffer(config: Dict[str, Any]) -> SharedReplayBuffer:
    """创建共享经验池"""
    buffer_config = config['shared_buffer']
    
    shared_buffer = SharedReplayBuffer(
        capacity=buffer_config.get('capacity', int(2e6)),
        n_experts=config['experts']['n_experts'],
        sequence_length=buffer_config.get('sequence_length', 32),
        compress_data=buffer_config.get('compress_data', True)
    )
    
    logging.info(f"共享经验池创建完成，容量: {buffer_config.get('capacity', int(2e6))}")
    
    return shared_buffer


def create_trainer(config: Dict[str, Any], 
                  expert_population: ExpertPopulation,
                  meta_router: MetaRouter,
                  shared_buffer: SharedReplayBuffer) -> MFPBTTrainer:
    """创建MF-PBT训练器"""
    trainer_config = config['trainer']
    
    trainer = MFPBTTrainer(
        expert_population=expert_population,
        meta_router=meta_router,
        shared_buffer=shared_buffer,
        population_size=config['experts']['n_experts'],
        diversity_threshold=trainer_config.get('diversity_threshold', 0.25),
        training_config=trainer_config.get('training_schedule', {})
    )
    
    logging.info("MF-PBT训练器创建完成")
    
    return trainer


def evaluation_callback(trainer: MFPBTTrainer) -> Dict[str, Any]:
    """评估回调函数"""
    # 获取当前性能指标
    population_metrics = trainer.expert_population.get_population_metrics()
    router_stats = trainer.meta_router.get_selection_statistics()
    
    # 简单的评估指标
    eval_results = {
        'avg_population_reward': population_metrics.get('avg_population_reward', 0),
        'diversity_score': population_metrics.get('diversity_score', 0),
        'router_entropy': router_stats.get('normalized_entropy', 0) if router_stats else 0,
        'step': trainer.global_step
    }
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(description='分层RL量化交易系统训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--log_level', type=str, default='INFO', help='日志级别')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, str(log_file))
    
    logger = logging.getLogger(__name__)
    logger.info("=== 分层RL量化交易系统训练开始 ===")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"随机种子: {args.seed}")
    
    try:
        # 设置随机种子
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        
        # 加载配置
        config = load_config(args.config)
        logger.info("配置文件加载完成")
        
        # 保存配置到输出目录
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 创建训练环境
        logger.info("创建训练环境...")
        base_env = create_training_environment(config)
        
        # 环境工厂函数
        def env_factory():
            return create_training_environment(config)
        
        # 创建专家种群
        logger.info("创建专家种群...")
        expert_population = create_expert_population(config, env_factory)
        
        # 创建元路由器
        logger.info("创建元路由器...")
        meta_router = create_meta_router(config, expert_population.experts, base_env)
        
        # 创建共享经验池
        logger.info("创建共享经验池...")
        shared_buffer = create_shared_buffer(config)
        
        # 创建训练器
        logger.info("创建MF-PBT训练器...")
        trainer = create_trainer(config, expert_population, meta_router, shared_buffer)
        
        # 恢复训练（如果指定）
        if args.resume:
            logger.info(f"恢复训练: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # 开始训练
        training_config = config.get('training', {})
        total_steps = training_config.get('total_steps', 100000)
        checkpoint_freq = training_config.get('checkpoint_freq', 10000)
        
        logger.info(f"开始训练，总步数: {total_steps}")
        
        training_results = trainer.train(
            total_steps=total_steps,
            save_path=str(output_dir),
            eval_callback=evaluation_callback,
            checkpoint_freq=checkpoint_freq
        )
        
        # 保存训练结果
        results_file = output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logger.info(f"训练完成！结果已保存到: {results_file}")
        
        # 输出最终统计
        logger.info("=== 训练结果摘要 ===")
        population_metrics = training_results.get('population_metrics', {})
        efficiency_metrics = training_results.get('efficiency_metrics', {})
        
        logger.info(f"种群平均奖励: {population_metrics.get('avg_population_reward', 0):.6f}")
        logger.info(f"种群多样性得分: {population_metrics.get('diversity_score', 0):.4f}")
        logger.info(f"总训练时间: {efficiency_metrics.get('total_training_time', 0):.2f}秒")
        logger.info(f"平均步骤时间: {efficiency_metrics.get('avg_step_time', 0):.4f}秒")
        
        logger.info("=== 训练成功完成 ===")
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        sys.exit(1)
    except (RuntimeError, ValueError, ImportError, FileNotFoundError) as e:
        logger.error(f"训练过程中发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()