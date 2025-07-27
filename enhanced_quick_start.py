#!/usr/bin/env python3
"""
增强系统快速开始
按照 quick_start.md 流程，使用增强组件进行训练和回测
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
import torch
import warnings
import yaml
from datetime import datetime
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入增强组件
from rl_agent.stable_cvar_ppo_agent import StableCVaRPPOAgent
from rl_agent.enhanced_trading_environment import EnhancedTradingEnvironment
from rl_agent.adaptive_safety_shield import AdaptiveSafetyShield
from rl_agent.enhanced_reward_system import EnhancedRewardSystem
from factors.advanced_alpha_factors import AdvancedAlphaFactors

# 导入基础组件
from data.data_manager import DataManager
from config.config_manager import ConfigManager
from backtest.backtest_engine import BacktestEngine
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def load_enhanced_config(config_file: str) -> dict:
    """加载增强配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_enhanced_systems(config: dict):
    """创建增强系统组件"""
    # 初始化数据管理器
    data_manager = DataManager(config.get('data', {}))
    
    # 创建回测引擎
    backtest_engine = BacktestEngine(config.get('backtest', {}))
    
    return {
        'data_manager': data_manager,
        'backtest_engine': backtest_engine
    }


def prepare_enhanced_data(data_manager, config):
    """准备增强数据"""
    logger.info("正在获取股票数据...")
    
    data_config = config.get('data', {})
    stock_data = data_manager.get_stock_data(
        start_time=data_config['start_date'],
        end_time=data_config['end_date'],
        include_fundamentals=True
    )
    
    if stock_data.empty:
        raise ValueError("未能获取到股票数据")
    
    # 获取价格和成交量数据
    price_data = stock_data['$close'].unstack().T
    volume_data = stock_data['$volume'].unstack().T
    
    logger.info(f"价格数据形状: {price_data.shape}")
    
    # 股票池筛选
    if 'selected_stocks.pkl' in os.listdir('./models') and os.path.exists('./models/selected_stocks.pkl'):
        import pickle
        with open('./models/selected_stocks.pkl', 'rb') as f:
            selected_stocks = pickle.load(f)
        logger.info(f"已加载训练时的股票池，包含 {len(selected_stocks)} 只股票")
        
        # 检查股票在当前数据中的可用性
        available_stocks = list(set(selected_stocks) & set(price_data.columns))
        logger.info(f"当前期间可用股票: {len(available_stocks)} 只")
        
        if len(available_stocks) < len(selected_stocks) * 0.8:
            logger.warning(f"可用股票比例较低: {len(available_stocks)}/{len(selected_stocks)}")
        
        price_data = price_data[available_stocks]
        volume_data = volume_data[available_stocks]
        
    else:
        # 首次训练，进行股票筛选
        logger.info("正在筛选低波动股票池...")
        
        # 简单的股票筛选逻辑
        returns = price_data.pct_change().dropna()
        volatility = returns.std()
        
        # 选择波动率适中的股票
        vol_threshold = volatility.quantile(0.7)  # 选择波动率在70%分位以下的股票
        selected_stocks = volatility[volatility <= vol_threshold].index.tolist()
        
        # 进一步筛选：确保有足够的交易数据
        min_data_points = len(price_data) * 0.8
        valid_stocks = []
        for stock in selected_stocks:
            if price_data[stock].notna().sum() >= min_data_points:
                valid_stocks.append(stock)
        
        selected_stocks = valid_stocks[:800]  # 限制最大股票数
        logger.info(f"筛选出 {len(selected_stocks)} 只股票")
        
        # 保存股票池
        os.makedirs('./models', exist_ok=True)
        import pickle
        with open('./models/selected_stocks.pkl', 'wb') as f:
            pickle.dump(selected_stocks, f)
        
        price_data = price_data[selected_stocks]
        volume_data = volume_data[selected_stocks]
    
    # 计算增强因子
    logger.info("正在计算增强因子...")
    enhanced_factors = AdvancedAlphaFactors(config.get('factors', {}))
    
    # 计算可用的因子子集（避免计算过于复杂的因子）
    available_factors = [
        'momentum_reversal_5d', 'momentum_reversal_20d', 'momentum_trend_strength',
        'price_acceleration', 'relative_strength_index', 'momentum_quality',
        'technical_alpha', 'volume_price_correlation'
    ]
    
    factor_data = enhanced_factors.calculate_all_factors(
        price_data, volume_data, factors=available_factors
    )
    
    logger.info(f"因子计算完成! 因子数据形状: {factor_data.shape}")
    
    return price_data, factor_data


def train_enhanced_system(config_file: str):
    """训练增强系统"""
    logger.info("🚀 开始训练增强系统")
    
    # 加载配置
    config = load_enhanced_config(config_file)
    
    # 创建系统组件
    systems = create_enhanced_systems(config)
    
    # 准备数据
    price_data, factor_data = prepare_enhanced_data(systems['data_manager'], config)
    
    # 创建增强交易环境
    logger.info("创建增强交易环境...")
    enhanced_env = EnhancedTradingEnvironment(
        factor_data=factor_data,
        price_data=price_data,
        config=config,
        train_universe=list(price_data.columns)
    )
    
    # 创建稳定CVaR-PPO智能体
    logger.info("创建稳定CVaR-PPO智能体...")
    agent_config = config.get('agent', {})
    enhanced_agent = StableCVaRPPOAgent(
        state_dim=enhanced_env.observation_space.shape[0],
        action_dim=enhanced_env.action_space.shape[0],
        config=agent_config
    )
    
    # 开始训练
    training_config = config.get('training', {})
    total_episodes = training_config.get('total_episodes', 400)
    save_frequency = training_config.get('save_frequency', 50)
    
    logger.info(f"开始训练: 总共 {total_episodes} 个episode")
    
    best_performance = {
        'episode': 0,
        'annual_return': -np.inf,
        'sharpe_ratio': -np.inf,
        'model_path': None
    }
    
    for episode in range(total_episodes):
        # 重置环境
        observation, info = enhanced_env.reset()
        episode_reward = 0.0
        episode_steps = 0
        
        while True:
            # 获取动作
            action, log_prob, value = enhanced_agent.get_action(observation)
            
            # 执行动作
            next_observation, reward, terminated, truncated, info = enhanced_env.step(action)
            
            # 存储经验
            enhanced_agent.store_experience(
                observation, action, reward, log_prob, value, terminated
            )
            
            episode_reward += reward
            episode_steps += 1
            observation = next_observation
            
            if terminated or truncated:
                break
        
        # 更新智能体
        if len(enhanced_agent.memory['states']) >= enhanced_agent.batch_size:
            update_info = enhanced_agent.update()
        
        # 定期保存和评估
        if (episode + 1) % save_frequency == 0:
            # 获取当前性能
            if 'performance_metrics' in info:
                annual_return = info['performance_metrics'].get('annualized_return', 0.0)
                sharpe_ratio = info['performance_metrics'].get('sharpe_ratio', 0.0)
                
                logger.info(
                    f"Episode {episode + 1}/{total_episodes}: "
                    f"奖励={episode_reward:.2f}, "
                    f"年化收益={annual_return:.2%}, "
                    f"夏普比率={sharpe_ratio:.2f}"
                )
                
                # 更新最佳模型
                if annual_return > best_performance['annual_return']:
                    best_performance.update({
                        'episode': episode + 1,
                        'annual_return': annual_return,
                        'sharpe_ratio': sharpe_ratio
                    })
                    
                    # 保存最佳模型
                    model_dir = config.get('model', {}).get('save_dir', './models/enhanced_csi300_2020_2022')
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = f"{model_dir}/best_enhanced_agent_episode_{episode + 1}.pth"
                    enhanced_agent.save_model(model_path)
                    best_performance['model_path'] = model_path
                    
                    logger.info(f"保存最佳模型: {model_path}, 年化收益: {annual_return:.2%}")
    
    # 训练完成
    logger.info("✅ 训练完成")
    logger.info(f"最佳性能: Episode {best_performance['episode']}, "
               f"年化收益: {best_performance['annual_return']:.2%}, "
               f"夏普比率: {best_performance['sharpe_ratio']:.2f}")
    
    return best_performance


def backtest_enhanced_system(config_file: str, model_path: str):
    """回测增强系统"""
    logger.info("📊 开始回测增强系统")
    
    # 加载配置
    config = load_enhanced_config(config_file)
    
    # 创建系统组件
    systems = create_enhanced_systems(config)
    
    # 准备回测数据
    price_data, factor_data = prepare_enhanced_data(systems['data_manager'], config)
    
    # 创建回测环境
    logger.info("创建回测环境...")
    backtest_env = EnhancedTradingEnvironment(
        factor_data=factor_data,
        price_data=price_data,
        config=config
    )
    
    # 加载训练好的模型
    logger.info(f"加载模型: {model_path}")
    enhanced_agent = StableCVaRPPOAgent(
        state_dim=backtest_env.observation_space.shape[0],
        action_dim=backtest_env.action_space.shape[0],
        config=config.get('agent', {})
    )
    enhanced_agent.load_model(model_path)
    
    # 运行回测
    logger.info("开始运行回测...")
    observation, info = backtest_env.reset()
    
    backtest_results = {
        'daily_returns': [],
        'portfolio_values': [],
        'actions': [],
        'rewards': []
    }
    
    step_count = 0
    while True:
        # 获取动作（确定性，不探索）
        action, _, _ = enhanced_agent.get_action(observation, deterministic=True)
        
        # 执行动作
        next_observation, reward, terminated, truncated, info = backtest_env.step(action)
        
        # 记录结果
        backtest_results['actions'].append(action.copy())
        backtest_results['rewards'].append(reward)
        
        if 'portfolio_value' in info:
            backtest_results['portfolio_values'].append(info['portfolio_value'])
        
        observation = next_observation
        step_count += 1
        
        # 定期输出进度
        if step_count % 50 == 0:
            current_return = info.get('cumulative_return', 0.0)
            current_drawdown = info.get('current_drawdown', 0.0)
            logger.info(f"回测进度: {step_count} 步, 累积收益: {current_return:.2%}, 回撤: {current_drawdown:.2%}")
        
        if terminated or truncated:
            break
    
    # 获取最终报告
    final_report = backtest_env.get_final_report()
    final_performance = final_report['final_performance']
    
    # 输出回测结果
    logger.info("📈 回测完成")
    logger.info("=" * 60)
    logger.info("增强系统回测报告")
    logger.info("=" * 60)
    
    print(f"\n【回测概要】")
    print(f"回测期间: {config['data']['start_date']} 至 {config['data']['end_date']}")
    print(f"回测步数: {step_count}")
    print(f"最终组合价值: {backtest_results['portfolio_values'][-1]:,.0f}" if backtest_results['portfolio_values'] else "N/A")
    
    print(f"\n【核心指标】")
    print(f"年化收益率: {final_performance.get('annualized_return', 0.0):.2%}")
    print(f"夏普比率: {final_performance.get('sharpe_ratio', 0.0):.2f}")
    print(f"最大回撤: {final_performance.get('max_drawdown', 0.0):.2%}")
    print(f"胜率: {final_performance.get('win_rate', 0.0):.1%}")
    print(f"卡尔玛比率: {final_performance.get('calmar_ratio', 0.0):.2f}")
    
    print(f"\n【目标达成评估】")
    annual_return = final_performance.get('annualized_return', 0.0)
    target_return = 0.08
    target_achieved = annual_return >= target_return
    
    print(f"目标年化收益: {target_return:.1%}")
    print(f"实际年化收益: {annual_return:.2%}")
    print(f"目标达成: {'✅ 是' if target_achieved else '❌ 否'}")
    print(f"目标完成度: {annual_return / target_return:.1%}")
    
    if target_achieved:
        print(f"🎉 恭喜！增强系统成功达到8%年化收益目标！")
    else:
        gap = target_return - annual_return
        print(f"📈 距离目标还需提升: {gap:.2%}")
    
    return final_performance


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强系统快速开始')
    parser.add_argument('--mode', choices=['train', 'backtest', 'full'], 
                       default='full', help='运行模式')
    parser.add_argument('--train_config', default='config_enhanced_train.yaml',
                       help='训练配置文件')
    parser.add_argument('--backtest_config', default='config_enhanced_backtest.yaml',
                       help='回测配置文件')
    parser.add_argument('--model', help='模型路径（回测模式需要）')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logger()
    
    logger.info("🚀 增强系统快速开始")
    
    best_model_path = None
    
    if args.mode in ['train', 'full']:
        logger.info("=" * 60)
        logger.info("第一步：训练增强系统 (2020-2022)")
        logger.info("=" * 60)
        
        best_performance = train_enhanced_system(args.train_config)
        best_model_path = best_performance['model_path']
        
        logger.info(f"训练完成，最佳模型: {best_model_path}")
    
    if args.mode in ['backtest', 'full']:
        logger.info("=" * 60)
        logger.info("第二步：回测增强系统 (2023)")
        logger.info("=" * 60)
        
        # 确定使用的模型路径
        if args.model:
            model_path = args.model
        elif best_model_path:
            model_path = best_model_path
        else:
            # 查找最新的模型
            model_dir = './models/enhanced_csi300_2020_2022'
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                if model_files:
                    model_path = os.path.join(model_dir, sorted(model_files)[-1])
                else:
                    raise ValueError("未找到可用的模型文件")
            else:
                raise ValueError("未找到模型目录，请先运行训练")
        
        logger.info(f"使用模型: {model_path}")
        
        # 运行回测
        backtest_results = backtest_enhanced_system(args.backtest_config, model_path)
        
        logger.info("回测完成")
    
    logger.info("✅ 增强系统快速开始完成")


if __name__ == "__main__":
    main()