"""
A股强化学习量化交易系统主程序
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager, get_default_config
from data import DataManager
from factors import FactorEngine
from rl_agent import TradingEnvironment, CVaRPPOAgent, SafetyShield
from risk_control import RiskController
from backtest import BacktestEngine


def setup_logging(config):
    """设置日志"""
    log_config = config.get('logging', {})
    log_dir = log_config.get('log_dir', './logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'trading_system_{timestamp}.log')
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger


def initialize_system(config_manager):
    """初始化系统组件"""
    logger = logging.getLogger(__name__)
    
    # 数据管理器
    data_config = config_manager.get_data_config()
    data_manager = DataManager(data_config)
    logger.info("数据管理器初始化完成")
    
    # 因子引擎
    factor_config = config_manager.get_config('factors')
    factor_engine = FactorEngine(factor_config)
    logger.info("因子引擎初始化完成")
    
    # 风险控制器
    risk_config = config_manager.get_risk_control_config()
    risk_controller = RiskController(risk_config)
    logger.info("风险控制器初始化完成")
    
    # 回测引擎
    backtest_config = config_manager.get_backtest_config()
    backtest_engine = BacktestEngine(backtest_config)
    logger.info("回测引擎初始化完成")
    
    return {
        'data_manager': data_manager,
        'factor_engine': factor_engine,
        'risk_controller': risk_controller,
        'backtest_engine': backtest_engine
    }


def prepare_data(data_manager, factor_engine, config):
    """准备训练和回测数据"""
    logger = logging.getLogger(__name__)
    
    data_config = config.get_data_config()
    
    # 获取股票数据
    logger.info("正在获取股票数据...")
    stock_data = data_manager.get_stock_data(
        start_time=data_config['start_date'],
        end_time=data_config['end_date']
    )
    
    if stock_data.empty:
        raise ValueError("未能获取到股票数据")
    
    # 获取价格数据
    price_data = stock_data['$close'].unstack()
    logger.info(f"价格数据形状: {price_data.shape}")
    
    # 筛选低波动股票池
    low_vol_stocks = factor_engine.filter_low_volatility_universe(
        price_data,
        threshold=config.get_value('factors.low_vol_threshold', 0.2),
        window=config.get_value('factors.low_vol_window', 60)
    )
    
    # 过滤价格数据
    if low_vol_stocks:
        price_data = price_data[low_vol_stocks]
        logger.info(f"筛选后股票数量: {len(low_vol_stocks)}")
    
    # 计算因子
    logger.info("正在计算因子...")
    factor_data = factor_engine.calculate_all_factors(price_data)
    
    if factor_data.empty:
        raise ValueError("未能计算出因子数据")
    
    logger.info(f"因子数据形状: {factor_data.shape}")
    
    return price_data, factor_data


def train_agent(price_data, factor_data, config_manager, systems):
    """训练强化学习智能体"""
    logger = logging.getLogger(__name__)
    logger.info("开始训练强化学习智能体...")
    
    # 创建交易环境
    env_config = config_manager.get_environment_config()
    environment = TradingEnvironment(factor_data, price_data, env_config)
    
    # 创建智能体
    agent_config = config_manager.get_agent_config()
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.shape[0]
    agent = CVaRPPOAgent(state_dim, action_dim, agent_config)
    
    # 创建安全保护层
    shield_config = config_manager.get_config('safety_shield')
    safety_shield = SafetyShield(shield_config)
    
    # 训练配置
    training_config = config_manager.get_training_config()
    total_episodes = training_config['total_episodes']
    update_frequency = training_config['update_frequency']
    save_frequency = training_config['save_frequency']
    
    # 训练循环
    best_sharpe = float('-inf')
    episode_rewards = []
    
    for episode in range(total_episodes):
        state, info = environment.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            # 获取动作
            action, log_prob, value, cvar_estimate = agent.get_action(state)
            
            # 应用安全保护
            safe_action = safety_shield.shield_action(
                action, info, price_data, environment.portfolio_weights
            )
            
            # 执行动作
            next_state, reward, terminated, truncated, info = environment.step(safe_action)
            
            # 存储经验
            agent.store_transition(state, safe_action, reward, value, log_prob, terminated or truncated, cvar_estimate)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        
        # 定期更新智能体
        if episode % update_frequency == 0 and episode > 0:
            update_stats = agent.update()
            logger.info(f"Episode {episode}, 平均奖励: {np.mean(episode_rewards[-update_frequency:]):.4f}")
        
        # 评估和保存模型
        if episode % save_frequency == 0 and episode > 0:
            # 简单评估
            current_sharpe = info.get('sharpe_ratio', 0)
            
            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe
                model_dir = config_manager.get_value('model.save_dir', './models')
                os.makedirs(model_dir, exist_ok=True)
                
                model_path = os.path.join(model_dir, f"best_agent_episode_{episode}.pth")
                agent.save_model(model_path)
                logger.info(f"保存最佳模型: {model_path}, Sharpe: {current_sharpe:.4f}")
    
    logger.info("训练完成")
    return agent, environment, safety_shield


def run_backtest(agent, environment, safety_shield, config_manager, systems):
    """运行回测"""
    logger = logging.getLogger(__name__)
    logger.info("开始运行回测...")
    
    backtest_engine = systems['backtest_engine']
    risk_controller = systems['risk_controller']
    
    # 回测期间
    backtest_config = config_manager.get_backtest_config()
    start_date = config_manager.get_value('data.start_date')
    end_date = config_manager.get_value('data.end_date')
    
    # 执行回测
    results = backtest_engine.run_backtest(
        agent=agent,
        env=environment,
        start_date=start_date,
        end_date=end_date
    )
    
    # 生成报告
    report = backtest_engine.generate_backtest_report(results)
    logger.info("回测报告:\n" + report)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, f'backtest_results_{timestamp}.pkl')
    backtest_engine.save_results(results, results_path)
    
    logger.info("回测完成")
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='A股强化学习量化交易系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'backtest', 'full'], 
                       default='full', help='运行模式')
    parser.add_argument('--model', type=str, help='加载预训练模型路径')
    
    args = parser.parse_args()
    
    # 初始化配置
    config_manager = ConfigManager(args.config)
    
    if not config_manager.validate_config():
        print("配置验证失败，程序退出")
        sys.exit(1)
    
    # 设置日志
    logger = setup_logging(config_manager.config)
    
    # 打印配置摘要
    config_manager.print_config_summary()
    
    # 初始化系统
    systems = initialize_system(config_manager)
    
    # 准备数据
    price_data, factor_data = prepare_data(
        systems['data_manager'], 
        systems['factor_engine'], 
        config_manager
    )
    
    agent = None
    environment = None
    safety_shield = None
    
    # 根据模式执行相应操作
    if args.mode in ['train', 'full']:
        agent, environment, safety_shield = train_agent(
            price_data, factor_data, config_manager, systems
        )
    
    if args.mode in ['backtest', 'full']:
        if agent is None and args.model:
            # 加载预训练模型
            logger.info(f"加载预训练模型: {args.model}")
            env_config = config_manager.get_environment_config()
            environment = TradingEnvironment(factor_data, price_data, env_config)
            
            agent_config = config_manager.get_agent_config()
            state_dim = environment.observation_space.shape[0]
            action_dim = environment.action_space.shape[0]
            agent = CVaRPPOAgent(state_dim, action_dim, agent_config)
            agent.load_model(args.model)
            
            shield_config = config_manager.get_config('safety_shield')
            safety_shield = SafetyShield(shield_config)
        
        if agent is not None:
            results = run_backtest(agent, environment, safety_shield, config_manager, systems)
        else:
            logger.error("未找到可用的智能体模型")
            sys.exit(1)
    
    logger.info("程序执行完成")


if __name__ == "__main__":
    main()