"""
A股强化学习量化交易系统主程序
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pickle
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


def setup_logging(config, mode='train', verbose=False):
    """设置日志"""
    log_config = config.get('logging', {})
    log_dir = log_config.get('log_dir', './logs')
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 训练模式使用更安静的日志级别（除非显式启用verbose）
    if mode == 'train' and not verbose:
        base_level = logging.INFO  # 保持INFO级别以显示进度
        # 只抑制特定的噪音日志
        noisy_loggers = [
            'rl_agent.safety_shield', 
            'rl_agent.cvar_ppo_agent'
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        # 保持重要进度日志可见
        logging.getLogger('__main__').setLevel(logging.INFO)
    else:
        base_level = getattr(logging, log_config.get('level', 'INFO'))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'trading_system_{timestamp}.log')
    
    logging.basicConfig(
        level=base_level,
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    if mode == 'train' and not verbose:
        logger.warning(f"训练模式 - 使用简化日志输出，详细日志请查看: {log_file}")
        logger.warning("提示: 使用 --verbose 或 -v 参数启用详细日志输出")
    else:
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


def prepare_data(data_manager, factor_engine, config, mode='train'):
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
    price_data = price_data.T
    logger.info(f"价格数据形状: {price_data.shape}")
    
    # 根据模式处理股票池
    if mode == 'backtest':
        # 回测模式：加载训练时保存的股票池
        import pickle
        try:
            with open('./models/selected_stocks.pkl', 'rb') as f:
                selected_stocks = pickle.load(f)
            logger.info(f"已加载训练时的股票池，包含 {len(selected_stocks)} 只股票")
            # 过滤价格数据到相同的股票池
            available_stocks = [s for s in selected_stocks if s in price_data.columns]
            if len(available_stocks) < len(selected_stocks):
                logger.warning(f"部分股票在回测期间不可用，使用 {len(available_stocks)} 只股票")
            
            # 如果可用股票太少，直接报错退出
            if len(available_stocks) == 0:
                logger.error("数据不匹配：训练时使用的股票池在回测期间完全不可用！")
                logger.error("请检查数据源或重新训练模型。")
                sys.exit(1)
            
            price_data = price_data[available_stocks]
            low_vol_stocks = available_stocks
        except FileNotFoundError:
            logger.warning("未找到训练时的股票池文件，重新筛选")
            # 如果没有保存的股票池，则重新筛选
            low_vol_stocks = factor_engine.filter_low_volatility_universe(
                price_data,
                threshold=config.get_value('factors.low_vol_threshold', 0.5),
                window=config.get_value('factors.low_vol_window', 60)
            )
            if low_vol_stocks:
                price_data = price_data[low_vol_stocks]
            else:
                price_data = price_data.iloc[:, :100]
                low_vol_stocks = list(price_data.columns)
    else:
        # 训练模式：正常筛选低波动股票池
        logger.info("正在筛选低波动股票池...")
        low_vol_stocks = factor_engine.filter_low_volatility_universe(
            price_data,
            threshold=config.get_value('factors.low_vol_threshold', 0.5),  # 放宽到50%
            window=config.get_value('factors.low_vol_window', 60)
        )
        logger.info("股票池筛选完成")
        
        # 过滤价格数据
        if low_vol_stocks:
            price_data = price_data[low_vol_stocks]
            logger.info(f"筛选后股票数量: {len(low_vol_stocks)}")
            # 保存股票池信息供回测使用
            selected_stocks = low_vol_stocks
        else:
            # 如果没有筛选出股票，使用前100只股票
            logger.warning("低波动筛选未找到股票，使用前100只股票")
            price_data = price_data.iloc[:, :100]
            selected_stocks = list(price_data.columns)
            logger.info(f"使用前100只股票: {price_data.shape[1]}")
        
        # 保存股票池到文件
        import pickle
        os.makedirs('./models', exist_ok=True)
        with open('./models/selected_stocks.pkl', 'wb') as f:
            pickle.dump(selected_stocks, f)
        logger.info(f"已保存股票池信息到 ./models/selected_stocks.pkl")
    
    # 计算因子
    logger.info("正在计算因子...")
    # 从原始股票数据中提取成交量数据
    logger.info("正在提取成交量数据...")
    volume_data = stock_data['$volume'].unstack()
    volume_data = volume_data.T
    # 使用相同的股票池过滤成交量数据
    volume_data = volume_data[low_vol_stocks]
    
    logger.info("正在计算技术因子（预计需要10-30秒）...")
    factor_data = factor_engine.calculate_all_factors(price_data, volume_data)
    
    if isinstance(factor_data.index, pd.MultiIndex):
        # The factor engine now returns a correctly shaped dataframe
        # We just need to align it with the price data
        factor_data = factor_data.unstack()
        aligned_price, aligned_factor = price_data.align(factor_data, join='inner', axis=0)
        price_data = aligned_price
        factor_data = aligned_factor.stack().reorder_levels(['datetime', 'instrument'])
    
    logger.info(f"因子计算完成! 因子数据形状: {factor_data.shape}")
    
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
    
    logger.info(f"开始训练: 总共 {total_episodes} 个episode")
    
    for episode in range(total_episodes):
        # 定期显示进度
        if episode % 10 == 0:
            logger.info(f"训练进度: Episode {episode}/{total_episodes} ({episode/total_episodes*100:.1f}%)")
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
                
                # 确保路径是绝对路径
                if not os.path.isabs(model_dir):
                    # 获取main.py文件所在的目录作为项目根目录
                    project_root = os.path.dirname(os.path.abspath(__file__))
                    model_dir = os.path.join(project_root, model_dir)

                os.makedirs(model_dir, exist_ok=True)
                
                model_path = os.path.join(model_dir, f"best_agent_episode_{episode}.pth")
                agent.save_model(model_path)
                
                # 增加验证步骤
                if os.path.exists(model_path):
                    logger.info(f"成功保存并验证模型: {model_path}, Sharpe: {current_sharpe:.4f}")
                else:
                    logger.error(f"严重错误: 模型保存失败，文件未找到: {model_path}")
    
    # 生成训练统计报告
    logger.warning("=" * 60)
    logger.warning("训练完成统计报告")
    logger.warning("=" * 60)
    
    # 安全保护层统计
    if hasattr(safety_shield, 'get_risk_event_summary'):
        risk_summary = safety_shield.get_risk_event_summary()
        logger.warning(f"安全保护层统计:\n{risk_summary}")
    
    # 智能体数值稳定性统计  
    if hasattr(agent, 'get_numerical_issues_summary'):
        numerical_summary = agent.get_numerical_issues_summary()
        logger.warning(f"智能体数值稳定性:\n{numerical_summary}")
    
    logger.warning("=" * 60)
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
        end_date=end_date,
        safety_shield=safety_shield
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
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='启用详细日志输出（训练模式下默认使用简化日志）')
    
    args = parser.parse_args()
    
    # 初始化配置
    config_manager = ConfigManager(args.config)
    
    if not config_manager.validate_config():
        print("配置验证失败，程序退出")
        sys.exit(1)
    
    # 设置日志
    logger = setup_logging(config_manager.config, args.mode, args.verbose)
    
    # 打印配置摘要
    config_manager.print_config_summary()
    
    # 初始化系统
    systems = initialize_system(config_manager)
    
    # 准备数据
    price_data, factor_data = prepare_data(
        systems['data_manager'], 
        systems['factor_engine'], 
        config_manager,
        args.mode  # 传递模式参数
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
            
            # 加载训练时的股票池
            with open('./models/selected_stocks.pkl', 'rb') as f:
                train_universe = pickle.load(f)

            env_config = config_manager.get_environment_config()
            environment = TradingEnvironment(factor_data, price_data, env_config, train_universe=train_universe)
            
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