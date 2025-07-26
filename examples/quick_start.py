"""
快速开始示例
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager
from data import DataManager
from factors import FactorEngine
from rl_agent import TradingEnvironment, CVaRPPOAgent, SafetyShield
from risk_control import RiskController
from backtest import BacktestEngine
from utils import setup_logger
import pandas as pd
import numpy as np


def quick_start_demo():
    """快速开始演示"""
    print("=== A股强化学习量化交易系统快速演示 ===\n")
    
    # 1. 初始化配置
    print("1. 初始化配置管理器...")
    config_manager = ConfigManager()
    config_manager.print_config_summary()
    
    # 2. 设置日志
    logger = setup_logger('quick_start')
    logger.info("快速演示开始")
    
    # 3. 创建模拟数据（用于演示）
    print("\n2. 创建模拟数据...")
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    n_stocks = 10
    stock_names = [f'stock_{i:03d}' for i in range(n_stocks)]
    
    # 模拟价格数据
    np.random.seed(42)
    price_data = pd.DataFrame(
        index=dates,
        columns=stock_names,
        data=100 + np.cumsum(np.random.normal(0, 0.02, (len(dates), n_stocks)), axis=0)
    )
    
    # 模拟因子数据
    factor_data = pd.DataFrame(
        index=dates,
        columns=['momentum', 'volatility', 'volume', 'trend'],
        data=np.random.normal(0, 1, (len(dates), 4))
    )
    
    print(f"价格数据形状: {price_data.shape}")
    print(f"因子数据形状: {factor_data.shape}")
    
    # 4. 创建交易环境
    print("\n3. 创建强化学习交易环境...")
    env_config = config_manager.get_environment_config()
    environment = TradingEnvironment(factor_data, price_data, env_config)
    
    print(f"状态空间维度: {environment.observation_space.shape}")
    print(f"动作空间维度: {environment.action_space.shape}")
    
    # 5. 创建智能体
    print("\n4. 创建CVaR-PPO智能体...")
    agent_config = config_manager.get_agent_config()
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.shape[0]
    agent = CVaRPPOAgent(state_dim, action_dim, agent_config)
    
    print(f"网络隐藏层维度: {agent.hidden_dim}")
    print(f"学习率: {agent.lr}")
    
    # 6. 创建安全保护层
    print("\n5. 创建安全保护层...")
    shield_config = config_manager.get_config('safety_shield')
    safety_shield = SafetyShield(shield_config)
    
    # 7. 简单训练演示
    print("\n6. 运行简单训练演示...")
    n_episodes = 5
    
    for episode in range(n_episodes):
        state, info = environment.reset()
        episode_reward = 0
        step_count = 0
        
        while step_count < 50:  # 限制步数用于演示
            # 获取动作
            action, log_prob, value, cvar_estimate = agent.get_action(state)
            
            # 应用安全保护
            safe_action = safety_shield.shield_action(action, info)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = environment.step(safe_action)
            
            # 存储经验
            agent.store_transition(state, safe_action, reward, value, log_prob, terminated or truncated, cvar_estimate)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode + 1}: 奖励 = {episode_reward:.4f}, 净值 = {info.get('portfolio_value', 1.0):.4f}")
    
    # 8. 简单回测演示
    print("\n7. 运行简单回测演示...")
    backtest_config = config_manager.get_backtest_config()
    backtest_engine = BacktestEngine(backtest_config)
    
    # 模拟回测
    results = backtest_engine.run_backtest(
        agent=agent,
        env=environment,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    
    # 生成报告
    report = backtest_engine.generate_backtest_report(results)
    print("\n=== 回测报告 ===")
    print(report)
    
    # 9. 风险控制演示
    print("\n8. 风险控制演示...")
    risk_config = config_manager.get_risk_control_config()
    risk_controller = RiskController(risk_config)
    
    # 模拟权重处理
    raw_weights = np.random.uniform(-0.05, 0.05, n_stocks)
    processed_weights = risk_controller.process_weights(
        raw_weights, price_data, 1.0, {'max_drawdown': 0.02}
    )
    
    print(f"原始权重范围: [{raw_weights.min():.4f}, {raw_weights.max():.4f}]")
    print(f"处理后权重范围: [{processed_weights.min():.4f}, {processed_weights.max():.4f}]")
    print(f"总杠杆: {np.sum(np.abs(processed_weights)):.4f}")
    
    # 10. 完成
    print("\n=== 快速演示完成 ===")
    print("系统各模块运行正常！")
    print("您可以使用以下命令运行完整系统:")
    print("python main.py --mode train  # 训练模式")
    print("python main.py --mode backtest --model path/to/model.pth  # 回测模式")
    print("python main.py --mode full  # 完整流程")


if __name__ == "__main__":
    quick_start_demo()