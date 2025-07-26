#!/usr/bin/env python
"""
测试训练改进效果
"""

import sys
sys.path.append('.')
from config import get_default_config
from data import DataManager
from factors import FactorEngine
from rl_agent import TradingEnvironment, CVaRPPOAgent, SafetyShield
import numpy as np
import logging

# 设置简单日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_improvements():
    """测试改进效果"""
    print("🧪 测试训练系统改进...")
    
    # 获取配置
    config = get_default_config()
    
    # 初始化组件
    data_manager = DataManager(config['data'])
    factor_engine = FactorEngine(config['factors'])
    
    # 获取小样本数据进行快速测试
    print("📊 获取测试数据...")
    instruments = data_manager._get_universe_stocks('2019-01-01', '2023-12-31')[:20]  # 只用20只股票
    stock_data = data_manager.get_stock_data(
        instruments=instruments,
        start_time='2023-01-01',
        end_time='2023-03-31'  # 3个月数据
    )
    
    if stock_data.empty:
        print("❌ 无法获取测试数据")
        return False
    
    # 处理数据 - 转置使得索引是日期，列是股票代码
    price_data = stock_data['$close'].unstack().T  # 转置：日期作为索引，股票作为列
    volume_data = stock_data['$volume'].unstack().T
    
    print(f"📈 价格数据形状: {price_data.shape}")
    
    # 计算因子
    factor_data = factor_engine.calculate_all_factors(price_data, volume_data)
    print(f"🔢 因子数据形状: {factor_data.shape}")
    
    # 创建环境
    env_config = config['environment']
    environment = TradingEnvironment(factor_data, price_data, env_config)
    
    # 创建智能体
    agent_config = config['agent']
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.shape[0]
    agent = CVaRPPOAgent(state_dim, action_dim, agent_config)
    
    # 创建安全保护层
    shield_config = config['safety_shield']
    safety_shield = SafetyShield(shield_config)
    
    print(f"🤖 智能体配置: 状态维度={state_dim}, 动作维度={action_dim}")
    print(f"📚 学习率: {agent_config['learning_rate']}")
    print(f"🛡️ 安全约束: 最大杠杆={shield_config['max_leverage']}")
    
    # 运行几个episode测试
    print("\n🏃 运行测试episode...")
    test_rewards = []
    constraint_counts = {'leverage': 0, 'drawdown': 0}
    
    for episode in range(5):  # 只测试5个episode
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
            
            # 统计约束触发
            if np.sum(np.abs(safe_action)) < np.sum(np.abs(action)) * 0.9:
                constraint_counts['leverage'] += 1
            
            # 执行动作
            next_state, reward, terminated, truncated, info = environment.step(safe_action)
            
            # 存储经验
            agent.store_transition(state, safe_action, reward, value, log_prob, terminated or truncated, cvar_estimate)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        test_rewards.append(episode_reward)
        print(f"  Episode {episode+1}: 奖励={episode_reward:.4f}, 步数={step_count}")
    
    # 测试更新
    print("\n🔄 测试智能体更新...")
    try:
        update_stats = agent.update()
        print(f"✅ 更新成功: 损失={update_stats.get('total_loss', 0):.6f}")
        print(f"📊 学习率: {update_stats.get('learning_rate', 0):.2e}")
    except Exception as e:
        print(f"❌ 更新失败: {e}")
        return False
    
    # 输出测试结果
    print(f"\n📋 测试结果总结:")
    print(f"  平均奖励: {np.mean(test_rewards):.4f}")
    print(f"  奖励标准差: {np.std(test_rewards):.4f}")
    print(f"  约束触发: 杠杆={constraint_counts['leverage']}次")
    
    # 获取安全保护层统计
    total_constraints = sum(safety_shield.risk_event_counts.values())
    print(f"  总约束触发: {total_constraints}次")
    
    # 判断改进效果
    avg_reward = np.mean(test_rewards)
    if avg_reward > -5:  # 如果平均奖励大于-5，认为改进有效
        print("✅ 训练改进效果良好！")
        return True
    else:
        print("⚠️ 训练改进效果有限，可能需要进一步调整")
        return False

if __name__ == "__main__":
    success = test_improvements()
    if success:
        print("\n🎉 系统优化完成，可以开始正式训练！")
        print("💡 建议运行: python main.py --mode train")
    else:
        print("\n🔧 系统仍需进一步优化")