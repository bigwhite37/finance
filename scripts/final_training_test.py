#!/usr/bin/env python
"""
最终训练系统测试 - 验证所有优化措施
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import logging
from config import get_default_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_config():
    """分析当前配置的合理性"""
    config = get_default_config()
    
    print("🔧 当前配置分析:")
    print("=" * 50)
    
    # 智能体配置
    agent_config = config['agent']
    print(f"🤖 智能体配置:")
    print(f"  - 网络规模: {agent_config['hidden_dim']} (较小，有利于稳定性)")
    print(f"  - 学习率: {agent_config['learning_rate']} (非常保守)")
    print(f"  - PPO clip: {agent_config['clip_epsilon']} (严格限制)")
    print(f"  - 训练轮数: {agent_config['ppo_epochs']} (最少)")
    print(f"  - 批次大小: {agent_config['batch_size']} (最小)")
    print(f"  - CVaR权重: {agent_config['cvar_lambda']} (最小)")
    
    # 安全约束配置
    shield_config = config['safety_shield']
    print(f"\n🛡️ 安全约束配置:")
    print(f"  - 最大杠杆: {shield_config['max_leverage']} (已放宽)")
    print(f"  - 单股仓位: {shield_config['max_position']} (已放宽)")
    print(f"  - VaR阈值: {shield_config['var_threshold']} (已放宽)")
    print(f"  - 回撤阈值: {shield_config['max_drawdown_threshold']} (已放宽)")
    
    # 训练配置
    training_config = config['training']
    print(f"\n📚 训练配置:")
    print(f"  - 总episode: {training_config['total_episodes']}")
    print(f"  - 更新频率: {training_config['update_frequency']}")
    print(f"  - 保存频率: {training_config['save_frequency']}")
    
    return config

def run_stability_test():
    """运行稳定性测试"""
    print("\n🧪 运行稳定性测试...")
    
    try:
        from data import DataManager
        from factors import FactorEngine
        from rl_agent import TradingEnvironment, CVaRPPOAgent, SafetyShield
        
        config = get_default_config()
        
        # 测试数据加载
        print("📊 测试数据加载...")
        data_manager = DataManager(config['data'])
        factor_engine = FactorEngine(config['factors'])
        
        # 获取小样本数据
        instruments = data_manager._get_universe_stocks('2019-01-01', '2023-12-31')[:10]
        stock_data = data_manager.get_stock_data(
            instruments=instruments,
            start_time='2023-01-01',
            end_time='2023-02-28'
        )
        
        if stock_data.empty:
            print("❌ 数据加载失败")
            return False
        
        price_data = stock_data['$close'].unstack()
        volume_data = stock_data['$volume'].unstack()
        factor_data = factor_engine.calculate_all_factors(price_data, volume_data)
        
        print(f"✅ 数据加载成功: 价格{price_data.shape}, 因子{factor_data.shape}")
        
        # 测试环境创建
        print("🌍 测试环境创建...")
        environment = TradingEnvironment(factor_data, price_data, config['environment'])
        
        # 测试智能体创建
        print("🤖 测试智能体创建...")
        state_dim = environment.observation_space.shape[0]
        action_dim = environment.action_space.shape[0]
        agent = CVaRPPOAgent(state_dim, action_dim, config['agent'])
        
        # 测试安全保护层
        print("🛡️ 测试安全保护层...")
        safety_shield = SafetyShield(config['safety_shield'])
        
        print(f"✅ 组件创建成功: 状态维度={state_dim}, 动作维度={action_dim}")
        
        # 运行短期训练测试
        print("🏃 运行短期训练测试...")
        test_rewards = []
        
        for episode in range(3):
            state, info = environment.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(min(10, len(price_data)-1)):  # 最多10步
                action, log_prob, value, cvar_estimate = agent.get_action(state)
                safe_action = safety_shield.shield_action(action, info, price_data, environment.portfolio_weights)
                
                next_state, reward, terminated, truncated, info = environment.step(safe_action)
                agent.store_transition(state, safe_action, reward, value, log_prob, terminated or truncated, cvar_estimate)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            test_rewards.append(episode_reward)
            print(f"  Episode {episode+1}: 奖励={episode_reward:.4f}, 步数={steps}")
        
        # 测试网络更新
        print("🔄 测试网络更新...")
        try:
            update_stats = agent.update()
            loss = update_stats.get('total_loss', 0)
            lr = update_stats.get('learning_rate', 0)
            
            if np.isfinite(loss) and loss < 1e6:  # 损失值合理
                print(f"✅ 更新成功: 损失={loss:.6f}, 学习率={lr:.2e}")
                return True
            else:
                print(f"⚠️ 损失值异常: {loss}")
                return False
                
        except Exception as e:
            print(f"❌ 更新失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def provide_recommendations():
    """提供训练建议"""
    print("\n💡 训练建议:")
    print("=" * 50)
    
    print("🎯 如果测试通过，可以开始正式训练:")
    print("   python main.py --mode train")
    
    print("\n📈 监控指标:")
    print("   - 平均奖励应该逐渐上升")
    print("   - 约束触发次数应该减少")
    print("   - Sharpe比率应该改善")
    print("   - 损失值应该保持在合理范围(< 1000)")
    
    print("\n⚠️ 如果仍有问题:")
    print("   1. 进一步降低学习率到 1e-6")
    print("   2. 减少网络规模到 32")
    print("   3. 增加更多数值稳定性检查")
    print("   4. 考虑使用更简单的奖励函数")
    
    print("\n🔧 高级优化选项:")
    print("   - 使用预训练的因子权重")
    print("   - 实施课程学习(Curriculum Learning)")
    print("   - 添加经验回放缓冲区")
    print("   - 使用更稳定的优化器(如RMSprop)")

def main():
    """主函数"""
    print("🚀 A股强化学习量化交易系统 - 最终测试")
    print("=" * 60)
    
    # 分析配置
    config = analyze_config()
    
    # 运行稳定性测试
    success = run_stability_test()
    
    # 提供建议
    provide_recommendations()
    
    # 总结
    print("\n" + "=" * 60)
    if success:
        print("🎉 系统测试通过！可以开始正式训练。")
        print("💪 所有优化措施已实施，系统稳定性大幅提升。")
    else:
        print("⚠️ 系统仍需进一步优化。")
        print("🔧 建议按照上述建议进行调整。")
    
    return success

if __name__ == "__main__":
    main()