#!/usr/bin/env python3
"""
奖励函数分析的红色阶段TDD测试
验证奖励函数是否过于稳定导致零回撤
"""

import pytest
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig


class TestRewardFunctionAnalysisRed:
    """测试奖励函数分析 - Red阶段"""
    
    def test_reward_function_amplification_effect(self):
        """Red: 测试奖励函数放大效应"""
        print("=== Red: 分析奖励函数的放大效应 ===")
        
        # 模拟实际观察到的数据
        portfolio_returns = [0.001740, 0.001741, 0.001740, 0.001742, 0.001739]  # 约0.174%
        market_returns = [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]  # 假设基准收益为0
        transaction_costs = [0.000001] * 5  # 很小的交易成本
        
        calculated_rewards = []
        
        for i in range(len(portfolio_returns)):
            port_ret = portfolio_returns[i]
            market_ret = market_returns[i]
            trans_cost = transaction_costs[i]
            
            # 复制奖励函数的核心逻辑
            excess_return = port_ret - market_ret
            transaction_cost_ratio = trans_cost
            alpha_reward = (excess_return - transaction_cost_ratio) * 1000  # 1000倍放大！
            
            # 假设其他奖励组件很小
            exploration_bonus = 1.0  # 假设固定值
            penalties = 0.5  # 假设固定的小惩罚
            
            final_reward = alpha_reward + exploration_bonus - penalties
            calculated_rewards.append(final_reward)
            
            print(f"组合收益: {port_ret:.6f}, 基准收益: {market_ret:.6f}")
            print(f"超额收益: {excess_return:.6f}, Alpha奖励: {alpha_reward:.4f}")
            print(f"最终奖励: {final_reward:.4f}")
            print()
        
        # 分析奖励的变异性
        reward_variance = np.var(calculated_rewards)
        reward_std = np.std(calculated_rewards)
        reward_range = max(calculated_rewards) - min(calculated_rewards)
        
        print(f"奖励方差: {reward_variance:.6f}")
        print(f"奖励标准差: {reward_std:.6f}")
        print(f"奖励范围: {reward_range:.6f}")
        print(f"平均奖励: {np.mean(calculated_rewards):.4f}")
        
        # 问题1：1000倍放大导致微小差异被巨大化
        print("❌ 问题1：1000倍放大因子导致：")
        print(f"  0.0001%的超额收益差异 -> {0.000001 * 1000:.2f}的奖励差异")
        
        # 问题2：如果基准收益始终为0或很小，超额收益会很稳定
        print("❌ 问题2：基准收益如果始终接近0，组合收益的小幅波动导致稳定的超额收益")
        
        # 问题3：固定的bonus和penalty使得奖励进一步稳定化
        print("❌ 问题3：固定的exploration_bonus和小惩罚进一步稳定化奖励")
        
        # 这个测试应该失败，显示奖励过于稳定的问题
        expected_min_variance = 1.0  # 期望最小方差
        assert reward_variance >= expected_min_variance, \
            f"奖励函数变异性过低 ({reward_variance:.6f})，导致训练回撤始终为0"
    
    def test_market_benchmark_return_analysis(self):
        """Red: 测试市场基准收益率分析"""
        print("=== Red: 分析市场基准收益率 ===")
        
        # 创建测试环境配置
        config = PortfolioConfig(
            stock_pool=["600519.SH", "600036.SH", "601318.SH"],
            initial_cash=1000000.0
        )
        
        # 创建模拟的基准数据
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        
        # 情况1：基准价格完全不变（异常情况）
        benchmark_data_static = pd.DataFrame({
            'close': [3000.0] * 10  # 完全不变
        }, index=dates)
        
        # 情况2：基准价格有微小变化
        benchmark_data_micro = pd.DataFrame({
            'close': [3000.0 + i * 0.001 for i in range(10)]  # 极微小变化
        }, index=dates)
        
        # 情况3：基准价格有正常变化
        benchmark_data_normal = pd.DataFrame({
            'close': [3000.0, 3015.0, 2995.0, 3020.0, 3010.0, 2980.0, 3030.0, 3005.0, 2990.0, 3025.0]
        }, index=dates)
        
        for name, benchmark_data in [
            ("静态基准", benchmark_data_static),
            ("微变基准", benchmark_data_micro),
            ("正常基准", benchmark_data_normal)
        ]:
            print(f"\n{name}基准收益率分析:")
            
            returns = []
            for i in range(1, len(benchmark_data)):
                current_price = benchmark_data.iloc[i]['close']
                previous_price = benchmark_data.iloc[i-1]['close']
                market_return = (current_price - previous_price) / previous_price
                returns.append(market_return)
                print(f"  第{i}期: {previous_price:.3f} -> {current_price:.3f}, 收益率: {market_return:.6f}")
            
            if returns:
                variance = np.var(returns)
                print(f"  基准收益率方差: {variance:.8f}")
                print(f"  基准收益率范围: {max(returns) - min(returns):.6f}")
                
                # 如果基准收益率变异性过低，会导致超额收益稳定
                if variance < 0.0001:
                    print(f"  ❌ {name}变异性过低，会导致超额收益过于稳定！")
    
    def test_realistic_market_scenario_should_produce_variable_rewards(self):
        """Red: 测试真实市场场景应该产生可变奖励"""
        print("=== Red: 验证真实市场场景应产生可变奖励 ===")
        
        # 模拟真实的市场数据
        realistic_portfolio_returns = [
            0.02,   # 2%收益（好日子）
            -0.01,  # -1%损失（坏日子）
            0.005,  # 0.5%小收益
            -0.015, # -1.5%较大损失
            0.03,   # 3%大收益
            0.001,  # 0.1%微小收益
            -0.02,  # -2%损失
            0.015,  # 1.5%收益
            -0.005, # -0.5%小损失
            0.008   # 0.8%收益
        ]
        
        realistic_market_returns = [
            0.015,  # 基准1.5%收益
            -0.008, # 基准-0.8%损失
            0.003,  # 基准0.3%收益
            -0.012, # 基准-1.2%损失
            0.025,  # 基准2.5%收益
            0.002,  # 基准0.2%收益
            -0.018, # 基准-1.8%损失
            0.010,  # 基准1.0%收益
            -0.003, # 基准-0.3%损失
            0.006   # 基准0.6%收益
        ]
        
        calculated_rewards = []
        
        for i in range(len(realistic_portfolio_returns)):
            port_ret = realistic_portfolio_returns[i]
            market_ret = realistic_market_returns[i]
            
            # 应用奖励函数逻辑
            excess_return = port_ret - market_ret
            transaction_cost_ratio = 0.0001  # 0.01%交易成本
            alpha_reward = (excess_return - transaction_cost_ratio) * 1000
            
            # 简化其他组件
            exploration_bonus = 1.0
            penalties = 0.5
            
            final_reward = alpha_reward + exploration_bonus - penalties
            calculated_rewards.append(final_reward)
            
            print(f"组合收益: {port_ret:7.3f}%, 基准: {market_ret:7.3f}%, "
                  f"超额: {excess_return:7.3f}%, 奖励: {final_reward:8.2f}")
        
        # 真实场景应该产生高变异性奖励
        reward_variance = np.var(calculated_rewards)
        reward_range = max(calculated_rewards) - min(calculated_rewards)
        
        print(f"\n真实场景奖励统计:")
        print(f"方差: {reward_variance:.2f}")
        print(f"范围: {reward_range:.2f}")
        print(f"最小奖励: {min(calculated_rewards):.2f}")
        print(f"最大奖励: {max(calculated_rewards):.2f}")
        
        # 真实场景应该有显著的奖励变异性
        assert reward_variance >= 100.0, f"真实市场场景应产生高变异性奖励，但方差只有 {reward_variance:.2f}"
        assert reward_range >= 20.0, f"真实市场场景应产生大范围奖励，但范围只有 {reward_range:.2f}"
        
        print("✅ 真实市场场景确实产生了高变异性奖励")
    
    def test_diagnosis_of_current_training_reward_stability(self):
        """Red: 诊断当前训练奖励稳定性的根本原因"""
        print("=== Red: 诊断当前训练奖励稳定性根本原因 ===")
        
        # 从日志观察到的数据
        observed_rewards = [174.12, 174.07, 174.12, 174.14, 173.94, 174.09, 173.94, 174.15, 174.15, 174.00]
        
        print("观察到的episode奖励:")
        for i, reward in enumerate(observed_rewards):
            print(f"  Episode {(i+1)*20}: {reward:.2f}")
        
        # 逆向工程：从最终奖励推断alpha_reward
        # final_reward ≈ alpha_reward + exploration_bonus - penalties
        # 假设 exploration_bonus ≈ 1.0, penalties ≈ 0.5
        estimated_alpha_rewards = [r - 1.0 + 0.5 for r in observed_rewards]  # r - bonus + penalty
        
        print("\n估算的Alpha奖励:")
        for i, alpha_reward in enumerate(estimated_alpha_rewards):
            print(f"  Episode {(i+1)*20}: {alpha_reward:.2f}")
        
        # 从alpha_reward推断超额收益
        # alpha_reward = (excess_return - transaction_cost_ratio) * 1000
        # 假设 transaction_cost_ratio ≈ 0.0001
        estimated_excess_returns = [(alpha + 0.0001) / 1000 for alpha in estimated_alpha_rewards]
        
        print("\n估算的超额收益率:")
        for i, excess_ret in enumerate(estimated_excess_returns):
            print(f"  Episode {(i+1)*20}: {excess_ret:.6f} ({excess_ret*100:.4f}%)")
        
        # 分析估算结果
        excess_variance = np.var(estimated_excess_returns)
        excess_std = np.std(estimated_excess_returns)
        excess_range = max(estimated_excess_returns) - min(estimated_excess_returns)
        
        print(f"\n超额收益率统计:")
        print(f"方差: {excess_variance:.8f}")
        print(f"标准差: {excess_std:.6f}")
        print(f"范围: {excess_range:.6f}")
        print(f"平均: {np.mean(estimated_excess_returns):.6f}")
        
        # 诊断结论
        print("\n❌ 诊断结论:")
        if excess_variance < 0.000001:
            print("1. 超额收益率变异性极低，说明：")
            print("   - 投资组合收益率极其稳定")
            print("   - 或基准收益率极其稳定")
            print("   - 或两者都极其稳定")
        
        if np.mean(estimated_excess_returns) > 0.001:
            print("2. 平均超额收益率较高，说明：")
            print("   - 基准收益率可能过低（接近0）")
            print("   - 或投资组合收益率被高估")
        
        print("3. 需要检查的问题：")
        print("   - 基准数据是否正确加载和计算")
        print("   - 投资组合收益率计算是否过于理想化")
        print("   - 奖励函数1000倍放大是否合理")
        
        # 这个测试应该揭示问题
        assert excess_variance >= 0.000001, \
            f"超额收益率变异性过低 ({excess_variance:.8f})，这是奖励稳定导致零回撤的根本原因"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])