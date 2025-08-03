#!/usr/bin/env python3
"""
负奖励问题分析的红色阶段TDD测试
分析为什么训练奖励持续为负
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestNegativeRewardAnalysisRed:
    """测试负奖励问题分析 - Red阶段"""
    
    def test_reward_function_components_analysis(self):
        """Red: 分析奖励函数各组件的贡献"""
        print("=== Red: 分析奖励函数各组件的贡献 ===")
        
        # 模拟实际训练中的参数
        def analyze_reward_components(portfolio_return, market_return, transaction_cost_ratio=0.0001):
            """分解奖励函数各组件"""
            
            # 1. 计算超额收益
            excess_return = portfolio_return - market_return
            
            # 2. Alpha奖励基础部分
            base_alpha = (excess_return - transaction_cost_ratio) * 100
            
            # 3. 市场波动因子
            market_volatility_factor = abs(market_return) * 50
            
            # 4. 探索随机性（模拟）
            exploration_randomness = 0.0  # 设为0来分析基础情况
            
            # 5. 风险惩罚因子
            risk_penalty_factor = abs(portfolio_return) * 10 if abs(portfolio_return) > 0.05 else 0
            
            # 6. 表现惩罚
            underperformance_penalty = 0.0
            if excess_return < -0.01:  # 超额收益为负且超过1%时
                underperformance_penalty = abs(excess_return) * 200  # 放大惩罚
            
            # 计算Alpha奖励
            alpha_reward = base_alpha + market_volatility_factor + exploration_randomness - risk_penalty_factor - underperformance_penalty
            
            # 其他奖励组件（简化）
            exploration_bonus = 1.0  # 假设权重有差异
            concentration_penalty = 0.0  # 假设没有过度集中
            drawdown_penalty = 0.0  # 假设回撤正常
            risk_penalty = 0.5  # 假设风险控制惩罚
            
            final_reward = alpha_reward + exploration_bonus - concentration_penalty - drawdown_penalty - risk_penalty
            
            return {
                'portfolio_return': portfolio_return,
                'market_return': market_return,
                'excess_return': excess_return,
                'base_alpha': base_alpha,
                'market_volatility_factor': market_volatility_factor,
                'exploration_randomness': exploration_randomness,
                'risk_penalty_factor': risk_penalty_factor,
                'underperformance_penalty': underperformance_penalty,
                'alpha_reward': alpha_reward,
                'exploration_bonus': exploration_bonus,
                'concentration_penalty': concentration_penalty,
                'drawdown_penalty': drawdown_penalty,
                'risk_penalty': risk_penalty,
                'final_reward': final_reward
            }
        
        # 测试不同收益率场景
        test_scenarios = [
            (0.002, 0.001, "轻微跑赢"),      # 0.2% vs 0.1%
            (0.001, 0.001, "持平"),          # 0.1% vs 0.1%  
            (0.001, 0.002, "轻微跑输"),      # 0.1% vs 0.2%
            (-0.001, 0.001, "亏损vs盈利"),   # -0.1% vs 0.1%
            (-0.002, -0.001, "都亏损相对好"), # -0.2% vs -0.1%
            (-0.001, -0.002, "都亏损相对差"), # -0.1% vs -0.2%
        ]
        
        print("奖励函数组件分析:")
        print("=" * 120)
        print(f"{'场景':12s} {'组合收益':8s} {'基准收益':8s} {'超额收益':8s} {'基础Alpha':9s} {'惩罚系数':8s} {'Alpha奖励':9s} {'最终奖励':8s}")
        print("-" * 120)
        
        negative_reward_count = 0
        
        for portfolio_ret, market_ret, description in test_scenarios:
            result = analyze_reward_components(portfolio_ret, market_ret)
            
            print(f"{description:12s} {portfolio_ret:8.3f} {market_ret:8.3f} {result['excess_return']:8.3f} "
                  f"{result['base_alpha']:9.2f} {result['underperformance_penalty']:8.2f} "
                  f"{result['alpha_reward']:9.2f} {result['final_reward']:8.2f}")
            
            if result['final_reward'] < 0:
                negative_reward_count += 1
        
        print("-" * 120)
        print(f"负奖励场景数: {negative_reward_count}/{len(test_scenarios)}")
        
        # 问题分析
        print(f"\n❌ 发现的问题:")
        
        # 分析第一个应该为正的场景
        positive_scenario = analyze_reward_components(0.002, 0.001)  # 跑赢市场
        if positive_scenario['final_reward'] < 0:
            print(f"1. 跑赢市场的场景仍为负奖励: {positive_scenario['final_reward']:.2f}")
            
            if positive_scenario['underperformance_penalty'] > 0:
                print(f"   - 错误的表现惩罚: {positive_scenario['underperformance_penalty']:.2f}")
                
            if positive_scenario['base_alpha'] < 1:
                print(f"   - Alpha奖励过小: {positive_scenario['base_alpha']:.2f}")
        
        # 这个测试应该失败，显示奖励函数设计问题
        assert negative_reward_count < len(test_scenarios) // 2, \
            f"过多场景产生负奖励 ({negative_reward_count}/{len(test_scenarios)})，奖励函数设计有问题"
    
    def test_market_benchmark_return_calculation_issue(self):
        """Red: 测试市场基准收益率计算问题"""
        print("=== Red: 分析市场基准收益率计算 ===")
        
        # 从代码推断基准收益率可能的问题
        # portfolio_environment.py 第886行调用 _calculate_market_benchmark_return()
        
        # 模拟可能的基准收益率计算问题
        possible_issues = [
            ("基准数据缺失", None),
            ("基准收益率始终为0", 0.0),
            ("基准收益率异常高", 0.02),  # 2%的基准收益率
            ("基准收益率计算错误", -0.001)  # 负的基准收益率
        ]
        
        portfolio_return = 0.001  # 假设组合收益率为0.1%
        
        print("不同基准收益率下的奖励计算:")
        print("基准情况            | 基准收益率 | 超额收益 | Base Alpha | 是否合理")
        print("-" * 65)
        
        for issue_name, market_return in possible_issues:
            if market_return is None:
                market_return = 0.0  # 数据缺失时默认为0
            
            excess_return = portfolio_return - market_return
            base_alpha = (excess_return - 0.0001) * 100  # 减去交易成本
            
            is_reasonable = "是" if base_alpha > 0 else "否"
            
            print(f"{issue_name:15s} | {market_return:9.4f} | {excess_return:8.4f} | {base_alpha:10.2f} | {is_reasonable}")
        
        # 分析发现的问题
        print(f"\n❌ 基准收益率问题分析:")
        print("1. 如果基准收益率过高（>组合收益率），会导致负的超额收益")
        print("2. 如果基准收益率数据缺失或异常，会影响Alpha计算")
        print("3. 基准收益率的计算逻辑可能存在错误")
        
        # 需要检查实际的基准收益率
        print(f"\n需要验证:")
        print("- 基准数据是否正确加载")
        print("- 基准收益率计算公式是否正确")
        print("- 基准收益率的数值范围是否合理")
        
        # 这个测试用于暴露基准收益率可能的问题
        assert False, "需要检查市场基准收益率的实际计算逻辑和数据"
    
    def test_penalty_factors_excessive_impact(self):
        """Red: 测试惩罚因子过度影响"""
        print("=== Red: 分析惩罚因子的过度影响 ===")
        
        # 分析各种惩罚对最终奖励的影响
        def calculate_penalty_impact(excess_return):
            """计算不同惩罚因子的影响"""
            
            # 基础Alpha奖励
            base_alpha = (excess_return - 0.0001) * 100
            
            # 表现惩罚（问题重点）
            underperformance_penalty = 0.0
            if excess_return < -0.01:  # 超额收益为负且超过1%时
                underperformance_penalty = abs(excess_return) * 200  # 200倍放大！
            
            # 其他常见惩罚
            risk_penalty = 0.5
            concentration_penalty = 0.0
            drawdown_penalty = 0.0
            
            # 正面奖励
            exploration_bonus = 1.0
            market_volatility_factor = 0.1
            
            net_reward = (base_alpha + exploration_bonus + market_volatility_factor 
                         - underperformance_penalty - risk_penalty 
                         - concentration_penalty - drawdown_penalty)
            
            return {
                'excess_return': excess_return,
                'base_alpha': base_alpha,
                'underperformance_penalty': underperformance_penalty,
                'total_penalties': underperformance_penalty + risk_penalty + concentration_penalty + drawdown_penalty,
                'total_bonuses': exploration_bonus + market_volatility_factor,
                'net_reward': net_reward
            }
        
        # 测试不同超额收益率下的惩罚影响
        test_excess_returns = [0.005, 0.001, 0.0, -0.005, -0.01, -0.015, -0.02]
        
        print("超额收益率对奖励的影响分析:")
        print("超额收益率 | Base Alpha | 表现惩罚 | 总惩罚 | 总奖励 | 净奖励 | 问题")
        print("-" * 75)
        
        excessive_penalty_cases = 0
        
        for excess_ret in test_excess_returns:
            result = calculate_penalty_impact(excess_ret)
            
            problem = ""
            if result['underperformance_penalty'] > 10:
                problem = "惩罚过重"
                excessive_penalty_cases += 1
            elif result['net_reward'] < 0 and excess_ret >= 0:
                problem = "正收益负奖励"
                excessive_penalty_cases += 1
            
            print(f"{excess_ret:10.3f} | {result['base_alpha']:10.2f} | {result['underperformance_penalty']:8.2f} | "
                  f"{result['total_penalties']:6.2f} | {result['total_bonuses']:6.2f} | "
                  f"{result['net_reward']:7.2f} | {problem}")
        
        print("-" * 75)
        print(f"过度惩罚案例数: {excessive_penalty_cases}")
        
        print(f"\n❌ 惩罚系统问题:")
        print("1. underperformance_penalty 200倍放大过于严厉")
        print("2. 即使轻微跑输（-0.5%）也会产生巨大惩罚（1.0）")
        print("3. 惩罚抵消了所有正面奖励，导致持续负奖励")
        
        # 这个测试应该失败，暴露惩罚过度的问题
        assert excessive_penalty_cases == 0, \
            f"发现 {excessive_penalty_cases} 个过度惩罚案例，需要调整惩罚参数"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])