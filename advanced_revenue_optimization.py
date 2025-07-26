#!/usr/bin/env python3
"""
高级收益优化分析器
基于当前-0.86%的结果，分析并优化参数以实现8%目标
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class AdvancedRevenueOptimizer:
    """高级收益优化器"""
    
    def __init__(self):
        """初始化优化器"""
        self.current_performance = {
            'annual_return': -0.0086,
            'max_drawdown': 0.0107,
            'sharpe_ratio': -7.51,
            'volatility': 0.0051,
            'average_leverage': 0.36
        }
        
        self.target_performance = {
            'annual_return': 0.08,
            'max_drawdown_limit': 0.12,
            'min_sharpe_ratio': 0.8,
            'target_volatility': 0.15
        }
        
        # 当前参数配置
        self.current_params = {
            'max_position': 0.17,
            'max_leverage': 1.4,
            'var_threshold': 0.035,
            'max_drawdown_threshold': 0.12,
            'volatility_threshold': 0.25,
            'lambda1': 1.5,  # 回撤惩罚
            'lambda2': 0.7,  # CVaR惩罚
            'cvar_threshold': -0.015,
            'reward_amplification': 10.0  # 当前奖励放大倍数
        }
    
    def analyze_performance_gap(self) -> Dict:
        """分析性能差距"""
        analysis = {}
        
        # 收益率差距分析
        return_gap = self.target_performance['annual_return'] - self.current_performance['annual_return']
        analysis['return_gap'] = {
            'absolute_gap': return_gap,
            'percentage_gap': return_gap / abs(self.current_performance['annual_return']) * 100,
            'required_improvement': f"{return_gap*100:.2f}%"
        }
        
        # 风险利用率分析
        current_dd = abs(self.current_performance['max_drawdown'])
        dd_budget = self.target_performance['max_drawdown_limit']
        dd_utilization = current_dd / dd_budget
        
        analysis['risk_utilization'] = {
            'current_drawdown': f"{current_dd*100:.2f}%",
            'drawdown_budget': f"{dd_budget*100:.2f}%",
            'utilization_rate': f"{dd_utilization*100:.1f}%",
            'available_risk_budget': f"{(dd_budget - current_dd)*100:.2f}%",
            'risk_efficiency': 'VERY_LOW' if dd_utilization < 0.2 else 'LOW' if dd_utilization < 0.5 else 'MODERATE'
        }
        
        # 杠杆利用率分析
        current_leverage = self.current_performance['average_leverage']
        max_leverage = self.current_params['max_leverage']
        leverage_utilization = current_leverage / max_leverage
        
        analysis['leverage_utilization'] = {
            'current_leverage': f"{current_leverage:.2f}x",
            'max_leverage': f"{max_leverage:.2f}x",
            'utilization_rate': f"{leverage_utilization*100:.1f}%",
            'leverage_efficiency': 'VERY_LOW' if leverage_utilization < 0.3 else 'LOW' if leverage_utilization < 0.6 else 'MODERATE'
        }
        
        # 波动率利用率分析
        current_vol = self.current_performance['volatility']
        target_vol = self.target_performance['target_volatility']
        vol_utilization = current_vol / target_vol
        
        analysis['volatility_utilization'] = {
            'current_volatility': f"{current_vol*100:.2f}%",
            'target_volatility': f"{target_vol*100:.2f}%",
            'utilization_rate': f"{vol_utilization*100:.1f}%",
            'volatility_efficiency': 'EXTREMELY_LOW' if vol_utilization < 0.1 else 'VERY_LOW' if vol_utilization < 0.3 else 'LOW'
        }
        
        return analysis
    
    def generate_optimization_strategy(self) -> Dict:
        """生成优化策略"""
        
        # 基于性能差距计算所需的参数调整
        return_gap = 0.0886  # 需要提升8.86%
        risk_utilization = 0.107 / 0.12  # 当前风险利用率约9%
        
        # 策略1: 渐进式风险提升
        progressive_strategy = {
            'name': '渐进式风险提升策略',
            'approach': '逐步提升风险容忍度，保持安全边际',
            'params': {
                'max_position': 0.20,  # 提升至20%
                'max_leverage': 1.6,   # 提升至1.6倍
                'var_threshold': 0.045, # 放宽VaR至4.5%
                'lambda1': 1.0,        # 进一步降低回撤惩罚
                'lambda2': 0.5,        # 进一步降低CVaR惩罚
                'reward_amplification': 15.0,  # 提升奖励放大至15倍
                'training_episodes': 300  # 增加训练深度
            },
            'expected_improvement': '3-5%年化收益率',
            'risk_level': '低-中等'
        }
        
        # 策略2: 积极收益优化
        aggressive_strategy = {
            'name': '积极收益优化策略',
            'approach': '大幅提升收益信号，适度增加风险容忍',
            'params': {
                'max_position': 0.25,  # 提升至25%
                'max_leverage': 1.8,   # 提升至1.8倍
                'var_threshold': 0.055, # 放宽VaR至5.5%
                'lambda1': 0.8,        # 大幅降低回撤惩罚
                'lambda2': 0.3,        # 大幅降低CVaR惩罚
                'reward_amplification': 20.0,  # 提升奖励放大至20倍
                'momentum_weight': 2.0, # 增加动量奖励权重
                'training_episodes': 400  # 增加训练深度
            },
            'expected_improvement': '5-8%年化收益率',
            'risk_level': '中等'
        }
        
        # 策略3: 智能动态调整
        dynamic_strategy = {
            'name': '智能动态调整策略',
            'approach': '基于市场状态动态调整参数',
            'params': {
                'max_position': 0.22,  # 适中提升
                'max_leverage': 1.7,   # 适中提升
                'var_threshold': 0.050, # 适中放宽
                'lambda1': 0.9,        # 适中降低
                'lambda2': 0.4,        # 适中降低
                'reward_amplification': 18.0,  # 提升奖励放大
                'use_market_regime_detection': True,  # 启用市场状态检测
                'adaptive_risk_scaling': True,        # 启用自适应风险缩放
                'training_episodes': 350
            },
            'expected_improvement': '4-7%年化收益率',
            'risk_level': '中等-可控'
        }
        
        return {
            'progressive': progressive_strategy,
            'aggressive': aggressive_strategy,
            'dynamic': dynamic_strategy
        }
    
    def recommend_optimal_strategy(self) -> Dict:
        """推荐最优策略"""
        strategies = self.generate_optimization_strategy()
        
        # 基于当前状况推荐策略
        recommendation = {
            'primary_recommendation': 'dynamic',
            'reasoning': [
                '当前风险利用率极低(9%)，有大量风险预算可用',
                '波动率利用率极低(3.4%)，可以大幅提升',
                '杠杆利用率低(26%)，有提升空间',
                '智能动态调整可以平衡收益和风险',
                '当前系统已证明风险控制能力强，可以适度激进'
            ],
            'implementation_priority': [
                '1. 首先实施渐进式策略验证方向正确',
                '2. 如效果良好，升级到动态策略',
                '3. 根据市场表现考虑积极策略'
            ],
            'key_modifications': [
                '提升奖励放大倍数至18倍',
                '降低回撤和CVaR惩罚系数',
                '增加最大仓位至22%',
                '延长训练至350个episodes',
                '启用市场状态感知功能'
            ]
        }
        
        return recommendation
    
    def calculate_risk_reward_profile(self, strategy_params: Dict) -> Dict:
        """计算风险收益配置"""
        
        # 基于参数变化预测性能影响
        position_boost = strategy_params.get('max_position', 0.17) / 0.17
        leverage_boost = strategy_params.get('max_leverage', 1.4) / 1.4
        reward_boost = strategy_params.get('reward_amplification', 10.0) / 10.0
        penalty_reduction = (1.5 / strategy_params.get('lambda1', 1.5)) * (0.7 / strategy_params.get('lambda2', 0.7))
        
        # 预测收益率提升
        estimated_return_boost = (
            position_boost * 0.3 +      # 仓位影响30%
            leverage_boost * 0.25 +     # 杠杆影响25%  
            reward_boost * 0.35 +       # 奖励信号影响35%
            penalty_reduction * 0.1     # 惩罚减少影响10%
        )
        
        base_return = -0.0086
        estimated_return = base_return + (estimated_return_boost - 1.0) * 0.12  # 假设12%基础提升潜力
        
        # 预测风险变化
        risk_increase_factor = (position_boost + leverage_boost) / 2
        estimated_drawdown = 0.0107 * risk_increase_factor
        estimated_volatility = 0.0051 * risk_increase_factor * 3  # 波动率通常变化更大
        
        # 计算风险调整后收益
        estimated_sharpe = estimated_return / estimated_volatility if estimated_volatility > 0 else 0
        
        return {
            'estimated_annual_return': estimated_return,
            'estimated_max_drawdown': min(estimated_drawdown, 0.12),  # 不超过12%限制
            'estimated_volatility': estimated_volatility,
            'estimated_sharpe_ratio': estimated_sharpe,
            'risk_budget_utilization': min(estimated_drawdown / 0.12, 1.0),
            'target_achievement_probability': min((estimated_return + 0.0086) / 0.0886, 1.0)
        }


def main():
    """主函数"""
    print("🎯 高级收益优化分析")
    print("="*50)
    
    optimizer = AdvancedRevenueOptimizer()
    
    # 1. 分析性能差距
    print("\n📊 性能差距分析:")
    gap_analysis = optimizer.analyze_performance_gap()
    
    print(f"\n收益率差距:")
    return_gap = gap_analysis['return_gap']
    print(f"  需要提升: {return_gap['required_improvement']}")
    print(f"  提升幅度: {return_gap['percentage_gap']:.1f}%")
    
    print(f"\n风险利用率:")
    risk_util = gap_analysis['risk_utilization']
    print(f"  当前回撤: {risk_util['current_drawdown']}")
    print(f"  可用预算: {risk_util['available_risk_budget']}")
    print(f"  利用效率: {risk_util['risk_efficiency']}")
    
    print(f"\n杠杆利用率:")
    lev_util = gap_analysis['leverage_utilization']
    print(f"  当前杠杆: {lev_util['current_leverage']}")
    print(f"  利用率: {lev_util['utilization_rate']}")
    print(f"  效率评级: {lev_util['leverage_efficiency']}")
    
    print(f"\n波动率利用率:")
    vol_util = gap_analysis['volatility_utilization']
    print(f"  当前波动率: {vol_util['current_volatility']}")
    print(f"  利用率: {vol_util['utilization_rate']}")
    print(f"  效率评级: {vol_util['volatility_efficiency']}")
    
    # 2. 生成优化策略
    print("\n🚀 优化策略方案:")
    strategies = optimizer.generate_optimization_strategy()
    
    for name, strategy in strategies.items():
        print(f"\n策略: {strategy['name']}")
        print(f"  方法: {strategy['approach']}")
        print(f"  预期提升: {strategy['expected_improvement']}")
        print(f"  风险等级: {strategy['risk_level']}")
        
        # 计算风险收益配置
        risk_reward = optimizer.calculate_risk_reward_profile(strategy['params'])
        print(f"  预测年化收益: {risk_reward['estimated_annual_return']*100:.2f}%")
        print(f"  预测最大回撤: {risk_reward['estimated_max_drawdown']*100:.2f}%")
        print(f"  目标达成概率: {risk_reward['target_achievement_probability']*100:.1f}%")
    
    # 3. 推荐最优策略
    print("\n💡 最优策略推荐:")
    recommendation = optimizer.recommend_optimal_strategy()
    
    print(f"推荐策略: {strategies[recommendation['primary_recommendation']]['name']}")
    print(f"\n推荐理由:")
    for reason in recommendation['reasoning']:
        print(f"  • {reason}")
    
    print(f"\n实施优先级:")
    for priority in recommendation['implementation_priority']:
        print(f"  {priority}")
    
    print(f"\n关键修改项:")
    for modification in recommendation['key_modifications']:
        print(f"  ✓ {modification}")
    
    # 4. 生成配置建议
    recommended_strategy = strategies[recommendation['primary_recommendation']]
    print(f"\n⚙️  推荐配置参数:")
    print("```yaml")
    print("# 收益优化配置")
    print("safety_shield:")
    for param, value in recommended_strategy['params'].items():
        if param in ['max_position', 'max_leverage', 'var_threshold']:
            print(f"  {param}: {value}")
    
    print("\nenvironment:")
    print(f"  lambda1: {recommended_strategy['params']['lambda1']}")
    print(f"  lambda2: {recommended_strategy['params']['lambda2']}")
    
    print("\ntraining:")
    print(f"  total_episodes: {recommended_strategy['params']['training_episodes']}")
    print("```")
    
    print(f"\n🎯 预期结果:")
    target_config = optimizer.calculate_risk_reward_profile(recommended_strategy['params'])
    print(f"  目标年化收益率: {target_config['estimated_annual_return']*100:.2f}%")
    print(f"  预期最大回撤: {target_config['estimated_max_drawdown']*100:.2f}%")
    print(f"  8%目标达成概率: {target_config['target_achievement_probability']*100:.1f}%")
    
    print(f"\n✅ 下一步行动:")
    print("  1. 应用推荐配置参数")
    print("  2. 重新训练模型(350 episodes)")
    print("  3. 验证回测结果")
    print("  4. 根据结果进行微调")


if __name__ == "__main__":
    main()