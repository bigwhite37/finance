#!/usr/bin/env python3
"""
风险控制参数优化分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import logging

class RiskParamOptimizer:
    """风险参数优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 当前参数配置
        self.current_params = {
            'max_position': 0.15,
            'max_leverage': 1.2,
            'var_threshold': 0.025,
            'max_drawdown_threshold': 0.08,
            'volatility_threshold': 0.20,
            'lookback_window': 20
        }
        
        # 环境参数
        self.environment_params = {
            'max_position': 0.1,
            'max_leverage': 1.2,
            'lambda1': 2.0,
            'lambda2': 1.0,
            'max_dd_threshold': 0.05
        }
        
    def analyze_current_constraints(self) -> Dict:
        """分析当前约束的严格程度"""
        analysis = {
            'constraint_analysis': {},
            'optimization_recommendations': {}
        }
        
        # 1. 仓位约束分析
        max_pos_env = self.environment_params['max_position']
        max_pos_shield = self.current_params['max_position']
        
        analysis['constraint_analysis']['position'] = {
            'environment_limit': max_pos_env,
            'shield_limit': max_pos_shield,
            'effective_limit': min(max_pos_env, max_pos_shield),
            'restrictiveness': 'HIGH' if max_pos_shield <= 0.15 else 'MEDIUM',
            'recommendation': 'RELAX' if max_pos_shield <= 0.12 else 'MAINTAIN'
        }
        
        # 2. 杠杆约束分析
        leverage_ratio = self.current_params['max_leverage']
        analysis['constraint_analysis']['leverage'] = {
            'current_limit': leverage_ratio,
            'restrictiveness': 'HIGH' if leverage_ratio <= 1.2 else 'MEDIUM',
            'utilization_capacity': leverage_ratio * 100,  # 可用仓位百分比
            'recommendation': 'INCREASE' if leverage_ratio < 1.5 else 'MAINTAIN'
        }
        
        # 3. 风险度量约束分析
        var_threshold = self.current_params['var_threshold']
        analysis['constraint_analysis']['var'] = {
            'threshold': var_threshold,
            'daily_loss_limit': f"{var_threshold*100:.1f}%",
            'annualized_equivalent': f"{var_threshold * np.sqrt(252) * 100:.1f}%",
            'restrictiveness': 'HIGH' if var_threshold <= 0.025 else 'MEDIUM',
            'recommendation': 'RELAX' if var_threshold <= 0.02 else 'MAINTAIN'
        }
        
        # 4. 回撤约束分析
        dd_threshold = self.current_params['max_drawdown_threshold']
        analysis['constraint_analysis']['drawdown'] = {
            'threshold': dd_threshold,
            'early_warning_level': dd_threshold * 0.8,
            'restrictiveness': 'HIGH' if dd_threshold <= 0.08 else 'MEDIUM',
            'recommendation': 'INCREASE' if dd_threshold < 0.12 else 'MAINTAIN'
        }
        
        # 5. 波动率约束分析
        vol_threshold = self.current_params['volatility_threshold']
        analysis['constraint_analysis']['volatility'] = {
            'threshold': vol_threshold,
            'annualized_vol_limit': f"{vol_threshold*100:.0f}%",
            'restrictiveness': 'MEDIUM' if vol_threshold >= 0.20 else 'HIGH',
            'recommendation': 'MAINTAIN' if vol_threshold >= 0.20 else 'INCREASE'
        }
        
        return analysis
    
    def generate_optimized_params(self, target_return: float = 0.10) -> Dict:
        """
        生成优化的参数配置
        
        Args:
            target_return: 目标年化收益率
        """
        # 基于目标收益率调整参数
        risk_appetite = min(max(target_return / 0.15, 0.5), 2.0)  # 风险偏好系数
        
        optimized_params = {
            # 仓位限制：允许更大的单股票仓位以获取alpha
            'max_position': min(0.20, 0.12 + target_return * 0.5),
            
            # 杠杆限制：适度提高以增强收益潜力
            'max_leverage': min(2.0, 1.2 + target_return * 2.0),
            
            # VaR阈值：放宽以允许更多风险承担
            'var_threshold': min(0.04, 0.025 + target_return * 0.1),
            
            # 回撤阈值：设置更合理的止损线
            'max_drawdown_threshold': min(0.15, 0.08 + target_return * 0.4),
            
            # 波动率阈值：匹配目标收益的合理波动率
            'volatility_threshold': min(0.30, 0.20 + target_return * 0.5),
            
            # 环境参数优化
            'environment_max_position': min(0.15, 0.10 + target_return * 0.3),
            'environment_max_leverage': min(1.8, 1.2 + target_return * 2.0),
            
            # 调整奖励函数权重
            'lambda1': max(0.5, 2.0 - target_return * 5.0),  # 降低回撤惩罚
            'lambda2': max(0.2, 1.0 - target_return * 3.0),  # 降低CVaR惩罚
            
            # CVaR参数调整
            'cvar_threshold': min(-0.01, -0.02 + target_return * 0.05),
            'cvar_lambda': max(0.005, 0.01 - target_return * 0.02)
        }
        
        return optimized_params
    
    def validate_param_consistency(self, params: Dict) -> Tuple[bool, List[str]]:
        """验证参数一致性"""
        issues = []
        
        # 检查仓位限制一致性
        if 'environment_max_position' in params and 'max_position' in params:
            if params['environment_max_position'] > params['max_position']:
                issues.append("环境仓位限制大于安全保护层限制，可能导致不一致")
        
        # 检查杠杆限制一致性
        if 'environment_max_leverage' in params and 'max_leverage' in params:
            if params['environment_max_leverage'] > params['max_leverage']:
                issues.append("环境杠杆限制大于安全保护层限制，可能导致不一致")
        
        # 检查阈值合理性
        if params.get('var_threshold', 0) > 0.05:
            issues.append("VaR阈值可能过于宽松，存在风险")
        
        if params.get('max_drawdown_threshold', 0) > 0.20:
            issues.append("最大回撤阈值可能过于宽松")
        
        if params.get('max_position', 0) > 0.25:
            issues.append("单股票仓位限制可能过于宽松")
        
        return len(issues) == 0, issues
    
    def generate_risk_profile_summary(self, params: Dict) -> str:
        """生成风险配置摘要"""
        summary = []
        summary.append("优化后风险控制参数摘要")
        summary.append("=" * 40)
        
        # 仓位配置
        summary.append(f"单股票最大仓位: {params.get('max_position', 0)*100:.1f}%")
        summary.append(f"最大杠杆倍数: {params.get('max_leverage', 1.0):.1f}x")
        
        # 风险度量
        summary.append(f"VaR阈值: {params.get('var_threshold', 0)*100:.1f}%")
        summary.append(f"最大回撤阈值: {params.get('max_drawdown_threshold', 0)*100:.1f}%")
        summary.append(f"波动率阈值: {params.get('volatility_threshold', 0)*100:.1f}%")
        
        # 奖励函数权重
        if 'lambda1' in params:
            summary.append(f"回撤惩罚系数: {params['lambda1']:.2f}")
        if 'lambda2' in params:
            summary.append(f"CVaR惩罚系数: {params['lambda2']:.2f}")
        
        # 风险等级评估
        risk_score = (params.get('max_position', 0.1) * 5 + 
                     params.get('max_leverage', 1.0) * 2 + 
                     params.get('var_threshold', 0.02) * 100)
        
        if risk_score < 1.5:
            risk_level = "保守型"
        elif risk_score < 2.5:
            risk_level = "平衡型"
        else:
            risk_level = "积极型"
        
        summary.append(f"\n风险等级: {risk_level} (评分: {risk_score:.2f})")
        
        return "\\n".join(summary)

def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)
    
    optimizer = RiskParamOptimizer()
    
    print("=== 当前风险控制参数分析 ===")
    analysis = optimizer.analyze_current_constraints()
    
    for category, details in analysis['constraint_analysis'].items():
        print(f"\\n{category.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")
    
    print("\\n=== 参数优化建议 ===")
    
    # 针对不同目标收益率生成优化参数
    target_returns = [0.08, 0.10, 0.12]
    
    for target_return in target_returns:
        print(f"\\n目标年化收益率: {target_return*100:.0f}%")
        optimized = optimizer.generate_optimized_params(target_return)
        
        # 验证参数一致性
        is_valid, issues = optimizer.validate_param_consistency(optimized)
        
        print(f"参数验证: {'通过' if is_valid else '存在问题'}")
        if issues:
            for issue in issues:
                print(f"  警告: {issue}")
        
        # 生成摘要
        summary = optimizer.generate_risk_profile_summary(optimized)
        print(f"\\n{summary}")
        
        print("\\n详细参数:")
        for key, value in sorted(optimized.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\\n=== 推荐方案 ===")
    print("基于目标10%年化收益率的平衡配置:")
    recommended = optimizer.generate_optimized_params(0.10)
    
    print("\\n配置文件更新建议:")
    print("safety_shield:")
    print(f"  max_position: {recommended['max_position']:.3f}")
    print(f"  max_leverage: {recommended['max_leverage']:.1f}")
    print(f"  var_threshold: {recommended['var_threshold']:.3f}")
    print(f"  max_drawdown_threshold: {recommended['max_drawdown_threshold']:.3f}")
    print(f"  volatility_threshold: {recommended['volatility_threshold']:.3f}")
    
    print("\\nenvironment:")
    print(f"  max_position: {recommended['environment_max_position']:.3f}")
    print(f"  max_leverage: {recommended['environment_max_leverage']:.1f}")
    print(f"  lambda1: {recommended['lambda1']:.2f}")
    print(f"  lambda2: {recommended['lambda2']:.2f}")

if __name__ == "__main__":
    main()