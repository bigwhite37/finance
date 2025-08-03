"""
奖励函数优化器使用示例

该示例展示了如何使用RewardOptimizer来计算风险调整后的奖励，
包括回撤惩罚机制和多样化奖励机制。
"""

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl_trading_system.risk_control.reward_optimizer import (
    RewardOptimizer,
    RewardConfig
)


def main():
    """主函数：演示奖励函数优化器的使用"""
    
    print("=== 奖励函数优化器使用示例 ===\n")
    
    # 1. 创建奖励函数配置
    config = RewardConfig(
        base_return_weight=1.0,
        risk_aversion_coefficient=0.5,
        drawdown_penalty_factor=2.0,
        drawdown_threshold=0.02,
        diversification_bonus=0.1,
        concentration_penalty=0.5,
        max_single_position=0.2,
        dynamic_penalty_enabled=True,
        time_decay_enabled=True
    )
    
    # 2. 创建奖励优化器
    optimizer = RewardOptimizer(config)
    
    print("1. 奖励函数配置:")
    print(f"   - 基础收益权重: {config.base_return_weight}")
    print(f"   - 回撤惩罚因子: {config.drawdown_penalty_factor}")
    print(f"   - 回撤惩罚阈值: {config.drawdown_threshold}")
    print(f"   - 多样化奖励系数: {config.diversification_bonus}")
    print(f"   - 集中度惩罚系数: {config.concentration_penalty}")
    print()
    
    # 3. 模拟交易场景
    print("2. 模拟交易场景:")
    
    # 场景数据
    scenarios = [
        # (收益率, 回撤, 持仓, 回撤阶段, 描述)
        (0.02, -0.01, {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}, 'NORMAL', '正常盈利，分散持仓'),
        (-0.01, -0.03, {'AAPL': 0.5, 'GOOGL': 0.5}, 'DRAWDOWN_START', '开始亏损，适度集中'),
        (-0.02, -0.05, {'AAPL': 0.8, 'GOOGL': 0.2}, 'DRAWDOWN_CONTINUE', '持续亏损，高度集中'),
        (-0.015, -0.06, {'AAPL': 0.7, 'GOOGL': 0.3}, 'DRAWDOWN_CONTINUE', '回撤恶化，集中持仓'),
        (0.005, -0.04, {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}, 'RECOVERY', '开始恢复，增加分散'),
        (0.015, -0.02, {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.2}, 'RECOVERY', '继续恢复，更加分散'),
        (0.01, -0.01, {'AAPL': 0.2, 'GOOGL': 0.2, 'MSFT': 0.2, 'TSLA': 0.2, 'NVDA': 0.2}, 'NORMAL', '回到正常，高度分散')
    ]
    
    rewards = []
    timestamps = []
    
    for i, (returns, drawdown, positions, phase, description) in enumerate(scenarios):
        timestamp = datetime.now() + timedelta(days=i)
        
        # 计算风险调整奖励
        reward = optimizer.calculate_risk_adjusted_reward(
            returns=returns,
            drawdown=drawdown,
            positions=positions,
            drawdown_phase=phase,
            timestamp=timestamp
        )
        
        rewards.append(reward)
        timestamps.append(timestamp)
        
        print(f"   场景 {i+1}: {description}")
        print(f"     收益率: {returns:+.3f}, 回撤: {drawdown:+.3f}, 阶段: {phase}")
        print(f"     持仓: {positions}")
        print(f"     奖励: {reward:+.6f}")
        print()
    
    # 4. 分析奖励组件
    print("3. 奖励组件分析:")
    
    # 选择一个典型场景进行详细分析
    test_positions = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.2, 'TSLA': 0.1}
    test_drawdown = -0.05
    
    # 计算各个组件
    basic_diversification = optimizer._calculate_basic_diversification_reward(test_positions)
    dynamic_diversification = optimizer._calculate_dynamic_diversification_reward(test_positions)
    concentration_penalty = optimizer._calculate_enhanced_concentration_penalty(test_positions)
    correlation_adjustment = optimizer._calculate_correlation_adjustment_reward(test_positions)
    drawdown_penalty = optimizer._calculate_enhanced_drawdown_penalty(test_drawdown, 'DRAWDOWN_CONTINUE')
    
    print(f"   测试持仓: {test_positions}")
    print(f"   测试回撤: {test_drawdown}")
    print(f"   - 基础多样化奖励: {basic_diversification:+.6f}")
    print(f"   - 动态多样化奖励: {dynamic_diversification:+.6f}")
    print(f"   - 相关性调整奖励: {correlation_adjustment:+.6f}")
    print(f"   - 集中度惩罚: {concentration_penalty:+.6f}")
    print(f"   - 回撤惩罚: {drawdown_penalty:+.6f}")
    print()
    
    # 5. 多样化指标分析
    print("4. 多样化指标分析:")
    
    test_portfolios = [
        ({'AAPL': 1.0}, '完全集中'),
        ({'AAPL': 0.5, 'GOOGL': 0.5}, '二元分散'),
        ({'AAPL': 0.33, 'GOOGL': 0.33, 'MSFT': 0.34}, '三元分散'),
        ({'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}, '四元均衡'),
        ({'AAPL': 0.2, 'GOOGL': 0.2, 'MSFT': 0.2, 'TSLA': 0.2, 'NVDA': 0.2}, '五元均衡')
    ]
    
    for positions, description in test_portfolios:
        metrics = optimizer.calculate_portfolio_diversification_metrics(positions)
        print(f"   {description}:")
        print(f"     赫芬达尔指数: {metrics['herfindahl_index']:.4f}")
        print(f"     有效资产数量: {metrics['effective_number_of_assets']:.2f}")
        print(f"     最大权重: {metrics['max_weight']:.4f}")
        print(f"     多样化比率: {metrics['diversification_ratio']:.4f}")
        print()
    
    # 6. 性能统计
    print("5. 性能统计:")
    performance = optimizer.get_performance_summary()
    
    print(f"   总交易次数: {performance['episodes']}")
    print(f"   平均收益: {performance['avg_return']:+.6f}")
    print(f"   总收益: {performance['total_return']:+.6f}")
    print(f"   平均回撤: {performance['avg_drawdown']:+.6f}")
    print(f"   最大回撤: {performance['max_drawdown']:+.6f}")
    print(f"   夏普比率: {performance['risk_metrics']['sharpe_ratio']:+.4f}")
    print(f"   卡尔玛比率: {performance['risk_metrics']['calmar_ratio']:+.4f}")
    print()
    
    # 7. 惩罚机制分析
    print("6. 惩罚机制分析:")
    penalty_analysis = optimizer.get_penalty_analysis()
    
    if 'status' not in penalty_analysis:
        print(f"   总惩罚次数: {penalty_analysis['total_penalties']}")
        print(f"   平均惩罚: {penalty_analysis['avg_penalty']:+.6f}")
        print(f"   最大惩罚: {penalty_analysis['max_penalty']:+.6f}")
        print(f"   惩罚趋势: {penalty_analysis['penalty_trend']}")
        print(f"   连续亏损次数: {penalty_analysis['consecutive_losses']}")
    else:
        print(f"   {penalty_analysis['status']}")
    print()
    
    # 8. 参数优化建议
    print("7. 参数优化建议:")
    target_metrics = {
        'target_sharpe': 2.0,
        'target_calmar': 1.5,
        'max_drawdown': 0.08,
        'target_volatility': 0.15
    }
    
    optimized_config = optimizer.optimize_reward_parameters(target_metrics)
    print(f"   回撤惩罚因子调整: {config.drawdown_penalty_factor:.2f} -> {optimized_config.drawdown_penalty_factor:.2f}")
    print(f"   风险厌恶系数调整: {config.risk_aversion_coefficient:.2f} -> {optimized_config.risk_aversion_coefficient:.2f}")
    print(f"   多样化奖励调整: {config.diversification_bonus:.2f} -> {optimized_config.diversification_bonus:.2f}")
    print()
    
    # 9. 多样化参数优化
    print("8. 多样化参数优化:")
    diversification_suggestions = optimizer.optimize_diversification_parameters(target_diversification=0.8)
    
    if 'status' not in diversification_suggestions:
        print(f"   当前多样化水平: {diversification_suggestions['current_diversification']:.4f}")
        print(f"   目标多样化水平: {diversification_suggestions['target_diversification']:.4f}")
        print(f"   多样化差距: {diversification_suggestions['diversification_gap']:+.4f}")
        print(f"   多样化奖励调整建议: {diversification_suggestions['diversification_bonus_adjustment']:.2f}x")
        print(f"   集中度惩罚调整建议: {diversification_suggestions['concentration_penalty_adjustment']:.2f}x")
        print(f"   最大单一持仓建议: {diversification_suggestions['max_position_adjustment']:.2f}")
    else:
        print(f"   {diversification_suggestions['status']}")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()