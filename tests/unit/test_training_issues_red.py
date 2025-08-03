#!/usr/bin/env python3
"""
训练问题的红色阶段TDD测试
复现学习率持续下降和负累积奖励下零回撤问题
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import DrawdownEarlyStopping


class TestTrainingIssuesRed:
    """测试训练问题 - Red阶段"""
    
    def test_learning_rate_decay_issue(self):
        """Red: 测试学习率因子异常持续下降问题"""
        print("=== Red: 分析学习率因子异常持续下降 ===")
        
        # 模拟从训练日志中观察到的性能数据
        # Episode 50前的平均奖励约-31.67，之后持续下降
        episode_rewards = [
            # Episodes 1-20: 平均约-31.67
            -26.71, -30.0, -35.0, -28.5, -33.0, -29.0, -32.0, -31.5, -30.5, -33.5,
            -28.0, -34.0, -31.0, -29.5, -32.5, -30.0, -33.0, -31.5, -28.5, -32.0,
            
            # Episodes 21-40: 平均约-35.72  
            -33.92, -38.0, -36.5, -34.0, -37.0, -35.5, -36.0, -35.0, -37.5, -34.5,
            -36.0, -35.0, -38.0, -34.0, -36.5, -35.0, -37.0, -36.0, -34.5, -37.5,
            
            # Episodes 41-50: 继续下降趋势
            -40.0, -38.5, -42.0, -39.0, -41.5, -38.0, -43.0, -40.5, -39.5, -42.5
        ]
        
        # 模拟自适应学习率调整逻辑
        def simulate_lr_adaptation(performance_history, current_lr_factor=1.0):
            """模拟学习率自适应调整"""
            lr_adaptation_factor = 0.9  # 从配置推断
            performance_threshold_down = 0.95  # 假设阈值
            min_lr_factor = 0.01
            
            if len(performance_history) < 50:
                return current_lr_factor
            
            recent_performance = np.mean(performance_history[-20:])
            long_term_performance = np.mean(performance_history[-50:])
            
            print(f"Recent (last 20): {recent_performance:.2f}")
            print(f"Long-term (last 50): {long_term_performance:.2f}")
            print(f"Threshold check: {recent_performance} < {long_term_performance * performance_threshold_down:.2f}")
            
            if recent_performance < long_term_performance * performance_threshold_down:
                new_lr_factor = max(
                    min_lr_factor,
                    current_lr_factor * lr_adaptation_factor
                )
                print(f"性能下降: {current_lr_factor:.4f} -> {new_lr_factor:.4f}")
                return new_lr_factor
            
            return current_lr_factor
        
        # 测试学习率调整序列
        lr_factor = 1.0
        lr_history = [lr_factor]
        
        print("学习率因子变化序列:")
        for i, reward in enumerate(episode_rewards, 1):
            # 在episode 50及之后开始连续调整
            if i >= 50:
                lr_factor = simulate_lr_adaptation(episode_rewards[:i], lr_factor)
                lr_history.append(lr_factor)
                
                if i % 5 == 0:  # 每5个episode显示一次
                    print(f"Episode {i}: 奖励={reward:.2f}, 学习率因子={lr_factor:.4f}")
        
        # 问题分析：学习率因子是否过度下降
        final_lr_factor = lr_history[-1]
        lr_decay_episodes = len([lr for lr in lr_history if lr < 1.0])
        
        print(f"\n问题分析:")
        print(f"最终学习率因子: {final_lr_factor:.4f}")
        print(f"下降次数: {lr_decay_episodes}")
        print(f"是否过度下降: {'是' if final_lr_factor < 0.1 else '否'}")
        
        # 这个测试应该揭示学习率过度下降的问题
        assert final_lr_factor >= 0.1, \
            f"学习率因子过度下降到 {final_lr_factor:.4f}，可能导致训练停滞"
    
    def test_negative_cumulative_reward_zero_drawdown_issue(self):
        """Red: 测试负累积奖励下回撤仍为零的问题"""
        print("=== Red: 分析负累积奖励下零回撤问题 ===")
        
        # 从训练日志中提取的实际数据
        observed_data = [
            # (episode, cumulative_reward, expected_drawdown)
            (50, -1692.1631, "应该>0"),
            (100, -3446.5452, "应该>0")
        ]
        
        # 模拟累积奖励的演进过程
        # 如果episode奖励持续为负，累积奖励应该一直下降
        episode_rewards = [-33.0] * 50  # 模拟前50个episode都为负
        episode_rewards.extend([-35.0] * 50)  # 后50个episode更负
        
        cumulative_rewards = []
        cumulative_sum = 0.0
        
        for reward in episode_rewards:
            cumulative_sum += reward
            cumulative_rewards.append(cumulative_sum)
        
        print("累积奖励演进过程:")
        for i in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:  # 每10个episode显示
            print(f"Episode {i+1}: 累积奖励 {cumulative_rewards[i]:.4f}")
        
        # 使用DrawdownEarlyStopping测试回撤计算
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.3, patience=10)
        calculated_drawdowns = []
        
        for cumulative in cumulative_rewards:
            drawdown_monitor.step(cumulative)
            current_drawdown = drawdown_monitor.get_current_drawdown()
            calculated_drawdowns.append(current_drawdown)
        
        # 分析关键点的回撤
        key_episodes = [49, 99]  # Episode 50, 100 (0-indexed)
        
        print(f"\n关键点回撤分析:")
        for episode_idx in key_episodes:
            episode_num = episode_idx + 1
            cumulative = cumulative_rewards[episode_idx]
            drawdown = calculated_drawdowns[episode_idx]
            
            print(f"Episode {episode_num}:")
            print(f"  累积奖励: {cumulative:.4f}")
            print(f"  计算回撤: {drawdown:.4f}")
            print(f"  预期: 应该>0 (因为累积奖励为负)")
        
        # 检查回撤监控器的状态
        print(f"\n回撤监控器状态:")
        print(f"  当前峰值: {drawdown_monitor.peak_value}")
        print(f"  当前回撤: {drawdown_monitor.get_current_drawdown():.4f}")
        
        # 问题分析：为什么负累积奖励下回撤为0？
        print(f"\n❌ 问题分析:")
        if drawdown_monitor.peak_value <= 0:
            print("  问题根源: 峰值为负数或零，导致回撤计算异常")
            print("  峰值只有在累积奖励为正时才更新")
            print("  当累积奖励持续为负时，峰值保持初始值(0)，导致回撤计算错误")
        
        # 这个测试应该失败，暴露负累积奖励下回撤计算的问题
        episode_50_drawdown = calculated_drawdowns[49]
        episode_100_drawdown = calculated_drawdowns[99]
        
        assert episode_50_drawdown > 0.1, \
            f"Episode 50累积奖励为负({cumulative_rewards[49]:.2f})时回撤应该>10%，实际: {episode_50_drawdown:.4f}"
        
        assert episode_100_drawdown > 0.2, \
            f"Episode 100累积奖励为负({cumulative_rewards[99]:.2f})时回撤应该>20%，实际: {episode_100_drawdown:.4f}"
    
    def test_drawdown_calculation_logic_for_negative_cumulative_rewards(self):
        """Red: 测试负累积奖励场景下的回撤计算逻辑"""
        print("=== Red: 测试负累积奖励的回撤计算逻辑 ===")
        
        # 测试场景：从正数开始下降到负数
        test_sequence = [
            100.0,   # 初始正值，设为峰值
            80.0,    # 下降，回撤应为20%
            60.0,    # 继续下降，回撤应为40%
            40.0,    # 继续下降，回撤应为60%
            0.0,     # 归零，回撤应为100%
            -20.0,   # 变负，回撤应为120%？
            -40.0,   # 更负，回撤应为140%？
            -60.0    # 最负，回撤应为160%？
        ]
        
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.5, patience=5)
        
        print("回撤计算序列:")
        for i, value in enumerate(test_sequence):
            drawdown_monitor.step(value)
            current_drawdown = drawdown_monitor.get_current_drawdown()
            
            # 手动计算期望回撤
            peak = drawdown_monitor.peak_value
            if peak > 0:
                expected_drawdown = (peak - value) / peak
            else:
                expected_drawdown = 0.0
            
            print(f"步骤 {i+1}: 值={value:6.1f}, 峰值={peak:6.1f}, "
                  f"实际回撤={current_drawdown:.4f}, 期望回撤={expected_drawdown:.4f}")
        
        # 分析问题：负值时的回撤计算
        final_drawdown = drawdown_monitor.get_current_drawdown()
        final_peak = drawdown_monitor.peak_value
        final_value = test_sequence[-1]
        
        print(f"\n最终状态:")
        print(f"  最终值: {final_value}")
        print(f"  最终峰值: {final_peak}")
        print(f"  最终回撤: {final_drawdown:.4f}")
        
        # 当前实现的回撤公式: (peak - current) / peak if peak > 0 else 0
        # 问题：当current为负数时，(peak - current)会变得更大，但这是否合理？
        
        if final_value < 0 and final_peak > 0:
            manual_calculation = (final_peak - final_value) / final_peak
            print(f"  手动计算回撤: {manual_calculation:.4f}")
            
            # 这揭示了一个概念问题：当累积奖励变为负数时，回撤应该如何定义？
            assert manual_calculation > 1.0, \
                "负累积奖励导致回撤计算>100%，这在概念上可能是错误的"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])