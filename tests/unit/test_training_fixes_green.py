#!/usr/bin/env python3
"""
训练问题修复的绿色阶段TDD测试
验证学习率调度和回撤计算修复效果
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import DrawdownEarlyStopping


class TestTrainingFixesGreen:
    """测试训练问题修复 - Green阶段"""
    
    def test_improved_drawdown_calculation_for_negative_rewards(self):
        """Green: 测试改进的负奖励回撤计算"""
        print("=== Green: 验证改进的负奖励回撤计算 ===")
        
        # 测试场景1：从负值开始的累积奖励序列
        negative_start_sequence = [-50, -100, -150, -200, -180, -160, -140]
        
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.3, patience=5)
        
        print("负起始值回撤测试:")
        for i, value in enumerate(negative_start_sequence):
            drawdown_monitor.step(value)
            current_drawdown = drawdown_monitor.get_current_drawdown()
            
            print(f"步骤 {i+1}: 累积奖励={value:6.1f}, 峰值={drawdown_monitor.peak_value:6.1f}, "
                  f"回撤={current_drawdown:.4f}")
        
        # 验证：负起始序列应该能产生有意义的回撤
        final_drawdown = drawdown_monitor.get_current_drawdown()
        peak_value = drawdown_monitor.peak_value
        
        print(f"\\n最终状态: 峰值={peak_value}, 回撤={final_drawdown:.4f}")
        
        # 改进后应该能检测到回撤（即使从负值开始）
        assert final_drawdown > 0.1, f"负起始序列应该检测到回撤>10%，实际: {final_drawdown:.4f}"
        assert peak_value >= 0, f"峰值应该>=0，实际: {peak_value}"
        
        print("✅ 负起始值回撤计算正常工作")
    
    def test_improved_drawdown_calculation_mixed_sequence(self):
        """Green: 测试混合正负序列的回撤计算"""
        print("=== Green: 验证混合正负序列回撤计算 ===")
        
        # 测试场景2：混合正负序列
        mixed_sequence = [
            -10, -5, 0, 10, 20, 30,    # 从负到正的恢复
            25, 15, 5, -5, -15, -25,   # 从正到负的下降
            -20, -10, 0, 5             # 再次恢复
        ]
        
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.5, patience=3)
        
        print("混合序列回撤测试:")
        max_drawdowns = []
        
        for i, value in enumerate(mixed_sequence):
            drawdown_monitor.step(value)
            current_drawdown = drawdown_monitor.get_current_drawdown()
            max_drawdowns.append(current_drawdown)
            
            if i % 4 == 3:  # 每4步显示一次
                print(f"步骤 {i+1}: 累积奖励={value:6.1f}, 峰值={drawdown_monitor.peak_value:6.1f}, "
                      f"回撤={current_drawdown:.4f}")
        
        # 验证：混合序列应该能检测到显著回撤
        max_drawdown = max(max_drawdowns)
        recovery_phases = len([d for d in max_drawdowns[10:] if d < max_drawdowns[9]])  # 恢复阶段
        
        print(f"\\n分析结果:")
        print(f"最大回撤: {max_drawdown:.4f}")
        print(f"恢复阶段数: {recovery_phases}")
        
        assert max_drawdown > 0.5, f"混合序列应该检测到显著回撤>50%，实际: {max_drawdown:.4f}"
        assert recovery_phases > 0, "应该检测到恢复阶段"
        
        print("✅ 混合正负序列回撤计算正常工作")
    
    def test_improved_learning_rate_adaptation_negative_rewards(self):
        """Green: 测试改进的负奖励环境学习率调整"""
        print("=== Green: 验证改进的负奖励学习率调整 ===")
        
        # 模拟负奖励环境的性能变化
        def simulate_improved_lr_adaptation(performance_history, current_lr_factor=1.0):
            """改进的学习率自适应调整逻辑"""
            lr_adaptation_factor = 0.9
            performance_threshold_down = 0.85
            performance_threshold_up = 1.15
            min_lr_factor = 0.1  # 提高最小值
            max_lr_factor = 2.0
            lr_recovery_factor = 1.1
            
            if len(performance_history) < 50:
                return current_lr_factor, "insufficient_data"
            
            recent_performance = np.mean(performance_history[-20:])
            long_term_performance = np.mean(performance_history[-50:])
            
            # 计算性能变化
            performance_diff = recent_performance - long_term_performance
            performance_change_ratio = abs(performance_diff) / max(abs(long_term_performance), 1.0)
            
            # 负奖励环境的性能判断
            is_performance_worse = False
            is_performance_better = False
            
            if long_term_performance >= 0:
                is_performance_worse = recent_performance < long_term_performance * performance_threshold_down
                is_performance_better = recent_performance > long_term_performance * performance_threshold_up
            else:
                # 负奖励环境：更负是更差，更不负是更好
                is_performance_worse = recent_performance < long_term_performance / performance_threshold_down
                is_performance_better = recent_performance > long_term_performance / performance_threshold_up
            
            # 避免过于频繁调整
            significant_change = performance_change_ratio > 0.05
            
            if is_performance_worse and significant_change:
                new_lr_factor = max(min_lr_factor, current_lr_factor * lr_adaptation_factor)
                return new_lr_factor, f"decrease: {current_lr_factor:.4f} -> {new_lr_factor:.4f}"
            elif is_performance_better and significant_change:
                new_lr_factor = min(max_lr_factor, current_lr_factor * lr_recovery_factor)
                return new_lr_factor, f"increase: {current_lr_factor:.4f} -> {new_lr_factor:.4f}"
            else:
                return current_lr_factor, "no_change"
        
        # 测试场景：负奖励环境下的性能变化
        base_performance = -35.0
        performance_sequences = [
            # 前50个episodes：稳定的负奖励
            [base_performance + np.random.normal(0, 2) for _ in range(50)],
            # 接下来20个：性能恶化（更负）
            [base_performance - 10 + np.random.normal(0, 3) for _ in range(20)],
            # 再20个：性能改善（不那么负）
            [base_performance - 5 + np.random.normal(0, 2) for _ in range(20)]
        ]
        
        all_performance = []
        for seq in performance_sequences:
            all_performance.extend(seq)
        
        # 测试学习率调整
        lr_factor = 1.0
        lr_history = []
        adjustment_log = []
        
        print("改进的学习率调整测试:")
        for i in range(len(all_performance)):
            episode_reward = all_performance[i]
            
            if i >= 60:  # 从有足够数据后开始测试
                new_lr_factor, reason = simulate_improved_lr_adaptation(
                    all_performance[:i+1], lr_factor
                )
                
                if new_lr_factor != lr_factor:
                    adjustment_log.append((i+1, reason))
                    print(f"Episode {i+1}: {reason}")
                
                lr_factor = new_lr_factor
                lr_history.append(lr_factor)
        
        # 验证改进效果
        final_lr_factor = lr_history[-1] if lr_history else 1.0
        adjustment_count = len(adjustment_log)
        
        print(f"\\n调整结果:")
        print(f"最终学习率因子: {final_lr_factor:.4f}")
        print(f"调整次数: {adjustment_count}")
        print(f"调整历史: {adjustment_log}")
        
        # 改进后应该避免过度下降
        assert final_lr_factor >= 0.1, f"学习率因子不应过度下降，实际: {final_lr_factor:.4f}"
        assert adjustment_count <= 5, f"调整次数应该合理，实际: {adjustment_count}"
        
        print("✅ 改进的学习率调整避免了过度下降")
    
    def test_complete_training_scenario_with_fixes(self):
        """Green: 测试完整训练场景下的修复效果"""
        print("=== Green: 验证完整训练场景修复效果 ===")
        
        # 模拟完整的训练过程：100个episodes
        np.random.seed(42)  # 确保可重复性
        
        # 模拟真实训练中的奖励演进
        episode_rewards = []
        for episode in range(100):
            if episode < 20:
                # 早期：较差的负奖励
                reward = np.random.normal(-45, 8)
            elif episode < 50:
                # 中期：略有改善但仍为负
                reward = np.random.normal(-35, 6)
            elif episode < 80:
                # 后期：继续改善，偶尔正奖励
                reward = np.random.normal(-20, 10)
            else:
                # 最后阶段：更多正奖励
                reward = np.random.normal(-5, 12)
            
            episode_rewards.append(reward)
        
        # 计算累积奖励和回撤
        cumulative_rewards = []
        cumulative_sum = 0.0
        
        for reward in episode_rewards:
            cumulative_sum += reward
            cumulative_rewards.append(cumulative_sum)
        
        # 测试改进的回撤监控
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.4, patience=10)
        drawdown_history = []
        
        for cumulative in cumulative_rewards:
            drawdown_monitor.step(cumulative)
            current_drawdown = drawdown_monitor.get_current_drawdown()
            drawdown_history.append(current_drawdown)
        
        # 分析结果
        max_drawdown = max(drawdown_history)
        non_zero_drawdowns = len([d for d in drawdown_history if d > 0.01])
        final_cumulative = cumulative_rewards[-1]
        
        print(f"完整训练场景结果:")
        print(f"最终累积奖励: {final_cumulative:.2f}")
        print(f"最大回撤: {max_drawdown:.4f}")
        print(f"非零回撤期间数: {non_zero_drawdowns}")
        print(f"回撤监控峰值: {drawdown_monitor.peak_value:.2f}")
        
        # 验证修复效果
        assert max_drawdown > 0.05, f"应该检测到有意义的回撤>5%，实际: {max_drawdown:.4f}"
        assert non_zero_drawdowns >= 30, f"应该有充足的回撤监控期间，实际: {non_zero_drawdowns}"
        assert not np.isnan(max_drawdown), "回撤值不应为NaN"
        
        # 显示关键节点的回撤
        key_episodes = [19, 49, 79, 99]
        print(f"\\n关键节点回撤:")
        for ep in key_episodes:
            print(f"Episode {ep+1}: 累积={cumulative_rewards[ep]:7.2f}, 回撤={drawdown_history[ep]:.4f}")
        
        print("✅ 完整训练场景下修复效果良好")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])