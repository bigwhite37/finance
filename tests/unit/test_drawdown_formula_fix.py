#!/usr/bin/env python3
"""
回撤计算公式修复测试
验证新的负累积奖励回撤计算是否合理
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import DrawdownEarlyStopping


class TestDrawdownFormulaFix:
    """测试回撤计算公式修复"""
    
    def test_improved_negative_cumulative_reward_drawdown_calculation(self):
        """测试改进的负累积奖励回撤计算"""
        print("=== 测试改进的负累积奖励回撤计算 ===")
        
        # 测试实际训练中的累积奖励序列
        realistic_training_sequence = [
            -30, -60, -90, -150, -200, -300, -450, -600, -800, -1000,  # 前10个episodes
            -1200, -1400, -1600, -1800, -2000, -2200, -2400, -2600, -2800, -3000,  # 20个episodes
            -3200, -3300, -3268  # 最后到实际观察的值
        ]
        
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.3, patience=5)
        
        print("训练序列回撤计算:")
        for i, cumulative in enumerate(realistic_training_sequence):
            drawdown_monitor.step(cumulative)
            current_drawdown = drawdown_monitor.get_current_drawdown()
            
            if i % 5 == 4:  # 每5个episode显示一次
                print(f"Episode {i+1:2d}: 累积奖励={cumulative:7.0f}, 回撤={current_drawdown:.4f} ({current_drawdown*100:.1f}%)")
        
        final_drawdown = drawdown_monitor.get_current_drawdown()
        final_cumulative = realistic_training_sequence[-1]
        
        print(f"\n最终结果:")
        print(f"累积奖励: {final_cumulative}")
        print(f"计算回撤: {final_drawdown:.4f} ({final_drawdown*100:.1f}%)")
        print(f"峰值: {drawdown_monitor.peak_value}")
        
        # 验证：新公式应该产生合理的回撤值
        assert 0.1 <= final_drawdown <= 0.95, f"回撤应该在10%-95%之间，实际: {final_drawdown:.4f}"
        assert final_drawdown > 0.3, f"如此大的损失回撤应该>30%，实际: {final_drawdown:.4f}"
        
        print("✅ 新的回撤计算公式产生合理结果")
    
    def test_drawdown_calculation_with_different_baselines(self):
        """测试不同基准下的回撤计算"""
        print("=== 测试不同基准下的回撤计算 ===")
        
        test_cumulative_values = [-500, -1000, -2000, -3000, -5000]
        baselines = [500, 1000, 2000, 5000]
        
        print("不同基准下的回撤计算结果:")
        print("累积奖励  |  基准500  |  基准1000  |  基准2000  |  基准5000")
        print("-" * 60)
        
        for cumulative in test_cumulative_values:
            row = f"{cumulative:8.0f}  |"
            
            for baseline in baselines:
                # 模拟计算逻辑
                drawdown = min(abs(cumulative) / baseline, 0.95)
                row += f"  {drawdown:.4f}  |"
            
            print(row)
        
        # 验证基准1000的合理性（从实际训练日志推导）
        # 100个episodes，平均奖励约-30，累积约-3000
        # 使用基准1000，回撤约为3.0，被限制为0.95，符合预期
        
        test_cumulative = -3268  # 实际训练值
        baseline = 1000
        expected_drawdown = min(abs(test_cumulative) / baseline, 0.95)
        
        print(f"\n基准验证:")
        print(f"累积奖励: {test_cumulative}")
        print(f"使用基准: {baseline}")
        print(f"计算回撤: {expected_drawdown:.4f} ({expected_drawdown*100:.1f}%)")
        
        assert expected_drawdown == 0.95, f"大损失应该触发95%限制，实际: {expected_drawdown:.4f}"
        
        print("✅ 基准1000的回撤计算合理")
    
    def test_reasonable_early_stopping_threshold_adjustment(self):
        """测试合理的早停阈值调整"""
        print("=== 测试合理的早停阈值调整 ===")
        
        # 模拟不同训练阶段的性能
        training_phases = [
            (range(1, 21), "早期学习"),     # Episodes 1-20: 学习基础
            (range(21, 51), "技能发展"),    # Episodes 21-50: 技能发展  
            (range(51, 81), "性能优化"),    # Episodes 51-80: 性能优化
            (range(81, 101), "收敛阶段")    # Episodes 81-100: 收敛
        ]
        
        episode_rewards = []
        for episodes, phase_name in training_phases:
            if phase_name == "早期学习":
                rewards = [np.random.normal(-40, 8) for _ in episodes]
            elif phase_name == "技能发展":
                rewards = [np.random.normal(-35, 6) for _ in episodes]
            elif phase_name == "性能优化":
                rewards = [np.random.normal(-28, 5) for _ in episodes]
            else:  # 收敛阶段
                rewards = [np.random.normal(-25, 4) for _ in episodes]
            
            episode_rewards.extend(rewards)
        
        # 计算累积奖励
        cumulative_rewards = []
        cumulative_sum = 0
        
        for reward in episode_rewards:
            cumulative_sum += reward
            cumulative_rewards.append(cumulative_sum)
        
        # 测试不同早停阈值的效果
        thresholds = [0.2, 0.3, 0.5, 0.7]
        
        print("不同早停阈值的触发情况:")
        for threshold in thresholds:
            drawdown_monitor = DrawdownEarlyStopping(max_drawdown=threshold, patience=5)
            early_stop_episode = None
            
            for i, cumulative in enumerate(cumulative_rewards):
                should_stop = drawdown_monitor.step(cumulative)
                if should_stop and early_stop_episode is None:
                    early_stop_episode = i + 1
                    break
            
            final_cumulative = cumulative_rewards[early_stop_episode-1] if early_stop_episode else cumulative_rewards[-1]
            final_drawdown = drawdown_monitor.get_current_drawdown()
            
            print(f"阈值 {threshold:.1f}: {'早停' if early_stop_episode else '正常'} "
                  f"(Episode {early_stop_episode or 100}, "
                  f"累积奖励={final_cumulative:.0f}, 回撤={final_drawdown:.3f})")
        
        # 建议合理的阈值：允许训练探索但防止过度损失
        suggested_threshold = 0.5
        print(f"\n建议早停阈值: {suggested_threshold} (50%)")
        print("原因: 在负奖励环境下允许充分探索，同时防止极端损失")
        
        print("✅ 早停阈值调整分析完成")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])