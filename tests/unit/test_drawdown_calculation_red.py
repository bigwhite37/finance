#!/usr/bin/env python3
"""
回撤计算问题的红色阶段TDD测试
验证回撤计算逻辑是否存在问题
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import DrawdownEarlyStopping


class TestDrawdownCalculationRed:
    """测试回撤计算问题 - Red阶段"""
    
    def test_drawdown_should_detect_realistic_trading_scenarios(self):
        """Red: 测试回撤计算应该能检测真实的交易场景"""
        print("=== Red: 验证回撤计算能检测真实交易场景 ===")
        
        # 模拟真实的累积奖励序列（应该有起伏）
        realistic_cumulative_rewards = [
            100,    # 起始
            120,    # 增长
            150,    # 继续增长
            140,    # 小幅下降 -> 应该产生回撤
            130,    # 继续下降 -> 回撤加大
            145,    # 反弹
            160,    # 新高
            155,    # 小幅下降
            170,    # 新高
            165     # 再次下降
        ]
        
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.2, patience=5)
        
        calculated_drawdowns = []
        
        for reward in realistic_cumulative_rewards:
            drawdown_monitor.step(reward)
            current_drawdown = drawdown_monitor.get_current_drawdown()
            calculated_drawdowns.append(current_drawdown)
            print(f"累积奖励: {reward}, 回撤: {current_drawdown:.4f}")
        
        # 验证回撤计算结果
        print(f"所有计算出的回撤: {calculated_drawdowns}")
        
        # 应该在下降期间检测到回撤
        # 从150下降到130，回撤应该是 (150-130)/150 = 0.1333
        expected_drawdown_at_index_4 = (150 - 130) / 150  # 约0.1333
        actual_drawdown_at_index_4 = calculated_drawdowns[4]
        
        print(f"期望回撤 (索引4): {expected_drawdown_at_index_4:.4f}")
        print(f"实际回撤 (索引4): {actual_drawdown_at_index_4:.4f}")
        
        assert abs(actual_drawdown_at_index_4 - expected_drawdown_at_index_4) < 0.001, \
            f"回撤计算错误: 期望 {expected_drawdown_at_index_4:.4f}, 实际 {actual_drawdown_at_index_4:.4f}"
        
        # 应该有非零回撤
        non_zero_drawdowns = [d for d in calculated_drawdowns if d > 0.001]
        assert len(non_zero_drawdowns) > 0, f"在真实场景中应该检测到回撤，但所有回撤都接近0: {calculated_drawdowns}"
        
        print("✅ 回撤计算逻辑对真实场景工作正常")
    
    def test_constantly_increasing_rewards_should_have_zero_drawdown(self):
        """Red: 测试持续增长的奖励应该产生零回撤"""
        print("=== Red: 验证持续增长奖励的回撤为0 ===")
        
        # 模拟训练日志中的情况：持续增长
        constantly_increasing_rewards = [174.0 * i for i in range(1, 201)]  # 模拟200个episode
        
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.2, patience=5)
        
        all_drawdowns = []
        
        for i, reward in enumerate(constantly_increasing_rewards, 1):
            drawdown_monitor.step(reward)
            current_drawdown = drawdown_monitor.get_current_drawdown()
            all_drawdowns.append(current_drawdown)
            
            if i % 50 == 0:  # 每50个episode记录一次
                print(f"Episode {i}: 累积奖励 {reward:.4f}, 回撤 {current_drawdown:.4f}")
        
        # 对于持续增长的序列，回撤应该始终为0
        max_drawdown = max(all_drawdowns)
        assert max_drawdown < 0.001, f"持续增长序列的最大回撤应该接近0，但得到 {max_drawdown:.6f}"
        
        print("✅ 持续增长序列确实产生零回撤（符合预期）")
    
    def test_realistic_training_should_not_have_constant_positive_rewards(self):
        """Red: 测试真实训练不应该有恒定的正奖励"""
        print("=== Red: 验证真实训练不应该有恒定正奖励 ===")
        
        # 从训练日志中提取的实际数据模式
        observed_episode_rewards = [174.12, 174.07, 174.12, 174.14, 173.94, 174.09, 173.94, 174.15, 174.15, 174.00]
        
        # 计算奖励的方差和范围
        reward_variance = np.var(observed_episode_rewards)
        reward_range = max(observed_episode_rewards) - min(observed_episode_rewards)
        reward_std = np.std(observed_episode_rewards)
        
        print(f"观察到的episode奖励: {observed_episode_rewards}")
        print(f"奖励方差: {reward_variance:.6f}")
        print(f"奖励范围: {reward_range:.6f}")
        print(f"奖励标准差: {reward_std:.6f}")
        
        # 真实的交易奖励应该有更大的变异性
        # 在强化学习中，especially in early training，我们期望看到更大的波动
        expected_min_variance = 1.0  # 至少应该有一些变化
        expected_min_range = 5.0     # 奖励范围应该至少有5个单位
        
        print(f"期望最小方差: {expected_min_variance}")
        print(f"期望最小范围: {expected_min_range}")
        
        # 这个测试应该失败，暴露奖励过于稳定的问题
        assert reward_variance >= expected_min_variance, \
            f"奖励变异性过低 ({reward_variance:.6f})，可能表示奖励计算有问题"
        
        assert reward_range >= expected_min_range, \
            f"奖励范围过小 ({reward_range:.6f})，真实交易应该有更大波动"
        
        print("✅ 奖励有足够的变异性（符合真实交易预期）")
    
    def test_drawdown_calculation_logic_correctness(self):
        """Red: 测试回撤计算逻辑的正确性"""
        print("=== Red: 验证回撤计算逻辑正确性 ===")
        
        drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.3, patience=5)
        
        # 测试用例：已知的回撤计算
        test_cases = [
            # (cumulative_reward, expected_drawdown_after_peak, description)
            (100, 0.0, "初始值"),
            (150, 0.0, "创新高"),
            (120, (150-120)/150, "从150下降到120"),  # 应该是0.2
            (110, (150-110)/150, "进一步下降到110"),  # 应该是0.2667
            (160, 0.0, "创新高，回撤重置"),
            (140, (160-140)/160, "从160下降到140"),   # 应该是0.125
        ]
        
        for i, (reward, expected_drawdown, description) in enumerate(test_cases):
            drawdown_monitor.step(reward)
            actual_drawdown = drawdown_monitor.get_current_drawdown()
            
            print(f"步骤 {i+1}: {description}")
            print(f"  累积奖励: {reward}")
            print(f"  期望回撤: {expected_drawdown:.4f}")
            print(f"  实际回撤: {actual_drawdown:.4f}")
            
            assert abs(actual_drawdown - expected_drawdown) < 0.001, \
                f"步骤 {i+1} 回撤计算错误: 期望 {expected_drawdown:.4f}, 实际 {actual_drawdown:.4f}"
        
        print("✅ 回撤计算逻辑完全正确")
    
    def test_identify_root_cause_of_zero_drawdown(self):
        """Red: 测试识别零回撤的根本原因"""
        print("=== Red: 识别零回撤的根本原因 ===")
        
        # 分析问题：为什么训练中一直是零回撤？
        
        # 原因1：奖励过于稳定
        stable_rewards = [174] * 200  # 完全相同的奖励
        cumulative_stable = [sum(stable_rewards[:i+1]) for i in range(len(stable_rewards))]
        
        # 原因2：奖励始终为正且相似
        similar_positive_rewards = [174 + np.random.normal(0, 0.1) for _ in range(200)]  # 极小变化
        cumulative_similar = [sum(similar_positive_rewards[:i+1]) for i in range(len(similar_positive_rewards))]
        
        # 测试这两种情况的回撤
        for scenario_name, cumulative_rewards in [
            ("完全稳定奖励", cumulative_stable),
            ("微小变化正奖励", cumulative_similar)
        ]:
            drawdown_monitor = DrawdownEarlyStopping(max_drawdown=0.2, patience=5)
            drawdowns = []
            
            for reward in cumulative_rewards:
                drawdown_monitor.step(reward)
                drawdowns.append(drawdown_monitor.get_current_drawdown())
            
            max_drawdown = max(drawdowns)
            print(f"{scenario_name}: 最大回撤 = {max_drawdown:.6f}")
            
            # 这些场景确实应该产生很小的回撤
            assert max_drawdown < 0.01, f"{scenario_name} 应该产生接近零的回撤"
        
        # 根本问题：需要检查奖励函数实现
        print("❌ 根本问题：训练中的奖励过于稳定，缺乏真实的市场波动")
        print("❌ 需要检查：")
        print("  1. 奖励函数的计算逻辑")
        print("  2. 是否正确反映了交易盈亏")
        print("  3. 是否包含了市场风险和波动性")
        
        # 这个断言应该失败，指出真正的问题
        assert False, "发现根本问题：奖励函数过于稳定，无法反映真实交易的不确定性"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])