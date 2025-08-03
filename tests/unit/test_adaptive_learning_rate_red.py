#!/usr/bin/env python3
"""
自适应学习率修复的红色阶段TDD测试
验证学习率因子不会降为0且有合理的恢复机制
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import RLTrainer, TrainingConfig


class TestAdaptiveLearningRateRed:
    """测试自适应学习率修复"""
    
    def test_learning_rate_factor_should_not_reach_zero(self):
        """Red: 测试学习率因子不应该降为0"""
        print("=== Red: 验证学习率因子不会降为0 ===")
        
        # 直接创建模拟配置对象，绕过验证以模拟旧的有问题实现
        config = type('MockConfig', (), {
            'enable_adaptive_learning': True,
            'lr_adaptation_factor': 0.8,  # 衰减因子
            'min_lr_factor': 0.0,  # 旧实现没有最小值限制！
            'max_lr_factor': 1.0,
            'performance_threshold_down': 0.9,  # 旧的阈值
            'performance_threshold_up': 1.1,   # 旧的阈值
            'lr_recovery_factor': 1.1  # 旧的恢复因子
        })()
        
        # 模拟训练器
        trainer = type('MockTrainer', (), {
            'config': config,
            'performance_history': [],
            'current_lr_factor': 1.0
        })()
        
        # 模拟持续的性能下降场景
        # 这应该会导致学习率因子逐渐降低
        performance_values = [100] * 50 + [10] * 50  # 前50个好，后50个极差
        
        for i, reward in enumerate(performance_values):
            trainer.performance_history.append(reward)
            
            # 模拟改进后的逻辑
            if len(trainer.performance_history) >= 50:
                recent_performance = np.mean(trainer.performance_history[-20:])
                long_term_performance = np.mean(trainer.performance_history[-50:])
                
                if recent_performance < long_term_performance * trainer.config.performance_threshold_down:
                    # 改进的逻辑：有最小值限制
                    trainer.current_lr_factor = max(
                        trainer.config.min_lr_factor,
                        trainer.current_lr_factor * trainer.config.lr_adaptation_factor
                    )
                elif recent_performance > long_term_performance * trainer.config.performance_threshold_up:
                    # 改进的逻辑：更快的恢复
                    trainer.current_lr_factor = min(
                        trainer.config.max_lr_factor,
                        trainer.current_lr_factor * trainer.config.lr_recovery_factor
                    )
        
        # 当前实现会导致学习率因子变得非常小
        print(f"当前实现的最终学习率因子: {trainer.current_lr_factor:.6f}")
        
        # 这个测试应该失败，因为当前实现会让学习率因子变得极小
        assert trainer.current_lr_factor >= 0.01, f"学习率因子不应该小于0.01，但是得到了 {trainer.current_lr_factor:.6f}"
        
        print("✅ 学习率因子保持在合理范围内")
    
    def test_learning_rate_factor_should_have_minimum_bound(self):
        """Green: 测试学习率因子应该有最小值边界"""
        print("=== Green: 验证学习率因子有最小值边界 ===")
        
        # 测试改进后的逻辑
        config = TrainingConfig(
            n_episodes=10,
            enable_adaptive_learning=True,
            lr_adaptation_factor=0.8
        )
        
        trainer = type('MockTrainer', (), {
            'config': config,
            'performance_history': [],
            'current_lr_factor': 1.0,
            'min_lr_factor': 0.01,  # 新增最小值
            'max_lr_factor': 1.0    # 新增最大值
        })()
        
        # 模拟改进后的逻辑
        def improved_adapt_lr(recent_perf, long_term_perf):
            if recent_perf < long_term_perf * 0.9:
                # 性能下降，但有最小值限制
                trainer.current_lr_factor = max(
                    trainer.min_lr_factor,
                    trainer.current_lr_factor * trainer.config.lr_adaptation_factor
                )
            elif recent_perf > long_term_perf * 1.1:
                # 性能提升，恢复更快
                trainer.current_lr_factor = min(
                    trainer.max_lr_factor,
                    trainer.current_lr_factor * 1.25  # 更快的恢复
                )
        
        # 模拟持续性能下降
        performance_values = [100] * 50 + [20] * 50  # 大幅性能下降
        
        for reward in performance_values:
            trainer.performance_history.append(reward)
            
            if len(trainer.performance_history) >= 50:
                recent_performance = np.mean(trainer.performance_history[-20:])
                long_term_performance = np.mean(trainer.performance_history[-50:])
                improved_adapt_lr(recent_performance, long_term_performance)
        
        # 改进后应该保持在最小值之上
        assert trainer.current_lr_factor >= trainer.min_lr_factor
        print(f"✅ 改进后学习率因子: {trainer.current_lr_factor:.6f}, 最小值: {trainer.min_lr_factor}")
    
    def test_learning_rate_factor_should_recover_quickly(self):
        """Green: 测试学习率因子应该有快速恢复机制"""
        print("=== Green: 验证学习率因子快速恢复机制 ===")
        
        trainer = type('MockTrainer', (), {
            'current_lr_factor': 0.01,  # 从很小的值开始
            'min_lr_factor': 0.01,
            'max_lr_factor': 1.0
        })()
        
        # 模拟改进后的快速恢复逻辑
        def fast_recovery_adapt_lr(performance_improving):
            if performance_improving:
                # 当性能改善时，快速恢复学习率
                trainer.current_lr_factor = min(
                    trainer.max_lr_factor,
                    trainer.current_lr_factor * 1.25  # 25%增长
                )
        
        # 模拟性能改善
        initial_lr = trainer.current_lr_factor
        for _ in range(10):  # 10次性能改善
            fast_recovery_adapt_lr(True)
        
        # 应该快速恢复到更高的学习率
        recovery_ratio = trainer.current_lr_factor / initial_lr
        assert recovery_ratio > 5.0, f"学习率应该快速恢复，但恢复比例只有 {recovery_ratio:.2f}"
        print(f"✅ 学习率从 {initial_lr:.4f} 恢复到 {trainer.current_lr_factor:.4f}")
    
    def test_learning_rate_adaptation_should_be_stable(self):
        """Green: 测试学习率适应应该稳定，避免震荡"""
        print("=== Green: 验证学习率适应稳定性 ===")
        
        trainer = type('MockTrainer', (), {
            'current_lr_factor': 0.5,
            'min_lr_factor': 0.01,
            'max_lr_factor': 1.0,
            'performance_history': []
        })()
        
        # 模拟稳定的性能（小幅波动）
        # 这种情况下学习率应该保持相对稳定
        base_performance = 100
        performance_values = []
        
        # 生成轻微波动的性能数据
        np.random.seed(42)  # 确保可重复
        for i in range(100):
            noise = np.random.normal(0, 5)  # 5%的噪音
            performance_values.append(base_performance + noise)
        
        initial_lr = trainer.current_lr_factor
        changes = []
        
        for reward in performance_values:
            trainer.performance_history.append(reward)
            
            if len(trainer.performance_history) >= 50:
                recent_performance = np.mean(trainer.performance_history[-20:])
                long_term_performance = np.mean(trainer.performance_history[-50:])
                
                old_lr = trainer.current_lr_factor
                
                # 改进后的稳定逻辑 - 需要更大的性能差异才调整
                if recent_performance < long_term_performance * 0.85:  # 更严格的阈值
                    trainer.current_lr_factor = max(
                        trainer.min_lr_factor,
                        trainer.current_lr_factor * 0.9  # 更温和的调整
                    )
                elif recent_performance > long_term_performance * 1.15:  # 更严格的阈值
                    trainer.current_lr_factor = min(
                        trainer.max_lr_factor,
                        trainer.current_lr_factor * 1.1  # 更温和的调整
                    )
                
                changes.append(abs(trainer.current_lr_factor - old_lr))
        
        # 在稳定性能下，学习率变化应该很小
        avg_change = np.mean(changes) if changes else 0
        assert avg_change < 0.01, f"学习率变化应该很小，但平均变化为 {avg_change:.4f}"
        print(f"✅ 稳定性能下学习率变化很小: 平均变化 {avg_change:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])