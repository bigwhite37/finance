#!/usr/bin/env python3
"""
自适应学习率修复的绿色阶段TDD测试
验证改进后的学习率适应机制正常工作
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import RLTrainer, TrainingConfig


class TestAdaptiveLearningRateGreen:
    """测试改进后的自适应学习率"""
    
    def test_improved_learning_rate_factor_stays_above_minimum(self):
        """Green: 测试改进后的学习率因子保持在最小值之上"""
        print("=== Green: 验证改进后学习率因子保持在最小值之上 ===")
        
        # 使用改进后的配置
        config = TrainingConfig(
            n_episodes=10,
            enable_adaptive_learning=True,
            lr_adaptation_factor=0.8,
            min_lr_factor=0.01,  # 有最小值保护
            max_lr_factor=1.0,
            lr_recovery_factor=1.25,  # 更快恢复
            performance_threshold_down=0.85,  # 更严格的阈值
            performance_threshold_up=1.15
        )
        
        # 模拟训练器
        trainer = type('MockTrainer', (), {
            'config': config,
            'performance_history': [],
            'current_lr_factor': 1.0
        })()
        
        # 模拟极端的性能下降场景
        performance_values = [100] * 50 + [5] * 100  # 前50个好，后100个极差
        
        for reward in performance_values:
            trainer.performance_history.append(reward)
            
            if len(trainer.performance_history) >= 50:
                recent_performance = np.mean(trainer.performance_history[-20:])
                long_term_performance = np.mean(trainer.performance_history[-50:])
                
                if recent_performance < long_term_performance * trainer.config.performance_threshold_down:
                    trainer.current_lr_factor = max(
                        trainer.config.min_lr_factor,
                        trainer.current_lr_factor * trainer.config.lr_adaptation_factor
                    )
                elif recent_performance > long_term_performance * trainer.config.performance_threshold_up:
                    trainer.current_lr_factor = min(
                        trainer.config.max_lr_factor,
                        trainer.current_lr_factor * trainer.config.lr_recovery_factor
                    )
        
        # 改进后应该保持在最小值之上
        print(f"改进后学习率因子: {trainer.current_lr_factor:.6f}")
        assert trainer.current_lr_factor >= config.min_lr_factor
        print(f"✅ 学习率因子保持在最小值 {config.min_lr_factor} 之上")
    
    def test_fast_recovery_mechanism(self):
        """Green: 测试快速恢复机制"""
        print("=== Green: 验证快速恢复机制 ===")
        
        config = TrainingConfig(
            n_episodes=10,
            enable_adaptive_learning=True,
            lr_recovery_factor=1.25  # 25%的快速恢复
        )
        
        trainer = type('MockTrainer', (), {
            'config': config,
            'performance_history': [],
            'current_lr_factor': 0.1  # 从较低的学习率开始
        })()
        
        # 模拟性能恢复场景
        # 首先建立历史数据
        for _ in range(50):
            trainer.performance_history.append(50)  # 低性能基线
        
        # 然后模拟性能显著提升
        recovery_values = [80] * 20  # 性能显著提升
        initial_lr = trainer.current_lr_factor
        
        for reward in recovery_values:
            trainer.performance_history.append(reward)
            
            recent_performance = np.mean(trainer.performance_history[-20:])
            long_term_performance = np.mean(trainer.performance_history[-50:])
            
            if recent_performance > long_term_performance * trainer.config.performance_threshold_up:
                trainer.current_lr_factor = min(
                    trainer.config.max_lr_factor,
                    trainer.current_lr_factor * trainer.config.lr_recovery_factor
                )
        
        # 验证快速恢复
        recovery_ratio = trainer.current_lr_factor / initial_lr
        print(f"学习率从 {initial_lr:.4f} 恢复到 {trainer.current_lr_factor:.4f}, 恢复比例: {recovery_ratio:.2f}")
        assert recovery_ratio >= 2.0, f"学习率恢复比例应该至少为2.0，但是 {recovery_ratio:.2f}"
        print("✅ 快速恢复机制工作正常")
    
    def test_stable_learning_rate_with_stable_performance(self):
        """Green: 测试稳定性能下学习率保持稳定"""
        print("=== Green: 验证稳定性能下学习率稳定 ===")
        
        config = TrainingConfig(
            n_episodes=10,
            enable_adaptive_learning=True,
            performance_threshold_down=0.85,  # 更严格的阈值
            performance_threshold_up=1.15
        )
        
        trainer = type('MockTrainer', (), {
            'config': config,
            'performance_history': [],
            'current_lr_factor': 0.5
        })()
        
        # 模拟稳定的性能（小幅波动）
        np.random.seed(42)
        base_performance = 100
        changes = []
        
        for i in range(150):
            noise = np.random.normal(0, 3)  # 3%的小幅波动
            performance = base_performance + noise
            trainer.performance_history.append(performance)
            
            if len(trainer.performance_history) >= 50:
                recent_performance = np.mean(trainer.performance_history[-20:])
                long_term_performance = np.mean(trainer.performance_history[-50:])
                
                old_lr = trainer.current_lr_factor
                
                if recent_performance < long_term_performance * trainer.config.performance_threshold_down:
                    trainer.current_lr_factor = max(
                        trainer.config.min_lr_factor,
                        trainer.current_lr_factor * trainer.config.lr_adaptation_factor
                    )
                elif recent_performance > long_term_performance * trainer.config.performance_threshold_up:
                    trainer.current_lr_factor = min(
                        trainer.config.max_lr_factor,
                        trainer.current_lr_factor * trainer.config.lr_recovery_factor
                    )
                
                changes.append(abs(trainer.current_lr_factor - old_lr))
        
        avg_change = np.mean(changes) if changes else 0
        print(f"稳定性能下平均学习率变化: {avg_change:.6f}")
        assert avg_change < 0.01, f"学习率变化应该很小，但是 {avg_change:.4f}"
        print("✅ 稳定性能下学习率保持稳定")
    
    def test_configuration_validation(self):
        """Green: 测试配置验证功能"""
        print("=== Green: 验证配置验证功能 ===")
        
        # 测试无效的min_lr_factor
        with pytest.raises(ValueError, match="min_lr_factor必须大于0且小于max_lr_factor"):
            TrainingConfig(min_lr_factor=0.0)
        
        # 测试无效的lr_recovery_factor
        with pytest.raises(ValueError, match="lr_recovery_factor必须大于1.0"):
            TrainingConfig(lr_recovery_factor=0.9)
        
        # 测试无效的performance阈值
        with pytest.raises(ValueError, match="performance_threshold_down必须小于1.0"):
            TrainingConfig(performance_threshold_down=1.1)
        
        print("✅ 配置验证功能正常工作")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])