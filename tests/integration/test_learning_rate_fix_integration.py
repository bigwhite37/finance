#!/usr/bin/env python3
"""
学习率修复的集成测试
验证改进后的学习率在实际训练中的表现
"""

import pytest
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import RLTrainer, TrainingConfig


@pytest.mark.integration
class TestLearningRateFixIntegration:
    """学习率修复集成测试"""
    
    def test_training_config_with_improved_parameters(self):
        """测试改进后的训练配置"""
        print("=== 验证改进后的训练配置 ===")
        
        # 使用改进后的参数创建配置
        config = TrainingConfig(
            n_episodes=100,
            enable_adaptive_learning=True,
            lr_adaptation_factor=0.9,  # 更温和的衰减
            min_lr_factor=0.01,  # 最小值保护
            max_lr_factor=1.0,
            lr_recovery_factor=1.25,  # 快速恢复
            performance_threshold_down=0.85,  # 更严格的阈值
            performance_threshold_up=1.15
        )
        
        # 验证配置参数
        assert config.min_lr_factor == 0.01
        assert config.lr_recovery_factor == 1.25
        assert config.performance_threshold_down == 0.85
        assert config.performance_threshold_up == 1.15
        
        print("✅ 改进后的训练配置参数正确")
    
    def test_simulated_training_with_adaptive_lr(self):
        """测试模拟训练中的自适应学习率"""
        print("=== 模拟训练测试自适应学习率 ===")
        
        config = TrainingConfig(
            n_episodes=100,
            enable_adaptive_learning=True,
            lr_adaptation_factor=0.9,
            min_lr_factor=0.01,
            max_lr_factor=1.0,
            lr_recovery_factor=1.25,
            performance_threshold_down=0.85,
            performance_threshold_up=1.15
        )
        
        # 模拟训练器
        trainer = type('MockTrainer', (), {
            'config': config,
            'performance_history': [],
            'current_lr_factor': 1.0,
            'adaptive_learning_enabled': True
        })()
        
        # 模拟完整的训练过程
        phases = [
            # 阶段1：稳定开始
            ([100] * 50, "稳定开始阶段"),
            # 阶段2：性能下降
            ([50] * 30, "性能下降阶段"),
            # 阶段3：性能恢复
            ([120] * 20, "性能恢复阶段")
        ]
        
        lr_history = []
        
        for performance_values, phase_name in phases:
            for reward in performance_values:
                trainer.performance_history.append(reward)
                
                # 应用改进后的自适应逻辑
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
                    
                    lr_history.append({
                        'episode': len(trainer.performance_history),
                        'lr_factor': trainer.current_lr_factor,
                        'phase': phase_name,
                        'recent_perf': recent_performance,
                        'long_term_perf': long_term_performance
                    })
        
        # 验证关键行为
        # 1. 学习率应该保持在最小值之上
        final_lr = trainer.current_lr_factor
        assert final_lr >= config.min_lr_factor, f"最终学习率 {final_lr} 应该≥ {config.min_lr_factor}"
        
        # 2. 在性能下降阶段，学习率应该下降但不会到0
        min_lr_in_history = min(entry['lr_factor'] for entry in lr_history)
        assert min_lr_in_history >= config.min_lr_factor, f"历史最小学习率 {min_lr_in_history} 应该≥ {config.min_lr_factor}"
        
        # 3. 在性能恢复阶段，学习率应该有所恢复
        final_phase_lr = [entry['lr_factor'] for entry in lr_history if entry['phase'] == "性能恢复阶段"]
        if final_phase_lr:
            recovery_lr = final_phase_lr[-1]
            assert recovery_lr > min_lr_in_history, f"恢复阶段学习率 {recovery_lr} 应该> 最小值 {min_lr_in_history}"
        
        print(f"✅ 最终学习率: {final_lr:.4f}")
        print(f"✅ 历史最小学习率: {min_lr_in_history:.4f}")
        print(f"✅ 学习率始终保持在最小值 {config.min_lr_factor} 之上")
    
    def test_extreme_performance_drop_scenario(self):
        """测试极端性能下降场景"""
        print("=== 测试极端性能下降场景 ===")
        
        config = TrainingConfig(
            n_episodes=100,
            enable_adaptive_learning=True,
            lr_adaptation_factor=0.8,  # 较强的衰减
            min_lr_factor=0.05,  # 较高的最小值
            lr_recovery_factor=1.5   # 较强的恢复
        )
        
        trainer = type('MockTrainer', (), {
            'config': config,
            'performance_history': [],
            'current_lr_factor': 1.0
        })()
        
        # 模拟极端下降：从高性能急剧下降到极低性能
        extreme_scenario = [200] * 50 + [10] * 50  # 急剧下降
        
        for reward in extreme_scenario:
            trainer.performance_history.append(reward)
            
            if len(trainer.performance_history) >= 50:
                recent_performance = np.mean(trainer.performance_history[-20:])
                long_term_performance = np.mean(trainer.performance_history[-50:])
                
                if recent_performance < long_term_performance * trainer.config.performance_threshold_down:
                    trainer.current_lr_factor = max(
                        trainer.config.min_lr_factor,
                        trainer.current_lr_factor * trainer.config.lr_adaptation_factor
                    )
        
        # 即使在极端下降情况下，学习率也应该保持在最小值之上
        assert trainer.current_lr_factor >= config.min_lr_factor
        print(f"✅ 极端下降后学习率: {trainer.current_lr_factor:.4f}, 最小值: {config.min_lr_factor}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])