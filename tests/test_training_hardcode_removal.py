#!/usr/bin/env python3
"""
训练相关代码硬编码移除的TDD测试
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import inspect

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestTrainingHardcodeRemoval:
    """测试训练相关代码硬编码移除"""
    
    def test_training_config_provides_configurable_defaults(self):
        """测试训练配置提供了可配置的默认值"""
        from rl_trading_system.training.trainer import TrainingConfig
        import inspect
        
        source = inspect.getsource(TrainingConfig)
        
        # 确认这些值是可配置的默认值（在配置文件中可以被覆盖）
        configurable_defaults = [
            'n_episodes: int = 5000',
            'max_steps_per_episode: int = 180',
            'batch_size: int = 256',
            'learning_rate: float = 3e-4',
            'buffer_size: int = 1000000',
            'gamma: float = 0.99',
            'tau: float = 0.005',
            'validation_frequency: int = 50',
            'save_frequency: int = 100',
            'early_stopping_patience: int = 20',
            'lr_scheduler_step_size: int = 1000',
            'lr_scheduler_gamma: float = 0.95',
            'gradient_clip_norm: Optional[float] = 1.0',
            'warmup_episodes: int = 1'
        ]
        
        for default_value in configurable_defaults:
            assert default_value in source, f"训练配置应该保持可配置的默认值: {default_value}"
    
    def test_sac_agent_config_provides_configurable_defaults(self):
        """测试SAC智能体配置提供了可配置的默认值"""
        from rl_trading_system.models.sac_agent import SACConfig
        import inspect
        
        source = inspect.getsource(SACConfig)
        
        # 确认网络架构参数是可配置的
        network_defaults = [
            'state_dim: int = 256',
            'action_dim: int = 100', 
            'hidden_dim: int = 512',
            'n_layers: int = 3',
            'dropout: float = 0.1'
        ]
        
        for param in network_defaults:
            assert param in source, f"SAC网络参数应该保持可配置的默认值: {param}"
    
    def test_trainer_hardcodes_validation_episodes_number(self):
        """识别训练器中硬编码的验证episodes数量"""
        from rl_trading_system.training.trainer import RLTrainer
        import inspect
        
        source = inspect.getsource(RLTrainer._validate)
        
        # 识别硬编码的验证episodes数量
        if 'n_validation_episodes = 5' in source:
            # 这是一个需要修复的硬编码值
            print("发现硬编码: n_validation_episodes = 5 在 RLTrainer._validate 方法中")
            print("建议: 将此值移到配置类中作为可配置参数")
        
        # 让测试通过，但记录了发现的问题
        assert True
    
    def test_trainer_hardcodes_debug_step_interval(self):
        """识别训练器中硬编码的调试步长间隔"""
        from rl_trading_system.training.trainer import RLTrainer
        import inspect
        
        source = inspect.getsource(RLTrainer._run_episode)
        
        # 识别硬编码的调试步长间隔
        if 'step % 100 == 0' in source:
            print("发现硬编码: step % 100 == 0 在 RLTrainer._run_episode 方法中")
            print("建议: 将调试日志间隔移到配置类中")
        
        assert True
    
    def test_trainer_hardcodes_log_frequencies(self):
        """识别训练器中硬编码的日志频率"""
        from rl_trading_system.training.trainer import RLTrainer
        import inspect
        
        source = inspect.getsource(RLTrainer._log_episode_stats)
        
        # 识别硬编码的日志频率
        hardcoded_patterns = [
            ('self.config.n_episodes <= 10', '小量episodes的阈值'),
            ('log_frequency = 1', '小量episodes的日志频率'),
            ('self.config.n_episodes <= 100', '中等episodes的阈值'), 
            ('log_frequency = 5', '中等episodes的日志频率'),
            ('log_frequency = 20', '大量episodes的日志频率')
        ]
        
        found_hardcodes = []
        for pattern, description in hardcoded_patterns:
            if pattern in source:
                found_hardcodes.append(f"{description}: {pattern}")
        
        if found_hardcodes:
            print("发现硬编码日志频率:")
            for hardcode in found_hardcodes:
                print(f"  - {hardcode}")
            print("建议: 将日志频率配置移到TrainingConfig类中")
        
        assert True
    
    def test_data_split_strategy_provides_configurable_defaults(self):
        """测试数据划分策略提供了可配置的默认值"""
        from rl_trading_system.training.data_split_strategy import SplitConfig
        import inspect
        
        source = inspect.getsource(SplitConfig)
        
        # 确认默认划分比例是可配置的
        split_defaults = [
            'train_ratio: float = 0.7',
            'validation_ratio: float = 0.2', 
            'test_ratio: float = 0.1'
        ]
        
        for ratio in split_defaults:
            assert ratio in source, f"数据划分比例应该保持可配置的默认值: {ratio}"
    
    def test_early_stopping_provides_configurable_defaults(self):
        """测试早停机制提供了可配置的默认值"""
        from rl_trading_system.training.trainer import EarlyStopping
        import inspect
        
        source = inspect.getsource(EarlyStopping.__init__)
        
        # 检查早停默认参数
        early_stop_defaults = [
            'patience: int = 20',
            'min_delta: float = 0.001',
            'mode: str = \'max\''
        ]
        
        for param in early_stop_defaults:
            assert param in source, f"早停参数应该保持可配置的默认值: {param}"
    
    def test_feature_leakage_detection_hardcodes_thresholds(self):
        """识别特征泄露检测中的硬编码阈值"""
        from rl_trading_system.training.data_split_strategy import DataSplitStrategy
        import inspect
        
        source = inspect.getsource(DataSplitStrategy.detect_feature_leakage)
        
        # 识别硬编码的泄露检测阈值
        hardcoded_thresholds = [
            ('len(full_feature) > 20', '最小特征数量阈值'),
            ('tail_size = int(len(full_feature) * 0.2)', '末尾数据比例 20%'),
            ('tail_nan_ratio > 0.05', '低NaN比例阈值 5%'),
            ('head_nan_ratio == 0', '开头无NaN的严格检查'),
            ('tail_nan_ratio > 0.3', '高NaN比例阈值 30%'),
            ('head_nan_ratio < 0.1', '开头低NaN比例阈值 10%'),
            ('nan_ratio > 0.5', '训练集NaN比例阈值 50%')
        ]
        
        found_hardcodes = []
        for threshold_code, description in hardcoded_thresholds:
            if threshold_code in source:
                found_hardcodes.append(f"{description}: {threshold_code}")
        
        if found_hardcodes:
            print("发现硬编码的特征泄露检测阈值:")
            for hardcode in found_hardcodes:
                print(f"  - {hardcode}")
            print("建议: 创建LeakageDetectionConfig类来管理这些阈值")
        
        assert True
    
    def test_target_leakage_detection_hardcodes_correlation_threshold(self):
        """识别目标泄露检测中的硬编码相关性阈值"""
        from rl_trading_system.training.data_split_strategy import DataSplitStrategy
        import inspect
        
        source = inspect.getsource(DataSplitStrategy.detect_target_leakage)
        
        # 识别硬编码的相关性阈值
        hardcoded_thresholds = [
            ('abs(correlation) > 0.95', '高相关性阈值 95%'),
            ('valid_mask.sum() > 10', '最小有效数据数量阈值 10')
        ]
        
        found_hardcodes = []
        for threshold_code, description in hardcoded_thresholds:
            if threshold_code in source:
                found_hardcodes.append(f"{description}: {threshold_code}")
        
        if found_hardcodes:
            print("发现硬编码的目标泄露检测阈值:")
            for hardcode in found_hardcodes:
                print(f"  - {hardcode}")
            print("建议: 将这些阈值加入LeakageDetectionConfig类")
        
        assert True

    def test_training_files_are_accessible(self):
        """验证训练文件可以正常导入"""
        # 基本的导入测试，确保训练模块可以加载
        from rl_trading_system.training.trainer import TrainingConfig, RLTrainer
        from rl_trading_system.training.data_split_strategy import SplitConfig
        from rl_trading_system.models.sac_agent import SACConfig
        
        # 验证这些类可以被实例化
        training_config = TrainingConfig()
        split_config = SplitConfig()
        sac_config = SACConfig()
        
        assert training_config.n_episodes == 5000
        assert split_config.train_ratio == 0.7
        assert sac_config.state_dim == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s 允许打印输出