#!/usr/bin/env python3
"""
多核训练优化的集成测试
验证多核训练的端到端功能
"""

import pytest
import time
from pathlib import Path
import sys
import numpy as np
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import RLTrainer, TrainingConfig
from rl_trading_system.training.data_split_strategy import SplitResult


@pytest.mark.integration
class TestMulticoreTrainingIntegration:
    """多核训练集成测试"""
    
    def test_multicore_training_configuration_end_to_end(self):
        """测试多核训练配置的端到端功能"""
        print("=== 集成测试: 多核训练配置端到端 ===")
        
        # 创建测试配置
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                n_episodes=5,  # 少量episodes用于快速测试
                enable_multiprocessing=True,
                num_workers=2,
                parallel_environments=1,  # 简化测试
                data_loader_workers=1,
                save_dir=temp_dir
            )
            
            # 验证配置正确加载
            assert config.enable_multiprocessing is True
            assert config.num_workers == 2
            assert config.parallel_environments == 1
            
            print(f"✅ 多核配置正确创建:")
            print(f"  - 多进程: {config.enable_multiprocessing}")
            print(f"  - 工作进程: {config.num_workers}")
            print(f"  - 并行环境: {config.parallel_environments}")
    
    def test_multicore_features_initialization(self):
        """测试多核特性初始化"""
        print("=== 集成测试: 多核特性初始化 ===")
        
        # 创建简化的模拟组件
        class MockEnvironment:
            def reset(self):
                return np.random.randn(10)
            
            def step(self, action):
                return np.random.randn(10), np.random.randn(), False, {}
            
            def close(self):
                pass
        
        class MockAgent:
            def act(self, state):
                return np.random.randn(3)
            
            def get_action(self, state, deterministic=False):
                return np.random.randn(3)
            
            def update(self, *args, **kwargs):
                return {'actor_loss': 0.1, 'critic_loss': 0.2}
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        class MockDataSplit:
            def __init__(self):
                self.train_indices = list(range(100))
                self.val_indices = list(range(100, 150))
                self.test_indices = list(range(150, 200))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                n_episodes=2,
                enable_multiprocessing=True,
                num_workers=2,
                save_dir=temp_dir
            )
            
            # 创建训练器
            trainer = RLTrainer(
                config=config,
                environment=MockEnvironment(),
                agent=MockAgent(),
                data_split=MockDataSplit()
            )
            
            # 验证多核组件已初始化
            assert hasattr(trainer, 'parallel_env_manager')
            assert hasattr(trainer, 'scaler')  # 混合精度缩放器
            
            print("✅ 多核特性初始化成功")
    
    def test_training_with_multicore_optimization_disabled(self):
        """测试禁用多核优化的训练"""
        print("=== 集成测试: 禁用多核优化的训练 ===")
        
        # 创建简化的训练环境
        class MockEnvironment:
            def __init__(self):
                self.step_count = 0
            
            def reset(self):
                self.step_count = 0
                return np.random.randn(10)
            
            def step(self, action):
                self.step_count += 1
                done = self.step_count >= 5  # 5步后结束
                reward = np.random.randn()
                next_state = np.random.randn(10)
                return next_state, reward, done, {}
        
        class MockAgent:
            def act(self, state):
                return np.random.randn(3)
            
            def get_action(self, state, deterministic=False):
                return np.random.randn(3)
            
            def update(self, *args, **kwargs):
                return {'actor_loss': 0.1, 'critic_loss': 0.2}
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        class MockDataSplit:
            def __init__(self):
                self.train_indices = list(range(100))
                self.val_indices = list(range(100, 150))
                self.test_indices = list(range(150, 200))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 禁用多核优化
            config = TrainingConfig(
                n_episodes=2,
                enable_multiprocessing=False,
                save_dir=temp_dir
            )
            
            trainer = RLTrainer(
                config=config,
                environment=MockEnvironment(),
                agent=MockAgent(),
                data_split=MockDataSplit()
            )
            
            # 执行训练
            start_time = time.time()
            training_stats = trainer.train()
            training_time = time.time() - start_time
            
            # 验证训练完成
            assert 'mean_reward' in training_stats
            assert training_stats['total_episodes'] == 2
            
            print(f"✅ 禁用多核优化训练完成，用时: {training_time:.3f}秒")
    
    def test_training_with_multicore_optimization_enabled(self):
        """测试启用多核优化的训练"""
        print("=== 集成测试: 启用多核优化的训练 ===")
        
        # 创建简化的训练环境
        class MockEnvironment:
            def __init__(self):
                self.step_count = 0
            
            def reset(self):
                self.step_count = 0
                return np.random.randn(10)
            
            def step(self, action):
                self.step_count += 1
                done = self.step_count >= 5  # 5步后结束
                reward = np.random.randn()
                next_state = np.random.randn(10)
                return next_state, reward, done, {}
        
        class MockAgent:
            def act(self, state):
                return np.random.randn(3)
            
            def get_action(self, state, deterministic=False):
                return np.random.randn(3)
            
            def update(self, *args, **kwargs):
                return {'actor_loss': 0.1, 'critic_loss': 0.2}
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        class MockDataSplit:
            def __init__(self):
                self.train_indices = list(range(100))
                self.val_indices = list(range(100, 150))
                self.test_indices = list(range(150, 200))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 启用多核优化
            config = TrainingConfig(
                n_episodes=2,
                enable_multiprocessing=True,
                num_workers=2,
                parallel_environments=1,  # 简化以避免环境复制问题
                data_loader_workers=1,
                save_dir=temp_dir
            )
            
            trainer = RLTrainer(
                config=config,
                environment=MockEnvironment(),
                agent=MockAgent(),
                data_split=MockDataSplit()
            )
            
            # 执行训练
            start_time = time.time()
            training_stats = trainer.train()
            training_time = time.time() - start_time
            
            # 验证训练完成
            assert 'mean_reward' in training_stats
            assert training_stats['total_episodes'] == 2
            
            print(f"✅ 启用多核优化训练完成，用时: {training_time:.3f}秒")
    
    def test_multicore_configuration_validation_in_real_training(self):
        """测试实际训练中的多核配置验证"""
        print("=== 集成测试: 实际训练中的多核配置验证 ===")
        
        # 测试各种配置组合
        test_configs = [
            {"enable_multiprocessing": True, "num_workers": 1, "parallel_environments": 1},
            {"enable_multiprocessing": True, "num_workers": 2, "parallel_environments": 1},
            {"enable_multiprocessing": False, "num_workers": 0, "parallel_environments": 1},
        ]
        
        class MockEnvironment:
            def __init__(self):
                self.step_count = 0
            
            def reset(self):
                self.step_count = 0
                return np.random.randn(10)
            
            def step(self, action):
                self.step_count += 1
                done = self.step_count >= 3
                return np.random.randn(10), np.random.randn(), done, {}
        
        class MockAgent:
            def act(self, state):
                return np.random.randn(3)
            
            def get_action(self, state, deterministic=False):
                return np.random.randn(3)
            
            def update(self, *args, **kwargs):
                return {'actor_loss': 0.1, 'critic_loss': 0.2}
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        class MockDataSplit:
            def __init__(self):
                self.train_indices = list(range(50))
                self.val_indices = list(range(50, 75))
                self.test_indices = list(range(75, 100))
        
        for i, config_params in enumerate(test_configs):
            print(f"  测试配置 {i+1}: {config_params}")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                config = TrainingConfig(
                    n_episodes=1,
                    save_dir=temp_dir,
                    **config_params
                )
                
                trainer = RLTrainer(
                    config=config,
                    environment=MockEnvironment(),
                    agent=MockAgent(),
                    data_split=MockDataSplit()
                )
                
                # 验证配置不会导致错误
                training_stats = trainer.train()
                assert 'mean_reward' in training_stats
                
                print(f"    ✅ 配置 {i+1} 训练成功")
        
        print("✅ 所有多核配置验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])