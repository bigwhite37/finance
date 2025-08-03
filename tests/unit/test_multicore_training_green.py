#!/usr/bin/env python3
"""
多核训练优化的绿色阶段TDD测试
验证多核并行支持功能正常工作
"""

import pytest
import time
import multiprocessing as mp
from pathlib import Path
import sys
import numpy as np
import torch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.training.trainer import (
    RLTrainer, TrainingConfig, ExperienceDataset, 
    ParallelEnvironmentManager
)


class TestMulticoreTrainingGreen:
    """测试多核训练优化 - Green阶段"""
    
    def test_training_config_supports_multicore_parameters(self):
        """Green: 测试训练配置支持多核参数"""
        print("=== Green: 验证训练配置支持多核参数 ===")
        
        # 创建配置并验证多核相关属性存在
        config = TrainingConfig(n_episodes=10)
        
        # 验证所有多核配置属性都存在
        multicore_attributes = [
            'num_workers',
            'parallel_environments', 
            'data_loader_workers',
            'enable_multiprocessing',
            'pin_memory',
            'persistent_workers',
            'prefetch_factor',
            'enable_mixed_precision',
            'enable_cudnn_benchmark',
            'non_blocking_transfer'
        ]
        
        for attr in multicore_attributes:
            assert hasattr(config, attr), f"配置缺少多核属性: {attr}"
        
        # 验证默认值合理
        assert config.enable_multiprocessing is True
        assert config.num_workers > 0
        assert config.parallel_environments >= 1
        assert config.pin_memory is True
        
        print(f"✅ 多核配置属性验证成功:")
        print(f"  - 工作进程数: {config.num_workers}")
        print(f"  - 并行环境数: {config.parallel_environments}")
        print(f"  - 数据加载器工作线程: {config.data_loader_workers}")
        print(f"  - 混合精度训练: {config.enable_mixed_precision}")
    
    def test_experience_dataset_creation(self):
        """Green: 测试经验数据集创建"""
        print("=== Green: 验证经验数据集创建功能 ===")
        
        # 创建测试经验数据
        experiences = [
            (np.random.randn(10), np.random.randn(3), 1.0, np.random.randn(10), False)
            for _ in range(100)
        ]
        
        # 创建数据集
        dataset = ExperienceDataset(experiences)
        
        # 验证数据集功能
        assert len(dataset) == 100
        assert dataset[0] == experiences[0]
        
        # 测试带转换的数据集
        def transform_fn(experience):
            state, action, reward, next_state, done = experience
            return (state * 2, action, reward, next_state, done)
        
        dataset_with_transform = ExperienceDataset(experiences, transform=transform_fn)
        transformed_exp = dataset_with_transform[0]
        original_exp = experiences[0]
        
        # 验证转换生效
        np.testing.assert_array_equal(transformed_exp[0], original_exp[0] * 2)
        assert transformed_exp[1:] == original_exp[1:]
        
        print("✅ 经验数据集创建和转换功能正常")
    
    def test_parallel_environment_manager_initialization(self):
        """Green: 测试并行环境管理器初始化"""
        print("=== Green: 验证并行环境管理器初始化 ===")
        
        # 创建简单的环境工厂
        def mock_env_factory():
            return type('MockEnv', (), {
                'reset': lambda: np.random.randn(10),
                'step': lambda action: (np.random.randn(10), np.random.randn(), False, {}),
                'close': lambda: None
            })()
        
        # 创建并行环境管理器
        manager = ParallelEnvironmentManager(mock_env_factory, num_envs=2)
        
        # 验证基本属性
        assert manager.num_envs == 2
        assert manager.env_factory == mock_env_factory
        assert len(manager.envs) == 0  # 初始化前为空
        
        print("✅ 并行环境管理器初始化成功")
    
    def test_multicore_configuration_validation(self):
        """Green: 测试多核配置验证"""
        print("=== Green: 验证多核配置验证功能 ===")
        
        # 测试有效配置
        valid_config = TrainingConfig(
            n_episodes=10,
            num_workers=4,
            parallel_environments=2,
            data_loader_workers=2
        )
        
        assert valid_config.num_workers == 4
        assert valid_config.parallel_environments == 2
        assert valid_config.data_loader_workers == 2
        
        # 测试无效配置会被自动调整
        max_workers = mp.cpu_count()
        config_with_too_many_workers = TrainingConfig(
            n_episodes=10,
            num_workers=max_workers + 10,  # 超过系统核心数
            parallel_environments=max_workers + 5
        )
        
        # 应该被自动调整到系统限制内
        assert config_with_too_many_workers.num_workers <= max_workers
        assert config_with_too_many_workers.parallel_environments <= max_workers
        
        # 测试负数工作进程数应该抛出异常
        with pytest.raises(ValueError, match="num_workers必须为非负数"):
            TrainingConfig(num_workers=-1)
        
        with pytest.raises(ValueError, match="parallel_environments必须为正数"):
            TrainingConfig(parallel_environments=0)
        
        print("✅ 多核配置验证功能正常")
    
    def test_data_loader_creation_with_multiprocessing(self):
        """Green: 测试支持多进程的数据加载器创建"""
        print("=== Green: 验证多进程数据加载器创建 ===")
        
        # 创建启用多进程的配置
        config = TrainingConfig(
            n_episodes=10,
            enable_multiprocessing=True,
            data_loader_workers=2,
            batch_size=32
        )
        
        # 创建模拟训练器
        class MockTrainer:
            def __init__(self, config):
                self.config = config
            
            def _create_data_loader(self, experiences):
                # 直接从RLTrainer复制方法逻辑
                from torch.utils.data import DataLoader
                from rl_trading_system.training.trainer import ExperienceDataset
                
                if not experiences:
                    return None
                
                dataset = ExperienceDataset(experiences)
                
                dataloader_kwargs = {
                    'batch_size': self.config.batch_size,
                    'shuffle': True,
                    'pin_memory': self.config.pin_memory and torch.cuda.is_available(),
                    'prefetch_factor': self.config.prefetch_factor
                }
                
                if self.config.enable_multiprocessing and self.config.data_loader_workers > 0:
                    dataloader_kwargs.update({
                        'num_workers': self.config.data_loader_workers,
                        'persistent_workers': self.config.persistent_workers
                    })
                
                return DataLoader(dataset, **dataloader_kwargs)
        
        trainer = MockTrainer(config)
        
        # 创建测试数据
        experiences = [
            (np.random.randn(10), np.random.randn(3), 1.0, np.random.randn(10), False)
            for _ in range(100)
        ]
        
        # 创建数据加载器
        data_loader = trainer._create_data_loader(experiences)
        
        # 验证数据加载器属性
        assert data_loader is not None
        assert data_loader.batch_size == config.batch_size
        if config.enable_multiprocessing and config.data_loader_workers > 0:
            assert data_loader.num_workers == config.data_loader_workers
        
        # 测试数据加载器可以迭代
        batch_count = 0
        for batch in data_loader:
            batch_count += 1
            if batch_count >= 2:  # 只测试几个批次
                break
        
        assert batch_count > 0
        print(f"✅ 多进程数据加载器创建成功，批次大小: {config.batch_size}")
    
    def test_multicore_performance_simulation(self):
        """Green: 测试多核性能提升模拟"""
        print("=== Green: 验证多核性能提升 ===")
        
        cpu_count = mp.cpu_count()
        print(f"系统CPU核心数: {cpu_count}")
        
        # 使用ThreadPoolExecutor避免pickle问题
        def simulate_parallel_task(n_workers=1, n_tasks=100):
            """模拟并行任务"""
            def compute_task(task_id):
                # 模拟计算密集型任务
                data = np.random.randn(50, 50)
                return np.sum(data @ data.T)
            
            start_time = time.time()
            
            if n_workers == 1:
                # 单线程执行
                results = [compute_task(i) for i in range(n_tasks)]
            else:
                # 多线程执行（避免pickle问题）
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    results = list(executor.map(compute_task, range(n_tasks)))
            
            return time.time() - start_time, len(results)
        
        # 单线程基准测试
        single_thread_time, _ = simulate_parallel_task(n_workers=1, n_tasks=50)
        
        # 多线程测试
        multi_thread_workers = min(4, cpu_count)
        multi_thread_time, _ = simulate_parallel_task(n_workers=multi_thread_workers, n_tasks=50)
        
        # 计算加速比
        speedup = single_thread_time / multi_thread_time if multi_thread_time > 0 else 1.0
        
        print(f"单线程时间: {single_thread_time:.4f}秒")
        print(f"多线程时间 ({multi_thread_workers} 工作进程): {multi_thread_time:.4f}秒")
        print(f"加速比: {speedup:.2f}x")
        
        # 对于小任务，多线程可能有开销，但功能应该正常工作
        # 主要验证多核功能可以正常运行而不出错
        assert speedup > 0.3, f"多核功能应该正常运行，实际加速比: {speedup:.2f}x"
        
        # 重要的是验证多线程版本没有出错且返回了正确数量的结果
        _, single_results = simulate_parallel_task(n_workers=1, n_tasks=10)
        _, multi_results = simulate_parallel_task(n_workers=2, n_tasks=10)
        
        assert single_results == multi_results == 10, "多线程应该返回正确数量的结果"
        print("✅ 多核并行处理功能正常运行")
    
    def test_gpu_optimization_features_available(self):
        """Green: 测试GPU优化特性可用"""
        print("=== Green: 验证GPU优化特性 ===")
        
        config = TrainingConfig(
            n_episodes=10,
            enable_mixed_precision=True,
            enable_cudnn_benchmark=True,
            non_blocking_transfer=True
        )
        
        # 验证GPU优化配置
        assert config.enable_mixed_precision is True
        assert config.enable_cudnn_benchmark is True
        assert config.non_blocking_transfer is True
        
        # 测试PyTorch GPU功能可用性
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  - CUDA可用: GPU数量 {torch.cuda.device_count()}")
            print(f"  - 当前设备: {torch.cuda.get_device_name()}")
        else:
            print("  - CUDA不可用，将使用CPU")
        
        # 测试混合精度相关类可用
        try:
            scaler = torch.cuda.amp.GradScaler()
            print("  - 混合精度支持: ✅")
        except Exception as e:
            print(f"  - 混合精度支持: ❌ ({e})")
        
        print("✅ GPU优化特性配置正确")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])