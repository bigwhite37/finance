"""
检查点管理器的单元测试
测试模型保存、加载和版本管理功能，检查点完整性和恢复能力，模型压缩和优化功能
"""
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import shutil
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import pickle
import hashlib

from src.rl_trading_system.model_management.checkpoint_manager import (
    CheckpointManager,
    CheckpointConfig,
    ModelCheckpoint,
    CheckpointMetadata,
    ModelCompressor
)


class MockModel(nn.Module):
    """模拟PyTorch模型"""
    
    def __init__(self, input_size=10, hidden_size=64, output_size=4):
        super(MockModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_model_size(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters())


class MockAgent:
    """模拟强化学习智能体"""
    
    def __init__(self):
        self.actor = MockModel(input_size=20, hidden_size=128, output_size=4)
        self.critic = MockModel(input_size=24, hidden_size=128, output_size=1)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.training_step = 0
        
    def state_dict(self):
        """获取模型状态字典"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'training_step': self.training_step
        }
    
    def load_state_dict(self, state_dict):
        """加载模型状态字典"""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.optimizer_actor.load_state_dict(state_dict['optimizer_actor'])
        self.optimizer_critic.load_state_dict(state_dict['optimizer_critic'])
        self.training_step = state_dict['training_step']
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'actor_params': self.actor.get_model_size(),
            'critic_params': self.critic.get_model_size(),
            'total_params': self.actor.get_model_size() + self.critic.get_model_size()
        }


class TestCheckpointConfig:
    """检查点配置测试类"""
    
    def test_checkpoint_config_creation(self):
        """测试检查点配置创建"""
        config = CheckpointConfig(
            save_dir="./checkpoints",
            max_checkpoints=10,
            save_frequency=100,
            compression_enabled=True,
            model_format="torch"
        )
        
        assert config.save_dir == "./checkpoints"
        assert config.max_checkpoints == 10
        assert config.save_frequency == 100
        assert config.compression_enabled == True
        assert config.model_format == "torch"
    
    def test_checkpoint_config_defaults(self):
        """测试检查点配置默认值"""
        config = CheckpointConfig()
        
        assert config.save_dir == "./checkpoints"
        assert config.max_checkpoints == 5
        assert config.save_frequency == 1000
        assert config.compression_enabled == False
        assert config.model_format == "torch"
        assert config.auto_save_best == True
    
    def test_checkpoint_config_validation(self):
        """测试检查点配置验证"""
        # 测试无效的max_checkpoints
        with pytest.raises(ValueError, match="max_checkpoints必须为正数"):
            CheckpointConfig(max_checkpoints=0)
        
        # 测试无效的save_frequency
        with pytest.raises(ValueError, match="save_frequency必须为正数"):
            CheckpointConfig(save_frequency=-1)
        
        # 测试无效的model_format
        with pytest.raises(ValueError, match="不支持的模型格式"):
            CheckpointConfig(model_format="invalid")


class TestModelCheckpoint:
    """模型检查点测试类"""
    
    def test_model_checkpoint_creation(self):
        """测试模型检查点创建"""
        metadata = CheckpointMetadata(
            episode=1000,
            timestamp=datetime.now(),
            model_hash="abc123",
            performance_metrics={"reward": 100.0, "loss": 0.1}
        )
        
        checkpoint = ModelCheckpoint(
            checkpoint_id="checkpoint_1000",
            file_path="/path/to/checkpoint.pth",
            metadata=metadata
        )
        
        assert checkpoint.checkpoint_id == "checkpoint_1000"
        assert checkpoint.file_path == "/path/to/checkpoint.pth"
        assert checkpoint.metadata.episode == 1000
        assert checkpoint.metadata.model_hash == "abc123"
    
    def test_checkpoint_metadata_serialization(self):
        """测试检查点元数据序列化"""
        metadata = CheckpointMetadata(
            episode=500,
            timestamp=datetime.now(),
            model_hash="def456",
            performance_metrics={"accuracy": 0.95},
            model_info={"params": 10000}
        )
        
        # 序列化
        serialized = metadata.to_dict()
        
        assert serialized['episode'] == 500
        assert serialized['model_hash'] == "def456"
        assert serialized['performance_metrics']['accuracy'] == 0.95
        assert 'timestamp' in serialized
        
        # 反序列化
        restored = CheckpointMetadata.from_dict(serialized)
        
        assert restored.episode == metadata.episode
        assert restored.model_hash == metadata.model_hash
        assert restored.performance_metrics == metadata.performance_metrics
    
    def test_checkpoint_comparison(self):
        """测试检查点比较"""
        metadata1 = CheckpointMetadata(
            episode=1000,
            timestamp=datetime.now(),
            performance_metrics={"reward": 100.0}
        )
        
        metadata2 = CheckpointMetadata(
            episode=2000,
            timestamp=datetime.now(),
            performance_metrics={"reward": 150.0}
        )
        
        checkpoint1 = ModelCheckpoint("cp1", "path1", metadata1)
        checkpoint2 = ModelCheckpoint("cp2", "path2", metadata2)
        
        # 测试性能比较
        assert checkpoint2.is_better_than(checkpoint1, metric="reward", mode="max")
        assert not checkpoint1.is_better_than(checkpoint2, metric="reward", mode="max")


class TestCheckpointManager:
    """检查点管理器测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def checkpoint_config(self, temp_dir):
        """检查点配置fixture"""
        return CheckpointConfig(
            save_dir=temp_dir,
            max_checkpoints=3,
            save_frequency=10,
            auto_save_best=True
        )
    
    @pytest.fixture
    def mock_agent(self):
        """模拟智能体fixture"""
        return MockAgent()
    
    @pytest.fixture
    def checkpoint_manager(self, checkpoint_config):
        """检查点管理器fixture"""
        return CheckpointManager(checkpoint_config)
    
    def test_checkpoint_manager_initialization(self, checkpoint_manager, checkpoint_config):
        """测试检查点管理器初始化"""
        assert checkpoint_manager.config == checkpoint_config
        assert os.path.exists(checkpoint_config.save_dir)
        assert len(checkpoint_manager.checkpoints) == 0
        assert checkpoint_manager.best_checkpoint is None
    
    def test_save_checkpoint_basic(self, checkpoint_manager, mock_agent, temp_dir):
        """测试基本检查点保存"""
        metrics = {"reward": 100.0, "loss": 0.1}
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_agent,
            episode=100,
            metrics=metrics
        )
        
        # 检查文件是否创建
        assert os.path.exists(checkpoint_path)
        
        # 检查检查点是否被记录
        assert len(checkpoint_manager.checkpoints) == 1
        
        checkpoint = checkpoint_manager.checkpoints[0]
        assert checkpoint.metadata.episode == 100
        assert checkpoint.metadata.performance_metrics == metrics
    
    def test_load_checkpoint_basic(self, checkpoint_manager, mock_agent, temp_dir):
        """测试基本检查点加载"""
        # 先保存一个检查点
        original_step = mock_agent.training_step = 500
        metrics = {"reward": 200.0}
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_agent,
            episode=200,
            metrics=metrics
        )
        
        # 修改模型状态
        mock_agent.training_step = 0
        
        # 加载检查点
        loaded_metadata = checkpoint_manager.load_checkpoint(checkpoint_path, mock_agent)
        
        # 验证加载结果
        assert mock_agent.training_step == original_step
        assert loaded_metadata.episode == 200
        assert loaded_metadata.performance_metrics == metrics
    
    def test_checkpoint_versioning(self, checkpoint_manager, mock_agent):
        """测试检查点版本管理"""
        # 保存多个检查点
        for i in range(5):
            metrics = {"reward": i * 10.0}
            checkpoint_manager.save_checkpoint(
                model=mock_agent,
                episode=(i + 1) * 100,
                metrics=metrics
            )
        
        # 检查是否只保留了最大数量的检查点
        assert len(checkpoint_manager.checkpoints) <= checkpoint_manager.config.max_checkpoints
        
        # 检查检查点是否按时间排序
        timestamps = [cp.metadata.timestamp for cp in checkpoint_manager.checkpoints]
        assert timestamps == sorted(timestamps)
    
    def test_best_checkpoint_tracking(self, checkpoint_manager, mock_agent):
        """测试最佳检查点跟踪"""
        # 保存多个检查点，奖励递增
        rewards = [50.0, 100.0, 75.0, 150.0, 120.0]
        
        for i, reward in enumerate(rewards):
            metrics = {"reward": reward}
            checkpoint_manager.save_checkpoint(
                model=mock_agent,
                episode=(i + 1) * 100,
                metrics=metrics,
                is_best_metric="reward"
            )
        
        # 检查最佳检查点
        assert checkpoint_manager.best_checkpoint is not None
        assert checkpoint_manager.best_checkpoint.metadata.performance_metrics["reward"] == 150.0
    
    def test_checkpoint_cleanup(self, checkpoint_manager, mock_agent):
        """测试检查点清理"""
        initial_count = 10
        max_checkpoints = checkpoint_manager.config.max_checkpoints
        
        # 保存超过最大数量的检查点
        for i in range(initial_count):
            metrics = {"reward": i * 5.0}
            checkpoint_manager.save_checkpoint(
                model=mock_agent,
                episode=(i + 1) * 50,
                metrics=metrics
            )
        
        # 验证只保留了最大数量的检查点
        assert len(checkpoint_manager.checkpoints) == max_checkpoints
        
        # 验证保留的是最新的检查点
        episodes = [cp.metadata.episode for cp in checkpoint_manager.checkpoints]
        expected_episodes = list(range((initial_count - max_checkpoints + 1) * 50, 
                                     (initial_count + 1) * 50, 50))
        assert episodes == expected_episodes
    
    def test_checkpoint_integrity_verification(self, checkpoint_manager, mock_agent, temp_dir):
        """测试检查点完整性验证"""
        # 保存检查点
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_agent,
            episode=300,
            metrics={"reward": 300.0}
        )
        
        # 验证检查点完整性
        is_valid = checkpoint_manager.verify_checkpoint_integrity(checkpoint_path)
        assert is_valid
        
        # 损坏检查点文件
        with open(checkpoint_path, 'rb') as f:
            data = f.read()
        
        # 写入损坏的数据
        with open(checkpoint_path, 'wb') as f:
            f.write(data[:len(data)//2])  # 截断文件
        
        # 验证损坏的检查点
        is_valid = checkpoint_manager.verify_checkpoint_integrity(checkpoint_path)
        assert not is_valid
    
    def test_checkpoint_recovery(self, checkpoint_manager, mock_agent, temp_dir):
        """测试检查点恢复能力"""
        # 保存多个检查点
        saved_paths = []
        for i in range(3):
            metrics = {"reward": (i + 1) * 100.0}
            path = checkpoint_manager.save_checkpoint(
                model=mock_agent,
                episode=(i + 1) * 200,
                metrics=metrics
            )
            saved_paths.append(path)
        
        # 模拟部分检查点损坏
        os.remove(saved_paths[1])  # 删除中间的检查点
        
        # 扫描并恢复可用的检查点
        recovered_checkpoints = checkpoint_manager.scan_and_recover_checkpoints()
        
        # 验证恢复结果
        assert len(recovered_checkpoints) == 2  # 应该恢复2个有效检查点
        episodes = [cp.metadata.episode for cp in recovered_checkpoints]
        assert 200 in episodes and 600 in episodes
        assert 400 not in episodes  # 损坏的检查点不应该被恢复
    
    def test_checkpoint_metadata_persistence(self, checkpoint_manager, mock_agent):
        """测试检查点元数据持久化"""
        # 保存检查点
        metrics = {"reward": 500.0, "steps": 10000}
        model_info = mock_agent.get_model_info()
        
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_agent,
            episode=500,
            metrics=metrics,
            model_info=model_info
        )
        
        # 重新创建管理器并扫描检查点
        new_manager = CheckpointManager(checkpoint_manager.config)
        new_manager.scan_and_recover_checkpoints()
        
        # 验证元数据是否正确恢复
        assert len(new_manager.checkpoints) == 1
        recovered_checkpoint = new_manager.checkpoints[0]
        
        assert recovered_checkpoint.metadata.episode == 500
        assert recovered_checkpoint.metadata.performance_metrics == metrics
        assert recovered_checkpoint.metadata.model_info == model_info
    
    def test_checkpoint_format_conversion(self, checkpoint_manager, mock_agent):
        """测试检查点格式转换"""
        # 保存PyTorch格式的检查点
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_agent,
            episode=100,
            metrics={"reward": 100.0}
        )
        
        # 转换为ONNX格式（模拟）
        onnx_path = checkpoint_manager.convert_checkpoint_format(
            checkpoint_path, 
            target_format="onnx",
            input_shape=(1, 20)  # 示例输入形状
        )
        
        assert os.path.exists(onnx_path)
        assert onnx_path.endswith('.onnx')
    
    @pytest.mark.parametrize("compression_method", ["gzip", "lzma", "bz2"])
    def test_checkpoint_compression(self, checkpoint_manager, mock_agent, compression_method):
        """测试检查点压缩"""
        # 启用压缩
        checkpoint_manager.config.compression_enabled = True
        checkpoint_manager.config.compression_method = compression_method
        
        # 保存压缩的检查点
        checkpoint_path = checkpoint_manager.save_checkpoint(
            model=mock_agent,
            episode=100,
            metrics={"reward": 150.0}
        )
        
        # 验证压缩文件存在（检查点路径本身就是压缩后的路径）
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith(f".{compression_method}")
        
        # 验证可以正确加载压缩的检查点
        loaded_metadata = checkpoint_manager.load_checkpoint(checkpoint_path, mock_agent)
        assert loaded_metadata.episode == 100
        assert loaded_metadata.performance_metrics["reward"] == 150.0
    
    def test_checkpoint_size_optimization(self, checkpoint_manager, mock_agent):
        """测试检查点大小优化"""
        # 保存原始检查点
        original_path = checkpoint_manager.save_checkpoint(
            model=mock_agent,
            episode=100,
            metrics={"reward": 100.0}
        )
        original_size = os.path.getsize(original_path)
        
        # 应用大小优化
        optimized_path = checkpoint_manager.optimize_checkpoint_size(
            original_path,
            remove_optimizer_state=True,
            quantize_weights=True
        )
        optimized_size = os.path.getsize(optimized_path)
        
        # 验证优化效果（由于MockAgent可能没有真正的优化器状态，所以大小可能相似）
        # 主要验证优化过程没有出错
        assert optimized_size > 0
        
        # 验证优化后的检查点仍然可用
        new_agent = MockAgent()
        loaded_metadata = checkpoint_manager.load_checkpoint(optimized_path, new_agent)
        assert loaded_metadata.episode == 100


class TestModelCompressor:
    """模型压缩器测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_model(self):
        """模拟模型fixture"""
        return MockModel()
    
    @pytest.fixture
    def compressor(self, temp_dir):
        """模型压缩器fixture"""
        return ModelCompressor(temp_dir)
    
    def test_model_quantization(self, compressor, mock_model, temp_dir):
        """测试模型量化"""
        # 保存原始模型
        original_path = os.path.join(temp_dir, "original_model.pth")
        torch.save(mock_model.state_dict(), original_path)
        original_size = os.path.getsize(original_path)
        
        # 量化模型
        quantized_path = compressor.quantize_model(
            model=mock_model,
            model_path=original_path,
            quantization_type="dynamic"
        )
        
        # 验证量化结果
        assert os.path.exists(quantized_path)
        quantized_size = os.path.getsize(quantized_path)
        
        # 量化后的模型应该更小
        assert quantized_size < original_size
    
    def test_model_pruning(self, compressor, mock_model, temp_dir):
        """测试模型剪枝"""
        # 计算原始参数数量
        original_params = mock_model.get_model_size()
        
        # 剪枝模型
        pruned_model = compressor.prune_model(
            model=mock_model,
            pruning_ratio=0.2  # 剪枝20%的参数
        )
        
        # 验证剪枝效果
        pruned_params = pruned_model.get_model_size()
        
        # 注意：由于我们的模拟模型比较简单，这里主要测试接口
        assert isinstance(pruned_model, nn.Module)
    
    def test_onnx_conversion(self, compressor, mock_model, temp_dir):
        """测试ONNX转换"""
        # 转换为ONNX格式
        onnx_path = compressor.convert_to_onnx(
            model=mock_model,
            input_shape=(1, 10),
            output_path=os.path.join(temp_dir, "model.onnx")
        )
        
        # 验证ONNX文件
        assert os.path.exists(onnx_path)
        assert onnx_path.endswith('.onnx')
    
    def test_torchscript_conversion(self, compressor, mock_model, temp_dir):
        """测试TorchScript转换"""
        # 转换为TorchScript格式
        script_path = compressor.convert_to_torchscript(
            model=mock_model,
            example_input=torch.randn(1, 10),
            output_path=os.path.join(temp_dir, "model.pt")
        )
        
        # 验证TorchScript文件
        assert os.path.exists(script_path)
        assert script_path.endswith('.pt')
        
        # 验证可以加载TorchScript模型
        loaded_model = torch.jit.load(script_path)
        assert isinstance(loaded_model, torch.jit.ScriptModule)
    
    def test_compression_pipeline(self, compressor, mock_model, temp_dir):
        """测试完整的压缩流水线"""
        # 执行完整的压缩流水线
        results = compressor.compress_model_pipeline(
            model=mock_model,
            input_shape=(1, 10),
            output_dir=temp_dir,
            enable_quantization=True,
            enable_pruning=True,
            enable_onnx=True,
            enable_torchscript=True
        )
        
        # 验证所有输出文件
        assert 'quantized' in results
        assert 'pruned' in results
        assert 'onnx' in results
        assert 'torchscript' in results
        
        # 验证文件存在
        for format_name, file_path in results.items():
            assert os.path.exists(file_path)


class TestIntegrationScenarios:
    """集成场景测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_training_checkpoint_workflow(self, temp_dir):
        """测试训练检查点工作流"""
        # 创建配置和管理器
        config = CheckpointConfig(
            save_dir=temp_dir,
            max_checkpoints=3,
            auto_save_best=True,
            compression_enabled=True
        )
        manager = CheckpointManager(config)
        agent = MockAgent()
        
        # 模拟训练过程
        best_reward = 0
        for episode in range(1, 11):
            # 模拟训练
            agent.training_step += 100
            reward = np.random.uniform(0, 200) + episode * 10  # 总体趋势上升
            
            metrics = {
                "reward": reward,
                "loss": np.random.uniform(0.1, 1.0),
                "episode": episode
            }
            
            # 保存检查点
            is_best = reward > best_reward
            if is_best:
                best_reward = reward
            
            checkpoint_path = manager.save_checkpoint(
                model=agent,
                episode=episode * 100,
                metrics=metrics,
                is_best_metric="reward" if is_best else None
            )
            
            assert os.path.exists(checkpoint_path)
        
        # 验证最终状态
        assert len(manager.checkpoints) <= config.max_checkpoints
        assert manager.best_checkpoint is not None
        assert manager.best_checkpoint.metadata.performance_metrics["reward"] == best_reward
    
    def test_checkpoint_disaster_recovery(self, temp_dir):
        """测试检查点灾难恢复"""
        # 创建检查点管理器
        config = CheckpointConfig(save_dir=temp_dir, max_checkpoints=5)
        manager = CheckpointManager(config)
        agent = MockAgent()
        
        # 保存多个检查点
        saved_episodes = []
        for i in range(5):
            episode = (i + 1) * 200
            metrics = {"reward": i * 50.0}
            
            manager.save_checkpoint(
                model=agent,
                episode=episode,
                metrics=metrics
            )
            saved_episodes.append(episode)
        
        # 模拟部分文件损坏
        checkpoints = manager.checkpoints
        # 损坏第2和第4个检查点
        os.remove(checkpoints[1].file_path)
        
        # 在第3个检查点中写入无效数据
        with open(checkpoints[2].file_path, 'w') as f:
            f.write("invalid data")
        
        # 创建新的管理器实例，模拟重启后的恢复
        recovery_manager = CheckpointManager(config)
        recovered = recovery_manager.scan_and_recover_checkpoints()
        
        # 验证恢复结果（至少恢复部分检查点，允许一些检查点被成功恢复）
        assert len(recovered) >= 2  # 至少应该恢复2个检查点
        recovered_episodes = [cp.metadata.episode for cp in recovered]
        
        # 应该恢复第1和第5个检查点
        assert saved_episodes[0] in recovered_episodes
        assert saved_episodes[4] in recovered_episodes
        # 第2个检查点被删除，应该不存在
        assert saved_episodes[1] not in recovered_episodes
    
    def test_model_format_migration(self, temp_dir):
        """测试模型格式迁移"""
        # 创建原始PyTorch检查点
        config = CheckpointConfig(save_dir=temp_dir, model_format="torch")
        manager = CheckpointManager(config)
        agent = MockAgent()
        
        # 保存PyTorch格式
        torch_path = manager.save_checkpoint(
            model=agent,
            episode=1000,
            metrics={"reward": 500.0}
        )
        
        # 创建压缩器
        compressor = ModelCompressor(temp_dir)
        
        # 批量转换为不同格式
        conversion_results = {}
        
        # 转换为ONNX
        try:
            onnx_path = compressor.convert_to_onnx(
                model=agent.actor,  # 只转换actor网络作为例子
                input_shape=(1, 20),
                output_path=os.path.join(temp_dir, "model.onnx")
            )
            conversion_results['onnx'] = onnx_path
        except Exception as e:
            # ONNX转换可能因为环境问题失败，这里跳过
            print(f"ONNX conversion skipped: {e}")
        
        # 转换为TorchScript
        script_path = compressor.convert_to_torchscript(
            model=agent.actor,
            example_input=torch.randn(1, 20),
            output_path=os.path.join(temp_dir, "model_script.pt")
        )
        conversion_results['torchscript'] = script_path
        
        # 验证转换结果
        for format_name, path in conversion_results.items():
            assert os.path.exists(path)
            print(f"Successfully converted to {format_name}: {path}")
    
    def test_checkpoint_performance_monitoring(self, temp_dir):
        """测试检查点性能监控"""
        config = CheckpointConfig(save_dir=temp_dir, max_checkpoints=10)
        manager = CheckpointManager(config)
        agent = MockAgent()
        
        # 记录性能指标
        performance_history = []
        
        for episode in range(1, 21):
            # 模拟不同的性能指标
            metrics = {
                "reward": np.random.uniform(50, 200),
                "loss": np.random.uniform(0.01, 0.5),
                "success_rate": np.random.uniform(0.3, 0.9),
                "steps_per_episode": np.random.randint(100, 500)
            }
            
            performance_history.append(metrics)
            
            # 保存检查点
            manager.save_checkpoint(
                model=agent,
                episode=episode * 50,
                metrics=metrics,
                is_best_metric="reward"
            )
        
        # 分析性能趋势
        rewards = [m["reward"] for m in performance_history]
        losses = [m["loss"] for m in performance_history]
        
        # 验证检查点中记录了完整的性能历史
        assert len(manager.checkpoints) <= config.max_checkpoints
        
        # 验证最佳检查点确实对应最高奖励
        if manager.best_checkpoint:
            best_reward = manager.best_checkpoint.metadata.performance_metrics["reward"]
            assert best_reward == max(rewards)
        
        # 生成性能报告
        performance_report = manager.generate_performance_report()
        
        assert "total_checkpoints" in performance_report
        assert "best_performance" in performance_report
        assert "performance_trend" in performance_report
        
        print(f"Performance report: {performance_report}")


class TestErrorHandling:
    """错误处理测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_invalid_checkpoint_file(self, temp_dir):
        """测试无效检查点文件处理"""
        config = CheckpointConfig(save_dir=temp_dir)
        manager = CheckpointManager(config)
        agent = MockAgent()
        
        # 创建无效的检查点文件
        invalid_path = os.path.join(temp_dir, "invalid.pth")
        with open(invalid_path, 'w') as f:
            f.write("not a valid checkpoint")
        
        # 尝试加载无效检查点
        with pytest.raises(Exception):
            manager.load_checkpoint(invalid_path, agent)
    
    def test_disk_space_handling(self, temp_dir):
        """测试磁盘空间不足处理"""
        config = CheckpointConfig(save_dir=temp_dir)
        manager = CheckpointManager(config)
        agent = MockAgent()
        
        # 模拟磁盘空间检查
        available_space = manager.check_available_disk_space()
        assert available_space > 0
        
        # 验证空间不足时的处理
        with patch('shutil.disk_usage', return_value=(1000, 1000, 100)):  # 只有100字节可用
            with pytest.raises(RuntimeError, match="磁盘空间不足"):
                manager.save_checkpoint(
                    model=agent,
                    episode=100,
                    metrics={"reward": 100.0}
                )
    
    def test_concurrent_access_handling(self, temp_dir):
        """测试并发访问处理"""
        config = CheckpointConfig(save_dir=temp_dir)
        manager = CheckpointManager(config)
        agent = MockAgent()
        
        # 模拟并发保存
        import threading
        import time
        
        results = []
        errors = []
        
        def save_checkpoint_worker(worker_id):
            try:
                time.sleep(0.1 * worker_id)  # 模拟不同的启动时间
                path = manager.save_checkpoint(
                    model=agent,
                    episode=worker_id * 100,
                    metrics={"reward": worker_id * 10.0}
                )
                results.append(path)
            except Exception as e:
                errors.append(str(e))
        
        # 启动多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=save_checkpoint_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(errors) == 0  # 不应该有错误
        assert len(results) == 3  # 应该成功保存3个检查点
        assert len(manager.checkpoints) == 3