"""
训练状态持久化管理器

实现训练检查点系统，支持任意阶段的中断恢复，
添加模型版本管理，追踪不同训练阶段的模型状态，
创建训练历史记录和配置快照功能。
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import logging
import os
import json
import pickle
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """模型版本信息"""
    version_id: str
    phase: str
    timestamp: datetime
    model_hash: str
    performance_metrics: Dict[str, float]
    config_snapshot: Dict[str, Any]
    checkpoint_path: str
    parent_version: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class TrainingSnapshot:
    """训练快照"""
    snapshot_id: str
    timestamp: datetime
    phase: str
    iteration: int
    model_state: Dict[str, Any]
    optimizer_states: Dict[str, Any]
    training_metrics: Dict[str, Any]
    config: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CheckpointConfig:
    """检查点配置"""
    base_dir: str = "checkpoints"
    max_versions: int = 10
    max_snapshots: int = 50
    auto_cleanup: bool = True
    compression: bool = True
    backup_interval: int = 3600  # 备份间隔（秒）
    version_retention_days: int = 30
    enable_cloud_backup: bool = False
    cloud_backup_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.cloud_backup_config is None:
            self.cloud_backup_config = {}


class CheckpointManager:
    """
    训练状态持久化管理器
    
    提供完整的训练状态管理功能，包括：
    - 模型版本管理
    - 训练快照
    - 配置历史
    - 自动备份和清理
    """
    
    def __init__(self, config: CheckpointConfig):
        """
        初始化检查点管理器
        
        Args:
            config: 检查点配置
        """
        self.config = config
        
        # 创建目录结构
        self.base_dir = Path(config.base_dir)
        self.models_dir = self.base_dir / "models"
        self.snapshots_dir = self.base_dir / "snapshots"
        self.configs_dir = self.base_dir / "configs"
        self.metadata_dir = self.base_dir / "metadata"
        self.backups_dir = self.base_dir / "backups"
        
        for dir_path in [self.models_dir, self.snapshots_dir, self.configs_dir, 
                        self.metadata_dir, self.backups_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 版本和快照管理
        self.versions: Dict[str, ModelVersion] = {}
        self.snapshots: Dict[str, TrainingSnapshot] = {}
        
        # 加载现有数据
        self._load_existing_data()
        
        logger.info(f"检查点管理器初始化完成 - 基础目录: {self.base_dir}")
        
    def save_model_version(self,
                          model_state: Dict[str, Any],
                          phase: str,
                          performance_metrics: Dict[str, float],
                          config: Dict[str, Any],
                          tags: Optional[List[str]] = None,
                          parent_version: Optional[str] = None) -> str:
        """
        保存模型版本
        
        Args:
            model_state: 模型状态字典
            phase: 训练阶段
            performance_metrics: 性能指标
            config: 配置信息
            tags: 版本标签
            parent_version: 父版本ID
            
        Returns:
            版本ID
        """
        # 生成版本ID
        version_id = self._generate_version_id(phase)
        
        # 计算模型哈希
        model_hash = self._compute_model_hash(model_state)
        
        # 保存模型文件
        model_filename = f"model_{version_id}.pth"
        model_path = self.models_dir / model_filename
        
        model_data = {
            'version_id': version_id,
            'model_state_dict': model_state,
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'model_hash': model_hash,
            'performance_metrics': performance_metrics,
            'config': config,
            'parent_version': parent_version,
            'tags': tags or []
        }
        
        if self.config.compression:
            torch.save(model_data, model_path, _use_new_zipfile_serialization=True)
        else:
            torch.save(model_data, model_path)
        
        # 创建版本记录
        version = ModelVersion(
            version_id=version_id,
            phase=phase,
            timestamp=datetime.now(),
            model_hash=model_hash,
            performance_metrics=performance_metrics,
            config_snapshot=config.copy(),
            checkpoint_path=str(model_path),
            parent_version=parent_version,
            tags=tags or []
        )
        
        self.versions[version_id] = version
        
        # 保存版本元数据
        self._save_version_metadata(version)
        
        # 清理旧版本
        if self.config.auto_cleanup:
            self._cleanup_old_versions()
        
        logger.info(f"模型版本已保存: {version_id} (阶段: {phase})")
        return version_id
        
    def load_model_version(self, version_id: str) -> Dict[str, Any]:
        """
        加载模型版本
        
        Args:
            version_id: 版本ID
            
        Returns:
            模型数据
        """
        if version_id not in self.versions:
            raise ValueError(f"版本不存在: {version_id}")
        
        version = self.versions[version_id]
        
        if not os.path.exists(version.checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {version.checkpoint_path}")
        
        model_data = torch.load(version.checkpoint_path, map_location='cpu')
        
        logger.info(f"模型版本已加载: {version_id}")
        return model_data
        
    def create_training_snapshot(self,
                               phase: str,
                               iteration: int,
                               model_state: Dict[str, Any],
                               optimizer_states: Dict[str, Any],
                               training_metrics: Dict[str, Any],
                               config: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        创建训练快照
        
        Args:
            phase: 训练阶段
            iteration: 迭代次数
            model_state: 模型状态
            optimizer_states: 优化器状态
            training_metrics: 训练指标
            config: 配置信息
            metadata: 额外元数据
            
        Returns:
            快照ID
        """
        # 生成快照ID
        snapshot_id = self._generate_snapshot_id(phase, iteration)
        
        # 保存快照文件
        snapshot_filename = f"snapshot_{snapshot_id}.pkl"
        snapshot_path = self.snapshots_dir / snapshot_filename
        
        snapshot_data = {
            'snapshot_id': snapshot_id,
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'iteration': iteration,
            'model_state': model_state,
            'optimizer_states': optimizer_states,
            'training_metrics': training_metrics,
            'config': config,
            'metadata': metadata or {}
        }
        
        if self.config.compression:
            with open(snapshot_path, 'wb') as f:
                pickle.dump(snapshot_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(snapshot_path, 'wb') as f:
                pickle.dump(snapshot_data, f)
        
        # 创建快照记录
        snapshot = TrainingSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            phase=phase,
            iteration=iteration,
            model_state=model_state,
            optimizer_states=optimizer_states,
            training_metrics=training_metrics,
            config=config,
            metadata=metadata or {}
        )
        
        self.snapshots[snapshot_id] = snapshot
        
        # 保存快照元数据
        self._save_snapshot_metadata(snapshot)
        
        # 清理旧快照
        if self.config.auto_cleanup:
            self._cleanup_old_snapshots()
        
        logger.info(f"训练快照已创建: {snapshot_id} (阶段: {phase}, 迭代: {iteration})")
        return snapshot_id
        
    def load_training_snapshot(self, snapshot_id: str) -> TrainingSnapshot:
        """
        加载训练快照
        
        Args:
            snapshot_id: 快照ID
            
        Returns:
            训练快照
        """
        if snapshot_id not in self.snapshots:
            raise ValueError(f"快照不存在: {snapshot_id}")
        
        snapshot_path = self.snapshots_dir / f"snapshot_{snapshot_id}.pkl"
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"快照文件不存在: {snapshot_path}")
        
        with open(snapshot_path, 'rb') as f:
            snapshot_data = pickle.load(f)
        
        logger.info(f"训练快照已加载: {snapshot_id}")
        return TrainingSnapshot(**snapshot_data)
        
    def save_config_snapshot(self, config: Dict[str, Any], phase: str, version: str) -> str:
        """
        保存配置快照
        
        Args:
            config: 配置字典
            phase: 训练阶段
            version: 版本标识
            
        Returns:
            配置快照ID
        """
        config_id = f"{phase}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        config_filename = f"config_{config_id}.json"
        config_path = self.configs_dir / config_filename
        
        config_data = {
            'config_id': config_id,
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'version': version,
            'config': config
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"配置快照已保存: {config_id}")
        return config_id
        
    def load_config_snapshot(self, config_id: str) -> Dict[str, Any]:
        """
        加载配置快照
        
        Args:
            config_id: 配置ID
            
        Returns:
            配置数据
        """
        config_path = self.configs_dir / f"config_{config_id}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        logger.info(f"配置快照已加载: {config_id}")
        return config_data
        
    def get_latest_version(self, phase: Optional[str] = None) -> Optional[ModelVersion]:
        """
        获取最新版本
        
        Args:
            phase: 训练阶段过滤
            
        Returns:
            最新版本信息
        """
        filtered_versions = self.versions.values()
        
        if phase:
            filtered_versions = [v for v in filtered_versions if v.phase == phase]
        
        if not filtered_versions:
            return None
        
        return max(filtered_versions, key=lambda v: v.timestamp)
        
    def get_latest_snapshot(self, phase: Optional[str] = None) -> Optional[TrainingSnapshot]:
        """
        获取最新快照
        
        Args:
            phase: 训练阶段过滤
            
        Returns:
            最新快照信息
        """
        filtered_snapshots = self.snapshots.values()
        
        if phase:
            filtered_snapshots = [s for s in filtered_snapshots if s.phase == phase]
        
        if not filtered_snapshots:
            return None
        
        return max(filtered_snapshots, key=lambda s: s.timestamp)
        
    def list_versions(self, phase: Optional[str] = None, tags: Optional[List[str]] = None) -> List[ModelVersion]:
        """
        列出版本
        
        Args:
            phase: 阶段过滤
            tags: 标签过滤
            
        Returns:
            版本列表
        """
        versions = list(self.versions.values())
        
        if phase:
            versions = [v for v in versions if v.phase == phase]
        
        if tags:
            versions = [v for v in versions if any(tag in v.tags for tag in tags)]
        
        # 按时间排序
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        
        return versions
        
    def list_snapshots(self, phase: Optional[str] = None, limit: Optional[int] = None) -> List[TrainingSnapshot]:
        """
        列出快照
        
        Args:
            phase: 阶段过滤
            limit: 数量限制
            
        Returns:
            快照列表
        """
        snapshots = list(self.snapshots.values())
        
        if phase:
            snapshots = [s for s in snapshots if s.phase == phase]
        
        # 按时间排序
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        
        if limit:
            snapshots = snapshots[:limit]
        
        return snapshots
        
    def delete_version(self, version_id: str):
        """
        删除版本
        
        Args:
            version_id: 版本ID
        """
        if version_id not in self.versions:
            logger.warning(f"版本不存在: {version_id}")
            return
        
        version = self.versions[version_id]
        
        # 删除模型文件
        if os.path.exists(version.checkpoint_path):
            os.remove(version.checkpoint_path)
        
        # 删除元数据文件
        metadata_path = self.metadata_dir / f"version_{version_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # 从内存中删除
        del self.versions[version_id]
        
        logger.info(f"版本已删除: {version_id}")
        
    def delete_snapshot(self, snapshot_id: str):
        """
        删除快照
        
        Args:
            snapshot_id: 快照ID
        """
        if snapshot_id not in self.snapshots:
            logger.warning(f"快照不存在: {snapshot_id}")
            return
        
        # 删除快照文件
        snapshot_path = self.snapshots_dir / f"snapshot_{snapshot_id}.pkl"
        if snapshot_path.exists():
            snapshot_path.unlink()
        
        # 删除元数据文件
        metadata_path = self.metadata_dir / f"snapshot_{snapshot_id}.json"
        if metadata_path.exists():
            metadata_path.unlink()
        
        # 从内存中删除
        del self.snapshots[snapshot_id]
        
        logger.info(f"快照已删除: {snapshot_id}")
        
    def create_backup(self) -> str:
        """
        创建完整备份
        
        Returns:
            备份路径
        """
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.backups_dir / backup_id
        
        # 创建备份目录
        backup_path.mkdir(exist_ok=True)
        
        # 复制所有文件
        for source_dir in [self.models_dir, self.snapshots_dir, self.configs_dir, self.metadata_dir]:
            target_dir = backup_path / source_dir.name
            if source_dir.exists():
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        
        # 创建备份清单
        manifest = {
            'backup_id': backup_id,
            'timestamp': datetime.now().isoformat(),
            'versions_count': len(self.versions),
            'snapshots_count': len(self.snapshots),
            'total_size': self._calculate_directory_size(backup_path)
        }
        
        with open(backup_path / "manifest.json", 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"备份已创建: {backup_path}")
        return str(backup_path)
        
    def restore_from_backup(self, backup_path: str):
        """
        从备份恢复
        
        Args:
            backup_path: 备份路径
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"备份路径不存在: {backup_path}")
        
        # 检查备份清单
        manifest_path = backup_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"备份清单不存在: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        logger.info(f"开始从备份恢复: {manifest['backup_id']}")
        
        # 清空当前数据
        self._clear_all_data()
        
        # 恢复文件
        for target_dir in [self.models_dir, self.snapshots_dir, self.configs_dir, self.metadata_dir]:
            source_dir = backup_path / target_dir.name
            if source_dir.exists():
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        
        # 重新加载数据
        self._load_existing_data()
        
        logger.info(f"备份恢复完成: {len(self.versions)} 个版本, {len(self.snapshots)} 个快照")
        
    def get_storage_statistics(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            存储统计
        """
        stats = {
            'total_versions': len(self.versions),
            'total_snapshots': len(self.snapshots),
            'storage_usage': {
                'models': self._calculate_directory_size(self.models_dir),
                'snapshots': self._calculate_directory_size(self.snapshots_dir),
                'configs': self._calculate_directory_size(self.configs_dir),
                'metadata': self._calculate_directory_size(self.metadata_dir),
                'backups': self._calculate_directory_size(self.backups_dir)
            },
            'phase_distribution': {},
            'recent_activity': []
        }
        
        # 阶段分布统计
        for version in self.versions.values():
            phase = version.phase
            if phase not in stats['phase_distribution']:
                stats['phase_distribution'][phase] = 0
            stats['phase_distribution'][phase] += 1
        
        # 最近活动
        all_items = []
        for version in self.versions.values():
            all_items.append(('version', version.version_id, version.timestamp))
        for snapshot in self.snapshots.values():
            all_items.append(('snapshot', snapshot.snapshot_id, snapshot.timestamp))
        
        all_items.sort(key=lambda x: x[2], reverse=True)
        stats['recent_activity'] = [
            {'type': item[0], 'id': item[1], 'timestamp': item[2].isoformat()}
            for item in all_items[:10]
        ]
        
        return stats
        
    def _generate_version_id(self, phase: str) -> str:
        """生成版本ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"{phase}_{timestamp}_{unique_id}"
        
    def _generate_snapshot_id(self, phase: str, iteration: int) -> str:
        """生成快照ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{phase}_iter{iteration}_{timestamp}"
        
    def _compute_model_hash(self, model_state: Dict[str, Any]) -> str:
        """计算模型哈希"""
        # 将模型状态转换为字节串并计算哈希
        model_bytes = pickle.dumps(model_state, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(model_bytes).hexdigest()[:16]
        
    def _save_version_metadata(self, version: ModelVersion):
        """保存版本元数据"""
        metadata_path = self.metadata_dir / f"version_{version.version_id}.json"
        
        metadata = {
            'version_id': version.version_id,
            'phase': version.phase,
            'timestamp': version.timestamp.isoformat(),
            'model_hash': version.model_hash,
            'performance_metrics': version.performance_metrics,
            'checkpoint_path': version.checkpoint_path,
            'parent_version': version.parent_version,
            'tags': version.tags
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
    def _save_snapshot_metadata(self, snapshot: TrainingSnapshot):
        """保存快照元数据"""
        metadata_path = self.metadata_dir / f"snapshot_{snapshot.snapshot_id}.json"
        
        metadata = {
            'snapshot_id': snapshot.snapshot_id,
            'timestamp': snapshot.timestamp.isoformat(),
            'phase': snapshot.phase,
            'iteration': snapshot.iteration,
            'metadata': snapshot.metadata
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
    def _load_existing_data(self):
        """加载现有数据"""
        # 加载版本元数据
        for metadata_file in self.metadata_dir.glob("version_*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                version = ModelVersion(
                    version_id=metadata['version_id'],
                    phase=metadata['phase'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    model_hash=metadata['model_hash'],
                    performance_metrics=metadata['performance_metrics'],
                    config_snapshot={},  # 配置快照单独存储
                    checkpoint_path=metadata['checkpoint_path'],
                    parent_version=metadata.get('parent_version'),
                    tags=metadata.get('tags', [])
                )
                
                self.versions[version.version_id] = version
                
            except Exception as e:
                logger.warning(f"加载版本元数据失败 {metadata_file}: {e}")
        
        # 加载快照元数据
        for metadata_file in self.metadata_dir.glob("snapshot_*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                snapshot = TrainingSnapshot(
                    snapshot_id=metadata['snapshot_id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    phase=metadata['phase'],
                    iteration=metadata['iteration'],
                    model_state={},  # 实际数据在快照文件中
                    optimizer_states={},
                    training_metrics={},
                    config={},
                    metadata=metadata.get('metadata', {})
                )
                
                self.snapshots[snapshot.snapshot_id] = snapshot
                
            except Exception as e:
                logger.warning(f"加载快照元数据失败 {metadata_file}: {e}")
        
        logger.info(f"加载完成: {len(self.versions)} 个版本, {len(self.snapshots)} 个快照")
        
    def _cleanup_old_versions(self):
        """清理旧版本"""
        if len(self.versions) <= self.config.max_versions:
            return
        
        # 按时间排序，保留最新的版本
        sorted_versions = sorted(self.versions.values(), key=lambda v: v.timestamp, reverse=True)
        versions_to_delete = sorted_versions[self.config.max_versions:]
        
        for version in versions_to_delete:
            self.delete_version(version.version_id)
        
        logger.info(f"清理了 {len(versions_to_delete)} 个旧版本")
        
    def _cleanup_old_snapshots(self):
        """清理旧快照"""
        if len(self.snapshots) <= self.config.max_snapshots:
            return
        
        # 按时间排序，保留最新的快照
        sorted_snapshots = sorted(self.snapshots.values(), key=lambda s: s.timestamp, reverse=True)
        snapshots_to_delete = sorted_snapshots[self.config.max_snapshots:]
        
        for snapshot in snapshots_to_delete:
            self.delete_snapshot(snapshot.snapshot_id)
        
        logger.info(f"清理了 {len(snapshots_to_delete)} 个旧快照")
        
    def _calculate_directory_size(self, directory: Path) -> int:
        """计算目录大小"""
        if not directory.exists():
            return 0
        
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
        
    def _clear_all_data(self):
        """清空所有数据"""
        # 清空内存数据
        self.versions.clear()
        self.snapshots.clear()
        
        # 清空文件
        for directory in [self.models_dir, self.snapshots_dir, self.configs_dir, self.metadata_dir]:
            if directory.exists():
                shutil.rmtree(directory)
                directory.mkdir(exist_ok=True)