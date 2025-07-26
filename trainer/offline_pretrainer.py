"""
离线预训练器

实现纯离线训练逻辑，支持行为克隆和TD学习，
提供预训练检查点保存和加载功能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
import logging
import os
from pathlib import Path

from data.offline_dataset import OfflineDataset
from rl_agent.cvar_ppo_agent import CVaRPPOAgent

logger = logging.getLogger(__name__)


@dataclass
class OfflinePretrainerConfig:
    """离线预训练器配置"""
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    behavior_cloning_weight: float = 0.5
    td_learning_weight: float = 0.5
    value_learning_weight: float = 1.0
    gradient_clip_norm: float = 0.5
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 1e-4
    checkpoint_interval: int = 10
    validation_split: float = 0.2
    use_data_augmentation: bool = True
    warmup_epochs: int = 5
    warmup_lr_factor: float = 0.1


class OfflinePretrainer:
    """
    离线预训练器
    
    实现纯离线训练逻辑，支持行为克隆和TD学习目标，
    结合价值函数训练，提供预训练检查点保存和加载功能。
    """
    
    def __init__(self, agent: CVaRPPOAgent, config: OfflinePretrainerConfig):
        """
        初始化离线预训练器
        
        Args:
            agent: CVaR-PPO智能体实例
            config: 预训练器配置
        """
        self.agent = agent
        self.config = config
        self.device = agent.device
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'bc_loss': [],
            'td_loss': [],
            'value_loss': []
        }
        
        # 学习率调度器
        self.scheduler = None
        
        logger.info(f"初始化离线预训练器 - 配置: {config}")
        
    def pretrain(self, 
                offline_dataset: OfflineDataset,
                validation_dataset: Optional[OfflineDataset] = None,
                checkpoint_dir: str = "checkpoints/offline_pretrain") -> Dict[str, Any]:
        """
        执行离线预训练
        
        Args:
            offline_dataset: 离线训练数据集
            validation_dataset: 验证数据集（可选）
            checkpoint_dir: 检查点保存目录
            
        Returns:
            训练结果统计
        """
        logger.info("开始离线预训练...")
        
        # 创建检查点目录
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        train_loader, val_loader = self._prepare_data_loaders(
            offline_dataset, validation_dataset
        )
        
        if len(train_loader) == 0:
            logger.error("训练数据为空，无法进行预训练")
            return {'status': 'failed', 'reason': 'empty_dataset'}
        
        # 设置优化器和调度器
        self._setup_optimizer_and_scheduler()
        
        # 训练循环
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # 训练阶段
            train_metrics = self._train_epoch(train_loader)
            
            # 验证阶段
            val_metrics = self._validate_epoch(val_loader) if val_loader else {}
            
            # 记录训练历史
            self._update_training_history(train_metrics, val_metrics)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 日志输出
            self._log_epoch_results(epoch, train_metrics, val_metrics)
            
            # 保存检查点
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, epoch, train_metrics, val_metrics)
            
            # 早停检查
            current_loss = val_metrics.get('total_loss', train_metrics['total_loss'])
            if self._check_early_stopping(current_loss):
                logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                break
        
        # 保存最终模型
        final_checkpoint_path = self._save_final_checkpoint(checkpoint_dir)
        
        # 训练结果
        result = {
            'status': 'completed',
            'epochs_trained': self.current_epoch + 1,
            'best_loss': self.best_loss,
            'final_checkpoint': final_checkpoint_path,
            'training_history': self.training_history
        }
        
        logger.info(f"离线预训练完成: {result}")
        return result
        
    def _prepare_data_loaders(self, 
                            offline_dataset: OfflineDataset,
                            validation_dataset: Optional[OfflineDataset] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """准备训练和验证数据加载器"""
        
        # 创建行为克隆数据集
        behavior_dataset = offline_dataset.create_behavior_dataset()
        
        if len(behavior_dataset) == 0:
            logger.warning("行为克隆数据集为空")
            return DataLoader([]), None
        
        # 应用数据增强
        if self.config.use_data_augmentation:
            behavior_dataset = offline_dataset.apply_data_augmentation(behavior_dataset)
        
        # 分割训练和验证集
        if validation_dataset is None and self.config.validation_split > 0:
            dataset_size = len(behavior_dataset)
            val_size = int(dataset_size * self.config.validation_split)
            train_size = dataset_size - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                behavior_dataset, [train_size, val_size]
            )
        else:
            train_dataset = behavior_dataset
            val_dataset = validation_dataset.create_behavior_dataset() if validation_dataset else None
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                drop_last=False
            )
        
        logger.info(f"数据加载器准备完成 - 训练: {len(train_loader)} 批次, 验证: {len(val_loader) if val_loader else 0} 批次")
        return train_loader, val_loader
        
    def _setup_optimizer_and_scheduler(self):
        """设置优化器和学习率调度器"""
        # 使用统一优化器进行预训练
        self.optimizer = torch.optim.Adam(
            self.agent.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-8
        )
        
        # 学习率调度器：预热 + 余弦退火
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=self.config.warmup_lr_factor,
            total_iters=self.config.warmup_epochs
        )
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs,
            eta_min=self.config.learning_rate * 0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs]
        )
        
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.agent.network.train()
        
        epoch_losses = {
            'total_loss': [],
            'bc_loss': [],
            'td_loss': [],
            'value_loss': []
        }
        
        for batch_idx, (states, expert_actions) in enumerate(train_loader):
            states = states.to(self.device)
            expert_actions = expert_actions.to(self.device)
            
            # 前向传播
            action_mean, action_std, values, cvar_estimates = self.agent.network(states)
            
            # 计算损失
            bc_loss = self.behavior_cloning_loss(action_mean, expert_actions)
            td_loss = self.td_learning_loss(states, expert_actions, values)
            value_loss = self._compute_value_loss(values, states, expert_actions)
            
            # 总损失
            total_loss = (
                self.config.behavior_cloning_weight * bc_loss +
                self.config.td_learning_weight * td_loss +
                self.config.value_learning_weight * value_loss
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.agent.network.parameters(),
                self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            
            # 记录损失
            epoch_losses['total_loss'].append(total_loss.item())
            epoch_losses['bc_loss'].append(bc_loss.item())
            epoch_losses['td_loss'].append(td_loss.item())
            epoch_losses['value_loss'].append(value_loss.item())
        
        # 计算平均损失
        return {key: np.mean(values) for key, values in epoch_losses.items()}
        
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        if not val_loader:
            return {}
            
        self.agent.network.eval()
        
        epoch_losses = {
            'total_loss': [],
            'bc_loss': [],
            'td_loss': [],
            'value_loss': []
        }
        
        with torch.no_grad():
            for states, expert_actions in val_loader:
                states = states.to(self.device)
                expert_actions = expert_actions.to(self.device)
                
                # 前向传播
                action_mean, action_std, values, cvar_estimates = self.agent.network(states)
                
                # 计算损失
                bc_loss = self.behavior_cloning_loss(action_mean, expert_actions)
                td_loss = self.td_learning_loss(states, expert_actions, values)
                value_loss = self._compute_value_loss(values, states, expert_actions)
                
                # 总损失
                total_loss = (
                    self.config.behavior_cloning_weight * bc_loss +
                    self.config.td_learning_weight * td_loss +
                    self.config.value_learning_weight * value_loss
                )
                
                # 记录损失
                epoch_losses['total_loss'].append(total_loss.item())
                epoch_losses['bc_loss'].append(bc_loss.item())
                epoch_losses['td_loss'].append(td_loss.item())
                epoch_losses['value_loss'].append(value_loss.item())
        
        # 计算平均损失
        return {key: np.mean(values) for key, values in epoch_losses.items()}
        
    def behavior_cloning_loss(self, predicted_actions: torch.Tensor, expert_actions: torch.Tensor) -> torch.Tensor:
        """
        行为克隆损失函数，支持监督学习初始化
        
        Args:
            predicted_actions: 预测的动作 (batch_size, action_dim)
            expert_actions: 专家动作 (batch_size, action_dim)
            
        Returns:
            行为克隆损失
        """
        # 确保维度匹配
        if predicted_actions.dim() != expert_actions.dim():
            if expert_actions.dim() == 1:
                expert_actions = expert_actions.unsqueeze(-1)
            elif predicted_actions.dim() == 1:
                predicted_actions = predicted_actions.unsqueeze(-1)
        
        # 使用均方误差损失
        mse_loss = F.mse_loss(predicted_actions, expert_actions)
        
        # 添加L1正则化，鼓励稀疏动作
        l1_regularization = torch.mean(torch.abs(predicted_actions))
        
        # 组合损失
        total_bc_loss = mse_loss + 0.01 * l1_regularization
        
        return total_bc_loss
        
    def td_learning_loss(self, 
                        states: torch.Tensor, 
                        actions: torch.Tensor, 
                        values: torch.Tensor) -> torch.Tensor:
        """
        时序差分学习损失，结合价值函数训练
        
        Args:
            states: 状态 (batch_size, state_dim)
            actions: 动作 (batch_size, action_dim)
            values: 价值估计 (batch_size, 1)
            
        Returns:
            TD学习损失
        """
        batch_size = states.shape[0]
        
        # 模拟下一状态（在离线设置中，我们使用简单的状态转移模型）
        next_states = self._simulate_next_states(states, actions)
        
        # 计算下一状态的价值
        with torch.no_grad():
            _, _, next_values, _ = self.agent.network(next_states)
        
        # 模拟奖励（基于动作和状态变化）
        rewards = self._simulate_rewards(states, actions, next_states)
        
        # TD目标
        gamma = self.agent.gamma
        td_targets = rewards + gamma * next_values.squeeze()
        
        # TD误差
        td_errors = td_targets - values.squeeze()
        
        # TD损失（Huber损失，对异常值更鲁棒）
        td_loss = F.smooth_l1_loss(values.squeeze(), td_targets)
        
        return td_loss
        
    def _simulate_next_states(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        模拟下一状态（简化版本）
        
        Args:
            states: 当前状态
            actions: 动作
            
        Returns:
            模拟的下一状态
        """
        # 简单的线性状态转移模型
        # 在实际应用中，这里可以使用更复杂的市场动态模型
        
        # 添加小的随机扰动来模拟市场变化
        noise = torch.randn_like(states) * 0.01
        
        # 动作对状态的影响（简化模型）
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
        
        # 扩展动作维度以匹配状态维度
        action_effect = actions.repeat(1, states.shape[1] // actions.shape[1])
        if action_effect.shape[1] != states.shape[1]:
            action_effect = action_effect[:, :states.shape[1]]
        
        next_states = states + 0.1 * action_effect + noise
        
        return next_states
        
    def _simulate_rewards(self, 
                         states: torch.Tensor, 
                         actions: torch.Tensor, 
                         next_states: torch.Tensor) -> torch.Tensor:
        """
        模拟奖励函数
        
        Args:
            states: 当前状态
            actions: 动作
            next_states: 下一状态
            
        Returns:
            模拟的奖励
        """
        # 基于状态变化和动作的简单奖励模型
        state_change = torch.mean(next_states - states, dim=1)
        
        if actions.dim() > 1:
            action_magnitude = torch.mean(actions, dim=1)
        else:
            action_magnitude = actions
        
        # 奖励 = 状态改善 - 动作成本
        rewards = state_change - 0.01 * torch.abs(action_magnitude)
        
        return rewards
        
    def _compute_value_loss(self, 
                          values: torch.Tensor, 
                          states: torch.Tensor, 
                          actions: torch.Tensor) -> torch.Tensor:
        """
        计算价值函数损失
        
        Args:
            values: 价值估计
            states: 状态
            actions: 动作
            
        Returns:
            价值函数损失
        """
        # 使用TD学习的目标作为价值函数的监督信号
        next_states = self._simulate_next_states(states, actions)
        rewards = self._simulate_rewards(states, actions, next_states)
        
        with torch.no_grad():
            _, _, next_values, _ = self.agent.network(next_states)
            value_targets = rewards + self.agent.gamma * next_values.squeeze()
        
        value_loss = F.mse_loss(values.squeeze(), value_targets)
        
        return value_loss
        
    def _update_training_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """更新训练历史"""
        self.training_history['epoch'].append(self.current_epoch)
        self.training_history['train_loss'].append(train_metrics['total_loss'])
        self.training_history['bc_loss'].append(train_metrics['bc_loss'])
        self.training_history['td_loss'].append(train_metrics['td_loss'])
        self.training_history['value_loss'].append(train_metrics['value_loss'])
        
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics['total_loss'])
        else:
            self.training_history['val_loss'].append(train_metrics['total_loss'])
            
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """记录epoch结果"""
        train_loss = train_metrics['total_loss']
        val_loss = val_metrics.get('total_loss', train_loss)
        
        logger.info(
            f"Epoch {epoch + 1}/{self.config.epochs} - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"BC Loss: {train_metrics['bc_loss']:.6f}, "
            f"TD Loss: {train_metrics['td_loss']:.6f}, "
            f"Value Loss: {train_metrics['value_loss']:.6f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        )
        
    def _check_early_stopping(self, current_loss: float) -> bool:
        """检查早停条件"""
        if current_loss < self.best_loss - self.config.early_stopping_threshold:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.early_stopping_patience
            
    def _save_checkpoint(self, 
                        checkpoint_dir: str, 
                        epoch: int, 
                        train_metrics: Dict[str, float], 
                        val_metrics: Dict[str, float]):
        """保存训练检查点"""
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        
        checkpoint_data = {
            'epoch': epoch,
            'network_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
        
    def _save_final_checkpoint(self, checkpoint_dir: str) -> str:
        """保存最终模型检查点"""
        final_path = os.path.join(checkpoint_dir, "final_pretrained_model.pth")
        
        final_data = {
            'network_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'best_loss': self.best_loss,
            'epochs_trained': self.current_epoch + 1
        }
        
        torch.save(final_data, final_path)
        logger.info(f"最终预训练模型已保存: {final_path}")
        
        return final_path
        
    def save_checkpoint(self, filepath: str):
        """
        保存预训练检查点
        
        Args:
            filepath: 保存路径
        """
        checkpoint_data = {
            'epoch': self.current_epoch,
            'network_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config,
            'agent_config': self.agent.config
        }
        
        torch.save(checkpoint_data, filepath)
        logger.info(f"预训练检查点已保存: {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """
        加载预训练检查点
        
        Args:
            filepath: 检查点文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
            
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载网络参数
        self.agent.network.load_state_dict(checkpoint['network_state_dict'])
        
        # 加载训练状态
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'bc_loss': [], 'td_loss': [], 'value_loss': []
        })
        
        # 加载优化器状态（如果存在）
        if hasattr(self, 'optimizer') and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # 加载调度器状态（如果存在）
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logger.info(f"预训练检查点已加载: {filepath}")
        
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.training_history['epoch']:
            return {'status': 'not_trained'}
            
        return {
            'status': 'trained',
            'epochs_completed': len(self.training_history['epoch']),
            'best_loss': self.best_loss,
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
            'training_history': self.training_history
        }