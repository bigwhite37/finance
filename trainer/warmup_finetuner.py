"""
热身微调器

实现短期在线适应，支持Critic专用更新，冻结Actor参数，
开发价值基线重估算法，添加收敛检测和失败回退机制。
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
import time
from pathlib import Path

from buffers.online_replay_buffer import OnlineReplayBuffer, TrajectoryData
from rl_agent.cvar_ppo_agent import CVaRPPOAgent

logger = logging.getLogger(__name__)


@dataclass
class WarmUpFinetunerConfig:
    """热身微调器配置"""
    warmup_days: int = 60
    warmup_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    critic_only_updates: bool = True
    convergence_threshold: float = 1e-4
    convergence_patience: int = 5
    max_iterations: int = 100
    gradient_clip_norm: float = 0.5
    value_loss_weight: float = 1.0
    cvar_loss_weight: float = 0.5
    baseline_reestimation_window: int = 30
    failure_threshold: float = 0.1
    enable_early_stopping: bool = True
    checkpoint_interval: int = 5
    min_data_samples: int = 100


class WarmUpFinetuner:
    """
    热身微调器
    
    实现短期在线适应，冻结Actor参数仅更新Critic网络，
    使用最近市场数据重新估计Q值和价值基线，
    添加收敛检测和失败回退机制。
    """
    
    def __init__(self, agent: CVaRPPOAgent, config: WarmUpFinetunerConfig):
        """
        初始化热身微调器
        
        Args:
            agent: CVaR-PPO智能体实例
            config: 热身微调器配置
        """
        self.agent = agent
        self.config = config
        self.device = agent.device
        
        # 训练状态
        self.current_iteration = 0
        self.convergence_counter = 0
        self.best_loss = float('inf')
        self.baseline_values = {}
        
        # 训练历史
        self.training_history = {
            'iteration': [],
            'value_loss': [],
            'cvar_loss': [],
            'total_loss': [],
            'convergence_metric': []
        }
        
        # 备份原始模型状态（用于失败回退）
        self.backup_state = None
        
        # 优化器（仅用于Critic）
        self.critic_optimizer = None
        self.critic_scheduler = None
        
        logger.info(f"初始化热身微调器 - 配置: {config}")
        
    def finetune(self, 
                recent_data: OnlineReplayBuffer, 
                days: Optional[int] = None,
                checkpoint_dir: str = "checkpoints/warmup") -> Dict[str, Any]:
        """
        执行热身微调
        
        Args:
            recent_data: 最近的在线数据缓冲区
            days: 数据时间窗口（天数）
            checkpoint_dir: 检查点保存目录
            
        Returns:
            微调结果统计
        """
        if days is None:
            days = self.config.warmup_days
            
        logger.info(f"开始热身微调，使用最近{days}天的数据...")
        
        # 创建检查点目录
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # 备份原始模型状态
        self._backup_model_state()
        
        # 收集最近的交互数据
        recent_trajectories = recent_data.get_recent_trajectory(window=days)
        
        if len(recent_trajectories) < self.config.min_data_samples:
            logger.warning(f"数据样本不足: {len(recent_trajectories)} < {self.config.min_data_samples}")
            return {
                'status': 'failed',
                'reason': 'insufficient_data',
                'samples_available': len(recent_trajectories),
                'samples_required': self.config.min_data_samples
            }
        
        # 准备数据加载器
        data_loader = self._prepare_data_loader(recent_trajectories)
        
        # 设置Critic专用优化器
        self._setup_critic_optimizer()
        
        # 冻结Actor参数
        if self.config.critic_only_updates:
            self.agent.freeze_actor()
            logger.info("Actor参数已冻结，仅更新Critic")
        
        # 微调循环
        try:
            result = self._finetune_loop(data_loader, checkpoint_dir)
            
            # 重新估计价值基线
            if result['status'] == 'converged':
                self._reestimate_value_baseline(recent_trajectories)
                result['baseline_updated'] = True
            
        except Exception as e:
            logger.error(f"热身微调过程中发生错误: {e}")
            result = self._handle_failure()
            
        finally:
            # 解冻Actor参数
            if self.config.critic_only_updates:
                self.agent.unfreeze_actor()
                logger.info("Actor参数已解冻")
        
        logger.info(f"热身微调完成: {result}")
        return result
        
    def _backup_model_state(self):
        """备份原始模型状态"""
        self.backup_state = {
            'network_state_dict': self.agent.network.state_dict().copy(),
            'optimizer_states': self.agent.get_optimizer_states()
        }
        logger.info("模型状态已备份")
        
    def _prepare_data_loader(self, trajectories: List[TrajectoryData]) -> DataLoader:
        """准备数据加载器"""
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        values_list = []
        
        for traj in trajectories:
            states_list.extend(traj.states)
            actions_list.extend(traj.actions)
            rewards_list.extend(traj.rewards)
            next_states_list.extend(traj.next_states)
            dones_list.extend(traj.dones)
            
            # 如果轨迹包含价值估计，使用它们；否则重新计算
            if traj.values is not None:
                values_list.extend(traj.values)
            else:
                # 使用当前网络重新计算价值
                with torch.no_grad():
                    states_tensor = torch.FloatTensor(traj.states).to(self.device)
                    _, _, values, _ = self.agent.network(states_tensor)
                    values_list.extend(values.cpu().numpy().flatten())
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states_list)
        actions_tensor = torch.FloatTensor(actions_list)
        rewards_tensor = torch.FloatTensor(rewards_list)
        next_states_tensor = torch.FloatTensor(next_states_list)
        dones_tensor = torch.FloatTensor(dones_list)
        values_tensor = torch.FloatTensor(values_list)
        
        # 创建数据集
        dataset = TensorDataset(
            states_tensor, actions_tensor, rewards_tensor,
            next_states_tensor, dones_tensor, values_tensor
        )
        
        # 创建数据加载器
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        logger.info(f"数据加载器准备完成: {len(dataset)} 个样本, {len(data_loader)} 个批次")
        return data_loader
        
    def _setup_critic_optimizer(self):
        """设置Critic专用优化器"""
        # 确保智能体使用分离优化器
        if not self.agent.use_split_optimizers:
            self.agent.split_optimizers()
        
        # 创建专门的Critic优化器（可能与智能体的不同学习率）
        self.critic_optimizer = torch.optim.Adam(
            self.agent.network.critic_parameters(),
            lr=self.config.learning_rate,
            eps=1e-8
        )
        
        # 学习率调度器
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        logger.info(f"Critic优化器设置完成，学习率: {self.config.learning_rate}")
        
    def _finetune_loop(self, data_loader: DataLoader, checkpoint_dir: str) -> Dict[str, Any]:
        """微调循环"""
        self.current_iteration = 0
        self.convergence_counter = 0
        self.best_loss = float('inf')
        
        for iteration in range(self.config.max_iterations):
            self.current_iteration = iteration
            
            # 训练一个epoch
            epoch_metrics = self._train_epoch(data_loader)
            
            # 更新训练历史
            self._update_training_history(epoch_metrics)
            
            # 学习率调度
            if self.critic_scheduler:
                self.critic_scheduler.step(epoch_metrics['total_loss'])
            
            # 日志输出
            self._log_iteration_results(iteration, epoch_metrics)
            
            # 保存检查点
            if (iteration + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, iteration, epoch_metrics)
            
            # 收敛检查
            if self._check_convergence(epoch_metrics['total_loss']):
                logger.info(f"收敛检测成功，在第 {iteration + 1} 次迭代停止")
                return {
                    'status': 'converged',
                    'iterations': iteration + 1,
                    'final_loss': epoch_metrics['total_loss'],
                    'training_history': self.training_history
                }
            
            # 失败检查
            if self._check_failure(epoch_metrics):
                logger.warning(f"检测到训练失败，在第 {iteration + 1} 次迭代停止")
                return self._handle_failure()
        
        # 达到最大迭代次数
        logger.info(f"达到最大迭代次数 {self.config.max_iterations}")
        return {
            'status': 'max_iterations',
            'iterations': self.config.max_iterations,
            'final_loss': self.training_history['total_loss'][-1] if self.training_history['total_loss'] else float('inf'),
            'training_history': self.training_history
        }
        
    def _train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.agent.network.train()
        
        epoch_losses = {
            'value_loss': [],
            'cvar_loss': [],
            'total_loss': []
        }
        
        for batch in data_loader:
            states, actions, rewards, next_states, dones, old_values = batch
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            old_values = old_values.to(self.device)
            
            # 前向传播
            _, _, current_values, cvar_estimates = self.agent.network(states)
            _, _, next_values, _ = self.agent.network(next_states)
            
            # 计算目标值
            with torch.no_grad():
                value_targets = rewards + self.agent.gamma * next_values.squeeze() * (1 - dones)
                cvar_targets = self._compute_cvar_targets(rewards)
            
            # 计算损失
            value_loss = self._compute_value_loss(current_values, value_targets)
            cvar_loss = self._compute_cvar_loss(cvar_estimates, cvar_targets)
            
            total_loss = (
                self.config.value_loss_weight * value_loss +
                self.config.cvar_loss_weight * cvar_loss
            )
            
            # 反向传播（仅更新Critic）
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.agent.network.critic_parameters(),
                self.config.gradient_clip_norm
            )
            
            self.critic_optimizer.step()
            
            # 记录损失
            epoch_losses['value_loss'].append(value_loss.item())
            epoch_losses['cvar_loss'].append(cvar_loss.item())
            epoch_losses['total_loss'].append(total_loss.item())
        
        # 计算平均损失
        return {key: np.mean(values) for key, values in epoch_losses.items()}
        
    def _compute_value_loss(self, predicted_values: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """计算价值函数损失"""
        # 确保维度匹配
        predicted_values = predicted_values.squeeze()
        target_values = target_values.squeeze()
        
        # 使用Huber损失，对异常值更鲁棒
        value_loss = F.smooth_l1_loss(predicted_values, target_values)
        
        return value_loss
        
    def _compute_cvar_loss(self, predicted_cvar: torch.Tensor, target_cvar: torch.Tensor) -> torch.Tensor:
        """计算CVaR损失"""
        # 确保维度匹配
        predicted_cvar = predicted_cvar.squeeze()
        target_cvar = target_cvar.squeeze()
        
        # CVaR损失
        cvar_loss = F.mse_loss(predicted_cvar, target_cvar)
        
        # 添加CVaR约束惩罚
        cvar_penalty = F.relu(predicted_cvar.mean() - self.agent.cvar_threshold)
        
        return cvar_loss + 2.0 * cvar_penalty
        
    def _compute_cvar_targets(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算CVaR目标值"""
        # 计算分位数
        var_quantile = torch.quantile(rewards, self.agent.cvar_alpha)
        
        # 计算CVaR
        tail_rewards = rewards[rewards <= var_quantile]
        if len(tail_rewards) > 0:
            cvar = torch.mean(tail_rewards)
        else:
            cvar = var_quantile
        
        return cvar.repeat(len(rewards))
        
    def _update_training_history(self, metrics: Dict[str, float]):
        """更新训练历史"""
        self.training_history['iteration'].append(self.current_iteration)
        self.training_history['value_loss'].append(metrics['value_loss'])
        self.training_history['cvar_loss'].append(metrics['cvar_loss'])
        self.training_history['total_loss'].append(metrics['total_loss'])
        
        # 计算收敛指标
        convergence_metric = self._compute_convergence_metric(metrics)
        self.training_history['convergence_metric'].append(convergence_metric)
        
    def _compute_convergence_metric(self, metrics: Dict[str, float]) -> float:
        """计算收敛指标"""
        # 使用损失变化率作为收敛指标
        if len(self.training_history['total_loss']) < 2:
            return float('inf')
        
        current_loss = metrics['total_loss']
        previous_loss = self.training_history['total_loss'][-1]
        
        # 相对变化率
        if previous_loss != 0:
            change_rate = abs(current_loss - previous_loss) / abs(previous_loss)
        else:
            change_rate = abs(current_loss)
        
        return change_rate
        
    def _log_iteration_results(self, iteration: int, metrics: Dict[str, float]):
        """记录迭代结果"""
        logger.info(
            f"热身微调 Iter {iteration + 1}/{self.config.max_iterations} - "
            f"Total Loss: {metrics['total_loss']:.6f}, "
            f"Value Loss: {metrics['value_loss']:.6f}, "
            f"CVaR Loss: {metrics['cvar_loss']:.6f}, "
            f"LR: {self.critic_optimizer.param_groups[0]['lr']:.2e}"
        )
        
    def _check_convergence(self, current_loss: float) -> bool:
        """检查收敛条件"""
        if not self.config.enable_early_stopping:
            return False
        
        # 检查损失改善
        if current_loss < self.best_loss - self.config.convergence_threshold:
            self.best_loss = current_loss
            self.convergence_counter = 0
            return False
        else:
            self.convergence_counter += 1
            return self.convergence_counter >= self.config.convergence_patience
            
    def _check_failure(self, metrics: Dict[str, float]) -> bool:
        """检查失败条件"""
        # 检查损失是否过大或包含NaN
        if (metrics['total_loss'] > self.config.failure_threshold or
            np.isnan(metrics['total_loss']) or
            np.isinf(metrics['total_loss'])):
            return True
        
        # 检查损失是否持续增长
        if len(self.training_history['total_loss']) >= 5:
            recent_losses = self.training_history['total_loss'][-5:]
            if all(recent_losses[i] < recent_losses[i+1] for i in range(len(recent_losses)-1)):
                logger.warning("检测到损失持续增长")
                return True
        
        return False
        
    def _handle_failure(self) -> Dict[str, Any]:
        """处理失败情况"""
        logger.warning("热身微调失败，回退到原始模型状态")
        
        if self.backup_state:
            # 恢复网络参数
            self.agent.network.load_state_dict(self.backup_state['network_state_dict'])
            
            # 恢复优化器状态
            self.agent.load_optimizer_states(self.backup_state['optimizer_states'])
            
            logger.info("模型状态已回退")
        
        return {
            'status': 'failed',
            'reason': 'training_failure',
            'iterations': self.current_iteration + 1,
            'backup_restored': self.backup_state is not None,
            'training_history': self.training_history
        }
        
    def _save_checkpoint(self, checkpoint_dir: str, iteration: int, metrics: Dict[str, float]):
        """保存检查点"""
        checkpoint_path = os.path.join(checkpoint_dir, f"warmup_checkpoint_iter_{iteration + 1}.pth")
        
        checkpoint_data = {
            'iteration': iteration,
            'network_state_dict': self.agent.network.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict() if self.critic_scheduler else None,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config,
            'metrics': metrics
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        logger.debug(f"热身微调检查点已保存: {checkpoint_path}")
        
    def freeze_actor_update_critic(self, batch: Dict[str, torch.Tensor]):
        """
        冻结Actor，仅更新Critic
        
        Args:
            batch: 训练批次数据
        """
        # 确保Actor被冻结
        if not self.agent.actor_frozen:
            self.agent.freeze_actor()
        
        # 提取批次数据
        states = batch['states'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # 前向传播
        _, _, current_values, cvar_estimates = self.agent.network(states)
        
        with torch.no_grad():
            _, _, next_values, _ = self.agent.network(next_states)
        
        # 计算目标值
        value_targets = rewards + self.agent.gamma * next_values.squeeze() * (1 - dones)
        cvar_targets = self._compute_cvar_targets(rewards)
        
        # 计算损失
        value_loss = self._compute_value_loss(current_values, value_targets)
        cvar_loss = self._compute_cvar_loss(cvar_estimates, cvar_targets)
        
        total_loss = (
            self.config.value_loss_weight * value_loss +
            self.config.cvar_loss_weight * cvar_loss
        )
        
        # 仅更新Critic
        if self.agent.use_split_optimizers and self.agent.critic_optimizer:
            self.agent.critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.network.critic_parameters(),
                self.config.gradient_clip_norm
            )
            self.agent.critic_optimizer.step()
        else:
            logger.warning("智能体未使用分离优化器，无法执行Critic专用更新")
        
    def reestimate_value_baseline(self, recent_trajectories: List[TrajectoryData]):
        """
        重新估计价值基线，适应最近市场数据
        
        Args:
            recent_trajectories: 最近的轨迹数据
        """
        self._reestimate_value_baseline(recent_trajectories)
        
    def _reestimate_value_baseline(self, recent_trajectories: List[TrajectoryData]):
        """重新估计价值基线的内部实现"""
        logger.info("开始重新估计价值基线...")
        
        if not recent_trajectories:
            logger.warning("没有可用的轨迹数据进行基线重估")
            return
        
        self.agent.network.eval()
        
        # 收集所有状态和实际回报
        all_states = []
        all_returns = []
        
        for traj in recent_trajectories:
            states = traj.states
            rewards = traj.rewards
            
            # 计算折扣回报
            returns = self._compute_discounted_returns(rewards)
            
            all_states.extend(states)
            all_returns.extend(returns)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(all_states).to(self.device)
        returns_tensor = torch.FloatTensor(all_returns).to(self.device)
        
        # 计算新的价值估计
        with torch.no_grad():
            _, _, new_values, _ = self.agent.network(states_tensor)
            new_values = new_values.squeeze()
        
        # 计算基线统计
        old_baseline_mean = returns_tensor.mean().item()
        new_baseline_mean = new_values.mean().item()
        baseline_mse = F.mse_loss(new_values, returns_tensor).item()
        
        # 保存基线信息
        self.baseline_values = {
            'old_mean': old_baseline_mean,
            'new_mean': new_baseline_mean,
            'mse': baseline_mse,
            'samples': len(all_states),
            'window_days': self.config.baseline_reestimation_window
        }
        
        logger.info(
            f"价值基线重估完成 - "
            f"旧基线均值: {old_baseline_mean:.4f}, "
            f"新基线均值: {new_baseline_mean:.4f}, "
            f"MSE: {baseline_mse:.6f}, "
            f"样本数: {len(all_states)}"
        )
        
    def _compute_discounted_returns(self, rewards: np.ndarray) -> np.ndarray:
        """计算折扣回报"""
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.agent.gamma * running_return
            returns[t] = running_return
        
        return returns
        
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.training_history['iteration']:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'iterations_completed': len(self.training_history['iteration']),
            'best_loss': self.best_loss,
            'final_loss': self.training_history['total_loss'][-1] if self.training_history['total_loss'] else None,
            'convergence_achieved': self.convergence_counter >= self.config.convergence_patience,
            'baseline_values': self.baseline_values,
            'training_history': self.training_history
        }
        
    def load_checkpoint(self, filepath: str):
        """
        加载热身微调检查点
        
        Args:
            filepath: 检查点文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载网络参数
        self.agent.network.load_state_dict(checkpoint['network_state_dict'])
        
        # 加载训练状态
        self.current_iteration = checkpoint.get('iteration', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'iteration': [], 'value_loss': [], 'cvar_loss': [],
            'total_loss': [], 'convergence_metric': []
        })
        
        logger.info(f"热身微调检查点已加载: {filepath}")