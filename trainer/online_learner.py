"""
在线学习器

支持持续在线优化，集成混合批次处理和重要性权重应用，
实现信任域约束的在线更新算法，添加动态风险权重调整。
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
from collections import deque

from sampler.mixture_sampler import MixtureSampler
from rl_agent.cvar_ppo_agent import CVaRPPOAgent

logger = logging.getLogger(__name__)


@dataclass
class OnlineLearnerConfig:
    """在线学习器配置"""
    batch_size: int = 256
    learning_rate: float = 3e-4
    trust_region_beta: float = 1.0
    beta_decay: float = 0.99
    beta_min: float = 0.1
    kl_target: float = 0.01
    kl_threshold: float = 0.05
    max_kl_violations: int = 3
    gradient_clip_norm: float = 0.5
    update_frequency: int = 1
    max_updates_per_step: int = 10
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    cvar_loss_weight: float = 1.0
    entropy_weight: float = 0.01
    risk_weight_base: float = 1.0
    risk_weight_scale: float = 2.0
    enable_trust_region: bool = True
    enable_dynamic_risk: bool = True
    checkpoint_interval: int = 50
    performance_window: int = 100


class OnlineLearner:
    """
    在线学习器
    
    支持持续在线优化，集成混合批次处理和重要性权重应用，
    实现信任域约束的在线更新算法，添加动态风险权重调整。
    """
    
    def __init__(self, agent: CVaRPPOAgent, config: OnlineLearnerConfig):
        """
        初始化在线学习器
        
        Args:
            agent: CVaR-PPO智能体实例
            config: 在线学习器配置
        """
        self.agent = agent
        self.config = config
        self.device = agent.device
        
        # 信任域约束相关
        self.trust_region_beta = config.trust_region_beta
        self.kl_violation_count = 0
        self.old_policy_params = None
        
        # 动态风险权重
        self.current_risk_weight = config.risk_weight_base
        self.performance_history = deque(maxlen=config.performance_window)
        
        # 训练统计
        self.update_count = 0
        self.total_updates = 0
        self.training_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'cvar_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'trust_region_beta': [],
            'risk_weight': [],
            'importance_weight_stats': []
        }
        
        # 优化器设置
        self._setup_optimizers()
        
        logger.info(f"初始化在线学习器 - 配置: {config}")
        
    def _setup_optimizers(self):
        """设置优化器"""
        # 确保使用分离优化器
        if not self.agent.use_split_optimizers:
            self.agent.split_optimizers()
        
        # 更新学习率
        if self.agent.actor_optimizer:
            for param_group in self.agent.actor_optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate
        
        if self.agent.critic_optimizer:
            for param_group in self.agent.critic_optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate
        
        logger.info(f"优化器设置完成，学习率: {self.config.learning_rate}")
        
    def online_update(self, 
                     mixed_batch: Dict[str, Any], 
                     importance_weights: np.ndarray,
                     rho: float) -> Dict[str, Any]:
        """
        在线更新策略
        
        Args:
            mixed_batch: 混合批次数据
            importance_weights: 重要性权重
            rho: 当前在线采样比例
            
        Returns:
            更新统计信息
        """
        if len(mixed_batch.get('states', [])) == 0:
            logger.warning("混合批次为空，跳过更新")
            return {'status': 'skipped', 'reason': 'empty_batch'}
        
        # 保存旧策略参数用于KL散度计算
        self._save_old_policy_params()
        
        # 转换数据为张量
        batch_tensors = self._prepare_batch_tensors(mixed_batch, importance_weights)
        
        # 计算动态风险权重
        dynamic_risk_weight = self._compute_dynamic_risk_weight(rho)
        
        # 执行更新
        update_metrics = self._perform_update(batch_tensors, dynamic_risk_weight, rho)
        
        # 信任域约束检查
        kl_divergence = self._compute_kl_divergence(batch_tensors)
        trust_region_metrics = self._apply_trust_region_constraint(kl_divergence)
        
        # 合并指标
        update_metrics.update(trust_region_metrics)
        update_metrics['dynamic_risk_weight'] = dynamic_risk_weight
        update_metrics['rho'] = rho
        
        # 更新训练统计
        self._update_training_statistics(update_metrics)
        
        # 更新计数
        self.update_count += 1
        self.total_updates += 1
        
        return update_metrics
        
    def _save_old_policy_params(self):
        """保存旧策略参数"""
        self.old_policy_params = {}
        for name, param in self.agent.network.named_parameters():
            if 'actor' in name or 'shared' in name:  # 策略相关参数
                self.old_policy_params[name] = param.data.clone()
                
    def _prepare_batch_tensors(self, 
                              mixed_batch: Dict[str, Any], 
                              importance_weights: np.ndarray) -> Dict[str, torch.Tensor]:
        """准备批次张量"""
        batch_tensors = {}
        
        # 基本数据
        batch_tensors['states'] = torch.FloatTensor(mixed_batch['states']).to(self.device)
        batch_tensors['actions'] = torch.FloatTensor(mixed_batch['actions']).to(self.device)
        batch_tensors['rewards'] = torch.FloatTensor(mixed_batch['rewards']).to(self.device)
        batch_tensors['next_states'] = torch.FloatTensor(mixed_batch['next_states']).to(self.device)
        batch_tensors['dones'] = torch.FloatTensor(mixed_batch['dones']).to(self.device)
        
        # 重要性权重
        batch_tensors['importance_weights'] = torch.FloatTensor(importance_weights).to(self.device)
        
        # 可选数据
        if 'log_probs' in mixed_batch and len(mixed_batch['log_probs']) > 0:
            batch_tensors['old_log_probs'] = torch.FloatTensor(mixed_batch['log_probs']).to(self.device)
        else:
            # 如果没有旧的log_probs，使用当前策略计算
            with torch.no_grad():
                action_mean, action_std, _, _ = self.agent.network(batch_tensors['states'])
                dist = torch.distributions.Normal(action_mean, action_std)
                batch_tensors['old_log_probs'] = dist.log_prob(batch_tensors['actions']).sum(dim=-1)
        
        if 'values' in mixed_batch and len(mixed_batch['values']) > 0:
            batch_tensors['old_values'] = torch.FloatTensor(mixed_batch['values']).to(self.device)
        else:
            # 计算旧的价值估计
            with torch.no_grad():
                _, _, old_values, _ = self.agent.network(batch_tensors['states'])
                batch_tensors['old_values'] = old_values.squeeze()
        
        return batch_tensors
        
    def _compute_dynamic_risk_weight(self, rho: float) -> float:
        """
        计算动态风险权重 λ(ρ)
        
        Args:
            rho: 在线采样比例
            
        Returns:
            动态风险权重
        """
        if not self.config.enable_dynamic_risk:
            return self.config.risk_weight_base
        
        # 基础公式: λ(ρ) = λ_base + λ_scale * ρ
        base_weight = self.config.risk_weight_base
        scale_weight = self.config.risk_weight_scale * rho
        
        # 根据性能历史调整
        if len(self.performance_history) > 10:
            recent_performance = np.mean(list(self.performance_history)[-10:])
            if recent_performance < -0.05:  # 性能下降
                scale_weight *= 1.5  # 增加风险约束
            elif recent_performance > 0.05:  # 性能提升
                scale_weight *= 0.8  # 适度放松风险约束
        
        dynamic_weight = base_weight + scale_weight
        
        # 确保权重在合理范围内
        dynamic_weight = np.clip(dynamic_weight, 0.1, 5.0)
        
        return dynamic_weight
        
    def _perform_update(self, 
                       batch_tensors: Dict[str, torch.Tensor],
                       dynamic_risk_weight: float,
                       rho: float) -> Dict[str, Any]:
        """执行策略更新"""
        self.agent.network.train()
        
        # 前向传播
        action_mean, action_std, values, cvar_estimates = self.agent.network(batch_tensors['states'])
        
        # 计算新的动作概率
        dist = torch.distributions.Normal(action_mean, action_std)
        new_log_probs = dist.log_prob(batch_tensors['actions']).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # 计算优势函数
        advantages = self._compute_advantages(batch_tensors, values)
        
        # 计算各项损失
        policy_loss = self._compute_policy_loss(
            new_log_probs, batch_tensors['old_log_probs'], 
            advantages, batch_tensors['importance_weights']
        )
        
        value_loss = self._compute_value_loss(
            values, batch_tensors, batch_tensors['importance_weights']
        )
        
        cvar_loss = self._compute_cvar_loss(
            cvar_estimates, batch_tensors['rewards'], 
            batch_tensors['importance_weights'], dynamic_risk_weight
        )
        
        # 总损失
        total_loss = (
            self.config.policy_loss_weight * policy_loss +
            self.config.value_loss_weight * value_loss +
            self.config.cvar_loss_weight * cvar_loss -
            self.config.entropy_weight * entropy
        )
        
        # 反向传播和优化
        if self.agent.actor_optimizer and self.agent.critic_optimizer:
            # 分离优化器更新
            self.agent.actor_optimizer.zero_grad()
            self.agent.critic_optimizer.zero_grad()
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.agent.network.parameters(),
                self.config.gradient_clip_norm
            )
            
            self.agent.actor_optimizer.step()
            self.agent.critic_optimizer.step()
        else:
            logger.warning("优化器未正确设置，跳过参数更新")
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'cvar_loss': cvar_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropy.item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
        }
        
    def _compute_advantages(self, 
                           batch_tensors: Dict[str, torch.Tensor],
                           values: torch.Tensor) -> torch.Tensor:
        """计算优势函数"""
        rewards = batch_tensors['rewards']
        next_states = batch_tensors['next_states']
        dones = batch_tensors['dones']
        
        # 计算下一状态的价值
        with torch.no_grad():
            _, _, next_values, _ = self.agent.network(next_states)
            next_values = next_values.squeeze()
        
        # TD目标
        td_targets = rewards + self.agent.gamma * next_values * (1 - dones)
        
        # 优势函数
        advantages = td_targets - values.squeeze()
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
        
    def _compute_policy_loss(self, 
                            new_log_probs: torch.Tensor,
                            old_log_probs: torch.Tensor,
                            advantages: torch.Tensor,
                            importance_weights: torch.Tensor) -> torch.Tensor:
        """计算策略损失"""
        # PPO比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 应用重要性权重
        weighted_advantages = advantages * importance_weights
        
        # PPO裁剪损失
        surr1 = ratio * weighted_advantages
        surr2 = torch.clamp(ratio, 1 - self.agent.clip_epsilon, 1 + self.agent.clip_epsilon) * weighted_advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
        
    def _compute_value_loss(self, 
                           values: torch.Tensor,
                           batch_tensors: Dict[str, torch.Tensor],
                           importance_weights: torch.Tensor) -> torch.Tensor:
        """计算价值损失"""
        rewards = batch_tensors['rewards']
        next_states = batch_tensors['next_states']
        dones = batch_tensors['dones']
        
        # 计算价值目标
        with torch.no_grad():
            _, _, next_values, _ = self.agent.network(next_states)
            value_targets = rewards + self.agent.gamma * next_values.squeeze() * (1 - dones)
        
        # 加权价值损失
        value_errors = (values.squeeze() - value_targets) ** 2
        weighted_value_loss = (value_errors * importance_weights).mean()
        
        return weighted_value_loss
        
    def _compute_cvar_loss(self, 
                          cvar_estimates: torch.Tensor,
                          rewards: torch.Tensor,
                          importance_weights: torch.Tensor,
                          dynamic_risk_weight: float) -> torch.Tensor:
        """计算CVaR损失"""
        # 计算CVaR目标
        var_quantile = torch.quantile(rewards, self.agent.cvar_alpha)
        tail_rewards = rewards[rewards <= var_quantile]
        
        if len(tail_rewards) > 0:
            cvar_target = torch.mean(tail_rewards)
        else:
            cvar_target = var_quantile
        
        # CVaR预测损失
        cvar_pred_loss = F.mse_loss(cvar_estimates.squeeze(), cvar_target.repeat(len(cvar_estimates)))
        
        # CVaR约束惩罚
        cvar_penalty = F.relu(cvar_estimates.squeeze().mean() - self.agent.cvar_threshold)
        
        # 动态风险权重调整
        total_cvar_loss = dynamic_risk_weight * (cvar_pred_loss + 2.0 * cvar_penalty)
        
        return total_cvar_loss
        
    def _compute_kl_divergence(self, batch_tensors: Dict[str, torch.Tensor]) -> float:
        """计算KL散度"""
        if self.old_policy_params is None:
            return 0.0
        
        states = batch_tensors['states']
        
        # 当前策略分布
        action_mean_new, action_std_new, _, _ = self.agent.network(states)
        
        # 临时恢复旧参数计算旧策略分布
        current_params = {}
        for name, param in self.agent.network.named_parameters():
            if name in self.old_policy_params:
                current_params[name] = param.data.clone()
                param.data.copy_(self.old_policy_params[name])
        
        with torch.no_grad():
            action_mean_old, action_std_old, _, _ = self.agent.network(states)
        
        # 恢复当前参数
        for name, param in self.agent.network.named_parameters():
            if name in current_params:
                param.data.copy_(current_params[name])
        
        # 计算KL散度
        old_dist = torch.distributions.Normal(action_mean_old, action_std_old)
        new_dist = torch.distributions.Normal(action_mean_new, action_std_new)
        
        kl_div = torch.distributions.kl_divergence(new_dist, old_dist).sum(dim=-1).mean()
        
        return kl_div.item()
        
    def apply_trust_region_constraint(self, new_policy_loss: torch.Tensor, old_policy) -> torch.Tensor:
        """
        应用信任域约束
        
        Args:
            new_policy_loss: 新策略损失
            old_policy: 旧策略（未使用，保持接口兼容）
            
        Returns:
            约束后的损失
        """
        if not self.config.enable_trust_region:
            return new_policy_loss
        
        # 信任域惩罚项已在_apply_trust_region_constraint中处理
        return new_policy_loss
        
    def _apply_trust_region_constraint(self, kl_divergence: float) -> Dict[str, Any]:
        """应用信任域约束"""
        trust_region_metrics = {
            'kl_divergence': kl_divergence,
            'trust_region_beta': self.trust_region_beta,
            'kl_violation': False
        }
        
        if not self.config.enable_trust_region:
            return trust_region_metrics
        
        # 检查KL散度是否超过阈值
        if kl_divergence > self.config.kl_threshold:
            self.kl_violation_count += 1
            trust_region_metrics['kl_violation'] = True
            
            logger.warning(f"KL散度超过阈值: {kl_divergence:.6f} > {self.config.kl_threshold}")
            
            # 如果违规次数过多，回退到之前的参数
            if self.kl_violation_count >= self.config.max_kl_violations:
                self._rollback_policy_params()
                logger.warning("KL违规次数过多，回退策略参数")
                trust_region_metrics['policy_rollback'] = True
                self.kl_violation_count = 0
        else:
            self.kl_violation_count = 0
        
        # 调整信任域参数
        self.anneal_trust_region(self.update_count)
        
        return trust_region_metrics
        
    def _rollback_policy_params(self):
        """回退策略参数"""
        if self.old_policy_params is None:
            logger.warning("没有旧参数可以回退")
            return
        
        for name, param in self.agent.network.named_parameters():
            if name in self.old_policy_params:
                param.data.copy_(self.old_policy_params[name])
        
        logger.info("策略参数已回退")
        
    def anneal_trust_region(self, episode: int):
        """
        退火信任域参数
        
        Args:
            episode: 当前episode
        """
        # 指数衰减
        self.trust_region_beta = max(
            self.config.beta_min,
            self.trust_region_beta * self.config.beta_decay
        )
        
        # 更新智能体的信任域参数
        self.agent.trust_region_beta = self.trust_region_beta
        
    def _update_training_statistics(self, metrics: Dict[str, Any]):
        """更新训练统计"""
        self.training_metrics['policy_loss'].append(metrics.get('policy_loss', 0.0))
        self.training_metrics['value_loss'].append(metrics.get('value_loss', 0.0))
        self.training_metrics['cvar_loss'].append(metrics.get('cvar_loss', 0.0))
        self.training_metrics['total_loss'].append(metrics.get('total_loss', 0.0))
        self.training_metrics['kl_divergence'].append(metrics.get('kl_divergence', 0.0))
        self.training_metrics['trust_region_beta'].append(self.trust_region_beta)
        self.training_metrics['risk_weight'].append(metrics.get('dynamic_risk_weight', 1.0))
        
        # 重要性权重统计
        if 'importance_weights' in metrics:
            self.training_metrics['importance_weight_stats'].append(metrics['importance_weights'])
        
        # 保持历史长度
        max_history = 1000
        for key in self.training_metrics:
            if len(self.training_metrics[key]) > max_history:
                self.training_metrics[key] = self.training_metrics[key][-max_history//2:]
                
    def update_performance_history(self, performance: float):
        """
        更新性能历史
        
        Args:
            performance: 性能指标
        """
        self.performance_history.append(performance)
        
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        if not self.training_metrics['total_loss']:
            return {'status': 'not_trained'}
        
        # 计算统计指标
        recent_window = 50
        recent_losses = self.training_metrics['total_loss'][-recent_window:]
        recent_kl = self.training_metrics['kl_divergence'][-recent_window:]
        
        return {
            'status': 'trained',
            'total_updates': self.total_updates,
            'current_trust_region_beta': self.trust_region_beta,
            'current_risk_weight': self.current_risk_weight,
            'kl_violation_count': self.kl_violation_count,
            'recent_loss_mean': np.mean(recent_losses),
            'recent_loss_std': np.std(recent_losses),
            'recent_kl_mean': np.mean(recent_kl),
            'recent_kl_std': np.std(recent_kl),
            'performance_history_length': len(self.performance_history),
            'training_metrics': self.training_metrics
        }
        
    def save_checkpoint(self, filepath: str):
        """
        保存在线学习器检查点
        
        Args:
            filepath: 保存路径
        """
        checkpoint_data = {
            'update_count': self.update_count,
            'total_updates': self.total_updates,
            'trust_region_beta': self.trust_region_beta,
            'kl_violation_count': self.kl_violation_count,
            'current_risk_weight': self.current_risk_weight,
            'performance_history': list(self.performance_history),
            'training_metrics': self.training_metrics,
            'config': self.config,
            'old_policy_params': self.old_policy_params
        }
        
        torch.save(checkpoint_data, filepath)
        logger.info(f"在线学习器检查点已保存: {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """
        加载在线学习器检查点
        
        Args:
            filepath: 检查点文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载状态
        self.update_count = checkpoint.get('update_count', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
        self.trust_region_beta = checkpoint.get('trust_region_beta', self.config.trust_region_beta)
        self.kl_violation_count = checkpoint.get('kl_violation_count', 0)
        self.current_risk_weight = checkpoint.get('current_risk_weight', self.config.risk_weight_base)
        
        # 加载历史数据
        if 'performance_history' in checkpoint:
            self.performance_history = deque(checkpoint['performance_history'], 
                                           maxlen=self.config.performance_window)
        
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
            
        if 'old_policy_params' in checkpoint:
            self.old_policy_params = checkpoint['old_policy_params']
        
        # 更新智能体的信任域参数
        self.agent.trust_region_beta = self.trust_region_beta
        
        logger.info(f"在线学习器检查点已加载: {filepath}")
        
    def reset_training_state(self):
        """重置训练状态"""
        self.update_count = 0
        self.total_updates = 0
        self.trust_region_beta = self.config.trust_region_beta
        self.kl_violation_count = 0
        self.current_risk_weight = self.config.risk_weight_base
        self.performance_history.clear()
        self.old_policy_params = None
        
        # 重置训练指标
        for key in self.training_metrics:
            self.training_metrics[key].clear()
        
        logger.info("在线学习器训练状态已重置")