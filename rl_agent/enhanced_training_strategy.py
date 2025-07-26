"""
增强的训练策略
集成先进的深度强化学习训练技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import deque
import random
from .enhanced_architecture import EnhancedActorCriticNetwork, AdaptiveLearningRateScheduler, CurriculumLearningStrategy

logger = logging.getLogger(__name__)


class PrioritizedExperienceReplay:
    """优先经验回放 - 重要转移经验优先学习"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        
        self.experiences = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, experience: Dict, priority: float = None):
        """添加经验"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
            
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
            self.priorities.append(priority)
        else:
            self.experiences[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """按优先级采样"""
        if len(self.experiences) < batch_size:
            return [], np.array([]), np.array([])
        
        # 计算采样概率
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.experiences), batch_size, p=probabilities)
        
        # 计算重要性权重
        weights = (len(self.experiences) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        
        # 获取经验
        experiences = [self.experiences[i] for i in indices]
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # 防止零优先级


class MultiTimeFrameTrainer:
    """多时间框架训练器 - 学习不同时间尺度的模式"""
    
    def __init__(self, timeframes: List[int] = [1, 5, 20]):
        self.timeframes = timeframes
        self.buffers = {tf: deque(maxlen=1000) for tf in timeframes}
        
    def add_experience(self, experience: Dict):
        """添加经验到所有时间框架"""
        for tf in self.timeframes:
            # 根据时间框架调整奖励
            adjusted_exp = experience.copy()
            if tf > 1:
                # 累积奖励
                adjusted_exp['reward'] = experience['reward'] * tf
            self.buffers[tf].append(adjusted_exp)
    
    def sample_batch(self, batch_size: int, timeframe: int) -> List[Dict]:
        """从指定时间框架采样"""
        if timeframe not in self.buffers or len(self.buffers[timeframe]) < batch_size:
            return []
        return random.sample(list(self.buffers[timeframe]), batch_size)


class EnsembleTraining:
    """集成训练 - 训练多个模型并组合预测"""
    
    def __init__(self, num_models: int = 3, diversity_weight: float = 0.1):
        self.num_models = num_models
        self.diversity_weight = diversity_weight
        self.models = []
        self.optimizers = []
        
    def add_model(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """添加模型到集成"""
        self.models.append(model)
        self.optimizers.append(optimizer)
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """集成训练步骤"""
        total_loss = 0.0
        individual_losses = []
        
        # 训练每个模型
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(batch['states'])
            
            # 计算基础损失
            base_loss = self._compute_base_loss(outputs, batch)
            
            # 计算多样性损失（鼓励模型差异）
            diversity_loss = self._compute_diversity_loss(outputs, i)
            
            # 总损失
            total_model_loss = base_loss + self.diversity_weight * diversity_loss
            
            # 反向传播
            total_model_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            individual_losses.append(total_model_loss.item())
            total_loss += total_model_loss.item()
        
        return {
            'ensemble_loss': total_loss / len(self.models),
            'individual_losses': individual_losses
        }
    
    def predict(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """集成预测"""
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(state)
                predictions.append(pred)
        
        # 平均预测
        ensemble_pred = {}
        for key in predictions[0].keys():
            if key in ['action_mean', 'value', 'cvar_estimate']:
                ensemble_pred[key] = torch.stack([p[key] for p in predictions]).mean(dim=0)
            elif key == 'action_std':
                # 标准差需要特殊处理
                stds = torch.stack([p[key] for p in predictions])
                ensemble_pred[key] = torch.sqrt(torch.mean(stds**2, dim=0))
        
        return ensemble_pred
    
    def _compute_base_loss(self, outputs: Dict, batch: Dict) -> torch.Tensor:
        """计算基础损失"""
        # 实现具体的损失计算逻辑
        # 这里需要根据具体的训练目标实现
        pass
    
    def _compute_diversity_loss(self, outputs: Dict, model_idx: int) -> torch.Tensor:
        """计算多样性损失"""
        # 鼓励模型输出的多样性
        diversity_loss = 0.0
        
        for other_idx, other_model in enumerate(self.models):
            if other_idx != model_idx:
                with torch.no_grad():
                    other_outputs = other_model(outputs['features'])
                
                # 计算输出差异
                action_diff = F.mse_loss(outputs['action_mean'], other_outputs['action_mean'])
                diversity_loss += torch.exp(-action_diff)  # 差异越大，损失越小
        
        return diversity_loss / (len(self.models) - 1)


class AdvancedPPOTrainer:
    """高级PPO训练器 - 集成多种训练技术"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 初始化增强网络
        self.network = EnhancedActorCriticNetwork(
            state_dim, action_dim, 
            hidden_dim=config.get('hidden_dim', 256),
            num_attention_heads=config.get('num_attention_heads', 8),
            num_residual_blocks=config.get('num_residual_blocks', 3)
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), 
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 自适应学习率调度
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            self.optimizer,
            patience=config.get('lr_patience', 10),
            factor=config.get('lr_decay_factor', 0.5)
        )
        
        # 课程学习
        self.curriculum = CurriculumLearningStrategy(
            total_episodes=config.get('total_episodes', 1000)
        )
        
        # 优先经验回放
        self.per_buffer = PrioritizedExperienceReplay(
            capacity=config.get('per_capacity', 10000),
            alpha=config.get('per_alpha', 0.6),
            beta=config.get('per_beta', 0.4)
        )
        
        # 多时间框架训练
        self.mtf_trainer = MultiTimeFrameTrainer(
            timeframes=config.get('timeframes', [1, 5, 20])
        )
        
        # 集成训练（如果启用）
        self.use_ensemble = config.get('use_ensemble', False)
        if self.use_ensemble:
            self.ensemble = EnsembleTraining(
                num_models=config.get('num_ensemble_models', 3)
            )
        
        # 训练统计
        self.training_stats = {
            'episode': 0,
            'actor_losses': [],
            'critic_losses': [],
            'kl_divergences': [],
            'explained_variances': [],
            'entropy_values': []
        }
        
        # 性能监控
        self.performance_window = deque(maxlen=100)
        
    def update_with_batch(self, batch: Dict, importance_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """使用批次数据更新网络"""
        
        # 提取批次数据
        states = torch.FloatTensor(batch['states'])
        actions = torch.FloatTensor(batch['actions'])
        rewards = torch.FloatTensor(batch['rewards'])
        old_log_probs = torch.FloatTensor(batch['log_probs'])
        old_values = torch.FloatTensor(batch['values'])
        dones = torch.FloatTensor(batch['dones'])
        
        # 前向传播
        outputs = self.network(states)
        action_mean = outputs['action_mean']
        action_std = outputs['action_std']
        values = outputs['value']
        cvar_estimates = outputs['cvar_estimate']
        
        # 计算新的log概率
        dist = torch.distributions.Normal(action_mean, action_std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # 计算GAE优势
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO损失计算
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Actor损失（PPO clip）
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.get('clip_epsilon', 0.2), 
                           1 + self.config.get('clip_epsilon', 0.2)) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 应用重要性权重（如果使用PER）
        if importance_weights is not None:
            actor_loss = (actor_loss * importance_weights).mean()
        
        # Critic损失（Huber损失，更稳定）
        value_pred_clipped = old_values + torch.clamp(values - old_values, 
                                                     -self.config.get('clip_epsilon', 0.2),
                                                     self.config.get('clip_epsilon', 0.2))
        value_loss1 = F.huber_loss(values, returns, reduction='none')
        value_loss2 = F.huber_loss(value_pred_clipped, returns, reduction='none')
        critic_loss = torch.max(value_loss1, value_loss2).mean()
        
        # CVaR损失
        cvar_target = self._compute_cvar_target(rewards, self.config.get('cvar_alpha', 0.05))
        cvar_loss = F.mse_loss(cvar_estimates.squeeze(), cvar_target)
        
        # 熵奖励（鼓励探索）
        entropy = dist.entropy().mean()
        entropy_loss = -self.config.get('entropy_coef', 0.01) * entropy
        
        # 总损失
        total_loss = (actor_loss + 
                     self.config.get('value_coef', 0.5) * critic_loss +
                     self.config.get('cvar_coef', 0.1) * cvar_loss +
                     entropy_loss)
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(), 
            max_norm=self.config.get('max_grad_norm', 1.0)
        )
        
        self.optimizer.step()
        
        # 更新学习率
        self.lr_scheduler.step(-total_loss.item())  # 使用负损失作为性能指标
        
        # 计算统计信息
        with torch.no_grad():
            kl_div = torch.mean(old_log_probs - new_log_probs)
            explained_var = 1 - torch.var(returns - values) / torch.var(returns)
        
        # 更新训练统计
        stats = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'cvar_loss': cvar_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_div.item(),
            'explained_variance': explained_var.item(),
            'grad_norm': grad_norm.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # 更新PER优先级（如果使用）
        if importance_weights is not None:
            td_errors = torch.abs(returns - values.squeeze()).detach().cpu().numpy()
            return stats, td_errors
        
        return stats
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                     dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        next_value = 0
        
        gamma = self.config.get('gamma', 0.99)
        lambda_gae = self.config.get('lambda_gae', 0.95)
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * lambda_gae * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def _compute_cvar_target(self, rewards: torch.Tensor, alpha: float) -> torch.Tensor:
        """计算CVaR目标"""
        sorted_rewards, _ = torch.sort(rewards)
        cutoff_idx = int(alpha * len(rewards))
        cvar_target = sorted_rewards[:cutoff_idx].mean()
        return torch.full_like(rewards, cvar_target)
    
    def train_episode(self, episode: int, experience_buffer: List[Dict]) -> Dict[str, float]:
        """训练一个episode"""
        
        # 更新课程学习
        self.curriculum.update_episode(episode)
        difficulty_config = self.curriculum.get_current_difficulty()
        
        # 准备训练批次
        if self.config.get('use_per', False):
            # 使用优先经验回放
            experiences, indices, weights = self.per_buffer.sample(
                self.config.get('batch_size', 64)
            )
            if not experiences:
                return {}
            
            batch = self._prepare_batch(experiences)
            weights_tensor = torch.FloatTensor(weights)
            
            stats, td_errors = self.update_with_batch(batch, weights_tensor)
            
            # 更新优先级
            self.per_buffer.update_priorities(indices, td_errors)
            
        else:
            # 标准训练
            batch = self._prepare_batch(experience_buffer)
            stats = self.update_with_batch(batch)
        
        # 更新训练统计
        self.training_stats['episode'] = episode
        for key, value in stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
        
        # 性能监控
        if 'episode_return' in stats:
            self.performance_window.append(stats['episode_return'])
        
        return stats
    
    def _prepare_batch(self, experiences: List[Dict]) -> Dict[str, np.ndarray]:
        """准备训练批次"""
        batch = {
            'states': np.array([exp['state'] for exp in experiences]),
            'actions': np.array([exp['action'] for exp in experiences]),
            'rewards': np.array([exp['reward'] for exp in experiences]),
            'log_probs': np.array([exp['log_prob'] for exp in experiences]),
            'values': np.array([exp['value'] for exp in experiences]),
            'dones': np.array([exp['done'] for exp in experiences])
        }
        return batch
    
    def get_training_summary(self) -> Dict[str, float]:
        """获取训练摘要"""
        summary = {}
        
        # 计算最近性能
        if self.performance_window:
            summary['recent_avg_return'] = np.mean(list(self.performance_window))
            summary['recent_std_return'] = np.std(list(self.performance_window))
        
        # 计算损失趋势
        for key in ['actor_losses', 'critic_losses', 'kl_divergences']:
            if key in self.training_stats and self.training_stats[key]:
                recent_values = self.training_stats[key][-20:]  # 最近20个值
                summary[f'recent_avg_{key[:-2]}'] = np.mean(recent_values)
        
        return summary