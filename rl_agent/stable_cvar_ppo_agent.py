"""
数值稳定的CVaR-PPO智能体
解决NaN梯度问题，大幅提升训练稳定性和收益能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple
import logging
try:
    from .numerical_stability_fixes import (
        compute_stable_ratio, stable_mse_loss, compute_stable_advantages,
        apply_gradient_clipping, initialize_network_weights, create_stable_optimizer,
        check_tensor_health, StableLossFunction
    )
except ImportError:
    from rl_agent.numerical_stability_fixes import (
        compute_stable_ratio, stable_mse_loss, compute_stable_advantages,
        apply_gradient_clipping, initialize_network_weights, create_stable_optimizer,
        check_tensor_health, StableLossFunction
    )

logger = logging.getLogger(__name__)


class StableActorCriticNetwork(nn.Module):
    """数值稳定的Actor-Critic网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # 使用更深的网络提升学习能力
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加LayerNorm提升稳定性
            nn.ReLU(),
            nn.Dropout(0.1),  # 轻微dropout防止过拟合
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            
            # 添加残差连接层
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Actor网络 - 输出动作均值和标准差
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # 使用网络参数而非固定参数来学习标准差
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)

        # Critic网络 
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # CVaR估计网络
        self.cvar_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 改进的权重初始化
        initialize_network_weights(self, gain=1.0)
        
        # Actor输出层特殊初始化
        nn.init.xavier_uniform_(self.actor_mean.weight, gain=0.1)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.xavier_uniform_(self.actor_log_std.weight, gain=0.1)
        nn.init.constant_(self.actor_log_std.bias, -1.0)  # 初始化为较小的标准差

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 输入数值稳定性检查
        if not check_tensor_health(state, "input_state"):
            state = torch.clamp(state, min=-10.0, max=10.0)
        
        features = self.shared_layers(state)
        
        # 添加残差连接
        if features.shape[-1] == state.shape[-1]:
            features = features + state
        
        # Actor输出
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std(features)
        
        # 稳定的标准差计算
        action_std = torch.exp(torch.clamp(action_log_std, min=-5.0, max=2.0)) + 1e-6
        
        # Critic输出
        value = self.critic(features)
        
        # CVaR估计
        cvar_estimate = self.cvar_estimator(features)
        
        # 输出数值稳定性检查
        action_mean = torch.clamp(action_mean, min=-5.0, max=5.0)
        value = torch.clamp(value, min=-100.0, max=100.0)
        cvar_estimate = torch.clamp(cvar_estimate, min=-1.0, max=1.0)
        
        return action_mean, action_std, value, cvar_estimate


class StableCVaRPPOAgent:
    """数值稳定的CVaR-PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 超参数配置
        self.lr = config.get('learning_rate', 1e-4)  # 降低学习率提升稳定性
        self.gamma = config.get('gamma', 0.99)
        self.lambda_gae = config.get('lambda_gae', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 8)  # 增加训练轮次
        self.batch_size = config.get('batch_size', 64)
        self.cvar_alpha = config.get('cvar_alpha', 0.05)
        self.cvar_lambda = config.get('cvar_lambda', 0.1)
        self.cvar_threshold = config.get('cvar_threshold', -0.02)
        
        # 网络初始化
        hidden_dim = config.get('hidden_dim', 256)
        self.network = StableActorCriticNetwork(state_dim, action_dim, hidden_dim)
        
        # 稳定的优化器
        self.optimizer = create_stable_optimizer(self.network, lr=self.lr)
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.8, patience=20
        )
        
        # 损失函数
        self.loss_function = StableLossFunction(
            self.clip_epsilon, self.cvar_lambda, self.cvar_threshold
        )
        
        # 经验缓存
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': [],
            'cvar_estimates': []
        }
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'episodes': 0,
            'nan_gradients': 0,
            'nan_losses': 0,
            'gradient_norms': [],
            'loss_history': []
        }
        
        logger.info(f"初始化稳定CVaR-PPO智能体 - 状态维度: {state_dim}, 动作维度: {action_dim}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, torch.Tensor]:
        """获取动作"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        
        self.network.eval()
        with torch.no_grad():
            action_mean, action_std, value, cvar_estimate = self.network(state)
            
            if deterministic:
                action = action_mean
                log_prob = None
            else:
                # 创建正态分布
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # 数值稳定性检查
                if torch.isnan(log_prob).any():
                    log_prob = torch.zeros_like(log_prob)
        
        self.network.train()
        
        action_np = action.squeeze().numpy()
        log_prob_value = log_prob.item() if log_prob is not None else 0.0
        
        return action_np, log_prob_value, value.squeeze()
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        log_prob: float, value: torch.Tensor, done: bool):
        """存储经验"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['log_probs'].append(log_prob)
        self.memory['values'].append(value.item())
        self.memory['dones'].append(done)
        
        # 计算CVaR估计
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            _, _, _, cvar_est = self.network(state_tensor)
            self.memory['cvar_estimates'].append(cvar_est.item())
    
    def update(self) -> Dict[str, float]:
        """更新网络参数"""
        if len(self.memory['states']) < self.batch_size:
            return {'message': 'Not enough samples for update'}
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.memory['states']))
        actions = torch.FloatTensor(np.array(self.memory['actions']))
        rewards = torch.FloatTensor(self.memory['rewards'])
        old_log_probs = torch.FloatTensor(self.memory['log_probs'])
        values = torch.FloatTensor(self.memory['values'])
        dones = torch.BoolTensor(self.memory['dones'])
        
        # 计算returns和advantages
        returns = self._compute_returns(rewards, values, dones)
        advantages = compute_stable_advantages(rewards, values, self.gamma, self.lambda_gae)
        
        # 计算CVaR目标
        returns_np = returns.numpy()
        cvar_target = torch.FloatTensor([
            np.percentile(returns_np, self.cvar_alpha * 100) 
            for _ in range(len(returns_np))
        ])
        
        # 训练统计
        total_losses = []
        policy_losses = []
        value_losses = []
        cvar_losses = []
        
        # 多轮训练
        for epoch in range(self.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_cvar_target = cvar_target[batch_indices]
                
                # 前向传播
                action_mean, action_std, batch_values, batch_cvar_pred = self.network(batch_states)
                
                # 计算新的log概率
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # 计算损失
                total_loss = self.loss_function.compute_total_loss(
                    new_log_probs, batch_old_log_probs, batch_advantages,
                    batch_values, batch_returns, batch_cvar_pred, batch_cvar_target
                )
                
                # 检查损失是否健康
                if torch.isnan(total_loss):
                    self.stats['nan_losses'] += 1
                    logger.warning(f"检测到NaN损失，跳过此次更新 (第{self.stats['nan_losses']}次)")
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 应用梯度裁剪
                grad_norm = apply_gradient_clipping(self.network, max_norm=1.0)
                
                # 检查梯度健康状况
                has_nan_grad = any(torch.isnan(p.grad).any() for p in self.network.parameters() 
                                 if p.grad is not None)
                
                if has_nan_grad:
                    self.stats['nan_gradients'] += 1
                    logger.warning(f"检测到NaN梯度，跳过此次更新 (第{self.stats['nan_gradients']}次)")
                    continue
                
                # 更新参数
                self.optimizer.step()
                
                # 记录统计信息
                total_losses.append(total_loss.item())
                self.stats['gradient_norms'].append(grad_norm)
        
        # 更新学习率
        avg_loss = np.mean(total_losses) if total_losses else 0.0
        self.scheduler.step(-avg_loss)  # 负值因为我们想要最小化损失
        
        # 清空缓存
        self._clear_memory()
        
        # 更新统计
        self.stats['total_steps'] += len(states)
        self.stats['episodes'] += 1
        self.stats['loss_history'].append(avg_loss)
        
        return {
            'total_loss': avg_loss,
            'policy_loss': np.mean(policy_losses) if policy_losses else 0.0,
            'value_loss': np.mean(value_losses) if value_losses else 0.0,
            'cvar_loss': np.mean(cvar_losses) if cvar_losses else 0.0,
            'gradient_norm': np.mean(self.stats['gradient_norms'][-10:]) if self.stats['gradient_norms'] else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'nan_gradients': self.stats['nan_gradients'],
            'nan_losses': self.stats['nan_losses']
        }
    
    def _compute_returns(self, rewards: torch.Tensor, values: torch.Tensor, 
                        dones: torch.BoolTensor) -> torch.Tensor:
        """计算折扣回报"""
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0.0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _clear_memory(self):
        """清空经验缓存"""
        for key in self.memory:
            self.memory[key] = []
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'stats': self.stats
        }, filepath)
        logger.info(f"模型已保存至: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'stats' in checkpoint:
            self.stats.update(checkpoint['stats'])
        logger.info(f"模型已从 {filepath} 加载")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'total_steps': self.stats['total_steps'],
            'episodes': self.stats['episodes'],
            'nan_gradients_rate': self.stats['nan_gradients'] / max(1, self.stats['total_steps']),
            'nan_losses_rate': self.stats['nan_losses'] / max(1, self.stats['total_steps']),
            'avg_gradient_norm': np.mean(self.stats['gradient_norms']) if self.stats['gradient_norms'] else 0.0,
            'recent_loss': np.mean(self.stats['loss_history'][-10:]) if self.stats['loss_history'] else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }