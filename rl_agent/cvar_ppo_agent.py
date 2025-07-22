"""
CVaR约束的PPO智能体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from typing import Dict, List, Optional, Tuple
import logging
from utils.logging_utils import statistical_warning, throttled_warning

logger = logging.getLogger(__name__)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 共享特征层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络 (策略网络)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_std = nn.Parameter(torch.ones(action_dim) * 0.1)
        
        # Critic网络 (价值网络)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # CVaR估计网络
        self.cvar_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化网络权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # 特殊初始化actor输出层
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.shared_layers(state)
        
        # Actor输出
        action_mean = self.actor_mean(features)
        action_std = F.softplus(self.actor_std) + 1e-6
        
        # Critic输出
        value = self.critic(features)
        
        # CVaR估计
        cvar_estimate = self.cvar_estimator(features)
        
        return action_mean, action_std, value, cvar_estimate


class CVaRPPOAgent:
    """条件风险价值约束的PPO智能体"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 config: Dict):
        """
        初始化CVaR-PPO智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            config: 智能体配置
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 网络参数
        self.hidden_dim = config.get('hidden_dim', 256)
        self.lr = config.get('learning_rate', 3e-4)
        
        # PPO参数
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        self.lambda_gae = config.get('lambda_gae', 0.95)
        
        # CVaR参数
        self.cvar_alpha = config.get('cvar_alpha', 0.05)
        self.cvar_lambda = config.get('cvar_lambda', 1.0)
        self.cvar_threshold = config.get('cvar_threshold', -0.02)
        
        # 网络初始化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = ActorCriticNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-8)
        
        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        
        # 经验缓存
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'cvar_estimates': []
        }
        
        # 数值稳定性统计
        self.numerical_issues = {
            'nan_states': 0,
            'nan_actions': 0, 
            'nan_losses': 0,
            'nan_gradients': 0
        }
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        根据状态获取动作
        
        Args:
            state: 当前状态
            deterministic: 是否确定性动作
            
        Returns:
            action: 动作
            log_prob: 对数概率
            value: 状态价值
        """
        # 检查输入状态是否包含NaN
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            self.numerical_issues['nan_states'] += 1
            
            # 使用统计性日志
            statistical_warning(
                logger,
                "输入状态数值异常",
                f"第{self.numerical_issues['nan_states']}次状态NaN/Inf检测",
                report_interval=50  # 每50次报告一次
            )
            
            state = np.zeros_like(state)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, action_std, value, cvar_estimate = self.network(state_tensor)
            
            # 检查网络输出是否包含NaN
            if torch.any(torch.isnan(action_mean)) or torch.any(torch.isnan(action_std)):
                logger.error("网络输出包含NaN值，重新初始化网络")
                self.network = ActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
                action_mean, action_std, value, cvar_estimate = self.network(state_tensor)
            
            # 确保action_std不会太小
            action_std = torch.clamp(action_std, min=1e-6, max=1.0)
            
            if deterministic:
                action = action_mean
                log_prob = torch.tensor(0.0)
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                
                # 检查采样结果
                if torch.any(torch.isnan(action)) or torch.any(torch.isnan(log_prob)):
                    self.numerical_issues['nan_actions'] += 1
                    
                    # 使用限制器控制日志频率
                    throttled_warning(
                        logger,
                        "动作采样产生NaN，使用确定性动作",
                        "nan_action_sampling",
                        min_interval=30.0,  # 30秒间隔
                        max_per_minute=1    # 每分钟最多1次
                    )
                    
                    action = action_mean
                    log_prob = torch.tensor(0.0)
        
        return (action.cpu().numpy().flatten(), 
                log_prob.cpu().item(), 
                value.cpu().item(),
                cvar_estimate.cpu().item())
    
    def store_transition(self, 
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        value: float,
                        log_prob: float,
                        done: bool,
                        cvar_estimate: float):
        """存储转移经验"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
        self.memory['cvar_estimates'].append(cvar_estimate)
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.memory['states']) == 0:
            return {}
            
        # 转换为张量
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.memory['actions'])).to(self.device)
        rewards = torch.FloatTensor(self.memory['rewards']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        dones = torch.FloatTensor(self.memory['dones']).to(self.device)
        cvar_estimates = torch.FloatTensor(self.memory['cvar_estimates']).to(self.device)
        
        # 计算GAE优势函数
        advantages, returns = self._compute_gae(rewards, old_values, dones)
        
        # 计算CVaR约束
        cvar_target = self._compute_cvar_target(rewards)
        
        # 数据集
        dataset = torch.utils.data.TensorDataset(
            states, actions, old_log_probs, advantages, returns, cvar_target
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # PPO更新
        total_losses = []
        for epoch in range(self.ppo_epochs):
            epoch_losses = []
            for batch in dataloader:
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_cvar_target = batch
                
                # 前向传播
                action_mean, action_std, values, cvar_pred = self.network(batch_states)
                
                # 检查网络输出是否包含NaN
                if torch.any(torch.isnan(action_mean)) or torch.any(torch.isnan(action_std)):
                    logger.error("训练中网络输出包含NaN，跳过此批次")
                    continue
                
                # 确保action_std不会太小或太大
                action_std = torch.clamp(action_std, min=1e-6, max=1.0)
                
                # 计算新的对数概率
                dist = Normal(action_mean, action_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # 计算损失
                loss = self._compute_loss(
                    new_log_probs, batch_old_log_probs, batch_advantages,
                    values, batch_returns, cvar_pred, batch_cvar_target
                )
                
                # 检查损失是否为NaN
                if torch.isnan(loss):
                    self.numerical_issues['nan_losses'] += 1
                    
                    # 使用统计性日志
                    statistical_warning(
                        logger,
                        "训练损失数值异常",
                        f"第{self.numerical_issues['nan_losses']}次损失NaN",
                        report_interval=20  # 每20次报告一次
                    )
                    
                    continue
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 检查梯度是否包含NaN
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                if torch.isnan(grad_norm):
                    self.numerical_issues['nan_gradients'] += 1
                    
                    # 使用限制器控制日志频率
                    throttled_warning(
                        logger,
                        "梯度包含NaN，跳过此次更新",
                        "nan_gradients",
                        min_interval=20.0,  # 20秒间隔  
                        max_per_minute=2    # 每分钟最多2次
                    )
                    
                    continue
                
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            total_losses.extend(epoch_losses)
        
        # 更新学习率
        self.scheduler.step()
        
        # 清空记忆
        self._clear_memory()
        
        # 检查网络健康状态
        self._check_network_health()
        
        return {
            'total_loss': np.mean(total_losses) if total_losses else 0.0,
            'avg_cvar_estimate': torch.mean(cvar_estimates).item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算广义优势估计(GAE)"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def _compute_cvar_target(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算CVaR目标值"""
        # 计算分位数
        var_quantile = torch.quantile(rewards, self.cvar_alpha)
        
        # 计算CVaR
        tail_rewards = rewards[rewards <= var_quantile]
        if len(tail_rewards) > 0:
            cvar = torch.mean(tail_rewards)
        else:
            cvar = var_quantile
            
        return cvar.repeat(len(rewards))
    
    def _compute_loss(self, 
                     new_log_probs: torch.Tensor,
                     old_log_probs: torch.Tensor,
                     advantages: torch.Tensor,
                     values: torch.Tensor,
                     returns: torch.Tensor,
                     cvar_pred: torch.Tensor,
                     cvar_target: torch.Tensor) -> torch.Tensor:
        """计算总损失函数"""
        # PPO策略损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值函数损失
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # CVaR约束损失
        cvar_loss = F.mse_loss(cvar_pred.squeeze(), cvar_target)
        
        # CVaR惩罚项
        cvar_penalty = F.relu(cvar_pred.squeeze().mean() - self.cvar_threshold)
        
        # 总损失
        total_loss = (
            policy_loss + 
            0.5 * value_loss + 
            self.cvar_lambda * cvar_loss +
            2.0 * cvar_penalty  # 强化CVaR约束
        )
        
        return total_loss
    
    def _clear_memory(self):
        """清空经验缓存"""
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'cvar_estimates': []
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filepath)
        
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def _check_network_health(self):
        """检查网络健康状态"""
        for name, param in self.network.named_parameters():
            if torch.any(torch.isnan(param)):
                logger.error(f"网络参数 {name} 包含NaN值，重新初始化网络")
                self.network = ActorCriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-8)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
                break
    
    def get_numerical_issues_summary(self) -> str:
        """获取数值稳定性问题统计"""
        total_issues = sum(self.numerical_issues.values())
        if total_issues == 0:
            return "训练过程中无数值稳定性问题"
        
        summary_lines = [f"数值稳定性问题统计 (总计: {total_issues})"]
        for issue_type, count in self.numerical_issues.items():
            if count > 0:
                issue_name = {
                    'nan_states': 'NaN状态输入',
                    'nan_actions': 'NaN动作采样', 
                    'nan_losses': 'NaN训练损失',
                    'nan_gradients': 'NaN梯度'
                }.get(issue_type, issue_type)
                percentage = (count / total_issues) * 100
                summary_lines.append(f"  - {issue_name}: {count}次 ({percentage:.1f}%)")
        
        return '\n'.join(summary_lines)
        
    def reset_numerical_issues_counts(self):
        """重置数值稳定性问题计数"""
        for key in self.numerical_issues:
            self.numerical_issues[key] = 0