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
from utils.memory_optimizer import (
    MemoryMonitor, GradientAccumulator, ModelParallelWrapper,
    MemoryEfficientBuffer, optimize_tensor_memory, memory_profiler,
    clear_gpu_cache, force_garbage_collection
)

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

    def actor_parameters(self):
        """获取Actor网络参数"""
        actor_params = []
        # 共享层参数（Actor和Critic都需要）
        actor_params.extend(self.shared_layers.parameters())
        # Actor特有参数
        actor_params.extend(self.actor_mean.parameters())
        actor_params.append(self.actor_std)
        return actor_params

    def critic_parameters(self):
        """获取Critic网络参数"""
        critic_params = []
        # 共享层参数（Actor和Critic都需要）
        critic_params.extend(self.shared_layers.parameters())
        # Critic特有参数
        critic_params.extend(self.critic.parameters())
        critic_params.extend(self.cvar_estimator.parameters())
        return critic_params


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

        # O2O学习参数
        self.actor_lr = config.get('actor_lr', 3e-4)
        self.critic_lr = config.get('critic_lr', 1e-3)
        self.use_split_optimizers = config.get('use_split_optimizers', False)

        # 信任域约束参数
        self.trust_region_beta = config.get('trust_region_beta', 1.0)
        self.beta_decay = config.get('beta_decay', 0.99)
        self.kl_target = config.get('kl_target', 0.01)
        self.kl_threshold = config.get('kl_threshold', 0.05)
        self.use_trust_region = config.get('use_trust_region', False)
        
        # 策略发散检测
        self.policy_divergence_threshold = config.get('policy_divergence_threshold', 0.1)
        self.max_kl_violations = config.get('max_kl_violations', 3)
        self.kl_violation_count = 0
        
        # 保存旧策略参数用于KL散度计算
        self.old_policy_params = None

        # 网络初始化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = ActorCriticNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)
        
        # 内存优化和并行计算
        self.memory_monitor = MemoryMonitor(check_interval=2.0, alert_threshold=0.8)
        self.use_gradient_accumulation = config.get('use_gradient_accumulation', False)
        self.accumulation_steps = config.get('accumulation_steps', 4)
        self.use_model_parallel = config.get('use_model_parallel', False)
        self.use_mixed_precision = config.get('use_mixed_precision', torch.cuda.is_available())
        
        # 梯度累积器
        if self.use_gradient_accumulation:
            self.gradient_accumulator = GradientAccumulator(
                self.network, self.accumulation_steps
            )
        
        # 模型并行包装
        if self.use_model_parallel and torch.cuda.device_count() > 1:
            self.network = ModelParallelWrapper(self.network)
            logger.info(f"启用模型并行，使用 {torch.cuda.device_count()} 个GPU")
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # 内存优化的经验缓存
        self.memory_efficient_buffer = MemoryEfficientBuffer(
            capacity=config.get('memory_buffer_capacity', 10000),
            compress_threshold=config.get('compress_threshold', 5000)
        )
        
        # 初始化优化器（默认使用统一优化器）
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, eps=1e-8)
        self.actor_optimizer = None
        self.critic_optimizer = None

        # 添加学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        self.actor_scheduler = None
        self.critic_scheduler = None

        # 参数冻结状态跟踪
        self.actor_frozen = False
        self.critic_frozen = False

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
        
        # 启动内存监控
        if config.get('enable_memory_monitoring', True):
            self.memory_monitor.start_monitoring()

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
        """存储转移经验（内存优化版本）"""
        # 使用内存优化的缓冲区
        transition = {
            'state': optimize_tensor_memory(torch.from_numpy(state).float()),
            'action': optimize_tensor_memory(torch.from_numpy(action).float()),
            'reward': reward,
            'value': value,
            'log_prob': log_prob,
            'done': done,
            'cvar_estimate': cvar_estimate
        }
        
        self.memory_efficient_buffer.add(transition)
        
        # 同时保持原有格式以兼容现有代码
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
        self.memory['cvar_estimates'].append(cvar_estimate)
        
        # 定期清理GPU缓存
        if len(self.memory['states']) % 100 == 0:
            clear_gpu_cache()

    def update(self) -> Dict[str, float]:
        """更新网络（标准PPO更新，不使用重要性权重）"""
        return self.update_with_importance_weights(importance_weights=None)

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
        ratio = torch.exp(new_log_probs - old_log_probs + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值函数损失 - 确保维度匹配
        values_flat = values.squeeze()
        if values_flat.dim() == 0:
            values_flat = values_flat.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        value_loss = F.mse_loss(values_flat, returns)

        # CVaR约束损失 - 确保维度匹配
        cvar_pred_flat = cvar_pred.squeeze()
        if cvar_pred_flat.dim() == 0:
            cvar_pred_flat = cvar_pred_flat.unsqueeze(0)
        if cvar_target.dim() == 0:
            cvar_target = cvar_target.unsqueeze(0)
        cvar_loss = F.mse_loss(cvar_pred_flat, cvar_target)

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
                
                # 重新初始化优化器
                if self.use_split_optimizers:
                    self.split_optimizers()
                else:
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

    def split_optimizers(self):
        """
        分离Actor和Critic优化器，支持O2O学习
        
        Returns:
            Dict: 包含优化器信息的字典
        """
        if self.use_split_optimizers:
            logger.info("优化器已经分离，跳过重复操作")
            return {
                'status': 'already_split',
                'actor_lr': self.actor_lr,
                'critic_lr': self.critic_lr
            }

        # 创建分离的优化器
        self.actor_optimizer = torch.optim.Adam(
            self.network.actor_parameters(),
            lr=self.actor_lr,
            eps=1e-8
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.network.critic_parameters(),
            lr=self.critic_lr,
            eps=1e-8
        )

        # 创建分离的学习率调度器
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, 
            step_size=100, 
            gamma=0.95
        )
        
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(
            self.critic_optimizer, 
            step_size=100, 
            gamma=0.95
        )

        # 更新状态
        self.use_split_optimizers = True
        
        logger.info(f"优化器分离完成 - Actor LR: {self.actor_lr}, Critic LR: {self.critic_lr}")
        
        return {
            'status': 'split_success',
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'actor_params': len(list(self.network.actor_parameters())),
            'critic_params': len(list(self.network.critic_parameters()))
        }

    def freeze_actor(self):
        """
        冻结Actor参数，用于热身微调阶段
        """
        if self.actor_frozen:
            logger.info("Actor参数已经冻结")
            return

        for param in self.network.actor_parameters():
            param.requires_grad = False
            
        self.actor_frozen = True
        logger.info("Actor参数已冻结")

    def unfreeze_actor(self):
        """
        解冻Actor参数，恢复正常训练
        """
        if not self.actor_frozen:
            logger.info("Actor参数未冻结")
            return

        for param in self.network.actor_parameters():
            param.requires_grad = True
            
        self.actor_frozen = False
        logger.info("Actor参数已解冻")

    def freeze_critic(self):
        """
        冻结Critic参数
        """
        if self.critic_frozen:
            logger.info("Critic参数已经冻结")
            return

        for param in self.network.critic_parameters():
            param.requires_grad = False
            
        self.critic_frozen = True
        logger.info("Critic参数已冻结")

    def unfreeze_critic(self):
        """
        解冻Critic参数
        """
        if not self.critic_frozen:
            logger.info("Critic参数未冻结")
            return

        for param in self.network.critic_parameters():
            param.requires_grad = True
            
        self.critic_frozen = False
        logger.info("Critic参数已解冻")

    def get_optimizer_states(self) -> Dict:
        """
        获取优化器状态，用于保存和恢复
        
        Returns:
            Dict: 优化器状态字典
        """
        states = {
            'use_split_optimizers': self.use_split_optimizers,
            'actor_frozen': self.actor_frozen,
            'critic_frozen': self.critic_frozen
        }

        if self.use_split_optimizers:
            if self.actor_optimizer is not None:
                states['actor_optimizer'] = self.actor_optimizer.state_dict()
            if self.critic_optimizer is not None:
                states['critic_optimizer'] = self.critic_optimizer.state_dict()
            if self.actor_scheduler is not None:
                states['actor_scheduler'] = self.actor_scheduler.state_dict()
            if self.critic_scheduler is not None:
                states['critic_scheduler'] = self.critic_scheduler.state_dict()
        else:
            states['unified_optimizer'] = self.optimizer.state_dict()
            states['unified_scheduler'] = self.scheduler.state_dict()

        return states

    def load_optimizer_states(self, states: Dict):
        """
        加载优化器状态
        
        Args:
            states: 优化器状态字典
        """
        self.use_split_optimizers = states.get('use_split_optimizers', False)
        self.actor_frozen = states.get('actor_frozen', False)
        self.critic_frozen = states.get('critic_frozen', False)

        if self.use_split_optimizers:
            # 确保分离优化器已创建
            if self.actor_optimizer is None or self.critic_optimizer is None:
                self.split_optimizers()
            
            # 加载分离优化器状态
            if 'actor_optimizer' in states and self.actor_optimizer is not None:
                self.actor_optimizer.load_state_dict(states['actor_optimizer'])
            if 'critic_optimizer' in states and self.critic_optimizer is not None:
                self.critic_optimizer.load_state_dict(states['critic_optimizer'])
            if 'actor_scheduler' in states and self.actor_scheduler is not None:
                self.actor_scheduler.load_state_dict(states['actor_scheduler'])
            if 'critic_scheduler' in states and self.critic_scheduler is not None:
                self.critic_scheduler.load_state_dict(states['critic_scheduler'])
        else:
            # 加载统一优化器状态
            if 'unified_optimizer' in states:
                self.optimizer.load_state_dict(states['unified_optimizer'])
            if 'unified_scheduler' in states:
                self.scheduler.load_state_dict(states['unified_scheduler'])

        # 应用冻结状态
        if self.actor_frozen:
            self.freeze_actor()
        if self.critic_frozen:
            self.freeze_critic()

        logger.info(f"优化器状态加载完成 - 分离模式: {self.use_split_optimizers}, "
                   f"Actor冻结: {self.actor_frozen}, Critic冻结: {self.critic_frozen}")

    def save_model_with_optimizers(self, filepath: str):
        """
        保存模型和优化器状态
        
        Args:
            filepath: 保存路径
        """
        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_states': self.get_optimizer_states(),
            'config': self.config,
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'trust_region_beta': self.trust_region_beta,
            'use_trust_region': self.use_trust_region,
            'kl_violation_count': self.kl_violation_count,
            'old_policy_params': self.old_policy_params
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"模型和优化器状态已保存到: {filepath}")

    def load_model_with_optimizers(self, filepath: str):
        """
        加载模型和优化器状态
        
        Args:
            filepath: 加载路径
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载网络参数
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        # 更新学习率配置
        if 'actor_lr' in checkpoint:
            self.actor_lr = checkpoint['actor_lr']
        if 'critic_lr' in checkpoint:
            self.critic_lr = checkpoint['critic_lr']
        
        # 加载信任域参数
        if 'trust_region_beta' in checkpoint:
            self.trust_region_beta = checkpoint['trust_region_beta']
        if 'use_trust_region' in checkpoint:
            self.use_trust_region = checkpoint['use_trust_region']
        if 'kl_violation_count' in checkpoint:
            self.kl_violation_count = checkpoint['kl_violation_count']
        if 'old_policy_params' in checkpoint:
            self.old_policy_params = checkpoint['old_policy_params']
        
        # 加载优化器状态
        if 'optimizer_states' in checkpoint:
            self.load_optimizer_states(checkpoint['optimizer_states'])
        else:
            # 向后兼容：加载旧格式的优化器状态
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"模型和优化器状态已从 {filepath} 加载")

    def update_with_importance_weights(self, importance_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        使用重要性权重更新网络，支持O2O混合数据训练
        
        Args:
            importance_weights: 重要性权重数组，形状为 (batch_size,)
                               如果为None，则使用标准PPO更新
        
        Returns:
            Dict: 训练统计信息
        """
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

        # 处理重要性权重
        if importance_weights is not None:
            importance_weights = torch.FloatTensor(importance_weights).to(self.device)
            # 权重归一化和数值稳定性保护
            importance_weights = self._normalize_importance_weights(importance_weights)
        else:
            importance_weights = torch.ones(len(states)).to(self.device)

        # 计算GAE优势函数
        advantages, returns = self._compute_gae(rewards, old_values, dones)

        # 计算CVaR约束
        cvar_target = self._compute_cvar_target(rewards)

        # 数据集
        dataset = torch.utils.data.TensorDataset(
            states, actions, old_log_probs, advantages, returns, cvar_target, importance_weights
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # PPO更新
        total_losses = []
        weight_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        for epoch in range(self.ppo_epochs):
            epoch_losses = []
            epoch_weights = []
            
            for batch in dataloader:
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_cvar_target, batch_weights = batch

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

                # 计算加权损失
                loss = self._compute_weighted_loss(
                    new_log_probs, batch_old_log_probs, batch_advantages,
                    values, batch_returns, cvar_pred, batch_cvar_target,
                    batch_weights
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

                # 内存优化的反向传播和优化器更新
                if self.use_gradient_accumulation:
                    # 使用梯度累积
                    with self.gradient_accumulator.accumulate():
                        if self.use_mixed_precision:
                            with torch.cuda.amp.autocast():
                                scaled_loss = loss / self.accumulation_steps
                                self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss = loss / self.accumulation_steps
                            scaled_loss.backward()
                    
                    self.gradient_accumulator.add_loss(loss)
                    
                    # 检查是否应该更新参数
                    if self.gradient_accumulator.should_update():
                        # 检查梯度是否包含NaN
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                        if torch.isnan(grad_norm):
                            self.numerical_issues['nan_gradients'] += 1
                            throttled_warning(
                                logger,
                                "梯度包含NaN，跳过此次更新",
                                "nan_gradients",
                                min_interval=20.0,
                                max_per_minute=2
                            )
                            continue
                        
                        # 更新参数
                        if self.use_split_optimizers:
                            if not self.actor_frozen and self.actor_optimizer is not None:
                                self.gradient_accumulator.update_and_reset(
                                    self.actor_optimizer, self.scaler
                                )
                            if not self.critic_frozen and self.critic_optimizer is not None:
                                self.gradient_accumulator.update_and_reset(
                                    self.critic_optimizer, self.scaler
                                )
                        else:
                            self.gradient_accumulator.update_and_reset(
                                self.optimizer, self.scaler
                            )
                else:
                    # 标准反向传播
                    if self.use_split_optimizers:
                        # 使用分离优化器
                        if not self.actor_frozen and self.actor_optimizer is not None:
                            self.actor_optimizer.zero_grad()
                        if not self.critic_frozen and self.critic_optimizer is not None:
                            self.critic_optimizer.zero_grad()
                        
                        if self.use_mixed_precision:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)

                        # 检查梯度是否包含NaN
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                        if torch.isnan(grad_norm):
                            self.numerical_issues['nan_gradients'] += 1
                            throttled_warning(
                                logger,
                                "梯度包含NaN，跳过此次更新",
                                "nan_gradients",
                                min_interval=20.0,
                                max_per_minute=2
                            )
                            continue

                        # 分别更新Actor和Critic
                        if not self.actor_frozen and self.actor_optimizer is not None:
                            if self.use_mixed_precision:
                                self.scaler.step(self.actor_optimizer)
                            else:
                                self.actor_optimizer.step()
                        if not self.critic_frozen and self.critic_optimizer is not None:
                            if self.use_mixed_precision:
                                self.scaler.step(self.critic_optimizer)
                            else:
                                self.critic_optimizer.step()
                        
                        if self.use_mixed_precision:
                            self.scaler.update()
                    else:
                        # 使用统一优化器
                        self.optimizer.zero_grad()
                        
                        if self.use_mixed_precision:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        # 检查梯度是否包含NaN
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                        if torch.isnan(grad_norm):
                            self.numerical_issues['nan_gradients'] += 1
                            throttled_warning(
                                logger,
                                "梯度包含NaN，跳过此次更新",
                                "nan_gradients",
                                min_interval=20.0,
                                max_per_minute=2
                            )
                            continue

                        if self.use_mixed_precision:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                epoch_losses.append(loss.item())
                epoch_weights.extend(batch_weights.cpu().numpy())

            total_losses.extend(epoch_losses)
            
            # 计算权重统计
            if epoch_weights:
                weight_stats = {
                    'mean': np.mean(epoch_weights),
                    'std': np.std(epoch_weights),
                    'min': np.min(epoch_weights),
                    'max': np.max(epoch_weights)
                }

        # 更新学习率
        if self.use_split_optimizers:
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()
        else:
            self.scheduler.step()

        # 清空记忆
        self._clear_memory()

        # 检查网络健康状态
        self._check_network_health()

        # 准备返回信息
        result = {
            'total_loss': np.mean(total_losses) if total_losses else 0.0,
            'avg_cvar_estimate': torch.mean(cvar_estimates).item(),
            'importance_weight_stats': weight_stats
        }

        if self.use_split_optimizers:
            if self.actor_scheduler is not None:
                result['actor_learning_rate'] = self.actor_scheduler.get_last_lr()[0]
            if self.critic_scheduler is not None:
                result['critic_learning_rate'] = self.critic_scheduler.get_last_lr()[0]
            result['actor_frozen'] = self.actor_frozen
            result['critic_frozen'] = self.critic_frozen
        else:
            result['learning_rate'] = self.scheduler.get_last_lr()[0]

        return result

    def _normalize_importance_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """
        归一化重要性权重并提供数值稳定性保护
        
        Args:
            weights: 原始重要性权重
            
        Returns:
            torch.Tensor: 归一化后的权重
        """
        # 检查权重是否包含NaN或Inf
        if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
            logger.warning("重要性权重包含NaN或Inf，使用均匀权重")
            return torch.ones_like(weights)
        
        # 裁剪极端值
        weights = torch.clamp(weights, min=1e-8, max=100.0)
        
        # 归一化
        weights = weights / torch.mean(weights)
        
        # 再次裁剪以确保稳定性
        weights = torch.clamp(weights, min=0.1, max=10.0)
        
        return weights

    def _compute_weighted_loss(self,
                              new_log_probs: torch.Tensor,
                              old_log_probs: torch.Tensor,
                              advantages: torch.Tensor,
                              values: torch.Tensor,
                              returns: torch.Tensor,
                              cvar_pred: torch.Tensor,
                              cvar_target: torch.Tensor,
                              importance_weights: torch.Tensor) -> torch.Tensor:
        """
        计算加权损失函数，支持重要性权重
        
        Args:
            new_log_probs: 新策略的对数概率
            old_log_probs: 旧策略的对数概率
            advantages: 优势函数
            values: 价值函数预测
            returns: 回报
            cvar_pred: CVaR预测
            cvar_target: CVaR目标
            importance_weights: 重要性权重
            
        Returns:
            torch.Tensor: 总损失
        """
        # PPO策略损失（加权）
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        # 应用重要性权重
        weighted_policy_loss = -torch.mean(importance_weights * torch.min(surr1, surr2))

        # 价值函数损失（加权）- 确保维度匹配
        values_flat = values.squeeze()
        if values_flat.dim() == 0:
            values_flat = values_flat.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        
        value_errors = (values_flat - returns) ** 2
        weighted_value_loss = torch.mean(importance_weights * value_errors)

        # CVaR约束损失（加权）- 确保维度匹配
        cvar_pred_flat = cvar_pred.squeeze()
        if cvar_pred_flat.dim() == 0:
            cvar_pred_flat = cvar_pred_flat.unsqueeze(0)
        if cvar_target.dim() == 0:
            cvar_target = cvar_target.unsqueeze(0)
        
        cvar_errors = (cvar_pred_flat - cvar_target) ** 2
        weighted_cvar_loss = torch.mean(importance_weights * cvar_errors)

        # CVaR惩罚项（不加权，因为这是全局约束）
        cvar_penalty = F.relu(cvar_pred.squeeze().mean() - self.cvar_threshold)

        # 总损失
        total_loss = (
            weighted_policy_loss +
            0.5 * weighted_value_loss +
            self.cvar_lambda * weighted_cvar_loss +
            2.0 * cvar_penalty  # 强化CVaR约束
        )

        return total_loss

    def enable_trust_region(self, beta: float = 1.0):
        """
        启用信任域约束
        
        Args:
            beta: 信任域约束强度
        """
        self.use_trust_region = True
        self.trust_region_beta = beta
        self._save_old_policy_params()
        logger.info(f"信任域约束已启用，beta = {beta}")

    def disable_trust_region(self):
        """禁用信任域约束"""
        self.use_trust_region = False
        self.old_policy_params = None
        logger.info("信任域约束已禁用")

    def _save_old_policy_params(self):
        """保存当前策略参数作为旧策略"""
        self.old_policy_params = {}
        for name, param in self.network.named_parameters():
            if 'actor' in name or 'shared' in name:  # 只保存与策略相关的参数
                self.old_policy_params[name] = param.data.clone()

    def _compute_kl_divergence(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        计算新旧策略之间的KL散度
        
        Args:
            states: 状态批次
            actions: 动作批次
            
        Returns:
            torch.Tensor: KL散度
        """
        if self.old_policy_params is None:
            return torch.tensor(0.0).to(self.device)

        # 当前策略分布
        action_mean, action_std, _, _ = self.network(states)
        current_dist = Normal(action_mean, action_std)

        # 临时保存当前参数
        current_params = {}
        for name, param in self.network.named_parameters():
            if name in self.old_policy_params:
                current_params[name] = param.data.clone()

        # 加载旧策略参数
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if name in self.old_policy_params:
                    param.data.copy_(self.old_policy_params[name])

            # 计算旧策略分布
            old_action_mean, old_action_std, _, _ = self.network(states)
            old_dist = Normal(old_action_mean, old_action_std)

        # 恢复当前参数
        for name, param in self.network.named_parameters():
            if name in current_params:
                param.data.copy_(current_params[name])

        # 计算KL散度
        kl_div = torch.distributions.kl_divergence(current_dist, old_dist).mean()
        
        return kl_div

    def _compute_trust_region_loss(self, 
                                  new_log_probs: torch.Tensor,
                                  old_log_probs: torch.Tensor,
                                  advantages: torch.Tensor,
                                  states: torch.Tensor,
                                  actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算带信任域约束的策略损失
        
        Args:
            new_log_probs: 新策略对数概率
            old_log_probs: 旧策略对数概率
            advantages: 优势函数
            states: 状态
            actions: 动作
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (策略损失, KL散度)
        """
        # 标准PPO损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 计算KL散度
        kl_div = self._compute_kl_divergence(states, actions)

        if self.use_trust_region:
            # 添加信任域约束项
            trust_region_penalty = self.trust_region_beta * kl_div
            policy_loss = policy_loss + trust_region_penalty

        return policy_loss, kl_div

    def _update_trust_region_beta(self, kl_div: float):
        """
        自适应调整信任域参数beta
        
        Args:
            kl_div: 当前KL散度
        """
        if not self.use_trust_region:
            return

        if kl_div > self.kl_threshold:
            # KL散度过大，增加约束强度
            self.trust_region_beta *= 1.5
            self.kl_violation_count += 1
            logger.warning(f"KL散度过大 ({kl_div:.6f} > {self.kl_threshold}), "
                          f"增加信任域约束: beta = {self.trust_region_beta:.4f}")
        elif kl_div < self.kl_target:
            # KL散度过小，减少约束强度
            self.trust_region_beta *= self.beta_decay
            self.trust_region_beta = max(self.trust_region_beta, 0.1)  # 最小值限制
        
        # 检查策略发散
        if self.kl_violation_count >= self.max_kl_violations:
            logger.error(f"策略发散检测：连续{self.kl_violation_count}次KL违规")
            return True  # 表示需要回退
        
        return False

    def _check_policy_divergence(self, kl_div: float) -> bool:
        """
        检查策略是否发散
        
        Args:
            kl_div: KL散度
            
        Returns:
            bool: 是否发散
        """
        if kl_div > self.policy_divergence_threshold:
            logger.error(f"策略发散检测：KL散度 {kl_div:.6f} 超过阈值 {self.policy_divergence_threshold}")
            return True
        return False

    def reset_policy_to_checkpoint(self):
        """
        将策略重置到上一个检查点（旧策略参数）
        """
        if self.old_policy_params is None:
            logger.warning("没有可用的策略检查点")
            return False

        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if name in self.old_policy_params:
                    param.data.copy_(self.old_policy_params[name])

        # 重置违规计数
        self.kl_violation_count = 0
        logger.info("策略已重置到上一个检查点")
        return True

    def update_with_trust_region(self, importance_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        使用信任域约束的更新方法
        
        Args:
            importance_weights: 重要性权重
            
        Returns:
            Dict: 训练统计信息
        """
        if len(self.memory['states']) == 0:
            return {}

        # 保存当前策略作为旧策略
        if self.use_trust_region:
            self._save_old_policy_params()

        # 转换为张量
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.memory['actions'])).to(self.device)
        rewards = torch.FloatTensor(self.memory['rewards']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        dones = torch.FloatTensor(self.memory['dones']).to(self.device)
        cvar_estimates = torch.FloatTensor(self.memory['cvar_estimates']).to(self.device)

        # 处理重要性权重
        if importance_weights is not None:
            importance_weights = torch.FloatTensor(importance_weights).to(self.device)
            importance_weights = self._normalize_importance_weights(importance_weights)
        else:
            importance_weights = torch.ones(len(states)).to(self.device)

        # 计算GAE优势函数
        advantages, returns = self._compute_gae(rewards, old_values, dones)

        # 计算CVaR约束
        cvar_target = self._compute_cvar_target(rewards)

        # 数据集
        dataset = torch.utils.data.TensorDataset(
            states, actions, old_log_probs, advantages, returns, cvar_target, importance_weights
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # PPO更新
        total_losses = []
        kl_divs = []
        weight_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        policy_diverged = False
        
        for epoch in range(self.ppo_epochs):
            epoch_losses = []
            epoch_kls = []
            epoch_weights = []
            
            for batch in dataloader:
                batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_cvar_target, batch_weights = batch

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

                # 计算带信任域约束的策略损失
                if self.use_trust_region:
                    policy_loss, kl_div = self._compute_trust_region_loss(
                        new_log_probs, batch_old_log_probs, batch_advantages,
                        batch_states, batch_actions
                    )
                    epoch_kls.append(kl_div.item())
                    
                    # 检查策略发散
                    if self._check_policy_divergence(kl_div.item()):
                        policy_diverged = True
                        break
                else:
                    # 标准PPO策略损失
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    kl_div = torch.tensor(0.0)

                # 应用重要性权重到策略损失
                if importance_weights is not None:
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    weighted_policy_loss = -torch.mean(batch_weights * torch.min(surr1, surr2))
                    
                    if self.use_trust_region:
                        trust_region_penalty = self.trust_region_beta * kl_div
                        policy_loss = weighted_policy_loss + trust_region_penalty
                    else:
                        policy_loss = weighted_policy_loss

                # 价值函数损失（加权）
                values_flat = values.squeeze()
                if values_flat.dim() == 0:
                    values_flat = values_flat.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                
                value_errors = (values_flat - batch_returns) ** 2
                weighted_value_loss = torch.mean(batch_weights * value_errors)

                # CVaR约束损失（加权）
                cvar_pred_flat = cvar_pred.squeeze()
                if cvar_pred_flat.dim() == 0:
                    cvar_pred_flat = cvar_pred_flat.unsqueeze(0)
                if batch_cvar_target.dim() == 0:
                    batch_cvar_target = batch_cvar_target.unsqueeze(0)
                
                cvar_errors = (cvar_pred_flat - batch_cvar_target) ** 2
                weighted_cvar_loss = torch.mean(batch_weights * cvar_errors)

                # CVaR惩罚项
                cvar_penalty = F.relu(cvar_pred.squeeze().mean() - self.cvar_threshold)

                # 总损失
                total_loss = (
                    policy_loss +
                    0.5 * weighted_value_loss +
                    self.cvar_lambda * weighted_cvar_loss +
                    2.0 * cvar_penalty
                )

                # 检查损失是否为NaN
                if torch.isnan(total_loss):
                    self.numerical_issues['nan_losses'] += 1
                    statistical_warning(
                        logger,
                        "训练损失数值异常",
                        f"第{self.numerical_issues['nan_losses']}次损失NaN",
                        report_interval=20
                    )
                    continue

                # 反向传播和优化器更新
                if self.use_split_optimizers:
                    if not self.actor_frozen and self.actor_optimizer is not None:
                        self.actor_optimizer.zero_grad()
                    if not self.critic_frozen and self.critic_optimizer is not None:
                        self.critic_optimizer.zero_grad()
                    
                    total_loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    if torch.isnan(grad_norm):
                        self.numerical_issues['nan_gradients'] += 1
                        throttled_warning(
                            logger,
                            "梯度包含NaN，跳过此次更新",
                            "nan_gradients",
                            min_interval=20.0,
                            max_per_minute=2
                        )
                        continue

                    if not self.actor_frozen and self.actor_optimizer is not None:
                        self.actor_optimizer.step()
                    if not self.critic_frozen and self.critic_optimizer is not None:
                        self.critic_optimizer.step()
                else:
                    self.optimizer.zero_grad()
                    total_loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    if torch.isnan(grad_norm):
                        self.numerical_issues['nan_gradients'] += 1
                        throttled_warning(
                            logger,
                            "梯度包含NaN，跳过此次更新",
                            "nan_gradients",
                            min_interval=20.0,
                            max_per_minute=2
                        )
                        continue

                    self.optimizer.step()

                epoch_losses.append(total_loss.item())
                epoch_weights.extend(batch_weights.cpu().numpy())

            if policy_diverged:
                logger.error("检测到策略发散，停止训练并回退")
                self.reset_policy_to_checkpoint()
                break

            total_losses.extend(epoch_losses)
            kl_divs.extend(epoch_kls)
            
            # 计算权重统计
            if epoch_weights:
                weight_stats = {
                    'mean': np.mean(epoch_weights),
                    'std': np.std(epoch_weights),
                    'min': np.min(epoch_weights),
                    'max': np.max(epoch_weights)
                }

        # 更新信任域参数
        if self.use_trust_region and kl_divs:
            avg_kl = np.mean(kl_divs)
            need_rollback = self._update_trust_region_beta(avg_kl)
            if need_rollback:
                self.reset_policy_to_checkpoint()

        # 更新学习率
        if self.use_split_optimizers:
            if self.actor_scheduler is not None:
                self.actor_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()
        else:
            self.scheduler.step()

        # 清空记忆
        self._clear_memory()

        # 检查网络健康状态
        self._check_network_health()

        # 准备返回信息
        result = {
            'total_loss': np.mean(total_losses) if total_losses else 0.0,
            'avg_cvar_estimate': torch.mean(cvar_estimates).item(),
            'importance_weight_stats': weight_stats,
            'policy_diverged': policy_diverged
        }

        if self.use_trust_region:
            result.update({
                'avg_kl_divergence': np.mean(kl_divs) if kl_divs else 0.0,
                'trust_region_beta': self.trust_region_beta,
                'kl_violation_count': self.kl_violation_count
            })

        if self.use_split_optimizers:
            if self.actor_scheduler is not None:
                result['actor_learning_rate'] = self.actor_scheduler.get_last_lr()[0]
            if self.critic_scheduler is not None:
                result['critic_learning_rate'] = self.critic_scheduler.get_last_lr()[0]
            result['actor_frozen'] = self.actor_frozen
            result['critic_frozen'] = self.critic_frozen
        else:
            result['learning_rate'] = self.scheduler.get_last_lr()[0]

        return result
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存使用统计"""
        stats = self.memory_monitor.get_memory_stats()
        return {
            'total_memory_gb': stats.total_memory,
            'used_memory_gb': stats.used_memory,
            'memory_percent': stats.memory_percent,
            'gpu_memory_used_gb': stats.gpu_memory_used or 0.0,
            'gpu_memory_total_gb': stats.gpu_memory_total or 0.0,
            'buffer_size': len(self.memory_efficient_buffer)
        }
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        # 清理GPU缓存
        clear_gpu_cache()
        
        # 强制垃圾回收
        force_garbage_collection()
        
        # 清理内存缓冲区
        self.memory_efficient_buffer.clear()
        
        # 重置数值问题计数
        self.reset_numerical_issues_counts()
        
        logger.info("内存优化完成")
    
    def enable_memory_profiling(self, enabled: bool = True):
        """启用/禁用内存分析"""
        if enabled:
            self.memory_monitor.start_monitoring()
        else:
            self.memory_monitor.stop_monitoring()
    
    def get_peak_memory_usage(self) -> Optional[Dict[str, float]]:
        """获取峰值内存使用"""
        peak_stats = self.memory_monitor.get_peak_memory_usage()
        if peak_stats:
            return {
                'peak_memory_percent': peak_stats.memory_percent,
                'peak_used_memory_gb': peak_stats.used_memory,
                'peak_gpu_memory_gb': peak_stats.gpu_memory_used or 0.0
            }
        return None
    
    def update_with_memory_profiling(self, importance_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """带内存分析的更新方法"""
        with memory_profiler("CVaR-PPO Update"):
            result = self.update_with_importance_weights(importance_weights)
            
            # 添加内存统计到结果
            memory_stats = self.get_memory_stats()
            result.update({
                'memory_usage_percent': memory_stats['memory_percent'],
                'gpu_memory_usage_gb': memory_stats['gpu_memory_used_gb']
            })
            
            return result