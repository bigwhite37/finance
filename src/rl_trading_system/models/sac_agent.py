"""
SAC (Soft Actor-Critic) 智能体实现
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import json
from collections import deque

from .actor_network import Actor, ActorConfig
from .critic_network import CriticWithTargetNetwork, CriticConfig
from .replay_buffer import (
    BaseReplayBuffer, 
    ReplayBuffer, 
    PrioritizedReplayBuffer, 
    Experience, 
    ReplayBufferConfig,
    create_replay_buffer
)
from .transformer import TimeSeriesTransformer, TransformerConfig


@dataclass
class SACConfig:
    """SAC智能体配置"""
    # 网络架构
    state_dim: int = 256
    action_dim: int = 100
    hidden_dim: int = 512
    n_layers: int = 3
    activation: str = 'relu'
    dropout: float = 0.1
    
    # 学习率
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    
    # SAC算法参数
    gamma: float = 0.99  # 折扣因子
    tau: float = 0.005   # 软更新系数
    alpha: float = 0.2   # 初始温度参数
    target_entropy: Optional[float] = None  # 目标熵，默认为-action_dim
    auto_alpha: bool = True  # 是否自动调整温度参数
    
    # 训练参数
    batch_size: int = 256
    buffer_capacity: int = 1000000
    learning_starts: int = 100   # 开始学习的最小经验数（降低以便测试时更早开始学习）
    train_freq: int = 1  # 训练频率
    target_update_freq: int = 1  # 目标网络更新频率
    gradient_steps: int = 1  # 每次更新的梯度步数
    
    # 优先级回放参数
    use_prioritized_replay: bool = False
    alpha_replay: float = 0.6
    beta_replay: float = 0.4
    beta_increment: float = 0.001
    
    # 设备和其他
    device: str = 'cpu'
    seed: Optional[int] = None
    
    # Transformer配置
    use_transformer: bool = True  # 是否使用Transformer编码观察
    transformer_config: Optional[TransformerConfig] = None
    
    # 日志和保存
    log_interval: int = 1000
    save_interval: int = 10000
    
    def __post_init__(self):
        """后处理配置"""
        if self.target_entropy is None:
            self.target_entropy = -float(self.action_dim)


class SACAgent(nn.Module):
    """
    SAC (Soft Actor-Critic) 智能体
    
    实现完整的SAC算法，包括：
    - Actor网络（策略网络）
    - 双Critic网络（价值网络）
    - 自动温度参数调整
    - 经验回放缓冲区
    - 目标网络软更新
    """
    
    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # 设置随机种子
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        
        # 初始化Transformer（如果启用）
        self.transformer = None
        if config.use_transformer and config.transformer_config is not None:
            self.transformer = TimeSeriesTransformer(config.transformer_config).to(self.device)
        
        # 初始化网络
        self._build_networks()
        
        # 初始化优化器
        self._build_optimizers()
        
        # 初始化回放缓冲区
        self._build_replay_buffer()
        
        # 训练统计
        self.training_step = 0
        self.episode_count = 0
        self.total_env_steps = 0
        
        # 性能统计
        self.training_stats = {
            'actor_loss': deque(maxlen=1000),
            'critic_loss': deque(maxlen=1000),
            'alpha_loss': deque(maxlen=1000),
            'alpha_value': deque(maxlen=1000),
            'q_value': deque(maxlen=1000),
            'policy_entropy': deque(maxlen=1000)
        }
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
    def _build_networks(self):
        """构建神经网络"""
        # Actor网络配置
        actor_config = ActorConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            activation=self.config.activation,
            dropout=self.config.dropout
        )
        
        # Critic网络配置
        critic_config = CriticConfig(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            activation=self.config.activation,
            dropout=self.config.dropout
        )
        
        # 创建网络
        self.actor = Actor(actor_config).to(self.device)
        self.critic = CriticWithTargetNetwork(critic_config).to(self.device)
        
        # 温度参数
        if self.config.auto_alpha:
            self.log_alpha = nn.Parameter(
                torch.log(torch.tensor(self.config.alpha, device=self.device))
            )
        else:
            self.register_buffer(
                'log_alpha', 
                torch.log(torch.tensor(self.config.alpha, device=self.device))
            )
    
    def _build_optimizers(self):
        """构建优化器"""
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=self.config.lr_actor
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.critic.get_parameters(), 
            lr=self.config.lr_critic
        )
        
        if self.config.auto_alpha:
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], 
                lr=self.config.lr_alpha
            )
        else:
            self.alpha_optimizer = None
    
    def _build_replay_buffer(self):
        """构建回放缓冲区"""
        buffer_config = ReplayBufferConfig(
            capacity=self.config.buffer_capacity,
            batch_size=self.config.batch_size,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            device=self.config.device
        )
        
        if self.config.use_prioritized_replay:
            buffer_config.alpha = self.config.alpha_replay
            buffer_config.beta = self.config.beta_replay
            buffer_config.beta_increment = self.config.beta_increment
        
        self.replay_buffer = create_replay_buffer(buffer_config)
    
    @property
    def alpha(self) -> torch.Tensor:
        """当前温度参数"""
        return torch.exp(self.log_alpha)
    
    def preprocess_observation(self, obs):
        """
        预处理观察，准备送入Transformer
        
        Args:
            obs: 字典观察或张量
            
        Returns:
            torch.Tensor: 预处理后的观察张量
        """
        if isinstance(obs, dict):
            # 提取特征数据
            features = obs['features']  # (seq_len, n_stocks, n_features_per_stock)
            
            if isinstance(features, np.ndarray):
                features = torch.from_numpy(features).float()
            
            # 确保特征是正确的维度：(batch_size, seq_len, n_stocks, n_features_per_stock)
            if features.dim() == 3:
                features = features.unsqueeze(0)  # 添加批次维度
            
            return features
        else:
            # 如果不是字典，直接返回
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float()
            return obs
    
    def encode_observation(self, obs, training=False):
        """
        编码观察为低维表示
        
        Args:
            obs: 原始观察
            training: 是否在训练模式（影响梯度计算）
            
        Returns:
            torch.Tensor: 编码后的观察
        """
        if self.transformer is not None:
            # 使用Transformer编码
            processed_obs = self.preprocess_observation(obs)
            
            if training:
                # 训练模式：保持梯度
                encoded = self.transformer(processed_obs)
            else:
                # 推理模式：分离梯度以避免重复反向传播
                with torch.no_grad():
                    encoded = self.transformer(processed_obs)
            
            # 如果编码结果是3D的，需要展平或选择
            if encoded.dim() == 3:
                # 对股票维度进行平均池化，得到 (batch_size, d_model)
                encoded = encoded.mean(dim=1)
            
            # 移除批次维度（如果只有一个样本）
            if encoded.size(0) == 1:
                encoded = encoded.squeeze(0)
                
            return encoded
        else:
            # 没有Transformer时，使用传统的展平方法
            return self._flatten_dict_observation(obs)
    
    def get_action_from_encoded(self, encoded_obs, deterministic: bool = False):
        """
        从编码后的观察生成动作
        
        Args:
            encoded_obs: 编码后的观察
            deterministic: 是否使用确定性策略
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 动作和对数概率
        """
        # 确保是批次格式
        if encoded_obs.dim() == 1:
            encoded_obs = encoded_obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        encoded_obs = encoded_obs.to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(encoded_obs, deterministic=deterministic)
        
        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        
        return action, log_prob

    def _flatten_dict_observation(self, obs):
        """
        将字典观察转换为扁平化的张量
        
        Args:
            obs: 字典观察或张量
            
        Returns:
            torch.Tensor: 扁平化的状态张量
        """
        if isinstance(obs, dict):
            # 处理字典观察
            features = []
            
            # 添加特征数据（展平）
            if 'features' in obs:
                feat = obs['features']
                if isinstance(feat, np.ndarray):
                    feat = torch.from_numpy(feat)
                features.append(feat.flatten())
            
            # 添加持仓信息
            if 'positions' in obs:
                pos = obs['positions'] 
                if isinstance(pos, np.ndarray):
                    pos = torch.from_numpy(pos)
                features.append(pos.flatten())
            
            # 添加市场状态
            if 'market_state' in obs:
                market = obs['market_state']
                if isinstance(market, np.ndarray):
                    market = torch.from_numpy(market)
                features.append(market.flatten())
            
            # 拼接所有特征
            if features:
                return torch.cat(features, dim=0).float()
            else:
                raise ValueError("观察字典中没有找到有效特征")
        else:
            # 如果不是字典，直接返回张量
            if isinstance(obs, np.ndarray):
                return torch.from_numpy(obs).float()
            elif isinstance(obs, torch.Tensor):
                return obs.float()
            else:
                raise ValueError(f"不支持的观察类型: {type(obs)}")
    
    def get_action(self, 
                   state, 
                   deterministic: bool = False,
                   return_log_prob: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        获取动作
        
        Args:
            state: 状态张量或字典观察
            deterministic: 是否使用确定性策略
            return_log_prob: 是否返回对数概率
            
        Returns:
            action: 动作张量
            log_prob: 对数概率（如果return_log_prob=True）  
        """
        # 处理不同类型的观察 - 使用新的编码方法（推理模式）
        state_tensor = self.encode_observation(state, training=False)
        
        # 确保状态是批次格式
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        state_tensor = state_tensor.to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state_tensor, deterministic=deterministic)
        
        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        
        if return_log_prob:
            return action, log_prob
        else:
            return action
    
    def add_experience(self, experience: Experience) -> None:
        """
        添加经验到回放缓冲区
        
        Args:
            experience: 经验数据
        """
        # 确保张量在正确设备上
        experience.state = experience.state.to('cpu')
        experience.action = experience.action.to('cpu')
        experience.next_state = experience.next_state.to('cpu')
        
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # 计算TD误差作为优先级
            with torch.no_grad():
                priority = self._compute_td_error(experience)
            self.replay_buffer.add(experience, priority=priority)
        else:
            self.replay_buffer.add(experience)
        
        self.total_env_steps += 1
    
    def _compute_td_error(self, experience: Experience) -> float:
        """计算TD误差用于优先级回放"""
        state = experience.state.unsqueeze(0).to(self.device)
        action = experience.action.unsqueeze(0).to(self.device)
        reward = torch.tensor([experience.reward], device=self.device)
        next_state = experience.next_state.unsqueeze(0).to(self.device)
        done = torch.tensor([experience.done], dtype=torch.float32, device=self.device)
        
        # 计算当前Q值
        current_q1, current_q2 = self.critic.get_main_q_values(state, action)
        current_q = torch.min(current_q1, current_q2)
        
        # 计算目标Q值
        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(next_state)
            target_q = self.critic.get_target_min_q_value(next_state, next_action)
            target_q = target_q - self.alpha * next_log_prob.unsqueeze(1)
            target_q = reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.config.gamma * target_q
        
        # 计算TD误差
        td_error = torch.abs(current_q - target_q).item()
        return max(td_error, 1e-6)  # 防止零优先级
    
    def can_update(self) -> bool:
        """检查是否可以更新"""
        return (self.replay_buffer.can_sample() and 
                self.total_env_steps >= self.config.learning_starts)
    
    def update(self, update_actor: bool = True) -> Dict[str, float]:
        """
        更新网络参数
        
        Args:
            update_actor: 是否更新Actor网络
            
        Returns:
            losses: 损失字典
        """
        if not self.can_update():
            return {}
        
        losses = {}
        
        for _ in range(self.config.gradient_steps):
            # 采样批次
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                batch, indices, weights = self.replay_buffer.sample()
                weights = weights.to(self.device)
            else:
                batch = self.replay_buffer.sample()
                indices = None
                weights = None
            
            # 准备批次数据
            states = torch.stack([exp.state for exp in batch]).to(self.device)
            actions = torch.stack([exp.action for exp in batch]).to(self.device)
            rewards = torch.tensor([exp.reward for exp in batch], 
                                 dtype=torch.float32, device=self.device)
            next_states = torch.stack([exp.next_state for exp in batch]).to(self.device)
            dones = torch.tensor([exp.done for exp in batch], 
                               dtype=torch.float32, device=self.device)
            
            # 更新Critic
            critic_loss, td_errors = self._update_critic(
                states, actions, rewards, next_states, dones, weights
            )
            losses['critic_loss'] = critic_loss
            
            # 更新优先级
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and indices is not None:
                self.replay_buffer.update_priorities(indices, td_errors.detach().cpu())
            
            # 更新Actor
            if update_actor:
                actor_loss, policy_entropy = self._update_actor(states)
                losses['actor_loss'] = actor_loss
                losses['policy_entropy'] = policy_entropy
                
                # 更新温度参数
                if self.config.auto_alpha:
                    alpha_loss = self._update_alpha(states)
                    losses['alpha_loss'] = alpha_loss
            
            # 软更新目标网络
            if self.training_step % self.config.target_update_freq == 0:
                self.critic.soft_update(self.config.tau)
            
            self.training_step += 1
        
        # 记录统计信息
        losses['alpha'] = self.alpha.item()
        self._update_stats(losses)
        
        # 定期日志
        if self.training_step % self.config.log_interval == 0:
            self._log_training_stats()
        
        return losses
    
    def _update_critic(self, states, actions, rewards, next_states, dones, weights=None):
        """更新Critic网络"""
        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.get_action(next_states)
            target_q = self.critic.get_target_min_q_value(next_states, next_actions)
            target_q = target_q - self.alpha * next_log_probs.unsqueeze(1)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.config.gamma * target_q
        
        # 计算当前Q值
        current_q1, current_q2 = self.critic.get_main_q_values(states, actions)
        
        # 计算损失
        td_errors1 = torch.abs(current_q1 - target_q)
        td_errors2 = torch.abs(current_q2 - target_q)
        td_errors = torch.max(td_errors1, td_errors2).squeeze()
        
        if weights is not None:
            # 加权损失（用于优先级回放）
            critic_loss = torch.mean(weights * (td_errors1.squeeze() ** 2 + td_errors2.squeeze() ** 2))
        else:
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 反向传播
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.get_parameters(), max_norm=1.0)
        
        self.critic_optimizer.step()
        
        return critic_loss.item(), td_errors
    
    def _update_actor(self, states):
        """更新Actor网络"""
        # 生成动作和对数概率
        actions, log_probs = self.actor.get_action(states)
        
        # 计算Q值
        q_values = self.critic.main_network.get_min_q_value(states, actions)
        
        # 计算Actor损失
        actor_loss = torch.mean(self.alpha * log_probs - q_values.squeeze())
        
        # 反向传播
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        
        self.actor_optimizer.step()
        
        # 计算策略熵（正值表示高熵，负值表示低熵）
        policy_entropy = -torch.mean(log_probs).item()
        
        return actor_loss.item(), policy_entropy
    
    def _update_alpha(self, states):
        """更新温度参数"""
        with torch.no_grad():
            _, log_probs = self.actor.get_action(states)
        
        # 计算alpha损失
        alpha_loss = -torch.mean(self.log_alpha * (log_probs + self.config.target_entropy))
        
        # 反向传播
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        return alpha_loss.item()
    
    def _update_stats(self, losses):
        """更新训练统计"""
        for key, value in losses.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)
    
    def _log_training_stats(self):
        """记录训练统计"""
        stats_str = f"Step {self.training_step}: "
        for key, values in self.training_stats.items():
            if values:
                avg_value = np.mean(list(values)[-100:])  # 最近100步的平均值
                stats_str += f"{key}={avg_value:.4f} "
        
        self.logger.info(stats_str)
    
    def get_training_stats(self) -> Dict[str, float]:
        """获取训练统计"""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_recent"] = np.mean(list(values)[-100:]) if len(values) >= 100 else np.mean(values)
        
        stats['training_step'] = self.training_step
        stats['total_env_steps'] = self.total_env_steps
        stats['buffer_size'] = self.replay_buffer.size
        
        return stats
    
    def save(self, path: Union[str, Path]) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存网络参数
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.alpha_optimizer else None,
            'training_step': self.training_step,
            'total_env_steps': self.total_env_steps,
            'config': self.config
        }, path / 'model.pt')
        
        # 保存回放缓冲区（如果需要）
        if hasattr(self.replay_buffer, 'state_dict'):
            torch.save(self.replay_buffer.state_dict(), path / 'replay_buffer.pt')
        
        # 保存配置
        with open(path / 'config.json', 'w') as f:
            # 将配置转换为可序列化的字典
            config_dict = {
                k: v for k, v in self.config.__dict__.items() 
                if isinstance(v, (int, float, str, bool, type(None)))
            }
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"模型已保存到 {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        path = Path(path)
        
        # 加载模型参数
        checkpoint = torch.load(path / 'model.pt', map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.log_alpha.data = checkpoint['log_alpha'].to(self.device)
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if checkpoint['alpha_optimizer_state_dict'] and self.alpha_optimizer:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.training_step = checkpoint['training_step']
        self.total_env_steps = checkpoint['total_env_steps']
        
        # 加载回放缓冲区（如果存在）
        buffer_path = path / 'replay_buffer.pt'
        if buffer_path.exists() and hasattr(self.replay_buffer, 'load_state_dict'):
            buffer_state = torch.load(buffer_path, map_location='cpu')
            self.replay_buffer.load_state_dict(buffer_state)
        
        self.logger.info(f"模型已从 {path} 加载")
    
    def eval(self):
        """设置为评估模式"""
        super().eval()
        self.actor.eval()
        self.critic.eval_mode()
        return self
    
    def train(self, mode: bool = True):
        """设置为训练模式"""
        super().train(mode)
        self.actor.train(mode)
        if mode:
            self.critic.train_mode()
        else:
            self.critic.eval_mode()
        return self
    
    def reset_training_stats(self):
        """重置训练统计"""
        for key in self.training_stats:
            self.training_stats[key].clear()
    
    def get_policy_state_dict(self) -> Dict[str, Any]:
        """获取策略网络状态字典（用于部署）"""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'log_alpha': self.log_alpha,
            'config': self.config
        }
    
    def load_policy_state_dict(self, state_dict: Dict[str, Any]):
        """加载策略网络状态字典"""
        self.actor.load_state_dict(state_dict['actor_state_dict'])
        self.log_alpha.data = state_dict['log_alpha'].to(self.device)