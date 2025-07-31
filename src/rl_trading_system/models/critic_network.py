"""
Critic网络实现 - SAC算法的价值网络
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CriticConfig:
    """Critic网络配置"""
    state_dim: int = 256
    action_dim: int = 100
    hidden_dim: int = 512
    n_layers: int = 3
    activation: str = 'relu'
    dropout: float = 0.1


class Critic(nn.Module):
    """
    SAC Critic网络
    
    实现状态-动作价值函数Q(s,a)
    采用双Q网络架构以减少过估计偏差
    """
    
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        
        # 激活函数映射
        activation_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'gelu': nn.GELU,
            'leaky_relu': nn.LeakyReLU
        }
        
        if config.activation not in activation_map:
            raise ValueError(f"不支持的激活函数: {config.activation}")
        
        activation_fn = activation_map[config.activation]
        
        # 状态编码器
        state_layers = []
        input_dim = config.state_dim
        
        for i in range(config.n_layers - 1):
            state_layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                activation_fn(),
                nn.Dropout(config.dropout)
            ])
            input_dim = config.hidden_dim
        
        self.state_encoder = nn.Sequential(*state_layers)
        
        # 动作编码器
        action_layers = []
        input_dim = config.action_dim
        
        for i in range(config.n_layers - 1):
            action_layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                activation_fn(),
                nn.Dropout(config.dropout)
            ])
            input_dim = config.hidden_dim
        
        self.action_encoder = nn.Sequential(*action_layers)
        
        # Q网络：融合状态和动作特征
        self.q_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            activation_fn(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            activation_fn(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        
        # 对最后一层使用较小的初始化
        final_layer = self.q_network[-1]
        nn.init.uniform_(final_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(final_layer.bias, -3e-3, 3e-3)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            
        Returns:
            q_value: Q值 [batch_size, 1]
        """
        # 编码状态和动作
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        
        # 融合特征
        combined_features = torch.cat([state_features, action_features], dim=1)
        
        # 计算Q值
        q_value = self.q_network(combined_features)
        
        return q_value
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        获取Q值（与forward相同，提供更明确的接口）
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            
        Returns:
            q_value: Q值 [batch_size, 1]
        """
        return self.forward(state, action)


class DoubleCritic(nn.Module):
    """
    双Critic网络
    
    实现双Q网络架构，包含两个独立的Critic网络
    用于减少Q值过估计问题
    """
    
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        
        # 两个独立的Critic网络
        self.critic1 = Critic(config)
        self.critic2 = Critic(config)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            
        Returns:
            q1: 第一个Critic的Q值 [batch_size, 1]
            q2: 第二个Critic的Q值 [batch_size, 1]
        """
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        
        return q1, q2
    
    def get_q_values(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取两个Q值
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            
        Returns:
            q1: 第一个Critic的Q值 [batch_size, 1]
            q2: 第二个Critic的Q值 [batch_size, 1]
        """
        return self.forward(state, action)
    
    def get_min_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        获取两个Q值中的最小值（用于减少过估计）
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            
        Returns:
            min_q: 最小Q值 [batch_size, 1]
        """
        q1, q2 = self.forward(state, action)
        min_q = torch.min(q1, q2)
        
        return min_q
    
    def get_target_q_value(self, state: torch.Tensor, action: torch.Tensor, 
                          log_prob: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        计算目标Q值（用于SAC训练）
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            log_prob: 动作对数概率 [batch_size]
            alpha: 温度参数
            
        Returns:
            target_q: 目标Q值 [batch_size, 1]
        """
        min_q = self.get_min_q_value(state, action)
        target_q = min_q - alpha * log_prob.unsqueeze(1)
        
        return target_q


class CriticWithTargetNetwork(nn.Module):
    """
    带目标网络的Critic
    
    包含主网络和目标网络，支持软更新
    """
    
    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config
        
        # 主网络
        self.main_network = DoubleCritic(config)
        
        # 目标网络
        self.target_network = DoubleCritic(config)
        
        # 初始化目标网络参数
        self.hard_update()
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """主网络前向传播"""
        return self.main_network(state, action)
    
    def target_forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """目标网络前向传播"""
        return self.target_network(state, action)
    
    def get_main_q_values(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取主网络Q值"""
        return self.main_network.get_q_values(state, action)
    
    def get_target_q_values(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取目标网络Q值"""
        return self.target_network.get_q_values(state, action)
    
    def get_target_min_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """获取目标网络最小Q值"""
        return self.target_network.get_min_q_value(state, action)
    
    def soft_update(self, tau: float = 0.005):
        """
        软更新目标网络参数
        
        Args:
            tau: 更新系数，target = tau * main + (1 - tau) * target
        """
        for target_param, main_param in zip(
            self.target_network.parameters(), 
            self.main_network.parameters()
        ):
            target_param.data.copy_(
                tau * main_param.data + (1.0 - tau) * target_param.data
            )
    
    def hard_update(self):
        """硬更新目标网络参数（完全复制）"""
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def get_parameters(self):
        """获取主网络参数（用于优化器）"""
        return self.main_network.parameters()
    
    def train_mode(self):
        """设置为训练模式"""
        self.main_network.train()
        self.target_network.eval()  # 目标网络始终为评估模式
    
    def eval_mode(self):
        """设置为评估模式"""
        self.main_network.eval()
        self.target_network.eval()