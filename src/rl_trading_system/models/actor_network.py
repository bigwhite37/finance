"""
Actor网络实现 - SAC算法的策略网络
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class ActorConfig:
    """Actor网络配置"""
    state_dim: int = 256
    action_dim: int = 100
    hidden_dim: int = 512
    n_layers: int = 3
    activation: str = 'relu'
    dropout: float = 0.1
    log_std_min: float = -20
    log_std_max: float = 2
    epsilon: float = 1e-6


class Actor(nn.Module):
    """
    SAC Actor网络
    
    实现策略网络，输出投资组合权重分布
    使用重参数化技巧支持梯度反向传播
    """
    
    def __init__(self, config: ActorConfig):
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
        
        # 构建共享层
        layers = []
        input_dim = config.state_dim
        
        for i in range(config.n_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                activation_fn(),
                nn.Dropout(config.dropout)
            ])
            input_dim = config.hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # 均值头
        self.mean_head = nn.Linear(config.hidden_dim, config.action_dim)
        
        # 标准差头
        self.log_std_head = nn.Linear(config.hidden_dim, config.action_dim)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        
        # 对输出层使用较小的初始化
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        # 为均值头设置不同的初始偏置，打破对称性
        nn.init.uniform_(self.mean_head.bias, -0.1, 0.1)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            log_std: 动作对数标准差 [batch_size, action_dim]
        """
        # 检查输入是否包含NaN或无穷值
        if torch.any(torch.isnan(state)) or torch.any(torch.isinf(state)):
            # 将NaN和无穷值替换为零
            state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 共享特征提取
        features = self.shared_layers(state)
        
        # 检查特征是否包含NaN
        if torch.any(torch.isnan(features)):
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 计算均值和对数标准差
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # 确保输出没有NaN值
        if torch.any(torch.isnan(mean)):
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1e6, neginf=-1e6)
        if torch.any(torch.isnan(log_std)):
            log_std = torch.nan_to_num(log_std, nan=self.config.log_std_min, posinf=self.config.log_std_max, neginf=self.config.log_std_min)
        
        # 限制log_std的范围以确保数值稳定性
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成动作和对数概率
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 投资组合权重 [batch_size, action_dim]
            log_prob: 动作对数概率 [batch_size]
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            # 确定性动作：直接使用均值，跳过随机采样
            action_raw = mean
            # 对于确定性动作，log_prob设为0
            log_prob = torch.zeros(state.size(0), device=state.device)
            
            # 应用tanh变换确保动作在有界范围内
            action_tanh = torch.tanh(action_raw)
            
            # 将tanh输出转换为投资组合权重（非负且和为1）
            action = self._to_portfolio_weights(action_tanh)
            
        else:
            # 随机动作：从正态分布采样
            std = torch.exp(log_std)
            normal_dist = Normal(mean, std)
            action_raw = normal_dist.rsample()  # 重参数化采样
            
            # 计算对数概率
            log_prob = normal_dist.log_prob(action_raw).sum(dim=1)
            
            # 应用tanh变换确保动作在有界范围内
            action_tanh = torch.tanh(action_raw)
            
            # 计算tanh变换的雅可比行列式修正
            log_prob -= torch.sum(
                torch.log(1 - action_tanh.pow(2) + self.config.epsilon), 
                dim=1
            )
            
            # 将tanh输出转换为投资组合权重（非负且和为1）
            action = self._to_portfolio_weights(action_tanh)
        
        return action, log_prob
    
    def _to_portfolio_weights(self, action_tanh: torch.Tensor) -> torch.Tensor:
        """
        将tanh输出转换为投资组合权重
        
        Args:
            action_tanh: tanh变换后的动作 [batch_size, action_dim]
            
        Returns:
            weights: 投资组合权重 [batch_size, action_dim]
        """
        # 使用更低的温度参数来增强权重差异，鼓励更多样化的投资组合
        temperature = 0.5  # 降低温度以增强探索和权重差异
        
        # 在softmax之前添加偏置，打破对称性
        bias = torch.randn_like(action_tanh) * 0.1  # 小的随机偏置
        action_biased = action_tanh + bias
        
        weights = F.softmax(action_biased / temperature, dim=1)
        
        # 确保权重没有NaN值
        if torch.any(torch.isnan(weights)):
            # 如果出现NaN，回退到均匀分布
            weights = torch.ones_like(weights) / weights.size(1)
        
        return weights
    
    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        计算给定状态和动作的对数概率
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            
        Returns:
            log_prob: 对数概率 [batch_size]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # 从投资组合权重反推tanh前的值
        # 这是一个近似过程，因为softmax变换不可逆
        action_normalized = action / (action.sum(dim=1, keepdim=True) + self.config.epsilon)
        
        # 使用逆softmax近似（通过对数）
        action_log = torch.log(action_normalized + self.config.epsilon)
        action_centered = action_log - action_log.mean(dim=1, keepdim=True)
        
        # 限制在tanh的有效范围内
        action_positive = torch.clamp(action_centered, -0.999, 0.999)
        action_raw = torch.atanh(action_positive)
        
        # 计算正态分布的对数概率
        normal_dist = Normal(mean, std)
        log_prob = normal_dist.log_prob(action_raw).sum(dim=1)
        
        # 减去tanh变换的雅可比行列式
        log_prob -= torch.sum(
            torch.log(1 - action_positive.pow(2) + self.config.epsilon), 
            dim=1
        )
        
        return log_prob
    
    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算策略熵
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            entropy: 策略熵 [batch_size]
        """
        _, log_std = self.forward(state)
        
        # 正态分布的熵：0.5 * log(2π) + log_std
        entropy = 0.5 * (1.0 + torch.log(2 * torch.pi)) + log_std
        entropy = entropy.sum(dim=1)
        
        return entropy
    
    def get_deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        获取确定性动作（用于评估）
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            action: 确定性动作 [batch_size, action_dim]
        """
        action, _ = self.get_action(state, deterministic=True)
        return action