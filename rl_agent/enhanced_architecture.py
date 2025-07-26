"""
增强的网络架构设计
专门为金融时序预测和投资组合优化设计的神经网络架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制 - 用于捕获因子间的相互作用"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return output


class FactorEncoder(nn.Module):
    """因子编码器 - 专门处理金融因子特征"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_factor_groups: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_factor_groups = num_factor_groups
        
        # 因子分组（技术、基本面、宏观、情绪）
        group_size = input_dim // num_factor_groups
        self.factor_groups = nn.ModuleList([
            nn.Sequential(
                nn.Linear(group_size, hidden_dim // num_factor_groups),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // num_factor_groups, hidden_dim // num_factor_groups)
            ) for _ in range(num_factor_groups)
        ])
        
        # 因子交互层
        self.factor_interaction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 因子重要性权重
        self.factor_weights = nn.Parameter(torch.ones(num_factor_groups))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        group_size = self.input_dim // self.num_factor_groups
        
        # 分组处理因子
        group_outputs = []
        for i, group_net in enumerate(self.factor_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < len(self.factor_groups) - 1 else self.input_dim
            group_input = x[:, start_idx:end_idx]
            group_output = group_net(group_input)
            # 应用学习到的权重
            weighted_output = group_output * torch.softmax(self.factor_weights, dim=0)[i]
            group_outputs.append(weighted_output)
        
        # 合并分组输出
        combined = torch.cat(group_outputs, dim=1)
        
        # 因子交互
        output = self.factor_interaction(combined)
        
        return output


class ResidualBlock(nn.Module):
    """残差块 - 改善深度网络的训练"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layers(x)
        out = out + residual  # 残差连接
        out = self.norm(out)  # 层归一化
        return F.relu(out)


class RiskAwareHead(nn.Module):
    """风险感知输出头 - 集成多种风险度量"""
    
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        
        # 主要风险预测
        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 不确定性估计（用于贝叶斯方法）
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        risk_pred = self.risk_predictor(x)
        uncertainty = torch.exp(self.uncertainty_estimator(x))  # 确保正值
        return risk_pred, uncertainty


class EnhancedActorCriticNetwork(nn.Module):
    """增强的Actor-Critic网络架构"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 num_attention_heads: int = 8, num_residual_blocks: int = 3):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 因子编码器
        self.factor_encoder = FactorEncoder(state_dim, hidden_dim)
        
        # 多头注意力层
        self.attention = MultiHeadAttention(hidden_dim, num_attention_heads)
        
        # 残差块堆叠
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor网络（策略网络）- 使用风险感知设计
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 动态标准差（学习状态相关的探索）
        self.actor_std_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim)
        )
        
        # Critic网络（价值网络）- 多头输出
        self.value_head = RiskAwareHead(hidden_dim, 1)
        
        # CVaR估计网络 - 增强设计
        self.cvar_head = RiskAwareHead(hidden_dim, 1)
        
        # 市场状态感知层
        self.market_state_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 3种市场状态：牛市、熊市、震荡
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """改进的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化用于大部分层
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # 特殊初始化策略输出层（小初始值以确保稳定性）
        for module in self.actor_mean.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = state.size(0)
        
        # 因子编码
        encoded_features = self.factor_encoder(state)
        
        # 为注意力机制添加序列维度
        features_seq = encoded_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # 多头注意力
        attended_features = self.attention(features_seq)
        attended_features = attended_features.squeeze(1)  # [batch, hidden_dim]
        
        # 残差块处理
        deep_features = attended_features
        for residual_block in self.residual_blocks:
            deep_features = residual_block(deep_features)
        
        # 特征融合
        final_features = self.feature_fusion(deep_features + attended_features)  # 跳跃连接
        
        # Actor输出
        action_mean = self.actor_mean(final_features)
        action_std_logits = self.actor_std_net(final_features)
        action_std = F.softplus(action_std_logits) + 1e-6
        
        # Critic输出（带不确定性）
        value, value_uncertainty = self.value_head(final_features)
        
        # CVaR估计（带不确定性）
        cvar_estimate, cvar_uncertainty = self.cvar_head(final_features)
        
        # 市场状态检测
        market_state_logits = self.market_state_detector(final_features)
        market_state_probs = F.softmax(market_state_logits, dim=-1)
        
        return {
            'action_mean': action_mean,
            'action_std': action_std,
            'value': value,
            'value_uncertainty': value_uncertainty,
            'cvar_estimate': cvar_estimate,
            'cvar_uncertainty': cvar_uncertainty,
            'market_state_probs': market_state_probs,
            'features': final_features  # 用于额外分析
        }
    
    def get_attention_weights(self, state: torch.Tensor) -> torch.Tensor:
        """获取注意力权重以便解释模型决策"""
        with torch.no_grad():
            encoded_features = self.factor_encoder(state)
            features_seq = encoded_features.unsqueeze(1)
            
            # 手动计算注意力权重
            Q = self.attention.w_q(features_seq)
            K = self.attention.w_k(features_seq)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention.scale
            attention_weights = F.softmax(scores, dim=-1)
            
            return attention_weights.squeeze(1)


class AdaptiveLearningRateScheduler:
    """自适应学习率调度器"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 patience: int = 10, factor: float = 0.5, 
                 min_lr: float = 1e-6, performance_window: int = 20):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.performance_window = performance_window
        
        self.performance_history = []
        self.wait_count = 0
        self.best_performance = float('-inf')
        
    def step(self, performance_metric: float):
        """基于性能指标调整学习率"""
        self.performance_history.append(performance_metric)
        
        # 保持窗口大小
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
        
        # 计算最近性能趋势
        if len(self.performance_history) >= self.performance_window:
            recent_avg = np.mean(self.performance_history[-self.performance_window//2:])
            
            if recent_avg > self.best_performance:
                self.best_performance = recent_avg
                self.wait_count = 0
            else:
                self.wait_count += 1
                
            # 如果性能停滞，降低学习率
            if self.wait_count >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    
                    if new_lr < old_lr:
                        print(f"学习率衰减: {old_lr:.2e} -> {new_lr:.2e}")
                
                self.wait_count = 0


class CurriculumLearningStrategy:
    """课程学习策略 - 逐渐增加任务难度"""
    
    def __init__(self, total_episodes: int, difficulty_levels: int = 4):
        self.total_episodes = total_episodes
        self.difficulty_levels = difficulty_levels
        self.current_episode = 0
        
        # 定义难度级别
        self.difficulty_configs = [
            {'transaction_cost': 0.0005, 'market_noise': 0.1, 'volatility_multiplier': 0.8},
            {'transaction_cost': 0.001, 'market_noise': 0.15, 'volatility_multiplier': 1.0},
            {'transaction_cost': 0.0015, 'market_noise': 0.2, 'volatility_multiplier': 1.2},
            {'transaction_cost': 0.002, 'market_noise': 0.25, 'volatility_multiplier': 1.5}
        ]
    
    def get_current_difficulty(self) -> Dict:
        """获取当前难度配置"""
        progress = self.current_episode / self.total_episodes
        difficulty_level = min(int(progress * self.difficulty_levels), self.difficulty_levels - 1)
        return self.difficulty_configs[difficulty_level]
    
    def update_episode(self, episode: int):
        """更新当前episode"""
        self.current_episode = episode