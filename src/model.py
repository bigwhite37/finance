"""
TimeSeriesTransformer特征提取器和自定义策略网络
用于Stable-Baselines3的SAC/PPO算法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from typing import Dict, List, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
import gymnasium as gym
import logging

logger = logging.getLogger(__name__)


class TimeSeriesTransformer(BaseFeaturesExtractor):
    """
    时间序列Transformer特征提取器
    
    专门为金融时间序列数据设计，包含：
    - 多头自注意力机制捕获时间依赖
    - 位置编码
    - 残差连接和层归一化
    - Dropout防止过拟合
    """
    
    def __init__(self, 
                 observation_space: gym.Space,
                 lookback_window: int = 30,
                 num_stocks: int = 50,
                 num_features: int = 5,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 features_dim: int = 256):
        """
        初始化TimeSeriesTransformer
        
        Args:
            observation_space: 观察空间
            lookback_window: 历史窗口长度
            num_stocks: 股票数量
            num_features: 每只股票的特征数量
            d_model: Transformer模型维度
            nhead: 多头注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout概率
            features_dim: 输出特征维度
        """
        super().__init__(observation_space, features_dim)
        
        self.lookback_window = lookback_window
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.d_model = d_model
        
        # 计算输入维度
        hist_dim = lookback_window * num_stocks * num_features
        weight_dim = num_stocks
        state_dim = 3  # 总价值、回撤、波动率
        total_dim = hist_dim + weight_dim + state_dim
        
        if observation_space.shape[0] != total_dim:
            logger.warning(f"观察空间维度不匹配: 期望{total_dim}, 实际{observation_space.shape[0]}")
        
        # 输入投影层
        self.input_projection = nn.Linear(num_features, d_model)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_len=lookback_window)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 股票级别的注意力聚合
        self.stock_attention = MultiHeadStockAttention(d_model, nhead, dropout)
        
        # 当前状态处理
        self.weight_projection = nn.Linear(weight_dim, d_model // 2)
        self.state_projection = nn.Linear(state_dim, d_model // 2)
        
        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observations: 观察张量 [batch_size, obs_dim]
            
        Returns:
            提取的特征 [batch_size, features_dim]
        """
        batch_size = observations.shape[0]
        
        # 分解观察空间
        hist_dim = self.lookback_window * self.num_stocks * self.num_features
        weight_dim = self.num_stocks
        
        hist_data = observations[:, :hist_dim]
        weights = observations[:, hist_dim:hist_dim + weight_dim]
        state = observations[:, hist_dim + weight_dim:]
        
        # 重塑历史数据 [batch_size, lookback_window, num_stocks, num_features]
        hist_data = hist_data.reshape(batch_size, self.lookback_window, self.num_stocks, self.num_features)
        
        # 处理历史数据
        hist_features = self._process_historical_data(hist_data)
        
        # 处理当前状态
        weight_features = F.relu(self.weight_projection(weights))
        state_features = F.relu(self.state_projection(state))
        current_features = torch.cat([weight_features, state_features], dim=-1)
        
        # 特征融合
        combined_features = torch.cat([hist_features, current_features], dim=-1)
        output = self.fusion(combined_features)
        
        return output
    
    def _process_historical_data(self, hist_data: torch.Tensor) -> torch.Tensor:
        """
        处理历史时间序列数据
        
        Args:
            hist_data: [batch_size, lookback_window, num_stocks, num_features]
            
        Returns:
            处理后的特征 [batch_size, d_model]
        """
        batch_size, seq_len, num_stocks, num_features = hist_data.shape
        
        # 重塑为 [batch_size * num_stocks, seq_len, num_features]
        reshaped_data = hist_data.reshape(batch_size * num_stocks, seq_len, num_features)
        
        # 输入投影 [batch_size * num_stocks, seq_len, d_model]
        projected = self.input_projection(reshaped_data)
        
        # 添加位置编码
        projected = self.positional_encoding(projected)
        
        # Transformer编码 [batch_size * num_stocks, seq_len, d_model]
        encoded = self.transformer(projected)
        
        # 时间维度聚合（取最后一个时间步）
        stock_features = encoded[:, -1, :]  # [batch_size * num_stocks, d_model]
        
        # 重塑为 [batch_size, num_stocks, d_model]
        stock_features = stock_features.reshape(batch_size, num_stocks, self.d_model)
        
        # 股票级别注意力聚合
        aggregated_features = self.stock_attention(stock_features)
        
        return aggregated_features


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


class MultiHeadStockAttention(nn.Module):
    """多头股票注意力机制"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 学习全局查询向量
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, stock_features: torch.Tensor) -> torch.Tensor:
        """
        股票级别注意力聚合
        
        Args:
            stock_features: [batch_size, num_stocks, d_model]
            
        Returns:
            聚合特征 [batch_size, d_model]
        """
        batch_size = stock_features.shape[0]
        
        # 扩展全局查询
        query = self.global_query.expand(batch_size, -1, -1)
        
        # 注意力计算
        attended, _ = self.attention(query, stock_features, stock_features)
        attended = self.dropout(attended)
        
        # 残差连接和归一化
        output = self.norm(attended + query)
        
        return output.squeeze(1)  # [batch_size, d_model]


class TradingPolicy(ActorCriticPolicy):
    """
    自定义交易策略网络
    使用TimeSeriesTransformer作为特征提取器
    """
    
    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 lr_schedule: Schedule,
                 net_arch: List[Union[int, Dict[str, List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 lookback_window: int = 30,
                 num_stocks: int = 50,
                 num_features: int = 5,
                 *args, **kwargs):
        
        # 特征提取器参数
        self.lookback_window = lookback_window
        self.num_stocks = num_stocks  
        self.num_features = num_features
        
        # 默认网络架构
        if net_arch is None:
            net_arch = [dict(pi=[256, 128], vf=[256, 128])]
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args, **kwargs
        )
    
    def _build_mlp_extractor(self) -> None:
        """构建特征提取器"""
        self.features_extractor = TimeSeriesTransformer(
            self.observation_space,
            lookback_window=self.lookback_window,
            num_stocks=self.num_stocks,
            num_features=self.num_features,
            features_dim=256
        )


class RiskAwareRewardWrapper:
    """
    风险感知奖励包装器
    动态调整奖励函数中的风险惩罚系数
    """
    
    def __init__(self, 
                 base_penalty: float = 2.0,
                 volatility_threshold: float = 0.02,
                 drawdown_threshold: float = 0.05,
                 adaptive_penalty: bool = True):
        self.base_penalty = base_penalty
        self.volatility_threshold = volatility_threshold
        self.drawdown_threshold = drawdown_threshold
        self.adaptive_penalty = adaptive_penalty
        
        # 自适应参数
        self.recent_volatility = deque(maxlen=20)
        self.recent_drawdowns = deque(maxlen=20)
    
    def calculate_reward(self,
                        portfolio_return: float,
                        current_drawdown: float,
                        volatility: float,
                        trade_cost: float,
                        total_value: float) -> float:
        """
        计算调整后的奖励
        
        Args:
            portfolio_return: 组合收益率
            current_drawdown: 当前回撤
            volatility: 当前波动率
            trade_cost: 交易成本
            total_value: 组合总价值
            
        Returns:
            调整后的奖励
        """
        # 更新历史记录
        if self.adaptive_penalty:
            self.recent_volatility.append(volatility)
            self.recent_drawdowns.append(current_drawdown)
        
        # 基础收益奖励
        reward = portfolio_return
        
        # 动态回撤惩罚
        if current_drawdown > self.drawdown_threshold:
            if self.adaptive_penalty and len(self.recent_drawdowns) > 5:
                # 基于历史回撤调整惩罚强度
                avg_drawdown = np.mean(list(self.recent_drawdowns))
                penalty_multiplier = min(3.0, 1 + avg_drawdown / self.drawdown_threshold)
            else:
                penalty_multiplier = 1.0
            
            drawdown_penalty = self.base_penalty * penalty_multiplier * (current_drawdown - self.drawdown_threshold)
            reward -= drawdown_penalty
        
        # 波动率惩罚
        if volatility > self.volatility_threshold:
            volatility_penalty = 0.5 * (volatility - self.volatility_threshold)
            reward -= volatility_penalty
        
        # 交易成本惩罚
        cost_penalty = trade_cost / total_value
        reward -= cost_penalty
        
        # 奖励平滑（避免极端值）
        reward = np.clip(reward, -0.1, 0.1)
        
        return reward


class PortfolioMetrics:
    """组合性能指标计算工具"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(values: np.ndarray) -> float:
        """计算最大回撤"""
        if len(values) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown)
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
        """计算Calmar比率"""
        if max_drawdown == 0:
            return np.inf
        
        annual_return = (1 + np.mean(returns)) ** 252 - 1
        return annual_return / max_drawdown
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """计算Sortino比率"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        return np.mean(excess_returns) * np.sqrt(252) / downside_deviation
    
    @staticmethod
    def calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """计算信息比率"""
        if len(returns) != len(benchmark_returns) or len(returns) == 0:
            return 0.0
        
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return np.mean(excess_returns) / tracking_error * np.sqrt(252)


if __name__ == "__main__":
    # 测试TimeSeriesTransformer
    print("测试TimeSeriesTransformer...")
    
    # 创建模拟观察空间
    lookback_window = 30
    num_stocks = 10
    num_features = 5
    obs_dim = lookback_window * num_stocks * num_features + num_stocks + 3
    
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    
    # 创建特征提取器
    extractor = TimeSeriesTransformer(
        observation_space,
        lookback_window=lookback_window,
        num_stocks=num_stocks,
        num_features=num_features,
        features_dim=256
    )
    
    # 测试前向传播
    batch_size = 32
    obs = torch.randn(batch_size, obs_dim)
    
    with torch.no_grad():
        features = extractor(obs)
        print(f"输入形状: {obs.shape}")
        print(f"输出特征形状: {features.shape}")
        print(f"特征提取器参数数量: {sum(p.numel() for p in extractor.parameters())}")
    
    # 测试奖励包装器
    print("\n测试风险感知奖励包装器...")
    reward_wrapper = RiskAwareRewardWrapper()
    
    # 模拟几个奖励计算
    test_cases = [
        (0.01, 0.02, 0.015, 100, 1000000),  # 正常情况
        (0.005, 0.08, 0.025, 150, 1000000),  # 高回撤
        (-0.01, 0.15, 0.03, 80, 980000),    # 极端情况
    ]
    
    for i, (ret, dd, vol, cost, value) in enumerate(test_cases):
        reward = reward_wrapper.calculate_reward(ret, dd, vol, cost, value)
        print(f"案例{i+1}: 收益={ret:.3f}, 回撤={dd:.3f}, 波动={vol:.3f} -> 奖励={reward:.4f}")
    
    print("\n测试完成!")
    
    from collections import deque