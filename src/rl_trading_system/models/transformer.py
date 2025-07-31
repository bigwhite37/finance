"""
时序Transformer编码器实现
包括完整的Transformer架构，用于处理金融时序数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .positional_encoding import PositionalEncoding
from .temporal_attention import MultiHeadTemporalAttention


@dataclass
class TransformerConfig:
    """Transformer配置类"""
    d_model: int = 256          # 模型维度
    n_heads: int = 8            # 注意力头数
    n_layers: int = 6           # 编码器层数
    d_ff: int = 1024           # 前馈网络维度
    dropout: float = 0.1        # dropout概率
    max_seq_len: int = 252      # 最大序列长度（一年交易日）
    n_features: int = 50        # 输入特征数
    activation: str = 'gelu'    # 激活函数类型
    
    def __post_init__(self):
        """配置验证"""
        assert self.d_model % self.n_heads == 0, "d_model必须能被n_heads整除"
        assert self.activation in ['relu', 'gelu', 'swish'], f"不支持的激活函数: {self.activation}"


class FeedForwardNetwork(nn.Module):
    """
    前馈网络
    实现Transformer中的位置前馈网络
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度
            d_ff: 前馈网络隐藏层维度
            dropout: dropout概率
            activation: 激活函数类型
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    标准的Transformer多头注意力实现
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        初始化多头注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]，可选
            
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 线性变换
        Q = self.w_q(x)  # [batch_size, seq_len, d_model]
        K = self.w_k(x)  # [batch_size, seq_len, d_model]
        V = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        
        # 计算注意力
        attention_output = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # 重塑回原始形状
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 输出投影
        output = self.w_o(attention_output)
        
        return output
    
    def _scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        缩放点积注意力
        
        Args:
            Q: 查询张量 [batch_size, n_heads, seq_len, d_k]
            K: 键张量 [batch_size, n_heads, seq_len, d_k]
            V: 值张量 [batch_size, n_heads, seq_len, d_k]
            mask: 掩码张量，可选
            
        Returns:
            注意力输出 [batch_size, n_heads, seq_len, d_k]
        """
        d_k = Q.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            # 扩展掩码维度以匹配多头
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            scores = scores + mask
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含多头自注意力和前馈网络，以及残差连接和层归一化
    """
    
    def __init__(self, config: TransformerConfig):
        """
        初始化编码器层
        
        Args:
            config: Transformer配置
        """
        super().__init__()
        self.config = config
        
        # 多头自注意力
        self.self_attention = MultiHeadAttention(
            config.d_model, config.n_heads, config.dropout
        )
        
        # 前馈网络
        self.feed_forward = FeedForwardNetwork(
            config.d_model, config.d_ff, config.dropout, config.activation
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码，可选
            
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TimeSeriesTransformer(nn.Module):
    """
    时序Transformer编码器
    专门用于处理金融时序数据的Transformer架构
    """
    
    def __init__(self, config: TransformerConfig):
        """
        初始化时序Transformer
        
        Args:
            config: Transformer配置
        """
        super().__init__()
        self.config = config
        
        # 输入投影层
        self.input_projection = nn.Linear(config.n_features, config.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.n_layers)
        ])
        
        # 时间注意力聚合
        self.temporal_attention = MultiHeadTemporalAttention(
            config.d_model, config.n_heads, config.dropout
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, n_stocks, n_features]
            mask: 序列掩码 [batch_size, seq_len]，可选
            
        Returns:
            输出张量 [batch_size, n_stocks, d_model]
        """
        batch_size, seq_len, n_stocks, n_features = x.shape
        
        # 检查序列长度
        if seq_len > self.config.max_seq_len:
            raise IndexError(f"序列长度 {seq_len} 超过最大长度 {self.config.max_seq_len}")
        
        # 重塑输入：[batch_size * n_stocks, seq_len, n_features]
        x = x.view(batch_size * n_stocks, seq_len, n_features)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size * n_stocks, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 处理掩码
        attention_mask = None
        if mask is not None:
            # 扩展掩码以匹配重塑后的批次大小
            attention_mask = mask.repeat_interleave(n_stocks, dim=0)  # [batch_size * n_stocks, seq_len]
            # 转换为注意力掩码格式
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size * n_stocks, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, 1, seq_len, -1)  # [batch_size * n_stocks, 1, seq_len, seq_len]
            attention_mask = attention_mask.squeeze(1)  # [batch_size * n_stocks, seq_len, seq_len]
        
        # 通过编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)
        
        # 重塑回股票维度：[batch_size, n_stocks, seq_len, d_model]
        x = x.view(batch_size, n_stocks, seq_len, self.config.d_model)
        
        # 对每只股票应用时间注意力聚合
        stock_representations = []
        for i in range(n_stocks):
            stock_seq = x[:, i, :, :]  # [batch_size, seq_len, d_model]
            stock_mask = mask if mask is not None else None
            stock_repr = self.temporal_attention(stock_seq, stock_mask)  # [batch_size, d_model]
            stock_representations.append(stock_repr)
        
        # 堆叠所有股票的表示
        output = torch.stack(stock_representations, dim=1)  # [batch_size, n_stocks, d_model]
        
        # 输出投影
        output = self.output_projection(output)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor, 
                            mask: Optional[torch.Tensor] = None) -> dict:
        """
        获取注意力权重用于可视化
        
        Args:
            x: 输入张量 [batch_size, seq_len, n_stocks, n_features]
            mask: 序列掩码，可选
            
        Returns:
            包含注意力权重的字典
        """
        batch_size, seq_len, n_stocks, n_features = x.shape
        
        # 重塑输入
        x = x.view(batch_size * n_stocks, seq_len, n_features)
        
        # 输入投影和位置编码
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # 收集每层的注意力权重
        layer_attentions = []
        
        # 处理掩码
        attention_mask = None
        if mask is not None:
            attention_mask = mask.repeat_interleave(n_stocks, dim=0)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(-1, 1, seq_len, -1)
            attention_mask = attention_mask.squeeze(1)
        
        # 通过编码器层（这里简化，实际需要修改编码器层以返回注意力权重）
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)
        
        # 重塑并获取时间注意力权重
        x = x.view(batch_size, n_stocks, seq_len, self.config.d_model)
        
        temporal_attentions = []
        for i in range(n_stocks):
            stock_seq = x[:, i, :, :]
            stock_mask = mask if mask is not None else None
            _, head_attentions = self.temporal_attention.forward_with_attention_weights(
                stock_seq, stock_mask
            )
            temporal_attentions.append(head_attentions)
        
        return {
            'temporal_attentions': temporal_attentions,
            'n_stocks': n_stocks,
            'seq_len': seq_len
        }
    
    def encode_sequence(self, x: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        仅编码序列，不进行时间聚合
        
        Args:
            x: 输入张量 [batch_size, seq_len, n_stocks, n_features]
            mask: 序列掩码，可选
            
        Returns:
            编码后的序列 [batch_size, seq_len, n_stocks, d_model]
        """
        batch_size, seq_len, n_stocks, n_features = x.shape
        
        # 重塑输入
        x = x.view(batch_size * n_stocks, seq_len, n_features)
        
        # 输入投影和位置编码
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # 处理掩码
        attention_mask = None
        if mask is not None:
            attention_mask = mask.repeat_interleave(n_stocks, dim=0)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(-1, 1, seq_len, -1)
            attention_mask = attention_mask.squeeze(1)
        
        # 通过编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask)
        
        # 重塑回原始形状
        x = x.view(batch_size, seq_len, n_stocks, self.config.d_model)
        
        return x
    
    def get_model_size(self) -> dict:
        """
        获取模型大小信息
        
        Returns:
            包含模型大小信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'config': self.config
        }