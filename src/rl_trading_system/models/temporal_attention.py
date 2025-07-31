"""
时间注意力机制实现
包括缩放点积注意力、时间注意力聚合和多头时间注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    
    def __init__(self, dropout: float = 0.1):
        """
        初始化缩放点积注意力
        
        Args:
            dropout: dropout概率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, d_k]
            key: 键张量 [batch_size, seq_len, d_k]
            value: 值张量 [batch_size, seq_len, d_k]
            mask: 掩码张量 [batch_size, seq_len, seq_len]，可选
            
        Returns:
            output: 注意力输出 [batch_size, seq_len, d_k]
            attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
        """
        d_k = query.size(-1)
        
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores + mask
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class TemporalAttention(nn.Module):
    """
    时间注意力机制
    用于将时序特征聚合为固定维度的表示
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        初始化时间注意力
        
        Args:
            d_model: 模型维度
            dropout: dropout概率
        """
        super().__init__()
        self.d_model = d_model
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 注意力机制
        self.attention = ScaledDotProductAttention(dropout)
        
        # 用于时间聚合的查询向量
        self.temporal_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len]，可选
            
        Returns:
            output: 聚合后的输出 [batch_size, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 线性变换
        k = self.w_k(x)  # [batch_size, seq_len, d_model]
        v = self.w_v(x)  # [batch_size, seq_len, d_model]
        
        # 扩展时间查询向量
        q = self.temporal_query.expand(batch_size, 1, d_model)  # [batch_size, 1, d_model]
        q = self.w_q(q)
        
        # 处理掩码
        attention_mask = None
        if mask is not None:
            # 将1D掩码转换为2D注意力掩码
            attention_mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            attention_mask = attention_mask.expand(batch_size, 1, seq_len)
        
        # 应用注意力机制
        attended, attention_weights = self.attention(q, k, v, attention_mask)
        
        # 输出投影
        output = self.w_o(attended)  # [batch_size, 1, d_model]
        output = output.squeeze(1)   # [batch_size, d_model]
        
        # 残差连接和层归一化（使用时间查询向量）
        query_squeezed = q.squeeze(1)  # [batch_size, d_model]
        output = self.layer_norm(output + query_squeezed)
        
        return self.dropout(output)
    
    def forward_with_attention(self, x: torch.Tensor, 
                             mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播并返回注意力权重
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len]，可选
            
        Returns:
            output: 聚合后的输出 [batch_size, d_model]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 线性变换
        k = self.w_k(x)
        v = self.w_v(x)
        
        # 扩展时间查询向量
        q = self.temporal_query.expand(batch_size, 1, d_model)
        q = self.w_q(q)
        
        # 处理掩码
        attention_mask = None
        if mask is not None:
            attention_mask = mask.unsqueeze(1)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len)
        
        # 应用注意力机制
        attended, attention_weights = self.attention(q, k, v, attention_mask)
        
        # 输出投影
        output = self.w_o(attended)
        output = output.squeeze(1)
        
        # 残差连接和层归一化
        query_squeezed = q.squeeze(1)
        output = self.layer_norm(output + query_squeezed)
        output = self.dropout(output)
        
        # 压缩注意力权重维度
        attention_weights = attention_weights.squeeze(1)  # [batch_size, seq_len]
        
        return output, attention_weights


class MultiHeadTemporalAttention(nn.Module):
    """
    多头时间注意力机制
    使用多个注意力头来捕捉不同的时间模式
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        """
        初始化多头时间注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 创建多个注意力头
        self.heads = nn.ModuleList([
            TemporalAttention(self.d_k, dropout) for _ in range(n_heads)
        ])
        
        # 输入投影层
        self.input_projection = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len]，可选
            
        Returns:
            output: 聚合后的输出 [batch_size, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 输入投影
        x_proj = self.input_projection(x)
        
        # 分割为多个头
        x_heads = x_proj.view(batch_size, seq_len, self.n_heads, self.d_k)
        x_heads = x_heads.transpose(1, 2)  # [batch_size, n_heads, seq_len, d_k]
        
        # 对每个头应用时间注意力
        head_outputs = []
        for i in range(self.n_heads):
            head_input = x_heads[:, i, :, :]  # [batch_size, seq_len, d_k]
            head_output = self.heads[i](head_input, mask)  # [batch_size, d_k]
            head_outputs.append(head_output)
        
        # 拼接所有头的输出
        concatenated = torch.cat(head_outputs, dim=-1)  # [batch_size, d_model]
        
        # 输出投影
        output = self.output_projection(concatenated)
        
        # 残差连接和层归一化
        # 使用输入的时间平均作为残差
        residual = x.mean(dim=1)  # [batch_size, d_model]
        output = self.layer_norm(output + residual)
        
        return self.dropout(output)
    
    def forward_with_heads(self, x: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播并返回各个头的输出
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len]，可选
            
        Returns:
            output: 最终输出 [batch_size, d_model]
            head_outputs: 各个头的输出列表，每个元素形状为 [batch_size, d_k]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 输入投影
        x_proj = self.input_projection(x)
        
        # 分割为多个头
        x_heads = x_proj.view(batch_size, seq_len, self.n_heads, self.d_k)
        x_heads = x_heads.transpose(1, 2)
        
        # 对每个头应用时间注意力
        head_outputs = []
        for i in range(self.n_heads):
            head_input = x_heads[:, i, :, :]
            head_output = self.heads[i](head_input, mask)
            head_outputs.append(head_output)
        
        # 拼接所有头的输出
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # 输出投影
        output = self.output_projection(concatenated)
        
        # 残差连接和层归一化
        residual = x.mean(dim=1)
        output = self.layer_norm(output + residual)
        output = self.dropout(output)
        
        return output, head_outputs
    
    def forward_with_attention_weights(self, x: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        前向传播并返回各个头的注意力权重
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len]，可选
            
        Returns:
            output: 最终输出 [batch_size, d_model]
            head_attentions: 各个头的注意力权重列表，每个元素形状为 [batch_size, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 输入投影
        x_proj = self.input_projection(x)
        
        # 分割为多个头
        x_heads = x_proj.view(batch_size, seq_len, self.n_heads, self.d_k)
        x_heads = x_heads.transpose(1, 2)
        
        # 对每个头应用时间注意力
        head_outputs = []
        head_attentions = []
        for i in range(self.n_heads):
            head_input = x_heads[:, i, :, :]
            head_output, head_attention = self.heads[i].forward_with_attention(head_input, mask)
            head_outputs.append(head_output)
            head_attentions.append(head_attention)
        
        # 拼接所有头的输出
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # 输出投影
        output = self.output_projection(concatenated)
        
        # 残差连接和层归一化
        residual = x.mean(dim=1)
        output = self.layer_norm(output + residual)
        output = self.dropout(output)
        
        return output, head_attentions


class AdaptiveTemporalAttention(nn.Module):
    """
    自适应时间注意力机制
    可以根据输入动态调整注意力模式
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, 
                 use_position_bias: bool = True, dropout: float = 0.1):
        """
        初始化自适应时间注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            use_position_bias: 是否使用位置偏置
            dropout: dropout概率
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_position_bias = use_position_bias
        
        # 多头时间注意力
        self.multi_head_attention = MultiHeadTemporalAttention(d_model, n_heads, dropout)
        
        # 位置偏置
        if use_position_bias:
            self.position_bias = nn.Parameter(torch.randn(1, 1, 512) * 0.1)  # 最大序列长度512
        
        # 自适应权重
        self.adaptive_weight = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len]，可选
            
        Returns:
            output: 聚合后的输出 [batch_size, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 添加位置偏置
        if self.use_position_bias:
            pos_bias = self.position_bias[:, :, :seq_len].transpose(1, 2)  # [1, seq_len, 1]
            x = x + pos_bias
        
        # 多头时间注意力
        attention_output = self.multi_head_attention(x, mask)
        
        # 计算自适应权重
        global_context = x.mean(dim=1)  # [batch_size, d_model]
        adaptive_weight = self.adaptive_weight(global_context)  # [batch_size, 1]
        
        # 简单的时间平均作为备选
        simple_average = x.mean(dim=1)  # [batch_size, d_model]
        
        # 自适应组合
        output = adaptive_weight * attention_output + (1 - adaptive_weight) * simple_average
        
        return self.dropout(output)
    
    def get_attention_visualization(self, x: torch.Tensor, 
                                  mask: Optional[torch.Tensor] = None) -> dict:
        """
        获取注意力可视化信息
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, seq_len]，可选
            
        Returns:
            visualization_data: 包含注意力权重和其他可视化信息的字典
        """
        batch_size, seq_len, d_model = x.shape
        
        # 添加位置偏置
        if self.use_position_bias:
            pos_bias = self.position_bias[:, :, :seq_len].transpose(1, 2)
            x = x + pos_bias
        
        # 获取多头注意力权重
        _, head_attentions = self.multi_head_attention.forward_with_attention_weights(x, mask)
        
        # 计算自适应权重
        global_context = x.mean(dim=1)
        adaptive_weight = self.adaptive_weight(global_context)
        
        return {
            'head_attentions': head_attentions,
            'adaptive_weights': adaptive_weight.detach().cpu().numpy(),
            'sequence_length': seq_len,
            'n_heads': self.n_heads
        }