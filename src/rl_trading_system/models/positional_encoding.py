"""
位置编码组件实现
支持多种位置编码方式：正弦余弦编码、可学习位置编码、相对位置编码
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    正弦余弦位置编码
    使用不同频率的正弦和余弦函数来编码位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout概率
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer，不参与梯度计算
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise IndexError(f"序列长度 {seq_len} 超过最大长度 {self.max_len}")
        
        # 添加位置编码
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码
    位置编码参数通过训练学习得到
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化可学习位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: dropout概率
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建可学习的位置编码参数
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise IndexError(f"序列长度 {seq_len} 超过最大长度 {self.max_len}")
        
        # 添加可学习位置编码
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码
    基于相对位置而非绝对位置的编码方式
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 128, dropout: float = 0.1):
        """
        初始化相对位置编码
        
        Args:
            d_model: 模型维度
            max_relative_position: 最大相对位置距离
            dropout: dropout概率
        """
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(p=dropout)
        
        # 相对位置编码表
        # 大小为 (2 * max_relative_position + 1, d_model)
        # 索引 max_relative_position 对应相对位置 0
        vocab_size = 2 * max_relative_position + 1
        self.relative_pe = nn.Parameter(torch.randn(vocab_size, d_model) * 0.1)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """
        计算相对位置矩阵
        
        Args:
            seq_len: 序列长度
            
        Returns:
            相对位置矩阵 [seq_len, seq_len]
        """
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # 裁剪到最大相对位置范围
        distance_mat = torch.clamp(distance_mat, 
                                 -self.max_relative_position, 
                                 self.max_relative_position)
        
        return distance_mat
    
    def get_attention_bias(self, seq_len: int) -> torch.Tensor:
        """
        获取注意力偏置矩阵
        
        Args:
            seq_len: 序列长度
            
        Returns:
            注意力偏置矩阵 [seq_len, seq_len]
        """
        relative_positions = self._get_relative_positions(seq_len)
        
        # 转换为相对位置编码表的索引
        relative_position_indices = relative_positions + self.max_relative_position
        
        # 获取相对位置编码
        relative_embeddings = self.relative_pe[relative_position_indices]
        
        # 计算注意力偏置（这里简化为求和，实际实现可能更复杂）
        attention_bias = relative_embeddings.sum(dim=-1)
        
        return attention_bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            处理后的张量 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 获取相对位置矩阵
        relative_positions = self._get_relative_positions(seq_len)
        relative_position_indices = relative_positions + self.max_relative_position
        
        # 获取相对位置编码
        relative_embeddings = self.relative_pe[relative_position_indices]  # [seq_len, seq_len, d_model]
        
        # 应用相对位置编码（简化实现）
        # 实际应用中，相对位置编码通常在注意力机制中使用
        # 这里我们简单地将其加到输入上作为演示
        position_encoding = relative_embeddings.mean(dim=1)  # [seq_len, d_model]
        x = x + position_encoding.unsqueeze(0)
        
        return self.dropout(x)


class AdaptivePositionalEncoding(nn.Module):
    """
    自适应位置编码
    结合多种位置编码方式，可以根据需要选择或组合使用
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, 
                 max_relative_position: int = 128,
                 encoding_type: str = 'sinusoidal',
                 dropout: float = 0.1):
        """
        初始化自适应位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            max_relative_position: 最大相对位置距离
            encoding_type: 编码类型 ('sinusoidal', 'learnable', 'relative', 'hybrid')
            dropout: dropout概率
        """
        super().__init__()
        self.encoding_type = encoding_type
        self.d_model = d_model
        
        if encoding_type == 'sinusoidal':
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        elif encoding_type == 'learnable':
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_len, dropout)
        elif encoding_type == 'relative':
            self.pos_encoding = RelativePositionalEncoding(d_model, max_relative_position, dropout)
        elif encoding_type == 'hybrid':
            # 混合编码：结合正弦余弦和可学习编码
            self.sinusoidal_pe = PositionalEncoding(d_model, max_len, 0.0)
            self.learnable_pe = LearnablePositionalEncoding(d_model, max_len, 0.0)
            self.dropout = nn.Dropout(p=dropout)
            self.mix_weight = nn.Parameter(torch.tensor(0.5))
        else:
            raise ValueError(f"不支持的编码类型: {encoding_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        if self.encoding_type == 'hybrid':
            # 混合编码
            sin_encoded = self.sinusoidal_pe(x.clone())
            learn_encoded = self.learnable_pe(x.clone())
            
            # 加权组合
            weight = torch.sigmoid(self.mix_weight)
            x = weight * sin_encoded + (1 - weight) * learn_encoded
            return self.dropout(x)
        else:
            return self.pos_encoding(x)
    
    def get_attention_bias(self, seq_len: int) -> Optional[torch.Tensor]:
        """
        获取注意力偏置（仅对相对位置编码有效）
        
        Args:
            seq_len: 序列长度
            
        Returns:
            注意力偏置矩阵或None
        """
        if self.encoding_type == 'relative':
            return self.pos_encoding.get_attention_bias(seq_len)
        return None