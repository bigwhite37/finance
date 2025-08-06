"""
TimesNet特征编码器
基于Transformer的金融时序特征提取，支持LoRA微调
参考claude.md中的TimesNetEncoder实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 残差连接的输入
        residual = query
        
        # 线性变换和重塑
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.w_o(context)
        
        # 残差连接和层归一化
        return self.layer_norm(output + residual)


class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.layer_norm(x + residual)


class InformerBlock(nn.Module):
    """Informer块，TimesNet的核心组件"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        x = self.self_attn(x, x, x, mask)
        # 前馈网络
        x = self.feed_forward(x)
        return x


class TimesNetEncoder(nn.Module):
    """
    TimesNet特征编码器
    金融时序数据的Transformer编码，带LoRA微调
    参考claude.md实现细节
    """
    
    def __init__(self, 
                 input_dim: int = 100, 
                 d_model: int = 64, 
                 n_heads: int = 4, 
                 n_layers: int = 2,
                 d_ff: int = 256,
                 lora_r: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 252):  # 一年交易日
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.lora_r = lora_r
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 多尺度时序卷积 - 提取不同时间尺度的特征
        conv_kernels = [3, 5, 7, 9]  # 对应3日、5日、7日、9日的时间模式
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(input_dim, d_model // len(conv_kernels), 
                     kernel_size=k, padding=k//2)
            for k in conv_kernels
        ])
        
        # TimesNet核心层
        self.transformer_blocks = nn.ModuleList([
            InformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # LoRA适配器参数
        self.lora_enabled = lora_r > 0
        if self.lora_enabled:
            # 为每个Transformer层添加LoRA参数
            self.lora_A = nn.ParameterList([
                nn.Parameter(torch.randn(d_model, lora_r) * 0.01)
                for _ in range(n_layers)
            ])
            self.lora_B = nn.ParameterList([
                nn.Parameter(torch.zeros(lora_r, d_model))
                for _ in range(n_layers)
            ])
            self.lora_scale = 1.0
        
        # 输出层
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            timestamps: 时间戳 [batch_size, seq_len] (可选)
            mask: 注意力掩码 [batch_size, seq_len] (可选)
            
        Returns:
            编码后的特征 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 多尺度卷积特征提取
        # 转置为 [batch, features, seq_len] 用于1D卷积
        x_transpose = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        multi_scale_features = []
        for conv in self.multi_scale_conv:
            # 应用1D卷积并转回 [batch, seq_len, channels]
            conv_out = conv(x_transpose).transpose(1, 2)
            multi_scale_features.append(conv_out)
        
        # 融合多尺度特征
        x_fused = torch.cat(multi_scale_features, dim=2)  # [batch, seq_len, d_model]
        
        # 2. 如果维度不匹配，通过线性投影调整
        if x_fused.size(-1) != self.d_model:
            x_fused = self.input_projection(x)
        
        # 3. 添加位置编码
        x_encoded = self.pos_encoding(x_fused)
        x_encoded = self.dropout(x_encoded)
        
        # 4. 通过Transformer层
        hidden_states = x_encoded
        for i, transformer_block in enumerate(self.transformer_blocks):
            # 标准Transformer处理
            hidden_states = transformer_block(hidden_states, mask)
            
            # 应用LoRA适配器
            if self.lora_enabled:
                # LoRA: h = h + (h @ A @ B) * scale
                lora_out = torch.matmul(
                    torch.matmul(hidden_states, self.lora_A[i]), 
                    self.lora_B[i]
                ) * self.lora_scale
                hidden_states = hidden_states + lora_out
        
        # 5. 最终归一化
        output = self.output_norm(hidden_states)
        
        return output
    
    def get_sequence_embedding(self, x: torch.Tensor, 
                             timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取序列级别的嵌入（用于下游任务）
        
        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            timestamps: 时间戳 (可选)
            
        Returns:
            序列嵌入 [batch_size, d_model]
        """
        # 获取完整的序列编码
        sequence_output = self.forward(x, timestamps)
        
        # 使用最后一个时间步的输出作为序列表示
        # 对于金融数据，最新的信息通常最重要
        sequence_embedding = sequence_output[:, -1, :]  # [batch_size, d_model]
        
        return sequence_embedding
    
    def freeze_base_model(self):
        """冻结基础模型参数，仅训练LoRA"""
        if not self.lora_enabled:
            return
        
        for name, param in self.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True


class MutualInfoEstimator(nn.Module):
    """
    互信息估计器
    用于DIAYN中估计I(a; z)，促进技能多样性
    """
    
    def __init__(self, skill_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.skill_dim = skill_dim
        self.action_dim = action_dim
        
        # 网络结构：将动作和技能向量映射到同一空间
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.skill_encoder = nn.Sequential(
            nn.Linear(skill_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 互信息估计头（InfoNCE）
        self.mi_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, actions: torch.Tensor, skills: torch.Tensor) -> torch.Tensor:
        """
        估计互信息 I(a; z)
        
        Args:
            actions: 动作 [batch_size, action_dim]
            skills: 技能向量 [batch_size, skill_dim]
            
        Returns:
            互信息估计值 [batch_size, 1]
        """
        # 编码动作和技能
        action_features = self.action_encoder(actions)  # [batch, hidden_dim]
        skill_features = self.skill_encoder(skills)     # [batch, hidden_dim]
        
        # 拼接特征
        combined_features = torch.cat([action_features, skill_features], dim=1)
        
        # 估计互信息
        mi_estimate = self.mi_head(combined_features)
        
        return mi_estimate
    
    def estimate_mi(self, actions: torch.Tensor, skills: torch.Tensor) -> float:
        """
        估计互信息（返回标量值）
        """
        with torch.no_grad():
            mi_tensor = self.forward(actions, skills)
            return mi_tensor.mean().item()


if __name__ == "__main__":
    # 测试TimesNet编码器
    batch_size = 32
    seq_len = 60  # 60个交易日
    input_dim = 100  # 100个特征
    
    # 创建模型
    encoder = TimesNetEncoder(
        input_dim=input_dim,
        d_model=64,
        n_heads=4,
        n_layers=2,
        lora_r=8
    )
    
    # 测试数据
    x = torch.randn(batch_size, seq_len, input_dim)
    timestamps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = encoder(x, timestamps)
    print(f"输出形状: {output.shape}")
    
    # 序列嵌入
    seq_embedding = encoder.get_sequence_embedding(x, timestamps)
    print(f"序列嵌入形状: {seq_embedding.shape}")
    
    # 测试互信息估计器
    mi_estimator = MutualInfoEstimator(skill_dim=10, action_dim=20)
    actions = torch.randn(batch_size, 20)
    skills = torch.randn(batch_size, 10)
    
    mi_estimate = mi_estimator(actions, skills)
    print(f"互信息估计形状: {mi_estimate.shape}")
    print(f"互信息估计值: {mi_estimator.estimate_mi(actions, skills):.4f}")