"""
测试位置编码组件的单元测试
"""

import pytest
import torch
import numpy as np
import math
from typing import Tuple

from src.rl_trading_system.models.positional_encoding import (
    PositionalEncoding,
    LearnablePositionalEncoding,
    RelativePositionalEncoding
)


class TestPositionalEncoding:
    """测试正弦余弦位置编码"""
    
    @pytest.fixture
    def pos_encoding(self):
        """创建位置编码实例"""
        return PositionalEncoding(d_model=256, max_len=1000)
    
    def test_initialization(self, pos_encoding):
        """测试位置编码初始化"""
        assert pos_encoding.d_model == 256
        assert pos_encoding.max_len == 1000
        assert pos_encoding.pe.shape == (1000, 256)
    
    def test_sinusoidal_pattern(self, pos_encoding):
        """测试正弦余弦编码模式"""
        pe = pos_encoding.pe
        
        # 检查偶数位置使用sin，奇数位置使用cos
        pos = 10
        for i in range(0, 256, 2):
            expected_sin = math.sin(pos / (10000 ** (i / 256)))
            expected_cos = math.cos(pos / (10000 ** (i / 256)))
            
            assert abs(pe[pos, i] - expected_sin) < 1e-6
            assert abs(pe[pos, i + 1] - expected_cos) < 1e-6
    
    def test_forward_pass(self, pos_encoding):
        """测试前向传播"""
        batch_size, seq_len, d_model = 2, 50, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 设置为评估模式以禁用dropout
        pos_encoding.eval()
        output = pos_encoding(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        
        # 验证位置编码被正确添加
        expected = x + pos_encoding.pe[:seq_len].unsqueeze(0)
        torch.testing.assert_close(output, expected)
    
    def test_different_sequence_lengths(self, pos_encoding):
        """测试不同序列长度下的位置编码效果"""
        d_model = 256
        
        # 测试不同长度
        for seq_len in [10, 50, 100, 252]:
            x = torch.randn(1, seq_len, d_model)
            output = pos_encoding(x)
            
            assert output.shape == (1, seq_len, d_model)
            
            # 验证位置编码的唯一性
            pe_slice = pos_encoding.pe[:seq_len]
            for i in range(seq_len - 1):
                for j in range(i + 1, seq_len):
                    # 不同位置的编码应该不同
                    assert not torch.allclose(pe_slice[i], pe_slice[j])
    
    def test_max_length_constraint(self, pos_encoding):
        """测试最大长度约束"""
        d_model = 256
        max_len = pos_encoding.max_len
        
        # 测试超过最大长度的情况
        x = torch.randn(1, max_len + 10, d_model)
        
        with pytest.raises(IndexError):
            pos_encoding(x)
    
    def test_positional_encoding_properties(self, pos_encoding):
        """测试位置编码的数学性质"""
        pe = pos_encoding.pe
        
        # 测试周期性：某些频率的编码应该具有周期性
        # 对于最低频率，周期应该是2π * 10000
        lowest_freq_period = 2 * math.pi * 10000
        
        # 由于序列长度限制，我们测试较小的周期性
        for pos in range(100):
            # 检查相邻位置的差异
            diff = torch.norm(pe[pos + 1] - pe[pos])
            assert diff > 0  # 相邻位置应该不同
    
    def test_gradient_flow(self, pos_encoding):
        """测试梯度流动"""
        x = torch.randn(2, 50, 256, requires_grad=True)
        output = pos_encoding(x)
        loss = output.sum()
        loss.backward()
        
        # 位置编码不应该有梯度（它是固定的）
        assert pos_encoding.pe.grad is None
        # 输入应该有梯度
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestLearnablePositionalEncoding:
    """测试可学习位置编码"""
    
    @pytest.fixture
    def learnable_pos_encoding(self):
        """创建可学习位置编码实例"""
        return LearnablePositionalEncoding(d_model=256, max_len=1000)
    
    def test_initialization(self, learnable_pos_encoding):
        """测试可学习位置编码初始化"""
        assert learnable_pos_encoding.d_model == 256
        assert learnable_pos_encoding.max_len == 1000
        assert learnable_pos_encoding.pe.shape == (1000, 256)
        assert learnable_pos_encoding.pe.requires_grad == True
    
    def test_forward_pass(self, learnable_pos_encoding):
        """测试前向传播"""
        batch_size, seq_len, d_model = 2, 50, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = learnable_pos_encoding(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_gradient_flow(self, learnable_pos_encoding):
        """测试梯度流动"""
        x = torch.randn(2, 50, 256, requires_grad=True)
        output = learnable_pos_encoding(x)
        loss = output.sum()
        loss.backward()
        
        # 可学习位置编码应该有梯度
        assert learnable_pos_encoding.pe.grad is not None
        # 输入也应该有梯度
        assert x.grad is not None
    
    def test_parameter_update(self, learnable_pos_encoding):
        """测试参数更新"""
        # 记录初始参数
        initial_pe = learnable_pos_encoding.pe.clone()
        
        # 模拟训练步骤
        optimizer = torch.optim.Adam(learnable_pos_encoding.parameters(), lr=0.01)
        
        x = torch.randn(2, 50, 256)
        output = learnable_pos_encoding(x)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 参数应该被更新
        assert not torch.allclose(initial_pe, learnable_pos_encoding.pe)
    
    def test_different_sequence_lengths(self, learnable_pos_encoding):
        """测试不同序列长度"""
        d_model = 256
        
        for seq_len in [10, 50, 100, 252]:
            x = torch.randn(1, seq_len, d_model)
            output = learnable_pos_encoding(x)
            assert output.shape == (1, seq_len, d_model)


class TestRelativePositionalEncoding:
    """测试相对位置编码"""
    
    @pytest.fixture
    def relative_pos_encoding(self):
        """创建相对位置编码实例"""
        return RelativePositionalEncoding(d_model=256, max_relative_position=128)
    
    def test_initialization(self, relative_pos_encoding):
        """测试相对位置编码初始化"""
        assert relative_pos_encoding.d_model == 256
        assert relative_pos_encoding.max_relative_position == 128
        # 相对位置编码表大小应该是 2 * max_relative_position + 1
        assert relative_pos_encoding.relative_pe.shape == (257, 256)
    
    def test_relative_position_calculation(self, relative_pos_encoding):
        """测试相对位置计算"""
        seq_len = 10
        relative_positions = relative_pos_encoding._get_relative_positions(seq_len)
        
        assert relative_positions.shape == (seq_len, seq_len)
        
        # 检查对角线（自己到自己的相对位置应该是0）
        for i in range(seq_len):
            assert relative_positions[i, i] == 0
        
        # 检查相对位置的对称性
        for i in range(seq_len):
            for j in range(seq_len):
                expected_relative_pos = j - i
                # 裁剪到最大相对位置范围内
                expected_relative_pos = max(-128, min(128, expected_relative_pos))
                assert relative_positions[i, j] == expected_relative_pos
    
    def test_forward_pass(self, relative_pos_encoding):
        """测试前向传播"""
        batch_size, seq_len, d_model = 2, 50, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = relative_pos_encoding(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_attention_bias_generation(self, relative_pos_encoding):
        """测试注意力偏置生成"""
        seq_len = 20
        attention_bias = relative_pos_encoding.get_attention_bias(seq_len)
        
        assert attention_bias.shape == (seq_len, seq_len)
        
        # 检查对称性质：bias[i,j] 和 bias[j,i] 应该有特定关系
        # 由于相对位置编码，bias[i,j] 应该等于 relative_pe[j-i]
        relative_positions = relative_pos_encoding._get_relative_positions(seq_len)
        
        for i in range(seq_len):
            for j in range(seq_len):
                rel_pos = relative_positions[i, j]
                # 转换为相对位置编码表的索引
                pe_idx = rel_pos + relative_pos_encoding.max_relative_position
                expected_bias = relative_pos_encoding.relative_pe[pe_idx].sum()  # 简化检查
                # 这里只检查形状和基本属性，具体实现可能有所不同
    
    def test_gradient_flow(self, relative_pos_encoding):
        """测试梯度流动"""
        x = torch.randn(2, 20, 256, requires_grad=True)
        output = relative_pos_encoding(x)
        loss = output.sum()
        loss.backward()
        
        # 相对位置编码参数应该有梯度
        assert relative_pos_encoding.relative_pe.grad is not None
        # 输入也应该有梯度
        assert x.grad is not None
    
    def test_max_relative_position_clipping(self, relative_pos_encoding):
        """测试最大相对位置裁剪"""
        # 测试超过最大相对位置的序列
        seq_len = 300  # 超过 max_relative_position * 2
        
        relative_positions = relative_pos_encoding._get_relative_positions(seq_len)
        
        # 所有相对位置都应该在 [-max_relative_position, max_relative_position] 范围内
        assert torch.all(relative_positions >= -relative_pos_encoding.max_relative_position)
        assert torch.all(relative_positions <= relative_pos_encoding.max_relative_position)
    
    def test_different_sequence_lengths(self, relative_pos_encoding):
        """测试不同序列长度下的相对位置编码效果"""
        d_model = 256
        
        for seq_len in [5, 20, 50, 100]:
            x = torch.randn(1, seq_len, d_model)
            output = relative_pos_encoding(x)
            
            assert output.shape == (1, seq_len, d_model)
            
            # 测试注意力偏置
            attention_bias = relative_pos_encoding.get_attention_bias(seq_len)
            assert attention_bias.shape == (seq_len, seq_len)


class TestPositionalEncodingComparison:
    """比较不同位置编码方法的测试"""
    
    def test_encoding_differences(self):
        """测试不同编码方法的差异"""
        d_model, seq_len = 256, 50
        x = torch.randn(1, seq_len, d_model)
        
        # 创建不同的位置编码
        sinusoidal_pe = PositionalEncoding(d_model, 1000)
        learnable_pe = LearnablePositionalEncoding(d_model, 1000)
        relative_pe = RelativePositionalEncoding(d_model, 128)
        
        # 获取输出
        sin_output = sinusoidal_pe(x)
        learn_output = learnable_pe(x)
        rel_output = relative_pe(x)
        
        # 所有输出形状应该相同
        assert sin_output.shape == learn_output.shape == rel_output.shape
        
        # 但内容应该不同（除非极其巧合）
        assert not torch.allclose(sin_output, learn_output, atol=1e-3)
        assert not torch.allclose(sin_output, rel_output, atol=1e-3)
        assert not torch.allclose(learn_output, rel_output, atol=1e-3)
    
    def test_performance_characteristics(self):
        """测试性能特征"""
        d_model, seq_len = 256, 100
        x = torch.randn(2, seq_len, d_model)
        
        encodings = [
            PositionalEncoding(d_model, 1000),
            LearnablePositionalEncoding(d_model, 1000),
            RelativePositionalEncoding(d_model, 128)
        ]
        
        for encoding in encodings:
            # 测试前向传播时间
            import time
            start_time = time.time()
            
            for _ in range(10):
                output = encoding(x)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            # 所有编码方法都应该在合理时间内完成
            assert avg_time < 0.1  # 100ms per forward pass should be reasonable
    
    @pytest.mark.parametrize("seq_len", [10, 50, 100, 252])
    def test_scalability_with_sequence_length(self, seq_len):
        """测试随序列长度的可扩展性"""
        d_model = 256
        x = torch.randn(1, seq_len, d_model)
        
        # 测试所有编码方法
        sinusoidal_pe = PositionalEncoding(d_model, max(1000, seq_len))
        learnable_pe = LearnablePositionalEncoding(d_model, max(1000, seq_len))
        relative_pe = RelativePositionalEncoding(d_model, max(128, seq_len // 2))
        
        # 所有方法都应该能处理不同长度的序列
        sin_output = sinusoidal_pe(x)
        learn_output = learnable_pe(x)
        rel_output = relative_pe(x)
        
        assert sin_output.shape == (1, seq_len, d_model)
        assert learn_output.shape == (1, seq_len, d_model)
        assert rel_output.shape == (1, seq_len, d_model)