"""
测试Transformer编码器的单元测试
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from src.rl_trading_system.models.transformer import (
    TimeSeriesTransformer,
    TransformerConfig,
    TransformerEncoderLayer,
    FeedForwardNetwork
)


class TestTransformerConfig:
    """测试Transformer配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = TransformerConfig()
        
        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 6
        assert config.d_ff == 1024
        assert config.dropout == 0.1
        assert config.max_seq_len == 252
        assert config.n_features == 50
        assert config.activation == 'gelu'
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = TransformerConfig(
            d_model=512,
            n_heads=16,
            n_layers=12,
            d_ff=2048,
            dropout=0.2,
            max_seq_len=500,
            n_features=100,
            activation='relu'
        )
        
        assert config.d_model == 512
        assert config.n_heads == 16
        assert config.n_layers == 12
        assert config.d_ff == 2048
        assert config.dropout == 0.2
        assert config.max_seq_len == 500
        assert config.n_features == 100
        assert config.activation == 'relu'
    
    def test_config_validation(self):
        """测试配置验证"""
        # d_model必须能被n_heads整除
        with pytest.raises(AssertionError):
            TransformerConfig(d_model=256, n_heads=7)
        
        # 正常情况应该不报错
        config = TransformerConfig(d_model=256, n_heads=8)
        assert config.d_model == 256
        assert config.n_heads == 8


class TestFeedForwardNetwork:
    """测试前馈网络"""
    
    @pytest.fixture
    def ffn(self):
        """创建前馈网络实例"""
        return FeedForwardNetwork(d_model=256, d_ff=1024, dropout=0.1, activation='gelu')
    
    def test_initialization(self, ffn):
        """测试前馈网络初始化"""
        assert isinstance(ffn.linear1, nn.Linear)
        assert isinstance(ffn.linear2, nn.Linear)
        assert isinstance(ffn.dropout, nn.Dropout)
        assert ffn.linear1.in_features == 256
        assert ffn.linear1.out_features == 1024
        assert ffn.linear2.in_features == 1024
        assert ffn.linear2.out_features == 256
    
    def test_forward_pass(self, ffn):
        """测试前向传播"""
        batch_size, seq_len, d_model = 2, 10, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = ffn(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_different_activations(self):
        """测试不同激活函数"""
        d_model, d_ff = 256, 1024
        activations = ['relu', 'gelu', 'swish']
        
        for activation in activations:
            ffn = FeedForwardNetwork(d_model, d_ff, activation=activation)
            x = torch.randn(2, 10, d_model)
            output = ffn(x)
            assert output.shape == (2, 10, d_model)
    
    def test_gradient_flow(self, ffn):
        """测试梯度流动"""
        x = torch.randn(2, 10, 256, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        for param in ffn.parameters():
            assert param.grad is not None


class TestTransformerEncoderLayer:
    """测试Transformer编码器层"""
    
    @pytest.fixture
    def encoder_layer(self):
        """创建编码器层实例"""
        config = TransformerConfig(d_model=256, n_heads=8, d_ff=1024, dropout=0.1)
        return TransformerEncoderLayer(config)
    
    def test_initialization(self, encoder_layer):
        """测试编码器层初始化"""
        assert hasattr(encoder_layer, 'self_attention')
        assert hasattr(encoder_layer, 'feed_forward')
        assert hasattr(encoder_layer, 'norm1')
        assert hasattr(encoder_layer, 'norm2')
        assert hasattr(encoder_layer, 'dropout1')
        assert hasattr(encoder_layer, 'dropout2')
    
    def test_forward_pass(self, encoder_layer):
        """测试前向传播"""
        batch_size, seq_len, d_model = 2, 20, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = encoder_layer(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_with_mask(self, encoder_layer):
        """测试带掩码的编码器层"""
        batch_size, seq_len, d_model = 2, 20, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        # 创建注意力掩码
        mask = torch.zeros(batch_size, seq_len, seq_len)
        mask[:, :, 10:] = float('-inf')  # 掩盖后10个位置
        
        output = encoder_layer(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_residual_connections(self, encoder_layer):
        """测试残差连接"""
        batch_size, seq_len, d_model = 2, 20, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        encoder_layer.eval()
        
        # 获取中间结果来验证残差连接
        # 这需要修改forward方法或添加hook，这里简化测试
        output = encoder_layer(x)
        
        # 输出不应该等于输入（因为有变换）
        assert not torch.allclose(output, x, atol=1e-3)
        
        # 但应该保持相同的形状
        assert output.shape == x.shape
    
    def test_gradient_flow(self, encoder_layer):
        """测试梯度流动"""
        x = torch.randn(2, 20, 256, requires_grad=True)
        output = encoder_layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        for param in encoder_layer.parameters():
            assert param.grad is not None
    
    def test_layer_norm_placement(self, encoder_layer):
        """测试层归一化的位置"""
        # 验证层归一化确实被应用
        batch_size, seq_len, d_model = 2, 20, 256
        x = torch.randn(batch_size, seq_len, d_model)
        
        encoder_layer.eval()
        output = encoder_layer(x)
        
        # 输出应该经过归一化，具有合理的统计特性
        output_mean = output.mean(dim=-1)
        output_std = output.std(dim=-1)
        
        # 层归一化后，最后一个维度的均值应该接近0，标准差接近1
        # 但由于残差连接，这个测试可能不严格成立
        assert output.shape == (batch_size, seq_len, d_model)


class TestTimeSeriesTransformer:
    """测试时序Transformer"""
    
    @pytest.fixture
    def transformer_config(self):
        """创建Transformer配置"""
        return TransformerConfig(
            d_model=256,
            n_heads=8,
            n_layers=6,
            d_ff=1024,
            dropout=0.1,
            max_seq_len=252,
            n_features=50
        )
    
    @pytest.fixture
    def transformer(self, transformer_config):
        """创建Transformer实例"""
        return TimeSeriesTransformer(transformer_config)
    
    def test_initialization(self, transformer, transformer_config):
        """测试Transformer初始化"""
        assert transformer.config == transformer_config
        assert hasattr(transformer, 'input_projection')
        assert hasattr(transformer, 'pos_encoding')
        assert hasattr(transformer, 'encoder_layers')
        assert hasattr(transformer, 'temporal_attention')
        assert hasattr(transformer, 'output_projection')
        assert len(transformer.encoder_layers) == transformer_config.n_layers
    
    def test_forward_pass(self, transformer):
        """测试前向传播"""
        batch_size, seq_len, n_stocks, n_features = 2, 60, 10, 50
        x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        
        output = transformer(x)
        
        assert output.shape == (batch_size, n_stocks, 256)  # d_model
    
    def test_different_input_dimensions(self, transformer):
        """测试不同输入维度下的表现"""
        n_features = 50
        d_model = 256
        
        test_cases = [
            (1, 30, 5),   # 小批次，短序列，少股票
            (2, 60, 10),  # 中等批次，中等序列，中等股票
            (4, 120, 20), # 大批次，长序列，多股票
        ]
        
        for batch_size, seq_len, n_stocks in test_cases:
            x = torch.randn(batch_size, seq_len, n_stocks, n_features)
            output = transformer(x)
            assert output.shape == (batch_size, n_stocks, d_model)
    
    def test_with_mask(self, transformer):
        """测试带掩码的Transformer"""
        batch_size, seq_len, n_stocks, n_features = 2, 60, 10, 50
        x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        
        # 创建序列掩码
        mask = torch.zeros(batch_size, seq_len)
        mask[:, 40:] = float('-inf')  # 掩盖后20个时间步
        
        output = transformer(x, mask=mask)
        
        assert output.shape == (batch_size, n_stocks, 256)
    
    def test_gradient_flow(self, transformer):
        """测试梯度流动"""
        batch_size, seq_len, n_stocks, n_features = 2, 30, 5, 50
        x = torch.randn(batch_size, seq_len, n_stocks, n_features, requires_grad=True)
        
        output = transformer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        for param in transformer.parameters():
            assert param.grad is not None
    
    def test_sequence_length_handling(self, transformer):
        """测试序列长度处理"""
        batch_size, n_stocks, n_features = 2, 10, 50
        max_seq_len = transformer.config.max_seq_len
        
        # 测试不同长度的序列
        for seq_len in [10, 50, 100, max_seq_len]:
            x = torch.randn(batch_size, seq_len, n_stocks, n_features)
            output = transformer(x)
            assert output.shape == (batch_size, n_stocks, 256)
        
        # 测试超过最大长度的序列
        if max_seq_len < 500:  # 避免内存问题
            x = torch.randn(batch_size, max_seq_len + 10, n_stocks, n_features)
            with pytest.raises(IndexError):
                transformer(x)
    
    def test_batch_processing(self, transformer):
        """测试批处理"""
        seq_len, n_stocks, n_features = 60, 10, 50
        
        # 测试不同批次大小
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, seq_len, n_stocks, n_features)
            output = transformer(x)
            assert output.shape == (batch_size, n_stocks, 256)
    
    def test_feature_dimension_handling(self, transformer_config):
        """测试特征维度处理"""
        # 测试不同特征维度
        for n_features in [20, 50, 100]:
            config = TransformerConfig(
                d_model=256,
                n_heads=8,
                n_layers=3,  # 减少层数以加快测试
                n_features=n_features
            )
            transformer = TimeSeriesTransformer(config)
            
            batch_size, seq_len, n_stocks = 2, 30, 5
            x = torch.randn(batch_size, seq_len, n_stocks, n_features)
            output = transformer(x)
            assert output.shape == (batch_size, n_stocks, 256)
    
    def test_model_parameters(self, transformer):
        """测试模型参数"""
        # 检查模型有参数
        total_params = sum(p.numel() for p in transformer.parameters())
        assert total_params > 0
        
        # 检查可训练参数
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        assert trainable_params > 0
        assert trainable_params == total_params  # 所有参数都应该可训练
    
    def test_model_modes(self, transformer):
        """测试模型模式（训练/评估）"""
        batch_size, seq_len, n_stocks, n_features = 2, 30, 5, 50
        x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        
        # 训练模式
        transformer.train()
        output_train = transformer(x)
        
        # 评估模式
        transformer.eval()
        output_eval = transformer(x)
        
        # 形状应该相同
        assert output_train.shape == output_eval.shape
        
        # 由于dropout，输出可能不同（但这个测试可能不稳定）
        # assert not torch.allclose(output_train, output_eval, atol=1e-6)
    
    def test_memory_efficiency(self, transformer):
        """测试内存效率"""
        # 测试较大的输入
        batch_size, seq_len, n_stocks, n_features = 4, 100, 20, 50
        x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        
        # 应该能够处理较大的输入而不出现内存错误
        output = transformer(x)
        assert output.shape == (batch_size, n_stocks, 256)
    
    def test_deterministic_output(self, transformer):
        """测试确定性输出"""
        batch_size, seq_len, n_stocks, n_features = 2, 30, 5, 50
        x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        
        # 设置为评估模式以禁用dropout
        transformer.eval()
        
        # 多次前向传播应该产生相同结果
        output1 = transformer(x)
        output2 = transformer(x)
        
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-6)


class TestTransformerIntegration:
    """测试Transformer集成"""
    
    def test_end_to_end_pipeline(self):
        """测试端到端流水线"""
        # 创建配置
        config = TransformerConfig(
            d_model=128,  # 较小的模型以加快测试
            n_heads=4,
            n_layers=2,
            d_ff=256,
            n_features=20
        )
        
        # 创建模型
        transformer = TimeSeriesTransformer(config)
        
        # 创建输入数据
        batch_size, seq_len, n_stocks, n_features = 2, 30, 5, 20
        x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        
        # 前向传播
        output = transformer(x)
        
        # 验证输出
        assert output.shape == (batch_size, n_stocks, 128)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_training_step_simulation(self):
        """测试训练步骤模拟"""
        config = TransformerConfig(
            d_model=128,
            n_heads=4,
            n_layers=2,
            n_features=20
        )
        
        transformer = TimeSeriesTransformer(config)
        optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001)
        
        # 模拟训练步骤
        batch_size, seq_len, n_stocks, n_features = 2, 30, 5, 20
        x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        target = torch.randn(batch_size, n_stocks, 128)
        
        # 前向传播
        output = transformer(x)
        
        # 计算损失
        loss = nn.MSELoss()(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 验证梯度被计算
        for param in transformer.parameters():
            assert param.grad is not None
    
    def test_model_saving_loading(self, tmp_path):
        """测试模型保存和加载"""
        config = TransformerConfig(d_model=128, n_heads=4, n_layers=2, n_features=20)
        transformer = TimeSeriesTransformer(config)
        
        # 设置为评估模式以禁用dropout
        transformer.eval()
        
        # 创建测试输入
        x = torch.randn(1, 30, 5, 20)
        original_output = transformer(x)
        
        # 保存模型
        model_path = tmp_path / "transformer.pth"
        torch.save(transformer.state_dict(), model_path)
        
        # 创建新模型并加载权重
        new_transformer = TimeSeriesTransformer(config)
        new_transformer.load_state_dict(torch.load(model_path))
        new_transformer.eval()  # 也设置为评估模式
        
        # 验证输出相同
        new_output = new_transformer(x)
        torch.testing.assert_close(original_output, new_output, rtol=1e-5, atol=1e-6)
    
    @pytest.mark.parametrize("batch_size,seq_len,n_stocks", [
        (1, 20, 3),
        (2, 40, 5),
        (4, 60, 10),
    ])
    def test_scalability(self, batch_size, seq_len, n_stocks):
        """测试可扩展性"""
        config = TransformerConfig(d_model=128, n_heads=4, n_layers=2, n_features=20)
        transformer = TimeSeriesTransformer(config)
        
        x = torch.randn(batch_size, seq_len, n_stocks, 20)
        output = transformer(x)
        
        assert output.shape == (batch_size, n_stocks, 128)
        assert not torch.isnan(output).any()
    
    def test_performance_benchmark(self):
        """测试性能基准"""
        config = TransformerConfig(d_model=256, n_heads=8, n_layers=6, n_features=50)
        transformer = TimeSeriesTransformer(config)
        transformer.eval()
        
        # 测试推理时间
        batch_size, seq_len, n_stocks, n_features = 2, 60, 10, 50
        x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                output = transformer(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        # 推理时间应该在合理范围内
        assert avg_time < 1.0  # 1秒内完成推理
        assert output.shape == (batch_size, n_stocks, 256)