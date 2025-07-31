"""
Transformer编码器使用示例
演示如何使用时序Transformer处理金融数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.rl_trading_system.models.transformer import TimeSeriesTransformer, TransformerConfig


def main():
    """主函数：演示Transformer编码器的使用"""
    
    print("=== Transformer编码器使用示例 ===\n")
    
    # 1. 创建配置
    config = TransformerConfig(
        d_model=256,        # 模型维度
        n_heads=8,          # 注意力头数
        n_layers=6,         # 编码器层数
        d_ff=1024,         # 前馈网络维度
        dropout=0.1,        # dropout概率
        max_seq_len=252,    # 最大序列长度（一年交易日）
        n_features=50,      # 输入特征数
        activation='gelu'   # 激活函数
    )
    
    print(f"配置信息:")
    print(f"  模型维度: {config.d_model}")
    print(f"  注意力头数: {config.n_heads}")
    print(f"  编码器层数: {config.n_layers}")
    print(f"  特征数: {config.n_features}")
    print()
    
    # 2. 创建模型
    transformer = TimeSeriesTransformer(config)
    transformer.eval()  # 设置为评估模式
    
    # 打印模型信息
    model_info = transformer.get_model_size()
    print(f"模型信息:")
    print(f"  总参数数: {model_info['total_parameters']:,}")
    print(f"  可训练参数数: {model_info['trainable_parameters']:,}")
    print(f"  模型大小: {model_info['model_size_mb']:.2f} MB")
    print()
    
    # 3. 创建示例数据
    batch_size = 2      # 批次大小
    seq_len = 60        # 序列长度（60个交易日）
    n_stocks = 10       # 股票数量
    n_features = 50     # 特征数量
    
    # 模拟金融时序数据
    # 形状: [batch_size, seq_len, n_stocks, n_features]
    x = torch.randn(batch_size, seq_len, n_stocks, n_features)
    
    print(f"输入数据形状: {x.shape}")
    print(f"  批次大小: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  股票数量: {n_stocks}")
    print(f"  特征数量: {n_features}")
    print()
    
    # 4. 前向传播
    with torch.no_grad():
        output = transformer(x)
    
    print(f"输出形状: {output.shape}")
    print(f"  批次大小: {output.shape[0]}")
    print(f"  股票数量: {output.shape[1]}")
    print(f"  编码维度: {output.shape[2]}")
    print()
    
    # 5. 演示带掩码的处理
    print("=== 带掩码的处理 ===")
    
    # 创建序列掩码（掩盖最后10个时间步）
    mask = torch.zeros(batch_size, seq_len)
    mask[:, -10:] = float('-inf')  # 掩盖最后10个时间步
    
    with torch.no_grad():
        masked_output = transformer(x, mask=mask)
    
    print(f"掩码形状: {mask.shape}")
    print(f"掩码输出形状: {masked_output.shape}")
    print(f"输出是否相同: {torch.allclose(output, masked_output)}")
    print()
    
    # 6. 演示注意力权重可视化
    print("=== 注意力权重可视化 ===")
    
    with torch.no_grad():
        attention_info = transformer.get_attention_weights(x[:1])  # 只取第一个样本
    
    print(f"时间注意力权重信息:")
    print(f"  股票数量: {attention_info['n_stocks']}")
    print(f"  序列长度: {attention_info['seq_len']}")
    print(f"  注意力头数: {len(attention_info['temporal_attentions'][0])}")
    print()
    
    # 7. 演示不同输入尺寸的处理
    print("=== 不同输入尺寸的处理 ===")
    
    test_cases = [
        (1, 30, 5),    # 小批次，短序列，少股票
        (4, 120, 20),  # 大批次，长序列，多股票
    ]
    
    for batch_size, seq_len, n_stocks in test_cases:
        test_x = torch.randn(batch_size, seq_len, n_stocks, n_features)
        
        with torch.no_grad():
            test_output = transformer(test_x)
        
        print(f"输入 {test_x.shape} -> 输出 {test_output.shape}")
    
    print()
    
    # 8. 演示序列编码（不进行时间聚合）
    print("=== 序列编码（不聚合）===")
    
    x_small = torch.randn(1, 30, 5, n_features)
    
    with torch.no_grad():
        encoded_sequence = transformer.encode_sequence(x_small)
    
    print(f"输入形状: {x_small.shape}")
    print(f"编码序列形状: {encoded_sequence.shape}")
    print("注意：编码序列保持了时间维度，没有进行聚合")
    print()
    
    # 9. 性能测试
    print("=== 性能测试 ===")
    
    import time
    
    test_x = torch.randn(2, 60, 10, n_features)
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = transformer(test_x)
    
    # 测试推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = transformer(test_x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"每秒可处理样本数: {1/avg_time:.1f}")
    print()
    
    print("=== 示例完成 ===")


if __name__ == "__main__":
    main()