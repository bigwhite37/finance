# Transformer配置指南

本指南提供了针对A股量化交易系统的Transformer配置推荐。

## 配置概述

基于当前系统分析，推荐配置参数如下：

### 🎯 推荐配置 (生产环境)

```python
from rl_trading_system.models.transformer import TransformerConfig

# 标准配置：平衡性能与准确性
transformer_config = TransformerConfig(
    d_model=256,           # 模型维度
    n_heads=8,             # 注意力头数
    n_layers=6,            # 编码器层数
    d_ff=1024,             # 前馈网络维度
    dropout=0.1,           # Dropout率
    max_seq_len=252,       # 最大序列长度（一年交易日）
    n_features=37,         # 每只股票的特征数
    activation='gelu'      # 激活函数
)
```

### 🚀 高性能配置 (GPU充足)

```python
# 高容量配置：更强的表达能力
transformer_config = TransformerConfig(
    d_model=512,           # 更大的模型维度
    n_heads=16,            # 更多注意力头
    n_layers=8,            # 更深的网络
    d_ff=2048,             # 更大的前馈网络
    dropout=0.1,
    max_seq_len=252,
    n_features=37,
    activation='gelu'
)
```

### ⚡ 轻量配置 (资源受限)

```python
# 轻量配置：快速训练和推理
transformer_config = TransformerConfig(
    d_model=128,           # 较小的模型维度
    n_heads=4,             # 较少注意力头
    n_layers=3,            # 较浅的网络
    d_ff=512,              # 较小的前馈网络
    dropout=0.1,
    max_seq_len=252,
    n_features=37,
    activation='gelu'
)
```

## 参数详细说明

### 核心参数

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `d_model` | 模型维度，影响表达能力 | 128/256/512 | 更大=更强表达力，但计算成本高 |
| `n_heads` | 注意力头数，必须能被d_model整除 | 4/8/16 | 更多=更好的注意力多样性 |
| `n_layers` | 编码器层数，影响模型深度 | 3/6/8 | 更深=更强特征提取，但易过拟合 |
| `d_ff` | 前馈网络维度，通常是d_model的2-4倍 | d_model×2~4 | 影响非线性变换能力 |
| `n_features` | 每只股票的特征数 | 37 | 必须与特征工程输出一致 |

### 数据相关参数

| 参数 | 说明 | 推荐值 | 注意事项 |
|------|------|--------|----------|
| `max_seq_len` | 最大序列长度 | 252 | 对应一年交易日，影响内存使用 |
| `n_features` | 输入特征维度 | 37 | 必须与FeatureEngineer输出匹配 |

### 训练相关参数

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `dropout` | Dropout概率 | 0.1 | 防止过拟合，过高影响学习能力 |
| `activation` | 激活函数 | 'gelu' | GELU在Transformer中表现更好 |

## 使用示例

### 1. 在SAC配置中使用

```python
from rl_trading_system.models.sac_agent import SACConfig
from rl_trading_system.models.transformer import TransformerConfig

# 创建Transformer配置
transformer_config = TransformerConfig(
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    dropout=0.1,
    max_seq_len=252,
    n_features=37,
    activation='gelu'
)

# 创建SAC配置
sac_config = SACConfig(
    state_dim=256,                    # 必须与transformer的d_model一致
    action_dim=3,                     # 股票数量
    hidden_dim=512,
    use_transformer=True,
    transformer_config=transformer_config,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

### 2. 配置文件方式

创建 `config/transformer_config.yaml`:

```yaml
transformer:
  d_model: 256
  n_heads: 8
  n_layers: 6
  d_ff: 1024
  dropout: 0.1
  max_seq_len: 252
  n_features: 37
  activation: 'gelu'

sac:
  state_dim: 256  # 必须与transformer.d_model一致
  use_transformer: true
```

### 3. 环境变量配置

```bash
# 设置环境变量
export MODEL_TRANSFORMER_D_MODEL=256
export MODEL_TRANSFORMER_N_HEADS=8
export MODEL_TRANSFORMER_N_LAYERS=6
export MODEL_TRANSFORMER_D_FF=1024
export MODEL_TRANSFORMER_N_FEATURES=37
export MODEL_SAC_STATE_DIM=256
```

## 配置优化建议

### 🎯 性能调优

1. **内存优化**：
   - 减少`max_seq_len`和`d_model`可降低内存使用
   - 使用梯度检查点技术处理大模型

2. **训练稳定性**：
   - 初期使用较小的`dropout`(0.05-0.1)
   - 逐步增加模型复杂度

3. **推理速度**：
   - 生产环境优先考虑`d_model=128`或`256`
   - 减少`n_layers`可显著提升推理速度

### ⚠️ 常见问题

1. **维度不匹配**：
   ```python
   # 确保SAC的state_dim与Transformer的d_model一致
   assert sac_config.state_dim == transformer_config.d_model
   ```

2. **特征数不匹配**：
   ```python
   # 确保特征数与FeatureEngineer输出一致
   # 当前系统：37个特征/股票
   assert transformer_config.n_features == 37
   ```

3. **注意力头数问题**：
   ```python
   # d_model必须能被n_heads整除
   assert transformer_config.d_model % transformer_config.n_heads == 0
   ```

## 模型复杂度对比

| 配置类型 | 参数量 | 训练时间 | 推理速度 | 内存使用 | 性能 |
|----------|--------|----------|----------|----------|------|
| 轻量 | ~0.5M | 快 | 很快 | 低 | 良好 |
| 标准 | ~2M | 中等 | 快 | 中等 | 很好 |
| 高性能 | ~8M | 慢 | 中等 | 高 | 优秀 |

## 最佳实践

1. **开发阶段**：使用轻量配置快速迭代
2. **训练阶段**：使用标准配置获得良好性能
3. **生产部署**：根据硬件条件选择合适配置
4. **模型保存**：确保保存完整的transformer_config
5. **配置验证**：使用配置验证脚本检查参数合理性

## 验证脚本

```python
def validate_transformer_config(transformer_config, sac_config):
    """验证Transformer配置的合理性"""
    
    # 基本验证
    assert transformer_config.d_model > 0, "d_model必须大于0"
    assert transformer_config.n_heads > 0, "n_heads必须大于0"
    assert transformer_config.d_model % transformer_config.n_heads == 0, "d_model必须能被n_heads整除"
    
    # 与SAC配置的一致性
    assert sac_config.state_dim == transformer_config.d_model, "SAC state_dim必须与Transformer d_model一致"
    
    # 特征维度检查
    assert transformer_config.n_features == 37, "当前系统要求n_features=37"
    
    # 合理性检查
    assert transformer_config.dropout <= 0.5, "dropout不应超过0.5"
    assert transformer_config.d_ff >= transformer_config.d_model, "d_ff应该至少等于d_model"
    
    print("✅ Transformer配置验证通过")
```

这份配置指南应该能帮助你根据不同场景选择合适的Transformer配置。建议从标准配置开始，然后根据实际性能需求进行调整。