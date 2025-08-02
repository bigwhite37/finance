# 🎯 Transformer配置快速指南

## 🚀 快速开始

### 1. 使用配置生成器（推荐）

```bash
# 生成标准配置
python scripts/generate_transformer_config.py --preset standard --validate --analyze

# 生成轻量配置用于快速测试
python scripts/generate_transformer_config.py --preset lightweight --format python

# 自定义配置
python scripts/generate_transformer_config.py --preset standard --d-model 128 --n-heads 4
```

### 2. 直接在代码中配置

```python
from rl_trading_system.models.transformer import TransformerConfig
from rl_trading_system.models.sac_agent import SACConfig, SACAgent

# 推荐的标准配置
transformer_config = TransformerConfig(
    d_model=256,           # 模型维度
    n_heads=8,             # 注意力头数
    n_layers=6,            # 编码器层数
    d_ff=1024,             # 前馈网络维度
    dropout=0.1,           # Dropout率
    max_seq_len=252,       # 最大序列长度
    n_features=37,         # 特征数量（固定）
    activation='gelu'      # 激活函数
)

sac_config = SACConfig(
    state_dim=256,         # 必须与d_model一致
    action_dim=3,
    hidden_dim=512,
    use_transformer=True,
    transformer_config=transformer_config
)

agent = SACAgent(sac_config)
```

## 📊 配置对比表

| 配置类型 | d_model | n_heads | n_layers | 参数量 | 内存(MB) | 适用场景 |
|----------|---------|---------|----------|--------|----------|----------|
| **调试** | 64 | 2 | 2 | 105K | 0.4 | 快速测试 |
| **轻量** | 128 | 4 | 3 | 1.2M | 4.6 | 资源受限 |
| **标准** | 256 | 8 | 6 | 4.8M | 18.3 | 推荐使用 |
| **高性能** | 512 | 16 | 8 | 25.5M | 97.1 | GPU充足 |

## ⚡ 关键参数说明

### 必须匹配的参数
- `sac_config.state_dim` == `transformer_config.d_model`
- `transformer_config.n_features` == 37 (当前特征工程输出)
- `transformer_config.d_model` % `transformer_config.n_heads` == 0

### 性能调优参数
- **d_model**: 128/256/512，影响表达能力和计算成本
- **n_layers**: 3/6/8，更深的网络表达力更强但易过拟合
- **n_heads**: 4/8/16，必须能被d_model整除
- **d_ff**: 通常是d_model的2-4倍

## 🛠️ 使用技巧

### 开发阶段
```bash
# 使用调试配置快速迭代
python scripts/generate_transformer_config.py --preset debug --format python
```

### 生产部署
```bash
# 使用标准配置平衡性能和资源
python scripts/generate_transformer_config.py --preset standard --validate
```

### 性能优化
```bash
# 分析配置复杂度
python scripts/generate_transformer_config.py --preset high_performance --analyze
```

## ❗ 常见问题

### 1. 维度不匹配错误
```
AttributeError: mat1 and mat2 shapes cannot be multiplied
```
**解决**: 确保 `state_dim == d_model`

### 2. 注意力头数错误
```
AssertionError: d_model必须能被n_heads整除
```
**解决**: 调整n_heads，例如d_model=256时使用n_heads=8

### 3. 特征数不匹配
```
RuntimeError: Expected input[1] to have size 37, but got size XX
```
**解决**: 确保 `n_features=37` 与特征工程输出一致

## 🔧 故障排除工具

```bash
# 验证配置合理性
python scripts/generate_transformer_config.py --preset standard --validate

# 列出所有可用配置
python scripts/generate_transformer_config.py --list-presets

# 查看详细的配置指南
cat docs/transformer_config_guide.md
```

## 💡 最佳实践

1. **开发**: 从`debug`配置开始快速迭代
2. **训练**: 使用`standard`配置获得良好性能  
3. **生产**: 根据硬件资源选择合适配置
4. **保存**: 始终保存完整的`transformer_config`到模型文件
5. **验证**: 使用配置生成器的验证功能检查参数

---

📖 **详细文档**: `docs/transformer_config_guide.md`  
🛠️ **配置生成器**: `scripts/generate_transformer_config.py`