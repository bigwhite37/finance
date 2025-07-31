# 快速训练指南

本指南将帮助您使用当前代码快速训练一个强化学习交易模型。

## 环境准备

### 1. 安装依赖

确保已安装所有必需的依赖：

```bash
# 安装Python依赖
pip install -r requirements.txt

# 或者使用setup.py
pip install -e .
```

### 2. 数据准备

系统支持两种数据提供商：
- **qlib** (推荐): 需要安装微软的Qlib
- **akshare**: 使用Akshare获取数据

确保在 `config/trading_config.yaml` 中配置了正确的数据提供商：

```yaml
trading:
  data:
    provider: "qlib"  # 或 "akshare"
    cache_enabled: true
    cache_dir: "./data_cache"
```

## 快速开始训练

### 1. 使用默认配置训练

最简单的训练命令：

```bash
python scripts/train.py
```

### 2. 自定义训练参数

#### 修改训练轮数
```bash
python scripts/train.py --episodes 5000
```

#### 使用自定义配置文件
```bash
python scripts/train.py --config config/my_model_config.yaml --data-config config/my_trading_config.yaml
```

#### 指定输出目录
```bash
python scripts/train.py --output-dir ./my_experiment
```

### 3. 配置文件详解

#### 模型配置 (`config/model_config.yaml`)

```yaml
# 关键参数
model:
  transformer:
    d_model: 256      # 模型维度
    n_heads: 8        # 注意力头数
    n_layers: 6       # 编码器层数
    max_seq_len: 252  # 最大序列长度（一年交易日）
  
  sac:
    state_dim: 256    # 状态维度
    action_dim: 100   # 动作维度（股票数量）
    hidden_dim: 512   # 隐藏层维度
    
training:
  n_episodes: 10000   # 训练轮数
  eval_freq: 100      # 评估频率
  patience: 50        # 早停耐心值
  checkpoint_dir: "./checkpoints"
```

#### 交易配置 (`config/trading_config.yaml`)

```yaml
trading:
  environment:
    lookback_window: 60         # 回看窗口
    initial_cash: 1000000.0     # 初始资金（100万）
    commission_rate: 0.001      # 手续费率 0.1%
    stamp_tax_rate: 0.001       # 印花税率 0.1%
    
  data:
    provider: "qlib"            # 数据提供商
    cache_enabled: true         # 启用缓存
    cache_dir: "./data_cache"   # 缓存目录

backtest:
  start_date: "2020-01-01"    # 回测开始日期
  end_date: "2023-12-31"      # 回测结束日期
  benchmark: "000300.SH"      # 基准指数
```

## 高级训练选项

### 1. 环境变量配置

支持通过环境变量覆盖配置：

```bash
# 模型参数
export MODEL_TRANSFORMER_D_MODEL=512
export MODEL_SAC_STATE_DIM=512
export TRAINING_N_EPISODES=20000

# 交易参数
export TRADING_ENVIRONMENT_INITIAL_CASH=2000000.0
export BACKTEST_START_DATE="2022-01-01"
```

### 2. GPU训练

系统会自动检测GPU，如需强制使用CPU：

```bash
# 在Python中设置设备
export CUDA_VISIBLE_DEVICES=""  # 禁用GPU
```

### 3. 多GPU训练

如需使用多GPU训练，可以修改trainer配置：

```python
# 在训练脚本中设置
training_config = {
    "device": "cuda",  # 或 "cpu"
    "batch_size": 512  # 增大批次大小
}
```

## 训练监控

### 1. 日志查看

训练过程中的日志会输出到控制台，包含：
- 每轮奖励
- 损失值
- 验证分数
- 训练时间

### 2. 检查点保存

训练过程中会自动保存检查点到 `checkpoints/` 目录：
- `checkpoint_latest.pt`: 最新模型
- `checkpoint_best.pt`: 最佳验证分数模型
- `training_metrics.pkl`: 训练指标

### 3. TensorBoard监控

如需使用TensorBoard：

```bash
# 启动TensorBoard
tensorboard --logdir=./logs

# 然后在浏览器访问 http://localhost:6006
```

## 训练中断与恢复

### 1. 自动恢复

训练中断后，系统会自动从最新检查点恢复：

```bash
# 重新运行训练命令即可恢复
python scripts/train.py
```

### 2. 手动恢复

如需从特定检查点恢复：

```python
# 在trainer.py中修改
trainer = RLTrainer(...)
trainer.load_checkpoint("./checkpoints/checkpoint_5000.pt")
trainer.train(resume_from=5000)
```

## 性能优化建议

### 1. 数据缓存

首次运行时会下载和缓存数据，后续运行会更快：

```yaml
trading:
  data:
    cache_enabled: true
    cache_dir: "./data_cache"
```

### 2. 批次大小调整

根据GPU内存调整批次大小：

```yaml
training:
  batch_size: 256  # 根据GPU内存调整
```

### 3. 多进程数据加载

如需启用多进程数据加载，可以：

```python
# 在数据加载部分设置
config = {
    "num_workers": 4,  # 根据CPU核心数调整
    "pin_memory": True  # GPU训练时启用
}
```

## 故障排除

### 1. 内存不足

如果遇到内存不足错误：
- 减小 `batch_size`
- 减小 `max_seq_len`
- 使用 `gradient_accumulation_steps`

### 2. 训练发散

如果训练发散：
- 降低学习率
- 增加 `warmup_episodes`
- 检查数据质量

### 3. 数据下载失败

如果数据下载失败：
- 检查网络连接
- 更换数据提供商
- 手动下载数据到缓存目录

## 示例训练会话

完整的训练会话示例：

```bash
# 1. 克隆项目
git clone <repository-url>
cd rl-trading-system

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境（可选）
cp config/model_config.yaml config/my_config.yaml
cp config/trading_config.yaml config/my_trading.yaml

# 4. 编辑配置文件（可选）
# vim config/my_config.yaml

# 5. 开始训练
python scripts/train.py \
  --config config/my_config.yaml \
  --data-config config/my_trading.yaml \
  --episodes 10000 \
  --output-dir ./experiments/run_001

# 6. 监控训练
# 查看日志输出
# 访问TensorBoard: http://localhost:6006

# 7. 训练完成后，模型保存在 checkpoints/ 目录
```

## 下一步

训练完成后，您可以：
1. **评估模型**: 使用 `scripts/evaluate.py` 评估训练好的模型
2. **回测**: 使用 `scripts/backtest.py` 进行历史回测
3. **部署**: 使用 `scripts/deploy.py` 部署到生产环境
4. **监控**: 使用 `scripts/monitor.py` 监控系统性能

如需了解更多高级功能，请参考：
- [API文档](../api/)
- [开发者指南](../developer_guide/architecture.md)
- [系统架构](../README.md)