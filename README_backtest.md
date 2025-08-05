# 强化学习投资策略回测脚本使用说明

## 概述

`run_backtest.py` 是一个专业的回测脚本，用于测试训练好的强化学习投资模型。脚本集成了 Qlib 数据平台和专业的可视化分析功能。

## 功能特性

- ✅ 支持 PPO/SAC 等强化学习算法模型
- ✅ 自动读取配置文件，确保测试环境与训练环境一致
- ✅ 集成 Qlib 数据平台，支持多频数据
- ✅ 专业的性能分析和风险评估
- ✅ 多种可视化选项：标准图表 + Qlib专业分析
- ✅ 支持多种输出格式：文本报告、图表、Excel
- ✅ 灵活的命令行参数配置

## 安装依赖

```bash
pip install qlib stable-baselines3 gymnasium pandas numpy matplotlib seaborn openpyxl
```

## 基本用法

### 1. 基础回测

```bash
python run_backtest.py --model-path models/ppo_model.zip
```

### 2. 指定配置文件

```bash
python run_backtest.py \
    --model-path models/ppo_model.zip \
    --config configs/ppo_daily.yaml
```

### 3. 自定义测试期间

```bash
python run_backtest.py \
    --model-path models/ppo_model.zip \
    --test-start 2023-07-01 \
    --test-end 2023-12-31
```

### 4. 启用Qlib专业可视化

```bash
python run_backtest.py \
    --model-path models/ppo_model.zip \
    --enable-qlib-viz
```

### 5. 完整参数示例

```bash
python run_backtest.py \
    --model-path models/ppo_model.zip \
    --config configs/ppo_daily.yaml \
    --output-dir results/backtest_20240105 \
    --test-start 2023-07-01 \
    --test-end 2023-12-31 \
    --stocks 50 \
    --initial-cash 1000000 \
    --enable-qlib-viz \
    --log-level INFO
```

## 命令行参数详解

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model-path` | 训练好的模型文件路径（.zip格式） | `models/ppo_model.zip` |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/ppo_daily.yaml` | 配置文件路径 |
| `--output-dir` | `results` | 结果输出目录 |
| `--test-start` | 配置文件中的值 | 测试开始时间 (YYYY-MM-DD) |
| `--test-end` | 配置文件中的值 | 测试结束时间 (YYYY-MM-DD) |
| `--stocks` | 配置文件中的值 | 股票数量限制 |
| `--initial-cash` | `1000000` | 初始资金 |
| `--qlib-uri` | 配置文件中的值 | Qlib数据URI |
| `--log-level` | `INFO` | 日志级别 (DEBUG/INFO/WARNING/ERROR) |

### 控制选项

| 参数 | 说明 |
|------|------|
| `--no-plot` | 跳过标准绘图，仅生成报告 |
| `--no-export` | 跳过Excel导出 |
| `--enable-qlib-viz` | 启用Qlib风格的专业可视化分析 |

## 输出文件说明

运行回测后，会在输出目录生成以下文件：

### 1. 文本报告 (`backtest_report_TIMESTAMP.txt`)
- 策略表现对比
- 关键指标统计
- 风险分析
- 月度收益分析

### 2. 标准可视化图表 (`backtest_plots_TIMESTAMP.png`)
- 净值曲线对比
- 回撤分析
- 收益率分布
- 滚动指标分析
- 持仓权重变化
- 风险收益散点图

### 3. Qlib专业分析 (`qlib_analysis_TIMESTAMP.png`)
启用 `--enable-qlib-viz` 时生成，包含：
- 综合净值分析
- 详细回撤分析
- 收益率分布对比
- 月度收益热力图
- 市场状态分析
- 交易频率分析
- 波动率分析
- 收益归因分析
- 业绩统计表
- 更多专业指标

### 4. Excel数据 (`backtest_results_TIMESTAMP.xlsx`)
包含多个工作表：
- 性能指标对比
- 净值序列
- 收益率序列
- 持仓权重（如果有）

## 配置文件要求

确保配置文件包含以下必要部分：

```yaml
# 数据配置
data:
  test_start: "2023-07-01"
  test_end: "2023-12-31"
  stock_limit: 50
  fields:
    - "$close"
    - "$open"
    - "$high"
    - "$low"
    - "$volume"
    - "$change"
    - "$factor"

# 环境配置  
environment:
  lookback_window: 30
  transaction_cost: 0.003
  features:
    - "$close"
    - "$open" 
    - "$high"
    - "$low"
    - "$volume"
    - "$change"
    - "$factor"

# 模型配置
model:
  algorithm: "PPO"  # 或 "SAC"
```

## 使用注意事项

### 1. 模型兼容性
- 确保模型文件是.zip格式
- 模型必须是用stable-baselines3训练的PPO或SAC模型
- 测试环境参数必须与训练时一致

### 2. 数据要求
- 需要Qlib格式的股票数据
- 数据索引格式为 (instrument, datetime)
- 确保测试期间有足够的数据

### 3. 性能优化
- 大数据集建议限制股票数量
- 可以使用 `--no-plot` 跳过图表生成以提高速度
- 使用适当的日志级别控制输出详细程度

### 4. 故障排除

**常见错误及解决方案：**

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| 模型加载失败 | 模型文件损坏或格式错误 | 检查模型文件完整性和格式 |
| 观察空间不匹配 | 测试环境参数与训练不一致 | 检查配置文件中的环境参数 |
| 数据加载失败 | Qlib数据路径错误 | 检查配置文件中的数据路径 |
| 内存不足 | 数据量过大 | 减少股票数量或缩短测试期间 |

## 高级用法

### 1. 批量回测
可以编写脚本批量测试多个模型：

```bash
#!/bin/bash
for model in models/*.zip; do
    echo "Testing $model"
    python run_backtest.py --model-path "$model" --output-dir "results/$(basename $model .zip)"
done
```

### 2. 参数研究
测试不同的交易成本或其他参数对性能的影响：

```bash
# 测试不同的初始资金
for cash in 500000 1000000 2000000; do
    python run_backtest.py --model-path models/ppo_model.zip --initial-cash $cash --output-dir "results/cash_$cash"
done
```

### 3. 时间序列分析
测试模型在不同时间段的表现：

```bash
# 按季度测试
python run_backtest.py --model-path models/ppo_model.zip --test-start 2023-01-01 --test-end 2023-03-31
python run_backtest.py --model-path models/ppo_model.zip --test-start 2023-04-01 --test-end 2023-06-30
python run_backtest.py --model-path models/ppo_model.zip --test-start 2023-07-01 --test-end 2023-09-30
python run_backtest.py --model-path models/ppo_model.zip --test-start 2023-10-01 --test-end 2023-12-31
```

## 日志信息

脚本会生成详细的日志信息，包括：
- 参数验证结果
- 数据加载进度
- 模型加载状态
- 回测执行进度
- 性能指标摘要
- 文件生成位置

日志同时输出到控制台和文件 (`logs/backtest_TIMESTAMP.log`)。

## 技术支持

如果遇到问题，请检查：
1. 依赖包版本兼容性
2. 配置文件格式正确性
3. 数据文件完整性
4. 模型文件有效性
5. 日志文件中的详细错误信息