# 快速开始教程（专业版）

本教程将指导您完成一个标准的量化策略研发流程：**1. 使用历史数据进行离线预训练；2. 在未来的未知数据上进行严格的样本外测试**。这个流程是客观评估模型泛化能力的基础。

## 前提条件

请确保您已严格按照[环境搭建指南](setup_guide.md)的所有步骤完成了环境配置，特别是`qlib`的数据下载。

## 第一步：创建训练和回测配置文件

为了客观地评估模型性能，我们必须将训练数据和测试数据严格分开。因此，我们创建两个配置文件。

**1. 训练配置文件 (`config_train.yaml`):**

```yaml
# config_train.yaml
# 用于在2020-2022年的数据上进行离线预训练

data:
  provider_uri: "~/.qlib/qlib_data/cn_data"
  start_date: "2020-01-01"
  end_date: "2022-12-31"
  universe: "csi300"

agent:
  learning_rate: 3.0e-4
  cvar_threshold: -0.02 # CVaR约束，目标是尾部风险不劣于-2%

risk_control:
  target_volatility: 0.12 # 目标年化波动率12%
  max_leverage: 1.2

training:
  total_episodes: 500 # 增加训练回合数以充分学习

model:
  save_dir: "./models/csi300_2020_2022"
```

**2. 回测配置文件 (`config_backtest.yaml`):**

```yaml
# config_backtest.yaml
# 用于在2023年的全新数据上回测模型

data:
  provider_uri: "~/.qlib/qlib_data/cn_data"
  start_date: "2023-01-01"
  end_date: "2023-12-31"
  universe: "csi300"

# 回测时不需要训练和agent配置，但需要risk_control来管理风险
risk_control:
  target_volatility: 0.12
  max_leverage: 1.2
```

> **为什么要分离训练和测试周期？**
> 这是为了模拟真实投资场景。我们只能用过去的数据来构建策略，而策略是否有效，必须由它在“未来”（即测试集）的表现来证明。如果训练和测试数据重叠，模型会“记住”答案，导致回测结果虚高，不具备参考价值。

## 第二步：离线预训练

现在，我们使用`config_train.yaml`和`train`模式来训练智能体。这个过程会比较耗时，请耐心等待。

```bash
# 使用训练配置进行模型训练
python main.py --mode train --config config_train.yaml
```

**预期输出：**
终端会持续输出训练日志。当训练完成时，您会看到类似`保存最佳模型: ./models/csi300_2020_2022/best_agent_episode_490.pth`的日志。**请记下这个模型路径**，我们下一步回测时需要它。

> **提示**: 500个回合的训练在现代CPU上可能需要30-90分钟，在GPU上会快很多。对于初次体验，您可以将`total_episodes`改回100来快速完成流程。

## 第三步：样本外测试（Backtest）

这是最关键的一步。我们使用上一步训练好的模型，在它从未见过的2023年的数据上运行回测。

```bash
# 使用回测配置和训练好的模型进行回测
# 将 --model 参数替换为您上一步记下的实际模型路径
python main.py --mode backtest --config config_backtest.yaml --model ./models/csi300_2020_2022/best_agent_episode_490.pth
```

## 第四步：解读客观的回测报告

这次在终端输出的回测报告，是在**样本外（Out-of-Sample）**数据上得到的，它能更真实地反映模型的泛化能力。

由于是在未知数据上运行，其性能指标（如夏普比率、年化收益）通常会比在训练集上看到的要低，这是完全正常的，也正是我们进行样本外测试的意义所在。

## 进阶：何时以及如何使用O2O训练？

以上流程创建的是一个**静态模型**：它在2020-2022年的数据上完成学习后，其参数就固定不变了。在真实的投资中，市场风格是不断变化的（例如，从成长股风格切换到价值股风格），静态模型可能无法适应这种**分布漂移（Distribution Drift）**，导致性能逐渐衰退。

**O2O（Offline-to-Online，离线到在线）** 训练就是为了解决这个问题而设计的。

#### 何时使用O2O？
当您需要模拟或执行一个**持续学习、动态适应**的策略时，就应该使用O2O。例如，您想模拟一个从2023年1月1日开始运行的基金，它每个月都能利用上一个月的新数据来微调自己，那么O2O就是最佳选择。

#### O2O如何工作？
它将训练过程分为三个阶段：
1.  **离线预训练 (Offline Pre-training)**: 与我们上面执行的`train`模式完全一样，用长周期历史数据完成基础模型的训练。
2.  **热身微调 (Warm-up Finetuning)**: 在进入在线学习前，使用最近的一小段数据（如过去60天）对模型进行快速微调，使其适应当前的市场环境。
3.  **在线学习 (Online Learning)**: 在模拟的交易过程中，持续收集新的交易数据，并用这些新数据与部分旧数据混合，不断地对模型进行迭代更新。

#### O2O配置示例

启用O2O需要在配置文件中加入`o2o`部分。系统有专门的`O2OTrainingCoordinator`来管理这个复杂流程。

```yaml
# 一个简化的O2O配置片段
o2o:
  # 离线预训练配置
  offline_pretraining:
    epochs: 100
  
  # 热身微调配置
  warmup_finetuning:
    days: 60 # 使用最近60天数据热身
    epochs: 20

  # 在线学习配置
  online_learning:
    initial_rho: 0.2 # 初始时，20%的数据来自新样本
    update_frequency: 10 # 每10个交易日更新一次模型
```

> 要了解关于O2O的完整配置和使用方法，请参考 **[O2O强化学习使用指南](../o2o_rl_guide.md)**。

--- 

**恭喜！** 您已经完成了一次规范的量化策略研发流程，并了解了静态模型与O2O动态模型的区别。现在，您可以开始探索如何优化您的离线模型，或者尝试挑战更高级的O2O训练流程。
