# CVaR-PPO智能体API文档

## CVaRPPOAgent 类

CVaR-PPO智能体是系统的决策核心，它基于深度强化学习，通过与交易环境的交互来学习最优的投资策略。该智能体经过特殊设计，能够在追求收益的同时，严格控制投资组合的尾部风险（CVaR）。

### 类签名
```python
class CVaRPPOAgent:
    def __init__(self, state_dim: int, action_dim: int, config: dict)
```

### 初始化参数
-   **state_dim** (int): 状态空间的维度，即智能体在做决策时需要考虑的输入特征数量。
-   **action_dim** (int): 动作空间的维度，通常等于股票池中的股票数量。
-   **config** (dict): 智能体的配置字典，包含学习率、PPO参数、CVaR约束参数等。

### 主要方法

#### get_action
根据当前环境状态，获取智能体决策的动作（即投资组合权重）。

```python
def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float, float]
```

**参数:**
-   **state** (np.ndarray): 当前环境状态的向量。
-   **deterministic** (bool, 可选): 是否采用确定性策略。`True`表示直接使用策略网络的输出均值，`False`表示从策略分布中采样，用于训练过程中的探索。

**返回值:**
-   **Tuple**: 包含动作、动作的对数概率、状态价值估计和CVaR估计的元组。

#### store_transition
存储一次完整的状态转移经验，用于后续的策略学习。

```python
def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, value: float, log_prob: float, done: bool, cvar_estimate: float)
```

#### update_with_importance_weights
使用收集到的经验数据来更新策略网络和价值网络。这是智能体学习的核心步骤。

```python
def update_with_importance_weights(self, importance_weights: Optional[np.ndarray] = None) -> Dict[str, float]
```

**参数:**
-   **importance_weights** (Optional[np.ndarray]): 重要性权重，用于O2O训练中平衡离线和在线数据的重要性。默认为None。

**返回值:**
-   **Dict[str, float]**: 包含训练损失、学习率等信息的字典。

#### save_model / load_model
保存或加载训练好的模型检查点。

```python
def save_model(self, filepath: str)
def load_model(self, filepath: str)
```

### O2O特定方法

#### split_optimizers
为Actor和Critic网络分离优化器，允许在O2O训练的不同阶段使用不同的学习率。

```python
def split_optimizers(self)
```

#### freeze_actor / unfreeze_actor
冻结或解冻Actor（策略）网络的参数。在热身微调阶段，通常只更新Critic（价值）网络，因此会冻结Actor。

```python
def freeze_actor()
def unfreeze_actor()
```

### 配置参数

智能体支持丰富的配置选项，以适应不同需求：

```yaml
agent_config:
  # 网络结构
  hidden_dim: 256

  # PPO 核心参数
  learning_rate: 3.0e-4
  clip_epsilon: 0.2
  ppo_epochs: 10
  batch_size: 64
  gamma: 0.99 # 折扣因子
  lambda_gae: 0.95 # GAE参数

  # CVaR 风险约束参数
  cvar_alpha: 0.05 # CVaR分位数
  cvar_lambda: 1.0 # CVaR损失权重
  cvar_threshold: -0.02 # 可接受的最大CVaR

  # O2O 学习参数
  use_split_optimizers: True
  actor_lr: 1.0e-4
  critic_lr: 3.0e-4

  # 信任域约束 (用于在线学习)
  use_trust_region: True
  trust_region_beta: 1.0
  kl_target: 0.01
```

### 性能与内存优化

智能体内部集成了多项优化措施：
-   **混合精度训练**: 在支持的GPU上自动使用`torch.cuda.amp`减少内存占用并加速训练。
-   **梯度累积**: 允许在有限的显存下模拟更大的批次大小。
-   **内存高效缓冲区**: 使用`MemoryEfficientBuffer`来存储经验，并对旧数据进行压缩。
-   **内存监控**: 内置`MemoryMonitor`实时监控内存使用情况，并提供预警。
