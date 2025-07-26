# 交易环境API文档

## TradingEnvironment 类

`TradingEnvironment`是为强化学习智能体设计的、遵循`gymnasium`接口标准的A股市场模拟环境。它负责处理市场数据、执行交易、计算收益和奖励，并将市场状态反馈给智能体。

### 类签名
```python
class TradingEnvironment(gym.Env):
    def __init__(self, factor_data: pd.DataFrame, price_data: pd.DataFrame, config: dict)
```

### 初始化参数
-   **factor_data** (pd.DataFrame): 因子数据，索引为日期，列为不同因子。
-   **price_data** (pd.DataFrame): 价格数据，索引为日期，列为不同股票代码。
-   **config** (dict): 环境配置字典，包含交易成本、回看窗口等设置。

### 核心方法 (Gymnasium接口)

#### reset
重置环境到初始状态，开始新一轮的回合（episode）。

```python
def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]
```

**返回值:**
-   **Tuple**: 包含初始状态观测（observation）和环境信息（info）的元组。

#### step
在环境中执行一个动作，并推进一个时间步。

```python
def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]
```

**参数:**
-   **action** (np.ndarray): 智能体输出的动作，即目标投资组合权重。

**返回值:**
-   **Tuple**: 包含新状态、奖励、是否终止(terminated)、是否截断(truncated)和环境信息的元组。

### 状态空间 (Observation Space)

状态空间是环境提供给智能体的信息，经过精心设计以全面反映市场情况：
-   **因子特征**: 由因子引擎计算的Alpha和风险因子。
-   **宏观特征**: 模拟的VIX指数、北向资金流等。
-   **组合状态**: 当前持仓权重、近期波动率、最大回撤等。
-   **市场制度信号**: 由`DynamicLowVolFilter`判断的当前市场波动制度（高、中、低）。
-   **时间信息**: 年内进度、月份、季度等周期性特征。

### 动作空间 (Action Space)

-   **类型**: 连续的`Box`空间。
-   **定义**: 一个N维向量，N是股票池大小，每个元素代表对应股票的目标权重。
-   **约束**: 内部通过`_constrain_action`方法对动作进行约束，包括：
    -   单只股票最大持仓限制。
    -   总杠杆限制。
    -   应用动态低波筛选器的可交易掩码。

### 奖励函数 (Reward Function)

奖励函数的设计旨在引导智能体学习稳健的策略：
`Reward = returns + profit_bonus - drawdown_penalty - cvar_penalty - cost_penalty`
-   **核心**: 周期收益率 `returns`。
-   **激励**: 对正收益给予额外奖励 `profit_bonus`。
-   **惩罚**: 对超过阈值的回撤、CVaR和交易成本进行惩罚。

### O2O特定方法

#### set_mode / get_mode
设置或获取环境的运行模式（`offline`或`online`）。

```python
def set_mode(self, mode: str)
def get_mode(self) -> str
```

#### collect_trajectory
在`online`模式下，收集并存储完整的状态转移轨迹，用于在线学习和分析。

```python
def collect_trajectory(self, state, action, reward, next_state, done, info)
```

#### get_recent_trajectory
获取最近一段时间的轨迹数据，用于热身微调等场景。

```python
def get_recent_trajectory(self, window: int = 60) -> List[Dict]
```

### 配置参数

```yaml
env_config:
  lookback_window: 20 # 回看窗口期
  transaction_cost: 0.001 # 交易成本
  max_position: 0.1 # 单股票最大仓位
  max_leverage: 1.2 # 最大杠杆

  # 奖励函数参数
  lambda1: 2.0 # 回撤惩罚系数
  lambda2: 1.0 # CVaR惩罚系数
  max_dd_threshold: 0.05 # 最大回撤惩罚阈值

  # O2O配置
  trajectory_buffer_size: 10000 # 轨迹缓冲区大小

  # 动态低波筛选器配置
  dynamic_lowvol:
    # ... (参考DynamicLowVolFilter的配置)
```
