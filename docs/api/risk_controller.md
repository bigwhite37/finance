# 风险控制器API文档

## RiskController 类

`RiskController`是一个统一的风险管理模块，它在智能体决策之后、实际执行之前，对投资组合权重进行一系列的风险审查和调整，构成了系统的“安全盾”。

### 类签名
```python
class RiskController:
    def __init__(self, config: dict, data_manager=None)
```

### 初始化参数
-   **config** (dict): 风险控制模块的配置字典。
-   **data_manager** (DataManager, 可选): 数据管理器实例，主要用于初始化内部的`DynamicLowVolFilter`。

### 核心方法

#### process_weights
这是风险控制器的核心入口。它接收智能体输出的原始权重，并按顺序执行一系列风险控制流程。

```python
def process_weights(self, raw_weights: np.ndarray, price_data: pd.DataFrame, current_nav: float, state: Dict) -> np.ndarray
```

**参数:**
-   **raw_weights** (np.ndarray): RL智能体输出的原始投资组合权重。
-   **price_data** (pd.DataFrame): 价格数据，用于计算波动率等指标。
-   **current_nav** (float): 当前的投资组合净值。
-   **state** (Dict): 当前环境的状态信息，包含回撤等。

**返回值:**
-   **np.ndarray**: 经过所有风险控制流程调整后的最终权重。

### 风险控制流程

`process_weights`方法内部的执行顺序如下：

1.  **基础约束**: 应用单股票最大仓位和总杠杆限制。
2.  **动态止损**: 检查是否触发全局止损条件，如果触发则进行减仓。
3.  **目标波动率控制**: 使用`TargetVolatilityController`根据市场情况调整组合杠杆，以匹配自适应的目标波动率。
4.  **动态低波筛选**: 如果启用，使用`DynamicLowVolFilter`获取当前可交易的股票掩码，并将不可交易的股票权重设为0。
5.  **风险平价优化 (可选)**: 如果启用，将原始权重与风险平价优化后的权重进行混合，以改善组合的风险分散。
6.  **最终安全检查**: 基于当前回撤等状态进行最终的仓位调整，作为最后一道防线。

### 其他主要方法

#### calculate_portfolio_risk
计算给定权重下的投资组合各项风险指标。

```python
def calculate_portfolio_risk(self, weights: np.ndarray, price_data: pd.DataFrame) -> Dict
```

#### check_risk_limits
检查给定权重是否违反了预设的风险限制。

```python
def check_risk_limits(self, weights: np.ndarray, price_data: pd.DataFrame) -> Tuple[bool, List[str]]
```

#### get_risk_report
生成一份全面的风险报告，包含当前风险状态、止损触发次数、动态低波筛选器状态等。

```python
def get_risk_report(self) -> Dict
```

### 配置参数

```yaml
risk_control_config:
  # 基础约束
  max_position: 0.1
  max_leverage: 1.2

  # 目标波动率
  target_volatility: 0.12

  # 动态止损
  enable_stop_loss: True
  stop_loss_threshold: 0.15
  stop_loss_window: 60

  # 风险平价
  enable_risk_parity: False
  alpha_weight: 0.7 # 原始权重和RP权重的混合比例

  # 动态低波筛选器
  enable_dynamic_lowvol: True
  dynamic_lowvol:
    # ... (参考DynamicLowVolFilter的配置)
```
