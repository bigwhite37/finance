# 风险控制教程

本教程将指导您如何配置和使用系统的多层风险控制机制。风险控制是本系统的核心理念之一，它在智能体决策的基础上增加了一层“安全气囊”，确保投资行为始终在预设的风险边界内。

## 1. 风险控制的核心：`RiskController`

系统的所有风险管理功能都统一由`RiskController`模块来协调。它像一个经验丰富的风控总监，在智能体（年轻的交易员）做出交易决策后，进行最终的审核和调整。

**工作流程:**
```mermaid
graph TD
    A[RL智能体产生原始权重] --> B[RiskController];
    subgraph RiskController 内部流程
        B --> C{1. 基础约束}; 
        C --> D{2. 动态止损}; 
        D --> E{3. 目标波动率}; 
        E --> F{4. 低波筛选}; 
        F --> G{5. 风险平价(可选)};
        G --> H{6. 最终安全检查};
    end
    H --> I[最终执行的权重];
```

## 2. 配置风险控制器

您可以在主配置文件中对`RiskController`的各项功能进行详细设置。

**示例配置 (`config.yaml`):**
```yaml
risk_control:
  # 基础约束
  max_position: 0.1         # 单只股票最大仓位 (10%)
  max_leverage: 1.2         # 最大总杠杆 (1.2x)

  # 目标波动率控制
  target_volatility: 0.12   # 目标年化波动率 (12%)

  # 动态止损
  enable_stop_loss: True
  stop_loss_threshold: 0.15 # 触发止损的最大回撤阈值 (15%)
  stop_loss_window: 60      # 计算回撤的窗口期（60天）

  # 风险平价优化
  enable_risk_parity: False # 是否启用风险平价
  alpha_weight: 0.7         # 智能体权重与RP权重的混合比例

  # 动态低波动筛选
  enable_dynamic_lowvol: True
  # dynamic_lowvol的具体配置... (此处省略)
```

## 3. 各风险模块详解

### 3.1 目标波动率控制 (`TargetVolatilityController`)

-   **作用**: 动态调整投资组合的杠杆，使其年化波动率逼近您设定的`target_volatility`。
-   **如何工作**: 它会计算投资组合过去一段时间的已实现波动率，如果低于目标，则按比例增加杠杆；如果高于目标，则降低杠杆。
-   **自适应目标**: 当与`DynamicLowVolFilter`一同使用时，目标波动率不再是固定值，而是会根据市场制度（高波/低波）自适应调整，实现更智能的风险管理。

### 3.2 动态止损 (`DynamicStopLoss`)

-   **作用**: 作为最终的保护机制，当投资组合出现大幅回撤时，强制降低仓位。
-   **如何工作**: 它会持续监控投资组合净值在`stop_loss_window`内的最大回撤。一旦回撤超过`stop_loss_threshold`，它会触发减仓信号（例如，将总仓位降低50%）。

### 3.3 动态低波动筛选 (`DynamicLowVolFilter`)

-   **作用**: 这是一个更主动的风险管理工具。它通过分析市场整体的波动性，判断当前处于“高波动”还是“低波动”制度。
-   **如何工作**:
    -   在高波动市场，它会筛选出那些相对“稳定”的股票，生成一个“可交易股票”的白名单，智能体只能在这些股票中进行交易。
    -   它还会动态调整`TargetVolatilityController`的目标波动率，在市场动荡时自动采取更保守的姿态。
-   **配置**: 这是最复杂的风险子模块，其详细配置请参考其自身的文档和示例。

## 4. 在回测和实盘中应用

`RiskController`（作为安全盾）通常在回测或实盘执行的顶层循环中被调用。

**回测示例:**
```python
from risk_control.risk_controller import RiskController

# 初始化
risk_controller_config = config.get_config('risk_control')
risk_controller = RiskController(risk_controller_config, data_manager)

# 在回测循环中
state, info = env.reset()
done = False
while not done:
    raw_weights = agent.get_action(state, deterministic=True)[0]

    # <<< 应用风险控制 >>>
    adjusted_weights = risk_controller.process_weights(
        raw_weights=raw_weights,
        price_data=env.get_price_data_slice(), # 获取当前价格数据
        current_nav=info['portfolio_value'],
        state=info
    )

    next_state, reward, terminated, truncated, info = env.step(adjusted_weights)
    state = next_state
    done = terminated or truncated
```

## 5. 查看风险报告

您可以随时调用`get_risk_report`方法来获取一份关于当前风险状况的详细报告。

```python
report = risk_controller.get_risk_report()
import json
print(json.dumps(report, indent=2))
```

这份报告将为您提供关于止损触发次数、动态低波筛选器的当前状态、自适应目标波动率等关键信息，帮助您全面了解系统的风险管理行为。
