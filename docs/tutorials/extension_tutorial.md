# 自定义扩展教程

本教程将向您展示如何扩展系统的核心功能，包括添加新的因子、自定义风险控制规则以及集成新的数据源。系统的模块化设计使得这些扩展变得相对简单。

## 1. 添加新因子

这是最常见的扩展需求。您可以将自己的投研成果（因子）无缝集成到系统中。

**回顾：开发自定义因子的三个步骤**

1.  **创建因子类**: 继承`factors.base.BaseFactor`并实现`compute`方法。
2.  **注册因子**: 调用`FactorEngine`实例的`register_custom_factor`方法。
3.  **配置因子**: 在`config.yaml`的`factors.alpha_factors`或`factors.risk_factors`列表中添加您的因子。

**详细示例请参考[因子工程教程](factor_tutorial.md)的第4节。**

## 2. 自定义风险控制规则

虽然`RiskController`提供了多种内置的风险控制模块，但您可能希望实现自己独特的风险管理逻辑。

### 方式一：创建新的风险控制子模块

这是最规范的方式。您可以创建一个新的风险控制类，然后在`RiskController`中调用它。

**第一步：创建自定义风险规则类**

```python
# my_risk_rules.py
import numpy as np

class MyMarketSentimentRule:
    def __init__(self, config):
        self.sentiment_threshold = config.get('sentiment_threshold', 0.3)

    def adjust_weights(self, weights, market_sentiment):
        """如果市场情绪低落，则减仓"""
        if market_sentiment < self.sentiment_threshold:
            print("市场情绪低迷，触发减仓规则！")
            return weights * 0.8 # 仓位减少20%
        return weights
```

**第二步：修改`RiskController`**

您需要修改`risk_control/risk_controller.py`来集成您的新规则。

```python
# risk_control/risk_controller.py

# 导入您的新规则
from my_risk_rules import MyMarketSentimentRule

class RiskController:
    def __init__(self, config: dict, data_manager=None):
        # ... 其他初始化代码 ...
        
        # 初始化您的规则
        self.my_sentiment_rule = MyMarketSentimentRule(config.get('my_rule_config', {}))

    def process_weights(self, raw_weights, ...):
        # ...
        # 在风险控制流程的某个环节调用您的规则
        # 假设state中包含了market_sentiment
        market_sentiment = state.get('market_sentiment', 0.5)
        weights = self.my_sentiment_rule.adjust_weights(weights, market_sentiment)
        # ...
        return weights
```

### 方式二：在现有流程中快速实现（不推荐用于复杂逻辑）

对于非常简单的规则，您可以直接在`RiskController`的`_final_safety_check`等方法中添加几行代码。这种方式不适合复杂的逻辑，因为它会破坏原有代码的结构。

## 3. 集成新数据源

这是一个更高级的扩展，需要修改`DataManager`。

**核心思路**: 修改`data/data_manager.py`，在`get_stock_data`等方法中加入对新数据源的处理逻辑。

**示例：添加一个CSV文件作为数据源**

```python
# data/data_manager.py

class DataManager:
    def __init__(self, config: Dict):
        self.config = config
        self.provider = config.get('provider', 'qlib')
        # ...
        # 为新数据源添加配置
        self.csv_path = config.get('csv_path')

    def get_stock_data(self, ...):
        # 如果provider是新的自定义类型
        if self.provider == 'csv':
            if not self.csv_path:
                raise ValueError("CSV provider需要配置csv_path")
            
            # 从CSV加载数据
            all_data = pd.read_csv(self.csv_path, index_col='date', parse_dates=True)
            
            # 根据传入的参数进行筛选
            # ... (此处省略筛选逻辑)
            
            return filtered_data
        
        # 原有的qlib逻辑
        # ...
```

之后，您就可以在配置文件中这样使用：

```yaml
data:
  provider: "csv"
  csv_path: "/path/to/my/stock_data.csv"
  # ...
```

**重要提示**: 当您对核心模块（如`RiskController`, `DataManager`）进行修改时，请务必：
1.  **备份原文件**。
2.  **充分理解原有代码的逻辑**。
3.  **为您添加的新功能编写单元测试**，以确保它不会对系统其他部分产生意料之外的影响。

通过以上方式，您可以将本系统作为框架，不断地将自己的想法和策略融入其中，构建出完全属于您自己的量化交易系统。
