# 因子工程教程

本教程将指导您如何使用系统的因子工程模块，包括计算内置因子、预处理因子以及开发自定义因子。

## 1. 因子工程的角色

因子是驱动强化学习智能体决策的核心输入。高质量的因子是策略成功的基石。系统的因子工程模块（主要是`FactorEngine`和`AlphaFactors`）负责：

-   **计算因子**: 基于原始的价格和成交量数据，计算出有预测能力的Alpha因子和衡量风险的风险因子。
-   **因子管理**: 提供统一的接口来调用和管理所有因子。
-   **因子预处理**: 对计算出的原始因子值进行标准化、去极值等操作，使其更适合输入到机器学习模型中。
-   **自定义扩展**: 提供简单的接口，让您可以方便地集成自己开发的因子。

## 2. 配置因子计算

您可以在配置文件中灵活地选择要计算哪些因子，并为它们设置不同的参数。

**示例配置 (`config.yaml`):**
```yaml
factors:
  # 要计算的Alpha因子列表
  alpha_factors:
    - name: 'Momentum'
      params:
        window: 20 # 计算20日动量
    - name: 'RSI'
      params:
        window: 14 # 计算14日RSI
    - name: 'MyCustomFactor' # 也可以包含自定义因子
      params:
        my_param: 123

  # 要计算的风险因子列表
  risk_factors:
    - name: 'Volatility'
      params:
        window: 60 # 计算60日波动率

  # 因子预处理流程
  preprocessing:
    - method: 'z_score' # 先进行Z-Score标准化
      group_by: 'date' # 按日期对所有股票进行截面标准化
    - method: 'winsorize' # 然后进行去极值处理
      lower_quantile: 0.01
      upper_quantile: 0.99
```

## 3. 计算因子

`FactorEngine`是执行因子计算的核心类。

```python
from factors.factor_engine import FactorEngine
from data.data_manager import DataManager
from config import ConfigManager

# 1. 初始化依赖项
config_manager = ConfigManager('my_config.yaml')
data_manager = DataManager(config_manager.get_data_config())

# 2. 初始化因子引擎
factor_config = config_manager.get_config('factors')
factor_engine = FactorEngine(factor_config, data_manager)

# 3. 获取股票池和日期范围
universe = data_manager._get_universe_stocks('2023-01-01', '2023-12-31')

# 4. 计算所有在配置中启用的因子
factor_data = factor_engine.calculate_factors(
    universe=universe,
    start_date='2023-01-01',
    end_date='2023-12-31'
)

print(factor_data.head())
```
`calculate_factors`的返回结果是一个多级索引的DataFrame，包含了在指定时间段和股票池上所有计算和预处理完成的因子值。

## 4. 开发自定义因子

系统允许您方便地添加自己的因子。只需遵循以下步骤：

**第一步：创建自定义因子类**

您的因子类需要继承自`factors.base.BaseFactor`并实现`compute`方法。

```python
# my_factors.py
from factors.base import BaseFactor
import pandas as pd

class MyMomentumFactor(BaseFactor):
    # 在构造函数中接收参数
    def __init__(self, window=20):
        self.window = window

    # 实现核心计算逻辑
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        # 'data' 是一个包含 'close' 列的DataFrame
        # 索引是 (datetime, instrument)
        close_prices = data['close'].unstack(level='instrument')
        
        # 计算动量
        momentum = close_prices.pct_change(self.window).stack()
        
        # 返回一个DataFrame，列名为因子名
        return momentum.to_frame(name=self.name)
```

**第二步：注册自定义因子**

在您的主流程中，将这个新因子注册到`FactorEngine`。

```python
from my_factors import MyMomentumFactor

# ... 初始化 factor_engine ...

# 注册
factor_engine.register_custom_factor('MyCustomFactor', MyMomentumFactor)
```

**第三步：在配置中使用**

现在您就可以像内置因子一样在`config.yaml`中使用`MyCustomFactor`了，如本教程第2节所示。

通过这种方式，您可以将自己的投研成果快速集成到系统中，并利用后续的强化学习和风险控制模块来验证其效果。
