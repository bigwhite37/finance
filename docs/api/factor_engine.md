# 因子引擎API文档

## FactorEngine 类

因子引擎是系统的核心计算单元之一，负责根据原始数据计算、合成和管理各种Alpha因子和风险因子。

### 类签名
```python
class FactorEngine:
    def __init__(self, config: dict, data_manager: DataManager)
```

### 初始化参数
-   **config** (dict): 因子配置字典，包含因子列表、参数等设置。
-   **data_manager** (DataManager): 一个DataManager实例，用于获取计算因子所需的底层数据。

### 主要方法

#### calculate_factors
计算所有在配置中启用的因子。

```python
def calculate_factors(self, start_date: str, end_date: str, 
                      universe: List[str]) -> pd.DataFrame
```

**参数:**
-   **universe** (List[str]): 股票代码列表。
-   **start_date** (str): 开始日期，格式 'YYYY-MM-DD'。
-   **end_date** (str): 结束日期，格式 'YYYY-MM-DD'。

**返回值:**
-   **pd.DataFrame**: 多级索引的数据框，索引为 (date, symbol)，列为各个因子的值。

**示例:**
```python
from data import DataManager
from factors import FactorEngine

# 假设config和data_manager已初始化
factor_engine = FactorEngine(config.get_config('factors'), data_manager)

# 获取股票池
universe = data_manager.get_universe(date='2023-12-31', market='csi300')

# 计算因子
factor_data = factor_engine.calculate_factors(
    universe=universe,
    start_date='2023-01-01',
    end_date='2023-12-31'
)

print(factor_data.head())
```

#### get_alpha_factors
获取Alpha因子列表。

```python
def get_alpha_factors(self) -> List[str]
```

**返回值:**
-   **List[str]**: 系统当前支持的Alpha因子名称列表。

#### get_risk_factors
获取风险因子列表。

```python
def get_risk_factors(self) -> List[str]
```

**返回值:**
-   **List[str]**: 系统当前支持的风险因子名称列表。

#### register_custom_factor
注册一个用户自定义的因子计算类。

```python
def register_custom_factor(self, factor_name: str, factor_class: Type[BaseFactor])
```

**参数:**
-   **factor_name** (str): 自定义因子的名称。
-   **factor_class** (Type[BaseFactor]): 继承自`factors.BaseFactor`的自定义因子类。

**示例:**
```python
# 假设已定义了MyCustomFactor
from my_factors import MyCustomFactor
factor_engine.register_custom_factor('MyMomentum', MyCustomFactor)
```

### 配置参数

因子引擎支持以下配置参数：

```yaml
# factors_config.yaml
# Alpha因子列表和参数
alpha_factors:
  - name: 'Momentum'
    params:
      window: 20
  - name: 'RSI'
    params:
      window: 14
  - name: 'MyCustomFactor' # 自定义因子
    params:
      param1: 'value1'

# 风险因子列表和参数
risk_factors:
  - name: 'Volatility'
    params:
      window: 60
  - name: 'Beta'
    params:
      window: 120

# 因子预处理
preprocessing:
  - method: 'z_score' # 标准化
    group_by: 'date'
  - method: 'winsorize' # 去极值
    lower_quantile: 0.01
    upper_quantile: 0.99

# 性能优化
performance:
  use_cache: True
  parallel: True
  n_jobs: -1 # 使用所有可用CPU
```

### 异常处理

```python
from factors import FactorCalculationError

try:
    factors = factor_engine.calculate_factors(universe, start_date, end_date)
except FactorCalculationError as e:
    print(f"因子计算错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 自定义因子开发指南

1.  创建一个继承自 `factors.BaseFactor` 的新类。
2.  实现 `compute` 方法，该方法接受 `pd.DataFrame` 格式的原始数据，并返回计算出的因子值。
3.  使用 `register_custom_factor` 方法将您的新因子注册到引擎中。
4.  在配置文件中添加您的自定义因子及其参数。

```python
from factors.base import BaseFactor
import pandas as pd

class MyMomentumFactor(BaseFactor):
    def __init__(self, window=20):
        self.window = window

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        # 'data' 是一个包含 'close' 列的DataFrame
        close_prices = data['close'].unstack(level='symbol')
        momentum = close_prices.pct_change(self.window).stack()
        return momentum.to_frame(name=self.name)
```
