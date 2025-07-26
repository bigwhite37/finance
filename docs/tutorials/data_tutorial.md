# 数据管理教程

本教程将指导您如何使用`DataManager`模块来获取、管理和预处理量化策略所需的数据。

## 1. 理解DataManager的角色

`DataManager`是整个系统的数据中枢。它负责：
-   **连接数据源**: 目前主要通过`qlib`库获取A股数据。
-   **提供统一接口**: 无论底层数据源是什么，都提供一致的函数调用方式。
-   **管理股票池**: 获取如沪深300、中证500等不同指数的成分股。
-   **数据清洗**: 对原始数据进行填充缺失值、处理异常值等预处理操作。
-   **缓存机制**: 内置简单的缓存，避免重复从硬盘读取数据，提高效率。

## 2. 配置DataManager

在运行策略之前，您需要确保`DataManager`的配置是正确的。这通常在主配置文件`config.yaml`中完成。

**示例配置:**
```yaml
data:
  provider_uri: "~/.qlib/qlib_data/cn_data" # qlib数据存储路径
  provider: "qlib" # 数据提供商
  region: "cn" # 地区
  universe: "csi300" # 默认使用的股票池
  start_date: "2020-01-01"
  end_date: "2023-12-31"
```

-   **`provider_uri`**: 这是最重要的配置，请确保它指向您存放`qlib`数据的正确路径。
-   **`universe`**: 定义了当您不指定股票列表时，默认使用哪个股票池。可选值包括`csi300`, `csi500`, `csi1000`, `all`。

## 3. 基本使用方法

### 3.1 初始化
```python
from config import ConfigManager
from data.data_manager import DataManager

# 加载配置
config_manager = ConfigManager('my_config.yaml')
data_config = config_manager.get_data_config()

# 初始化数据管理器
data_manager = DataManager(data_config)
```

### 3.2 获取股票数据

使用`get_stock_data`方法可以获取多只股票在一段时间内的多种数据字段。

```python
# 定义要获取的股票和字段
my_stocks = ['SH600519', 'SZ000001']
my_fields = ['$open', '$close', '$volume'] # qlib风格的字段名

# 获取数据
stock_data = data_manager.get_stock_data(
    instruments=my_stocks,
    start_time='2023-01-01',
    end_time='2023-03-31',
    fields=my_fields
)

# 返回的是一个多级索引的DataFrame
print(stock_data.head())
```

### 3.3 获取股票池

使用`_get_universe_stocks`方法（注意：这是一个内部方法，但在实践中常用）可以获取指定股票池的列表。

```python
# 获取沪深300成分股
csi300_stocks = data_manager._get_universe_stocks('2023-01-01', '2023-12-31')
print(f"沪深300股票数量: {len(csi300_stocks)}")
print(f"示例: {csi300_stocks[:5]}")
```

### 3.4 获取交易日历

```python
calendar = data_manager.get_trading_calendar(
    start_time='2023-01-01',
    end_time='2023-01-31'
)
print(calendar)
```

## 4. 数据质量

`DataManager`在返回数据前会自动进行清洗：
-   将无穷大的值替换为NaN。
-   将价格字段中的0或负值替换为NaN。
-   使用前向填充（`ffill`）和后向填充（`bfill`）来处理缺失值。

虽然有自动清洗，但数据的原始质量仍然至关重要。请确保您的`qlib`数据是完整和准确的。

## 5. 注意事项

-   **股票代码格式**: `qlib`内部使用的股票代码格式为`SH600000`或`SZ000001`，请在调用时遵循此格式。
-   **字段名称**: `qlib`的字段名称以`$`开头，如`$close`, `$volume`。
-   **性能**: 对于大规模的数据提取，建议分块进行，并合理利用`DataManager`的缓存机制。
