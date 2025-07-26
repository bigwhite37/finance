# 数据管理器API文档

## DataManager 类

数据管理器是系统的数据访问层，负责股票数据的获取、预处理和管理。

### 类签名
```python
class DataManager:
    def __init__(self, config: dict)
```

### 初始化参数
- **config** (dict): 数据配置字典，包含数据源、时间范围等设置

### 主要方法

#### get_data
获取指定股票的历史数据。

```python
def get_data(self, symbols: List[str], start_date: str, end_date: str, 
             fields: List[str] = None) -> pd.DataFrame
```

**参数:**
- **symbols** (List[str]): 股票代码列表，如 ['000001.SZ', '000002.SZ']
- **start_date** (str): 开始日期，格式 'YYYY-MM-DD'
- **end_date** (str): 结束日期，格式 'YYYY-MM-DD'
- **fields** (List[str], 可选): 数据字段列表，如 ['close', 'volume']

**返回值:**
- **pd.DataFrame**: 多级索引的数据框，索引为 (date, symbol)

**示例:**
```python
from data import DataManager

# 初始化数据管理器
config = {
    'provider': 'qlib',
    'market': 'csi300',
    'cache_dir': './cache'
}
data_manager = DataManager(config)

# 获取数据
symbols = ['000001.SZ', '000002.SZ']
data = data_manager.get_data(
    symbols=symbols,
    start_date='2023-01-01',
    end_date='2023-12-31',
    fields=['close', 'volume', 'high', 'low']
)

print(data.head())
```

#### get_universe
获取股票池列表。

```python
def get_universe(self, date: str = None, market: str = None) -> List[str]
```

**参数:**
- **date** (str, 可选): 指定日期，格式 'YYYY-MM-DD'
- **market** (str, 可选): 市场类型，如 'csi300', 'csi500'

**返回值:**
- **List[str]**: 股票代码列表

**示例:**
```python
# 获取CSI300成分股
universe = data_manager.get_universe(
    date='2023-12-31', 
    market='csi300'
)
print(f"股票池大小: {len(universe)}")
```

#### preprocess_data
数据预处理方法。

```python
def preprocess_data(self, data: pd.DataFrame, 
                   fill_method: str = 'forward') -> pd.DataFrame
```

**参数:**
- **data** (pd.DataFrame): 原始数据
- **fill_method** (str): 缺失值填充方法，'forward'、'backward'或'drop'

**返回值:**
- **pd.DataFrame**: 预处理后的数据

**示例:**
```python
# 数据预处理
clean_data = data_manager.preprocess_data(
    data=raw_data,
    fill_method='forward'
)
```

#### calculate_returns
计算收益率。

```python
def calculate_returns(self, prices: pd.DataFrame, 
                     periods: List[int] = [1]) -> pd.DataFrame
```

**参数:**
- **prices** (pd.DataFrame): 价格数据
- **periods** (List[int]): 收益率周期列表，如 [1, 5, 20]

**返回值:**
- **pd.DataFrame**: 收益率数据

**示例:**
```python
# 计算多周期收益率
returns = data_manager.calculate_returns(
    prices=price_data,
    periods=[1, 5, 20]  # 1日、5日、20日收益率
)
```

### 配置参数

数据管理器支持以下配置参数：

```python
config = {
    # 数据源设置
    'provider': 'qlib',           # 数据提供商: 'qlib', 'yahoo', 'local'
    'market': 'csi300',           # 股票池: 'csi300', 'csi500', 'all'
    
    # 缓存设置
    'cache_dir': './cache',       # 缓存目录
    'use_cache': True,            # 是否使用缓存
    'cache_expiry': 24,           # 缓存过期时间(小时)
    
    # 数据预处理
    'fill_method': 'forward',     # 缺失值填充方法
    'remove_outliers': True,      # 是否移除异常值
    'outlier_threshold': 3,       # 异常值阈值(标准差倍数)
    
    # 性能优化
    'batch_size': 100,            # 批处理大小
    'parallel': True,             # 是否并行处理
    'n_jobs': 4                   # 并行进程数
}
```

### 异常处理

```python
from data import DataManager, DataError

try:
    data = data_manager.get_data(symbols, start_date, end_date)
except DataError as e:
    print(f"数据获取错误: {e}")
except ValueError as e:
    print(f"参数错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 性能优化建议

1. **使用缓存**: 启用数据缓存以提高重复查询性能
2. **批量获取**: 一次性获取多只股票数据而非逐个获取
3. **合理字段**: 只获取需要的数据字段
4. **并行处理**: 对于大量数据处理启用并行模式

### 注意事项

1. 股票代码需要包含交易所后缀(.SZ, .SH)
2. 日期格式必须为 'YYYY-MM-DD'
3. 数据获取受网络状况影响，建议使用缓存
4. 大量数据查询可能消耗较多内存，注意资源控制