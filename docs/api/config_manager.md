# 配置管理器API文档

## ConfigManager 类

`ConfigManager`是整个系统的配置中心。它负责加载、管理、验证和提供所有模块的配置参数。它支持从YAML或JSON文件加载配置，并与一套默认配置进行合并，确保系统的稳健运行。

### 类签名
```python
class ConfigManager:
    def __init__(self, config_path: Optional[str] = None)
```

### 初始化参数
-   **config_path** (Optional[str]): 配置文件的路径。如果提供，将加载该文件并与默认配置合并。如果为None，则只使用默认配置。

### 核心方法

#### load_config / save_config
从指定路径加载或保存配置文件。

```python
def load_config(self, config_path: str)
def save_config(self, config_path: str)
```

#### get_config
获取指定节（section）的配置，或完整的配置字典。

```python
def get_config(self, section: Optional[str] = None) -> Dict[str, Any]
```
**示例:**
```python
# 获取完整的配置
all_config = config_manager.get_config()

# 获取agent节的配置
agent_config = config_manager.get_config('agent')
```

#### get_value / set_value
通过点分隔的路径（dot-separated path）方便地获取或设置单个配置项。

```python
def get_value(self, key_path: str, default: Any = None) -> Any
def set_value(self, key_path: str, value: Any)
```
**示例:**
```python
# 获取学习率
lr = config_manager.get_value('agent.learning_rate', default=0.001)

# 设置目标波动率
config_manager.set_value('risk_control.target_volatility', 0.10)
```

#### validate_config
对当前加载的配置进行全面的有效性验证。它会检查必需的配置节是否存在，关键参数是否在合理范围内，并对`dynamic_lowvol`和`o2o`等复杂模块的配置进行专项验证。

```python
def validate_config(self) -> bool
```

#### print_config_summary
在控制台打印一份清晰的配置摘要，方便快速检查关键参数。

```python
def print_config_summary(self)
```

### 便捷的配置获取方法

`ConfigManager`提供了一系列便捷方法来直接获取特定模块的配置，例如：
-   `get_data_config()`
-   `get_agent_config()`
-   `get_environment_config()`
-   `get_risk_control_config()`
-   `get_backtest_config()`
-   `get_o2o_config()`

### O2O配置管理

`ConfigManager`为复杂的O2O（离线到在线）训练流程提供了强大的支持：

-   **`create_o2o_template()`**: 生成一份完整的O2O配置模板。
-   **`migrate_config_to_o2o()`**: 将现有的标准配置无缝迁移到支持O2O的版本。
-   **`get_o2o_validation_report()`**: 生成一份详细的O2O配置验证报告。
-   **`get_o2o_optimization_suggestions()`**: 基于当前配置，提供一系列优化建议。

### 使用示例

```python
from config import ConfigManager

# 1. 使用默认配置初始化
config_manager = ConfigManager()

# 2. 加载自定义配置文件，它会覆盖默认值
config_manager.load_config('my_strategy.yaml')

# 3. 在代码中动态修改配置
config_manager.set_value('agent.batch_size', 128)

# 4. 验证最终配置的有效性
try:
    config_manager.validate_config()
    print("配置验证通过！")
except ValueError as e:
    print(f"配置错误: {e}")

# 5. 打印摘要
config_manager.print_config_summary()

# 6. 将最终配置传递给其他模块
agent_config = config_manager.get_agent_config()
# my_agent = Agent(agent_config)
```
