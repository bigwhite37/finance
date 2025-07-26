# 配置管理高级教程

本教程将介绍`ConfigManager`的一些高级用法，帮助您更灵活地管理复杂策略的配置，特别是进行参数调优和多环境管理。

## 1. 点路径操作：快速读写配置

当配置文件结构很深时，使用点分隔的路径（dot-separated path）是访问特定配置项最方便的方式。

```python
from config import ConfigManager

config_manager = ConfigManager('my_config.yaml')

# 读取深层配置
initial_rho = config_manager.get_value('o2o.online_learning.initial_rho')
print(f"初始在线学习采样率: {initial_rho}")

# 修改深层配置
print("将KL散度阈值调整得更敏感...")
config_manager.set_value('o2o.drift_detection.kl_threshold', 0.08)

# 即使中间的字典不存在，set_value也会自动创建
config_manager.set_value('new_module.new_param.value', 123)
```

## 2. O2O配置管理

O2O（离线到在线）训练流程的配置非常复杂。`ConfigManager`提供了一套专门的工具来简化这个过程。

### 2.1 从标准配置迁移到O2O

如果您有一个标准的配置文件，想要为其增加O2O功能，可以使用`migrate_config_to_o2o`方法。

```python
# 加载一个不含o2o节的配置文件
config_manager = ConfigManager('standard_config.yaml')

# 执行迁移（会自动备份原文件）
config_manager.migrate_config_to_o2o()

# 迁移后，配置中会自动加入完整的o2o配置节
# 并将原有的agent, env等配置进行适当调整以兼容O2O

# 可以将迁移后的完整配置保存到新文件
config_manager.save_config('o2o_enabled_config.yaml')
```

### 2.2 O2O配置验证与建议

由于O2O参数众多且相互关联，手动调整很容易出错。`ConfigManager`集成了`O2OConfigValidator`，可以对您的O2O配置进行“体检”。

```python
# 加载一个O2O配置文件
config_manager = ConfigManager('my_o2o_config.yaml')

# 获取验证报告
report = config_manager.get_o2o_validation_report()
print("--- O2O配置验证报告 ---")
print(report)

# 获取优化建议
suggestions = config_manager.get_o2o_optimization_suggestions()
print("\n--- O2O配置优化建议 ---")
for category, advice_list in suggestions.items():
    print(f"[{category.upper()}]:")
    for advice in advice_list:
        print(f"- {advice}")
```
这个功能对于调试和优化复杂的O2O策略至关重要。

## 3. 多环境与参数调优

在进行策略研究时，您经常需要在不同的配置（例如，不同的股票池、不同的风险偏好）下运行实验。`ConfigManager`可以帮助您轻松管理这些变体。

### 3.1 基础配置 + 增量配置

一个最佳实践是维护一个`base_config.yaml`文件，包含所有策略共享的配置。然后为每个实验创建一个小的增量配置文件，只包含与基础配置不同的部分。

**`base_config.yaml`:**
```yaml
agent:
  hidden_dim: 256
  gamma: 0.99
risk_control:
  max_position: 0.1
# ... 其他共享配置
```

**`experiment_aggressive.yaml`:**
```yaml
# 这个文件只定义与base不同的部分
agent:
  learning_rate: 5.0e-4 # 更高的学习率
risk_control:
  target_volatility: 0.15 # 更高的目标波动率
```

**加载方式:**
```python
# 先加载基础配置
config_manager = ConfigManager('base_config.yaml')

# 再加载实验配置，它会自动覆盖基础配置中的同名项
config_manager.load_config('experiment_aggressive.yaml')

# 现在config_manager中的配置就是合并后的结果
# learning_rate是5.0e-4, target_volatility是0.15, hidden_dim依然是256
```

### 3.2 脚本中进行参数扫描

您可以利用`ConfigManager`在Python脚本中轻松实现参数扫描，以进行超参数调优。

```python
import itertools

# 定义要扫描的参数
param_grid = {
    'agent.learning_rate': [1e-4, 3e-4, 5e-4],
    'risk_control.target_volatility': [0.10, 0.12, 0.15]
}

keys, values = zip(*param_grid.items())

# 遍历所有参数组合
for v in itertools.product(*values):
    experiment_params = dict(zip(keys, v))
    
    # 为每次实验创建一个新的ConfigManager实例
    config_manager = ConfigManager('base_config.yaml')
    
    print(f"\n--- 开始新实验: {experiment_params} ---")
    
    # 应用当前实验的参数
    for key_path, value in experiment_params.items():
        config_manager.set_value(key_path, value)
    
    # 在这里运行你的训练和回测流程...
    # run_training_and_backtest(config_manager)
```

通过这些高级功能，`ConfigManager`不仅仅是一个配置文件加载器，更是您进行严肃策略研究和参数调优的得力助手。

