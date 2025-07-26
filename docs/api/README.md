# API文档

本目录包含系统各模块的详细API文档。

## 核心模块API

### 数据管理模块 (data/)
- [数据管理器API](data_manager.md) - 数据获取、预处理和管理接口
- [离线数据集API](offline_dataset.md) - 离线数据集处理接口

### 因子工程模块 (factors/)
- [因子引擎API](factor_engine.md) - 因子计算和管理接口
- [Alpha因子API](alpha_factors.md) - Alpha因子计算接口
- [风险因子API](risk_factors.md) - 风险因子计算接口

### 强化学习模块 (rl_agent/)
- [交易环境API](trading_environment.md) - 强化学习交易环境接口
- [CVaR-PPO智能体API](cvar_ppo_agent.md) - 强化学习智能体接口
- [安全保护层API](safety_shield.md) - 风险控制和安全约束接口

### 风险控制模块 (risk_control/)
- [风险控制器API](risk_controller.md) - 主风险控制接口
- [目标波动率控制API](target_volatility.md) - 波动率控制接口
- [风险平价API](risk_parity.md) - 风险平价优化接口
- [动态止损API](stop_loss.md) - 止损管理接口

### 回测评估模块 (backtest/)
- [回测引擎API](backtest_engine.md) - 回测系统核心接口
- [性能分析器API](performance_analyzer.md) - 绩效分析接口
- [舒适度指标API](comfort_metrics.md) - 心理舒适度计算接口

### 配置管理模块 (config/)
- [配置管理器API](config_manager.md) - 系统配置管理接口

### 工具模块 (utils/)
- [日志工具API](logger.md) - 日志系统接口
- [指标计算API](metrics.md) - 各类指标计算接口
- [可视化工具API](visualization.md) - 图表和可视化接口

## 使用说明

### 基本使用方式
```python
from config import ConfigManager
from data import DataManager
from factors import FactorEngine

# 初始化配置
config = ConfigManager()

# 初始化数据管理器
data_manager = DataManager(config.get_data_config())

# 初始化因子引擎
factor_engine = FactorEngine(config.get_config('factors'))
```

### 接口设计原则
1. **一致性**: 所有模块遵循统一的接口设计规范
2. **简洁性**: 提供简单易用的高级接口
3. **灵活性**: 支持高级用户的自定义需求
4. **安全性**: 内置参数验证和异常处理

### 错误处理
系统采用直接异常抛出的方式，便于调试和问题定位：
```python
try:
    result = data_manager.get_data(symbols, start_date, end_date)
except ValueError as e:
    print(f"参数错误: {e}")
except Exception as e:
    print(f"系统错误: {e}")
```

### 配置管理
所有模块都支持通过配置文件进行参数设置：
```python
# 使用默认配置
config = ConfigManager()

# 使用自定义配置文件
config = ConfigManager('my_config.yaml')

# 获取特定模块配置
data_config = config.get_data_config()
```

## 开发指南

### 添加新模块API文档
1. 在相应目录下创建模块API文档
2. 包含类、方法、参数的详细说明
3. 提供完整的使用示例
4. 更新本README文件的索引

### 文档格式规范
- 使用Markdown格式
- 包含方法签名、参数说明、返回值说明
- 提供实际使用示例
- 注明注意事项和限制条件