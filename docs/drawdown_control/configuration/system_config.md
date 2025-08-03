# 系统配置文档

## 概述

本文档详细说明回撤控制系统的配置管理，包括系统级配置、组件配置、环境配置以及配置的最佳实践。

## 配置文件结构

### 主配置文件
系统使用 YAML 格式的层次化配置文件，主要包含以下几个部分：

```yaml
# config/drawdown_control_config.yaml
system:
  version: "2.0.0"
  debug_mode: false
  log_level: "INFO"
  
drawdown_control:
  max_drawdown_threshold: 0.15
  drawdown_warning_threshold: 0.08
  
risk_management:
  base_risk_budget: 0.10
  max_position_size: 0.15
  
monitoring:
  enable_real_time: true
  update_interval: 60
  
performance:
  enable_vectorization: true
  cache_size: 1000
```

## 系统级配置

### 基础系统配置

```python
@dataclass
class SystemConfig:
    """系统基础配置"""
    
    # 系统信息
    version: str = "2.0.0"
    environment: str = "production"  # development, testing, production
    debug_mode: bool = False
    
    # 日志配置
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_path: str = "./logs/drawdown_control.log"
    log_max_size: int = 100 * 1024 * 1024  # 100MB
    log_backup_count: int = 5
    
    # 时区和语言
    timezone: str = "Asia/Shanghai"
    locale: str = "zh_CN.UTF-8"
    
    # 安全配置
    enable_authentication: bool = True
    token_expiry_hours: int = 24
    max_api_requests_per_minute: int = 1000
```

### 数据配置

```python
@dataclass
class DataConfig:
    """数据源和存储配置"""
    
    # 数据源配置
    primary_data_source: str = "qlib"  # qlib, akshare, tushare
    backup_data_source: str = "akshare"
    data_update_interval: int = 300  # 5分钟
    
    # 数据库配置
    database_url: str = "sqlite:///./data/drawdown_control.db"
    connection_pool_size: int = 10
    connection_timeout: int = 30
    
    # 缓存配置
    cache_backend: str = "redis"  # redis, memory, file
    cache_host: str = "localhost"
    cache_port: int = 6379
    cache_db: int = 0
    cache_expire_seconds: int = 3600
    
    # 数据存储路径
    data_storage_path: str = "./data"
    backup_storage_path: str = "./data/backup"
    temp_storage_path: str = "/tmp/drawdown_control"
```

### 计算配置

```python
@dataclass
class ComputationConfig:
    """计算和性能配置"""
    
    # 并行计算配置
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    
    # 向量化计算配置
    enable_vectorization: bool = True
    numpy_threads: int = 4
    use_gpu: bool = False
    gpu_device_id: int = 0
    
    # 内存管理配置
    max_memory_usage_gb: float = 8.0
    enable_memory_optimization: bool = True
    garbage_collection_interval: int = 100
    
    # 缓存配置
    enable_computation_cache: bool = True
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 1800
```

## 组件配置

### 回撤监控配置

```yaml
drawdown_monitoring:
  # 基础监控参数
  rolling_window_days: 252
  min_observation_days: 30
  update_frequency: "daily"  # realtime, daily, hourly
  
  # 阈值配置
  warning_threshold: 0.05
  critical_threshold: 0.10
  max_tolerable_drawdown: 0.15
  
  # 计算配置
  calculation_method: "peak_to_trough"  # peak_to_trough, rolling_max
  precision_decimals: 4
  enable_risk_adjustment: true
  
  # 预警配置
  enable_email_alerts: true
  enable_sms_alerts: false
  alert_cooldown_minutes: 30
```

### 动态止损配置

```yaml
dynamic_stop_loss:
  # 基础止损参数
  base_stop_loss_pct: 5.0
  volatility_multiplier: 2.0
  min_stop_loss_pct: 2.0
  max_stop_loss_pct: 15.0
  
  # 追踪止损配置
  enable_trailing_stop: true
  trailing_distance_pct: 3.0
  trailing_activation_pct: 5.0
  
  # 组合级止损配置
  portfolio_stop_loss_pct: 12.0
  sector_stop_loss_pct: 8.0
  enable_correlation_adjustment: true
  
  # 执行配置
  execution_method: "adaptive"  # market, limit, adaptive
  max_market_impact_pct: 1.0
  batch_execution_size: 1000
```

### 风险预算配置

```yaml
risk_budget:
  # 基础风险预算
  base_risk_budget_pct: 10.0
  risk_scaling_factor: 0.5
  recovery_speed_factor: 0.1
  
  # 仓位限制配置
  max_single_position_pct: 15.0
  max_sector_exposure_pct: 30.0
  min_cash_reserve_pct: 5.0
  
  # 动态调整配置
  enable_dynamic_adjustment: true
  adjustment_frequency: "daily"
  lookback_period_days: 60
  volatility_adjustment_factor: 1.5
```

## 环境配置

### 开发环境配置

```yaml
# config/development.yaml
system:
  environment: "development"
  debug_mode: true
  log_level: "DEBUG"

drawdown_control:
  max_drawdown_threshold: 0.20  # 更宽松的阈值用于测试
  
monitoring:
  enable_real_time: false
  update_interval: 300  # 5分钟更新
  
performance:
  enable_vectorization: false  # 便于调试
  max_worker_threads: 2
```

### 测试环境配置

```yaml
# config/testing.yaml
system:
  environment: "testing"
  debug_mode: false
  log_level: "WARNING"

data:
  database_url: "sqlite:///./test_data/test.db"
  use_mock_data: true
  
drawdown_control:
  max_drawdown_threshold: 0.10
  
monitoring:
  enable_email_alerts: false
  enable_sms_alerts: false
```

### 生产环境配置

```yaml
# config/production.yaml
system:
  environment: "production"
  debug_mode: false
  log_level: "INFO"
  enable_authentication: true

data:
  database_url: "postgresql://user:pass@host:5432/drawdown_control"
  cache_backend: "redis"
  
monitoring:
  enable_real_time: true
  update_interval: 60
  enable_email_alerts: true
  
performance:
  enable_vectorization: true
  enable_parallel_processing: true
  max_worker_threads: 8
```

## 配置管理

### 配置加载

```python
from rl_trading_system.config import ConfigManager

# 创建配置管理器
config_manager = ConfigManager()

# 加载配置文件
config = config_manager.load_config("config/drawdown_control_config.yaml")

# 加载环境特定配置
config = config_manager.load_config_with_environment("production")

# 验证配置
is_valid, errors = config_manager.validate_config(config)
if not is_valid:
    print(f"配置验证失败: {errors}")
```

### 动态配置更新

```python
# 热更新配置（不重启系统）
config_manager.update_config_runtime({
    "drawdown_control.max_drawdown_threshold": 0.12,
    "monitoring.update_interval": 90
})

# 配置变更监听
def on_config_changed(key, old_value, new_value):
    print(f"配置变更: {key} {old_value} -> {new_value}")

config_manager.add_change_listener(on_config_changed)
```

### 配置备份和版本控制

```python
# 创建配置备份
backup_id = config_manager.create_backup()

# 回滚到指定备份
config_manager.restore_from_backup(backup_id)

# 获取配置历史
history = config_manager.get_config_history()
```

## 配置验证

### 验证规则

```python
from rl_trading_system.config.validators import ConfigValidator

class DrawdownControlConfigValidator(ConfigValidator):
    """回撤控制配置验证器"""
    
    def validate(self, config):
        errors = []
        
        # 验证阈值范围
        if not 0 < config.max_drawdown_threshold <= 1:
            errors.append("max_drawdown_threshold 必须在 0 和 1 之间")
        
        # 验证警告阈值小于最大阈值
        if config.drawdown_warning_threshold >= config.max_drawdown_threshold:
            errors.append("drawdown_warning_threshold 必须小于 max_drawdown_threshold")
        
        # 验证更新频率
        valid_frequencies = ["realtime", "daily", "hourly"]
        if config.update_frequency not in valid_frequencies:
            errors.append(f"update_frequency 必须是 {valid_frequencies} 之一")
        
        return len(errors) == 0, errors
```

### 配置约束检查

```python
# 注册验证器
config_manager.register_validator("drawdown_control", DrawdownControlConfigValidator())

# 验证配置
is_valid, errors = config_manager.validate_all()
if not is_valid:
    for error in errors:
        print(f"配置错误: {error}")
```

## 配置最佳实践

### 1. 分层配置管理
- **基础配置**: 包含所有默认值的基础配置文件
- **环境配置**: 不同环境（开发、测试、生产）的专用配置
- **用户配置**: 用户或实例特定的配置覆盖

### 2. 敏感信息管理
```python
# 使用环境变量存储敏感信息
import os

database_password = os.getenv('DB_PASSWORD')
api_secret_key = os.getenv('API_SECRET_KEY')

# 配置文件中引用环境变量
database_url: "${DB_HOST}:${DB_PORT}/${DB_NAME}"
```

### 3. 配置文档化
```yaml
# 在配置文件中添加注释说明
drawdown_control:
  # 最大回撤阈值，超过此值将触发风险控制措施
  # 范围: 0.05-0.30，推荐值: 0.15
  max_drawdown_threshold: 0.15
```

### 4. 配置测试
```python
# 为配置编写单元测试
def test_config_validation():
    config = load_test_config()
    assert config.max_drawdown_threshold > 0
    assert config.max_drawdown_threshold <= 1
    assert config.drawdown_warning_threshold < config.max_drawdown_threshold
```

## 常见配置问题

### 1. 性能相关问题
```yaml
# 问题：系统响应慢
# 解决：调整计算配置
performance:
  enable_vectorization: true      # 启用向量化计算
  enable_parallel_processing: true # 启用并行处理
  max_worker_threads: 8          # 增加工作线程数
```

### 2. 内存使用问题
```yaml
# 问题：内存使用过高
# 解决：调整缓存和内存配置
computation:
  max_memory_usage_gb: 4.0       # 限制最大内存使用
  cache_size_mb: 256             # 减少缓存大小
  enable_memory_optimization: true # 启用内存优化
```

### 3. 数据同步问题
```yaml
# 问题：数据更新不及时
# 解决：调整监控频率
monitoring:
  update_interval: 30            # 减少更新间隔到30秒
  enable_real_time: true         # 启用实时监控
```

## 相关文档

- [组件配置文档](./component_config.md) - 各组件详细配置
- [参数调优指南](./parameter_tuning.md) - 性能调优建议
- [部署配置指南](../deployment/deployment_guide.md) - 生产环境部署配置