# 回撤监控组件 (DrawdownMonitor)

## 概述

DrawdownMonitor 是回撤控制系统的核心组件，负责实时监控投资组合的回撤情况，提供准确的回撤计算、阶段识别和预警功能。

## 功能特性

### 1. 实时回撤计算
- **滚动最大回撤计算**: 基于滑动窗口的最大回撤监控
- **当前回撤水平**: 实时计算当前回撤相对于历史峰值的百分比
- **回撤持续时间**: 追踪回撤开始以来的持续时间
- **恢复时间预估**: 基于历史数据预估回撤恢复所需时间

### 2. 回撤阶段识别
- **开始阶段**: 检测回撤开始的时点和触发条件
- **持续阶段**: 监控回撤深化过程和风险累积
- **恢复阶段**: 识别回撤结束和组合价值恢复
- **稳定阶段**: 确认回撤完全恢复并进入新的增长周期

### 3. 多维度监控
- **绝对回撤**: 基于组合净值的绝对回撤计算
- **相对回撤**: 相对于基准指数的超额回撤
- **风险调整回撤**: 考虑波动率调整的风险回撤指标
- **分层回撤**: 按资产类别、行业、个股的分层回撤分析

## API 接口

### 初始化

```python
from rl_trading_system.risk_control.drawdown_monitor import DrawdownMonitor

# 创建监控器实例
monitor = DrawdownMonitor(
    window_size=252,  # 监控窗口大小（交易日）
    threshold=0.05,   # 回撤警告阈值
    enable_real_time=True  # 启用实时监控
)
```

### 核心方法

#### 计算回撤指标

```python
# 更新组合价值并计算回撤
drawdown_metrics = monitor.update_portfolio_value(
    current_value=1050000.0,
    timestamp=datetime.now()
)

# 批量计算历史回撤
historical_drawdowns = monitor.calculate_historical_drawdown(
    portfolio_values=value_series,
    dates=date_series
)
```

#### 回撤阶段识别

```python
# 获取当前回撤阶段
current_stage = monitor.get_current_drawdown_stage()

# 检查是否处于回撤状态
is_in_drawdown = monitor.is_in_drawdown()

# 获取回撤持续时间
duration = monitor.get_drawdown_duration()
```

#### 预警和通知

```python
# 检查是否触发警告
warnings = monitor.check_warning_conditions()

# 获取风险等级
risk_level = monitor.get_risk_level()

# 生成监控报告
report = monitor.generate_monitoring_report()
```

## 配置参数

### 基础配置

```python
@dataclass
class DrawdownMonitorConfig:
    # 监控窗口配置
    rolling_window: int = 252          # 滚动窗口大小
    min_observation_days: int = 30     # 最小观察天数
    
    # 回撤阈值配置
    warning_threshold: float = 0.05    # 警告阈值
    critical_threshold: float = 0.10   # 严重阈值
    max_tolerable_drawdown: float = 0.15  # 最大可容忍回撤
    
    # 计算精度配置
    precision: int = 4                 # 计算精度（小数位数）
    update_frequency: str = "daily"    # 更新频率
    
    # 阶段识别配置
    stage_confirmation_days: int = 3   # 阶段确认天数
    recovery_confirmation: float = 0.02  # 恢复确认阈值
```

### 高级配置

```python
# 性能优化配置
enable_vectorized_calculation: bool = True  # 启用向量化计算
enable_caching: bool = True                 # 启用缓存
cache_size: int = 1000                      # 缓存大小

# 实时监控配置
real_time_enabled: bool = False             # 实时监控开关
monitoring_interval: int = 60               # 监控间隔（秒）
async_processing: bool = False              # 异步处理开关
```

## 数据结构

### 回撤指标结构

```python
@dataclass
class DrawdownMetrics:
    current_drawdown: float           # 当前回撤水平
    max_drawdown: float              # 历史最大回撤
    drawdown_duration: int           # 回撤持续天数
    days_to_recovery: Optional[int]  # 预估恢复天数
    peak_value: float                # 历史峰值
    trough_value: float              # 回撤谷底值
    current_stage: str               # 当前阶段
    risk_level: str                  # 风险等级
    last_updated: datetime           # 最后更新时间
```

### 监控状态结构

```python
@dataclass
class MonitoringState:
    is_monitoring: bool              # 监控状态
    start_time: datetime            # 监控开始时间
    total_observations: int         # 总观察次数
    warning_count: int              # 警告次数
    last_warning_time: Optional[datetime]  # 最后警告时间
```

## 使用示例

### 基础使用

```python
import pandas as pd
from datetime import datetime, timedelta
from rl_trading_system.risk_control.drawdown_monitor import DrawdownMonitor

# 创建监控器
monitor = DrawdownMonitor(
    window_size=252,
    threshold=0.05
)

# 模拟组合价值序列
dates = pd.date_range('2023-01-01', periods=100, freq='D')
portfolio_values = [1000000 * (1 + np.random.normal(0.001, 0.02)) 
                   for _ in range(100)]

# 逐日更新监控
for date, value in zip(dates, portfolio_values):
    metrics = monitor.update_portfolio_value(value, date)
    
    if metrics.current_drawdown > 0.05:
        print(f"警告: {date.strftime('%Y-%m-%d')} 回撤达到 {metrics.current_drawdown:.2%}")
```

### 高级使用

```python
# 启用实时监控
monitor = DrawdownMonitor(
    window_size=252,
    threshold=0.05,
    enable_real_time=True,
    monitoring_interval=60
)

# 设置回调函数
def on_warning(metrics):
    print(f"回撤警告: {metrics.current_drawdown:.2%}")

def on_critical(metrics):
    print(f"严重回撤: {metrics.current_drawdown:.2%}")

monitor.set_warning_callback(on_warning)
monitor.set_critical_callback(on_critical)

# 开始实时监控
monitor.start_real_time_monitoring()
```

### 批量分析

```python
# 批量分析历史数据
historical_data = pd.read_csv('portfolio_history.csv')

analysis_result = monitor.analyze_historical_drawdowns(
    portfolio_values=historical_data['portfolio_value'],
    dates=historical_data['date'],
    benchmark_values=historical_data['benchmark_value']
)

print(f"历史最大回撤: {analysis_result.max_drawdown:.2%}")
print(f"平均回撤持续时间: {analysis_result.avg_duration} 天")
print(f"回撤频率: {analysis_result.drawdown_frequency} 次/年")
```

## 性能特征

### 计算复杂度
- **时间复杂度**: O(n) 对于 n 个观察值
- **空间复杂度**: O(w) 对于窗口大小 w
- **更新复杂度**: O(1) 对于单次更新

### 性能基准
- **处理速度**: >10,000 数据点/秒
- **内存使用**: <100MB 对于252天窗口
- **延迟**: <1ms 对于单次计算

### 扩展性
- 支持多线程并行计算
- 支持分布式计算（通过Redis缓存）
- 支持流式数据处理

## 注意事项

### 数据质量要求
1. **连续性**: 确保价值序列连续，避免缺失值
2. **时间戳**: 提供准确的时间戳信息
3. **数值精度**: 使用足够精度的浮点数

### 性能优化建议
1. **批量更新**: 尽量使用批量更新减少计算开销
2. **缓存利用**: 启用缓存提升重复计算性能
3. **窗口设置**: 根据需求合理设置监控窗口大小

### 常见问题
1. **初始期处理**: 观察期不足时使用可用数据计算
2. **异常值处理**: 自动检测和处理异常的价值变动
3. **时区处理**: 确保时间戳时区一致性

## 相关组件

- [归因分析组件](./attribution_analyzer.md) - 回撤原因分析
- [动态止损组件](./dynamic_stop_loss.md) - 基于回撤的止损控制
- [风险预算组件](./adaptive_risk_budget.md) - 风险预算调整
- [市场状态感知组件](./market_regime_detector.md) - 市场环境分析