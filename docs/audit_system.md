# 审计日志系统文档

## 概述

审计日志系统是量化交易系统的重要组成部分，负责记录所有交易决策、执行过程和风险事件，确保系统的透明度、可追溯性和合规性。

## 系统架构

### 核心组件

1. **AuditLogger**: 主要的审计日志记录器
2. **AuditQueryInterface**: 审计记录查询接口
3. **DataRetentionManager**: 数据保留管理器
4. **ComplianceReportGenerator**: 合规报告生成器

### 数据存储

- **InfluxDB**: 时序数据存储，用于高频审计数据
- **PostgreSQL**: 关系数据存储，用于结构化查询和报告

## 功能特性

### 1. 交易决策记录

系统自动记录每个交易决策的完整信息：

- 输入状态（市场特征、持仓状态）
- 输出动作（目标权重、置信度）
- 模型输出（Q值、损失函数值）
- 特征重要性分析
- 执行时间统计

```python
await audit_logger.log_trading_decision(
    session_id="session_001",
    model_version="v1.0.0",
    input_state=trading_state,
    output_action=trading_action,
    model_outputs={"q_values": [0.1, 0.2, 0.3]},
    feature_importance={"rsi": 0.3, "macd": 0.2},
    execution_time_ms=15.5
)
```

### 2. 交易执行记录

记录实际交易执行的详细信息：

- 交易基本信息（股票代码、数量、价格）
- 成本分析（手续费、印花税、滑点）
- 执行详情（订单ID、成交比例、执行场所）

```python
await audit_logger.log_transaction_execution(
    session_id="session_001",
    transaction=transaction_record,
    execution_details={
        "order_id": "ORD_001",
        "fill_ratio": 1.0,
        "execution_venue": "SZSE"
    }
)
```

### 3. 风险违规记录

自动检测和记录风险违规事件：

- 持仓集中度超限
- 单一持仓权重过高
- 流动性风险
- 模型异常行为

```python
await audit_logger.log_risk_violation(
    session_id="session_001",
    model_version="v1.0.0",
    violation_type="concentration_limit",
    violation_details={
        "threshold": 0.3,
        "actual_value": 0.45,
        "severity": "high"
    }
)
```

### 4. 审计记录查询

提供灵活的查询接口：

```python
# 按时间范围查询
records = await query_interface.query_by_time_range(
    start_time=start_time,
    end_time=end_time,
    event_type="trading_decision"
)

# 按会话查询
records = await query_interface.query_by_session("session_001")

# 按模型版本查询
records = await query_interface.query_by_model_version("v1.0.0")

# 获取决策详情
decision = await query_interface.get_decision_details("decision_001")
```

### 5. 数据保留管理

自动管理历史数据：

- 按配置的保留期限自动清理过期数据
- 提供数据统计和存储分析
- 支持数据备份和恢复

```python
# 获取数据统计
stats = await retention_manager.get_data_statistics()

# 手动清理过期数据
await retention_manager.cleanup_expired_data()
```

### 6. 合规报告生成

自动生成合规分析报告：

- 决策统计分析
- 风险违规汇总
- 持仓集中度分析
- 模型性能评估
- 合规分数计算

```python
report = await report_generator.generate_compliance_report(
    period_start=start_time,
    period_end=end_time
)
```

## 数据模型

### AuditRecord（审计记录）

```python
@dataclass
class AuditRecord:
    record_id: str
    timestamp: datetime
    event_type: str  # trading_decision, transaction_execution, risk_violation
    user_id: str
    session_id: str
    model_version: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
```

### DecisionRecord（决策记录）

```python
@dataclass
class DecisionRecord:
    decision_id: str
    timestamp: datetime
    model_version: str
    input_state: TradingState
    output_action: TradingAction
    model_outputs: Dict[str, Any]
    feature_importance: Dict[str, float]
    risk_metrics: Dict[str, float]
    execution_time_ms: Optional[float]
```

### ComplianceReport（合规报告）

```python
@dataclass
class ComplianceReport:
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_decisions: int
    risk_violations: List[Dict[str, Any]]
    concentration_analysis: Dict[str, float]
    model_performance: Dict[str, float]
    compliance_score: float
```

## 配置说明

### 基本配置

```yaml
# 数据库配置
influxdb:
  url: "http://localhost:8086"
  token: "${INFLUXDB_TOKEN}"
  org: "trading"
  bucket: "audit"

relational_db:
  url: "${POSTGRES_URL}"
  pool_size: 10

# 审计日志器配置
audit_logger:
  batch_size: 100
  flush_interval: 60
  max_retries: 3

# 数据保留策略
data_retention:
  retention_days: 1825  # 5年
  cleanup_interval_hours: 24
```

### 环境变量

系统需要以下环境变量：

- `INFLUXDB_TOKEN`: InfluxDB访问令牌
- `POSTGRES_URL`: PostgreSQL连接字符串
- `AUDIT_ENCRYPTION_KEY`: 数据加密密钥（可选）

## 部署指南

### 1. 数据库准备

#### InfluxDB设置

```bash
# 启动InfluxDB
docker run -d -p 8086:8086 \
  -v influxdb-storage:/var/lib/influxdb2 \
  influxdb:2.0

# 创建组织和bucket
influx org create -n trading
influx bucket create -n audit -o trading
```

#### PostgreSQL设置

```bash
# 启动PostgreSQL
docker run -d -p 5432:5432 \
  -e POSTGRES_DB=audit \
  -e POSTGRES_USER=audit_user \
  -e POSTGRES_PASSWORD=audit_password \
  -v postgres-data:/var/lib/postgresql/data \
  postgres:13
```

### 2. 系统启动

```python
import asyncio
from src.rl_trading_system.audit.audit_logger import AuditLogger

async def main():
    config = {
        'influxdb': {
            'url': 'http://localhost:8086',
            'token': 'your_token',
            'org': 'trading',
            'bucket': 'audit'
        },
        'relational_db_url': 'postgresql://user:password@localhost:5432/audit',
        'batch_size': 100,
        'flush_interval': 60
    }
    
    audit_logger = AuditLogger(config)
    await audit_logger.start()
    
    # 系统运行...
    
    await audit_logger.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 性能优化

### 1. 批处理优化

- 调整`batch_size`参数以平衡内存使用和写入性能
- 根据系统负载调整`flush_interval`

### 2. 数据库优化

- 为常用查询字段创建索引
- 定期分析和优化查询性能
- 考虑数据分区策略

### 3. 内存管理

- 监控批处理缓冲区大小
- 及时释放不需要的对象引用
- 使用连接池管理数据库连接

## 监控和告警

### 关键指标

- 审计记录写入延迟
- 数据库连接池状态
- 批处理队列长度
- 存储空间使用率

### 告警规则

- 写入失败率超过阈值
- 查询响应时间过长
- 存储空间不足
- 数据库连接异常

## 故障排除

### 常见问题

1. **数据库连接失败**
   - 检查数据库服务状态
   - 验证连接字符串和认证信息
   - 检查网络连接

2. **写入性能问题**
   - 调整批处理大小
   - 检查数据库性能
   - 优化数据结构

3. **查询超时**
   - 检查查询条件和索引
   - 优化查询语句
   - 考虑数据分页

### 日志分析

系统提供详细的日志信息，包括：

- 操作执行时间
- 错误详情和堆栈跟踪
- 性能统计信息
- 数据库操作记录

## 合规要求

### 数据保留

- 交易决策记录：至少保留5年
- 风险违规记录：至少保留7年
- 合规报告：至少保留7年

### 数据安全

- 支持数据加密存储
- 访问日志记录
- 数据完整性验证

### 审计追踪

- 所有操作都有完整的审计记录
- 支持按时间、用户、操作类型查询
- 提供数据变更历史

## API参考

详细的API文档请参考代码中的docstring和类型注解。主要接口包括：

- `AuditLogger`: 审计日志记录
- `AuditQueryInterface`: 审计记录查询
- `DataRetentionManager`: 数据保留管理
- `ComplianceReportGenerator`: 合规报告生成

## 最佳实践

1. **及时记录**: 在关键操作后立即记录审计信息
2. **结构化数据**: 使用标准化的数据格式和字段名
3. **错误处理**: 确保审计记录不影响主业务流程
4. **性能监控**: 定期监控系统性能和资源使用
5. **定期备份**: 建立完善的数据备份和恢复机制