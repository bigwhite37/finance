# API 文档

## 概述

本文档详细描述了强化学习量化交易系统的所有API接口，包括数据接口、模型接口、交易接口和监控接口。

## 目录

- [数据接口](data_interfaces.md) - 数据获取和处理相关API
- [模型接口](model_interfaces.md) - 机器学习模型相关API
- [交易接口](trading_interfaces.md) - 交易执行和管理相关API
- [监控接口](monitoring_interfaces.md) - 系统监控和告警相关API
- [配置接口](config_interfaces.md) - 系统配置管理相关API
- [审计接口](audit_interfaces.md) - 审计日志和合规相关API

## 通用约定

### 响应格式

所有API响应都遵循统一的JSON格式：

```json
{
  "success": true,
  "data": {},
  "message": "操作成功",
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

### 错误处理

错误响应格式：

```json
{
  "success": false,
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "参数验证失败",
    "details": {
      "field": "symbol",
      "reason": "股票代码格式不正确"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

### 认证

系统支持两种认证方式：

1. **API密钥认证**
```http
Authorization: Bearer your_api_key
```

2. **JWT令牌认证**
```http
Authorization: JWT your_jwt_token
```

### 限流

- 默认限制：每分钟100次请求
- 超出限制时返回HTTP 429状态码
- 响应头包含限流信息：
  - `X-RateLimit-Limit`: 限制次数
  - `X-RateLimit-Remaining`: 剩余次数
  - `X-RateLimit-Reset`: 重置时间

## 快速开始

### 1. 获取API密钥

```bash
curl -X POST http://localhost:8000/api/v1/auth/api-key \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### 2. 测试连接

```bash
curl -X GET http://localhost:8000/api/v1/health \
  -H "Authorization: Bearer your_api_key"
```

### 3. 获取股票列表

```bash
curl -X GET http://localhost:8000/api/v1/data/stocks \
  -H "Authorization: Bearer your_api_key"
```

## 版本控制

API使用语义化版本控制，当前版本为v1。版本信息包含在URL路径中：

- `/api/v1/` - 当前稳定版本
- `/api/v2/` - 下一个主要版本（开发中）

## 支持

如有问题，请联系：
- 邮箱：support@trading-system.com
- 文档：https://docs.trading-system.com
- GitHub：https://github.com/trading-system/api