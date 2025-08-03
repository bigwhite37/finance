# RESTful API 接口文档

## API 概述

回撤控制系统提供完整的RESTful API接口，支持实时监控、风险控制、数据查询和系统管理等功能。所有API都遵循REST规范，使用JSON格式进行数据交换。

### 基础信息

- **Base URL**: `https://api.drawdown-control.com/v1`
- **认证方式**: Bearer Token / API Key
- **内容类型**: `application/json`
- **字符编码**: UTF-8
- **API版本**: v1.0

### 通用响应格式

所有API响应都遵循统一的格式：

```json
{
    "success": true,
    "code": 200,
    "message": "操作成功",
    "data": {
        // 具体的业务数据
    },
    "timestamp": "2025-08-02T10:30:00Z",
    "request_id": "req_12345678"
}
```

### 错误响应格式

```json
{
    "success": false,
    "code": 400,
    "message": "请求参数错误",
    "error": {
        "type": "ValidationError",
        "details": [
            {
                "field": "portfolio_id",
                "message": "投资组合ID不能为空"
            }
        ]
    },
    "timestamp": "2025-08-02T10:30:00Z",
    "request_id": "req_12345678"
}
```

## 1. 认证接口

### 1.1 获取访问令牌

**请求**:
```http
POST /auth/token
Content-Type: application/json

{
    "username": "user123",
    "password": "password123",
    "grant_type": "password"
}
```

**响应**:
```json
{
    "success": true,
    "data": {
        "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "scope": "read write"
    }
}
```

### 1.2 刷新访问令牌

**请求**:
```http
POST /auth/refresh
Content-Type: application/json
Authorization: Bearer {refresh_token}

{
    "grant_type": "refresh_token"
}
```

## 2. 回撤监控接口

### 2.1 获取实时回撤数据

**请求**:
```http
GET /drawdown/monitor/{portfolio_id}
Authorization: Bearer {access_token}
```

**参数**:
- `portfolio_id` (string): 投资组合ID

**响应**:
```json
{
    "success": true,
    "data": {
        "portfolio_id": "portfolio_001",
        "current_drawdown": -0.0856,
        "max_drawdown": -0.1234,
        "drawdown_duration": 15,
        "peak_value": 1000000.0,
        "current_value": 914400.0,
        "drawdown_phase": "recovery",
        "risk_level": "medium",
        "volatility": 0.0234,
        "sharpe_ratio": 1.45,
        "calmar_ratio": 2.18,
        "last_updated": "2025-08-02T10:29:45Z"
    }
}
```

### 2.2 更新投资组合数据

**请求**:
```http
POST /drawdown/monitor/{portfolio_id}/update
Authorization: Bearer {access_token}
Content-Type: application/json

{
    "portfolio_values": [950000, 960000, 940000, 955000],
    "timestamps": [
        "2025-08-02T10:25:00Z",
        "2025-08-02T10:26:00Z", 
        "2025-08-02T10:27:00Z",
        "2025-08-02T10:28:00Z"
    ],
    "positions": {
        "AAPL": 0.25,
        "MSFT": 0.20,
        "GOOGL": 0.15,
        "TSLA": 0.10,
        "CASH": 0.30
    }
}
```

**响应**:
```json
{
    "success": true,
    "data": {
        "updated_metrics": {
            "current_drawdown": -0.0789,
            "max_drawdown": -0.1234,
            "risk_level": "medium"
        },
        "alerts_triggered": [
            {
                "type": "drawdown_warning",
                "severity": "medium",
                "message": "回撤水平接近预警阈值",
                "threshold": 0.08,
                "current_value": 0.0789
            }
        ]
    }
}
```

### 2.3 获取历史回撤数据

**请求**:
```http
GET /drawdown/monitor/{portfolio_id}/history
Authorization: Bearer {access_token}
```

**查询参数**:
- `start_date` (string): 开始日期 (YYYY-MM-DD)
- `end_date` (string): 结束日期 (YYYY-MM-DD)
- `granularity` (string): 数据粒度 (minute/hour/day)
- `limit` (integer): 返回记录数限制

**响应**:
```json
{
    "success": true,
    "data": {
        "portfolio_id": "portfolio_001",
        "granularity": "hour",
        "total_records": 168,
        "drawdown_history": [
            {
                "timestamp": "2025-08-01T00:00:00Z",
                "portfolio_value": 1000000.0,
                "drawdown": 0.0,
                "cumulative_return": 0.0
            },
            {
                "timestamp": "2025-08-01T01:00:00Z",
                "portfolio_value": 995000.0,
                "drawdown": -0.005,
                "cumulative_return": -0.005
            }
        ]
    }
}
```

## 3. 归因分析接口

### 3.1 获取回撤归因分析

**请求**:
```http
GET /attribution/analyze/{portfolio_id}
Authorization: Bearer {access_token}
```

**查询参数**:
- `analysis_period` (string): 分析周期 (1d/1w/1m/3m)
- `include_breakdown` (boolean): 是否包含详细分解

**响应**:
```json
{
    "success": true,
    "data": {
        "portfolio_id": "portfolio_001",
        "analysis_period": "1m",
        "total_drawdown": -0.1234,
        "attribution_breakdown": {
            "stock_specific": {
                "contribution": -0.0456,
                "percentage": 36.9,
                "top_contributors": [
                    {
                        "symbol": "TSLA",
                        "contribution": -0.0234,
                        "weight": 0.10
                    },
                    {
                        "symbol": "NVDA", 
                        "contribution": -0.0156,
                        "weight": 0.08
                    }
                ]
            },
            "sector_allocation": {
                "contribution": -0.0345,
                "percentage": 28.0,
                "breakdown": [
                    {
                        "sector": "Technology",
                        "contribution": -0.0234,
                        "weight": 0.45
                    },
                    {
                        "sector": "Healthcare",
                        "contribution": -0.0111,
                        "weight": 0.20
                    }
                ]
            },
            "factor_exposure": {
                "contribution": -0.0234,
                "percentage": 19.0,
                "factors": [
                    {
                        "factor": "Market Beta",
                        "exposure": 1.25,
                        "contribution": -0.0123
                    },
                    {
                        "factor": "Size Factor",
                        "exposure": -0.34,
                        "contribution": -0.0067
                    }
                ]
            },
            "interaction_effects": {
                "contribution": -0.0123,
                "percentage": 10.0
            },
            "unexplained": {
                "contribution": -0.0076,
                "percentage": 6.1
            }
        },
        "confidence_level": 0.95,
        "generated_at": "2025-08-02T10:30:00Z"
    }
}
```

## 4. 动态止损接口

### 4.1 获取止损设置

**请求**:
```http
GET /stop-loss/{portfolio_id}/settings
Authorization: Bearer {access_token}
```

**响应**:
```json
{
    "success": true,
    "data": {
        "portfolio_id": "portfolio_001",
        "stop_loss_config": {
            "enabled": true,
            "strategy": "dynamic_volatility",
            "base_stop_loss": 0.05,
            "trailing_stop_enabled": true,
            "trailing_distance": 0.02,
            "volatility_adjustment": true,
            "max_stop_loss": 0.15,
            "min_stop_loss": 0.02
        },
        "current_stop_levels": {
            "portfolio_level": {
                "stop_loss_price": 914400.0,
                "stop_loss_ratio": 0.0856,
                "triggered": false
            },
            "position_level": [
                {
                    "symbol": "AAPL",
                    "current_price": 175.50,
                    "stop_loss_price": 166.73,
                    "stop_loss_ratio": 0.0500,
                    "triggered": false
                },
                {
                    "symbol": "TSLA",
                    "current_price": 245.30,
                    "stop_loss_price": 225.02,
                    "stop_loss_ratio": 0.0827,
                    "triggered": false
                }
            ]
        }
    }
}
```

### 4.2 更新止损设置

**请求**:
```http
PUT /stop-loss/{portfolio_id}/settings
Authorization: Bearer {access_token}
Content-Type: application/json

{
    "strategy": "dynamic_volatility",
    "base_stop_loss": 0.06,
    "trailing_stop_enabled": true,
    "trailing_distance": 0.025,
    "volatility_adjustment": true,
    "portfolio_level_enabled": true,
    "position_level_enabled": true
}
```

### 4.3 手动触发止损

**请求**:
```http
POST /stop-loss/{portfolio_id}/trigger
Authorization: Bearer {access_token}
Content-Type: application/json

{
    "trigger_type": "manual",
    "scope": "portfolio", // or "position"
    "symbol": null, // 仅在scope为position时需要
    "reason": "手动风险控制"
}
```

## 5. 风险预算接口

### 5.1 获取当前风险预算

**请求**:
```http
GET /risk-budget/{portfolio_id}
Authorization: Bearer {access_token}
```

**响应**:
```json
{
    "success": true,
    "data": {
        "portfolio_id": "portfolio_001",
        "current_risk_budget": 0.12,
        "base_risk_budget": 0.10,
        "adjustment_factors": {
            "performance_factor": 1.15,
            "market_factor": 0.95,
            "volatility_factor": 1.08,
            "drawdown_factor": 0.92
        },
        "risk_utilization": {
            "used_budget": 0.089,
            "available_budget": 0.031,
            "utilization_rate": 0.742
        },
        "recent_adjustments": [
            {
                "timestamp": "2025-08-02T09:00:00Z",
                "old_budget": 0.11,
                "new_budget": 0.12,
                "reason": "市场波动率降低",
                "adjustment_type": "automatic"
            }
        ]
    }
}
```

### 5.2 手动调整风险预算

**请求**:
```http
POST /risk-budget/{portfolio_id}/adjust
Authorization: Bearer {access_token}
Content-Type: application/json

{
    "new_risk_budget": 0.08,
    "adjustment_type": "manual",
    "reason": "市场不确定性增加",
    "effective_immediately": true
}
```

## 6. 市场制度检测接口

### 6.1 获取当前市场制度

**请求**:
```http
GET /market-regime/current
Authorization: Bearer {access_token}
```

**响应**:
```json
{
    "success": true,
    "data": {
        "current_regime": "bull_market",
        "confidence": 0.87,
        "regime_probability": {
            "bull_market": 0.87,
            "bear_market": 0.05,
            "sideways": 0.06,
            "high_volatility": 0.02
        },
        "market_indicators": {
            "trend_strength": 0.73,
            "volatility_level": 0.18,
            "uncertainty_index": 0.32,
            "correlation_breakdown": false
        },
        "regime_duration": 45,
        "expected_transition_probability": {
            "remain_bull": 0.92,
            "to_sideways": 0.05,
            "to_bear": 0.02,
            "to_crisis": 0.01
        },
        "last_updated": "2025-08-02T10:30:00Z"
    }
}
```

## 7. 压力测试接口

### 7.1 执行压力测试

**请求**:
```http
POST /stress-test/{portfolio_id}/run
Authorization: Bearer {access_token}
Content-Type: application/json

{
    "test_scenarios": [
        "market_crash_2008",
        "covid_crash_2020", 
        "inflation_spike",
        "liquidity_crisis"
    ],
    "custom_scenario": {
        "name": "自定义情景",
        "market_shock": -0.20,
        "volatility_spike": 2.5,
        "correlation_increase": 0.9,
        "duration_days": 30
    },
    "monte_carlo_runs": 10000,
    "confidence_levels": [0.95, 0.99, 0.999]
}
```

**响应**:
```json
{
    "success": true,
    "data": {
        "test_id": "stress_test_20250802_103000",
        "portfolio_id": "portfolio_001",
        "test_status": "completed",
        "summary_results": {
            "worst_case_loss": -0.285,
            "var_95": -0.156,
            "var_99": -0.223,
            "var_99_9": -0.285,
            "expected_shortfall_95": -0.189,
            "max_drawdown_estimate": -0.312
        },
        "scenario_results": [
            {
                "scenario": "market_crash_2008",
                "probability": 0.02,
                "estimated_loss": -0.245,
                "recovery_time_estimate": "18个月"
            },
            {
                "scenario": "covid_crash_2020",
                "probability": 0.05,
                "estimated_loss": -0.189,
                "recovery_time_estimate": "8个月"
            }
        ],
        "risk_recommendations": [
            "建议降低风险预算至8%",
            "增加防御性资产配置",
            "考虑对冲策略"
        ],
        "generated_at": "2025-08-02T10:35:00Z"
    }
}
```

## 8. 监控和告警接口

### 8.1 获取告警历史

**请求**:
```http
GET /alerts/{portfolio_id}
Authorization: Bearer {access_token}
```

**查询参数**:
- `severity` (string): 告警级别 (low/medium/high/critical)
- `status` (string): 告警状态 (active/acknowledged/resolved)
- `limit` (integer): 返回记录数

**响应**:
```json
{
    "success": true,
    "data": {
        "total_alerts": 15,
        "active_alerts": 3,
        "alerts": [
            {
                "alert_id": "alert_001",
                "type": "drawdown_threshold",
                "severity": "high",
                "status": "active",
                "message": "投资组合回撤超过15%阈值",
                "current_value": -0.1678,
                "threshold": -0.15,
                "triggered_at": "2025-08-02T09:45:00Z",
                "acknowledged_at": null,
                "acknowledged_by": null
            }
        ]
    }
}
```

### 8.2 确认告警

**请求**:
```http
POST /alerts/{alert_id}/acknowledge
Authorization: Bearer {access_token}
Content-Type: application/json

{
    "acknowledged_by": "user123",
    "comment": "已知悉，正在采取措施"
}
```

## 9. 系统管理接口

### 9.1 获取系统状态

**请求**:
```http
GET /system/health
Authorization: Bearer {access_token}
```

**响应**:
```json
{
    "success": true,
    "data": {
        "overall_status": "healthy",
        "components": {
            "drawdown_monitor": {
                "status": "healthy",
                "response_time_ms": 15,
                "last_update": "2025-08-02T10:30:00Z"
            },
            "risk_budget": {
                "status": "healthy",
                "response_time_ms": 8,
                "last_update": "2025-08-02T10:29:55Z"
            },
            "database": {
                "status": "healthy",
                "connection_pool": "80%",
                "query_latency_ms": 5
            },
            "cache": {
                "status": "healthy",
                "hit_rate": "94.5%",
                "memory_usage": "67%"
            }
        },
        "performance_metrics": {
            "requests_per_second": 145,
            "average_response_time_ms": 12,
            "error_rate": 0.002
        }
    }
}
```

### 9.2 获取系统配置

**请求**:
```http
GET /system/config
Authorization: Bearer {access_token}
```

**响应**:
```json
{
    "success": true,
    "data": {
        "api_version": "1.0",
        "supported_features": [
            "real_time_monitoring",
            "attribution_analysis", 
            "dynamic_stop_loss",
            "adaptive_risk_budget",
            "stress_testing"
        ],
        "rate_limits": {
            "requests_per_minute": 1000,
            "burst_limit": 100
        },
        "data_retention": {
            "real_time_data": "7天",
            "historical_data": "5年",
            "logs": "1年"
        }
    }
}
```

## API 使用示例

### JavaScript 示例

```javascript
// 获取访问令牌
const getAccessToken = async () => {
    const response = await fetch('https://api.drawdown-control.com/v1/auth/token', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username: 'your_username',
            password: 'your_password',
            grant_type: 'password'
        })
    });
    
    const data = await response.json();
    return data.data.access_token;
};

// 获取回撤监控数据
const getDrawdownData = async (portfolioId, token) => {
    const response = await fetch(`https://api.drawdown-control.com/v1/drawdown/monitor/${portfolioId}`, {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    });
    
    return await response.json();
};

// 使用示例
const main = async () => {
    try {
        const token = await getAccessToken();
        const drawdownData = await getDrawdownData('portfolio_001', token);
        console.log('当前回撤水平:', drawdownData.data.current_drawdown);
    } catch (error) {
        console.error('API调用失败:', error);
    }
};
```

### Python 示例

```python
import requests
import json

class DrawdownControlAPI:
    def __init__(self, base_url, username, password):
        self.base_url = base_url
        self.access_token = self._get_access_token(username, password)
    
    def _get_access_token(self, username, password):
        url = f"{self.base_url}/auth/token"
        data = {
            "username": username,
            "password": password,
            "grant_type": "password"
        }
        response = requests.post(url, json=data)
        return response.json()["data"]["access_token"]
    
    def get_drawdown_data(self, portfolio_id):
        url = f"{self.base_url}/drawdown/monitor/{portfolio_id}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        response = requests.get(url, headers=headers)
        return response.json()
    
    def update_portfolio_data(self, portfolio_id, portfolio_values, timestamps, positions):
        url = f"{self.base_url}/drawdown/monitor/{portfolio_id}/update"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        data = {
            "portfolio_values": portfolio_values,
            "timestamps": timestamps,
            "positions": positions
        }
        response = requests.post(url, headers=headers, json=data)
        return response.json()

# 使用示例
api = DrawdownControlAPI(
    "https://api.drawdown-control.com/v1",
    "your_username", 
    "your_password"
)

# 获取回撤数据
drawdown_data = api.get_drawdown_data("portfolio_001")
print(f"当前回撤: {drawdown_data['data']['current_drawdown']:.2%}")
```

## 错误代码参考

| 错误代码 | 说明 | 解决方案 |
|---------|------|----------|
| 400 | 请求参数错误 | 检查请求参数格式和内容 |
| 401 | 认证失败 | 检查访问令牌是否有效 |
| 403 | 权限不足 | 确认用户具有相应权限 |
| 404 | 资源不存在 | 检查请求URL和资源ID |
| 429 | 请求频率过高 | 降低请求频率或联系管理员 |
| 500 | 服务器内部错误 | 联系技术支持 |
| 503 | 服务不可用 | 稍后重试或检查系统状态 |

---

*更多API详情请参考在线API文档和SDK示例。*