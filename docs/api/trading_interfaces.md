# 交易接口 API

## 概述

交易接口提供投资组合管理、交易执行、风险控制和绩效分析功能。

## 投资组合管理接口

### 获取投资组合状态

获取当前投资组合的详细状态。

**请求**
```http
GET /api/v1/trading/portfolio
```

**参数**
- `session_id` (string, optional): 交易会话ID

**响应**
```json
{
  "success": true,
  "data": {
    "portfolio": {
      "session_id": "session_001",
      "timestamp": "2024-01-01T09:30:00Z",
      "total_value": 1050000.0,
      "cash": 50000.0,
      "positions": [
        {
          "symbol": "000001.SZ",
          "quantity": 1000,
          "market_value": 10500.0,
          "weight": 0.01,
          "unrealized_pnl": 500.0,
          "cost_basis": 10.0,
          "current_price": 10.5
        }
      ],
      "performance": {
        "total_return": 0.05,
        "daily_return": 0.002,
        "sharpe_ratio": 1.85,
        "max_drawdown": 0.08,
        "volatility": 0.15
      }
    }
  }
}
```

### 更新投资组合权重

更新目标投资组合权重。

**请求**
```http
POST /api/v1/trading/portfolio/rebalance
```

**请求体**
```json
{
  "session_id": "session_001",
  "target_weights": {
    "000001.SZ": 0.3,
    "000002.SZ": 0.4,
    "000858.SZ": 0.2,
    "cash": 0.1
  },
  "rebalance_options": {
    "execution_style": "gradual",
    "max_trade_size": 0.05,
    "time_horizon_minutes": 30,
    "allow_partial_fills": true
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "rebalance_plan": {
      "plan_id": "plan_001",
      "created_at": "2024-01-01T09:30:00Z",
      "trades": [
        {
          "symbol": "000001.SZ",
          "action": "buy",
          "target_quantity": 500,
          "estimated_cost": 5250.0,
          "priority": 1
        },
        {
          "symbol": "000002.SZ",
          "action": "sell",
          "target_quantity": 200,
          "estimated_proceeds": 4000.0,
          "priority": 2
        }
      ],
      "estimated_total_cost": 125.5,
      "estimated_completion_time": "2024-01-01T10:00:00Z"
    }
  }
}
```

## 交易执行接口

### 提交交易订单

提交单个交易订单。

**请求**
```http
POST /api/v1/trading/orders
```

**请求体**
```json
{
  "session_id": "session_001",
  "symbol": "000001.SZ",
  "side": "buy",
  "quantity": 1000,
  "order_type": "market",
  "price": null,
  "time_in_force": "DAY",
  "execution_options": {
    "max_participation_rate": 0.1,
    "urgency": "medium",
    "allow_partial_fills": true
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "order": {
      "order_id": "order_001",
      "symbol": "000001.SZ",
      "side": "buy",
      "quantity": 1000,
      "order_type": "market",
      "status": "submitted",
      "submitted_at": "2024-01-01T09:30:00Z",
      "estimated_fill_price": 10.5,
      "estimated_total_cost": 10521.0
    }
  }
}
```

### 获取订单状态

查询订单的当前状态。

**请求**
```http
GET /api/v1/trading/orders/{order_id}
```

**响应**
```json
{
  "success": true,
  "data": {
    "order": {
      "order_id": "order_001",
      "symbol": "000001.SZ",
      "side": "buy",
      "quantity": 1000,
      "filled_quantity": 800,
      "remaining_quantity": 200,
      "status": "partially_filled",
      "average_fill_price": 10.52,
      "total_commission": 8.42,
      "fills": [
        {
          "fill_id": "fill_001",
          "quantity": 500,
          "price": 10.50,
          "timestamp": "2024-01-01T09:31:00Z",
          "commission": 5.25
        },
        {
          "fill_id": "fill_002",
          "quantity": 300,
          "price": 10.55,
          "timestamp": "2024-01-01T09:32:00Z",
          "commission": 3.17
        }
      ]
    }
  }
}
```

### 取消订单

取消未完成的订单。

**请求**
```http
DELETE /api/v1/trading/orders/{order_id}
```

**响应**
```json
{
  "success": true,
  "data": {
    "message": "订单已取消",
    "order_id": "order_001",
    "cancelled_at": "2024-01-01T09:35:00Z",
    "cancelled_quantity": 200
  }
}
```

### 获取交易历史

获取历史交易记录。

**请求**
```http
GET /api/v1/trading/trades
```

**参数**
- `session_id` (string, optional): 交易会话ID
- `symbol` (string, optional): 股票代码过滤
- `start_date` (string, optional): 开始日期
- `end_date` (string, optional): 结束日期
- `limit` (int, optional): 返回记录数量限制
- `offset` (int, optional): 偏移量

**响应**
```json
{
  "success": true,
  "data": {
    "trades": [
      {
        "trade_id": "trade_001",
        "order_id": "order_001",
        "symbol": "000001.SZ",
        "side": "buy",
        "quantity": 500,
        "price": 10.50,
        "timestamp": "2024-01-01T09:31:00Z",
        "commission": 5.25,
        "stamp_tax": 0.0,
        "net_amount": 5255.25,
        "execution_venue": "SZSE"
      }
    ],
    "total": 150,
    "summary": {
      "total_volume": 150000,
      "total_commission": 150.5,
      "total_stamp_tax": 75.2,
      "net_pnl": 2500.0
    }
  }
}
```

## 风险控制接口

### 获取风险指标

获取当前的风险指标。

**请求**
```http
GET /api/v1/trading/risk/metrics
```

**参数**
- `session_id` (string, optional): 交易会话ID

**响应**
```json
{
  "success": true,
  "data": {
    "risk_metrics": {
      "timestamp": "2024-01-01T09:30:00Z",
      "portfolio_var": {
        "1_day_95": 15000.0,
        "1_day_99": 25000.0,
        "10_day_95": 47000.0
      },
      "concentration_risk": {
        "max_single_position": 0.15,
        "top_5_concentration": 0.65,
        "herfindahl_index": 0.08
      },
      "sector_exposure": {
        "金融": 0.3,
        "科技": 0.25,
        "消费": 0.2,
        "工业": 0.15,
        "其他": 0.1
      },
      "liquidity_risk": {
        "avg_daily_volume_ratio": 0.05,
        "illiquid_positions_ratio": 0.02
      },
      "leverage": {
        "gross_leverage": 1.0,
        "net_leverage": 0.95
      }
    }
  }
}
```

### 风险检查

对交易计划进行风险检查。

**请求**
```http
POST /api/v1/trading/risk/check
```

**请求体**
```json
{
  "session_id": "session_001",
  "proposed_trades": [
    {
      "symbol": "000001.SZ",
      "side": "buy",
      "quantity": 2000,
      "price": 10.5
    }
  ],
  "check_types": [
    "position_limits",
    "concentration_limits",
    "sector_limits",
    "liquidity_limits",
    "var_limits"
  ]
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "risk_check_result": {
      "overall_status": "warning",
      "checks": {
        "position_limits": {
          "status": "pass",
          "details": "所有持仓在限制范围内"
        },
        "concentration_limits": {
          "status": "warning",
          "details": "单一持仓权重将达到18%，接近20%限制",
          "affected_positions": ["000001.SZ"]
        },
        "sector_limits": {
          "status": "pass",
          "details": "行业暴露在限制范围内"
        },
        "var_limits": {
          "status": "fail",
          "details": "预计VaR将超过限制25%",
          "current_var": 22000.0,
          "projected_var": 31000.0,
          "limit": 25000.0
        }
      },
      "recommendations": [
        "考虑减少000001.SZ的买入数量",
        "增加其他股票的配置以分散风险"
      ]
    }
  }
}
```

### 设置风险限制

设置或更新风险控制参数。

**请求**
```http
POST /api/v1/trading/risk/limits
```

**请求体**
```json
{
  "session_id": "session_001",
  "limits": {
    "max_single_position_weight": 0.2,
    "max_sector_weight": 0.4,
    "max_daily_var": 25000.0,
    "max_drawdown": 0.15,
    "min_liquidity_ratio": 0.1,
    "max_leverage": 1.5
  },
  "effective_date": "2024-01-01T00:00:00Z"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "message": "风险限制已更新",
    "limits_id": "limits_001",
    "effective_date": "2024-01-01T00:00:00Z",
    "updated_limits": {
      "max_single_position_weight": 0.2,
      "max_sector_weight": 0.4,
      "max_daily_var": 25000.0
    }
  }
}
```

## 绩效分析接口

### 获取绩效报告

获取投资组合的绩效分析报告。

**请求**
```http
GET /api/v1/trading/performance
```

**参数**
- `session_id` (string, optional): 交易会话ID
- `start_date` (string, optional): 开始日期
- `end_date` (string, optional): 结束日期
- `benchmark` (string, optional): 基准指数，默认"000300.SH"

**响应**
```json
{
  "success": true,
  "data": {
    "performance_report": {
      "period": {
        "start_date": "2024-01-01",
        "end_date": "2024-03-31"
      },
      "returns": {
        "total_return": 0.15,
        "annualized_return": 0.62,
        "benchmark_return": 0.08,
        "excess_return": 0.07,
        "daily_returns": [0.001, 0.002, -0.001]
      },
      "risk_metrics": {
        "volatility": 0.18,
        "max_drawdown": 0.12,
        "var_95": 0.025,
        "cvar_95": 0.035,
        "downside_deviation": 0.12
      },
      "risk_adjusted_metrics": {
        "sharpe_ratio": 1.85,
        "sortino_ratio": 2.15,
        "calmar_ratio": 1.25,
        "information_ratio": 0.95,
        "treynor_ratio": 0.08
      },
      "trading_metrics": {
        "turnover_rate": 2.5,
        "avg_holding_period": 15,
        "win_rate": 0.58,
        "profit_factor": 1.35,
        "total_trades": 120,
        "total_commission": 1250.0
      }
    }
  }
}
```

### 获取归因分析

获取绩效归因分析。

**请求**
```http
GET /api/v1/trading/performance/attribution
```

**参数**
- `session_id` (string, optional): 交易会话ID
- `start_date` (string): 开始日期，必填
- `end_date` (string): 结束日期，必填
- `attribution_model` (string, optional): 归因模型，默认"brinson"

**响应**
```json
{
  "success": true,
  "data": {
    "attribution_analysis": {
      "model": "brinson",
      "period": {
        "start_date": "2024-01-01",
        "end_date": "2024-03-31"
      },
      "total_excess_return": 0.07,
      "attribution_breakdown": {
        "asset_allocation": 0.02,
        "security_selection": 0.04,
        "interaction": 0.01
      },
      "sector_attribution": [
        {
          "sector": "金融",
          "portfolio_weight": 0.3,
          "benchmark_weight": 0.35,
          "portfolio_return": 0.12,
          "benchmark_return": 0.08,
          "allocation_effect": -0.002,
          "selection_effect": 0.012,
          "total_effect": 0.01
        }
      ],
      "top_contributors": [
        {
          "symbol": "000001.SZ",
          "contribution": 0.015,
          "weight": 0.05,
          "return": 0.25
        }
      ],
      "top_detractors": [
        {
          "symbol": "000002.SZ",
          "contribution": -0.008,
          "weight": 0.03,
          "return": -0.15
        }
      ]
    }
  }
}
```

## 交易会话管理接口

### 创建交易会话

创建新的交易会话。

**请求**
```http
POST /api/v1/trading/sessions
```

**请求体**
```json
{
  "session_name": "SAC_Agent_v1.0_Session",
  "model_id": "model_001",
  "initial_capital": 1000000.0,
  "trading_universe": ["000001.SZ", "000002.SZ", "000858.SZ"],
  "session_config": {
    "max_position_weight": 0.2,
    "rebalance_frequency": "daily",
    "risk_budget": 0.15,
    "transaction_cost_model": "almgren_chriss"
  },
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "session": {
      "session_id": "session_001",
      "session_name": "SAC_Agent_v1.0_Session",
      "status": "created",
      "created_at": "2024-01-01T00:00:00Z",
      "model_id": "model_001",
      "initial_capital": 1000000.0,
      "current_value": 1000000.0,
      "trading_universe": ["000001.SZ", "000002.SZ", "000858.SZ"]
    }
  }
}
```

### 启动交易会话

启动交易会话开始自动交易。

**请求**
```http
POST /api/v1/trading/sessions/{session_id}/start
```

**响应**
```json
{
  "success": true,
  "data": {
    "message": "交易会话已启动",
    "session_id": "session_001",
    "started_at": "2024-01-01T09:30:00Z",
    "status": "running"
  }
}
```

### 停止交易会话

停止交易会话。

**请求**
```http
POST /api/v1/trading/sessions/{session_id}/stop
```

**请求体**
```json
{
  "reason": "用户手动停止",
  "liquidate_positions": false
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "message": "交易会话已停止",
    "session_id": "session_001",
    "stopped_at": "2024-01-01T15:00:00Z",
    "final_value": 1050000.0,
    "total_return": 0.05
  }
}
```

## 错误码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| INSUFFICIENT_FUNDS | 资金不足 | 检查账户余额 |
| POSITION_LIMIT_EXCEEDED | 持仓限制超出 | 调整交易数量 |
| MARKET_CLOSED | 市场未开放 | 等待市场开放时间 |
| INVALID_ORDER_TYPE | 无效订单类型 | 使用支持的订单类型 |
| RISK_LIMIT_VIOLATED | 风险限制违反 | 调整交易计划 |
| SESSION_NOT_FOUND | 交易会话不存在 | 检查会话ID |

## 使用示例

### Python客户端示例

```python
import requests
import json
import time

class TradingAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def create_session(self, session_name, model_id, initial_capital, trading_universe):
        """创建交易会话"""
        data = {
            'session_name': session_name,
            'model_id': model_id,
            'initial_capital': initial_capital,
            'trading_universe': trading_universe,
            'session_config': {
                'max_position_weight': 0.2,
                'rebalance_frequency': 'daily',
                'risk_budget': 0.15
            }
        }
        
        response = requests.post(
            f'{self.base_url}/api/v1/trading/sessions',
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def rebalance_portfolio(self, session_id, target_weights):
        """重新平衡投资组合"""
        data = {
            'session_id': session_id,
            'target_weights': target_weights,
            'rebalance_options': {
                'execution_style': 'gradual',
                'max_trade_size': 0.05
            }
        }
        
        response = requests.post(
            f'{self.base_url}/api/v1/trading/portfolio/rebalance',
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def get_portfolio_status(self, session_id):
        """获取投资组合状态"""
        response = requests.get(
            f'{self.base_url}/api/v1/trading/portfolio',
            params={'session_id': session_id},
            headers=self.headers
        )
        return response.json()
    
    def check_risk(self, session_id, proposed_trades):
        """风险检查"""
        data = {
            'session_id': session_id,
            'proposed_trades': proposed_trades,
            'check_types': [
                'position_limits',
                'concentration_limits',
                'var_limits'
            ]
        }
        
        response = requests.post(
            f'{self.base_url}/api/v1/trading/risk/check',
            json=data,
            headers=self.headers
        )
        return response.json()

# 使用示例
api = TradingAPI('http://localhost:8000', 'your_api_key')

# 创建交易会话
session = api.create_session(
    session_name='Demo_Session',
    model_id='model_001',
    initial_capital=1000000.0,
    trading_universe=['000001.SZ', '000002.SZ', '000858.SZ']
)

session_id = session['data']['session']['session_id']
print(f"交易会话已创建: {session_id}")

# 重新平衡投资组合
target_weights = {
    '000001.SZ': 0.4,
    '000002.SZ': 0.3,
    '000858.SZ': 0.2,
    'cash': 0.1
}

rebalance_result = api.rebalance_portfolio(session_id, target_weights)
print(f"重新平衡计划: {rebalance_result['data']['rebalance_plan']['plan_id']}")

# 检查投资组合状态
portfolio = api.get_portfolio_status(session_id)
print(f"投资组合总价值: {portfolio['data']['portfolio']['total_value']}")

# 风险检查
proposed_trades = [
    {
        'symbol': '000001.SZ',
        'side': 'buy',
        'quantity': 1000,
        'price': 10.5
    }
]

risk_check = api.check_risk(session_id, proposed_trades)
print(f"风险检查状态: {risk_check['data']['risk_check_result']['overall_status']}")
```