# 数据接口 API

## 概述

数据接口提供股票市场数据获取、特征工程和数据质量管理功能。

## 股票数据接口

### 获取股票列表

获取支持的股票代码列表。

**请求**
```http
GET /api/v1/data/stocks
```

**参数**
- `market` (string, optional): 市场类型，默认为"A"
  - "A": A股市场
  - "HK": 港股市场
  - "US": 美股市场

**响应**
```json
{
  "success": true,
  "data": {
    "stocks": [
      {
        "symbol": "000001.SZ",
        "name": "平安银行",
        "market": "A",
        "sector": "金融",
        "industry": "银行"
      }
    ],
    "total": 4500
  }
}
```

### 获取历史价格数据

获取股票的历史价格数据。

**请求**
```http
GET /api/v1/data/prices
```

**参数**
- `symbols` (array): 股票代码列表，必填
- `start_date` (string): 开始日期，格式YYYY-MM-DD，必填
- `end_date` (string): 结束日期，格式YYYY-MM-DD，必填
- `frequency` (string, optional): 数据频率，默认为"1d"
  - "1d": 日频
  - "1h": 小时频
  - "5m": 5分钟频

**示例请求**
```bash
curl -X GET "http://localhost:8000/api/v1/data/prices?symbols=000001.SZ,000002.SZ&start_date=2024-01-01&end_date=2024-01-31" \
  -H "Authorization: Bearer your_api_key"
```

**响应**
```json
{
  "success": true,
  "data": {
    "prices": [
      {
        "symbol": "000001.SZ",
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 10.50,
        "high": 10.80,
        "low": 10.30,
        "close": 10.65,
        "volume": 1000000,
        "amount": 10650000.0
      }
    ]
  }
}
```

### 获取实时行情

获取股票的实时行情数据。

**请求**
```http
GET /api/v1/data/realtime
```

**参数**
- `symbols` (array): 股票代码列表，必填

**响应**
```json
{
  "success": true,
  "data": {
    "quotes": [
      {
        "symbol": "000001.SZ",
        "timestamp": "2024-01-01T09:30:00Z",
        "price": 10.65,
        "change": 0.15,
        "change_pct": 1.43,
        "volume": 50000,
        "bid": 10.64,
        "ask": 10.66,
        "bid_size": 1000,
        "ask_size": 2000
      }
    ]
  }
}
```

## 特征工程接口

### 计算技术指标

计算股票的技术指标。

**请求**
```http
POST /api/v1/data/features/technical
```

**请求体**
```json
{
  "symbols": ["000001.SZ", "000002.SZ"],
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "indicators": [
    {
      "name": "RSI",
      "params": {"period": 14}
    },
    {
      "name": "MACD",
      "params": {"fast": 12, "slow": 26, "signal": 9}
    },
    {
      "name": "BOLLINGER_BANDS",
      "params": {"period": 20, "std": 2}
    }
  ]
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "features": [
      {
        "symbol": "000001.SZ",
        "timestamp": "2024-01-01T00:00:00Z",
        "indicators": {
          "RSI": 65.5,
          "MACD": 0.12,
          "MACD_signal": 0.08,
          "MACD_histogram": 0.04,
          "BB_upper": 11.20,
          "BB_middle": 10.65,
          "BB_lower": 10.10
        }
      }
    ]
  }
}
```

### 计算基本面因子

计算股票的基本面因子。

**请求**
```http
POST /api/v1/data/features/fundamental
```

**请求体**
```json
{
  "symbols": ["000001.SZ", "000002.SZ"],
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "factors": [
    "PE_ratio",
    "PB_ratio",
    "ROE",
    "ROA",
    "debt_to_equity",
    "current_ratio"
  ]
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "factors": [
      {
        "symbol": "000001.SZ",
        "timestamp": "2024-01-01T00:00:00Z",
        "factors": {
          "PE_ratio": 8.5,
          "PB_ratio": 0.9,
          "ROE": 12.5,
          "ROA": 1.2,
          "debt_to_equity": 0.8,
          "current_ratio": 1.5
        }
      }
    ]
  }
}
```

### 特征标准化

对特征数据进行标准化处理。

**请求**
```http
POST /api/v1/data/features/normalize
```

**请求体**
```json
{
  "features": [
    {
      "symbol": "000001.SZ",
      "timestamp": "2024-01-01T00:00:00Z",
      "values": {
        "RSI": 65.5,
        "MACD": 0.12,
        "PE_ratio": 8.5
      }
    }
  ],
  "method": "z_score",
  "lookback_window": 252
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "normalized_features": [
      {
        "symbol": "000001.SZ",
        "timestamp": "2024-01-01T00:00:00Z",
        "values": {
          "RSI": 0.25,
          "MACD": 1.2,
          "PE_ratio": -0.8
        }
      }
    ]
  }
}
```

## 数据质量接口

### 数据质量检查

检查数据质量并返回质量报告。

**请求**
```http
POST /api/v1/data/quality/check
```

**请求体**
```json
{
  "symbols": ["000001.SZ", "000002.SZ"],
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "checks": [
    "missing_values",
    "outliers",
    "data_consistency",
    "price_anomalies"
  ]
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "quality_report": {
      "overall_score": 0.95,
      "checks": {
        "missing_values": {
          "score": 0.98,
          "issues": [
            {
              "symbol": "000001.SZ",
              "date": "2024-01-15",
              "field": "volume",
              "issue": "missing_value"
            }
          ]
        },
        "outliers": {
          "score": 0.92,
          "issues": [
            {
              "symbol": "000002.SZ",
              "date": "2024-01-20",
              "field": "price",
              "value": 50.0,
              "expected_range": [8.0, 12.0]
            }
          ]
        }
      }
    }
  }
}
```

### 数据修复

修复检测到的数据质量问题。

**请求**
```http
POST /api/v1/data/quality/fix
```

**请求体**
```json
{
  "issues": [
    {
      "symbol": "000001.SZ",
      "date": "2024-01-15",
      "field": "volume",
      "issue": "missing_value",
      "fix_method": "interpolation"
    }
  ]
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "fixed_issues": [
      {
        "symbol": "000001.SZ",
        "date": "2024-01-15",
        "field": "volume",
        "original_value": null,
        "fixed_value": 850000,
        "method": "interpolation"
      }
    ]
  }
}
```

## 数据缓存接口

### 缓存状态查询

查询数据缓存的状态。

**请求**
```http
GET /api/v1/data/cache/status
```

**响应**
```json
{
  "success": true,
  "data": {
    "cache_status": {
      "total_size": "2.5GB",
      "used_size": "1.8GB",
      "hit_rate": 0.85,
      "entries": 15000,
      "last_cleanup": "2024-01-01T06:00:00Z"
    }
  }
}
```

### 清理缓存

清理过期的缓存数据。

**请求**
```http
POST /api/v1/data/cache/cleanup
```

**请求体**
```json
{
  "max_age_hours": 24,
  "force": false
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "cleanup_result": {
      "removed_entries": 500,
      "freed_space": "200MB",
      "remaining_entries": 14500
    }
  }
}
```

## 错误码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| DATA_SOURCE_UNAVAILABLE | 数据源不可用 | 检查数据源连接状态 |
| INVALID_SYMBOL | 无效的股票代码 | 使用正确的股票代码格式 |
| DATE_RANGE_INVALID | 日期范围无效 | 确保开始日期早于结束日期 |
| FEATURE_CALCULATION_FAILED | 特征计算失败 | 检查输入数据和参数 |
| DATA_QUALITY_CHECK_FAILED | 数据质量检查失败 | 查看详细的质量报告 |

## 使用示例

### Python客户端示例

```python
import requests
import json

class TradingDataAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def get_stock_prices(self, symbols, start_date, end_date):
        """获取股票价格数据"""
        params = {
            'symbols': ','.join(symbols),
            'start_date': start_date,
            'end_date': end_date
        }
        response = requests.get(
            f'{self.base_url}/api/v1/data/prices',
            params=params,
            headers=self.headers
        )
        return response.json()
    
    def calculate_technical_indicators(self, symbols, start_date, end_date, indicators):
        """计算技术指标"""
        data = {
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'indicators': indicators
        }
        response = requests.post(
            f'{self.base_url}/api/v1/data/features/technical',
            json=data,
            headers=self.headers
        )
        return response.json()

# 使用示例
api = TradingDataAPI('http://localhost:8000', 'your_api_key')

# 获取价格数据
prices = api.get_stock_prices(
    symbols=['000001.SZ', '000002.SZ'],
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# 计算技术指标
indicators = api.calculate_technical_indicators(
    symbols=['000001.SZ', '000002.SZ'],
    start_date='2024-01-01',
    end_date='2024-01-31',
    indicators=[
        {'name': 'RSI', 'params': {'period': 14}},
        {'name': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}
    ]
)
```

### JavaScript客户端示例

```javascript
class TradingDataAPI {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async getStockPrices(symbols, startDate, endDate) {
        const params = new URLSearchParams({
            symbols: symbols.join(','),
            start_date: startDate,
            end_date: endDate
        });
        
        const response = await fetch(
            `${this.baseUrl}/api/v1/data/prices?${params}`,
            { headers: this.headers }
        );
        
        return await response.json();
    }
    
    async calculateTechnicalIndicators(symbols, startDate, endDate, indicators) {
        const response = await fetch(
            `${this.baseUrl}/api/v1/data/features/technical`,
            {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify({
                    symbols,
                    start_date: startDate,
                    end_date: endDate,
                    indicators
                })
            }
        );
        
        return await response.json();
    }
}

// 使用示例
const api = new TradingDataAPI('http://localhost:8000', 'your_api_key');

// 获取价格数据
const prices = await api.getStockPrices(
    ['000001.SZ', '000002.SZ'],
    '2024-01-01',
    '2024-01-31'
);

// 计算技术指标
const indicators = await api.calculateTechnicalIndicators(
    ['000001.SZ', '000002.SZ'],
    '2024-01-01',
    '2024-01-31',
    [
        { name: 'RSI', params: { period: 14 } },
        { name: 'MACD', params: { fast: 12, slow: 26, signal: 9 } }
    ]
);
```