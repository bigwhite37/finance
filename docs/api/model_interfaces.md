# 模型接口 API

## 概述

模型接口提供机器学习模型的训练、推理、管理和评估功能，包括Transformer编码器和SAC智能体。

## 模型管理接口

### 获取模型列表

获取系统中所有可用的模型。

**请求**
```http
GET /api/v1/models
```

**参数**
- `status` (string, optional): 模型状态过滤
  - "active": 活跃模型
  - "training": 训练中
  - "archived": 已归档
- `model_type` (string, optional): 模型类型过滤
  - "transformer": Transformer模型
  - "sac_agent": SAC智能体

**响应**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "model_id": "model_001",
        "name": "SAC_Trading_Agent_v1.0",
        "version": "1.0.0",
        "type": "sac_agent",
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-15T12:00:00Z",
        "performance": {
          "sharpe_ratio": 1.85,
          "max_drawdown": 0.12,
          "annual_return": 0.15
        },
        "metadata": {
          "training_samples": 100000,
          "validation_score": 0.92,
          "model_size": "45MB"
        }
      }
    ],
    "total": 5
  }
}
```

### 获取模型详情

获取特定模型的详细信息。

**请求**
```http
GET /api/v1/models/{model_id}
```

**响应**
```json
{
  "success": true,
  "data": {
    "model": {
      "model_id": "model_001",
      "name": "SAC_Trading_Agent_v1.0",
      "version": "1.0.0",
      "type": "sac_agent",
      "status": "active",
      "config": {
        "state_dim": 256,
        "action_dim": 100,
        "hidden_dim": 512,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "tau": 0.005
      },
      "architecture": {
        "encoder": {
          "type": "transformer",
          "layers": 6,
          "heads": 8,
          "d_model": 256
        },
        "actor": {
          "hidden_layers": [512, 256, 128],
          "activation": "relu"
        },
        "critic": {
          "hidden_layers": [512, 256, 128],
          "activation": "relu"
        }
      },
      "training_history": [
        {
          "epoch": 100,
          "loss": 0.025,
          "reward": 0.15,
          "timestamp": "2024-01-15T12:00:00Z"
        }
      ]
    }
  }
}
```

## 模型训练接口

### 开始训练

启动模型训练任务。

**请求**
```http
POST /api/v1/models/train
```

**请求体**
```json
{
  "model_config": {
    "name": "SAC_Trading_Agent_v1.1",
    "type": "sac_agent",
    "config": {
      "state_dim": 256,
      "action_dim": 100,
      "hidden_dim": 512,
      "learning_rate": 0.0003,
      "gamma": 0.99,
      "tau": 0.005,
      "batch_size": 256,
      "buffer_size": 1000000
    }
  },
  "training_config": {
    "max_episodes": 1000,
    "max_steps_per_episode": 252,
    "evaluation_frequency": 50,
    "save_frequency": 100,
    "early_stopping": {
      "patience": 100,
      "min_delta": 0.001
    }
  },
  "data_config": {
    "symbols": ["000001.SZ", "000002.SZ"],
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "validation_split": 0.2,
    "test_split": 0.1
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "training_job": {
      "job_id": "job_001",
      "model_id": "model_002",
      "status": "started",
      "created_at": "2024-01-01T00:00:00Z",
      "estimated_duration": "2 hours",
      "progress": {
        "current_episode": 0,
        "total_episodes": 1000,
        "current_loss": null,
        "best_reward": null
      }
    }
  }
}
```

### 获取训练状态

查询训练任务的当前状态。

**请求**
```http
GET /api/v1/models/train/{job_id}
```

**响应**
```json
{
  "success": true,
  "data": {
    "training_job": {
      "job_id": "job_001",
      "model_id": "model_002",
      "status": "training",
      "started_at": "2024-01-01T00:00:00Z",
      "progress": {
        "current_episode": 250,
        "total_episodes": 1000,
        "progress_percentage": 25.0,
        "current_loss": 0.045,
        "best_reward": 0.12,
        "estimated_remaining": "1.5 hours"
      },
      "metrics": {
        "episode_rewards": [0.08, 0.09, 0.11, 0.12],
        "losses": [0.08, 0.06, 0.05, 0.045],
        "validation_scores": [0.85, 0.87, 0.89, 0.91]
      }
    }
  }
}
```

### 停止训练

停止正在进行的训练任务。

**请求**
```http
POST /api/v1/models/train/{job_id}/stop
```

**请求体**
```json
{
  "save_checkpoint": true,
  "reason": "用户手动停止"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "message": "训练任务已停止",
    "final_status": "stopped",
    "checkpoint_saved": true,
    "final_metrics": {
      "episodes_completed": 250,
      "best_reward": 0.12,
      "final_loss": 0.045
    }
  }
}
```

## 模型推理接口

### 单次预测

使用模型进行单次预测。

**请求**
```http
POST /api/v1/models/{model_id}/predict
```

**请求体**
```json
{
  "input_data": {
    "features": [
      [
        [0.1, 0.2, 0.3],
        [0.15, 0.25, 0.35],
        [0.12, 0.22, 0.32]
      ]
    ],
    "positions": [0.3, 0.4, 0.3],
    "market_state": [0.5, 0.6, 0.7, 0.8, 0.9]
  },
  "options": {
    "deterministic": true,
    "return_confidence": true,
    "return_attention_weights": false
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "prediction": {
      "target_weights": [0.35, 0.45, 0.20],
      "confidence": 0.85,
      "model_outputs": {
        "q_values": [0.12, 0.15, 0.08],
        "action_log_probs": [-1.2, -0.9, -1.8]
      },
      "metadata": {
        "model_id": "model_001",
        "model_version": "1.0.0",
        "inference_time_ms": 15.5,
        "timestamp": "2024-01-01T00:00:00Z"
      }
    }
  }
}
```

### 批量预测

使用模型进行批量预测。

**请求**
```http
POST /api/v1/models/{model_id}/predict/batch
```

**请求体**
```json
{
  "batch_data": [
    {
      "id": "sample_001",
      "features": [[[0.1, 0.2, 0.3]]],
      "positions": [0.3, 0.4, 0.3],
      "market_state": [0.5, 0.6, 0.7, 0.8, 0.9]
    },
    {
      "id": "sample_002",
      "features": [[[0.2, 0.3, 0.4]]],
      "positions": [0.2, 0.5, 0.3],
      "market_state": [0.4, 0.7, 0.6, 0.9, 0.8]
    }
  ],
  "options": {
    "deterministic": true,
    "return_confidence": true
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "predictions": [
      {
        "id": "sample_001",
        "target_weights": [0.35, 0.45, 0.20],
        "confidence": 0.85
      },
      {
        "id": "sample_002",
        "target_weights": [0.25, 0.55, 0.20],
        "confidence": 0.78
      }
    ],
    "batch_metadata": {
      "total_samples": 2,
      "successful_predictions": 2,
      "failed_predictions": 0,
      "average_inference_time_ms": 12.3
    }
  }
}
```

## 模型评估接口

### 模型评估

对模型进行全面评估。

**请求**
```http
POST /api/v1/models/{model_id}/evaluate
```

**请求体**
```json
{
  "evaluation_config": {
    "test_data": {
      "symbols": ["000001.SZ", "000002.SZ"],
      "start_date": "2024-01-01",
      "end_date": "2024-03-31"
    },
    "metrics": [
      "sharpe_ratio",
      "max_drawdown",
      "annual_return",
      "win_rate",
      "profit_factor"
    ],
    "benchmark": "000300.SH"
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "evaluation_result": {
      "evaluation_id": "eval_001",
      "model_id": "model_001",
      "period": {
        "start_date": "2024-01-01",
        "end_date": "2024-03-31"
      },
      "performance_metrics": {
        "sharpe_ratio": 1.85,
        "max_drawdown": 0.12,
        "annual_return": 0.15,
        "win_rate": 0.58,
        "profit_factor": 1.35,
        "volatility": 0.18,
        "calmar_ratio": 1.25
      },
      "benchmark_comparison": {
        "benchmark": "000300.SH",
        "benchmark_return": 0.08,
        "excess_return": 0.07,
        "information_ratio": 0.95,
        "tracking_error": 0.12
      },
      "detailed_analysis": {
        "monthly_returns": [0.02, 0.01, 0.03],
        "drawdown_periods": [
          {
            "start": "2024-02-01",
            "end": "2024-02-15",
            "max_drawdown": 0.08
          }
        ],
        "sector_exposure": {
          "金融": 0.3,
          "科技": 0.25,
          "消费": 0.2,
          "其他": 0.25
        }
      }
    }
  }
}
```

### 获取评估历史

获取模型的历史评估记录。

**请求**
```http
GET /api/v1/models/{model_id}/evaluations
```

**参数**
- `limit` (int, optional): 返回记录数量限制，默认10
- `offset` (int, optional): 偏移量，默认0

**响应**
```json
{
  "success": true,
  "data": {
    "evaluations": [
      {
        "evaluation_id": "eval_001",
        "created_at": "2024-01-01T00:00:00Z",
        "period": {
          "start_date": "2024-01-01",
          "end_date": "2024-03-31"
        },
        "summary_metrics": {
          "sharpe_ratio": 1.85,
          "max_drawdown": 0.12,
          "annual_return": 0.15
        }
      }
    ],
    "total": 5,
    "has_more": true
  }
}
```

## 模型版本管理接口

### 创建模型版本

创建模型的新版本。

**请求**
```http
POST /api/v1/models/{model_id}/versions
```

**请求体**
```json
{
  "version": "1.1.0",
  "description": "改进了特征工程和风险控制",
  "changes": [
    "添加了新的技术指标",
    "优化了风险控制参数",
    "提升了模型稳定性"
  ],
  "checkpoint_path": "/models/checkpoints/model_001_v1.1.0.pth"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "version": {
      "version_id": "version_002",
      "model_id": "model_001",
      "version": "1.1.0",
      "status": "created",
      "created_at": "2024-01-01T00:00:00Z",
      "description": "改进了特征工程和风险控制",
      "changes": [
        "添加了新的技术指标",
        "优化了风险控制参数",
        "提升了模型稳定性"
      ]
    }
  }
}
```

### 切换模型版本

切换到指定的模型版本。

**请求**
```http
POST /api/v1/models/{model_id}/versions/{version_id}/activate
```

**响应**
```json
{
  "success": true,
  "data": {
    "message": "模型版本已激活",
    "active_version": {
      "version_id": "version_002",
      "version": "1.1.0",
      "activated_at": "2024-01-01T00:00:00Z"
    }
  }
}
```

## 模型解释性接口

### 获取特征重要性

获取模型的特征重要性分析。

**请求**
```http
POST /api/v1/models/{model_id}/explain/feature_importance
```

**请求体**
```json
{
  "input_data": {
    "features": [[[0.1, 0.2, 0.3]]],
    "positions": [0.3, 0.4, 0.3],
    "market_state": [0.5, 0.6, 0.7, 0.8, 0.9]
  },
  "method": "shap",
  "options": {
    "baseline_samples": 100,
    "return_raw_values": false
  }
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "feature_importance": {
      "method": "shap",
      "importance_scores": {
        "RSI": 0.25,
        "MACD": 0.18,
        "PE_ratio": 0.12,
        "volume": 0.08,
        "price_momentum": 0.15
      },
      "visualization_data": {
        "waterfall_data": [
          {"feature": "RSI", "contribution": 0.05},
          {"feature": "MACD", "contribution": 0.03}
        ]
      }
    }
  }
}
```

### 获取注意力权重

获取Transformer模型的注意力权重。

**请求**
```http
POST /api/v1/models/{model_id}/explain/attention
```

**请求体**
```json
{
  "input_data": {
    "features": [[[0.1, 0.2, 0.3]]],
    "positions": [0.3, 0.4, 0.3],
    "market_state": [0.5, 0.6, 0.7, 0.8, 0.9]
  },
  "layer": 5,
  "head": 0
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "attention_weights": {
      "layer": 5,
      "head": 0,
      "weights": [
        [0.1, 0.2, 0.3, 0.4],
        [0.15, 0.25, 0.35, 0.25],
        [0.2, 0.3, 0.2, 0.3]
      ],
      "sequence_length": 60,
      "attention_patterns": {
        "recent_focus": 0.6,
        "long_term_focus": 0.4,
        "volatility_focus": 0.3
      }
    }
  }
}
```

## 错误码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| MODEL_NOT_FOUND | 模型不存在 | 检查模型ID是否正确 |
| MODEL_TRAINING_FAILED | 模型训练失败 | 检查训练数据和参数 |
| INFERENCE_ERROR | 推理错误 | 检查输入数据格式 |
| MODEL_VERSION_CONFLICT | 模型版本冲突 | 使用不同的版本号 |
| INSUFFICIENT_RESOURCES | 资源不足 | 等待资源释放或增加资源 |

## 使用示例

### Python客户端示例

```python
import requests
import json
import numpy as np

class ModelAPI:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def predict(self, model_id, features, positions, market_state):
        """进行模型预测"""
        data = {
            'input_data': {
                'features': features.tolist(),
                'positions': positions.tolist(),
                'market_state': market_state.tolist()
            },
            'options': {
                'deterministic': True,
                'return_confidence': True
            }
        }
        
        response = requests.post(
            f'{self.base_url}/api/v1/models/{model_id}/predict',
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def start_training(self, model_config, training_config, data_config):
        """开始模型训练"""
        data = {
            'model_config': model_config,
            'training_config': training_config,
            'data_config': data_config
        }
        
        response = requests.post(
            f'{self.base_url}/api/v1/models/train',
            json=data,
            headers=self.headers
        )
        return response.json()
    
    def get_training_status(self, job_id):
        """获取训练状态"""
        response = requests.get(
            f'{self.base_url}/api/v1/models/train/{job_id}',
            headers=self.headers
        )
        return response.json()

# 使用示例
api = ModelAPI('http://localhost:8000', 'your_api_key')

# 模型预测
features = np.random.rand(1, 60, 50)  # [batch, sequence, features]
positions = np.array([0.3, 0.4, 0.3])
market_state = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

prediction = api.predict('model_001', features, positions, market_state)
print(f"预测权重: {prediction['data']['prediction']['target_weights']}")

# 开始训练
training_job = api.start_training(
    model_config={
        'name': 'SAC_Agent_v2.0',
        'type': 'sac_agent',
        'config': {
            'state_dim': 256,
            'action_dim': 100,
            'learning_rate': 0.0003
        }
    },
    training_config={
        'max_episodes': 1000,
        'evaluation_frequency': 50
    },
    data_config={
        'symbols': ['000001.SZ', '000002.SZ'],
        'start_date': '2020-01-01',
        'end_date': '2023-12-31'
    }
)

job_id = training_job['data']['training_job']['job_id']
print(f"训练任务ID: {job_id}")

# 监控训练进度
import time
while True:
    status = api.get_training_status(job_id)
    progress = status['data']['training_job']['progress']
    print(f"训练进度: {progress['progress_percentage']:.1f}%")
    
    if status['data']['training_job']['status'] in ['completed', 'failed', 'stopped']:
        break
    
    time.sleep(30)  # 每30秒检查一次
```