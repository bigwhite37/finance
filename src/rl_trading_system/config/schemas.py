"""
配置验证模式定义

定义各种配置文件的验证模式，包括类型检查、范围约束和默认值
需求: 10.1
"""

from typing import Dict, Any


def validate_stock_code(field: str, value: str, error_callback) -> None:
    """验证股票代码格式"""
    if not isinstance(value, str):
        error_callback(field, "股票代码必须是字符串")
        return
    
    if len(value) != 9:
        error_callback(field, "股票代码长度必须是9位")
        return
    
    if not (value.endswith('.SZ') or value.endswith('.SH')):
        error_callback(field, "股票代码必须以.SZ或.SH结尾")


def validate_positive_float(field: str, value: float, error_callback) -> None:
    """验证正浮点数"""
    if not isinstance(value, (int, float)):
        error_callback(field, "必须是数值类型")
        return
    
    if value <= 0:
        error_callback(field, "必须是正数")


def validate_probability(field: str, value: float, error_callback) -> None:
    """验证概率值（0-1之间）"""
    if not isinstance(value, (int, float)):
        error_callback(field, "必须是数值类型")
        return
    
    if not (0 <= value <= 1):
        error_callback(field, "概率值必须在0-1之间")


# 模型配置验证模式
MODEL_CONFIG_SCHEMA: Dict[str, Any] = {
    'model': {
        'required': True,
        'type': dict,
        'schema': {
            'transformer': {
                'required': True,
                'type': dict,
                'schema': {
                    'd_model': {
                        'required': True,
                        'type': int,
                        'min': 64,
                        'max': 2048,
                        'default': 256
                    },
                    'n_heads': {
                        'required': True,
                        'type': int,
                        'min': 1,
                        'max': 32,
                        'default': 8
                    },
                    'n_layers': {
                        'required': True,
                        'type': int,
                        'min': 1,
                        'max': 24,
                        'default': 6
                    },
                    'd_ff': {
                        'required': True,
                        'type': int,
                        'min': 128,
                        'max': 8192,
                        'default': 1024
                    },
                    'dropout': {
                        'required': True,
                        'type': float,
                        'min': 0.0,
                        'max': 0.9,
                        'default': 0.1,
                        'validator': validate_probability
                    },
                    'max_seq_len': {
                        'required': True,
                        'type': int,
                        'min': 10,
                        'max': 1000,
                        'default': 252
                    },
                    'n_features': {
                        'required': True,
                        'type': int,
                        'min': 1,
                        'max': 200,
                        'default': 50
                    }
                }
            },
            'sac': {
                'required': True,
                'type': dict,
                'schema': {
                    'state_dim': {
                        'required': True,
                        'type': int,
                        'min': 64,
                        'max': 2048,
                        'default': 256
                    },
                    'action_dim': {
                        'required': True,
                        'type': int,
                        'min': 1,
                        'max': 1000,
                        'default': 100
                    },
                    'hidden_dim': {
                        'required': True,
                        'type': int,
                        'min': 128,
                        'max': 2048,
                        'default': 512
                    },
                    'lr_actor': {
                        'required': True,
                        'type': float,
                        'min': 1e-6,
                        'max': 1e-1,
                        'default': 3e-4,
                        'validator': validate_positive_float
                    },
                    'lr_critic': {
                        'required': True,
                        'type': float,
                        'min': 1e-6,
                        'max': 1e-1,
                        'default': 3e-4,
                        'validator': validate_positive_float
                    },
                    'lr_alpha': {
                        'required': True,
                        'type': float,
                        'min': 1e-6,
                        'max': 1e-1,
                        'default': 3e-4,
                        'validator': validate_positive_float
                    },
                    'gamma': {
                        'required': True,
                        'type': float,
                        'min': 0.9,
                        'max': 0.999,
                        'default': 0.99,
                        'validator': validate_probability
                    },
                    'tau': {
                        'required': True,
                        'type': float,
                        'min': 0.001,
                        'max': 0.1,
                        'default': 0.005,
                        'validator': validate_positive_float
                    },
                    'alpha': {
                        'required': True,
                        'type': float,
                        'min': 0.01,
                        'max': 1.0,
                        'default': 0.2,
                        'validator': validate_positive_float
                    },
                    'target_entropy': {
                        'required': True,
                        'type': float,
                        'max': 0,
                        'default': -100
                    },
                    'buffer_size': {
                        'required': True,
                        'type': int,
                        'min': 10000,
                        'max': 10000000,
                        'default': 1000000
                    },
                    'batch_size': {
                        'required': True,
                        'type': int,
                        'min': 16,
                        'max': 1024,
                        'default': 256
                    }
                }
            }
        }
    },
    'training': {
        'required': True,
        'type': dict,
        'schema': {
            'n_episodes': {
                'required': True,
                'type': int,
                'min': 100,
                'max': 100000,
                'default': 10000
            },
            'eval_freq': {
                'required': True,
                'type': int,
                'min': 10,
                'max': 1000,
                'default': 100
            },
            'patience': {
                'required': True,
                'type': int,
                'min': 10,
                'max': 500,
                'default': 50
            },
            'min_delta': {
                'required': True,
                'type': float,
                'min': 1e-6,
                'max': 0.1,
                'default': 0.001,
                'validator': validate_positive_float
            },
            'checkpoint_dir': {
                'required': True,
                'type': str,
                'default': "./checkpoints"
            }
        }
    }
}


# 交易配置验证模式
TRADING_CONFIG_SCHEMA: Dict[str, Any] = {
    'trading': {
        'required': True,
        'type': dict,
        'schema': {
            'environment': {
                'required': True,
                'type': dict,
                'schema': {
                    'stock_pool': {
                        'required': True,
                        'type': list,
                        'default': []
                    },
                    'lookback_window': {
                        'required': True,
                        'type': int,
                        'min': 5,
                        'max': 500,
                        'default': 60
                    },
                    'initial_cash': {
                        'required': True,
                        'type': float,
                        'min': 10000,
                        'max': 100000000,
                        'default': 1000000.0,
                        'validator': validate_positive_float
                    },
                    'commission_rate': {
                        'required': True,
                        'type': float,
                        'min': 0.0,
                        'max': 0.01,
                        'default': 0.001,
                        'validator': validate_probability
                    },
                    'stamp_tax_rate': {
                        'required': True,
                        'type': float,
                        'min': 0.0,
                        'max': 0.01,
                        'default': 0.001,
                        'validator': validate_probability
                    },
                    'risk_aversion': {
                        'required': True,
                        'type': float,
                        'min': 0.0,
                        'max': 10.0,
                        'default': 0.1,
                        'validator': validate_positive_float
                    },
                    'max_drawdown_penalty': {
                        'required': True,
                        'type': float,
                        'min': 0.0,
                        'max': 10.0,
                        'default': 1.0,
                        'validator': validate_positive_float
                    }
                }
            },
            'cost_model': {
                'required': True,
                'type': dict,
                'schema': {
                    'almgren_chriss': {
                        'required': True,
                        'type': dict,
                        'schema': {
                            'permanent_impact': {
                                'required': True,
                                'type': float,
                                'min': 0.0,
                                'max': 1.0,
                                'default': 0.1,
                                'validator': validate_positive_float
                            },
                            'temporary_impact': {
                                'required': True,
                                'type': float,
                                'min': 0.0,
                                'max': 1.0,
                                'default': 0.05,
                                'validator': validate_positive_float
                            }
                        }
                    }
                }
            },
            'data': {
                'required': True,
                'type': dict,
                'schema': {
                    'provider': {
                        'required': True,
                        'type': str,
                        'allowed': ['qlib', 'akshare'],
                        'default': 'qlib'
                    },
                    'cache_enabled': {
                        'required': True,
                        'type': bool,
                        'default': True
                    },
                    'cache_dir': {
                        'required': True,
                        'type': str,
                        'default': './data_cache'
                    }
                }
            }
        }
    },
    'backtest': {
        'required': True,
        'type': dict,
        'schema': {
            'freq': {
                'required': True,
                'type': str,
                'allowed': ['1d', '1h', '1min'],
                'default': '1d'
            },
            'rebalance_freq': {
                'required': True,
                'type': str,
                'allowed': ['1d', '1h', '1min'],
                'default': '1d'
            },
            'start_date': {
                'required': True,
                'type': str,
                'default': '2020-01-01'
            },
            'end_date': {
                'required': True,
                'type': str,
                'default': '2023-12-31'
            },
            'benchmark': {
                'required': True,
                'type': str,
                'default': '000300.SH',
                'validator': validate_stock_code
            }
        }
    }
}


# 合规配置验证模式
COMPLIANCE_CONFIG_SCHEMA: Dict[str, Any] = {
    'compliance': {
        'required': True,
        'type': dict,
        'schema': {
            'audit': {
                'required': True,
                'type': dict,
                'schema': {
                    'retention_years': {
                        'required': True,
                        'type': int,
                        'min': 1,
                        'max': 20,
                        'default': 5
                    },
                    'database_url': {
                        'required': True,
                        'type': str,
                        'default': 'influxdb://localhost:8086/trading_audit'
                    }
                }
            },
            'risk_control': {
                'required': True,
                'type': dict,
                'schema': {
                    'max_position_concentration': {
                        'required': True,
                        'type': float,
                        'min': 0.01,
                        'max': 1.0,
                        'default': 0.1,
                        'validator': validate_probability
                    },
                    'max_sector_exposure': {
                        'required': True,
                        'type': float,
                        'min': 0.1,
                        'max': 1.0,
                        'default': 0.3,
                        'validator': validate_probability
                    },
                    'stop_loss_threshold': {
                        'required': True,
                        'type': float,
                        'min': -0.5,
                        'max': 0.0,
                        'default': -0.05
                    }
                }
            },
            'explainability': {
                'required': True,
                'type': dict,
                'schema': {
                    'shap_enabled': {
                        'required': True,
                        'type': bool,
                        'default': True
                    },
                    'lime_enabled': {
                        'required': True,
                        'type': bool,
                        'default': True
                    },
                    'attention_visualization': {
                        'required': True,
                        'type': bool,
                        'default': True
                    }
                }
            },
            'reporting': {
                'required': True,
                'type': dict,
                'schema': {
                    'daily_report': {
                        'required': True,
                        'type': bool,
                        'default': True
                    },
                    'weekly_report': {
                        'required': True,
                        'type': bool,
                        'default': True
                    },
                    'monthly_report': {
                        'required': True,
                        'type': bool,
                        'default': True
                    },
                    'report_recipients': {
                        'required': True,
                        'type': list,
                        'default': []
                    }
                }
            }
        }
    }
}


# 监控配置验证模式
MONITORING_CONFIG_SCHEMA: Dict[str, Any] = {
    'monitoring': {
        'required': True,
        'type': dict,
        'schema': {
            'prometheus': {
                'required': True,
                'type': dict,
                'schema': {
                    'port': {
                        'required': True,
                        'type': int,
                        'min': 1024,
                        'max': 65535,
                        'default': 8000
                    },
                    'metrics_path': {
                        'required': True,
                        'type': str,
                        'default': '/metrics'
                    }
                }
            },
            'thresholds': {
                'required': True,
                'type': dict,
                'schema': {
                    'max_drawdown': {
                        'required': True,
                        'type': float,
                        'min': 0.01,
                        'max': 1.0,
                        'default': 0.15,
                        'validator': validate_probability
                    },
                    'min_sharpe_ratio': {
                        'required': True,
                        'type': float,
                        'min': -5.0,
                        'max': 10.0,
                        'default': 0.5
                    },
                    'max_var_95': {
                        'required': True,
                        'type': float,
                        'min': 0.001,
                        'max': 0.5,
                        'default': 0.03,
                        'validator': validate_probability
                    },
                    'lookback_days': {
                        'required': True,
                        'type': int,
                        'min': 7,
                        'max': 365,
                        'default': 90
                    }
                }
            },
            'alerts': {
                'required': True,
                'type': dict,
                'schema': {
                    'channels': {
                        'required': True,
                        'type': list,
                        'default': ['email']
                    },
                    'email': {
                        'required': False,
                        'type': dict,
                        'schema': {
                            'smtp_server': {
                                'required': True,
                                'type': str,
                                'default': 'smtp.gmail.com'
                            },
                            'smtp_port': {
                                'required': True,
                                'type': int,
                                'min': 1,
                                'max': 65535,
                                'default': 587
                            },
                            'username': {
                                'required': True,
                                'type': str,
                                'default': ''
                            },
                            'password': {
                                'required': True,
                                'type': str,
                                'default': ''
                            },
                            'recipients': {
                                'required': True,
                                'type': list,
                                'default': []
                            }
                        }
                    },
                    'dingtalk': {
                        'required': False,
                        'type': dict,
                        'schema': {
                            'webhook_url': {
                                'required': True,
                                'type': str,
                                'default': ''
                            }
                        }
                    }
                }
            }
        }
    },
    'logging': {
        'required': True,
        'type': dict,
        'schema': {
            'level': {
                'required': True,
                'type': str,
                'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                'default': 'INFO'
            },
            'format': {
                'required': True,
                'type': str,
                'default': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}'
            },
            'rotation': {
                'required': True,
                'type': str,
                'default': '1 day'
            },
            'retention': {
                'required': True,
                'type': str,
                'default': '30 days'
            },
            'compression': {
                'required': True,
                'type': str,
                'allowed': ['gz', 'bz2', 'xz'],
                'default': 'gz'
            }
        }
    }
}


# 所有配置模式的映射
CONFIG_SCHEMAS = {
    'model': MODEL_CONFIG_SCHEMA,
    'trading': TRADING_CONFIG_SCHEMA,
    'compliance': COMPLIANCE_CONFIG_SCHEMA,
    'monitoring': MONITORING_CONFIG_SCHEMA
}