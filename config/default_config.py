"""
默认配置参数
"""

def get_default_config() -> dict:
    """获取默认配置"""
    
    config = {
        # 数据配置
        'data': {
            'provider': 'yahoo',
            'region': 'cn',
            'universe': 'csi300',
            'provider_uri': '~/.qlib/qlib_data/cn_data',
            'start_date': '2019-01-01',  # 提前开始时间以获得更多历史数据
            'end_date': '2023-12-31'
        },
        
        # 因子配置
        'factors': {
            'default_factors': [
                'return_5d', 'return_20d', 
                'volume_ratio', 'price_momentum',
                'volatility_20d', 'rsi_14d',
                'ma_ratio_10d', 'turnover_rate'
            ],
            'low_vol_threshold': 0.5,  # 放宽阈值
            'low_vol_window': 20  # 缩短窗口
        },
        
        # 强化学习环境配置
        'environment': {
            'lookback_window': 20,
            'transaction_cost': 0.001,
            'max_position': 0.1,
            'max_leverage': 1.2,
            'lambda1': 2.0,  # 回撤惩罚系数
            'lambda2': 1.0,  # CVaR惩罚系数
            'max_dd_threshold': 0.05
        },
        
        # CVaR-PPO智能体配置
        'agent': {
            'hidden_dim': 256,
            'learning_rate': 1e-5,  # 大幅降低学习率防止梯度爆炸
            'clip_epsilon': 0.2,
            'ppo_epochs': 5,  # 减少训练轮数
            'batch_size': 32,  # 减小批次大小
            'gamma': 0.99,
            'lambda_gae': 0.95,
            'cvar_alpha': 0.05,
            'cvar_lambda': 0.5,  # 减少CVaR权重
            'cvar_threshold': -0.02
        },
        
        # 安全保护层配置
        'safety_shield': {
            'max_position': 0.1,
            'max_leverage': 1.2,
            'var_threshold': 0.015,
            'max_drawdown_threshold': 0.05,
            'volatility_threshold': 0.15,
            'lookback_window': 20
        },
        
        # 风险控制配置
        'risk_control': {
            'target_volatility': 0.12,
            'max_position': 0.1,
            'max_leverage': 1.2,
            'max_drawdown_threshold': 0.10,
            'enable_risk_parity': False,
            'alpha_weight': 0.7
        },
        
        # 目标波动率控制配置
        'target_volatility': {
            'target_volatility': 0.12,
            'vol_lookback': 20,
            'max_leverage_multiplier': 2.0,
            'min_leverage_multiplier': 0.5
        },
        
        # 风险平价配置
        'risk_parity': {
            'rp_method': 'inverse_volatility',
            'rp_max_weight': 0.2,
            'rp_min_weight': 0.0
        },
        
        # 动态止损配置
        'stop_loss': {
            'stop_loss_pct': 0.03,
            'trailing_stop_pct': 0.05,
            'max_drawdown_stop': 0.08,
            'rebalance_threshold': 0.05,
            'rebalance_frequency': 20
        },
        
        # 回测配置
        'backtest': {
            'initial_capital': 1000000,
            'transaction_cost': 0.001,
            'slippage': 0.0001,
            'commission': 0.0005,
            'risk_free_rate': 0.03
        },
        
        # 心理舒适度阈值配置
        'comfort_thresholds': {
            'monthly_dd_threshold': 0.05,
            'max_consecutive_losses': 5,
            'max_loss_ratio': 0.4,
            'var_95_threshold': 0.01
        },
        
        # 训练配置
        'training': {
            'total_episodes': 500,  # 减少总训练轮数
            'max_steps_per_episode': 252,
            'update_frequency': 50,  # 更频繁的更新
            'save_frequency': 100,  # 更频繁的保存
            'evaluation_frequency': 25,
            'early_stopping_patience': 50,
            'min_improvement': 0.01
        },
        
        # 模型保存配置
        'model': {
            'save_dir': './models',
            'model_name': 'cvar_ppo_agent',
            'save_best_only': True,
            'monitor_metric': 'sharpe_ratio'
        },
        
        # 日志配置
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_dir': './logs'
        }
    }
    
    return config


def get_training_config() -> dict:
    """获取训练专用配置"""
    base_config = get_default_config()
    
    # 训练特定的调整
    training_adjustments = {
        'data': {
            'start_date': '2018-01-01',
            'end_date': '2022-12-31'
        },
        'agent': {
            'learning_rate': 5e-4,  # 稍高的学习率用于训练
            'ppo_epochs': 15,
            'batch_size': 128
        },
        'training': {
            'total_episodes': 2000,
            'evaluation_frequency': 25
        }
    }
    
    # 深度合并配置
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(base_config, training_adjustments)
    return base_config


def get_backtest_config() -> dict:
    """获取回测专用配置"""
    base_config = get_default_config()
    
    # 回测特定的调整
    backtest_adjustments = {
        'data': {
            'start_date': '2022-01-01',
            'end_date': '2023-12-31'
        },
        'agent': {
            'learning_rate': 1e-4,  # 较低学习率用于微调
        },
        'environment': {
            'transaction_cost': 0.0015,  # 更现实的交易成本
        },
        'backtest': {
            'transaction_cost': 0.0015,
            'commission': 0.0008
        }
    }
    
    # 深度合并配置
    def deep_update(base_dict, update_dict):
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    deep_update(base_config, backtest_adjustments)
    return base_config