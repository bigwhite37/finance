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
                'return_5d', 'return_20d', 'return_60d',
                'volume_ratio', 'price_momentum', 'price_reversal',
                'volatility_20d', 'rsi_14d', 'bollinger_position',
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
        
        # CVaR-PPO智能体配置 - 极度保守设置
        'agent': {
            'hidden_dim': 32,   # 最小网络规模
            'learning_rate': 1e-6,  # 极低学习率
            'clip_epsilon': 0.01,   # 极小clip范围
            'ppo_epochs': 1,    # 单轮训练
            'batch_size': 4,    # 极小批次
            'gamma': 0.9,       # 更低折扣因子
            'lambda_gae': 0.8,  # 更低GAE参数
            'cvar_alpha': 0.05,
            'cvar_lambda': 0.01,  # 极小CVaR权重
            'cvar_threshold': -0.02
        },
        
        # 安全保护层配置
        'safety_shield': {
            'max_position': 0.15,  # 放宽单股票仓位限制
            'max_leverage': 1.2,   # 与risk_control保持一致
            'var_threshold': 0.025, # 放宽VaR阈值
            'max_drawdown_threshold': 0.08, # 放宽回撤阈值
            'volatility_threshold': 0.20,   # 放宽波动率阈值
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
        
        # 动态低波筛选器配置
        'dynamic_lowvol': {
            # 滚动分位筛选配置
            'rolling_windows': [20, 60],
            'percentile_thresholds': {
                '低': 0.4,
                '中': 0.3, 
                '高': 0.2
            },
            
            # GARCH预测配置
            'garch_window': 250,
            'forecast_horizon': 5,
            'enable_ml_predictor': False,
            'ml_predictor_type': 'lightgbm',  # 'lightgbm', 'lstm'
            
            # IVOL约束配置
            'ivol_bad_threshold': 0.3,
            'ivol_good_threshold': 0.6,
            'five_factor_window': 252,
            
            # 市场状态检测配置
            'regime_detection_window': 60,
            'regime_model_type': 'HMM',  # 'HMM', 'MS-GARCH'
            'regime_states': 3,
            
            # 性能优化配置
            'enable_caching': True,
            'cache_expiry_days': 1,
            'parallel_processing': True,
            'max_workers': 4,
            
            # 数据质量配置
            'min_data_length': 100,
            'max_missing_ratio': 0.1,
            'outlier_threshold': 5.0,
            
            # 模型收敛配置
            'max_garch_iterations': 1000,
            'garch_convergence_tolerance': 1e-6,
            'hmm_convergence_tolerance': 1e-4,
            'max_hmm_iterations': 100
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
        },
        
        # O2O强化学习配置
        'o2o': {
            # 离线预训练配置
            'offline_pretraining': {
                'epochs': 100,
                'behavior_cloning_weight': 0.5,
                'td_learning_weight': 0.5,
                'learning_rate': 1e-4,
                'batch_size': 256,
                'save_checkpoints': True,
                'checkpoint_frequency': 20
            },
            
            # 热身微调配置
            'warmup_finetuning': {
                'days': 60,
                'epochs': 20,
                'critic_only_updates': True,
                'learning_rate': 5e-5,
                'batch_size': 128,
                'convergence_threshold': 1e-4,
                'max_no_improvement': 5
            },
            
            # 在线学习配置
            'online_learning': {
                'initial_rho': 0.2,
                'rho_increment': 0.01,
                'max_rho': 1.0,
                'trust_region_beta': 1.0,
                'beta_decay': 0.99,
                'min_beta': 0.1,
                'learning_rate': 3e-4,
                'batch_size': 64,
                'update_frequency': 10
            },
            
            # 漂移检测配置
            'drift_detection': {
                'kl_threshold': 0.1,
                'sharpe_drop_threshold': 0.2,
                'sharpe_window': 30,
                'cvar_breach_threshold': -0.02,
                'cvar_consecutive_days': 3,
                'monitoring_frequency': 5,
                'enable_auto_retraining': True
            },
            
            # 缓冲区配置
            'buffer_config': {
                'online_buffer_size': 10000,
                'priority_alpha': 0.6,
                'priority_beta': 0.4,
                'time_decay_factor': 0.99,
                'min_offline_ratio': 0.1,
                'fifo_eviction': True
            },
            
            # 风险约束配置
            'risk_constraints': {
                'dynamic_cvar_lambda': True,
                'base_cvar_lambda': 1.0,
                'lambda_scaling_factor': 1.5,
                'emergency_risk_multiplier': 2.0,
                'regime_aware_adjustment': True
            },
            
            # 训练流程配置
            'training_flow': {
                'enable_offline_pretraining': True,
                'enable_warmup_finetuning': True,
                'enable_online_learning': True,
                'auto_stage_transition': True,
                'save_intermediate_models': True,
                'model_versioning': True
            },
            
            # 监控和日志配置
            'monitoring': {
                'log_sampling_ratio': True,
                'log_drift_metrics': True,
                'log_performance_metrics': True,
                'save_training_history': True,
                'generate_reports': True,
                'report_frequency': 50
            }
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