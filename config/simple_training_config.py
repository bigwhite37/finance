"""
简化的训练配置 - 专注于数值稳定性
"""

def get_stable_config() -> dict:
    """获取数值稳定的配置"""
    
    config = {
        # 数据配置 - 使用更少的股票和更短的时间窗口
        'data': {
            'provider': 'yahoo',
            'region': 'cn',
            'universe': 'csi300',
            'provider_uri': '~/.qlib/qlib_data/cn_data',
            'start_date': '2023-01-01',  # 使用更近期的数据
            'end_date': '2023-06-30',    # 更短的时间窗口
            'max_stocks': 20             # 限制股票数量
        },
        
        # 简化的因子配置
        'factors': {
            'default_factors': [
                'return_5d',      # 只使用最基本的因子
                'ma_ratio_10d',   # 简单技术指标
                'volume_ratio'    # 成交量指标
            ],
            'low_vol_threshold': 1.0,  # 不进行波动率筛选
            'low_vol_window': 10
        },
        
        # 简化的环境配置
        'environment': {
            'lookback_window': 10,     # 更短的回望窗口
            'transaction_cost': 0.0001, # 极低交易成本
            'max_position': 0.2,       # 放宽仓位限制
            'max_leverage': 1.0,       # 无杠杆
            'lambda1': 0.1,            # 极低惩罚系数
            'lambda2': 0.1,
            'max_dd_threshold': 0.2    # 放宽回撤阈值
        },
        
        # 极简的智能体配置
        'agent': {
            'hidden_dim': 16,          # 极小网络
            'learning_rate': 1e-7,     # 极低学习率
            'clip_epsilon': 0.001,     # 极小clip
            'ppo_epochs': 1,           # 单轮训练
            'batch_size': 2,           # 极小批次
            'gamma': 0.8,              # 低折扣因子
            'lambda_gae': 0.7,
            'cvar_alpha': 0.05,
            'cvar_lambda': 0.001,      # 几乎忽略CVaR
            'cvar_threshold': -0.1     # 放宽CVaR阈值
        },
        
        # 极宽松的安全约束
        'safety_shield': {
            'max_position': 0.3,       # 很宽松的仓位限制
            'max_leverage': 1.5,       # 适中杠杆
            'var_threshold': 0.1,      # 很宽松的VaR
            'max_drawdown_threshold': 0.3, # 很宽松的回撤
            'volatility_threshold': 0.5,   # 很宽松的波动率
            'lookback_window': 5       # 很短的回望窗口
        },
        
        # 简化的训练配置
        'training': {
            'total_episodes': 100,     # 更少的episode
            'max_steps_per_episode': 50, # 更少的步数
            'update_frequency': 20,    # 更频繁更新
            'save_frequency': 50,
            'evaluation_frequency': 10,
            'early_stopping_patience': 20,
            'min_improvement': 0.001
        },
        
        # 其他必要配置
        'risk_control': {
            'target_volatility': 0.2,
            'max_position': 0.2,
            'max_leverage': 1.0,
            'max_drawdown_threshold': 0.3,
            'enable_risk_parity': False,
            'alpha_weight': 0.5
        },
        
        'backtest': {
            'initial_capital': 1000000,
            'transaction_cost': 0.0001,
            'slippage': 0.00001,
            'commission': 0.0001,
            'risk_free_rate': 0.03
        },
        
        'model': {
            'save_dir': './models',
            'model_name': 'simple_agent',
            'save_best_only': True,
            'monitor_metric': 'total_return'  # 使用更简单的指标
        },
        
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_dir': './logs'
        }
    }
    
    return config