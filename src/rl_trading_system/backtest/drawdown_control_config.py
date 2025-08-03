"""
回撤控制配置模块

定义回撤控制策略的配置参数和数据结构。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import yaml
import json
from pathlib import Path


@dataclass
class DrawdownControlConfig:
    """回撤控制配置类"""
    
    # 回撤监控参数
    max_drawdown_threshold: float = 0.15        # 最大回撤阈值
    drawdown_warning_threshold: float = 0.08    # 回撤警告阈值
    drawdown_calculation_window: int = 252      # 回撤计算窗口
    
    # 动态止损参数
    base_stop_loss: float = 0.05               # 基础止损阈值
    volatility_multiplier: float = 2.0         # 波动率乘数
    trailing_stop_distance: float = 0.03       # 追踪止损距离
    portfolio_stop_loss: float = 0.12          # 组合级止损
    enable_trailing_stop: bool = True          # 启用追踪止损
    
    # 仓位管理参数
    max_position_size: float = 0.15            # 最大单一持仓
    min_position_size: float = 0.01            # 最小持仓
    max_sector_exposure: float = 0.30          # 最大行业暴露
    cash_reserve_ratio: float = 0.05           # 现金储备比例
    
    # 风险预算参数
    base_risk_budget: float = 0.10             # 基础风险预算
    risk_scaling_factor: float = 0.5           # 风险缩放因子
    recovery_speed: float = 0.1                # 恢复速度
    
    # 奖励函数参数
    drawdown_penalty_factor: float = 2.0       # 回撤惩罚因子
    risk_aversion_coefficient: float = 0.5     # 风险厌恶系数
    diversification_bonus: float = 0.1         # 多样化奖励
    sharpe_target: float = 1.5                 # 目标夏普比率
    
    # 市场状态感知参数
    enable_market_regime_detection: bool = True  # 启用市场状态识别
    volatility_lookback: int = 20               # 波动率回望期
    trend_lookback: int = 30                    # 趋势回望期
    
    # 回测验证参数
    enable_parameter_optimization: bool = False  # 启用参数优化
    optimization_method: str = "grid_search"    # 优化方法: grid_search, bayesian
    validation_split: float = 0.2               # 验证集比例
    cross_validation_folds: int = 5             # 交叉验证折数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'drawdown_warning_threshold': self.drawdown_warning_threshold,
            'drawdown_calculation_window': self.drawdown_calculation_window,
            'base_stop_loss': self.base_stop_loss,
            'volatility_multiplier': self.volatility_multiplier,
            'trailing_stop_distance': self.trailing_stop_distance,
            'portfolio_stop_loss': self.portfolio_stop_loss,
            'enable_trailing_stop': self.enable_trailing_stop,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'max_sector_exposure': self.max_sector_exposure,
            'cash_reserve_ratio': self.cash_reserve_ratio,
            'base_risk_budget': self.base_risk_budget,
            'risk_scaling_factor': self.risk_scaling_factor,
            'recovery_speed': self.recovery_speed,
            'drawdown_penalty_factor': self.drawdown_penalty_factor,
            'risk_aversion_coefficient': self.risk_aversion_coefficient,
            'diversification_bonus': self.diversification_bonus,
            'sharpe_target': self.sharpe_target,
            'enable_market_regime_detection': self.enable_market_regime_detection,
            'volatility_lookback': self.volatility_lookback,
            'trend_lookback': self.trend_lookback,
            'enable_parameter_optimization': self.enable_parameter_optimization,
            'optimization_method': self.optimization_method,
            'validation_split': self.validation_split,
            'cross_validation_folds': self.cross_validation_folds
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DrawdownControlConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DrawdownControlConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict.get('drawdown_control', {}))
    
    @classmethod
    def from_json(cls, json_path: str) -> 'DrawdownControlConfig':
        """从JSON文件加载配置"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict.get('drawdown_control', {}))
    
    def save_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        config_dict = {'drawdown_control': self.to_dict()}
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def save_json(self, json_path: str):
        """保存配置到JSON文件"""
        config_dict = {'drawdown_control': self.to_dict()}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def validate(self) -> List[str]:
        """验证配置参数的合理性"""
        errors = []
        
        # 验证回撤阈值
        if not 0 < self.max_drawdown_threshold <= 1:
            errors.append("最大回撤阈值必须在0到1之间")
        
        if not 0 < self.drawdown_warning_threshold <= self.max_drawdown_threshold:
            errors.append("回撤警告阈值必须在0到最大回撤阈值之间")
        
        # 验证止损参数
        if not 0 < self.base_stop_loss <= 1:
            errors.append("基础止损阈值必须在0到1之间")
        
        if self.volatility_multiplier <= 0:
            errors.append("波动率乘数必须大于0")
        
        # 验证仓位参数
        if not 0 < self.max_position_size <= 1:
            errors.append("最大单一持仓必须在0到1之间")
        
        if not 0 <= self.min_position_size <= self.max_position_size:
            errors.append("最小持仓必须在0到最大持仓之间")
        
        # 验证其他参数
        if not 0 <= self.cash_reserve_ratio <= 1:
            errors.append("现金储备比例必须在0到1之间")
        
        if not 0 < self.validation_split < 1:
            errors.append("验证集比例必须在0到1之间")
        
        if self.cross_validation_folds < 2:
            errors.append("交叉验证折数必须至少为2")
        
        return errors
    
    def get_reward_config(self):
        """获取奖励优化器配置"""
        from ..risk_control.reward_optimizer import RewardConfig
        
        return RewardConfig(
            drawdown_penalty_factor=self.drawdown_penalty_factor,
            risk_aversion_coefficient=self.risk_aversion_coefficient,
            diversification_bonus=self.diversification_bonus,
            sharpe_target=self.sharpe_target
        )
    
    def create_parameter_grid(self) -> Dict[str, List[Any]]:
        """创建参数网格搜索的参数范围"""
        return {
            'max_drawdown_threshold': [0.10, 0.12, 0.15, 0.18, 0.20],
            'drawdown_warning_threshold': [0.05, 0.06, 0.08, 0.10],
            'base_stop_loss': [0.03, 0.05, 0.07, 0.10],
            'volatility_multiplier': [1.5, 2.0, 2.5, 3.0],
            'trailing_stop_distance': [0.02, 0.03, 0.04, 0.05],
            'max_position_size': [0.10, 0.12, 0.15, 0.18],
            'drawdown_penalty_factor': [1.0, 1.5, 2.0, 2.5, 3.0],
            'risk_aversion_coefficient': [0.3, 0.5, 0.7, 1.0],
            'diversification_bonus': [0.05, 0.1, 0.15, 0.2]
        }
    
    def create_bayesian_bounds(self) -> Dict[str, tuple]:
        """创建贝叶斯优化的参数边界"""
        return {
            'max_drawdown_threshold': (0.08, 0.25),
            'drawdown_warning_threshold': (0.03, 0.12),
            'base_stop_loss': (0.02, 0.15),
            'volatility_multiplier': (1.0, 5.0),
            'trailing_stop_distance': (0.01, 0.08),
            'max_position_size': (0.08, 0.25),
            'drawdown_penalty_factor': (0.5, 5.0),
            'risk_aversion_coefficient': (0.1, 2.0),
            'diversification_bonus': (0.01, 0.5)
        }


@dataclass 
class BacktestComparisonConfig:
    """回测对比配置"""
    baseline_strategies: List[str] = field(default_factory=lambda: ['buy_and_hold', 'equal_weight'])
    benchmarks: List[str] = field(default_factory=lambda: ['000300.SH', '000905.SH'])
    comparison_metrics: List[str] = field(default_factory=lambda: [
        'total_return', 'annual_return', 'volatility', 'sharpe_ratio', 
        'max_drawdown', 'calmar_ratio', 'win_rate', 'profit_factor'
    ])
    significance_test_methods: List[str] = field(default_factory=lambda: ['t_test', 'mann_whitney'])
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'baseline_strategies': self.baseline_strategies,
            'benchmarks': self.benchmarks,
            'comparison_metrics': self.comparison_metrics,
            'significance_test_methods': self.significance_test_methods,
            'confidence_level': self.confidence_level
        }