"""
配置参数验证器

实现回撤控制配置的参数验证、类型检查和合理性验证。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果数据类"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'valid': self.valid,
            'errors': self.errors,
            'warnings': self.warnings
        }


@dataclass
class ParameterRange:
    """参数范围定义"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_values: Optional[List[Any]] = None
    required: bool = True
    data_type: type = float
    
    def validate(self, value: Any, param_name: str) -> List[str]:
        """验证参数值"""
        errors = []
        
        # 检查是否为必需参数
        if self.required and value is None:
            errors.append(f"参数 {param_name} 为必需参数，不能为空")
            return errors
        
        if value is None:
            return errors  # 非必需参数可以为空
        
        # 检查数据类型
        if not isinstance(value, self.data_type):
            try:
                value = self.data_type(value)
            except (ValueError, TypeError):
                errors.append(f"参数 {param_name} 应为 {self.data_type.__name__} 类型，当前为 {type(value).__name__}")
                return errors
        
        # 检查有效值列表
        if self.valid_values is not None and value not in self.valid_values:
            errors.append(f"参数 {param_name} 的值 {value} 不在有效值列表中: {self.valid_values}")
        
        # 检查数值范围
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                errors.append(f"参数 {param_name} 的值 {value} 小于最小值 {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"参数 {param_name} 的值 {value} 大于最大值 {self.max_value}")
        
        return errors


class ConfigValidator:
    """
    配置参数验证器
    
    提供完整的配置参数验证功能：
    - 参数类型和范围验证
    - 参数合理性检查
    - 参数依赖关系验证
    - 业务逻辑一致性检查
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 定义回撤控制配置参数验证规则
        self.drawdown_control_rules = {
            # 基础配置
            'max_drawdown_threshold': ParameterRange(min_value=0.01, max_value=0.5, data_type=float, required=False),
            'portfolio_stop_loss': ParameterRange(min_value=0.05, max_value=0.8, data_type=float, required=False),
            'drawdown_calculation_window': ParameterRange(min_value=10, max_value=1000, data_type=int, required=False),
            
            # 止损配置
            'base_stop_loss': ParameterRange(min_value=0.01, max_value=0.3, data_type=float, required=False),
            'volatility_multiplier': ParameterRange(min_value=0.5, max_value=5.0, data_type=float, required=False),
            'trailing_stop_distance': ParameterRange(min_value=0.001, max_value=0.1, data_type=float, required=False),
            
            # 奖励优化配置
            'drawdown_penalty_factor': ParameterRange(min_value=0.1, max_value=10.0, data_type=float, required=False),
            'risk_aversion_coefficient': ParameterRange(min_value=0.1, max_value=10.0, data_type=float, required=False),
            
            # 风险预算配置
            'base_risk_budget': ParameterRange(min_value=0.01, max_value=1.0, data_type=float, required=False),
            'risk_budget_adjustment_speed': ParameterRange(min_value=0.01, max_value=1.0, data_type=float, required=False),
            
            # 市场状态检测配置
            'enable_market_regime_detection': ParameterRange(valid_values=[True, False], data_type=bool, required=False),
            'lookback_window': ParameterRange(min_value=20, max_value=500, data_type=int, required=False),
            
            # 压力测试配置
            'stress_test_scenarios': ParameterRange(min_value=1, max_value=100, data_type=int, required=False),
            'confidence_level': ParameterRange(min_value=0.9, max_value=0.999, data_type=float, required=False),
        }
        
        self.logger.info("配置验证器初始化完成")
    
    def validate_drawdown_control_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证回撤控制配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            验证结果字典
        """
        errors = []
        warnings = []
        
        try:
            # 基础参数验证
            basic_validation = self._validate_basic_parameters(config_dict)
            errors.extend(basic_validation.errors)
            warnings.extend(basic_validation.warnings)
            
            # 参数关系验证
            relationship_validation = self._validate_parameter_relationships(config_dict)
            errors.extend(relationship_validation.errors)
            warnings.extend(relationship_validation.warnings)
            
            # 业务逻辑验证
            business_validation = self._validate_business_logic(config_dict)
            errors.extend(business_validation.errors)
            warnings.extend(business_validation.warnings)
            
            # 性能影响验证
            performance_validation = self._validate_performance_impact(config_dict)
            warnings.extend(performance_validation.warnings)
            
            is_valid = len(errors) == 0
            
            result = ValidationResult(
                valid=is_valid,
                errors=errors,
                warnings=warnings
            )
            
            if is_valid:
                self.logger.info("配置验证通过")
            else:
                self.logger.error(f"配置验证失败: {len(errors)} 个错误")
            
            return result.to_dict()
            
        except Exception as e:
            self.logger.error(f"配置验证过程中发生错误: {e}")
            return {
                'valid': False,
                'errors': [f"验证过程异常: {e}"],
                'warnings': []
            }
    
    def _validate_basic_parameters(self, config_dict: Dict[str, Any]) -> ValidationResult:
        """验证基础参数"""
        errors = []
        warnings = []
        
        for param_name, rule in self.drawdown_control_rules.items():
            value = config_dict.get(param_name)
            param_errors = rule.validate(value, param_name)
            errors.extend(param_errors)
        
        # 检查未知参数
        known_params = set(self.drawdown_control_rules.keys())
        config_params = set(config_dict.keys())
        unknown_params = config_params - known_params
        
        for unknown_param in unknown_params:
            warnings.append(f"发现未知参数: {unknown_param}")
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_parameter_relationships(self, config_dict: Dict[str, Any]) -> ValidationResult:
        """验证参数关系"""
        errors = []
        warnings = []
        
        # 检查回撤阈值关系
        max_drawdown = config_dict.get('max_drawdown_threshold', 0.2)
        portfolio_stop = config_dict.get('portfolio_stop_loss', 0.3)
        
        if max_drawdown and portfolio_stop and max_drawdown >= portfolio_stop:
            errors.append(f"最大回撤阈值 ({max_drawdown}) 应小于组合止损 ({portfolio_stop})")
        
        # 检查止损参数关系
        base_stop = config_dict.get('base_stop_loss', 0.05)
        trailing_distance = config_dict.get('trailing_stop_distance', 0.02)
        
        if base_stop and trailing_distance and trailing_distance >= base_stop:
            warnings.append(f"追踪止损距离 ({trailing_distance}) 过大，可能影响止损效果")
        
        # 检查风险预算参数
        base_budget = config_dict.get('base_risk_budget', 0.1)
        adjustment_speed = config_dict.get('risk_budget_adjustment_speed', 0.1)
        
        if base_budget and adjustment_speed and adjustment_speed > base_budget:
            warnings.append(f"风险预算调整速度 ({adjustment_speed}) 大于基础预算 ({base_budget})，可能导致过度调整")
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_business_logic(self, config_dict: Dict[str, Any]) -> ValidationResult:
        """验证业务逻辑一致性"""
        errors = []
        warnings = []
        
        # 检查奖励函数参数合理性
        penalty_factor = config_dict.get('drawdown_penalty_factor', 1.0)
        risk_aversion = config_dict.get('risk_aversion_coefficient', 1.0)
        
        if penalty_factor and risk_aversion:
            if penalty_factor > 5.0 and risk_aversion > 5.0:
                warnings.append("回撤惩罚因子和风险厌恶系数都很高，可能导致过于保守的策略")
            elif penalty_factor < 0.5 and risk_aversion < 0.5:
                warnings.append("回撤惩罚因子和风险厌恶系数都很低，可能导致过于激进的策略")
        
        # 检查时间窗口参数
        drawdown_window = config_dict.get('drawdown_calculation_window', 252)
        lookback_window = config_dict.get('lookback_window', 60)
        
        if drawdown_window and lookback_window:
            if lookback_window > drawdown_window:
                warnings.append(f"回望窗口 ({lookback_window}) 大于回撤计算窗口 ({drawdown_window})，可能影响计算准确性")
        
        # 检查市场状态检测配置
        enable_regime = config_dict.get('enable_market_regime_detection', False)
        if not enable_regime:
            warnings.append("未启用市场状态检测，可能降低风险控制的适应性")
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_performance_impact(self, config_dict: Dict[str, Any]) -> ValidationResult:
        """验证性能影响"""
        warnings = []
        
        # 检查计算密集型参数
        drawdown_window = config_dict.get('drawdown_calculation_window', 252)
        lookback_window = config_dict.get('lookback_window', 60)
        stress_scenarios = config_dict.get('stress_test_scenarios', 0)
        
        if drawdown_window and drawdown_window > 1000:
            warnings.append(f"回撤计算窗口 ({drawdown_window}) 很大，可能影响计算性能")
        
        if lookback_window and lookback_window > 200:
            warnings.append(f"回望窗口 ({lookback_window}) 很大，可能影响实时性能")
        
        if stress_scenarios and stress_scenarios > 50:
            warnings.append(f"压力测试场景数 ({stress_scenarios}) 很多，可能影响测试性能")
        
        return ValidationResult(valid=True, errors=[], warnings=warnings)
    
    def validate_parameter_update(self, param_name: str, new_value: Any, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证单个参数更新
        
        Args:
            param_name: 参数名称
            new_value: 新值
            current_config: 当前配置
            
        Returns:
            验证结果
        """
        if param_name not in self.drawdown_control_rules:
            return {
                'valid': False,
                'errors': [f"未知参数: {param_name}"],
                'warnings': []
            }
        
        # 创建临时配置用于验证
        temp_config = current_config.copy()
        temp_config[param_name] = new_value
        
        # 验证单个参数
        rule = self.drawdown_control_rules[param_name]
        param_errors = rule.validate(new_value, param_name)
        
        if param_errors:
            return {
                'valid': False,
                'errors': param_errors,
                'warnings': []
            }
        
        # 验证参数关系
        relationship_validation = self._validate_parameter_relationships(temp_config)
        
        return {
            'valid': len(relationship_validation.errors) == 0,
            'errors': relationship_validation.errors,
            'warnings': relationship_validation.warnings
        }
    
    def get_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """
        获取参数信息
        
        Args:
            param_name: 参数名称
            
        Returns:
            参数信息字典
        """
        if param_name not in self.drawdown_control_rules:
            return None
        
        rule = self.drawdown_control_rules[param_name]
        
        return {
            'name': param_name,
            'data_type': rule.data_type.__name__,
            'required': rule.required,
            'min_value': rule.min_value,
            'max_value': rule.max_value,
            'valid_values': rule.valid_values,
            'description': self._get_parameter_description(param_name)
        }
    
    def _get_parameter_description(self, param_name: str) -> str:
        """获取参数描述"""
        descriptions = {
            'max_drawdown_threshold': '最大回撤阈值，超过此值将触发风险控制',
            'portfolio_stop_loss': '组合级止损阈值，触发时执行止损操作',
            'drawdown_calculation_window': '回撤计算的时间窗口（天数）',
            'base_stop_loss': '基础止损比例',
            'volatility_multiplier': '波动率调整乘数，用于动态调整止损',
            'trailing_stop_distance': '追踪止损距离',
            'drawdown_penalty_factor': '回撤惩罚因子，用于奖励函数调整',
            'risk_aversion_coefficient': '风险厌恶系数',
            'base_risk_budget': '基础风险预算比例',
            'risk_budget_adjustment_speed': '风险预算调整速度',
            'enable_market_regime_detection': '是否启用市场状态检测',
            'lookback_window': '市场状态检测的回望窗口',
            'stress_test_scenarios': '压力测试场景数量',
            'confidence_level': '置信水平'
        }
        
        return descriptions.get(param_name, '未知参数')
    
    def get_all_parameters_info(self) -> List[Dict[str, Any]]:
        """获取所有参数信息"""
        return [self.get_parameter_info(param_name) for param_name in self.drawdown_control_rules.keys()]