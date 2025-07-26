"""
O2O配置验证器
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class O2OConfigValidator:
    """O2O配置验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.validation_rules = self._setup_validation_rules()
        self.dependency_rules = self._setup_dependency_rules()
        self.optimization_suggestions = self._setup_optimization_suggestions()
    
    def validate_o2o_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证O2O配置的完整性和有效性
        
        Args:
            config: O2O配置字典
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        try:
            # 验证必需的配置节
            required_sections = [
                'offline_pretraining', 'warmup_finetuning', 'online_learning',
                'drift_detection', 'buffer_config', 'risk_constraints',
                'training_flow', 'monitoring'
            ]
            
            for section in required_sections:
                if section not in config:
                    errors.append(f"缺少必需的配置节: {section}")
            
            # 验证各个配置节
            if 'offline_pretraining' in config:
                errors.extend(self._validate_offline_pretraining(config['offline_pretraining']))
            
            if 'warmup_finetuning' in config:
                errors.extend(self._validate_warmup_finetuning(config['warmup_finetuning']))
            
            if 'online_learning' in config:
                errors.extend(self._validate_online_learning(config['online_learning']))
            
            if 'drift_detection' in config:
                errors.extend(self._validate_drift_detection(config['drift_detection']))
            
            if 'buffer_config' in config:
                errors.extend(self._validate_buffer_config(config['buffer_config']))
            
            if 'risk_constraints' in config:
                errors.extend(self._validate_risk_constraints(config['risk_constraints']))
            
            if 'training_flow' in config:
                errors.extend(self._validate_training_flow(config['training_flow']))
            
            if 'monitoring' in config:
                errors.extend(self._validate_monitoring(config['monitoring']))
            
            # 验证配置间的依赖关系
            errors.extend(self._validate_dependencies(config))
            
            # 验证参数范围
            errors.extend(self._validate_parameter_ranges(config))
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"配置验证过程中发生错误: {e}")
            return False, [f"验证过程异常: {str(e)}"]
    
    def _validate_offline_pretraining(self, config: Dict[str, Any]) -> List[str]:
        """验证离线预训练配置"""
        errors = []
        
        # 验证epochs
        epochs = config.get('epochs', 0)
        if not isinstance(epochs, int) or epochs <= 0:
            errors.append("离线预训练epochs必须是正整数")
        elif epochs > 1000:
            errors.append("离线预训练epochs过大，建议不超过1000")
        
        # 验证权重
        bc_weight = config.get('behavior_cloning_weight', 0.5)
        td_weight = config.get('td_learning_weight', 0.5)
        
        if not (0 <= bc_weight <= 1):
            errors.append("行为克隆权重必须在[0,1]范围内")
        if not (0 <= td_weight <= 1):
            errors.append("TD学习权重必须在[0,1]范围内")
        if abs(bc_weight + td_weight - 1.0) > 1e-6:
            errors.append("行为克隆权重和TD学习权重之和应该等于1.0")
        
        # 验证学习率
        lr = config.get('learning_rate', 1e-4)
        if not (1e-6 <= lr <= 1e-2):
            errors.append("离线预训练学习率建议在[1e-6, 1e-2]范围内")
        
        # 验证批次大小
        batch_size = config.get('batch_size', 256)
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append("批次大小必须是正整数")
        elif batch_size < 32:
            errors.append("批次大小过小，建议至少32")
        elif batch_size > 1024:
            errors.append("批次大小过大，建议不超过1024")
        
        return errors
    
    def _validate_warmup_finetuning(self, config: Dict[str, Any]) -> List[str]:
        """验证热身微调配置"""
        errors = []
        
        # 验证天数
        days = config.get('days', 60)
        if not isinstance(days, int) or days <= 0:
            errors.append("热身天数必须是正整数")
        elif days < 20:
            errors.append("热身天数过短，建议至少20天")
        elif days > 120:
            errors.append("热身天数过长，建议不超过120天")
        
        # 验证epochs
        epochs = config.get('epochs', 20)
        if not isinstance(epochs, int) or epochs <= 0:
            errors.append("热身epochs必须是正整数")
        elif epochs > 100:
            errors.append("热身epochs过大，建议不超过100")
        
        # 验证学习率
        lr = config.get('learning_rate', 5e-5)
        if not (1e-7 <= lr <= 1e-3):
            errors.append("热身学习率建议在[1e-7, 1e-3]范围内")
        
        # 验证收敛阈值
        conv_threshold = config.get('convergence_threshold', 1e-4)
        if not (1e-6 <= conv_threshold <= 1e-2):
            errors.append("收敛阈值建议在[1e-6, 1e-2]范围内")
        
        return errors
    
    def _validate_online_learning(self, config: Dict[str, Any]) -> List[str]:
        """验证在线学习配置"""
        errors = []
        
        # 验证初始rho
        initial_rho = config.get('initial_rho', 0.2)
        if not (0 < initial_rho <= 1):
            errors.append("初始在线采样比例必须在(0,1]范围内")
        elif initial_rho > 0.5:
            errors.append("初始在线采样比例过高，建议不超过0.5")
        
        # 验证rho增量
        rho_increment = config.get('rho_increment', 0.01)
        if not (0 < rho_increment <= 0.1):
            errors.append("采样比例增量建议在(0, 0.1]范围内")
        
        # 验证信任域参数
        trust_beta = config.get('trust_region_beta', 1.0)
        if not (0.1 <= trust_beta <= 10.0):
            errors.append("信任域beta参数建议在[0.1, 10.0]范围内")
        
        beta_decay = config.get('beta_decay', 0.99)
        if not (0.9 <= beta_decay <= 1.0):
            errors.append("beta衰减因子建议在[0.9, 1.0]范围内")
        
        # 验证学习率
        lr = config.get('learning_rate', 3e-4)
        if not (1e-6 <= lr <= 1e-2):
            errors.append("在线学习率建议在[1e-6, 1e-2]范围内")
        
        return errors
    
    def _validate_drift_detection(self, config: Dict[str, Any]) -> List[str]:
        """验证漂移检测配置"""
        errors = []
        
        # 验证KL阈值
        kl_threshold = config.get('kl_threshold', 0.1)
        if not (0.01 <= kl_threshold <= 1.0):
            errors.append("KL散度阈值建议在[0.01, 1.0]范围内")
        
        # 验证夏普率下降阈值
        sharpe_threshold = config.get('sharpe_drop_threshold', 0.2)
        if not (0.05 <= sharpe_threshold <= 0.5):
            errors.append("夏普率下降阈值建议在[0.05, 0.5]范围内")
        
        # 验证夏普率窗口
        sharpe_window = config.get('sharpe_window', 30)
        if not isinstance(sharpe_window, int) or not (10 <= sharpe_window <= 60):
            errors.append("夏普率窗口建议在[10, 60]天范围内")
        
        # 验证CVaR阈值
        cvar_threshold = config.get('cvar_breach_threshold', -0.02)
        if not (-0.1 <= cvar_threshold <= 0):
            errors.append("CVaR突破阈值建议在[-0.1, 0]范围内")
        
        # 验证连续天数
        consecutive_days = config.get('cvar_consecutive_days', 3)
        if not isinstance(consecutive_days, int) or not (1 <= consecutive_days <= 10):
            errors.append("CVaR连续天数建议在[1, 10]范围内")
        
        return errors
    
    def _validate_buffer_config(self, config: Dict[str, Any]) -> List[str]:
        """验证缓冲区配置"""
        errors = []
        
        # 验证缓冲区大小
        buffer_size = config.get('online_buffer_size', 10000)
        if not isinstance(buffer_size, int) or buffer_size <= 0:
            errors.append("在线缓冲区大小必须是正整数")
        elif buffer_size < 1000:
            errors.append("在线缓冲区大小过小，建议至少1000")
        elif buffer_size > 100000:
            errors.append("在线缓冲区大小过大，建议不超过100000")
        
        # 验证优先级参数
        priority_alpha = config.get('priority_alpha', 0.6)
        if not (0 < priority_alpha <= 1):
            errors.append("优先级alpha参数必须在(0,1]范围内")
        
        priority_beta = config.get('priority_beta', 0.4)
        if not (0 < priority_beta <= 1):
            errors.append("优先级beta参数必须在(0,1]范围内")
        
        # 验证时间衰减因子
        decay_factor = config.get('time_decay_factor', 0.99)
        if not (0.9 <= decay_factor <= 1.0):
            errors.append("时间衰减因子建议在[0.9, 1.0]范围内")
        
        # 验证最小离线比例
        min_offline_ratio = config.get('min_offline_ratio', 0.1)
        if not (0 <= min_offline_ratio <= 0.5):
            errors.append("最小离线比例建议在[0, 0.5]范围内")
        
        return errors
    
    def _validate_risk_constraints(self, config: Dict[str, Any]) -> List[str]:
        """验证风险约束配置"""
        errors = []
        
        # 验证基础CVaR lambda
        base_lambda = config.get('base_cvar_lambda', 1.0)
        if not (0.1 <= base_lambda <= 10.0):
            errors.append("基础CVaR lambda建议在[0.1, 10.0]范围内")
        
        # 验证缩放因子
        scaling_factor = config.get('lambda_scaling_factor', 1.5)
        if not (1.0 <= scaling_factor <= 5.0):
            errors.append("lambda缩放因子建议在[1.0, 5.0]范围内")
        
        # 验证紧急风险倍数
        emergency_multiplier = config.get('emergency_risk_multiplier', 2.0)
        if not (1.5 <= emergency_multiplier <= 10.0):
            errors.append("紧急风险倍数建议在[1.5, 10.0]范围内")
        
        return errors
    
    def _validate_training_flow(self, config: Dict[str, Any]) -> List[str]:
        """验证训练流程配置"""
        errors = []
        
        # 验证布尔值配置
        bool_configs = [
            'enable_offline_pretraining', 'enable_warmup_finetuning',
            'enable_online_learning', 'auto_stage_transition',
            'save_intermediate_models', 'model_versioning'
        ]
        
        for key in bool_configs:
            value = config.get(key, True)
            if not isinstance(value, bool):
                errors.append(f"{key}必须是布尔值")
        
        # 验证至少启用一个训练阶段
        if not any([
            config.get('enable_offline_pretraining', True),
            config.get('enable_warmup_finetuning', True),
            config.get('enable_online_learning', True)
        ]):
            errors.append("至少需要启用一个训练阶段")
        
        return errors
    
    def _validate_monitoring(self, config: Dict[str, Any]) -> List[str]:
        """验证监控配置"""
        errors = []
        
        # 验证报告频率
        report_freq = config.get('report_frequency', 50)
        if not isinstance(report_freq, int) or report_freq <= 0:
            errors.append("报告频率必须是正整数")
        elif report_freq > 1000:
            errors.append("报告频率过高，建议不超过1000")
        
        return errors
    
    def _validate_dependencies(self, config: Dict[str, Any]) -> List[str]:
        """验证配置间的依赖关系"""
        errors = []
        
        # 检查训练流程依赖
        training_flow = config.get('training_flow', {})
        
        # 如果启用在线学习，必须启用漂移检测
        if (training_flow.get('enable_online_learning', True) and 
            not config.get('drift_detection', {}).get('enable_auto_retraining', True)):
            errors.append("启用在线学习时建议启用自动重训练")
        
        # 如果启用热身微调，必须有足够的热身天数
        if training_flow.get('enable_warmup_finetuning', True):
            warmup_days = config.get('warmup_finetuning', {}).get('days', 60)
            if warmup_days < 30:
                errors.append("启用热身微调时建议至少30天的热身期")
        
        # 检查采样比例一致性
        online_config = config.get('online_learning', {})
        initial_rho = online_config.get('initial_rho', 0.2)
        rho_increment = online_config.get('rho_increment', 0.01)
        max_rho = online_config.get('max_rho', 1.0)
        
        if initial_rho + rho_increment > max_rho:
            errors.append("初始rho + 增量不应超过最大rho")
        
        return errors
    
    def _validate_parameter_ranges(self, config: Dict[str, Any]) -> List[str]:
        """验证参数范围的合理性"""
        errors = []
        
        # 验证学习率的相对关系
        offline_lr = config.get('offline_pretraining', {}).get('learning_rate', 1e-4)
        warmup_lr = config.get('warmup_finetuning', {}).get('learning_rate', 5e-5)
        online_lr = config.get('online_learning', {}).get('learning_rate', 3e-4)
        
        if warmup_lr > offline_lr:
            errors.append("热身学习率通常应小于离线预训练学习率")
        
        if online_lr < warmup_lr:
            errors.append("在线学习率通常应大于热身学习率")
        
        # 验证批次大小的相对关系
        offline_batch = config.get('offline_pretraining', {}).get('batch_size', 256)
        warmup_batch = config.get('warmup_finetuning', {}).get('batch_size', 128)
        online_batch = config.get('online_learning', {}).get('batch_size', 64)
        
        if offline_batch < online_batch:
            errors.append("离线预训练批次大小通常应大于在线学习批次大小")
        
        return errors
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """设置验证规则"""
        return {
            'learning_rates': {
                'offline': (1e-6, 1e-2),
                'warmup': (1e-7, 1e-3),
                'online': (1e-6, 1e-2)
            },
            'batch_sizes': {
                'min': 16,
                'max': 1024,
                'recommended_min': 32
            },
            'epochs': {
                'offline_max': 1000,
                'warmup_max': 100
            },
            'rho_params': {
                'initial_max': 0.5,
                'increment_max': 0.1
            }
        }
    
    def _setup_dependency_rules(self) -> List[Dict[str, Any]]:
        """设置依赖关系规则"""
        return [
            {
                'condition': 'online_learning_enabled',
                'requires': 'drift_detection_enabled',
                'message': '启用在线学习时建议启用漂移检测'
            },
            {
                'condition': 'warmup_enabled',
                'requires': 'min_warmup_days',
                'message': '热身微调需要足够的热身天数'
            }
        ]
    
    def _setup_optimization_suggestions(self) -> Dict[str, List[str]]:
        """设置优化建议"""
        return {
            'performance': [
                '使用较大的离线预训练批次大小以提高训练稳定性',
                '在线学习阶段使用较小的批次大小以提高适应性',
                '根据数据量调整缓冲区大小'
            ],
            'stability': [
                '设置适当的信任域约束以防止策略发散',
                '使用渐进式的rho增长策略',
                '启用自动重训练以应对分布漂移'
            ],
            'efficiency': [
                '启用检查点保存以支持训练中断恢复',
                '使用模型版本管理追踪训练进度',
                '合理设置监控频率以平衡性能和可观测性'
            ]
        }
    
    def get_optimization_suggestions(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        根据配置提供优化建议
        
        Args:
            config: O2O配置
            
        Returns:
            优化建议字典
        """
        suggestions = {
            'performance': [],
            'stability': [],
            'efficiency': []
        }
        
        # 性能优化建议
        offline_batch = config.get('offline_pretraining', {}).get('batch_size', 256)
        if offline_batch < 128:
            suggestions['performance'].append('建议增加离线预训练批次大小到至少128')
        
        online_batch = config.get('online_learning', {}).get('batch_size', 64)
        if online_batch > 128:
            suggestions['performance'].append('建议减少在线学习批次大小到128以下')
        
        # 稳定性建议
        initial_rho = config.get('online_learning', {}).get('initial_rho', 0.2)
        if initial_rho > 0.3:
            suggestions['stability'].append('建议降低初始在线采样比例到0.3以下')
        
        trust_beta = config.get('online_learning', {}).get('trust_region_beta', 1.0)
        if trust_beta < 0.5:
            suggestions['stability'].append('信任域参数过小，可能导致学习过慢')
        elif trust_beta > 5.0:
            suggestions['stability'].append('信任域参数过大，可能导致训练不稳定')
        
        # 效率建议
        if not config.get('training_flow', {}).get('save_intermediate_models', True):
            suggestions['efficiency'].append('建议启用中间模型保存以支持训练恢复')
        
        report_freq = config.get('monitoring', {}).get('report_frequency', 50)
        if report_freq < 10:
            suggestions['efficiency'].append('报告频率过高，可能影响训练性能')
        elif report_freq > 200:
            suggestions['efficiency'].append('报告频率过低，可能错过重要信息')
        
        return suggestions
    
    def generate_validation_report(self, config: Dict[str, Any]) -> str:
        """
        生成详细的验证报告
        
        Args:
            config: O2O配置
            
        Returns:
            验证报告字符串
        """
        is_valid, errors = self.validate_o2o_config(config)
        suggestions = self.get_optimization_suggestions(config)
        
        report = []
        report.append("=== O2O配置验证报告 ===")
        report.append(f"验证状态: {'通过' if is_valid else '失败'}")
        report.append("")
        
        if errors:
            report.append("错误信息:")
            for i, error in enumerate(errors, 1):
                report.append(f"  {i}. {error}")
            report.append("")
        
        if any(suggestions.values()):
            report.append("优化建议:")
            for category, items in suggestions.items():
                if items:
                    report.append(f"  {category.upper()}:")
                    for item in items:
                        report.append(f"    - {item}")
            report.append("")
        
        report.append("=== 报告结束 ===")
        return "\n".join(report)


def validate_o2o_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    验证O2O配置的便捷函数
    
    Args:
        config: O2O配置字典
        
    Returns:
        (是否有效, 错误信息列表)
    """
    validator = O2OConfigValidator()
    return validator.validate_o2o_config(config)


def get_o2o_optimization_suggestions(config: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    获取O2O配置优化建议的便捷函数
    
    Args:
        config: O2O配置字典
        
    Returns:
        优化建议字典
    """
    validator = O2OConfigValidator()
    return validator.get_optimization_suggestions(config)