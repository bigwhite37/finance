"""
配置管理器
"""

import yaml
import json
import os
from typing import Dict, Any, Optional, List, Tuple
import logging
from .default_config import get_default_config
from .dynamic_lowvol_validator import validate_dynamic_lowvol_config
from .o2o_config_validator import O2OConfigValidator
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = get_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        file_ext = os.path.splitext(config_path)[1].lower()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                user_config = yaml.safe_load(f)
            elif file_ext == '.json':
                user_config = json.load(f)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")
        
        # 深度合并配置
        self.config = self._deep_merge(self.config, user_config)
        logger.info(f"配置文件已加载: {config_path}")
    
    def save_config(self, config_path: str):
        """
        保存配置文件
        
        Args:
            config_path: 保存路径
        """
        file_ext = os.path.splitext(config_path)[1].lower()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            elif file_ext == '.json':
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的配置文件格式: {file_ext}")
        
        logger.info(f"配置文件已保存: {config_path}")
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        获取配置
        
        Args:
            section: 配置节名，None则返回完整配置
            
        Returns:
            配置字典
        """
        if section is None:
            return self.config.copy()
        
        return self.config.get(section, {}).copy()
    
    def update_config(self, updates: Dict[str, Any], section: Optional[str] = None):
        """
        更新配置
        
        Args:
            updates: 更新的配置项
            section: 配置节名，None则更新根级配置
        """
        if section is None:
            self.config = self._deep_merge(self.config, updates)
        else:
            if section not in self.config:
                self.config[section] = {}
            self.config[section] = self._deep_merge(self.config[section], updates)
        
        logger.info(f"配置已更新: {section or 'root'}")
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        通过点号分隔的路径获取配置值
        
        Args:
            key_path: 配置键路径，如 'agent.learning_rate'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_value(self, key_path: str, value: Any):
        """
        通过点号分隔的路径设置配置值
        
        Args:
            key_path: 配置键路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到父级字典
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
        logger.debug(f"配置值已设置: {key_path} = {value}")
    
    def validate_config(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
        """
        required_sections = [
            'data', 'factors', 'environment', 'agent', 
            'risk_control', 'backtest'
        ]
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"缺少必需的配置节: {section}")
                return False
        
        # 验证关键参数
        validations = [
            ('agent.learning_rate', lambda x: 0 < x < 1),
            ('environment.max_position', lambda x: 0 < x <= 1),
            ('risk_control.target_volatility', lambda x: 0 < x < 1),
            ('backtest.initial_capital', lambda x: x > 0)
        ]
        
        for key_path, validator in validations:
            value = self.get_value(key_path)
            if value is None or not validator(value):
                logger.error(f"配置值无效: {key_path} = {value}")
                return False
        
        # 验证动态低波筛选器配置（如果存在）
        if 'dynamic_lowvol' in self.config:
            validate_dynamic_lowvol_config(self.config['dynamic_lowvol'])
            logger.info("动态低波筛选器配置验证通过")
        
        # 验证O2O配置（如果存在）
        if 'o2o' in self.config:
            validator = O2OConfigValidator()
            is_valid, error_messages = validator.validate_o2o_config(self.config['o2o'])
            if not is_valid:
                error_str = '; '.join(error_messages)
                logger.error(f"O2O配置错误: {error_str}")
                raise ValueError(f"O2O配置验证失败: {error_str}")
            logger.info("O2O配置验证通过")
        
        logger.info("配置验证通过")
        return True
    
    def create_run_config(self, mode: str = 'default') -> Dict[str, Any]:
        """
        创建运行时配置
        
        Args:
            mode: 运行模式 ('training', 'backtest', 'default')
            
        Returns:
            运行时配置
        """
        run_config = self.config.copy()
        
        if mode == 'training':
            # 训练模式调整
            run_config['data']['end_date'] = '2022-12-31'  # 训练数据截止
            run_config['agent']['learning_rate'] = 5e-4
            run_config['training']['total_episodes'] = 2000
        elif mode == 'backtest':
            # 回测模式调整
            run_config['data']['start_date'] = '2022-01-01'  # 回测数据开始
            run_config['environment']['transaction_cost'] = 0.0015
        
        return run_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """
        深度合并两个字典
        
        Args:
            base: 基础字典
            update: 更新字典
            
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in update.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_data_config(self) -> Dict:
        """获取数据配置"""
        return self.get_config('data')
    
    def get_agent_config(self) -> Dict:
        """获取智能体配置"""
        return self.get_config('agent')
    
    def get_environment_config(self) -> Dict:
        """获取环境配置"""
        return self.get_config('environment')
    
    def get_risk_control_config(self) -> Dict:
        """获取风险控制配置"""
        return self.get_config('risk_control')
    
    def get_backtest_config(self) -> Dict:
        """获取回测配置"""
        return self.get_config('backtest')
    
    def get_training_config(self) -> Dict:
        """获取训练配置"""
        return self.get_config('training')
    
    def get_comfort_config(self) -> Dict:
        """获取心理舒适度配置"""
        return self.get_config('comfort_thresholds')
    
    def get_dynamic_lowvol_config(self) -> Dict:
        """获取动态低波筛选器配置"""
        return self.get_config('dynamic_lowvol')
    
    def get_o2o_config(self) -> Dict:
        """获取O2O配置"""
        return self.get_config('o2o')
    
    def _validate_o2o_config(self, o2o_config: Dict) -> bool:
        """
        验证O2O配置的有效性（已弃用，使用O2OConfigValidator）
        
        Args:
            o2o_config: O2O配置字典
            
        Returns:
            配置是否有效
            
        Raises:
            ValueError: 配置无效时抛出异常
        """
        validator = O2OConfigValidator()
        is_valid, errors = validator.validate_o2o_config(o2o_config)
        
        if not is_valid:
            raise ValueError(f"O2O配置验证失败: {'; '.join(errors)}")
        
        return True
    
    def create_o2o_template(self) -> Dict[str, Any]:
        """
        创建O2O配置模板
        
        Returns:
            O2O配置模板
        """
        return {
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
    
    def migrate_config_to_o2o(self, backup: bool = True) -> bool:
        """
        将现有配置迁移到支持O2O的版本
        
        Args:
            backup: 是否备份原配置
            
        Returns:
            迁移是否成功
        """
        try:
            # 备份原配置
            if backup and self.config_path:
                backup_path = f"{self.config_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"原配置已备份到: {backup_path}")
            
            # 添加O2O配置模板
            o2o_template = self.create_o2o_template()
            self.config = self._deep_merge(self.config, o2o_template)
            
            # 调整现有配置以兼容O2O
            self._adjust_config_for_o2o()
            
            # 保存更新后的配置
            if self.config_path:
                self.save_config(self.config_path)
            
            logger.info("配置已成功迁移到O2O版本")
            return True
            
        except Exception as e:
            logger.error(f"配置迁移失败: {e}")
            raise
    
    def _adjust_config_for_o2o(self):
        """调整现有配置以兼容O2O"""
        # 调整智能体配置
        agent_config = self.config.get('agent', {})
        if 'split_optimizers' not in agent_config:
            agent_config['split_optimizers'] = True
            agent_config['actor_lr'] = agent_config.get('learning_rate', 3e-4)
            agent_config['critic_lr'] = agent_config.get('learning_rate', 1e-3)
        
        # 调整环境配置
        env_config = self.config.get('environment', {})
        if 'mode_switching' not in env_config:
            env_config['mode_switching'] = True
            env_config['trajectory_collection'] = True
            env_config['regime_detection'] = True
        
        # 调整训练配置
        training_config = self.config.get('training', {})
        if 'o2o_enabled' not in training_config:
            training_config['o2o_enabled'] = True
            training_config['stage_management'] = True
            training_config['checkpoint_management'] = True
    
    def update_o2o_config(self, updates: Dict[str, Any]):
        """
        更新O2O配置
        
        Args:
            updates: 更新的配置项
        """
        self.update_config(updates, 'o2o')
        
        # 验证更新后的配置
        if 'o2o' in self.config:
            self._validate_o2o_config(self.config['o2o'])
    
    def get_o2o_stage_config(self, stage: str) -> Dict[str, Any]:
        """
        获取特定O2O阶段的配置
        
        Args:
            stage: 阶段名称 ('offline_pretraining', 'warmup_finetuning', 'online_learning')
            
        Returns:
            阶段配置
        """
        o2o_config = self.get_o2o_config()
        return o2o_config.get(stage, {})
    
    def enable_o2o_hot_update(self) -> bool:
        """
        启用O2O配置热更新功能
        
        Returns:
            是否成功启用
        """
        try:
            # 确保O2O配置存在
            if 'o2o' not in self.config:
                self.config['o2o'] = self.create_o2o_template()['o2o']
            
            # 设置配置热更新标志
            self.config['o2o']['hot_update'] = {
                'enabled': True,
                'watch_config_file': True,
                'auto_reload': True,
                'validation_on_update': True
            }
            
            logger.info("O2O配置热更新已启用")
            return True
            
        except Exception as e:
            logger.error(f"启用O2O配置热更新失败: {e}")
            raise
    
    def get_o2o_validation_report(self) -> str:
        """
        获取O2O配置验证报告
        
        Returns:
            验证报告字符串
        """
        if 'o2o' not in self.config:
            return "未找到O2O配置"
        
        try:
            validator = O2OConfigValidator()
            return validator.generate_validation_report(self.config['o2o'])
        except Exception as e:
            logger.error(f"生成O2O验证报告失败: {e}")
            raise
    
    def get_o2o_optimization_suggestions(self) -> Dict[str, List[str]]:
        """
        获取O2O配置优化建议
        
        Returns:
            优化建议字典
        """
        if 'o2o' not in self.config:
            return {}
        
        try:
            validator = O2OConfigValidator()
            return validator.get_optimization_suggestions(self.config['o2o'])
        except Exception as e:
            logger.error(f"获取O2O优化建议失败: {e}")
            raise
    
    def validate_o2o_config_with_suggestions(self) -> Tuple[bool, List[str], Dict[str, List[str]]]:
        """
        验证O2O配置并返回优化建议
        
        Returns:
            (是否有效, 错误信息列表, 优化建议字典)
        """
        if 'o2o' not in self.config:
            return False, ["未找到O2O配置"], {}
        
        try:
            validator = O2OConfigValidator()
            is_valid, errors = validator.validate_o2o_config(self.config['o2o'])
            suggestions = validator.get_optimization_suggestions(self.config['o2o'])
            return is_valid, errors, suggestions
        except Exception as e:
            logger.error(f"O2O配置验证失败: {e}")
            raise
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("=== 配置摘要 ===")
        
        # 数据配置
        data_config = self.get_data_config()
        print(f"数据源: {data_config.get('provider', 'N/A')}")
        print(f"股票池: {data_config.get('universe', 'N/A')}")
        print(f"数据期间: {data_config.get('start_date', 'N/A')} ~ {data_config.get('end_date', 'N/A')}")
        
        # 智能体配置
        agent_config = self.get_agent_config()
        print(f"学习率: {agent_config.get('learning_rate', 'N/A')}")
        print(f"CVaR阈值: {agent_config.get('cvar_threshold', 'N/A')}")
        
        # 风险控制
        risk_config = self.get_risk_control_config()
        print(f"目标波动率: {risk_config.get('target_volatility', 'N/A'):.1%}")
        print(f"最大杠杆: {risk_config.get('max_leverage', 'N/A')}")
        
        # O2O配置（如果存在）
        if 'o2o' in self.config:
            o2o_config = self.get_o2o_config()
            print(f"O2O模式: {'启用' if o2o_config else '禁用'}")
            if o2o_config:
                print(f"离线预训练轮数: {o2o_config.get('offline_pretraining', {}).get('epochs', 'N/A')}")
                print(f"热身天数: {o2o_config.get('warmup_finetuning', {}).get('days', 'N/A')}")
                print(f"初始在线比例: {o2o_config.get('online_learning', {}).get('initial_rho', 'N/A')}")
        
        print("=" * 20)