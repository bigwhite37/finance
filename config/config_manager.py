"""
配置管理器
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
import logging
from .default_config import get_default_config

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
        
        print("=" * 20)