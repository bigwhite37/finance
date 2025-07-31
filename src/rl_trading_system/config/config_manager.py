"""
配置管理器

实现YAML配置文件加载、环境变量覆盖、配置验证和默认值应用
需求: 10.1
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Callable
from copy import deepcopy
import yaml
from loguru import logger


class ConfigLoadError(Exception):
    """配置加载错误"""
    pass


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigManager:
    """配置管理器
    
    提供YAML配置文件加载、环境变量覆盖、配置验证和默认值应用功能
    """
    
    def __init__(self):
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._file_timestamps: Dict[str, float] = {}
    
    def load_config(self, config_path: Union[str, Path], 
                   enable_env_override: bool = True,
                   use_cache: bool = False) -> Dict[str, Any]:
        """加载单个配置文件
        
        Args:
            config_path: 配置文件路径
            enable_env_override: 是否启用环境变量覆盖
            use_cache: 是否使用缓存
            
        Returns:
            配置字典
            
        Raises:
            ConfigLoadError: 配置加载失败
        """
        config_path = Path(config_path)
        cache_key = str(config_path.absolute())
        
        # 检查缓存
        if use_cache and self._is_cache_valid(cache_key, config_path):
            logger.debug(f"使用缓存配置: {config_path}")
            return deepcopy(self._config_cache[cache_key])
        
        try:
            # 检查文件是否存在
            if not config_path.exists():
                raise ConfigLoadError(f"配置文件不存在: {config_path}")
            
            # 读取YAML文件
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            logger.info(f"成功加载配置文件: {config_path}")
            
            # 应用环境变量覆盖
            if enable_env_override:
                config = self._apply_env_overrides(config)
            
            # 更新缓存
            if use_cache:
                self._config_cache[cache_key] = deepcopy(config)
                self._file_timestamps[cache_key] = config_path.stat().st_mtime
            
            return config
            
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML解析错误: {e}")
        except (OSError, IOError) as e:
            raise ConfigLoadError(f"文件读取错误: {e}")
    
    def load_configs(self, config_paths: List[Union[str, Path]], 
                    enable_env_override: bool = True,
                    use_cache: bool = False) -> Dict[str, Any]:
        """加载多个配置文件并合并
        
        Args:
            config_paths: 配置文件路径列表
            enable_env_override: 是否启用环境变量覆盖
            use_cache: 是否使用缓存
            
        Returns:
            合并后的配置字典
        """
        merged_config = {}
        
        for config_path in config_paths:
            config = self.load_config(config_path, enable_env_override, use_cache)
            merged_config = self._deep_merge(merged_config, config)
        
        return merged_config
    
    def validate_config(self, config: Dict[str, Any], 
                       schema: Dict[str, Any]) -> None:
        """验证配置
        
        Args:
            config: 配置字典
            schema: 验证模式
            
        Raises:
            ConfigValidationError: 配置验证失败
        """
        self._validate_dict(config, schema, "")
    
    def apply_defaults(self, config: Dict[str, Any], 
                      schema: Dict[str, Any]) -> Dict[str, Any]:
        """应用默认值
        
        Args:
            config: 配置字典
            schema: 包含默认值的模式
            
        Returns:
            应用默认值后的配置字典
        """
        result = deepcopy(config)
        self._apply_defaults_recursive(result, schema)
        return result
    
    def load_and_validate_config(self, config_path: Union[str, Path],
                                schema: Dict[str, Any],
                                enable_env_override: bool = True,
                                use_cache: bool = False) -> Dict[str, Any]:
        """加载、验证并应用默认值的完整流程
        
        Args:
            config_path: 配置文件路径
            schema: 验证模式
            enable_env_override: 是否启用环境变量覆盖
            use_cache: 是否使用缓存
            
        Returns:
            处理后的配置字典
        """
        # 加载配置
        config = self.load_config(config_path, enable_env_override, use_cache)
        
        # 应用默认值
        config = self.apply_defaults(config, schema)
        
        # 验证配置
        self.validate_config(config, schema)
        
        return config
    
    def reload_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """强制重新加载配置（忽略缓存）
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            重新加载的配置字典
        """
        cache_key = str(Path(config_path).absolute())
        
        # 清除缓存
        if cache_key in self._config_cache:
            del self._config_cache[cache_key]
        if cache_key in self._file_timestamps:
            del self._file_timestamps[cache_key]
        
        return self.load_config(config_path, use_cache=True)
    
    def _is_cache_valid(self, cache_key: str, config_path: Path) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self._config_cache:
            return False
        
        if cache_key not in self._file_timestamps:
            return False
        
        try:
            current_mtime = config_path.stat().st_mtime
            cached_mtime = self._file_timestamps[cache_key]
            return current_mtime <= cached_mtime
        except OSError:
            return False
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖
        
        环境变量命名规则：
        - 使用下划线分隔嵌套层级
        - 全部大写
        - 例如：MODEL_TRANSFORMER_D_MODEL 对应 config['model']['transformer']['d_model']
        """
        result = deepcopy(config)
        
        # 获取所有相关的环境变量
        env_overrides = {}
        for key, value in os.environ.items():
            if self._is_config_env_var(key):
                env_overrides[key] = value
        
        # 应用环境变量覆盖
        for env_key, env_value in env_overrides.items():
            self._apply_single_env_override(result, env_key, env_value)
        
        return result
    
    def _is_config_env_var(self, env_key: str) -> bool:
        """判断是否是配置相关的环境变量"""
        # 简单的启发式规则：包含常见的配置前缀
        config_prefixes = ['MODEL_', 'TRADING_', 'DATA_', 'MONITORING_', 'COMPLIANCE_', 'FEATURE_']
        return any(env_key.startswith(prefix) for prefix in config_prefixes)
    
    def _apply_single_env_override(self, config: Dict[str, Any], 
                                  env_key: str, env_value: str) -> None:
        """应用单个环境变量覆盖"""
        # 将环境变量键转换为配置路径
        # 需要智能处理下划线，避免将 D_MODEL 拆分成 D 和 MODEL
        path_parts = self._parse_env_key_path(env_key)
        
        # 导航到目标位置
        current = config
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # 如果不是字典，创建新的字典
                current[part] = {}
            current = current[part]
        
        # 获取原始值的类型（如果存在）
        final_key = path_parts[-1]
        original_type = type(current.get(final_key)) if final_key in current else str
        
        # 转换环境变量值
        converted_value = self._convert_env_value(env_value, original_type)
        
        # 设置值
        current[final_key] = converted_value
        
        logger.debug(f"环境变量覆盖: {env_key} = {converted_value}")
    
    def _parse_env_key_path(self, env_key: str) -> List[str]:
        """解析环境变量键为配置路径
        
        使用预定义的映射来正确解析环境变量路径
        """
        # 转换为小写
        key_lower = env_key.lower()
        
        # 预定义的环境变量到配置路径的映射
        env_mappings = {
            # Model config mappings
            'model_transformer_d_model': ['model', 'transformer', 'd_model'],
            'model_transformer_n_heads': ['model', 'transformer', 'n_heads'],
            'model_transformer_n_layers': ['model', 'transformer', 'n_layers'],
            'model_transformer_d_ff': ['model', 'transformer', 'd_ff'],
            'model_transformer_dropout': ['model', 'transformer', 'dropout'],
            'model_transformer_max_seq_len': ['model', 'transformer', 'max_seq_len'],
            'model_transformer_n_features': ['model', 'transformer', 'n_features'],
            'model_sac_state_dim': ['model', 'sac', 'state_dim'],
            'model_sac_action_dim': ['model', 'sac', 'action_dim'],
            'model_sac_hidden_dim': ['model', 'sac', 'hidden_dim'],
            'model_sac_lr_actor': ['model', 'sac', 'lr_actor'],
            'model_sac_lr_critic': ['model', 'sac', 'lr_critic'],
            'model_sac_lr_alpha': ['model', 'sac', 'lr_alpha'],
            'model_sac_gamma': ['model', 'sac', 'gamma'],
            'model_sac_tau': ['model', 'sac', 'tau'],
            'model_sac_alpha': ['model', 'sac', 'alpha'],
            'model_sac_target_entropy': ['model', 'sac', 'target_entropy'],
            'model_sac_buffer_size': ['model', 'sac', 'buffer_size'],
            'model_sac_batch_size': ['model', 'sac', 'batch_size'],
            
            # Trading config mappings
            'trading_environment_stock_pool': ['trading', 'environment', 'stock_pool'],
            'trading_environment_lookback_window': ['trading', 'environment', 'lookback_window'],
            'trading_environment_initial_cash': ['trading', 'environment', 'initial_cash'],
            'trading_environment_commission_rate': ['trading', 'environment', 'commission_rate'],
            'trading_environment_stamp_tax_rate': ['trading', 'environment', 'stamp_tax_rate'],
            'trading_environment_risk_aversion': ['trading', 'environment', 'risk_aversion'],
            'trading_environment_max_drawdown_penalty': ['trading', 'environment', 'max_drawdown_penalty'],
            
            # Feature config mappings
            'feature_enabled': ['feature', 'enabled'],
            'feature_debug': ['feature', 'debug'],
            
            # Stock pool mapping
            'trading_stock_pool': ['trading', 'stock_pool']
        }
        
        # 检查是否有预定义的映射
        if key_lower in env_mappings:
            return env_mappings[key_lower]
        
        # 如果没有预定义映射，使用简单的下划线分割
        return key_lower.split('_')
    
    def _convert_env_value(self, env_value: str, target_type: type) -> Any:
        """转换环境变量值到目标类型"""
        if target_type == bool:
            return env_value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            try:
                return int(env_value)
            except ValueError:
                return int(float(env_value))  # 处理 "1.0" 这样的情况
        elif target_type == float:
            return float(env_value)
        elif target_type == list:
            return [item.strip() for item in env_value.split(',')]
        else:
            return env_value
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典"""
        result = deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _validate_dict(self, config: Dict[str, Any], 
                      schema: Dict[str, Any], path: str) -> None:
        """递归验证字典"""
        for key, field_schema in schema.items():
            current_path = f"{path}.{key}" if path else key
            
            # 检查必需字段
            if field_schema.get('required', False) and key not in config:
                raise ConfigValidationError(f"必需字段缺失: {current_path}")
            
            if key not in config:
                continue
            
            value = config[key]
            
            # 类型检查
            expected_type = field_schema.get('type')
            if expected_type and not isinstance(value, expected_type):
                raise ConfigValidationError(
                    f"类型错误: {current_path}, 期望 {expected_type.__name__}, 实际 {type(value).__name__}"
                )
            
            # 范围检查
            if 'min' in field_schema and value < field_schema['min']:
                raise ConfigValidationError(f"值超出范围: {current_path} < {field_schema['min']}")
            
            if 'max' in field_schema and value > field_schema['max']:
                raise ConfigValidationError(f"值超出范围: {current_path} > {field_schema['max']}")
            
            # 允许值检查
            if 'allowed' in field_schema and value not in field_schema['allowed']:
                raise ConfigValidationError(
                    f"不允许的值: {current_path} = {value}, 允许的值: {field_schema['allowed']}"
                )
            
            # 自定义验证器
            if 'validator' in field_schema:
                validator = field_schema['validator']
                try:
                    validator(current_path, value, self._validation_error)
                except ConfigValidationError:
                    raise
                except Exception as e:
                    raise ConfigValidationError(f"验证器错误: {current_path}, {e}")
            
            # 递归验证嵌套字典
            if isinstance(value, dict) and 'schema' in field_schema:
                self._validate_dict(value, field_schema['schema'], current_path)
    
    def _validation_error(self, field: str, message: str) -> None:
        """验证错误回调"""
        raise ConfigValidationError(f"{field}: {message}")
    
    def _apply_defaults_recursive(self, config: Dict[str, Any], 
                                 schema: Dict[str, Any]) -> None:
        """递归应用默认值"""
        for key, field_schema in schema.items():
            # 如果字段不存在且有默认值，则应用默认值
            if key not in config and 'default' in field_schema:
                config[key] = deepcopy(field_schema['default'])
            
            # 递归处理嵌套字典
            if (key in config and isinstance(config[key], dict) and 
                'schema' in field_schema):
                self._apply_defaults_recursive(config[key], field_schema['schema'])


# 全局配置管理器实例
config_manager = ConfigManager()


def load_config(config_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """便捷函数：加载配置"""
    return config_manager.load_config(config_path, **kwargs)


def load_configs(config_paths: List[Union[str, Path]], **kwargs) -> Dict[str, Any]:
    """便捷函数：加载多个配置"""
    return config_manager.load_configs(config_paths, **kwargs)


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """便捷函数：验证配置"""
    return config_manager.validate_config(config, schema)