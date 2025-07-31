"""
配置管理器测试用例

测试YAML配置文件加载和验证、环境变量覆盖机制、配置参数类型检查和默认值
需求: 10.1
"""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, mock_open

import yaml

from src.rl_trading_system.config.config_manager import (
    ConfigManager,
    ConfigValidationError,
    ConfigLoadError
)


# 全局fixtures
@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """示例配置字典"""
    return {
        'model': {
            'transformer': {
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 6,
                'dropout': 0.1,
                'max_seq_len': 252
            },
            'sac': {
                'lr_actor': 3e-4,
                'lr_critic': 3e-4,
                'gamma': 0.99,
                'buffer_size': 1000000
            }
        },
        'trading': {
            'environment': {
                'initial_cash': 1000000.0,
                'commission_rate': 0.001,
                'lookback_window': 60
            }
        }
    }

@pytest.fixture
def sample_yaml_content(sample_config_dict) -> str:
    """示例YAML配置内容"""
    return yaml.dump(sample_config_dict, default_flow_style=False)

@pytest.fixture
def temp_config_file(sample_yaml_content) -> Path:
    """创建临时配置文件"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sample_yaml_content)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # 清理临时文件
    if temp_path.exists():
        temp_path.unlink()

@pytest.fixture
def config_manager() -> ConfigManager:
    """配置管理器实例"""
    return ConfigManager()


class TestConfigManager:
    """配置管理器测试类"""
    pass


class TestConfigLoading:
    """配置加载测试"""
    
    def test_load_yaml_file_success(self, config_manager, temp_config_file, sample_config_dict):
        """测试成功加载YAML文件"""
        config = config_manager.load_config(temp_config_file)
        
        assert config == sample_config_dict
        assert config['model']['transformer']['d_model'] == 256
        assert config['trading']['environment']['initial_cash'] == 1000000.0
    
    def test_load_nonexistent_file(self, config_manager):
        """测试加载不存在的文件"""
        with pytest.raises(ConfigLoadError, match="配置文件不存在"):
            config_manager.load_config("nonexistent.yaml")
    
    def test_load_invalid_yaml(self, config_manager):
        """测试加载无效的YAML文件"""
        invalid_yaml = "invalid: yaml: content: ["
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ConfigLoadError, match="YAML解析错误"):
                config_manager.load_config(temp_path)
        finally:
            temp_path.unlink()
    
    def test_load_empty_file(self, config_manager):
        """测试加载空文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = Path(f.name)
        
        try:
            config = config_manager.load_config(temp_path)
            assert config == {}
        finally:
            temp_path.unlink()
    
    def test_load_multiple_configs(self, config_manager, sample_config_dict):
        """测试加载多个配置文件并合并"""
        config1 = {'model': {'transformer': {'d_model': 256}}}
        config2 = {'trading': {'environment': {'initial_cash': 1000000.0}}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1:
            yaml.dump(config1, f1)
            temp_path1 = Path(f1.name)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            yaml.dump(config2, f2)
            temp_path2 = Path(f2.name)
        
        try:
            merged_config = config_manager.load_configs([temp_path1, temp_path2])
            
            assert 'model' in merged_config
            assert 'trading' in merged_config
            assert merged_config['model']['transformer']['d_model'] == 256
            assert merged_config['trading']['environment']['initial_cash'] == 1000000.0
        finally:
            temp_path1.unlink()
            temp_path2.unlink()


class TestEnvironmentVariableOverride:
    """环境变量覆盖测试"""
    
    def test_simple_env_override(self, config_manager, temp_config_file):
        """测试简单环境变量覆盖"""
        with patch.dict(os.environ, {'MODEL_TRANSFORMER_D_MODEL': '512'}):
            config = config_manager.load_config(temp_config_file, enable_env_override=True)
            
            assert config['model']['transformer']['d_model'] == 512
    
    def test_nested_env_override(self, config_manager, temp_config_file):
        """测试嵌套路径环境变量覆盖"""
        env_vars = {
            'TRADING_ENVIRONMENT_INITIAL_CASH': '2000000.0',
            'MODEL_SAC_LR_ACTOR': '1e-3'
        }
        
        with patch.dict(os.environ, env_vars):
            config = config_manager.load_config(temp_config_file, enable_env_override=True)
            
            assert config['trading']['environment']['initial_cash'] == 2000000.0
            assert config['model']['sac']['lr_actor'] == 1e-3
    
    def test_env_override_type_conversion(self, config_manager, temp_config_file):
        """测试环境变量类型转换"""
        env_vars = {
            'MODEL_TRANSFORMER_N_HEADS': '16',  # int
            'MODEL_TRANSFORMER_DROPOUT': '0.2',  # float
            'TRADING_ENVIRONMENT_COMMISSION_RATE': '0.002'  # float
        }
        
        with patch.dict(os.environ, env_vars):
            config = config_manager.load_config(temp_config_file, enable_env_override=True)
            
            assert config['model']['transformer']['n_heads'] == 16
            assert isinstance(config['model']['transformer']['n_heads'], int)
            assert config['model']['transformer']['dropout'] == 0.2
            assert isinstance(config['model']['transformer']['dropout'], float)
            assert config['trading']['environment']['commission_rate'] == 0.002
    
    def test_env_override_boolean_values(self, config_manager):
        """测试布尔值环境变量覆盖"""
        config_dict = {'feature': {'enabled': False, 'debug': True}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = Path(f.name)
        
        env_vars = {
            'FEATURE_ENABLED': 'true',
            'FEATURE_DEBUG': 'false'
        }
        
        try:
            with patch.dict(os.environ, env_vars):
                config = config_manager.load_config(temp_path, enable_env_override=True)
                
                assert config['feature']['enabled'] is True
                assert config['feature']['debug'] is False
        finally:
            temp_path.unlink()
    
    def test_env_override_list_values(self, config_manager):
        """测试列表值环境变量覆盖"""
        config_dict = {'trading': {'stock_pool': ['000001.SZ', '000002.SZ']}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = Path(f.name)
        
        try:
            with patch.dict(os.environ, {'TRADING_STOCK_POOL': '000001.SZ,000002.SZ,000003.SZ'}):
                config = config_manager.load_config(temp_path, enable_env_override=True)
                
                expected_pool = ['000001.SZ', '000002.SZ', '000003.SZ']
                assert config['trading']['stock_pool'] == expected_pool
        finally:
            temp_path.unlink()
    
    def test_env_override_disabled(self, config_manager, temp_config_file):
        """测试禁用环境变量覆盖"""
        with patch.dict(os.environ, {'MODEL_TRANSFORMER_D_MODEL': '512'}):
            config = config_manager.load_config(temp_config_file, enable_env_override=False)
            
            # 应该保持原始值
            assert config['model']['transformer']['d_model'] == 256


class TestConfigValidation:
    """配置验证测试"""
    
    def test_validate_required_fields(self, config_manager):
        """测试必需字段验证"""
        schema = {
            'model': {
                'required': True,
                'type': dict,
                'schema': {
                    'transformer': {
                        'required': True,
                        'type': dict,
                        'schema': {
                            'd_model': {'required': True, 'type': int, 'min': 1}
                        }
                    }
                }
            }
        }
        
        # 缺少必需字段
        invalid_config = {'model': {'transformer': {}}}
        
        with pytest.raises(ConfigValidationError, match="必需字段缺失"):
            config_manager.validate_config(invalid_config, schema)
    
    def test_validate_type_checking(self, config_manager):
        """测试类型检查"""
        schema = {
            'model': {
                'type': dict,
                'schema': {
                    'lr': {'type': float, 'min': 0.0, 'max': 1.0},
                    'epochs': {'type': int, 'min': 1},
                    'enabled': {'type': bool}
                }
            }
        }
        
        # 类型错误的配置
        invalid_configs = [
            {'model': {'lr': 'invalid'}},  # 应该是float
            {'model': {'epochs': 0.5}},    # 应该是int
            {'model': {'enabled': 'yes'}}  # 应该是bool
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ConfigValidationError, match="类型错误"):
                config_manager.validate_config(invalid_config, schema)
    
    def test_validate_range_constraints(self, config_manager):
        """测试范围约束验证"""
        schema = {
            'model': {
                'type': dict,
                'schema': {
                    'lr': {'type': float, 'min': 0.0, 'max': 1.0},
                    'batch_size': {'type': int, 'min': 1, 'max': 1024}
                }
            }
        }
        
        # 超出范围的配置
        invalid_configs = [
            {'model': {'lr': -0.1}},      # 小于最小值
            {'model': {'lr': 1.5}},       # 大于最大值
            {'model': {'batch_size': 0}}, # 小于最小值
            {'model': {'batch_size': 2048}} # 大于最大值
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ConfigValidationError, match="值超出范围"):
                config_manager.validate_config(invalid_config, schema)
    
    def test_validate_allowed_values(self, config_manager):
        """测试允许值验证"""
        schema = {
            'data': {
                'type': dict,
                'schema': {
                    'provider': {'type': str, 'allowed': ['qlib', 'akshare']},
                    'freq': {'type': str, 'allowed': ['1d', '1h', '1min']}
                }
            }
        }
        
        # 不允许的值
        invalid_configs = [
            {'data': {'provider': 'yahoo'}},
            {'data': {'freq': '5min'}}
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ConfigValidationError, match="不允许的值"):
                config_manager.validate_config(invalid_config, schema)
    
    def test_validate_custom_validator(self, config_manager):
        """测试自定义验证器"""
        def validate_stock_code(field, value, error):
            """验证股票代码格式"""
            if not isinstance(value, str) or len(value) != 9:
                error(field, "股票代码格式错误")
            if not (value.endswith('.SZ') or value.endswith('.SH')):
                error(field, "股票代码必须以.SZ或.SH结尾")
        
        schema = {
            'trading': {
                'type': dict,
                'schema': {
                    'benchmark': {
                        'type': str,
                        'validator': validate_stock_code
                    }
                }
            }
        }
        
        # 无效的股票代码
        invalid_configs = [
            {'trading': {'benchmark': '000300'}},      # 长度不够
            {'trading': {'benchmark': '000300.XX'}},   # 错误后缀
            {'trading': {'benchmark': 123}}            # 错误类型
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ConfigValidationError):
                config_manager.validate_config(invalid_config, schema)


class TestDefaultValues:
    """默认值测试"""
    
    def test_apply_default_values(self, config_manager):
        """测试应用默认值"""
        schema = {
            'model': {
                'type': dict,
                'default': {},
                'schema': {
                    'lr': {'type': float, 'default': 1e-3},
                    'batch_size': {'type': int, 'default': 32},
                    'enabled': {'type': bool, 'default': True}
                }
            }
        }
        
        # 空配置
        config = {}
        filled_config = config_manager.apply_defaults(config, schema)
        
        expected = {
            'model': {
                'lr': 1e-3,
                'batch_size': 32,
                'enabled': True
            }
        }
        
        assert filled_config == expected
    
    def test_partial_default_application(self, config_manager):
        """测试部分默认值应用"""
        schema = {
            'model': {
                'type': dict,
                'schema': {
                    'lr': {'type': float, 'default': 1e-3},
                    'batch_size': {'type': int, 'default': 32},
                    'epochs': {'type': int, 'default': 100}
                }
            }
        }
        
        # 部分配置
        config = {'model': {'lr': 2e-3, 'epochs': 200}}
        filled_config = config_manager.apply_defaults(config, schema)
        
        expected = {
            'model': {
                'lr': 2e-3,      # 保持用户设置的值
                'batch_size': 32, # 应用默认值
                'epochs': 200     # 保持用户设置的值
            }
        }
        
        assert filled_config == expected
    
    def test_nested_default_values(self, config_manager):
        """测试嵌套默认值"""
        schema = {
            'model': {
                'type': dict,
                'default': {},
                'schema': {
                    'transformer': {
                        'type': dict,
                        'default': {},
                        'schema': {
                            'd_model': {'type': int, 'default': 256},
                            'n_heads': {'type': int, 'default': 8}
                        }
                    }
                }
            }
        }
        
        config = {}
        filled_config = config_manager.apply_defaults(config, schema)
        
        expected = {
            'model': {
                'transformer': {
                    'd_model': 256,
                    'n_heads': 8
                }
            }
        }
        
        assert filled_config == expected


class TestConfigManagerIntegration:
    """配置管理器集成测试"""
    
    def test_full_workflow(self, config_manager, sample_config_dict):
        """测试完整工作流程"""
        # 创建配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config_dict, f)
            temp_path = Path(f.name)
        
        # 定义验证模式
        schema = {
            'model': {
                'required': True,
                'type': dict,
                'schema': {
                    'transformer': {
                        'type': dict,
                        'schema': {
                            'd_model': {'type': int, 'min': 1, 'default': 256}
                        }
                    }
                }
            }
        }
        
        try:
            # 设置环境变量
            with patch.dict(os.environ, {'MODEL_TRANSFORMER_D_MODEL': '512'}):
                # 加载、验证和应用默认值
                config = config_manager.load_and_validate_config(
                    temp_path, 
                    schema, 
                    enable_env_override=True
                )
                
                # 验证结果
                assert config['model']['transformer']['d_model'] == 512  # 环境变量覆盖
                assert 'sac' in config['model']  # 原始配置保留
                
        finally:
            temp_path.unlink()
    
    def test_config_caching(self, config_manager, temp_config_file):
        """测试配置缓存"""
        # 首次加载
        config1 = config_manager.load_config(temp_config_file, use_cache=True)
        
        # 再次加载（应该使用缓存，但返回深拷贝以确保安全）
        config2 = config_manager.load_config(temp_config_file, use_cache=True)
        
        assert config1 is not config2  # 应该是不同对象（深拷贝）
        assert config1 == config2      # 但内容相同
        
        # 禁用缓存
        config3 = config_manager.load_config(temp_config_file, use_cache=False)
        
        assert config1 is not config3  # 应该是不同对象
        assert config1 == config3      # 但内容相同
    
    def test_config_reload(self, config_manager, sample_config_dict):
        """测试配置重新加载"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_config_dict, f)
            temp_path = Path(f.name)
        
        try:
            # 首次加载
            config1 = config_manager.load_config(temp_path, use_cache=True)
            
            # 修改文件
            modified_config = sample_config_dict.copy()
            modified_config['model']['transformer']['d_model'] = 512
            
            with open(temp_path, 'w') as f:
                yaml.dump(modified_config, f)
            
            # 强制重新加载
            config2 = config_manager.reload_config(temp_path)
            
            assert config1['model']['transformer']['d_model'] == 256
            assert config2['model']['transformer']['d_model'] == 512
            
        finally:
            temp_path.unlink()


@pytest.mark.parametrize("env_value,expected_type,expected_value", [
    ("123", int, 123),
    ("123.45", float, 123.45),
    ("true", bool, True),
    ("false", bool, False),
    ("hello", str, "hello"),
    ("1,2,3", list, ["1", "2", "3"]),
])
def test_env_value_type_conversion(env_value, expected_type, expected_value):
    """参数化测试环境变量类型转换"""
    from src.rl_trading_system.config.config_manager import ConfigManager
    
    manager = ConfigManager()
    converted = manager._convert_env_value(env_value, expected_type)
    
    assert type(converted) == expected_type
    assert converted == expected_value


class TestErrorHandling:
    """错误处理测试"""
    
    def test_file_permission_error(self, config_manager):
        """测试文件权限错误"""
        # 创建一个无权限读取的文件（在支持的系统上）
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # 尝试移除读权限
            temp_path.chmod(0o000)
            
            with pytest.raises(ConfigLoadError):
                config_manager.load_config(temp_path)
                
        except (OSError, PermissionError):
            # 如果无法设置权限，跳过此测试
            pytest.skip("无法设置文件权限")
        finally:
            # 恢复权限并删除文件
            try:
                temp_path.chmod(0o644)
                temp_path.unlink()
            except (OSError, PermissionError):
                pass
    
    def test_circular_reference_detection(self, config_manager):
        """测试循环引用检测"""
        # 这个测试可能需要在实际实现中根据具体的引用机制来调整
        pass