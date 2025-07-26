"""
动态低波筛选器配置系统单元测试
"""

import unittest
import copy
from config.dynamic_lowvol_validator import (
    DynamicLowVolConfigValidator,
    validate_dynamic_lowvol_config,
    get_default_dynamic_lowvol_config
)
from config.config_manager import ConfigManager
from risk_control.dynamic_lowvol_filter import ConfigurationException


class TestDynamicLowVolConfigValidator(unittest.TestCase):
    """动态低波筛选器配置验证器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.valid_config = get_default_dynamic_lowvol_config()
    
    def test_valid_default_config(self):
        """测试默认配置是否有效"""
        # 不应该抛出异常
        validate_dynamic_lowvol_config(self.valid_config)
    
    def test_invalid_config_type(self):
        """测试无效的配置类型"""
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config("invalid_config")
        
        self.assertIn("dynamic_lowvol配置必须是字典类型", str(cm.exception))
    
    def test_rolling_percentile_config_validation(self):
        """测试滚动分位筛选配置验证"""
        
        # 测试无效的rolling_windows类型
        config = copy.deepcopy(self.valid_config)
        config['rolling_windows'] = "invalid"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("rolling_windows必须是列表类型", str(cm.exception))
        
        # 测试空的rolling_windows
        config = copy.deepcopy(self.valid_config)
        config['rolling_windows'] = []
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("rolling_windows不能为空", str(cm.exception))
        
        # 测试无效的窗口长度
        config = copy.deepcopy(self.valid_config)
        config['rolling_windows'] = [0, 20]
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("rolling_windows中的窗口长度必须是正整数", str(cm.exception))
        
        # 测试窗口长度过小
        config = copy.deepcopy(self.valid_config)
        config['rolling_windows'] = [3, 20]
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("rolling_windows中的窗口长度不能小于5", str(cm.exception))
        
        # 测试窗口长度过大
        config = copy.deepcopy(self.valid_config)
        config['rolling_windows'] = [20, 600]
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("rolling_windows中的窗口长度不能大于500", str(cm.exception))
        
        # 测试无效的percentile_thresholds类型
        config = copy.deepcopy(self.valid_config)
        config['percentile_thresholds'] = "invalid"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("percentile_thresholds必须是字典类型", str(cm.exception))
        
        # 测试缺少必需的状态
        config = copy.deepcopy(self.valid_config)
        del config['percentile_thresholds']['低']
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("percentile_thresholds缺少必需的状态: 低", str(cm.exception))
        
        # 测试无效的阈值类型
        config = copy.deepcopy(self.valid_config)
        config['percentile_thresholds']['低'] = "invalid"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("percentile_thresholds[低]必须是数值类型", str(cm.exception))
        
        # 测试阈值超出范围
        config = copy.deepcopy(self.valid_config)
        config['percentile_thresholds']['低'] = 1.5
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("percentile_thresholds[低]必须在0.05-0.95范围内", str(cm.exception))
        
        # 测试阈值逻辑关系错误
        config = copy.deepcopy(self.valid_config)
        config['percentile_thresholds']['高'] = 0.5
        config['percentile_thresholds']['中'] = 0.3
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("高波动状态的分位数阈值应该小于中等波动状态", str(cm.exception))
    
    def test_garch_config_validation(self):
        """测试GARCH预测配置验证"""
        
        # 测试无效的garch_window类型
        config = copy.deepcopy(self.valid_config)
        config['garch_window'] = "invalid"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("garch_window必须是正整数", str(cm.exception))
        
        # 测试garch_window过小
        config = copy.deepcopy(self.valid_config)
        config['garch_window'] = 50
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("garch_window不能小于100", str(cm.exception))
        
        # 测试garch_window过大
        config = copy.deepcopy(self.valid_config)
        config['garch_window'] = 1500
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("garch_window不能大于1000", str(cm.exception))
        
        # 测试无效的forecast_horizon
        config = copy.deepcopy(self.valid_config)
        config['forecast_horizon'] = 0
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("forecast_horizon必须是正整数", str(cm.exception))
        
        # 测试forecast_horizon过大
        config = copy.deepcopy(self.valid_config)
        config['forecast_horizon'] = 25
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("forecast_horizon不能大于20", str(cm.exception))
        
        # 测试无效的enable_ml_predictor类型
        config = copy.deepcopy(self.valid_config)
        config['enable_ml_predictor'] = "invalid"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("enable_ml_predictor必须是布尔类型", str(cm.exception))
        
        # 测试无效的ml_predictor_type
        config = copy.deepcopy(self.valid_config)
        config['ml_predictor_type'] = "invalid_model"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("ml_predictor_type必须是['lightgbm', 'lstm']之一", str(cm.exception))
    
    def test_ivol_config_validation(self):
        """测试IVOL约束配置验证"""
        
        # 测试无效的ivol_bad_threshold类型
        config = copy.deepcopy(self.valid_config)
        config['ivol_bad_threshold'] = "invalid"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("ivol_bad_threshold必须是数值类型", str(cm.exception))
        
        # 测试ivol_bad_threshold超出范围
        config = copy.deepcopy(self.valid_config)
        config['ivol_bad_threshold'] = 1.5
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("ivol_bad_threshold必须在0.05-0.95范围内", str(cm.exception))
        
        # 测试阈值逻辑关系错误
        config = copy.deepcopy(self.valid_config)
        config['ivol_bad_threshold'] = 0.7
        config['ivol_good_threshold'] = 0.6
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("ivol_bad_threshold应该小于ivol_good_threshold", str(cm.exception))
        
        # 测试无效的five_factor_window
        config = copy.deepcopy(self.valid_config)
        config['five_factor_window'] = 30
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("five_factor_window不能小于60", str(cm.exception))
    
    def test_regime_config_validation(self):
        """测试市场状态检测配置验证"""
        
        # 测试无效的regime_detection_window
        config = copy.deepcopy(self.valid_config)
        config['regime_detection_window'] = 10
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("regime_detection_window不能小于20", str(cm.exception))
        
        # 测试无效的regime_model_type
        config = copy.deepcopy(self.valid_config)
        config['regime_model_type'] = "INVALID_MODEL"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("regime_model_type必须是['HMM', 'MS-GARCH']之一", str(cm.exception))
        
        # 测试无效的regime_states
        config = copy.deepcopy(self.valid_config)
        config['regime_states'] = 1
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("regime_states不能小于2", str(cm.exception))
        
        # 测试regime_states过大
        config = copy.deepcopy(self.valid_config)
        config['regime_states'] = 10
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("regime_states不能大于5", str(cm.exception))
    
    def test_performance_config_validation(self):
        """测试性能优化配置验证"""
        
        # 测试无效的enable_caching类型
        config = copy.deepcopy(self.valid_config)
        config['enable_caching'] = "invalid"
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("enable_caching必须是布尔类型", str(cm.exception))
        
        # 测试无效的cache_expiry_days
        config = copy.deepcopy(self.valid_config)
        config['cache_expiry_days'] = 50
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("cache_expiry_days不能大于30", str(cm.exception))
        
        # 测试无效的max_workers
        config = copy.deepcopy(self.valid_config)
        config['max_workers'] = 20
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("max_workers不能大于16", str(cm.exception))
    
    def test_data_quality_config_validation(self):
        """测试数据质量配置验证"""
        
        # 测试无效的min_data_length
        config = copy.deepcopy(self.valid_config)
        config['min_data_length'] = 30
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("min_data_length不能小于50", str(cm.exception))
        
        # 测试无效的max_missing_ratio
        config = copy.deepcopy(self.valid_config)
        config['max_missing_ratio'] = 0.8
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("max_missing_ratio必须在0.0-0.5范围内", str(cm.exception))
        
        # 测试无效的outlier_threshold
        config = copy.deepcopy(self.valid_config)
        config['outlier_threshold'] = 15.0
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("outlier_threshold必须在2.0-10.0范围内", str(cm.exception))
    
    def test_convergence_config_validation(self):
        """测试模型收敛配置验证"""
        
        # 测试无效的max_garch_iterations
        config = copy.deepcopy(self.valid_config)
        config['max_garch_iterations'] = 50
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("max_garch_iterations不能小于100", str(cm.exception))
        
        # 测试无效的garch_convergence_tolerance
        config = copy.deepcopy(self.valid_config)
        config['garch_convergence_tolerance'] = 1e-10
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("garch_convergence_tolerance必须在1e-8到1e-3范围内", str(cm.exception))
        
        # 测试无效的hmm_convergence_tolerance
        config = copy.deepcopy(self.valid_config)
        config['hmm_convergence_tolerance'] = 1e-1
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("hmm_convergence_tolerance必须在1e-6到1e-2范围内", str(cm.exception))
        
        # 测试无效的max_hmm_iterations
        config = copy.deepcopy(self.valid_config)
        config['max_hmm_iterations'] = 5
        with self.assertRaises(ConfigurationException) as cm:
            validate_dynamic_lowvol_config(config)
        self.assertIn("max_hmm_iterations不能小于10", str(cm.exception))


class TestConfigManagerIntegration(unittest.TestCase):
    """配置管理器集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config_manager = ConfigManager()
    
    def test_get_dynamic_lowvol_config(self):
        """测试获取动态低波筛选器配置"""
        config = self.config_manager.get_dynamic_lowvol_config()
        
        # 验证配置不为空
        self.assertIsInstance(config, dict)
        self.assertGreater(len(config), 0)
        
        # 验证关键配置项存在
        required_keys = [
            'rolling_windows', 'percentile_thresholds', 'garch_window',
            'forecast_horizon', 'ivol_bad_threshold', 'ivol_good_threshold',
            'regime_detection_window', 'regime_model_type'
        ]
        
        for key in required_keys:
            self.assertIn(key, config, f"缺少必需的配置项: {key}")
    
    def test_config_validation_with_dynamic_lowvol(self):
        """测试包含动态低波筛选器配置的整体配置验证"""
        # 默认配置应该通过验证
        self.assertTrue(self.config_manager.validate_config())
    
    def test_config_validation_with_invalid_dynamic_lowvol(self):
        """测试包含无效动态低波筛选器配置的验证"""
        # 设置无效的配置
        invalid_config = {'rolling_windows': "invalid"}
        self.config_manager.update_config(invalid_config, 'dynamic_lowvol')
        
        # 验证应该失败
        with self.assertRaises(ConfigurationException):
            self.config_manager.validate_config()
    
    def test_update_dynamic_lowvol_config(self):
        """测试更新动态低波筛选器配置"""
        # 更新配置
        updates = {
            'garch_window': 300,
            'forecast_horizon': 3
        }
        self.config_manager.update_config(updates, 'dynamic_lowvol')
        
        # 验证更新成功
        config = self.config_manager.get_dynamic_lowvol_config()
        self.assertEqual(config['garch_window'], 300)
        self.assertEqual(config['forecast_horizon'], 3)
        
        # 验证配置仍然有效
        self.assertTrue(self.config_manager.validate_config())
    
    def test_get_value_from_dynamic_lowvol_config(self):
        """测试通过路径获取动态低波筛选器配置值"""
        # 测试获取嵌套配置值
        value = self.config_manager.get_value('dynamic_lowvol.garch_window')
        self.assertEqual(value, 250)
        
        value = self.config_manager.get_value('dynamic_lowvol.percentile_thresholds.低')
        self.assertEqual(value, 0.4)
        
        # 测试不存在的配置项
        value = self.config_manager.get_value('dynamic_lowvol.non_existent', 'default')
        self.assertEqual(value, 'default')
    
    def test_set_value_in_dynamic_lowvol_config(self):
        """测试通过路径设置动态低波筛选器配置值"""
        # 设置配置值
        self.config_manager.set_value('dynamic_lowvol.garch_window', 400)
        
        # 验证设置成功
        value = self.config_manager.get_value('dynamic_lowvol.garch_window')
        self.assertEqual(value, 400)
        
        # 验证配置仍然有效
        self.assertTrue(self.config_manager.validate_config())


class TestDefaultConfigIntegrity(unittest.TestCase):
    """默认配置完整性测试"""
    
    def test_default_config_has_dynamic_lowvol_section(self):
        """测试默认配置包含动态低波筛选器配置节"""
        from config.default_config import get_default_config
        
        config = get_default_config()
        self.assertIn('dynamic_lowvol', config)
        
        # 验证动态低波筛选器配置有效
        validate_dynamic_lowvol_config(config['dynamic_lowvol'])
    
    def test_default_dynamic_lowvol_config_completeness(self):
        """测试默认动态低波筛选器配置的完整性"""
        config = get_default_dynamic_lowvol_config()
        
        # 验证所有必需的配置节都存在
        required_sections = [
            'rolling_windows', 'percentile_thresholds', 'garch_window',
            'forecast_horizon', 'enable_ml_predictor', 'ml_predictor_type',
            'ivol_bad_threshold', 'ivol_good_threshold', 'five_factor_window',
            'regime_detection_window', 'regime_model_type', 'regime_states',
            'enable_caching', 'cache_expiry_days', 'parallel_processing',
            'max_workers', 'min_data_length', 'max_missing_ratio',
            'outlier_threshold', 'max_garch_iterations',
            'garch_convergence_tolerance', 'hmm_convergence_tolerance',
            'max_hmm_iterations'
        ]
        
        for section in required_sections:
            self.assertIn(section, config, f"默认配置缺少必需的配置项: {section}")
        
        # 验证配置有效性
        validate_dynamic_lowvol_config(config)


if __name__ == '__main__':
    unittest.main()