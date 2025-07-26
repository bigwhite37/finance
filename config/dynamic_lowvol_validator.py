"""
动态低波筛选器配置验证模块
"""

from typing import Dict, Any, List
from risk_control.dynamic_lowvol_filter.exceptions import ConfigurationException


class DynamicLowVolConfigValidator:
    """动态低波筛选器配置验证器"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        验证动态低波筛选器配置
        
        Args:
            config: 动态低波筛选器配置字典
            
        Raises:
            ConfigurationException: 配置参数不正确时抛出
        """
        if not isinstance(config, dict):
            raise ConfigurationException("dynamic_lowvol配置必须是字典类型")
        
        # 首先验证必需参数
        DynamicLowVolConfigValidator._validate_required_parameters(config)
        
        # 验证滚动分位筛选配置
        DynamicLowVolConfigValidator._validate_rolling_percentile_config(config)
        
        # 验证GARCH预测配置
        DynamicLowVolConfigValidator._validate_garch_config(config)
        
        # 验证IVOL约束配置
        DynamicLowVolConfigValidator._validate_ivol_config(config)
        
        # 验证市场状态检测配置
        DynamicLowVolConfigValidator._validate_regime_config(config)
        
        # 验证性能优化配置
        DynamicLowVolConfigValidator._validate_performance_config(config)
        
        # 验证数据质量配置
        DynamicLowVolConfigValidator._validate_data_quality_config(config)
        
        # 验证模型收敛配置
        DynamicLowVolConfigValidator._validate_convergence_config(config)
    
    @staticmethod
    def _validate_required_parameters(config: Dict[str, Any]) -> None:
        """验证必需参数"""
        required_params = [
            'rolling_windows',
            'percentile_thresholds'
        ]
        
        for param in required_params:
            if param not in config:
                raise ConfigurationException(f"缺少必需参数: {param}")
    
    @staticmethod
    def _validate_rolling_percentile_config(config: Dict[str, Any]) -> None:
        """验证滚动分位筛选配置"""
        # 验证rolling_windows
        rolling_windows = config['rolling_windows']
        if not isinstance(rolling_windows, list):
            raise ConfigurationException("rolling_windows必须是列表类型")
        
        if len(rolling_windows) == 0:
            raise ConfigurationException("rolling_windows不能为空")
        
        for window in rolling_windows:
            if not isinstance(window, int) or window <= 0:
                raise ConfigurationException(f"rolling_windows中的窗口长度必须是正整数，得到: {window}")
            
            if window < 5:
                raise ConfigurationException(f"rolling_windows中的窗口长度不能小于5，得到: {window}")
            
            if window > 500:
                raise ConfigurationException(f"rolling_windows中的窗口长度不能大于500，得到: {window}")
        
        # 验证percentile_thresholds
        percentile_thresholds = config['percentile_thresholds']
        if not isinstance(percentile_thresholds, dict):
            raise ConfigurationException("percentile_thresholds必须是字典类型")
        
        required_regimes = ['低', '中', '高']
        for regime in required_regimes:
            if regime not in percentile_thresholds:
                raise ConfigurationException(f"percentile_thresholds缺少必需的状态: {regime}")
            
            threshold = percentile_thresholds[regime]
            if not isinstance(threshold, (int, float)):
                raise ConfigurationException(f"percentile_thresholds[{regime}]必须是数值类型，得到: {threshold}")
            
            if not 0.05 <= threshold <= 0.95:
                raise ConfigurationException(f"percentile_thresholds[{regime}]必须在0.05-0.95范围内，得到: {threshold}")
        
        # 验证阈值的逻辑关系：高波动状态应该有更严格的阈值
        if percentile_thresholds['高'] >= percentile_thresholds['中']:
            raise ConfigurationException("高波动状态的分位数阈值应该小于中等波动状态")
        
        if percentile_thresholds['中'] >= percentile_thresholds['低']:
            raise ConfigurationException("中等波动状态的分位数阈值应该小于低波动状态")
    
    @staticmethod
    def _validate_garch_config(config: Dict[str, Any]) -> None:
        """验证GARCH预测配置"""
        # 验证garch_window
        garch_window = config.get('garch_window', 250)
        if not isinstance(garch_window, int) or garch_window <= 0:
            raise ConfigurationException(f"garch_window必须是正整数，得到: {garch_window}")
        
        if garch_window < 100:
            raise ConfigurationException(f"garch_window不能小于100，得到: {garch_window}")
        
        if garch_window > 1000:
            raise ConfigurationException(f"garch_window不能大于1000，得到: {garch_window}")
        
        # 验证forecast_horizon
        forecast_horizon = config.get('forecast_horizon', 5)
        if not isinstance(forecast_horizon, int) or forecast_horizon <= 0:
            raise ConfigurationException(f"forecast_horizon必须是正整数，得到: {forecast_horizon}")
        
        if forecast_horizon > 20:
            raise ConfigurationException(f"forecast_horizon不能大于20，得到: {forecast_horizon}")
        
        # 验证enable_ml_predictor
        enable_ml_predictor = config.get('enable_ml_predictor', False)
        if not isinstance(enable_ml_predictor, bool):
            raise ConfigurationException(f"enable_ml_predictor必须是布尔类型，得到: {enable_ml_predictor}")
        
        # 验证ml_predictor_type
        ml_predictor_type = config.get('ml_predictor_type', 'lightgbm')
        if not isinstance(ml_predictor_type, str):
            raise ConfigurationException(f"ml_predictor_type必须是字符串类型，得到: {ml_predictor_type}")
        
        valid_ml_types = ['lightgbm', 'lstm']
        if ml_predictor_type not in valid_ml_types:
            raise ConfigurationException(f"ml_predictor_type必须是{valid_ml_types}之一，得到: {ml_predictor_type}")
    
    @staticmethod
    def _validate_ivol_config(config: Dict[str, Any]) -> None:
        """验证IVOL约束配置"""
        # 验证ivol_bad_threshold
        ivol_bad_threshold = config.get('ivol_bad_threshold', 0.3)
        if not isinstance(ivol_bad_threshold, (int, float)):
            raise ConfigurationException(f"ivol_bad_threshold必须是数值类型，得到: {ivol_bad_threshold}")
        
        if not 0.05 <= ivol_bad_threshold <= 0.95:
            raise ConfigurationException(f"ivol_bad_threshold必须在0.05-0.95范围内，得到: {ivol_bad_threshold}")
        
        # 验证ivol_good_threshold
        ivol_good_threshold = config.get('ivol_good_threshold', 0.6)
        if not isinstance(ivol_good_threshold, (int, float)):
            raise ConfigurationException(f"ivol_good_threshold必须是数值类型，得到: {ivol_good_threshold}")
        
        if not 0.05 <= ivol_good_threshold <= 0.95:
            raise ConfigurationException(f"ivol_good_threshold必须在0.05-0.95范围内，得到: {ivol_good_threshold}")
        
        # 验证阈值逻辑关系
        if ivol_bad_threshold >= ivol_good_threshold:
            raise ConfigurationException("ivol_bad_threshold应该小于ivol_good_threshold")
        
        # 验证five_factor_window
        five_factor_window = config.get('five_factor_window', 252)
        if not isinstance(five_factor_window, int) or five_factor_window <= 0:
            raise ConfigurationException(f"five_factor_window必须是正整数，得到: {five_factor_window}")
        
        if five_factor_window < 60:
            raise ConfigurationException(f"five_factor_window不能小于60，得到: {five_factor_window}")
        
        if five_factor_window > 500:
            raise ConfigurationException(f"five_factor_window不能大于500，得到: {five_factor_window}")
    
    @staticmethod
    def _validate_regime_config(config: Dict[str, Any]) -> None:
        """验证市场状态检测配置"""
        # 验证regime_detection_window
        regime_detection_window = config.get('regime_detection_window', 60)
        if not isinstance(regime_detection_window, int) or regime_detection_window <= 0:
            raise ConfigurationException(f"regime_detection_window必须是正整数，得到: {regime_detection_window}")
        
        if regime_detection_window < 20:
            raise ConfigurationException(f"regime_detection_window不能小于20，得到: {regime_detection_window}")
        
        if regime_detection_window > 200:
            raise ConfigurationException(f"regime_detection_window不能大于200，得到: {regime_detection_window}")
        
        # 验证regime_model_type
        regime_model_type = config.get('regime_model_type', 'HMM')
        if not isinstance(regime_model_type, str):
            raise ConfigurationException(f"regime_model_type必须是字符串类型，得到: {regime_model_type}")
        
        valid_model_types = ['HMM', 'MS-GARCH']
        if regime_model_type not in valid_model_types:
            raise ConfigurationException(f"regime_model_type必须是{valid_model_types}之一，得到: {regime_model_type}")
        
        # 验证regime_states
        regime_states = config.get('regime_states', 3)
        if not isinstance(regime_states, int) or regime_states <= 0:
            raise ConfigurationException(f"regime_states必须是正整数，得到: {regime_states}")
        
        if regime_states < 2:
            raise ConfigurationException(f"regime_states不能小于2，得到: {regime_states}")
        
        if regime_states > 5:
            raise ConfigurationException(f"regime_states不能大于5，得到: {regime_states}")
    
    @staticmethod
    def _validate_performance_config(config: Dict[str, Any]) -> None:
        """验证性能优化配置"""
        # 验证enable_caching
        enable_caching = config.get('enable_caching', True)
        if not isinstance(enable_caching, bool):
            raise ConfigurationException(f"enable_caching必须是布尔类型，得到: {enable_caching}")
        
        # 验证cache_expiry_days
        cache_expiry_days = config.get('cache_expiry_days', 1)
        if not isinstance(cache_expiry_days, int) or cache_expiry_days <= 0:
            raise ConfigurationException(f"cache_expiry_days必须是正整数，得到: {cache_expiry_days}")
        
        if cache_expiry_days > 30:
            raise ConfigurationException(f"cache_expiry_days不能大于30，得到: {cache_expiry_days}")
        
        # 验证parallel_processing
        parallel_processing = config.get('parallel_processing', True)
        if not isinstance(parallel_processing, bool):
            raise ConfigurationException(f"parallel_processing必须是布尔类型，得到: {parallel_processing}")
        
        # 验证max_workers
        max_workers = config.get('max_workers', 4)
        if not isinstance(max_workers, int) or max_workers <= 0:
            raise ConfigurationException(f"max_workers必须是正整数，得到: {max_workers}")
        
        if max_workers > 16:
            raise ConfigurationException(f"max_workers不能大于16，得到: {max_workers}")
    
    @staticmethod
    def _validate_data_quality_config(config: Dict[str, Any]) -> None:
        """验证数据质量配置"""
        # 验证min_data_length
        min_data_length = config.get('min_data_length', 100)
        if not isinstance(min_data_length, int) or min_data_length <= 0:
            raise ConfigurationException(f"min_data_length必须是正整数，得到: {min_data_length}")
        
        if min_data_length < 50:
            raise ConfigurationException(f"min_data_length不能小于50，得到: {min_data_length}")
        
        if min_data_length > 1000:
            raise ConfigurationException(f"min_data_length不能大于1000，得到: {min_data_length}")
        
        # 验证max_missing_ratio
        max_missing_ratio = config.get('max_missing_ratio', 0.1)
        if not isinstance(max_missing_ratio, (int, float)):
            raise ConfigurationException(f"max_missing_ratio必须是数值类型，得到: {max_missing_ratio}")
        
        if not 0.0 <= max_missing_ratio <= 0.5:
            raise ConfigurationException(f"max_missing_ratio必须在0.0-0.5范围内，得到: {max_missing_ratio}")
        
        # 验证outlier_threshold
        outlier_threshold = config.get('outlier_threshold', 5.0)
        if not isinstance(outlier_threshold, (int, float)):
            raise ConfigurationException(f"outlier_threshold必须是数值类型，得到: {outlier_threshold}")
        
        if not 2.0 <= outlier_threshold <= 10.0:
            raise ConfigurationException(f"outlier_threshold必须在2.0-10.0范围内，得到: {outlier_threshold}")
    
    @staticmethod
    def _validate_convergence_config(config: Dict[str, Any]) -> None:
        """验证模型收敛配置"""
        # 验证max_garch_iterations
        max_garch_iterations = config.get('max_garch_iterations', 1000)
        if not isinstance(max_garch_iterations, int) or max_garch_iterations <= 0:
            raise ConfigurationException(f"max_garch_iterations必须是正整数，得到: {max_garch_iterations}")
        
        if max_garch_iterations < 100:
            raise ConfigurationException(f"max_garch_iterations不能小于100，得到: {max_garch_iterations}")
        
        if max_garch_iterations > 5000:
            raise ConfigurationException(f"max_garch_iterations不能大于5000，得到: {max_garch_iterations}")
        
        # 验证garch_convergence_tolerance
        garch_convergence_tolerance = config.get('garch_convergence_tolerance', 1e-6)
        if not isinstance(garch_convergence_tolerance, (int, float)):
            raise ConfigurationException(f"garch_convergence_tolerance必须是数值类型，得到: {garch_convergence_tolerance}")
        
        if not 1e-8 <= garch_convergence_tolerance <= 1e-3:
            raise ConfigurationException(f"garch_convergence_tolerance必须在1e-8到1e-3范围内，得到: {garch_convergence_tolerance}")
        
        # 验证hmm_convergence_tolerance
        hmm_convergence_tolerance = config.get('hmm_convergence_tolerance', 1e-4)
        if not isinstance(hmm_convergence_tolerance, (int, float)):
            raise ConfigurationException(f"hmm_convergence_tolerance必须是数值类型，得到: {hmm_convergence_tolerance}")
        
        if not 1e-6 <= hmm_convergence_tolerance <= 1e-2:
            raise ConfigurationException(f"hmm_convergence_tolerance必须在1e-6到1e-2范围内，得到: {hmm_convergence_tolerance}")
        
        # 验证max_hmm_iterations
        max_hmm_iterations = config.get('max_hmm_iterations', 100)
        if not isinstance(max_hmm_iterations, int) or max_hmm_iterations <= 0:
            raise ConfigurationException(f"max_hmm_iterations必须是正整数，得到: {max_hmm_iterations}")
        
        if max_hmm_iterations < 10:
            raise ConfigurationException(f"max_hmm_iterations不能小于10，得到: {max_hmm_iterations}")
        
        if max_hmm_iterations > 1000:
            raise ConfigurationException(f"max_hmm_iterations不能大于1000，得到: {max_hmm_iterations}")


def validate_dynamic_lowvol_config(config: Dict[str, Any], data_manager=...) -> None:
    """
    验证动态低波筛选器配置的便捷函数
    
    Args:
        config: 动态低波筛选器配置字典
        data_manager: 数据管理器（可选）
        
    Raises:
        ConfigurationException: 配置参数不正确时抛出
        DataQualityException: 数据管理器无效时抛出
    """
    # 验证配置
    DynamicLowVolConfigValidator.validate_config(config)
    
    # 验证数据管理器（如果显式传递了参数）
    if data_manager is not ...:  # 使用Ellipsis作为默认值来检测是否传递了参数
        from risk_control.dynamic_lowvol_filter.exceptions import DataQualityException
        if data_manager is None:
            raise DataQualityException("数据管理器不能为空")


def get_default_dynamic_lowvol_config() -> Dict[str, Any]:
    """
    获取默认的动态低波筛选器配置
    
    Returns:
        默认配置字典
    """
    return {
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
        'ml_predictor_type': 'lightgbm',
        
        # IVOL约束配置
        'ivol_bad_threshold': 0.3,
        'ivol_good_threshold': 0.6,
        'five_factor_window': 252,
        
        # 市场状态检测配置
        'regime_detection_window': 60,
        'regime_model_type': 'HMM',
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
    }