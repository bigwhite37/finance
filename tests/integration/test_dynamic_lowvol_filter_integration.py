"""
动态低波筛选器主控制器集成测试

测试DynamicLowVolFilter主控制器的完整功能，包括：
- 四层筛选流水线协调
- 市场状态检测和阈值调整
- 统计信息收集和报告
- 与数据管理器的集成
- 异常处理和错误恢复
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.dynamic_lowvol_validator import validate_dynamic_lowvol_config
from risk_control.dynamic_lowvol_filter import (
    DynamicLowVolFilter,
    DynamicLowVolConfig,
    DataQualityException,
    InsufficientDataException,
    ModelFittingException,
    RegimeDetectionException,
    ConfigurationException
)


class TestDynamicLowVolFilterIntegration(unittest.TestCase):
    """动态低波筛选器主控制器集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试配置
        self.config = {
            'rolling_windows': [20, 60],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            'garch_window': 250,
            'forecast_horizon': 5,
            'enable_ml_predictor': False,
            'ivol_bad_threshold': 0.3,
            'ivol_good_threshold': 0.6,
            'regime_detection_window': 60,
            'regime_model_type': "HMM",
            'enable_caching': True,
            'cache_expiry_days': 1,
            'parallel_processing': False
        }
        
        # 创建模拟数据管理器
        self.mock_data_manager = self._create_mock_data_manager()
        
        # 创建测试日期
        self.test_date = pd.Timestamp('2023-06-01')
        
        # 创建测试数据
        self.test_data = self._create_test_data()
    
    def _create_mock_data_manager(self):
        """创建模拟数据管理器"""
        mock_manager = Mock()
        
        # 模拟价格数据
        dates = pd.date_range('2022-01-01', '2023-06-01', freq='D')
        stocks = [f'stock_{i:03d}' for i in range(100)]
        
        # 生成模拟价格数据
        np.random.seed(42)
        price_data = pd.DataFrame(
            np.random.randn(len(dates), len(stocks)).cumsum(axis=0) + 100,
            index=dates,
            columns=stocks
        )
        
        # 生成模拟成交量数据
        volume_data = pd.DataFrame(
            np.random.exponential(1000000, (len(dates), len(stocks))),
            index=dates,
            columns=stocks
        )
        
        # 生成模拟因子数据
        factor_names = ['market', 'size', 'value', 'profitability', 'investment']
        factor_data = pd.DataFrame(
            np.random.randn(len(dates), len(factor_names)) * 0.02,
            index=dates,
            columns=factor_names
        )
        
        # 生成模拟市场数据
        market_data = pd.DataFrame({
            'returns': np.random.randn(len(dates)) * 0.02,
            'volatility': np.random.exponential(0.2, len(dates))
        }, index=dates)
        
        # 设置模拟方法返回值
        mock_manager.get_price_data.return_value = price_data
        mock_manager.get_volume_data.return_value = volume_data
        mock_manager.get_factor_data.return_value = factor_data
        mock_manager.get_market_data.return_value = market_data
        
        return mock_manager
    
    def _create_test_data(self):
        """创建测试数据"""
        dates = pd.date_range('2022-01-01', '2023-06-01', freq='D')
        stocks = [f'stock_{i:03d}' for i in range(50)]
        
        # 生成具有不同波动特征的股票数据
        np.random.seed(42)
        returns_data = {}
        
        for i, stock in enumerate(stocks):
            if i < 15:  # 低波动股票
                vol = 0.15 + np.random.random() * 0.1
            elif i < 35:  # 中等波动股票
                vol = 0.25 + np.random.random() * 0.15
            else:  # 高波动股票
                vol = 0.45 + np.random.random() * 0.2
            
            returns_data[stock] = np.random.randn(len(dates)) * vol / np.sqrt(252)
        
        return pd.DataFrame(returns_data, index=dates)
    
    def test_initialization_success(self):
        """测试成功初始化"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 验证配置正确加载
        self.assertIsInstance(filter_instance.config, DynamicLowVolConfig)
        self.assertEqual(filter_instance.config.rolling_windows, [20, 60])
        self.assertEqual(filter_instance.config.garch_window, 250)
        
        # 验证组件正确初始化
        self.assertIsNotNone(filter_instance.data_preprocessor)
        self.assertIsNotNone(filter_instance.rolling_filter)
        self.assertIsNotNone(filter_instance.garch_predictor)
        self.assertIsNotNone(filter_instance.ivol_filter)
        self.assertIsNotNone(filter_instance.regime_detector)
        self.assertIsNotNone(filter_instance.threshold_adjuster)
        
        # 验证初始状态
        self.assertEqual(filter_instance._current_regime, "中")
        self.assertEqual(filter_instance._current_regime_confidence, 0.5)
        self.assertEqual(filter_instance._current_market_volatility, 0.3)
        self.assertIsNone(filter_instance._last_update_date)
        self.assertIsNone(filter_instance._current_tradable_mask)
    
    def test_initialization_invalid_config(self):
        """测试无效配置初始化"""
        # 测试非字典配置
        with self.assertRaises(ConfigurationException):
            validate_dynamic_lowvol_config("invalid_config")
        
        # 测试空数据管理器
        with self.assertRaises(DataQualityException):
            validate_dynamic_lowvol_config(self.config, None)
        
        # 测试无效配置参数
        invalid_config = self.config.copy()
        invalid_config['rolling_windows'] = [-20, 60]  # 负数窗口
        
        with self.assertRaises(ConfigurationException):
            validate_dynamic_lowvol_config(invalid_config)
    
    @patch('risk_control.dynamic_lowvol_filter.core.MarketRegimeDetector')
    @patch('risk_control.dynamic_lowvol_filter.core.RegimeAwareThresholdAdjuster')
    @patch('risk_control.dynamic_lowvol_filter.core.IVOLConstraintFilter')
    @patch('risk_control.dynamic_lowvol_filter.core.GARCHVolatilityPredictor')
    @patch('risk_control.dynamic_lowvol_filter.core.RollingPercentileFilter')
    def test_update_tradable_mask_success(self, mock_rolling, mock_garch, 
                                        mock_ivol, mock_adjuster, mock_detector):
        """测试成功更新可交易掩码"""
        # 设置模拟组件
        mock_rolling_instance = Mock()
        mock_rolling_instance.apply_percentile_filter.return_value = np.array([True] * 50 + [False] * 50)
        mock_rolling.return_value = mock_rolling_instance
        
        mock_garch_instance = Mock()
        mock_garch_instance.predict_batch_volatility.return_value = pd.Series([0.3] * 60 + [0.5] * 40)  # 前60个低于阈值
        mock_garch.return_value = mock_garch_instance
        
        mock_ivol_instance = Mock()
        mock_ivol_instance.apply_ivol_constraint.return_value = np.array([True] * 70 + [False] * 30)
        mock_ivol.return_value = mock_ivol_instance
        
        mock_detector_instance = Mock()
        mock_detector_instance.detect_regime.return_value = "高"
        mock_detector_instance.get_regime_probabilities.return_value = {"高": 0.8, "中": 0.15, "低": 0.05}
        mock_detector.return_value = mock_detector_instance
        
        mock_adjuster_instance = Mock()
        mock_adjuster_instance.adjust_thresholds.return_value = {
            'percentile_cut': 0.2,
            'target_vol': 0.35,
            'ivol_bad_threshold': 0.25,
            'ivol_good_threshold': 0.55
        }
        # 设置 adjustment_history 为一个包含阈值的列表
        mock_adjuster_instance.adjustment_history = [
            {'thresholds': {
                'percentile_cut': 0.2,
                'target_vol': 0.35,
                'ivol_bad_threshold': 0.25,
                'ivol_good_threshold': 0.55
            }}
        ]
        mock_adjuster.return_value = mock_adjuster_instance
        
        # 创建筛选器实例
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 执行更新
        result_mask = filter_instance.update_tradable_mask(self.test_date)
        
        # 验证结果
        self.assertIsInstance(result_mask, np.ndarray)
        self.assertEqual(len(result_mask), 100)  # 应该有100只股票
        self.assertTrue(result_mask.dtype == bool)
        
        # 验证内部状态更新
        self.assertEqual(filter_instance._current_regime, "高")
        self.assertEqual(filter_instance._current_regime_confidence, 0.8)
        self.assertEqual(filter_instance._last_update_date, self.test_date)
        self.assertIsNotNone(filter_instance._current_tradable_mask)
        
        # 验证组件方法被正确调用
        mock_rolling_instance.apply_percentile_filter.assert_called_once()
        mock_garch_instance.predict_batch_volatility.assert_called_once()
        mock_ivol_instance.apply_ivol_constraint.assert_called_once()
        mock_detector_instance.detect_regime.assert_called_once()
        mock_adjuster_instance.adjust_thresholds.assert_called_once()
    
    def test_get_current_regime(self):
        """测试获取当前市场状态"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 测试初始状态
        self.assertEqual(filter_instance.get_current_regime(), "中")
        
        # 更新内部状态
        filter_instance._current_regime = "高"
        self.assertEqual(filter_instance.get_current_regime(), "高")
        
        filter_instance._current_regime = "低"
        self.assertEqual(filter_instance.get_current_regime(), "低")
    
    def test_get_adaptive_target_volatility(self):
        """测试获取自适应目标波动率"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 测试不同市场状态下的目标波动率
        test_cases = [
            ("低", 0.2, 0.8, 0.45 * 1.05),  # 低波动状态，低市场波动率，高置信度
            ("中", 0.3, 0.9, 0.40),         # 中等波动状态，正常市场波动率，高置信度
            ("高", 0.6, 0.9, 0.35 * 0.9),   # 高波动状态，高市场波动率，高置信度
            ("高", 0.3, 0.5, None),         # 高波动状态，正常市场波动率，低置信度
        ]
        
        for regime, market_vol, confidence, expected_base in test_cases:
            filter_instance._current_regime = regime
            filter_instance._current_market_volatility = market_vol
            filter_instance._current_regime_confidence = confidence
            
            result = filter_instance.get_adaptive_target_volatility()
            
            # 验证结果在合理范围内
            self.assertGreaterEqual(result, 0.25)
            self.assertLessEqual(result, 0.60)
            self.assertIsInstance(result, float)
            
            if expected_base is not None:
                # 对于高置信度情况，验证结果接近预期
                self.assertAlmostEqual(result, expected_base, delta=0.05)
    
    def test_get_filter_statistics(self):
        """测试获取筛选统计信息"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 测试初始统计信息
        stats = filter_instance.get_filter_statistics()
        
        # 验证统计信息结构
        self.assertIn('current_state', stats)
        self.assertIn('total_updates', stats)
        self.assertIn('regime_history', stats)
        self.assertIn('filter_pass_rates', stats)
        self.assertIn('performance_metrics', stats)
        self.assertIn('filter_summary', stats)
        self.assertIn('regime_distribution', stats)
        
        # 验证当前状态信息
        current_state = stats['current_state']
        self.assertEqual(current_state['regime'], "中")
        self.assertEqual(current_state['regime_confidence'], 0.5)
        self.assertEqual(current_state['market_volatility'], 0.3)
        self.assertIsNone(current_state['last_update_date'])
        self.assertEqual(current_state['tradable_stocks_count'], 0)
        self.assertEqual(current_state['total_stocks_count'], 0)
        
        # 验证初始性能指标
        self.assertEqual(stats['total_updates'], 0)
        self.assertEqual(len(stats['regime_history']), 0)
        self.assertEqual(stats['performance_metrics']['error_count'], 0)
        
        # 模拟一些更新后再测试
        filter_instance._filter_statistics['total_updates'] = 5
        filter_instance._filter_statistics['regime_history'] = ["高", "高", "中", "低", "中"]
        filter_instance._filter_statistics['filter_pass_rates']['final_combined'] = [0.3, 0.25, 0.35, 0.4, 0.3]
        filter_instance._current_tradable_mask = np.array([True] * 30 + [False] * 70)
        
        updated_stats = filter_instance.get_filter_statistics()
        
        # 验证更新后的统计信息
        self.assertEqual(updated_stats['total_updates'], 5)
        self.assertEqual(len(updated_stats['regime_history']), 5)
        self.assertEqual(updated_stats['current_state']['tradable_stocks_count'], 30)
        self.assertEqual(updated_stats['current_state']['total_stocks_count'], 100)
        
        # 验证筛选摘要统计
        self.assertAlmostEqual(updated_stats['filter_summary']['avg_pass_rate'], 0.32, places=2)
        self.assertEqual(updated_stats['filter_summary']['min_pass_rate'], 0.25)
        self.assertEqual(updated_stats['filter_summary']['max_pass_rate'], 0.4)
        
        # 验证状态分布统计
        expected_distribution = {"高": 0.4, "中": 0.4, "低": 0.2}
        for regime, expected_ratio in expected_distribution.items():
            self.assertAlmostEqual(
                updated_stats['regime_distribution'][regime], 
                expected_ratio, 
                places=2
            )
    
    def test_get_current_tradable_stocks(self):
        """测试获取当前可交易股票列表"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 测试初始状态（无数据）
        self.assertEqual(filter_instance.get_current_tradable_stocks(), [])
        
        # 设置模拟数据
        stock_universe = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        tradable_mask = np.array([True, False, True, True, False])
        
        filter_instance._current_stock_universe = stock_universe
        filter_instance._current_tradable_mask = tradable_mask
        
        result = filter_instance.get_current_tradable_stocks()
        expected = ['AAPL', 'MSFT', 'TSLA']
        
        self.assertEqual(result, expected)
    
    def test_get_regime_transition_probability(self):
        """测试获取状态转换概率"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 测试无历史数据的情况
        result = filter_instance.get_regime_transition_probability()
        expected_default = {"低": 0.33, "中": 0.34, "高": 0.33}
        
        for regime in expected_default:
            self.assertAlmostEqual(result[regime], expected_default[regime], places=2)
        
        # 设置历史数据
        filter_instance._filter_statistics['regime_history'] = [
            "高", "高", "中", "中", "低", "低", "中", "高"
        ]
        filter_instance._current_regime = "高"
        
        result = filter_instance.get_regime_transition_probability()
        
        # 验证结果结构
        self.assertIsInstance(result, dict)
        self.assertTrue(all(isinstance(v, float) for v in result.values()))
        self.assertAlmostEqual(sum(result.values()), 1.0, places=2)
    
    def test_reset_statistics(self):
        """测试重置统计信息"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 添加一些统计数据
        filter_instance._filter_statistics['total_updates'] = 10
        filter_instance._filter_statistics['regime_history'] = ["高", "中", "低"]
        filter_instance._filter_statistics['performance_metrics']['error_count'] = 5
        
        # 重置统计信息
        filter_instance.reset_statistics()
        
        # 验证重置结果
        self.assertEqual(filter_instance._filter_statistics['total_updates'], 0)
        self.assertEqual(len(filter_instance._filter_statistics['regime_history']), 0)
        self.assertEqual(filter_instance._filter_statistics['performance_metrics']['error_count'], 0)
        
        # 验证所有筛选通过率历史被清空
        for key in filter_instance._filter_statistics['filter_pass_rates']:
            self.assertEqual(len(filter_instance._filter_statistics['filter_pass_rates'][key]), 0)
    
    @patch('risk_control.dynamic_lowvol_filter.core.DataPreprocessor')
    def test_data_preprocessing_error_handling(self, mock_preprocessor):
        """测试数据预处理错误处理"""
        # 模拟数据预处理异常
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.preprocess_price_data.side_effect = DataQualityException("数据质量问题")
        mock_preprocessor.return_value = mock_preprocessor_instance
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 验证异常被正确抛出
        with self.assertRaises(DataQualityException):
            filter_instance.update_tradable_mask(self.test_date)
        
        # 验证错误计数增加
        self.assertEqual(filter_instance._filter_statistics['performance_metrics']['error_count'], 1)
    
    @patch('risk_control.dynamic_lowvol_filter.core.MarketRegimeDetector')
    def test_regime_detection_error_handling(self, mock_detector):
        """测试市场状态检测错误处理"""
        # 模拟状态检测异常
        mock_detector_instance = Mock()
        mock_detector_instance.detect_regime.side_effect = RegimeDetectionException("状态检测失败")
        mock_detector.return_value = mock_detector_instance
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 验证异常被正确抛出
        with self.assertRaises(RegimeDetectionException):
            filter_instance.update_tradable_mask(self.test_date)
        
        # 验证错误计数增加
        self.assertEqual(filter_instance._filter_statistics['performance_metrics']['error_count'], 1)
    
    def test_data_manager_integration(self):
        """测试与数据管理器的集成"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 模拟数据管理器方法调用
        with patch.object(filter_instance, '_preprocess_data') as mock_preprocess, \
             patch.object(filter_instance, '_detect_market_regime') as mock_detect, \
             patch.object(filter_instance, '_adjust_thresholds') as mock_adjust, \
             patch.object(filter_instance, '_execute_filter_pipeline') as mock_execute:
            
            # 设置模拟返回值
            mock_preprocess.return_value = {'returns_data': self.test_data}
            mock_detect.return_value = "中"
            mock_adjust.return_value = {'percentile_cut': 0.3}
            mock_execute.return_value = np.array([True] * 50 + [False] * 50)
            
            # 执行更新
            result = filter_instance.update_tradable_mask(self.test_date)
            
            # 验证数据管理器方法被调用
            self.mock_data_manager.get_price_data.assert_called_once()
            self.mock_data_manager.get_volume_data.assert_called_once()
            self.mock_data_manager.get_factor_data.assert_called_once()
            self.mock_data_manager.get_market_data.assert_called_once()
            
            # 验证内部方法被正确调用
            mock_preprocess.assert_called_once()
            mock_detect.assert_called_once()
            mock_adjust.assert_called_once()
            mock_execute.assert_called_once()
            
            # 验证结果
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 100)
    
    def test_performance_tracking(self):
        """测试性能跟踪"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 模拟快速执行
        with patch('time.time', side_effect=[0.0, 0.1, 0.1, 0.1, 0.1]):  # 100ms执行时间，提供足够的值
            with patch.object(filter_instance, '_preprocess_data') as mock_preprocess, \
                 patch.object(filter_instance, '_detect_market_regime') as mock_detect, \
                 patch.object(filter_instance, '_adjust_thresholds') as mock_adjust, \
                 patch.object(filter_instance, '_execute_filter_pipeline') as mock_execute:
                
                # 设置模拟返回值
                mock_preprocess.return_value = {'returns_data': self.test_data}
                mock_detect.return_value = "中"
                mock_adjust.return_value = {'percentile_cut': 0.3}
                mock_execute.return_value = np.array([True] * 50 + [False] * 50)
                
                # 执行更新
                filter_instance.update_tradable_mask(self.test_date)
        
        # 验证性能统计
        stats = filter_instance.get_filter_statistics()
        self.assertEqual(stats['total_updates'], 1)
        self.assertAlmostEqual(stats['performance_metrics']['avg_update_time'], 0.1, places=2)
    
    def test_filter_pipeline_coordination(self):
        """测试筛选流水线协调"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 创建模拟数据
        test_returns = pd.DataFrame(
            np.random.randn(100, 50) * 0.02,
            index=pd.date_range('2023-01-01', periods=100),
            columns=[f'stock_{i}' for i in range(50)]
        )
        
        test_factors = pd.DataFrame(
            np.random.randn(100, 5) * 0.01,
            index=pd.date_range('2023-01-01', periods=100),
            columns=['market', 'size', 'value', 'profitability', 'investment']
        )
        
        processed_data = {
            'returns_data': test_returns,
            'factor_data': test_factors,
            'market_data': pd.DataFrame({'returns': np.random.randn(100) * 0.02})
        }
        
        thresholds = {
            'percentile_cut': 0.3,
            'target_vol': 0.4,
            'ivol_bad_threshold': 0.3,
            'ivol_good_threshold': 0.6
        }
        
        # 模拟各层筛选结果
        with patch.object(filter_instance.rolling_filter, 'apply_percentile_filter') as mock_rolling, \
             patch.object(filter_instance.garch_predictor, 'predict_batch_volatility') as mock_garch, \
             patch.object(filter_instance.ivol_filter, 'apply_ivol_constraint') as mock_ivol:
            
            # 设置不同的筛选结果
            mock_rolling.return_value = np.array([True] * 30 + [False] * 20)  # 60%通过
            # GARCH预测器返回波动率值，然后会与target_vol比较
            mock_garch.return_value = pd.Series([0.3] * 25 + [0.5] * 25)  # 前25个低于0.4阈值，后25个高于
            mock_ivol.return_value = np.array([True] * 35 + [False] * 15)     # 70%通过
            
            # 执行筛选流水线
            result = filter_instance._execute_filter_pipeline(
                processed_data, self.test_date, thresholds
            )
            
            # 验证各层筛选被调用
            mock_rolling.assert_called_once()
            mock_garch.assert_called_once()
            mock_ivol.assert_called_once()
            
            # 验证结果是交集
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), 50)
            
            # 验证统计信息被更新
            stats = filter_instance.get_filter_statistics()
            self.assertEqual(len(stats['filter_pass_rates']['rolling_percentile']), 1)
            self.assertEqual(len(stats['filter_pass_rates']['garch_prediction']), 1)
            self.assertEqual(len(stats['filter_pass_rates']['ivol_constraint']), 1)
            self.assertEqual(len(stats['filter_pass_rates']['final_combined']), 1)


class TestDynamicLowVolFilterEdgeCases(unittest.TestCase):
    """动态低波筛选器边界情况测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'rolling_windows': [20],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            'enable_caching': False  # 禁用缓存以简化测试
        }
        self.mock_data_manager = Mock()
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        # 模拟空数据返回
        self.mock_data_manager.get_price_data.return_value = pd.DataFrame()
        self.mock_data_manager.get_volume_data.return_value = pd.DataFrame()
        self.mock_data_manager.get_factor_data.return_value = pd.DataFrame()
        self.mock_data_manager.get_market_data.return_value = pd.DataFrame()
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        with self.assertRaises(DataQualityException):
            filter_instance.update_tradable_mask(pd.Timestamp('2023-06-01'))
    
    def test_insufficient_data_handling(self):
        """测试数据不足处理"""
        # 创建数据长度不足的数据
        short_data = pd.DataFrame(
            np.random.randn(10, 5),  # 只有10天数据，不足以满足窗口要求
            index=pd.date_range('2023-01-01', periods=10),
            columns=['A', 'B', 'C', 'D', 'E']
        )
        
        self.mock_data_manager.get_price_data.return_value = short_data
        self.mock_data_manager.get_volume_data.return_value = short_data
        self.mock_data_manager.get_factor_data.return_value = short_data
        self.mock_data_manager.get_market_data.return_value = short_data
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        with self.assertRaises(InsufficientDataException):
            filter_instance.update_tradable_mask(pd.Timestamp('2023-01-10'))
    
    def test_extreme_market_conditions(self):
        """测试极端市场条件"""
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.mock_data_manager)
        
        # 测试极高波动率
        filter_instance._current_market_volatility = 2.0
        filter_instance._current_regime = "高"
        filter_instance._current_regime_confidence = 0.9
        
        target_vol = filter_instance.get_adaptive_target_volatility()
        self.assertGreaterEqual(target_vol, 0.25)
        self.assertLessEqual(target_vol, 0.60)
        
        # 测试极低波动率
        filter_instance._current_market_volatility = 0.05
        filter_instance._current_regime = "低"
        filter_instance._current_regime_confidence = 0.9
        
        target_vol = filter_instance.get_adaptive_target_volatility()
        self.assertGreaterEqual(target_vol, 0.25)
        self.assertLessEqual(target_vol, 0.60)
    
    def test_configuration_edge_cases(self):
        """测试配置边界情况"""
        # 测试最小配置
        minimal_config = {}
        filter_instance = DynamicLowVolFilter(minimal_config, self.mock_data_manager)
        self.assertIsNotNone(filter_instance.config)
        
        # 测试极端配置值
        extreme_config = {
            'rolling_windows': [1],  # 最小窗口
            'percentile_thresholds': {"低": 0.99, "中": 0.5, "高": 0.01},  # 极端阈值
            'garch_window': 100,  # 最小GARCH窗口
            'regime_detection_window': 20  # 最小检测窗口
        }
        
        filter_instance = DynamicLowVolFilter(extreme_config, self.mock_data_manager)
        self.assertIsNotNone(filter_instance.config)


if __name__ == '__main__':
    unittest.main()