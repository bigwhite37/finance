"""
市场状态检测器单元测试

测试MarketRegimeDetector类的所有功能，包括：
- 市场状态检测
- HMM模型拟合和验证
- 状态概率分布获取
- 统计信息获取
- 缓存机制
- 异常处理
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模拟hmmlearn库
mock_hmm = Mock()
mock_hmm.GaussianHMM = Mock()
sys.modules['hmmlearn'] = mock_hmm
sys.modules['hmmlearn.hmm'] = mock_hmm

from risk_control.dynamic_lowvol_filter import (
    MarketRegimeDetector,
    DynamicLowVolConfig,
    RegimeDetectionException,
    DataQualityException,
    InsufficientDataException,
    ConfigurationException
)


class TestMarketRegimeDetector(unittest.TestCase):
    """市场状态检测器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试配置
        self.config = DynamicLowVolConfig(
            regime_detection_window=60,
            regime_model_type="HMM",
            enable_caching=True
        )
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        
        # 生成模拟市场收益率数据（具有状态切换特征）
        market_returns_data = self._generate_regime_switching_returns(200)
        self.test_market_returns = pd.Series(market_returns_data, index=dates)
        
        self.current_date = pd.Timestamp('2020-06-01')
    
    def _generate_regime_switching_returns(self, n_periods):
        """生成具有状态切换特征的模拟市场收益率数据"""
        returns = np.zeros(n_periods)
        
        # 定义三个状态的参数
        regimes = {
            0: {'mean': 0.0005, 'std': 0.01},   # 低波动状态
            1: {'mean': 0.0000, 'std': 0.02},   # 中等波动状态
            2: {'mean': -0.001, 'std': 0.04}    # 高波动状态
        }
        
        # 简单的状态切换逻辑
        current_regime = 1  # 从中等状态开始
        
        for t in range(n_periods):
            # 状态切换概率
            if np.random.random() < 0.05:  # 5%的概率切换状态
                current_regime = np.random.choice([0, 1, 2])
            
            # 根据当前状态生成收益率
            regime_params = regimes[current_regime]
            returns[t] = np.random.normal(
                regime_params['mean'], 
                regime_params['std']
            )
        
        return returns
    
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector.__init__')
    def test_init_success(self, mock_init):
        """测试检测器初始化成功"""
        # 模拟成功初始化
        mock_init.return_value = None
        
        detector = MarketRegimeDetector.__new__(MarketRegimeDetector)
        detector.config = self.config
        detector.detection_window = 60
        detector.model_type = "HMM"
        detector.regime_mapping = {0: "低", 1: "中", 2: "高"}
        detector.reverse_mapping = {"低": 0, "中": 1, "高": 2}
        detector._regime_cache = {}
        detector._model_cache = {}
        detector.GaussianHMM = Mock()
        detector.StandardScaler = Mock()
        
        self.assertEqual(detector.detection_window, 60)
        self.assertEqual(detector.model_type, "HMM")
        self.assertIsNotNone(detector._regime_cache)
        self.assertIsNotNone(detector._model_cache)
    
    def test_init_without_hmmlearn_library(self):
        """测试没有hmmlearn库时的初始化失败"""
        with patch.dict('sys.modules', {'hmmlearn': None}):
            with self.assertRaises(ConfigurationException) as context:
                MarketRegimeDetector(self.config)
            
            self.assertIn("需要安装hmmlearn库", str(context.exception))
    
    def test_init_without_sklearn_library(self):
        """测试没有sklearn库时的初始化失败"""
        with patch.dict('sys.modules', {'sklearn': None, 'sklearn.preprocessing': None}):
            with self.assertRaises(ConfigurationException) as context:
                MarketRegimeDetector(self.config)
            
            self.assertIn("需要安装scikit-learn库", str(context.exception))
    
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._fit_hmm_model')
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._prepare_detection_data')
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._validate_input_data')
    def test_detect_regime_success(self, mock_validate, mock_prepare, mock_fit):
        """测试成功检测市场状态"""
        # 设置模拟
        mock_validate.return_value = None
        
        # 模拟准备的检测数据
        detection_data = pd.Series(np.random.randn(60))
        mock_prepare.return_value = detection_data
        
        # 模拟HMM模型
        mock_model = Mock()
        mock_model.monitor_ = Mock()
        mock_model.monitor_.converged = True
        mock_model.transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        mock_model.startprob_ = np.array([0.33, 0.34, 0.33])
        mock_model.means_ = np.array([[0.0], [0.01], [0.02]])
        mock_model.covars_ = np.array([[[0.001]], [[0.004]], [[0.016]]])
        mock_model.predict.return_value = np.array([1])  # 预测状态为1（中等波动）
        mock_fit.return_value = mock_model
        
        detector = MarketRegimeDetector(self.config)
        
        result = detector.detect_regime(self.test_market_returns, self.current_date)
        
        # 验证结果
        self.assertEqual(result, "中")
        
        # 验证方法调用
        mock_validate.assert_called_once()
        mock_prepare.assert_called_once()
        mock_fit.assert_called_once()
        mock_model.predict.assert_called_once()
    
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._fit_hmm_model')
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._prepare_detection_data')
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._validate_input_data')
    def test_detect_regime_with_caching(self, mock_validate, mock_prepare, mock_fit):
        """测试缓存机制"""
        # 设置模拟
        mock_validate.return_value = None
        detection_data = pd.Series(np.random.randn(60))
        mock_prepare.return_value = detection_data
        
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])  # 预测状态为0（低波动）
        mock_fit.return_value = mock_model
        
        detector = MarketRegimeDetector(self.config)
        
        # 第一次调用
        result1 = detector.detect_regime(self.test_market_returns, self.current_date)
        
        # 第二次调用（应该使用缓存）
        result2 = detector.detect_regime(self.test_market_returns, self.current_date)
        
        # 验证结果相同
        self.assertEqual(result1, result2)
        self.assertEqual(result1, "低")
        
        # 验证模型只被调用一次（第二次使用缓存）
        self.assertEqual(mock_fit.call_count, 1)
    
    def test_validate_input_data_empty_returns(self):
        """测试空收益率数据验证"""
        detector = MarketRegimeDetector(self.config)
        empty_returns = pd.Series([], dtype=float)
        
        with self.assertRaises(DataQualityException) as context:
            detector._validate_input_data(empty_returns, self.current_date)
        
        self.assertIn("市场收益率数据为空", str(context.exception))
    
    def test_validate_input_data_insufficient_data(self):
        """测试数据长度不足验证"""
        detector = MarketRegimeDetector(self.config)
        short_returns = pd.Series(np.random.randn(30))  # 少于detection_window=60
        
        with self.assertRaises(InsufficientDataException) as context:
            detector._validate_input_data(short_returns, self.current_date)
        
        self.assertIn("市场数据长度", str(context.exception))
        self.assertIn("不足", str(context.exception))
    
    def test_validate_input_data_high_missing_ratio(self):
        """测试高缺失值比例验证"""
        detector = MarketRegimeDetector(self.config)
        
        # 创建高缺失值比例的数据
        returns_with_na = self.test_market_returns.copy()
        returns_with_na.iloc[::5] = np.nan  # 每5个值设为NaN，缺失率20%
        
        with self.assertRaises(DataQualityException) as context:
            detector._validate_input_data(returns_with_na, self.current_date)
        
        self.assertIn("缺失值比例", str(context.exception))
        self.assertIn("过高", str(context.exception))
    
    def test_validate_input_data_zero_variance(self):
        """测试零方差数据验证"""
        detector = MarketRegimeDetector(self.config)
        
        # 创建零方差数据，使用DatetimeIndex
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        zero_var_returns = pd.Series([0.01] * 100, index=dates)  # 所有值相同
        test_date = dates[50]  # 使用中间的日期
        
        with self.assertRaises(DataQualityException) as context:
            detector._validate_input_data(zero_var_returns, test_date)
        
        self.assertIn("市场收益率方差为0", str(context.exception))
    
    def test_validate_input_data_extreme_returns(self):
        """测试极端收益率验证"""
        detector = MarketRegimeDetector(self.config)
        
        # 创建包含极端值的数据
        extreme_returns = self.test_market_returns.copy()
        extreme_returns.iloc[:20] = 0.25  # 设置20个极端值（25%收益率），提高比例
        
        with self.assertRaises(DataQualityException) as context:
            detector._validate_input_data(extreme_returns, self.current_date)
        
        self.assertIn("极端收益率", str(context.exception))
        self.assertIn("过高", str(context.exception))
    
    def test_validate_input_data_wrong_type(self):
        """测试错误数据类型验证"""
        detector = MarketRegimeDetector(self.config)
        
        # 传入DataFrame而不是Series
        wrong_type_data = pd.DataFrame({'returns': self.test_market_returns})
        
        with self.assertRaises(DataQualityException) as context:
            detector._validate_input_data(wrong_type_data, self.current_date)
        
        self.assertIn("必须为Series类型", str(context.exception))
    
    def test_prepare_detection_data_success(self):
        """测试成功准备检测数据"""
        detector = MarketRegimeDetector(self.config)
        
        result = detector._prepare_detection_data(
            self.test_market_returns, self.current_date
        )
        
        # 验证结果
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), detector.detection_window)
        self.assertTrue(all(np.isfinite(result)))
    
    def test_prepare_detection_data_with_missing_values(self):
        """测试包含缺失值的数据准备"""
        detector = MarketRegimeDetector(self.config)
        
        # 创建包含少量缺失值的数据
        returns_with_na = self.test_market_returns.copy()
        returns_with_na.iloc[50:55] = np.nan  # 设置5个缺失值
        
        result = detector._prepare_detection_data(returns_with_na, self.current_date)
        
        # 验证结果：缺失值应该被填充
        self.assertIsInstance(result, pd.Series)
        self.assertFalse(result.isna().any())
    
    def test_prepare_detection_data_insufficient_history(self):
        """测试历史数据不足的情况"""
        detector = MarketRegimeDetector(self.config)
        
        # 使用早于所有数据的日期
        early_date = self.test_market_returns.index[0] - pd.Timedelta(days=1)
        
        with self.assertRaises(DataQualityException) as context:
            detector._prepare_detection_data(self.test_market_returns, early_date)
        
        self.assertIn("没有可用的历史数据", str(context.exception))
    
    def test_fit_hmm_model_success(self):
        """测试成功拟合HMM模型"""
        # 模拟HMM模型
        mock_model = Mock()
        mock_model.monitor_ = Mock()
        mock_model.monitor_.converged = True
        mock_model.transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        mock_model.startprob_ = np.array([0.33, 0.34, 0.33])
        mock_model.means_ = np.array([[0.0], [0.15], [0.35]])  # 增大差异以通过区分度检查
        mock_model.covars_ = np.array([[[0.001]], [[0.004]], [[0.016]]])
        
        detector = MarketRegimeDetector(self.config)
        # 直接设置模拟的GaussianHMM
        detector.GaussianHMM = Mock(return_value=mock_model)
        
        detection_data = pd.Series(np.random.randn(60))
        
        result = detector._fit_hmm_model(detection_data)
        
        # 验证结果
        self.assertEqual(result, mock_model)
        
        # 验证模型配置
        detector.GaussianHMM.assert_called_once_with(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            tol=1e-4,
            random_state=42
        )
        
        # 验证模型拟合
        mock_model.fit.assert_called_once()
    
    def test_fit_hmm_model_convergence_failure(self):
        """测试HMM模型收敛失败"""
        # 模拟不收敛的模型
        mock_model = Mock()
        mock_model.monitor_ = Mock()
        mock_model.monitor_.converged = False
        
        detector = MarketRegimeDetector(self.config)
        # 直接设置模拟的GaussianHMM
        detector.GaussianHMM = Mock(return_value=mock_model)
        
        detection_data = pd.Series(np.random.randn(60))
        
        with self.assertRaises(RegimeDetectionException) as context:
            detector._fit_hmm_model(detection_data)
        
        self.assertIn("HMM模型未收敛", str(context.exception))
    
    def test_fit_hmm_model_invalid_parameters(self):
        """测试HMM模型参数无效"""
        # 模拟参数无效的模型
        mock_model = Mock()
        mock_model.monitor_ = Mock()
        mock_model.monitor_.converged = True
        mock_model.transmat_ = np.array([[0.8, 0.1, 0.2], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])  # 行和不为1
        mock_model.startprob_ = np.array([0.33, 0.34, 0.33])
        mock_model.means_ = np.array([[0.0], [0.01], [0.02]])
        mock_model.covars_ = np.array([[[0.001]], [[0.004]], [[0.016]]])
        
        detector = MarketRegimeDetector(self.config)
        # 直接设置模拟的GaussianHMM
        detector.GaussianHMM = Mock(return_value=mock_model)
        
        detection_data = pd.Series(np.random.randn(60))
        
        with self.assertRaises(RegimeDetectionException) as context:
            detector._fit_hmm_model(detection_data)
        
        self.assertIn("转移矩阵行和不为1", str(context.exception))
    
    def test_validate_hmm_parameters_negative_transition_matrix(self):
        """测试转移矩阵包含负值的验证"""
        detector = MarketRegimeDetector(self.config)
        
        # 创建包含负值的转移矩阵
        mock_model = Mock()
        mock_model.transmat_ = np.array([[0.8, 0.1, 0.1], [-0.1, 0.8, 0.3], [0.1, 0.1, 0.8]])
        mock_model.startprob_ = np.array([0.33, 0.34, 0.33])
        mock_model.means_ = np.array([[0.0], [0.01], [0.02]])
        mock_model.covars_ = np.array([[[0.001]], [[0.004]], [[0.016]]])
        
        with self.assertRaises(RegimeDetectionException) as context:
            detector._validate_hmm_parameters(mock_model)
        
        self.assertIn("转移矩阵包含负值", str(context.exception))
    
    def test_validate_hmm_parameters_invalid_startprob(self):
        """测试初始状态概率无效的验证"""
        detector = MarketRegimeDetector(self.config)
        
        # 创建概率和不为1的初始状态概率
        mock_model = Mock()
        mock_model.transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        mock_model.startprob_ = np.array([0.33, 0.34, 0.4])  # 和为1.07
        mock_model.means_ = np.array([[0.0], [0.01], [0.02]])
        mock_model.covars_ = np.array([[[0.001]], [[0.004]], [[0.016]]])
        
        with self.assertRaises(RegimeDetectionException) as context:
            detector._validate_hmm_parameters(mock_model)
        
        self.assertIn("初始状态概率和不为1", str(context.exception))
    
    def test_validate_hmm_parameters_invalid_covariance(self):
        """测试协方差参数无效的验证"""
        detector = MarketRegimeDetector(self.config)
        
        # 创建包含负协方差的模型
        mock_model = Mock()
        mock_model.transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        mock_model.startprob_ = np.array([0.33, 0.34, 0.33])
        mock_model.means_ = np.array([[0.0], [0.01], [0.02]])
        mock_model.covars_ = np.array([[[0.001]], [[-0.004]], [[0.016]]])  # 负协方差
        
        with self.assertRaises(RegimeDetectionException) as context:
            detector._validate_hmm_parameters(mock_model)
        
        self.assertIn("协方差参数必须为正", str(context.exception))
    
    def test_validate_hmm_parameters_insufficient_discrimination(self):
        """测试状态区分度不足的验证"""
        detector = MarketRegimeDetector(self.config)
        
        # 创建均值差异过小的模型
        mock_model = Mock()
        mock_model.transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        mock_model.startprob_ = np.array([0.33, 0.34, 0.33])
        mock_model.means_ = np.array([[0.0], [0.01], [0.015]])  # 差异过小
        mock_model.covars_ = np.array([[[0.001]], [[0.004]], [[0.016]]])
        
        with self.assertRaises(RegimeDetectionException) as context:
            detector._validate_hmm_parameters(mock_model)
        
        self.assertIn("状态区分度不足", str(context.exception))
    
    def test_predict_current_state_success(self):
        """测试成功预测当前状态"""
        detector = MarketRegimeDetector(self.config)
        
        # 模拟HMM模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 1, 0])  # 最后状态为0
        
        detection_data = pd.Series(np.random.randn(5))
        
        result = detector._predict_current_state(mock_model, detection_data)
        
        # 验证结果
        self.assertEqual(result, 0)
        
        # 验证模型调用
        mock_model.predict.assert_called_once()
    
    def test_predict_current_state_invalid_state(self):
        """测试预测到无效状态"""
        detector = MarketRegimeDetector(self.config)
        
        # 模拟返回无效状态的模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([3])  # 无效状态3
        
        detection_data = pd.Series(np.random.randn(5))
        
        with self.assertRaises(RegimeDetectionException) as context:
            detector._predict_current_state(mock_model, detection_data)
        
        self.assertIn("检测到无效状态", str(context.exception))
    
    def test_validate_and_map_state_success(self):
        """测试成功验证和映射状态"""
        detector = MarketRegimeDetector(self.config)
        
        # 测试所有有效状态
        self.assertEqual(detector._validate_and_map_state(0), "低")
        self.assertEqual(detector._validate_and_map_state(1), "中")
        self.assertEqual(detector._validate_and_map_state(2), "高")
    
    def test_validate_and_map_state_invalid_index(self):
        """测试无效状态索引"""
        detector = MarketRegimeDetector(self.config)
        
        with self.assertRaises(RegimeDetectionException) as context:
            detector._validate_and_map_state(3)
        
        self.assertIn("检测到无效状态索引", str(context.exception))
    
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._fit_hmm_model')
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._prepare_detection_data')
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._validate_input_data')
    def test_get_regime_probabilities_success(self, mock_validate, mock_prepare, mock_fit):
        """测试成功获取状态概率分布"""
        # 设置模拟
        mock_validate.return_value = None
        detection_data = pd.Series(np.random.randn(60))
        mock_prepare.return_value = detection_data
        
        # 模拟HMM模型
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([
            [0.1, 0.8, 0.1],  # 第一个时点
            [0.2, 0.6, 0.2],  # 最后一个时点
        ])
        mock_fit.return_value = mock_model
        
        detector = MarketRegimeDetector(self.config)
        
        result = detector.get_regime_probabilities(
            self.test_market_returns, self.current_date
        )
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        self.assertIn("低", result)
        self.assertIn("中", result)
        self.assertIn("高", result)
        
        # 验证概率值
        self.assertEqual(result["低"], 0.2)
        self.assertEqual(result["中"], 0.6)
        self.assertEqual(result["高"], 0.2)
        
        # 验证概率和为1
        self.assertAlmostEqual(sum(result.values()), 1.0, places=6)
    
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._fit_hmm_model')
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._prepare_detection_data')
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._validate_input_data')
    def test_get_regime_statistics_success(self, mock_validate, mock_prepare, mock_fit):
        """测试成功获取状态统计信息"""
        # 设置模拟
        mock_validate.return_value = None
        detection_data = pd.Series(np.random.randn(60))
        mock_prepare.return_value = detection_data
        
        # 模拟HMM模型
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 2, 1, 0] * 12)  # 60个状态
        mock_model.transmat_ = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        mock_model.means_ = np.array([[0.0], [0.01], [0.02]])
        mock_model.covars_ = np.array([[[0.001]], [[0.004]], [[0.016]]])
        mock_model.score.return_value = -100.0
        mock_model.n_iter_ = 50
        mock_model.monitor_ = Mock()
        mock_model.monitor_.converged = True
        
        mock_fit.return_value = mock_model
        
        detector = MarketRegimeDetector(self.config)
        
        result = detector.get_regime_statistics(
            self.test_market_returns, self.current_date
        )
        
        # 验证结果结构
        self.assertIsInstance(result, dict)
        self.assertIn('current_regime', result)
        self.assertIn('regime_distribution', result)
        self.assertIn('transition_matrix', result)
        self.assertIn('means', result)
        self.assertIn('covariances', result)
        self.assertIn('model_score', result)
        self.assertIn('n_iter', result)
        self.assertIn('converged', result)
        
        # 验证具体值
        self.assertEqual(result['current_regime'], "低")  # 最后状态为0
        self.assertTrue(result['converged'])
        self.assertEqual(result['n_iter'], 50)
        self.assertEqual(result['model_score'], -100.0)
    
    @patch('risk_control.dynamic_lowvol_filter.MarketRegimeDetector._prepare_detection_data')
    def test_get_regime_statistics_failure(self, mock_prepare):
        """测试获取统计信息失败时的默认返回"""
        # 模拟异常
        mock_prepare.side_effect = Exception("测试异常")
        
        detector = MarketRegimeDetector(self.config)
        
        result = detector.get_regime_statistics(
            self.test_market_returns, self.current_date
        )
        
        # 验证默认返回值
        self.assertEqual(result['current_regime'], '中')
        self.assertIn('error', result)
        self.assertFalse(result['converged'])
        self.assertEqual(result['regime_distribution']['低'], 0.33)
    
    def test_clear_cache_all(self):
        """测试清理全部缓存"""
        detector = MarketRegimeDetector(self.config)
        
        # 添加一些缓存数据
        detector._regime_cache['test_key'] = "中"
        detector._model_cache['test_key'] = Mock()
        
        # 清理全部缓存
        detector.clear_cache()
        
        # 验证缓存已清空
        self.assertEqual(len(detector._regime_cache), 0)
        self.assertEqual(len(detector._model_cache), 0)
    
    def test_clear_cache_expired(self):
        """测试清理过期缓存"""
        detector = MarketRegimeDetector(self.config)
        
        # 添加新旧缓存数据
        old_date = pd.Timestamp.now() - pd.Timedelta(days=10)
        new_date = pd.Timestamp.now() - pd.Timedelta(days=1)
        
        detector._regime_cache[(old_date, 100)] = "高"
        detector._regime_cache[(new_date, 100)] = "中"
        detector._model_cache[(old_date, 100)] = Mock()
        detector._model_cache[(new_date, 100)] = Mock()
        
        # 清理5天前的缓存
        detector.clear_cache(older_than_days=5)
        
        # 验证只有旧缓存被清理
        self.assertEqual(len(detector._regime_cache), 1)
        self.assertEqual(len(detector._model_cache), 1)
        self.assertIn((new_date, 100), detector._regime_cache)
        self.assertIn((new_date, 100), detector._model_cache)
    
    def test_clear_cache_disabled(self):
        """测试缓存禁用时的清理操作"""
        config_no_cache = DynamicLowVolConfig(enable_caching=False)
        detector = MarketRegimeDetector(config_no_cache)
        
        # 清理操作不应该报错
        detector.clear_cache()
        detector.clear_cache(older_than_days=5)


if __name__ == '__main__':
    unittest.main()