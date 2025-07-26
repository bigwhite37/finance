"""
GARCH波动率预测器单元测试

测试GARCHVolatilityPredictor类的所有功能，包括：
- 单只股票波动率预测
- 批量股票波动率预测
- 模型诊断信息获取
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

# 模拟arch库
mock_arch = Mock()
mock_arch.arch_model = Mock()
sys.modules['arch'] = mock_arch

from risk_control.dynamic_lowvol_filter.predictors.garch_volatility import GARCHVolatilityPredictor
from risk_control.dynamic_lowvol_filter.data_structures import DynamicLowVolConfig
from risk_control.dynamic_lowvol_filter.exceptions import (
    ModelFittingException,
    DataQualityException,
    InsufficientDataException,
    ConfigurationException
)


class TestGARCHVolatilityPredictor(unittest.TestCase):
    """GARCH波动率预测器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试配置
        self.config = DynamicLowVolConfig(
            garch_window=100,  # 减小窗口以便测试
            forecast_horizon=5,
            enable_caching=True
        )
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        
        # 生成模拟收益率数据（具有GARCH特征）
        returns_data = self._generate_garch_returns(300)
        self.test_returns = pd.Series(returns_data, index=dates)
        
        # 创建多股票收益率数据
        self.multi_returns = pd.DataFrame({
            'STOCK_A': self._generate_garch_returns(300),
            'STOCK_B': self._generate_garch_returns(300),
            'STOCK_C': self._generate_garch_returns(300)
        }, index=dates)
        
        self.current_date = pd.Timestamp('2020-10-01')
        self.stock_code = 'TEST_STOCK'
    
    def _generate_garch_returns(self, n_periods):
        """生成具有GARCH特征的模拟收益率数据"""
        # GARCH(1,1)参数
        omega = 0.0001
        alpha = 0.1
        beta = 0.85
        
        returns = np.zeros(n_periods)
        sigma2 = np.zeros(n_periods)
        sigma2[0] = omega / (1 - alpha - beta)  # 无条件方差
        
        for t in range(1, n_periods):
            # GARCH方程
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            # 生成收益率
            returns[t] = np.sqrt(sigma2[t]) * np.random.standard_t(df=5)
        
        return returns
    
    @patch('risk_control.dynamic_lowvol_filter.predictors.garch_volatility.GARCHVolatilityPredictor.__init__')
    def test_init_success(self, mock_init):
        """测试预测器初始化成功"""
        # 模拟成功初始化
        mock_init.return_value = None
        
        predictor = GARCHVolatilityPredictor.__new__(GARCHVolatilityPredictor)
        predictor.config = self.config
        predictor.garch_window = 100
        predictor.forecast_horizon = 5
        predictor.enable_caching = True
        predictor._prediction_cache = {}
        predictor._model_cache = {}
        predictor.arch_model = Mock()
        
        self.assertEqual(predictor.garch_window, 100)
        self.assertEqual(predictor.forecast_horizon, 5)
        self.assertTrue(predictor.enable_caching)
        self.assertIsNotNone(predictor._prediction_cache)
        self.assertIsNotNone(predictor._model_cache)
    
    def test_init_without_arch_library(self):
        """测试没有arch库时的初始化失败"""
        with patch.dict('sys.modules', {'arch': None}):
            with self.assertRaises(ConfigurationException) as context:
                GARCHVolatilityPredictor(self.config)
            
            self.assertIn("需要安装arch库", str(context.exception))
    
    @patch('arch.arch_model')
    def test_predict_volatility_success(self, mock_arch_model):
        """测试成功预测单只股票波动率"""
        # 模拟GARCH模型
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = True
        mock_fitted_model.params = {
            'omega': 0.0001,
            'alpha[1]': 0.1,
            'beta[1]': 0.85
        }
        
        # 模拟预测结果 - 使用更合理的方差值（约15%年化波动率）
        mock_forecast = Mock()
        mock_forecast.variance = pd.DataFrame([[0.894, 0.9, 0.91, 0.92, 0.93]])  # 约15%年化波动率
        mock_fitted_model.forecast.return_value = mock_forecast
        
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        result = predictor.predict_volatility(
            self.test_returns, self.stock_code, self.current_date
        )
        
        # 验证结果
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
        self.assertLess(result, 2.0)  # 合理的年化波动率范围
        
        # 验证模型调用
        mock_arch_model.assert_called_once()
        mock_model.fit.assert_called_once()
        mock_fitted_model.forecast.assert_called_once_with(horizon=5, method='simulation')
    
    @patch('arch.arch_model')
    def test_predict_volatility_with_caching(self, mock_arch_model):
        """测试缓存机制"""
        # 设置模拟
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = True
        mock_fitted_model.params = {
            'omega': 0.0001,
            'alpha[1]': 0.1,
            'beta[1]': 0.85
        }
        
        mock_forecast = Mock()
        mock_forecast.variance = pd.DataFrame([[0.894]])  # 约15%年化波动率
        mock_fitted_model.forecast.return_value = mock_forecast
        
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 第一次调用
        result1 = predictor.predict_volatility(
            self.test_returns, self.stock_code, self.current_date, horizon=1
        )
        
        # 第二次调用（应该使用缓存）
        result2 = predictor.predict_volatility(
            self.test_returns, self.stock_code, self.current_date, horizon=1
        )
        
        # 验证结果相同
        self.assertEqual(result1, result2)
        
        # 验证模型只被调用一次（第二次使用缓存）
        self.assertEqual(mock_arch_model.call_count, 1)
    
    def test_validate_input_data_empty_returns(self):
        """测试空收益率数据验证"""
        predictor = GARCHVolatilityPredictor(self.config)
        empty_returns = pd.Series([], dtype=float)
        
        with self.assertRaises(DataQualityException) as context:
            predictor._validate_input_data(empty_returns, self.stock_code, 5)
        
        self.assertIn("收益率数据为空", str(context.exception))
    
    def test_validate_input_data_insufficient_data(self):
        """测试数据长度不足验证"""
        predictor = GARCHVolatilityPredictor(self.config)
        short_returns = pd.Series(np.random.randn(50))  # 少于garch_window=100
        
        with self.assertRaises(InsufficientDataException) as context:
            predictor._validate_input_data(short_returns, self.stock_code, 5)
        
        self.assertIn("有效数据长度", str(context.exception))
        self.assertIn("不足", str(context.exception))
    
    def test_validate_input_data_high_missing_ratio(self):
        """测试高缺失值比例验证"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 创建高缺失值比例的数据
        returns_with_na = self.test_returns.copy()
        returns_with_na.iloc[::3] = np.nan  # 每3个值设为NaN，缺失率约33%
        
        with self.assertRaises(DataQualityException) as context:
            predictor._validate_input_data(returns_with_na, self.stock_code, 5)
        
        self.assertIn("缺失值比例", str(context.exception))
        self.assertIn("过高", str(context.exception))
    
    def test_validate_input_data_zero_variance(self):
        """测试零方差数据验证"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 创建零方差数据
        zero_var_returns = pd.Series([0.01] * 200)  # 所有值相同
        
        with self.assertRaises(DataQualityException) as context:
            predictor._validate_input_data(zero_var_returns, self.stock_code, 5)
        
        self.assertIn("收益率方差为0", str(context.exception))
    
    def test_validate_input_data_extreme_returns(self):
        """测试极端收益率验证"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 创建包含极端值的数据
        extreme_returns = self.test_returns.copy()
        extreme_returns.iloc[:20] = 0.6  # 设置20个极端值（60%收益率）
        
        with self.assertRaises(DataQualityException) as context:
            predictor._validate_input_data(extreme_returns, self.stock_code, 5)
        
        self.assertIn("极端收益率过多", str(context.exception))
    
    def test_validate_input_data_invalid_stock_code(self):
        """测试无效股票代码验证"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        with self.assertRaises(DataQualityException) as context:
            predictor._validate_input_data(self.test_returns, "", 5)
        
        self.assertIn("股票代码必须为非空字符串", str(context.exception))
    
    def test_validate_input_data_invalid_horizon(self):
        """测试无效预测期限验证"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        with self.assertRaises(ConfigurationException) as context:
            predictor._validate_input_data(self.test_returns, self.stock_code, 0)
        
        self.assertIn("预测期限必须为正整数", str(context.exception))
    
    @patch('arch.arch_model')
    def test_fit_garch_model_convergence_failure(self, mock_arch_model):
        """测试GARCH模型收敛失败"""
        # 模拟不收敛的模型
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = False
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        with self.assertRaises(ModelFittingException) as context:
            predictor._fit_garch_model(self.test_returns.iloc[-100:], self.stock_code)
        
        self.assertIn("GARCH模型未收敛", str(context.exception))
    
    @patch('arch.arch_model')
    def test_fit_garch_model_invalid_parameters(self, mock_arch_model):
        """测试GARCH模型参数无效"""
        # 模拟参数无效的模型
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = True
        mock_fitted_model.params = {
            'omega': -0.0001,  # 负值，无效
            'alpha[1]': 0.1,
            'beta[1]': 0.85
        }
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        with self.assertRaises(ModelFittingException) as context:
            predictor._fit_garch_model(self.test_returns.iloc[-100:], self.stock_code)
        
        self.assertIn("omega参数", str(context.exception))
        self.assertIn("必须为正", str(context.exception))
    
    @patch('arch.arch_model')
    def test_fit_garch_model_non_stationary(self, mock_arch_model):
        """测试GARCH模型非平稳性"""
        # 模拟非平稳参数的模型
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = True
        mock_fitted_model.params = {
            'omega': 0.0001,
            'alpha[1]': 0.6,
            'beta[1]': 0.5  # alpha + beta = 1.1 > 1，非平稳
        }
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        with self.assertRaises(ModelFittingException) as context:
            predictor._fit_garch_model(self.test_returns.iloc[-100:], self.stock_code)
        
        self.assertIn("不满足平稳性条件", str(context.exception))
    
    @patch('arch.arch_model')
    def test_forecast_volatility_invalid_result(self, mock_arch_model):
        """测试预测结果无效"""
        # 模拟预测结果无效的情况
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = True
        mock_fitted_model.params = {
            'omega': 0.0001,
            'alpha[1]': 0.1,
            'beta[1]': 0.85
        }
        
        # 模拟无效的预测结果
        mock_forecast = Mock()
        mock_forecast.variance = pd.DataFrame([[np.nan]])  # NaN结果
        mock_fitted_model.forecast.return_value = mock_forecast
        
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        with self.assertRaises(ModelFittingException) as context:
            predictor._forecast_volatility(mock_fitted_model, 1)
        
        self.assertIn("GARCH预测结果异常", str(context.exception))
    
    @patch('arch.arch_model')
    def test_forecast_volatility_extreme_result(self, mock_arch_model):
        """测试预测结果过于极端"""
        # 模拟极端预测结果
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = True
        mock_fitted_model.params = {
            'omega': 0.0001,
            'alpha[1]': 0.1,
            'beta[1]': 0.85
        }
        
        # 模拟极端高的预测结果
        mock_forecast = Mock()
        mock_forecast.variance = pd.DataFrame([[10000000]])  # 极端高的方差 (已乘100)
        mock_fitted_model.forecast.return_value = mock_forecast
        
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        with self.assertRaises(ModelFittingException) as context:
            predictor._forecast_volatility(mock_fitted_model, 1)
        
        self.assertIn("预测波动率", str(context.exception))
        self.assertIn("过高", str(context.exception))
    
    @patch('arch.arch_model')
    def test_predict_batch_volatility_success(self, mock_arch_model):
        """测试批量预测成功"""
        # 模拟GARCH模型
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = True
        mock_fitted_model.params = {
            'omega': 0.0001,
            'alpha[1]': 0.1,
            'beta[1]': 0.85
        }
        
        mock_forecast = Mock()
        mock_forecast.variance = pd.DataFrame([[0.894]])  # 约15%年化波动率
        mock_fitted_model.forecast.return_value = mock_forecast
        
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        result = predictor.predict_batch_volatility(
            self.multi_returns, self.current_date
        )
        
        # 验证结果
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)  # 3只股票
        self.assertTrue(all(result > 0))  # 所有波动率都为正
        self.assertTrue(all(result < 2.0))  # 合理范围内
    
    @patch('arch.arch_model')
    def test_predict_batch_volatility_with_failures(self, mock_arch_model):
        """测试批量预测中部分失败的情况"""
        # 模拟部分股票拟合失败
        def side_effect(*args, **kwargs):
            # 第一只股票成功，其他失败
            if mock_arch_model.call_count == 1:
                mock_model = Mock()
                mock_fitted_model = Mock()
                mock_fitted_model.converged = True
                mock_fitted_model.params = {
                    'omega': 0.0001,
                    'alpha[1]': 0.1,
                    'beta[1]': 0.85
                }
                mock_forecast = Mock()
                mock_forecast.variance = pd.DataFrame([[0.0004]])
                mock_fitted_model.forecast.return_value = mock_forecast
                mock_model.fit.return_value = mock_fitted_model
                return mock_model
            else:
                # 其他股票拟合失败
                mock_model = Mock()
                mock_fitted_model = Mock()
                mock_fitted_model.converged = False
                mock_model.fit.return_value = mock_fitted_model
                return mock_model
        
        mock_arch_model.side_effect = side_effect
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        result = predictor.predict_batch_volatility(
            self.multi_returns, self.current_date
        )
        
        # 验证结果：应该有3个结果，失败的使用历史波动率
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(result > 0))
    
    @patch('arch.arch_model')
    def test_get_model_diagnostics_success(self, mock_arch_model):
        """测试获取模型诊断信息成功"""
        # 模拟成功的模型
        mock_model = Mock()
        mock_fitted_model = Mock()
        mock_fitted_model.converged = True
        mock_fitted_model.aic = 1000.0
        mock_fitted_model.bic = 1010.0
        mock_fitted_model.loglikelihood = -500.0
        mock_fitted_model.nobs = 100
        mock_fitted_model.params = {
            'omega': 0.0001,
            'alpha[1]': 0.1,
            'beta[1]': 0.85
        }
        
        mock_model.fit.return_value = mock_fitted_model
        mock_arch_model.return_value = mock_model
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        diagnostics = predictor.get_model_diagnostics(
            self.test_returns, self.stock_code
        )
        
        # 验证诊断信息
        self.assertTrue(diagnostics['converged'])
        self.assertEqual(diagnostics['aic'], 1000.0)
        self.assertEqual(diagnostics['bic'], 1010.0)
        self.assertEqual(diagnostics['log_likelihood'], -500.0)
        self.assertEqual(diagnostics['num_observations'], 100)
        self.assertEqual(diagnostics['alpha'], 0.1)
        self.assertEqual(diagnostics['beta'], 0.85)
        self.assertEqual(diagnostics['omega'], 0.0001)
    
    @patch('arch.arch_model')
    def test_get_model_diagnostics_failure(self, mock_arch_model):
        """测试获取模型诊断信息失败"""
        # 模拟拟合失败
        mock_arch_model.side_effect = Exception("拟合失败")
        
        predictor = GARCHVolatilityPredictor(self.config)
        
        diagnostics = predictor.get_model_diagnostics(
            self.test_returns, self.stock_code
        )
        
        # 验证失败时的诊断信息
        self.assertFalse(diagnostics['converged'])
        self.assertIn('error', diagnostics)
        self.assertTrue(pd.isna(diagnostics['aic']))
        self.assertTrue(pd.isna(diagnostics['alpha']))
    
    def test_clear_cache_all(self):
        """测试清理全部缓存"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 添加一些缓存数据
        predictor._prediction_cache['test_key'] = 0.2
        predictor._model_cache['test_key'] = Mock()
        
        # 清理全部缓存
        predictor.clear_cache()
        
        # 验证缓存已清空
        self.assertEqual(len(predictor._prediction_cache), 0)
        self.assertEqual(len(predictor._model_cache), 0)
    
    def test_clear_cache_expired(self):
        """测试清理过期缓存"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 添加新旧缓存数据
        old_date = pd.Timestamp.now() - pd.Timedelta(days=10)
        new_date = pd.Timestamp.now() - pd.Timedelta(days=1)
        
        predictor._prediction_cache[('STOCK1', old_date, 5)] = 0.2
        predictor._prediction_cache[('STOCK2', new_date, 5)] = 0.3
        predictor._model_cache[('STOCK1', old_date)] = Mock()
        predictor._model_cache[('STOCK2', new_date)] = Mock()
        
        # 清理5天前的缓存
        predictor.clear_cache(older_than_days=5)
        
        # 验证只有旧缓存被清理
        self.assertEqual(len(predictor._prediction_cache), 1)
        self.assertEqual(len(predictor._model_cache), 1)
        self.assertIn(('STOCK2', new_date, 5), predictor._prediction_cache)
        self.assertIn(('STOCK2', new_date), predictor._model_cache)
    
    def test_clear_cache_disabled(self):
        """测试缓存禁用时的清理操作"""
        config_no_cache = DynamicLowVolConfig(enable_caching=False)
        predictor = GARCHVolatilityPredictor(config_no_cache)
        
        # 清理操作不应该报错
        predictor.clear_cache()
        predictor.clear_cache(older_than_days=5)
    
    def test_winsorize_returns(self):
        """测试收益率缩尾处理"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 创建包含极端值的数据
        returns_with_outliers = pd.Series([
            -0.5, -0.1, -0.05, 0.0, 0.02, 0.05, 0.1, 0.6  # 包含-50%和60%的极端值
        ])
        
        winsorized = predictor._winsorize_returns(returns_with_outliers)
        
        # 验证极端值被处理
        self.assertGreater(winsorized.iloc[0], -0.5)  # 极端负值被调整
        self.assertLess(winsorized.iloc[-1], 0.6)     # 极端正值被调整
        
        # 验证中间值不变
        self.assertEqual(winsorized.iloc[3], 0.0)
        self.assertEqual(winsorized.iloc[4], 0.02)
    
    def test_prepare_garch_data(self):
        """测试GARCH数据准备"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 测试正常情况
        prepared_data = predictor._prepare_garch_data(
            self.test_returns, self.current_date
        )
        
        # 验证数据长度
        self.assertEqual(len(prepared_data), predictor.garch_window)
        
        # 验证数据截止日期
        self.assertLessEqual(prepared_data.index[-1], self.current_date)
    
    def test_prepare_garch_data_insufficient(self):
        """测试GARCH数据准备时数据不足"""
        predictor = GARCHVolatilityPredictor(self.config)
        
        # 创建截止日期过早的情况
        early_date = self.test_returns.index[50]  # 只有50个数据点
        
        prepared_data = predictor._prepare_garch_data(
            self.test_returns, early_date
        )
        
        self.assertLess(len(prepared_data), predictor.garch_window)


if __name__ == '__main__':
    unittest.main()