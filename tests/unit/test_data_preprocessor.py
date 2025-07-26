"""
DataPreprocessor单元测试

测试数据预处理模块的各项功能，包括数据清洗、收益率计算、
滚动窗口准备和数据质量验证。
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_control.dynamic_lowvol_filter import (
    DataPreprocessor, 
    DynamicLowVolConfig,
    DataQualityException,
    InsufficientDataException,
    ConfigurationException
)


class TestDataPreprocessor(unittest.TestCase):
    """DataPreprocessor测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = DynamicLowVolConfig()
        self.preprocessor = DataPreprocessor(self.config)
        
        # 创建测试数据 - 需要足够长度满足 max(rolling_windows) + garch_window = 60 + 250 = 310
        self.dates = pd.date_range('2020-01-01', periods=350, freq='D')
        self.stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D']
        
        # 生成正常价格数据
        np.random.seed(42)
        self.normal_price_data = pd.DataFrame(
            np.random.randn(350, 4).cumsum(axis=0) + 100,
            index=self.dates,
            columns=self.stocks
        )
        self.normal_price_data = np.abs(self.normal_price_data)  # 确保价格为正
    
    def test_init(self):
        """测试初始化"""
        self.assertIsInstance(self.preprocessor.config, DynamicLowVolConfig)
        self.assertEqual(self.preprocessor.missing_threshold, 0.1)
        self.assertEqual(self.preprocessor.outlier_threshold, 5.0)
    
    def test_preprocess_price_data_normal(self):
        """测试正常价格数据预处理"""
        result = self.preprocessor.preprocess_price_data(self.normal_price_data)
        
        # 检查返回类型和形状
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, self.normal_price_data.shape)
        
        # 检查无缺失值
        self.assertFalse(result.isna().any().any())
        
        # 检查数据为正值
        self.assertTrue((result > 0).all().all())
    
    def test_preprocess_price_data_empty(self):
        """测试空数据处理"""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.preprocess_price_data(empty_data)
        
        self.assertIn("价格数据为空", str(context.exception))
    
    def test_preprocess_price_data_insufficient_length(self):
        """测试数据长度不足"""
        short_data = self.normal_price_data.iloc[:50]  # 只有50天数据
        
        with self.assertRaises(InsufficientDataException) as context:
            self.preprocessor.preprocess_price_data(short_data)
        
        self.assertIn("价格数据长度", str(context.exception))
        self.assertIn("不足", str(context.exception))
    
    def test_preprocess_price_data_high_missing_ratio(self):
        """测试高缺失值比例"""
        missing_data = self.normal_price_data.copy()
        # 为STOCK_A设置15%的缺失值（超过10%阈值）
        missing_indices = np.random.choice(
            missing_data.index, 
            size=int(len(missing_data) * 0.15), 
            replace=False
        )
        missing_data.loc[missing_indices, 'STOCK_A'] = np.nan
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.preprocess_price_data(missing_data)
        
        self.assertIn("缺失值比例超过", str(context.exception))
        self.assertIn("STOCK_A", str(context.exception))
    
    def test_preprocess_price_data_with_missing_values(self):
        """测试含少量缺失值的数据处理"""
        missing_data = self.normal_price_data.copy()
        # 设置5%的缺失值（低于10%阈值）
        missing_indices = np.random.choice(
            missing_data.index, 
            size=int(len(missing_data) * 0.05), 
            replace=False
        )
        missing_data.loc[missing_indices, 'STOCK_A'] = np.nan
        
        result = self.preprocessor.preprocess_price_data(missing_data)
        
        # 检查缺失值已被填充
        self.assertFalse(result.isna().any().any())
        
        # 检查数据形状不变
        self.assertEqual(result.shape, missing_data.shape)
    
    def test_calculate_returns_simple(self):
        """测试简单收益率计算"""
        returns = self.preprocessor.calculate_returns(
            self.normal_price_data, 
            return_type='simple'
        )
        
        # 检查返回类型和形状
        self.assertIsInstance(returns, pd.DataFrame)
        self.assertEqual(returns.shape[0], self.normal_price_data.shape[0] - 1)
        self.assertEqual(returns.shape[1], self.normal_price_data.shape[1])
        
        # 检查收益率计算正确性
        expected_returns = self.normal_price_data.pct_change().iloc[1:]
        pd.testing.assert_frame_equal(returns, expected_returns)
    
    def test_calculate_returns_log(self):
        """测试对数收益率计算"""
        returns = self.preprocessor.calculate_returns(
            self.normal_price_data, 
            return_type='log'
        )
        
        # 检查返回类型和形状
        self.assertIsInstance(returns, pd.DataFrame)
        self.assertEqual(returns.shape[0], self.normal_price_data.shape[0] - 1)
        
        # 检查对数收益率计算正确性
        expected_returns = np.log(self.normal_price_data / self.normal_price_data.shift(1)).iloc[1:]
        pd.testing.assert_frame_equal(returns, expected_returns)
    
    def test_calculate_returns_empty_data(self):
        """测试空数据收益率计算"""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.calculate_returns(empty_data)
        
        self.assertIn("价格数据为空", str(context.exception))
    
    def test_calculate_returns_invalid_type(self):
        """测试无效收益率类型"""
        with self.assertRaises(ConfigurationException) as context:
            self.preprocessor.calculate_returns(
                self.normal_price_data, 
                return_type='invalid'
            )
        
        self.assertIn("不支持的收益率类型", str(context.exception))
    
    def test_calculate_returns_negative_prices(self):
        """测试负价格数据"""
        negative_data = self.normal_price_data.copy()
        negative_data.iloc[0, 0] = -10  # 设置负价格
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.calculate_returns(negative_data)
        
        self.assertIn("价格数据包含非正值", str(context.exception))
    
    def test_prepare_rolling_windows_normal(self):
        """测试正常滚动窗口准备"""
        windows = [20, 60]
        result = self.preprocessor.prepare_rolling_windows(
            self.normal_price_data, 
            windows
        )
        
        # 检查返回类型
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), len(windows))
        
        # 检查每个窗口的数据
        for window in windows:
            self.assertIn(window, result)
            self.assertIsInstance(result[window], pd.DataFrame)
            self.assertEqual(result[window].shape, self.normal_price_data.shape)
    
    def test_prepare_rolling_windows_empty_data(self):
        """测试空数据滚动窗口准备"""
        empty_data = pd.DataFrame()
        windows = [20, 60]
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.prepare_rolling_windows(empty_data, windows)
        
        self.assertIn("输入数据为空", str(context.exception))
    
    def test_prepare_rolling_windows_insufficient_data(self):
        """测试数据长度不足的滚动窗口准备"""
        short_data = self.normal_price_data.iloc[:30]  # 只有30天数据
        windows = [20, 60]  # 需要60天数据
        
        with self.assertRaises(InsufficientDataException) as context:
            self.preprocessor.prepare_rolling_windows(short_data, windows)
        
        self.assertIn("数据长度", str(context.exception))
        self.assertIn("不足", str(context.exception))
    
    def test_validate_data_quality_normal(self):
        """测试正常数据质量验证"""
        # 正常数据不应抛出异常
        try:
            self.preprocessor.validate_data_quality(self.normal_price_data, "测试数据")
        except Exception as e:
            self.fail(f"正常数据验证失败: {e}")
    
    def test_validate_data_quality_empty(self):
        """测试空数据质量验证"""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.validate_data_quality(empty_data, "空数据")
        
        self.assertIn("空数据为空", str(context.exception))
    
    def test_validate_data_quality_wrong_type(self):
        """测试错误数据类型验证"""
        wrong_data = "not a dataframe"
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.validate_data_quality(wrong_data, "错误类型")
        
        self.assertIn("必须为DataFrame类型", str(context.exception))
    
    def test_validate_data_quality_high_missing_ratio(self):
        """测试高缺失值比例验证"""
        missing_data = self.normal_price_data.copy()
        # 设置15%的缺失值
        total_cells = missing_data.size
        missing_count = int(total_cells * 0.15)
        
        # 随机设置缺失值
        for _ in range(missing_count):
            row = np.random.randint(0, missing_data.shape[0])
            col = np.random.randint(0, missing_data.shape[1])
            missing_data.iloc[row, col] = np.nan
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.validate_data_quality(missing_data, "高缺失数据")
        
        self.assertIn("缺失值比例", str(context.exception))
        self.assertIn("超过阈值", str(context.exception))
    
    def test_validate_data_quality_infinite_values(self):
        """测试无穷值验证"""
        inf_data = self.normal_price_data.copy()
        inf_data.iloc[0, 0] = np.inf
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor.validate_data_quality(inf_data, "无穷值数据")
        
        self.assertIn("包含", str(context.exception))
        self.assertIn("个无穷值", str(context.exception))
    
    def test_handle_outliers(self):
        """测试异常值处理"""
        outlier_data = self.normal_price_data.copy()
        
        # 添加异常值
        outlier_data.iloc[10, 0] = outlier_data.iloc[10, 0] + 10 * outlier_data.iloc[:, 0].std()
        
        result = self.preprocessor._handle_outliers(outlier_data)
        
        # 检查异常值被处理
        self.assertNotEqual(result.iloc[10, 0], outlier_data.iloc[10, 0])
        
        # 检查其他值未变
        pd.testing.assert_series_equal(
            result.iloc[:, 1], 
            outlier_data.iloc[:, 1]
        )
    
    def test_validate_returns_normal(self):
        """测试正常收益率验证"""
        returns = self.preprocessor.calculate_returns(self.normal_price_data)
        
        # 正常收益率不应抛出异常
        try:
            validated_returns = self.preprocessor._validate_returns(returns)
            pd.testing.assert_frame_equal(returns, validated_returns)
        except Exception as e:
            self.fail(f"正常收益率验证失败: {e}")
    
    def test_validate_returns_extreme_values(self):
        """测试极端收益率验证"""
        extreme_data = self.normal_price_data.copy()
        
        # 创建大量极端收益率以超过1%阈值
        # 总观测值 = 349 * 4 = 1396，1%阈值 = 13.96，需要至少14个极端值
        extreme_indices = range(1, 20)  # 创建19个极端值
        
        for i in extreme_indices:
            # 创建>50%的极端收益率
            extreme_data.iloc[i, 0] = extreme_data.iloc[i-1, 0] * 2  # 100%收益率
            if i < 15:  # 为多个股票创建极端值
                extreme_data.iloc[i, 1] = extreme_data.iloc[i-1, 1] * 1.8  # 80%收益率
        
        returns = extreme_data.pct_change().iloc[1:]
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor._validate_returns(returns)
        
        self.assertIn("极端收益率", str(context.exception))
    
    def test_validate_returns_zero_variance(self):
        """测试零方差收益率验证"""
        constant_data = self.normal_price_data.copy()
        constant_data.iloc[:, 0] = 100  # 价格不变
        
        returns = constant_data.pct_change().iloc[1:]
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor._validate_returns(returns)
        
        self.assertIn("收益率方差为0", str(context.exception))
    
    def test_validate_returns_too_many_zeros(self):
        """测试过多零收益率验证"""
        zero_data = self.normal_price_data.copy()
        
        # 设置大量零收益率（模拟停牌）
        for i in range(1, int(len(zero_data) * 0.6)):
            zero_data.iloc[i, 0] = zero_data.iloc[i-1, 0]  # 价格不变
        
        returns = zero_data.pct_change().iloc[1:]
        
        with self.assertRaises(DataQualityException) as context:
            self.preprocessor._validate_returns(returns)
        
        self.assertIn("零收益率比例", str(context.exception))
        self.assertIn("过高", str(context.exception))
    
    def test_integration_full_preprocessing_pipeline(self):
        """测试完整预处理流水线集成"""
        # 创建带有少量问题的数据
        test_data = self.normal_price_data.copy()
        
        # 添加少量缺失值
        test_data.iloc[10:12, 0] = np.nan
        
        # 添加轻微异常值
        test_data.iloc[50, 1] = test_data.iloc[50, 1] * 3
        
        # 执行完整预处理流水线
        cleaned_data = self.preprocessor.preprocess_price_data(test_data)
        returns = self.preprocessor.calculate_returns(cleaned_data)
        rolling_data = self.preprocessor.prepare_rolling_windows(
            returns, 
            self.config.rolling_windows
        )
        
        # 验证结果
        self.assertFalse(cleaned_data.isna().any().any())
        self.assertEqual(returns.shape[0], cleaned_data.shape[0] - 1)
        self.assertEqual(len(rolling_data), len(self.config.rolling_windows))
        
        # 验证数据质量
        self.preprocessor.validate_data_quality(cleaned_data, "清洗后数据")
        self.preprocessor.validate_data_quality(returns, "收益率数据")


if __name__ == '__main__':
    unittest.main()