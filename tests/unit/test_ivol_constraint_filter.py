"""
IVOL约束筛选器单元测试

测试IVOLConstraintFilter类的各项功能，包括：
- 五因子回归分解特异性波动
- 好波动和坏波动的区分逻辑
- IVOL双重约束筛选功能
- 异常处理和边界条件
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
    IVOLConstraintFilter,
    DynamicLowVolConfig,
    DataQualityException,
    InsufficientDataException,
    ModelFittingException,
    ConfigurationException
)


class TestIVOLConstraintFilter(unittest.TestCase):
    """IVOL约束筛选器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试配置
        self.config = DynamicLowVolConfig(
            ivol_bad_threshold=0.3,
            ivol_good_threshold=0.6,
            enable_caching=True
        )
        
        # 创建筛选器实例
        self.filter = IVOLConstraintFilter(self.config)
        
        # 创建测试数据
        self.dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']
        
        # 生成模拟收益率数据
        np.random.seed(42)
        returns_data = np.random.normal(0, 0.02, (len(self.dates), len(self.stocks)))
        self.returns = pd.DataFrame(
            returns_data, 
            index=self.dates, 
            columns=self.stocks
        )
        
        # 生成模拟因子数据
        factor_data = np.random.normal(0, 0.01, (len(self.dates), 3))
        self.factor_data = pd.DataFrame(
            factor_data,
            index=self.dates,
            columns=['factor_1', 'factor_2', 'factor_3']
        )
        
        # 生成市场数据
        market_data = np.random.normal(0, 0.015, len(self.dates))
        self.market_data = pd.DataFrame(
            market_data,
            index=self.dates,
            columns=['market_index']
        )
        
        self.current_date = self.dates[-1]
    
    def test_init(self):
        """测试初始化"""
        # 测试正常初始化
        filter_obj = IVOLConstraintFilter(self.config)
        self.assertEqual(filter_obj.ivol_bad_threshold, 0.3)
        self.assertEqual(filter_obj.ivol_good_threshold, 0.6)
        self.assertIsNotNone(filter_obj._ivol_cache)
        self.assertIsNotNone(filter_obj._factor_cache)
        
        # 测试禁用缓存
        config_no_cache = DynamicLowVolConfig(enable_caching=False)
        filter_no_cache = IVOLConstraintFilter(config_no_cache)
        self.assertIsNone(filter_no_cache._ivol_cache)
        self.assertIsNone(filter_no_cache._factor_cache)
    
    def test_apply_ivol_constraint_normal(self):
        """测试正常的IVOL约束筛选"""
        result = self.filter.apply_ivol_constraint(
            self.returns,
            self.factor_data,
            self.current_date,
            self.market_data
        )
        
        # 验证结果格式
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.stocks))
        self.assertEqual(result.dtype, bool)
        
        # 验证至少有一些股票通过筛选
        self.assertTrue(result.sum() > 0)
        self.assertTrue(result.sum() < len(self.stocks))
    
    def test_apply_ivol_constraint_without_market_data(self):
        """测试不提供市场数据的IVOL约束筛选"""
        result = self.filter.apply_ivol_constraint(
            self.returns,
            self.factor_data,
            self.current_date,
            market_data=None
        )
        
        # 验证结果格式
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(self.stocks))
        self.assertEqual(result.dtype, bool)
    
    def test_decompose_ivol_normal(self):
        """测试正常的IVOL分解"""
        # 构建五因子数据
        five_factors = self.filter._construct_five_factors(
            self.returns, self.factor_data, self.market_data, self.current_date
        )
        
        # 分解IVOL
        ivol_good, ivol_bad = self.filter.decompose_ivol(self.returns, five_factors)
        
        # 验证结果格式
        self.assertIsInstance(ivol_good, pd.Series)
        self.assertIsInstance(ivol_bad, pd.Series)
        self.assertEqual(len(ivol_good), len(self.stocks))
        self.assertEqual(len(ivol_bad), len(self.stocks))
        
        # 验证所有值都是非负的
        self.assertTrue((ivol_good >= 0).all())
        self.assertTrue((ivol_bad >= 0).all())
        
        # 验证没有无穷值
        self.assertFalse(np.isinf(ivol_good).any())
        self.assertFalse(np.isinf(ivol_bad).any())
    
    def test_decompose_ivol_with_missing_data(self):
        """测试包含缺失数据的IVOL分解"""
        # 在收益率数据中引入缺失值
        returns_with_na = self.returns.copy()
        returns_with_na.iloc[10:15, 0] = np.nan
        returns_with_na.iloc[20:25, 1] = np.nan
        
        five_factors = self.filter._construct_five_factors(
            returns_with_na, self.factor_data, self.market_data, self.current_date
        )
        
        ivol_good, ivol_bad = self.filter.decompose_ivol(returns_with_na, five_factors)
        
        # 验证结果仍然有效
        self.assertEqual(len(ivol_good), len(self.stocks))
        self.assertEqual(len(ivol_bad), len(self.stocks))
        self.assertFalse(ivol_good.isna().all())
        self.assertFalse(ivol_bad.isna().all())
    
    def test_construct_five_factors(self):
        """测试五因子构建"""
        five_factors = self.filter._construct_five_factors(
            self.returns, self.factor_data, self.market_data, self.current_date
        )
        
        # 验证结果格式
        self.assertIsInstance(five_factors, pd.DataFrame)
        self.assertEqual(len(five_factors.columns), 5)
        expected_columns = ['Market', 'Size', 'Value', 'Profitability', 'Investment']
        self.assertListEqual(list(five_factors.columns), expected_columns)
        
        # 验证没有缺失值
        self.assertFalse(five_factors.isna().any().any())
        
        # 验证数据长度合理
        self.assertTrue(len(five_factors) > 0)
        self.assertTrue(len(five_factors) <= len(self.returns))
    
    def test_construct_five_factors_without_market_data(self):
        """测试不提供市场数据时的五因子构建"""
        five_factors = self.filter._construct_five_factors(
            self.returns, self.factor_data, None, self.current_date
        )
        
        # 验证结果格式
        self.assertIsInstance(five_factors, pd.DataFrame)
        self.assertEqual(len(five_factors.columns), 5)
        
        # 市场因子应该是等权重市场收益率
        market_factor = five_factors['Market']
        expected_market = self.returns.mean(axis=1)
        
        # 验证市场因子的合理性（允许一定误差）
        self.assertTrue(len(market_factor) > 0)
        self.assertFalse(market_factor.isna().any())
    
    def test_decompose_residual_volatility(self):
        """测试残差波动分解"""
        # 创建测试残差数据
        residuals = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005, -0.025, 0.02])
        
        good_vol, bad_vol = self.filter._decompose_residual_volatility(residuals)
        
        # 验证结果类型
        self.assertIsInstance(good_vol, float)
        self.assertIsInstance(bad_vol, float)
        
        # 验证结果为非负
        self.assertGreaterEqual(good_vol, 0)
        self.assertGreaterEqual(bad_vol, 0)
        
        # 验证结果不是无穷值
        self.assertFalse(np.isinf(good_vol))
        self.assertFalse(np.isinf(bad_vol))
    
    def test_decompose_residual_volatility_edge_cases(self):
        """测试残差波动分解的边界情况"""
        # 测试只有正残差
        positive_residuals = pd.Series([0.01, 0.02, 0.015])
        good_vol, bad_vol = self.filter._decompose_residual_volatility(positive_residuals)
        self.assertGreater(good_vol, 0)
        self.assertEqual(bad_vol, 0)
        
        # 测试只有负残差
        negative_residuals = pd.Series([-0.01, -0.02, -0.015])
        good_vol, bad_vol = self.filter._decompose_residual_volatility(negative_residuals)
        self.assertEqual(good_vol, 0)
        self.assertGreater(bad_vol, 0)
        
        # 测试单个残差
        single_residual = pd.Series([0.01])
        good_vol, bad_vol = self.filter._decompose_residual_volatility(single_residual)
        self.assertEqual(good_vol, 0)  # 单个值无法计算标准差
        self.assertEqual(bad_vol, 0)
    
    def test_calculate_ivol_percentiles(self):
        """测试IVOL分位数计算"""
        # 创建测试IVOL数据
        ivol_data = pd.Series([0.1, 0.2, 0.15, 0.3, 0.25], index=self.stocks)
        
        percentiles = self.filter._calculate_ivol_percentiles(ivol_data)
        
        # 验证结果格式
        self.assertIsInstance(percentiles, pd.Series)
        self.assertEqual(len(percentiles), len(self.stocks))
        
        # 验证分位数范围
        self.assertTrue((percentiles >= 0).all())
        self.assertTrue((percentiles <= 1).all())
        
        # 验证排序正确性
        sorted_ivol = ivol_data.sort_values()
        sorted_percentiles = percentiles.loc[sorted_ivol.index]
        self.assertTrue(sorted_percentiles.is_monotonic_increasing)
    
    def test_calculate_ivol_percentiles_with_missing(self):
        """测试包含缺失值的IVOL分位数计算"""
        # 创建包含缺失值的IVOL数据
        ivol_data = pd.Series([0.1, np.nan, 0.15, 0.3, np.nan], index=self.stocks)
        
        percentiles = self.filter._calculate_ivol_percentiles(ivol_data)
        
        # 验证结果格式
        self.assertEqual(len(percentiles), len(self.stocks))
        
        # 验证缺失值被设置为1.0
        self.assertEqual(percentiles.iloc[1], 1.0)  # 第二个股票
        self.assertEqual(percentiles.iloc[4], 1.0)  # 第五个股票
        
        # 验证非缺失值的分位数正确
        valid_percentiles = percentiles.dropna()
        valid_percentiles = valid_percentiles[valid_percentiles < 1.0]
        self.assertTrue(len(valid_percentiles) > 0)
    
    def test_fill_missing_ivol(self):
        """测试IVOL缺失值填充"""
        # 创建包含缺失值的IVOL数据
        ivol_data = pd.Series([0.1, np.nan, 0.15, np.nan, 0.2], index=self.stocks)
        
        # 测试好波动填充
        filled_good = self.filter._fill_missing_ivol(ivol_data, self.returns, 'good')
        self.assertFalse(filled_good.isna().any())
        self.assertTrue((filled_good > 0).all())
        
        # 测试坏波动填充
        filled_bad = self.filter._fill_missing_ivol(ivol_data, self.returns, 'bad')
        self.assertFalse(filled_bad.isna().any())
        self.assertTrue((filled_bad > 0).all())
        
        # 验证好波动的填充值小于坏波动
        for i in range(len(self.stocks)):
            if ivol_data.iloc[i] != ivol_data.iloc[i]:  # 检查是否为NaN
                self.assertLess(filled_good.iloc[i], filled_bad.iloc[i])
    
    def test_get_ivol_statistics(self):
        """测试IVOL统计信息获取"""
        stats = self.filter.get_ivol_statistics(
            self.returns, self.factor_data, self.current_date
        )
        
        # 验证统计信息结构
        expected_keys = [
            'ivol_good_mean', 'ivol_good_std', 'ivol_good_median',
            'ivol_bad_mean', 'ivol_bad_std', 'ivol_bad_median',
            'good_bad_correlation', 'valid_stocks_count', 'total_stocks_count'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # 验证统计值的合理性
        self.assertGreaterEqual(stats['ivol_good_mean'], 0)
        self.assertGreaterEqual(stats['ivol_bad_mean'], 0)
        self.assertGreaterEqual(stats['valid_stocks_count'], 0)
        self.assertEqual(stats['total_stocks_count'], len(self.stocks))
        
        # 验证相关系数在合理范围内
        if not np.isnan(stats['good_bad_correlation']):
            self.assertGreaterEqual(stats['good_bad_correlation'], -1)
            self.assertLessEqual(stats['good_bad_correlation'], 1)
    
    def test_validate_input_data_normal(self):
        """测试正常输入数据验证"""
        # 正常数据应该不抛出异常
        try:
            self.filter._validate_input_data(
                self.returns, self.factor_data, self.current_date
            )
        except Exception as e:
            self.fail(f"正常数据验证失败: {e}")
    
    def test_validate_input_data_empty_returns(self):
        """测试空收益率数据验证"""
        empty_returns = pd.DataFrame()
        
        with self.assertRaises(DataQualityException) as context:
            self.filter._validate_input_data(
                empty_returns, self.factor_data, self.current_date
            )
        
        self.assertIn("收益率数据为空", str(context.exception))
    
    def test_validate_input_data_empty_factors(self):
        """测试空因子数据验证"""
        empty_factors = pd.DataFrame()
        
        with self.assertRaises(DataQualityException) as context:
            self.filter._validate_input_data(
                self.returns, empty_factors, self.current_date
            )
        
        self.assertIn("因子数据为空", str(context.exception))
    
    def test_validate_input_data_insufficient_length(self):
        """测试数据长度不足验证"""
        # 创建长度不足的数据
        short_returns = self.returns.iloc[:30]  # 只有30天数据
        short_factors = self.factor_data.iloc[:30]
        
        with self.assertRaises(InsufficientDataException) as context:
            self.filter._validate_input_data(
                short_returns, short_factors, short_returns.index[-1]
            )
        
        self.assertIn("不足，需要至少60个观测值", str(context.exception))
    
    def test_validate_input_data_misaligned_data(self):
        """测试数据不对齐验证"""
        # 创建不对齐的数据
        misaligned_factors = self.factor_data.iloc[50:]  # 因子数据从中间开始
        
        with self.assertRaises(InsufficientDataException) as context:
            self.filter._validate_input_data(
                self.returns, misaligned_factors, self.current_date
            )
        
        self.assertIn("对齐后长度", str(context.exception))
    
    def test_validate_input_data_invalid_date(self):
        """测试无效日期验证"""
        invalid_date = pd.Timestamp('2025-01-01')  # 超出数据范围的日期
        
        with self.assertRaises(DataQualityException) as context:
            self.filter._validate_input_data(
                self.returns, self.factor_data, invalid_date
            )
        
        self.assertIn("超出收益率数据范围", str(context.exception))
    
    def test_validate_constraint_result_normal(self):
        """测试正常约束结果验证"""
        # 创建正常的约束结果
        constraint_mask = pd.Series([True, False, True, False, True], index=self.stocks)
        
        # 正常结果应该不抛出异常
        try:
            self.filter._validate_constraint_result(constraint_mask, self.returns.columns)
        except Exception as e:
            self.fail(f"正常约束结果验证失败: {e}")
    
    def test_validate_constraint_result_length_mismatch(self):
        """测试约束结果长度不匹配验证"""
        # 创建长度不匹配的约束结果
        wrong_length_mask = pd.Series([True, False, True])  # 长度为3，但股票有5只
        
        with self.assertRaises(DataQualityException) as context:
            self.filter._validate_constraint_result(wrong_length_mask, self.returns.columns)
        
        self.assertIn("长度", str(context.exception))
        self.assertIn("不匹配", str(context.exception))
    
    def test_validate_constraint_result_empty_selection(self):
        """测试空筛选结果验证"""
        # 创建全为False的约束结果
        empty_mask = pd.Series([False] * len(self.stocks), index=self.stocks)
        
        with self.assertRaises(DataQualityException) as context:
            self.filter._validate_constraint_result(empty_mask, self.returns.columns)
        
        self.assertIn("筛选结果为空", str(context.exception))
    
    def test_validate_constraint_result_too_high_selection(self):
        """测试筛选比例过高验证"""
        # 创建筛选比例过高的约束结果（>90%）
        high_selection_mask = pd.Series([True] * len(self.stocks), index=self.stocks)
        
        with self.assertRaises(DataQualityException) as context:
            self.filter._validate_constraint_result(high_selection_mask, self.returns.columns)
        
        self.assertIn("筛选比例", str(context.exception))
        self.assertIn("过高", str(context.exception))
    
    def test_validate_constraint_result_wrong_dtype(self):
        """测试约束结果数据类型错误验证"""
        # 创建非布尔类型的约束结果
        wrong_dtype_mask = pd.Series([1, 0, 1, 0, 1], index=self.stocks)  # 整数类型
        
        with self.assertRaises(DataQualityException) as context:
            self.filter._validate_constraint_result(wrong_dtype_mask, self.returns.columns)
        
        self.assertIn("必须为布尔类型", str(context.exception))
    
    def test_caching_mechanism(self):
        """测试缓存机制"""
        # 启用缓存的筛选器
        cached_filter = IVOLConstraintFilter(self.config)
        
        # 第一次调用
        result1 = cached_filter.apply_ivol_constraint(
            self.returns, self.factor_data, self.current_date, self.market_data
        )
        
        # 检查缓存是否被初始化（可能为空但不为None）
        self.assertIsNotNone(cached_filter._factor_cache)
        
        # 第二次调用相同参数
        result2 = cached_filter.apply_ivol_constraint(
            self.returns, self.factor_data, self.current_date, self.market_data
        )
        
        # 结果应该相同
        np.testing.assert_array_equal(result1, result2)
        
        # 验证缓存机制工作正常（通过检查缓存字典存在）
        self.assertIsInstance(cached_filter._factor_cache, dict)
        self.assertIsInstance(cached_filter._ivol_cache, dict)
    
    def test_performance_with_large_dataset(self):
        """测试大数据集性能"""
        # 创建更大的测试数据集
        large_dates = pd.date_range('2022-01-01', periods=500, freq='D')
        large_stocks = [f'STOCK_{i:03d}' for i in range(100)]
        
        # 生成大数据集
        np.random.seed(42)
        large_returns = pd.DataFrame(
            np.random.normal(0, 0.02, (len(large_dates), len(large_stocks))),
            index=large_dates,
            columns=large_stocks
        )
        
        large_factors = pd.DataFrame(
            np.random.normal(0, 0.01, (len(large_dates), 5)),
            index=large_dates,
            columns=[f'factor_{i}' for i in range(5)]
        )
        
        # 测试性能（应该在合理时间内完成）
        import time
        start_time = time.time()
        
        result = self.filter.apply_ivol_constraint(
            large_returns, large_factors, large_dates[-1]
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证结果
        self.assertEqual(len(result), len(large_stocks))
        self.assertTrue(result.sum() > 0)
        
        # 性能要求：处理100只股票500天数据应在10秒内完成
        self.assertLess(execution_time, 10.0, f"执行时间{execution_time:.2f}秒过长")


if __name__ == '__main__':
    unittest.main()