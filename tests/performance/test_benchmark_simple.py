#!/usr/bin/env python3
"""
简化的性能基准测试

专注于测试基本计算操作的性能，不依赖完整的筛选器集成。
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import unittest
from typing import Dict, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.unit.test_backtest_utils import BacktestDataGenerator, BacktestMetricsCalculator


class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试"""
    
    def setUp(self):
        """测试前准备"""
        self.performance_threshold_ms = 100  # 100ms性能要求
        
        # 生成测试数据
        self.dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        self.dates = self.dates[self.dates.weekday < 5]  # 只保留工作日
        self.stocks = [f'stock_{i:04d}' for i in range(1000)]  # 1000只股票
        
        # 生成真实规模的数据
        np.random.seed(42)
        self.returns_data = BacktestDataGenerator.generate_realistic_stock_returns(
            self.dates, self.stocks
        )
        self.price_data = (1 + self.returns_data).cumprod() * 100
        
    def test_data_generation_performance(self):
        """测试数据生成性能"""
        # 测试大规模数据生成时间
        test_stocks = [f'test_stock_{i}' for i in range(500)]
        test_dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        test_dates = test_dates[test_dates.weekday < 5]  # 工作日
        
        start_time = time.perf_counter()
        
        generated_data = BacktestDataGenerator.generate_realistic_stock_returns(
            test_dates, test_stocks
        )
        
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # 验证结果
        self.assertIsInstance(generated_data, pd.DataFrame)
        self.assertEqual(generated_data.shape, (len(test_dates), len(test_stocks)))
        
        # 性能要求：500股票×250个交易日的数据生成应在合理时间内完成
        self.assertLess(execution_time_ms, 5000, f"数据生成耗时 {execution_time_ms:.2f}ms 超过5秒阈值")
        
        print(f"数据生成性能: {len(test_stocks)}只股票×{len(test_dates)}天 = {execution_time_ms:.2f}ms")
        
    def test_metrics_calculation_performance(self):
        """测试指标计算性能"""
        # 测试大量股票的指标计算
        execution_times = []
        
        for i in range(10):  # 多次测试
            # 随机选择一只股票的收益率数据
            stock_returns = self.returns_data.iloc[:, i].values
            
            start_time = time.perf_counter()
            
            metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(stock_returns)
            
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            execution_times.append(execution_time_ms)
            
            # 验证结果完整性
            self.assertIsInstance(metrics, dict)
            self.assertIn('annual_return', metrics)
            self.assertIn('sharpe_ratio', metrics)
            self.assertTrue(np.isfinite(metrics['annual_return']))
        
        avg_time = np.mean(execution_times)
        max_time = np.max(execution_times)
        
        # 性能要求：单个股票指标计算应在很短时间内完成
        self.assertLess(avg_time, 50, f"指标计算平均耗时 {avg_time:.2f}ms 超过50ms阈值")
        self.assertLess(max_time, 100, f"指标计算最大耗时 {max_time:.2f}ms 超过100ms阈值")
        
        print(f"指标计算性能: 平均 {avg_time:.2f}ms, 最大 {max_time:.2f}ms")
        
    def test_vectorized_operations_performance(self):
        """测试向量化操作性能"""
        # 测试大规模向量化计算
        data_matrix = self.returns_data.values  # 转为numpy数组
        
        # 测试滚动窗口计算
        window_size = 20
        start_time = time.perf_counter()
        
        # 计算滚动标准差（模拟波动率计算）
        rolling_std = pd.DataFrame(data_matrix, columns=self.stocks).rolling(window_size).std()
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # 验证结果
        self.assertIsInstance(rolling_std, pd.DataFrame)
        self.assertEqual(rolling_std.shape, data_matrix.shape)
        
        # 性能要求：1000只股票的滚动计算应在合理时间内完成
        self.assertLess(execution_time_ms, 1000, f"滚动计算耗时 {execution_time_ms:.2f}ms 应小于1000ms")
        
        print(f"向量化操作性能: {len(self.stocks)}只股票滚动计算 = {execution_time_ms:.2f}ms")
        
    def test_percentile_calculation_performance(self):
        """测试分位数计算性能"""
        # 测试大规模分位数计算（模拟筛选逻辑中的关键操作）
        data_matrix = self.returns_data.values
        
        percentiles = [5, 25, 50, 75, 95]
        
        start_time = time.perf_counter()
        
        # 计算各分位数
        percentile_results = {}
        for p in percentiles:
            percentile_results[p] = np.percentile(data_matrix, p, axis=0)
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # 验证结果
        for p in percentiles:
            self.assertEqual(len(percentile_results[p]), len(self.stocks))
            self.assertTrue(np.all(np.isfinite(percentile_results[p])))
        
        # 性能要求：分位数计算应该很快
        self.assertLess(execution_time_ms, 200, f"分位数计算耗时 {execution_time_ms:.2f}ms 应小于200ms")
        
        print(f"分位数计算性能: {len(self.stocks)}只股票×{len(percentiles)}个分位数 = {execution_time_ms:.2f}ms")
        
    def test_correlation_calculation_performance(self):
        """测试相关性计算性能"""
        # 测试股票间相关性计算（模拟市场状态分析）
        # 为了控制计算时间，只选择部分股票
        subset_data = self.returns_data.iloc[:, :100]  # 100只股票
        
        start_time = time.perf_counter()
        
        correlation_matrix = subset_data.corr()
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # 验证结果
        self.assertEqual(correlation_matrix.shape, (100, 100))
        self.assertTrue(np.allclose(np.diag(correlation_matrix), 1.0))  # 对角线应为1
        
        # 性能要求：100×100相关性矩阵计算应在合理时间内
        self.assertLess(execution_time_ms, 500, f"相关性计算耗时 {execution_time_ms:.2f}ms 应小于500ms")
        
        print(f"相关性计算性能: 100×100矩阵 = {execution_time_ms:.2f}ms")
        
    def test_large_dataset_memory_efficiency(self):
        """测试大数据集的内存效率"""
        # 监控内存使用情况
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 获取初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建大规模数据集
        large_dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        large_dates = large_dates[large_dates.weekday < 5]
        large_stocks = [f'stock_{i:05d}' for i in range(2000)]  # 2000只股票
        
        # 分批生成数据以控制内存使用
        batch_size = 500
        all_data = []
        
        for i in range(0, len(large_stocks), batch_size):
            batch_stocks = large_stocks[i:i+batch_size]
            batch_data = BacktestDataGenerator.generate_realistic_stock_returns(
                large_dates[:252], batch_stocks  # 只用一年数据测试
            )
            all_data.append(batch_data)
            
            # 强制垃圾回收
            gc.collect()
        
        # 合并数据
        combined_data = pd.concat(all_data, axis=1)
        
        # 获取峰值内存使用
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # 验证数据完整性
        self.assertEqual(combined_data.shape[1], len(large_stocks))
        self.assertEqual(combined_data.shape[0], 252)
        
        # 内存使用合理性检查（应该不超过合理范围）
        expected_memory_mb = len(large_stocks) * 252 * 8 / 1024 / 1024  # 粗略估算（8字节/float64）
        self.assertLess(memory_increase, expected_memory_mb * 5, 
                       f"内存使用 {memory_increase:.1f}MB 超过预期 {expected_memory_mb * 5:.1f}MB")
        
        print(f"内存效率测试: {len(large_stocks)}只股票×{252}天, 内存增加 {memory_increase:.1f}MB")
        
        # 清理数据
        del combined_data, all_data
        gc.collect()
        
    def test_computation_stability(self):
        """测试计算稳定性"""
        # 测试在不同数据条件下的计算稳定性
        test_cases = [
            ("normal", np.random.normal(0, 0.02, 252)),
            ("high_vol", np.random.normal(0, 0.05, 252)),
            ("trending", np.linspace(-0.001, 0.001, 252) + np.random.normal(0, 0.01, 252)),
            ("volatile", np.random.normal(0, 0.01, 252) * np.random.choice([-1, 1], 252))
        ]
        
        for case_name, returns in test_cases:
            with self.subTest(case=case_name):
                start_time = time.perf_counter()
                
                metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(returns)
                
                end_time = time.perf_counter()
                execution_time_ms = (end_time - start_time) * 1000
                
                # 验证计算结果的合理性和稳定性
                self.assertTrue(np.isfinite(metrics['annual_return']))
                self.assertTrue(np.isfinite(metrics['annual_volatility']))
                self.assertTrue(np.isfinite(metrics['sharpe_ratio']))
                self.assertGreaterEqual(metrics['annual_volatility'], 0)
                self.assertGreaterEqual(metrics['max_drawdown'], 0)
                self.assertLessEqual(metrics['win_rate'], 1)
                
                # 性能应该稳定
                self.assertLess(execution_time_ms, 100, 
                               f"案例 {case_name} 计算耗时 {execution_time_ms:.2f}ms 超过100ms")
                
                print(f"稳定性测试 {case_name}: {execution_time_ms:.2f}ms")


if __name__ == '__main__':
    unittest.main()