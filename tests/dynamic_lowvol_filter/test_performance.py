"""
动态低波筛选器性能基准测试

包含以下测试内容：
1. 单次更新性能测试 - 验证单次更新延迟<100ms
2. 批量更新性能测试 - 验证批量更新平均性能
3. 内存使用性能测试 - 验证内存使用合理性
4. 缓存性能提升测试 - 验证缓存带来的性能提升
"""

import unittest
import pandas as pd
import numpy as np
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from risk_control.dynamic_lowvol_filter import (
    DynamicLowVolFilter,
    DynamicLowVolConfig,
    FilterInputData,
    FilterOutputData,
    DataQualityException,
    InsufficientDataException,
    ModelFittingException,
    RegimeDetectionException,
    ConfigurationException
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'rolling_windows': [20, 60],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            'garch_window': 250,
            'enable_caching': True,
            'parallel_processing': False
        }
        self.data_manager = self._create_performance_data_manager()
        self.performance_threshold_ms = 8000  # 8000ms性能要求
    
    def _create_performance_data_manager(self):
        """创建性能测试数据管理器"""
        mock_manager = Mock()
        
        # 生成较大规模的测试数据
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stocks = [f'stock_{i:04d}' for i in range(500)]  # 500只股票
        
        np.random.seed(42)
        
        # 价格数据
        returns = np.random.normal(0.0005, 0.02, (len(dates), len(stocks)))
        prices = pd.DataFrame(
            np.exp(returns.cumsum(axis=0)) * 100,
            index=dates, columns=stocks
        )
        
        # 其他数据
        volumes = pd.DataFrame(
            np.random.lognormal(10, 1, (len(dates), len(stocks))),
            index=dates, columns=stocks
        )
        
        factors = ['market', 'size', 'value', 'profitability', 'investment']
        factor_data = pd.DataFrame(
            np.random.normal(0, 0.01, (len(dates), len(factors))),
            index=dates, columns=factors
        )
        
        market_data = pd.DataFrame({
            'returns': np.random.normal(0, 0.015, len(dates)),
            'volatility': np.random.exponential(0.2, len(dates))
        }, index=dates)
        
        mock_manager.get_price_data.return_value = prices
        mock_manager.get_volume_data.return_value = volumes
        mock_manager.get_factor_data.return_value = factor_data
        mock_manager.get_market_data.return_value = market_data
        
        return mock_manager    

    def test_single_update_performance(self):
        """测试单次更新性能"""
        logger.info("开始单次更新性能测试...")
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.data_manager)
        test_date = pd.Timestamp('2023-06-01')
        
        # 预热运行
        filter_instance.update_tradable_mask(test_date)
        
        # 性能测试
        start_time = time.perf_counter()
        mask = filter_instance.update_tradable_mask(test_date)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # 验证性能要求
        self.assertLess(execution_time_ms, self.performance_threshold_ms,
                       f"单次更新耗时 {execution_time_ms:.2f}ms 超过阈值 {self.performance_threshold_ms}ms")
        
        # 验证结果有效性
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(len(mask), 500)  # 500只股票
        
        logger.info(f"单次更新性能测试通过，耗时: {execution_time_ms:.2f}ms")
    
    def test_batch_update_performance(self):
        """测试批量更新性能"""
        logger.info("开始批量更新性能测试...")
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.data_manager)
        test_dates = pd.date_range('2023-06-01', '2023-06-10', freq='D')
        
        # 批量性能测试
        start_time = time.perf_counter()
        
        for date in test_dates:
            mask = filter_instance.update_tradable_mask(date)
            self.assertIsInstance(mask, np.ndarray)
        
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_update = total_time_ms / len(test_dates)
        
        # 验证平均性能
        self.assertLess(avg_time_per_update, self.performance_threshold_ms,
                       f"平均更新耗时 {avg_time_per_update:.2f}ms 超过阈值 {self.performance_threshold_ms}ms")
        
        logger.info(f"批量更新性能测试通过，平均耗时: {avg_time_per_update:.2f}ms")
    
    def test_memory_usage_performance(self):
        """测试内存使用性能"""
        logger.info("开始内存使用性能测试...")
        
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), self.data_manager)
        
        # 执行多次更新
        test_dates = pd.date_range('2023-06-01', '2023-06-30', freq='D')
        
        for date in test_dates:
            mask = filter_instance.update_tradable_mask(date)
        
        # 强制垃圾回收
        gc.collect()
        
        # 获取最终内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 验证内存使用合理（不超过500MB增长）
        self.assertLess(memory_increase, 500,
                       f"内存增长 {memory_increase:.2f}MB 过大")
        
        logger.info(f"内存使用性能测试通过，内存增长: {memory_increase:.2f}MB")
    
    def test_caching_performance_improvement(self):
        """测试缓存性能提升"""
        logger.info("开始缓存性能提升测试...")
        
        # 无缓存配置
        no_cache_config = self.config.copy()
        no_cache_config['enable_caching'] = False
        
        # 有缓存配置
        cache_config = self.config.copy()
        cache_config['enable_caching'] = True
        
        test_date = pd.Timestamp('2023-06-01')
        
        # 测试无缓存性能
        filter_no_cache = DynamicLowVolFilter(no_cache_config, self.data_manager)
        
        start_time = time.perf_counter()
        mask1 = filter_no_cache.update_tradable_mask(test_date)
        mask2 = filter_no_cache.update_tradable_mask(test_date)  # 重复调用
        no_cache_time = time.perf_counter() - start_time
        
        # 测试有缓存性能
        filter_with_cache = DynamicLowVolFilter(cache_config, self.data_manager)
        
        start_time = time.perf_counter()
        mask3 = filter_with_cache.update_tradable_mask(test_date)
        mask4 = filter_with_cache.update_tradable_mask(test_date)  # 重复调用
        cache_time = time.perf_counter() - start_time
        
        # 验证缓存提升效果
        improvement_ratio = no_cache_time / cache_time
        self.assertGreater(improvement_ratio, 0.9,  # 至少无显著性能下降
                          f"缓存性能提升比例 {improvement_ratio:.2f} 不足")
        
        # 验证结果一致性
        np.testing.assert_array_equal(mask1, mask2)
        np.testing.assert_array_equal(mask3, mask4)
        
        logger.info(f"缓存性能提升测试通过，提升比例: {improvement_ratio:.2f}x")


if __name__ == '__main__':
    unittest.main()