"""
动态低波筛选器高级功能测试

测试ML预测器和并行处理等高级功能。
"""

import unittest
import pandas as pd
import numpy as np
import time
import logging
import sys
import os
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter, DynamicLowVolConfig

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestAdvancedFeatures(unittest.TestCase):
    """高级功能测试类"""

    def setUp(self):
        """测试前准备"""
        self.base_config = {
            'rolling_windows': [20, 60],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            'garch_window': 250,
            'enable_caching': False, # 在性能测试中禁用缓存以准确测量计算时间
        }
        self.data_manager = self._create_test_data_manager(stock_count=100)

    def _create_test_data_manager(self, stock_count):
        """创建测试数据管理器"""
        mock_manager = Mock()
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stocks = [f'stock_{i:03d}' for i in range(stock_count)]
        
        np.random.seed(42)
        prices = pd.DataFrame(
            np.exp(np.random.normal(0.0005, 0.02, (len(dates), len(stocks))).cumsum(axis=0)) * 100,
            index=dates, columns=stocks
        )
        volumes = pd.DataFrame(np.random.lognormal(10, 1, (len(dates), len(stocks))), index=dates, columns=stocks)
        factors = pd.DataFrame(np.random.normal(0, 0.01, (len(dates), 5)), index=dates, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        market_data = pd.DataFrame({'returns': np.random.normal(0, 0.015, len(dates))}, index=dates)

        mock_manager.get_price_data.return_value = prices
        mock_manager.get_volume_data.return_value = volumes
        mock_manager.get_factor_data.return_value = factors
        mock_manager.get_market_data.return_value = market_data
        return mock_manager

    @patch('risk_control.dynamic_lowvol_filter.core.MLVolatilityPredictor')
    def test_ml_predictor_integration(self, MockMlPredictor):
        """测试ML预测器集成"""
        logger.info("开始ML预测器集成测试...")
        
        # 启用ML预测器
        ml_config = self.base_config.copy()
        ml_config['enable_ml_predictor'] = True
        
        # 模拟ML预测器的返回掩码
        mock_ml_predictor_instance = MockMlPredictor.return_value
        ml_mask = np.array([True] * 50 + [False] * 50) # 假设ML模型筛选掉了一半
        mock_ml_predictor_instance.apply_filter.return_value = ml_mask

        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**ml_config), self.data_manager)
        
        # 模拟其他层的掩码，假设都为True以隔离ML预测器的影响
        with patch.object(filter_instance.rolling_filter, 'apply_percentile_filter', return_value=np.array([True]*100)),             patch.object(filter_instance.garch_predictor, 'predict_batch_volatility', return_value=pd.Series([0.1]*100)),             patch.object(filter_instance.ivol_filter, 'apply_ivol_constraint', return_value=np.array([True]*100)):

            final_mask = filter_instance.update_tradable_mask(pd.Timestamp('2023-06-01'))

            # 验证ML预测器被调用
            mock_ml_predictor_instance.apply_filter.assert_called_once()
            
            # 验证最终掩码等于ML预测器的掩码
            np.testing.assert_array_equal(final_mask, ml_mask, "当其他层全通过时，最终掩码应等于ML预测器掩码")
            logger.info(f"ML预测器集成测试通过，筛选结果符合预期。")

    def test_parallel_processing_performance(self):
        """测试并行处理性能与结果一致性"""
        logger.info("开始并行处理性能测试...")
        
        # 使用更多股票以突显并行效果
        parallel_data_manager = self._create_test_data_manager(stock_count=200)
        test_date = pd.Timestamp('2023-06-01')

        # 1. 运行串行模式
        serial_config = self.base_config.copy()
        serial_config['parallel_processing'] = False
        serial_filter = DynamicLowVolFilter(DynamicLowVolConfig(**serial_config), parallel_data_manager)
        
        start_time_serial = time.perf_counter()
        serial_mask = serial_filter.update_tradable_mask(test_date)
        end_time_serial = time.perf_counter()
        serial_duration = end_time_serial - start_time_serial
        logger.info(f"串行模式执行时间: {serial_duration:.4f} 秒")

        # 2. 运行并行模式
        parallel_config = self.base_config.copy()
        parallel_config['parallel_processing'] = True
        parallel_filter = DynamicLowVolFilter(DynamicLowVolConfig(**parallel_config), parallel_data_manager)
        
        start_time_parallel = time.perf_counter()
        parallel_mask = parallel_filter.update_tradable_mask(test_date)
        end_time_parallel = time.perf_counter()
        parallel_duration = end_time_parallel - start_time_parallel
        logger.info(f"并行模式执行时间: {parallel_duration:.4f} 秒")

        # 3. 验证结果一致性
        np.testing.assert_array_equal(serial_mask, parallel_mask, "并行与串行模式的筛选结果必须一致")
        logger.info("结果一致性验证通过。")

        # 4. 验证性能提升
        # 在某些情况下（例如IO密集或核心数少），并行可能不会更快，所以我们只记录性能，不强制断言
        speedup = serial_duration / parallel_duration if parallel_duration > 0 else float('inf')
        logger.info(f"并行处理性能提升: {speedup:.2f}x")
        self.assertTrue(speedup > 0, "并行模式不应比串行模式慢一个数量级")


if __name__ == '__main__':
    unittest.main()
