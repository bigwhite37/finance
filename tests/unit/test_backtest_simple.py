#!/usr/bin/env python3
"""
简化的回测验证测试

专注于测试回测工具函数和指标计算，不依赖复杂的模块集成。
"""

import sys
import os
import pandas as pd
import numpy as np
import unittest
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.unit.test_backtest_utils import (
    BacktestDataGenerator, 
    BacktestMetricsCalculator,
    BacktestConfigFactory,
    BacktestExpectedMetrics
)


class TestBacktestUtilities(unittest.TestCase):
    """测试回测工具类"""
    
    def setUp(self):
        """测试前准备"""
        self.dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
        self.dates = self.dates[self.dates.weekday < 5]  # 只保留工作日
        self.stocks = [f'stock_{i:03d}' for i in range(20)]  # 20只股票用于测试
        
    def test_backtest_data_generator(self):
        """测试回测数据生成器"""
        # 测试股票收益率生成
        returns_data = BacktestDataGenerator.generate_realistic_stock_returns(
            self.dates, self.stocks
        )
        
        # 验证数据格式
        self.assertIsInstance(returns_data, pd.DataFrame)
        self.assertEqual(returns_data.shape, (len(self.dates), len(self.stocks)))
        self.assertTrue(returns_data.index.equals(self.dates))
        self.assertTrue(returns_data.columns.equals(pd.Index(self.stocks)))
        
        # 验证数据合理性
        self.assertTrue(np.all(np.isfinite(returns_data.values)))
        self.assertTrue(np.all(np.abs(returns_data.values) < 0.5))  # 日收益率不超过50%
        
        # 测试市场状态数据生成
        market_data = BacktestDataGenerator.generate_market_regime_data(self.dates)
        
        # 验证市场数据格式
        self.assertIsInstance(market_data, pd.DataFrame)
        self.assertEqual(len(market_data), len(self.dates))
        self.assertIn('market_return', market_data.columns)
        self.assertIn('market_volatility', market_data.columns)
        
        # 验证市场数据合理性
        self.assertTrue(np.all(np.isfinite(market_data['market_return'].values)))
        self.assertTrue(np.all(market_data['market_volatility'].values > 0))
        
    def test_backtest_metrics_calculator(self):
        """测试回测指标计算器"""
        # 生成测试收益率数据
        np.random.seed(42)  # 固定随机种子确保可重复性
        returns = np.random.normal(0.0005, 0.02, 252)  # 252个交易日
        
        # 计算指标
        metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(returns)
        
        # 验证指标类型和合理性
        self.assertIsInstance(metrics, dict)
        
        # 检查必需的指标
        required_metrics = [
            'annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown',
            'skewness', 'kurtosis', 'var_95', 'cvar_95', 'calmar_ratio',
            'sortino_ratio', 'total_return', 'win_rate', 'tail_loss_frequency'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertTrue(np.isfinite(metrics[metric]))
        
        # 验证指标合理性
        self.assertGreaterEqual(metrics['annual_volatility'], 0)
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 1)
        self.assertGreaterEqual(metrics['max_drawdown'], 0)
        self.assertLessEqual(metrics['max_drawdown'], 1)
        
        # 测试空输入
        empty_metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(np.array([]))
        self.assertEqual(empty_metrics['annual_return'], 0.0)
        self.assertEqual(empty_metrics['annual_volatility'], 0.0)
        
    def test_backtest_config_factory(self):
        """测试回测配置工厂"""
        # 测试配置创建
        config = BacktestConfigFactory.create_backtest_config()
        
        # 验证配置结构
        self.assertIsInstance(config, dict)
        
        # 检查必需的配置项
        required_configs = [
            'rolling_windows', 'percentile_thresholds', 'garch_window',
            'forecast_horizon', 'enable_ml_predictor', 'ivol_bad_threshold',
            'ivol_good_threshold', 'regime_detection_window', 'regime_model_type',
            'enable_caching', 'cache_expiry_days', 'parallel_processing'
        ]
        
        for config_key in required_configs:
            self.assertIn(config_key, config)
        
        # 验证配置值合理性
        self.assertIsInstance(config['rolling_windows'], list)
        self.assertGreater(len(config['rolling_windows']), 0)
        self.assertIsInstance(config['percentile_thresholds'], dict)
        self.assertGreater(config['garch_window'], 0)
        self.assertGreater(config['forecast_horizon'], 0)
        self.assertIsInstance(config['enable_ml_predictor'], bool)
        
        # 测试数据管理器Mock创建
        data_manager = BacktestConfigFactory.create_benchmark_data_manager()
        
        # 验证Mock对象有必需的方法
        self.assertTrue(hasattr(data_manager, 'get_price_data'))
        self.assertTrue(hasattr(data_manager, 'get_volume_data'))
        self.assertTrue(hasattr(data_manager, 'get_market_data'))
        
        # 测试Mock返回数据
        price_data = data_manager.get_price_data()
        self.assertIsInstance(price_data, pd.DataFrame)
        self.assertGreater(len(price_data), 0)
        self.assertGreater(len(price_data.columns), 0)
        
    def test_backtest_expected_metrics(self):
        """测试回测预期指标"""
        expected = BacktestExpectedMetrics.EXPECTED_METRICS
        
        # 验证预期指标结构
        self.assertIsInstance(expected, dict)
        
        # 检查必需的预期指标
        required_expected = [
            'annual_return_min', 'annual_volatility_max', 'max_drawdown_max',
            'sharpe_ratio_min', 'tail_loss_reduction_min'
        ]
        
        for expected_key in required_expected:
            self.assertIn(expected_key, expected)
            self.assertIsInstance(expected[expected_key], (int, float))
            self.assertGreater(expected[expected_key], 0)
        
        # 验证预期指标合理性
        self.assertGreater(expected['annual_return_min'], 0)
        self.assertLess(expected['annual_return_min'], 1)  # 年化收益小于100%
        self.assertLess(expected['annual_volatility_max'], 1)  # 年化波动小于100%
        self.assertLess(expected['max_drawdown_max'], 1)  # 最大回撤小于100%
        self.assertGreater(expected['sharpe_ratio_min'], 0)
        self.assertLess(expected['tail_loss_reduction_min'], 1)  # 尾部亏损降低小于100%
        
    def test_portfolio_metrics_edge_cases(self):
        """测试组合指标计算的边界情况"""
        # 测试全零收益率
        zero_returns = np.zeros(100)
        zero_metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(zero_returns)
        self.assertEqual(zero_metrics['annual_return'], 0.0)
        self.assertEqual(zero_metrics['annual_volatility'], 0.0)
        self.assertEqual(zero_metrics['sharpe_ratio'], 0.0)
        self.assertEqual(zero_metrics['max_drawdown'], 0.0)
        
        # 测试单一值收益率
        single_return = np.array([0.01])
        single_metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(single_return)
        self.assertIsInstance(single_metrics['annual_return'], (int, float))
        self.assertTrue(np.isfinite(single_metrics['annual_return']))
        
        # 测试极端值收益率
        extreme_returns = np.array([0.1, -0.05, 0.08, -0.12, 0.03])
        extreme_metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(extreme_returns)
        self.assertTrue(np.isfinite(extreme_metrics['max_drawdown']))
        self.assertGreaterEqual(extreme_metrics['max_drawdown'], 0)
        self.assertLessEqual(extreme_metrics['max_drawdown'], 1)
        
    def test_data_consistency(self):
        """测试数据生成的一致性"""
        # 设置相同的随机种子，生成两次数据应该相同
        np.random.seed(123)
        returns1 = BacktestDataGenerator.generate_realistic_stock_returns(
            self.dates[:10], self.stocks[:5]
        )
        
        np.random.seed(123)
        returns2 = BacktestDataGenerator.generate_realistic_stock_returns(
            self.dates[:10], self.stocks[:5]
        )
        
        # 验证数据一致性
        pd.testing.assert_frame_equal(returns1, returns2)
        
        # 设置不同的随机种子，生成的数据应该不同
        np.random.seed(456)
        returns3 = BacktestDataGenerator.generate_realistic_stock_returns(
            self.dates[:10], self.stocks[:5]
        )
        
        # 验证数据差异性（不应该完全相等）
        self.assertFalse(returns1.equals(returns3))


if __name__ == '__main__':
    unittest.main()