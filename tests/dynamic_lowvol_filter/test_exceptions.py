"""
动态低波筛选器异常处理测试

包含以下测试内容：
1. 数据质量异常测试 - 验证数据质量问题的异常处理
2. 模型拟合异常测试 - 验证模型拟合失败的异常处理
3. 状态检测异常测试 - 验证状态检测失败的异常处理
4. 配置异常测试 - 验证配置错误的异常处理
5. 异常传播测试 - 验证异常正确传播
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

from config.dynamic_lowvol_validator import validate_dynamic_lowvol_config
from risk_control.dynamic_lowvol_filter.core import DynamicLowVolFilter
from risk_control.dynamic_lowvol_filter.data_structures import DynamicLowVolConfig
from risk_control.dynamic_lowvol_filter.exceptions import (
    FilterException,
    DataQualityException,
    ModelFittingException,
    RegimeDetectionException,
    InsufficientDataException,
    ConfigurationException
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestExceptionHandling(unittest.TestCase):
    """异常处理测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'rolling_windows': [20, 60],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            'garch_window': 250,
            'enable_caching': True
        }
    
    def test_data_quality_exceptions(self):
        """测试数据质量异常"""
        logger.info("开始数据质量异常测试...")
        
        # 创建有问题的数据管理器
        mock_manager = Mock()
        
        # 测试1: 数据不足异常
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')  # 只有10天数据
        stocks = ['stock_001', 'stock_002']
        
        insufficient_prices = pd.DataFrame(
            np.random.randn(len(dates), len(stocks)) * 0.02 + 100,
            index=dates, columns=stocks
        )
        
        mock_manager.get_price_data.return_value = insufficient_prices
        mock_manager.get_volume_data.return_value = insufficient_prices
        mock_manager.get_factor_data.return_value = pd.DataFrame()
        mock_manager.get_market_data.return_value = pd.DataFrame()
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), mock_manager)
        
        with self.assertRaises(InsufficientDataException):
            filter_instance.update_tradable_mask(pd.Timestamp('2023-01-10'))
        
        # 测试2: 数据质量异常（大量缺失值）
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        
        poor_quality_prices = pd.DataFrame(
            np.random.randn(len(dates), len(stocks)) * 0.02 + 100,
            index=dates, columns=stocks
        )
        # 引入大量缺失值
        poor_quality_prices.iloc[::2, :] = np.nan  # 50%缺失
        
        mock_manager.get_price_data.return_value = poor_quality_prices
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), mock_manager)
        
        with self.assertRaises(DataQualityException):
            filter_instance.update_tradable_mask(pd.Timestamp('2023-06-01'))
        
        logger.info("数据质量异常测试通过")
    
    def test_model_fitting_exceptions(self):
        """测试模型拟合异常"""
        logger.info("开始模型拟合异常测试...")
        
        mock_manager = Mock()
        
        # 创建会导致GARCH拟合失败的数据
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stocks = ['stock_001', 'stock_002']
        
        # 常数序列（无波动）会导致GARCH拟合失败
        constant_prices = pd.DataFrame(
            np.ones((len(dates), len(stocks))) * 100,
            index=dates, columns=stocks
        )
        
        mock_manager.get_price_data.return_value = constant_prices
        mock_manager.get_volume_data.return_value = constant_prices
        mock_manager.get_factor_data.return_value = pd.DataFrame(
            np.random.normal(0, 0.01, (len(dates), 5)),
            index=dates, columns=['market', 'size', 'value', 'profitability', 'investment']
        )
        mock_manager.get_market_data.return_value = pd.DataFrame({
            'returns': np.zeros(len(dates)),
            'volatility': np.zeros(len(dates))
        }, index=dates)
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), mock_manager)
        
        with self.assertRaises(ModelFittingException):
            filter_instance.update_tradable_mask(pd.Timestamp('2023-06-01'))
        
        logger.info("模型拟合异常测试通过")
    
    def test_regime_detection_exceptions(self):
        """测试状态检测异常"""
        logger.info("开始状态检测异常测试...")
        
        mock_manager = Mock()
        
        # 创建会导致状态检测失败的数据
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stocks = ['stock_001', 'stock_002']
        
        normal_prices = pd.DataFrame(
            np.random.randn(len(dates), len(stocks)).cumsum(axis=0) + 100,
            index=dates, columns=stocks
        )
        
        # 异常的市场数据（全为NaN）
        abnormal_market_data = pd.DataFrame({
            'returns': np.full(len(dates), np.nan),
            'volatility': np.full(len(dates), np.nan)
        }, index=dates)
        
        mock_manager.get_price_data.return_value = normal_prices
        mock_manager.get_volume_data.return_value = normal_prices
        mock_manager.get_factor_data.return_value = pd.DataFrame(
            np.random.normal(0, 0.01, (len(dates), 5)),
            index=dates, columns=['market', 'size', 'value', 'profitability', 'investment']
        )
        mock_manager.get_market_data.return_value = abnormal_market_data
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), mock_manager)
        
        with self.assertRaises(RegimeDetectionException):
            filter_instance.update_tradable_mask(pd.Timestamp('2023-06-01'))
        
        logger.info("状态检测异常测试通过")
    
    def test_configuration_exceptions(self):
        """测试配置异常"""
        logger.info("开始配置异常测试...")
        
        mock_manager = Mock()
        
        # 测试1: 无效的分位数阈值
        invalid_config1 = self.config.copy()
        invalid_config1['percentile_thresholds'] = {"低": 1.5, "中": 0.3, "高": 0.2}  # >1.0
        
        with self.assertRaises(ConfigurationException):
            validate_dynamic_lowvol_config(invalid_config1)
        
        # 测试2: 无效的滚动窗口
        invalid_config2 = self.config.copy()
        invalid_config2['rolling_windows'] = [0, 60]  # 窗口为0
        
        with self.assertRaises(ConfigurationException):
            validate_dynamic_lowvol_config(invalid_config2)
        
        # 测试3: 缺失必需参数
        invalid_config3 = self.config.copy()
        del invalid_config3['rolling_windows']
        
        with self.assertRaises(ConfigurationException):
            validate_dynamic_lowvol_config(invalid_config3)
        
        logger.info("配置异常测试通过")
    
    def test_exception_propagation(self):
        """测试异常传播"""
        logger.info("开始异常传播测试...")
        
        mock_manager = Mock()
        
        # 模拟数据管理器抛出异常
        mock_manager.get_price_data.side_effect = Exception("数据获取失败")
        
        filter_instance = DynamicLowVolFilter(DynamicLowVolConfig(**self.config), mock_manager)
        
        # 验证异常正确传播
        with self.assertRaises(Exception) as context:
            filter_instance.update_tradable_mask(pd.Timestamp('2023-06-01'))
        
        self.assertIn("数据获取失败", str(context.exception))
        
        logger.info("异常传播测试通过")


if __name__ == '__main__':
    unittest.main()