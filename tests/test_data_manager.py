"""
数据管理器测试
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataManager


class TestDataManager(unittest.TestCase):
    """数据管理器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'provider': 'yahoo',
            'region': 'cn',
            'universe': 'csi300',
            'provider_uri': 'mock://test_data'
        }
        
    @patch('data.data_manager.qlib')
    def test_init_qlib(self, mock_qlib):
        """测试qlib初始化"""
        # 创建数据管理器
        data_manager = DataManager(self.config)
        
        # 验证qlib.init被调用
        mock_qlib.init.assert_called_once_with(
            provider_uri='mock://test_data',
            region='cn'
        )
        
    def test_clean_data(self):
        """测试数据清洗功能"""
        # 创建测试数据
        data = pd.DataFrame({
            'A': [1, 2, np.inf, 4, 5],
            'B': [np.nan, 2, 3, -np.inf, 5],
            'C': [1, np.nan, 3, 4, np.nan]
        })
        
        # 创建数据管理器
        with patch('data.data_manager.qlib'):
            data_manager = DataManager(self.config)
            
        # 清洗数据
        cleaned_data = data_manager._clean_data(data)
        
        # 验证无穷值被替换
        self.assertFalse(np.isinf(cleaned_data.values).any())
        
        # 验证数据形状保持不变
        self.assertEqual(cleaned_data.shape, data.shape)
        
    def test_get_universe_stocks(self):
        """测试股票池获取"""
        with patch('data.data_manager.qlib'), \
             patch('data.data_manager.D') as mock_D:
            
            # 模拟返回股票列表
            mock_instruments = ['000001.SZ', '000002.SZ', '600000.SH']
            mock_D.instruments.return_value = mock_instruments
            
            # 创建数据管理器
            data_manager = DataManager(self.config)
            
            # 获取股票池
            stocks = data_manager._get_universe_stocks('2023-01-01', '2023-12-31')
            
            # 验证结果
            self.assertEqual(stocks, mock_instruments)
            mock_D.instruments.assert_called_once_with(market='csi300')
    
    def test_get_trading_calendar(self):
        """测试交易日历获取"""
        with patch('data.data_manager.qlib'), \
             patch('data.data_manager.D') as mock_D:
            
            # 模拟返回交易日
            mock_calendar = pd.DatetimeIndex(['2023-01-03', '2023-01-04', '2023-01-05'])
            mock_D.calendar.return_value = mock_calendar
            
            # 创建数据管理器
            data_manager = DataManager(self.config)
            
            # 获取交易日历
            calendar = data_manager.get_trading_calendar('2023-01-01', '2023-01-10')
            
            # 验证结果
            expected = ['2023-01-03', '2023-01-04', '2023-01-05']
            self.assertEqual(calendar, expected)
    
    def test_get_data_info(self):
        """测试数据信息获取"""
        with patch('data.data_manager.qlib'):
            data_manager = DataManager(self.config)
            
            # 添加一些缓存数据
            data_manager._cache['test_key'] = pd.DataFrame()
            
            # 获取数据信息
            info = data_manager.get_data_info()
            
            # 验证信息内容
            self.assertEqual(info['provider'], 'yahoo')
            self.assertEqual(info['region'], 'cn')
            self.assertEqual(info['universe'], 'csi300')
            self.assertEqual(info['cache_size'], 1)
            self.assertIn('test_key', info['cache_keys'])


if __name__ == '__main__':
    unittest.main()