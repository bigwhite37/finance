"""
离线数据集测试

测试OfflineDataset的数据加载、清洗、特征创建和行为克隆数据集生成功能。
"""

import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime, timedelta

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.offline_dataset import OfflineDataset, OfflineDataConfig
from data.data_manager import DataManager


class TestOfflineDataset(unittest.TestCase):
    """离线数据集测试类"""
    
    def setUp(self):
        """测试前置设置"""
        # 模拟数据管理器
        self.mock_data_manager = Mock(spec=DataManager)
        
        # 创建测试配置
        self.config = OfflineDataConfig(
            start_date="2022-01-01",
            end_date="2022-12-31",
            lookback_window=20,
            prediction_horizon=1,
            min_samples_per_stock=30,  # 降低最小样本要求
            normalize_features=True,
            include_market_features=True,
            behavior_cloning_threshold=0.02
        )
        
        # 创建模拟股票数据
        self.mock_stock_data = self._create_mock_stock_data()
        self.mock_market_data = self._create_mock_market_data()
        
        # 设置模拟返回值
        self.mock_data_manager.get_stock_data.return_value = self.mock_stock_data
        self.mock_data_manager.get_market_data.return_value = self.mock_market_data
        
    def _create_mock_stock_data(self) -> pd.DataFrame:
        """创建模拟股票数据"""
        # 创建时间序列
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
        dates = [d for d in dates if d.weekday() < 5][:250]  # 只取工作日
        
        # 创建股票列表
        stocks = ['000001.SZ', '000002.SZ', '600000.SH']
        
        # 创建多级索引
        index_tuples = [(date, stock) for date in dates for stock in stocks]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['datetime', 'instrument'])
        
        # 创建更真实的价格数据 - 每只股票独立的价格走势
        np.random.seed(42)  # 确保可重复性
        
        all_data = []
        base_prices = [50, 75, 100]  # 每只股票的基础价格
        
        for i, stock in enumerate(stocks):
            base_price = base_prices[i]
            stock_data = []
            
            for j, date in enumerate(dates):
                if j == 0:
                    # 第一天的价格
                    price = base_price
                else:
                    # 基于前一天价格的随机游走
                    change = np.random.normal(0, 0.02)  # 2%日波动率
                    price = stock_data[-1]['$close'] * (1 + change)
                    price = max(price, 1.0)  # 确保价格为正
                
                # 生成当日的OHLC数据
                daily_volatility = 0.01
                high = price * (1 + abs(np.random.normal(0, daily_volatility)))
                low = price * (1 - abs(np.random.normal(0, daily_volatility)))
                open_price = price * (1 + np.random.normal(0, daily_volatility/2))
                close_price = price
                volume = np.random.uniform(10000, 1000000)
                vwap = (high + low + open_price + close_price) / 4
                
                stock_data.append({
                    '$open': open_price,
                    '$high': high,
                    '$low': low,
                    '$close': close_price,
                    '$volume': volume,
                    '$vwap': vwap
                })
            
            all_data.extend(stock_data)
        
        # 创建DataFrame
        data_dict = {col: [row[col] for row in all_data] for col in all_data[0].keys()}
        df = pd.DataFrame(data_dict, index=multi_index)
        return df
        
    def _create_mock_market_data(self) -> pd.DataFrame:
        """创建模拟市场数据"""
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
        dates = [d for d in dates if d.weekday() < 5][:250]
        
        indices = ["SH000300", "SH000905", "SH000852"]
        index_tuples = [(date, idx) for date in dates for idx in indices]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['datetime', 'instrument'])
        
        np.random.seed(42)
        data = {
            '$close': np.random.uniform(3000, 5000, len(index_tuples)),
            '$volume': np.random.uniform(1000000, 10000000, len(index_tuples))
        }
        
        return pd.DataFrame(data, index=multi_index)
        
    def test_initialization(self):
        """测试初始化"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        
        self.assertEqual(dataset.data_manager, self.mock_data_manager)
        self.assertEqual(dataset.config, self.config)
        self.assertIsNone(dataset.raw_data)
        self.assertIsNone(dataset.processed_data)
        self.assertIsNone(dataset.features)
        self.assertIsNone(dataset.targets)
        
    def test_load_historical_data(self):
        """测试历史数据加载"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        result = dataset.load_historical_data()
        
        # 验证返回值（链式调用）
        self.assertEqual(result, dataset)
        
        # 验证数据管理器被正确调用
        self.mock_data_manager.get_stock_data.assert_called_once()
        self.mock_data_manager.get_market_data.assert_called_once()
        
        # 验证数据被正确加载
        self.assertIsNotNone(dataset.raw_data)
        self.assertIsNotNone(dataset.processed_data)
        self.assertEqual(dataset.raw_data.shape, self.mock_stock_data.shape)
        
    def test_data_cleaning(self):
        """测试数据清洗功能"""
        # 创建包含异常值的数据
        dirty_data = self.mock_stock_data.copy()
        dirty_data.iloc[0, 0] = np.inf  # 添加无穷大
        dirty_data.iloc[1, 1] = -np.inf  # 添加负无穷大
        dirty_data.iloc[2, 2] = np.nan   # 添加NaN
        dirty_data.iloc[3, 0] = -10      # 添加负价格
        
        self.mock_data_manager.get_stock_data.return_value = dirty_data
        
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        # 验证清洗后的数据
        self.assertFalse(np.isinf(dataset.processed_data.values).any())
        self.assertFalse(np.isnan(dataset.processed_data.values).any())
        
        # 验证数据在合理范围内（标准化后的数据通常在[-5, 5]范围内）
        self.assertTrue((np.abs(dataset.processed_data.values) <= 10).all())
                
    def test_feature_creation(self):
        """测试特征和目标变量创建"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        # 验证特征和目标变量被创建
        self.assertIsNotNone(dataset.features)
        self.assertIsNotNone(dataset.targets)
        
        # 验证特征维度
        expected_feature_dim = len(self.mock_stock_data.columns) * self.config.lookback_window
        self.assertEqual(dataset.features.shape[1], expected_feature_dim)
        
        # 验证样本数量合理
        self.assertGreater(len(dataset.features), 0)
        self.assertEqual(len(dataset.features), len(dataset.targets))
        
        # 验证目标变量是收益率（应该在合理范围内）
        self.assertTrue(np.abs(dataset.targets).max() < 5.0)  # 收益率不应超过500%（极端情况）
        
    def test_normalization(self):
        """测试特征标准化"""
        # 测试标准化功能
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        # 由于使用了滚动标准化，检查数据是否在合理范围内
        # 标准化后的数据应该大部分在[-3, 3]范围内
        feature_values = dataset.features.flatten()
        within_range = np.sum(np.abs(feature_values) <= 3) / len(feature_values)
        self.assertGreater(within_range, 0.8)  # 至少80%的数据在3个标准差内
        
    def test_behavior_dataset_creation(self):
        """测试行为克隆数据集创建"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        # 测试无专家动作的情况
        behavior_dataset = dataset.create_behavior_dataset()
        
        # 验证数据集类型
        self.assertIsInstance(behavior_dataset, torch.utils.data.TensorDataset)
        
        # 验证数据集大小
        self.assertGreater(len(behavior_dataset), 0)
        
        # 验证张量维度
        features, actions = behavior_dataset.tensors
        self.assertEqual(len(features), len(actions))
        self.assertEqual(features.shape[1], dataset.features.shape[1])
        
        # 验证动作值在合理范围内（[-1, 1]）
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))
        
    def test_behavior_dataset_with_expert_actions(self):
        """测试使用专家动作的行为克隆数据集"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        # 创建模拟专家动作
        dates = pd.date_range('2022-01-01', '2022-12-31', freq='D')
        dates = [d for d in dates if d.weekday() < 5][:250]
        stocks = ['000001.SZ', '000002.SZ', '600000.SH']
        
        np.random.seed(42)
        expert_actions = pd.DataFrame(
            np.random.uniform(-1, 1, (len(dates), len(stocks))),
            index=dates,
            columns=stocks
        )
        
        behavior_dataset = dataset.create_behavior_dataset(expert_actions)
        
        # 验证使用了专家动作
        self.assertIsInstance(behavior_dataset, torch.utils.data.TensorDataset)
        self.assertGreater(len(behavior_dataset), 0)
        
    def test_tensor_dataset_creation(self):
        """测试标准张量数据集创建"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        tensor_dataset = dataset.create_tensor_dataset()
        
        # 验证数据集类型和大小
        self.assertIsInstance(tensor_dataset, torch.utils.data.TensorDataset)
        self.assertEqual(len(tensor_dataset), len(dataset.features))
        
        # 验证张量类型
        features, targets = tensor_dataset.tensors
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(targets.dtype, torch.float32)
        
    def test_data_augmentation(self):
        """测试数据增强功能"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        # 创建原始数据集
        original_dataset = dataset.create_tensor_dataset()
        original_size = len(original_dataset)
        
        # 应用数据增强
        augmented_dataset = dataset.apply_data_augmentation(original_dataset)
        
        # 验证增强后的数据集更大
        self.assertIsInstance(augmented_dataset, torch.utils.data.TensorDataset)
        self.assertGreater(len(augmented_dataset), original_size)
        self.assertLessEqual(len(augmented_dataset), original_size * 3)  # 最多3倍
        
    def test_statistics(self):
        """测试统计信息获取"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        
        # 测试未初始化时的统计
        stats = dataset.get_statistics()
        self.assertEqual(stats['total_samples'], 0)
        self.assertEqual(stats['feature_dim'], 0)
        
        # 加载数据后的统计
        dataset.load_historical_data()
        stats = dataset.get_statistics()
        
        # 验证统计信息
        self.assertGreater(stats['total_samples'], 0)
        self.assertGreater(stats['feature_dim'], 0)
        self.assertEqual(stats['date_range'], (self.config.start_date, self.config.end_date))
        self.assertIn('target_mean', stats)
        self.assertIn('target_std', stats)
        self.assertIn('target_min', stats)
        self.assertIn('target_max', stats)
        
    def test_len_and_getitem(self):
        """测试长度和索引访问"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        dataset.load_historical_data()
        
        # 测试长度
        length = len(dataset)
        self.assertGreater(length, 0)
        self.assertEqual(length, len(dataset.features))
        
        # 测试索引访问
        feature, target = dataset[0]
        self.assertIsInstance(feature, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        self.assertEqual(feature.shape[0], dataset.features.shape[1])
        self.assertEqual(target.shape[0], 1)
        
    def test_edge_cases(self):
        """测试边界条件"""
        # 测试空数据
        empty_data = pd.DataFrame()
        self.mock_data_manager.get_stock_data.return_value = empty_data
        
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        
        # 空数据应该不会抛出异常，但会产生空的特征
        dataset.load_historical_data()
        self.assertEqual(len(dataset), 0)
        
    def test_error_handling(self):
        """测试错误处理"""
        dataset = OfflineDataset(self.mock_data_manager, self.config)
        
        # 测试在未加载数据时调用方法
        with self.assertRaises(ValueError):
            dataset.create_tensor_dataset()
            
        with self.assertRaises(ValueError):
            dataset.create_behavior_dataset()
            
        with self.assertRaises(ValueError):
            dataset[0]
            
    def test_config_validation(self):
        """测试配置参数验证"""
        # 测试不同的配置参数
        config = OfflineDataConfig(
            start_date="2022-01-01",
            end_date="2022-01-31",  # 短时间窗口
            lookback_window=5,
            prediction_horizon=1,
            min_samples_per_stock=10
        )
        
        dataset = OfflineDataset(self.mock_data_manager, config)
        dataset.load_historical_data()
        
        # 即使是短时间窗口也应该能工作
        self.assertGreaterEqual(len(dataset), 0)


if __name__ == '__main__':
    unittest.main()