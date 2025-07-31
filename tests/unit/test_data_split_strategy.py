"""
数据划分策略的单元测试
测试时序数据的训练/验证/测试划分、滚动窗口划分和数据泄露防护
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, MagicMock

from src.rl_trading_system.training.data_split_strategy import (
    DataSplitStrategy, 
    TimeSeriesSplitStrategy,
    RollingWindowSplitStrategy,
    FixedSplitStrategy,
    SplitConfig,
    SplitResult
)


class TestSplitConfig:
    """数据划分配置测试类"""
    
    def test_split_config_creation(self):
        """测试配置创建"""
        config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            min_train_samples=100,
            min_validation_samples=50,
            gap_days=1,
            rolling_window_size=252
        )
        
        assert config.train_ratio == 0.7
        assert config.validation_ratio == 0.2
        assert config.test_ratio == 0.1
        assert config.min_train_samples == 100
        assert config.min_validation_samples == 50
        assert config.gap_days == 1
        assert config.rolling_window_size == 252
    
    def test_split_config_validation(self):
        """测试配置验证"""
        # 测试比例和不为1的情况
        with pytest.raises(ValueError, match="比例之和必须为1"):
            SplitConfig(
                train_ratio=0.6,
                validation_ratio=0.2,
                test_ratio=0.1
            )
        
        # 测试负比例
        with pytest.raises(ValueError, match="比例不能为负数"):
            SplitConfig(
                train_ratio=-0.1,
                validation_ratio=0.6,
                test_ratio=0.5
            )
        
        # 测试最小样本数
        with pytest.raises(ValueError, match="最小样本数必须为正数"):
            SplitConfig(
                train_ratio=0.7,
                validation_ratio=0.2,
                test_ratio=0.1,
                min_train_samples=0
            )


class TestSplitResult:
    """数据划分结果测试类"""
    
    def test_split_result_creation(self):
        """测试结果创建"""
        train_indices = np.array([0, 1, 2, 3, 4])
        val_indices = np.array([5, 6, 7])
        test_indices = np.array([8, 9])
        
        result = SplitResult(
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices,
            split_dates={
                'train_start': '2023-01-01',
                'train_end': '2023-06-30',
                'val_start': '2023-07-01',
                'val_end': '2023-09-30',
                'test_start': '2023-10-01',
                'test_end': '2023-12-31'
            }
        )
        
        assert np.array_equal(result.train_indices, train_indices)
        assert np.array_equal(result.validation_indices, val_indices)
        assert np.array_equal(result.test_indices, test_indices)
        assert len(result.split_dates) == 6
    
    def test_split_result_validation(self):
        """测试结果验证"""
        # 测试索引重叠
        with pytest.raises(ValueError, match="训练和验证索引不能重叠"):
            SplitResult(
                train_indices=np.array([0, 1, 2]),
                validation_indices=np.array([2, 3, 4]),  # 与训练重叠
                test_indices=np.array([5, 6])
            )
        
        # 测试索引重叠
        with pytest.raises(ValueError, match="验证和测试索引不能重叠"):
            SplitResult(
                train_indices=np.array([0, 1, 2]),
                validation_indices=np.array([3, 4, 5]),
                test_indices=np.array([5, 6, 7])  # 与验证重叠
            )
    
    def test_get_metrics(self):
        """测试获取统计指标"""
        result = SplitResult(
            train_indices=np.array([0, 1, 2, 3, 4]),
            validation_indices=np.array([5, 6, 7]),
            test_indices=np.array([8, 9])
        )
        
        metrics = result.get_metrics()
        
        assert metrics['train_size'] == 5
        assert metrics['validation_size'] == 3
        assert metrics['test_size'] == 2
        assert metrics['total_size'] == 10
        assert abs(metrics['train_ratio'] - 0.5) < 1e-6
        assert abs(metrics['validation_ratio'] - 0.3) < 1e-6
        assert abs(metrics['test_ratio'] - 0.2) < 1e-6


class TestTimeSeriesSplitStrategy:
    """时序数据划分策略测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本时序数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'TEST',
            'price': np.random.randn(len(dates)) * 10 + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        return data.set_index(['datetime', 'symbol'])
    
    @pytest.fixture
    def split_config(self):
        """创建划分配置"""
        return SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            min_train_samples=50,
            min_validation_samples=20,
            gap_days=1
        )
    
    def test_time_series_split_basic(self, sample_data, split_config):
        """测试基本时序划分"""
        strategy = TimeSeriesSplitStrategy(split_config)
        result = strategy.split(sample_data)
        
        # 检查基本属性
        assert isinstance(result, SplitResult)
        assert len(result.train_indices) > 0
        assert len(result.validation_indices) > 0
        assert len(result.test_indices) > 0
        
        # 检查时序顺序
        assert result.train_indices[-1] < result.validation_indices[0]
        assert result.validation_indices[-1] < result.test_indices[0]
        
        # 检查比例
        metrics = result.get_metrics()
        assert abs(metrics['train_ratio'] - 0.7) < 0.1
        assert abs(metrics['validation_ratio'] - 0.2) < 0.1
        assert abs(metrics['test_ratio'] - 0.1) < 0.1
    
    def test_time_series_split_with_gap(self, sample_data):
        """测试带间隔的时序划分"""
        config = SplitConfig(
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            gap_days=5  # 5天间隔
        )
        
        strategy = TimeSeriesSplitStrategy(config)
        result = strategy.split(sample_data)
        
        # 检查间隔
        train_end_date = sample_data.index.get_level_values('datetime')[result.train_indices[-1]]
        val_start_date = sample_data.index.get_level_values('datetime')[result.validation_indices[0]]
        gap = (val_start_date - train_end_date).days
        assert gap >= config.gap_days
        
        val_end_date = sample_data.index.get_level_values('datetime')[result.validation_indices[-1]]
        test_start_date = sample_data.index.get_level_values('datetime')[result.test_indices[0]]
        gap = (test_start_date - val_end_date).days
        assert gap >= config.gap_days
    
    def test_time_series_split_minimum_samples(self, split_config):
        """测试最小样本数约束"""
        # 创建小数据集
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        small_data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'TEST',
            'price': np.random.randn(len(dates))
        }).set_index(['datetime', 'symbol'])
        
        strategy = TimeSeriesSplitStrategy(split_config)
        
        # 应该抛出异常，因为数据太少
        with pytest.raises(ValueError, match="数据量不足"):
            strategy.split(small_data)
    
    def test_data_leakage_prevention(self, sample_data, split_config):
        """测试数据泄露防护"""
        strategy = TimeSeriesSplitStrategy(split_config)
        result = strategy.split(sample_data)
        
        # 检查时间顺序，确保没有未来数据泄露
        train_dates = sample_data.index.get_level_values('datetime')[result.train_indices]
        val_dates = sample_data.index.get_level_values('datetime')[result.validation_indices]
        test_dates = sample_data.index.get_level_values('datetime')[result.test_indices]
        
        # 训练数据应该在验证数据之前
        assert train_dates.max() < val_dates.min()
        
        # 验证数据应该在测试数据之前
        assert val_dates.max() < test_dates.min()
        
        # 检查索引的时序性
        assert np.all(np.diff(result.train_indices) >= 0)
        assert np.all(np.diff(result.validation_indices) >= 0)
        assert np.all(np.diff(result.test_indices) >= 0)


class TestRollingWindowSplitStrategy:
    """滚动窗口划分策略测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本时序数据"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'TEST',
            'price': np.random.randn(len(dates)) * 10 + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        return data.set_index(['datetime', 'symbol'])
    
    @pytest.fixture
    def rolling_config(self):
        """创建滚动窗口配置"""
        return SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            rolling_window_size=252,  # 一年的交易日
            step_size=63,  # 季度步长
            min_train_samples=100
        )
    
    def test_rolling_window_split_basic(self, sample_data, rolling_config):
        """测试基本滚动窗口划分"""
        strategy = RollingWindowSplitStrategy(rolling_config)
        splits = strategy.split_rolling(sample_data)
        
        # 应该产生多个划分
        assert len(splits) > 1
        
        # 检查每个划分
        for i, result in enumerate(splits):
            assert isinstance(result, SplitResult)
            assert len(result.train_indices) > 0
            assert len(result.validation_indices) > 0
            assert len(result.test_indices) > 0
            
            # 检查时序顺序
            assert result.train_indices[-1] < result.validation_indices[0]
            assert result.validation_indices[-1] < result.test_indices[0]
    
    def test_rolling_window_progression(self, sample_data, rolling_config):
        """测试滚动窗口的时间推进"""
        strategy = RollingWindowSplitStrategy(rolling_config)
        splits = strategy.split_rolling(sample_data)
        
        # 检查窗口推进
        for i in range(1, len(splits)):
            prev_split = splits[i-1]
            curr_split = splits[i]
            
            # 当前划分的开始应该在前一个划分之后
            prev_train_start = sample_data.index.get_level_values('datetime')[prev_split.train_indices[0]]
            curr_train_start = sample_data.index.get_level_values('datetime')[curr_split.train_indices[0]]
            
            assert curr_train_start > prev_train_start
    
    def test_rolling_window_overlap_prevention(self, sample_data, rolling_config):
        """测试滚动窗口的重叠防护"""
        strategy = RollingWindowSplitStrategy(rolling_config)
        splits = strategy.split_rolling(sample_data)
        
        # 检查相邻窗口的时间推进
        for i in range(1, len(splits)):
            prev_split = splits[i-1]
            curr_split = splits[i]
            
            # 当前窗口的开始应该在前一个窗口开始之后（允许训练数据重叠，但窗口整体应该推进）
            prev_window_start = sample_data.index.get_level_values('datetime')[prev_split.train_indices[0]]
            curr_window_start = sample_data.index.get_level_values('datetime')[curr_split.train_indices[0]]
            
            assert curr_window_start > prev_window_start
            
            # 检查步长推进是否合理
            time_diff = (curr_window_start - prev_window_start).days
            expected_step = rolling_config.step_size or (rolling_config.rolling_window_size // 4)
            # 允许一定的偏差，因为步长是按日期计算的
            assert time_diff >= expected_step * 0.5
    
    def test_rolling_window_size_consistency(self, sample_data, rolling_config):
        """测试滚动窗口大小一致性"""
        strategy = RollingWindowSplitStrategy(rolling_config)
        splits = strategy.split_rolling(sample_data)
        
        # 每个窗口的总大小应该接近配置的窗口大小
        for result in splits[:-1]:  # 最后一个窗口可能较小
            total_size = (len(result.train_indices) + 
                         len(result.validation_indices) + 
                         len(result.test_indices))
            
            # 允许一定的偏差
            assert abs(total_size - rolling_config.rolling_window_size) < 50


class TestFixedSplitStrategy:
    """固定划分策略测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本时序数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'TEST',
            'price': np.random.randn(len(dates)) * 10 + 100
        })
        return data.set_index(['datetime', 'symbol'])
    
    def test_fixed_split_by_date(self, sample_data):
        """测试按日期固定划分"""
        config = SplitConfig(
            train_end_date='2023-08-31',
            validation_end_date='2023-10-31'
        )
        
        strategy = FixedSplitStrategy(config)
        result = strategy.split(sample_data)
        
        # 检查日期边界
        train_dates = sample_data.index.get_level_values('datetime')[result.train_indices]
        val_dates = sample_data.index.get_level_values('datetime')[result.validation_indices]
        test_dates = sample_data.index.get_level_values('datetime')[result.test_indices]
        
        assert train_dates.max() <= pd.Timestamp(config.train_end_date)
        assert val_dates.max() <= pd.Timestamp(config.validation_end_date)
        assert test_dates.min() > pd.Timestamp(config.validation_end_date)
    
    def test_fixed_split_by_ratio(self, sample_data):
        """测试按比例固定划分"""
        config = SplitConfig(
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2
        )
        
        strategy = FixedSplitStrategy(config)
        result = strategy.split(sample_data)
        
        metrics = result.get_metrics()
        
        # 检查比例
        assert abs(metrics['train_ratio'] - 0.6) < 0.05
        assert abs(metrics['validation_ratio'] - 0.2) < 0.05
        assert abs(metrics['test_ratio'] - 0.2) < 0.05


class TestDataLeakageDetection:
    """数据泄露检测测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'TEST',
            'feature1': np.random.randn(len(dates)),
            'feature2': np.random.randn(len(dates)),
            'target': np.random.randn(len(dates))
        })
        return data.set_index(['datetime', 'symbol'])
    
    def test_temporal_leakage_detection(self, sample_data):
        """测试时间泄露检测"""
        config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1
        )
        
        strategy = TimeSeriesSplitStrategy(config)
        result = strategy.split(sample_data)
        
        # 检测是否存在时间泄露
        leakage_detected = strategy.detect_temporal_leakage(result, sample_data)
        assert not leakage_detected  # 正确的划分不应该有泄露
    
    def test_feature_leakage_detection(self, sample_data):
        """测试特征泄露检测"""
        # 创建有泄露的特征（使用未来信息）
        future_feature = sample_data['feature1'].shift(-5)  # 使用5天后的数据
        sample_data['leaked_feature'] = future_feature
        
        config = SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
        strategy = TimeSeriesSplitStrategy(config)
        result = strategy.split(sample_data)
        
        # 检测特征泄露
        leakage_detected = strategy.detect_feature_leakage(
            result, sample_data, ['leaked_feature']
        )
        assert leakage_detected  # 应该检测到泄露
    
    def test_target_leakage_detection(self, sample_data):
        """测试目标变量泄露检测"""
        # 创建使用未来目标的特征
        sample_data['target_lag'] = sample_data['target'].shift(-1)
        
        config = SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
        strategy = TimeSeriesSplitStrategy(config)
        result = strategy.split(sample_data)
        
        # 检测目标泄露
        leakage_detected = strategy.detect_target_leakage(
            result, sample_data, 'target', ['target_lag']
        )
        assert leakage_detected  # 应该检测到泄露


class TestSplitStrategyComparison:
    """划分策略对比测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'TEST',
            'price': np.cumsum(np.random.randn(len(dates))) + 100,
            'return': np.random.randn(len(dates)) * 0.02
        })
        return data.set_index(['datetime', 'symbol'])
    
    def test_strategy_consistency(self, sample_data):
        """测试不同策略的一致性"""
        config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1
        )
        
        # 测试时序策略
        ts_strategy = TimeSeriesSplitStrategy(config)
        ts_result = ts_strategy.split(sample_data)
        
        # 测试固定策略
        fixed_strategy = FixedSplitStrategy(config)
        fixed_result = fixed_strategy.split(sample_data)
        
        # 两种策略的结果应该相似（但不完全相同）
        ts_metrics = ts_result.get_metrics()
        fixed_metrics = fixed_result.get_metrics()
        
        assert abs(ts_metrics['train_ratio'] - fixed_metrics['train_ratio']) < 0.1
        assert abs(ts_metrics['validation_ratio'] - fixed_metrics['validation_ratio']) < 0.1
        assert abs(ts_metrics['test_ratio'] - fixed_metrics['test_ratio']) < 0.1
    
    def test_strategy_robustness(self, sample_data):
        """测试策略稳健性"""
        config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            random_seed=42
        )
        
        strategy = TimeSeriesSplitStrategy(config)
        
        # 多次运行应该产生相同结果
        result1 = strategy.split(sample_data)
        result2 = strategy.split(sample_data)
        
        assert np.array_equal(result1.train_indices, result2.train_indices)
        assert np.array_equal(result1.validation_indices, result2.validation_indices)
        assert np.array_equal(result1.test_indices, result2.test_indices)
    
    @pytest.mark.parametrize("strategy_name,strategy_class", [
        ("time_series", TimeSeriesSplitStrategy),
        ("fixed", FixedSplitStrategy)
    ])
    def test_strategy_validity(self, sample_data, strategy_name, strategy_class):
        """测试不同策略的有效性"""
        config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1
        )
        
        strategy = strategy_class(config)
        result = strategy.split(sample_data)
        
        # 基本有效性检查
        assert len(result.train_indices) > 0
        assert len(result.validation_indices) > 0
        assert len(result.test_indices) > 0
        
        # 无重叠检查
        assert len(np.intersect1d(result.train_indices, result.validation_indices)) == 0
        assert len(np.intersect1d(result.validation_indices, result.test_indices)) == 0
        assert len(np.intersect1d(result.train_indices, result.test_indices)) == 0
        
        # 覆盖完整性检查
        all_indices = np.concatenate([
            result.train_indices, 
            result.validation_indices, 
            result.test_indices
        ])
        expected_indices = np.arange(len(sample_data))
        
        # 允许有间隔，但总体应该覆盖大部分数据
        coverage = len(all_indices) / len(expected_indices)
        assert coverage > 0.8  # 至少覆盖80%的数据


class TestEdgeCases:
    """边界情况测试类"""
    
    def test_single_symbol_data(self):
        """测试单一股票数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'SINGLE',
            'price': np.random.randn(len(dates))
        }).set_index(['datetime', 'symbol'])
        
        config = SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
        strategy = TimeSeriesSplitStrategy(config)
        result = strategy.split(data)
        
        assert len(result.train_indices) > 0
        assert len(result.validation_indices) > 0
        assert len(result.test_indices) > 0
    
    def test_multiple_symbols_data(self):
        """测试多股票数据"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        symbols = ['A', 'B', 'C']
        
        data_list = []
        for symbol in symbols:
            symbol_data = pd.DataFrame({
                'datetime': dates,
                'symbol': symbol,
                'price': np.random.randn(len(dates))
            })
            data_list.append(symbol_data)
        
        data = pd.concat(data_list).set_index(['datetime', 'symbol'])
        
        config = SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
        strategy = TimeSeriesSplitStrategy(config)
        result = strategy.split(data)
        
        # 应该能够处理多股票数据
        assert len(result.train_indices) > 0
        assert len(result.validation_indices) > 0
        assert len(result.test_indices) > 0
    
    def test_irregular_time_series(self):
        """测试非规律时序数据"""
        # 创建有缺失日期的数据
        all_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        # 随机删除一些日期
        keep_indices = np.random.choice(len(all_dates), size=int(len(all_dates) * 0.8), replace=False)
        dates = all_dates[np.sort(keep_indices)]
        
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'IRREGULAR',
            'price': np.random.randn(len(dates))
        }).set_index(['datetime', 'symbol'])
        
        config = SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
        strategy = TimeSeriesSplitStrategy(config)
        result = strategy.split(data)
        
        # 应该能够处理非规律时序
        assert len(result.train_indices) > 0
        assert len(result.validation_indices) > 0
        assert len(result.test_indices) > 0
    
    def test_minimum_data_requirements(self):
        """测试最小数据要求"""
        # 创建极小数据集
        dates = pd.date_range(start='2023-01-01', end='2023-01-05', freq='D')
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': 'TINY',
            'price': [1, 2, 3, 4, 5]
        }).set_index(['datetime', 'symbol'])
        
        config = SplitConfig(
            train_ratio=0.6,
            validation_ratio=0.2,
            test_ratio=0.2,
            min_train_samples=10,  # 要求最少10个训练样本
            min_validation_samples=2
        )
        
        strategy = TimeSeriesSplitStrategy(config)
        
        with pytest.raises(ValueError, match="数据量不足"):
            strategy.split(data)