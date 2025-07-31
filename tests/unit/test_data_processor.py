"""
数据预处理管道测试用例
测试数据清洗、预处理流水线、数据质量检查和缓存功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import shutil

from src.rl_trading_system.data.data_processor import DataProcessor
from src.rl_trading_system.data.data_cache import DataCache
from src.rl_trading_system.data.data_quality import DataQualityChecker
from src.rl_trading_system.data.data_models import MarketData, FeatureVector
from src.rl_trading_system.data.feature_engineer import FeatureEngineer


class TestDataProcessor:
    """数据预处理器测试类"""
    
    @pytest.fixture
    def sample_price_data(self):
        """创建样本价格数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = {
            'open': 100 + np.random.randn(100) * 2,
            'high': 102 + np.random.randn(100) * 2,
            'low': 98 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': np.random.randint(1000000, 10000000, 100),
            'amount': np.random.randint(100000000, 1000000000, 100)
        }
        
        # 确保价格关系正确
        for i in range(100):
            data['high'][i] = max(data['open'][i], data['high'][i], 
                                data['low'][i], data['close'][i])
            data['low'][i] = min(data['open'][i], data['high'][i], 
                               data['low'][i], data['close'][i])
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def sample_fundamental_data(self):
        """创建样本基本面数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = {
            'pe_ratio': 10 + np.random.randn(100) * 3,
            'pb_ratio': 1.5 + np.random.randn(100) * 0.5,
            'roe': 0.1 + np.random.randn(100) * 0.05,
            'roa': 0.05 + np.random.randn(100) * 0.02,
            'revenue_growth': 0.1 + np.random.randn(100) * 0.1,
            'profit_growth': 0.15 + np.random.randn(100) * 0.15
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def dirty_price_data(self):
        """创建包含脏数据的价格数据"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        data = {
            'open': [100, 101, np.nan, 103, -5, 105, 106, 107, 108, 109] + [100] * 40,
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111] + [102] * 40,
            'low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107] + [98] * 40,
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] + [101] * 40,
            'volume': [1000000, 2000000, np.nan, 4000000, -1000000, 
                      6000000, 7000000, 8000000, 9000000, 10000000] + [1000000] * 40,
            'amount': [100000000, 200000000, 300000000, 400000000, 500000000,
                      600000000, 700000000, 800000000, 900000000, 1000000000] + [100000000] * 40
        }
        
        # 故意制造价格关系错误
        data['low'][5] = 120  # 最低价高于最高价
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    @pytest.fixture
    def data_processor(self):
        """创建数据预处理器实例"""
        return DataProcessor()
    
    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_data_processor_initialization(self, data_processor):
        """测试数据预处理器初始化"""
        assert data_processor is not None
        assert hasattr(data_processor, 'feature_engineer')
        assert hasattr(data_processor, 'quality_checker')
        assert hasattr(data_processor, 'cache')
    
    def test_process_clean_data(self, data_processor, sample_price_data):
        """测试处理干净数据"""
        result = data_processor.process_data(
            data=sample_price_data,
            symbols=['000001.SZ'],
            data_type='price'
        )
        
        assert isinstance(result, dict)
        assert 'processed_data' in result
        assert 'quality_report' in result
        assert 'feature_vectors' in result
        
        # 检查处理后的数据
        processed_data = result['processed_data']
        assert not processed_data.empty
        assert len(processed_data) <= len(sample_price_data)
        
        # 检查质量报告
        quality_report = result['quality_report']
        assert quality_report['status'] in ['good', 'warning', 'error']
        assert 'score' in quality_report
        assert isinstance(quality_report['score'], float)
    
    def test_process_dirty_data(self, data_processor, dirty_price_data):
        """测试处理脏数据"""
        result = data_processor.process_data(
            data=dirty_price_data,
            symbols=['000001.SZ'],
            data_type='price',
            clean_strategy='aggressive'
        )
        
        # 检查数据清洗效果
        processed_data = result['processed_data']
        assert len(processed_data) < len(dirty_price_data)  # 应该删除了一些脏数据
        
        # 检查没有负值
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in processed_data.columns:
                assert (processed_data[col] >= 0).all()
        
        # 检查价格关系
        if all(col in processed_data.columns for col in ['high', 'low']):
            assert (processed_data['high'] >= processed_data['low']).all()
    
    def test_batch_processing(self, data_processor, sample_price_data):
        """测试批处理功能"""
        # 创建多个股票的数据
        symbols = ['000001.SZ', '000002.SZ', '600000.SH']
        batch_data = {}
        
        for symbol in symbols:
            # 为每个股票创建略有不同的数据
            data = sample_price_data.copy()
            data = data * (1 + np.random.randn() * 0.1)  # 添加随机变化
            batch_data[symbol] = data
        
        result = data_processor.process_batch(
            batch_data=batch_data,
            data_type='price',
            parallel=True
        )
        
        assert isinstance(result, dict)
        assert len(result) == len(symbols)
        
        for symbol in symbols:
            assert symbol in result
            assert 'processed_data' in result[symbol]
            assert 'quality_report' in result[symbol]
    
    def test_feature_engineering_integration(self, data_processor, sample_price_data):
        """测试特征工程集成"""
        result = data_processor.process_data(
            data=sample_price_data,
            symbols=['000001.SZ'],
            data_type='price',
            calculate_features=True
        )
        
        assert 'feature_vectors' in result
        feature_vectors = result['feature_vectors']
        assert len(feature_vectors) > 0
        
        # 检查特征向量结构
        first_vector = feature_vectors[0]
        assert isinstance(first_vector, FeatureVector)
        assert first_vector.symbol == '000001.SZ'
        assert len(first_vector.technical_indicators) > 0
        assert len(first_vector.fundamental_factors) > 0
        assert len(first_vector.market_microstructure) > 0
    
    def test_data_normalization(self, data_processor, sample_price_data):
        """测试数据标准化"""
        result = data_processor.process_data(
            data=sample_price_data,
            symbols=['000001.SZ'],
            data_type='price',
            normalize=True,
            normalization_method='zscore'
        )
        
        processed_data = result['processed_data']
        
        # 检查标准化效果（技术指标应该被标准化）
        if 'sma_20' in processed_data.columns:
            sma_mean = processed_data['sma_20'].mean()
            sma_std = processed_data['sma_20'].std()
            assert abs(sma_mean) < 0.1  # 均值接近0
            assert abs(sma_std - 1.0) < 0.1  # 标准差接近1
    
    def test_missing_value_handling(self, data_processor):
        """测试缺失值处理"""
        # 创建包含缺失值的数据
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'open': [100, np.nan, 102, 103, np.nan] * 4,
            'high': [102, 103, np.nan, 105, 106] * 4,
            'low': [98, 99, 100, np.nan, 102] * 4,
            'close': [101, 102, 103, 104, np.nan] * 4,
            'volume': [1000000] * 20,
            'amount': [100000000] * 20
        }, index=dates)
        
        # 测试前向填充
        result_ffill = data_processor.process_data(
            data=data,
            symbols=['000001.SZ'],
            data_type='price',
            missing_value_method='ffill'
        )
        
        processed_ffill = result_ffill['processed_data']
        assert processed_ffill.isnull().sum().sum() < data.isnull().sum().sum()
        
        # 测试删除缺失值
        result_drop = data_processor.process_data(
            data=data,
            symbols=['000001.SZ'],
            data_type='price',
            missing_value_method='drop'
        )
        
        processed_drop = result_drop['processed_data']
        assert processed_drop.isnull().sum().sum() == 0
        assert len(processed_drop) < len(data)
    
    def test_outlier_detection_and_treatment(self, data_processor):
        """测试异常值检测和处理"""
        # 创建包含异常值的数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 2,
            'high': 102 + np.random.randn(100) * 2,
            'low': 98 + np.random.randn(100) * 2,
            'close': 100 + np.random.randn(100) * 2,
            'volume': np.random.randint(1000000, 10000000, 100),
            'amount': np.random.randint(100000000, 1000000000, 100)
        }, index=dates)
        
        # 人为添加异常值
        data.loc[data.index[10], 'close'] = 1000  # 极大异常值
        data.loc[data.index[20], 'volume'] = 100000000  # 极大成交量
        
        result = data_processor.process_data(
            data=data,
            symbols=['000001.SZ'],
            data_type='price',
            outlier_treatment='clip'
        )
        
        processed_data = result['processed_data']
        
        # 检查异常值是否被处理
        close_max = processed_data['close'].max()
        assert close_max < 200  # 异常值应该被裁剪
    
    def test_data_validation(self, data_processor, sample_price_data):
        """测试数据验证"""
        # 测试有效数据
        is_valid = data_processor.validate_data(
            data=sample_price_data,
            data_type='price'
        )
        assert is_valid
        
        # 测试无效数据
        invalid_data = sample_price_data.copy()
        invalid_data['high'] = invalid_data['low'] - 10  # 制造价格关系错误
        
        is_valid = data_processor.validate_data(
            data=invalid_data,
            data_type='price'
        )
        assert not is_valid
    
    def test_pipeline_configuration(self, data_processor):
        """测试流水线配置"""
        config = {
            'clean_strategy': 'conservative',
            'missing_value_method': 'ffill',
            'outlier_treatment': 'clip',
            'normalize': True,
            'normalization_method': 'minmax',
            'calculate_features': True,
            'feature_selection': True,
            'cache_enabled': True
        }
        
        data_processor.configure_pipeline(config)
        
        assert data_processor.config['clean_strategy'] == 'conservative'
        assert data_processor.config['normalize'] is True
        assert data_processor.config['cache_enabled'] is True


class TestDataProcessorCache:
    """数据预处理器缓存测试类"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def cache_processor(self, temp_cache_dir):
        """创建带缓存的数据预处理器"""
        cache = DataCache(cache_dir=temp_cache_dir, default_ttl=3600)
        processor = DataProcessor(cache=cache)
        return processor
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        data = pd.DataFrame({
            'open': 100 + np.random.randn(50),
            'high': 102 + np.random.randn(50),
            'low': 98 + np.random.randn(50),
            'close': 100 + np.random.randn(50),
            'volume': np.random.randint(1000000, 10000000, 50),
            'amount': np.random.randint(100000000, 1000000000, 50)
        }, index=dates)
        
        # 确保价格关系正确
        for i in range(50):
            data.iloc[i, 1] = max(data.iloc[i, [0, 1, 2, 3]])  # high
            data.iloc[i, 2] = min(data.iloc[i, [0, 1, 2, 3]])  # low
        
        return data
    
    def test_cache_hit(self, cache_processor, sample_data):
        """测试缓存命中"""
        # 第一次处理，应该缓存结果
        result1 = cache_processor.process_data(
            data=sample_data,
            symbols=['000001.SZ'],
            data_type='price',
            use_cache=True
        )
        
        # 第二次处理，应该从缓存获取
        with patch.object(cache_processor.feature_engineer, 'calculate_technical_indicators') as mock_calc:
            result2 = cache_processor.process_data(
                data=sample_data,
                symbols=['000001.SZ'],
                data_type='price',
                use_cache=True
            )
            
            # 如果从缓存获取，不应该调用特征计算
            mock_calc.assert_not_called()
        
        # 结果应该相同
        pd.testing.assert_frame_equal(
            result1['processed_data'], 
            result2['processed_data']
        )
    
    def test_cache_miss_on_different_params(self, cache_processor, sample_data):
        """测试不同参数导致的缓存未命中"""
        # 使用不同参数处理
        result1 = cache_processor.process_data(
            data=sample_data,
            symbols=['000001.SZ'],
            data_type='price',
            normalize=True,
            use_cache=True
        )
        
        result2 = cache_processor.process_data(
            data=sample_data,
            symbols=['000001.SZ'],
            data_type='price',
            normalize=False,  # 不同的参数
            use_cache=True
        )
        
        # 结果应该不同
        assert not result1['processed_data'].equals(result2['processed_data'])
    
    def test_cache_expiration(self, temp_cache_dir, sample_data):
        """测试缓存过期"""
        # 创建短TTL的缓存
        cache = DataCache(cache_dir=temp_cache_dir, default_ttl=1)  # 1秒TTL
        processor = DataProcessor(cache=cache)
        
        # 第一次处理
        result1 = processor.process_data(
            data=sample_data,
            symbols=['000001.SZ'],
            data_type='price',
            use_cache=True
        )
        
        # 等待缓存过期
        import time
        time.sleep(2)
        
        # 第二次处理，应该重新计算
        with patch.object(processor.feature_engineer, 'calculate_technical_indicators') as mock_calc:
            mock_calc.return_value = pd.DataFrame()  # 模拟返回
            
            result2 = processor.process_data(
                data=sample_data,
                symbols=['000001.SZ'],
                data_type='price',
                use_cache=True
            )
            
            # 应该重新调用特征计算
            mock_calc.assert_called()
    
    def test_cache_disable(self, cache_processor, sample_data):
        """测试禁用缓存"""
        # 禁用缓存处理
        result1 = cache_processor.process_data(
            data=sample_data,
            symbols=['000001.SZ'],
            data_type='price',
            use_cache=False
        )
        
        # 再次处理，应该重新计算
        with patch.object(cache_processor.feature_engineer, 'calculate_technical_indicators') as mock_calc:
            mock_calc.return_value = pd.DataFrame()
            
            result2 = cache_processor.process_data(
                data=sample_data,
                symbols=['000001.SZ'],
                data_type='price',
                use_cache=False
            )
            
            # 应该调用特征计算
            mock_calc.assert_called()


class TestDataProcessorQuality:
    """数据预处理器质量检查测试类"""
    
    @pytest.fixture
    def quality_processor(self):
        """创建数据预处理器"""
        return DataProcessor()
    
    def test_quality_check_good_data(self, quality_processor):
        """测试高质量数据的质量检查"""
        # 创建高质量数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(100) * 0.5,
            'high': 101 + np.random.randn(100) * 0.5,
            'low': 99 + np.random.randn(100) * 0.5,
            'close': 100 + np.random.randn(100) * 0.5,
            'volume': np.random.randint(1000000, 2000000, 100),
            'amount': np.random.randint(100000000, 200000000, 100)
        }, index=dates)
        
        # 确保价格关系正确
        for i in range(100):
            data.loc[data.index[i], 'high'] = max(
                data.loc[data.index[i], ['open', 'high', 'low', 'close']]
            )
            data.loc[data.index[i], 'low'] = min(
                data.loc[data.index[i], ['open', 'high', 'low', 'close']]
            )
        
        quality_report = quality_processor.check_data_quality(
            data=data,
            data_type='price'
        )
        
        assert quality_report['status'] == 'good'
        assert quality_report['score'] >= 0.8
        assert len(quality_report['issues']) == 0
    
    def test_quality_check_poor_data(self, quality_processor):
        """测试低质量数据的质量检查"""
        # 创建低质量数据
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        data = pd.DataFrame({
            'open': [100, np.nan, -50, 103, np.nan] * 4,
            'high': [102, np.nan, 104, 105, 106] * 4,
            'low': [150, 99, 100, 101, 102] * 4,  # 故意制造错误关系
            'close': [101, np.nan, 103, 104, np.nan] * 4,
            'volume': [-1000000, 2000000, np.nan, 4000000, 5000000] * 4,
            'amount': [100000000] * 20
        }, index=dates)
        
        quality_report = quality_processor.check_data_quality(
            data=data,
            data_type='price'
        )
        
        assert quality_report['status'] in ['warning', 'error']
        assert quality_report['score'] < 0.6
        assert len(quality_report['issues']) > 0
    
    def test_quality_improvement_after_cleaning(self, quality_processor):
        """测试清洗后数据质量改善"""
        # 创建脏数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        dirty_data = pd.DataFrame({
            'open': [100, np.nan, -50, 103] + [100] * 46,
            'high': [102, 103, 104, 105] + [102] * 46,
            'low': [98, 99, 100, 101] + [98] * 46,
            'close': [101, np.nan, 103, 104] + [101] * 46,
            'volume': [-1000000, 2000000, 3000000, 4000000] + [1000000] * 46,
            'amount': [100000000] * 50
        }, index=dates)
        
        # 清洗前的质量
        quality_before = quality_processor.check_data_quality(
            data=dirty_data,
            data_type='price'
        )
        
        # 清洗数据
        result = quality_processor.process_data(
            data=dirty_data,
            symbols=['000001.SZ'],
            data_type='price',
            clean_strategy='aggressive'
        )
        
        # 清洗后的质量
        quality_after = result['quality_report']
        
        # 质量应该有所改善或至少不变差
        assert quality_after['score'] >= quality_before['score'] * 0.9  # 允许轻微下降
        assert len(quality_after['issues']) <= len(quality_before['issues'])
    
    def test_quality_metrics_calculation(self, quality_processor):
        """测试质量指标计算"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(100),
            'high': 102 + np.random.randn(100),
            'low': 98 + np.random.randn(100),
            'close': 100 + np.random.randn(100),
            'volume': np.random.randint(1000000, 10000000, 100),
            'amount': np.random.randint(100000000, 1000000000, 100)
        }, index=dates)
        
        # 添加一些缺失值
        data.iloc[10:15, 0] = np.nan  # 5个缺失值
        
        quality_report = quality_processor.check_data_quality(
            data=data,
            data_type='price'
        )
        
        # 检查统计信息
        stats = quality_report['statistics']
        assert 'row_count' in stats
        assert 'missing_values' in stats
        assert stats['row_count'] == 100
        assert stats['missing_values']['open'] == 5


class TestDataProcessorIntegration:
    """数据预处理器集成测试类"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def full_processor(self, temp_cache_dir):
        """创建完整配置的数据预处理器"""
        cache = DataCache(cache_dir=temp_cache_dir)
        processor = DataProcessor(cache=cache)
        
        config = {
            'clean_strategy': 'conservative',
            'missing_value_method': 'ffill',
            'outlier_treatment': 'clip',
            'normalize': True,
            'normalization_method': 'zscore',
            'calculate_features': True,
            'feature_selection': True,
            'cache_enabled': True
        }
        processor.configure_pipeline(config)
        
        return processor
    
    def test_end_to_end_processing(self, full_processor):
        """测试端到端处理流程"""
        # 创建真实场景的数据
        dates = pd.date_range('2023-01-01', periods=252, freq='D')  # 一年交易日
        np.random.seed(42)
        
        # 模拟股价随机游走
        returns = np.random.randn(252) * 0.02
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(252) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(252)) * 0.002),
            'low': prices * (1 - np.abs(np.random.randn(252)) * 0.002),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 252),
            'amount': prices * np.random.randint(1000000, 10000000, 252)
        }, index=dates)
        
        # 确保价格关系正确
        for i in range(252):
            data.iloc[i, 1] = max(data.iloc[i, [0, 1, 2, 3]])  # high
            data.iloc[i, 2] = min(data.iloc[i, [0, 1, 2, 3]])  # low
        
        # 添加一些真实的数据问题
        data.iloc[50:55, 0] = np.nan  # 缺失值
        data.iloc[100, 3] = data.iloc[100, 3] * 1.5  # 异常值
        
        # 处理数据
        result = full_processor.process_data(
            data=data,
            symbols=['000001.SZ'],
            data_type='price'
        )
        
        # 验证结果完整性
        assert 'processed_data' in result
        assert 'quality_report' in result
        assert 'feature_vectors' in result
        
        processed_data = result['processed_data']
        quality_report = result['quality_report']
        feature_vectors = result['feature_vectors']
        
        # 验证数据质量
        assert not processed_data.empty
        assert quality_report['status'] in ['good', 'warning']
        assert len(feature_vectors) > 0
        
        # 验证特征工程
        first_vector = feature_vectors[0]
        assert len(first_vector.technical_indicators) > 5
        assert len(first_vector.fundamental_factors) >= 1
        assert len(first_vector.market_microstructure) > 0
        
        # 验证数据标准化
        if 'sma_20' in processed_data.columns:
            sma_mean = processed_data['sma_20'].mean()
            assert abs(sma_mean) < 0.2  # 标准化后均值接近0
    
    def test_multi_symbol_processing(self, full_processor):
        """测试多股票处理"""
        symbols = ['000001.SZ', '000002.SZ', '600000.SH']
        batch_data = {}
        
        # 为每个股票创建数据
        for i, symbol in enumerate(symbols):
            dates = pd.date_range('2023-01-01', periods=100, freq='D')
            base_price = 100 + i * 50  # 不同的基础价格
            
            data = pd.DataFrame({
                'open': base_price + np.random.randn(100) * 2,
                'high': base_price + 2 + np.random.randn(100) * 2,
                'low': base_price - 2 + np.random.randn(100) * 2,
                'close': base_price + np.random.randn(100) * 2,
                'volume': np.random.randint(1000000, 10000000, 100),
                'amount': np.random.randint(100000000, 1000000000, 100)
            }, index=dates)
            
            # 确保价格关系正确
            for j in range(100):
                data.iloc[j, 1] = max(data.iloc[j, [0, 1, 2, 3]])
                data.iloc[j, 2] = min(data.iloc[j, [0, 1, 2, 3]])
            
            batch_data[symbol] = data
        
        # 批处理
        results = full_processor.process_batch(
            batch_data=batch_data,
            data_type='price',
            parallel=True
        )
        
        # 验证结果
        assert len(results) == len(symbols)
        
        for symbol in symbols:
            assert symbol in results
            result = results[symbol]
            assert 'processed_data' in result
            assert 'quality_report' in result
            assert 'feature_vectors' in result
            
            # 验证每个股票的处理结果
            assert not result['processed_data'].empty
            assert len(result['feature_vectors']) > 0
    
    def test_performance_monitoring(self, full_processor):
        """测试性能监控"""
        # 创建大量数据测试性能
        dates = pd.date_range('2023-01-01', periods=1000, freq='D')
        data = pd.DataFrame({
            'open': 100 + np.random.randn(1000),
            'high': 102 + np.random.randn(1000),
            'low': 98 + np.random.randn(1000),
            'close': 100 + np.random.randn(1000),
            'volume': np.random.randint(1000000, 10000000, 1000),
            'amount': np.random.randint(100000000, 1000000000, 1000)
        }, index=dates)
        
        # 确保价格关系正确
        for i in range(1000):
            data.iloc[i, 1] = max(data.iloc[i, [0, 1, 2, 3]])
            data.iloc[i, 2] = min(data.iloc[i, [0, 1, 2, 3]])
        
        import time
        start_time = time.time()
        
        result = full_processor.process_data(
            data=data,
            symbols=['000001.SZ'],
            data_type='price'
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证处理时间合理（应该在几秒内完成）
        assert processing_time < 30  # 30秒内完成
        
        # 验证结果质量
        assert not result['processed_data'].empty
        assert result['quality_report']['status'] in ['good', 'warning']
        
        # 验证缓存效果
        start_time2 = time.time()
        result2 = full_processor.process_data(
            data=data,
            symbols=['000001.SZ'],
            data_type='price',
            use_cache=True
        )
        end_time2 = time.time()
        cached_time = end_time2 - start_time2
        
        # 缓存应该提高速度或至少不慢太多
        assert cached_time <= processing_time * 1.2  # 允许20%的误差