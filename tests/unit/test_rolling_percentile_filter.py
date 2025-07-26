"""
滚动分位筛选器单元测试

测试RollingPercentileFilter类的各项功能，包括：
- 基本筛选功能
- 动态阈值调整
- 多窗口组合筛选
- 异常处理
- 缓存机制
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from risk_control.dynamic_lowvol_filter import (
    RollingPercentileFilter,
    DynamicLowVolConfig,
    DataQualityException,
    InsufficientDataException,
    ConfigurationException
)


class TestRollingPercentileFilter:
    """滚动分位筛选器测试类"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return DynamicLowVolConfig(
            rolling_windows=[20, 60],
            percentile_thresholds={"低": 0.4, "中": 0.3, "高": 0.2},
            enable_caching=True
        )
    
    @pytest.fixture
    def filter_instance(self, config):
        """创建筛选器实例"""
        return RollingPercentileFilter(config)
    
    @pytest.fixture
    def sample_returns(self):
        """创建样本收益率数据"""
        # 创建100天的数据，5只股票
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C', 'STOCK_D', 'STOCK_E']
        
        # 生成不同波动率特征的收益率数据
        np.random.seed(42)
        returns_data = {}
        
        # 低波动股票
        returns_data['STOCK_A'] = np.random.normal(0, 0.01, 100)
        returns_data['STOCK_B'] = np.random.normal(0, 0.015, 100)
        
        # 中等波动股票
        returns_data['STOCK_C'] = np.random.normal(0, 0.025, 100)
        
        # 高波动股票
        returns_data['STOCK_D'] = np.random.normal(0, 0.04, 100)
        returns_data['STOCK_E'] = np.random.normal(0, 0.05, 100)
        
        returns_df = pd.DataFrame(returns_data, index=dates)
        return returns_df
    
    @pytest.fixture
    def market_volatility(self):
        """创建市场波动率数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        # 模拟市场波动率从低到高的变化
        vol_values = np.linspace(0.15, 0.35, 100) + np.random.normal(0, 0.02, 100)
        return pd.Series(vol_values, index=dates)
    
    def test_initialization(self, config):
        """测试筛选器初始化"""
        filter_obj = RollingPercentileFilter(config)
        
        assert filter_obj.config == config
        assert filter_obj.rolling_windows == [20, 60]
        assert filter_obj.percentile_thresholds == {"低": 0.4, "中": 0.3, "高": 0.2}
        assert filter_obj._volatility_cache == {}
        assert filter_obj._percentile_cache == {}
    
    def test_initialization_no_cache(self):
        """测试禁用缓存的初始化"""
        config = DynamicLowVolConfig(enable_caching=False)
        filter_obj = RollingPercentileFilter(config)
        
        assert filter_obj._volatility_cache is None
        assert filter_obj._percentile_cache is None
    
    def test_apply_percentile_filter_basic(self, filter_instance, sample_returns):
        """测试基本分位筛选功能"""
        current_date = sample_returns.index[50]  # 选择中间日期
        
        result = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=20, percentile_threshold=0.3
        )
        
        # 验证结果格式
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == len(sample_returns.columns)
        
        # 验证筛选逻辑：应该有股票被选中，且选中比例合理
        selected_count = result.sum()
        assert selected_count > 0, "应该有股票被选中"
        assert selected_count <= len(result) * 0.5, "选中比例不应过高"
        
        # 验证低波动股票更可能被选中（STOCK_A波动率最低）
        assert result[0] == True, "最低波动股票应该被选中"
    
    def test_apply_percentile_filter_different_windows(self, filter_instance, sample_returns):
        """测试不同窗口长度的筛选效果"""
        current_date = sample_returns.index[70]
        
        # 20日窗口
        result_20 = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=20, percentile_threshold=0.3
        )
        
        # 60日窗口
        result_60 = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=60, percentile_threshold=0.3
        )
        
        # 验证结果格式
        assert isinstance(result_20, np.ndarray) and isinstance(result_60, np.ndarray)
        assert result_20.dtype == bool and result_60.dtype == bool
        assert len(result_20) == len(result_60) == len(sample_returns.columns)
        
        # 验证两个窗口都有合理的筛选结果
        assert result_20.sum() > 0 and result_60.sum() > 0
        
        # 低波动股票在两种窗口下都应该被选中
        assert result_20[0] == True and result_60[0] == True  # STOCK_A
    
    def test_apply_percentile_filter_by_industry(self, filter_instance, sample_returns):
        """测试按行业分组的分位筛选"""
        current_date = sample_returns.index[50]
        
        # 创建行业映射
        industry_mapping = {
            'STOCK_A': '金融',
            'STOCK_B': '金融', 
            'STOCK_C': '科技',
            'STOCK_D': '科技',
            'STOCK_E': '消费'
        }
        
        result = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=20, percentile_threshold=0.5,
            by_industry=True, industry_mapping=industry_mapping
        )
        
        # 验证结果格式
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == len(sample_returns.columns)
        
        # 在行业内排名，应该有更多股票通过筛选
        selected_count = result.sum()
        assert selected_count >= 2  # 至少有2只股票被选中
    
    def test_calculate_dynamic_threshold(self, filter_instance, market_volatility):
        """测试动态阈值计算"""
        base_threshold = 0.3
        sensitivity = 0.5
        
        result = filter_instance.calculate_dynamic_threshold(
            market_volatility, base_threshold, sensitivity
        )
        
        # 验证结果类型和范围
        assert isinstance(result, float)
        assert 0.1 <= result <= 0.5
        
        # 测试不同敏感度
        result_high_sens = filter_instance.calculate_dynamic_threshold(
            market_volatility, base_threshold, sensitivity=1.0
        )
        result_low_sens = filter_instance.calculate_dynamic_threshold(
            market_volatility, base_threshold, sensitivity=0.1
        )
        
        # 高敏感度应该产生更大的调整
        assert abs(result_high_sens - base_threshold) >= abs(result_low_sens - base_threshold)
    
    def test_get_multi_window_filter_intersection(self, filter_instance, sample_returns):
        """测试多窗口交集筛选"""
        current_date = sample_returns.index[70]
        
        result = filter_instance.get_multi_window_filter(
            sample_returns, current_date, windows=[20, 60], 
            combination_method='intersection'
        )
        
        # 验证结果格式
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == len(sample_returns.columns)
        
        # 交集应该比单个窗口更严格
        single_window_result = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=20
        )
        
        # 交集选中的股票数量应该不超过单窗口
        assert result.sum() <= single_window_result.sum()
    
    def test_get_multi_window_filter_union(self, filter_instance, sample_returns):
        """测试多窗口并集筛选"""
        current_date = sample_returns.index[70]
        
        result = filter_instance.get_multi_window_filter(
            sample_returns, current_date, windows=[20, 60],
            combination_method='union'
        )
        
        # 验证结果格式
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        
        # 并集应该比单个窗口更宽松
        single_window_result = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=20
        )
        
        # 并集选中的股票数量应该不少于单窗口
        assert result.sum() >= single_window_result.sum()
    
    def test_get_multi_window_filter_weighted(self, filter_instance, sample_returns):
        """测试多窗口加权筛选"""
        current_date = sample_returns.index[70]
        
        result = filter_instance.get_multi_window_filter(
            sample_returns, current_date, windows=[20, 60],
            combination_method='weighted'
        )
        
        # 验证结果格式
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        
        # 加权结果应该介于交集和并集之间
        intersection_result = filter_instance.get_multi_window_filter(
            sample_returns, current_date, windows=[20, 60],
            combination_method='intersection'
        )
        union_result = filter_instance.get_multi_window_filter(
            sample_returns, current_date, windows=[20, 60],
            combination_method='union'
        )
        
        assert intersection_result.sum() <= result.sum() <= union_result.sum()
    
    def test_caching_mechanism(self, filter_instance, sample_returns):
        """测试缓存机制"""
        current_date = sample_returns.index[50]
        window = 20
        
        # 第一次调用
        result1 = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=window
        )
        
        # 验证缓存中有数据
        cache_key = (current_date, window)
        assert cache_key in filter_instance._volatility_cache
        
        # 第二次调用应该使用缓存
        result2 = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=window
        )
        
        # 结果应该相同
        assert np.array_equal(result1, result2)
    
    def test_data_quality_exceptions(self, filter_instance):
        """测试数据质量异常处理"""
        current_date = pd.Timestamp('2023-01-01')
        
        # 测试空DataFrame
        with pytest.raises(DataQualityException, match="收益率数据为空"):
            filter_instance.apply_percentile_filter(
                pd.DataFrame(), current_date
            )
        
        # 测试非DataFrame输入
        with pytest.raises(DataQualityException, match="收益率数据必须为DataFrame类型"):
            filter_instance.apply_percentile_filter(
                "not_a_dataframe", current_date
            )
    
    def test_insufficient_data_exception(self, filter_instance):
        """测试数据不足异常"""
        # 创建数据长度不足的DataFrame
        short_data = pd.DataFrame({
            'STOCK_A': [0.01, 0.02, 0.01],
            'STOCK_B': [0.02, 0.01, 0.03]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        current_date = short_data.index[-1]
        
        with pytest.raises(InsufficientDataException, match="数据长度.*不足以计算.*期滚动窗口"):
            filter_instance.apply_percentile_filter(
                short_data, current_date, window=20
            )
    
    def test_configuration_exceptions(self, filter_instance, sample_returns):
        """测试配置参数异常"""
        current_date = sample_returns.index[50]
        
        # 测试无效窗口长度
        with pytest.raises(ConfigurationException, match="滚动窗口长度必须为正数"):
            filter_instance.apply_percentile_filter(
                sample_returns, current_date, window=0
            )
        
        # 测试无效分位数阈值
        with pytest.raises(ConfigurationException, match="分位数阈值必须在\\(0,1\\)范围内"):
            filter_instance.apply_percentile_filter(
                sample_returns, current_date, percentile_threshold=1.5
            )
        
        # 测试动态阈值计算的参数异常
        market_vol = pd.Series([0.2, 0.3, 0.25])
        
        with pytest.raises(ConfigurationException, match="基础阈值必须在\\(0,1\\)范围内"):
            filter_instance.calculate_dynamic_threshold(
                market_vol, base_threshold=1.5
            )
        
        with pytest.raises(ConfigurationException, match="敏感度必须在\\[0,1\\]范围内"):
            filter_instance.calculate_dynamic_threshold(
                market_vol, sensitivity=2.0
            )
    
    def test_invalid_combination_method(self, filter_instance, sample_returns):
        """测试无效的组合方法"""
        current_date = sample_returns.index[50]
        
        with pytest.raises(ConfigurationException, match="不支持的组合方法"):
            filter_instance.get_multi_window_filter(
                sample_returns, current_date, combination_method='invalid_method'
            )
    
    def test_empty_market_volatility(self, filter_instance):
        """测试空的市场波动率数据"""
        empty_vol = pd.Series([], dtype=float)
        
        with pytest.raises(DataQualityException, match="市场波动率数据为空"):
            filter_instance.calculate_dynamic_threshold(empty_vol)
    
    def test_missing_date_handling(self, filter_instance, sample_returns):
        """测试缺失日期的处理"""
        # 使用不存在的日期
        future_date = sample_returns.index[-1] + timedelta(days=10)
        
        # 应该使用最近的可用日期
        result = filter_instance.apply_percentile_filter(
            sample_returns, future_date, window=20
        )
        
        # 应该能正常返回结果
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_returns.columns)
    
    def test_all_nan_volatility_handling(self, filter_instance):
        """测试全NaN波动率的处理"""
        # 创建会产生NaN波动率的数据（所有收益率为0）
        zero_returns = pd.DataFrame({
            'STOCK_A': [0.0] * 50,
            'STOCK_B': [0.0] * 50
        }, index=pd.date_range('2023-01-01', periods=50))
        
        current_date = zero_returns.index[-1]
        
        # 零收益率会导致零波动率，进而导致筛选结果为空
        with pytest.raises(DataQualityException, match="筛选结果为空"):
            filter_instance.apply_percentile_filter(
                zero_returns, current_date, window=20
            )
    
    def test_filter_result_validation(self, filter_instance):
        """测试筛选结果验证"""
        # 创建会导致筛选结果为空的数据
        high_vol_returns = pd.DataFrame({
            'STOCK_A': np.random.normal(0, 0.1, 50),
            'STOCK_B': np.random.normal(0, 0.1, 50)
        }, index=pd.date_range('2023-01-01', periods=50))
        
        current_date = high_vol_returns.index[-1]
        
        # 使用极低的阈值，应该导致没有股票通过筛选
        with pytest.raises(DataQualityException, match="筛选结果为空"):
            filter_instance.apply_percentile_filter(
                high_vol_returns, current_date, window=20, percentile_threshold=0.01
            )
    
    def test_industry_mapping_edge_cases(self, filter_instance, sample_returns):
        """测试行业映射的边界情况"""
        current_date = sample_returns.index[50]
        
        # 测试部分股票没有行业信息
        incomplete_mapping = {
            'STOCK_A': '金融',
            'STOCK_B': '金融'
            # STOCK_C, STOCK_D, STOCK_E 没有行业信息
        }
        
        result = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=20, percentile_threshold=0.3,
            by_industry=True, industry_mapping=incomplete_mapping
        )
        
        # 应该能正常处理（没有行业信息的股票使用全市场排名）
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_returns.columns)
    
    def test_single_stock_industry(self, filter_instance, sample_returns):
        """测试单股票行业的处理"""
        current_date = sample_returns.index[50]
        
        # 创建每个股票都在不同行业的映射
        single_stock_mapping = {
            'STOCK_A': '金融',
            'STOCK_B': '科技',
            'STOCK_C': '消费',
            'STOCK_D': '医药',
            'STOCK_E': '能源'
        }
        
        result = filter_instance.apply_percentile_filter(
            sample_returns, current_date, window=20, percentile_threshold=0.3,
            by_industry=True, industry_mapping=single_stock_mapping
        )
        
        # 单股票行业应该回退到全市场排名
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_returns.columns)
    
    def test_performance_with_large_dataset(self, filter_instance):
        """测试大数据集的性能"""
        # 创建较大的数据集
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        stocks = [f'STOCK_{i:03d}' for i in range(100)]
        
        # 生成随机收益率数据
        np.random.seed(42)
        returns_data = np.random.normal(0, 0.02, (1000, 100))
        large_returns = pd.DataFrame(returns_data, index=dates, columns=stocks)
        
        current_date = large_returns.index[500]
        
        # 测试基本筛选功能
        result = filter_instance.apply_percentile_filter(
            large_returns, current_date, window=20, percentile_threshold=0.3
        )
        
        # 验证结果
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        assert result.dtype == bool
        
        # 验证筛选比例合理
        selection_ratio = result.sum() / len(result)
        assert 0.1 <= selection_ratio <= 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])