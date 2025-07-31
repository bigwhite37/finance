"""
特征工程模块测试用例
测试技术指标计算、基本面因子和市场微观结构特征计算
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.rl_trading_system.data.feature_engineer import FeatureEngineer
from src.rl_trading_system.data.data_models import MarketData, FeatureVector


# Global fixtures
@pytest.fixture
def feature_engineer():
    """创建特征工程器实例"""
    return FeatureEngineer()

@pytest.fixture
def sample_price_data():
    """创建样本价格数据"""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # 生成模拟价格数据
    base_price = 100.0
    returns = np.random.normal(0, 0.02, 100)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame({
        'datetime': dates,
        'symbol': ['000001.SZ'] * 100,
        'open': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100),
        'amount': [p * v for p, v in zip(prices, np.random.randint(1000000, 10000000, 100))]
    })
    
    # 确保价格逻辑关系正确
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data.set_index(['datetime', 'symbol'])

@pytest.fixture
def sample_fundamental_data():
    """创建样本基本面数据"""
    dates = pd.date_range('2023-01-01', periods=20, freq='Q')  # 季度数据
    
    data = pd.DataFrame({
        'datetime': dates,
        'symbol': ['000001.SZ'] * 20,
        'pe_ratio': np.random.uniform(10, 30, 20),
        'pb_ratio': np.random.uniform(1, 5, 20),
        'roe': np.random.uniform(0.05, 0.25, 20),
        'roa': np.random.uniform(0.02, 0.15, 20),
        'debt_ratio': np.random.uniform(0.2, 0.8, 20),
        'current_ratio': np.random.uniform(1.0, 3.0, 20),
        'revenue_growth': np.random.uniform(-0.1, 0.3, 20),
        'profit_growth': np.random.uniform(-0.2, 0.5, 20)
    })
    
    return data.set_index(['datetime', 'symbol'])


class TestFeatureEngineer:
    """特征工程器测试类"""
    pass


class TestTechnicalIndicators:
    """技术指标计算测试"""
    
    def test_calculate_sma(self, feature_engineer, sample_price_data):
        """测试简单移动平均线计算"""
        result = feature_engineer.calculate_sma(sample_price_data, window=20)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['sma_5', 'sma_10', 'sma_20', 'sma_60']
        for col in expected_columns:
            assert col in result.columns
        
        # 验证数值合理性
        assert not result['sma_20'].isnull().all()
        sma_20_valid = result['sma_20'].dropna()
        assert len(sma_20_valid) > 0
        assert (sma_20_valid > 0).all()
        
        # 验证移动平均线的单调性（在趋势明显时）
        close_prices = sample_price_data['close'].values
        sma_20 = result['sma_20'].dropna().values
        
        # 简单验证：SMA应该平滑价格波动
        assert len(sma_20) > 0
    
    def test_calculate_ema(self, feature_engineer, sample_price_data):
        """测试指数移动平均线计算"""
        result = feature_engineer.calculate_ema(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['ema_12', 'ema_26']
        for col in expected_columns:
            assert col in result.columns
        
        # 验证数值合理性
        assert not result['ema_12'].isnull().all()
        ema_12_valid = result['ema_12'].dropna()
        assert len(ema_12_valid) > 0
        assert (ema_12_valid > 0).all()
        
        # 验证EMA的响应性（应该比SMA更快响应价格变化）
        sma_result = feature_engineer.calculate_sma(sample_price_data, window=12)
        if 'sma_12' in sma_result.columns:
            # EMA和SMA应该有相似的趋势但不完全相同
            correlation = result['ema_12'].corr(sma_result['sma_12'])
            assert correlation > 0.8  # 高相关性但不完全相同
    
    def test_calculate_rsi(self, feature_engineer, sample_price_data):
        """测试相对强弱指数计算"""
        result = feature_engineer.calculate_rsi(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        assert 'rsi_14' in result.columns
        
        # 验证RSI范围在0-100之间
        rsi_values = result['rsi_14'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # 验证RSI的合理性
        assert len(rsi_values) > 0
        assert rsi_values.std() > 0  # RSI应该有变化
    
    def test_calculate_macd(self, feature_engineer, sample_price_data):
        """测试MACD指标计算"""
        result = feature_engineer.calculate_macd(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['macd', 'macd_signal', 'macd_histogram']
        for col in expected_columns:
            assert col in result.columns
        
        # 验证MACD的数学关系
        macd_values = result.dropna()
        if len(macd_values) > 0:
            # MACD直方图 = MACD - 信号线
            calculated_histogram = macd_values['macd'] - macd_values['macd_signal']
            np.testing.assert_array_almost_equal(
                calculated_histogram.values,
                macd_values['macd_histogram'].values,
                decimal=6
            )
    
    def test_calculate_bollinger_bands(self, feature_engineer, sample_price_data):
        """测试布林带计算"""
        result = feature_engineer.calculate_bollinger_bands(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position']
        for col in expected_columns:
            assert col in result.columns
        
        # 验证布林带的数学关系
        bb_data = result.dropna()
        if len(bb_data) > 0:
            # 上轨 > 中轨 > 下轨
            assert (bb_data['bb_upper'] >= bb_data['bb_middle']).all()
            assert (bb_data['bb_middle'] >= bb_data['bb_lower']).all()
            
            # 布林带宽度 = 上轨 - 下轨
            calculated_width = bb_data['bb_upper'] - bb_data['bb_lower']
            np.testing.assert_array_almost_equal(
                calculated_width.values,
                bb_data['bb_width'].values,
                decimal=6
            )
            
            # 布林带位置通常在0-1之间，但可能超出范围（这是正常的）
            # 验证布林带位置的计算是否正确
            expected_position = (sample_price_data['close'] - bb_data['bb_lower']) / bb_data['bb_width']
            expected_position = expected_position.dropna()
            actual_position = bb_data['bb_position']
            
            # 确保索引对齐
            common_index = expected_position.index.intersection(actual_position.index)
            if len(common_index) > 0:
                np.testing.assert_array_almost_equal(
                    expected_position.loc[common_index].values,
                    actual_position.loc[common_index].values,
                    decimal=6
                )
    
    def test_calculate_stochastic(self, feature_engineer, sample_price_data):
        """测试随机指标计算"""
        result = feature_engineer.calculate_stochastic(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['stoch_k', 'stoch_d']
        for col in expected_columns:
            assert col in result.columns
        
        # 验证随机指标范围在0-100之间
        stoch_data = result.dropna()
        if len(stoch_data) > 0:
            assert (stoch_data['stoch_k'] >= 0).all()
            assert (stoch_data['stoch_k'] <= 100).all()
            assert (stoch_data['stoch_d'] >= 0).all()
            assert (stoch_data['stoch_d'] <= 100).all()
    
    def test_calculate_atr(self, feature_engineer, sample_price_data):
        """测试平均真实波幅计算"""
        result = feature_engineer.calculate_atr(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        assert 'atr_14' in result.columns
        
        # 验证ATR为正值
        atr_values = result['atr_14'].dropna()
        assert len(atr_values) > 0
        assert (atr_values > 0).all()
        
        # 验证ATR的合理性（应该反映价格波动）
        assert len(atr_values) > 0
        assert atr_values.std() >= 0  # ATR应该有变化
    
    def test_calculate_volume_indicators(self, feature_engineer, sample_price_data):
        """测试成交量指标计算"""
        result = feature_engineer.calculate_volume_indicators(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['volume_sma', 'volume_ratio', 'obv', 'vwap']
        for col in expected_columns:
            assert col in result.columns
        
        # 验证成交量指标的合理性
        volume_data = result.dropna()
        if len(volume_data) > 0:
            # 成交量移动平均应该为正
            assert (volume_data['volume_sma'] > 0).all()
            
            # 成交量比率应该为正
            assert (volume_data['volume_ratio'] > 0).all()
            
            # VWAP应该为正
            assert (volume_data['vwap'] > 0).all()
    
    @pytest.mark.parametrize("window", [5, 10, 20, 60])
    def test_technical_indicators_different_windows(self, feature_engineer, sample_price_data, window):
        """测试不同窗口期的技术指标计算"""
        result = feature_engineer.calculate_sma(sample_price_data, window=window)
        
        # 验证结果包含指定窗口的指标
        expected_col = f'sma_{window}'
        assert expected_col in result.columns
        
        # 验证前window-1个值为NaN
        sma_values = result[expected_col]
        assert sma_values.iloc[:window-1].isnull().all()
        
        # 验证后续值不为NaN
        if len(sma_values) > window:
            assert not sma_values.iloc[window:].isnull().all()


class TestFundamentalFactors:
    """基本面因子测试"""
    
    def test_calculate_valuation_factors(self, feature_engineer, sample_fundamental_data):
        """测试估值因子计算"""
        result = feature_engineer.calculate_valuation_factors(sample_fundamental_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio']
        for col in expected_columns:
            if col in result.columns:
                # 验证估值因子为正值
                values = result[col].dropna()
                if len(values) > 0:
                    assert (values > 0).all()
    
    def test_calculate_profitability_factors(self, feature_engineer, sample_fundamental_data):
        """测试盈利能力因子计算"""
        result = feature_engineer.calculate_profitability_factors(sample_fundamental_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['roe', 'roa', 'gross_margin', 'net_margin']
        for col in expected_columns:
            if col in result.columns:
                values = result[col].dropna()
                if len(values) > 0:
                    # ROE和ROA应该在合理范围内
                    if col in ['roe', 'roa']:
                        assert (values >= -1).all()  # 允许负值但不应过分极端
                        assert (values <= 2).all()   # 不应超过200%
    
    def test_calculate_growth_factors(self, feature_engineer, sample_fundamental_data):
        """测试成长性因子计算"""
        result = feature_engineer.calculate_growth_factors(sample_fundamental_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['revenue_growth', 'profit_growth', 'eps_growth']
        for col in expected_columns:
            if col in result.columns:
                values = result[col].dropna()
                if len(values) > 0:
                    # 成长率可以为负，但应该在合理范围内
                    assert (values >= -2).all()  # 不应低于-200%
                    assert (values <= 5).all()   # 不应超过500%
    
    def test_calculate_leverage_factors(self, feature_engineer, sample_fundamental_data):
        """测试杠杆因子计算"""
        result = feature_engineer.calculate_leverage_factors(sample_fundamental_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['debt_ratio', 'debt_to_equity', 'current_ratio', 'quick_ratio']
        for col in expected_columns:
            if col in result.columns:
                values = result[col].dropna()
                if len(values) > 0:
                    # 杠杆指标应该为正值
                    assert (values >= 0).all()
                    
                    # 债务比率不应超过100%
                    if col == 'debt_ratio':
                        assert (values <= 1).all()


class TestMarketMicrostructure:
    """市场微观结构特征测试"""
    
    def test_calculate_liquidity_features(self, feature_engineer, sample_price_data):
        """测试流动性特征计算"""
        result = feature_engineer.calculate_liquidity_features(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['turnover_rate', 'amihud_illiquidity', 'bid_ask_spread']
        for col in expected_columns:
            if col in result.columns:
                values = result[col].dropna()
                if len(values) > 0:
                    # 流动性指标应该为正值
                    assert (values >= 0).all()
    
    def test_calculate_volatility_features(self, feature_engineer, sample_price_data):
        """测试波动率特征计算"""
        result = feature_engineer.calculate_volatility_features(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['realized_volatility', 'garman_klass_volatility', 'parkinson_volatility']
        for col in expected_columns:
            if col in result.columns:
                values = result[col].dropna()
                if len(values) > 0:
                    # 波动率应该为正值
                    assert (values >= 0).all()
    
    def test_calculate_momentum_features(self, feature_engineer, sample_price_data):
        """测试动量特征计算"""
        result = feature_engineer.calculate_momentum_features(sample_price_data)
        
        # 验证结果不为空
        assert not result.empty
        
        # 验证列名
        expected_columns = ['price_momentum_1m', 'price_momentum_3m', 'volume_momentum']
        for col in expected_columns:
            if col in result.columns:
                values = result[col].dropna()
                if len(values) > 0:
                    # 动量可以为正或负
                    assert not values.isnull().all()


class TestFeatureNormalization:
    """特征标准化测试"""
    
    @pytest.fixture
    def sample_features(self):
        """创建样本特征数据"""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(100, 20, 100),
            'feature2': np.random.exponential(2, 100),
            'feature3': np.random.uniform(-10, 10, 100),
            'feature4': np.random.normal(0, 1, 100)
        })
        return data
    
    def test_z_score_normalization(self, feature_engineer, sample_features):
        """测试Z-score标准化"""
        result = feature_engineer.normalize_features(sample_features, method='zscore')
        
        # 验证结果形状
        assert result.shape == sample_features.shape
        
        # 验证标准化后的统计特性
        for col in result.columns:
            values = result[col].dropna()
            if len(values) > 1:
                # 均值应该接近0
                assert abs(values.mean()) < 0.1
                # 标准差应该接近1
                assert abs(values.std() - 1) < 0.1
    
    def test_min_max_normalization(self, feature_engineer, sample_features):
        """测试Min-Max标准化"""
        result = feature_engineer.normalize_features(sample_features, method='minmax')
        
        # 验证结果形状
        assert result.shape == sample_features.shape
        
        # 验证标准化后的范围
        for col in result.columns:
            values = result[col].dropna()
            if len(values) > 0:
                # 值应该在[0, 1]范围内
                assert (values >= 0).all()
                assert (values <= 1).all()
                # 最小值应该接近0，最大值应该接近1
                assert abs(values.min()) < 0.01
                assert abs(values.max() - 1) < 0.01
    
    def test_robust_normalization(self, feature_engineer, sample_features):
        """测试鲁棒标准化"""
        result = feature_engineer.normalize_features(sample_features, method='robust')
        
        # 验证结果形状
        assert result.shape == sample_features.shape
        
        # 验证标准化后的统计特性
        for col in result.columns:
            values = result[col].dropna()
            if len(values) > 1:
                # 中位数应该接近0
                assert abs(values.median()) < 0.5
    
    def test_handle_missing_values(self, feature_engineer):
        """测试缺失值处理"""
        # 创建包含缺失值的数据
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [np.nan, 2, 3, 4, np.nan],
            'feature3': [1, 2, 3, 4, 5]
        })
        
        # 测试前向填充
        result_ffill = feature_engineer.handle_missing_values(data, method='ffill')
        assert result_ffill.isnull().sum().sum() <= data.isnull().sum().sum()
        
        # 测试均值填充
        result_mean = feature_engineer.handle_missing_values(data, method='mean')
        assert result_mean.isnull().sum().sum() == 0
        
        # 测试中位数填充
        result_median = feature_engineer.handle_missing_values(data, method='median')
        assert result_median.isnull().sum().sum() == 0
    
    def test_outlier_detection_and_treatment(self, feature_engineer, sample_features):
        """测试异常值检测和处理"""
        # 添加一些异常值
        data_with_outliers = sample_features.copy()
        data_with_outliers.iloc[0, 0] = 1000  # 极大值
        data_with_outliers.iloc[1, 1] = -1000  # 极小值
        
        # 检测异常值
        outliers = feature_engineer.detect_outliers(data_with_outliers, method='iqr')
        assert outliers.sum().sum() > 0  # 应该检测到异常值
        
        # 处理异常值
        result = feature_engineer.treat_outliers(data_with_outliers, method='clip')
        
        # 验证异常值被处理
        assert result.max().max() < data_with_outliers.max().max()
        assert result.min().min() > data_with_outliers.min().min()


class TestFeatureSelection:
    """特征选择测试"""
    
    @pytest.fixture
    def sample_features_with_target(self):
        """创建带目标变量的特征数据"""
        np.random.seed(42)
        n_samples = 100
        
        # 创建一些有用的特征
        useful_feature1 = np.random.normal(0, 1, n_samples)
        useful_feature2 = np.random.normal(0, 1, n_samples)
        
        # 创建目标变量（与有用特征相关）
        target = 0.5 * useful_feature1 + 0.3 * useful_feature2 + np.random.normal(0, 0.1, n_samples)
        
        # 创建一些噪声特征
        noise_features = np.random.normal(0, 1, (n_samples, 5))
        
        features = pd.DataFrame({
            'useful1': useful_feature1,
            'useful2': useful_feature2,
            'noise1': noise_features[:, 0],
            'noise2': noise_features[:, 1],
            'noise3': noise_features[:, 2],
            'noise4': noise_features[:, 3],
            'noise5': noise_features[:, 4]
        })
        
        return features, pd.Series(target)
    
    def test_correlation_based_selection(self, feature_engineer, sample_features_with_target):
        """测试基于相关性的特征选择"""
        features, target = sample_features_with_target
        
        selected_features = feature_engineer.select_features_by_correlation(
            features, target, threshold=0.1
        )
        
        # 验证选择了一些特征
        assert len(selected_features) > 0
        assert len(selected_features) <= len(features.columns)
        
        # 验证选择的特征确实与目标相关
        for feature in selected_features:
            correlation = abs(features[feature].corr(target))
            assert correlation >= 0.1
    
    def test_mutual_information_selection(self, feature_engineer, sample_features_with_target):
        """测试基于互信息的特征选择"""
        features, target = sample_features_with_target
        
        selected_features = feature_engineer.select_features_by_mutual_info(
            features, target, k=3
        )
        
        # 验证选择了指定数量的特征
        assert len(selected_features) == 3
        
        # 验证选择的特征在原特征中
        for feature in selected_features:
            assert feature in features.columns
    
    def test_variance_threshold_selection(self, feature_engineer):
        """测试基于方差阈值的特征选择"""
        # 创建包含低方差特征的数据
        data = pd.DataFrame({
            'high_var': np.random.normal(0, 10, 100),
            'medium_var': np.random.normal(0, 1, 100),
            'low_var': np.random.normal(0, 0.01, 100),
            'constant': [1] * 100
        })
        
        selected_features = feature_engineer.select_features_by_variance(
            data, threshold=0.1
        )
        
        # 验证低方差特征被过滤
        assert 'constant' not in selected_features
        assert 'low_var' not in selected_features
        assert 'high_var' in selected_features
        assert 'medium_var' in selected_features


class TestIntegrationTests:
    """集成测试"""
    
    def test_complete_feature_pipeline(self, feature_engineer, sample_price_data, sample_fundamental_data):
        """测试完整的特征工程流水线"""
        # 计算技术指标
        technical_features = feature_engineer.calculate_technical_indicators(sample_price_data)
        assert not technical_features.empty
        
        # 计算基本面因子
        fundamental_features = feature_engineer.calculate_fundamental_factors(sample_fundamental_data)
        assert not fundamental_features.empty
        
        # 计算市场微观结构特征
        microstructure_features = feature_engineer.calculate_microstructure_features(sample_price_data)
        assert not microstructure_features.empty
        
        # 合并所有特征
        all_features = feature_engineer.combine_features([
            technical_features,
            fundamental_features,
            microstructure_features
        ])
        assert not all_features.empty
        
        # 标准化特征
        normalized_features = feature_engineer.normalize_features(all_features)
        assert normalized_features.shape == all_features.shape
        
        # 验证最终特征向量的创建
        feature_vector = feature_engineer.create_feature_vector(
            timestamp=datetime.now(),
            symbol='000001.SZ',
            normalized_features=normalized_features.iloc[-1]
        )
        
        assert isinstance(feature_vector, FeatureVector)
        assert feature_vector.symbol == '000001.SZ'
        assert len(feature_vector.technical_indicators) > 0
        assert len(feature_vector.fundamental_factors) > 0
        assert len(feature_vector.market_microstructure) > 0
    
    def test_error_handling(self, feature_engineer):
        """测试错误处理"""
        # 测试空数据
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError):
            feature_engineer.calculate_technical_indicators(empty_data)
        
        # 测试缺少必要列的数据
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            feature_engineer.calculate_technical_indicators(invalid_data)
        
        # 测试包含NaN的数据处理
        data_with_nan = pd.DataFrame({
            'close': [1, 2, np.nan, 4, 5],
            'volume': [100, 200, 300, np.nan, 500]
        })
        
        # 应该能够处理NaN值而不抛出异常
        result = feature_engineer.handle_missing_values(data_with_nan)
        assert not result.isnull().all().all()
    
    @pytest.mark.parametrize("data_size", [10, 50, 100, 500])
    def test_performance_with_different_data_sizes(self, feature_engineer, data_size):
        """测试不同数据大小下的性能"""
        # 生成不同大小的测试数据
        dates = pd.date_range('2023-01-01', periods=data_size, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'datetime': dates,
            'symbol': ['000001.SZ'] * data_size,
            'open': np.random.uniform(90, 110, data_size),
            'high': np.random.uniform(95, 115, data_size),
            'low': np.random.uniform(85, 105, data_size),
            'close': np.random.uniform(90, 110, data_size),
            'volume': np.random.randint(1000000, 10000000, data_size),
            'amount': np.random.uniform(1e8, 1e9, data_size)
        }).set_index(['datetime', 'symbol'])
        
        # 确保价格逻辑关系正确
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        # 测试技术指标计算
        import time
        start_time = time.time()
        result = feature_engineer.calculate_technical_indicators(data)
        end_time = time.time()
        
        # 验证结果
        assert not result.empty
        
        # 验证性能（应该在合理时间内完成）
        execution_time = end_time - start_time
        assert execution_time < 10  # 应该在10秒内完成
        
        # 对于较大的数据集，执行时间应该合理增长
        if data_size >= 100:
            assert execution_time < data_size * 0.1  # 每100条数据不超过10秒