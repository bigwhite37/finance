"""
数据接口测试用例
测试DataInterface抽象类和具体实现，包括Qlib和Akshare数据获取功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.rl_trading_system.data.interfaces import DataInterface
from src.rl_trading_system.data.qlib_interface import QlibDataInterface
from src.rl_trading_system.data.akshare_interface import AkshareDataInterface
from src.rl_trading_system.data.data_models import MarketData


class TestDataInterface:
    """测试DataInterface抽象类"""
    
    def test_abstract_class_cannot_be_instantiated(self):
        """测试抽象类不能直接实例化"""
        with pytest.raises(TypeError):
            DataInterface()
    
    def test_abstract_methods_must_be_implemented(self):
        """测试抽象方法必须被实现"""
        class IncompleteInterface(DataInterface):
            pass
        
        with pytest.raises(TypeError):
            IncompleteInterface()
    
    def test_complete_implementation_can_be_instantiated(self):
        """测试完整实现可以被实例化"""
        class CompleteInterface(DataInterface):
            def get_stock_list(self, market: str = 'A') -> List[str]:
                return ['000001.SZ']
            
            def get_price_data(self, symbols: List[str], 
                              start_date: str, end_date: str) -> pd.DataFrame:
                return pd.DataFrame()
            
            def get_fundamental_data(self, symbols: List[str], 
                                   start_date: str, end_date: str) -> pd.DataFrame:
                return pd.DataFrame()
        
        interface = CompleteInterface()
        assert isinstance(interface, DataInterface)


class TestQlibDataInterface:
    """测试QlibDataInterface实现"""
    
    @pytest.fixture
    def qlib_interface(self):
        """创建QlibDataInterface实例"""
        return QlibDataInterface(provider_uri="test://provider")
    
    @pytest.fixture
    def sample_stock_list(self):
        """示例股票列表"""
        return ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
    
    @pytest.fixture
    def sample_price_data(self):
        """示例价格数据"""
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        symbols = ['000001.SZ', '000002.SZ']
        
        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'datetime': date,
                    'instrument': symbol,
                    '$open': 10.0 + np.random.random(),
                    '$high': 11.0 + np.random.random(),
                    '$low': 9.0 + np.random.random(),
                    '$close': 10.5 + np.random.random(),
                    '$volume': 1000000 + np.random.randint(0, 500000),
                    '$amount': 10000000 + np.random.randint(0, 5000000)
                })
        
        df = pd.DataFrame(data)
        df.set_index(['datetime', 'instrument'], inplace=True)
        return df
    
    @pytest.fixture
    def sample_fundamental_data(self):
        """示例基本面数据"""
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        symbols = ['000001.SZ', '000002.SZ']
        
        data = []
        for date in dates:
            for symbol in symbols:
                data.append({
                    'datetime': date,
                    'instrument': symbol,
                    'pe_ratio': 15.0 + np.random.random() * 10,
                    'pb_ratio': 1.5 + np.random.random() * 2,
                    'market_cap': 1000000000 + np.random.randint(0, 500000000),
                    'total_revenue': 100000000 + np.random.randint(0, 50000000)
                })
        
        df = pd.DataFrame(data)
        df.set_index(['datetime', 'instrument'], inplace=True)
        return df
    
    def test_initialization(self, qlib_interface):
        """测试初始化"""
        assert qlib_interface.provider_uri == "test://provider"
        assert isinstance(qlib_interface, DataInterface)
    
    def test_initialization_with_default_provider(self):
        """测试使用默认provider初始化"""
        interface = QlibDataInterface()
        assert interface.provider_uri is None
    
    @patch('qlib.init')
    @patch('qlib.D.instruments')
    def test_get_stock_list_success(self, mock_instruments, mock_init, 
                                   qlib_interface, sample_stock_list):
        """测试成功获取股票列表"""
        mock_instruments.return_value = sample_stock_list
        
        result = qlib_interface.get_stock_list('A')
        
        assert result == sample_stock_list
        mock_init.assert_called_once()
        mock_instruments.assert_called_once_with(market='A')
    
    @patch('qlib.init')
    @patch('qlib.D.instruments')
    def test_get_stock_list_different_markets(self, mock_instruments, mock_init, 
                                             qlib_interface):
        """测试获取不同市场的股票列表"""
        mock_instruments.return_value = ['000001.SZ']
        
        # 测试A股市场
        qlib_interface.get_stock_list('A')
        mock_instruments.assert_called_with(market='A')
        
        # 测试港股市场
        qlib_interface.get_stock_list('HK')
        mock_instruments.assert_called_with(market='HK')
    
    @patch('qlib.init')
    @patch('qlib.D.instruments')
    def test_get_stock_list_exception_handling(self, mock_instruments, mock_init, 
                                              qlib_interface):
        """测试获取股票列表异常处理"""
        mock_instruments.side_effect = Exception("Qlib connection error")
        
        with pytest.raises(Exception) as exc_info:
            qlib_interface.get_stock_list('A')
        
        assert "Qlib connection error" in str(exc_info.value)
    
    @patch('qlib.init')
    @patch('qlib.D.features')
    def test_get_price_data_success(self, mock_features, mock_init, 
                                   qlib_interface, sample_price_data):
        """测试成功获取价格数据"""
        mock_features.return_value = sample_price_data
        
        symbols = ['000001.SZ', '000002.SZ']
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        
        result = qlib_interface.get_price_data(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_init.assert_called_once()
        mock_features.assert_called_once()
    
    @patch('qlib.init')
    @patch('qlib.D.features')
    def test_get_price_data_empty_symbols(self, mock_features, mock_init, 
                                         qlib_interface):
        """测试空股票列表"""
        mock_features.return_value = pd.DataFrame()
        
        result = qlib_interface.get_price_data([], '2023-01-01', '2023-01-10')
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('qlib.init')
    @patch('qlib.D.features')
    def test_get_price_data_invalid_date_range(self, mock_features, mock_init, 
                                              qlib_interface):
        """测试无效日期范围"""
        mock_features.side_effect = ValueError("Invalid date range")
        
        with pytest.raises(ValueError) as exc_info:
            qlib_interface.get_price_data(['000001.SZ'], '2023-01-10', '2023-01-01')
        
        assert "Invalid date range" in str(exc_info.value)
    
    @patch('qlib.init')
    @patch('qlib.D.features')
    def test_get_fundamental_data_success(self, mock_features, mock_init, 
                                         qlib_interface, sample_fundamental_data):
        """测试成功获取基本面数据"""
        mock_features.return_value = sample_fundamental_data
        
        symbols = ['000001.SZ', '000002.SZ']
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        
        result = qlib_interface.get_fundamental_data(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_init.assert_called_once()
        mock_features.assert_called_once()
    
    @patch('qlib.init')
    @patch('qlib.D.features')
    def test_get_fundamental_data_missing_data(self, mock_features, mock_init, 
                                              qlib_interface):
        """测试基本面数据缺失"""
        mock_features.return_value = pd.DataFrame()
        
        result = qlib_interface.get_fundamental_data(['000001.SZ'], 
                                                    '2023-01-01', '2023-01-10')
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestAkshareDataInterface:
    """测试AkshareDataInterface实现"""
    
    @pytest.fixture
    def akshare_interface(self):
        """创建AkshareDataInterface实例"""
        return AkshareDataInterface()
    
    @pytest.fixture
    def sample_stock_list_akshare(self):
        """示例Akshare股票列表"""
        return pd.DataFrame({
            'code': ['000001', '000002', '600000', '600036'],
            'name': ['平安银行', '万科A', '浦发银行', '招商银行'],
            'market': ['sz', 'sz', 'sh', 'sh']
        })
    
    @pytest.fixture
    def sample_price_data_akshare(self):
        """示例Akshare价格数据"""
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        
        data = []
        for date in dates:
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': 10.0 + np.random.random(),
                'high': 11.0 + np.random.random(),
                'low': 9.0 + np.random.random(),
                'close': 10.5 + np.random.random(),
                'volume': 1000000 + np.random.randint(0, 500000),
                'amount': 10000000 + np.random.randint(0, 5000000)
            })
        
        return pd.DataFrame(data)
    
    def test_initialization(self, akshare_interface):
        """测试初始化"""
        assert isinstance(akshare_interface, DataInterface)
    
    @patch('akshare.stock_info_a_code_name')
    def test_get_stock_list_success(self, mock_stock_info, akshare_interface, 
                                   sample_stock_list_akshare):
        """测试成功获取股票列表"""
        mock_stock_info.return_value = sample_stock_list_akshare
        
        result = akshare_interface.get_stock_list('A')
        
        assert isinstance(result, list)
        assert len(result) > 0
        mock_stock_info.assert_called_once()
    
    @patch('akshare.stock_info_a_code_name')
    def test_get_stock_list_exception_handling(self, mock_stock_info, 
                                              akshare_interface):
        """测试获取股票列表异常处理"""
        mock_stock_info.side_effect = Exception("Akshare API error")
        
        with pytest.raises(Exception) as exc_info:
            akshare_interface.get_stock_list('A')
        
        assert "Akshare API error" in str(exc_info.value)
    
    @patch('akshare.stock_zh_a_hist')
    def test_get_price_data_success(self, mock_stock_hist, akshare_interface, 
                                   sample_price_data_akshare):
        """测试成功获取价格数据"""
        mock_stock_hist.return_value = sample_price_data_akshare
        
        symbols = ['000001']
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        
        result = akshare_interface.get_price_data(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        mock_stock_hist.assert_called()
    
    @patch('akshare.stock_zh_a_hist')
    def test_get_price_data_multiple_symbols(self, mock_stock_hist, 
                                            akshare_interface, sample_price_data_akshare):
        """测试获取多个股票的价格数据"""
        mock_stock_hist.return_value = sample_price_data_akshare
        
        symbols = ['000001', '000002', '600000']
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        
        result = akshare_interface.get_price_data(symbols, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        # 应该为每个股票调用一次API
        assert mock_stock_hist.call_count == len(symbols)
    
    @patch('akshare.stock_zh_a_hist')
    def test_get_price_data_api_failure(self, mock_stock_hist, akshare_interface):
        """测试API调用失败"""
        mock_stock_hist.side_effect = Exception("API rate limit exceeded")
        
        with pytest.raises(Exception) as exc_info:
            akshare_interface.get_price_data(['000001'], '2023-01-01', '2023-01-10')
        
        assert "API rate limit exceeded" in str(exc_info.value)
    
    def test_get_fundamental_data_not_implemented(self, akshare_interface):
        """测试基本面数据获取（当前未实现）"""
        result = akshare_interface.get_fundamental_data(['000001'], 
                                                       '2023-01-01', '2023-01-10')
        
        assert isinstance(result, pd.DataFrame)
        # 当前实现返回空DataFrame


class TestDataFormatUnification:
    """测试数据格式统一"""
    
    @pytest.fixture
    def qlib_interface(self):
        return QlibDataInterface()
    
    @pytest.fixture
    def akshare_interface(self):
        return AkshareDataInterface()
    
    def test_price_data_format_consistency(self, qlib_interface, akshare_interface):
        """测试价格数据格式一致性"""
        symbols = ['000001.SZ']
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        
        with patch('qlib.init'), patch('qlib.D.features') as mock_qlib_features:
            # 模拟Qlib数据格式
            qlib_data = pd.DataFrame({
                'datetime': pd.date_range('2023-01-01', '2023-01-10'),
                'instrument': ['000001.SZ'] * 10,
                '$open': np.random.random(10) * 10 + 10,
                '$high': np.random.random(10) * 10 + 11,
                '$low': np.random.random(10) * 10 + 9,
                '$close': np.random.random(10) * 10 + 10,
                '$volume': np.random.randint(1000000, 2000000, 10),
                '$amount': np.random.randint(10000000, 20000000, 10)
            }).set_index(['datetime', 'instrument'])
            mock_qlib_features.return_value = qlib_data
            
            qlib_result = qlib_interface.get_price_data(symbols, start_date, end_date)
        
        with patch('akshare.stock_zh_a_hist') as mock_akshare_hist:
            # 模拟Akshare数据格式
            akshare_data = pd.DataFrame({
                'date': pd.date_range('2023-01-01', '2023-01-10').strftime('%Y-%m-%d'),
                'open': np.random.random(10) * 10 + 10,
                'high': np.random.random(10) * 10 + 11,
                'low': np.random.random(10) * 10 + 9,
                'close': np.random.random(10) * 10 + 10,
                'volume': np.random.randint(1000000, 2000000, 10),
                'amount': np.random.randint(10000000, 20000000, 10)
            })
            mock_akshare_hist.return_value = akshare_data
            
            akshare_result = akshare_interface.get_price_data(['000001'], 
                                                             start_date, end_date)
        
        # 验证两个接口返回的数据格式一致
        assert isinstance(qlib_result, pd.DataFrame)
        assert isinstance(akshare_result, pd.DataFrame)
    
    def test_data_validation_with_market_data_model(self):
        """测试使用MarketData模型进行数据验证"""
        # 创建有效的市场数据
        valid_data = MarketData(
            timestamp=datetime.now(),
            symbol='000001.SZ',
            open_price=10.0,
            high_price=11.0,
            low_price=9.0,
            close_price=10.5,
            volume=1000000,
            amount=10500000
        )
        
        assert valid_data.symbol == '000001.SZ'
        assert valid_data.high_price > valid_data.low_price
        
        # 测试无效数据
        with pytest.raises(ValueError):
            MarketData(
                timestamp=datetime.now(),
                symbol='000001.SZ',
                open_price=10.0,
                high_price=9.0,  # 最高价低于最低价
                low_price=11.0,
                close_price=10.5,
                volume=1000000,
                amount=10500000
            )


class TestDataQualityChecks:
    """测试数据质量检查"""
    
    def test_missing_data_detection(self):
        """测试缺失数据检测"""
        # 创建包含缺失值的数据
        data = pd.DataFrame({
            'open': [10.0, np.nan, 12.0],
            'high': [11.0, 13.0, np.nan],
            'low': [9.0, 11.0, 11.5],
            'close': [10.5, 12.5, 12.0],
            'volume': [1000000, 1200000, 1100000]
        })
        
        # 检测缺失值
        missing_data = data.isnull().sum()
        assert missing_data['open'] == 1
        assert missing_data['high'] == 1
        assert missing_data['low'] == 0
    
    def test_outlier_detection(self):
        """测试异常值检测"""
        # 创建包含异常值的数据
        normal_prices = np.random.normal(10, 1, 100)
        outlier_prices = np.append(normal_prices, [100, -5])  # 添加异常值
        
        # 使用3σ规则检测异常值
        mean_price = np.mean(normal_prices)
        std_price = np.std(normal_prices)
        
        outliers = np.abs(outlier_prices - mean_price) > 3 * std_price
        assert np.sum(outliers) >= 2  # 至少检测到2个异常值
    
    def test_data_consistency_checks(self):
        """测试数据一致性检查"""
        # 测试价格关系一致性
        data = pd.DataFrame({
            'open': [10.0, 11.0, 12.0],
            'high': [11.0, 12.0, 13.0],
            'low': [9.0, 10.0, 11.0],
            'close': [10.5, 11.5, 12.5],
            'volume': [1000000, 1200000, 1100000]
        })
        
        # 检查high >= low
        assert (data['high'] >= data['low']).all()
        
        # 检查价格在合理范围内
        assert (data['high'] >= data['open']).all() or (data['high'] >= data['close']).all()
        assert (data['low'] <= data['open']).all() or (data['low'] <= data['close']).all()
    
    def test_volume_amount_consistency(self):
        """测试成交量和成交额一致性"""
        data = pd.DataFrame({
            'close': [10.0, 11.0, 12.0],
            'volume': [1000000, 1200000, 1100000],
            'amount': [10000000, 13200000, 13200000]
        })
        
        # 计算平均价格
        avg_price = data['amount'] / data['volume']
        
        # 验证平均价格在合理范围内（接近收盘价）
        price_diff_ratio = np.abs(avg_price - data['close']) / data['close']
        assert (price_diff_ratio < 0.1).all()  # 差异小于10%


class TestErrorHandling:
    """测试错误处理"""
    
    @pytest.fixture
    def qlib_interface(self):
        return QlibDataInterface()
    
    @pytest.fixture
    def akshare_interface(self):
        return AkshareDataInterface()
    
    def test_network_error_handling(self, qlib_interface):
        """测试网络错误处理"""
        with patch('qlib.init') as mock_init:
            mock_init.side_effect = ConnectionError("Network connection failed")
            
            with pytest.raises(ConnectionError):
                qlib_interface.get_stock_list('A')
    
    def test_api_rate_limit_handling(self, akshare_interface):
        """测试API限流处理"""
        with patch('akshare.stock_info_a_code_name') as mock_api:
            mock_api.side_effect = Exception("API rate limit exceeded")
            
            with pytest.raises(Exception) as exc_info:
                akshare_interface.get_stock_list('A')
            
            assert "rate limit" in str(exc_info.value).lower()
    
    def test_invalid_symbol_handling(self, qlib_interface):
        """测试无效股票代码处理"""
        with patch('qlib.init'), patch('qlib.D.features') as mock_features:
            mock_features.side_effect = ValueError("Invalid symbol")
            
            with pytest.raises(ValueError):
                qlib_interface.get_price_data(['INVALID'], '2023-01-01', '2023-01-10')
    
    def test_date_format_validation(self, qlib_interface):
        """测试日期格式验证"""
        with patch('qlib.init'), patch('qlib.D.features') as mock_features:
            mock_features.side_effect = ValueError("Invalid date format")
            
            with pytest.raises(ValueError):
                qlib_interface.get_price_data(['000001.SZ'], 'invalid-date', '2023-01-10')


@pytest.mark.integration
class TestDataInterfaceIntegration:
    """数据接口集成测试"""
    
    def test_data_pipeline_integration(self):
        """测试数据管道集成"""
        # 这是一个集成测试示例，实际运行需要真实的数据源
        pass
    
    def test_cache_mechanism_integration(self):
        """测试缓存机制集成"""
        # 测试数据缓存功能
        pass
    
    def test_data_source_failover(self):
        """测试数据源故障转移"""
        # 测试当主数据源失败时，自动切换到备用数据源
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])