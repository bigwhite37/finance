"""
数据加载器测试用例
测试数据初始化、加载和处理功能
"""
import pytest
import os
import sys
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import QlibDataLoader, split_data


class TestQlibDataLoader:
    """QlibDataLoader测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.data_root = "/tmp/test_qlib_data"
        self.loader = QlibDataLoader(self.data_root)
    
    def test_init(self):
        """测试初始化"""
        assert self.loader.data_root == self.data_root
        assert self.loader.is_initialized == False
    
    @patch('data_loader.qlib.init')
    @patch('data_loader.os.path.exists')
    def test_initialize_qlib_success(self, mock_exists, mock_qlib_init):
        """测试Qlib初始化成功"""
        # Mock目录存在
        mock_exists.return_value = True
        
        # Mock qlib初始化成功
        mock_qlib_init.return_value = None
        
        provider_uri = {
            "1min": "/tmp/test_data/1min",
            "day": "/tmp/test_data/day"
        }
        
        self.loader.initialize_qlib(provider_uri)
        
        assert self.loader.is_initialized == True
        mock_qlib_init.assert_called_once()
    
    @patch('data_loader.os.path.exists')
    def test_initialize_qlib_missing_directory(self, mock_exists):
        """测试目录缺失时的错误处理"""
        mock_exists.return_value = False
        
        provider_uri = {
            "1min": "/tmp/nonexistent",
            "day": "/tmp/nonexistent2"
        }
        
        with pytest.raises(FileNotFoundError):
            self.loader.initialize_qlib(provider_uri)
    
    @patch('data_loader.qlib.init')
    @patch('data_loader.os.path.exists')
    def test_initialize_qlib_failure(self, mock_exists, mock_qlib_init):
        """测试Qlib初始化失败"""
        mock_exists.return_value = True
        mock_qlib_init.side_effect = Exception("初始化失败")
        
        provider_uri = {"day": "/tmp/test"}
        
        with pytest.raises(RuntimeError, match="Qlib初始化失败"):
            self.loader.initialize_qlib(provider_uri)
    
    @patch('data_loader.D.instruments')
    def test_get_stock_list_success(self, mock_instruments):
        """测试获取股票列表成功"""
        self.loader.is_initialized = True
        
        # Mock返回股票字典
        mock_instruments.return_value = {
            "000001.SZ": {"sector": "bank"},
            "000002.SZ": {"sector": "tech"},
            "600000.SH": {"sector": "bank"}
        }
        
        stocks = self.loader.get_stock_list(market="all", limit=2)
        
        assert len(stocks) == 2
        assert all(stock in ["000001.SZ", "000002.SZ", "600000.SH"] for stock in stocks)
    
    def test_get_stock_list_not_initialized(self):
        """测试未初始化时获取股票列表"""
        with pytest.raises(RuntimeError, match="请先初始化Qlib"):
            self.loader.get_stock_list()
    
    @patch('data_loader.D.features')
    def test_load_data_success(self, mock_features):
        """测试加载数据成功"""
        self.loader.is_initialized = True
        
        # Mock返回数据
        mock_data = pd.DataFrame({
            "$close": [100, 101, 102],
            "$volume": [1000, 1100, 1200]
        }, index=pd.MultiIndex.from_tuples([
            ("2023-01-01", "000001.SZ"),
            ("2023-01-02", "000001.SZ"),
            ("2023-01-03", "000001.SZ")
        ], names=["datetime", "instrument"]))
        
        mock_features.return_value = mock_data
        
        result = self.loader.load_data(
            instruments=["000001.SZ"],
            start_time="2023-01-01",
            end_time="2023-01-03",
            fields=["$close", "$volume"]
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)
        assert "$close" in result.columns
        assert "$volume" in result.columns
    
    def test_load_data_not_initialized(self):
        """测试未初始化时加载数据"""
        with pytest.raises(RuntimeError, match="请先初始化Qlib"):
            self.loader.load_data(
                instruments=["000001.SZ"],
                start_time="2023-01-01",
                end_time="2023-01-03"
            )
    
    @patch('data_loader.D.calendar')
    def test_get_calendar_success(self, mock_calendar):
        """测试获取交易日历成功"""
        self.loader.is_initialized = True
        
        # Mock日历数据
        mock_calendar.return_value = pd.DatetimeIndex([
            "2023-01-01", "2023-01-02", "2023-01-03"
        ])
        
        calendar = self.loader.get_calendar(
            start_time="2023-01-01",
            end_time="2023-01-03"
        )
        
        assert len(calendar) == 3
        assert all(isinstance(date, str) for date in calendar)


class TestSplitData:
    """数据分割测试类"""
    
    def setup_method(self):
        """设置测试数据"""
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
        self.test_data = pd.DataFrame({
            "$close": np.random.randn(len(dates)) + 100,
            "$volume": np.random.randint(1000, 10000, len(dates))
        }, index=dates)
    
    def test_split_data_success(self):
        """测试数据分割成功"""
        train_data, valid_data, test_data = split_data(
            self.test_data,
            train_start="2020-01-01", train_end="2022-12-31",
            valid_start="2023-01-01", valid_end="2023-06-30", 
            test_start="2023-07-01", test_end="2023-12-31"
        )
        
        # 检查数据分割正确
        assert train_data.index.min() >= pd.Timestamp("2020-01-01")
        assert train_data.index.max() <= pd.Timestamp("2022-12-31")
        
        assert valid_data.index.min() >= pd.Timestamp("2023-01-01")
        assert valid_data.index.max() <= pd.Timestamp("2023-06-30")
        
        assert test_data.index.min() >= pd.Timestamp("2023-07-01")
        assert test_data.index.max() <= pd.Timestamp("2023-12-31")
    
    def test_split_data_empty_result(self):
        """测试分割后空结果"""
        # 使用不存在的日期范围
        train_data, valid_data, test_data = split_data(
            self.test_data,
            train_start="2030-01-01", train_end="2030-12-31",
            valid_start="2031-01-01", valid_end="2031-06-30",
            test_start="2031-07-01", test_end="2031-12-31"
        )
        
        assert train_data.empty
        assert valid_data.empty
        assert test_data.empty


class TestDataLoaderIntegration:
    """数据加载器集成测试"""
    
    @pytest.fixture
    def mock_qlib_environment(self):
        """Mock Qlib环境"""
        with patch('data_loader.qlib.init'), \
             patch('data_loader.os.path.exists', return_value=True), \
             patch('data_loader.D.instruments') as mock_instruments, \
             patch('data_loader.D.features') as mock_features, \
             patch('data_loader.D.calendar') as mock_calendar:
            
            # 设置Mock返回值
            mock_instruments.return_value = {
                f"00000{i}.SZ": {"sector": "test"} for i in range(1, 6)
            }
            
            # 创建模拟数据
            dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
            stocks = [f"00000{i}.SZ" for i in range(1, 4)]
            
            data_list = []
            for date in dates[:100]:  # 限制数据量
                for stock in stocks:
                    data_list.append({
                        "$close": 100 + np.random.randn(),
                        "$volume": np.random.randint(1000, 10000),
                        "datetime": date,
                        "instrument": stock
                    })
            
            mock_data = pd.DataFrame(data_list)
            mock_data = mock_data.set_index(["datetime", "instrument"])
            mock_features.return_value = mock_data
            
            mock_calendar.return_value = pd.DatetimeIndex(dates[:100])
            
            yield {
                'instruments': mock_instruments,
                'features': mock_features,
                'calendar': mock_calendar
            }
    
    def test_full_workflow(self, mock_qlib_environment):
        """测试完整工作流程"""
        loader = QlibDataLoader()
        
        # 1. 初始化
        loader.initialize_qlib()
        assert loader.is_initialized
        
        # 2. 获取股票列表
        stocks = loader.get_stock_list(limit=3)
        assert len(stocks) == 3
        
        # 3. 加载数据
        data = loader.load_data(
            instruments=stocks,
            start_time="2023-01-01",
            end_time="2023-03-31"
        )
        assert not data.empty
        assert data.shape[1] >= 2  # 至少有close和volume列
        
        # 4. 获取日历
        calendar = loader.get_calendar("2023-01-01", "2023-03-31")
        assert len(calendar) > 0
        
        # 5. 数据分割
        train_data, valid_data, test_data = split_data(
            data,
            train_start="2023-01-01", train_end="2023-02-28",
            valid_start="2023-03-01", valid_end="2023-03-15",
            test_start="2023-03-16", test_end="2023-03-31"
        )
        
        assert not train_data.empty
        assert not valid_data.empty
        assert not test_data.empty


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])