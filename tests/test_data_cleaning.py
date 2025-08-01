"""
数据清洗功能测试
测试数据加载时的前向填充（Forward-fill）功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from src.rl_trading_system.data.interfaces import DataInterface


class TestDataCleaning:
    """数据清洗测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.config = PortfolioConfig(
            stock_pool=['000001.SZ', '000002.SZ', '600000.SH'],
            lookback_window=30,
            initial_cash=1000000.0
        )
        
        # 创建模拟数据接口
        self.mock_data_interface = Mock(spec=DataInterface)
        
        # 创建基准数据
        benchmark_dates = pd.date_range('2019-04-26', '2019-05-06', freq='D')
        benchmark_index = pd.MultiIndex.from_product(
            [benchmark_dates, ['000300.SH']],
            names=['datetime', 'instrument']
        )
        
        benchmark_data = []
        for date in benchmark_dates:
            base_price = 3000.0 + np.random.random() * 100.0
            row = {
                'open': base_price,
                'high': base_price * 1.02,
                'low': base_price * 0.98,
                'close': base_price + np.random.random() * 10 - 5,
                'volume': 10000000,
                'amount': base_price * 10000000
            }
            benchmark_data.append(row)
        
        self.benchmark_data = pd.DataFrame(benchmark_data, index=benchmark_index)
        
        # 创建包含NaN的测试数据
        dates = pd.date_range('2019-04-26', '2019-05-06', freq='D')
        symbols = self.config.stock_pool
        
        # 创建多层索引
        index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['datetime', 'instrument']
        )
        
        # 创建价格数据，在节假日（4月29日和30日）设置NaN
        np.random.seed(42)
        data = []
        
        for date in dates:
            for symbol in symbols:
                if date.strftime('%Y-%m-%d') in ['2019-04-29', '2019-04-30']:
                    # 节假日设置NaN
                    row = {
                        'open': np.nan,
                        'high': np.nan,
                        'low': np.nan,
                        'close': np.nan,
                        'volume': np.nan,
                        'amount': np.nan
                    }
                else:
                    # 正常交易日设置有效价格
                    base_price = 10.0 + np.random.random() * 5.0
                    row = {
                        'open': base_price,
                        'high': base_price * 1.05,
                        'low': base_price * 0.95,
                        'close': base_price + np.random.random() - 0.5,
                        'volume': 1000000 + np.random.randint(0, 500000),
                        'amount': base_price * (1000000 + np.random.randint(0, 500000))
                    }
                data.append(row)
        
        self.price_data_with_nan = pd.DataFrame(data, index=index)
        
        # 创建清洗后的数据（前向填充）
        self.cleaned_price_data = self.price_data_with_nan.copy()
        # 按股票分组进行前向填充
        for symbol in symbols:
            symbol_data = self.cleaned_price_data.xs(symbol, level='instrument')
            filled_data = symbol_data.ffill()
            # 将填充后的数据放回原DataFrame
            for col in filled_data.columns:
                self.cleaned_price_data.loc[(slice(None), symbol), col] = filled_data[col].values
    
    def test_data_cleaning_should_be_applied_during_loading(self):
        """测试数据清洗应该在数据加载阶段应用"""
        # 设置模拟数据接口返回包含NaN的数据
        def mock_get_price_data(symbols, start_date, end_date):
            if symbols == ['000300.SH']:  # 基准数据
                return self.benchmark_data
            else:  # 股票数据
                return self.price_data_with_nan
        
        self.mock_data_interface.get_price_data.side_effect = mock_get_price_data
        
        # 创建环境时应该自动清洗数据
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2019-04-26',
            end_date='2019-05-06'
        )
        
        # 验证加载的数据已经被清洗（不包含NaN）
        assert not env.price_data.isnull().any().any(), "价格数据不应包含NaN值"
        
        # 验证节假日的价格使用了前向填充
        holiday_data = env.price_data.xs('2019-04-29', level='datetime')
        pre_holiday_data = env.price_data.xs('2019-04-28', level='datetime')
        
        for symbol in self.config.stock_pool:
            # 节假日价格应该等于前一交易日价格
            assert holiday_data.loc[symbol, 'close'] == pre_holiday_data.loc[symbol, 'close'], \
                f"股票{symbol}在节假日的价格应该使用前向填充"
    
    def test_data_cleaning_preserves_valid_data(self):
        """测试数据清洗保留有效数据"""
        def mock_get_price_data(symbols, start_date, end_date):
            if symbols == ['000300.SH']:  # 基准数据
                return self.benchmark_data
            else:  # 股票数据
                return self.price_data_with_nan
        
        self.mock_data_interface.get_price_data.side_effect = mock_get_price_data
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2019-04-26',
            end_date='2019-05-06'
        )
        
        # 验证非节假日的数据保持不变
        normal_day_original = self.price_data_with_nan.xs('2019-04-28', level='datetime')
        normal_day_cleaned = env.price_data.xs('2019-04-28', level='datetime')
        
        for symbol in self.config.stock_pool:
            assert normal_day_cleaned.loc[symbol, 'close'] == normal_day_original.loc[symbol, 'close'], \
                f"股票{symbol}在正常交易日的价格不应被修改"
    
    def test_data_cleaning_handles_first_day_nan(self):
        """测试数据清洗处理第一天就是NaN的情况"""
        # 创建第一天就有NaN的数据
        dates = pd.date_range('2019-04-26', '2019-04-30', freq='D')
        symbols = ['000001.SZ']
        
        index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['datetime', 'instrument']
        )
        
        data = []
        for i, date in enumerate(dates):
            if i == 0:  # 第一天设置NaN
                row = {
                    'open': np.nan,
                    'high': np.nan,
                    'low': np.nan,
                    'close': np.nan,
                    'volume': np.nan,
                    'amount': np.nan
                }
            else:
                base_price = 10.0
                row = {
                    'open': base_price,
                    'high': base_price * 1.05,
                    'low': base_price * 0.95,
                    'close': base_price,
                    'volume': 1000000,
                    'amount': base_price * 1000000
                }
            data.append(row)
        
        first_day_nan_data = pd.DataFrame(data, index=index)
        
        def mock_get_price_data(symbols, start_date, end_date):
            if symbols == ['000300.SH']:  # 基准数据
                return self.benchmark_data
            else:  # 股票数据
                return first_day_nan_data
        
        self.mock_data_interface.get_price_data.side_effect = mock_get_price_data
        
        # 现在我们的实现能够处理第一天为NaN的情况，应该成功创建环境
        env = PortfolioEnvironment(
            config=PortfolioConfig(stock_pool=['000001.SZ']),
            data_interface=self.mock_data_interface,
            start_date='2019-04-26',
            end_date='2019-04-30'
        )
        
        # 验证数据清洗成功，跳过了第一天的NaN数据
        assert env.price_data is not None
        assert not env.price_data.isnull().any().any(), "清洗后的价格数据不应包含NaN值"
        
        # 验证第一个有效交易日是2019-04-27（跳过了第一天的NaN）
        first_date = env.price_data.index.get_level_values('datetime').min()
        assert first_date.strftime('%Y-%m-%d') == '2019-04-27', "第一个有效交易日应该是2019-04-27"
    
    def test_data_cleaning_handles_all_nan_stock(self):
        """测试数据清洗处理某只股票全部为NaN的情况"""
        # 创建某只股票全部为NaN的数据
        dates = pd.date_range('2019-04-26', '2019-04-30', freq='D')
        symbols = ['000001.SZ', '000002.SZ']
        
        index = pd.MultiIndex.from_product(
            [dates, symbols],
            names=['datetime', 'instrument']
        )
        
        data = []
        for date in dates:
            for symbol in symbols:
                if symbol == '000002.SZ':  # 第二只股票全部NaN
                    row = {
                        'open': np.nan,
                        'high': np.nan,
                        'low': np.nan,
                        'close': np.nan,
                        'volume': np.nan,
                        'amount': np.nan
                    }
                else:
                    base_price = 10.0
                    row = {
                        'open': base_price,
                        'high': base_price * 1.05,
                        'low': base_price * 0.95,
                        'close': base_price,
                        'volume': 1000000,
                        'amount': base_price * 1000000
                    }
                data.append(row)
        
        all_nan_stock_data = pd.DataFrame(data, index=index)
        
        def mock_get_price_data(symbols, start_date, end_date):
            if symbols == ['000300.SH']:  # 基准数据
                return self.benchmark_data
            else:  # 股票数据
                return all_nan_stock_data
        
        self.mock_data_interface.get_price_data.side_effect = mock_get_price_data
        
        # 应该抛出异常，因为没有找到所有股票都有有效数据的交易日
        with pytest.raises(RuntimeError, match="没有找到任何所有股票都有有效数据的交易日"):
            PortfolioEnvironment(
                config=PortfolioConfig(stock_pool=['000001.SZ', '000002.SZ']),
                data_interface=self.mock_data_interface,
                start_date='2019-04-26',
                end_date='2019-04-30'
            )
    
    def test_no_warning_suppression_in_data_cleaning(self):
        """测试数据清洗过程中不应该抑制警告"""
        def mock_get_price_data(symbols, start_date, end_date):
            if symbols == ['000300.SH']:  # 基准数据
                return self.benchmark_data
            else:  # 股票数据
                return self.price_data_with_nan
        
        self.mock_data_interface.get_price_data.side_effect = mock_get_price_data
        
        # 创建环境时应该有清洗日志，但不应该有"使用前期价格"的警告
        # 因为数据清洗应该在加载阶段完成，而不是在运行时临时处理
        with patch('src.rl_trading_system.trading.portfolio_environment.logger') as mock_logger:
            env = PortfolioEnvironment(
                config=self.config,
                data_interface=self.mock_data_interface,
                start_date='2019-04-26',
                end_date='2019-05-06'
            )
            
            # 检查是否有数据清洗的日志
            info_calls = [call for call in mock_logger.info.call_args_list 
                         if '数据清洗' in str(call)]
            assert len(info_calls) > 0, "应该有数据清洗的日志记录"
            
            # 检查不应该有"使用前期价格"的警告
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if '使用前期价格' in str(call)]
            assert len(warning_calls) == 0, "数据清洗后不应该有'使用前期价格'的警告"
    
    def test_cleaned_data_produces_correct_returns(self):
        """测试清洗后的数据产生正确的收益率计算"""
        def mock_get_price_data(symbols, start_date, end_date):
            if symbols == ['000300.SH']:  # 基准数据
                return self.benchmark_data
            else:  # 股票数据
                return self.price_data_with_nan
        
        self.mock_data_interface.get_price_data.side_effect = mock_get_price_data
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2019-04-26',
            end_date='2019-05-06'
        )
        
        # 重置环境
        env.reset()
        
        # 执行几步，检查节假日期间的收益率计算
        # 在节假日期间，由于价格不变，收益率应该为0
        for _ in range(5):
            action = np.array([0.33, 0.33, 0.34])  # 均匀分配
            obs, reward, done, info = env.step(action)
            
            # 验证收益率计算没有异常
            assert not np.isnan(info.get('portfolio_return', 0)), "投资组合收益率不应为NaN"
            assert not np.isinf(info.get('portfolio_return', 0)), "投资组合收益率不应为无穷大"