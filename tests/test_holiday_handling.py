"""
测试A股节假日休市数据处理
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from src.rl_trading_system.data.interfaces import DataInterface
from typing import List


class MockDataInterfaceWithHolidays(DataInterface):
    """模拟包含节假日数据的数据接口"""
    
    def __init__(self):
        super().__init__()
        self.stock_pool = ['600519.SH', '000001.SZ', '000002.SZ']
    
    def get_stock_list(self, market: str = 'A') -> List[str]:
        return self.stock_pool
    
    def get_price_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """生成包含节假日的测试数据"""
        # 创建日期范围，包含节假日
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        for date in date_range:
            for symbol in symbols:
                # 模拟节假日情况：2018-01-01是元旦节假日
                if date.strftime('%Y-%m-%d') == '2018-01-01':
                    # 节假日设置NaN值
                    row = {
                        'datetime': date,
                        'instrument': symbol,
                        'open': np.nan,
                        'high': np.nan,
                        'low': np.nan,
                        'close': np.nan,
                        'volume': np.nan,
                        'amount': np.nan
                    }
                else:
                    # 正常交易日生成有效数据
                    base_price = 100.0 + hash(symbol) % 50
                    row = {
                        'datetime': date,
                        'instrument': symbol,
                        'open': base_price + np.random.normal(0, 1),
                        'high': base_price + np.random.normal(2, 1),
                        'low': base_price + np.random.normal(-2, 1),
                        'close': base_price + np.random.normal(0, 1),
                        'volume': np.random.randint(1000000, 10000000),
                        'amount': np.random.randint(100000000, 1000000000)
                    }
                data.append(row)
        
        df = pd.DataFrame(data)
        df = df.set_index(['datetime', 'instrument'])
        return df
    
    def get_fundamental_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame()


class TestHolidayHandling:
    """测试节假日处理"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = PortfolioConfig(
            stock_pool=['600519.SH', '000001.SZ', '000002.SZ'],
            initial_cash=1000000.0,
            lookback_window=10
        )
        self.data_interface = MockDataInterfaceWithHolidays()
    
    def test_holiday_data_cleaning_should_handle_first_day_nan(self):
        """测试：当第一个交易日包含NaN时，数据清洗应该能正确处理"""
        # 现在我们的实现能够处理第一天为NaN的情况，应该成功创建环境
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.data_interface,
            start_date='2018-01-01',  # 从节假日开始
            end_date='2018-01-05'
        )
        
        # 验证环境创建成功，且数据已正确清洗
        assert env.price_data is not None
        assert not env.price_data.isnull().any().any(), "清洗后的价格数据不应包含NaN值"
        
        # 验证第一个有效交易日是2018-01-02（跳过了节假日2018-01-01）
        first_date = env.price_data.index.get_level_values('datetime').min()
        assert first_date.strftime('%Y-%m-%d') == '2018-01-02', "第一个有效交易日应该是2018-01-02"
    
    def test_holiday_data_cleaning_should_skip_invalid_first_days(self):
        """测试：数据清洗应该跳过无效的第一天，从第一个有效交易日开始"""
        # 这个测试描述了我们期望的行为：能够处理第一天为节假日的情况
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.data_interface,
            start_date='2018-01-01',  # 从节假日开始
            end_date='2018-01-05'
        )
        
        # 验证数据清洗后第一个有效交易日的数据
        assert not env.price_data.isnull().any().any(), "清洗后的价格数据不应包含NaN值"
        
        # 验证第一个有效交易日是2018-01-02
        first_date = env.price_data.index.get_level_values('datetime').min()
        assert first_date.strftime('%Y-%m-%d') == '2018-01-02', "第一个有效交易日应该是2018-01-02"
    
    def test_holiday_data_cleaning_should_forward_fill_middle_holidays(self):
        """测试：数据清洗应该对中间的节假日进行前向填充"""
        
        # 创建包含中间节假日的数据
        class MockDataWithMiddleHoliday(MockDataInterfaceWithHolidays):
            def get_price_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
                df = super().get_price_data(symbols, start_date, end_date)
                # 将2018-01-03设为节假日（中间的日期）
                holiday_mask = df.index.get_level_values('datetime').strftime('%Y-%m-%d') == '2018-01-03'
                df.loc[holiday_mask, :] = np.nan
                return df
        
        data_interface = MockDataWithMiddleHoliday()
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=data_interface,
            start_date='2018-01-02',  # 从有效交易日开始
            end_date='2018-01-05'
        )
        
        # 验证中间节假日使用了前向填充
        holiday_data = env.price_data.xs('2018-01-03', level='datetime')
        pre_holiday_data = env.price_data.xs('2018-01-02', level='datetime')
        
        for symbol in self.config.stock_pool:
            assert holiday_data.loc[symbol, 'close'] == pre_holiday_data.loc[symbol, 'close'], \
                f"股票{symbol}在节假日的价格应该使用前向填充"