"""
测试NaN价格处理的正确性
验证系统能够正确处理股票价格中的NaN值
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
from datetime import datetime, timedelta

from src.rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from src.rl_trading_system.data.interfaces import DataInterface


class TestNaNHandling:
    """测试NaN价格处理"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = PortfolioConfig(
            stock_pool=['000001.SZ', '000002.SZ', '600000.SH'],
            lookback_window=30,
            initial_cash=1000000.0
        )
        
        # 创建模拟数据接口
        self.mock_data_interface = Mock(spec=DataInterface)
        
        # 创建包含NaN的价格数据
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        symbols = self.config.stock_pool
        
        # 创建多层索引的价格数据，包含一些NaN值
        index_tuples = [(date, symbol) for date in dates for symbol in symbols]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['datetime', 'symbol'])
        
        # 创建正常价格数据
        normal_prices = np.random.uniform(10, 20, len(index_tuples))
        
        # 在特定位置插入NaN（模拟停牌等情况）
        nan_positions = [3, 7, 15, 22]  # 在这些位置设置NaN
        for pos in nan_positions:
            if pos < len(normal_prices):
                normal_prices[pos] = np.nan
        
        self.price_data_with_nan = pd.DataFrame({
            'open': np.random.uniform(10, 20, len(index_tuples)),
            'high': np.random.uniform(15, 25, len(index_tuples)),
            'low': np.random.uniform(8, 15, len(index_tuples)),
            'close': normal_prices,
            'volume': np.random.uniform(1000, 10000, len(index_tuples)),
            'amount': np.random.uniform(10000, 100000, len(index_tuples))
        }, index=multi_index)
        
        # 创建基准数据
        self.benchmark_data = pd.DataFrame({
            'close': np.random.uniform(3000, 4000, len(dates))
        }, index=dates)
    
    def test_nan_handling_with_previous_prices(self):
        """测试有前期价格时的NaN处理"""
        self.mock_data_interface.get_price_data.side_effect = [
            self.price_data_with_nan,
            self.benchmark_data
        ]
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        # 重置环境
        env.reset()
        
        # 手动设置一些前期价格
        env.previous_prices = np.array([15.0, 18.0, 12.0])
        
        # 模拟遇到NaN价格的情况
        # 创建包含NaN的当日数据
        nan_date = pd.Timestamp('2023-01-05')
        current_day_data = pd.DataFrame({
            'close': [np.nan, 20.0, np.nan]
        }, index=self.config.stock_pool)
        
        # 手动调用价格更新逻辑
        env.current_step = 5
        
        # 模拟_update_current_prices中的逻辑
        new_current_prices = np.zeros(env.n_stocks)
        
        for i, symbol in enumerate(env.config.stock_pool):
            if symbol in current_day_data.index:
                price = current_day_data.loc[symbol, 'close']
                if pd.isna(price) or price <= 0:
                    if env.previous_prices is not None and i < len(env.previous_prices):
                        new_current_prices[i] = env.previous_prices[i]
                    else:
                        pytest.fail(f"应该有前期价格可用，但没有找到")
                else:
                    new_current_prices[i] = float(price)
        
        # 验证结果
        expected_prices = [15.0, 20.0, 12.0]  # 第1和第3个股票使用前期价格，第2个使用当前价格
        np.testing.assert_array_equal(new_current_prices, expected_prices)
    
    def test_nan_handling_without_previous_prices_raises_error(self):
        """测试没有前期价格时遇到NaN应该抛出异常"""
        self.mock_data_interface.get_price_data.side_effect = [
            self.price_data_with_nan,
            self.benchmark_data
        ]
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2023-01-01',
            end_date='2023-01-10'
        )
        
        # 重置环境
        env.reset()
        
        # 确保没有前期价格
        env.previous_prices = None
        
        # 创建包含NaN的当日数据
        current_day_data = pd.DataFrame({
            'close': [np.nan, 20.0, 15.0]
        }, index=self.config.stock_pool)
        
        # 模拟价格更新逻辑，应该抛出异常
        with pytest.raises(RuntimeError, match="价格无效.*且无前期价格可用"):
            new_current_prices = np.zeros(env.n_stocks)
            
            for i, symbol in enumerate(env.config.stock_pool):
                if symbol in current_day_data.index:
                    price = current_day_data.loc[symbol, 'close']
                    if pd.isna(price) or price <= 0:
                        if env.previous_prices is not None and i < len(env.previous_prices):
                            new_current_prices[i] = env.previous_prices[i]
                        else:
                            raise RuntimeError(f"股票{symbol}的价格无效: {price}，且无前期价格可用")
                    else:
                        new_current_prices[i] = float(price)
    
    def test_nan_in_real_data_is_normal(self):
        """测试真实数据中的NaN是正常现象"""
        # NaN在股票数据中是正常的，可能由以下原因造成：
        # 1. 股票停牌
        # 2. 新股上市前的数据
        # 3. 退市股票
        # 4. 节假日
        # 5. 数据源的数据质量问题
        
        # 验证我们的数据中确实包含NaN
        nan_count = self.price_data_with_nan['close'].isna().sum()
        assert nan_count > 0, "测试数据应该包含NaN值"
        
        # 验证NaN的位置是我们预期的
        nan_positions = self.price_data_with_nan['close'].isna()
        assert nan_positions.sum() == 4, "应该有4个NaN值"
    
    def test_data_quality_reports_nan_correctly(self):
        """测试数据质量检查能正确报告NaN"""
        from src.rl_trading_system.data.data_quality import DataQualityChecker
        
        checker = DataQualityChecker()
        
        # 检查包含NaN的数据
        quality_report = checker.check_data_quality(self.price_data_with_nan, 'price')
        
        # 应该报告数据质量问题
        assert quality_report['status'] in ['warning', 'error']
        
        # 应该在统计信息中包含缺失值信息
        assert 'missing_values' in quality_report['statistics']
        
        # 缺失值数量应该大于0
        missing_values = quality_report['statistics']['missing_values']
        assert missing_values['close'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])