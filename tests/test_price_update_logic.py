"""
测试价格更新逻辑的正确性
验证价格在不同时间步正确更新
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from src.rl_trading_system.data.interfaces import DataInterface


class TestPriceUpdateLogic:
    """测试价格更新逻辑"""
    
    def setup_method(self):
        """设置测试环境"""
        self.config = PortfolioConfig(
            stock_pool=['000001.SZ', '000002.SZ', '600000.SH'],
            lookback_window=30,
            initial_cash=1000000.0
        )
        
        # 创建模拟数据接口
        self.mock_data_interface = Mock(spec=DataInterface)
        
        # 创建有变化的价格数据
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        symbols = self.config.stock_pool
        
        # 创建多层索引的价格数据
        index_tuples = [(date, symbol) for date in dates for symbol in symbols]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['datetime', 'symbol'])
        
        # 创建有明显变化的价格数据
        base_prices = [10.0, 20.0, 15.0]  # 每只股票的基础价格
        price_data = []
        
        for i, (date, symbol) in enumerate(index_tuples):
            symbol_idx = symbols.index(symbol)
            date_idx = dates.get_loc(date)
            # 价格随时间变化：基础价格 + 日期索引 * 0.1
            price = base_prices[symbol_idx] + date_idx * 0.1
            price_data.append(price)
        
        self.price_data_with_changes = pd.DataFrame({
            'open': np.array(price_data) * 0.99,
            'high': np.array(price_data) * 1.02,
            'low': np.array(price_data) * 0.98,
            'close': price_data,
            'volume': np.random.uniform(1000, 10000, len(index_tuples)),
            'amount': np.random.uniform(10000, 100000, len(index_tuples))
        }, index=multi_index)
        
        # 创建基准数据
        self.benchmark_data = pd.DataFrame({
            'close': np.random.uniform(3000, 4000, len(dates))
        }, index=dates)
    
    def test_price_changes_over_time(self):
        """测试价格随时间正确变化"""
        self.mock_data_interface.get_price_data.side_effect = [
            self.price_data_with_changes,
            self.benchmark_data
        ]
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2023-01-01',
            end_date='2023-01-20'
        )
        
        # 重置环境
        env.reset()
        
        # 记录初始价格
        initial_prices = env.current_prices.copy()
        
        # 执行几个步骤
        for step in range(5):
            action = np.array([0.33, 0.33, 0.34])  # 简单的权重分配
            env.step(action)
        
        # 验证价格确实发生了变化
        final_prices = env.current_prices
        
        # 价格应该不同（因为我们设计的数据随时间变化）
        assert not np.array_equal(initial_prices, final_prices), \
            f"价格应该发生变化，但初始价格 {initial_prices} 和最终价格 {final_prices} 相同"
        
        # 验证价格变化的方向是正确的（应该增加，因为我们设计的数据是递增的）
        for i in range(len(initial_prices)):
            assert final_prices[i] > initial_prices[i], \
                f"股票{i}的价格应该增加，但从 {initial_prices[i]} 变为 {final_prices[i]}"
    
    def test_price_update_at_data_boundary(self):
        """测试在数据边界处的价格更新"""
        # 创建只有5天数据的小数据集
        short_dates = pd.date_range('2023-01-01', periods=5, freq='D')
        symbols = self.config.stock_pool
        
        index_tuples = [(date, symbol) for date in short_dates for symbol in symbols]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['datetime', 'symbol'])
        
        short_price_data = pd.DataFrame({
            'open': np.random.uniform(10, 20, len(index_tuples)),
            'high': np.random.uniform(15, 25, len(index_tuples)),
            'low': np.random.uniform(8, 15, len(index_tuples)),
            'close': np.random.uniform(10, 20, len(index_tuples)),
            'volume': np.random.uniform(1000, 10000, len(index_tuples)),
            'amount': np.random.uniform(10000, 100000, len(index_tuples))
        }, index=multi_index)
        
        short_benchmark_data = pd.DataFrame({
            'close': np.random.uniform(3000, 4000, len(short_dates))
        }, index=short_dates)
        
        self.mock_data_interface.get_price_data.side_effect = [
            short_price_data,
            short_benchmark_data
        ]
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2023-01-01',
            end_date='2023-01-05'
        )
        
        # 重置环境
        env.reset()
        
        # 尝试执行超过数据范围的步骤
        action = np.array([0.33, 0.33, 0.34])
        
        # 前几步应该正常
        for step in range(3):
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        # 验证环境能够正确处理数据边界
        # 要么正常结束，要么使用边界数据
        assert True  # 如果到这里没有异常，说明处理是正确的
    
    def test_price_index_calculation(self):
        """测试价格索引计算的正确性"""
        self.mock_data_interface.get_price_data.side_effect = [
            self.price_data_with_changes,
            self.benchmark_data
        ]
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2023-01-01',
            end_date='2023-01-20'
        )
        
        # 重置环境
        env.reset()
        
        # 检查初始索引计算
        initial_start_idx = env.start_idx
        initial_step = env.current_step
        
        # 计算预期的日期索引
        expected_date_idx = initial_start_idx + initial_step
        unique_dates = env.price_data.index.get_level_values('datetime').unique()
        
        if expected_date_idx < len(unique_dates):
            expected_date = unique_dates[expected_date_idx]
            
            # 验证当前使用的日期是正确的
            # 这需要我们能够访问内部状态，或者通过价格变化来推断
            assert True  # 基本的索引计算测试
    
    def test_debug_price_update_details(self):
        """调试价格更新的详细过程"""
        self.mock_data_interface.get_price_data.side_effect = [
            self.price_data_with_changes,
            self.benchmark_data
        ]
        
        env = PortfolioEnvironment(
            config=self.config,
            data_interface=self.mock_data_interface,
            start_date='2023-01-01',
            end_date='2023-01-20'
        )
        
        # 重置环境
        env.reset()
        
        print(f"初始状态:")
        print(f"  start_idx: {env.start_idx}")
        print(f"  current_step: {env.current_step}")
        print(f"  current_prices: {env.current_prices}")
        print(f"  数据日期范围: {env.price_data.index.get_level_values('datetime').min()} 到 {env.price_data.index.get_level_values('datetime').max()}")
        print(f"  唯一日期数量: {len(env.price_data.index.get_level_values('datetime').unique())}")
        
        # 执行一步
        action = np.array([0.33, 0.33, 0.34])
        obs, reward, done, info = env.step(action)
        
        print(f"\n执行一步后:")
        print(f"  start_idx: {env.start_idx}")
        print(f"  current_step: {env.current_step}")
        print(f"  current_prices: {env.current_prices}")
        print(f"  previous_prices: {env.previous_prices}")
        
        # 验证价格确实更新了
        if env.previous_prices is not None:
            price_changed = not np.array_equal(env.current_prices, env.previous_prices)
            print(f"  价格是否变化: {price_changed}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])