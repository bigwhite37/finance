"""
简单测试投资组合环境的基本功能
验证修复后的代码能正常工作
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from src.rl_trading_system.data.interfaces import DataInterface


def test_portfolio_environment_basic_functionality():
    """测试投资组合环境的基本功能"""
    config = PortfolioConfig(
        stock_pool=['000001.SZ', '000002.SZ', '600000.SH'],
        lookback_window=30,
        initial_cash=1000000.0
    )
    
    # 创建模拟数据接口
    mock_data_interface = Mock(spec=DataInterface)
    
    # 创建有效的价格数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    symbols = config.stock_pool
    
    # 创建多层索引的价格数据
    index_tuples = [(date, symbol) for date in dates for symbol in symbols]
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['datetime', 'symbol'])
    
    valid_price_data = pd.DataFrame({
        'open': np.random.uniform(10, 20, len(index_tuples)),
        'high': np.random.uniform(15, 25, len(index_tuples)),
        'low': np.random.uniform(8, 15, len(index_tuples)),
        'close': np.random.uniform(10, 20, len(index_tuples)),
        'volume': np.random.uniform(1000, 10000, len(index_tuples)),
        'amount': np.random.uniform(10000, 100000, len(index_tuples))
    }, index=multi_index)
    
    # 创建基准数据
    benchmark_data = pd.DataFrame({
        'close': np.random.uniform(3000, 4000, len(dates))
    }, index=dates)
    
    mock_data_interface.get_price_data.side_effect = [
        valid_price_data,  # 股票数据
        benchmark_data     # 基准数据
    ]
    
    # 创建环境
    env = PortfolioEnvironment(
        config=config,
        data_interface=mock_data_interface,
        start_date='2023-01-01',
        end_date='2023-04-10'
    )
    
    # 测试重置
    observation = env.reset()
    assert 'features' in observation
    assert 'positions' in observation
    assert 'market_state' in observation
    
    # 测试步进
    action = np.array([0.4, 0.3, 0.3])  # 权重分配
    next_obs, reward, done, info = env.step(action)
    
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert 'portfolio_return' in info
    assert 'total_value' in info


def test_portfolio_environment_error_handling():
    """测试投资组合环境的错误处理"""
    config = PortfolioConfig(
        stock_pool=['000001.SZ'],
        lookback_window=30,
        initial_cash=1000000.0
    )
    
    mock_data_interface = Mock(spec=DataInterface)
    
    # 测试空数据的情况
    mock_data_interface.get_price_data.return_value = pd.DataFrame()
    
    with pytest.raises(ValueError, match="无法获取股票数据"):
        env = PortfolioEnvironment(
            config=config,
            data_interface=mock_data_interface,
            start_date='2023-01-01',
            end_date='2023-04-10'
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])