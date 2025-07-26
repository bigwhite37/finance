#!/usr/bin/env python
"""
测试因子数据质量
"""

import sys
sys.path.append('.')
from data import DataManager
from factors import FactorEngine
from config import get_default_config
import numpy as np

def test_factor_data():
    # Test the factor data
    config = get_default_config()
    data_config = config['data']
    factor_config = config['factors']

    dm = DataManager(data_config)
    fe = FactorEngine(factor_config)

    # Get a small sample of data with longer time period
    instruments = dm._get_universe_stocks('2019-01-01', '2023-12-31')[:10]
    stock_data = dm.get_stock_data(
        instruments=instruments,
        start_time='2022-01-01',
        end_time='2022-03-31'  # 3 months of data
    )

    if not stock_data.empty:
        price_data = stock_data['$close'].unstack()
        volume_data = stock_data['$volume'].unstack()
        
        print('Price data shape:', price_data.shape)
        print('Price data NaN count:', price_data.isna().sum().sum())
        print('Price data Inf count:', np.isinf(price_data).sum().sum())
        
        # Test factor calculation
        factor_data = fe.calculate_all_factors(price_data, volume_data)
        print('Factor data shape:', factor_data.shape)
        print('Factor data NaN count:', factor_data.isna().sum().sum())
        print('Factor data Inf count:', np.isinf(factor_data).sum().sum())
        print('Factor data sample:')
        print(factor_data.head())
        
        # Check for specific problematic values
        print('\nFactor data statistics:')
        print(factor_data.describe())
        
        return factor_data
    else:
        print('No stock data available')
        return None

if __name__ == "__main__":
    test_factor_data()