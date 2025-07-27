

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager
from data import DataManager
from factors import FactorEngine

def setup_logging():
    """Sets up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def calculate_ic(factor_data: pd.DataFrame, price_data: pd.DataFrame, periods: list = [1, 5, 10]) -> pd.DataFrame:
    """
    Calculates the Information Coefficient (IC) for each factor.

    Args:
        factor_data: DataFrame with factor values. Index is (datetime, instrument), columns are factors.
        price_data: DataFrame with price data. Index is datetime, columns are instruments.
        periods: List of forward returns periods to calculate IC against.

    Returns:
        DataFrame with IC values for each factor and period.
    """
    logger = logging.getLogger(__name__)
    
    results = {}
    for period in periods:
        # Calculate forward returns and stack
        forward_returns = price_data.pct_change(periods=period).shift(-period)
        forward_returns_stacked = forward_returns.stack()
        forward_returns_stacked.name = 'return'

        # Join with factor data, this aligns on the (datetime, instrument) index
        data = factor_data.join(forward_returns_stacked, how='inner')
        
        if data.empty:
            logger.warning(f"No common data for period {period}d, skipping.")
            continue

        # Calculate IC for each factor for each day
        daily_ic = data.groupby(level='datetime').apply(
            lambda x: x.corr(method='spearman')['return']
        ).drop('return', axis=1, errors='ignore')

        # Average the daily ICs for the period
        results[f'ic_{period}d'] = daily_ic.mean()

    return pd.DataFrame(results)


def analyze_factor_performance(config_path: str, data_range: tuple):
    """
    Analyzes the performance of factors for a given data range.

    Args:
        config_path: Path to the config file.
        data_range: Tuple with start and end date.
    """
    logger = logging.getLogger(__name__)
    
    start_date, end_date = data_range
    
    logger.info(f"Analyzing factors for the period: {start_date} to {end_date}")

    # Initialize components
    config_manager = ConfigManager(config_path)
    config_manager.set_value('data.start_date', start_date)
    config_manager.set_value('data.end_date', end_date)
    
    data_manager = DataManager(config_manager.get_data_config())
    factor_engine = FactorEngine(config_manager.get_config('factors'))

    # Prepare data
    logger.info("Preparing data...")
    stock_data = data_manager.get_stock_data(start_time=start_date, end_time=end_date)
    price_data = stock_data['$close'].unstack().T
    volume_data = stock_data['$volume'].unstack().T
    
    logger.info("Calculating factors...")
    factor_data = factor_engine.calculate_all_factors(price_data, volume_data)

    # Calculate IC
    logger.info("Calculating Information Coefficient (IC)...")
    ic_df = calculate_ic(factor_data, price_data)
    
    # Print results
    logger.info("\n=== Factor IC Analysis ===\n")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(ic_df)


if __name__ == "__main__":
    setup_logging()
    
    # Analyze training period
    analyze_factor_performance(
        config_path='config_train.yaml',
        data_range=('2020-01-01', '2022-12-31')
    )

    # Analyze backtest period
    analyze_factor_performance(
        config_path='config_backtest.yaml',
        data_range=('2023-01-01', '2023-12-31')
    )
