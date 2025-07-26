import pickle
import pandas as pd
import sys
import os

# Add project path to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import DataManager
from config import ConfigManager

def debug_universe_mismatch():
    """
    Debugs the stock universe mismatch between training and backtesting.
    """
    print("--- 正在启动股票池不匹配问题调试 ---")

    # 1. 加载训练期间保存的股票池
    trained_stocks_path = './models/selected_stocks.pkl'
    try:
        with open(trained_stocks_path, 'rb') as f:
            trained_stocks = pickle.load(f)
        print(f"\n[1/3] 成功加载训练股票池: {trained_stocks_path}")
        print(f"    -> 包含 {len(trained_stocks)} 只股票。")
        print(f"    -> 股票代码示例: {trained_stocks[:5]}")
    except Exception as e:
        print(f"\n[错误] 加载股票池文件失败: {trained_stocks_path}")
        print(f"    -> 原因: {e}")
        return

    # 2. 加载回测期间的数据，获取可用股票列表
    try:
        print("\n[2/3] 正在加载回测期间的可用股票...")
        config_manager = ConfigManager('config_backtest.yaml')
        data_config = config_manager.get_data_config()
        data_manager = DataManager(data_config)
        
        stock_data = data_manager.get_stock_data(
            start_time=data_config['start_date'],
            end_time=data_config['end_date']
        )
        if stock_data.empty:
            print("\n[错误] 未能加载任何回测期间的股票数据。")
            return
            
        price_data = stock_data['$close'].unstack()
        backtest_stocks = list(price_data.columns)
        print(f"    -> 成功加载回测数据。")
        print(f"    -> 包含 {len(backtest_stocks)} 只股票。")
        print(f"    -> 股票代码示例: {backtest_stocks[:5]}")

    except Exception as e:
        print(f"\n[错误] 加载回测数据失败。")
        print(f"    -> 原因: {e}")
        return

    # 3. 分析和比较
    print("\n[3/3] 正在分析两个股票池的差异...")
    trained_set = set(trained_stocks)
    backtest_set = set(backtest_stocks)
    intersection = trained_set.intersection(backtest_set)
    
    print("\n--- 最终分析结果 ---")
    print(f"训练股票池数量: {len(trained_set)}")
    print(f"回测可用股票数量: {len(backtest_set)}")
    print(f"两个池子的交集数量: {len(intersection)}")

    if not intersection:
        print("\n[结论] 根本原因已确认：两个股票池的交集为空。")
        print("这表明训练选出的股票代码与回测数据中的股票代码格式完全不匹配。")
    else:
        print(f"\n[结论] 存在 {len(intersection)} 只重叠股票，问题可能更复杂。")

if __name__ == "__main__":
    debug_universe_mismatch()
