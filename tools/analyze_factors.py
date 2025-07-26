import sys
import os
import pandas as pd
import numpy as np

# Add project path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager
from data import DataManager
from factors import FactorEngine

def analyze_factors():
    """
    Analyzes the generated factor data for numerical stability issues.
    """
    print("--- 因子数据稳定性分析 ---")
    
    try:
        # 1. Load data and generate factors
        print("\n[1/3] 正在加载数据并计算因子...")
        config_manager = ConfigManager('config_train.yaml')
        data_config = config_manager.get_data_config()
        data_manager = DataManager(data_config)
        factor_engine = FactorEngine(config_manager.get_config('factors'))

        stock_data = data_manager.get_stock_data(
            start_time=data_config['start_date'],
            end_time=data_config['end_date']
        )
        
        price_data = stock_data['$close'].unstack().T
        volume_data = stock_data['$volume'].unstack().T
        
        low_vol_stocks = factor_engine.filter_low_volatility_universe(price_data)
        price_data = price_data[low_vol_stocks]
        volume_data = volume_data[low_vol_stocks]

        factor_data = factor_engine.calculate_all_factors(price_data, volume_data)
        print("    -> 因子数据生成完毕。")
        print(f"    -> 因子数据形状: {factor_data.shape}")

        # 2. Perform statistical analysis
        print("\n[2/3] 正在对每个因子进行统计分析...")
        
        # Use .describe() for a quick overview
        stats_summary = factor_data.describe(percentiles=[.01, .05, .25, .75, .95, .99])
        
        print("\n--- 统计摘要 ---")
        print(stats_summary)

        # 3. Check for NaN and Inf values
        print("\n[3/3] 正在检查 NaN 和 Inf 值...")
        nan_report = factor_data.isnull().sum()
        inf_report = (factor_data == np.inf).sum()
        neg_inf_report = (factor_data == -np.inf).sum()

        print("\n--- 不稳定值报告 ---")
        report_df = pd.DataFrame({
            'NaN_Count': nan_report,
            'Infinity_Count': inf_report,
            'Neg_Infinity_Count': neg_inf_report
        })
        print(report_df[report_df.sum(axis=1) > 0])
        
        print("\n--- 分析结论 ---")
        if report_df.sum().sum() > 0:
            print("[警告] 发现不稳定的值 (NaN/Inf)。这是导致训练失败的直接原因。")
        else:
            print("[信息] 未发现 NaN 或 Inf 值。")
            
        # Check for extreme values based on standard deviation
        extreme_value_threshold = 10  # 10个标准差以外的值被认为是极端值
        extreme_values = (np.abs(factor_data - factor_data.mean()) > extreme_value_threshold * factor_data.std()).sum()
        
        if extreme_values.sum() > 0:
            print(f"[警告] 发现极端值 (超过 {extreme_value_threshold} 倍标准差)。这可能导致梯度爆炸和训练不稳定。")
            print("极端值报告 (计数):")
            print(extreme_values[extreme_values > 0])
        else:
            print("[信息] 未发现明显的极端值。")

    except Exception as e:
        print(f"\n[错误] 分析过程中出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_factors()
