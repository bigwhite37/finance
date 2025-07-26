#!/usr/bin/env python3
"""
调试因子计算问题
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factors.alpha_factors import AlphaFactors
from factors.risk_factors import RiskFactors

def debug_factor_classification():
    """调试因子分类问题"""
    
    # 创建测试数据
    dates = pd.date_range('2022-01-01', periods=60, freq='D')
    stocks = [f'stock_{i}' for i in range(10)]
    
    np.random.seed(42)
    price_data = pd.DataFrame(
        np.random.randn(60, 10).cumsum(axis=0) + 100,
        index=dates,
        columns=stocks
    )
    volume_data = pd.DataFrame(
        np.random.randint(1000, 10000, (60, 10)),
        index=dates,
        columns=stocks
    )
    
    # 创建因子计算器
    alpha_calc = AlphaFactors({})
    risk_calc = RiskFactors({})
    
    # 增强因子列表
    enhanced_factors = [
        "momentum_20d", "momentum_60d", "price_reversal",
        "ma_ratio_20d", "ma_ratio_60d", "bollinger_position", 
        "williams_r", "rsi_14d",
        "volume_ratio", "turnover_rate", "volume_price_trend",
        "volatility_20d", "volatility_60d",
        "price_volume_correlation", "mean_reversion_5d", 
        "trend_strength", "volume_momentum"
    ]
    
    print("因子分类调试:")
    print("="*50)
    
    alpha_factors = []
    risk_factors = []
    
    for factor in enhanced_factors:
        has_alpha = hasattr(alpha_calc, f"calculate_{factor}")
        has_risk = hasattr(risk_calc, f"calculate_{factor}")
        
        print(f"{factor:25} Alpha: {has_alpha:5} Risk: {has_risk:5}")
        
        if has_alpha:
            alpha_factors.append(factor)
        if has_risk:
            risk_factors.append(factor)
    
    print(f"\nAlpha因子 ({len(alpha_factors)}个): {alpha_factors}")
    print(f"Risk因子 ({len(risk_factors)}个): {risk_factors}")
    
    # 测试分别计算
    print("\n测试分别计算:")
    print("="*50)
    
    try:
        print("计算Alpha因子...")
        if alpha_factors:
            alpha_data = alpha_calc.calculate_factors(price_data, volume_data, alpha_factors)
            print(f"Alpha因子数据形状: {alpha_data.shape}")
        else:
            alpha_data = pd.DataFrame()
            print("无Alpha因子")
    except Exception as e:
        print(f"Alpha因子计算失败: {e}")
        alpha_data = pd.DataFrame()
    
    try:
        print("计算Risk因子...")
        if risk_factors:
            risk_data = risk_calc.calculate_factors(price_data, volume_data, risk_factors)
            print(f"Risk因子数据形状: {risk_data.shape}")
        else:
            risk_data = pd.DataFrame()
            print("无Risk因子")
    except Exception as e:
        print(f"Risk因子计算失败: {e}")
        risk_data = pd.DataFrame()
    
    # 测试合并
    print("\n测试合并:")
    print("="*50)
    
    try:
        if not alpha_data.empty and not risk_data.empty:
            combined = pd.concat([alpha_data, risk_data], axis=1)
            print(f"合并成功，形状: {combined.shape}")
        elif not alpha_data.empty:
            combined = alpha_data
            print(f"只有Alpha因子，形状: {combined.shape}")
        elif not risk_data.empty:
            combined = risk_data
            print(f"只有Risk因子，形状: {combined.shape}")
        else:
            print("没有可用因子数据")
            combined = pd.DataFrame()
    except Exception as e:
        print(f"合并失败: {e}")

if __name__ == "__main__":
    debug_factor_classification()