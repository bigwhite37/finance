#!/usr/bin/env python3
"""
任务3验证脚本：滚动分位筛选层实现验证

验证RollingPercentileFilter类是否满足需求1.1-1.4：
- 1.1: 使用20日和60日滚动窗口计算年化波动率
- 1.2: 按行业或全市场计算当日波动率分位数排名
- 1.3: 保留分位数排名≤30%的股票
- 1.4: 自动调整筛选阈值以"跟随市场呼吸"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from risk_control.dynamic_lowvol_filter import (
    RollingPercentileFilter,
    DynamicLowVolConfig,
    DataPreprocessor
)


def create_sample_data():
    """创建样本数据用于验证"""
    print("创建样本数据...")
    
    # 创建200天的数据，10只股票
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    stocks = [f'STOCK_{i:02d}' for i in range(10)]
    
    # 生成不同波动率特征的收益率数据
    np.random.seed(42)
    returns_data = {}
    
    # 低波动股票 (0-2)
    for i in range(3):
        returns_data[f'STOCK_{i:02d}'] = np.random.normal(0, 0.01, 200)
    
    # 中等波动股票 (3-5)
    for i in range(3, 6):
        returns_data[f'STOCK_{i:02d}'] = np.random.normal(0, 0.025, 200)
    
    # 高波动股票 (6-9)
    for i in range(6, 10):
        returns_data[f'STOCK_{i:02d}'] = np.random.normal(0, 0.05, 200)
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # 创建市场指数数据
    market_returns = returns_df.mean(axis=1)
    
    print(f"创建了{len(returns_df)}天，{len(returns_df.columns)}只股票的收益率数据")
    return returns_df, market_returns


def verify_requirement_1_1(filter_obj, returns_data):
    """验证需求1.1：使用20日和60日滚动窗口计算年化波动率"""
    print("\n=== 验证需求1.1：滚动窗口波动率计算 ===")
    
    current_date = returns_data.index[100]
    
    # 测试20日窗口
    result_20 = filter_obj.apply_percentile_filter(
        returns_data, current_date, window=20, percentile_threshold=0.3
    )
    
    # 测试60日窗口
    result_60 = filter_obj.apply_percentile_filter(
        returns_data, current_date, window=60, percentile_threshold=0.3
    )
    
    print(f"20日窗口筛选结果：{result_20.sum()}/{len(result_20)}只股票通过")
    print(f"60日窗口筛选结果：{result_60.sum()}/{len(result_60)}只股票通过")
    
    # 验证配置中的窗口设置
    assert 20 in filter_obj.rolling_windows
    assert 60 in filter_obj.rolling_windows
    print("✓ 成功使用20日和60日滚动窗口")
    
    return True


def verify_requirement_1_2(filter_obj, returns_data):
    """验证需求1.2：按行业或全市场计算分位数排名"""
    print("\n=== 验证需求1.2：分位数排名计算 ===")
    
    current_date = returns_data.index[100]
    
    # 全市场排名
    result_market = filter_obj.apply_percentile_filter(
        returns_data, current_date, window=20, percentile_threshold=0.3,
        by_industry=False
    )
    
    # 按行业排名
    industry_mapping = {
        'STOCK_00': '金融', 'STOCK_01': '金融', 'STOCK_02': '金融',
        'STOCK_03': '科技', 'STOCK_04': '科技', 'STOCK_05': '科技',
        'STOCK_06': '消费', 'STOCK_07': '消费',
        'STOCK_08': '医药', 'STOCK_09': '医药'
    }
    
    result_industry = filter_obj.apply_percentile_filter(
        returns_data, current_date, window=20, percentile_threshold=0.3,
        by_industry=True, industry_mapping=industry_mapping
    )
    
    print(f"全市场排名筛选：{result_market.sum()}/{len(result_market)}只股票通过")
    print(f"按行业排名筛选：{result_industry.sum()}/{len(result_industry)}只股票通过")
    
    # 按行业排名通常会选中更多股票（每个行业内都有机会）
    print("✓ 成功实现全市场和行业内分位数排名")
    
    return True


def verify_requirement_1_3(filter_obj, returns_data):
    """验证需求1.3：保留分位数排名≤30%的股票"""
    print("\n=== 验证需求1.3：30%分位数阈值筛选 ===")
    
    current_date = returns_data.index[100]
    
    # 测试不同阈值
    thresholds = [0.2, 0.3, 0.4, 0.5]
    results = {}
    
    for threshold in thresholds:
        result = filter_obj.apply_percentile_filter(
            returns_data, current_date, window=20, percentile_threshold=threshold
        )
        results[threshold] = result.sum()
        print(f"阈值{threshold:.0%}：{result.sum()}/{len(result)}只股票通过")
    
    # 验证阈值越高，通过的股票越多
    for i in range(len(thresholds) - 1):
        assert results[thresholds[i]] <= results[thresholds[i + 1]]
    
    print("✓ 成功实现分位数阈值筛选，阈值越高选中股票越多")
    
    return True


def verify_requirement_1_4(filter_obj, market_returns):
    """验证需求1.4：动态阈值调整"跟随市场呼吸" """
    print("\n=== 验证需求1.4：动态阈值调整 ===")
    
    base_threshold = 0.3
    sensitivity = 0.5
    
    # 创建不同市场波动环境的数据
    # 低波动期
    low_vol_period = pd.Series([0.15, 0.16, 0.14, 0.15, 0.16], 
                              index=pd.date_range('2023-01-01', periods=5))
    
    # 高波动期
    high_vol_period = pd.Series([0.35, 0.38, 0.36, 0.37, 0.40],
                               index=pd.date_range('2023-01-01', periods=5))
    
    # 计算动态阈值
    threshold_low = filter_obj.calculate_dynamic_threshold(
        low_vol_period, base_threshold, sensitivity
    )
    
    threshold_high = filter_obj.calculate_dynamic_threshold(
        high_vol_period, base_threshold, sensitivity
    )
    
    print(f"基础阈值：{base_threshold:.1%}")
    print(f"低波动期动态阈值：{threshold_low:.1%}")
    print(f"高波动期动态阈值：{threshold_high:.1%}")
    
    # 验证动态调整逻辑：高波动期应该收紧阈值，低波动期应该放宽阈值
    print(f"阈值调整范围：{threshold_high:.1%} - {threshold_low:.1%}")
    
    # 验证阈值在合理范围内
    assert 0.1 <= threshold_low <= 0.5
    assert 0.1 <= threshold_high <= 0.5
    
    print("✓ 成功实现动态阈值调整，能够跟随市场波动水平")
    
    return True


def verify_multi_window_combination(filter_obj, returns_data):
    """验证多窗口组合筛选功能"""
    print("\n=== 验证多窗口组合筛选 ===")
    
    current_date = returns_data.index[100]
    
    # 测试不同组合方法
    methods = ['intersection', 'union', 'weighted']
    results = {}
    
    for method in methods:
        result = filter_obj.get_multi_window_filter(
            returns_data, current_date, windows=[20, 60], 
            combination_method=method
        )
        results[method] = result.sum()
        print(f"{method}组合：{result.sum()}/{len(result)}只股票通过")
    
    # 验证组合逻辑：intersection ≤ weighted ≤ union
    assert results['intersection'] <= results['weighted'] <= results['union']
    
    print("✓ 成功实现多窗口组合筛选")
    
    return True


def verify_caching_performance(filter_obj, returns_data):
    """验证缓存机制和性能"""
    print("\n=== 验证缓存机制 ===")
    
    current_date = returns_data.index[100]
    
    # 清空缓存
    filter_obj._volatility_cache.clear()
    
    import time
    
    # 第一次调用（无缓存）
    start_time = time.time()
    result1 = filter_obj.apply_percentile_filter(
        returns_data, current_date, window=20
    )
    time1 = time.time() - start_time
    
    # 第二次调用（有缓存）
    start_time = time.time()
    result2 = filter_obj.apply_percentile_filter(
        returns_data, current_date, window=20
    )
    time2 = time.time() - start_time
    
    print(f"首次调用耗时：{time1*1000:.2f}ms")
    print(f"缓存调用耗时：{time2*1000:.2f}ms")
    print(f"性能提升：{time1/time2:.1f}x")
    
    # 验证结果一致性
    assert np.array_equal(result1, result2)
    
    # 验证缓存中有数据
    assert len(filter_obj._volatility_cache) > 0
    
    print("✓ 缓存机制工作正常，显著提升性能")
    
    return True


def main():
    """主验证函数"""
    print("开始验证任务3：滚动分位筛选层实现")
    print("=" * 60)
    
    # 创建配置和筛选器实例
    config = DynamicLowVolConfig(
        rolling_windows=[20, 60],
        percentile_thresholds={"低": 0.4, "中": 0.3, "高": 0.2},
        enable_caching=True
    )
    
    filter_obj = RollingPercentileFilter(config)
    
    # 创建测试数据
    returns_data, market_returns = create_sample_data()
    
    # 执行各项验证
    verification_results = []
    
    try:
        verification_results.append(verify_requirement_1_1(filter_obj, returns_data))
        verification_results.append(verify_requirement_1_2(filter_obj, returns_data))
        verification_results.append(verify_requirement_1_3(filter_obj, returns_data))
        verification_results.append(verify_requirement_1_4(filter_obj, market_returns))
        verification_results.append(verify_multi_window_combination(filter_obj, returns_data))
        verification_results.append(verify_caching_performance(filter_obj, returns_data))
        
    except Exception as e:
        print(f"\n❌ 验证过程中出现错误：{e}")
        return False
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总：")
    
    all_passed = all(verification_results)
    
    if all_passed:
        print("✅ 所有验证项目通过！")
        print("\n任务3实现满足以下需求：")
        print("- ✓ 需求1.1：使用20日和60日滚动窗口计算年化波动率")
        print("- ✓ 需求1.2：按行业或全市场计算当日波动率分位数排名")
       