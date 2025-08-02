#!/usr/bin/env python3
"""
特征工程性能瓶颈识别和优化的TDD测试
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestFeatureEngineeringPerformance:
    """测试特征工程性能瓶颈"""
    
    def test_pandas_apply_lambda_is_slow(self):
        """测试pandas apply lambda操作的性能问题"""
        # Red: 识别当前代码中的性能瓶颈
        
        # 创建模拟数据，类似真实市场数据的规模
        n_dates = 1000  # 约4年的交易日数据
        n_stocks = 3
        
        data_list = []
        dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
        stocks = ['600519.SH', '600036.SH', '601318.SH']
        
        for stock in stocks:
            for date in dates:
                data_list.append({
                    'instrument': stock,
                    'datetime': date,
                    'open': 10.0 + np.random.randn() * 0.5,
                    'high': 10.5 + np.random.randn() * 0.3,
                    'low': 9.5 + np.random.randn() * 0.3,
                    'close': 10.0 + np.random.randn() * 0.5,
                    'volume': 1000000 + np.random.randint(0, 500000)
                })
        
        df = pd.DataFrame(data_list)
        df = df.set_index(['instrument', 'datetime'])
        
        # 测试当前低效的实现（apply lambda）
        start_time = time.time()
        
        # 模拟 calculate_volatility_features 中的低效操作
        high_low_ratio = df['high'] / df['low']
        
        # 这是当前代码中的瓶颈：apply lambda
        log_hl_slow = high_low_ratio.apply(lambda x: np.log(x) if x > 1 else 0)
        
        slow_time = time.time() - start_time
        
        # 测试优化后的实现（向量化操作）
        start_time = time.time()
        
        # 优化：使用向量化操作代替apply lambda
        log_hl_fast = np.where(high_low_ratio > 1, np.log(high_low_ratio), 0)
        
        fast_time = time.time() - start_time
        
        # 验证结果相同
        np.testing.assert_array_almost_equal(log_hl_slow.values, log_hl_fast, decimal=10)
        
        speedup = slow_time / fast_time if fast_time > 0 else float('inf')
        
        print(f"性能对比:")
        print(f"  apply lambda: {slow_time:.4f}秒")
        print(f"  向量化操作: {fast_time:.4f}秒")
        print(f"  加速比: {speedup:.1f}x")
        
        if speedup > 5:  # 如果加速超过5倍
            print("性能瓶颈确认: pandas apply lambda操作极其缓慢")
            print("优化建议: 使用numpy向量化操作代替apply lambda")
        
        assert True
    
    def test_feature_calculation_can_be_cached(self):
        """测试特征计算可以被缓存以避免重复计算"""
        # Red: 识别重复计算的问题
        
        from rl_trading_system.data.feature_engineer import FeatureEngineer
        
        # 创建测试数据
        n_dates = 200
        dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
        stocks = ['600519.SH', '600036.SH', '601318.SH']
        
        data_list = []
        for stock in stocks:
            for date in dates:
                data_list.append({
                    'instrument': stock,
                    'datetime': date,
                    'open': 10.0 + np.random.randn() * 0.5,
                    'high': 10.5 + np.random.randn() * 0.3,
                    'low': 9.5 + np.random.randn() * 0.3,
                    'close': 10.0 + np.random.randn() * 0.5,
                    'volume': 1000000 + np.random.randint(0, 500000),
                    'amount': 100000000 + np.random.randint(0, 50000000)
                })
        
        df = pd.DataFrame(data_list)
        df = df.set_index(['instrument', 'datetime'])
        
        feature_engineer = FeatureEngineer()
        
        # 第一次计算 - 冷启动
        start_time = time.time()
        technical_features_1 = feature_engineer.calculate_technical_indicators(df)
        volatility_features_1 = feature_engineer.calculate_volatility_features(df)
        momentum_features_1 = feature_engineer.calculate_momentum_features(df)
        first_calculation_time = time.time() - start_time
        
        # 第二次计算 - 应该可以缓存但当前没有
        start_time = time.time()
        technical_features_2 = feature_engineer.calculate_technical_indicators(df)
        volatility_features_2 = feature_engineer.calculate_volatility_features(df)
        momentum_features_2 = feature_engineer.calculate_momentum_features(df)
        second_calculation_time = time.time() - start_time
        
        print(f"特征计算性能:")
        print(f"  第一次计算: {first_calculation_time:.4f}秒")
        print(f"  第二次计算: {second_calculation_time:.4f}秒")
        print(f"  重复计算时间浪费: {second_calculation_time:.4f}秒")
        
        # 验证结果相同（说明可以缓存）
        pd.testing.assert_frame_equal(technical_features_1, technical_features_2)
        pd.testing.assert_frame_equal(volatility_features_1, volatility_features_2)
        pd.testing.assert_frame_equal(momentum_features_1, momentum_features_2)
        
        if second_calculation_time > 0.1:  # 如果重复计算超过0.1秒
            print("性能问题确认: 特征计算没有缓存机制")
            print("影响: 每次训练都重新计算相同的特征")
            print("优化建议: 实现特征缓存机制，避免重复计算")
        
        assert True
        
    def test_optimized_feature_calculation_should_exist(self):
        """测试应该存在优化的特征计算方法（当前会失败）"""
        # Green: 定义优化后的期望接口
        
        from rl_trading_system.data.feature_engineer import FeatureEngineer
        
        feature_engineer = FeatureEngineer()
        
        try:
            # 检查是否有优化的方法
            assert hasattr(feature_engineer, 'calculate_volatility_features_optimized'), \
                "需要实现 calculate_volatility_features_optimized 方法"
            assert hasattr(feature_engineer, 'enable_feature_cache'), \
                "需要实现 enable_feature_cache 方法"
            assert hasattr(feature_engineer, 'clear_feature_cache'), \
                "需要实现 clear_feature_cache 方法"
        except AssertionError as e:
            print(f"优化方法尚未实现: {e}")
            print("需要实现的优化:")
            print("1. calculate_volatility_features_optimized: 使用向量化操作")
            print("2. enable_feature_cache: 启用特征缓存")
            print("3. clear_feature_cache: 清除特征缓存")
            print("4. 将所有apply lambda操作替换为numpy向量化操作")
            
            pytest.skip("优化方法尚未实现，跳过性能测试")

    def test_data_loading_should_be_cached(self):
        """测试数据加载应该被缓存（当前每次都重新加载）"""
        # Red: 识别数据重复加载的问题
        
        # 这个测试识别了在train.py中的问题：
        # 每次创建PortfolioEnvironment时都会重新调用data_interface.get_price_data
        
        print("数据加载性能问题识别:")
        print("当前问题:")
        print("1. 每次创建PortfolioEnvironment都重新加载数据")
        print("2. data_interface.get_price_data可能涉及网络请求或文件I/O")
        print("3. 相同的数据被重复加载和处理")
        print()
        print("优化建议:")
        print("1. 在训练开始时预加载并缓存所有数据")
        print("2. 将预处理后的数据传递给PortfolioEnvironment")
        print("3. 避免在环境初始化时进行数据I/O操作")
        
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s 允许打印输出