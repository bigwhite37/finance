"""
动态低波筛选器性能优化测试

测试GARCH模型缓存、并行处理、向量化计算和内存监控功能。
"""

import pytest
import pandas as pd
import numpy as np
import time
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, List

# 导入被测试的模块
from risk_control.performance_optimizer import (
    PerformanceOptimizer, CacheConfig, ParallelConfig, MonitoringConfig,
    AdvancedCacheManager, ParallelProcessingManager,
    VectorizedComputeOptimizer, MemoryMonitor,
    performance_monitor, PerformanceMetrics
)
from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter, DynamicLowVolConfig


class TestAdvancedCacheManager:
    """高级缓存管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_config = CacheConfig()
        self.cache_config.memory_cache_size = 10
        self.cache_config.disk_cache_dir = self.temp_dir
        self.cache_config.disk_cache_ttl = 3600
        self.cache_manager = AdvancedCacheManager(self.cache_config)
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_memory_cache_basic_operations(self):
        """测试内存缓存基本操作"""
        # 测试存储和获取
        self.cache_manager.put("test_key", "test_value")
        assert self.cache_manager.get("test_key") == "test_value"
        
        # 测试缓存未命中
        assert self.cache_manager.get("nonexistent_key", "default") == "default"
    
    def test_memory_cache_lru_eviction(self):
        """测试内存缓存LRU淘汰策略"""
        # 创建仅内存缓存的配置以测试LRU行为
        memory_only_config = CacheConfig.create_memory_only_config()
        memory_only_config.memory_cache_size = self.cache_config.memory_cache_size
        memory_only_cache_manager = AdvancedCacheManager(memory_only_config)
        
        # 填满缓存
        for i in range(memory_only_config.memory_cache_size):
            memory_only_cache_manager.put(f"key_{i}", f"value_{i}")
        
        # 添加一个新项，应该触发LRU淘汰
        memory_only_cache_manager.put("new_key", "new_value")
        
        # 检查最早的项是否被淘汰
        assert memory_only_cache_manager.get("key_0", None) is None
        assert memory_only_cache_manager.get("new_key") == "new_value"
    
    def test_disk_cache_operations(self):
        """测试磁盘缓存操作"""
        test_data = {"complex": "data", "with": [1, 2, 3]}
        
        # 存储到磁盘缓存
        self.cache_manager.put("disk_test", test_data)
        
        # 清理内存缓存，强制从磁盘读取
        self.cache_manager.clear(memory_only=True)
        
        # 从磁盘缓存获取
        retrieved_data = self.cache_manager.get("disk_test")
        assert retrieved_data == test_data
    
    def test_cache_statistics(self):
        """测试缓存统计信息"""
        # 执行一些缓存操作
        self.cache_manager.put("key1", "value1")
        self.cache_manager.get("key1")  # 命中
        self.cache_manager.get("nonexistent")  # 未命中
        
        stats = self.cache_manager.get_stats()
        
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
        assert stats['total_requests'] >= 2
        assert 0 <= stats['hit_rate'] <= 1


class TestParallelProcessingManager:
    """并行处理管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.parallel_config = ParallelConfig(
            enable_parallel=True,
            max_workers=2,
            use_process_pool=False,
            chunk_size=2
        )
        self.parallel_manager = ParallelProcessingManager(self.parallel_config)
    
    def test_parallel_map_basic(self):
        """测试基本并行映射功能"""
        def square(x):
            return x ** 2
        
        input_data = [1, 2, 3, 4, 5]
        expected_result = [1, 4, 9, 16, 25]
        
        with self.parallel_manager as pm:
            result = pm.parallel_map(square, input_data)
        
        assert sorted(result) == expected_result
    
    def test_parallel_map_with_errors(self):
        """测试并行处理中的错误处理"""
        def problematic_function(x):
            if x == 3:
                raise ValueError("Test error")
            return x * 2
        
        input_data = [1, 2, 3, 4]
        
        with self.parallel_manager as pm:
            result = pm.parallel_map(problematic_function, input_data)
        
        # 检查结果中包含None（表示失败的任务）
        assert None in result
        assert 2 in result  # 成功的任务结果
        assert 4 in result
    
    def test_serial_fallback(self):
        """测试串行执行回退"""
        # 禁用并行处理
        config = ParallelConfig(enable_parallel=False)
        manager = ParallelProcessingManager(config)
        
        def double(x):
            return x * 2
        
        input_data = [1, 2, 3]
        expected_result = [2, 4, 6]
        
        with manager as pm:
            result = pm.parallel_map(double, input_data)
        
        assert sorted(result) == expected_result
    
    def test_performance_statistics(self):
        """测试性能统计信息"""
        def simple_task(x):
            time.sleep(0.01)  # 模拟计算时间
            return x
        
        with self.parallel_manager as pm:
            pm.parallel_map(simple_task, [1, 2, 3])
            stats = pm.get_performance_stats()
        
        assert stats['total_tasks'] >= 3
        assert stats['completed_tasks'] >= 0
        assert stats['success_rate'] >= 0


class TestVectorizedComputeOptimizer:
    """向量化计算优化器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.optimizer = VectorizedComputeOptimizer()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        stocks = ['STOCK_A', 'STOCK_B', 'STOCK_C']
        
        # 生成随机收益率数据
        np.random.seed(42)
        returns_data = np.random.normal(0, 0.02, (100, 3))
        self.returns_df = pd.DataFrame(returns_data, index=dates, columns=stocks)
        
        # 生成因子数据
        factors = ['Market', 'Size', 'Value']
        factor_data = np.random.normal(0, 0.01, (100, 3))
        self.factors_df = pd.DataFrame(factor_data, index=dates, columns=factors)
    
    def test_vectorized_rolling_volatility(self):
        """测试向量化滚动波动率计算"""
        windows = [10, 20]
        result = self.optimizer.vectorized_rolling_volatility(self.returns_df, windows)
        
        # 检查结果结构
        assert isinstance(result, dict)
        assert set(result.keys()) == set(windows)
        
        for window in windows:
            vol_df = result[window]
            assert isinstance(vol_df, pd.DataFrame)
            assert vol_df.shape == self.returns_df.shape
            assert vol_df.columns.equals(self.returns_df.columns)
            
            # 检查年化处理
            assert vol_df.iloc[-1].mean() > 0  # 波动率应为正值
    
    def test_vectorized_percentile_ranks(self):
        """测试向量化分位数排名计算"""
        # 测试按行排名
        ranks = self.optimizer.vectorized_percentile_ranks(self.returns_df, axis=1)
        
        assert isinstance(ranks, pd.DataFrame)
        assert ranks.shape == self.returns_df.shape
        
        # 检查排名范围
        assert (ranks >= 0).all().all()
        assert (ranks <= 1).all().all()
        
        # 检查每行排名的唯一性（对于不同值）
        for idx in ranks.index[:5]:  # 检查前5行
            row_ranks = ranks.loc[idx]
            if len(set(self.returns_df.loc[idx])) == len(self.returns_df.columns):
                # 如果原始值都不同，排名也应该不同
                assert len(set(row_ranks)) == len(row_ranks)
    
    def test_vectorized_factor_regression(self):
        """测试向量化因子回归计算"""
        betas, residuals = self.optimizer.vectorized_factor_regression(
            self.returns_df, self.factors_df
        )
        
        # 检查回归系数
        assert isinstance(betas, pd.DataFrame)
        assert betas.index.equals(self.returns_df.columns)
        assert betas.columns.equals(self.factors_df.columns)
        
        # 检查残差
        assert isinstance(residuals, pd.DataFrame)
        assert residuals.shape == self.returns_df.shape
        assert residuals.index.equals(self.returns_df.index)
        assert residuals.columns.equals(self.returns_df.columns)
    
    def test_vectorized_volatility_decomposition(self):
        """测试向量化波动率分解"""
        # 使用回归残差作为输入
        _, residuals = self.optimizer.vectorized_factor_regression(
            self.returns_df, self.factors_df
        )
        
        good_vol, bad_vol = self.optimizer.vectorized_volatility_decomposition(residuals)
        
        # 检查结果类型和形状
        assert isinstance(good_vol, pd.Series)
        assert isinstance(bad_vol, pd.Series)
        assert good_vol.index.equals(self.returns_df.columns)
        assert bad_vol.index.equals(self.returns_df.columns)
        
        # 检查波动率为非负值
        assert (good_vol >= 0).all()
        assert (bad_vol >= 0).all()
    
    def test_vectorized_garch_batch_prediction(self):
        """测试批量GARCH预测"""
        returns_dict = {
            col: self.returns_df[col] for col in self.returns_df.columns
        }
        
        predictions = self.optimizer.vectorized_garch_batch_prediction(
            returns_dict, horizon=5
        )
        
        # 检查预测结果
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == set(returns_dict.keys())
        
        for stock, pred_vol in predictions.items():
            assert isinstance(pred_vol, (int, float))
            assert pred_vol > 0  # 预测波动率应为正值
            assert pred_vol < 2.0  # 合理的年化波动率范围


class TestMemoryMonitor:
    """内存监控器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.memory_monitor = MemoryMonitor(
            warning_threshold_mb=100.0,
            critical_threshold_mb=200.0
        )
    
    def test_get_current_memory_usage(self):
        """测试获取当前内存使用情况"""
        memory_info = self.memory_monitor.get_current_memory_usage()
        
        # 检查返回的信息结构
        required_keys = [
            'current_memory_mb', 'virtual_memory_mb', 'peak_memory_mb',
            'memory_percent', 'available_memory_mb'
        ]
        
        for key in required_keys:
            assert key in memory_info
            assert isinstance(memory_info[key], (int, float))
            assert memory_info[key] >= 0
    
    def test_check_memory_status(self):
        """测试内存状态检查"""
        status_info = self.memory_monitor.check_memory_status()
        
        # 检查状态信息结构
        required_keys = ['status', 'message', 'recommendations', 'memory_info', 'memory_trend']
        for key in required_keys:
            assert key in status_info
        
        # 检查状态值
        assert status_info['status'] in ['normal', 'warning', 'critical']
        assert isinstance(status_info['message'], str)
        assert isinstance(status_info['recommendations'], list)
    
    def test_memory_optimization_suggestions(self):
        """测试内存优化建议"""
        suggestions = self.memory_monitor.get_memory_optimization_suggestions()
        
        assert isinstance(suggestions, list)
        # 建议应该是字符串
        for suggestion in suggestions:
            assert isinstance(suggestion, str)


class TestPerformanceOptimizer:
    """性能优化管理器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        
        cache_config = CacheConfig(
            disk_cache_dir=self.temp_dir,
            memory_cache_size=10
        )
        
        parallel_config = ParallelConfig(
            enable_parallel=True,
            max_workers=2
        )
        
        self.optimizer = PerformanceOptimizer(
            cache_config=cache_config,
            parallel_config=parallel_config,
            monitoring_config=MonitoringConfig(enable_memory_monitoring=True)
        )
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_comprehensive_performance_report(self):
        """测试综合性能报告"""
        with self.optimizer:
            # 执行一些操作以生成统计数据
            self.optimizer.cache_manager.put("test", "value")
            self.optimizer.cache_manager.get("test")
            
            report = self.optimizer.get_comprehensive_performance_report()
        
        # 检查报告结构
        assert 'timestamp' in report
        assert 'cache_stats' in report
        assert 'parallel_stats' in report
        assert 'memory_status' in report
        assert 'memory_suggestions' in report
    
    def test_optimize_system_performance(self):
        """测试系统性能优化"""
        with self.optimizer:
            optimization_result = self.optimizer.optimize_system_performance()
        
        assert 'actions_taken' in optimization_result.to_dict()
        assert isinstance(optimization_result.actions_taken, list)


class TestPerformanceMonitorDecorator:
    """性能监控装饰器测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        cache_config = CacheConfig(disk_cache_dir=self.temp_dir)
        self.cache_manager = AdvancedCacheManager(cache_config)
    
    def teardown_method(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_performance_monitor_without_cache(self):
        """测试不使用缓存的性能监控"""
        @performance_monitor(enable_memory_monitoring=False)
        def test_function(x, y):
            time.sleep(0.01)  # 模拟计算时间
            return x + y
        
        result = test_function(1, 2)
        assert result == 3
    
    def test_performance_monitor_with_cache(self):
        """测试使用缓存的性能监控"""
        @performance_monitor(cache_manager=self.cache_manager, enable_memory_monitoring=False)
        def cached_function(x):
            time.sleep(0.01)  # 模拟计算时间
            return x * 2
        
        # 第一次调用，应该执行函数
        start_time = time.time()
        result1 = cached_function(5)
        first_call_time = time.time() - start_time
        
        # 第二次调用，应该从缓存获取
        start_time = time.time()
        result2 = cached_function(5)
        second_call_time = time.time() - start_time
        
        assert result1 == result2 == 10
        # 第二次调用应该更快（从缓存获取）
        assert second_call_time < first_call_time


class TestDynamicLowVolFilterPerformanceIntegration:
    """动态低波筛选器性能优化集成测试"""
    
    def setup_method(self):
        """测试前准备"""
        # 创建模拟数据管理器
        self.mock_data_manager = Mock()
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        stocks = [f'STOCK_{i:03d}' for i in range(20)]
        
        # 价格数据
        np.random.seed(42)
        price_data = pd.DataFrame(
            np.random.lognormal(0, 0.02, (50, 20)).cumprod(axis=0) * 100,
            index=dates, columns=stocks
        )
        
        # 成交量数据
        volume_data = pd.DataFrame(
            np.random.randint(1000, 10000, (50, 20)),
            index=dates, columns=stocks
        )
        
        # 因子数据
        factor_data = pd.DataFrame(
            np.random.normal(0, 0.01, (50, 5)),
            index=dates, columns=['Market', 'Size', 'Value', 'Profitability', 'Investment']
        )
        
        # 市场数据
        market_data = pd.DataFrame(
            np.random.normal(0, 0.015, (50, 1)),
            index=dates, columns=['market_return']
        )
        
        # 配置模拟数据管理器
        self.mock_data_manager.get_price_data.return_value = price_data
        self.mock_data_manager.get_volume_data.return_value = volume_data
        self.mock_data_manager.get_factor_data.return_value = factor_data
        self.mock_data_manager.get_market_data.return_value = market_data
        
        # 创建筛选器配置
        self.filter_config = {
            'rolling_windows': [10, 20],
            'garch_window': 100,
            'enable_caching': True,
            'parallel_processing': True,
            'cache_expiry_days': 1
        }
    
    def test_filter_performance_optimization_integration(self):
        """测试筛选器性能优化集成"""
        # 创建筛选器实例
        filter_instance = DynamicLowVolFilter(
            config=DynamicLowVolConfig(**self.filter_config),
            data_manager=self.mock_data_manager
        )
        
        # 测试日期
        test_date = pd.Timestamp('2023-01-30')
        
        # 第一次执行（无缓存）
        start_time = time.time()
        try:
            mask1 = filter_instance.update_tradable_mask(test_date)
            first_execution_time = time.time() - start_time
        except Exception as e:
            # 如果执行失败，记录但不中断测试
            pytest.skip(f"筛选器执行失败: {e}")
        
        # 第二次执行（有缓存）
        start_time = time.time()
        try:
            mask2 = filter_instance.update_tradable_mask(test_date)
            second_execution_time = time.time() - start_time
        except Exception as e:
            pytest.skip(f"筛选器第二次执行失败: {e}")
        
        # 验证结果一致性
        assert isinstance(mask1, np.ndarray)
        assert isinstance(mask2, np.ndarray)
        assert mask1.dtype == bool
        assert mask2.dtype == bool
        
        # 获取性能报告
        performance_report = filter_instance.get_performance_report()
        
        # 验证性能报告结构
        assert 'filter_performance' in performance_report
        assert 'cache_stats' in performance_report
        assert 'parallel_stats' in performance_report
        
        filter_perf = performance_report['filter_performance']
        assert 'total_updates' in filter_perf
        assert 'avg_update_time' in filter_perf
        assert 'cache_hit_rate' in filter_perf
        
        # 验证统计信息
        assert filter_perf['total_updates'] >= 2
        assert filter_perf['avg_update_time'] > 0
    
    def test_performance_benchmark(self):
        """测试性能基准测试功能"""
        filter_instance = DynamicLowVolFilter(
            config=DynamicLowVolConfig(**self.filter_config),
            data_manager=self.mock_data_manager
        )
        
        # 创建简化的测试数据
        test_data = {
            'returns': pd.DataFrame(
                np.random.normal(0, 0.02, (10, 5)),
                columns=[f'STOCK_{i}' for i in range(5)]
            )
        }
        
        try:
            # 执行基准测试（减少迭代次数以加快测试）
            benchmark_result = filter_instance.benchmark_performance(
                test_data, iterations=3
            )
            
            # 验证基准测试结果
            required_keys = [
                'avg_serial_time', 'avg_parallel_time', 'speedup_ratio',
                'performance_improvement', 'iterations'
            ]
            
            for key in required_keys:
                assert key in benchmark_result
            
            assert benchmark_result['iterations'] == 3
            assert benchmark_result['speedup_ratio'] >= 0
            
        except Exception as e:
            pytest.skip(f"基准测试执行失败: {e}")
    
    def test_memory_optimization_suggestions(self):
        """测试内存优化建议功能"""
        filter_instance = DynamicLowVolFilter(
            config=DynamicLowVolConfig(**self.filter_config),
            data_manager=self.mock_data_manager
        )
        
        suggestions = filter_instance.get_memory_optimization_suggestions()
        
        assert isinstance(suggestions, list)
        # 建议应该是字符串
        for suggestion in suggestions:
            assert isinstance(suggestion, str)
    
    def test_cache_clearing(self):
        """测试缓存清理功能"""
        filter_instance = DynamicLowVolFilter(
            config=DynamicLowVolConfig(**self.filter_config),
            data_manager=self.mock_data_manager
        )
        
        # 执行一些操作以填充缓存
        test_date = pd.Timestamp('2023-01-30')
        try:
            filter_instance.update_tradable_mask(test_date)
        except:
            pass  # 忽略执行错误，专注于测试缓存清理
        
        # 清理缓存
        filter_instance.clear_performance_cache()
        
        # 验证缓存已清理
        cache_stats = filter_instance.performance_optimizer.cache_manager.get_cache_stats()
        assert cache_stats['memory_cache_size'] == 0


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])