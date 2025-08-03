"""
内存使用优化和缓存策略测试

该模块测试内存优化功能，包括：
- 内存效率测试
- 缓存策略测试
- 大数据集处理测试
- 内存泄漏检测测试
"""

import pytest
import numpy as np
import pandas as pd
import gc
import time
from typing import Dict, List
import psutil
from unittest.mock import Mock, patch

from src.rl_trading_system.performance.memory_optimizer import MemoryOptimizer, MemoryCache
from src.rl_trading_system.performance.memory_profiler import MemoryProfiler


class TestMemoryOptimizer:
    """内存优化器测试类"""
    
    @pytest.fixture
    def optimizer(self):
        """创建内存优化器实例"""
        return MemoryOptimizer()
    
    @pytest.fixture
    def large_dataframe(self):
        """创建大型测试DataFrame"""
        np.random.seed(42)
        n_rows = 10000
        
        data = {
            'int_col': np.random.randint(0, 1000, n_rows),
            'float_col': np.random.randn(n_rows),
            'category_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'string_col': [f'string_{i}' for i in range(n_rows)],
            'datetime_col': pd.date_range('2020-01-01', periods=n_rows, freq='H')
        }
        
        return pd.DataFrame(data)
    
    def test_dataframe_memory_optimization(self, optimizer, large_dataframe):
        """测试DataFrame内存优化"""
        original_memory = large_dataframe.memory_usage(deep=True).sum()
        
        optimized_df = optimizer.optimize_dataframe(large_dataframe)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        # 验证内存确实减少了
        assert optimized_memory < original_memory
        
        # 验证数据完整性
        assert len(optimized_df) == len(large_dataframe)
        assert list(optimized_df.columns) == list(large_dataframe.columns)
        
        # 验证数值精度
        np.testing.assert_array_almost_equal(
            optimized_df['float_col'].values,
            large_dataframe['float_col'].values,
            decimal=5
        )
        
        # 验证类别数据
        assert (optimized_df['category_col'].values == large_dataframe['category_col'].values).all()
    
    def test_memory_monitoring(self, optimizer):
        """测试内存监控功能"""
        initial_memory = optimizer.get_memory_usage()
        
        # 分配一些内存
        large_array = np.random.randn(1000000)
        
        current_memory = optimizer.get_memory_usage()
        
        # 验证内存增加
        assert current_memory > initial_memory
        
        # 清理
        del large_array
        gc.collect()
        
        final_memory = optimizer.get_memory_usage()
        
        # 验证内存回收（允许一些误差）
        assert abs(final_memory - initial_memory) < 50  # 50MB误差范围
    
    def test_garbage_collection_optimization(self, optimizer):
        """测试垃圾回收优化"""
        # 创建一些临时对象
        temp_objects = []
        for i in range(1000):
            temp_objects.append(np.random.randn(1000))
        
        memory_before_gc = optimizer.get_memory_usage()
        
        # 删除引用
        del temp_objects
        
        # 强制垃圾回收
        gc_result = optimizer.force_garbage_collection()
        
        # 验证回收结果
        assert gc_result['objects_collected'] >= 0
        assert 'memory_freed_mb' in gc_result
        assert 'memory_before_mb' in gc_result
        assert 'memory_after_mb' in gc_result
    
    def test_chunk_processing(self, optimizer):
        """测试分块处理功能"""
        # 创建大型数据集
        large_data = np.random.randn(100000, 50)
        
        chunk_size = 10000
        results = []
        
        # 分块处理
        for chunk in optimizer.process_in_chunks(large_data, chunk_size):
            chunk_result = {
                'mean': np.mean(chunk),
                'std': np.std(chunk),
                'size': chunk.shape[0]
            }
            results.append(chunk_result)
        
        # 验证分块结果
        assert len(results) == (len(large_data) + chunk_size - 1) // chunk_size
        
        # 验证每个块的大小
        for i, result in enumerate(results[:-1]):
            assert result['size'] == chunk_size
        
        # 最后一块可能小于chunk_size
        assert results[-1]['size'] <= chunk_size
    
    def test_memory_pool(self, optimizer):
        """测试内存池功能"""
        pool_size = 1000000  # 1M floats
        
        # 初始化内存池
        optimizer.initialize_memory_pool(pool_size)
        
        # 从内存池获取数组
        array1 = optimizer.get_array_from_pool(10000)
        array2 = optimizer.get_array_from_pool(20000)
        
        # 验证数组属性
        assert len(array1) == 10000
        assert len(array2) == 20000
        assert array1.dtype == np.float64
        assert array2.dtype == np.float64
        
        # 归还数组到内存池
        optimizer.return_array_to_pool(array1)
        optimizer.return_array_to_pool(array2)
        
        # 再次获取应该复用内存
        array3 = optimizer.get_array_from_pool(10000)
        assert len(array3) == 10000
    
    def test_memory_efficient_operations(self, optimizer):
        """测试内存高效操作"""
        # 创建大型矩阵
        matrix = np.random.randn(5000, 5000)
        
        # 内存高效的矩阵运算
        result = optimizer.memory_efficient_matrix_multiply(matrix, matrix.T)
        
        # 验证结果正确性
        expected = np.dot(matrix, matrix.T)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
        
        # 验证内存使用
        memory_usage = optimizer.get_memory_usage()
        assert memory_usage < 1000  # 应该控制在合理范围内


class TestMemoryCache:
    """内存缓存测试类"""
    
    @pytest.fixture
    def cache(self):
        """创建内存缓存实例"""
        return MemoryCache(max_size=100, ttl=3600)
    
    def test_cache_basic_operations(self, cache):
        """测试缓存基本操作"""
        # 设置缓存
        cache.set('key1', 'value1')
        cache.set('key2', {'nested': 'dict'})
        cache.set('key3', np.array([1, 2, 3]))
        
        # 获取缓存
        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == {'nested': 'dict'}
        np.testing.assert_array_equal(cache.get('key3'), np.array([1, 2, 3]))
        
        # 不存在的键
        assert cache.get('nonexistent') is None
        assert cache.get('nonexistent', 'default') == 'default'
    
    def test_cache_expiration(self, cache):
        """测试缓存过期"""
        # 创建短期缓存
        short_cache = MemoryCache(max_size=10, ttl=1)  # 1秒过期
        
        short_cache.set('temp_key', 'temp_value')
        assert short_cache.get('temp_key') == 'temp_value'
        
        # 等待过期
        time.sleep(1.1)
        
        assert short_cache.get('temp_key') is None
    
    def test_cache_size_limit(self, cache):
        """测试缓存大小限制"""
        # 创建小缓存
        small_cache = MemoryCache(max_size=3, ttl=3600)
        
        # 添加超过限制的项目
        small_cache.set('key1', 'value1')
        small_cache.set('key2', 'value2')
        small_cache.set('key3', 'value3')
        small_cache.set('key4', 'value4')  # 应该淘汰最旧的
        
        # 验证缓存大小
        assert len(small_cache._cache) <= 3
        
        # 最旧的项目应该被淘汰
        assert small_cache.get('key1') is None
        assert small_cache.get('key4') == 'value4'
    
    def test_cache_hit_rate(self, cache):
        """测试缓存命中率"""
        # 添加一些数据
        for i in range(10):
            cache.set(f'key{i}', f'value{i}')
        
        # 一些命中和未命中
        cache.get('key1')  # 命中
        cache.get('key2')  # 命中
        cache.get('nonexistent1')  # 未命中
        cache.get('key3')  # 命中
        cache.get('nonexistent2')  # 未命中
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 3
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.6
    
    def test_cache_clear(self, cache):
        """测试缓存清理"""
        # 添加数据
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        
        assert len(cache._cache) == 2
        
        # 清理缓存
        cache.clear()
        
        assert len(cache._cache) == 0
        assert cache.get('key1') is None
        assert cache.get('key2') is None


class TestMemoryProfiler:
    """内存分析器测试类"""
    
    @pytest.fixture
    def profiler(self):
        """创建内存分析器实例"""
        return MemoryProfiler()
    
    def test_function_memory_profiling(self, profiler):
        """测试函数内存分析"""
        def memory_intensive_function():
            """内存密集型函数"""
            large_array = np.random.randn(100000)
            result = np.sum(large_array ** 2)
            del large_array
            return result
        
        # 分析函数内存使用
        result, profile = profiler.profile_function(memory_intensive_function)
        
        # 验证结果
        assert isinstance(result, (int, float, np.number))
        assert 'peak_memory_mb' in profile
        assert 'memory_growth_mb' in profile
        assert 'execution_time' in profile
        
        # 验证内存分析结果
        assert profile['peak_memory_mb'] > 0
    
    def test_context_manager_profiling(self, profiler):
        """测试上下文管理器内存分析"""
        with profiler.profile_context() as profile:
            # 分配一些内存
            large_data = np.random.randn(50000)
            processed_data = large_data * 2
            del large_data, processed_data
        
        # 验证分析结果
        assert 'start_memory' in profile
        assert 'end_memory' in profile
        assert 'peak_memory' in profile
        assert 'memory_delta' in profile
    
    def test_memory_leak_detection(self, profiler):
        """测试内存泄漏检测"""
        def potentially_leaky_function():
            """可能有内存泄漏的函数"""
            data = []
            for i in range(1000):
                data.append(np.random.randn(100))
            return len(data)  # 返回而不清理，模拟泄漏
        
        # 执行多次并检测内存增长
        initial_memory = profiler.get_current_memory()
        
        for _ in range(5):
            potentially_leaky_function()
        
        final_memory = profiler.get_current_memory()
        memory_growth = final_memory - initial_memory
        
        # 检测是否有内存泄漏
        leak_detected = profiler.detect_memory_leak(
            initial_memory, final_memory, threshold_mb=10
        )
        
        # 根据内存增长判断
        if memory_growth > 10:  # 10MB阈值
            assert leak_detected
        
    def test_memory_hotspots_analysis(self, profiler):
        """测试内存热点分析"""
        def function_with_hotspots():
            """包含内存热点的函数"""
            # 热点1：大型数组分配
            big_array = np.random.randn(100000)
            
            # 热点2：重复的小分配
            small_arrays = []
            for i in range(1000):
                small_arrays.append(np.random.randn(100))
            
            # 热点3：DataFrame操作
            df = pd.DataFrame(np.random.randn(10000, 10))
            df_processed = df.apply(lambda x: x**2)
            
            return len(big_array) + len(small_arrays) + len(df_processed)
        
        # 分析内存热点
        hotspots = profiler.analyze_memory_hotspots(function_with_hotspots)
        
        # 验证热点分析结果
        assert isinstance(hotspots, dict)
        assert 'total_allocations' in hotspots
        assert 'peak_usage_mb' in hotspots
        assert 'allocation_pattern' in hotspots


class TestMemoryOptimizedDataStructures:
    """内存优化数据结构测试"""
    
    def test_memory_efficient_portfolio_data(self):
        """测试内存高效的投资组合数据结构"""
        from src.rl_trading_system.performance.memory_optimizer import MemoryEfficientPortfolioData
        
        # 创建大量投资组合数据
        n_portfolios = 1000
        n_assets = 100
        
        # 普通存储方式
        normal_data = []
        for i in range(n_portfolios):
            portfolio = {
                'weights': np.random.dirichlet(np.ones(n_assets)),
                'returns': np.random.randn(252),
                'metadata': {'id': i, 'name': f'portfolio_{i}'}
            }
            normal_data.append(portfolio)
        
        # 内存优化存储方式
        efficient_data = MemoryEfficientPortfolioData()
        
        for i in range(n_portfolios):
            efficient_data.add_portfolio(
                portfolio_id=i,
                weights=np.random.dirichlet(np.ones(n_assets)),
                returns=np.random.randn(252),
                metadata={'name': f'portfolio_{i}'}
            )
        
        # 比较内存使用
        normal_memory = sum(p['weights'].nbytes + p['returns'].nbytes for p in normal_data)
        efficient_memory = efficient_data.get_memory_usage()
        
        # 内存优化版本应该使用更少内存
        assert efficient_memory < normal_memory
        
        # 验证数据完整性
        assert efficient_data.get_portfolio_count() == n_portfolios
        
        # 随机检查一些投资组合
        for i in [0, 100, 500, 999]:
            portfolio = efficient_data.get_portfolio(i)
            assert len(portfolio['weights']) == n_assets
            assert len(portfolio['returns']) == 252
            assert portfolio['metadata']['name'] == f'portfolio_{i}'
    
    def test_sparse_matrix_optimization(self):
        """测试稀疏矩阵优化"""
        from src.rl_trading_system.performance.memory_optimizer import SparseMatrixOptimizer
        
        # 创建稀疏矩阵（大部分元素为0）
        n_rows, n_cols = 10000, 5000
        density = 0.01  # 1%非零元素
        
        # 生成稀疏数据
        np.random.seed(42)
        sparse_data = np.zeros((n_rows, n_cols))
        n_nonzero = int(n_rows * n_cols * density)
        
        for _ in range(n_nonzero):
            i = np.random.randint(0, n_rows)
            j = np.random.randint(0, n_cols)
            sparse_data[i, j] = np.random.randn()
        
        # 优化稀疏矩阵
        optimizer = SparseMatrixOptimizer()
        optimized_matrix = optimizer.optimize_sparse_matrix(sparse_data)
        
        # 验证内存节省
        original_memory = sparse_data.nbytes
        optimized_memory = optimizer.get_memory_usage(optimized_matrix)
        
        assert optimized_memory < original_memory
        
        # 验证数据完整性
        reconstructed = optimizer.to_dense(optimized_matrix)
        np.testing.assert_array_almost_equal(sparse_data, reconstructed)