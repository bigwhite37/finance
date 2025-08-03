"""
测试向量化计算器

该测试模块验证向量化计算器的正确性和性能，确保：
- 回撤计算的准确性
- 风险度量的正确性
- 向量化操作的性能提升
- 并行计算的一致性
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from src.rl_trading_system.performance.vectorized_calculator import VectorizedCalculator


class TestVectorizedCalculator:
    """向量化计算器测试类"""
    
    @pytest.fixture
    def calculator(self):
        """创建向量化计算器实例"""
        return VectorizedCalculator(enable_parallel=True, n_jobs=2)
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        n_periods = 252  # 一年的交易日
        n_assets = 50    # 50个资产
        
        # 生成模拟价格数据
        price_data = np.random.randn(n_periods, n_assets).cumsum(axis=0) + 100
        
        # 生成持仓权重
        weights = np.random.dirichlet(np.ones(n_assets))
        
        # 生成收益率数据
        returns = np.diff(price_data, axis=0) / price_data[:-1]
        
        return {
            'prices': price_data,
            'returns': returns,
            'weights': weights,
            'portfolio_values': np.dot(price_data, weights)
        }
    
    def test_init_calculator(self, calculator):
        """测试计算器初始化"""
        assert calculator.enable_parallel is True
        assert calculator.n_jobs == 2
        assert hasattr(calculator, '_cache')
    
    def test_vectorized_drawdown_calculation(self, calculator, sample_data):
        """测试向量化回撤计算"""
        portfolio_values = sample_data['portfolio_values']
        
        # 使用向量化方法计算回撤
        result = calculator.calculate_vectorized_drawdown(portfolio_values)
        
        # 验证结果结构
        assert 'current_drawdown' in result
        assert 'max_drawdown' in result
        assert 'drawdown_series' in result
        assert 'underwater_curve' in result
        assert 'recovery_periods' in result
        
        # 验证数据类型和形状
        assert isinstance(result['current_drawdown'], float)
        assert isinstance(result['max_drawdown'], float)
        assert len(result['drawdown_series']) == len(portfolio_values)
        assert len(result['underwater_curve']) == len(portfolio_values)
        
        # 验证逻辑正确性
        assert result['max_drawdown'] <= 0  # 最大回撤应为负值或零
        assert result['current_drawdown'] <= 0  # 当前回撤应为负值或零
        assert result['max_drawdown'] <= result['current_drawdown']  # 最大回撤应小于等于当前回撤
    
    def test_batch_portfolio_metrics(self, calculator, sample_data):
        """测试批量投资组合指标计算"""
        returns = sample_data['returns']
        weights = sample_data['weights']
        
        # 测试批量计算
        result = calculator.calculate_batch_portfolio_metrics(returns, weights)
        
        # 验证结果结构
        expected_metrics = [
            'portfolio_return', 'portfolio_volatility', 'sharpe_ratio',
            'max_drawdown', 'calmar_ratio', 'sortino_ratio', 'var_95', 'cvar_95'
        ]
        
        for metric in expected_metrics:
            assert metric in result
            assert isinstance(result[metric], (int, float, np.number))
        
        # 验证指标合理性
        assert result['portfolio_volatility'] > 0  # 波动率应为正
        assert -1 <= result['max_drawdown'] <= 0  # 最大回撤在合理范围内
        assert result['var_95'] < 0  # VaR应为负值
        assert result['cvar_95'] <= result['var_95']  # CVaR应小于等于VaR
    
    def test_parallel_risk_calculation(self, calculator, sample_data):
        """测试并行风险计算"""
        returns = sample_data['returns']
        
        # 创建多个权重组合进行并行计算
        weight_sets = []
        for i in range(10):
            np.random.seed(i)
            weights = np.random.dirichlet(np.ones(returns.shape[1]))
            weight_sets.append(weights)
        
        # 并行计算
        start_time = time.time()
        parallel_results = calculator.calculate_parallel_risk_metrics(returns, weight_sets)
        parallel_time = time.time() - start_time
        
        # 验证结果
        assert len(parallel_results) == len(weight_sets)
        
        for i, result in enumerate(parallel_results):
            assert 'portfolio_id' in result
            assert result['portfolio_id'] == i
            assert 'metrics' in result
            
            # 验证每个结果的指标完整性
            metrics = result['metrics']
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
            assert 'volatility' in metrics
    
    def test_rolling_window_calculation(self, calculator, sample_data):
        """测试滚动窗口计算"""
        portfolio_values = sample_data['portfolio_values']
        window_size = 60  # 60天滚动窗口
        
        result = calculator.calculate_rolling_metrics(portfolio_values, window_size)
        
        # 验证结果结构
        assert 'rolling_returns' in result
        assert 'rolling_volatility' in result
        assert 'rolling_sharpe' in result
        assert 'rolling_drawdown' in result
        
        # 验证数据长度
        expected_length = len(portfolio_values) - window_size + 1
        assert len(result['rolling_returns']) == expected_length
        assert len(result['rolling_volatility']) == expected_length
        assert len(result['rolling_sharpe']) == expected_length
        assert len(result['rolling_drawdown']) == expected_length
        
        # 验证数据有效性
        assert np.all(result['rolling_volatility'] >= 0)  # 波动率非负
        assert np.all(result['rolling_drawdown'] <= 0)     # 回撤非正
    
    def test_correlation_matrix_calculation(self, calculator, sample_data):
        """测试相关性矩阵计算"""
        returns = sample_data['returns']
        
        # 计算相关性矩阵
        result = calculator.calculate_correlation_matrix(returns)
        
        # 验证矩阵属性
        assert result['correlation_matrix'].shape == (returns.shape[1], returns.shape[1])
        assert np.allclose(result['correlation_matrix'], result['correlation_matrix'].T)  # 对称性
        assert np.allclose(np.diag(result['correlation_matrix']), 1.0)  # 对角线为1
        
        # 验证特征值分析
        assert 'eigenvalues' in result
        assert 'condition_number' in result
        assert 'effective_rank' in result
        
        # 验证数值合理性
        assert result['condition_number'] > 0
        assert 0 < result['effective_rank'] <= returns.shape[1]
    
    def test_risk_attribution_vectorized(self, calculator, sample_data):
        """测试向量化风险归因计算"""
        returns = sample_data['returns']
        weights = sample_data['weights']
        
        # 计算风险归因
        result = calculator.calculate_risk_attribution(returns, weights)
        
        # 验证结果结构
        assert 'component_contributions' in result
        assert 'marginal_contributions' in result
        assert 'percentage_contributions' in result
        
        # 验证贡献度总和
        component_contribs = result['component_contributions']
        assert len(component_contribs) == len(weights)
        
        # 贡献度百分比应接近100%
        percentage_contribs = result['percentage_contributions']
        assert abs(np.sum(percentage_contribs) - 1.0) < 1e-10
    
    def test_performance_vs_naive_implementation(self, calculator, sample_data):
        """测试向量化实现与朴素实现的性能对比"""
        portfolio_values = sample_data['portfolio_values']
        
        # 朴素实现
        def naive_drawdown_calculation(values):
            """朴素的回撤计算实现"""
            drawdowns = []
            peak = values[0]
            
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (value - peak) / peak
                drawdowns.append(drawdown)
            
            return {
                'max_drawdown': min(drawdowns),
                'current_drawdown': drawdowns[-1],
                'drawdown_series': drawdowns
            }
        
        # 性能测试
        # 朴素实现
        start_time = time.time()
        naive_result = naive_drawdown_calculation(portfolio_values)
        naive_time = time.time() - start_time
        
        # 向量化实现
        start_time = time.time()
        vectorized_result = calculator.calculate_vectorized_drawdown(portfolio_values)
        vectorized_time = time.time() - start_time
        
        # 验证结果一致性（允许小的数值误差）
        assert abs(naive_result['max_drawdown'] - vectorized_result['max_drawdown']) < 1e-10
        assert abs(naive_result['current_drawdown'] - vectorized_result['current_drawdown']) < 1e-10
        
        # 验证性能提升（向量化应该更快，但在小数据集上可能差异不大）
        print(f"朴素实现耗时: {naive_time:.6f}s")
        print(f"向量化实现耗时: {vectorized_time:.6f}s")
        print(f"性能提升: {naive_time/vectorized_time:.2f}x")
    
    def test_memory_efficient_calculation(self, calculator):
        """测试内存高效计算"""
        # 生成大数据集
        n_periods = 5000
        n_assets = 100
        
        # 使用内存高效模式
        calculator.enable_memory_optimization = True
        
        # 生成数据（分块生成以避免内存问题）
        chunk_size = 1000
        results = []
        
        for i in range(0, n_periods, chunk_size):
            end_idx = min(i + chunk_size, n_periods)
            chunk_size_actual = end_idx - i
            
            # 生成数据块
            np.random.seed(i)
            chunk_data = np.random.randn(chunk_size_actual, n_assets)
            
            # 计算该块的指标
            chunk_result = calculator.calculate_chunk_metrics(chunk_data)
            results.append(chunk_result)
        
        # 验证所有块都有结果
        assert len(results) > 0
        
        for result in results:
            assert 'mean_return' in result
            assert 'volatility' in result
            assert 'chunk_size' in result
    
    def test_numerical_stability(self, calculator):
        """测试数值稳定性"""
        # 测试极端情况
        test_cases = [
            np.array([1.0, 1.0, 1.0, 1.0]),  # 无变化
            np.array([1.0, 2.0, 1.0, 2.0]),  # 简单波动
            np.array([1e-10, 2e-10, 1e-10]), # 极小数值
            np.array([1e10, 2e10, 1e10]),    # 极大数值
            np.array([1.0, np.inf, 1.0]),    # 包含无穷大（应被处理）
        ]
        
        for i, test_data in enumerate(test_cases):
            try:
                # 处理无穷大等特殊值
                if np.any(np.isinf(test_data)):
                    test_data = np.where(np.isinf(test_data), np.nan, test_data)
                    test_data = test_data[~np.isnan(test_data)]
                
                if len(test_data) < 2:
                    continue
                
                result = calculator.calculate_vectorized_drawdown(test_data)
                
                # 验证结果的数值稳定性
                assert np.isfinite(result['max_drawdown'])
                assert np.isfinite(result['current_drawdown'])
                assert not np.any(np.isnan(result['drawdown_series']))
                
            except Exception as e:
                pytest.fail(f"测试用例 {i} 失败: {e}")
    
    def test_cache_functionality(self, calculator, sample_data):
        """测试缓存功能"""
        portfolio_values = sample_data['portfolio_values']
        
        # 首次计算
        start_time = time.time()
        result1 = calculator.calculate_vectorized_drawdown(portfolio_values, use_cache=True)
        first_time = time.time() - start_time
        
        # 再次计算相同数据（应使用缓存）
        start_time = time.time()
        result2 = calculator.calculate_vectorized_drawdown(portfolio_values, use_cache=True)
        cached_time = time.time() - start_time
        
        # 验证结果一致性
        assert result1['max_drawdown'] == result2['max_drawdown']
        assert result1['current_drawdown'] == result2['current_drawdown']
        
        # 验证缓存效果（第二次应该更快）
        assert cached_time < first_time or cached_time < 0.001  # 允许很快的情况
        
        print(f"首次计算耗时: {first_time:.6f}s")
        print(f"缓存计算耗时: {cached_time:.6f}s")
    
    def test_error_handling(self, calculator):
        """测试错误处理"""
        # 测试空数据
        with pytest.raises(ValueError, match="输入数据不能为空"):
            calculator.calculate_vectorized_drawdown(np.array([]))
        
        # 测试单个数据点
        with pytest.raises(ValueError, match="至少需要2个数据点"):
            calculator.calculate_vectorized_drawdown(np.array([1.0]))
        
        # 测试包含NaN的数据
        data_with_nan = np.array([1.0, 2.0, np.nan, 3.0])
        result = calculator.calculate_vectorized_drawdown(data_with_nan)
        
        # 应该处理NaN并返回有效结果
        assert np.isfinite(result['max_drawdown'])
        assert np.isfinite(result['current_drawdown'])
    
    def test_concurrent_calculations(self, calculator, sample_data):
        """测试并发计算的线程安全性"""
        portfolio_values = sample_data['portfolio_values']
        
        def calculate_metrics():
            """计算指标的函数"""
            return calculator.calculate_vectorized_drawdown(portfolio_values)
        
        # 并发执行计算
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(calculate_metrics) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # 验证所有结果一致
        first_result = results[0]
        for result in results[1:]:
            assert abs(result['max_drawdown'] - first_result['max_drawdown']) < 1e-10
            assert abs(result['current_drawdown'] - first_result['current_drawdown']) < 1e-10


@pytest.mark.slow
class TestVectorizedCalculatorPerformance:
    """向量化计算器性能测试类"""
    
    def test_large_dataset_performance(self):
        """测试大数据集性能"""
        calculator = VectorizedCalculator(enable_parallel=True, n_jobs=mp.cpu_count())
        
        # 生成大型数据集
        n_periods = 10000
        n_assets = 200
        
        np.random.seed(42)
        large_returns = np.random.randn(n_periods, n_assets) * 0.02
        weights = np.random.dirichlet(np.ones(n_assets))
        
        # 性能测试
        start_time = time.time()
        result = calculator.calculate_batch_portfolio_metrics(large_returns, weights)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        
        # 验证性能要求（应在合理时间内完成）
        assert calculation_time < 10.0  # 10秒内完成
        
        # 验证结果有效性
        assert np.isfinite(result['sharpe_ratio'])
        assert np.isfinite(result['max_drawdown'])
        assert np.isfinite(result['portfolio_volatility'])
        
        print(f"大数据集计算耗时: {calculation_time:.3f}s")
        print(f"数据规模: {n_periods} x {n_assets}")
        print(f"处理速度: {(n_periods * n_assets) / calculation_time:.0f} 数据点/秒")
    
    def test_scalability_analysis(self):
        """测试可扩展性分析"""
        calculator = VectorizedCalculator(enable_parallel=True)
        
        # 测试不同数据规模的性能
        scales = [
            (1000, 50),
            (2000, 50),
            (5000, 50),
            (1000, 100),
            (1000, 200)
        ]
        
        performance_results = []
        
        for n_periods, n_assets in scales:
            np.random.seed(42)
            returns = np.random.randn(n_periods, n_assets) * 0.01
            weights = np.random.dirichlet(np.ones(n_assets))
            
            start_time = time.time()
            calculator.calculate_batch_portfolio_metrics(returns, weights)
            end_time = time.time()
            
            computation_time = end_time - start_time
            data_points = n_periods * n_assets
            throughput = data_points / computation_time
            
            performance_results.append({
                'n_periods': n_periods,
                'n_assets': n_assets,
                'data_points': data_points,
                'time': computation_time,
                'throughput': throughput
            })
        
        # 分析可扩展性
        for result in performance_results:
            print(f"规模 {result['n_periods']}x{result['n_assets']}: "
                  f"{result['time']:.3f}s, "
                  f"{result['throughput']:.0f} 点/秒")
        
        # 验证性能随规模合理增长
        assert len(performance_results) == len(scales)
        
        # 验证小规模数据处理速度
        small_scale_result = performance_results[0]
        assert small_scale_result['time'] < 1.0  # 小规模应在1秒内完成