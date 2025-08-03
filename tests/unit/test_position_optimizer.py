"""仓位优化算法单元测试"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.rl_trading_system.risk_control.position_optimizer import (
    PositionOptimizer,
    OptimizationConfig,
    AssetData,
    OptimizationResult,
    OptimizationMethod,
    ObjectiveType
)


class TestPositionOptimizer(unittest.TestCase):
    """仓位优化算法测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = OptimizationConfig(
            method=OptimizationMethod.MEAN_VARIANCE,
            objective=ObjectiveType.MAXIMIZE_SHARPE,
            risk_aversion=1.0,
            min_weight=0.0,
            max_weight=0.5,
            transaction_cost_rate=0.001,
            max_iterations=100,
            tolerance=1e-6
        )
        self.optimizer = PositionOptimizer(self.config)
        
        # 创建测试资产数据
        self.asset_data = {
            'AAPL': AssetData(
                symbol='AAPL',
                expected_return=0.12,
                volatility=0.25,
                current_weight=0.2,
                market_cap=2e12,
                liquidity=0.9,
                sector='Technology',
                beta=1.2,
                price=150.0
            ),
            'GOOGL': AssetData(
                symbol='GOOGL',
                expected_return=0.10,
                volatility=0.30,
                current_weight=0.15,
                market_cap=1.5e12,
                liquidity=0.85,
                sector='Technology',
                beta=1.1,
                price=2500.0
            ),
            'JPM': AssetData(
                symbol='JPM',
                expected_return=0.08,
                volatility=0.20,
                current_weight=0.1,
                market_cap=400e9,
                liquidity=0.8,
                sector='Financial',
                beta=1.3,
                price=140.0
            ),
            'JNJ': AssetData(
                symbol='JNJ',
                expected_return=0.06,
                volatility=0.15,
                current_weight=0.05,
                market_cap=450e9,
                liquidity=0.75,
                sector='Healthcare',
                beta=0.8,
                price=170.0
            )
        }
        
        # 创建历史收益率数据
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        returns_data = {}
        for symbol, asset in self.asset_data.items():
            # 生成符合资产特征的随机收益率
            daily_returns = np.random.normal(
                asset.expected_return / 252,  # 日化收益率
                asset.volatility / np.sqrt(252),  # 日化波动率
                n_days
            )
            returns_data[symbol] = daily_returns
        
        self.historical_returns = pd.DataFrame(returns_data, index=dates)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.optimizer.config.method, OptimizationMethod.MEAN_VARIANCE)
        self.assertEqual(self.optimizer.config.objective, ObjectiveType.MAXIMIZE_SHARPE)
        self.assertEqual(len(self.optimizer.optimization_history), 0)
        self.assertEqual(self.optimizer.performance_stats['total_optimizations'], 0)
    
    def test_build_expected_returns_from_asset_data(self):
        """测试从资产数据构建预期收益率"""
        expected_returns = self.optimizer._build_expected_returns(self.asset_data, None)
        
        # 检查返回数组长度
        self.assertEqual(len(expected_returns), len(self.asset_data))
        
        # 检查收益率值
        symbols = list(self.asset_data.keys())
        for i, symbol in enumerate(symbols):
            self.assertAlmostEqual(
                expected_returns[i], 
                self.asset_data[symbol].expected_return,
                places=6
            )
    
    def test_build_expected_returns_from_historical_data(self):
        """测试从历史数据构建预期收益率"""
        expected_returns = self.optimizer._build_expected_returns(
            self.asset_data, self.historical_returns
        )
        
        # 检查返回数组长度
        self.assertEqual(len(expected_returns), len(self.asset_data))
        
        # 检查收益率为数值
        self.assertTrue(np.all(np.isfinite(expected_returns)))
    
    def test_build_covariance_matrix_from_asset_data(self):
        """测试从资产数据构建协方差矩阵"""
        covariance_matrix = self.optimizer._build_covariance_matrix(self.asset_data, None)
        
        # 检查矩阵维度
        n_assets = len(self.asset_data)
        self.assertEqual(covariance_matrix.shape, (n_assets, n_assets))
        
        # 检查对称性
        np.testing.assert_array_almost_equal(covariance_matrix, covariance_matrix.T)
        
        # 检查正定性（所有特征值为正）
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        self.assertTrue(np.all(eigenvalues > 0))
        
        # 检查对角线元素（方差）- 由于正则化和收缩，实际值可能与预期略有不同
        symbols = list(self.asset_data.keys())
        for i, symbol in enumerate(symbols):
            expected_variance = self.asset_data[symbol].volatility ** 2
            # 检查方差在合理范围内（考虑正则化和收缩的影响）
            self.assertGreater(covariance_matrix[i, i], 0)
            self.assertLess(covariance_matrix[i, i], expected_variance * 2)  # 不应该过大
    
    def test_build_covariance_matrix_from_historical_data(self):
        """测试从历史数据构建协方差矩阵"""
        covariance_matrix = self.optimizer._build_covariance_matrix(
            self.asset_data, self.historical_returns
        )
        
        # 检查矩阵维度
        n_assets = len(self.asset_data)
        self.assertEqual(covariance_matrix.shape, (n_assets, n_assets))
        
        # 检查对称性
        np.testing.assert_array_almost_equal(covariance_matrix, covariance_matrix.T)
        
        # 检查正定性
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        self.assertTrue(np.all(eigenvalues > 0))
    
    def test_calculate_transaction_costs(self):
        """测试交易成本计算"""
        current_weights = np.array([0.2, 0.15, 0.1, 0.05])
        new_weights = np.array([0.25, 0.2, 0.15, 0.1])
        symbols = list(self.asset_data.keys())
        
        transaction_costs = self.optimizer._calculate_transaction_costs(
            new_weights, current_weights, symbols
        )
        
        # 检查交易成本为正
        self.assertGreater(transaction_costs, 0)
        
        # 检查交易成本随换手率增加
        high_turnover_weights = np.array([0.4, 0.3, 0.2, 0.1])
        high_costs = self.optimizer._calculate_transaction_costs(
            high_turnover_weights, current_weights, symbols
        )
        self.assertGreater(high_costs, transaction_costs)
    
    def test_equal_weight_optimization(self):
        """测试等权重优化"""
        symbols = list(self.asset_data.keys())
        result = self.optimizer._equal_weight_optimization(symbols)
        
        # 检查结果类型
        self.assertIsInstance(result, OptimizationResult)
        
        # 检查权重和为1
        total_weight = sum(result.optimal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # 检查等权重
        expected_weight = 1.0 / len(symbols)
        for weight in result.optimal_weights.values():
            self.assertAlmostEqual(weight, expected_weight, places=6)
        
        # 检查状态
        self.assertEqual(result.optimization_status, 'fallback_equal_weight')
    
    def test_mean_variance_optimization(self):
        """测试均值方差优化"""
        result = self.optimizer.optimize_portfolio(self.asset_data)
        
        # 检查结果类型
        self.assertIsInstance(result, OptimizationResult)
        
        # 检查权重和为1
        total_weight = sum(result.optimal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # 检查权重约束
        for weight in result.optimal_weights.values():
            self.assertGreaterEqual(weight, self.config.min_weight - 1e-6)
            self.assertLessEqual(weight, self.config.max_weight + 1e-6)
        
        # 检查指标计算
        self.assertIsInstance(result.expected_return, float)
        self.assertIsInstance(result.expected_risk, float)
        self.assertIsInstance(result.sharpe_ratio, float)
        self.assertGreaterEqual(result.expected_risk, 0)
    
    def test_risk_parity_optimization(self):
        """测试风险平价优化"""
        config = OptimizationConfig(method=OptimizationMethod.RISK_PARITY)
        optimizer = PositionOptimizer(config)
        
        result = optimizer.optimize_portfolio(self.asset_data)
        
        # 检查结果类型
        self.assertIsInstance(result, OptimizationResult)
        
        # 检查权重和为1
        total_weight = sum(result.optimal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # 检查低波动率资产获得更高权重
        # JNJ波动率最低，应该获得相对较高的权重
        jnj_weight = result.optimal_weights['JNJ']
        googl_weight = result.optimal_weights['GOOGL']  # 波动率最高
        self.assertGreater(jnj_weight, googl_weight)
    
    def test_min_variance_optimization(self):
        """测试最小方差优化"""
        config = OptimizationConfig(method=OptimizationMethod.MIN_VARIANCE)
        optimizer = PositionOptimizer(config)
        
        result = optimizer.optimize_portfolio(self.asset_data)
        
        # 检查结果类型
        self.assertIsInstance(result, OptimizationResult)
        
        # 检查权重和为1
        total_weight = sum(result.optimal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # 最小方差优化应该偏向低波动率资产
        jnj_weight = result.optimal_weights['JNJ']  # 波动率最低
        self.assertGreater(jnj_weight, 0)
    
    def test_max_sharpe_optimization(self):
        """测试最大夏普比率优化"""
        config = OptimizationConfig(method=OptimizationMethod.MAX_SHARPE)
        optimizer = PositionOptimizer(config)
        
        result = optimizer.optimize_portfolio(self.asset_data)
        
        # 检查结果类型
        self.assertIsInstance(result, OptimizationResult)
        
        # 检查权重和为1
        total_weight = sum(result.optimal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # 检查夏普比率计算
        if result.expected_risk > 0:
            expected_sharpe = result.expected_return / result.expected_risk
            self.assertAlmostEqual(result.sharpe_ratio, expected_sharpe, places=6)
    
    def test_black_litterman_optimization(self):
        """测试Black-Litterman优化"""
        config = OptimizationConfig(method=OptimizationMethod.BLACK_LITTERMAN)
        optimizer = PositionOptimizer(config)
        
        # 提供基准权重
        benchmark_weights = {
            'AAPL': 0.3,
            'GOOGL': 0.25,
            'JPM': 0.25,
            'JNJ': 0.2
        }
        
        result = optimizer.optimize_portfolio(
            self.asset_data, 
            benchmark_weights=benchmark_weights
        )
        
        # 检查结果类型
        self.assertIsInstance(result, OptimizationResult)
        
        # 检查权重和为1
        total_weight = sum(result.optimal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_optimization_with_historical_data(self):
        """测试使用历史数据的优化"""
        result = self.optimizer.optimize_portfolio(
            self.asset_data, 
            historical_returns=self.historical_returns
        )
        
        # 检查结果类型
        self.assertIsInstance(result, OptimizationResult)
        
        # 检查权重和为1
        total_weight = sum(result.optimal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # 检查计算时间被记录
        self.assertGreater(result.computation_time, 0)
    
    def test_optimization_with_constraints(self):
        """测试带约束的优化"""
        # 设置目标收益率约束
        config = OptimizationConfig(
            method=OptimizationMethod.MEAN_VARIANCE,
            target_return=0.09,
            min_weight=0.05,
            max_weight=0.4
        )
        optimizer = PositionOptimizer(config)
        
        result = optimizer.optimize_portfolio(self.asset_data)
        
        # 检查权重约束
        for weight in result.optimal_weights.values():
            self.assertGreaterEqual(weight, config.min_weight - 1e-6)
            self.assertLessEqual(weight, config.max_weight + 1e-6)
        
        # 检查目标收益率约束（允许一定误差）
        if result.optimization_status == 'success':
            self.assertAlmostEqual(result.expected_return, config.target_return, places=2)
    
    def test_optimization_empty_asset_data(self):
        """测试空资产数据的异常处理"""
        # 空资产数据应该返回等权重结果而不是抛出异常
        result = self.optimizer.optimize_portfolio({})
        self.assertIn('fallback_equal_weight', result.optimization_status)
        self.assertEqual(len(result.optimal_weights), 0)
    
    def test_optimization_with_custom_constraints(self):
        """测试自定义约束"""
        # 创建自定义约束：AAPL + GOOGL <= 0.6
        def custom_constraint(weights):
            symbols = list(self.asset_data.keys())
            aapl_idx = symbols.index('AAPL')
            googl_idx = symbols.index('GOOGL')
            return 0.6 - (weights[aapl_idx] + weights[googl_idx])
        
        custom_constraints = [{'type': 'ineq', 'fun': custom_constraint}]
        
        result = self.optimizer.optimize_portfolio(
            self.asset_data,
            custom_constraints=custom_constraints
        )
        
        # 检查约束是否满足
        aapl_weight = result.optimal_weights['AAPL']
        googl_weight = result.optimal_weights['GOOGL']
        self.assertLessEqual(aapl_weight + googl_weight, 0.6 + 1e-6)
    
    def test_get_efficient_frontier(self):
        """测试有效前沿计算"""
        frontier = self.optimizer.get_efficient_frontier(self.asset_data, n_points=10)
        
        # 检查返回结果
        self.assertIsInstance(frontier, list)
        self.assertGreater(len(frontier), 0)
        
        # 检查每个点都是OptimizationResult
        for point in frontier:
            self.assertIsInstance(point, OptimizationResult)
            
            # 检查权重和为1
            total_weight = sum(point.optimal_weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=6)
        
        # 检查风险范围合理
        risks = [point.expected_risk for point in frontier]
        # 检查风险值都为正数且在合理范围内
        self.assertTrue(all(risk >= 0 for risk in risks))
        if len(set(risks)) > 1:  # 如果有不同的风险值
            self.assertGreater(max(risks), min(risks))
    
    def test_get_performance_attribution(self):
        """测试业绩归因分析"""
        result = self.optimizer.optimize_portfolio(self.asset_data)
        attribution = self.optimizer.get_performance_attribution(result, self.asset_data)
        
        # 检查归因结构
        self.assertIn('asset_contribution', attribution)
        self.assertIn('sector_contribution', attribution)
        self.assertIn('factor_contribution', attribution)
        self.assertIn('risk_contribution', attribution)
        
        # 检查资产贡献
        asset_contrib = attribution['asset_contribution']
        self.assertEqual(len(asset_contrib), len(self.asset_data))
        
        # 检查贡献值为数值
        for contrib in asset_contrib.values():
            self.assertIsInstance(contrib, (int, float))
        
        # 检查行业贡献
        sector_contrib = attribution['sector_contribution']
        self.assertGreater(len(sector_contrib), 0)
    
    def test_get_optimization_summary(self):
        """测试优化摘要"""
        # 先进行几次优化
        self.optimizer.optimize_portfolio(self.asset_data)
        
        config2 = OptimizationConfig(method=OptimizationMethod.RISK_PARITY)
        optimizer2 = PositionOptimizer(config2)
        optimizer2.optimize_portfolio(self.asset_data)
        
        summary = self.optimizer.get_optimization_summary()
        
        # 检查摘要结构
        self.assertIn('total_optimizations', summary)
        self.assertIn('performance_stats', summary)
        self.assertIn('recent_results', summary)
        
        # 检查统计信息
        self.assertEqual(summary['total_optimizations'], 1)
        self.assertIn('successful_optimizations', summary['performance_stats'])
        self.assertIn('average_computation_time', summary['performance_stats'])
        
        # 检查最近结果
        recent_results = summary['recent_results']
        self.assertEqual(len(recent_results), 1)
        self.assertIn('timestamp', recent_results[0])
        self.assertIn('method', recent_results[0])
        self.assertIn('status', recent_results[0])
    
    def test_reset_history(self):
        """测试重置历史"""
        # 先进行一次优化
        self.optimizer.optimize_portfolio(self.asset_data)
        
        # 确认有历史记录
        self.assertGreater(len(self.optimizer.optimization_history), 0)
        self.assertGreater(self.optimizer.performance_stats['total_optimizations'], 0)
        
        # 重置历史
        self.optimizer.reset_history()
        
        # 确认历史已清空
        self.assertEqual(len(self.optimizer.optimization_history), 0)
        self.assertEqual(self.optimizer.performance_stats['total_optimizations'], 0)
        self.assertEqual(self.optimizer.performance_stats['successful_optimizations'], 0)
        self.assertEqual(self.optimizer.performance_stats['average_computation_time'], 0.0)
    
    def test_optimization_methods_enum(self):
        """测试所有优化方法"""
        methods_to_test = [
            OptimizationMethod.MEAN_VARIANCE,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MIN_VARIANCE,
            OptimizationMethod.MAX_SHARPE,
            OptimizationMethod.BLACK_LITTERMAN,
            OptimizationMethod.EQUAL_WEIGHT
        ]
        
        for method in methods_to_test:
            with self.subTest(method=method):
                config = OptimizationConfig(method=method)
                optimizer = PositionOptimizer(config)
                
                result = optimizer.optimize_portfolio(self.asset_data)
                
                # 检查基本结果
                self.assertIsInstance(result, OptimizationResult)
                total_weight = sum(result.optimal_weights.values())
                self.assertAlmostEqual(total_weight, 1.0, places=6)
    
    def test_unsupported_optimization_method(self):
        """测试不支持的优化方法"""
        # 创建一个不存在的方法（通过直接修改config）
        config = OptimizationConfig()
        optimizer = PositionOptimizer(config)
        
        # 手动设置一个无效的方法（使用字符串而不是枚举）
        # 这会在_create_optimization_result中导致错误，然后回退到等权重
        class MockMethod:
            def __init__(self, value):
                self.value = value
        
        optimizer.config.method = MockMethod("INVALID_METHOD")
        
        # 应该回退到等权重
        result = optimizer.optimize_portfolio(self.asset_data)
        self.assertIn('fallback_equal_weight', result.optimization_status)
    
    def test_optimization_result_creation(self):
        """测试优化结果创建"""
        symbols = list(self.asset_data.keys())
        n_assets = len(symbols)
        
        optimal_weights = np.array([0.3, 0.3, 0.2, 0.2])
        expected_returns = np.array([0.12, 0.10, 0.08, 0.06])
        covariance_matrix = np.eye(n_assets) * 0.01
        current_weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        result = self.optimizer._create_optimization_result(
            optimal_weights, expected_returns, covariance_matrix, current_weights,
            symbols, 'success', 10
        )
        
        # 检查结果属性
        self.assertEqual(result.optimization_status, 'success')
        self.assertEqual(result.iterations, 10)
        self.assertEqual(len(result.optimal_weights), n_assets)
        
        # 检查权重字典
        for i, symbol in enumerate(symbols):
            self.assertAlmostEqual(result.optimal_weights[symbol], optimal_weights[i], places=6)
        
        # 检查计算指标
        self.assertIsInstance(result.expected_return, float)
        self.assertIsInstance(result.expected_risk, float)
        self.assertIsInstance(result.sharpe_ratio, float)
        self.assertIsInstance(result.turnover, float)
        self.assertIsInstance(result.transaction_costs, float)
    
    def test_performance_stats_tracking(self):
        """测试性能统计跟踪"""
        initial_stats = self.optimizer.performance_stats.copy()
        
        # 进行优化
        result = self.optimizer.optimize_portfolio(self.asset_data)
        
        # 检查统计更新
        self.assertEqual(
            self.optimizer.performance_stats['total_optimizations'],
            initial_stats['total_optimizations'] + 1
        )
        
        if result.optimization_status == 'success':
            self.assertEqual(
                self.optimizer.performance_stats['successful_optimizations'],
                initial_stats['successful_optimizations'] + 1
            )
        
        # 检查平均计算时间更新
        self.assertGreater(
            self.optimizer.performance_stats['average_computation_time'],
            initial_stats['average_computation_time']
        )
    
    def test_covariance_matrix_regularization(self):
        """测试协方差矩阵正则化"""
        # 创建一个可能奇异的协方差矩阵情况
        asset_data_singular = {
            'A': AssetData('A', 0.1, 0.2, 0.5, 1e12, 0.9, 'Tech'),
            'B': AssetData('B', 0.1, 0.2, 0.5, 1e12, 0.9, 'Tech')  # 相同的波动率
        }
        
        covariance_matrix = self.optimizer._build_covariance_matrix(asset_data_singular, None)
        
        # 检查正定性（正则化后应该是正定的）
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        self.assertTrue(np.all(eigenvalues > 0))
        
        # 检查对角线元素包含正则化项
        expected_variance = 0.2 ** 2
        self.assertGreater(covariance_matrix[0, 0], expected_variance)
    
    def test_shrinkage_estimation(self):
        """测试收缩估计"""
        config = OptimizationConfig(shrinkage_factor=0.5)
        optimizer = PositionOptimizer(config)
        
        covariance_matrix = optimizer._build_covariance_matrix(
            self.asset_data, self.historical_returns
        )
        
        # 检查矩阵维度和正定性
        n_assets = len(self.asset_data)
        self.assertEqual(covariance_matrix.shape, (n_assets, n_assets))
        
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        self.assertTrue(np.all(eigenvalues > 0))


if __name__ == '__main__':
    unittest.main()