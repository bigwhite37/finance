"""风险预算分配器单元测试"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta

from src.rl_trading_system.risk_control.risk_budget_allocator import (
    RiskBudgetAllocator,
    RiskBudgetConfig,
    AssetRiskMetrics,
    RiskBudgetAllocation,
    AllocationMethod
)


class TestRiskBudgetAllocator(unittest.TestCase):
    """风险预算分配器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = RiskBudgetConfig(
            base_risk_budget=0.10,
            max_risk_budget=0.20,
            min_risk_budget=0.02,
            drawdown_warning_threshold=0.05,
            drawdown_critical_threshold=0.10,
            risk_scaling_factor=0.5
        )
        self.allocator = RiskBudgetAllocator(self.config)
        
        # 创建测试资产风险指标
        self.asset_risk_metrics = {
            'AAPL': AssetRiskMetrics(
                symbol='AAPL',
                volatility=0.25,
                beta=1.2,
                var_95=-0.03,
                expected_shortfall=-0.05,
                correlation_with_market=0.8,
                liquidity_score=0.9,
                sector='Technology',
                timestamp=datetime.now()
            ),
            'GOOGL': AssetRiskMetrics(
                symbol='GOOGL',
                volatility=0.30,
                beta=1.1,
                var_95=-0.035,
                expected_shortfall=-0.055,
                correlation_with_market=0.75,
                liquidity_score=0.85,
                sector='Technology',
                timestamp=datetime.now()
            ),
            'JPM': AssetRiskMetrics(
                symbol='JPM',
                volatility=0.20,
                beta=1.3,
                var_95=-0.025,
                expected_shortfall=-0.04,
                correlation_with_market=0.85,
                liquidity_score=0.8,
                sector='Financial',
                timestamp=datetime.now()
            )
        }
        
        self.asset_universe = list(self.asset_risk_metrics.keys())
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.allocator.config.base_risk_budget, 0.10)
        self.assertEqual(self.allocator.current_risk_budget, 0.10)
        self.assertEqual(self.allocator.current_drawdown, 0.0)
        self.assertEqual(len(self.allocator.allocation_history), 0)
    
    def test_calculate_dynamic_risk_budget_low_drawdown(self):
        """测试低回撤时的动态风险预算计算"""
        # 低回撤情况
        current_drawdown = 0.02  # 2%回撤，低于警告阈值
        new_budget = self.allocator.calculate_dynamic_risk_budget(current_drawdown)
        
        # 低回撤时应保持基础风险预算
        self.assertAlmostEqual(new_budget, self.config.base_risk_budget, places=3)
        self.assertEqual(self.allocator.current_drawdown, 0.02)
    
    def test_calculate_dynamic_risk_budget_medium_drawdown(self):
        """测试中等回撤时的动态风险预算计算"""
        # 中等回撤情况
        current_drawdown = 0.07  # 7%回撤，在警告和临界阈值之间
        new_budget = self.allocator.calculate_dynamic_risk_budget(current_drawdown)
        
        # 中等回撤时应降低风险预算
        self.assertLess(new_budget, self.config.base_risk_budget)
        self.assertGreater(new_budget, self.config.min_risk_budget)
    
    def test_calculate_dynamic_risk_budget_high_drawdown(self):
        """测试高回撤时的动态风险预算计算"""
        # 高回撤情况
        current_drawdown = 0.15  # 15%回撤，超过临界阈值
        new_budget = self.allocator.calculate_dynamic_risk_budget(current_drawdown)
        
        # 高回撤时应大幅降低风险预算
        self.assertLess(new_budget, self.config.base_risk_budget)
        self.assertGreaterEqual(new_budget, self.config.min_risk_budget)
        
        # 检查风险预算确实被显著降低了（至少降低20%）
        reduction_ratio = (self.config.base_risk_budget - new_budget) / self.config.base_risk_budget
        self.assertGreater(reduction_ratio, 0.2)  # 至少降低20%
    
    def test_calculate_dynamic_risk_budget_with_market_volatility(self):
        """测试包含市场波动率的动态风险预算计算"""
        current_drawdown = 0.03
        market_volatility = 0.8  # 高波动率
        
        new_budget = self.allocator.calculate_dynamic_risk_budget(
            current_drawdown, market_volatility=market_volatility
        )
        
        # 高波动率应进一步降低风险预算
        baseline_budget = self.allocator.calculate_dynamic_risk_budget(current_drawdown)
        self.assertLess(new_budget, baseline_budget)
    
    def test_calculate_dynamic_risk_budget_with_performance_metrics(self):
        """测试包含历史表现的动态风险预算计算"""
        current_drawdown = 0.03
        performance_metrics = {'sharpe_ratio': 1.5}  # 良好的夏普比率
        
        new_budget = self.allocator.calculate_dynamic_risk_budget(
            current_drawdown, performance_metrics=performance_metrics
        )
        
        # 良好表现应适度增加风险预算
        baseline_budget = self.allocator.calculate_dynamic_risk_budget(current_drawdown)
        self.assertGreaterEqual(new_budget, baseline_budget)
    
    def test_equal_weight_allocation(self):
        """测试等权重分配"""
        weights = self.allocator._equal_weight_allocation(self.asset_universe)
        
        # 检查权重和为1
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        
        # 检查每个资产权重相等
        expected_weight = 1.0 / len(self.asset_universe)
        for weight in weights.values():
            self.assertAlmostEqual(weight, expected_weight, places=6)
    
    def test_risk_parity_allocation(self):
        """测试风险平价分配"""
        weights = self.allocator._risk_parity_allocation(
            self.asset_universe, self.asset_risk_metrics
        )
        
        # 检查权重和为1
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        
        # 检查低波动率资产获得更高权重
        # JPM波动率最低(0.20)，应获得最高权重
        self.assertGreater(weights['JPM'], weights['AAPL'])
        self.assertGreater(weights['JPM'], weights['GOOGL'])
    
    def test_volatility_weighted_allocation(self):
        """测试波动率加权分配"""
        weights = self.allocator._volatility_weighted_allocation(
            self.asset_universe, self.asset_risk_metrics
        )
        
        # 波动率加权应与风险平价相同
        risk_parity_weights = self.allocator._risk_parity_allocation(
            self.asset_universe, self.asset_risk_metrics
        )
        
        for asset in self.asset_universe:
            self.assertAlmostEqual(weights[asset], risk_parity_weights[asset], places=6)
    
    def test_correlation_adjusted_allocation(self):
        """测试相关性调整分配"""
        # 创建相关性矩阵
        correlation_matrix = np.array([
            [1.0, 0.8, 0.6],  # AAPL
            [0.8, 1.0, 0.5],  # GOOGL
            [0.6, 0.5, 1.0]   # JPM
        ])
        
        weights = self.allocator._correlation_adjusted_allocation(
            self.asset_universe, self.asset_risk_metrics, correlation_matrix
        )
        
        # 检查权重和为1
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        
        # 检查所有权重为正
        for weight in weights.values():
            self.assertGreater(weight, 0)
    
    def test_correlation_adjusted_allocation_without_matrix(self):
        """测试无相关性矩阵时的相关性调整分配"""
        weights = self.allocator._correlation_adjusted_allocation(
            self.asset_universe, self.asset_risk_metrics, None
        )
        
        # 应回退到风险平价
        risk_parity_weights = self.allocator._risk_parity_allocation(
            self.asset_universe, self.asset_risk_metrics
        )
        
        for asset in self.asset_universe:
            self.assertAlmostEqual(weights[asset], risk_parity_weights[asset], places=6)
    
    def test_drawdown_adjusted_allocation_low_drawdown(self):
        """测试低回撤时的回撤调整分配"""
        self.allocator.current_drawdown = 0.02  # 低于警告阈值
        
        weights = self.allocator._drawdown_adjusted_allocation(
            self.asset_universe, self.asset_risk_metrics
        )
        
        # 低回撤时应与风险平价相同
        risk_parity_weights = self.allocator._risk_parity_allocation(
            self.asset_universe, self.asset_risk_metrics
        )
        
        for asset in self.asset_universe:
            self.assertAlmostEqual(weights[asset], risk_parity_weights[asset], places=6)
    
    def test_drawdown_adjusted_allocation_high_drawdown(self):
        """测试高回撤时的回撤调整分配"""
        self.allocator.current_drawdown = 0.08  # 高于警告阈值
        
        weights = self.allocator._drawdown_adjusted_allocation(
            self.asset_universe, self.asset_risk_metrics
        )
        
        # 检查权重和为1
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        
        # 高回撤时应偏向低风险资产
        # JPM的VaR最小，应获得相对更高权重
        self.assertGreater(weights['JPM'], 0)
    
    def test_apply_allocation_constraints(self):
        """测试分配约束应用"""
        # 创建违反约束的权重
        weights = {
            'AAPL': 0.8,  # 超过最大权重限制
            'GOOGL': 0.15,
            'JPM': 0.05
        }
        
        constrained_weights = self.allocator._apply_allocation_constraints(weights)
        
        # 检查权重和为1
        self.assertAlmostEqual(sum(constrained_weights.values()), 1.0, places=6)
        
        # 检查单一资产权重限制
        max_weight = self.config.max_single_asset_budget / self.config.base_risk_budget
        for weight in constrained_weights.values():
            self.assertLessEqual(weight, max_weight + 1e-6)  # 允许小的数值误差
    
    def test_allocate_risk_budget_equal_weight(self):
        """测试等权重风险预算分配"""
        allocation = self.allocator.allocate_risk_budget(
            self.asset_universe,
            self.asset_risk_metrics,
            allocation_method=AllocationMethod.EQUAL_WEIGHT
        )
        
        # 检查分配结果
        self.assertIsInstance(allocation, RiskBudgetAllocation)
        self.assertEqual(allocation.allocation_method, AllocationMethod.EQUAL_WEIGHT)
        self.assertEqual(len(allocation.asset_allocations), len(self.asset_universe))
        
        # 检查总风险预算
        total_allocated = sum(allocation.asset_allocations.values())
        self.assertAlmostEqual(total_allocated, self.config.base_risk_budget, places=6)
        
        # 检查等权重
        expected_allocation = self.config.base_risk_budget / len(self.asset_universe)
        for allocation_value in allocation.asset_allocations.values():
            self.assertAlmostEqual(allocation_value, expected_allocation, places=6)
    
    def test_allocate_risk_budget_risk_parity(self):
        """测试风险平价风险预算分配"""
        allocation = self.allocator.allocate_risk_budget(
            self.asset_universe,
            self.asset_risk_metrics,
            allocation_method=AllocationMethod.RISK_PARITY
        )
        
        # 检查分配结果
        self.assertEqual(allocation.allocation_method, AllocationMethod.RISK_PARITY)
        
        # 检查总风险预算
        total_allocated = sum(allocation.asset_allocations.values())
        self.assertAlmostEqual(total_allocated, self.config.base_risk_budget, places=6)
        
        # 检查低波动率资产获得更多预算
        self.assertGreater(allocation.asset_allocations['JPM'], 
                          allocation.asset_allocations['GOOGL'])
    
    def test_allocate_risk_budget_with_correlation_matrix(self):
        """测试带相关性矩阵的风险预算分配"""
        correlation_matrix = np.array([
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.5],
            [0.6, 0.5, 1.0]
        ])
        
        allocation = self.allocator.allocate_risk_budget(
            self.asset_universe,
            self.asset_risk_metrics,
            correlation_matrix=correlation_matrix,
            allocation_method=AllocationMethod.CORRELATION_ADJUSTED
        )
        
        # 检查分配结果
        self.assertEqual(allocation.allocation_method, AllocationMethod.CORRELATION_ADJUSTED)
        
        # 检查总风险预算
        total_allocated = sum(allocation.asset_allocations.values())
        self.assertAlmostEqual(total_allocated, self.config.base_risk_budget, places=6)
    
    def test_allocate_risk_budget_empty_universe(self):
        """测试空资产列表的异常处理"""
        with self.assertRaises(ValueError):
            self.allocator.allocate_risk_budget([], self.asset_risk_metrics)
    
    def test_allocate_risk_budget_unsupported_method(self):
        """测试不支持的分配方法异常处理"""
        # 创建一个不存在的分配方法
        with self.assertRaises(ValueError):
            # 直接传入字符串而不是枚举值来触发异常
            self.allocator.allocate_risk_budget(
                self.asset_universe,
                self.asset_risk_metrics,
                allocation_method="UNSUPPORTED_METHOD"
            )
    
    def test_calculate_risk_contributions(self):
        """测试风险贡献计算"""
        weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'JPM': 0.3}
        
        risk_contributions = self.allocator._calculate_risk_contributions(
            weights, self.asset_risk_metrics
        )
        
        # 检查所有资产都有风险贡献
        self.assertEqual(len(risk_contributions), len(weights))
        
        # 检查风险贡献为正
        for contribution in risk_contributions.values():
            self.assertGreater(contribution, 0)
        
        # 检查高波动率资产有更高的风险贡献
        self.assertGreater(risk_contributions['GOOGL'], risk_contributions['JPM'])
    
    def test_calculate_expected_portfolio_risk(self):
        """测试预期组合风险计算"""
        weights = {'AAPL': 0.4, 'GOOGL': 0.3, 'JPM': 0.3}
        
        portfolio_risk = self.allocator._calculate_expected_portfolio_risk(
            weights, self.asset_risk_metrics
        )
        
        # 检查组合风险为正
        self.assertGreater(portfolio_risk, 0)
        
        # 检查组合风险在合理范围内
        self.assertLess(portfolio_risk, 1.0)
    
    def test_calculate_diversification_ratio(self):
        """测试多样化比率计算"""
        # 等权重情况
        equal_weights = {'AAPL': 0.33, 'GOOGL': 0.33, 'JPM': 0.34}
        equal_div_ratio = self.allocator._calculate_diversification_ratio(
            equal_weights, self.asset_risk_metrics
        )
        
        # 集中权重情况
        concentrated_weights = {'AAPL': 0.8, 'GOOGL': 0.1, 'JPM': 0.1}
        concentrated_div_ratio = self.allocator._calculate_diversification_ratio(
            concentrated_weights, self.asset_risk_metrics
        )
        
        # 等权重应有更高的多样化比率
        self.assertGreater(equal_div_ratio, concentrated_div_ratio)
        
        # 多样化比率应在0-1之间
        self.assertGreaterEqual(equal_div_ratio, 0)
        self.assertLessEqual(equal_div_ratio, 1)
    
    def test_get_allocation_summary_empty_history(self):
        """测试空历史的分配摘要"""
        summary = self.allocator.get_allocation_summary()
        
        self.assertEqual(summary['total_allocations'], 0)
        self.assertEqual(summary['current_risk_budget'], self.config.base_risk_budget)
        self.assertEqual(summary['current_drawdown'], 0.0)
    
    def test_get_allocation_summary_with_history(self):
        """测试有历史记录的分配摘要"""
        # 先进行一次分配
        self.allocator.allocate_risk_budget(
            self.asset_universe,
            self.asset_risk_metrics
        )
        
        summary = self.allocator.get_allocation_summary()
        
        self.assertEqual(summary['total_allocations'], 1)
        self.assertIn('latest_allocation', summary)
        self.assertIn('risk_budget_trend', summary)
    
    def test_reset_allocation_history(self):
        """测试重置分配历史"""
        # 先进行一些操作
        self.allocator.calculate_dynamic_risk_budget(0.05)
        self.allocator.allocate_risk_budget(
            self.asset_universe,
            self.asset_risk_metrics
        )
        
        # 确认有历史记录
        self.assertGreater(len(self.allocator.allocation_history), 0)
        self.assertGreater(len(self.allocator.risk_budget_history), 0)
        
        # 重置历史
        self.allocator.reset_allocation_history()
        
        # 确认历史已清空
        self.assertEqual(len(self.allocator.allocation_history), 0)
        self.assertEqual(len(self.allocator.risk_budget_history), 0)
        self.assertEqual(len(self.allocator.drawdown_history), 0)
    
    def test_risk_budget_bounds(self):
        """测试风险预算边界约束"""
        # 测试极端低回撤
        very_low_drawdown = -0.05  # 负回撤（盈利）
        budget = self.allocator.calculate_dynamic_risk_budget(very_low_drawdown)
        self.assertLessEqual(budget, self.config.max_risk_budget)
        
        # 测试极端高回撤
        very_high_drawdown = 0.50  # 50%回撤
        budget = self.allocator.calculate_dynamic_risk_budget(very_high_drawdown)
        self.assertGreaterEqual(budget, self.config.min_risk_budget)
    
    def test_allocation_history_tracking(self):
        """测试分配历史跟踪"""
        initial_count = len(self.allocator.allocation_history)
        
        # 进行多次分配
        for method in [AllocationMethod.EQUAL_WEIGHT, AllocationMethod.RISK_PARITY]:
            self.allocator.allocate_risk_budget(
                self.asset_universe,
                self.asset_risk_metrics,
                allocation_method=method
            )
        
        # 检查历史记录增加
        self.assertEqual(len(self.allocator.allocation_history), initial_count + 2)
        
        # 检查最新分配方法
        latest_allocation = self.allocator.allocation_history[-1]
        self.assertEqual(latest_allocation.allocation_method, AllocationMethod.RISK_PARITY)


if __name__ == '__main__':
    unittest.main()