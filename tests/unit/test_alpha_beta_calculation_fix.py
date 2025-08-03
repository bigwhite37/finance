"""
Alpha和Beta计算修复测试

测试Alpha和Beta计算的数值稳定性问题。
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.rl_trading_system.metrics.portfolio_metrics import PortfolioMetricsCalculator


class TestAlphaBetaCalculationFix:
    """Alpha和Beta计算修复测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.calculator = PortfolioMetricsCalculator()
        
        # 创建正常的测试数据
        np.random.seed(42)
        self.normal_portfolio_returns = np.random.normal(0.001, 0.02, 252)  # 年化收益约25%，波动率约32%
        self.normal_benchmark_returns = np.random.normal(0.0008, 0.015, 252)  # 年化收益约20%，波动率约24%
    
    def test_alpha_beta_calculation_with_normal_data_should_be_reasonable(self):
        """测试正常数据的Alpha和Beta计算应该得到合理结果"""
        alpha, beta = self.calculator.calculate_alpha_beta(
            self.normal_portfolio_returns,
            self.normal_benchmark_returns,
            risk_free_rate=0.03
        )
        
        # Alpha和Beta应该在合理范围内
        assert -1.0 <= alpha <= 1.0, f"Alpha {alpha} 超出合理范围 [-1.0, 1.0]"
        assert -5.0 <= beta <= 5.0, f"Beta {beta} 超出合理范围 [-5.0, 5.0]"
        assert not np.isnan(alpha), f"Alpha不应该是NaN: {alpha}"
        assert not np.isnan(beta), f"Beta不应该是NaN: {beta}"
        assert not np.isinf(alpha), f"Alpha不应该是无穷大: {alpha}"
        assert not np.isinf(beta), f"Beta不应该是无穷大: {beta}"
    
    def test_alpha_beta_calculation_with_extreme_data_should_fail_gracefully(self):
        """测试极端数据的Alpha和Beta计算应该优雅地失败或返回合理值"""
        # 创建极端数据：非常小的方差
        extreme_portfolio_returns = np.array([0.000001] * 252)  # 几乎无变化
        extreme_benchmark_returns = np.array([0.000001] * 252)  # 几乎无变化
        
        alpha, beta = self.calculator.calculate_alpha_beta(
            extreme_portfolio_returns,
            extreme_benchmark_returns,
            risk_free_rate=0.03
        )
        
        # 即使是极端情况，也不应该返回异常大的数值
        assert abs(alpha) < 1e6, f"Alpha {alpha} 数值过大，可能存在计算错误"
        assert abs(beta) < 1e6, f"Beta {beta} 数值过大，可能存在计算错误"
        assert not np.isnan(alpha), f"Alpha不应该是NaN: {alpha}"
        assert not np.isnan(beta), f"Beta不应该是NaN: {beta}"
        assert not np.isinf(alpha), f"Alpha不应该是无穷大: {alpha}"
        assert not np.isinf(beta), f"Beta不应该是无穷大: {beta}"
    
    def test_alpha_beta_calculation_with_zero_benchmark_variance_should_handle_gracefully(self):
        """测试基准方差为零时的Alpha和Beta计算应该优雅处理"""
        portfolio_returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.zeros(252)  # 基准收益率为零（方差为零）
        
        alpha, beta = self.calculator.calculate_alpha_beta(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate=0.03
        )
        
        # 基准方差为零时，Beta应该为0，Alpha应该等于投资组合的超额收益
        assert beta == 0.0, f"基准方差为零时Beta应该为0，实际为: {beta}"
        assert not np.isnan(alpha), f"Alpha不应该是NaN: {alpha}"
        assert not np.isinf(alpha), f"Alpha不应该是无穷大: {alpha}"
        assert abs(alpha) < 10.0, f"Alpha {alpha} 数值过大"
    
    def test_alpha_beta_calculation_with_realistic_market_data_simulation(self):
        """测试模拟真实市场数据的Alpha和Beta计算"""
        # 模拟真实的市场数据特征
        # 投资组合：年化收益12%，年化波动率18%
        portfolio_daily_return = 0.12 / 252
        portfolio_daily_vol = 0.18 / np.sqrt(252)
        portfolio_returns = np.random.normal(portfolio_daily_return, portfolio_daily_vol, 252)
        
        # 基准：年化收益8%，年化波动率15%
        benchmark_daily_return = 0.08 / 252
        benchmark_daily_vol = 0.15 / np.sqrt(252)
        benchmark_returns = np.random.normal(benchmark_daily_return, benchmark_daily_vol, 252)
        
        # 添加一些相关性
        correlation = 0.7
        benchmark_returns = correlation * portfolio_returns + np.sqrt(1 - correlation**2) * benchmark_returns
        
        alpha, beta = self.calculator.calculate_alpha_beta(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate=0.03
        )
        
        # 验证结果在合理范围内
        assert -0.5 <= alpha <= 0.5, f"Alpha {alpha} 超出合理范围 [-0.5, 0.5]"
        assert 0.0 <= beta <= 3.0, f"Beta {beta} 超出合理范围 [0.0, 3.0]"
        assert not np.isnan(alpha), f"Alpha不应该是NaN: {alpha}"
        assert not np.isnan(beta), f"Beta不应该是NaN: {beta}"
        assert not np.isinf(alpha), f"Alpha不应该是无穷大: {alpha}"
        assert not np.isinf(beta), f"Beta不应该是无穷大: {beta}"
    
    def test_alpha_beta_calculation_with_problematic_data_that_causes_overflow(self):
        """测试可能导致数值溢出的问题数据"""
        # 创建可能导致计算问题的数据
        # 非常大的收益率变化
        portfolio_returns = np.array([0.1, -0.1, 0.15, -0.12, 0.08] * 50 + [0.001, 0.002])  # 252个数据点
        benchmark_returns = np.array([0.001, 0.002, 0.001, 0.002, 0.001] * 50 + [0.001, 0.002])  # 252个数据点
        
        alpha, beta = self.calculator.calculate_alpha_beta(
            portfolio_returns,
            benchmark_returns,
            risk_free_rate=0.03
        )
        
        # 即使数据有问题，也不应该返回异常大的数值
        assert abs(alpha) < 1e6, f"Alpha {alpha} 数值过大，可能存在计算错误"
        assert abs(beta) < 1e6, f"Beta {beta} 数值过大，可能存在计算错误"
        assert not np.isnan(alpha), f"Alpha不应该是NaN: {alpha}"
        assert not np.isnan(beta), f"Beta不应该是NaN: {beta}"
        assert not np.isinf(alpha), f"Alpha不应该是无穷大: {alpha}"
        assert not np.isinf(beta), f"Beta不应该是无穷大: {beta}"
    
    def test_alpha_beta_calculation_with_empty_data_should_raise_error(self):
        """测试空数据应该抛出错误而不是返回异常值"""
        with pytest.raises(RuntimeError, match="无法计算Alpha和Beta"):
            self.calculator.calculate_alpha_beta(
                np.array([]),
                np.array([]),
                risk_free_rate=0.03
            )
    
    def test_alpha_beta_calculation_with_mismatched_length_should_raise_error(self):
        """测试长度不匹配的数据应该抛出错误"""
        portfolio_returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.random.normal(0.0008, 0.015, 100)  # 不同长度
        
        with pytest.raises(RuntimeError, match="投资组合收益率和基准收益率长度不匹配"):
            self.calculator.calculate_alpha_beta(
                portfolio_returns,
                benchmark_returns,
                risk_free_rate=0.03
            )


if __name__ == '__main__':
    pytest.main([__file__])