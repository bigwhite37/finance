"""
Alpha和Beta修复集成测试

验证修复后的Alpha和Beta计算在实际使用场景中的表现。
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.rl_trading_system.metrics.portfolio_metrics import PortfolioMetricsCalculator


class TestAlphaBetaFixIntegration:
    """Alpha和Beta修复集成测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.calculator = PortfolioMetricsCalculator()
    
    def test_alpha_beta_with_realistic_trading_scenario(self):
        """测试真实交易场景下的Alpha和Beta计算"""
        # 模拟一年的交易数据
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # 投资组合：模拟一个表现良好的策略
        # 年化收益15%，年化波动率20%
        np.random.seed(42)
        portfolio_daily_returns = np.random.normal(0.15/252, 0.20/np.sqrt(252), 252)
        portfolio_values = [1000000]
        for ret in portfolio_daily_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        # 基准：模拟沪深300指数
        # 年化收益8%，年化波动率18%
        benchmark_daily_returns = np.random.normal(0.08/252, 0.18/np.sqrt(252), 252)
        benchmark_values = [1000000]
        for ret in benchmark_daily_returns:
            benchmark_values.append(benchmark_values[-1] * (1 + ret))
        
        # 计算投资组合指标
        metrics = self.calculator.calculate_portfolio_metrics(
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_values,
            dates=dates.tolist(),
            risk_free_rate=0.03
        )
        
        # 验证Alpha和Beta在合理范围内
        assert -1.0 <= metrics.alpha <= 1.0, f"Alpha {metrics.alpha} 超出合理范围"
        assert -3.0 <= metrics.beta <= 3.0, f"Beta {metrics.beta} 超出合理范围"
        assert not np.isnan(metrics.alpha), f"Alpha不应该是NaN: {metrics.alpha}"
        assert not np.isnan(metrics.beta), f"Beta不应该是NaN: {metrics.beta}"
        assert not np.isinf(metrics.alpha), f"Alpha不应该是无穷大: {metrics.alpha}"
        assert not np.isinf(metrics.beta), f"Beta不应该是无穷大: {metrics.beta}"
        
        # 验证其他指标也正常
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.annualized_return, float)
        assert metrics.max_drawdown >= 0
    
    def test_alpha_beta_with_extreme_market_conditions(self):
        """测试极端市场条件下的Alpha和Beta计算"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # 模拟极端市场条件：高波动率市场
        np.random.seed(123)
        
        # 投资组合：在高波动市场中的表现
        portfolio_returns = []
        for i in range(252):
            if i % 20 == 0:  # 每20天一次大幅波动
                ret = np.random.normal(0, 0.05)  # 5%的日波动
            else:
                ret = np.random.normal(0.0005, 0.015)  # 正常波动
            portfolio_returns.append(ret)
        
        portfolio_values = [1000000]
        for ret in portfolio_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        # 基准：相对稳定
        benchmark_returns = np.random.normal(0.0003, 0.01, 252)
        benchmark_values = [1000000]
        for ret in benchmark_returns:
            benchmark_values.append(benchmark_values[-1] * (1 + ret))
        
        # 计算指标
        metrics = self.calculator.calculate_portfolio_metrics(
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_values,
            dates=dates.tolist(),
            risk_free_rate=0.03
        )
        
        # 即使在极端条件下，Alpha和Beta也应该在合理范围内
        assert abs(metrics.alpha) < 5.0, f"Alpha {metrics.alpha} 在极端条件下仍然过大"
        assert abs(metrics.beta) < 10.0, f"Beta {metrics.beta} 在极端条件下仍然过大"
        assert not np.isnan(metrics.alpha), f"Alpha不应该是NaN: {metrics.alpha}"
        assert not np.isnan(metrics.beta), f"Beta不应该是NaN: {metrics.beta}"
        assert not np.isinf(metrics.alpha), f"Alpha不应该是无穷大: {metrics.alpha}"
        assert not np.isinf(metrics.beta), f"Beta不应该是无穷大: {metrics.beta}"
    
    def test_alpha_beta_with_low_correlation_scenario(self):
        """测试低相关性场景下的Alpha和Beta计算"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # 投资组合：与基准低相关性的策略
        np.random.seed(456)
        portfolio_returns = np.random.normal(0.0008, 0.025, 252)  # 独立的收益率序列
        
        # 基准：标准市场表现
        benchmark_returns = np.random.normal(0.0005, 0.015, 252)
        
        # 构建价值序列
        portfolio_values = [1000000]
        for ret in portfolio_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        benchmark_values = [1000000]
        for ret in benchmark_returns:
            benchmark_values.append(benchmark_values[-1] * (1 + ret))
        
        # 计算指标
        metrics = self.calculator.calculate_portfolio_metrics(
            portfolio_values=portfolio_values,
            benchmark_values=benchmark_values,
            dates=dates.tolist(),
            risk_free_rate=0.03
        )
        
        # 低相关性情况下，Beta应该接近0，Alpha应该反映策略的独立表现
        assert abs(metrics.beta) < 2.0, f"低相关性情况下Beta {metrics.beta} 应该较小"
        assert abs(metrics.alpha) < 2.0, f"Alpha {metrics.alpha} 应该在合理范围内"
        assert not np.isnan(metrics.alpha), f"Alpha不应该是NaN: {metrics.alpha}"
        assert not np.isnan(metrics.beta), f"Beta不应该是NaN: {metrics.beta}"
    
    def test_alpha_beta_calculation_stability_over_multiple_runs(self):
        """测试Alpha和Beta计算在多次运行中的稳定性"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        alphas = []
        betas = []
        
        # 运行多次计算，验证结果的稳定性
        for seed in range(10):
            np.random.seed(seed)
            
            # 生成相似但略有不同的数据
            portfolio_returns = np.random.normal(0.001, 0.02, 252)
            benchmark_returns = np.random.normal(0.0008, 0.015, 252)
            
            portfolio_values = [1000000]
            for ret in portfolio_returns:
                portfolio_values.append(portfolio_values[-1] * (1 + ret))
            
            benchmark_values = [1000000]
            for ret in benchmark_returns:
                benchmark_values.append(benchmark_values[-1] * (1 + ret))
            
            metrics = self.calculator.calculate_portfolio_metrics(
                portfolio_values=portfolio_values,
                benchmark_values=benchmark_values,
                dates=dates.tolist(),
                risk_free_rate=0.03
            )
            
            alphas.append(metrics.alpha)
            betas.append(metrics.beta)
        
        # 验证所有结果都在合理范围内
        for i, (alpha, beta) in enumerate(zip(alphas, betas)):
            assert abs(alpha) < 2.0, f"第{i+1}次运行Alpha {alpha} 超出合理范围"
            assert abs(beta) < 5.0, f"第{i+1}次运行Beta {beta} 超出合理范围"
            assert not np.isnan(alpha), f"第{i+1}次运行Alpha不应该是NaN: {alpha}"
            assert not np.isnan(beta), f"第{i+1}次运行Beta不应该是NaN: {beta}"
            assert not np.isinf(alpha), f"第{i+1}次运行Alpha不应该是无穷大: {alpha}"
            assert not np.isinf(beta), f"第{i+1}次运行Beta不应该是无穷大: {beta}"
        
        # 验证结果的变异性在合理范围内
        alpha_std = np.std(alphas)
        beta_std = np.std(betas)
        
        assert alpha_std < 1.0, f"Alpha标准差 {alpha_std} 过大，计算可能不稳定"
        assert beta_std < 2.0, f"Beta标准差 {beta_std} 过大，计算可能不稳定"


if __name__ == '__main__':
    pytest.main([__file__])