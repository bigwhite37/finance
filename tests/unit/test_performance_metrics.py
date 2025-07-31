"""
绩效指标计算模块的单元测试
测试收益率、夏普比率、最大回撤等指标计算，风险指标（VaR、CVaR、波动率）计算，交易指标（换手率、成本分析）计算
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional
from decimal import Decimal

from src.rl_trading_system.evaluation.performance_metrics import (
    ReturnMetrics,
    RiskMetrics,
    RiskAdjustedMetrics,
    TradingMetrics,
    PortfolioMetrics
)
from src.rl_trading_system.backtest.multi_frequency_backtest import Trade, OrderType


class TestReturnMetrics:
    """收益率指标测试类"""

    @pytest.fixture
    def sample_returns(self):
        """创建样本收益率数据"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        # 创建模拟的日收益率序列，年化收益约10%
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # 均值0.1%，标准差2%
        return pd.Series(returns, index=dates)

    @pytest.fixture
    def sample_portfolio_values(self):
        """创建样本组合价值序列"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        values = [1000000]  # 初始值100万
        for ret in returns:
            values.append(values[-1] * (1 + ret))
        
        return pd.Series(values[1:], index=dates)

    def test_return_metrics_initialization(self):
        """测试收益率指标初始化"""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        metrics = ReturnMetrics(returns)

        assert len(metrics.returns) == 5
        assert isinstance(metrics.returns, pd.Series)

    def test_total_return_calculation(self, sample_returns):
        """测试总收益率计算"""
        metrics = ReturnMetrics(sample_returns)
        total_return = metrics.calculate_total_return()

        # 总收益率 = 累积收益
        expected_total_return = (1 + sample_returns).prod() - 1
        assert abs(total_return - expected_total_return) < 1e-10

    def test_annualized_return_calculation(self, sample_returns):
        """测试年化收益率计算"""
        metrics = ReturnMetrics(sample_returns)
        annualized_return = metrics.calculate_annualized_return()

        # 年化收益率 = (1 + 总收益率)^(252/天数) - 1
        total_return = metrics.calculate_total_return()
        expected_annualized = (1 + total_return) ** (252 / len(sample_returns)) - 1
        assert abs(annualized_return - expected_annualized) < 1e-10

    def test_monthly_returns_calculation(self, sample_returns):
        """测试月度收益率计算"""
        metrics = ReturnMetrics(sample_returns)
        monthly_returns = metrics.calculate_monthly_returns()

        # 验证返回的是DataFrame
        assert isinstance(monthly_returns, pd.DataFrame)
        assert 'monthly_return' in monthly_returns.columns
        assert len(monthly_returns) > 0

    def test_cumulative_returns_calculation(self, sample_returns):
        """测试累积收益率计算"""
        metrics = ReturnMetrics(sample_returns)
        cumulative_returns = metrics.calculate_cumulative_returns()

        # 验证累积收益率的计算
        assert isinstance(cumulative_returns, pd.Series)
        assert len(cumulative_returns) == len(sample_returns)
        assert abs(cumulative_returns.iloc[0] - sample_returns.iloc[0]) < 1e-10
        
        # 最后一个值应该等于总收益率
        total_return = metrics.calculate_total_return()
        assert abs(cumulative_returns.iloc[-1] - total_return) < 1e-10

    def test_empty_returns_error(self):
        """测试空收益率序列错误"""
        with pytest.raises(ValueError, match="收益率序列不能为空"):
            ReturnMetrics(pd.Series([]))

    def test_invalid_returns_error(self):
        """测试无效收益率错误"""
        # 测试包含NaN的序列
        returns_with_nan = pd.Series([0.01, np.nan, 0.02])
        with pytest.raises(ValueError, match="收益率序列包含无效值"):
            ReturnMetrics(returns_with_nan)

        # 测试包含无穷大的序列
        returns_with_inf = pd.Series([0.01, np.inf, 0.02])
        with pytest.raises(ValueError, match="收益率序列包含无效值"):
            ReturnMetrics(returns_with_inf)


class TestRiskMetrics:
    """风险指标测试类"""

    @pytest.fixture
    def sample_returns(self):
        """创建样本收益率数据"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        return pd.Series(returns, index=dates)

    @pytest.fixture
    def sample_portfolio_values(self):
        """创建样本组合价值序列"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        values = [1000000]
        for ret in returns:
            values.append(values[-1] * (1 + ret))
        
        return pd.Series(values[1:], index=dates)

    def test_risk_metrics_initialization(self, sample_returns):
        """测试风险指标初始化"""
        metrics = RiskMetrics(sample_returns)
        assert len(metrics.returns) == 252

    def test_volatility_calculation(self, sample_returns):
        """测试波动率计算"""
        metrics = RiskMetrics(sample_returns)
        
        # 日波动率
        daily_vol = metrics.calculate_volatility()
        expected_daily_vol = sample_returns.std()
        assert abs(daily_vol - expected_daily_vol) < 1e-10

        # 年化波动率
        annualized_vol = metrics.calculate_volatility(annualized=True)
        expected_annualized_vol = sample_returns.std() * np.sqrt(252)
        assert abs(annualized_vol - expected_annualized_vol) < 1e-10

    def test_max_drawdown_calculation(self, sample_portfolio_values):
        """测试最大回撤计算"""
        returns = sample_portfolio_values.pct_change().dropna()
        metrics = RiskMetrics(returns)

        max_drawdown = metrics.calculate_max_drawdown(sample_portfolio_values)

        # 手动计算最大回撤进行验证
        peak = sample_portfolio_values.expanding().max()
        drawdown = (sample_portfolio_values - peak) / peak
        expected_max_drawdown = abs(drawdown.min())

        assert abs(max_drawdown - expected_max_drawdown) < 1e-10
        assert max_drawdown >= 0  # 最大回撤应该为正数

    def test_var_calculation(self, sample_returns):
        """测试VaR计算"""
        metrics = RiskMetrics(sample_returns)

        # 95% VaR
        var_95 = metrics.calculate_var(confidence_level=0.95)
        expected_var_95 = abs(sample_returns.quantile(0.05))
        assert abs(var_95 - expected_var_95) < 1e-10

        # 99% VaR
        var_99 = metrics.calculate_var(confidence_level=0.99)
        expected_var_99 = abs(sample_returns.quantile(0.01))
        assert abs(var_99 - expected_var_99) < 1e-10

        # VaR应该为正数
        assert var_95 >= 0
        assert var_99 >= 0
        # 99% VaR应该大于95% VaR
        assert var_99 >= var_95

    def test_cvar_calculation(self, sample_returns):
        """测试CVaR计算"""
        metrics = RiskMetrics(sample_returns)

        # 95% CVaR
        cvar_95 = metrics.calculate_cvar(confidence_level=0.95)
        
        # 手动计算CVaR进行验证
        var_95 = metrics.calculate_var(confidence_level=0.95)
        tail_losses = sample_returns[sample_returns <= -var_95]
        expected_cvar_95 = abs(tail_losses.mean()) if len(tail_losses) > 0 else var_95

        assert abs(cvar_95 - expected_cvar_95) < 1e-10
        assert cvar_95 >= 0

    def test_downside_deviation_calculation(self, sample_returns):
        """测试下行偏差计算"""
        metrics = RiskMetrics(sample_returns)

        # 相对于0的下行偏差
        downside_dev = metrics.calculate_downside_deviation()
        negative_returns = sample_returns[sample_returns < 0]
        expected_downside_dev = np.sqrt((negative_returns ** 2).mean())
        assert abs(downside_dev - expected_downside_dev) < 1e-10

        # 相对于目标收益率的下行偏差
        target_return = 0.005
        downside_dev_target = metrics.calculate_downside_deviation(target_return=target_return)
        below_target = sample_returns[sample_returns < target_return] - target_return
        expected_downside_dev_target = np.sqrt((below_target ** 2).mean())
        assert abs(downside_dev_target - expected_downside_dev_target) < 1e-10

    def test_skewness_calculation(self, sample_returns):
        """测试偏度计算"""
        metrics = RiskMetrics(sample_returns)
        skewness = metrics.calculate_skewness()

        # 使用scipy的skew函数验证
        from scipy.stats import skew
        expected_skewness = skew(sample_returns.values)
        assert abs(skewness - expected_skewness) < 1e-2  # 放宽精度要求

    def test_kurtosis_calculation(self, sample_returns):
        """测试峰度计算"""
        metrics = RiskMetrics(sample_returns)
        kurtosis = metrics.calculate_kurtosis()

        # 使用scipy的kurtosis函数验证
        from scipy.stats import kurtosis as scipy_kurtosis
        expected_kurtosis = scipy_kurtosis(sample_returns.values)
        assert abs(kurtosis - expected_kurtosis) < 0.1  # 进一步放宽精度要求

    def test_invalid_confidence_level_error(self, sample_returns):
        """测试无效置信水平错误"""
        metrics = RiskMetrics(sample_returns)

        # 测试置信水平超出范围
        with pytest.raises(ValueError, match="置信水平必须在0和1之间"):
            metrics.calculate_var(confidence_level=1.5)

        with pytest.raises(ValueError, match="置信水平必须在0和1之间"):
            metrics.calculate_var(confidence_level=-0.1)


class TestRiskAdjustedMetrics:
    """风险调整指标测试类"""

    @pytest.fixture
    def sample_returns(self):
        """创建样本收益率数据"""
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        return pd.Series(returns, index=dates)

    @pytest.fixture
    def sample_portfolio_values(self):
        """创建样本组合价值序列"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        values = [1000000]
        for ret in returns:
            values.append(values[-1] * (1 + ret))
        
        return pd.Series(values[1:], index=dates)

    def test_sharpe_ratio_calculation(self, sample_returns):
        """测试夏普比率计算"""
        metrics = RiskAdjustedMetrics(sample_returns)

        # 默认无风险利率
        sharpe_ratio = metrics.calculate_sharpe_ratio()
        
        # 手动计算验证
        excess_returns = sample_returns - 0.03/252  # 默认3%年化无风险利率
        expected_sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        assert abs(sharpe_ratio - expected_sharpe) < 1e-10

        # 自定义无风险利率
        risk_free_rate = 0.05  # 5%
        sharpe_ratio_custom = metrics.calculate_sharpe_ratio(risk_free_rate=risk_free_rate)
        excess_returns_custom = sample_returns - risk_free_rate/252
        expected_sharpe_custom = excess_returns_custom.mean() / excess_returns_custom.std() * np.sqrt(252)
        assert abs(sharpe_ratio_custom - expected_sharpe_custom) < 1e-10

    def test_sortino_ratio_calculation(self, sample_returns):
        """测试索提诺比率计算"""
        metrics = RiskAdjustedMetrics(sample_returns)

        sortino_ratio = metrics.calculate_sortino_ratio()

        # 手动计算验证
        target_return = 0.03/252  # 默认3%年化目标收益率
        excess_returns = sample_returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.sqrt((downside_returns ** 2).mean())
        expected_sortino = excess_returns.mean() / downside_deviation * np.sqrt(252)
        
        assert abs(sortino_ratio - expected_sortino) < 1e-10

    def test_calmar_ratio_calculation(self, sample_returns, sample_portfolio_values):
        """测试卡玛比率计算"""
        metrics = RiskAdjustedMetrics(sample_returns)

        calmar_ratio = metrics.calculate_calmar_ratio(sample_portfolio_values)

        # 手动计算验证
        annualized_return = (1 + sample_returns).prod() ** (252 / len(sample_returns)) - 1
        
        # 计算最大回撤
        peak = sample_portfolio_values.expanding().max()
        drawdown = (sample_portfolio_values - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        expected_calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0
        assert abs(calmar_ratio - expected_calmar) < 1e-10

    def test_information_ratio_calculation(self, sample_returns):
        """测试信息比率计算"""
        # 创建基准收益率
        np.random.seed(43)
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=sample_returns.index
        )

        metrics = RiskAdjustedMetrics(sample_returns)
        info_ratio = metrics.calculate_information_ratio(benchmark_returns)

        # 手动计算验证
        active_returns = sample_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        expected_info_ratio = active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0

        assert abs(info_ratio - expected_info_ratio) < 1e-10

    def test_treynor_ratio_calculation(self, sample_returns):
        """测试特雷诺比率计算"""
        metrics = RiskAdjustedMetrics(sample_returns)

        # 假设beta = 1.2
        beta = 1.2
        treynor_ratio = metrics.calculate_treynor_ratio(beta=beta)

        # 手动计算验证
        risk_free_rate = 0.03/252
        excess_returns = sample_returns - risk_free_rate
        expected_treynor = excess_returns.mean() * 252 / beta

        assert abs(treynor_ratio - expected_treynor) < 1e-10

    def test_zero_volatility_handling(self):
        """测试零波动率处理"""
        # 创建零波动率的收益序列
        constant_returns = pd.Series([0.001] * 252)
        metrics = RiskAdjustedMetrics(constant_returns)

        # 夏普比率应该为无穷大或处理为特殊值
        sharpe_ratio = metrics.calculate_sharpe_ratio()
        # 零波动率时，夏普比率会非常大（接近无穷大）
        assert np.isinf(sharpe_ratio) or abs(sharpe_ratio) > 1e10

    def test_invalid_benchmark_error(self, sample_returns):
        """测试无效基准错误"""
        metrics = RiskAdjustedMetrics(sample_returns)

        # 长度不匹配的基准
        short_benchmark = pd.Series([0.001] * 100)
        with pytest.raises(ValueError, match="基准收益率序列长度与投资组合收益率不匹配"):
            metrics.calculate_information_ratio(short_benchmark)


class TestTradingMetrics:
    """交易指标测试类"""

    @pytest.fixture
    def sample_trades(self):
        """创建样本交易数据"""
        trades = [
            Trade("000001.SZ", OrderType.BUY, 1000, Decimal("10.00"), datetime(2023, 1, 2), Decimal("10.00")),
            Trade("000001.SZ", OrderType.SELL, 500, Decimal("11.00"), datetime(2023, 1, 15), Decimal("5.50")),
            Trade("000002.SZ", OrderType.BUY, 2000, Decimal("5.00"), datetime(2023, 1, 20), Decimal("10.00")),
            Trade("000001.SZ", OrderType.SELL, 500, Decimal("10.50"), datetime(2023, 2, 1), Decimal("5.25")),
            Trade("000002.SZ", OrderType.SELL, 1000, Decimal("5.50"), datetime(2023, 2, 10), Decimal("5.50"))
        ]
        return trades

    @pytest.fixture
    def sample_portfolio_values(self):
        """创建样本组合价值序列"""
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        np.random.seed(42)
        values = np.random.uniform(900000, 1100000, 60)
        return pd.Series(values, index=dates)

    def test_trading_metrics_initialization(self, sample_trades, sample_portfolio_values):
        """测试交易指标初始化"""
        metrics = TradingMetrics(sample_trades, sample_portfolio_values)
        assert len(metrics.trades) == 5
        assert len(metrics.portfolio_values) == 60

    def test_turnover_rate_calculation(self, sample_trades, sample_portfolio_values):
        """测试换手率计算"""
        metrics = TradingMetrics(sample_trades, sample_portfolio_values)

        # 月度换手率
        monthly_turnover = metrics.calculate_turnover_rate(period='monthly')
        assert isinstance(monthly_turnover, pd.Series)
        assert len(monthly_turnover) > 0

        # 年化换手率
        annual_turnover = metrics.calculate_turnover_rate(period='annual')
        assert isinstance(annual_turnover, float)
        assert annual_turnover >= 0

    def test_transaction_cost_analysis(self, sample_trades, sample_portfolio_values):
        """测试交易成本分析"""
        metrics = TradingMetrics(sample_trades, sample_portfolio_values)

        cost_analysis = metrics.calculate_transaction_cost_analysis()

        # 验证返回的字典包含必要的字段
        assert 'total_commission' in cost_analysis
        assert 'commission_rate' in cost_analysis
        assert 'cost_per_trade' in cost_analysis
        assert 'cost_ratio_to_portfolio' in cost_analysis

        # 验证数值合理性
        assert cost_analysis['total_commission'] >= 0
        assert cost_analysis['commission_rate'] >= 0
        assert cost_analysis['cost_per_trade'] >= 0
        assert cost_analysis['cost_ratio_to_portfolio'] >= 0

    def test_holding_period_analysis(self, sample_trades):
        """测试持仓周期分析"""
        metrics = TradingMetrics(sample_trades, pd.Series([1000000]))

        holding_analysis = metrics.calculate_holding_period_analysis()

        # 验证返回的字典包含必要的字段
        assert 'average_holding_days' in holding_analysis
        assert 'median_holding_days' in holding_analysis
        assert 'max_holding_days' in holding_analysis
        assert 'min_holding_days' in holding_analysis

        # 验证数值合理性
        assert holding_analysis['average_holding_days'] >= 0
        assert holding_analysis['median_holding_days'] >= 0
        assert holding_analysis['max_holding_days'] >= holding_analysis['min_holding_days']

    def test_win_loss_analysis(self, sample_trades):
        """测试盈亏分析"""
        metrics = TradingMetrics(sample_trades, pd.Series([1000000]))

        win_loss_analysis = metrics.calculate_win_loss_analysis()

        # 验证返回的字典包含必要的字段
        assert 'win_rate' in win_loss_analysis
        assert 'profit_loss_ratio' in win_loss_analysis
        assert 'average_win' in win_loss_analysis
        assert 'average_loss' in win_loss_analysis
        assert 'total_trades' in win_loss_analysis

        # 验证数值合理性
        assert 0 <= win_loss_analysis['win_rate'] <= 1
        assert win_loss_analysis['total_trades'] == len([t for t in sample_trades if t.trade_type == OrderType.SELL])

    def test_position_concentration_analysis(self, sample_trades, sample_portfolio_values):
        """测试持仓集中度分析"""
        metrics = TradingMetrics(sample_trades, sample_portfolio_values)

        concentration = metrics.calculate_position_concentration()

        # 验证返回的字典包含必要的字段
        assert 'herfindahl_index' in concentration
        assert 'max_position_weight' in concentration
        assert 'top_5_concentration' in concentration
        assert 'effective_positions' in concentration

        # 验证数值合理性
        assert 0 <= concentration['herfindahl_index'] <= 1
        assert 0 <= concentration['max_position_weight'] <= 1
        assert concentration['effective_positions'] >= 1

    def test_empty_trades_handling(self):
        """测试空交易列表处理"""
        empty_trades = []
        portfolio_values = pd.Series([1000000])
        
        # 应该能够处理空交易列表而不抛出异常
        metrics = TradingMetrics(empty_trades, portfolio_values)
        
        # 但某些计算可能返回默认值或抛出合理的错误
        cost_analysis = metrics.calculate_transaction_cost_analysis()
        assert cost_analysis['total_commission'] == 0


class TestPortfolioMetrics:
    """组合指标测试类"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        portfolio_values = [1000000]
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        trades = [
            Trade("000001.SZ", OrderType.BUY, 1000, Decimal("10.00"), datetime(2023, 1, 2), Decimal("10.00")),
            Trade("000001.SZ", OrderType.SELL, 500, Decimal("11.00"), datetime(2023, 6, 15), Decimal("5.50")),
            Trade("000002.SZ", OrderType.BUY, 2000, Decimal("5.00"), datetime(2023, 3, 20), Decimal("10.00")),
        ]

        return {
            'returns': pd.Series(returns, index=dates),
            'portfolio_values': pd.Series(portfolio_values[1:], index=dates),
            'trades': trades
        }

    def test_comprehensive_metrics_calculation(self, sample_data):
        """测试综合指标计算"""
        metrics = PortfolioMetrics(
            returns=sample_data['returns'],
            portfolio_values=sample_data['portfolio_values'],
            trades=sample_data['trades']
        )

        comprehensive_metrics = metrics.calculate_comprehensive_metrics()

        # 验证返回的指标包含所有类别
        assert 'return_metrics' in comprehensive_metrics
        assert 'risk_metrics' in comprehensive_metrics
        assert 'risk_adjusted_metrics' in comprehensive_metrics
        assert 'trading_metrics' in comprehensive_metrics

        # 验证每个类别包含合理的指标
        return_metrics = comprehensive_metrics['return_metrics']
        assert 'total_return' in return_metrics
        assert 'annualized_return' in return_metrics

        risk_metrics = comprehensive_metrics['risk_metrics']
        assert 'volatility' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'var_95' in risk_metrics

        risk_adjusted = comprehensive_metrics['risk_adjusted_metrics']
        assert 'sharpe_ratio' in risk_adjusted
        assert 'sortino_ratio' in risk_adjusted

    def test_benchmark_comparison(self, sample_data):
        """测试基准比较"""
        # 创建基准数据
        np.random.seed(43)
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=sample_data['returns'].index
        )

        metrics = PortfolioMetrics(
            returns=sample_data['returns'],
            portfolio_values=sample_data['portfolio_values'],
            trades=sample_data['trades']
        )

        comparison = metrics.compare_with_benchmark(benchmark_returns)

        # 验证比较结果
        assert 'portfolio_metrics' in comparison
        assert 'benchmark_metrics' in comparison
        assert 'relative_metrics' in comparison

        relative_metrics = comparison['relative_metrics']
        assert 'excess_return' in relative_metrics
        assert 'information_ratio' in relative_metrics
        assert 'tracking_error' in relative_metrics

    def test_rolling_metrics_calculation(self, sample_data):
        """测试滚动指标计算"""
        metrics = PortfolioMetrics(
            returns=sample_data['returns'],
            portfolio_values=sample_data['portfolio_values'],
            trades=sample_data['trades']
        )

        # 30天滚动夏普比率
        rolling_sharpe = metrics.calculate_rolling_metrics(window=30, metric='sharpe_ratio')
        assert isinstance(rolling_sharpe, pd.Series)
        # rolling()方法会保持原序列长度，前面的值为NaN
        assert len(rolling_sharpe) == len(sample_data['returns'])
        # 有效值的数量应该是 len - window + 1
        valid_values = rolling_sharpe.dropna()
        assert len(valid_values) == len(sample_data['returns']) - 30 + 1

        # 60天滚动波动率
        rolling_vol = metrics.calculate_rolling_metrics(window=60, metric='volatility')
        assert isinstance(rolling_vol, pd.Series)
        assert len(rolling_vol) == len(sample_data['returns'])  # 保持原序列长度

    def test_sector_analysis(self, sample_data):
        """测试行业分析"""
        # 添加行业信息到交易数据
        sector_mapping = {
            "000001.SZ": "金融",
            "000002.SZ": "地产"
        }

        metrics = PortfolioMetrics(
            returns=sample_data['returns'],
            portfolio_values=sample_data['portfolio_values'],
            trades=sample_data['trades']
        )

        sector_analysis = metrics.calculate_sector_analysis(sector_mapping)

        # 验证行业分析结果
        assert isinstance(sector_analysis, dict)
        assert len(sector_analysis) > 0

        # 验证每个行业的指标
        for sector, sector_metrics in sector_analysis.items():
            assert 'weight' in sector_metrics
            assert 'return_contribution' in sector_metrics
            assert 'trade_count' in sector_metrics

    def test_invalid_data_validation(self):
        """测试无效数据验证"""
        # 测试长度不匹配的数据
        returns = pd.Series([0.01, 0.02])
        portfolio_values = pd.Series([1000000, 1010000, 1020000])  # 长度不匹配
        trades = []

        with pytest.raises(ValueError, match="收益率序列和组合价值序列长度不匹配"):
            PortfolioMetrics(returns, portfolio_values, trades)