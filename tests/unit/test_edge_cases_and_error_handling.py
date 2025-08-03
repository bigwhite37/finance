"""
边界条件和异常处理测试

该模块专门测试回撤控制系统的边界条件、异常情况和错误处理，确保系统在各种极端情况下的稳定性。
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings

from src.rl_trading_system.risk_control.drawdown_monitor import DrawdownMonitor
from src.rl_trading_system.risk_control.drawdown_attribution_analyzer import DrawdownAttributionAnalyzer
from src.rl_trading_system.risk_control.dynamic_stop_loss import DynamicStopLoss
from src.rl_trading_system.risk_control.market_regime_detector import MarketRegimeDetector
from src.rl_trading_system.risk_control.adaptive_risk_budget import (
    AdaptiveRiskBudget, AdaptiveRiskBudgetConfig, PerformanceMetrics, MarketMetrics
)


class TestDrawdownMonitorEdgeCases:
    """回撤监控器边界条件测试"""
    
    @pytest.fixture
    def monitor(self):
        """创建回撤监控器"""
        return DrawdownMonitor(
            max_drawdown_threshold=0.20,
            recovery_threshold=0.05,
            monitoring_window=30
        )
    
    def test_empty_portfolio_values(self, monitor):
        """测试空投资组合数值"""
        with pytest.raises(ValueError, match="投资组合净值序列不能为空"):
            monitor.update_portfolio_value([])
    
    def test_single_value_portfolio(self, monitor):
        """测试单一数值投资组合"""
        result = monitor.update_portfolio_value([100.0])
        assert result['current_drawdown'] == 0.0
        assert result['max_drawdown'] == 0.0
    
    def test_negative_portfolio_values(self, monitor):
        """测试负数投资组合净值"""
        # 应该能处理负值，但会发出警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = monitor.update_portfolio_value([-100.0, -50.0, -25.0])
            assert len(w) > 0
            assert "负数净值" in str(w[0].message)
    
    def test_zero_portfolio_values(self, monitor):
        """测试零值投资组合"""
        with pytest.raises(ValueError, match="投资组合净值不能为零或负数"):
            monitor.update_portfolio_value([100.0, 0.0, 50.0])
    
    def test_nan_portfolio_values(self, monitor):
        """测试NaN值投资组合"""
        with pytest.raises(ValueError, match="投资组合净值包含无效数据"):
            monitor.update_portfolio_value([100.0, np.nan, 50.0])
    
    def test_inf_portfolio_values(self, monitor):
        """测试无穷大值投资组合"""
        with pytest.raises(ValueError, match="投资组合净值包含无效数据"):
            monitor.update_portfolio_value([100.0, np.inf, 50.0])
    
    def test_extreme_large_values(self, monitor):
        """测试极大数值"""
        large_values = [1e15, 1.1e15, 0.9e15]
        result = monitor.update_portfolio_value(large_values)
        assert np.isfinite(result['current_drawdown'])
        assert np.isfinite(result['max_drawdown'])
    
    def test_extreme_small_values(self, monitor):
        """测试极小数值"""
        small_values = [1e-10, 1.1e-10, 0.9e-10]
        result = monitor.update_portfolio_value(small_values)
        assert np.isfinite(result['current_drawdown'])
        assert np.isfinite(result['max_drawdown'])
    
    def test_identical_values(self, monitor):
        """测试相同数值序列"""
        identical_values = [100.0] * 100
        result = monitor.update_portfolio_value(identical_values)
        assert result['current_drawdown'] == 0.0
        assert result['max_drawdown'] == 0.0
    
    def test_monotonic_decreasing(self, monitor):
        """测试单调递减序列"""
        decreasing_values = [100.0 - i for i in range(50)]
        result = monitor.update_portfolio_value(decreasing_values)
        assert result['current_drawdown'] < -0.4  # 应该有显著回撤
        assert result['max_drawdown'] < -0.4
    
    def test_rapid_oscillation(self, monitor):
        """测试快速震荡序列"""
        oscillating_values = [100.0 + 10 * np.sin(i) for i in range(100)]
        result = monitor.update_portfolio_value(oscillating_values)
        assert np.isfinite(result['current_drawdown'])
        assert result['volatility'] > 0.05  # 应该检测到高波动性


class TestDrawdownAttributionAnalyzerEdgeCases:
    """回撤归因分析器边界条件测试"""
    
    @pytest.fixture
    def analyzer(self):
        """创建归因分析器"""
        return DrawdownAttributionAnalyzer()
    
    def test_empty_portfolio_data(self, analyzer):
        """测试空投资组合数据"""
        with pytest.raises(ValueError, match="投资组合数据不能为空"):
            analyzer.analyze_drawdown_attribution(
                portfolio_returns=pd.Series([]),
                holdings=pd.DataFrame(),
                benchmark_returns=pd.Series([])
            )
    
    def test_mismatched_data_lengths(self, analyzer):
        """测试数据长度不匹配"""
        portfolio_returns = pd.Series([0.01, 0.02, 0.03])
        holdings = pd.DataFrame({'A': [0.5, 0.6], 'B': [0.5, 0.4]})  # 长度不匹配
        benchmark_returns = pd.Series([0.01, 0.015, 0.02])
        
        with pytest.raises(ValueError, match="数据长度不匹配"):
            analyzer.analyze_drawdown_attribution(
                portfolio_returns=portfolio_returns,
                holdings=holdings,
                benchmark_returns=benchmark_returns
            )
    
    def test_negative_holdings(self, analyzer):
        """测试负持仓（做空）"""
        portfolio_returns = pd.Series([0.01, -0.02, 0.03])
        holdings = pd.DataFrame({
            'A': [0.6, 0.8, 0.7],
            'B': [-0.1, -0.3, -0.2]  # 负持仓（做空）
        })
        benchmark_returns = pd.Series([0.01, 0.00, 0.02])
        
        # 应该能处理负持仓
        result = analyzer.analyze_drawdown_attribution(
            portfolio_returns=portfolio_returns,
            holdings=holdings,
            benchmark_returns=benchmark_returns
        )
        assert 'individual_contributions' in result
        assert 'B' in result['individual_contributions']
    
    def test_holdings_not_sum_to_one(self, analyzer):
        """测试持仓比例不等于1"""
        portfolio_returns = pd.Series([0.01, -0.02, 0.03])
        holdings = pd.DataFrame({
            'A': [0.3, 0.4, 0.5],
            'B': [0.2, 0.3, 0.2]  # 总和不等于1
        })
        benchmark_returns = pd.Series([0.01, 0.00, 0.02])
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = analyzer.analyze_drawdown_attribution(
                portfolio_returns=portfolio_returns,
                holdings=holdings,
                benchmark_returns=benchmark_returns
            )
            # 应该发出警告但仍能运行
            assert any("持仓比例之和" in str(warning.message) for warning in w)
    
    def test_extreme_returns(self, analyzer):
        """测试极端收益率"""
        # 极端正收益和极端负收益
        portfolio_returns = pd.Series([0.5, -0.8, 2.0])  # 极端收益率
        holdings = pd.DataFrame({
            'A': [0.5, 0.5, 0.5],
            'B': [0.5, 0.5, 0.5]
        })
        benchmark_returns = pd.Series([0.01, 0.00, 0.02])
        
        result = analyzer.analyze_drawdown_attribution(
            portfolio_returns=portfolio_returns,
            holdings=holdings,
            benchmark_returns=benchmark_returns
        )
        
        # 结果应该是有限的
        assert np.isfinite(result['total_attribution'])
        assert all(np.isfinite(contrib) for contrib in result['individual_contributions'].values())


class TestDynamicStopLossEdgeCases:
    """动态止损控制器边界条件测试"""
    
    @pytest.fixture
    def stop_loss(self):
        """创建动态止损控制器"""
        return DynamicStopLoss(
            initial_stop_loss=0.05,
            trailing_stop_ratio=0.02,
            volatility_adjustment=True
        )
    
    def test_negative_initial_stop_loss(self):
        """测试负数初始止损"""
        with pytest.raises(ValueError, match="初始止损比例必须为正数"):
            DynamicStopLoss(initial_stop_loss=-0.05)
    
    def test_zero_stop_loss(self):
        """测试零止损"""
        with pytest.raises(ValueError, match="初始止损比例必须为正数"):
            DynamicStopLoss(initial_stop_loss=0.0)
    
    def test_extreme_large_stop_loss(self):
        """测试极大止损比例"""
        with pytest.raises(ValueError, match="止损比例不能超过"):
            DynamicStopLoss(initial_stop_loss=1.5)  # 150%止损不合理
    
    def test_empty_price_history(self, stop_loss):
        """测试空价格历史"""
        with pytest.raises(ValueError, match="价格历史不能为空"):
            stop_loss.update_stop_loss([])
    
    def test_single_price_point(self, stop_loss):
        """测试单一价格点"""
        result = stop_loss.update_stop_loss([100.0])
        assert result['stop_loss_price'] == 95.0  # 100 * (1 - 0.05)
        assert not result['stop_triggered']
    
    def test_nan_prices(self, stop_loss):
        """测试NaN价格"""
        with pytest.raises(ValueError, match="价格数据包含无效值"):
            stop_loss.update_stop_loss([100.0, np.nan, 105.0])
    
    def test_negative_prices(self, stop_loss):
        """测试负数价格"""
        with pytest.raises(ValueError, match="价格不能为负数"):
            stop_loss.update_stop_loss([100.0, -50.0, 105.0])
    
    def test_zero_prices(self, stop_loss):
        """测试零价格"""
        with pytest.raises(ValueError, match="价格不能为零"):
            stop_loss.update_stop_loss([100.0, 0.0, 105.0])
    
    def test_extreme_price_volatility(self, stop_loss):
        """测试极端价格波动"""
        # 极端波动的价格序列
        extreme_prices = [100, 200, 50, 300, 25, 400]
        result = stop_loss.update_stop_loss(extreme_prices)
        
        # 应该能处理极端波动
        assert np.isfinite(result['stop_loss_price'])
        assert result['volatility'] > 1.0  # 应该检测到极高波动性
    
    def test_stop_loss_with_gaps(self, stop_loss):
        """测试有跳空的价格序列"""
        # 模拟跳空（价格突然大幅变化）
        gap_prices = [100, 101, 102, 150, 151, 152]  # 有向上跳空
        result = stop_loss.update_stop_loss(gap_prices)
        
        assert np.isfinite(result['stop_loss_price'])
        # 追踪止损应该适应价格跳空


class TestMarketRegimeDetectorEdgeCases:
    """市场制度检测器边界条件测试"""
    
    @pytest.fixture
    def detector(self):
        """创建市场制度检测器"""
        return MarketRegimeDetector(
            lookback_window=20,
            volatility_threshold=0.02,
            trend_threshold=0.01
        )
    
    def test_insufficient_data(self, detector):
        """测试数据不足"""
        short_returns = pd.Series([0.01, 0.02])  # 少于lookback_window
        
        with pytest.raises(ValueError, match="数据量不足"):
            detector.detect_regime(short_returns)
    
    def test_all_zero_returns(self, detector):
        """测试全零收益率"""
        zero_returns = pd.Series([0.0] * 25)
        result = detector.detect_regime(zero_returns)
        
        assert result['regime'] == 'low_volatility'  # 应该识别为低波动制度
        assert result['volatility'] == 0.0
    
    def test_constant_positive_returns(self, detector):
        """测试恒定正收益率"""
        constant_returns = pd.Series([0.01] * 25)
        result = detector.detect_regime(constant_returns)
        
        assert result['trend'] > 0
        assert result['regime'] in ['bull_market', 'trending_up']
    
    def test_constant_negative_returns(self, detector):
        """测试恒定负收益率"""
        constant_returns = pd.Series([-0.01] * 25)
        result = detector.detect_regime(constant_returns)
        
        assert result['trend'] < 0
        assert result['regime'] in ['bear_market', 'trending_down']
    
    def test_extreme_outliers(self, detector):
        """测试极端异常值"""
        returns_with_outliers = pd.Series([0.001] * 20 + [0.5, -0.6, 0.001, 0.001, 0.001])
        result = detector.detect_regime(returns_with_outliers)
        
        # 应该检测到高波动制度
        assert result['volatility'] > 0.1
        assert 'high_volatility' in result['regime']
    
    def test_missing_data_handling(self, detector):
        """测试缺失数据处理"""
        returns_with_nan = pd.Series([0.01, 0.02, np.nan, 0.015, -0.005] + [0.001] * 20)
        
        # 应该能处理NaN值（通过清理或插值）
        result = detector.detect_regime(returns_with_nan)
        assert 'regime' in result
        assert np.isfinite(result['volatility'])


class TestAdaptiveRiskBudgetEdgeCases:
    """自适应风险预算边界条件测试"""
    
    @pytest.fixture
    def risk_budget(self):
        """创建自适应风险预算实例"""
        config = AdaptiveRiskBudgetConfig(
            base_risk_budget=0.10,
            min_risk_budget=0.01,
            max_risk_budget=0.30
        )
        return AdaptiveRiskBudget(config)
    
    def test_invalid_config_parameters(self):
        """测试无效配置参数"""
        # 最小风险预算大于基础风险预算
        with pytest.raises(ValueError, match="最小风险预算不能大于基础风险预算"):
            AdaptiveRiskBudgetConfig(
                base_risk_budget=0.05,
                min_risk_budget=0.10,
                max_risk_budget=0.20
            )
        
        # 基础风险预算大于最大风险预算
        with pytest.raises(ValueError, match="基础风险预算不能大于最大风险预算"):
            AdaptiveRiskBudgetConfig(
                base_risk_budget=0.25,
                min_risk_budget=0.01,
                max_risk_budget=0.20
            )
        
        # 负数风险预算
        with pytest.raises(ValueError, match="风险预算不能为负数"):
            AdaptiveRiskBudgetConfig(
                base_risk_budget=-0.05,
                min_risk_budget=0.01,
                max_risk_budget=0.20
            )
    
    def test_extreme_performance_metrics(self, risk_budget):
        """测试极端表现指标"""
        # 极端夏普比率
        extreme_metrics = PerformanceMetrics(
            sharpe_ratio=100.0,  # 不现实的高夏普比率
            return_rate=10.0,
            volatility=0.001,
            max_drawdown=0.0,
            consecutive_losses=0
        )
        
        risk_budget.update_performance_metrics(extreme_metrics)
        budget = risk_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 应该被限制在合理范围内
        assert risk_budget.config.min_risk_budget <= budget <= risk_budget.config.max_risk_budget
    
    def test_extreme_market_metrics(self, risk_budget):
        """测试极端市场指标"""
        extreme_market = MarketMetrics(
            market_volatility=5.0,  # 500%波动率
            market_trend=2.0,       # 200%趋势
            uncertainty_index=10.0,  # 超出正常范围
            correlation_breakdown=True
        )
        
        risk_budget.update_market_metrics(extreme_market)
        budget = risk_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 应该被限制在合理范围内
        assert risk_budget.config.min_risk_budget <= budget <= risk_budget.config.max_risk_budget
    
    def test_rapid_successive_updates(self, risk_budget):
        """测试快速连续更新"""
        # 模拟快速连续的更新
        for i in range(100):
            metrics = PerformanceMetrics(
                sharpe_ratio=np.random.normal(1.0, 0.5),
                consecutive_losses=np.random.randint(0, 10)
            )
            risk_budget.update_performance_metrics(metrics)
            
            # 每次都强制更新
            budget = risk_budget.calculate_adaptive_risk_budget(force_update=True)
            
            # 预算应该始终在有效范围内
            assert risk_budget.config.min_risk_budget <= budget <= risk_budget.config.max_risk_budget
    
    def test_memory_overflow_protection(self, risk_budget):
        """测试内存溢出保护"""
        # 添加大量历史数据测试内存限制
        for i in range(1000):  # 远超过历史限制
            metrics = PerformanceMetrics(sharpe_ratio=0.5 + i * 0.001)
            risk_budget.update_performance_metrics(metrics)
        
        # 历史数据应该被限制
        assert len(risk_budget.performance_history) <= 100  # 假设限制为100
        
        # 最新数据应该被保留
        assert risk_budget.performance_history[-1].sharpe_ratio > 1.0


class TestErrorHandlingAndRecovery:
    """错误处理和恢复测试"""
    
    def test_system_recovery_after_exception(self):
        """测试异常后的系统恢复"""
        monitor = DrawdownMonitor()
        
        # 模拟异常
        try:
            monitor.update_portfolio_value([])  # 会抛出异常
        except ValueError:
            pass
        
        # 系统应该能在异常后正常工作
        result = monitor.update_portfolio_value([100.0, 95.0, 105.0])
        assert 'current_drawdown' in result
    
    def test_concurrent_access_safety(self):
        """测试并发访问安全性"""
        import threading
        import time
        
        monitor = DrawdownMonitor()
        errors = []
        
        def update_worker():
            try:
                for i in range(50):
                    values = [100 + i, 105 + i, 95 + i]
                    monitor.update_portfolio_value(values)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # 启动多个并发线程
        threads = [threading.Thread(target=update_worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 不应该有并发错误
        assert len(errors) == 0
    
    def test_resource_cleanup_on_failure(self):
        """测试失败时的资源清理"""
        detector = MarketRegimeDetector()
        
        # 模拟资源分配
        original_memory_usage = detector.__sizeof__()
        
        # 尝试处理无效数据
        try:
            detector.detect_regime(pd.Series([np.inf, np.nan, -np.inf]))
        except (ValueError, RuntimeError):
            pass
        
        # 资源应该被正确清理（简化测试）
        assert detector.__sizeof__() <= original_memory_usage * 1.1  # 允许10%的内存增长
    
    @patch('logging.Logger.error')
    def test_error_logging(self, mock_logger):
        """测试错误日志记录"""
        analyzer = DrawdownAttributionAnalyzer()
        
        # 触发错误
        try:
            analyzer.analyze_drawdown_attribution(
                portfolio_returns=pd.Series([]),
                holdings=pd.DataFrame(),
                benchmark_returns=pd.Series([])
            )
        except ValueError:
            pass
        
        # 应该记录错误日志
        assert mock_logger.called
    
    def test_graceful_degradation(self):
        """测试优雅降级"""
        # 当某些功能不可用时，系统应该优雅降级
        risk_budget = AdaptiveRiskBudget(AdaptiveRiskBudgetConfig())
        
        # 模拟某些计算失败的情况
        with patch.object(risk_budget, '_calculate_performance_factor', side_effect=Exception("计算失败")):
            # 系统应该回退到默认行为
            budget = risk_budget.calculate_adaptive_risk_budget()
            
            # 应该返回基础预算或安全值
            assert budget == risk_budget.config.base_risk_budget or budget == risk_budget.config.min_risk_budget


if __name__ == "__main__":
    pytest.main([__file__, "-v"])