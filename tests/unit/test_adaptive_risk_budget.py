"""自适应风险预算系统单元测试"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from dataclasses import dataclass

from src.rl_trading_system.risk_control.adaptive_risk_budget import (
    AdaptiveRiskBudget,
    AdaptiveRiskBudgetConfig,
    PerformanceMetrics,
    MarketMetrics,
    MarketCondition,
    PerformanceRegime,
    RiskBudgetAdjustment
)


class TestAdaptiveRiskBudgetConfig:
    """测试自适应风险预算配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = AdaptiveRiskBudgetConfig()
        
        assert config.base_risk_budget == 0.10
        assert config.min_risk_budget == 0.02
        assert config.max_risk_budget == 0.25
        assert config.performance_lookback_days == 60
        assert config.market_lookback_days == 30
        assert config.smoothing_factor == 0.1
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = AdaptiveRiskBudgetConfig(
            base_risk_budget=0.15,
            min_risk_budget=0.05,
            max_risk_budget=0.30,
            performance_lookback_days=90
        )
        
        assert config.base_risk_budget == 0.15
        assert config.min_risk_budget == 0.05
        assert config.max_risk_budget == 0.30
        assert config.performance_lookback_days == 90


class TestPerformanceMetrics:
    """测试表现指标"""
    
    def test_performance_metrics_creation(self):
        """测试表现指标创建"""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown=0.1,
            volatility=0.15,
            win_rate=0.6,
            consecutive_losses=2
        )
        
        assert metrics.sharpe_ratio == 1.5
        assert metrics.calmar_ratio == 2.0
        assert metrics.max_drawdown == 0.1
        assert metrics.volatility == 0.15
        assert metrics.win_rate == 0.6
        assert metrics.consecutive_losses == 2
        assert isinstance(metrics.timestamp, datetime)


class TestMarketMetrics:
    """测试市场指标"""
    
    def test_market_metrics_creation(self):
        """测试市场指标创建"""
        metrics = MarketMetrics(
            market_volatility=0.2,
            market_trend=0.05,
            correlation_with_market=0.7,
            uncertainty_index=0.3
        )
        
        assert metrics.market_volatility == 0.2
        assert metrics.market_trend == 0.05
        assert metrics.correlation_with_market == 0.7
        assert metrics.uncertainty_index == 0.3
        assert isinstance(metrics.timestamp, datetime)


class TestAdaptiveRiskBudget:
    """测试自适应风险预算系统"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return AdaptiveRiskBudgetConfig(
            base_risk_budget=0.10,
            min_risk_budget=0.02,
            max_risk_budget=0.25,
            performance_lookback_days=30,
            market_lookback_days=20,
            smoothing_factor=0.2,
            max_daily_change=0.1
        )
    
    @pytest.fixture
    def adaptive_budget(self, config):
        """自适应风险预算实例"""
        return AdaptiveRiskBudget(config)
    
    def test_initialization(self, adaptive_budget, config):
        """测试初始化"""
        assert adaptive_budget.config == config
        assert adaptive_budget.current_risk_budget == config.base_risk_budget
        assert adaptive_budget.smoothed_risk_budget == config.base_risk_budget
        assert adaptive_budget.current_performance_regime == PerformanceRegime.AVERAGE
        assert adaptive_budget.current_market_condition == MarketCondition.SIDEWAYS
        assert len(adaptive_budget.performance_history) == 0
        assert len(adaptive_budget.market_history) == 0
    
    def test_update_performance_metrics(self, adaptive_budget):
        """测试更新表现指标"""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            calmar_ratio=2.0,
            max_drawdown=0.08,
            volatility=0.12
        )
        
        adaptive_budget.update_performance_metrics(metrics)
        
        assert len(adaptive_budget.performance_history) == 1
        assert adaptive_budget.performance_history[-1] == metrics
        assert adaptive_budget.current_performance_regime == PerformanceRegime.GOOD
        assert adaptive_budget._performance_cache == metrics
    
    def test_update_market_metrics(self, adaptive_budget):
        """测试更新市场指标"""
        metrics = MarketMetrics(
            market_volatility=0.15,
            market_trend=0.05,
            uncertainty_index=0.3
        )
        
        adaptive_budget.update_market_metrics(metrics)
        
        assert len(adaptive_budget.market_history) == 1
        assert adaptive_budget.market_history[-1] == metrics
        assert adaptive_budget.current_market_condition == MarketCondition.SIDEWAYS
    
    def test_classify_performance_regime(self, adaptive_budget):
        """测试表现状态分类"""
        # 优秀表现
        metrics = PerformanceMetrics(sharpe_ratio=2.5)
        regime = adaptive_budget._classify_performance_regime(metrics)
        assert regime == PerformanceRegime.EXCELLENT
        
        # 良好表现
        metrics = PerformanceMetrics(sharpe_ratio=1.2)
        regime = adaptive_budget._classify_performance_regime(metrics)
        assert regime == PerformanceRegime.GOOD
        
        # 一般表现
        metrics = PerformanceMetrics(sharpe_ratio=0.5)
        regime = adaptive_budget._classify_performance_regime(metrics)
        assert regime == PerformanceRegime.AVERAGE
        
        # 较差表现
        metrics = PerformanceMetrics(sharpe_ratio=-0.2)
        regime = adaptive_budget._classify_performance_regime(metrics)
        assert regime == PerformanceRegime.POOR
        
        # 糟糕表现
        metrics = PerformanceMetrics(sharpe_ratio=-0.8)
        regime = adaptive_budget._classify_performance_regime(metrics)
        assert regime == PerformanceRegime.TERRIBLE
    
    def test_classify_market_condition(self, adaptive_budget):
        """测试市场状态分类"""
        # 危机状态
        metrics = MarketMetrics(market_volatility=0.5, uncertainty_index=0.9)
        condition = adaptive_budget._classify_market_condition(metrics)
        assert condition == MarketCondition.CRISIS
        
        # 高波动
        metrics = MarketMetrics(market_volatility=0.3, uncertainty_index=0.3)
        condition = adaptive_budget._classify_market_condition(metrics)
        assert condition == MarketCondition.HIGH_VOLATILITY
        
        # 低波动
        metrics = MarketMetrics(market_volatility=0.05, uncertainty_index=0.2)
        condition = adaptive_budget._classify_market_condition(metrics)
        assert condition == MarketCondition.LOW_VOLATILITY
        
        # 牛市
        metrics = MarketMetrics(market_volatility=0.15, market_trend=0.15, uncertainty_index=0.3)
        condition = adaptive_budget._classify_market_condition(metrics)
        assert condition == MarketCondition.BULL
        
        # 熊市
        metrics = MarketMetrics(market_volatility=0.15, market_trend=-0.15, uncertainty_index=0.3)
        condition = adaptive_budget._classify_market_condition(metrics)
        assert condition == MarketCondition.BEAR
        
        # 震荡市
        metrics = MarketMetrics(market_volatility=0.15, market_trend=0.05, uncertainty_index=0.3)
        condition = adaptive_budget._classify_market_condition(metrics)
        assert condition == MarketCondition.SIDEWAYS
    
    def test_calculate_performance_factor(self, adaptive_budget):
        """测试表现调整因子计算"""
        # 优秀表现
        metrics = PerformanceMetrics(sharpe_ratio=2.5, calmar_ratio=3.0)
        factor = adaptive_budget._calculate_performance_factor(metrics)
        assert factor > 1.0
        
        # 良好表现
        metrics = PerformanceMetrics(sharpe_ratio=1.2, calmar_ratio=1.5)
        factor = adaptive_budget._calculate_performance_factor(metrics)
        assert factor > 1.0
        
        # 一般表现
        metrics = PerformanceMetrics(sharpe_ratio=0.5, calmar_ratio=1.0)
        factor = adaptive_budget._calculate_performance_factor(metrics)
        assert factor == 1.0
        
        # 较差表现
        metrics = PerformanceMetrics(sharpe_ratio=-0.2, calmar_ratio=0.3)
        factor = adaptive_budget._calculate_performance_factor(metrics)
        assert factor < 1.0
    
    def test_calculate_market_factor(self, adaptive_budget):
        """测试市场调整因子计算"""
        # 危机状态
        adaptive_budget.current_market_condition = MarketCondition.CRISIS
        factor = adaptive_budget._calculate_market_factor(MarketMetrics())
        assert factor == 0.5
        
        # 熊市
        adaptive_budget.current_market_condition = MarketCondition.BEAR
        factor = adaptive_budget._calculate_market_factor(MarketMetrics())
        assert factor == 0.7
        
        # 牛市
        adaptive_budget.current_market_condition = MarketCondition.BULL
        factor = adaptive_budget._calculate_market_factor(MarketMetrics())
        assert factor == 1.2
        
        # 震荡市
        adaptive_budget.current_market_condition = MarketCondition.SIDEWAYS
        factor = adaptive_budget._calculate_market_factor(MarketMetrics())
        assert factor == 1.0
    
    def test_calculate_consecutive_loss_factor(self, adaptive_budget):
        """测试连续亏损调整因子计算"""
        # 无连续亏损
        metrics = PerformanceMetrics(consecutive_losses=0)
        factor = adaptive_budget._calculate_consecutive_loss_factor(metrics)
        assert factor == 1.0
        
        # 少量连续亏损
        metrics = PerformanceMetrics(consecutive_losses=2)
        factor = adaptive_budget._calculate_consecutive_loss_factor(metrics)
        assert factor == 1.0
        
        # 超过阈值的连续亏损
        metrics = PerformanceMetrics(consecutive_losses=5)
        factor = adaptive_budget._calculate_consecutive_loss_factor(metrics)
        assert factor < 1.0
    
    def test_calculate_volatility_factor(self, adaptive_budget):
        """测试波动率调整因子计算"""
        # 高波动率
        metrics = MarketMetrics(market_volatility=0.3)
        factor = adaptive_budget._calculate_volatility_factor(metrics)
        assert factor < 1.0
        
        # 低波动率
        metrics = MarketMetrics(market_volatility=0.05)
        factor = adaptive_budget._calculate_volatility_factor(metrics)
        assert factor > 1.0
        
        # 正常波动率
        metrics = MarketMetrics(market_volatility=0.15)
        factor = adaptive_budget._calculate_volatility_factor(metrics)
        assert factor == 1.0
    
    def test_calculate_uncertainty_factor(self, adaptive_budget):
        """测试不确定性调整因子计算"""
        # 高不确定性
        metrics = MarketMetrics(uncertainty_index=0.8)
        factor = adaptive_budget._calculate_uncertainty_factor(metrics)
        assert factor < 1.0
        
        # 低不确定性
        metrics = MarketMetrics(uncertainty_index=0.3)
        factor = adaptive_budget._calculate_uncertainty_factor(metrics)
        assert factor == 1.0
    
    def test_apply_adjustments(self, adaptive_budget):
        """测试应用调整因子"""
        factors = {
            'performance': 1.2,
            'market': 0.8,
            'consecutive_loss': 1.0,
            'volatility': 0.9,
            'uncertainty': 1.0,
            'recovery': 1.0
        }
        
        new_budget = adaptive_budget._apply_adjustments(factors)
        
        # 应该在合理范围内
        assert adaptive_budget.config.min_risk_budget <= new_budget <= adaptive_budget.config.max_risk_budget
        
        # 应该反映调整因子的影响
        expected_factor = 1.2 * 0.8 * 1.0 * (0.9 ** 0.5) * 1.0 * 1.0
        expected_budget = adaptive_budget.config.base_risk_budget * expected_factor
        assert abs(new_budget - expected_budget) < 0.001
    
    def test_apply_smoothing(self, adaptive_budget):
        """测试平滑机制"""
        # 设置当前状态
        adaptive_budget.current_risk_budget = 0.10
        adaptive_budget.smoothed_risk_budget = 0.10
        
        # 测试小幅调整
        new_budget = 0.11
        smoothed = adaptive_budget._apply_smoothing(new_budget)
        
        # 应该被平滑
        assert smoothed != new_budget
        assert adaptive_budget.current_risk_budget < smoothed < new_budget
        
        # 测试大幅调整（超过最大变化率）
        new_budget = 0.20  # 100%增长，超过最大变化率
        smoothed = adaptive_budget._apply_smoothing(new_budget)
        
        # 应该被限制在最大变化率内
        max_change = adaptive_budget.current_risk_budget * adaptive_budget.config.max_daily_change
        assert smoothed <= adaptive_budget.current_risk_budget + max_change
    
    def test_detect_anomaly(self, adaptive_budget):
        """测试异常检测"""
        # 填充历史数据
        for _ in range(25):
            adaptive_budget.risk_budget_history.append(0.10)
        
        # 正常值（在2.5个标准差内，由于所有值都是0.10，标准差为0，所以任何不同的值都会被认为是异常）
        # 我们需要添加一些变化
        adaptive_budget.risk_budget_history.clear()
        for i in range(25):
            adaptive_budget.risk_budget_history.append(0.10 + np.random.normal(0, 0.005))
        
        # 正常值
        assert not adaptive_budget._detect_anomaly(0.105)
        
        # 异常值
        assert adaptive_budget._detect_anomaly(0.20)
        assert adaptive_budget._detect_anomaly(0.02)
    
    def test_handle_anomaly(self, adaptive_budget):
        """测试异常处理"""
        adaptive_budget.current_risk_budget = 0.10
        
        anomalous_budget = 0.25
        handled_budget = adaptive_budget._handle_anomaly(anomalous_budget)
        
        # 应该被调整到更保守的值
        assert adaptive_budget.current_risk_budget < handled_budget < anomalous_budget
        assert adaptive_budget.last_anomaly_time is not None
    
    def test_calculate_adaptive_risk_budget_no_data(self, adaptive_budget):
        """测试无数据时的风险预算计算"""
        budget = adaptive_budget.calculate_adaptive_risk_budget()
        
        # 应该返回当前预算
        assert budget == adaptive_budget.config.base_risk_budget
    
    def test_calculate_adaptive_risk_budget_with_data(self, adaptive_budget):
        """测试有数据时的风险预算计算"""
        # 添加表现指标
        performance_metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            calmar_ratio=2.0,
            consecutive_losses=0
        )
        adaptive_budget.update_performance_metrics(performance_metrics)
        
        # 添加市场指标
        market_metrics = MarketMetrics(
            market_volatility=0.15,
            market_trend=0.05,
            uncertainty_index=0.3
        )
        adaptive_budget.update_market_metrics(market_metrics)
        
        budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 应该有调整
        assert budget != adaptive_budget.config.base_risk_budget
        assert len(adaptive_budget.adjustment_history) > 0
        assert adaptive_budget.last_update_time is not None
    
    def test_calculate_adaptive_risk_budget_excellent_performance(self, adaptive_budget):
        """测试优秀表现时的风险预算计算"""
        # 优秀表现
        performance_metrics = PerformanceMetrics(
            sharpe_ratio=2.5,
            calmar_ratio=3.0,
            consecutive_losses=0
        )
        adaptive_budget.update_performance_metrics(performance_metrics)
        
        # 良好市场条件
        market_metrics = MarketMetrics(
            market_volatility=0.08,
            market_trend=0.12,
            uncertainty_index=0.2
        )
        adaptive_budget.update_market_metrics(market_metrics)
        
        budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 应该增加风险预算
        assert budget > adaptive_budget.config.base_risk_budget
    
    def test_calculate_adaptive_risk_budget_poor_performance(self, adaptive_budget):
        """测试较差表现时的风险预算计算"""
        # 较差表现
        performance_metrics = PerformanceMetrics(
            sharpe_ratio=-0.5,
            calmar_ratio=0.2,
            consecutive_losses=5
        )
        adaptive_budget.update_performance_metrics(performance_metrics)
        
        # 不利市场条件
        market_metrics = MarketMetrics(
            market_volatility=0.35,
            market_trend=-0.15,
            uncertainty_index=0.8
        )
        adaptive_budget.update_market_metrics(market_metrics)
        
        budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 应该降低风险预算
        assert budget < adaptive_budget.config.base_risk_budget
    
    def test_adjustment_delay(self, adaptive_budget):
        """测试调整延迟"""
        # 设置最近更新时间
        adaptive_budget.last_update_time = datetime.now() - timedelta(hours=12)
        
        # 添加数据
        performance_metrics = PerformanceMetrics(sharpe_ratio=1.5)
        adaptive_budget.update_performance_metrics(performance_metrics)
        
        market_metrics = MarketMetrics(market_volatility=0.15)
        adaptive_budget.update_market_metrics(market_metrics)
        
        # 不强制更新，应该返回当前预算
        budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=False)
        assert budget == adaptive_budget.current_risk_budget
        
        # 强制更新，应该计算新预算
        budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        assert len(adaptive_budget.adjustment_history) > 0
    
    def test_get_risk_budget_summary(self, adaptive_budget):
        """测试获取风险预算摘要"""
        # 添加一些历史数据
        adaptive_budget.risk_budget_history.extend([0.08, 0.09, 0.10, 0.11, 0.12])
        
        # 添加调整记录
        adjustment = RiskBudgetAdjustment(
            timestamp=datetime.now(),
            old_budget=0.10,
            new_budget=0.12,
            adjustment_reason="表现优秀",
            performance_regime=PerformanceRegime.GOOD,
            market_condition=MarketCondition.BULL,
            adjustment_factors={'performance': 1.2}
        )
        adaptive_budget.adjustment_history.append(adjustment)
        
        summary = adaptive_budget.get_risk_budget_summary()
        
        assert 'current_risk_budget' in summary
        assert 'base_risk_budget' in summary
        assert 'performance_regime' in summary
        assert 'market_condition' in summary
        assert 'total_adjustments' in summary
        assert 'budget_range' in summary
        assert 'recent_adjustments' in summary
        
        assert summary['total_adjustments'] == 1
        assert summary['budget_range']['min'] == 0.08
        assert summary['budget_range']['max'] == 0.12
        assert len(summary['recent_adjustments']) == 1
    
    def test_reset_system(self, adaptive_budget):
        """测试系统重置"""
        # 添加一些数据
        adaptive_budget.current_risk_budget = 0.15
        adaptive_budget.performance_history.append(PerformanceMetrics())
        adaptive_budget.market_history.append(MarketMetrics())
        adaptive_budget.risk_budget_history.append(0.15)
        adaptive_budget.adjustment_history.append(
            RiskBudgetAdjustment(
                timestamp=datetime.now(),
                old_budget=0.10,
                new_budget=0.15,
                adjustment_reason="测试",
                performance_regime=PerformanceRegime.GOOD,
                market_condition=MarketCondition.BULL,
                adjustment_factors={}
            )
        )
        
        # 重置
        adaptive_budget.reset_system()
        
        # 验证重置结果
        assert adaptive_budget.current_risk_budget == adaptive_budget.config.base_risk_budget
        assert adaptive_budget.smoothed_risk_budget == adaptive_budget.config.base_risk_budget
        assert adaptive_budget.last_update_time is None
        assert len(adaptive_budget.performance_history) == 0
        assert len(adaptive_budget.market_history) == 0
        assert len(adaptive_budget.risk_budget_history) == 0
        assert len(adaptive_budget.adjustment_history) == 0
        assert adaptive_budget.current_performance_regime == PerformanceRegime.AVERAGE
        assert adaptive_budget.current_market_condition == MarketCondition.SIDEWAYS


class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        config = AdaptiveRiskBudgetConfig(
            base_risk_budget=0.10,
            smoothing_factor=0.3,
            max_daily_change=0.2
        )
        adaptive_budget = AdaptiveRiskBudget(config)
        
        # 模拟一系列市场条件和表现
        scenarios = [
            # 良好开始
            (PerformanceMetrics(sharpe_ratio=1.5, consecutive_losses=0),
             MarketMetrics(market_volatility=0.12, market_trend=0.08)),
            
            # 市场恶化
            (PerformanceMetrics(sharpe_ratio=0.5, consecutive_losses=2),
             MarketMetrics(market_volatility=0.25, market_trend=-0.05)),
            
            # 危机状态
            (PerformanceMetrics(sharpe_ratio=-0.5, consecutive_losses=5),
             MarketMetrics(market_volatility=0.4, uncertainty_index=0.9)),
            
            # 恢复阶段
            (PerformanceMetrics(sharpe_ratio=1.0, consecutive_losses=0),
             MarketMetrics(market_volatility=0.15, market_trend=0.03)),
        ]
        
        budgets = []
        for perf_metrics, market_metrics in scenarios:
            adaptive_budget.update_performance_metrics(perf_metrics)
            adaptive_budget.update_market_metrics(market_metrics)
            budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
            budgets.append(budget)
        
        # 验证预算变化趋势
        assert len(budgets) == 4
        assert all(config.min_risk_budget <= b <= config.max_risk_budget for b in budgets)
        
        # 应该有调整记录
        assert len(adaptive_budget.adjustment_history) > 0
        
        # 危机时预算应该最低
        crisis_budget = budgets[2]
        assert crisis_budget < config.base_risk_budget
        
        # 获取摘要
        summary = adaptive_budget.get_risk_budget_summary()
        assert summary['total_adjustments'] > 0
        assert len(summary['recent_adjustments']) > 0


class TestExtendedCoverage:
    """扩展测试覆盖率"""
    
    @pytest.fixture
    def adaptive_budget(self):
        """创建自适应风险预算实例"""
        config = AdaptiveRiskBudgetConfig(
            base_risk_budget=0.10,
            smoothing_factor=0.5,
            max_daily_change=0.3
        )
        return AdaptiveRiskBudget(config)
    
    def test_recovery_factor_calculation(self, adaptive_budget):
        """测试恢复因子计算"""
        # 模拟连续亏损后的恢复
        poor_metrics = PerformanceMetrics(sharpe_ratio=-1.0, consecutive_losses=3)
        adaptive_budget.update_performance_metrics(poor_metrics)
        adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 开始恢复
        recovery_metrics = PerformanceMetrics(sharpe_ratio=0.5, consecutive_losses=0)
        adaptive_budget.update_performance_metrics(recovery_metrics)
        recovery_budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 验证恢复因子的作用
        assert recovery_budget >= adaptive_budget.config.base_risk_budget * 0.8
    
    def test_anomaly_detection_and_handling(self, adaptive_budget):
        """测试异常检测和处理"""
        # 设置正常状态
        normal_metrics = PerformanceMetrics(sharpe_ratio=1.0)
        adaptive_budget.update_performance_metrics(normal_metrics)
        adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 添加一些历史数据以建立基线
        for _ in range(5):
            adaptive_budget.risk_budget_history.append(0.10)
        
        # 创建一个异常值
        extreme_metrics = PerformanceMetrics(sharpe_ratio=-5.0, consecutive_losses=10)
        extreme_market = MarketMetrics(market_volatility=1.0, uncertainty_index=1.0)
        
        adaptive_budget.update_performance_metrics(extreme_metrics)
        adaptive_budget.update_market_metrics(extreme_market)
        
        # 计算预算，应该触发异常处理
        anomaly_budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        # 验证异常被检测并处理
        assert adaptive_budget.config.min_risk_budget <= anomaly_budget <= adaptive_budget.config.max_risk_budget
    
    def test_adjustment_reason_generation(self, adaptive_budget):
        """测试调整原因生成"""
        # 提供不同的因子组合
        factors = {
            'performance_factor': 0.8,
            'market_factor': 1.2,
            'consecutive_loss_factor': 0.7,
            'volatility_factor': 0.9,
            'uncertainty_factor': 1.1,
            'recovery_factor': 1.0
        }
        
        # 调用私有方法测试原因生成
        reason = adaptive_budget._generate_adjustment_reason(factors)
        
        # 验证原因包含相关信息
        assert isinstance(reason, str)
        assert len(reason) > 0
        assert any(keyword in reason for keyword in ['表现', '市场', '连续', '波动', '不确定'])
    
    def test_smoothing_with_extreme_values(self, adaptive_budget):
        """测试极值情况下的平滑处理"""
        # 设置当前预算
        adaptive_budget.current_risk_budget = 0.10
        adaptive_budget.smoothed_risk_budget = 0.10
        
        # 测试极大值平滑
        large_budget = 0.20
        smoothed_large = adaptive_budget._apply_smoothing(large_budget)
        assert smoothed_large < large_budget
        assert smoothed_large > adaptive_budget.smoothed_risk_budget
        
        # 测试极小值平滑
        small_budget = 0.02
        smoothed_small = adaptive_budget._apply_smoothing(small_budget)
        assert smoothed_small > small_budget
        assert smoothed_small < adaptive_budget.smoothed_risk_budget
    
    def test_uncertainty_factor_edge_cases(self, adaptive_budget):
        """测试不确定性因子的边界情况"""
        # 极低不确定性
        low_uncertain_market = MarketMetrics(uncertainty_index=0.0)
        low_factor = adaptive_budget._calculate_uncertainty_factor(low_uncertain_market)
        assert low_factor >= 1.0
        
        # 极高不确定性
        high_uncertain_market = MarketMetrics(uncertainty_index=1.0)
        high_factor = adaptive_budget._calculate_uncertainty_factor(high_uncertain_market)
        assert high_factor <= 1.0
        
        # 中等不确定性
        medium_uncertain_market = MarketMetrics(uncertainty_index=0.5)
        medium_factor = adaptive_budget._calculate_uncertainty_factor(medium_uncertain_market)
        assert 0.8 <= medium_factor <= 1.2
    
    def test_performance_regime_classification_edge_cases(self, adaptive_budget):
        """测试表现制度分类的边界情况"""
        # 测试边界值
        boundary_metrics = PerformanceMetrics(sharpe_ratio=0.5)  # 正好在边界上
        regime = adaptive_budget._classify_performance_regime(boundary_metrics)
        assert regime in [PerformanceRegime.AVERAGE, PerformanceRegime.GOOD]
        
        # 测试极值
        extreme_good = PerformanceMetrics(sharpe_ratio=5.0)
        regime = adaptive_budget._classify_performance_regime(extreme_good)
        assert regime == PerformanceRegime.EXCELLENT
        
        extreme_bad = PerformanceMetrics(sharpe_ratio=-5.0)
        regime = adaptive_budget._classify_performance_regime(extreme_bad)
        assert regime == PerformanceRegime.POOR
    
    def test_market_condition_classification_comprehensive(self, adaptive_budget):
        """测试市场条件分类的全面情况"""
        # 强牛市
        strong_bull = MarketMetrics(market_trend=0.15, market_volatility=0.10)
        condition = adaptive_budget._classify_market_condition(strong_bull)
        assert condition == MarketCondition.BULL
        
        # 强熊市
        strong_bear = MarketMetrics(market_trend=-0.15, market_volatility=0.30)
        condition = adaptive_budget._classify_market_condition(strong_bear)
        assert condition == MarketCondition.BEAR
        
        # 震荡市场
        sideways = MarketMetrics(market_trend=0.02, market_volatility=0.20)
        condition = adaptive_budget._classify_market_condition(sideways)
        assert condition == MarketCondition.SIDEWAYS
    
    def test_consecutive_loss_factor_calculation(self, adaptive_budget):
        """测试连续亏损因子计算"""
        # 无连续亏损
        no_loss = PerformanceMetrics(consecutive_losses=0)
        factor = adaptive_budget._calculate_consecutive_loss_factor(no_loss)
        assert factor == 1.0
        
        # 短期连续亏损
        short_loss = PerformanceMetrics(consecutive_losses=2)
        factor = adaptive_budget._calculate_consecutive_loss_factor(short_loss)
        assert 0.8 <= factor < 1.0
        
        # 长期连续亏损
        long_loss = PerformanceMetrics(consecutive_losses=10)
        factor = adaptive_budget._calculate_consecutive_loss_factor(long_loss)
        assert factor < 0.8
    
    def test_adjustment_factors_integration(self, adaptive_budget):
        """测试调整因子的整合"""
        perf_metrics = PerformanceMetrics(
            sharpe_ratio=0.8,
            consecutive_losses=1,
            max_drawdown=0.05
        )
        market_metrics = MarketMetrics(
            market_volatility=0.18,
            market_trend=0.03,
            uncertainty_index=0.4
        )
        
        adaptive_budget.update_performance_metrics(perf_metrics)
        adaptive_budget.update_market_metrics(market_metrics)
        
        # 计算调整因子
        perf_regime = adaptive_budget._classify_performance_regime(perf_metrics)
        market_condition = adaptive_budget._classify_market_condition(market_metrics)
        
        factors = adaptive_budget._calculate_adjustment_factors(
            perf_metrics, market_metrics, perf_regime, market_condition
        )
        
        # 验证所有因子都被计算
        expected_factors = [
            'performance_factor', 'market_factor', 'consecutive_loss_factor',
            'volatility_factor', 'uncertainty_factor', 'recovery_factor'
        ]
        
        for factor_name in expected_factors:
            assert factor_name in factors
            assert isinstance(factors[factor_name], (int, float))
            assert factors[factor_name] > 0
    
    def test_historical_data_management(self, adaptive_budget):
        """测试历史数据管理"""
        # 添加大量历史数据
        for i in range(150):  # 超过默认限制
            metrics = PerformanceMetrics(sharpe_ratio=0.5 + i * 0.01)
            adaptive_budget.update_performance_metrics(metrics)
        
        # 验证历史数据被正确限制
        assert len(adaptive_budget.performance_history) <= 100
        
        # 验证最新数据被保留
        latest_metrics = adaptive_budget.performance_history[-1]
        assert latest_metrics.sharpe_ratio > 1.0


if __name__ == "__main__":
    pytest.main([__file__])