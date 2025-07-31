"""
风险控制模块的单元测试
测试持仓集中度限制和行业暴露控制，止损机制和异常交易检测，风险控制规则的有效性
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
import json
from decimal import Decimal

from src.rl_trading_system.risk_control.risk_controller import (
    RiskController,
    PositionConcentrationController,
    SectorExposureController,
    StopLossController,
    AnomalousTradeDetector,
    RiskRule,
    RiskViolation,
    RiskLevel,
    RiskControlConfig,
    TradeDecision,
    Portfolio,
    Position,
    Trade
)


class TestRiskController:
    """风险控制器测试类"""

    @pytest.fixture
    def risk_config(self):
        """创建风险控制配置"""
        return RiskControlConfig(
            max_position_weight=0.1,  # 单个持仓最大权重10%
            max_sector_exposure=0.3,  # 单个行业最大暴露30%
            max_portfolio_concentration=0.6,  # 投资组合最大集中度60%
            stop_loss_threshold=0.05,  # 止损阈值5%
            max_daily_loss=0.02,  # 最大日损失2%
            max_drawdown=0.1,  # 最大回撤10%
            min_cash_ratio=0.05,  # 最小现金比例5%
            max_leverage=2.0,  # 最大杠杆2倍
            volatility_threshold=0.3,  # 波动率阈值30%
            liquidity_threshold=1000000  # 流动性阈值100万
        )

    @pytest.fixture
    def sample_portfolio(self):
        """创建样本投资组合"""
        positions = [
            Position("AAPL", 100, 150.0, "Technology", datetime.now()),
            Position("MSFT", 80, 300.0, "Technology", datetime.now()),
            Position("JPM", 50, 140.0, "Financial", datetime.now()),
            Position("JNJ", 60, 170.0, "Healthcare", datetime.now()),
            Position("XOM", 40, 60.0, "Energy", datetime.now())
        ]
        
        return Portfolio(
            positions=positions,
            cash=50000.0,
            total_value=100000.0,
            timestamp=datetime.now()
        )

    @pytest.fixture
    def risk_controller(self, risk_config):
        """创建风险控制器"""
        return RiskController(config=risk_config)

    def test_risk_controller_initialization(self, risk_controller, risk_config):
        """测试风险控制器初始化"""
        assert risk_controller.config == risk_config
        assert risk_controller.position_controller is not None
        assert risk_controller.sector_controller is not None
        assert risk_controller.stop_loss_controller is not None
        assert risk_controller.anomaly_detector is not None
        assert len(risk_controller.risk_rules) > 0
        assert len(risk_controller.violations_history) == 0

    def test_comprehensive_risk_check(self, risk_controller, sample_portfolio):
        """测试综合风险检查"""
        # 创建一个交易决策
        trade_decision = TradeDecision(
            symbol="TSLA",
            action="BUY",
            quantity=100,
            target_price=800.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        # 执行风险检查
        risk_check_result = risk_controller.check_trade_risk(
            trade_decision=trade_decision,
            current_portfolio=sample_portfolio
        )
        
        assert isinstance(risk_check_result, dict)
        assert 'approved' in risk_check_result
        assert 'violations' in risk_check_result
        assert 'risk_score' in risk_check_result
        assert 'recommendations' in risk_check_result
        
        # 验证风险评分在合理范围内
        assert 0 <= risk_check_result['risk_score'] <= 1.0

    def test_portfolio_risk_assessment(self, risk_controller, sample_portfolio):
        """测试投资组合风险评估"""
        risk_assessment = risk_controller.assess_portfolio_risk(sample_portfolio)
        
        assert isinstance(risk_assessment, dict)
        assert 'overall_risk_score' in risk_assessment
        assert 'concentration_risk' in risk_assessment
        assert 'sector_risk' in risk_assessment
        assert 'liquidity_risk' in risk_assessment
        assert 'volatility_risk' in risk_assessment
        
        # 验证各项风险评分
        for risk_type, score in risk_assessment.items():
            if isinstance(score, (int, float)):
                assert 0 <= score <= 1.0

    def test_risk_rule_management(self, risk_controller):
        """测试风险规则管理"""
        # 添加自定义风险规则
        custom_rule = RiskRule(
            rule_id="custom_volatility",
            rule_name="自定义波动率限制",
            risk_level=RiskLevel.MEDIUM,
            check_function=lambda portfolio, trade: True,
            description="测试用自定义规则"
        )
        
        risk_controller.add_risk_rule(custom_rule)
        
        # 验证规则已添加
        assert "custom_volatility" in [rule.rule_id for rule in risk_controller.risk_rules]
        
        # 移除规则
        risk_controller.remove_risk_rule("custom_volatility")
        assert "custom_volatility" not in [rule.rule_id for rule in risk_controller.risk_rules]

    def test_dynamic_risk_adjustment(self, risk_controller, sample_portfolio):
        """测试动态风险调整"""
        # 模拟市场波动率上升
        market_conditions = {
            'market_volatility': 0.4,
            'market_trend': 'declining',
            'liquidity_stress': 0.7
        }
        
        # 调整风险参数
        risk_controller.adjust_risk_parameters(market_conditions)
        
        # 验证参数已调整
        adjusted_config = risk_controller.get_current_config()
        assert adjusted_config.volatility_threshold != risk_controller.config.volatility_threshold

    def test_risk_violation_handling(self, risk_controller, sample_portfolio):
        """测试风险违规处理"""
        # 创建一个会触发违规的大额交易
        large_trade = TradeDecision(
            symbol="NVDA",
            action="BUY",
            quantity=1000,  # 大量购买
            target_price=500.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        risk_result = risk_controller.check_trade_risk(large_trade, sample_portfolio)
        
        # 如果有违规，验证违规记录
        if not risk_result['approved']:
            assert len(risk_result['violations']) > 0
            
            # 验证违规记录的完整性
            for violation in risk_result['violations']:
                assert violation.rule_id is not None
                assert violation.risk_level is not None
                assert violation.description is not None
                assert violation.timestamp is not None

    def test_emergency_risk_controls(self, risk_controller, sample_portfolio):
        """测试紧急风险控制"""
        # 模拟紧急情况
        emergency_conditions = {
            'market_crash': True,
            'portfolio_loss': 0.15,  # 15%损失
            'volatility_spike': 0.8
        }
        
        emergency_response = risk_controller.handle_emergency_situation(
            emergency_conditions, sample_portfolio
        )
        
        assert isinstance(emergency_response, dict)
        assert 'emergency_actions' in emergency_response
        assert 'risk_overrides' in emergency_response
        assert 'portfolio_protection' in emergency_response

    def test_risk_metrics_calculation(self, risk_controller, sample_portfolio):
        """测试风险指标计算"""
        risk_metrics = risk_controller.calculate_risk_metrics(sample_portfolio)
        
        expected_metrics = [
            'var_95', 'cvar_95', 'max_drawdown', 'volatility',
            'sharpe_ratio', 'beta', 'concentration_herfindahl',
            'sector_concentration', 'liquidity_score'
        ]
        
        for metric in expected_metrics:
            assert metric in risk_metrics
            assert isinstance(risk_metrics[metric], (int, float))

    def test_real_time_risk_monitoring(self, risk_controller, sample_portfolio):
        """测试实时风险监控"""
        # 开始监控
        risk_controller.start_real_time_monitoring(sample_portfolio)
        
        # 模拟价格变动
        price_updates = {
            "AAPL": 140.0,  # 下跌
            "MSFT": 310.0,  # 上涨
            "JPM": 130.0    # 下跌
        }
        
        # 更新投资组合
        updated_portfolio = risk_controller.update_portfolio_prices(
            sample_portfolio, price_updates
        )
        
        # 检查是否触发监控告警
        monitoring_alerts = risk_controller.get_monitoring_alerts()
        
        assert isinstance(monitoring_alerts, list)
        # 验证告警结构
        for alert in monitoring_alerts:
            assert 'alert_type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'timestamp' in alert

    def test_risk_budget_management(self, risk_controller, sample_portfolio):
        """测试风险预算管理"""
        # 设置风险预算
        risk_budget = {
            'total_var_budget': 0.02,
            'sector_budgets': {
                'Technology': 0.008,
                'Financial': 0.005,
                'Healthcare': 0.004,
                'Energy': 0.003
            }
        }
        
        risk_controller.set_risk_budget(risk_budget)
        
        # 检查风险预算使用情况
        budget_utilization = risk_controller.check_risk_budget_utilization(sample_portfolio)
        
        assert isinstance(budget_utilization, dict)
        assert 'total_utilization' in budget_utilization
        assert 'sector_utilization' in budget_utilization
        assert 'remaining_budget' in budget_utilization

    def test_stress_testing(self, risk_controller, sample_portfolio):
        """测试压力测试"""
        # 定义压力测试场景
        stress_scenarios = [
            {
                'name': 'market_crash',
                'market_shock': -0.2,
                'volatility_increase': 2.0,
                'correlation_increase': 0.3
            },
            {
                'name': 'sector_rotation',
                'tech_shock': -0.15,
                'finance_boost': 0.1,
                'duration_days': 30
            }
        ]
        
        stress_results = risk_controller.run_stress_tests(
            sample_portfolio, stress_scenarios
        )
        
        assert isinstance(stress_results, dict)
        assert len(stress_results) == len(stress_scenarios)
        
        for scenario_name, result in stress_results.items():
            assert 'portfolio_loss' in result
            assert 'var_impact' in result
            assert 'worst_positions' in result


class TestPositionConcentrationController:
    """持仓集中度控制器测试类"""

    @pytest.fixture
    def concentration_controller(self):
        """创建持仓集中度控制器"""
        return PositionConcentrationController(
            max_single_position=0.1,  # 单个持仓最大10%
            max_top5_concentration=0.4,  # 前5大持仓最大40%
            max_top10_concentration=0.6,  # 前10大持仓最大60%
            herfindahl_threshold=0.2  # Herfindahl指数阈值
        )

    @pytest.fixture
    def concentrated_portfolio(self):
        """创建集中度较高的投资组合"""
        positions = [
            Position("AAPL", 500, 200.0, "Technology", datetime.now()),  # 10万，50%
            Position("MSFT", 200, 250.0, "Technology", datetime.now()),  # 5万，25%
            Position("GOOGL", 100, 150.0, "Technology", datetime.now()),  # 1.5万，7.5%
            Position("AMZN", 50, 300.0, "Technology", datetime.now()),   # 1.5万，7.5%
        ]
        
        return Portfolio(
            positions=positions,
            cash=20000.0,
            total_value=200000.0,
            timestamp=datetime.now()
        )

    def test_single_position_concentration_check(self, concentration_controller, concentrated_portfolio):
        """测试单个持仓集中度检查"""
        concentration_result = concentration_controller.check_position_concentration(
            concentrated_portfolio
        )
        
        assert isinstance(concentration_result, dict)
        assert 'violations' in concentration_result
        assert 'concentration_metrics' in concentration_result
        
        # AAPL持仓50%应该违规
        violations = concentration_result['violations']
        assert len(violations) > 0
        
        # 验证违规信息
        single_position_violations = [v for v in violations if 'single_position' in v.rule_id]
        assert len(single_position_violations) > 0

    def test_top_positions_concentration(self, concentration_controller, concentrated_portfolio):
        """测试头部持仓集中度"""
        metrics = concentration_controller.calculate_concentration_metrics(concentrated_portfolio)
        
        assert 'top5_concentration' in metrics
        assert 'top10_concentration' in metrics
        assert 'herfindahl_index' in metrics
        
        # 验证计算结果的合理性
        assert metrics['top5_concentration'] >= metrics['top10_concentration']
        assert 0 <= metrics['herfindahl_index'] <= 1

    def test_herfindahl_index_calculation(self, concentration_controller):
        """测试Herfindahl指数计算"""
        # 创建不同集中度的投资组合进行测试
        
        # 高度集中的投资组合
        high_concentration_positions = [
            Position("STOCK1", 1000, 100.0, "Tech", datetime.now()),  # 90%
            Position("STOCK2", 100, 100.0, "Tech", datetime.now())    # 10%
        ]
        high_conc_portfolio = Portfolio(high_concentration_positions, 0, 110000.0, datetime.now())
        
        # 分散的投资组合
        diversified_positions = [
            Position(f"STOCK{i}", 50, 100.0, "Tech", datetime.now()) for i in range(20)
        ]  # 每个5%
        diversified_portfolio = Portfolio(diversified_positions, 0, 100000.0, datetime.now())
        
        # 计算HHI
        high_hhi = concentration_controller.calculate_herfindahl_index(high_conc_portfolio)
        diversified_hhi = concentration_controller.calculate_herfindahl_index(diversified_portfolio)
        
        # 高度集中的组合应该有更高的HHI
        assert high_hhi > diversified_hhi
        assert high_hhi > 0.8  # 应该接近1
        assert diversified_hhi < 0.1  # 应该接近0.05 (1/20)

    def test_concentration_violation_detection(self, concentration_controller):
        """测试集中度违规检测"""
        # 创建会违规的交易
        current_portfolio = Portfolio([], 100000.0, 100000.0, datetime.now())
        
        # 尝试买入超过限制的大仓位
        large_trade = TradeDecision(
            symbol="AAPL",
            action="BUY",
            quantity=600,  # 价值12万，超过10万组合的10%限制
            target_price=200.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        violation_result = concentration_controller.check_trade_impact(
            large_trade, current_portfolio
        )
        
        assert not violation_result['approved']
        assert len(violation_result['violations']) > 0

    def test_concentration_adjustment_recommendations(self, concentration_controller, concentrated_portfolio):
        """测试集中度调整建议"""
        recommendations = concentration_controller.get_rebalancing_recommendations(
            concentrated_portfolio
        )
        
        assert isinstance(recommendations, list)
        
        # 应该建议减少AAPL仓位
        aapl_recommendations = [r for r in recommendations if r['symbol'] == 'AAPL']
        assert len(aapl_recommendations) > 0
        assert aapl_recommendations[0]['action'] == 'REDUCE'

    def test_dynamic_concentration_limits(self, concentration_controller):
        """测试动态集中度限制"""
        # 在高波动市场环境下，应该降低集中度限制
        market_conditions = {
            'volatility': 0.4,
            'correlation': 0.8,
            'liquidity_stress': 0.6
        }
        
        concentration_controller.adjust_limits_for_market_conditions(market_conditions)
        
        # 验证限制已被收紧
        adjusted_limits = concentration_controller.get_current_limits()
        original_limit = 0.1
        
        assert adjusted_limits['max_single_position'] <= original_limit

    def test_sector_concentration_within_position_control(self, concentration_controller):
        """测试行业集中度在持仓控制中的考虑"""
        # 创建同一行业的多个持仓
        tech_heavy_positions = [
            Position("AAPL", 100, 150.0, "Technology", datetime.now()),
            Position("MSFT", 100, 250.0, "Technology", datetime.now()),
            Position("GOOGL", 100, 200.0, "Technology", datetime.now()),
            Position("META", 100, 300.0, "Technology", datetime.now())
        ]
        
        tech_portfolio = Portfolio(tech_heavy_positions, 10000.0, 100000.0, datetime.now())
        
        # 检查是否考虑了行业集中度风险
        sector_aware_check = concentration_controller.check_sector_aware_concentration(
            tech_portfolio
        )
        
        assert 'sector_concentration_risk' in sector_aware_check
        assert sector_aware_check['sector_concentration_risk'] > 0.5  # 应该识别出高风险


class TestSectorExposureController:
    """行业暴露控制器测试类"""

    @pytest.fixture
    def sector_controller(self):
        """创建行业暴露控制器"""
        return SectorExposureController(
            max_sector_exposure={
                'Technology': 0.4,    # 科技行业最大40%
                'Financial': 0.3,     # 金融行业最大30%
                'Healthcare': 0.25,   # 医疗行业最大25%
                'Energy': 0.2,        # 能源行业最大20%
                'Consumer': 0.3       # 消费行业最大30%
            },
            min_sector_diversification=5,  # 至少5个行业
            max_correlation_threshold=0.7   # 行业间最大相关性
        )

    @pytest.fixture
    def sector_biased_portfolio(self):
        """创建有行业偏向的投资组合"""
        positions = [
            Position("AAPL", 200, 150.0, "Technology", datetime.now()),
            Position("MSFT", 150, 200.0, "Technology", datetime.now()),
            Position("GOOGL", 100, 250.0, "Technology", datetime.now()),
            Position("NVDA", 80, 400.0, "Technology", datetime.now()),  # 科技股占主导
            Position("JPM", 50, 140.0, "Financial", datetime.now()),
            Position("BAC", 40, 35.0, "Financial", datetime.now())
        ]
        
        return Portfolio(positions, 15000.0, 150000.0, datetime.now())

    def test_sector_exposure_calculation(self, sector_controller, sector_biased_portfolio):
        """测试行业暴露度计算"""
        sector_exposures = sector_controller.calculate_sector_exposures(sector_biased_portfolio)
        
        assert isinstance(sector_exposures, dict)
        assert 'Technology' in sector_exposures
        assert 'Financial' in sector_exposures
        
        # 验证暴露度之和不超过100%（排除现金）
        total_exposure = sum(sector_exposures.values())
        assert total_exposure <= 1.0
        
        # 科技行业应该是最大的暴露
        assert sector_exposures['Technology'] > sector_exposures.get('Financial', 0)

    def test_sector_exposure_violations(self, sector_controller, sector_biased_portfolio):
        """测试行业暴露违规检测"""
        violation_check = sector_controller.check_sector_violations(sector_biased_portfolio)
        
        assert isinstance(violation_check, dict)
        assert 'violations' in violation_check
        assert 'exposure_summary' in violation_check
        
        # 科技行业可能超过40%限制
        violations = violation_check['violations']
        tech_violations = [v for v in violations if 'Technology' in v.description]
        
        if len(tech_violations) > 0:
            assert tech_violations[0].risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]

    def test_sector_diversification_check(self, sector_controller):
        """测试行业多样化检查"""
        # 创建缺乏多样化的投资组合
        undiversified_positions = [
            Position("AAPL", 300, 150.0, "Technology", datetime.now()),
            Position("MSFT", 200, 200.0, "Technology", datetime.now()),
            Position("GOOGL", 150, 250.0, "Technology", datetime.now())
        ]
        
        undiversified_portfolio = Portfolio(undiversified_positions, 10000.0, 150000.0, datetime.now())
        
        diversification_check = sector_controller.check_diversification(undiversified_portfolio)
        
        assert 'diversification_score' in diversification_check
        assert 'sectors_count' in diversification_check
        assert 'diversification_violations' in diversification_check
        
        # 应该检测到多样化不足
        assert diversification_check['sectors_count'] < sector_controller.min_sector_diversification

    def test_sector_correlation_analysis(self, sector_controller):
        """测试行业相关性分析"""
        # 提供历史价格数据进行相关性分析
        historical_returns = pd.DataFrame({
            'Technology': np.random.normal(0.01, 0.02, 100),
            'Financial': np.random.normal(0.008, 0.025, 100),
            'Healthcare': np.random.normal(0.007, 0.015, 100),
            'Energy': np.random.normal(0.005, 0.03, 100)
        })
        
        correlation_analysis = sector_controller.analyze_sector_correlations(historical_returns)
        
        assert isinstance(correlation_analysis, dict)
        assert 'correlation_matrix' in correlation_analysis
        assert 'high_correlation_pairs' in correlation_analysis
        assert 'correlation_risk_score' in correlation_analysis

    def test_dynamic_sector_limits(self, sector_controller):
        """测试动态行业限制调整"""
        # 模拟特定行业的负面事件
        sector_events = {
            'Technology': {
                'regulatory_risk': 0.8,
                'valuation_concern': 0.7,
                'earnings_uncertainty': 0.6
            }
        }
        
        sector_controller.adjust_sector_limits_for_events(sector_events)
        
        # 验证科技行业限制已被降低
        adjusted_limits = sector_controller.get_current_sector_limits()
        assert adjusted_limits['Technology'] < 0.4  # 应该低于原始40%限制

    def test_sector_rotation_recommendations(self, sector_controller, sector_biased_portfolio):
        """测试行业轮动建议"""
        # 提供宏观经济指标
        macro_indicators = {
            'interest_rates': 0.05,
            'inflation': 0.03,
            'gdp_growth': 0.025,
            'unemployment': 0.04
        }
        
        rotation_recommendations = sector_controller.get_sector_rotation_recommendations(
            sector_biased_portfolio, macro_indicators
        )
        
        assert isinstance(rotation_recommendations, list)
        
        for recommendation in rotation_recommendations:
            assert 'sector' in recommendation
            assert 'action' in recommendation  # INCREASE, DECREASE, MAINTAIN
            assert 'rationale' in recommendation
            assert 'target_allocation' in recommendation

    def test_sector_risk_budgeting(self, sector_controller):
        """测试行业风险预算"""
        # 设置行业风险预算
        risk_budget = {
            'Technology': 0.15,  # 科技行业风险预算15%
            'Financial': 0.10,   # 金融行业风险预算10%
            'Healthcare': 0.08,  # 医疗行业风险预算8%
            'Energy': 0.12,      # 能源行业风险预算12%
            'Consumer': 0.10     # 消费行业风险预算10%
        }
        
        sector_controller.set_sector_risk_budgets(risk_budget)
        
        # 创建测试投资组合
        test_positions = [
            Position("AAPL", 100, 200.0, "Technology", datetime.now()),
            Position("JPM", 80, 150.0, "Financial", datetime.now()),
            Position("JNJ", 60, 180.0, "Healthcare", datetime.now())
        ]
        test_portfolio = Portfolio(test_positions, 20000.0, 80000.0, datetime.now())
        
        # 检查风险预算使用情况
        budget_usage = sector_controller.check_risk_budget_usage(test_portfolio)
        
        assert isinstance(budget_usage, dict)
        for sector in risk_budget.keys():
            if sector in budget_usage:
                assert 'allocated_risk' in budget_usage[sector]
                assert 'used_risk' in budget_usage[sector]
                assert 'remaining_budget' in budget_usage[sector]

    def test_sector_concentration_stress_test(self, sector_controller, sector_biased_portfolio):
        """测试行业集中度压力测试"""
        # 定义行业冲击场景
        stress_scenarios = {
            'tech_crash': {'Technology': -0.3, 'Financial': -0.1},
            'financial_crisis': {'Financial': -0.4, 'Technology': -0.15},
            'regulatory_crackdown': {'Technology': -0.25, 'Healthcare': -0.2}
        }
        
        stress_results = sector_controller.run_sector_stress_tests(
            sector_biased_portfolio, stress_scenarios
        )
        
        assert isinstance(stress_results, dict)
        
        for scenario_name, result in stress_results.items():
            assert 'portfolio_impact' in result
            assert 'sector_contributions' in result
            assert 'risk_adjusted_impact' in result


class TestStopLossController:
    """止损控制器测试类"""

    @pytest.fixture
    def stop_loss_controller(self):
        """创建止损控制器"""
        return StopLossController(
            default_stop_loss=0.05,      # 默认5%止损
            trailing_stop_distance=0.03,  # 跟踪止损距离3%
            max_position_loss=0.10,      # 单个持仓最大损失10%
            portfolio_stop_loss=0.08,    # 投资组合止损8%
            volatility_adjusted=True,     # 启用波动率调整
            sector_specific_stops={       # 行业特定止损
                'Technology': 0.06,
                'Energy': 0.08,
                'Healthcare': 0.04
            }
        )

    @pytest.fixture
    def portfolio_with_losses(self):
        """创建有浮亏的投资组合"""
        positions = [
            Position("AAPL", 100, 150.0, "Technology", datetime.now() - timedelta(days=30),
                    cost_basis=160.0),  # 浮亏6.25%
            Position("TSLA", 50, 200.0, "Technology", datetime.now() - timedelta(days=15),
                    cost_basis=250.0),  # 浮亏20%
            Position("XOM", 200, 60.0, "Energy", datetime.now() - timedelta(days=45),
                    cost_basis=65.0),   # 浮亏7.69%
            Position("JNJ", 80, 170.0, "Healthcare", datetime.now() - timedelta(days=20),
                    cost_basis=165.0)   # 浮盈3.03%
        ]
        
        return Portfolio(positions, 20000.0, 100000.0, datetime.now())

    def test_position_stop_loss_check(self, stop_loss_controller, portfolio_with_losses):
        """测试单个持仓止损检查"""
        stop_loss_alerts = stop_loss_controller.check_position_stop_losses(portfolio_with_losses)
        
        assert isinstance(stop_loss_alerts, list)
        
        # TSLA应该触发止损（20%损失超过任何合理的止损线）
        tsla_alerts = [alert for alert in stop_loss_alerts if alert['symbol'] == 'TSLA']
        assert len(tsla_alerts) > 0
        assert tsla_alerts[0]['action'] == 'STOP_LOSS_TRIGGERED'

    def test_trailing_stop_loss(self, stop_loss_controller):
        """测试跟踪止损"""
        # 创建一个持仓的价格历史
        price_history = [
            {'date': datetime.now() - timedelta(days=10), 'price': 100.0},
            {'date': datetime.now() - timedelta(days=9), 'price': 105.0},
            {'date': datetime.now() - timedelta(days=8), 'price': 110.0},  # 峰值
            {'date': datetime.now() - timedelta(days=7), 'price': 108.0},
            {'date': datetime.now() - timedelta(days=6), 'price': 106.0},
            {'date': datetime.now() - timedelta(days=5), 'price': 104.0},
            {'date': datetime.now() - timedelta(days=4), 'price': 102.0},
            {'date': datetime.now() - timedelta(days=3), 'price': 100.0},
            {'date': datetime.now() - timedelta(days=2), 'price': 98.0},
            {'date': datetime.now() - timedelta(days=1), 'price': 96.0},
            {'date': datetime.now(), 'price': 94.0}
        ]
        
        position = Position("TEST", 100, 94.0, "Technology", datetime.now() - timedelta(days=10),
                          cost_basis=100.0)
        
        trailing_stop_result = stop_loss_controller.calculate_trailing_stop(
            position, price_history
        )
        
        assert isinstance(trailing_stop_result, dict)
        assert 'current_stop_price' in trailing_stop_result
        assert 'triggered' in trailing_stop_result
        assert 'peak_price' in trailing_stop_result
        
        # 从峰值110下跌到94，超过3%跟踪距离，应该触发
        assert trailing_stop_result['triggered'] is True

    def test_volatility_adjusted_stop_loss(self, stop_loss_controller):
        """测试波动率调整止损"""
        # 高波动率股票
        high_vol_position = Position("VOLATILE", 100, 100.0, "Technology", datetime.now(),
                                   cost_basis=100.0)
        
        # 提供波动率数据
        volatility_data = {
            'VOLATILE': 0.4,  # 40%年化波动率
            'market_volatility': 0.2
        }
        
        adjusted_stop = stop_loss_controller.calculate_volatility_adjusted_stop(
            high_vol_position, volatility_data
        )
        
        # 高波动率股票应该有更宽的止损
        assert adjusted_stop['stop_loss_percentage'] > stop_loss_controller.default_stop_loss

    def test_portfolio_level_stop_loss(self, stop_loss_controller, portfolio_with_losses):
        """测试投资组合级别止损"""
        # 添加投资组合历史价值数据
        portfolio_history = [
            {'date': datetime.now() - timedelta(days=30), 'value': 120000.0},
            {'date': datetime.now() - timedelta(days=20), 'value': 115000.0},
            {'date': datetime.now() - timedelta(days=10), 'value': 105000.0},
            {'date': datetime.now(), 'value': 100000.0}  # 当前价值
        ]
        
        portfolio_stop_check = stop_loss_controller.check_portfolio_stop_loss(
            portfolio_with_losses, portfolio_history
        )
        
        assert isinstance(portfolio_stop_check, dict)
        assert 'triggered' in portfolio_stop_check
        assert 'max_portfolio_value' in portfolio_stop_check
        assert 'current_drawdown' in portfolio_stop_check
        
        # 从12万跌到10万，回撤16.67%，超过8%止损线
        assert portfolio_stop_check['triggered'] is True

    def test_sector_specific_stop_losses(self, stop_loss_controller):
        """测试行业特定止损"""
        # 不同行业的持仓
        tech_position = Position("AAPL", 100, 94.0, "Technology", datetime.now(),
                               cost_basis=100.0)  # 6%损失
        energy_position = Position("XOM", 100, 90.0, "Energy", datetime.now(),
                                 cost_basis=100.0)  # 10%损失
        healthcare_position = Position("JNJ", 100, 96.0, "Healthcare", datetime.now(),
                                     cost_basis=100.0)  # 4%损失
        
        positions = [tech_position, energy_position, healthcare_position]
        
        sector_stop_results = []
        for position in positions:
            result = stop_loss_controller.check_sector_specific_stop(position)
            sector_stop_results.append(result)
        
        # 验证不同行业应用了不同的止损标准
        # 科技股6%损失应该触发（6% > 6%标准）
        tech_result = [r for r in sector_stop_results if r['symbol'] == 'AAPL'][0]
        assert tech_result['triggered'] is True
        
        # 能源股10%损失应该触发（10% > 8%标准）
        energy_result = [r for r in sector_stop_results if r['symbol'] == 'XOM'][0]
        assert energy_result['triggered'] is True
        
        # 医疗股4%损失应该触发（4% = 4%标准）
        healthcare_result = [r for r in sector_stop_results if r['symbol'] == 'JNJ'][0]
        assert healthcare_result['triggered'] is True

    def test_dynamic_stop_loss_adjustment(self, stop_loss_controller):
        """测试动态止损调整"""
        # 模拟市场条件变化
        market_conditions = {
            'market_volatility': 0.35,  # 高波动率
            'market_trend': 'declining',
            'correlation_increase': 0.8
        }
        
        stop_loss_controller.adjust_stops_for_market_conditions(market_conditions)
        
        # 在高波动和下跌市场中，止损应该更紧
        adjusted_stops = stop_loss_controller.get_current_stop_levels()
        
        # 验证止损已被调整
        assert adjusted_stops['default_stop_loss'] != stop_loss_controller.default_stop_loss

    def test_stop_loss_execution_simulation(self, stop_loss_controller):
        """测试止损执行模拟"""
        # 创建触发止损的持仓
        triggered_position = Position("LOSER", 100, 85.0, "Technology", datetime.now(),
                                    cost_basis=100.0)
        
        execution_plan = stop_loss_controller.create_stop_loss_execution_plan(triggered_position)
        
        assert isinstance(execution_plan, dict)
        assert 'symbol' in execution_plan
        assert 'action' in execution_plan
        assert 'quantity' in execution_plan
        assert 'order_type' in execution_plan
        assert 'execution_priority' in execution_plan
        assert 'estimated_slippage' in execution_plan

    def test_stop_loss_override_conditions(self, stop_loss_controller):
        """测试止损覆盖条件"""
        # 定义可能覆盖止损的条件
        override_conditions = {
            'earnings_announcement': True,
            'major_news_pending': True,
            'unusual_volume': True,
            'technical_support_level': 85.0
        }
        
        position = Position("NEWS_STOCK", 100, 85.0, "Technology", datetime.now(),
                          cost_basis=100.0)
        
        override_decision = stop_loss_controller.evaluate_stop_loss_override(
            position, override_conditions
        )
        
        assert isinstance(override_decision, dict)
        assert 'override_recommended' in override_decision
        assert 'override_reasons' in override_decision
        assert 'risk_assessment' in override_decision


class TestAnomalousTradeDetector:
    """异常交易检测器测试类"""

    @pytest.fixture
    def anomaly_detector(self):
        """创建异常交易检测器"""
        return AnomalousTradeDetector(
            volume_threshold_multiplier=3.0,    # 成交量异常阈值3倍
            price_movement_threshold=0.05,      # 价格异动阈值5%
            frequency_threshold=10,             # 频率阈值每小时10次
            size_threshold_percentile=95,       # 交易规模阈值95分位数
            pattern_detection_window=24,        # 模式检测窗口24小时
            ml_anomaly_threshold=0.8            # 机器学习异常阈值
        )

    @pytest.fixture
    def historical_trades(self):
        """创建历史交易数据"""
        trades = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(1000):
            trade = Trade(
                symbol=np.random.choice(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]),
                quantity=np.random.normal(100, 30),
                price=np.random.normal(200, 50),
                timestamp=base_time + timedelta(hours=i*0.72),  # 约每小时1.4个交易
                trade_type=np.random.choice(["BUY", "SELL"]),
                sector=np.random.choice(["Technology", "Consumer", "Healthcare"])
            )
            trades.append(trade)
        
        return trades

    def test_volume_anomaly_detection(self, anomaly_detector, historical_trades):
        """测试成交量异常检测"""
        # 添加历史数据
        anomaly_detector.add_historical_trades(historical_trades)
        
        # 创建异常大成交量的交易
        anomalous_trade = Trade(
            symbol="AAPL",
            quantity=1000,  # 异常大的数量
            price=150.0,
            timestamp=datetime.now(),
            trade_type="BUY",
            sector="Technology"
        )
        
        volume_anomaly_result = anomaly_detector.detect_volume_anomaly(anomalous_trade)
        
        assert isinstance(volume_anomaly_result, dict)
        assert 'is_anomalous' in volume_anomaly_result
        assert 'anomaly_score' in volume_anomaly_result
        assert 'historical_average' in volume_anomaly_result
        assert 'threshold' in volume_anomaly_result
        
        # 大成交量应该被标记为异常
        assert volume_anomaly_result['is_anomalous'] is True

    def test_price_movement_anomaly(self, anomaly_detector):
        """测试价格异动检测"""
        # 创建价格异动的交易
        normal_price_trade = Trade("AAPL", 100, 150.0, datetime.now(), "BUY", "Technology")
        
        # 模拟市场价格数据
        market_data = {
            "AAPL": {
                "current_price": 150.0,
                "previous_close": 145.0,
                "bid": 149.8,
                "ask": 150.2,
                "volume": 1000000
            }
        }
        
        anomaly_detector.update_market_data(market_data)
        
        # 创建偏离市场价格的交易
        price_anomaly_trade = Trade("AAPL", 100, 160.0, datetime.now(), "BUY", "Technology")
        
        price_anomaly_result = anomaly_detector.detect_price_anomaly(price_anomaly_trade)
        
        assert isinstance(price_anomaly_result, dict)
        assert 'is_anomalous' in price_anomaly_result
        assert 'price_deviation' in price_anomaly_result
        assert 'market_price' in price_anomaly_result
        
        # 偏离市场价格的交易应该被标记为异常
        assert price_anomaly_result['is_anomalous'] is True

    def test_trading_frequency_anomaly(self, anomaly_detector):
        """测试交易频率异常检测"""
        # 创建高频交易序列
        high_frequency_trades = []
        base_time = datetime.now()
        
        for i in range(15):  # 15分钟内15个交易，超过正常频率
            trade = Trade(
                symbol="AAPL",
                quantity=100,
                price=150.0 + i * 0.1,
                timestamp=base_time + timedelta(minutes=i),
                trade_type="BUY",
                sector="Technology"
            )
            high_frequency_trades.append(trade)
        
        frequency_anomaly_result = anomaly_detector.detect_frequency_anomaly(
            high_frequency_trades, time_window_hours=1
        )
        
        assert isinstance(frequency_anomaly_result, dict)
        assert 'is_anomalous' in frequency_anomaly_result
        assert 'trade_count' in frequency_anomaly_result
        assert 'threshold' in frequency_anomaly_result
        
        # 高频交易应该被标记为异常
        assert frequency_anomaly_result['is_anomalous'] is True

    def test_trade_size_anomaly(self, anomaly_detector, historical_trades):
        """测试交易规模异常检测"""
        anomaly_detector.add_historical_trades(historical_trades)
        
        # 创建异常大的交易
        large_trade = Trade(
            symbol="AAPL",
            quantity=5000,  # 异常大的交易规模
            price=150.0,
            timestamp=datetime.now(),
            trade_type="BUY",
            sector="Technology"
        )
        
        size_anomaly_result = anomaly_detector.detect_size_anomaly(large_trade)
        
        assert isinstance(size_anomaly_result, dict)
        assert 'is_anomalous' in size_anomaly_result
        assert 'size_percentile' in size_anomaly_result
        assert 'dollar_value' in size_anomaly_result
        
        # 大规模交易应该被标记为异常
        assert size_anomaly_result['is_anomalous'] is True

    def test_pattern_based_anomaly_detection(self, anomaly_detector, historical_trades):
        """测试基于模式的异常检测"""
        anomaly_detector.add_historical_trades(historical_trades)
        
        # 创建可疑的交易模式（例如：大量小额交易）
        suspicious_pattern_trades = []
        for i in range(20):
            trade = Trade(
                symbol="AAPL",
                quantity=10,  # 小额交易
                price=150.0,
                timestamp=datetime.now() + timedelta(minutes=i),
                trade_type="BUY",
                sector="Technology"
            )
            suspicious_pattern_trades.append(trade)
        
        pattern_anomaly_result = anomaly_detector.detect_pattern_anomaly(
            suspicious_pattern_trades
        )
        
        assert isinstance(pattern_anomaly_result, dict)
        assert 'suspicious_patterns' in pattern_anomaly_result
        assert 'pattern_scores' in pattern_anomaly_result
        assert 'overall_anomaly_score' in pattern_anomaly_result

    def test_machine_learning_anomaly_detection(self, anomaly_detector, historical_trades):
        """测试机器学习异常检测"""
        # 训练异常检测模型
        anomaly_detector.train_ml_model(historical_trades)
        
        # 创建测试交易
        test_trades = [
            Trade("AAPL", 100, 150.0, datetime.now(), "BUY", "Technology"),  # 正常
            Trade("AAPL", 10000, 200.0, datetime.now(), "BUY", "Technology"),  # 异常
            Trade("TSLA", 50, 250.0, datetime.now(), "SELL", "Technology")  # 正常
        ]
        
        ml_results = []
        for trade in test_trades:
            ml_result = anomaly_detector.detect_ml_anomaly(trade)
            ml_results.append(ml_result)
        
        # 验证ML模型能够识别异常
        assert len(ml_results) == 3
        
        # 第二个交易（大额异常交易）应该有高异常分数
        assert ml_results[1]['anomaly_score'] > ml_results[0]['anomaly_score']
        assert ml_results[1]['anomaly_score'] > ml_results[2]['anomaly_score']

    def test_comprehensive_anomaly_screening(self, anomaly_detector, historical_trades):
        """测试综合异常筛查"""
        anomaly_detector.add_historical_trades(historical_trades)
        
        # 创建多方面异常的交易
        highly_anomalous_trade = Trade(
            symbol="AAPL",
            quantity=8000,  # 大规模
            price=180.0,    # 价格偏高
            timestamp=datetime.now(),
            trade_type="BUY",
            sector="Technology"
        )
        
        comprehensive_result = anomaly_detector.comprehensive_anomaly_check(
            highly_anomalous_trade
        )
        
        assert isinstance(comprehensive_result, dict)
        assert 'overall_anomaly_score' in comprehensive_result
        assert 'individual_scores' in comprehensive_result
        assert 'risk_assessment' in comprehensive_result
        assert 'recommended_action' in comprehensive_result
        
        # 综合异常分数应该较高
        assert comprehensive_result['overall_anomaly_score'] > 0.7

    def test_real_time_anomaly_monitoring(self, anomaly_detector):
        """测试实时异常监控"""
        # 启动实时监控
        anomaly_detector.start_real_time_monitoring()
        
        # 模拟交易流
        trade_stream = [
            Trade("AAPL", 100, 150.0, datetime.now(), "BUY", "Technology"),
            Trade("AAPL", 5000, 155.0, datetime.now() + timedelta(seconds=30), "BUY", "Technology"),
            Trade("MSFT", 200, 250.0, datetime.now() + timedelta(seconds=60), "SELL", "Technology")
        ]
        
        real_time_alerts = []
        for trade in trade_stream:
            alert = anomaly_detector.process_real_time_trade(trade)
            if alert:
                real_time_alerts.append(alert)
        
        # 应该有异常告警（第二个大额交易）
        assert len(real_time_alerts) > 0
        
        # 验证告警结构
        for alert in real_time_alerts:
            assert 'trade_id' in alert
            assert 'anomaly_type' in alert
            assert 'severity' in alert
            assert 'timestamp' in alert

    def test_false_positive_reduction(self, anomaly_detector, historical_trades):
        """测试假阳性降低机制"""
        anomaly_detector.add_historical_trades(historical_trades)
        
        # 创建可能的假阳性案例（大交易但在合理范围内）
        potentially_false_positive_trade = Trade(
            symbol="AAPL",
            quantity=500,  # 较大但可能合理
            price=150.0,
            timestamp=datetime.now(),
            trade_type="BUY",
            sector="Technology"
        )
        
        # 提供额外的上下文信息
        context_info = {
            'market_volatility': 0.15,
            'recent_news': True,
            'earnings_season': True,
            'institutional_flow': True
        }
        
        refined_result = anomaly_detector.detect_anomaly_with_context(
            potentially_false_positive_trade, context_info
        )
        
        assert isinstance(refined_result, dict)
        assert 'adjusted_anomaly_score' in refined_result
        assert 'context_adjustments' in refined_result
        assert 'confidence_level' in refined_result
        
        # 有上下文的检测应该降低假阳性
        basic_result = anomaly_detector.comprehensive_anomaly_check(potentially_false_positive_trade)
        assert refined_result['adjusted_anomaly_score'] <= basic_result['overall_anomaly_score']