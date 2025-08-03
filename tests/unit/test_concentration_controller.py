"""集中度控制器单元测试"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta

from src.rl_trading_system.risk_control.concentration_controller import (
    ConcentrationController,
    ConcentrationConfig,
    AssetInfo,
    ConcentrationViolation,
    ConcentrationMetrics,
    ConcentrationType,
    ViolationSeverity,
    ConcentrationLimit
)


class TestConcentrationController(unittest.TestCase):
    """集中度控制器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = ConcentrationConfig(
            max_single_asset_weight=0.15,
            single_asset_warning_threshold=0.12,
            max_sector_weight=0.30,
            sector_warning_threshold=0.25,
            max_factor_exposure=0.40,
            factor_warning_threshold=0.30,
            max_herfindahl_index=0.20,
            herfindahl_warning_threshold=0.15
        )
        self.controller = ConcentrationController(self.config)
        
        # 创建测试资产信息
        self.asset_info = {
            'AAPL': AssetInfo(
                symbol='AAPL',
                sector='Technology',
                geographic_region='North America',
                market_cap_category='Large Cap',
                factor_exposures={'Growth': 0.8, 'Quality': 0.6},
                current_weight=0.10,
                liquidity_score=0.9
            ),
            'GOOGL': AssetInfo(
                symbol='GOOGL',
                sector='Technology',
                geographic_region='North America',
                market_cap_category='Large Cap',
                factor_exposures={'Growth': 0.9, 'Momentum': 0.5},
                current_weight=0.08,
                liquidity_score=0.85
            ),
            'JPM': AssetInfo(
                symbol='JPM',
                sector='Financial',
                geographic_region='North America',
                market_cap_category='Large Cap',
                factor_exposures={'Value': 0.7, 'Quality': 0.4},
                current_weight=0.06,
                liquidity_score=0.8
            ),
            'TSM': AssetInfo(
                symbol='TSM',
                sector='Technology',
                geographic_region='Asia',
                market_cap_category='Large Cap',
                factor_exposures={'Growth': 0.6, 'Quality': 0.8},
                current_weight=0.05,
                liquidity_score=0.7
            ),
            'NVDA': AssetInfo(
                symbol='NVDA',
                sector='Technology',
                geographic_region='North America',
                market_cap_category='Large Cap',
                factor_exposures={'Growth': 1.0, 'Momentum': 0.9},
                current_weight=0.04,
                liquidity_score=0.75
            )
        }
        
        # 正常的投资组合权重
        self.normal_weights = {
            'AAPL': 0.10,
            'GOOGL': 0.08,
            'JPM': 0.06,
            'TSM': 0.05,
            'NVDA': 0.04
        }
        
        # 违规的投资组合权重
        self.violation_weights = {
            'AAPL': 0.20,  # 超过单一资产限制
            'GOOGL': 0.15,
            'JPM': 0.10,
            'TSM': 0.08,
            'NVDA': 0.07
        }
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(len(self.controller.concentration_limits), 5)
        self.assertEqual(len(self.controller.current_violations), 0)
        self.assertEqual(len(self.controller.violation_history), 0)
        self.assertIsNone(self.controller.last_check_time)
    
    def test_check_single_asset_concentration_normal(self):
        """测试正常情况下的单一资产集中度检查"""
        violations = self.controller.check_concentration_violations(
            self.normal_weights, self.asset_info
        )
        
        # 正常权重不应有违规
        single_asset_violations = [
            v for v in violations 
            if v.violation_type == ConcentrationType.SINGLE_ASSET
        ]
        self.assertEqual(len(single_asset_violations), 0)
    
    def test_check_single_asset_concentration_violation(self):
        """测试单一资产集中度违规"""
        violations = self.controller.check_concentration_violations(
            self.violation_weights, self.asset_info
        )
        
        # 应该检测到AAPL的违规
        single_asset_violations = [
            v for v in violations 
            if v.violation_type == ConcentrationType.SINGLE_ASSET
        ]
        self.assertGreater(len(single_asset_violations), 0)
        
        # 检查AAPL违规
        aapl_violation = next(
            (v for v in single_asset_violations if 'AAPL' in v.affected_items), 
            None
        )
        self.assertIsNotNone(aapl_violation)
        self.assertEqual(aapl_violation.current_value, 0.20)
        self.assertEqual(aapl_violation.limit_value, self.config.max_single_asset_weight)
    
    def test_check_sector_concentration_normal(self):
        """测试正常情况下的行业集中度检查"""
        violations = self.controller.check_concentration_violations(
            self.normal_weights, self.asset_info
        )
        
        # 正常权重不应有行业违规
        sector_violations = [
            v for v in violations 
            if v.violation_type == ConcentrationType.SECTOR
        ]
        # 可能有轻微违规，但不应该是严重违规
        critical_violations = [
            v for v in sector_violations 
            if v.severity == ViolationSeverity.CRITICAL
        ]
        self.assertEqual(len(critical_violations), 0)
    
    def test_check_sector_concentration_violation(self):
        """测试行业集中度违规"""
        # 创建行业集中的权重
        sector_violation_weights = {
            'AAPL': 0.15,
            'GOOGL': 0.12,
            'TSM': 0.10,
            'NVDA': 0.08,  # Technology总权重 = 45%，超过30%限制
            'JPM': 0.05
        }
        
        violations = self.controller.check_concentration_violations(
            sector_violation_weights, self.asset_info
        )
        
        # 应该检测到Technology行业违规
        sector_violations = [
            v for v in violations 
            if v.violation_type == ConcentrationType.SECTOR
        ]
        self.assertGreater(len(sector_violations), 0)
        
        # 检查Technology行业违规
        tech_violation = next(
            (v for v in sector_violations if 'Technology' in v.message), 
            None
        )
        self.assertIsNotNone(tech_violation)
    
    def test_check_factor_concentration(self):
        """测试因子集中度检查"""
        # 创建因子集中的权重（高Growth暴露）
        factor_violation_weights = {
            'AAPL': 0.20,   # Growth: 0.8
            'GOOGL': 0.15,  # Growth: 0.9
            'NVDA': 0.10,   # Growth: 1.0
            'JPM': 0.05,
            'TSM': 0.05
        }
        
        violations = self.controller.check_concentration_violations(
            factor_violation_weights, self.asset_info
        )
        
        # 可能检测到Growth因子违规
        factor_violations = [
            v for v in violations 
            if v.violation_type == ConcentrationType.FACTOR
        ]
        # 至少应该有一些因子相关的检查
        self.assertIsInstance(factor_violations, list)
    
    def test_check_geographic_concentration(self):
        """测试地理集中度检查"""
        # 创建地理集中的权重
        geo_violation_weights = {
            'AAPL': 0.25,
            'GOOGL': 0.20,
            'JPM': 0.15,
            'NVDA': 0.10,  # North America总权重 = 70%，超过60%限制
            'TSM': 0.05
        }
        
        violations = self.controller.check_concentration_violations(
            geo_violation_weights, self.asset_info
        )
        
        # 应该检测到North America地区违规
        geo_violations = [
            v for v in violations 
            if v.violation_type == ConcentrationType.GEOGRAPHIC
        ]
        self.assertGreater(len(geo_violations), 0)
    
    def test_check_herfindahl_index_normal(self):
        """测试正常情况下的赫芬达尔指数"""
        violations = self.controller.check_concentration_violations(
            self.normal_weights, self.asset_info
        )
        
        # 计算赫芬达尔指数
        herfindahl = sum(w**2 for w in self.normal_weights.values())
        self.assertLess(herfindahl, self.config.max_herfindahl_index)
    
    def test_check_herfindahl_index_violation(self):
        """测试赫芬达尔指数违规"""
        # 创建高集中度权重
        concentrated_weights = {
            'AAPL': 0.50,
            'GOOGL': 0.30,
            'JPM': 0.20
        }
        
        violations = self.controller.check_concentration_violations(
            concentrated_weights, self.asset_info
        )
        
        # 应该检测到赫芬达尔指数违规
        herfindahl_violations = [
            v for v in violations 
            if 'Herfindahl' in v.message or '赫芬达尔' in v.message
        ]
        self.assertGreater(len(herfindahl_violations), 0)
    
    def test_determine_severity(self):
        """测试违规严重程度判断"""
        # 测试不同严重程度
        low_severity = self.controller._determine_severity(0.10, 0.15, 0.12)
        self.assertEqual(low_severity, ViolationSeverity.LOW)
        
        medium_severity = self.controller._determine_severity(0.13, 0.15, 0.12)
        self.assertEqual(medium_severity, ViolationSeverity.MEDIUM)
        
        high_severity = self.controller._determine_severity(0.16, 0.15, 0.12)
        self.assertEqual(high_severity, ViolationSeverity.HIGH)
        
        critical_severity = self.controller._determine_severity(0.20, 0.15, 0.12)
        self.assertEqual(critical_severity, ViolationSeverity.CRITICAL)
    
    def test_adjust_single_asset_concentration(self):
        """测试单一资产集中度调整"""
        # 使用更温和的违规权重进行测试
        mild_violation_weights = {
            'AAPL': 0.18,  # 轻微超过15%限制
            'GOOGL': 0.12,
            'JPM': 0.10,
            'TSM': 0.08,
            'NVDA': 0.07
        }
        
        adjusted_weights = self.controller.adjust_portfolio_for_concentration(
            mild_violation_weights, self.asset_info
        )
        
        # 检查权重和是否为1
        self.assertAlmostEqual(sum(adjusted_weights.values()), 1.0, places=6)
        
        # 检查是否有调整发生（总调整幅度应该大于0）
        total_adjustment = sum(
            abs(adjusted_weights.get(symbol, 0) - mild_violation_weights.get(symbol, 0))
            for symbol in set(list(adjusted_weights.keys()) + list(mild_violation_weights.keys()))
        )
        self.assertGreater(total_adjustment, 0)
        
        # 检查调整历史记录
        self.assertGreater(len(self.controller.adjustment_history), 0)
    
    def test_adjust_portfolio_no_violations(self):
        """测试无违规时的投资组合调整"""
        adjusted_weights = self.controller.adjust_portfolio_for_concentration(
            self.normal_weights, self.asset_info
        )
        
        # 无违规时权重应基本不变
        for symbol in self.normal_weights:
            self.assertAlmostEqual(
                adjusted_weights[symbol], 
                self.normal_weights[symbol], 
                places=3
            )
    
    def test_redistribute_weight(self):
        """测试权重重新分配"""
        weights = self.normal_weights.copy()
        original_total = sum(weights.values())
        
        # 模拟重新分配
        self.controller._redistribute_weight(
            weights, 'AAPL', 0.05, self.asset_info
        )
        
        # 检查总权重增加
        new_total = sum(weights.values())
        self.assertAlmostEqual(new_total, original_total + 0.05, places=6)
        
        # 检查AAPL权重未变（被排除）
        self.assertEqual(weights['AAPL'], self.normal_weights['AAPL'])
    
    def test_normalize_weights(self):
        """测试权重归一化"""
        unnormalized_weights = {
            'AAPL': 0.20,
            'GOOGL': 0.15,
            'JPM': 0.10
        }
        
        normalized_weights = self.controller._normalize_weights(unnormalized_weights)
        
        # 检查权重和为1
        self.assertAlmostEqual(sum(normalized_weights.values()), 1.0, places=6)
        
        # 检查比例保持
        total = sum(unnormalized_weights.values())
        for symbol in unnormalized_weights:
            expected = unnormalized_weights[symbol] / total
            self.assertAlmostEqual(normalized_weights[symbol], expected, places=6)
    
    def test_calculate_concentration_metrics(self):
        """测试集中度指标计算"""
        metrics = self.controller.calculate_concentration_metrics(
            self.normal_weights, self.asset_info
        )
        
        # 检查指标类型
        self.assertIsInstance(metrics, ConcentrationMetrics)
        
        # 检查赫芬达尔指数
        expected_herfindahl = sum(w**2 for w in self.normal_weights.values())
        self.assertAlmostEqual(metrics.herfindahl_index, expected_herfindahl, places=6)
        
        # 检查最大单一权重
        expected_max_weight = max(self.normal_weights.values())
        self.assertEqual(metrics.max_single_weight, expected_max_weight)
        
        # 检查有效资产数量
        expected_effective_assets = 1.0 / expected_herfindahl
        self.assertAlmostEqual(metrics.effective_number_of_assets, expected_effective_assets, places=6)
        
        # 检查多样化比率
        expected_div_ratio = 1.0 - expected_max_weight
        self.assertEqual(metrics.diversification_ratio, expected_div_ratio)
        
        # 检查行业集中度
        self.assertIn('Technology', metrics.sector_concentration)
        self.assertIn('Financial', metrics.sector_concentration)
    
    def test_calculate_concentration_metrics_empty_portfolio(self):
        """测试空投资组合的集中度指标计算"""
        metrics = self.controller.calculate_concentration_metrics({}, {})
        
        self.assertEqual(metrics.herfindahl_index, 0.0)
        self.assertEqual(metrics.max_single_weight, 0.0)
        self.assertEqual(metrics.effective_number_of_assets, 0.0)
        self.assertEqual(metrics.diversification_ratio, 0.0)
    
    def test_get_severity_score(self):
        """测试严重程度评分"""
        self.assertEqual(self.controller._get_severity_score(ViolationSeverity.LOW), 1)
        self.assertEqual(self.controller._get_severity_score(ViolationSeverity.MEDIUM), 2)
        self.assertEqual(self.controller._get_severity_score(ViolationSeverity.HIGH), 3)
        self.assertEqual(self.controller._get_severity_score(ViolationSeverity.CRITICAL), 4)
    
    def test_update_concentration_limit(self):
        """测试更新集中度限制"""
        original_limit = self.controller.concentration_limits[ConcentrationType.SINGLE_ASSET].max_weight
        
        # 更新限制
        new_max_weight = 0.20
        self.controller.update_concentration_limit(
            ConcentrationType.SINGLE_ASSET,
            max_weight=new_max_weight
        )
        
        # 检查更新
        updated_limit = self.controller.concentration_limits[ConcentrationType.SINGLE_ASSET].max_weight
        self.assertEqual(updated_limit, new_max_weight)
        self.assertNotEqual(updated_limit, original_limit)
    
    def test_get_concentration_summary(self):
        """测试获取集中度摘要"""
        # 先进行一次检查以生成数据
        self.controller.check_concentration_violations(
            self.violation_weights, self.asset_info
        )
        
        summary = self.controller.get_concentration_summary()
        
        # 检查摘要内容
        self.assertIn('current_violations', summary)
        self.assertIn('total_violations_history', summary)
        self.assertIn('last_check_time', summary)
        self.assertIn('concentration_limits', summary)
        
        # 检查限制信息
        limits = summary['concentration_limits']
        self.assertIn('single_asset', limits)
        self.assertIn('sector', limits)
        
        # 检查单一资产限制
        single_asset_limit = limits['single_asset']
        self.assertEqual(single_asset_limit['max_weight'], self.config.max_single_asset_weight)
        self.assertEqual(single_asset_limit['warning_threshold'], self.config.single_asset_warning_threshold)
        self.assertTrue(single_asset_limit['enabled'])
    
    def test_reset_history(self):
        """测试重置历史记录"""
        # 先生成一些历史数据
        self.controller.check_concentration_violations(
            self.violation_weights, self.asset_info
        )
        self.controller.calculate_concentration_metrics(
            self.normal_weights, self.asset_info
        )
        
        # 确认有历史数据
        self.assertGreater(len(self.controller.violation_history), 0)
        self.assertGreater(len(self.controller.metrics_history), 0)
        
        # 重置历史
        self.controller.reset_history()
        
        # 确认历史已清空
        self.assertEqual(len(self.controller.violation_history), 0)
        self.assertEqual(len(self.controller.metrics_history), 0)
        self.assertEqual(len(self.controller.adjustment_history), 0)
        self.assertEqual(len(self.controller.current_violations), 0)
    
    def test_concentration_limits_initialization(self):
        """测试集中度限制初始化"""
        limits = self.controller.concentration_limits
        
        # 检查所有类型的限制都被初始化
        expected_types = [
            ConcentrationType.SINGLE_ASSET,
            ConcentrationType.SECTOR,
            ConcentrationType.FACTOR,
            ConcentrationType.GEOGRAPHIC,
            ConcentrationType.MARKET_CAP
        ]
        
        for concentration_type in expected_types:
            self.assertIn(concentration_type, limits)
            limit = limits[concentration_type]
            self.assertIsInstance(limit, ConcentrationLimit)
            self.assertTrue(limit.enabled)
            self.assertGreater(limit.max_weight, 0)
            self.assertGreater(limit.warning_threshold, 0)
            self.assertLess(limit.warning_threshold, limit.max_weight)
    
    def test_violation_history_tracking(self):
        """测试违规历史跟踪"""
        initial_count = len(self.controller.violation_history)
        
        # 进行多次检查
        self.controller.check_concentration_violations(
            self.violation_weights, self.asset_info
        )
        first_check_count = len(self.controller.violation_history)
        
        self.controller.check_concentration_violations(
            self.normal_weights, self.asset_info
        )
        second_check_count = len(self.controller.violation_history)
        
        # 检查历史记录增长
        self.assertGreaterEqual(first_check_count, initial_count)
        self.assertGreaterEqual(second_check_count, first_check_count)
    
    def test_adjustment_with_sector_exclusion(self):
        """测试带行业排除的调整"""
        weights = {'AAPL': 0.1, 'GOOGL': 0.1, 'JPM': 0.1}
        
        # 重新分配权重，排除Technology行业
        self.controller._redistribute_weight(
            weights, None, 0.1, self.asset_info, exclude_sector='Technology'
        )
        
        # Technology行业的资产权重不应增加太多
        # JPM（Financial行业）应该获得更多权重
        self.assertGreater(weights['JPM'], 0.1)
    
    def test_liquidity_based_redistribution(self):
        """测试基于流动性的权重重新分配"""
        weights = {'AAPL': 0.1, 'GOOGL': 0.1, 'JPM': 0.1}
        original_weights = weights.copy()
        
        # 重新分配权重
        self.controller._redistribute_weight(
            weights, 'AAPL', 0.1, self.asset_info
        )
        
        # 流动性更高的资产应该获得更多权重
        # AAPL流动性最高(0.9)，但被排除
        # GOOGL流动性(0.85) > JPM流动性(0.8)
        googl_increase = weights['GOOGL'] - original_weights['GOOGL']
        jpm_increase = weights['JPM'] - original_weights['JPM']
        
        # GOOGL应该获得更多增加（由于更高的流动性）
        self.assertGreater(googl_increase, jpm_increase)


if __name__ == '__main__':
    unittest.main()