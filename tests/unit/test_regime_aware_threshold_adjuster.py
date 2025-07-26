"""
状态感知阈值调节器单元测试

测试RegimeAwareThresholdAdjuster类的各项功能，包括：
- 阈值调整逻辑
- 不同波动状态下的参数配置
- 动态阈值计算
- 异常处理
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_control.dynamic_lowvol_filter import (
    RegimeAwareThresholdAdjuster,
    DynamicLowVolConfig,
    ConfigurationException,
    DataQualityException
)


class TestRegimeAwareThresholdAdjuster(unittest.TestCase):
    """状态感知阈值调节器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建默认配置
        self.config = DynamicLowVolConfig(
            percentile_thresholds={"低": 0.4, "中": 0.3, "高": 0.2},
            ivol_bad_threshold=0.3,
            ivol_good_threshold=0.6
        )
        
        # 创建调节器实例
        self.adjuster = RegimeAwareThresholdAdjuster(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        # 验证配置正确加载
        self.assertEqual(self.adjuster.config, self.config)
        
        # 验证默认阈值设置
        self.assertIn("percentile_cut", self.adjuster.default_thresholds)
        self.assertIn("target_vol", self.adjuster.default_thresholds)
        self.assertIn("ivol_bad_threshold", self.adjuster.default_thresholds)
        self.assertIn("ivol_good_threshold", self.adjuster.default_thresholds)
        
        # 验证状态特定阈值配置
        for regime in ["低", "中", "高"]:
            self.assertIn(regime, self.adjuster.regime_thresholds)
            regime_config = self.adjuster.regime_thresholds[regime]
            self.assertIn("percentile_cut", regime_config)
            self.assertIn("target_vol", regime_config)
            self.assertIn("ivol_bad_threshold", regime_config)
            self.assertIn("ivol_good_threshold", regime_config)
            self.assertIn("garch_confidence", regime_config)
        
        # 验证阈值递减关系
        high_cut = self.adjuster.regime_thresholds["高"]["percentile_cut"]
        mid_cut = self.adjuster.regime_thresholds["中"]["percentile_cut"]
        low_cut = self.adjuster.regime_thresholds["低"]["percentile_cut"]
        self.assertLessEqual(high_cut, mid_cut)
        self.assertLessEqual(mid_cut, low_cut)
    
    def test_adjust_thresholds_basic(self):
        """测试基本阈值调整功能"""
        # 创建新的调节器实例以避免平滑效应
        fresh_adjuster = RegimeAwareThresholdAdjuster(self.config)
        
        # 测试高波动状态
        high_thresholds = fresh_adjuster.adjust_thresholds("高")
        self.assertIsInstance(high_thresholds, dict)
        self.assertIn("percentile_cut", high_thresholds)
        self.assertIn("target_vol", high_thresholds)
        self.assertEqual(high_thresholds["percentile_cut"], 0.2)
        
        # 创建另一个新实例测试中波动状态
        fresh_adjuster2 = RegimeAwareThresholdAdjuster(self.config)
        mid_thresholds = fresh_adjuster2.adjust_thresholds("中")
        self.assertEqual(mid_thresholds["percentile_cut"], 0.3)
        
        # 创建另一个新实例测试低波动状态
        fresh_adjuster3 = RegimeAwareThresholdAdjuster(self.config)
        low_thresholds = fresh_adjuster3.adjust_thresholds("低")
        self.assertEqual(low_thresholds["percentile_cut"], 0.4)
        
        # 验证阈值递减关系
        self.assertLessEqual(
            high_thresholds["percentile_cut"], 
            mid_thresholds["percentile_cut"]
        )
        self.assertLessEqual(
            mid_thresholds["percentile_cut"], 
            low_thresholds["percentile_cut"]
        )
    
    def test_adjust_thresholds_with_market_volatility(self):
        """测试带市场波动率的阈值调整"""
        # 使用新的调节器实例避免平滑效应
        adjuster1 = RegimeAwareThresholdAdjuster(self.config)
        adjuster2 = RegimeAwareThresholdAdjuster(self.config)
        adjuster3 = RegimeAwareThresholdAdjuster(self.config)
        
        # 测试高市场波动率（应收紧阈值）
        high_vol_thresholds = adjuster1.adjust_thresholds(
            "中", market_volatility=0.6
        )
        
        # 测试低市场波动率（应放宽阈值）
        low_vol_thresholds = adjuster2.adjust_thresholds(
            "中", market_volatility=0.15
        )
        
        # 测试正常市场波动率
        normal_vol_thresholds = adjuster3.adjust_thresholds(
            "中", market_volatility=0.3
        )
        
        # 验证调整方向正确
        # 高波动时阈值应更严格（更小）
        self.assertLess(
            high_vol_thresholds["percentile_cut"],
            normal_vol_thresholds["percentile_cut"]
        )
        
        # 低波动时阈值应更宽松（更大）
        self.assertGreater(
            low_vol_thresholds["percentile_cut"],
            normal_vol_thresholds["percentile_cut"]
        )
    
    def test_adjust_thresholds_with_confidence(self):
        """测试带置信度的阈值调整"""
        # 测试高置信度（应使用原始阈值）
        high_conf_thresholds = self.adjuster.adjust_thresholds(
            "高", regime_confidence=0.9
        )
        
        # 测试低置信度（应向中性状态靠拢）
        low_conf_thresholds = self.adjuster.adjust_thresholds(
            "高", regime_confidence=0.5
        )
        
        # 低置信度时应向中性状态靠拢
        neutral_threshold = self.adjuster.regime_thresholds["中"]["percentile_cut"]
        high_regime_threshold = self.adjuster.regime_thresholds["高"]["percentile_cut"]
        
        # 低置信度的阈值应介于高波动状态和中性状态之间
        self.assertGreater(
            low_conf_thresholds["percentile_cut"],
            high_regime_threshold
        )
        self.assertLess(
            low_conf_thresholds["percentile_cut"],
            neutral_threshold
        )
    
    def test_smoothing_mechanism(self):
        """测试平滑机制"""
        # 第一次调整
        first_thresholds = self.adjuster.adjust_thresholds("高")
        
        # 第二次调整到不同状态
        second_thresholds = self.adjuster.adjust_thresholds("低")
        
        # 验证平滑效果：第二次调整应该不会完全跳到新状态的阈值
        expected_low_threshold = self.adjuster.regime_thresholds["低"]["percentile_cut"]
        
        # 由于平滑，实际阈值应该介于前一次和目标阈值之间
        self.assertNotEqual(second_thresholds["percentile_cut"], expected_low_threshold)
        
        # 第三次调整应该更接近目标值
        third_thresholds = self.adjuster.adjust_thresholds("低")
        self.assertGreater(
            third_thresholds["percentile_cut"],
            second_thresholds["percentile_cut"]
        )
    
    def test_get_regime_specific_config(self):
        """测试获取特定状态配置"""
        # 测试各个状态的配置获取
        for regime in ["低", "中", "高"]:
            config = self.adjuster.get_regime_specific_config(regime)
            self.assertIsInstance(config, dict)
            self.assertIn("percentile_cut", config)
            self.assertIn("target_vol", config)
            self.assertIn("ivol_bad_threshold", config)
            self.assertIn("ivol_good_threshold", config)
            self.assertIn("garch_confidence", config)
        
        # 测试无效状态
        with self.assertRaises(ConfigurationException):
            self.adjuster.get_regime_specific_config("无效状态")
    
    def test_calculate_adaptive_percentile_threshold(self):
        """测试自适应分位数阈值计算"""
        # 测试正常压力水平
        normal_threshold = self.adjuster.calculate_adaptive_percentile_threshold(
            "中", market_stress_level=0.0
        )
        base_threshold = self.adjuster.regime_thresholds["中"]["percentile_cut"]
        self.assertEqual(normal_threshold, base_threshold)
        
        # 测试高压力水平（应收紧阈值）
        high_stress_threshold = self.adjuster.calculate_adaptive_percentile_threshold(
            "中", market_stress_level=0.8
        )
        self.assertLess(high_stress_threshold, base_threshold)
        
        # 测试低压力水平（应放宽阈值）
        low_stress_threshold = self.adjuster.calculate_adaptive_percentile_threshold(
            "中", market_stress_level=-0.8
        )
        self.assertGreater(low_stress_threshold, base_threshold)
        
        # 测试边界值
        min_threshold = self.adjuster.calculate_adaptive_percentile_threshold(
            "中", market_stress_level=1.0
        )
        max_threshold = self.adjuster.calculate_adaptive_percentile_threshold(
            "中", market_stress_level=-1.0
        )
        self.assertGreaterEqual(min_threshold, 0.1)
        self.assertLessEqual(max_threshold, 0.5)
    
    def test_adjustment_history_tracking(self):
        """测试调整历史记录"""
        # 初始状态应该没有历史记录
        self.assertEqual(len(self.adjuster.adjustment_history), 0)
        
        # 进行几次调整
        self.adjuster.adjust_thresholds("高")
        self.adjuster.adjust_thresholds("中", market_volatility=0.3)
        self.adjuster.adjust_thresholds("低", regime_confidence=0.8)
        
        # 验证历史记录
        self.assertEqual(len(self.adjuster.adjustment_history), 3)
        
        # 验证记录内容
        for record in self.adjuster.adjustment_history:
            self.assertIn("timestamp", record)
            self.assertIn("regime", record)
            self.assertIn("thresholds", record)
            self.assertIn("market_volatility", record)
            self.assertIn("regime_confidence", record)
        
        # 测试统计信息
        stats = self.adjuster.get_threshold_adjustment_statistics()
        self.assertEqual(stats["total_adjustments"], 3)
        self.assertIn("regime_distribution", stats)
        self.assertIn("average_thresholds", stats)
        self.assertIn("threshold_volatility", stats)
        
        # 测试重置历史
        self.adjuster.reset_adjustment_history()
        self.assertEqual(len(self.adjuster.adjustment_history), 0)
    
    def test_input_validation(self):
        """测试输入验证"""
        # 测试无效状态
        with self.assertRaises(ConfigurationException):
            self.adjuster.adjust_thresholds("无效状态")
        
        # 测试无效市场波动率
        with self.assertRaises(DataQualityException):
            self.adjuster.adjust_thresholds("中", market_volatility=-0.1)
        
        with self.assertRaises(DataQualityException):
            self.adjuster.adjust_thresholds("中", market_volatility=3.0)
        
        with self.assertRaises(DataQualityException):
            self.adjuster.adjust_thresholds("中", market_volatility="invalid")
        
        # 测试无效置信度
        with self.assertRaises(DataQualityException):
            self.adjuster.adjust_thresholds("中", regime_confidence=-0.1)
        
        with self.assertRaises(DataQualityException):
            self.adjuster.adjust_thresholds("中", regime_confidence=1.5)
        
        with self.assertRaises(DataQualityException):
            self.adjuster.adjust_thresholds("中", regime_confidence="invalid")
        
        # 测试自适应阈值计算的输入验证
        with self.assertRaises(ConfigurationException):
            self.adjuster.calculate_adaptive_percentile_threshold(
                "无效状态", market_stress_level=0.0
            )
        
        with self.assertRaises(ConfigurationException):
            self.adjuster.calculate_adaptive_percentile_threshold(
                "中", market_stress_level=2.0
            )
    
    def test_threshold_bounds_validation(self):
        """测试阈值边界验证"""
        # 创建极端配置来测试边界验证
        extreme_config = DynamicLowVolConfig(
            percentile_thresholds={"低": 0.9, "中": 0.5, "高": 0.1},
            ivol_bad_threshold=0.1,
            ivol_good_threshold=0.9
        )
        
        adjuster = RegimeAwareThresholdAdjuster(extreme_config)
        
        # 测试极端市场波动率调整
        thresholds = adjuster.adjust_thresholds(
            "中", market_volatility=2.0  # 极高波动率
        )
        
        # 验证调整后的阈值仍在合理范围内
        self.assertGreaterEqual(thresholds["percentile_cut"], 0.05)
        self.assertLessEqual(thresholds["percentile_cut"], 0.6)
        self.assertGreaterEqual(thresholds["ivol_bad_threshold"], 0.05)
        self.assertLessEqual(thresholds["ivol_bad_threshold"], 0.9)
    
    def test_configuration_validation(self):
        """测试配置验证"""
        # 测试无效的分位数阈值顺序
        with self.assertRaises(ConfigurationException):
            invalid_config = DynamicLowVolConfig(
                percentile_thresholds={"低": 0.2, "中": 0.3, "高": 0.4},  # 顺序错误
                ivol_bad_threshold=0.3,
                ivol_good_threshold=0.6
            )
            RegimeAwareThresholdAdjuster(invalid_config)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试零市场波动率
        thresholds = self.adjuster.adjust_thresholds(
            "中", market_volatility=0.0
        )
        self.assertIsInstance(thresholds, dict)
        
        # 测试零置信度 - 使用新的调节器实例避免平滑效应
        fresh_adjuster = RegimeAwareThresholdAdjuster(self.config)
        thresholds = fresh_adjuster.adjust_thresholds(
            "高", regime_confidence=0.0
        )
        # 零置信度应该完全向中性状态靠拢
        neutral_threshold = fresh_adjuster.regime_thresholds["中"]["percentile_cut"]
        # 由于置信度调整逻辑，实际值应该接近中性状态
        self.assertGreater(thresholds["percentile_cut"], 0.25)  # 应该大于高波动状态的0.2
        self.assertLess(thresholds["percentile_cut"], 0.35)     # 应该小于或接近中性状态的0.3
        
        # 测试完美置信度
        fresh_adjuster2 = RegimeAwareThresholdAdjuster(self.config)
        thresholds = fresh_adjuster2.adjust_thresholds(
            "高", regime_confidence=1.0
        )
        expected_threshold = fresh_adjuster2.regime_thresholds["高"]["percentile_cut"]
        self.assertAlmostEqual(
            thresholds["percentile_cut"], 
            expected_threshold, 
            places=3
        )
    
    def test_statistics_calculation(self):
        """测试统计信息计算"""
        # 空历史记录的统计
        stats = self.adjuster.get_threshold_adjustment_statistics()
        self.assertEqual(stats["total_adjustments"], 0)
        self.assertEqual(stats["regime_distribution"], {})
        
        # 添加一些调整记录
        regimes = ["高", "中", "低", "中", "高"]
        for regime in regimes:
            self.adjuster.adjust_thresholds(regime)
        
        stats = self.adjuster.get_threshold_adjustment_statistics()
        self.assertEqual(stats["total_adjustments"], 5)
        
        # 验证状态分布
        expected_distribution = {"高": 2/5, "中": 2/5, "低": 1/5}
        for regime, expected_ratio in expected_distribution.items():
            self.assertAlmostEqual(
                stats["regime_distribution"][regime], 
                expected_ratio, 
                places=3
            )
        
        # 验证平均阈值和波动率统计
        self.assertIn("average_thresholds", stats)
        self.assertIn("threshold_volatility", stats)
        self.assertIn("latest_regime", stats)
        self.assertIn("latest_thresholds", stats)


class TestRegimeAwareThresholdAdjusterIntegration(unittest.TestCase):
    """状态感知阈值调节器集成测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = DynamicLowVolConfig()
        self.adjuster = RegimeAwareThresholdAdjuster(self.config)
    
    def test_realistic_adjustment_scenario(self):
        """测试真实调整场景"""
        # 模拟市场从低波动到高波动的转换
        scenarios = [
            ("低", 0.15, 0.9),  # 低波动，低市场波动率，高置信度
            ("低", 0.20, 0.8),  # 低波动，波动率上升，置信度下降
            ("中", 0.30, 0.7),  # 转为中波动
            ("中", 0.45, 0.6),  # 中波动，波动率继续上升
            ("高", 0.60, 0.8),  # 转为高波动
            ("高", 0.55, 0.9),  # 高波动，波动率略降，置信度恢复
        ]
        
        previous_threshold = None
        for regime, market_vol, confidence in scenarios:
            thresholds = self.adjuster.adjust_thresholds(
                regime, market_volatility=market_vol, regime_confidence=confidence
            )
            
            # 验证阈值合理性
            self.assertGreater(thresholds["percentile_cut"], 0)
            self.assertLess(thresholds["percentile_cut"], 1)
            
            # 验证平滑性（相邻调整不应有剧烈变化）
            if previous_threshold is not None:
                threshold_change = abs(
                    thresholds["percentile_cut"] - previous_threshold
                )
                self.assertLess(threshold_change, 0.2)  # 单次调整不超过20%
            
            previous_threshold = thresholds["percentile_cut"]
        
        # 验证调整历史记录完整
        self.assertEqual(len(self.adjuster.adjustment_history), 6)
        
        # 验证统计信息
        stats = self.adjuster.get_threshold_adjustment_statistics()
        self.assertEqual(stats["total_adjustments"], 6)
        self.assertIn("低", stats["regime_distribution"])
        self.assertIn("中", stats["regime_distribution"])
        self.assertIn("高", stats["regime_distribution"])


if __name__ == '__main__':
    unittest.main()