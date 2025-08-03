"""
市场状态感知系统测试
测试MarketRegimeDetector的各项功能
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from typing import List
import numpy as np
import pandas as pd

from src.rl_trading_system.risk_control.market_regime_detector import (
    MarketRegimeDetector, MarketRegimeConfig, MarketRegime, 
    MarketIndicators, RegimeDetectionResult, MarketRegimeAnalyzer
)
from src.rl_trading_system.data.data_models import MarketData


class TestMarketRegimeDetector(unittest.TestCase):
    """市场状态检测器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = MarketRegimeConfig(
            ma_short_period=5,
            ma_long_period=10,
            volatility_window=5,
            regime_persistence=2
        )
        self.detector = MarketRegimeDetector(self.config)
        
        # 创建测试数据
        self.base_time = datetime(2024, 1, 1)
        self.test_prices = [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116]
        
    def _create_market_data(self, price: float, timestamp: datetime, volume: int = 1000000) -> MarketData:
        """创建测试用市场数据"""
        return MarketData(
            timestamp=timestamp,
            symbol="TEST",
            open_price=price * 0.99,
            high_price=price * 1.01,
            low_price=price * 0.98,
            close_price=price,
            volume=volume,
            amount=price * volume
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.config.ma_short_period, 5)
        self.assertEqual(self.detector.config.ma_long_period, 10)
        self.assertIsNone(self.detector.current_regime)
        self.assertEqual(len(self.detector.price_history), 0)
    
    def test_update_market_data_insufficient_data(self):
        """测试数据不足时的处理"""
        market_data = self._create_market_data(100, self.base_time)
        result = self.detector.update_market_data(market_data)
        
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertEqual(result.regime, MarketRegime.SIDEWAYS_MARKET)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(self.detector.price_history), 1)
    
    def test_update_market_data_sufficient_data(self):
        """测试有足够数据时的状态检测"""
        # 添加足够的历史数据
        for i, price in enumerate(self.test_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            result = self.detector.update_market_data(market_data)
        
        # 最后一次更新应该有有效的检测结果
        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIsInstance(result.regime, MarketRegime)
        self.assertGreater(result.confidence, 0)
        self.assertIsInstance(result.indicators, MarketIndicators)
        self.assertEqual(len(result.regime_probabilities), len(MarketRegime))
    
    def test_bull_market_detection(self):
        """测试牛市检测"""
        # 创建明显的上涨趋势数据
        bull_prices = [100, 102, 105, 108, 112, 116, 120, 125, 130, 135, 140, 145, 150, 155, 160]
        
        for i, price in enumerate(bull_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            result = self.detector.update_market_data(market_data)
        
        # 应该检测到牛市或至少有较高的牛市概率
        self.assertTrue(
            result.regime == MarketRegime.BULL_MARKET or 
            result.regime_probabilities[MarketRegime.BULL_MARKET] > 0.3
        )
    
    def test_bear_market_detection(self):
        """测试熊市检测"""
        # 创建明显的下跌趋势数据
        bear_prices = [160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90]
        
        for i, price in enumerate(bear_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            result = self.detector.update_market_data(market_data)
        
        # 应该检测到熊市或至少有较高的熊市概率
        self.assertTrue(
            result.regime == MarketRegime.BEAR_MARKET or 
            result.regime_probabilities[MarketRegime.BEAR_MARKET] > 0.3
        )
    
    def test_high_volatility_detection(self):
        """测试高波动率检测"""
        # 创建高波动率数据
        volatile_prices = [100, 110, 95, 115, 85, 120, 80, 125, 75, 130, 70, 135, 65, 140, 60]
        
        for i, price in enumerate(volatile_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            result = self.detector.update_market_data(market_data)
        
        # 应该检测到高波动率
        self.assertTrue(
            result.regime == MarketRegime.HIGH_VOLATILITY or
            result.indicators.volatility > self.config.high_vol_threshold
        )
    
    def test_calculate_indicators(self):
        """测试指标计算"""
        # 添加足够的数据
        for i, price in enumerate(self.test_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            self.detector.update_market_data(market_data)
        
        indicators = self.detector._calculate_indicators()
        
        self.assertIsInstance(indicators, MarketIndicators)
        self.assertGreater(indicators.ma_short, 0)
        self.assertGreater(indicators.ma_long, 0)
        self.assertGreaterEqual(indicators.rsi, 0)
        self.assertLessEqual(indicators.rsi, 100)
        self.assertGreaterEqual(indicators.volatility, 0)
        self.assertGreaterEqual(indicators.price_position, 0)
        self.assertLessEqual(indicators.price_position, 1)
    
    def test_rsi_calculation(self):
        """测试RSI计算"""
        prices = np.array([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])
        rsi = self.detector._calculate_rsi(prices, 5)
        
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        self.assertIsInstance(rsi, float)
    
    def test_bollinger_bands_calculation(self):
        """测试布林带计算"""
        prices = np.array(self.test_prices)
        upper, lower, width = self.detector._calculate_bollinger_bands(prices, 5, 2.0)
        
        self.assertGreater(upper, lower)
        self.assertGreater(width, 0)
        self.assertEqual(width, upper - lower)
    
    def test_volatility_calculation(self):
        """测试波动率计算"""
        prices = np.array(self.test_prices)
        volatility = self.detector._calculate_volatility(prices, 5)
        
        self.assertGreaterEqual(volatility, 0)
        self.assertIsInstance(volatility, float)
    
    def test_trend_indicators_calculation(self):
        """测试趋势指标计算"""
        # 上涨趋势
        uptrend_prices = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
        strength, direction = self.detector._calculate_trend_indicators(uptrend_prices)
        
        self.assertGreaterEqual(strength, 0)
        self.assertLessEqual(strength, 1)
        self.assertIn(direction, [-1, 0, 1])
        
        # 对于明显的上涨趋势，方向应该是1
        if strength > 0.5:
            self.assertEqual(direction, 1)
    
    def test_regime_persistence(self):
        """测试状态持续性检查"""
        # 设置较短的持续性要求
        self.detector.config.regime_persistence = 2
        
        # 第一次检测
        regime1 = self.detector._check_regime_persistence(MarketRegime.BULL_MARKET, 0.8)
        
        # 第二次检测相同状态
        regime2 = self.detector._check_regime_persistence(MarketRegime.BULL_MARKET, 0.8)
        
        # 应该确认为牛市
        self.assertEqual(regime2, MarketRegime.BULL_MARKET)
    
    def test_risk_adjustment_factor(self):
        """测试风险调整因子计算"""
        # 创建测试指标
        indicators = MarketIndicators(
            timestamp=self.base_time,
            price=100,
            ma_short=100,
            ma_long=100,
            price_position=0.5,
            rsi=50,
            bollinger_upper=102,
            bollinger_lower=98,
            bollinger_width=4,
            volatility=0.02,
            volatility_percentile=0.5,
            trend_strength=0.5,
            trend_direction=0,
            correlation_level=0.5,
            market_stress=0.3
        )
        
        # 测试不同状态的风险调整
        bull_factor = self.detector._calculate_risk_adjustment_factor(MarketRegime.BULL_MARKET, indicators)
        bear_factor = self.detector._calculate_risk_adjustment_factor(MarketRegime.BEAR_MARKET, indicators)
        crisis_factor = self.detector._calculate_risk_adjustment_factor(MarketRegime.CRISIS, indicators)
        
        self.assertGreater(bull_factor, bear_factor)
        self.assertGreater(bear_factor, crisis_factor)
        self.assertGreaterEqual(bull_factor, 0.1)
        self.assertLessEqual(bull_factor, 2.0)
    
    def test_recommended_actions_generation(self):
        """测试推荐行动生成"""
        indicators = MarketIndicators(
            timestamp=self.base_time,
            price=100,
            ma_short=100,
            ma_long=100,
            price_position=0.5,
            rsi=50,
            bollinger_upper=102,
            bollinger_lower=98,
            bollinger_width=4,
            volatility=0.02,
            volatility_percentile=0.5,
            trend_strength=0.5,
            trend_direction=0,
            correlation_level=0.5,
            market_stress=0.3
        )
        
        actions = self.detector._generate_recommended_actions(MarketRegime.BULL_MARKET, indicators)
        
        self.assertIsInstance(actions, list)
        self.assertGreater(len(actions), 0)
        self.assertTrue(all(isinstance(action, str) for action in actions))
    
    def test_adjust_risk_parameters(self):
        """测试风险参数调整"""
        base_params = {
            'max_position_size': 0.1,
            'stop_loss_threshold': 0.05,
            'volatility_target': 0.15
        }
        
        # 添加一些历史数据以便计算风险因子
        for i, price in enumerate(self.test_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            self.detector.update_market_data(market_data)
        
        adjusted_params = self.detector.adjust_risk_parameters(MarketRegime.CRISIS, base_params)
        
        self.assertIsInstance(adjusted_params, dict)
        self.assertIn('max_position_size', adjusted_params)
        self.assertIn('stop_loss_threshold', adjusted_params)
        self.assertIn('volatility_target', adjusted_params)
        
        # 危机模式下应该降低风险
        self.assertLessEqual(adjusted_params['max_position_size'], base_params['max_position_size'])
    
    def test_get_current_regime(self):
        """测试获取当前状态"""
        self.assertIsNone(self.detector.get_current_regime())
        
        # 添加数据并更新状态
        for i, price in enumerate(self.test_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            self.detector.update_market_data(market_data)
        
        current_regime = self.detector.get_current_regime()
        self.assertIsInstance(current_regime, MarketRegime)
    
    def test_get_regime_duration(self):
        """测试获取状态持续时间"""
        self.assertIsNone(self.detector.get_regime_duration())
        
        # 添加数据
        for i, price in enumerate(self.test_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            self.detector.update_market_data(market_data)
        
        duration = self.detector.get_regime_duration()
        if duration is not None:
            self.assertIsInstance(duration, timedelta)
    
    def test_get_market_stress_level(self):
        """测试获取市场压力水平"""
        self.assertEqual(self.detector.get_market_stress_level(), 0.0)
        
        # 添加数据
        for i, price in enumerate(self.test_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            self.detector.update_market_data(market_data)
        
        stress_level = self.detector.get_market_stress_level()
        self.assertGreaterEqual(stress_level, 0.0)
        self.assertLessEqual(stress_level, 1.0)
    
    def test_is_crisis_mode(self):
        """测试危机模式判断"""
        self.assertFalse(self.detector.is_crisis_mode())
        
        # 手动设置为危机模式
        self.detector.current_regime = MarketRegime.CRISIS
        self.assertTrue(self.detector.is_crisis_mode())
    
    def test_get_regime_statistics(self):
        """测试获取状态统计"""
        # 初始状态应该返回空字典
        stats = self.detector.get_regime_statistics()
        self.assertEqual(stats, {})
        
        # 添加数据
        for i, price in enumerate(self.test_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            self.detector.update_market_data(market_data)
        
        stats = self.detector.get_regime_statistics()
        
        self.assertIn('current_regime', stats)
        self.assertIn('regime_confidence', stats)
        self.assertIn('regime_frequencies', stats)
        self.assertIn('avg_volatility', stats)
        self.assertIn('total_observations', stats)
    
    def test_reset(self):
        """测试重置功能"""
        # 添加一些数据
        for i, price in enumerate(self.test_prices[:5]):
            timestamp = self.base_time + timedelta(days=i)
            market_data = self._create_market_data(price, timestamp)
            self.detector.update_market_data(market_data)
        
        self.assertGreater(len(self.detector.price_history), 0)
        
        # 重置
        self.detector.reset()
        
        self.assertEqual(len(self.detector.price_history), 0)
        self.assertEqual(len(self.detector.indicators_history), 0)
        self.assertIsNone(self.detector.current_regime)
        self.assertEqual(self.detector.regime_confidence, 0.0)


class TestMarketRegimeAnalyzer(unittest.TestCase):
    """市场状态分析器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = MarketRegimeConfig(
            ma_short_period=5,
            ma_long_period=10,
            volatility_window=5
        )
        self.detector = MarketRegimeDetector(self.config)
        self.analyzer = MarketRegimeAnalyzer(self.detector)
        
        self.base_time = datetime(2024, 1, 1)
        self.test_prices = [100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116]
    
    def _create_market_data_list(self) -> List[MarketData]:
        """创建测试用市场数据列表"""
        market_data_list = []
        for i, price in enumerate(self.test_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = MarketData(
                timestamp=timestamp,
                symbol="TEST",
                open_price=price * 0.99,
                high_price=price * 1.01,
                low_price=price * 0.98,
                close_price=price,
                volume=1000000,
                amount=price * 1000000
            )
            market_data_list.append(market_data)
        return market_data_list
    
    def test_analyze_historical_regimes(self):
        """测试历史状态分析"""
        market_data_list = self._create_market_data_list()
        df = self.analyzer.analyze_historical_regimes(market_data_list)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        expected_columns = [
            'timestamp', 'regime', 'confidence', 'risk_adjustment_factor',
            'price', 'volatility', 'trend_strength', 'trend_direction',
            'market_stress', 'rsi'
        ]
        
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_evaluate_regime_accuracy(self):
        """测试状态识别准确性评估"""
        market_data_list = self._create_market_data_list()
        df = self.analyzer.analyze_historical_regimes(market_data_list)
        
        # 创建模拟的实际状态
        actual_regimes = [MarketRegime.BULL_MARKET] * len(df)
        
        accuracy_metrics = self.analyzer.evaluate_regime_accuracy(df, actual_regimes)
        
        self.assertIn('overall_accuracy', accuracy_metrics)
        self.assertIn('regime_metrics', accuracy_metrics)
        self.assertGreaterEqual(accuracy_metrics['overall_accuracy'], 0)
        self.assertLessEqual(accuracy_metrics['overall_accuracy'], 1)
        
        # 检查各状态的指标
        for regime in MarketRegime:
            if regime.value in accuracy_metrics['regime_metrics']:
                metrics = accuracy_metrics['regime_metrics'][regime.value]
                self.assertIn('precision', metrics)
                self.assertIn('recall', metrics)
                self.assertIn('f1_score', metrics)
    
    def test_generate_regime_report(self):
        """测试生成状态分析报告"""
        market_data_list = self._create_market_data_list()
        df = self.analyzer.analyze_historical_regimes(market_data_list)
        
        report = self.analyzer.generate_regime_report(df)
        
        self.assertIsInstance(report, str)
        self.assertIn("市场状态分析报告", report)
        self.assertIn("分析期间", report)
        self.assertIn("各市场状态占比", report)
        self.assertIn("平均指标", report)
    
    def test_generate_regime_report_empty_data(self):
        """测试空数据的报告生成"""
        empty_df = pd.DataFrame()
        report = self.analyzer.generate_regime_report(empty_df)
        
        self.assertEqual(report, "无历史数据可供分析")
    
    def test_analyze_regime_transitions(self):
        """测试状态转换分析"""
        # 创建包含状态转换的测试数据
        test_data = pd.DataFrame({
            'regime': ['bull', 'bull', 'sideways', 'bear', 'bear', 'crisis', 'bull']
        })
        
        transitions = self.analyzer._analyze_regime_transitions(test_data)
        
        self.assertIsInstance(transitions, dict)
        self.assertIn('bull -> sideways', transitions)
        self.assertIn('sideways -> bear', transitions)
        self.assertIn('crisis -> bull', transitions)


class TestMarketRegimeConfig(unittest.TestCase):
    """市场状态配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MarketRegimeConfig()
        
        self.assertEqual(config.ma_short_period, 20)
        self.assertEqual(config.ma_long_period, 60)
        self.assertEqual(config.rsi_period, 14)
        self.assertEqual(config.high_vol_threshold, 0.02)
        self.assertEqual(config.low_vol_threshold, 0.01)
        self.assertEqual(config.regime_persistence, 5)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = MarketRegimeConfig(
            ma_short_period=10,
            ma_long_period=30,
            high_vol_threshold=0.03,
            regime_persistence=3
        )
        
        self.assertEqual(config.ma_short_period, 10)
        self.assertEqual(config.ma_long_period, 30)
        self.assertEqual(config.high_vol_threshold, 0.03)
        self.assertEqual(config.regime_persistence, 3)


class TestMarketRegimeIntegration(unittest.TestCase):
    """市场状态系统集成测试"""
    
    def setUp(self):
        """测试初始化"""
        # 使用适合测试的配置
        config = MarketRegimeConfig(
            ma_short_period=5,
            ma_long_period=10,
            volatility_window=5,
            regime_persistence=2
        )
        self.detector = MarketRegimeDetector(config)
        self.base_time = datetime(2024, 1, 1)
    
    def test_full_workflow(self):
        """测试完整工作流程"""
        # 创建不同市场状态的数据
        
        # 1. 牛市数据
        bull_prices = [100, 102, 105, 108, 112, 116, 120, 125, 130, 135]
        for i, price in enumerate(bull_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = MarketData(
                timestamp=timestamp,
                symbol="TEST",
                open_price=price * 0.99,
                high_price=price * 1.01,
                low_price=price * 0.98,
                close_price=price,
                volume=1000000,
                amount=price * 1000000
            )
            result = self.detector.update_market_data(market_data)
        
        # 检查是否检测到上涨趋势
        self.assertIsInstance(result, RegimeDetectionResult)
        
        # 2. 添加高波动率数据
        volatile_prices = [135, 145, 125, 150, 120, 155, 115, 160, 110, 165]
        for i, price in enumerate(volatile_prices):
            timestamp = self.base_time + timedelta(days=len(bull_prices) + i)
            market_data = MarketData(
                timestamp=timestamp,
                symbol="TEST",
                open_price=price * 0.99,
                high_price=price * 1.01,
                low_price=price * 0.98,
                close_price=price,
                volume=1000000,
                amount=price * 1000000
            )
            result = self.detector.update_market_data(market_data)
        
        # 检查是否检测到高波动率
        self.assertGreater(result.indicators.volatility, 0)
        
        # 3. 测试风险参数调整
        base_params = {
            'max_position_size': 0.1,
            'stop_loss_threshold': 0.05
        }
        
        adjusted_params = self.detector.adjust_risk_parameters(result.regime, base_params)
        self.assertIsInstance(adjusted_params, dict)
        
        # 4. 测试统计信息
        stats = self.detector.get_regime_statistics()
        self.assertIn('current_regime', stats)
        self.assertGreater(stats['total_observations'], 0)
    
    def test_crisis_detection_and_response(self):
        """测试危机检测和响应"""
        # 创建危机场景数据（大幅下跌 + 高波动）
        crisis_prices = [100, 95, 85, 90, 75, 80, 65, 70, 55, 60, 45, 50, 35, 40, 25]
        
        for i, price in enumerate(crisis_prices):
            timestamp = self.base_time + timedelta(days=i)
            market_data = MarketData(
                timestamp=timestamp,
                symbol="TEST",
                open_price=price * 0.99,
                high_price=price * 1.05,  # 更大的价格波动
                low_price=price * 0.95,
                close_price=price,
                volume=2000000,  # 更高的成交量
                amount=price * 2000000
            )
            result = self.detector.update_market_data(market_data)
        
        # 检查是否检测到危机或高风险状态
        self.assertTrue(
            result.regime in [MarketRegime.CRISIS, MarketRegime.BEAR_MARKET, MarketRegime.HIGH_VOLATILITY]
        )
        
        # 检查风险调整因子是否降低
        self.assertLess(result.risk_adjustment_factor, 1.0)
        
        # 检查推荐行动是否包含风险控制措施
        risk_actions = [action for action in result.recommended_actions 
                       if any(keyword in action for keyword in ['降低', '减仓', '止损', '风险'])]
        self.assertGreater(len(risk_actions), 0)


if __name__ == '__main__':
    unittest.main()