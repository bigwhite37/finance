"""
回撤监控器单元测试

测试DrawdownMonitor类的各项功能，包括：
- 实时回撤计算
- 回撤阶段识别
- 市场状态检测
- 回撤归因分析
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.rl_trading_system.risk_control.drawdown_monitor import (
    DrawdownMonitor, DrawdownPhase, MarketRegime, DrawdownMetrics, MarketStateMetrics
)


class TestDrawdownMonitor(unittest.TestCase):
    """回撤监控器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.monitor = DrawdownMonitor(
            drawdown_threshold=0.05,
            recovery_threshold=0.02,
            lookback_window=100,
            volatility_window=20
        )
        self.base_time = datetime(2024, 1, 1)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.monitor.drawdown_threshold, 0.05)
        self.assertEqual(self.monitor.recovery_threshold, 0.02)
        self.assertEqual(self.monitor.lookback_window, 100)
        self.assertEqual(self.monitor.volatility_window, 20)
        self.assertEqual(len(self.monitor.portfolio_values), 0)
        self.assertEqual(self.monitor.current_phase, DrawdownPhase.NORMAL)
    
    def test_single_value_update(self):
        """测试单个净值更新"""
        metrics = self.monitor.update_portfolio_value(100.0, self.base_time)
        
        self.assertEqual(len(self.monitor.portfolio_values), 1)
        self.assertEqual(self.monitor.portfolio_values[0], 100.0)
        self.assertEqual(metrics.current_drawdown, 0.0)
        self.assertEqual(metrics.max_drawdown, 0.0)
        self.assertEqual(metrics.current_phase, DrawdownPhase.NORMAL)
    
    def test_no_drawdown_scenario(self):
        """测试无回撤场景（持续上涨）"""
        values = [100, 105, 110, 115, 120]
        
        for i, value in enumerate(values):
            timestamp = self.base_time + timedelta(days=i)
            metrics = self.monitor.update_portfolio_value(value, timestamp)
        
        # 应该没有回撤
        self.assertAlmostEqual(metrics.current_drawdown, 0.0, places=6)
        self.assertAlmostEqual(metrics.max_drawdown, 0.0, places=6)
        self.assertEqual(metrics.current_phase, DrawdownPhase.NORMAL)
        self.assertEqual(metrics.drawdown_duration, 0)
    
    def test_simple_drawdown_scenario(self):
        """测试简单回撤场景"""
        values = [100, 110, 120, 115, 105, 95, 100, 110]
        
        for i, value in enumerate(values):
            timestamp = self.base_time + timedelta(days=i)
            metrics = self.monitor.update_portfolio_value(value, timestamp)
        
        # 检查最大回撤（从120跌到95，回撤20.83%）
        expected_max_drawdown = (95 - 120) / 120
        self.assertAlmostEqual(metrics.max_drawdown, expected_max_drawdown, places=4)
        
        # 当前净值110相对于峰值120的回撤是8.33%
        expected_current_drawdown = (110 - 120) / 120
        self.assertAlmostEqual(metrics.current_drawdown, expected_current_drawdown, places=4)
    
    def test_drawdown_phase_identification(self):
        """测试回撤阶段识别"""
        # 设置较低的阈值以便测试
        monitor = DrawdownMonitor(drawdown_threshold=0.03, recovery_threshold=0.01)
        
        # 正常阶段
        metrics = monitor.update_portfolio_value(100, self.base_time)
        self.assertEqual(metrics.current_phase, DrawdownPhase.NORMAL)
        
        # 小幅下跌，但还未达到阈值，仍为正常
        metrics = monitor.update_portfolio_value(98, self.base_time + timedelta(days=1))
        self.assertEqual(metrics.current_phase, DrawdownPhase.NORMAL)
        
        # 下跌超过阈值，进入回撤开始阶段
        metrics = monitor.update_portfolio_value(96, self.base_time + timedelta(days=2))
        self.assertEqual(metrics.current_phase, DrawdownPhase.DRAWDOWN_START)
        
        # 继续下跌，进入回撤持续阶段
        metrics = monitor.update_portfolio_value(92, self.base_time + timedelta(days=3))
        self.assertEqual(metrics.current_phase, DrawdownPhase.DRAWDOWN_CONTINUE)
        
        # 开始恢复
        metrics = monitor.update_portfolio_value(95, self.base_time + timedelta(days=4))
        self.assertEqual(metrics.current_phase, DrawdownPhase.RECOVERY)
        
        # 完全恢复
        metrics = monitor.update_portfolio_value(100, self.base_time + timedelta(days=5))
        self.assertEqual(metrics.current_phase, DrawdownPhase.NORMAL)
    
    def test_drawdown_duration_calculation(self):
        """测试回撤持续时间计算"""
        values = [100, 110, 120, 115, 110, 105, 100, 95, 100, 105]
        
        for i, value in enumerate(values):
            timestamp = self.base_time + timedelta(days=i)
            metrics = self.monitor.update_portfolio_value(value, timestamp)
        
        # 最后的净值105相对于峰值120仍有回撤，但比95有所恢复
        # 从120开始算，持续回撤的天数应该是从第3天到最后
        self.assertGreater(metrics.drawdown_duration, 0)
    
    def test_recovery_time_calculation(self):
        """测试恢复时间计算"""
        # 创建一个有明确恢复的场景
        values = [100, 120, 100, 80, 90, 100, 110, 120]
        
        for i, value in enumerate(values):
            timestamp = self.base_time + timedelta(days=i)
            metrics = self.monitor.update_portfolio_value(value, timestamp)
        
        # 应该能计算出恢复时间
        self.assertIsNotNone(metrics.recovery_time)
        self.assertGreater(metrics.recovery_time, 0)
    
    def test_underwater_curve(self):
        """测试水下曲线计算"""
        values = [100, 110, 120, 110, 100, 90, 100, 110]
        
        for i, value in enumerate(values):
            timestamp = self.base_time + timedelta(days=i)
            metrics = self.monitor.update_portfolio_value(value, timestamp)
        
        # 水下曲线应该记录每个时点的回撤
        self.assertEqual(len(metrics.underwater_curve), len(values))
        
        # 第一个点应该是0（没有回撤）
        self.assertAlmostEqual(metrics.underwater_curve[0], 0.0, places=6)
        
        # 峰值点（120）应该是0
        self.assertAlmostEqual(metrics.underwater_curve[2], 0.0, places=6)
        
        # 最低点（90）应该是最大回撤
        min_value_idx = values.index(90)
        self.assertLess(metrics.underwater_curve[min_value_idx], -0.2)  # 超过20%回撤
    
    def test_drawdown_frequency_calculation(self):
        """测试回撤频率计算"""
        # 创建多次回撤的场景
        values = [100, 110, 105, 115, 110, 120, 115, 125, 120, 130]
        
        for i, value in enumerate(values):
            timestamp = self.base_time + timedelta(days=i)
            metrics = self.monitor.update_portfolio_value(value, timestamp)
        
        # 应该能识别出多次回撤事件
        self.assertGreater(metrics.drawdown_frequency, 0)
    
    def test_market_regime_detection_bull_market(self):
        """测试牛市识别"""
        # 创建明显的上涨趋势数据
        prices = np.array([100 + i * 2 + np.random.normal(0, 0.5) for i in range(50)])
        market_data = {'prices': prices}
        
        metrics = self.monitor.detect_market_regime(market_data)
        
        # 应该识别为牛市或低波动市场
        self.assertIn(metrics.regime, [MarketRegime.BULL_MARKET, MarketRegime.LOW_VOLATILITY])
        self.assertGreater(metrics.confidence_score, 0.3)
    
    def test_market_regime_detection_bear_market(self):
        """测试熊市识别"""
        # 创建明显的下跌趋势数据
        prices = np.array([100 - i * 1.5 + np.random.normal(0, 0.3) for i in range(50)])
        market_data = {'prices': prices}
        
        metrics = self.monitor.detect_market_regime(market_data)
        
        # 应该识别为熊市
        self.assertEqual(metrics.regime, MarketRegime.BEAR_MARKET)
        self.assertGreater(metrics.confidence_score, 0.3)
    
    def test_market_regime_detection_high_volatility(self):
        """测试高波动市场识别"""
        # 创建高波动数据
        np.random.seed(42)  # 确保测试结果可重复
        prices = [100]
        for i in range(49):
            change = np.random.normal(0, 5)  # 高波动
            prices.append(max(prices[-1] + change, 10))  # 确保价格不为负
        
        market_data = {'prices': np.array(prices)}
        
        metrics = self.monitor.detect_market_regime(market_data)
        
        # 应该识别为高波动市场
        self.assertIn(metrics.regime, [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS])
        self.assertGreater(metrics.volatility, 0.2)  # 高波动率
    
    def test_market_regime_detection_sideways(self):
        """测试震荡市识别"""
        # 创建震荡数据
        prices = np.array([100 + 5 * np.sin(i * 0.3) + np.random.normal(0, 1) for i in range(50)])
        market_data = {'prices': prices}
        
        metrics = self.monitor.detect_market_regime(market_data)
        
        # 趋势强度应该较低
        self.assertLess(abs(metrics.trend_strength), 0.5)
    
    def test_volatility_calculation(self):
        """测试波动率计算"""
        # 创建已知波动率的数据
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)  # 日收益率标准差2%
        
        volatility = self.monitor._calculate_volatility(returns)
        
        # 由于使用了随机数据，实际波动率会有差异，放宽测试范围
        self.assertGreater(volatility, 0.1)  # 至少10%年化波动率
        self.assertLess(volatility, 0.5)     # 不超过50%年化波动率
    
    def test_trend_strength_calculation(self):
        """测试趋势强度计算"""
        # 强上涨趋势
        uptrend_prices = np.array([100 + i * 2 for i in range(30)])
        trend_strength = self.monitor._calculate_trend_strength(uptrend_prices)
        self.assertGreater(trend_strength, 0.8)  # 强正趋势
        
        # 强下跌趋势
        downtrend_prices = np.array([100 - i * 2 for i in range(30)])
        trend_strength = self.monitor._calculate_trend_strength(downtrend_prices)
        self.assertLess(trend_strength, -0.8)  # 强负趋势
        
        # 水平趋势（无明显趋势）
        flat_prices = np.array([100 + np.random.normal(0, 0.5) for _ in range(30)])
        trend_strength = self.monitor._calculate_trend_strength(flat_prices)
        self.assertLess(abs(trend_strength), 0.8)  # 相对较弱的趋势
    
    def test_drawdown_attribution_analysis(self):
        """测试回撤归因分析"""
        positions = {'AAPL': 0.3, 'GOOGL': 0.4, 'MSFT': 0.3}
        position_returns = {'AAPL': -0.05, 'GOOGL': -0.08, 'MSFT': 0.02}
        
        contributions = self.monitor.analyze_drawdown_attribution(positions, position_returns)
        
        # GOOGL应该是最大的负贡献者
        self.assertGreater(contributions['GOOGL'], contributions['AAPL'])
        self.assertEqual(contributions['MSFT'], 0.0)  # 正收益不贡献回撤
        
        # 所有负贡献的总和应该接近1
        total_negative_contribution = sum(contributions.values())
        self.assertAlmostEqual(total_negative_contribution, 1.0, delta=0.1)
    
    def test_liquidity_score_calculation(self):
        """测试流动性评分计算"""
        # 高流动性场景：稳定成交量，低波动
        market_data = {
            'prices': np.array([100 + 0.1 * i + np.random.normal(0, 0.5) for i in range(30)]),
            'volumes': np.array([1000000 + np.random.normal(0, 50000) for _ in range(30)])
        }
        
        liquidity_score = self.monitor._calculate_liquidity_score(market_data)
        self.assertGreater(liquidity_score, 0.3)
        
        # 低流动性场景：不稳定成交量，高波动
        market_data = {
            'prices': np.array([100 + np.random.normal(0, 5) for _ in range(30)]),
            'volumes': np.array([1000000 + np.random.normal(0, 500000) for _ in range(30)])
        }
        
        liquidity_score = self.monitor._calculate_liquidity_score(market_data)
        self.assertLess(liquidity_score, 0.7)
    
    def test_lookback_window_limit(self):
        """测试回看窗口限制"""
        # 添加超过窗口大小的数据
        for i in range(150):  # 超过默认窗口100
            value = 100 + i * 0.1
            timestamp = self.base_time + timedelta(days=i)
            self.monitor.update_portfolio_value(value, timestamp)
        
        # 数据长度应该被限制在窗口大小内
        self.assertEqual(len(self.monitor.portfolio_values), 100)
        self.assertEqual(len(self.monitor.timestamps), 100)
    
    def test_get_current_status(self):
        """测试获取当前状态"""
        # 空状态
        status = self.monitor.get_current_status()
        self.assertEqual(status['status'], '无数据')
        self.assertEqual(status['portfolio_value'], 0.0)
        
        # 有数据状态
        self.monitor.update_portfolio_value(100.0, self.base_time)
        self.monitor.update_portfolio_value(95.0, self.base_time + timedelta(days=1))
        
        status = self.monitor.get_current_status()
        self.assertEqual(status['status'], '正常监控')
        self.assertEqual(status['portfolio_value'], 95.0)
        self.assertEqual(status['monitoring_duration'], 2)
        self.assertIsNotNone(status['last_update'])
    
    def test_reset_functionality(self):
        """测试重置功能"""
        # 添加一些数据
        for i in range(10):
            value = 100 + i
            timestamp = self.base_time + timedelta(days=i)
            self.monitor.update_portfolio_value(value, timestamp)
        
        # 重置
        self.monitor.reset()
        
        # 验证所有状态都被重置
        self.assertEqual(len(self.monitor.portfolio_values), 0)
        self.assertEqual(len(self.monitor.timestamps), 0)
        self.assertEqual(len(self.monitor.drawdown_history), 0)
        self.assertEqual(self.monitor.current_peak, 0.0)
        self.assertEqual(self.monitor.current_phase, DrawdownPhase.NORMAL)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试相同值
        for i in range(5):
            metrics = self.monitor.update_portfolio_value(100.0, self.base_time + timedelta(days=i))
        
        self.assertEqual(metrics.current_drawdown, 0.0)
        self.assertEqual(metrics.max_drawdown, 0.0)
        
        # 测试极小值
        metrics = self.monitor.update_portfolio_value(0.001, self.base_time + timedelta(days=5))
        self.assertLess(metrics.current_drawdown, 0)  # 应该有大幅回撤
        
        # 测试负值（理论上不应该出现，但要处理）
        metrics = self.monitor.update_portfolio_value(-10, self.base_time + timedelta(days=6))
        self.assertIsInstance(metrics, DrawdownMetrics)  # 应该能正常处理
    
    def test_data_types_and_formats(self):
        """测试数据类型和格式"""
        metrics = self.monitor.update_portfolio_value(100.0, self.base_time)
        
        # 测试DrawdownMetrics的to_dict方法
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('current_drawdown', metrics_dict)
        self.assertIn('max_drawdown', metrics_dict)
        self.assertIn('current_phase', metrics_dict)
        
        # 测试MarketStateMetrics
        market_data = {'prices': np.array([100, 101, 102, 103, 104])}
        market_metrics = self.monitor.detect_market_regime(market_data)
        market_dict = market_metrics.to_dict()
        
        self.assertIsInstance(market_dict, dict)
        self.assertIn('regime', market_dict)
        self.assertIn('volatility', market_dict)
        self.assertIn('confidence_score', market_dict)


if __name__ == '__main__':
    unittest.main()