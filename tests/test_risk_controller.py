"""
风险控制器测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_control import RiskController, TargetVolatilityController, RiskParityOptimizer, DynamicStopLoss


class TestRiskController(unittest.TestCase):
    """风险控制器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'target_volatility': 0.12,
            'max_position': 0.1,
            'max_leverage': 1.2,
            'max_drawdown_threshold': 0.10,
            'enable_risk_parity': False,
            'alpha_weight': 0.7
        }
        
        self.risk_controller = RiskController(self.config)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        n_stocks = 5
        
        np.random.seed(42)
        self.price_data = pd.DataFrame(
            index=dates,
            columns=[f'stock_{i}' for i in range(n_stocks)],
            data=100 + np.cumsum(np.random.normal(0, 0.02, (len(dates), n_stocks)), axis=0)
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.risk_controller.target_volatility, 0.12)
        self.assertEqual(self.risk_controller.max_position, 0.1)
        self.assertEqual(self.risk_controller.max_leverage, 1.2)
        
        # 验证子模块初始化
        self.assertIsNotNone(self.risk_controller.target_vol_controller)
        self.assertIsNotNone(self.risk_controller.risk_parity_optimizer)
        self.assertIsNotNone(self.risk_controller.stop_loss_manager)
    
    def test_apply_basic_constraints(self):
        """测试基础约束"""
        # 测试单股票仓位限制
        weights = np.array([0.15, -0.12, 0.08, 0.05, -0.03])
        constrained = self.risk_controller._apply_basic_constraints(weights)
        
        # 验证单股票仓位限制
        self.assertTrue(np.all(np.abs(constrained) <= self.risk_controller.max_position))
        
        # 测试总杠杆限制
        high_leverage = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        constrained_leverage = self.risk_controller._apply_basic_constraints(high_leverage)
        
        # 验证总杠杆限制
        total_leverage = np.sum(np.abs(constrained_leverage))
        self.assertLessEqual(total_leverage, self.risk_controller.max_leverage + 1e-10)
    
    def test_calculate_portfolio_risk(self):
        """测试组合风险计算"""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        risk_metrics = self.risk_controller.calculate_portfolio_risk(weights, self.price_data)
        
        # 验证返回的指标
        expected_keys = [
            'volatility', 'var_95', 'cvar_95', 'max_drawdown', 
            'sharpe_ratio', 'total_leverage', 'max_position', 'num_positions'
        ]
        
        for key in expected_keys:
            self.assertIn(key, risk_metrics)
            self.assertIsInstance(risk_metrics[key], (int, float))
        
        # 验证杠杆计算
        self.assertAlmostEqual(risk_metrics['total_leverage'], np.sum(np.abs(weights)), places=6)
        self.assertAlmostEqual(risk_metrics['max_position'], np.max(np.abs(weights)), places=6)
    
    def test_check_risk_limits(self):
        """测试风险限制检查"""
        # 测试正常权重
        normal_weights = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        is_valid, violations = self.risk_controller.check_risk_limits(normal_weights, self.price_data)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(violations), 0)
        
        # 测试超限权重
        over_limit_weights = np.array([0.15, 0.15, 0.15, 0.15, 0.15])
        is_valid, violations = self.risk_controller.check_risk_limits(over_limit_weights, self.price_data)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(violations), 0)
    
    def test_process_weights(self):
        """测试权重处理"""
        raw_weights = np.array([0.08, -0.06, 0.04, 0.03, -0.02])
        
        processed_weights = self.risk_controller.process_weights(
            raw_weights, self.price_data, 1.0, {'max_drawdown': 0.02}
        )
        
        # 验证处理后权重满足约束
        self.assertTrue(np.all(np.abs(processed_weights) <= self.risk_controller.max_position))
        self.assertLessEqual(np.sum(np.abs(processed_weights)), self.risk_controller.max_leverage)
        
        # 验证权重形状保持
        self.assertEqual(len(processed_weights), len(raw_weights))
    
    def test_final_safety_check(self):
        """测试最终安全检查"""
        weights = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
        
        # 测试正常状态
        normal_state = {'max_drawdown': 0.02}
        safe_weights = self.risk_controller._final_safety_check(weights, normal_state)
        np.testing.assert_array_almost_equal(safe_weights, weights)
        
        # 测试高回撤状态
        high_drawdown_state = {'max_drawdown': 0.09}
        adjusted_weights = self.risk_controller._final_safety_check(weights, high_drawdown_state)
        
        # 权重应该被缩减
        self.assertTrue(np.all(np.abs(adjusted_weights) < np.abs(weights)))
    
    def test_get_risk_report(self):
        """测试风险报告生成"""
        # 添加一些历史数据
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.risk_controller._update_history(weights, 1.05, {'max_drawdown': 0.02, 'portfolio_volatility': 0.10})
        
        report = self.risk_controller.get_risk_report()
        
        # 验证报告内容
        expected_keys = [
            'current_nav', 'total_return', 'max_drawdown', 
            'avg_volatility', 'total_positions'
        ]
        
        for key in expected_keys:
            self.assertIn(key, report)
        
        self.assertEqual(report['current_nav'], 1.05)
        self.assertEqual(report['total_positions'], 1)


class TestTargetVolatilityController(unittest.TestCase):
    """目标波动率控制器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'target_volatility': 0.12,
            'vol_lookback': 20,
            'max_leverage_multiplier': 2.0,
            'min_leverage_multiplier': 0.5
        }
        
        self.controller = TargetVolatilityController(self.config)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.price_data = pd.DataFrame(
            index=dates,
            columns=['stock_1', 'stock_2'],
            data=100 + np.cumsum(np.random.normal(0, 0.02, (len(dates), 2)), axis=0)
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.controller.target_vol, 0.12)
        self.assertEqual(self.controller.lookback_window, 20)
        self.assertEqual(self.controller.max_leverage_multiplier, 2.0)
        self.assertEqual(self.controller.min_leverage_multiplier, 0.5)
    
    def test_estimate_portfolio_volatility(self):
        """测试组合波动率估计"""
        weights = np.array([0.5, 0.5])
        
        vol = self.controller.estimate_portfolio_volatility(weights, self.price_data)
        
        # 验证波动率为正数
        self.assertGreater(vol, 0)
        self.assertIsInstance(vol, float)
    
    def test_should_adjust(self):
        """测试调整判断"""
        # 当前波动率接近目标
        self.assertFalse(self.controller.should_adjust(0.12, tolerance=0.02))
        
        # 当前波动率偏离目标
        self.assertTrue(self.controller.should_adjust(0.08, tolerance=0.02))
        self.assertTrue(self.controller.should_adjust(0.16, tolerance=0.02))
    
    def test_get_volatility_regime(self):
        """测试波动率状态判断"""
        regime = self.controller.get_volatility_regime(self.price_data)
        
        # 验证返回值在预期范围内
        self.assertIn(regime, ['低', '中', '高'])
    
    def test_adaptive_target_volatility(self):
        """测试自适应目标波动率"""
        adaptive_target = self.controller.adaptive_target_volatility(self.price_data)
        
        # 验证返回值为正数且合理
        self.assertGreater(adaptive_target, 0)
        self.assertLess(adaptive_target, 1)  # 年化波动率应该小于100%


class TestDynamicStopLoss(unittest.TestCase):
    """动态止损测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'stop_loss_pct': 0.03,
            'trailing_stop_pct': 0.05,
            'max_drawdown_stop': 0.08,
            'rebalance_threshold': 0.05,
            'rebalance_frequency': 20
        }
        
        self.stop_loss = DynamicStopLoss(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.stop_loss.stop_loss_pct, 0.03)
        self.assertEqual(self.stop_loss.trailing_stop_pct, 0.05)
        self.assertEqual(self.stop_loss.max_drawdown_stop, 0.08)
        
        self.assertIsNone(self.stop_loss.trailing_high)
        self.assertIsNone(self.stop_loss.initial_value)
        self.assertFalse(self.stop_loss.stop_loss_triggered)
    
    def test_check_stop_loss_no_trigger(self):
        """测试止损检查 - 无触发"""
        # 测试正常上涨情况
        self.assertFalse(self.stop_loss.check_stop_loss(1.0))
        self.assertFalse(self.stop_loss.check_stop_loss(1.05))
        self.assertFalse(self.stop_loss.check_stop_loss(1.02))
        
        # 验证追踪高点更新
        self.assertEqual(self.stop_loss.trailing_high, 1.05)
    
    def test_check_stop_loss_trigger(self):
        """测试止损检查 - 触发"""
        # 设置初始值
        self.stop_loss.check_stop_loss(1.0)
        self.stop_loss.check_stop_loss(1.1)  # 新高
        
        # 测试移动止损触发
        triggered = self.stop_loss.check_stop_loss(1.04)  # 从1.1跌到1.04，跌幅>5%
        
        self.assertTrue(triggered)
        self.assertTrue(self.stop_loss.stop_loss_triggered)
        self.assertEqual(len(self.stop_loss.trigger_history), 1)
    
    def test_check_rebalance_signal(self):
        """测试再平衡信号"""
        current_weights = np.array([0.2, 0.3, 0.5])
        target_weights = np.array([0.25, 0.35, 0.4])
        
        # 测试权重偏离触发
        should_rebalance = self.stop_loss.check_rebalance_signal(
            current_weights, target_weights, 10
        )
        
        # 计算权重偏离
        deviation = np.sum(np.abs(current_weights - target_weights))
        if deviation > self.stop_loss.rebalance_threshold:
            self.assertTrue(should_rebalance)
        else:
            self.assertFalse(should_rebalance)
    
    def test_calculate_stop_loss_levels(self):
        """测试止损点位计算"""
        # 设置一些历史数据
        self.stop_loss.check_stop_loss(1.0)
        self.stop_loss.check_stop_loss(1.1)
        
        levels = self.stop_loss.calculate_stop_loss_levels(1.05)
        
        # 验证返回的止损点位
        expected_keys = ['fixed_stop', 'trailing_stop', 'max_drawdown_stop']
        for key in expected_keys:
            self.assertIn(key, levels)
            self.assertIsInstance(levels[key], float)
    
    def test_reset_stop_loss(self):
        """测试止损重置"""
        # 设置一些状态
        self.stop_loss.check_stop_loss(1.0)
        self.stop_loss.check_stop_loss(0.95)  # 触发止损
        
        self.assertTrue(self.stop_loss.stop_loss_triggered)
        
        # 重置
        self.stop_loss.reset_stop_loss(1.1)
        
        # 验证重置结果
        self.assertEqual(self.stop_loss.initial_value, 1.1)
        self.assertEqual(self.stop_loss.trailing_high, 1.1)
        self.assertFalse(self.stop_loss.stop_loss_triggered)


if __name__ == '__main__':
    unittest.main()