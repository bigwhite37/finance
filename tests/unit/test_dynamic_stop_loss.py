"""动态止损控制器单元测试"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import numpy as np

from src.rl_trading_system.risk_control.dynamic_stop_loss import (
    DynamicStopLoss, StopLossConfig, Position, Portfolio, StopLossOrder,
    StopLossType, StopLossStatus, TrailingStopRecord, PortfolioStopLossRecord
)


class TestDynamicStopLoss(unittest.TestCase):
    """动态止损控制器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = StopLossConfig(
            base_stop_loss=0.05,
            min_stop_loss=0.02,
            max_stop_loss=0.15,
            volatility_multiplier=2.0,
            trailing_stop_distance=0.03,
            trailing_activation_threshold=0.02,
            portfolio_stop_loss=0.12,
            portfolio_partial_stop=0.08,
            partial_stop_ratio=0.3
        )
        self.stop_loss_controller = DynamicStopLoss(self.config)
        
        # 创建测试持仓
        self.test_position = Position(
            symbol="AAPL",
            quantity=100,
            current_price=150.0,
            cost_basis=160.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        # 创建测试投资组合
        self.test_portfolio = Portfolio(
            positions=[self.test_position],
            cash=10000.0,
            total_value=25000.0,
            timestamp=datetime.now()
        )
    
    def test_calculate_fixed_stop_loss(self):
        """测试固定止损计算"""
        stop_price = self.stop_loss_controller._calculate_fixed_stop_loss(self.test_position)
        expected_price = 160.0 * (1 - 0.05)  # 152.0
        self.assertAlmostEqual(stop_price, expected_price, places=2)
    
    def test_calculate_percentage_stop_loss(self):
        """测试百分比止损计算"""
        stop_price = self.stop_loss_controller._calculate_percentage_stop_loss(self.test_position)
        expected_price = 150.0 * (1 - 0.05)  # 142.5
        self.assertAlmostEqual(stop_price, expected_price, places=2)
    
    def test_calculate_volatility_adjusted_stop_loss(self):
        """测试波动率调整止损计算"""
        # 测试高波动率情况
        high_volatility = 0.4
        stop_price = self.stop_loss_controller._calculate_volatility_adjusted_stop_loss(
            self.test_position, high_volatility
        )
        # 高波动率应该放宽止损
        self.assertLess(stop_price, self.stop_loss_controller._calculate_fixed_stop_loss(self.test_position))
        
        # 测试低波动率情况
        low_volatility = 0.05
        stop_price_low = self.stop_loss_controller._calculate_volatility_adjusted_stop_loss(
            self.test_position, low_volatility
        )
        # 低波动率应该收紧止损
        self.assertGreater(stop_price_low, self.stop_loss_controller._calculate_fixed_stop_loss(self.test_position))
    
    def test_calculate_adaptive_stop_loss(self):
        """测试自适应止损计算"""
        stop_levels = self.stop_loss_controller.calculate_adaptive_stop_loss(
            self.test_position, market_volatility=0.2
        )
        
        # 检查返回的止损水平包含所有类型
        expected_keys = ['fixed', 'percentage', 'volatility_adjusted', 'trailing']
        for key in expected_keys:
            self.assertIn(key, stop_levels)
        
        # 检查固定止损和百分比止损的值
        self.assertAlmostEqual(stop_levels['fixed'], 152.0, places=2)
        self.assertAlmostEqual(stop_levels['percentage'], 142.5, places=2)
        
        # 检查止损水平被缓存
        self.assertIn("AAPL", self.stop_loss_controller.stop_loss_levels)
    
    def test_update_trailing_stops(self):
        """测试追踪止损更新"""
        # 创建盈利的持仓
        profitable_position = Position(
            symbol="GOOGL",
            quantity=50,
            current_price=120.0,  # 成本100，当前120，盈利20%
            cost_basis=100.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        updated_stops = self.stop_loss_controller.update_trailing_stops([profitable_position])
        
        # 检查追踪止损是否被设置
        self.assertIn("GOOGL", updated_stops)
        # 由于自适应距离调整，实际距离可能与基础距离不同
        self.assertGreater(updated_stops["GOOGL"], 115.0)  # 应该在合理范围内
        self.assertLess(updated_stops["GOOGL"], 118.0)
        
        # 检查追踪止损记录是否被创建
        self.assertIn("GOOGL", self.stop_loss_controller.trailing_stop_records)
        record = self.stop_loss_controller.trailing_stop_records["GOOGL"]
        self.assertEqual(record.symbol, "GOOGL")
        self.assertEqual(record.initial_price, 100.0)  # 成本基础
        self.assertEqual(record.peak_price, 120.0)     # 当前价格峰值
        
        # 测试价格上涨时追踪止损的更新
        profitable_position.current_price = 130.0
        updated_stops_2 = self.stop_loss_controller.update_trailing_stops([profitable_position])
        
        # 追踪止损应该向上调整
        self.assertGreater(updated_stops_2["GOOGL"], 125.0)  # 应该在合理范围内
        self.assertLess(updated_stops_2["GOOGL"], 128.0)
        self.assertGreater(updated_stops_2["GOOGL"], updated_stops["GOOGL"])
        
        # 检查记录是否被更新
        updated_record = self.stop_loss_controller.trailing_stop_records["GOOGL"]
        self.assertEqual(updated_record.peak_price, 130.0)
        self.assertEqual(updated_record.update_count, 1)
    
    def test_update_trailing_stops_no_profit(self):
        """测试无盈利时不设置追踪止损"""
        # 使用亏损的持仓（当前价格150，成本160）
        updated_stops = self.stop_loss_controller.update_trailing_stops([self.test_position])
        
        # 应该没有设置追踪止损
        self.assertEqual(len(updated_stops), 0)
        self.assertNotIn("AAPL", self.stop_loss_controller.trailing_stops)
        self.assertNotIn("AAPL", self.stop_loss_controller.trailing_stop_records)
    
    def test_trailing_stop_price_decline(self):
        """测试价格下跌时追踪止损不向下调整"""
        # 创建盈利的持仓并设置追踪止损
        profitable_position = Position(
            symbol="MSFT",
            quantity=100,
            current_price=130.0,
            cost_basis=100.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        # 首次设置追踪止损
        self.stop_loss_controller.update_trailing_stops([profitable_position])
        initial_stop = self.stop_loss_controller.trailing_stops["MSFT"]
        
        # 价格下跌
        profitable_position.current_price = 125.0
        updated_stops = self.stop_loss_controller.update_trailing_stops([profitable_position])
        
        # 追踪止损不应该向下调整
        self.assertEqual(len(updated_stops), 0)  # 没有更新
        self.assertEqual(self.stop_loss_controller.trailing_stops["MSFT"], initial_stop)
    
    def test_adaptive_trailing_distance_volatility(self):
        """测试基于波动率的自适应追踪距离"""
        symbol = "VOLATILE_STOCK"
        
        # 设置高波动率
        self.stop_loss_controller.volatility_cache[symbol] = 0.4  # 40%波动率
        
        distance = self.stop_loss_controller._calculate_adaptive_trailing_distance(
            symbol, 100.0, 90.0, 0.11  # 11%利润
        )
        
        # 高波动率应该增加追踪距离
        self.assertGreater(distance, self.config.trailing_stop_distance)
        
        # 设置低波动率
        self.stop_loss_controller.volatility_cache[symbol] = 0.05  # 5%波动率
        
        distance_low = self.stop_loss_controller._calculate_adaptive_trailing_distance(
            symbol, 100.0, 90.0, 0.11
        )
        
        # 低波动率应该减少追踪距离
        self.assertLess(distance_low, self.config.trailing_stop_distance)
    
    def test_adaptive_trailing_distance_profit_tightening(self):
        """测试基于利润的追踪距离收紧"""
        symbol = "PROFIT_STOCK"
        
        # 设置中等波动率
        self.stop_loss_controller.volatility_cache[symbol] = 0.2
        
        # 低利润情况
        distance_low_profit = self.stop_loss_controller._calculate_adaptive_trailing_distance(
            symbol, 100.0, 90.0, 0.05  # 5%利润
        )
        
        # 高利润情况
        distance_high_profit = self.stop_loss_controller._calculate_adaptive_trailing_distance(
            symbol, 100.0, 90.0, 0.15  # 15%利润
        )
        
        # 高利润时应该收紧追踪距离
        self.assertLess(distance_high_profit, distance_low_profit)
    
    def test_get_trailing_stop_details(self):
        """测试获取追踪止损详细信息"""
        # 创建盈利持仓并设置追踪止损
        profitable_position = Position(
            symbol="DETAIL_TEST",
            quantity=100,
            current_price=120.0,
            cost_basis=100.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        self.stop_loss_controller.update_trailing_stops([profitable_position])
        
        # 获取详细信息
        details = self.stop_loss_controller.get_trailing_stop_details("DETAIL_TEST")
        
        # 检查详细信息
        self.assertIsNotNone(details)
        self.assertEqual(details['symbol'], "DETAIL_TEST")
        self.assertEqual(details['initial_price'], 100.0)  # 成本基础
        self.assertEqual(details['peak_price'], 120.0)     # 当前价格峰值
        self.assertGreater(details['profit_locked'], 0)
        self.assertGreater(details['locked_profit_ratio'], 0)
        
        # 测试不存在的股票
        details_none = self.stop_loss_controller.get_trailing_stop_details("NON_EXISTENT")
        self.assertIsNone(details_none)
    
    def test_get_all_trailing_stops_summary(self):
        """测试获取所有追踪止损汇总信息"""
        # 创建多个盈利持仓
        positions = [
            Position("STOCK1", 100, 120.0, 100.0, "Tech", datetime.now()),
            Position("STOCK2", 200, 110.0, 100.0, "Finance", datetime.now()),
            Position("STOCK3", 150, 105.0, 100.0, "Healthcare", datetime.now())
        ]
        
        # 设置追踪止损
        for pos in positions:
            self.stop_loss_controller.update_trailing_stops([pos])
        
        # 获取汇总信息
        summary = self.stop_loss_controller.get_all_trailing_stops_summary()
        
        # 检查汇总信息
        self.assertEqual(summary['total_active_trailing_stops'], 3)
        self.assertGreater(summary['total_locked_profit'], 0)
        self.assertGreater(summary['average_distance_ratio'], 0)
        self.assertEqual(len(summary['symbols']), 3)
        
        # 检查每个股票的信息
        symbols_info = summary['symbols']
        symbols = [info['symbol'] for info in symbols_info]
        self.assertIn("STOCK1", symbols)
        self.assertIn("STOCK2", symbols)
        self.assertIn("STOCK3", symbols)
    
    def test_trailing_stop_removal_on_loss(self):
        """测试亏损时移除追踪止损"""
        # 创建盈利持仓并设置追踪止损
        position = Position(
            symbol="REMOVE_TEST",
            quantity=100,
            current_price=120.0,
            cost_basis=100.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        # 设置追踪止损
        self.stop_loss_controller.update_trailing_stops([position])
        self.assertIn("REMOVE_TEST", self.stop_loss_controller.trailing_stops)
        
        # 价格跌破激活阈值
        position.current_price = 101.0  # 只有1%利润，低于2%阈值
        self.stop_loss_controller.update_trailing_stops([position])
        
        # 追踪止损应该被移除
        self.assertNotIn("REMOVE_TEST", self.stop_loss_controller.trailing_stops)
        self.assertNotIn("REMOVE_TEST", self.stop_loss_controller.trailing_stop_records)
    
    def test_select_positions_for_liquidation_worst_performers(self):
        """测试按最差表现选择清仓持仓"""
        # 创建多个持仓，有不同的表现
        positions = [
            Position("STOCK1", 100, 90.0, 100.0, "Tech", datetime.now()),      # 亏损10%
            Position("STOCK2", 200, 105.0, 100.0, "Finance", datetime.now()),  # 盈利5%
            Position("STOCK3", 150, 80.0, 100.0, "Healthcare", datetime.now()) # 亏损20%
        ]
        
        # 设置未实现损益
        positions[0].unrealized_pnl = -1000.0  # STOCK1
        positions[1].unrealized_pnl = 1000.0   # STOCK2
        positions[2].unrealized_pnl = -3000.0  # STOCK3
        
        portfolio = Portfolio(positions, 5000.0, 25000.0, datetime.now())
        
        # 测试选择30%的持仓进行清仓
        selected = self.stop_loss_controller._select_positions_for_liquidation(portfolio, 0.3)
        
        # 应该选择表现最差的STOCK3
        self.assertEqual(len(selected), 1)
        self.assertIn("STOCK3", selected)
    
    def test_select_positions_for_liquidation_largest_positions(self):
        """测试按最大持仓选择清仓持仓"""
        # 修改配置为按最大持仓清仓
        self.config.liquidation_strategy = "largest_positions"
        self.stop_loss_controller = DynamicStopLoss(self.config)
        
        positions = [
            Position("SMALL", 50, 100.0, 100.0, "Tech", datetime.now()),       # 市值5000
            Position("LARGE", 300, 100.0, 100.0, "Finance", datetime.now()),   # 市值30000
            Position("MEDIUM", 100, 100.0, 100.0, "Healthcare", datetime.now()) # 市值10000
        ]
        
        portfolio = Portfolio(positions, 5000.0, 50000.0, datetime.now())
        
        selected = self.stop_loss_controller._select_positions_for_liquidation(portfolio, 0.3)
        
        # 应该选择最大的持仓LARGE
        self.assertEqual(len(selected), 1)
        self.assertIn("LARGE", selected)
    
    def test_execute_portfolio_stop_loss(self):
        """测试执行组合级止损"""
        # 创建测试组合
        positions = [
            Position("STOCK1", 100, 90.0, 100.0, "Tech", datetime.now()),
            Position("STOCK2", 200, 95.0, 100.0, "Finance", datetime.now())
        ]
        portfolio = Portfolio(positions, 5000.0, 25000.0, datetime.now())
        
        # 模拟止损触发结果
        stop_loss_result = {
            'trigger_type': 'partial_stop',
            'positions_to_liquidate': ['STOCK1', 'STOCK2']
        }
        
        # 执行组合级止损
        executed_orders = self.stop_loss_controller.execute_portfolio_stop_loss(
            portfolio, stop_loss_result
        )
        
        # 检查执行结果
        self.assertEqual(len(executed_orders), 2)
        
        for order in executed_orders:
            self.assertEqual(order['action'], 'SELL')
            self.assertEqual(order['order_type'], 'MARKET')
            self.assertEqual(order['status'], 'EXECUTED')
            self.assertIn(order['symbol'], ['STOCK1', 'STOCK2'])
    
    def test_portfolio_recovery_detection(self):
        """测试组合恢复检测"""
        # 首先触发组合级止损
        peak_portfolio = Portfolio(
            positions=[Position("AAPL", 100, 200.0, 160.0, "Tech", datetime.now())],
            cash=5000.0,
            total_value=25000.0,
            timestamp=datetime.now()
        )
        self.stop_loss_controller.check_portfolio_stop_loss(peak_portfolio)
        
        # 触发止损
        losing_portfolio = Portfolio(
            positions=[Position("AAPL", 100, 100.0, 160.0, "Tech", datetime.now())],
            cash=5000.0,
            total_value=15000.0,
            timestamp=datetime.now()
        )
        self.stop_loss_controller.check_portfolio_stop_loss(losing_portfolio)
        self.assertTrue(self.stop_loss_controller.portfolio_stop_loss_active)
        
        # 模拟恢复
        recovery_portfolio = Portfolio(
            positions=[Position("AAPL", 100, 190.0, 160.0, "Tech", datetime.now())],
            cash=5000.0,
            total_value=24000.0,  # 恢复到接近峰值
            timestamp=datetime.now()
        )
        
        result = self.stop_loss_controller.check_portfolio_stop_loss(recovery_portfolio)
        
        # 检查恢复检测
        self.assertIsNotNone(result)
        self.assertEqual(result['trigger_type'], 'recovery')
        self.assertEqual(result['action'], 'recovery_detected')
        self.assertFalse(self.stop_loss_controller.portfolio_stop_loss_active)
    
    def test_get_portfolio_stop_loss_status(self):
        """测试获取组合级止损状态"""
        status = self.stop_loss_controller.get_portfolio_stop_loss_status()
        
        # 检查状态信息
        self.assertIn('active', status)
        self.assertIn('peak_value', status)
        self.assertIn('total_records', status)
        self.assertIn('current_thresholds', status)
        
        thresholds = status['current_thresholds']
        self.assertEqual(thresholds['full_stop'], -0.12)
        self.assertEqual(thresholds['partial_stop'], -0.08)
        self.assertEqual(thresholds['recovery'], -0.05)
    
    def test_get_portfolio_stop_loss_history(self):
        """测试获取组合级止损历史"""
        # 触发一次组合级止损
        peak_portfolio = Portfolio(
            positions=[Position("AAPL", 100, 200.0, 160.0, "Tech", datetime.now())],
            cash=5000.0,
            total_value=25000.0,
            timestamp=datetime.now()
        )
        self.stop_loss_controller.check_portfolio_stop_loss(peak_portfolio)
        
        losing_portfolio = Portfolio(
            positions=[Position("AAPL", 100, 100.0, 160.0, "Tech", datetime.now())],
            cash=5000.0,
            total_value=15000.0,
            timestamp=datetime.now()
        )
        self.stop_loss_controller.check_portfolio_stop_loss(losing_portfolio)
        
        # 获取历史记录
        history = self.stop_loss_controller.get_portfolio_stop_loss_history()
        
        # 检查历史记录
        self.assertEqual(len(history), 1)
        record = history[0]
        self.assertEqual(record['trigger_type'], 'full_stop')
        self.assertIn('trigger_time', record)
        self.assertIn('positions_liquidated', record)
        self.assertIn('positions_count', record)
        self.assertFalse(record['is_recovered'])
    
    def test_check_stop_loss_triggers_fixed(self):
        """测试固定止损触发"""
        # 设置止损水平
        self.stop_loss_controller.stop_loss_levels["AAPL"] = {
            'fixed': 152.0,
            'percentage': 142.5,
            'volatility_adjusted': 150.0,
            'trailing': 0.0
        }
        
        # 创建触发止损的持仓（价格跌到151）
        trigger_position = Position(
            symbol="AAPL",
            quantity=100,
            current_price=151.0,
            cost_basis=160.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        triggered_orders = self.stop_loss_controller.check_stop_loss_triggers([trigger_position])
        
        # 检查是否触发了固定止损
        self.assertEqual(len(triggered_orders), 1)
        order = triggered_orders[0]
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.stop_type, StopLossType.FIXED)
        self.assertEqual(order.status, StopLossStatus.TRIGGERED)
        self.assertAlmostEqual(order.stop_price, 152.0, places=2)
    
    def test_check_stop_loss_triggers_trailing(self):
        """测试追踪止损触发"""
        # 设置追踪止损
        self.stop_loss_controller.trailing_stops["AAPL"] = 148.0
        
        # 创建触发追踪止损的持仓
        trigger_position = Position(
            symbol="AAPL",
            quantity=100,
            current_price=147.0,  # 低于追踪止损价格148
            cost_basis=160.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        triggered_orders = self.stop_loss_controller.check_stop_loss_triggers([trigger_position])
        
        # 检查是否触发了追踪止损
        self.assertEqual(len(triggered_orders), 1)
        order = triggered_orders[0]
        self.assertEqual(order.symbol, "AAPL")
        self.assertEqual(order.stop_type, StopLossType.TRAILING)
        self.assertEqual(order.status, StopLossStatus.TRIGGERED)
        self.assertAlmostEqual(order.stop_price, 148.0, places=2)
    
    def test_check_stop_loss_no_trigger(self):
        """测试未触发止损的情况"""
        # 设置止损水平
        self.stop_loss_controller.stop_loss_levels["AAPL"] = {
            'fixed': 152.0,
            'percentage': 142.5,
            'volatility_adjusted': 150.0,
            'trailing': 0.0
        }
        
        # 创建未触发止损的持仓（价格153，高于所有止损线）
        no_trigger_position = Position(
            symbol="AAPL",
            quantity=100,
            current_price=153.0,
            cost_basis=160.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        
        triggered_orders = self.stop_loss_controller.check_stop_loss_triggers([no_trigger_position])
        
        # 检查没有触发止损
        self.assertEqual(len(triggered_orders), 0)
    
    def test_execute_stop_loss_orders(self):
        """测试止损订单执行"""
        # 创建止损订单
        stop_order = StopLossOrder(
            symbol="AAPL",
            quantity=100,
            stop_price=152.0,
            stop_type=StopLossType.FIXED,
            status=StopLossStatus.TRIGGERED,
            created_time=datetime.now(),
            triggered_time=datetime.now(),
            reason="测试止损"
        )
        
        executed_orders = self.stop_loss_controller.execute_stop_loss_orders([stop_order])
        
        # 检查订单执行结果
        self.assertEqual(len(executed_orders), 1)
        result = executed_orders[0]
        self.assertEqual(result['symbol'], "AAPL")
        self.assertEqual(result['quantity'], 100)
        self.assertEqual(result['status'], 'EXECUTED')
        self.assertAlmostEqual(result['executed_price'], 152.0, places=2)
        
        # 检查追踪止损是否被清除
        self.assertNotIn("AAPL", self.stop_loss_controller.trailing_stops)
    
    def test_check_portfolio_stop_loss_full(self):
        """测试组合级全部止损"""
        # 首先设置组合峰值
        peak_portfolio = Portfolio(
            positions=[Position("AAPL", 100, 200.0, 160.0, "Tech", datetime.now())],
            cash=5000.0,
            total_value=25000.0,
            timestamp=datetime.now()
        )
        self.stop_loss_controller.check_portfolio_stop_loss(peak_portfolio)  # 设置峰值
        
        # 创建严重回撤的投资组合
        losing_position = Position(
            symbol="AAPL",
            quantity=100,
            current_price=100.0,  # 价格从200跌到100
            cost_basis=160.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        losing_position.unrealized_pnl = -6000.0
        
        losing_portfolio = Portfolio(
            positions=[losing_position],
            cash=5000.0,
            total_value=15000.0,  # 从25000跌到15000，回撤40%
            timestamp=datetime.now()
        )
        
        result = self.stop_loss_controller.check_portfolio_stop_loss(losing_portfolio)
        
        # 检查是否触发全部清仓止损
        self.assertIsNotNone(result)
        self.assertEqual(result['trigger_type'], 'full_stop')
        self.assertEqual(result['action'], 'liquidate_all')
        self.assertIn('positions_to_liquidate', result)
        self.assertTrue(self.stop_loss_controller.portfolio_stop_loss_active)
    
    def test_check_portfolio_stop_loss_partial(self):
        """测试组合级部分止损"""
        # 首先设置组合峰值
        peak_portfolio = Portfolio(
            positions=[Position("AAPL", 100, 180.0, 160.0, "Tech", datetime.now())],
            cash=8000.0,
            total_value=26000.0,
            timestamp=datetime.now()
        )
        self.stop_loss_controller.check_portfolio_stop_loss(peak_portfolio)  # 设置峰值
        
        # 创建中等回撤的投资组合
        losing_position = Position(
            symbol="AAPL",
            quantity=100,
            current_price=160.0,  # 价格从180跌到160
            cost_basis=160.0,
            sector="Technology",
            timestamp=datetime.now()
        )
        losing_position.unrealized_pnl = 0.0  # 无未实现损益
        
        partial_losing_portfolio = Portfolio(
            positions=[losing_position],
            cash=8000.0,
            total_value=23900.0,  # 从26000跌到23900，回撤约8.1%（刚好触发8%部分止损）
            timestamp=datetime.now()
        )
        
        result = self.stop_loss_controller.check_portfolio_stop_loss(partial_losing_portfolio)
        
        # 检查是否触发部分减仓止损
        self.assertIsNotNone(result)
        self.assertEqual(result['trigger_type'], 'partial_stop')
        self.assertEqual(result['action'], 'partial_liquidation')
        self.assertEqual(result['liquidation_ratio'], 0.3)
        self.assertTrue(self.stop_loss_controller.portfolio_stop_loss_active)
    
    def test_check_portfolio_stop_loss_no_trigger(self):
        """测试组合级止损未触发"""
        result = self.stop_loss_controller.check_portfolio_stop_loss(self.test_portfolio)
        
        # 检查没有触发组合级止损
        self.assertIsNone(result)
    
    def test_update_price_history(self):
        """测试价格历史更新"""
        symbol = "AAPL"
        price = 150.0
        timestamp = datetime.now()
        
        # 更新价格历史
        self.stop_loss_controller.update_price_history(symbol, price, timestamp)
        
        # 检查价格历史是否被记录
        self.assertIn(symbol, self.stop_loss_controller.price_history)
        self.assertEqual(len(self.stop_loss_controller.price_history[symbol]), 1)
        self.assertEqual(self.stop_loss_controller.price_history[symbol][0], (timestamp, price))
        
        # 检查波动率缓存是否被清除
        self.assertNotIn(symbol, self.stop_loss_controller.volatility_cache)
    
    def test_calculate_volatility(self):
        """测试波动率计算"""
        symbol = "AAPL"
        
        # 添加一些历史价格数据
        base_time = datetime.now()
        prices = [100, 102, 98, 105, 103, 107, 104, 108, 106, 110]
        
        for i, price in enumerate(prices):
            timestamp = base_time + timedelta(days=i)
            self.stop_loss_controller.update_price_history(symbol, price, timestamp)
        
        # 计算波动率
        volatility = self.stop_loss_controller._calculate_volatility(symbol)
        
        # 检查波动率是否合理
        self.assertGreater(volatility, 0)
        self.assertLess(volatility, 1.0)  # 年化波动率应该小于100%
        
        # 检查波动率是否被缓存
        self.assertIn(symbol, self.stop_loss_controller.volatility_cache)
        self.assertEqual(self.stop_loss_controller.volatility_cache[symbol], volatility)
    
    def test_calculate_volatility_insufficient_data(self):
        """测试数据不足时的波动率计算"""
        symbol = "NEW_STOCK"
        
        # 没有历史数据时应该返回默认波动率
        volatility = self.stop_loss_controller._calculate_volatility(symbol)
        self.assertEqual(volatility, 0.2)  # 默认20%波动率
    
    def test_get_stop_loss_status(self):
        """测试获取止损状态"""
        symbol = "AAPL"
        
        # 设置一些止损数据
        self.stop_loss_controller.stop_loss_levels[symbol] = {
            'fixed': 152.0,
            'percentage': 142.5
        }
        self.stop_loss_controller.trailing_stops[symbol] = 148.0
        self.stop_loss_controller.volatility_cache[symbol] = 0.25
        
        status = self.stop_loss_controller.get_stop_loss_status(symbol)
        
        # 检查状态信息
        self.assertEqual(status['symbol'], symbol)
        self.assertTrue(status['has_active_stops'])
        self.assertEqual(status['stop_levels']['fixed'], 152.0)
        self.assertEqual(status['trailing_stop'], 148.0)
        self.assertEqual(status['volatility'], 0.25)
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        # 设置一些统计数据
        self.stop_loss_controller.stats['total_stops_triggered'] = 10
        self.stop_loss_controller.stats['successful_stops'] = 8
        self.stop_loss_controller.stats['false_stops'] = 2
        
        stats = self.stop_loss_controller.get_statistics()
        
        # 检查统计信息
        self.assertEqual(stats['total_stops_triggered'], 10)
        self.assertEqual(stats['successful_stops'], 8)
        self.assertEqual(stats['false_stops'], 2)
        self.assertEqual(stats['success_rate'], 0.8)
    
    def test_reset_stop_loss(self):
        """测试重置止损设置"""
        symbol = "AAPL"
        
        # 设置一些止损数据
        self.stop_loss_controller.stop_loss_levels[symbol] = {'fixed': 152.0}
        self.stop_loss_controller.trailing_stops[symbol] = 148.0
        
        # 重置止损
        self.stop_loss_controller.reset_stop_loss(symbol)
        
        # 检查数据是否被清除
        self.assertNotIn(symbol, self.stop_loss_controller.stop_loss_levels)
        self.assertNotIn(symbol, self.stop_loss_controller.trailing_stops)
    
    def test_disable_enable_stop_loss(self):
        """测试禁用和启用止损"""
        symbol = "AAPL"
        
        # 创建活跃订单
        order = StopLossOrder(
            symbol=symbol,
            quantity=100,
            stop_price=152.0,
            stop_type=StopLossType.FIXED,
            status=StopLossStatus.ACTIVE,
            created_time=datetime.now()
        )
        self.stop_loss_controller.active_orders[symbol] = order
        
        # 禁用止损
        self.stop_loss_controller.disable_stop_loss(symbol)
        self.assertEqual(self.stop_loss_controller.active_orders[symbol].status, StopLossStatus.DISABLED)
        
        # 启用止损
        self.stop_loss_controller.enable_stop_loss(symbol)
        self.assertEqual(self.stop_loss_controller.active_orders[symbol].status, StopLossStatus.ACTIVE)


class TestStopLossConfig(unittest.TestCase):
    """止损配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = StopLossConfig()
        
        # 检查默认值
        self.assertEqual(config.base_stop_loss, 0.05)
        self.assertEqual(config.min_stop_loss, 0.02)
        self.assertEqual(config.max_stop_loss, 0.15)
        self.assertEqual(config.volatility_multiplier, 2.0)
        self.assertEqual(config.trailing_stop_distance, 0.03)
        self.assertEqual(config.portfolio_stop_loss, 0.12)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = StopLossConfig(
            base_stop_loss=0.08,
            trailing_stop_distance=0.05,
            portfolio_stop_loss=0.15
        )
        
        # 检查自定义值
        self.assertEqual(config.base_stop_loss, 0.08)
        self.assertEqual(config.trailing_stop_distance, 0.05)
        self.assertEqual(config.portfolio_stop_loss, 0.15)


if __name__ == '__main__':
    unittest.main()