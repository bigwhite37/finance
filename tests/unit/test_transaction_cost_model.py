"""
测试交易成本计算模块
测试手续费、印花税和滑点成本计算，以及A股特有的交易规则和成本结构
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List

from src.rl_trading_system.trading.transaction_cost_model import (
    TransactionCostModel,
    CostParameters,
    CostBreakdown,
    TradeInfo
)
from src.rl_trading_system.trading.almgren_chriss_model import (
    AlmgrenChrissModel,
    MarketImpactParameters
)


class TestCostParameters:
    """测试成本参数类"""
    
    def test_parameters_creation(self):
        """测试参数正常创建"""
        params = CostParameters(
            commission_rate=0.001,
            stamp_tax_rate=0.001,
            min_commission=5.0,
            transfer_fee_rate=0.00002,
            market_impact_model=None
        )
        
        assert params.commission_rate == 0.001
        assert params.stamp_tax_rate == 0.001
        assert params.min_commission == 5.0
        assert params.transfer_fee_rate == 0.00002
        assert params.market_impact_model is None
    
    def test_parameters_validation_negative_rates(self):
        """测试负费率验证"""
        with pytest.raises(ValueError, match="手续费率不能为负数"):
            CostParameters(
                commission_rate=-0.001,
                stamp_tax_rate=0.001,
                min_commission=5.0,
                transfer_fee_rate=0.00002
            )
        
        with pytest.raises(ValueError, match="印花税率不能为负数"):
            CostParameters(
                commission_rate=0.001,
                stamp_tax_rate=-0.001,
                min_commission=5.0,
                transfer_fee_rate=0.00002
            )
    
    def test_parameters_validation_negative_min_commission(self):
        """测试负最小手续费验证"""
        with pytest.raises(ValueError, match="最小手续费不能为负数"):
            CostParameters(
                commission_rate=0.001,
                stamp_tax_rate=0.001,
                min_commission=-5.0,
                transfer_fee_rate=0.00002
            )
    
    def test_parameters_validation_excessive_rates(self):
        """测试过高费率验证"""
        with pytest.raises(ValueError, match="手续费率不能超过10%"):
            CostParameters(
                commission_rate=0.15,  # 15%
                stamp_tax_rate=0.001,
                min_commission=5.0,
                transfer_fee_rate=0.00002
            )


class TestTradeInfo:
    """测试交易信息类"""
    
    def test_trade_info_creation(self):
        """测试交易信息正常创建"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            timestamp=timestamp,
            market_volume=1000000,
            volatility=0.02
        )
        
        assert trade.symbol == "000001.SZ"
        assert trade.side == "buy"
        assert trade.quantity == 1000
        assert trade.price == 10.5
        assert trade.timestamp == timestamp
        assert trade.market_volume == 1000000
        assert trade.volatility == 0.02
    
    def test_trade_info_validation_invalid_side(self):
        """测试无效交易方向验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="交易方向必须是'buy'或'sell'"):
            TradeInfo(
                symbol="000001.SZ",
                side="invalid",
                quantity=1000,
                price=10.5,
                timestamp=timestamp
            )
    
    def test_trade_info_validation_negative_quantity(self):
        """测试负数量验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="交易数量不能为负数"):
            TradeInfo(
                symbol="000001.SZ",
                side="buy",
                quantity=-1000,
                price=10.5,
                timestamp=timestamp
            )
    
    def test_trade_info_validation_negative_price(self):
        """测试负价格验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="价格不能为负数"):
            TradeInfo(
                symbol="000001.SZ",
                side="buy",
                quantity=1000,
                price=-10.5,
                timestamp=timestamp
            )
    
    def test_trade_info_get_trade_value(self):
        """测试获取交易价值方法"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        assert trade.get_trade_value() == 10500.0
    
    def test_trade_info_is_buy_sell(self):
        """测试买卖判断方法"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        buy_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        sell_trade = TradeInfo(
            symbol="000001.SZ",
            side="sell",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        assert buy_trade.is_buy()
        assert not buy_trade.is_sell()
        assert sell_trade.is_sell()
        assert not sell_trade.is_buy()


class TestCostBreakdown:
    """测试成本分解类"""
    
    def test_cost_breakdown_creation(self):
        """测试成本分解正常创建"""
        breakdown = CostBreakdown(
            commission=10.5,
            stamp_tax=10.5,
            transfer_fee=2.1,
            market_impact=5.2,
            total_cost=28.3
        )
        
        assert breakdown.commission == 10.5
        assert breakdown.stamp_tax == 10.5
        assert breakdown.transfer_fee == 2.1
        assert breakdown.market_impact == 5.2
        assert breakdown.total_cost == 28.3
    
    def test_cost_breakdown_validation_negative_costs(self):
        """测试负成本验证"""
        with pytest.raises(ValueError, match="手续费不能为负数"):
            CostBreakdown(
                commission=-10.5,
                stamp_tax=10.5,
                transfer_fee=2.1,
                market_impact=5.2,
                total_cost=28.3
            )
    
    def test_cost_breakdown_validation_inconsistent_total(self):
        """测试总成本一致性验证"""
        with pytest.raises(ValueError, match="总成本应等于各项成本之和"):
            CostBreakdown(
                commission=10.5,
                stamp_tax=10.5,
                transfer_fee=2.1,
                market_impact=5.2,
                total_cost=50.0  # 不等于各项之和
            )
    
    def test_cost_breakdown_get_cost_ratio(self):
        """测试获取成本比率方法"""
        breakdown = CostBreakdown(
            commission=10.5,
            stamp_tax=10.5,
            transfer_fee=2.1,
            market_impact=5.2,
            total_cost=28.3
        )
        
        trade_value = 10500.0
        ratio = breakdown.get_cost_ratio(trade_value)
        expected_ratio = 28.3 / 10500.0
        assert abs(ratio - expected_ratio) < 1e-8
    
    def test_cost_breakdown_get_cost_basis_points(self):
        """测试获取基点成本方法"""
        breakdown = CostBreakdown(
            commission=10.5,
            stamp_tax=10.5,
            transfer_fee=2.1,
            market_impact=5.2,
            total_cost=28.3
        )
        
        trade_value = 10500.0
        bp = breakdown.get_cost_basis_points(trade_value)
        expected_bp = (28.3 / 10500.0) * 10000
        assert abs(bp - expected_bp) < 1e-6


class TestTransactionCostModel:
    """测试交易成本模型"""
    
    @pytest.fixture
    def default_cost_parameters(self):
        """默认成本参数"""
        return CostParameters(
            commission_rate=0.001,
            stamp_tax_rate=0.001,
            min_commission=5.0,
            transfer_fee_rate=0.00002
        )
    
    @pytest.fixture
    def almgren_chriss_model(self):
        """Almgren-Chriss模型"""
        params = MarketImpactParameters(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            volatility=0.02,
            daily_volume=1000000,
            participation_rate=0.1
        )
        return AlmgrenChrissModel(params)
    
    @pytest.fixture
    def cost_model(self, default_cost_parameters):
        """默认成本模型"""
        return TransactionCostModel(default_cost_parameters)
    
    @pytest.fixture
    def cost_model_with_impact(self, default_cost_parameters, almgren_chriss_model):
        """带市场冲击的成本模型"""
        default_cost_parameters.market_impact_model = almgren_chriss_model
        return TransactionCostModel(default_cost_parameters)
    
    def test_model_creation(self, default_cost_parameters):
        """测试模型正常创建"""
        model = TransactionCostModel(default_cost_parameters)
        assert model.parameters == default_cost_parameters
    
    def test_calculate_commission_basic(self, cost_model):
        """测试基本手续费计算"""
        trade_value = 10500.0
        commission = cost_model._calculate_commission(trade_value)
        
        # 手续费 = max(trade_value * rate, min_commission)
        expected = max(10500.0 * 0.001, 5.0)
        assert abs(commission - expected) < 1e-8
    
    def test_calculate_commission_minimum(self, cost_model):
        """测试最小手续费"""
        trade_value = 1000.0  # 小额交易
        commission = cost_model._calculate_commission(trade_value)
        
        # 应该使用最小手续费
        assert commission == 5.0
    
    def test_calculate_stamp_tax_buy(self, cost_model):
        """测试买入印花税（应为0）"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        buy_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        stamp_tax = cost_model._calculate_stamp_tax(buy_trade)
        assert stamp_tax == 0.0
    
    def test_calculate_stamp_tax_sell(self, cost_model):
        """测试卖出印花税"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        sell_trade = TradeInfo(
            symbol="000001.SZ",
            side="sell",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        stamp_tax = cost_model._calculate_stamp_tax(sell_trade)
        expected = 10500.0 * 0.001
        assert abs(stamp_tax - expected) < 1e-8
    
    def test_calculate_transfer_fee(self, cost_model):
        """测试过户费计算"""
        trade_value = 10500.0
        transfer_fee = cost_model._calculate_transfer_fee(trade_value)
        
        expected = 10500.0 * 0.00002
        assert abs(transfer_fee - expected) < 1e-8
    
    def test_calculate_market_impact_without_model(self, cost_model):
        """测试无市场冲击模型时的计算"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        impact = cost_model._calculate_market_impact(trade)
        assert impact == 0.0
    
    def test_calculate_market_impact_with_model(self, cost_model_with_impact):
        """测试有市场冲击模型时的计算"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=100000,
            price=10.5,
            timestamp=timestamp,
            market_volume=1000000,
            volatility=0.02
        )
        
        impact = cost_model_with_impact._calculate_market_impact(trade)
        assert impact > 0.0
    
    def test_calculate_cost_buy_trade(self, cost_model):
        """测试买入交易成本计算"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        buy_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        breakdown = cost_model.calculate_cost(buy_trade)
        
        # 验证各项成本
        assert breakdown.commission == 10.5  # max(10500 * 0.001, 5.0)
        assert breakdown.stamp_tax == 0.0    # 买入无印花税
        assert abs(breakdown.transfer_fee - 0.21) < 1e-8  # 10500 * 0.00002
        assert breakdown.market_impact == 0.0  # 无市场冲击模型
        
        expected_total = 10.5 + 0.0 + 0.21 + 0.0
        assert abs(breakdown.total_cost - expected_total) < 1e-8
    
    def test_calculate_cost_sell_trade(self, cost_model):
        """测试卖出交易成本计算"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        sell_trade = TradeInfo(
            symbol="000001.SZ",
            side="sell",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        breakdown = cost_model.calculate_cost(sell_trade)
        
        # 验证各项成本
        assert breakdown.commission == 10.5   # max(10500 * 0.001, 5.0)
        assert breakdown.stamp_tax == 10.5    # 10500 * 0.001
        assert abs(breakdown.transfer_fee - 0.21) < 1e-8 # 10500 * 0.00002
        assert breakdown.market_impact == 0.0 # 无市场冲击模型
        
        expected_total = 10.5 + 10.5 + 0.21 + 0.0
        assert abs(breakdown.total_cost - expected_total) < 1e-8
    
    def test_calculate_cost_with_market_impact(self, cost_model_with_impact):
        """测试包含市场冲击的成本计算"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=100000,
            price=10.5,
            timestamp=timestamp,
            market_volume=1000000,
            volatility=0.02
        )
        
        breakdown = cost_model_with_impact.calculate_cost(trade)
        
        # 验证市场冲击大于0
        assert breakdown.market_impact > 0.0
        
        # 验证总成本包含市场冲击
        expected_total = breakdown.commission + breakdown.stamp_tax + breakdown.transfer_fee + breakdown.market_impact
        assert abs(breakdown.total_cost - expected_total) < 1e-8
    
    def test_calculate_batch_costs(self, cost_model):
        """测试批量成本计算"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trades = [
            TradeInfo("000001.SZ", "buy", 1000, 10.5, timestamp),
            TradeInfo("000002.SZ", "sell", 2000, 15.2, timestamp),
            TradeInfo("000003.SZ", "buy", 500, 8.8, timestamp)
        ]
        
        breakdowns = cost_model.calculate_batch_costs(trades)
        
        assert len(breakdowns) == 3
        
        # 验证每个结果都是有效的
        for breakdown in breakdowns:
            assert isinstance(breakdown, CostBreakdown)
            assert breakdown.total_cost > 0
    
    def test_a_share_trading_rules_validation(self, cost_model):
        """测试A股交易规则验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        # 测试正常交易
        normal_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=100,  # 100股，符合A股最小交易单位
            price=10.5,
            timestamp=timestamp
        )
        
        breakdown = cost_model.calculate_cost(normal_trade)
        assert breakdown.total_cost > 0
        
        # 测试不符合最小交易单位的交易（应该被处理）
        odd_lot_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=150,  # 150股，不是100的倍数
            price=10.5,
            timestamp=timestamp
        )
        
        # 模型应该能处理这种情况
        breakdown = cost_model.calculate_cost(odd_lot_trade)
        assert breakdown.total_cost > 0
    
    def test_cost_model_different_scenarios(self, cost_model):
        """测试成本模型在各种交易场景下的表现"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        scenarios = [
            # 小额交易
            TradeInfo("000001.SZ", "buy", 100, 5.0, timestamp),
            # 大额交易
            TradeInfo("000002.SZ", "sell", 100000, 50.0, timestamp),
            # 高价股
            TradeInfo("000003.SZ", "buy", 100, 200.0, timestamp),
            # 低价股
            TradeInfo("000004.SZ", "sell", 10000, 2.0, timestamp)
        ]
        
        for trade in scenarios:
            breakdown = cost_model.calculate_cost(trade)
            
            # 验证成本合理性
            assert breakdown.total_cost > 0
            assert breakdown.commission >= 5.0  # 最小手续费
            
            # 验证印花税规则
            if trade.is_sell():
                assert breakdown.stamp_tax > 0
            else:
                assert breakdown.stamp_tax == 0
            
            # 验证成本比率在合理范围内
            cost_ratio = breakdown.get_cost_ratio(trade.get_trade_value())
            assert 0 < cost_ratio < 0.1  # 成本比率应在0-10%之间


class TestTransactionCostModelBoundaryConditions:
    """测试交易成本模型边界条件"""
    
    def test_zero_quantity_trade(self):
        """测试零数量交易"""
        params = CostParameters(
            commission_rate=0.001,
            stamp_tax_rate=0.001,
            min_commission=5.0,
            transfer_fee_rate=0.00002
        )
        model = TransactionCostModel(params)
        
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        zero_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=0,
            price=10.5,
            timestamp=timestamp
        )
        
        breakdown = model.calculate_cost(zero_trade)
        
        # 零数量交易仍应有最小手续费
        assert breakdown.commission == 5.0
        assert breakdown.stamp_tax == 0.0
        assert breakdown.transfer_fee == 0.0
    
    def test_very_high_price_trade(self):
        """测试极高价格交易"""
        params = CostParameters(
            commission_rate=0.001,
            stamp_tax_rate=0.001,
            min_commission=5.0,
            transfer_fee_rate=0.00002
        )
        model = TransactionCostModel(params)
        
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        high_price_trade = TradeInfo(
            symbol="000001.SZ",
            side="sell",
            quantity=100,
            price=10000.0,  # 极高价格
            timestamp=timestamp
        )
        
        breakdown = model.calculate_cost(high_price_trade)
        
        # 验证成本计算正确
        trade_value = 100 * 10000.0
        expected_commission = trade_value * 0.001
        expected_stamp_tax = trade_value * 0.001
        expected_transfer_fee = trade_value * 0.00002
        
        assert abs(breakdown.commission - expected_commission) < 1e-6
        assert abs(breakdown.stamp_tax - expected_stamp_tax) < 1e-6
        assert abs(breakdown.transfer_fee - expected_transfer_fee) < 1e-6
    
    def test_extreme_parameters(self):
        """测试极端参数"""
        # 极低费率
        low_params = CostParameters(
            commission_rate=1e-6,
            stamp_tax_rate=1e-6,
            min_commission=0.01,
            transfer_fee_rate=1e-8
        )
        
        model = TransactionCostModel(low_params)
        
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            timestamp=timestamp
        )
        
        breakdown = model.calculate_cost(trade)
        
        # 验证计算结果有效
        assert breakdown.total_cost > 0
        assert not np.isnan(breakdown.total_cost)
        assert not np.isinf(breakdown.total_cost)


class TestTransactionCostModelPerformance:
    """测试交易成本模型性能"""
    
    def test_batch_calculation_performance(self):
        """测试批量计算性能"""
        params = CostParameters(
            commission_rate=0.001,
            stamp_tax_rate=0.001,
            min_commission=5.0,
            transfer_fee_rate=0.00002
        )
        model = TransactionCostModel(params)
        
        # 生成大量交易数据
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        trades = []
        
        for i in range(1000):
            trade = TradeInfo(
                symbol=f"00000{i % 100}.SZ",
                side="buy" if i % 2 == 0 else "sell",
                quantity=np.random.randint(100, 10000),
                price=np.random.uniform(5.0, 50.0),
                timestamp=timestamp
            )
            trades.append(trade)
        
        import time
        start_time = time.time()
        
        breakdowns = model.calculate_batch_costs(trades)
        
        calculation_time = time.time() - start_time
        
        # 计算时间应该在合理范围内
        assert calculation_time < 2.0
        assert len(breakdowns) == 1000
        
        # 验证所有结果都有效
        for breakdown in breakdowns:
            assert breakdown.total_cost > 0
            assert not np.isnan(breakdown.total_cost)


class TestTransactionCostModelIntegration:
    """测试交易成本模型集成"""
    
    def test_integration_with_almgren_chriss(self):
        """测试与Almgren-Chriss模型的集成"""
        # 创建市场冲击模型
        impact_params = MarketImpactParameters(
            permanent_impact_coeff=0.08,
            temporary_impact_coeff=0.4,
            volatility=0.025,
            daily_volume=5000000,
            participation_rate=0.05
        )
        impact_model = AlmgrenChrissModel(impact_params)
        
        # 创建交易成本模型
        cost_params = CostParameters(
            commission_rate=0.001,
            stamp_tax_rate=0.001,
            min_commission=5.0,
            transfer_fee_rate=0.00002,
            market_impact_model=impact_model
        )
        cost_model = TransactionCostModel(cost_params)
        
        # 测试不同规模的交易
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        small_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.5,
            timestamp=timestamp,
            market_volume=5000000,
            volatility=0.025
        )
        
        large_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=100000,
            price=10.5,
            timestamp=timestamp,
            market_volume=5000000,
            volatility=0.025
        )
        
        small_breakdown = cost_model.calculate_cost(small_trade)
        large_breakdown = cost_model.calculate_cost(large_trade)
        
        # 验证大额交易有更高的市场冲击
        assert large_breakdown.market_impact > small_breakdown.market_impact
        
        # 验证总成本随交易规模增长
        small_ratio = small_breakdown.get_cost_ratio(small_trade.get_trade_value())
        large_ratio = large_breakdown.get_cost_ratio(large_trade.get_trade_value())
        assert large_ratio > small_ratio
    
    def test_real_world_cost_estimation(self):
        """测试真实世界成本估算"""
        # 使用真实的A股成本参数
        real_params = CostParameters(
            commission_rate=0.0003,  # 万三手续费
            stamp_tax_rate=0.001,   # 千一印花税
            min_commission=5.0,     # 5元最小手续费
            transfer_fee_rate=0.00002  # 万0.2过户费
        )
        
        model = TransactionCostModel(real_params)
        
        # 测试典型的A股交易
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        # 小额交易（1万元）
        small_trade = TradeInfo(
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=10.0,
            timestamp=timestamp
        )
        
        # 中等交易（10万元）
        medium_trade = TradeInfo(
            symbol="000002.SZ",
            side="sell",
            quantity=5000,
            price=20.0,
            timestamp=timestamp
        )
        
        # 大额交易（100万元）
        large_trade = TradeInfo(
            symbol="000003.SZ",
            side="buy",
            quantity=20000,
            price=50.0,
            timestamp=timestamp
        )
        
        small_breakdown = model.calculate_cost(small_trade)
        medium_breakdown = model.calculate_cost(medium_trade)
        large_breakdown = model.calculate_cost(large_trade)
        
        # 验证成本在合理范围内
        small_bp = small_breakdown.get_cost_basis_points(small_trade.get_trade_value())
        medium_bp = medium_breakdown.get_cost_basis_points(medium_trade.get_trade_value())
        large_bp = large_breakdown.get_cost_basis_points(large_trade.get_trade_value())
        
        # 小额交易成本较高（由于最小手续费）
        assert 5 <= small_bp <= 100  # 0.5-10bp
        
        # 中等交易成本适中
        assert 10 <= medium_bp <= 50   # 1-5bp
        
        # 大额交易成本相对较低
        assert 3 <= large_bp <= 30     # 0.3-3bp
        
        # 验证卖出交易有印花税
        assert medium_breakdown.stamp_tax > 0  # 卖出交易
        assert small_breakdown.stamp_tax == 0  # 买入交易
        assert large_breakdown.stamp_tax == 0  # 买入交易