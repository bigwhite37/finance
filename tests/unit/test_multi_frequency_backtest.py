"""
多频率回测引擎的单元测试
测试日频和分钟频回测功能、交易执行模拟和成交价格处理、回测结果的准确性和性能
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date, time
from typing import Dict, List, Tuple, Any, Optional
from decimal import Decimal
from enum import Enum

from src.rl_trading_system.backtest.multi_frequency_backtest import (
    MultiFrequencyBacktest,
    BacktestConfig,
    BacktestResult,
    ExecutionMode,
    PriceMode,
    OrderType,
    Order,
    Trade,
    Position,
    Portfolio
)


class TestExecutionMode:
    """交易执行模式测试类"""

    def test_execution_mode_enum_values(self):
        """测试交易执行模式枚举值"""
        assert ExecutionMode.NEXT_BAR == "next_bar"
        assert ExecutionMode.NEXT_CLOSE == "next_close"
        assert ExecutionMode.NEXT_OPEN == "next_open"
        assert ExecutionMode.MARKET_ORDER == "market_order"
        assert ExecutionMode.LIMIT_ORDER == "limit_order"


class TestPriceMode:
    """价格模式测试类"""

    def test_price_mode_enum_values(self):
        """测试价格模式枚举值"""
        assert PriceMode.CLOSE == "close"
        assert PriceMode.OPEN == "open"
        assert PriceMode.HIGH == "high"
        assert PriceMode.LOW == "low"
        assert PriceMode.VWAP == "vwap"
        assert PriceMode.TWAP == "twap"


class TestOrderType:
    """订单类型测试类"""

    def test_order_type_enum_values(self):
        """测试订单类型枚举值"""
        assert OrderType.BUY == "buy"
        assert OrderType.SELL == "sell"
        assert OrderType.SHORT == "short"
        assert OrderType.COVER == "cover"


class TestOrder:
    """订单测试类"""

    def test_order_creation(self):
        """测试订单创建"""
        order = Order(
            symbol="000001.SZ",
            order_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50"),
            timestamp=datetime.now()
        )

        assert order.symbol == "000001.SZ"
        assert order.order_type == OrderType.BUY
        assert order.quantity == 1000
        assert order.price == Decimal("10.50")
        assert isinstance(order.timestamp, datetime)
        assert order.order_id is not None
        assert order.status == "pending"

    def test_order_validation(self):
        """测试订单验证"""
        # 测试无效数量
        with pytest.raises(ValueError, match="订单数量必须为正数"):
            Order(
                symbol="000001.SZ",
                order_type=OrderType.BUY,
                quantity=0,
                price=Decimal("10.50")
            )

        # 测试无效价格
        with pytest.raises(ValueError, match="订单价格必须为正数"):
            Order(
                symbol="000001.SZ",
                order_type=OrderType.BUY,
                quantity=1000,
                price=Decimal("-10.50")
            )

        # 测试空股票代码
        with pytest.raises(ValueError, match="股票代码不能为空"):
            Order(
                symbol="",
                order_type=OrderType.BUY,
                quantity=1000,
                price=Decimal("10.50")
            )

    def test_order_execution_price_calculation(self):
        """测试订单执行价格计算"""
        order = Order(
            symbol="000001.SZ",
            order_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50")
        )

        # 模拟市场数据
        market_data = {
            'open': 10.45,
            'high': 10.60,
            'low': 10.40,
            'close': 10.55,
            'volume': 1000000,
            'vwap': 10.52
        }

        # 测试不同价格模式下的执行价格
        assert order.get_execution_price(market_data, PriceMode.CLOSE) == Decimal("10.55")
        assert order.get_execution_price(market_data, PriceMode.OPEN) == Decimal("10.45")
        assert order.get_execution_price(market_data, PriceMode.HIGH) == Decimal("10.60")
        assert order.get_execution_price(market_data, PriceMode.LOW) == Decimal("10.40")
        assert order.get_execution_price(market_data, PriceMode.VWAP) == Decimal("10.52")


class TestTrade:
    """交易测试类"""

    def test_trade_creation(self):
        """测试交易创建"""
        trade = Trade(
            symbol="000001.SZ",
            trade_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50"),
            timestamp=datetime.now(),
            commission=Decimal("10.50")
        )

        assert trade.symbol == "000001.SZ"
        assert trade.trade_type == OrderType.BUY
        assert trade.quantity == 1000
        assert trade.price == Decimal("10.50")
        assert trade.commission == Decimal("10.50")
        assert trade.trade_id is not None

    def test_trade_value_calculation(self):
        """测试交易价值计算"""
        trade = Trade(
            symbol="000001.SZ",
            trade_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50"),
            commission=Decimal("10.50")
        )

        # 买入交易价值 = -(quantity * price + commission)
        expected_value = -(1000 * Decimal("10.50") + Decimal("10.50"))
        assert trade.get_trade_value() == expected_value

        # 卖出交易
        sell_trade = Trade(
            symbol="000001.SZ",
            trade_type=OrderType.SELL,
            quantity=1000,
            price=Decimal("11.00"),
            commission=Decimal("11.00")
        )

        # 卖出交易价值 = quantity * price - commission
        expected_sell_value = 1000 * Decimal("11.00") - Decimal("11.00")
        assert sell_trade.get_trade_value() == expected_sell_value


class TestPosition:
    """持仓测试类"""

    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(symbol="000001.SZ")

        assert position.symbol == "000001.SZ"
        assert position.quantity == 0
        assert position.avg_price == Decimal("0")
        assert position.market_value == Decimal("0")
        assert position.unrealized_pnl == Decimal("0")

    def test_position_buy_operations(self):
        """测试持仓买入操作"""
        position = Position(symbol="000001.SZ")

        # 第一次买入
        position.update_position(
            trade_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50")
        )

        assert position.quantity == 1000
        assert position.avg_price == Decimal("10.50")

        # 第二次买入（不同价格）
        position.update_position(
            trade_type=OrderType.BUY,
            quantity=500,
            price=Decimal("11.00")
        )

        assert position.quantity == 1500
        # 平均价格 = (1000*10.50 + 500*11.00) / 1500
        expected_avg_price = (Decimal("10500") + Decimal("5500")) / Decimal("1500")
        assert abs(position.avg_price - expected_avg_price) < Decimal("0.01")

    def test_position_sell_operations(self):
        """测试持仓卖出操作"""
        position = Position(symbol="000001.SZ")

        # 先买入
        position.update_position(
            trade_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50")
        )

        # 部分卖出
        position.update_position(
            trade_type=OrderType.SELL,
            quantity=300,
            price=Decimal("11.00")
        )

        assert position.quantity == 700
        assert position.avg_price == Decimal("10.50")  # 卖出不改变平均成本

    def test_position_market_value_calculation(self):
        """测试持仓市值计算"""
        position = Position(symbol="000001.SZ")

        position.update_position(
            trade_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50")
        )

        # 更新市场价格
        current_price = Decimal("11.20")
        position.update_market_value(current_price)

        expected_market_value = 1000 * current_price
        assert position.market_value == expected_market_value

        # 未实现盈亏 = 市值 - 成本
        expected_cost = 1000 * Decimal("10.50")
        expected_unrealized_pnl = expected_market_value - expected_cost
        assert position.unrealized_pnl == expected_unrealized_pnl

    def test_position_insufficient_quantity_error(self):
        """测试持仓数量不足错误"""
        position = Position(symbol="000001.SZ")

        # 没有持仓却要卖出
        with pytest.raises(ValueError, match="持仓数量不足"):
            position.update_position(
                trade_type=OrderType.SELL,
                quantity=100,
                price=Decimal("10.50")
            )

        # 买入后卖出超过持仓数量
        position.update_position(
            trade_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50")
        )

        with pytest.raises(ValueError, match="持仓数量不足"):
            position.update_position(
                trade_type=OrderType.SELL,
                quantity=1500,
                price=Decimal("11.00")
            )


class TestPortfolio:
    """组合测试类"""

    def test_portfolio_creation(self):
        """测试组合创建"""
        portfolio = Portfolio(initial_cash=Decimal("1000000"))

        assert portfolio.cash == Decimal("1000000")
        assert portfolio.initial_cash == Decimal("1000000")
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == Decimal("1000000")

    def test_portfolio_order_execution(self):
        """测试组合订单执行"""
        portfolio = Portfolio(initial_cash=Decimal("1000000"))

        # 创建买入订单
        order = Order(
            symbol="000001.SZ",
            order_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50")
        )

        # 模拟市场数据
        market_data = pd.Series({
            'open': 10.45,
            'high': 10.60,
            'low': 10.40,
            'close': 10.55,
            'volume': 1000000
        })

        # 执行订单
        trade = portfolio.execute_order(
            order=order,
            market_data=market_data,
            price_mode=PriceMode.CLOSE,
            commission_rate=Decimal("0.001"),
            stamp_tax_rate=Decimal("0.001")
        )

        # 验证交易结果
        assert trade.symbol == "000001.SZ"
        assert trade.trade_type == OrderType.BUY
        assert trade.quantity == 1000
        assert trade.price == Decimal("10.55")

        # 验证组合状态
        assert "000001.SZ" in portfolio.positions
        assert portfolio.positions["000001.SZ"].quantity == 1000

        # 验证现金减少
        expected_cost = 1000 * Decimal("10.55") + trade.commission
        expected_cash = Decimal("1000000") - expected_cost
        assert abs(portfolio.cash - expected_cash) < Decimal("0.01")

    def test_portfolio_insufficient_cash_error(self):
        """测试组合现金不足错误"""
        portfolio = Portfolio(initial_cash=Decimal("10000"))  # 现金不足

        order = Order(
            symbol="000001.SZ",
            order_type=OrderType.BUY,
            quantity=10000,  # 需要约10万元
            price=Decimal("10.50")
        )

        market_data = pd.Series({
            'close': 10.55
        })

        # 应该抛出现金不足异常
        with pytest.raises(ValueError, match="现金不足"):
            portfolio.execute_order(
                order=order,
                market_data=market_data,
                price_mode=PriceMode.CLOSE,
                commission_rate=Decimal("0.001"),
                stamp_tax_rate=Decimal("0.001")
            )

    def test_portfolio_performance_calculation(self):
        """测试组合业绩计算"""
        portfolio = Portfolio(initial_cash=Decimal("1000000"))

        # 买入股票
        order = Order(
            symbol="000001.SZ",
            order_type=OrderType.BUY,
            quantity=1000,
            price=Decimal("10.50")
        )

        market_data = pd.Series({'close': 10.50})
        portfolio.execute_order(order, market_data, PriceMode.CLOSE, Decimal("0.001"), Decimal("0.001"))

        # 更新市场价格
        current_prices = {"000001.SZ": Decimal("11.50")}
        portfolio.update_market_values(current_prices)

        # 计算总价值和收益率
        performance = portfolio.get_performance_metrics()

        assert performance['total_value'] > portfolio.initial_cash
        assert performance['total_return'] > 0
        assert performance['cash_ratio'] < 1.0


class TestBacktestConfig:
    """回测配置测试类"""

    def test_backtest_config_creation(self):
        """测试回测配置创建"""
        config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=1000000.0,
            frequency="1d",
            execution_mode=ExecutionMode.NEXT_CLOSE,
            price_mode=PriceMode.CLOSE,
            commission_rate=0.001,
            stamp_tax_rate=0.001
        )

        assert config.start_date == date(2023, 1, 1)
        assert config.end_date == date(2023, 12, 31)
        assert config.initial_capital == 1000000.0
        assert config.frequency == "1d"
        assert config.execution_mode == ExecutionMode.NEXT_CLOSE
        assert config.price_mode == PriceMode.CLOSE
        assert config.commission_rate == 0.001
        assert config.stamp_tax_rate == 0.001

    def test_backtest_config_validation(self):
        """测试回测配置验证"""
        # 测试结束日期早于开始日期
        with pytest.raises(ValueError, match="结束日期不能早于开始日期"):
            BacktestConfig(
                start_date=date(2023, 12, 31),
                end_date=date(2023, 1, 1),
                initial_capital=1000000.0
            )

        # 测试无效的初始资金
        with pytest.raises(ValueError, match="初始资金必须为正数"):
            BacktestConfig(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                initial_capital=-1000.0
            )

        # 测试无效的佣金率
        with pytest.raises(ValueError, match="佣金率必须为非负数"):
            BacktestConfig(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                initial_capital=1000000.0,
                commission_rate=-0.001
            )

    def test_backtest_config_frequency_validation(self):
        """测试回测频率验证"""
        # 有效频率
        valid_frequencies = ["1d", "1h", "30min", "15min", "5min", "1min"]
        for freq in valid_frequencies:
            config = BacktestConfig(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                initial_capital=1000000.0,
                frequency=freq
            )
            assert config.frequency == freq

        # 无效频率
        with pytest.raises(ValueError, match="不支持的频率"):
            BacktestConfig(
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                initial_capital=1000000.0,
                frequency="invalid"
            )


class TestBacktestResult:
    """回测结果测试类"""

    def test_backtest_result_creation(self):
        """测试回测结果创建"""
        # 模拟交易历史
        trades = [
            Trade("000001.SZ", OrderType.BUY, 1000, Decimal("10.50"), datetime.now(), Decimal("10.50")),
            Trade("000001.SZ", OrderType.SELL, 500, Decimal("11.00"), datetime.now(), Decimal("5.50"))
        ]

        # 模拟组合价值历史
        portfolio_values = pd.Series(
            [1000000, 1005000, 1010000, 1008000, 1012000],
            index=pd.date_range('2023-01-01', periods=5, freq='D')
        )

        result = BacktestResult(
            trades=trades,
            portfolio_values=portfolio_values,
            positions={},
            final_cash=Decimal("950000")
        )

        assert len(result.trades) == 2
        assert len(result.portfolio_values) == 5
        assert result.final_cash == Decimal("950000")

    def test_backtest_result_performance_metrics(self):
        """测试回测结果性能指标计算"""
        # 创建一个简单的收益序列
        returns = [0.01, -0.005, 0.015, -0.01, 0.02]
        portfolio_values = [1000000]
        for r in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + r))

        portfolio_values = pd.Series(
            portfolio_values,
            index=pd.date_range('2023-01-01', periods=6, freq='D')
        )

        result = BacktestResult(
            trades=[],
            portfolio_values=portfolio_values,
            positions={},
            final_cash=Decimal("0")
        )

        metrics = result.calculate_performance_metrics()

        # 验证基本指标
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics

        # 验证总收益率
        expected_total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        assert abs(metrics['total_return'] - expected_total_return) < 0.0001

        # 验证最大回撤为正数
        assert metrics['max_drawdown'] >= 0


class TestMultiFrequencyBacktest:
    """多频率回测引擎测试类"""

    @pytest.fixture
    def sample_data(self):
        """创建样本市场数据"""
        dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
        symbols = ['000001.SZ', '000002.SZ']

        data_list = []
        for symbol in symbols:
            for date in dates:
                data_list.append({
                    'symbol': symbol,
                    'datetime': date,
                    'open': 10.0 + np.random.randn() * 0.1,
                    'high': 10.2 + np.random.randn() * 0.1,
                    'low': 9.8 + np.random.randn() * 0.1,
                    'close': 10.0 + np.random.randn() * 0.1,
                    'volume': 1000000 + np.random.randint(-100000, 100000)
                })

        df = pd.DataFrame(data_list)
        df = df.set_index(['datetime', 'symbol'])
        return df

    @pytest.fixture
    def backtest_config(self):
        """创建回测配置"""
        return BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
            initial_capital=1000000.0,
            frequency="1d",
            execution_mode=ExecutionMode.NEXT_CLOSE,
            price_mode=PriceMode.CLOSE,
            commission_rate=0.001,
            stamp_tax_rate=0.001
        )

    @pytest.fixture
    def simple_strategy(self):
        """创建简单的买入持有策略"""
        def strategy_func(data, portfolio, timestamp):
            """简单策略：第一天买入，最后一天卖出"""
            orders = []

            if timestamp == pd.Timestamp('2023-01-02'):  # 第二个交易日买入
                orders.append(Order(
                    symbol="000001.SZ",
                    order_type=OrderType.BUY,
                    quantity=1000,
                    price=Decimal("10.0")
                ))
            elif timestamp == pd.Timestamp('2023-01-09'):  # 倒数第二个交易日卖出
                if "000001.SZ" in portfolio.positions and portfolio.positions["000001.SZ"].quantity > 0:
                    orders.append(Order(
                        symbol="000001.SZ",
                        order_type=OrderType.SELL,
                        quantity=portfolio.positions["000001.SZ"].quantity,
                        price=Decimal("10.0")
                    ))

            return orders

        return strategy_func

    def test_backtest_engine_initialization(self, backtest_config):
        """测试回测引擎初始化"""
        engine = MultiFrequencyBacktest(backtest_config)

        assert engine.config == backtest_config
        assert engine.portfolio.initial_cash == Decimal(str(backtest_config.initial_capital))
        assert len(engine.trades) == 0
        assert len(engine.portfolio_values) == 0

    def test_backtest_data_validation(self, backtest_config):
        """测试回测数据验证"""
        engine = MultiFrequencyBacktest(backtest_config)

        # 测试空数据
        with pytest.raises(ValueError, match="回测数据不能为空"):
            engine.run(data=pd.DataFrame(), strategy=lambda *args: [])

        # 测试缺少必要列的数据
        invalid_data = pd.DataFrame({
            'symbol': ['000001.SZ'],
            'datetime': [pd.Timestamp('2023-01-01')],
            'close': [10.0]
            # 缺少 open, high, low, volume
        }).set_index(['datetime', 'symbol'])

        with pytest.raises(ValueError, match="数据缺少必要的列"):
            engine.run(data=invalid_data, strategy=lambda *args: [])

    def test_backtest_daily_frequency_execution(self, sample_data, backtest_config, simple_strategy):
        """测试日频回测执行"""
        engine = MultiFrequencyBacktest(backtest_config)
        result = engine.run(data=sample_data, strategy=simple_strategy)

        # 验证回测结果
        assert isinstance(result, BacktestResult)
        assert len(result.trades) >= 1  # 至少有买入交易
        assert len(result.portfolio_values) > 0

        # 验证组合价值序列的连续性
        assert result.portfolio_values.index.is_monotonic_increasing

        # 验证最终资金 + 持仓市值 = 总价值
        final_total_value = float(result.final_cash)
        for position in result.positions.values():
            final_total_value += float(position.market_value)

        assert abs(final_total_value - result.portfolio_values.iloc[-1]) < 1.0

    def test_backtest_minute_frequency_data_handling(self, backtest_config):
        """测试分钟频回测数据处理"""
        # 创建分钟频数据
        minute_dates = pd.date_range('2023-01-01 09:30:00', '2023-01-01 15:00:00', freq='1min')
        minute_data_list = []

        for ts in minute_dates:
            minute_data_list.append({
                'symbol': '000001.SZ',
                'datetime': ts,
                'open': 10.0 + np.random.randn() * 0.01,
                'high': 10.02 + np.random.randn() * 0.01,
                'low': 9.98 + np.random.randn() * 0.01,
                'close': 10.0 + np.random.randn() * 0.01,
                'volume': 1000 + np.random.randint(-100, 100)
            })

        minute_data = pd.DataFrame(minute_data_list).set_index(['datetime', 'symbol'])

        # 更新配置为分钟频
        minute_config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 1),
            initial_capital=1000000.0,
            frequency="1min",
            execution_mode=ExecutionMode.NEXT_CLOSE,
            price_mode=PriceMode.CLOSE
        )

        engine = MultiFrequencyBacktest(minute_config)

        # 简单策略：在第一分钟买入
        def minute_strategy(data, portfolio, timestamp):
            orders = []
            if timestamp == minute_dates[0]:
                orders.append(Order(
                    symbol="000001.SZ",
                    order_type=OrderType.BUY,
                    quantity=100,
                    price=Decimal("10.0")
                ))
            return orders

        result = engine.run(data=minute_data, strategy=minute_strategy)

        # 验证分钟频回测结果
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_values) > 100  # 分钟频应该有很多数据点

    def test_backtest_execution_modes(self, sample_data, simple_strategy):
        """测试不同执行模式"""
        base_config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
            initial_capital=1000000.0,
            frequency="1d"
        )

        execution_modes = [
            ExecutionMode.NEXT_CLOSE,
            ExecutionMode.NEXT_OPEN,
            ExecutionMode.MARKET_ORDER
        ]

        results = {}
        for mode in execution_modes:
            config = BacktestConfig(
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                initial_capital=base_config.initial_capital,
                frequency=base_config.frequency,
                execution_mode=mode,
                price_mode=PriceMode.CLOSE
            )

            engine = MultiFrequencyBacktest(config)
            result = engine.run(data=sample_data, strategy=simple_strategy)
            results[mode] = result

        # 验证不同执行模式都能正常运行
        for mode, result in results.items():
            assert isinstance(result, BacktestResult)
            assert len(result.portfolio_values) > 0

    def test_backtest_price_modes(self, sample_data, simple_strategy):
        """测试不同价格模式"""
        base_config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 10),
            initial_capital=1000000.0,
            frequency="1d",
            execution_mode=ExecutionMode.NEXT_CLOSE
        )

        price_modes = [PriceMode.CLOSE, PriceMode.OPEN, PriceMode.HIGH, PriceMode.LOW]

        results = {}
        for mode in price_modes:
            config = BacktestConfig(
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                initial_capital=base_config.initial_capital,
                frequency=base_config.frequency,
                execution_mode=base_config.execution_mode,
                price_mode=mode
            )

            engine = MultiFrequencyBacktest(config)
            result = engine.run(data=sample_data, strategy=simple_strategy)
            results[mode] = result

        # 验证不同价格模式都能正常运行，并且可能产生不同的结果
        for mode, result in results.items():
            assert isinstance(result, BacktestResult)
            assert len(result.trades) >= 1

    def test_backtest_transaction_cost_calculation(self, sample_data, backtest_config, simple_strategy):
        """测试交易成本计算"""
        # 设置较高的交易成本以便测试
        high_cost_config = BacktestConfig(
            start_date=backtest_config.start_date,
            end_date=backtest_config.end_date,
            initial_capital=backtest_config.initial_capital,
            frequency=backtest_config.frequency,
            execution_mode=backtest_config.execution_mode,
            price_mode=backtest_config.price_mode,
            commission_rate=0.01,  # 1% 佣金率
            stamp_tax_rate=0.01   # 1% 印花税率
        )

        engine = MultiFrequencyBacktest(high_cost_config)
        result = engine.run(data=sample_data, strategy=simple_strategy)

        # 验证交易成本被正确计算
        total_commission = sum(float(trade.commission) for trade in result.trades)
        assert total_commission > 0

        # 验证买入和卖出的佣金计算
        buy_trades = [t for t in result.trades if t.trade_type == OrderType.BUY]
        sell_trades = [t for t in result.trades if t.trade_type == OrderType.SELL]

        if buy_trades:
            buy_trade = buy_trades[0]
            expected_buy_commission = float(buy_trade.quantity) * float(buy_trade.price) * 0.01
            assert abs(float(buy_trade.commission) - expected_buy_commission) < 0.01

        if sell_trades:
            sell_trade = sell_trades[0]
            # 卖出佣金 = 佣金率 + 印花税率
            expected_sell_commission = float(sell_trade.quantity) * float(sell_trade.price) * 0.02
            assert abs(float(sell_trade.commission) - expected_sell_commission) < 0.01

    def test_backtest_performance_accuracy(self, sample_data, backtest_config):
        """测试回测性能准确性"""
        # 创建一个确定性策略以便验证准确性
        def deterministic_strategy(data, portfolio, timestamp):
            orders = []

            # 在特定日期执行特定交易
            if timestamp == pd.Timestamp('2023-01-02'):
                orders.append(Order(
                    symbol="000001.SZ",
                    order_type=OrderType.BUY,
                    quantity=1000,
                    price=Decimal("10.0")  # 指定价格
                ))

            return orders

        engine = MultiFrequencyBacktest(backtest_config)
        result = engine.run(data=sample_data, strategy=deterministic_strategy)

        # 手动计算预期结果并验证
        if result.trades:
            buy_trade = result.trades[0]
            expected_cost = float(buy_trade.quantity) * float(buy_trade.price) + float(buy_trade.commission)
            expected_remaining_cash = backtest_config.initial_capital - expected_cost

            # 验证现金余额的准确性（允许小的舍入误差）
            assert abs(float(result.final_cash) - expected_remaining_cash) < 1.0

    def test_backtest_data_frequency_mismatch_handling(self, backtest_config):
        """测试数据频率不匹配处理"""
        # 创建小时频数据但配置为日频回测
        hourly_dates = pd.date_range('2023-01-01 09:00:00', '2023-01-02 16:00:00', freq='1h')
        hourly_data_list = []

        for ts in hourly_dates:
            hourly_data_list.append({
                'symbol': '000001.SZ',
                'datetime': ts,
                'open': 10.0,
                'high': 10.1,
                'low': 9.9,
                'close': 10.0,
                'volume': 1000
            })

        hourly_data = pd.DataFrame(hourly_data_list).set_index(['datetime', 'symbol'])

        engine = MultiFrequencyBacktest(backtest_config)  # 日频配置

        # 策略
        def strategy(data, portfolio, timestamp):
            return []

        # 回测引擎应该能够处理频率不匹配，自动重采样
        result = engine.run(data=hourly_data, strategy=strategy)
        assert isinstance(result, BacktestResult)

    def test_backtest_edge_cases(self, backtest_config):
        """测试边界情况"""
        # 创建只有一天数据的情况
        single_day_data = pd.DataFrame({
            'symbol': ['000001.SZ'],
            'datetime': [pd.Timestamp('2023-01-01')],
            'open': [10.0],
            'high': [10.1],
            'low': [9.9],
            'close': [10.0],
            'volume': [1000]
        }).set_index(['datetime', 'symbol'])

        single_day_config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 1, 1),
            initial_capital=1000000.0,
            frequency="1d"
        )

        engine = MultiFrequencyBacktest(single_day_config)

        def empty_strategy(data, portfolio, timestamp):
            return []

        result = engine.run(data=single_day_data, strategy=empty_strategy)

        # 验证单日回测能正常运行
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_values) >= 1
        assert result.portfolio_values.iloc[0] == single_day_config.initial_capital

    def test_backtest_strategy_exception_handling(self, sample_data, backtest_config):
        """测试策略异常处理"""
        def faulty_strategy(data, portfolio, timestamp):
            # 故意抛出异常的策略
            raise RuntimeError("策略执行错误")

        engine = MultiFrequencyBacktest(backtest_config)

        # 根据开发规则1，不允许捕获异常，应该让异常暴露
        # 因此这里应该直接抛出异常而不是被捕获
        with pytest.raises(RuntimeError, match="策略执行错误"):
            engine.run(data=sample_data, strategy=faulty_strategy)

    def test_backtest_invalid_order_handling(self, sample_data, backtest_config):
        """测试无效订单处理"""
        def invalid_order_strategy(data, portfolio, timestamp):
            if timestamp == pd.Timestamp('2023-01-02'):
                # 返回无效订单（负数量）
                return [Order(
                    symbol="000001.SZ",
                    order_type=OrderType.BUY,
                    quantity=-1000,  # 无效数量
                    price=Decimal("10.0")
                )]
            return []

        engine = MultiFrequencyBacktest(backtest_config)

        # 根据开发规则1，不允许捕获异常，应该让异常暴露
        # 无效订单应该直接抛出ValueError
        with pytest.raises(ValueError, match="订单数量必须为正数"):
            engine.run(data=sample_data, strategy=invalid_order_strategy)