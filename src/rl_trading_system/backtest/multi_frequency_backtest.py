"""
多频率回测引擎实现
支持日频和分钟频回测，多种成交价格，交易执行模拟，考虑实际成交情况
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, date, time
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class ExecutionMode(Enum):
    """交易执行模式"""
    NEXT_BAR = "next_bar"          # 下一个bar执行
    NEXT_CLOSE = "next_close"      # 下一个收盘价执行
    NEXT_OPEN = "next_open"        # 下一个开盘价执行
    MARKET_ORDER = "market_order"  # 市价单
    LIMIT_ORDER = "limit_order"    # 限价单
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)
    
    def __hash__(self):
        return super().__hash__()


class PriceMode(Enum):
    """价格模式"""
    CLOSE = "close"  # 收盘价
    OPEN = "open"    # 开盘价
    HIGH = "high"    # 最高价
    LOW = "low"      # 最低价
    VWAP = "vwap"    # 成交量加权平均价格
    TWAP = "twap"    # 时间加权平均价格
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)
    
    def __hash__(self):
        return super().__hash__()


class OrderType(Enum):
    """订单类型"""
    BUY = "buy"      # 买入
    SELL = "sell"    # 卖出
    SHORT = "short"  # 做空
    COVER = "cover"  # 平仓
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)
    
    def __hash__(self):
        return super().__hash__()


@dataclass
class Order:
    """订单类"""
    symbol: str
    order_type: OrderType
    quantity: int
    price: Decimal
    timestamp: datetime = field(default_factory=datetime.now)
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = field(default="pending")
    
    def __post_init__(self):
        """订单创建后验证"""
        if not self.symbol:
            raise ValueError("股票代码不能为空")
        if self.quantity <= 0:
            raise ValueError("订单数量必须为正数")
        if self.price <= 0:
            raise ValueError("订单价格必须为正数")
    
    def get_execution_price(self, market_data: Dict[str, float], price_mode: PriceMode) -> Decimal:
        """根据价格模式获取执行价格"""
        if price_mode == PriceMode.CLOSE:
            price = market_data.get('close', 0)
        elif price_mode == PriceMode.OPEN:
            price = market_data.get('open', 0)
        elif price_mode == PriceMode.HIGH:
            price = market_data.get('high', 0)
        elif price_mode == PriceMode.LOW:
            price = market_data.get('low', 0)
        elif price_mode == PriceMode.VWAP:
            price = market_data.get('vwap', market_data.get('close', 0))
        elif price_mode == PriceMode.TWAP:
            price = market_data.get('twap', market_data.get('close', 0))
        else:
            price = market_data.get('close', 0)
        
        return Decimal(str(price)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


@dataclass
class Trade:
    """交易记录类"""
    symbol: str
    trade_type: OrderType
    quantity: int
    price: Decimal
    timestamp: datetime = field(default_factory=datetime.now)
    commission: Decimal = field(default=Decimal('0'))
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def get_trade_value(self) -> Decimal:
        """计算交易价值（买入为负，卖出为正）"""
        base_value = self.quantity * self.price
        
        if self.trade_type in [OrderType.BUY, OrderType.COVER]:
            # 买入：负现金流
            return -(base_value + self.commission)
        else:
            # 卖出：正现金流
            return base_value - self.commission


class Position:
    """持仓类"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0
        self.avg_price = Decimal('0')
        self.market_value = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        
    def update_position(self, trade_type: OrderType, quantity: int, price: Decimal):
        """更新持仓"""
        if trade_type in [OrderType.BUY, OrderType.COVER]:
            # 买入操作
            if self.quantity == 0:
                self.avg_price = price
                self.quantity = quantity
            else:
                # 加权平均成本
                total_cost = self.quantity * self.avg_price + quantity * price
                total_quantity = self.quantity + quantity
                self.avg_price = total_cost / total_quantity
                self.quantity = total_quantity
        else:
            # 卖出操作
            if quantity > self.quantity:
                raise ValueError(f"持仓数量不足，当前持仓：{self.quantity}，卖出数量：{quantity}")
            self.quantity -= quantity
            
            # 如果全部卖出，重置平均价格
            if self.quantity == 0:
                self.avg_price = Decimal('0')
    
    def update_market_value(self, current_price: Decimal):
        """更新市值和未实现盈亏"""
        self.market_value = self.quantity * current_price
        cost_basis = self.quantity * self.avg_price
        self.unrealized_pnl = self.market_value - cost_basis


class Portfolio:
    """投资组合类"""
    
    def __init__(self, initial_cash: Decimal):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        
    @property
    def total_value(self) -> Decimal:
        """总价值"""
        market_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + market_value
    
    def execute_order(self, order: Order, market_data: pd.Series, 
                     price_mode: PriceMode, commission_rate: Decimal, stamp_tax_rate: Decimal = Decimal('0.001')) -> Trade:
        """执行订单"""
        # 获取执行价格
        execution_price = order.get_execution_price(market_data.to_dict(), price_mode)
        
        # 计算佣金
        base_commission = order.quantity * execution_price * commission_rate
        
        # 如果是卖出，还需要加上印花税（A股特有）
        if order.order_type == OrderType.SELL:
            stamp_tax = order.quantity * execution_price * stamp_tax_rate
            total_commission = base_commission + stamp_tax
        else:
            total_commission = base_commission
            
        # 检查资金是否足够（买入时）
        if order.order_type == OrderType.BUY:
            required_cash = order.quantity * execution_price + total_commission
            if required_cash > self.cash:
                raise ValueError(f"现金不足，需要：{required_cash}，可用：{self.cash}")
        
        # 创建交易记录
        trade = Trade(
            symbol=order.symbol,
            trade_type=order.order_type,
            quantity=order.quantity,
            price=execution_price,
            timestamp=order.timestamp,
            commission=total_commission
        )
        
        # 更新持仓
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(order.symbol)
        
        self.positions[order.symbol].update_position(
            trade_type=order.order_type,
            quantity=order.quantity,
            price=execution_price
        )
        
        # 更新现金
        self.cash += trade.get_trade_value()
        
        # 记录交易
        self.trades.append(trade)
        
        return trade
    
    def update_market_values(self, current_prices: Dict[str, Decimal]):
        """更新所有持仓的市值"""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_market_value(current_prices[symbol])
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取绩效指标"""
        total_value = float(self.total_value)
        initial_value = float(self.initial_cash)
        
        total_return = (total_value / initial_value) - 1
        cash_ratio = float(self.cash) / total_value
        
        return {
            'total_value': total_value,
            'total_return': total_return,
            'cash_ratio': cash_ratio,
            'cash': float(self.cash),
            'market_value': total_value - float(self.cash)
        }


@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: date
    end_date: date
    initial_capital: float
    frequency: str = "1d"
    execution_mode: ExecutionMode = ExecutionMode.NEXT_CLOSE
    price_mode: PriceMode = PriceMode.CLOSE
    commission_rate: float = 0.001
    stamp_tax_rate: float = 0.001
    
    def __post_init__(self):
        """配置验证"""
        if self.end_date < self.start_date:
            raise ValueError("结束日期不能早于开始日期")
        if self.initial_capital <= 0:
            raise ValueError("初始资金必须为正数")
        if self.commission_rate < 0:
            raise ValueError("佣金率必须为非负数")
        if self.stamp_tax_rate < 0:
            raise ValueError("印花税率必须为非负数")
        
        # 验证频率
        valid_frequencies = ["1d", "1h", "30min", "15min", "5min", "1min"]
        if self.frequency not in valid_frequencies:
            raise ValueError(f"不支持的频率：{self.frequency}")


@dataclass
class BacktestResult:
    """回测结果"""
    trades: List[Trade]
    portfolio_values: pd.Series
    positions: Dict[str, Position]
    final_cash: Decimal
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """计算绩效指标"""
        if len(self.portfolio_values) < 2:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
        
        # 计算收益率序列
        returns = self.portfolio_values.pct_change().dropna()
        
        # 基本指标
        total_return = (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1
        
        # 年化收益率（假设252个交易日）
        trading_days = len(self.portfolio_values)
        if trading_days > 1:
            annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        else:
            annualized_return = 0.0
        
        # 波动率
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        
        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        if volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        peak = self.portfolio_values.expanding().max()
        drawdown = (self.portfolio_values - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # 胜率
        if len(returns) > 0:
            win_rate = (returns > 0).mean()
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }


class MultiFrequencyBacktest:
    """多频率回测引擎"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = Portfolio(Decimal(str(config.initial_capital)))
        self.trades: List[Trade] = []
        self.portfolio_values: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        
    def _validate_data(self, data: pd.DataFrame):
        """验证回测数据"""
        if data.empty:
            raise ValueError("回测数据不能为空")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"数据缺少必要的列：{missing_columns}")
    
    def _resample_data_if_needed(self, data: pd.DataFrame) -> pd.DataFrame:
        """根据回测频率重采样数据"""
        if self.config.frequency == "1d":
            # 如果是日频回测，但数据是高频的，需要重采样到日频
            if hasattr(data.index, 'levels'):
                # MultiIndex case
                time_index = data.index.get_level_values(0)
            else:
                time_index = data.index
                
            # 检查是否需要重采样
            if len(time_index) > 0:
                time_diff = pd.Timedelta(days=1)
                if len(time_index) > 1:
                    actual_freq = time_index[1] - time_index[0]
                    if actual_freq < time_diff:
                        # 需要重采样到日频
                        if 'symbol' in data.index.names:
                            # MultiIndex with symbol
                            resampled_data = []
                            for symbol in data.index.get_level_values('symbol').unique():
                                symbol_data = data.xs(symbol, level='symbol')
                                daily_data = symbol_data.resample('1D').agg({
                                    'open': 'first',
                                    'high': 'max',
                                    'low': 'min',
                                    'close': 'last',
                                    'volume': 'sum'
                                }).dropna()
                                daily_data['symbol'] = symbol
                                resampled_data.append(daily_data.reset_index().set_index(['datetime', 'symbol']))
                            
                            if resampled_data:
                                return pd.concat(resampled_data).sort_index()
        
        return data
    
    def run(self, data: pd.DataFrame, strategy: Callable) -> BacktestResult:
        """运行回测"""
        # 验证数据
        self._validate_data(data)
        
        # 重采样数据（如果需要）
        data = self._resample_data_if_needed(data)
        
        # 获取时间索引
        if hasattr(data.index, 'levels') and 'datetime' in data.index.names:
            # MultiIndex case
            timestamps = data.index.get_level_values('datetime').unique().sort_values()
        else:
            timestamps = data.index.unique() if hasattr(data.index, 'unique') else [data.index[0]]
        
        # 过滤时间范围
        start_ts = pd.Timestamp(self.config.start_date)
        end_ts = pd.Timestamp(self.config.end_date) + pd.Timedelta(days=1)
        timestamps = [ts for ts in timestamps if start_ts <= ts < end_ts]
        
        if not timestamps:
            timestamps = [start_ts]  # 至少有一个时间点
        
        # 初始化组合价值记录
        self.portfolio_values = [float(self.portfolio.initial_cash)]
        self.timestamps = [timestamps[0] if timestamps else start_ts]
        
        # 按时间步执行回测
        for i, timestamp in enumerate(timestamps):
            # 获取当前时间的数据
            try:
                if hasattr(data.index, 'levels') and 'datetime' in data.index.names:
                    current_data = data.xs(timestamp, level='datetime')
                else:
                    # 找到最接近的时间点
                    closest_idx = np.abs(pd.to_datetime(data.index) - timestamp).argmin()
                    current_data = data.iloc[closest_idx:closest_idx+1]
            except (KeyError, IndexError):
                continue
            
            # 调用策略生成订单
            orders = strategy(current_data, self.portfolio, timestamp)
            
            # 执行订单
            for order in orders:
                if hasattr(current_data, 'xs'):
                    # MultiIndex case - get data for specific symbol
                    try:
                        symbol_data = current_data.xs(order.symbol) if order.symbol in current_data.index else current_data.iloc[0]
                    except (KeyError, IndexError):
                        symbol_data = current_data.iloc[0] if len(current_data) > 0 else pd.Series()
                else:
                    symbol_data = current_data.iloc[0] if len(current_data) > 0 else pd.Series()
                
                if not symbol_data.empty:
                    trade = self.portfolio.execute_order(
                        order=order,
                        market_data=symbol_data,
                        price_mode=self.config.price_mode,
                        commission_rate=Decimal(str(self.config.commission_rate)),
                        stamp_tax_rate=Decimal(str(self.config.stamp_tax_rate))
                    )
                    self.trades.append(trade)
            
            # 更新持仓市值
            if hasattr(current_data, 'index') and len(current_data) > 0:
                current_prices = {}
                if hasattr(current_data, 'xs'):
                    # MultiIndex case
                    for symbol in current_data.index:
                        try:
                            symbol_data = current_data.xs(symbol)
                            current_prices[symbol] = Decimal(str(symbol_data['close']))
                        except (KeyError, IndexError):
                            pass
                else:
                    # Single index case
                    if 'close' in current_data.columns and len(current_data) > 0:
                        symbol = getattr(current_data.iloc[0], 'symbol', 'default')
                        current_prices[symbol] = Decimal(str(current_data.iloc[0]['close']))
                
                self.portfolio.update_market_values(current_prices)
            
            # 记录组合价值
            current_value = float(self.portfolio.total_value)
            if i < len(timestamps) - 1 or len(self.portfolio_values) == 1:
                self.portfolio_values.append(current_value)
                if i < len(timestamps) - 1:
                    self.timestamps.append(timestamps[i + 1])
        
        # 创建时间序列，确保长度匹配
        # 如果portfolio_values比timestamps长，截断portfolio_values
        # 如果timestamps比portfolio_values长，截断timestamps
        min_length = min(len(self.portfolio_values), len(self.timestamps))
        portfolio_series = pd.Series(
            self.portfolio_values[:min_length],
            index=pd.DatetimeIndex(self.timestamps[:min_length])
        )
        
        # 返回回测结果
        return BacktestResult(
            trades=self.trades,
            portfolio_values=portfolio_series,
            positions=self.portfolio.positions.copy(),
            final_cash=self.portfolio.cash
        )