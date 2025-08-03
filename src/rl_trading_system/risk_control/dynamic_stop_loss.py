"""动态止损控制器实现"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod


class StopLossType(Enum):
    """止损类型枚举"""
    FIXED = "fixed"                    # 固定止损
    PERCENTAGE = "percentage"          # 百分比止损
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # 波动率调整止损
    TRAILING = "trailing"              # 追踪止损


class StopLossStatus(Enum):
    """止损状态枚举"""
    ACTIVE = "active"                  # 激活状态
    TRIGGERED = "triggered"            # 已触发
    DISABLED = "disabled"              # 已禁用
    EXPIRED = "expired"                # 已过期


@dataclass
class Position:
    """持仓数据类"""
    symbol: str
    quantity: float
    current_price: float
    cost_basis: float
    sector: str
    timestamp: datetime
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    weight: Optional[float] = None
    
    def __post_init__(self):
        if self.market_value is None:
            self.market_value = self.quantity * self.current_price
        if self.unrealized_pnl is None:
            self.unrealized_pnl = (self.current_price - self.cost_basis) * self.quantity
        if self.weight is None:
            self.weight = 0.0


@dataclass
class Portfolio:
    """投资组合数据类"""
    positions: List[Position]
    cash: float
    total_value: float
    timestamp: datetime
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定股票的持仓"""
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None


@dataclass
class StopLossOrder:
    """止损订单数据类"""
    symbol: str
    quantity: float
    stop_price: float
    stop_type: StopLossType
    status: StopLossStatus
    created_time: datetime
    triggered_time: Optional[datetime] = None
    reason: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TrailingStopRecord:
    """追踪止损记录"""
    symbol: str
    initial_price: float                       # 初始价格
    peak_price: float                          # 峰值价格
    current_stop_price: float                  # 当前止损价格
    activation_time: datetime                  # 激活时间
    last_update_time: datetime                 # 最后更新时间
    profit_locked: float                       # 已锁定利润
    distance_ratio: float                      # 止损距离比例
    update_count: int = 0                      # 更新次数
    
    def calculate_locked_profit_ratio(self, cost_basis: float) -> float:
        """计算已锁定利润比例"""
        if cost_basis <= 0:
            return 0.0
        return max(0.0, (self.current_stop_price - cost_basis) / cost_basis)


@dataclass
class PortfolioStopLossRecord:
    """组合级止损记录"""
    trigger_time: datetime
    trigger_type: str                          # 'full_stop' 或 'partial_stop'
    portfolio_pnl_ratio: float                # 触发时的组合损益比例
    threshold: float                           # 触发阈值
    action_taken: str                          # 采取的行动
    positions_liquidated: List[str]            # 被清仓的持仓
    recovery_threshold: float                  # 恢复阈值
    recovery_time: Optional[datetime] = None   # 恢复时间
    is_recovered: bool = False                 # 是否已恢复


@dataclass
class StopLossConfig:
    """止损配置参数"""
    # 基础止损参数
    base_stop_loss: float = 0.05               # 基础止损阈值 (5%)
    min_stop_loss: float = 0.02                # 最小止损阈值 (2%)
    max_stop_loss: float = 0.15                # 最大止损阈值 (15%)
    
    # 波动率调整参数
    volatility_multiplier: float = 2.0         # 波动率乘数
    volatility_lookback_days: int = 20         # 波动率计算回望天数
    volatility_adjustment_factor: float = 0.5  # 波动率调整因子
    
    # 追踪止损参数
    trailing_stop_distance: float = 0.03       # 追踪止损距离 (3%)
    trailing_activation_threshold: float = 0.02 # 追踪止损激活阈值 (2%盈利)
    trailing_update_frequency: int = 1          # 追踪止损更新频率 (分钟)
    trailing_min_distance: float = 0.01        # 最小追踪距离 (1%)
    trailing_max_distance: float = 0.08        # 最大追踪距离 (8%)
    trailing_acceleration_factor: float = 1.5  # 追踪加速因子
    trailing_deceleration_threshold: float = 0.1 # 追踪减速阈值
    
    # 自适应追踪止损参数
    adaptive_trailing_enabled: bool = True     # 启用自适应追踪止损
    volatility_based_distance: bool = True     # 基于波动率的距离调整
    profit_based_tightening: bool = True       # 基于利润的收紧机制
    time_based_adjustment: bool = True         # 基于时间的调整机制
    
    # 组合级止损参数
    portfolio_stop_loss: float = 0.12          # 组合级止损阈值 (12%)
    portfolio_partial_stop: float = 0.08       # 组合部分止损阈值 (8%)
    partial_stop_ratio: float = 0.3            # 部分止损比例 (30%)
    portfolio_recovery_threshold: float = 0.05 # 组合恢复阈值 (5%)
    
    # 组合级止损策略参数
    liquidation_strategy: str = "worst_performers"  # 清仓策略: worst_performers, largest_positions, sector_based
    max_liquidation_per_batch: int = 5         # 每批最大清仓数量
    liquidation_delay_minutes: int = 5         # 清仓间隔时间（分钟）
    recovery_monitoring_enabled: bool = True   # 启用恢复监控
    auto_rebalance_on_recovery: bool = True    # 恢复时自动再平衡
    
    # 市场状态调整参数
    high_volatility_threshold: float = 0.3     # 高波动率阈值
    low_volatility_threshold: float = 0.1      # 低波动率阈值
    bear_market_adjustment: float = 1.2        # 熊市调整因子
    bull_market_adjustment: float = 0.8        # 牛市调整因子


class DynamicStopLoss:
    """动态止损控制器"""
    
    def __init__(self, config: StopLossConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 止损状态跟踪
        self.stop_loss_levels: Dict[str, Dict[str, float]] = {}  # 各类型止损水平
        self.trailing_stops: Dict[str, float] = {}               # 追踪止损价格
        self.trailing_stop_records: Dict[str, TrailingStopRecord] = {}  # 追踪止损详细记录
        self.stop_loss_history: List[StopLossOrder] = []         # 止损历史记录
        self.active_orders: Dict[str, StopLossOrder] = {}        # 活跃止损订单
        
        # 组合级止损状态
        self.portfolio_stop_loss_active: bool = False           # 组合级止损是否激活
        self.portfolio_stop_loss_records: List[PortfolioStopLossRecord] = []  # 组合级止损记录
        self.portfolio_peak_value: float = 0.0                  # 组合峰值
        self.last_liquidation_time: Optional[datetime] = None   # 最后清仓时间
        
        # 市场数据缓存
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.volatility_cache: Dict[str, float] = {}
        
        # 统计信息
        self.stats = {
            'total_stops_triggered': 0,
            'successful_stops': 0,
            'false_stops': 0,
            'average_stop_loss': 0.0
        }
    
    def calculate_adaptive_stop_loss(self, position: Position, 
                                   market_volatility: Optional[float] = None) -> Dict[str, float]:
        """计算自适应止损水平"""
        symbol = position.symbol
        
        # 计算各种类型的止损水平
        stop_levels = {}
        
        # 1. 固定止损
        stop_levels['fixed'] = self._calculate_fixed_stop_loss(position)
        
        # 2. 百分比止损
        stop_levels['percentage'] = self._calculate_percentage_stop_loss(position)
        
        # 3. 波动率调整止损
        volatility = market_volatility or self._calculate_volatility(symbol)
        stop_levels['volatility_adjusted'] = self._calculate_volatility_adjusted_stop_loss(
            position, volatility
        )
        
        # 4. 追踪止损
        if symbol in self.trailing_stops:
            stop_levels['trailing'] = self.trailing_stops[symbol]
        else:
            stop_levels['trailing'] = 0.0
        
        # 更新止损水平缓存
        self.stop_loss_levels[symbol] = stop_levels
        
        self.logger.debug(f"计算 {symbol} 止损水平: {stop_levels}")
        
        return stop_levels
    
    def _calculate_fixed_stop_loss(self, position: Position) -> float:
        """计算固定止损价格"""
        return position.cost_basis * (1 - self.config.base_stop_loss)
    
    def _calculate_percentage_stop_loss(self, position: Position) -> float:
        """计算百分比止损价格"""
        # 基于当前价格的百分比止损
        return position.current_price * (1 - self.config.base_stop_loss)
    
    def _calculate_volatility_adjusted_stop_loss(self, position: Position, 
                                               volatility: float) -> float:
        """计算波动率调整止损价格"""
        # 基础止损阈值
        base_threshold = self.config.base_stop_loss
        
        # 根据波动率调整止损阈值
        if volatility > self.config.high_volatility_threshold:
            # 高波动率时放宽止损
            adjusted_threshold = base_threshold * (1 + volatility * self.config.volatility_adjustment_factor)
        elif volatility < self.config.low_volatility_threshold:
            # 低波动率时收紧止损
            adjusted_threshold = base_threshold * (1 - volatility * self.config.volatility_adjustment_factor)
        else:
            # 正常波动率
            adjusted_threshold = base_threshold * (1 + (volatility - self.config.low_volatility_threshold) * 
                                                 self.config.volatility_multiplier)
        
        # 限制在合理范围内
        adjusted_threshold = np.clip(adjusted_threshold, 
                                   self.config.min_stop_loss, 
                                   self.config.max_stop_loss)
        
        return position.cost_basis * (1 - adjusted_threshold)
    
    def _calculate_volatility(self, symbol: str) -> float:
        """计算股票的历史波动率"""
        if symbol in self.volatility_cache:
            return self.volatility_cache[symbol]
        
        # 获取历史价格数据
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            # 如果没有足够的历史数据，使用默认波动率
            default_volatility = 0.2  # 20%年化波动率
            self.volatility_cache[symbol] = default_volatility
            return default_volatility
        
        # 计算收益率
        prices = [price for _, price in self.price_history[symbol][-self.config.volatility_lookback_days:]]
        if len(prices) < 2:
            default_volatility = 0.2
            self.volatility_cache[symbol] = default_volatility
            return default_volatility
        
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        # 计算年化波动率
        if returns:
            daily_volatility = np.std(returns)
            annual_volatility = daily_volatility * np.sqrt(252)  # 假设252个交易日
        else:
            annual_volatility = 0.2
        
        self.volatility_cache[symbol] = annual_volatility
        return annual_volatility
    
    def update_trailing_stops(self, positions: List[Position]) -> Dict[str, float]:
        """更新追踪止损价格"""
        updated_stops = {}
        current_time = datetime.now()
        
        for position in positions:
            if position.quantity <= 0:
                continue
            
            symbol = position.symbol
            current_price = position.current_price
            cost_basis = position.cost_basis
            
            # 检查是否满足追踪止损激活条件（盈利超过阈值）
            profit_ratio = (current_price - cost_basis) / cost_basis
            if profit_ratio < self.config.trailing_activation_threshold:
                # 如果不满足激活条件，移除现有的追踪止损
                if symbol in self.trailing_stops:
                    del self.trailing_stops[symbol]
                if symbol in self.trailing_stop_records:
                    del self.trailing_stop_records[symbol]
                continue
            
            # 计算自适应追踪止损距离
            trailing_distance = self._calculate_adaptive_trailing_distance(
                symbol, current_price, cost_basis, profit_ratio
            )
            
            # 计算新的追踪止损价格
            new_stop_price = current_price * (1 - trailing_distance)
            
            # 初始化或更新追踪止损记录
            if symbol not in self.trailing_stop_records:
                # 首次激活追踪止损
                self.trailing_stop_records[symbol] = TrailingStopRecord(
                    symbol=symbol,
                    initial_price=cost_basis,  # 使用成本基础而不是当前价格
                    peak_price=current_price,
                    current_stop_price=new_stop_price,
                    activation_time=current_time,
                    last_update_time=current_time,
                    profit_locked=max(0.0, new_stop_price - cost_basis),
                    distance_ratio=trailing_distance
                )
                self.trailing_stops[symbol] = new_stop_price
                updated_stops[symbol] = new_stop_price
                
                self.logger.info(f"激活 {symbol} 追踪止损: 价格={current_price:.2f}, 止损={new_stop_price:.2f}, 距离={trailing_distance:.2%}")
            
            else:
                # 更新现有追踪止损
                record = self.trailing_stop_records[symbol]
                
                # 更新峰值价格
                if current_price > record.peak_price:
                    record.peak_price = current_price
                
                # 只有当新止损价格更高时才更新（向上追踪）
                if new_stop_price > record.current_stop_price:
                    old_stop = record.current_stop_price
                    record.current_stop_price = new_stop_price
                    record.last_update_time = current_time
                    record.profit_locked = new_stop_price - cost_basis
                    record.distance_ratio = trailing_distance
                    record.update_count += 1
                    
                    self.trailing_stops[symbol] = new_stop_price
                    updated_stops[symbol] = new_stop_price
                    
                    self.logger.info(f"更新 {symbol} 追踪止损: {old_stop:.2f} -> {new_stop_price:.2f}, 距离={trailing_distance:.2%}")
        
        return updated_stops
    
    def _calculate_adaptive_trailing_distance(self, symbol: str, current_price: float, 
                                            cost_basis: float, profit_ratio: float) -> float:
        """计算自适应追踪止损距离"""
        base_distance = self.config.trailing_stop_distance
        
        if not self.config.adaptive_trailing_enabled:
            return base_distance
        
        # 1. 基于波动率的距离调整
        if self.config.volatility_based_distance:
            volatility = self._calculate_volatility(symbol)
            if volatility > self.config.high_volatility_threshold:
                # 高波动率时增加追踪距离
                volatility_adjustment = 1 + (volatility - self.config.high_volatility_threshold) * 2
            elif volatility < self.config.low_volatility_threshold:
                # 低波动率时减少追踪距离
                volatility_adjustment = 1 - (self.config.low_volatility_threshold - volatility) * 0.5
            else:
                volatility_adjustment = 1.0
            
            base_distance *= volatility_adjustment
        
        # 2. 基于利润的收紧机制
        if self.config.profit_based_tightening:
            if profit_ratio > self.config.trailing_deceleration_threshold:
                # 高利润时逐渐收紧追踪距离
                tightening_factor = 1 - (profit_ratio - self.config.trailing_deceleration_threshold) * 0.3
                base_distance *= max(tightening_factor, 0.5)  # 最多收紧50%
        
        # 3. 基于时间的调整机制
        if self.config.time_based_adjustment and symbol in self.trailing_stop_records:
            record = self.trailing_stop_records[symbol]
            holding_duration = (datetime.now() - record.activation_time).total_seconds() / 3600  # 小时
            
            if holding_duration > 24:  # 持有超过24小时
                # 长期持有时逐渐收紧追踪距离
                time_adjustment = 1 - min(holding_duration / 168, 0.3)  # 最多收紧30%（一周后）
                base_distance *= time_adjustment
        
        # 限制在合理范围内
        return np.clip(base_distance, self.config.trailing_min_distance, self.config.trailing_max_distance)
    
    def get_trailing_stop_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取追踪止损详细信息"""
        if symbol not in self.trailing_stop_records:
            return None
        
        record = self.trailing_stop_records[symbol]
        current_time = datetime.now()
        
        return {
            'symbol': symbol,
            'initial_price': record.initial_price,
            'peak_price': record.peak_price,
            'current_stop_price': record.current_stop_price,
            'activation_time': record.activation_time,
            'last_update_time': record.last_update_time,
            'holding_duration_hours': (current_time - record.activation_time).total_seconds() / 3600,
            'profit_locked': record.profit_locked,
            'distance_ratio': record.distance_ratio,
            'update_count': record.update_count,
            'locked_profit_ratio': max(0.0, (record.current_stop_price - record.initial_price) / record.initial_price)
        }
    
    def get_all_trailing_stops_summary(self) -> Dict[str, Any]:
        """获取所有追踪止损的汇总信息"""
        if not self.trailing_stop_records:
            return {
                'total_active_trailing_stops': 0,
                'total_locked_profit': 0.0,
                'average_distance_ratio': 0.0,
                'symbols': []
            }
        
        total_locked_profit = sum(record.profit_locked for record in self.trailing_stop_records.values())
        average_distance = np.mean([record.distance_ratio for record in self.trailing_stop_records.values()])
        
        symbols_info = []
        for symbol, record in self.trailing_stop_records.items():
            symbols_info.append({
                'symbol': symbol,
                'stop_price': record.current_stop_price,
                'locked_profit': record.profit_locked,
                'distance_ratio': record.distance_ratio,
                'update_count': record.update_count
            })
        
        return {
            'total_active_trailing_stops': len(self.trailing_stop_records),
            'total_locked_profit': total_locked_profit,
            'average_distance_ratio': average_distance,
            'symbols': symbols_info
        }
    
    def check_stop_loss_triggers(self, positions: List[Position]) -> List[StopLossOrder]:
        """检查止损触发条件"""
        triggered_orders = []
        
        for position in positions:
            if position.quantity <= 0:
                continue
            
            symbol = position.symbol
            current_price = position.current_price
            
            # 获取各类型止损水平
            stop_levels = self.stop_loss_levels.get(symbol, {})
            
            # 检查各种止损类型
            triggered_stop = None
            
            # 1. 检查固定止损
            if 'fixed' in stop_levels and current_price <= stop_levels['fixed']:
                triggered_stop = StopLossOrder(
                    symbol=symbol,
                    quantity=position.quantity,
                    stop_price=stop_levels['fixed'],
                    stop_type=StopLossType.FIXED,
                    status=StopLossStatus.TRIGGERED,
                    created_time=datetime.now(),
                    triggered_time=datetime.now(),
                    reason=f"价格{current_price:.2f}触发固定止损{stop_levels['fixed']:.2f}"
                )
            
            # 2. 检查波动率调整止损
            elif 'volatility_adjusted' in stop_levels and current_price <= stop_levels['volatility_adjusted']:
                triggered_stop = StopLossOrder(
                    symbol=symbol,
                    quantity=position.quantity,
                    stop_price=stop_levels['volatility_adjusted'],
                    stop_type=StopLossType.VOLATILITY_ADJUSTED,
                    status=StopLossStatus.TRIGGERED,
                    created_time=datetime.now(),
                    triggered_time=datetime.now(),
                    reason=f"价格{current_price:.2f}触发波动率调整止损{stop_levels['volatility_adjusted']:.2f}"
                )
            
            # 3. 检查追踪止损
            elif symbol in self.trailing_stops and current_price <= self.trailing_stops[symbol]:
                triggered_stop = StopLossOrder(
                    symbol=symbol,
                    quantity=position.quantity,
                    stop_price=self.trailing_stops[symbol],
                    stop_type=StopLossType.TRAILING,
                    status=StopLossStatus.TRIGGERED,
                    created_time=datetime.now(),
                    triggered_time=datetime.now(),
                    reason=f"价格{current_price:.2f}触发追踪止损{self.trailing_stops[symbol]:.2f}"
                )
            
            if triggered_stop:
                triggered_orders.append(triggered_stop)
                self.stop_loss_history.append(triggered_stop)
                self.stats['total_stops_triggered'] += 1
                
                self.logger.warning(f"止损触发: {triggered_stop.reason}")
        
        return triggered_orders
    
    def execute_stop_loss_orders(self, triggered_orders: List[StopLossOrder]) -> List[Dict[str, Any]]:
        """执行止损订单"""
        executed_orders = []
        
        for order in triggered_orders:
            try:
                # 模拟订单执行
                execution_result = {
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'executed_price': order.stop_price,
                    'execution_time': datetime.now(),
                    'order_type': 'MARKET_SELL',
                    'stop_type': order.stop_type.value,
                    'status': 'EXECUTED',
                    'reason': order.reason
                }
                
                executed_orders.append(execution_result)
                
                # 更新订单状态
                order.status = StopLossStatus.TRIGGERED
                self.active_orders[order.symbol] = order
                
                # 清除追踪止损记录
                if order.symbol in self.trailing_stops:
                    del self.trailing_stops[order.symbol]
                if order.symbol in self.trailing_stop_records:
                    del self.trailing_stop_records[order.symbol]
                
                self.logger.info(f"执行止损订单: {order.symbol} {order.quantity}股 @ {order.stop_price:.2f}")
                
            except Exception as e:
                self.logger.error(f"执行止损订单失败 {order.symbol}: {str(e)}")
                execution_result = {
                    'symbol': order.symbol,
                    'status': 'FAILED',
                    'error': str(e)
                }
                executed_orders.append(execution_result)
        
        return executed_orders
    
    def check_portfolio_stop_loss(self, portfolio: Portfolio) -> Optional[Dict[str, Any]]:
        """检查组合级止损"""
        if not portfolio.positions:
            return None
        
        # 更新组合峰值
        if portfolio.total_value > self.portfolio_peak_value:
            self.portfolio_peak_value = portfolio.total_value
        
        # 计算组合总的未实现损益
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in portfolio.positions)
        portfolio_pnl_ratio = total_unrealized_pnl / portfolio.total_value if portfolio.total_value > 0 else 0
        
        # 计算从峰值的回撤
        drawdown_from_peak = (portfolio.total_value - self.portfolio_peak_value) / self.portfolio_peak_value if self.portfolio_peak_value > 0 else 0
        
        # 检查是否已经在组合级止损状态
        if self.portfolio_stop_loss_active:
            return self._check_portfolio_recovery(portfolio, portfolio_pnl_ratio, drawdown_from_peak)
        
        # 检查是否触发组合级止损
        if drawdown_from_peak <= -self.config.portfolio_stop_loss:
            return self._trigger_portfolio_stop_loss(
                portfolio, 'full_stop', drawdown_from_peak, -self.config.portfolio_stop_loss
            )
        
        # 检查是否触发部分止损
        elif drawdown_from_peak <= -self.config.portfolio_partial_stop:
            return self._trigger_portfolio_stop_loss(
                portfolio, 'partial_stop', drawdown_from_peak, -self.config.portfolio_partial_stop
            )
        
        return None
    
    def _trigger_portfolio_stop_loss(self, portfolio: Portfolio, trigger_type: str, 
                                   pnl_ratio: float, threshold: float) -> Dict[str, Any]:
        """触发组合级止损"""
        current_time = datetime.now()
        
        # 确定清仓策略和目标持仓
        if trigger_type == 'full_stop':
            action = 'liquidate_all'
            positions_to_liquidate = [pos.symbol for pos in portfolio.positions if pos.quantity > 0]
            liquidation_ratio = 1.0
        else:
            action = 'partial_liquidation'
            positions_to_liquidate = self._select_positions_for_liquidation(
                portfolio, self.config.partial_stop_ratio
            )
            liquidation_ratio = self.config.partial_stop_ratio
        
        # 创建组合级止损记录
        record = PortfolioStopLossRecord(
            trigger_time=current_time,
            trigger_type=trigger_type,
            portfolio_pnl_ratio=pnl_ratio,
            threshold=threshold,
            action_taken=action,
            positions_liquidated=positions_to_liquidate,
            recovery_threshold=self.config.portfolio_recovery_threshold
        )
        
        self.portfolio_stop_loss_records.append(record)
        self.portfolio_stop_loss_active = True
        self.last_liquidation_time = current_time
        
        self.logger.warning(f"触发组合级止损: {trigger_type}, 回撤={pnl_ratio:.2%}, 清仓{len(positions_to_liquidate)}个持仓")
        
        return {
            'trigger_type': trigger_type,
            'pnl_ratio': pnl_ratio,
            'threshold': threshold,
            'action': action,
            'positions_to_liquidate': positions_to_liquidate,
            'liquidation_ratio': liquidation_ratio,
            'liquidation_strategy': self.config.liquidation_strategy,
            'reason': f"组合回撤{pnl_ratio:.2%}触发{trigger_type}止损",
            'record_id': len(self.portfolio_stop_loss_records) - 1
        }
    
    def _select_positions_for_liquidation(self, portfolio: Portfolio, liquidation_ratio: float) -> List[str]:
        """选择需要清仓的持仓"""
        positions_with_loss = [pos for pos in portfolio.positions if pos.quantity > 0]
        
        if not positions_with_loss:
            return []
        
        if self.config.liquidation_strategy == "worst_performers":
            # 按未实现损益排序，优先清仓表现最差的
            positions_with_loss.sort(key=lambda x: x.unrealized_pnl)
            
        elif self.config.liquidation_strategy == "largest_positions":
            # 按市值排序，优先清仓最大的持仓
            positions_with_loss.sort(key=lambda x: x.market_value, reverse=True)
            
        elif self.config.liquidation_strategy == "sector_based":
            # 按行业分散清仓
            sector_positions = {}
            for pos in positions_with_loss:
                if pos.sector not in sector_positions:
                    sector_positions[pos.sector] = []
                sector_positions[pos.sector].append(pos)
            
            # 从每个行业选择表现最差的持仓
            selected_positions = []
            for sector_pos in sector_positions.values():
                sector_pos.sort(key=lambda x: x.unrealized_pnl)
                selected_positions.extend(sector_pos)
            positions_with_loss = selected_positions
        
        # 计算需要清仓的数量
        total_positions = len(positions_with_loss)
        positions_to_liquidate_count = max(1, int(total_positions * liquidation_ratio))
        positions_to_liquidate_count = min(positions_to_liquidate_count, self.config.max_liquidation_per_batch)
        
        return [pos.symbol for pos in positions_with_loss[:positions_to_liquidate_count]]
    
    def _check_portfolio_recovery(self, portfolio: Portfolio, pnl_ratio: float, 
                                drawdown_from_peak: float) -> Optional[Dict[str, Any]]:
        """检查组合恢复状态"""
        if not self.config.recovery_monitoring_enabled:
            return None
        
        # 检查是否满足恢复条件
        if drawdown_from_peak >= -self.config.portfolio_recovery_threshold:
            # 标记恢复
            if self.portfolio_stop_loss_records:
                latest_record = self.portfolio_stop_loss_records[-1]
                if not latest_record.is_recovered:
                    latest_record.is_recovered = True
                    latest_record.recovery_time = datetime.now()
                    
                    self.portfolio_stop_loss_active = False
                    
                    self.logger.info(f"组合级止损恢复: 当前回撤={drawdown_from_peak:.2%}")
                    
                    return {
                        'trigger_type': 'recovery',
                        'pnl_ratio': pnl_ratio,
                        'drawdown_from_peak': drawdown_from_peak,
                        'action': 'recovery_detected',
                        'recovery_time': latest_record.recovery_time,
                        'auto_rebalance': self.config.auto_rebalance_on_recovery,
                        'reason': f"组合回撤恢复至{drawdown_from_peak:.2%}"
                    }
        
        return None
    
    def execute_portfolio_stop_loss(self, portfolio: Portfolio, 
                                  stop_loss_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """执行组合级止损"""
        executed_orders = []
        positions_to_liquidate = stop_loss_result.get('positions_to_liquidate', [])
        
        for symbol in positions_to_liquidate:
            position = portfolio.get_position(symbol)
            if position and position.quantity > 0:
                try:
                    # 创建清仓订单
                    liquidation_order = {
                        'symbol': symbol,
                        'quantity': position.quantity,
                        'action': 'SELL',
                        'order_type': 'MARKET',
                        'price': position.current_price,
                        'execution_time': datetime.now(),
                        'reason': f"组合级止损清仓 - {stop_loss_result['trigger_type']}",
                        'status': 'EXECUTED'
                    }
                    
                    executed_orders.append(liquidation_order)
                    
                    # 清除该股票的个股止损设置
                    self.reset_stop_loss(symbol)
                    
                    self.logger.info(f"执行组合级止损清仓: {symbol} {position.quantity}股")
                    
                except Exception as e:
                    self.logger.error(f"执行组合级止损失败 {symbol}: {str(e)}")
                    error_order = {
                        'symbol': symbol,
                        'status': 'FAILED',
                        'error': str(e),
                        'reason': '组合级止损执行失败'
                    }
                    executed_orders.append(error_order)
        
        return executed_orders
    
    def get_portfolio_stop_loss_status(self) -> Dict[str, Any]:
        """获取组合级止损状态"""
        return {
            'active': self.portfolio_stop_loss_active,
            'peak_value': self.portfolio_peak_value,
            'total_records': len(self.portfolio_stop_loss_records),
            'last_liquidation_time': self.last_liquidation_time,
            'recovery_monitoring_enabled': self.config.recovery_monitoring_enabled,
            'current_thresholds': {
                'full_stop': -self.config.portfolio_stop_loss,
                'partial_stop': -self.config.portfolio_partial_stop,
                'recovery': -self.config.portfolio_recovery_threshold
            }
        }
    
    def get_portfolio_stop_loss_history(self) -> List[Dict[str, Any]]:
        """获取组合级止损历史"""
        history = []
        for record in self.portfolio_stop_loss_records:
            history.append({
                'trigger_time': record.trigger_time,
                'trigger_type': record.trigger_type,
                'portfolio_pnl_ratio': record.portfolio_pnl_ratio,
                'threshold': record.threshold,
                'action_taken': record.action_taken,
                'positions_liquidated': record.positions_liquidated,
                'positions_count': len(record.positions_liquidated),
                'recovery_time': record.recovery_time,
                'is_recovered': record.is_recovered,
                'duration_minutes': (
                    (record.recovery_time - record.trigger_time).total_seconds() / 60
                    if record.recovery_time else None
                )
            })
        return history
    
    def update_price_history(self, symbol: str, price: float, timestamp: datetime):
        """更新价格历史数据"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append((timestamp, price))
        
        # 保持历史数据在合理范围内
        max_history_length = self.config.volatility_lookback_days * 2
        if len(self.price_history[symbol]) > max_history_length:
            self.price_history[symbol] = self.price_history[symbol][-max_history_length:]
        
        # 清除过期的波动率缓存
        if symbol in self.volatility_cache:
            del self.volatility_cache[symbol]
    
    def get_stop_loss_status(self, symbol: str) -> Dict[str, Any]:
        """获取股票的止损状态"""
        status = {
            'symbol': symbol,
            'has_active_stops': symbol in self.stop_loss_levels,
            'stop_levels': self.stop_loss_levels.get(symbol, {}),
            'trailing_stop': self.trailing_stops.get(symbol, 0.0),
            'volatility': self.volatility_cache.get(symbol, 0.0),
            'active_order': self.active_orders.get(symbol)
        }
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取止损统计信息"""
        if self.stats['total_stops_triggered'] > 0:
            success_rate = self.stats['successful_stops'] / self.stats['total_stops_triggered']
        else:
            success_rate = 0.0
        
        return {
            'total_stops_triggered': self.stats['total_stops_triggered'],
            'successful_stops': self.stats['successful_stops'],
            'false_stops': self.stats['false_stops'],
            'success_rate': success_rate,
            'average_stop_loss': self.stats['average_stop_loss'],
            'active_trailing_stops': len(self.trailing_stops),
            'total_symbols_monitored': len(self.stop_loss_levels)
        }
    
    def reset_stop_loss(self, symbol: str):
        """重置股票的止损设置"""
        if symbol in self.stop_loss_levels:
            del self.stop_loss_levels[symbol]
        if symbol in self.trailing_stops:
            del self.trailing_stops[symbol]
        if symbol in self.trailing_stop_records:
            del self.trailing_stop_records[symbol]
        if symbol in self.active_orders:
            del self.active_orders[symbol]
        
        self.logger.info(f"重置 {symbol} 的止损设置")
    
    def disable_stop_loss(self, symbol: str):
        """禁用股票的止损"""
        if symbol in self.active_orders:
            self.active_orders[symbol].status = StopLossStatus.DISABLED
        
        self.logger.info(f"禁用 {symbol} 的止损")
    
    def enable_stop_loss(self, symbol: str):
        """启用股票的止损"""
        if symbol in self.active_orders:
            self.active_orders[symbol].status = StopLossStatus.ACTIVE
        
        self.logger.info(f"启用 {symbol} 的止损")