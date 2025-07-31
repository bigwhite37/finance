"""
交易成本计算模块
实现手续费、印花税、过户费和市场冲击成本的计算，支持A股特有的交易规则
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Union
import numpy as np

from .almgren_chriss_model import AlmgrenChrissModel


@dataclass
class CostParameters:
    """成本参数配置"""
    commission_rate: float          # 手续费率
    stamp_tax_rate: float          # 印花税率
    min_commission: float          # 最小手续费
    transfer_fee_rate: float       # 过户费率
    market_impact_model: Optional[AlmgrenChrissModel] = None  # 市场冲击模型
    
    def __post_init__(self):
        """参数验证"""
        self._validate()
    
    def _validate(self):
        """验证参数有效性"""
        if self.commission_rate < 0:
            raise ValueError("手续费率不能为负数")
        
        if self.commission_rate > 0.1:
            raise ValueError("手续费率不能超过10%")
        
        if self.stamp_tax_rate < 0:
            raise ValueError("印花税率不能为负数")
        
        if self.stamp_tax_rate > 0.1:
            raise ValueError("印花税率不能超过10%")
        
        if self.min_commission < 0:
            raise ValueError("最小手续费不能为负数")
        
        if self.transfer_fee_rate < 0:
            raise ValueError("过户费率不能为负数")


@dataclass
class TradeInfo:
    """交易信息"""
    symbol: str                    # 股票代码
    side: str                      # 交易方向：'buy' 或 'sell'
    quantity: int                  # 交易数量
    price: float                   # 交易价格
    timestamp: datetime            # 交易时间
    market_volume: Optional[int] = None      # 市场成交量（用于市场冲击计算）
    volatility: Optional[float] = None       # 股票波动率（用于市场冲击计算）
    
    def __post_init__(self):
        """数据验证"""
        self._validate()
    
    def _validate(self):
        """验证数据有效性"""
        if self.side not in ['buy', 'sell']:
            raise ValueError("交易方向必须是'buy'或'sell'")
        
        if self.quantity < 0:
            raise ValueError("交易数量不能为负数")
        
        if self.price < 0:
            raise ValueError("价格不能为负数")
        
        if self.market_volume is not None and self.market_volume < 0:
            raise ValueError("市场成交量不能为负数")
        
        if self.volatility is not None and self.volatility < 0:
            raise ValueError("波动率不能为负数")
    
    def get_trade_value(self) -> float:
        """获取交易价值"""
        return self.quantity * self.price
    
    def is_buy(self) -> bool:
        """判断是否为买入交易"""
        return self.side == 'buy'
    
    def is_sell(self) -> bool:
        """判断是否为卖出交易"""
        return self.side == 'sell'


@dataclass
class CostBreakdown:
    """成本分解结构"""
    commission: float      # 手续费
    stamp_tax: float       # 印花税
    transfer_fee: float    # 过户费
    market_impact: float   # 市场冲击成本
    total_cost: float      # 总成本
    
    def __post_init__(self):
        """数据验证"""
        self._validate()
    
    def _validate(self):
        """验证数据有效性"""
        if self.commission < 0:
            raise ValueError("手续费不能为负数")
        
        if self.stamp_tax < 0:
            raise ValueError("印花税不能为负数")
        
        if self.transfer_fee < 0:
            raise ValueError("过户费不能为负数")
        
        if self.market_impact < 0:
            raise ValueError("市场冲击成本不能为负数")
        
        if self.total_cost < 0:
            raise ValueError("总成本不能为负数")
        
        # 验证总成本等于各项成本之和
        expected_total = self.commission + self.stamp_tax + self.transfer_fee + self.market_impact
        if abs(self.total_cost - expected_total) > 1e-8:
            raise ValueError("总成本应等于各项成本之和")
    
    def get_cost_ratio(self, trade_value: float) -> float:
        """获取成本比率"""
        if trade_value == 0:
            return 0.0
        return self.total_cost / trade_value
    
    def get_cost_basis_points(self, trade_value: float) -> float:
        """获取成本（基点）"""
        return self.get_cost_ratio(trade_value) * 10000
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'commission': self.commission,
            'stamp_tax': self.stamp_tax,
            'transfer_fee': self.transfer_fee,
            'market_impact': self.market_impact,
            'total_cost': self.total_cost
        }


class TransactionCostModel:
    """
    交易成本计算模型
    
    该模型计算A股交易的各项成本，包括：
    1. 手续费：双边收取，有最小手续费限制
    2. 印花税：仅卖出时收取
    3. 过户费：双边收取
    4. 市场冲击：可选，使用Almgren-Chriss模型计算
    
    A股交易成本特点：
    - 手续费：通常为万分之三，最低5元
    - 印花税：千分之一，仅卖出收取
    - 过户费：万分之0.2，双边收取
    """
    
    def __init__(self, parameters: CostParameters):
        """
        初始化交易成本模型
        
        Args:
            parameters: 成本参数配置
        """
        self.parameters = parameters
    
    def calculate_cost(self, trade: TradeInfo) -> CostBreakdown:
        """
        计算单笔交易的成本
        
        Args:
            trade: 交易信息
            
        Returns:
            CostBreakdown: 成本分解结果
        """
        trade_value = trade.get_trade_value()
        
        # 计算各项成本
        commission = self._calculate_commission(trade_value)
        stamp_tax = self._calculate_stamp_tax(trade)
        transfer_fee = self._calculate_transfer_fee(trade_value)
        market_impact = self._calculate_market_impact(trade)
        
        # 计算总成本
        total_cost = commission + stamp_tax + transfer_fee + market_impact
        
        return CostBreakdown(
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            market_impact=market_impact,
            total_cost=total_cost
        )
    
    def calculate_batch_costs(self, trades: List[TradeInfo]) -> List[CostBreakdown]:
        """
        批量计算交易成本
        
        Args:
            trades: 交易信息列表
            
        Returns:
            List[CostBreakdown]: 成本分解结果列表
        """
        return [self.calculate_cost(trade) for trade in trades]
    
    def _calculate_commission(self, trade_value: float) -> float:
        """
        计算手续费
        
        手续费 = max(交易价值 * 手续费率, 最小手续费)
        
        Args:
            trade_value: 交易价值
            
        Returns:
            float: 手续费
        """
        commission = trade_value * self.parameters.commission_rate
        return max(commission, self.parameters.min_commission)
    
    def _calculate_stamp_tax(self, trade: TradeInfo) -> float:
        """
        计算印花税
        
        印花税仅在卖出时收取
        印花税 = 交易价值 * 印花税率 (仅卖出)
        
        Args:
            trade: 交易信息
            
        Returns:
            float: 印花税
        """
        if trade.is_sell():
            return trade.get_trade_value() * self.parameters.stamp_tax_rate
        else:
            return 0.0
    
    def _calculate_transfer_fee(self, trade_value: float) -> float:
        """
        计算过户费
        
        过户费 = 交易价值 * 过户费率
        
        Args:
            trade_value: 交易价值
            
        Returns:
            float: 过户费
        """
        return trade_value * self.parameters.transfer_fee_rate
    
    def _calculate_market_impact(self, trade: TradeInfo) -> float:
        """
        计算市场冲击成本
        
        如果配置了市场冲击模型，则使用模型计算；否则返回0
        
        Args:
            trade: 交易信息
            
        Returns:
            float: 市场冲击成本（以价格比例表示）
        """
        if self.parameters.market_impact_model is None:
            return 0.0
        
        # 使用Almgren-Chriss模型计算市场冲击
        impact_result = self.parameters.market_impact_model.calculate_impact(
            trade_volume=trade.quantity,
            market_volume=trade.market_volume,
            volatility=trade.volatility
        )
        
        # 将冲击转换为绝对成本
        trade_value = trade.get_trade_value()
        return impact_result.total_impact * trade_value
    
    def estimate_round_trip_cost(self, 
                                symbol: str,
                                quantity: int,
                                price: float,
                                timestamp: datetime,
                                market_volume: Optional[int] = None,
                                volatility: Optional[float] = None) -> dict:
        """
        估算往返交易成本（买入+卖出）
        
        Args:
            symbol: 股票代码
            quantity: 交易数量
            price: 交易价格
            timestamp: 交易时间
            market_volume: 市场成交量
            volatility: 波动率
            
        Returns:
            dict: 包含买入、卖出和总成本的字典
        """
        # 创建买入交易
        buy_trade = TradeInfo(
            symbol=symbol,
            side='buy',
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            market_volume=market_volume,
            volatility=volatility
        )
        
        # 创建卖出交易
        sell_trade = TradeInfo(
            symbol=symbol,
            side='sell',
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            market_volume=market_volume,
            volatility=volatility
        )
        
        # 计算成本
        buy_cost = self.calculate_cost(buy_trade)
        sell_cost = self.calculate_cost(sell_trade)
        
        # 计算总成本
        total_cost = CostBreakdown(
            commission=buy_cost.commission + sell_cost.commission,
            stamp_tax=buy_cost.stamp_tax + sell_cost.stamp_tax,
            transfer_fee=buy_cost.transfer_fee + sell_cost.transfer_fee,
            market_impact=buy_cost.market_impact + sell_cost.market_impact,
            total_cost=buy_cost.total_cost + sell_cost.total_cost
        )
        
        return {
            'buy_cost': buy_cost,
            'sell_cost': sell_cost,
            'total_cost': total_cost,
            'round_trip_basis_points': total_cost.get_cost_basis_points(buy_trade.get_trade_value())
        }
    
    def get_cost_estimate(self, 
                         trade_value: float,
                         side: str = 'buy') -> dict:
        """
        快速成本估算（不需要完整的交易信息）
        
        Args:
            trade_value: 交易价值
            side: 交易方向
            
        Returns:
            dict: 成本估算结果
        """
        # 计算各项成本
        commission = max(trade_value * self.parameters.commission_rate, 
                        self.parameters.min_commission)
        
        stamp_tax = (trade_value * self.parameters.stamp_tax_rate 
                    if side == 'sell' else 0.0)
        
        transfer_fee = trade_value * self.parameters.transfer_fee_rate
        
        total_cost = commission + stamp_tax + transfer_fee
        
        return {
            'commission': commission,
            'stamp_tax': stamp_tax,
            'transfer_fee': transfer_fee,
            'market_impact': 0.0,  # 快速估算不包含市场冲击
            'total_cost': total_cost,
            'cost_ratio': total_cost / trade_value if trade_value > 0 else 0,
            'cost_basis_points': (total_cost / trade_value * 10000) if trade_value > 0 else 0
        }
    
    def update_parameters(self, new_parameters: CostParameters) -> None:
        """
        更新成本参数
        
        Args:
            new_parameters: 新的成本参数
        """
        self.parameters = new_parameters
    
    def get_parameters(self) -> CostParameters:
        """
        获取当前成本参数
        
        Returns:
            CostParameters: 当前参数
        """
        return self.parameters
    
    def validate_trade(self, trade: TradeInfo) -> List[str]:
        """
        验证交易是否符合A股交易规则
        
        Args:
            trade: 交易信息
            
        Returns:
            List[str]: 验证错误信息列表，空列表表示验证通过
        """
        errors = []
        
        # 检查最小交易单位（A股通常为100股的倍数）
        if trade.quantity % 100 != 0 and trade.quantity >= 100:
            errors.append(f"交易数量{trade.quantity}不是100的倍数")
        
        # 检查价格精度（A股价格通常精确到分）
        if round(trade.price, 2) != trade.price:
            errors.append(f"价格{trade.price}精度超过2位小数")
        
        # 检查涨跌停限制（简化检查，实际需要前一日收盘价）
        if trade.price <= 0:
            errors.append(f"价格{trade.price}不能为零或负数")
        
        # 检查交易时间（简化检查）
        hour = trade.timestamp.hour
        minute = trade.timestamp.minute
        
        # A股交易时间：9:30-11:30, 13:00-15:00
        morning_session = (9, 30) <= (hour, minute) <= (11, 30)
        afternoon_session = (13, 0) <= (hour, minute) <= (15, 0)
        
        if not (morning_session or afternoon_session):
            errors.append(f"交易时间{trade.timestamp}不在交易时段内")
        
        return errors
    
    def calculate_optimal_lot_size(self, 
                                  target_value: float,
                                  price: float,
                                  max_cost_ratio: float = 0.01) -> int:
        """
        计算最优交易批次大小
        
        在给定最大成本比率约束下，计算最优的交易数量
        
        Args:
            target_value: 目标交易价值
            price: 股票价格
            max_cost_ratio: 最大成本比率
            
        Returns:
            int: 建议的交易数量
        """
        target_quantity = int(target_value / price)
        
        # 从目标数量开始，逐步调整直到满足成本约束
        for quantity in range(target_quantity, 0, -100):  # 以100股为单位递减
            trade_value = quantity * price
            cost_estimate = self.get_cost_estimate(trade_value)
            
            if cost_estimate['cost_ratio'] <= max_cost_ratio:
                return quantity
        
        # 如果无法满足成本约束，返回最小交易单位
        return 100