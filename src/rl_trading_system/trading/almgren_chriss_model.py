"""
Almgren-Chriss市场冲击模型
实现永久冲击（线性）和临时冲击（平方根）模型，用于估算交易成本
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import math


@dataclass
class MarketImpactParameters:
    """市场冲击参数"""
    permanent_impact_coeff: float  # 永久冲击系数
    temporary_impact_coeff: float  # 临时冲击系数
    volatility: float              # 股票波动率
    daily_volume: int              # 日均成交量
    participation_rate: float      # 市场参与度
    
    def __post_init__(self):
        """参数验证"""
        self._validate()
    
    def _validate(self):
        """验证参数有效性"""
        if self.permanent_impact_coeff < 0:
            raise ValueError("永久冲击系数不能为负数")
        
        if self.temporary_impact_coeff < 0:
            raise ValueError("临时冲击系数不能为负数")
        
        if self.volatility < 0:
            raise ValueError("波动率不能为负数")
        
        if self.daily_volume < 0:
            raise ValueError("日均成交量不能为负数")
        
        if not (0 <= self.participation_rate <= 1):
            raise ValueError("市场参与度必须在0到1之间")


@dataclass
class ImpactResult:
    """冲击结果"""
    permanent_impact: float  # 永久冲击
    temporary_impact: float  # 临时冲击
    total_impact: float      # 总冲击
    trade_volume: int        # 交易量
    market_volume: int       # 市场成交量
    
    def __post_init__(self):
        """结果验证"""
        self._validate()
    
    def _validate(self):
        """验证结果有效性"""
        if self.permanent_impact < 0:
            raise ValueError("永久冲击不能为负数")
        
        if self.temporary_impact < 0:
            raise ValueError("临时冲击不能为负数")
        
        if self.total_impact < 0:
            raise ValueError("总冲击不能为负数")
        
        # 验证总冲击等于永久冲击和临时冲击之和
        expected_total = self.permanent_impact + self.temporary_impact
        if abs(self.total_impact - expected_total) > 1e-8:
            raise ValueError("总冲击应等于永久冲击和临时冲击之和")
        
        if self.trade_volume < 0:
            raise ValueError("交易量不能为负数")
        
        if self.market_volume < 0:
            raise ValueError("市场成交量不能为负数")
    
    def get_participation_rate(self) -> float:
        """获取实际参与度"""
        if self.market_volume == 0:
            return 0.0
        return self.trade_volume / self.market_volume
    
    def get_cost_basis_points(self) -> float:
        """获取成本（基点）"""
        return self.total_impact * 10000


class AlmgrenChrissModel:
    """
    Almgren-Chriss市场冲击模型
    
    该模型将市场冲击分解为两个组成部分：
    1. 永久冲击（线性）：反映信息泄露和价格发现的影响
    2. 临时冲击（平方根）：反映流动性消耗的影响
    
    永久冲击 = α * (V / V_daily)
    临时冲击 = β * σ * sqrt(V / V_daily)
    
    其中：
    - α: 永久冲击系数
    - β: 临时冲击系数  
    - σ: 股票波动率
    - V: 交易量
    - V_daily: 日均成交量
    """
    
    def __init__(self, parameters: MarketImpactParameters):
        """
        初始化Almgren-Chriss模型
        
        Args:
            parameters: 市场冲击参数
        """
        self.parameters = parameters
    
    def calculate_impact(self, 
                        trade_volume: int,
                        market_volume: Optional[int] = None,
                        volatility: Optional[float] = None) -> ImpactResult:
        """
        计算市场冲击
        
        Args:
            trade_volume: 交易量
            market_volume: 市场成交量（可选，默认使用参数中的日均成交量）
            volatility: 波动率（可选，默认使用参数中的波动率）
            
        Returns:
            ImpactResult: 冲击结果
        """
        # 使用默认值或提供的值
        market_vol = market_volume if market_volume is not None else self.parameters.daily_volume
        vol = volatility if volatility is not None else self.parameters.volatility
        
        # 计算永久冲击（线性）
        permanent_impact = self._calculate_permanent_impact(trade_volume, market_vol)
        
        # 计算临时冲击（平方根）
        temporary_impact = self._calculate_temporary_impact(trade_volume, market_vol, vol)
        
        # 计算总冲击
        total_impact = permanent_impact + temporary_impact
        
        return ImpactResult(
            permanent_impact=permanent_impact,
            temporary_impact=temporary_impact,
            total_impact=total_impact,
            trade_volume=trade_volume,
            market_volume=market_vol
        )
    
    def _calculate_permanent_impact(self, trade_volume: int, market_volume: int) -> float:
        """
        计算永久冲击（线性模型）
        
        永久冲击反映了交易对股价的持久影响，通常由信息泄露引起。
        模型：permanent_impact = α * (V / V_daily)
        
        Args:
            trade_volume: 交易量
            market_volume: 市场成交量
            
        Returns:
            float: 永久冲击
        """
        if market_volume == 0:
            return 0.0
        
        participation_rate = trade_volume / market_volume
        return self.parameters.permanent_impact_coeff * participation_rate
    
    def _calculate_temporary_impact(self, trade_volume: int, market_volume: int, volatility: float) -> float:
        """
        计算临时冲击（平方根模型）
        
        临时冲击反映了交易对流动性的消耗，随着时间推移会恢复。
        模型：temporary_impact = β * σ * sqrt(V / V_daily)
        
        Args:
            trade_volume: 交易量
            market_volume: 市场成交量
            volatility: 波动率
            
        Returns:
            float: 临时冲击
        """
        if market_volume == 0:
            return 0.0
        
        participation_rate = trade_volume / market_volume
        return self.parameters.temporary_impact_coeff * volatility * math.sqrt(participation_rate)
    
    def update_parameters(self, new_parameters: MarketImpactParameters) -> None:
        """
        更新模型参数
        
        Args:
            new_parameters: 新的市场冲击参数
        """
        self.parameters = new_parameters
    
    def get_parameters(self) -> MarketImpactParameters:
        """
        获取当前模型参数
        
        Returns:
            MarketImpactParameters: 当前参数
        """
        return self.parameters
    
    def estimate_optimal_trade_size(self, 
                                   target_volume: int,
                                   time_horizon: int,
                                   risk_aversion: float = 1.0) -> int:
        """
        估算最优交易规模
        
        基于Almgren-Chriss模型的最优执行策略，平衡市场冲击和时间风险。
        
        Args:
            target_volume: 目标交易总量
            time_horizon: 交易时间窗口（分钟）
            risk_aversion: 风险厌恶系数
            
        Returns:
            int: 建议的单次交易规模
        """
        if time_horizon <= 0:
            return target_volume
        
        # 简化的最优交易规模公式
        # 在实际应用中，这需要更复杂的优化算法
        alpha = self.parameters.permanent_impact_coeff
        beta = self.parameters.temporary_impact_coeff
        sigma = self.parameters.volatility
        
        # 最优交易速度（简化版本）
        optimal_rate = math.sqrt(alpha / (beta * sigma * risk_aversion))
        optimal_size = int(target_volume * optimal_rate / time_horizon)
        
        # 确保交易规模在合理范围内
        min_size = max(1, target_volume // (time_horizon * 10))
        max_size = target_volume // max(1, time_horizon // 10)
        
        return max(min_size, min(optimal_size, max_size))
    
    def calculate_execution_cost(self, 
                               trade_schedule: list,
                               market_volumes: Optional[list] = None) -> dict:
        """
        计算执行计划的总成本
        
        Args:
            trade_schedule: 交易计划列表，每个元素为交易量
            market_volumes: 对应的市场成交量列表（可选）
            
        Returns:
            dict: 包含总成本、永久冲击、临时冲击等信息的字典
        """
        total_permanent = 0.0
        total_temporary = 0.0
        total_volume = 0
        
        results = []
        
        for i, trade_vol in enumerate(trade_schedule):
            market_vol = (market_volumes[i] if market_volumes and i < len(market_volumes) 
                         else self.parameters.daily_volume)
            
            result = self.calculate_impact(trade_vol, market_vol)
            results.append(result)
            
            total_permanent += result.permanent_impact
            total_temporary += result.temporary_impact
            total_volume += trade_vol
        
        return {
            'total_permanent_impact': total_permanent,
            'total_temporary_impact': total_temporary,
            'total_impact': total_permanent + total_temporary,
            'total_volume': total_volume,
            'average_impact': (total_permanent + total_temporary) / len(trade_schedule) if trade_schedule else 0,
            'detailed_results': results
        }