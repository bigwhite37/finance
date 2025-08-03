"""风险预算分配器实现"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import logging
from abc import ABC, abstractmethod


class AllocationMethod(Enum):
    """分配方法枚举"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    VOLATILITY_WEIGHTED = "volatility_weighted"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    DRAWDOWN_ADJUSTED = "drawdown_adjusted"


@dataclass
class RiskBudgetConfig:
    """风险预算配置"""
    # 基础风险预算参数
    base_risk_budget: float = 0.10                    # 基础风险预算 (10%)
    max_risk_budget: float = 0.20                     # 最大风险预算 (20%)
    min_risk_budget: float = 0.02                     # 最小风险预算 (2%)
    
    # 回撤调整参数
    drawdown_warning_threshold: float = 0.05          # 回撤警告阈值 (5%)
    drawdown_critical_threshold: float = 0.10         # 回撤临界阈值 (10%)
    risk_scaling_factor: float = 0.5                  # 风险缩放因子
    recovery_speed: float = 0.1                       # 恢复速度
    
    # 分配方法参数
    default_allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY
    volatility_lookback_days: int = 20                # 波动率回看天数
    correlation_lookback_days: int = 60               # 相关性回看天数
    
    # 约束参数
    max_single_asset_budget: float = 0.15             # 单一资产最大风险预算
    min_single_asset_budget: float = 0.005            # 单一资产最小风险预算
    concentration_penalty: float = 2.0                # 集中度惩罚因子


@dataclass
class AssetRiskMetrics:
    """资产风险指标"""
    symbol: str
    volatility: float                                 # 波动率
    beta: float                                       # 贝塔系数
    var_95: float                                     # 95% VaR
    expected_shortfall: float                         # 期望损失
    correlation_with_market: float                    # 与市场相关性
    liquidity_score: float                           # 流动性评分
    sector: str                                      # 行业
    timestamp: datetime


@dataclass
class RiskBudgetAllocation:
    """风险预算分配结果"""
    total_risk_budget: float                         # 总风险预算
    asset_allocations: Dict[str, float]              # 资产分配
    allocation_method: AllocationMethod              # 分配方法
    risk_contributions: Dict[str, float]             # 风险贡献
    expected_portfolio_risk: float                   # 预期组合风险
    diversification_ratio: float                     # 多样化比率
    allocation_timestamp: datetime                   # 分配时间戳
    metadata: Dict[str, Any]                        # 元数据


class RiskBudgetAllocator:
    """风险预算分配器"""
    
    def __init__(self, config: RiskBudgetConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 历史数据存储
        self.drawdown_history: List[float] = []
        self.risk_budget_history: List[float] = []
        self.allocation_history: List[RiskBudgetAllocation] = []
        
        # 当前状态
        self.current_drawdown: float = 0.0
        self.current_risk_budget: float = config.base_risk_budget
        self.last_allocation_time: Optional[datetime] = None
        
        # 资产风险缓存
        self.asset_risk_cache: Dict[str, AssetRiskMetrics] = {}
        self.correlation_matrix_cache: Optional[np.ndarray] = None
        self.cache_timestamp: Optional[datetime] = None
    
    def calculate_dynamic_risk_budget(self, current_drawdown: float, 
                                    market_volatility: float = None,
                                    performance_metrics: Dict[str, float] = None) -> float:
        """
        基于回撤水平动态计算风险预算
        
        Args:
            current_drawdown: 当前回撤水平
            market_volatility: 市场波动率（可选）
            performance_metrics: 历史表现指标（可选）
            
        Returns:
            调整后的风险预算
        """
        self.current_drawdown = abs(current_drawdown)  # 确保为正值
        
        # 基于回撤水平调整风险预算
        if self.current_drawdown <= self.config.drawdown_warning_threshold:
            # 低回撤：保持或略微增加风险预算
            risk_adjustment = 1.0
        elif self.current_drawdown <= self.config.drawdown_critical_threshold:
            # 中等回撤：线性降低风险预算
            excess_drawdown = self.current_drawdown - self.config.drawdown_warning_threshold
            max_excess = self.config.drawdown_critical_threshold - self.config.drawdown_warning_threshold
            reduction_factor = (excess_drawdown / max_excess) * self.config.risk_scaling_factor
            risk_adjustment = 1.0 - reduction_factor
        else:
            # 高回撤：大幅降低风险预算
            risk_adjustment = 1.0 - self.config.risk_scaling_factor
        
        # 市场波动率调整（如果提供）
        if market_volatility is not None:
            # 高波动率时进一步降低风险预算
            volatility_adjustment = max(0.5, 1.0 - market_volatility)
            risk_adjustment *= volatility_adjustment
        
        # 历史表现调整（如果提供）
        if performance_metrics:
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            if sharpe_ratio > 1.0:
                # 表现良好时适度增加风险预算
                performance_adjustment = min(1.2, 1.0 + (sharpe_ratio - 1.0) * 0.1)
            else:
                # 表现不佳时降低风险预算
                performance_adjustment = max(0.8, sharpe_ratio)
            risk_adjustment *= performance_adjustment
        
        # 计算新的风险预算
        new_risk_budget = self.config.base_risk_budget * risk_adjustment
        
        # 应用边界约束
        new_risk_budget = np.clip(
            new_risk_budget,
            self.config.min_risk_budget,
            self.config.max_risk_budget
        )
        
        # 平滑调整（避免剧烈变化）
        if self.current_risk_budget > 0:
            max_change = self.current_risk_budget * 0.2  # 最大20%变化
            change = new_risk_budget - self.current_risk_budget
            if abs(change) > max_change:
                change = np.sign(change) * max_change
            new_risk_budget = self.current_risk_budget + change
        
        self.current_risk_budget = new_risk_budget
        self.risk_budget_history.append(new_risk_budget)
        self.drawdown_history.append(self.current_drawdown)
        
        self.logger.info(f"风险预算调整: 回撤={current_drawdown:.2%}, "
                        f"新风险预算={new_risk_budget:.2%}, "
                        f"调整因子={risk_adjustment:.3f}")
        
        return new_risk_budget
    
    def allocate_risk_budget(self, 
                           asset_universe: List[str],
                           asset_risk_metrics: Dict[str, AssetRiskMetrics],
                           correlation_matrix: np.ndarray = None,
                           allocation_method: AllocationMethod = None) -> RiskBudgetAllocation:
        """
        在不同资产间分配风险预算
        
        Args:
            asset_universe: 资产列表
            asset_risk_metrics: 资产风险指标
            correlation_matrix: 相关性矩阵
            allocation_method: 分配方法
            
        Returns:
            风险预算分配结果
        """
        if not asset_universe:
            raise ValueError("资产列表不能为空")
        
        # 使用默认分配方法
        if allocation_method is None:
            allocation_method = self.config.default_allocation_method
        
        # 更新资产风险缓存
        self.asset_risk_cache.update(asset_risk_metrics)
        
        # 根据分配方法计算权重
        if allocation_method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight_allocation(asset_universe)
        elif allocation_method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity_allocation(asset_universe, asset_risk_metrics)
        elif allocation_method == AllocationMethod.VOLATILITY_WEIGHTED:
            weights = self._volatility_weighted_allocation(asset_universe, asset_risk_metrics)
        elif allocation_method == AllocationMethod.CORRELATION_ADJUSTED:
            weights = self._correlation_adjusted_allocation(
                asset_universe, asset_risk_metrics, correlation_matrix
            )
        elif allocation_method == AllocationMethod.DRAWDOWN_ADJUSTED:
            weights = self._drawdown_adjusted_allocation(asset_universe, asset_risk_metrics)
        else:
            raise ValueError(f"不支持的分配方法: {allocation_method}")
        
        # 应用约束
        weights = self._apply_allocation_constraints(weights)
        
        # 计算风险预算分配
        asset_allocations = {
            asset: weight * self.current_risk_budget 
            for asset, weight in weights.items()
        }
        
        # 计算风险贡献
        risk_contributions = self._calculate_risk_contributions(
            weights, asset_risk_metrics, correlation_matrix
        )
        
        # 计算预期组合风险
        expected_portfolio_risk = self._calculate_expected_portfolio_risk(
            weights, asset_risk_metrics, correlation_matrix
        )
        
        # 计算多样化比率
        diversification_ratio = self._calculate_diversification_ratio(
            weights, asset_risk_metrics, correlation_matrix
        )
        
        # 创建分配结果
        allocation = RiskBudgetAllocation(
            total_risk_budget=self.current_risk_budget,
            asset_allocations=asset_allocations,
            allocation_method=allocation_method,
            risk_contributions=risk_contributions,
            expected_portfolio_risk=expected_portfolio_risk,
            diversification_ratio=diversification_ratio,
            allocation_timestamp=datetime.now(),
            metadata={
                'current_drawdown': self.current_drawdown,
                'asset_count': len(asset_universe),
                'max_weight': max(weights.values()),
                'min_weight': min(weights.values())
            }
        )
        
        self.allocation_history.append(allocation)
        self.last_allocation_time = allocation.allocation_timestamp
        
        self.logger.info(f"风险预算分配完成: 方法={allocation_method.value}, "
                        f"资产数量={len(asset_universe)}, "
                        f"预期风险={expected_portfolio_risk:.2%}, "
                        f"多样化比率={diversification_ratio:.3f}")
        
        return allocation
    
    def _equal_weight_allocation(self, asset_universe: List[str]) -> Dict[str, float]:
        """等权重分配"""
        weight = 1.0 / len(asset_universe)
        return {asset: weight for asset in asset_universe}
    
    def _risk_parity_allocation(self, 
                              asset_universe: List[str],
                              asset_risk_metrics: Dict[str, AssetRiskMetrics]) -> Dict[str, float]:
        """风险平价分配"""
        # 计算逆波动率权重
        inv_volatilities = {}
        for asset in asset_universe:
            if asset in asset_risk_metrics:
                vol = asset_risk_metrics[asset].volatility
                inv_volatilities[asset] = 1.0 / max(vol, 0.01)  # 避免除零
            else:
                inv_volatilities[asset] = 1.0
        
        # 归一化权重
        total_inv_vol = sum(inv_volatilities.values())
        weights = {
            asset: inv_vol / total_inv_vol 
            for asset, inv_vol in inv_volatilities.items()
        }
        
        return weights
    
    def _volatility_weighted_allocation(self, 
                                      asset_universe: List[str],
                                      asset_risk_metrics: Dict[str, AssetRiskMetrics]) -> Dict[str, float]:
        """波动率加权分配（低波动率资产获得更高权重）"""
        return self._risk_parity_allocation(asset_universe, asset_risk_metrics)
    
    def _correlation_adjusted_allocation(self, 
                                       asset_universe: List[str],
                                       asset_risk_metrics: Dict[str, AssetRiskMetrics],
                                       correlation_matrix: np.ndarray = None) -> Dict[str, float]:
        """相关性调整分配"""
        if correlation_matrix is None:
            # 如果没有相关性矩阵，回退到风险平价
            return self._risk_parity_allocation(asset_universe, asset_risk_metrics)
        
        # 基于风险平价的基础权重
        base_weights = self._risk_parity_allocation(asset_universe, asset_risk_metrics)
        
        # 相关性调整
        adjusted_weights = {}
        for i, asset in enumerate(asset_universe):
            base_weight = base_weights[asset]
            
            # 计算与其他资产的平均相关性
            if i < len(correlation_matrix):
                avg_correlation = np.mean(np.abs(correlation_matrix[i, :]))
                # 低相关性资产获得更高权重
                correlation_adjustment = 1.0 / (1.0 + avg_correlation)
                adjusted_weights[asset] = base_weight * correlation_adjustment
            else:
                adjusted_weights[asset] = base_weight
        
        # 重新归一化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                asset: weight / total_weight 
                for asset, weight in adjusted_weights.items()
            }
        
        return adjusted_weights
    
    def _drawdown_adjusted_allocation(self, 
                                    asset_universe: List[str],
                                    asset_risk_metrics: Dict[str, AssetRiskMetrics]) -> Dict[str, float]:
        """回撤调整分配"""
        # 基于风险平价的基础权重
        base_weights = self._risk_parity_allocation(asset_universe, asset_risk_metrics)
        
        # 根据当前回撤水平调整权重分布
        if self.current_drawdown > self.config.drawdown_warning_threshold:
            # 高回撤时更加保守，增加低风险资产权重
            adjusted_weights = {}
            for asset in asset_universe:
                base_weight = base_weights[asset]
                if asset in asset_risk_metrics:
                    # 基于VaR调整权重
                    var_95 = asset_risk_metrics[asset].var_95
                    risk_adjustment = 1.0 / (1.0 + abs(var_95))
                    adjusted_weights[asset] = base_weight * risk_adjustment
                else:
                    adjusted_weights[asset] = base_weight
            
            # 重新归一化
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {
                    asset: weight / total_weight 
                    for asset, weight in adjusted_weights.items()
                }
            return adjusted_weights
        else:
            return base_weights
    
    def _apply_allocation_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用分配约束"""
        constrained_weights = weights.copy()
        
        # 应用单一资产权重限制
        for asset, weight in constrained_weights.items():
            max_weight = self.config.max_single_asset_budget / self.current_risk_budget
            min_weight = self.config.min_single_asset_budget / self.current_risk_budget
            constrained_weights[asset] = np.clip(weight, min_weight, max_weight)
        
        # 重新归一化以确保权重和为1
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                asset: weight / total_weight 
                for asset, weight in constrained_weights.items()
            }
        
        return constrained_weights
    
    def _calculate_risk_contributions(self, 
                                    weights: Dict[str, float],
                                    asset_risk_metrics: Dict[str, AssetRiskMetrics],
                                    correlation_matrix: np.ndarray = None) -> Dict[str, float]:
        """计算风险贡献"""
        risk_contributions = {}
        
        for asset, weight in weights.items():
            if asset in asset_risk_metrics:
                # 简化的风险贡献计算：权重 × 波动率
                volatility = asset_risk_metrics[asset].volatility
                risk_contributions[asset] = weight * volatility
            else:
                risk_contributions[asset] = weight * 0.1  # 默认波动率
        
        return risk_contributions
    
    def _calculate_expected_portfolio_risk(self, 
                                         weights: Dict[str, float],
                                         asset_risk_metrics: Dict[str, AssetRiskMetrics],
                                         correlation_matrix: np.ndarray = None) -> float:
        """计算预期组合风险"""
        if not weights:
            return 0.0
        
        # 简化计算：加权平均波动率
        weighted_volatility = 0.0
        for asset, weight in weights.items():
            if asset in asset_risk_metrics:
                volatility = asset_risk_metrics[asset].volatility
                weighted_volatility += weight * volatility
            else:
                weighted_volatility += weight * 0.1  # 默认波动率
        
        # 如果有相关性矩阵，可以进行更精确的计算
        if correlation_matrix is not None and len(weights) == len(correlation_matrix):
            # 这里可以实现更复杂的组合风险计算
            pass
        
        return weighted_volatility
    
    def _calculate_diversification_ratio(self, 
                                       weights: Dict[str, float],
                                       asset_risk_metrics: Dict[str, AssetRiskMetrics],
                                       correlation_matrix: np.ndarray = None) -> float:
        """计算多样化比率"""
        if not weights:
            return 0.0
        
        # 简化的多样化比率：1 - 最大权重
        max_weight = max(weights.values())
        diversification_ratio = 1.0 - max_weight
        
        return diversification_ratio
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """获取分配摘要"""
        if not self.allocation_history:
            return {
                'total_allocations': 0,
                'current_risk_budget': self.current_risk_budget,
                'current_drawdown': self.current_drawdown
            }
        
        latest_allocation = self.allocation_history[-1]
        
        return {
            'total_allocations': len(self.allocation_history),
            'current_risk_budget': self.current_risk_budget,
            'current_drawdown': self.current_drawdown,
            'latest_allocation': {
                'method': latest_allocation.allocation_method.value,
                'asset_count': len(latest_allocation.asset_allocations),
                'expected_risk': latest_allocation.expected_portfolio_risk,
                'diversification_ratio': latest_allocation.diversification_ratio,
                'timestamp': latest_allocation.allocation_timestamp
            },
            'risk_budget_trend': {
                'min': min(self.risk_budget_history) if self.risk_budget_history else 0,
                'max': max(self.risk_budget_history) if self.risk_budget_history else 0,
                'current': self.current_risk_budget
            }
        }
    
    def reset_allocation_history(self):
        """重置分配历史"""
        self.allocation_history.clear()
        self.risk_budget_history.clear()
        self.drawdown_history.clear()
        self.logger.info("风险预算分配历史已重置")