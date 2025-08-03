"""自适应风险预算系统实现"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from collections import deque
import warnings


class MarketCondition(Enum):
    """市场状态枚举"""
    BULL = "bull"                    # 牛市
    BEAR = "bear"                    # 熊市
    SIDEWAYS = "sideways"            # 震荡市
    HIGH_VOLATILITY = "high_vol"     # 高波动
    LOW_VOLATILITY = "low_vol"       # 低波动
    CRISIS = "crisis"                # 危机


class PerformanceRegime(Enum):
    """表现状态枚举"""
    EXCELLENT = "excellent"          # 优秀
    GOOD = "good"                   # 良好
    AVERAGE = "average"             # 一般
    POOR = "poor"                   # 较差
    TERRIBLE = "terrible"           # 糟糕


@dataclass
class PerformanceMetrics:
    """历史表现指标"""
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 1.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    total_return: float = 0.0
    downside_deviation: float = 0.0
    sortino_ratio: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketMetrics:
    """市场指标"""
    market_volatility: float = 0.0
    market_trend: float = 0.0        # 正值表示上涨趋势
    correlation_with_market: float = 0.0
    liquidity_score: float = 1.0
    uncertainty_index: float = 0.0   # 不确定性指数
    regime_stability: float = 1.0    # 状态稳定性
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptiveRiskBudgetConfig:
    """自适应风险预算配置"""
    # 基础参数
    base_risk_budget: float = 0.10                    # 基础风险预算
    min_risk_budget: float = 0.02                     # 最小风险预算
    max_risk_budget: float = 0.25                     # 最大风险预算
    
    # 表现调整参数
    performance_lookback_days: int = 60               # 表现回看天数
    sharpe_threshold_excellent: float = 2.0           # 优秀表现阈值
    sharpe_threshold_good: float = 1.0                # 良好表现阈值
    sharpe_threshold_poor: float = 0.0                # 较差表现阈值
    performance_adjustment_factor: float = 0.3        # 表现调整因子
    
    # 市场条件调整参数
    market_lookback_days: int = 30                    # 市场回看天数
    volatility_threshold_high: float = 0.25           # 高波动阈值
    volatility_threshold_low: float = 0.10            # 低波动阈值
    uncertainty_threshold: float = 0.7                # 不确定性阈值
    market_adjustment_factor: float = 0.4             # 市场调整因子
    
    # 连续亏损调整参数
    consecutive_loss_threshold: int = 3               # 连续亏损阈值
    loss_penalty_factor: float = 0.2                 # 亏损惩罚因子
    max_loss_penalty: float = 0.5                    # 最大亏损惩罚
    
    # 平滑机制参数
    smoothing_factor: float = 0.1                    # 平滑因子 (EMA)
    max_daily_change: float = 0.05                   # 最大日变化率
    adjustment_delay_days: int = 1                   # 调整延迟天数
    
    # 异常检测参数
    anomaly_detection_window: int = 20               # 异常检测窗口
    anomaly_threshold_std: float = 2.0               # 异常阈值（标准差倍数）
    anomaly_adjustment_factor: float = 0.5           # 异常调整因子
    
    # 恢复机制参数
    recovery_speed_fast: float = 0.05                # 快速恢复速度
    recovery_speed_slow: float = 0.02                # 慢速恢复速度
    recovery_threshold: float = 0.5                  # 恢复阈值


@dataclass
class RiskBudgetAdjustment:
    """风险预算调整记录"""
    timestamp: datetime
    old_budget: float
    new_budget: float
    adjustment_reason: str
    performance_regime: PerformanceRegime
    market_condition: MarketCondition
    adjustment_factors: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRiskBudget:
    """自适应风险预算系统"""
    
    def __init__(self, config: AdaptiveRiskBudgetConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 当前状态
        self.current_risk_budget: float = config.base_risk_budget
        self.smoothed_risk_budget: float = config.base_risk_budget
        self.last_update_time: Optional[datetime] = None
        
        # 历史数据
        self.performance_history: deque = deque(maxlen=config.performance_lookback_days)
        self.market_history: deque = deque(maxlen=config.market_lookback_days)
        self.risk_budget_history: deque = deque(maxlen=252)  # 一年历史
        self.adjustment_history: List[RiskBudgetAdjustment] = []
        
        # 状态跟踪
        self.current_performance_regime: PerformanceRegime = PerformanceRegime.AVERAGE
        self.current_market_condition: MarketCondition = MarketCondition.SIDEWAYS
        self.consecutive_adjustments: int = 0
        self.last_anomaly_time: Optional[datetime] = None
        
        # 缓存
        self._performance_cache: Optional[PerformanceMetrics] = None
        self._market_cache: Optional[MarketMetrics] = None
        self._cache_timestamp: Optional[datetime] = None
    
    def update_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """更新历史表现指标"""
        self.performance_history.append(metrics)
        self._performance_cache = metrics
        self._cache_timestamp = datetime.now()
        
        # 更新表现状态
        self.current_performance_regime = self._classify_performance_regime(metrics)
        
        self.logger.debug(f"更新表现指标: Sharpe={metrics.sharpe_ratio:.3f}, "
                         f"状态={self.current_performance_regime.value}")
    
    def update_market_metrics(self, metrics: MarketMetrics) -> None:
        """更新市场指标"""
        self.market_history.append(metrics)
        self._market_cache = metrics
        
        # 更新市场状态
        self.current_market_condition = self._classify_market_condition(metrics)
        
        self.logger.debug(f"更新市场指标: 波动率={metrics.market_volatility:.3f}, "
                         f"状态={self.current_market_condition.value}")
    
    def calculate_adaptive_risk_budget(self, 
                                     force_update: bool = False) -> float:
        """
        计算自适应风险预算
        
        Args:
            force_update: 是否强制更新
            
        Returns:
            调整后的风险预算
        """
        current_time = datetime.now()
        
        # 检查是否需要更新
        if not force_update and self.last_update_time:
            time_diff = current_time - self.last_update_time
            if time_diff.days < self.config.adjustment_delay_days:
                return self.current_risk_budget
        
        # 获取当前指标
        performance_metrics = self._get_current_performance_metrics()
        market_metrics = self._get_current_market_metrics()
        
        # 严格遵守规则6：无法获取数据时立即抛出RuntimeError
        if not performance_metrics:
            raise RuntimeError("无法获取必要的表现指标数据：需要历史表现数据来计算自适应风险预算")
        
        if not market_metrics:
            raise RuntimeError("无法获取必要的市场指标数据：需要市场状态数据来计算自适应风险预算")
        
        # 计算各种调整因子
        adjustment_factors = self._calculate_adjustment_factors(
            performance_metrics, market_metrics
        )
        
        # 计算新的风险预算
        new_risk_budget = self._apply_adjustments(adjustment_factors)
        
        # 异常检测
        if self._detect_anomaly(new_risk_budget):
            self.logger.warning(f"检测到异常风险预算调整: {new_risk_budget:.3f}")
            new_risk_budget = self._handle_anomaly(new_risk_budget)
        
        # 应用平滑机制
        smoothed_budget = self._apply_smoothing(new_risk_budget)
        
        # 记录调整
        if abs(smoothed_budget - self.current_risk_budget) > 1e-6:
            adjustment = RiskBudgetAdjustment(
                timestamp=current_time,
                old_budget=self.current_risk_budget,
                new_budget=smoothed_budget,
                adjustment_reason=self._generate_adjustment_reason(adjustment_factors),
                performance_regime=self.current_performance_regime,
                market_condition=self.current_market_condition,
                adjustment_factors=adjustment_factors
            )
            self.adjustment_history.append(adjustment)
            
            self.logger.info(f"风险预算调整: {self.current_risk_budget:.3f} -> "
                           f"{smoothed_budget:.3f}, 原因: {adjustment.adjustment_reason}")
        
        # 更新状态
        self.current_risk_budget = smoothed_budget
        self.smoothed_risk_budget = smoothed_budget
        self.last_update_time = current_time
        self.risk_budget_history.append(smoothed_budget)
        
        return smoothed_budget
    
    def _classify_performance_regime(self, metrics: PerformanceMetrics) -> PerformanceRegime:
        """分类表现状态"""
        sharpe = metrics.sharpe_ratio
        
        if sharpe >= self.config.sharpe_threshold_excellent:
            return PerformanceRegime.EXCELLENT
        elif sharpe >= self.config.sharpe_threshold_good:
            return PerformanceRegime.GOOD
        elif sharpe >= self.config.sharpe_threshold_poor:
            return PerformanceRegime.AVERAGE
        elif sharpe >= -0.5:
            return PerformanceRegime.POOR
        else:
            return PerformanceRegime.TERRIBLE
    
    def _classify_market_condition(self, metrics: MarketMetrics) -> MarketCondition:
        """分类市场状态"""
        volatility = metrics.market_volatility
        uncertainty = metrics.uncertainty_index
        trend = metrics.market_trend
        
        # 危机检测
        if uncertainty > 0.8 or volatility > 0.4:
            return MarketCondition.CRISIS
        
        # 波动率分类
        if volatility > self.config.volatility_threshold_high:
            return MarketCondition.HIGH_VOLATILITY
        elif volatility < self.config.volatility_threshold_low:
            return MarketCondition.LOW_VOLATILITY
        
        # 趋势分类
        if trend > 0.1:
            return MarketCondition.BULL
        elif trend < -0.1:
            return MarketCondition.BEAR
        else:
            return MarketCondition.SIDEWAYS
    
    def _calculate_adjustment_factors(self, 
                                    performance_metrics: PerformanceMetrics,
                                    market_metrics: MarketMetrics) -> Dict[str, float]:
        """计算调整因子"""
        factors = {}
        
        # 1. 表现调整因子
        factors['performance'] = self._calculate_performance_factor(performance_metrics)
        
        # 2. 市场条件调整因子
        factors['market'] = self._calculate_market_factor(market_metrics)
        
        # 3. 连续亏损调整因子
        factors['consecutive_loss'] = self._calculate_consecutive_loss_factor(performance_metrics)
        
        # 4. 波动率调整因子
        factors['volatility'] = self._calculate_volatility_factor(market_metrics)
        
        # 5. 不确定性调整因子
        factors['uncertainty'] = self._calculate_uncertainty_factor(market_metrics)
        
        # 6. 恢复调整因子
        factors['recovery'] = self._calculate_recovery_factor()
        
        return factors
    
    def _calculate_performance_factor(self, metrics: PerformanceMetrics) -> float:
        """计算表现调整因子"""
        sharpe = metrics.sharpe_ratio
        calmar = metrics.calmar_ratio
        
        # 基于夏普比率的调整
        if sharpe >= self.config.sharpe_threshold_excellent:
            performance_factor = 1.0 + self.config.performance_adjustment_factor
        elif sharpe >= self.config.sharpe_threshold_good:
            performance_factor = 1.0 + self.config.performance_adjustment_factor * 0.5
        elif sharpe >= self.config.sharpe_threshold_poor:
            performance_factor = 1.0
        else:
            performance_factor = 1.0 - self.config.performance_adjustment_factor
        
        # 卡尔玛比率调整
        if calmar > 2.0:
            performance_factor *= 1.1
        elif calmar < 0.5:
            performance_factor *= 0.9
        
        return np.clip(performance_factor, 0.5, 1.5)
    
    def _calculate_market_factor(self, metrics: MarketMetrics) -> float:
        """计算市场调整因子"""
        condition = self.current_market_condition
        
        if condition == MarketCondition.CRISIS:
            return 0.5  # 危机时大幅降低风险预算
        elif condition == MarketCondition.BEAR:
            return 0.7  # 熊市时降低风险预算
        elif condition == MarketCondition.HIGH_VOLATILITY:
            return 0.8  # 高波动时降低风险预算
        elif condition == MarketCondition.BULL:
            return 1.2  # 牛市时适度增加风险预算
        elif condition == MarketCondition.LOW_VOLATILITY:
            return 1.1  # 低波动时适度增加风险预算
        else:
            return 1.0  # 震荡市保持不变
    
    def _calculate_consecutive_loss_factor(self, metrics: PerformanceMetrics) -> float:
        """计算连续亏损调整因子"""
        consecutive_losses = metrics.consecutive_losses
        
        if consecutive_losses >= self.config.consecutive_loss_threshold:
            penalty = min(
                consecutive_losses * self.config.loss_penalty_factor,
                self.config.max_loss_penalty
            )
            return 1.0 - penalty
        
        return 1.0
    
    def _calculate_volatility_factor(self, metrics: MarketMetrics) -> float:
        """计算波动率调整因子"""
        volatility = metrics.market_volatility
        
        if volatility > self.config.volatility_threshold_high:
            # 高波动时降低风险预算
            excess_vol = volatility - self.config.volatility_threshold_high
            return 1.0 - excess_vol * 2.0
        elif volatility < self.config.volatility_threshold_low:
            # 低波动时适度增加风险预算
            vol_deficit = self.config.volatility_threshold_low - volatility
            return 1.0 + vol_deficit * 0.5
        
        return 1.0
    
    def _calculate_uncertainty_factor(self, metrics: MarketMetrics) -> float:
        """计算不确定性调整因子"""
        uncertainty = metrics.uncertainty_index
        
        if uncertainty > self.config.uncertainty_threshold:
            # 高不确定性时采用保守策略
            excess_uncertainty = uncertainty - self.config.uncertainty_threshold
            return 1.0 - excess_uncertainty * self.config.market_adjustment_factor
        
        return 1.0
    
    def _calculate_recovery_factor(self) -> float:
        """计算恢复调整因子"""
        if len(self.risk_budget_history) < 10:
            return 1.0
        
        # 检查是否处于恢复阶段
        recent_budgets = list(self.risk_budget_history)[-10:]
        current_budget = self.current_risk_budget
        base_budget = self.config.base_risk_budget
        
        if current_budget < base_budget * 0.8:
            # 如果当前预算远低于基础预算，考虑恢复
            trend = np.polyfit(range(len(recent_budgets)), recent_budgets, 1)[0]
            if trend > 0:  # 上升趋势
                return 1.0 + self.config.recovery_speed_fast
            else:
                return 1.0 + self.config.recovery_speed_slow
        
        return 1.0
    
    def _apply_adjustments(self, factors: Dict[str, float]) -> float:
        """应用调整因子"""
        # 计算综合调整因子
        total_factor = 1.0
        
        for factor_name, factor_value in factors.items():
            if factor_name == 'performance':
                total_factor *= factor_value
            elif factor_name == 'market':
                total_factor *= factor_value
            elif factor_name == 'consecutive_loss':
                total_factor *= factor_value
            elif factor_name == 'volatility':
                total_factor *= factor_value ** 0.5  # 降低波动率因子的影响
            elif factor_name == 'uncertainty':
                total_factor *= factor_value
            elif factor_name == 'recovery':
                total_factor *= factor_value
        
        # 应用到基础风险预算
        new_budget = self.config.base_risk_budget * total_factor
        
        # 应用边界约束
        new_budget = np.clip(
            new_budget,
            self.config.min_risk_budget,
            self.config.max_risk_budget
        )
        
        return new_budget
    
    def _apply_smoothing(self, new_budget: float) -> float:
        """应用平滑机制"""
        # EMA平滑
        alpha = self.config.smoothing_factor
        smoothed = alpha * new_budget + (1 - alpha) * self.smoothed_risk_budget
        
        # 限制最大变化率
        max_change = self.current_risk_budget * self.config.max_daily_change
        change = smoothed - self.current_risk_budget
        
        if abs(change) > max_change:
            change = np.sign(change) * max_change
            smoothed = self.current_risk_budget + change
        
        return smoothed
    
    def _detect_anomaly(self, new_budget: float) -> bool:
        """检测异常调整"""
        if len(self.risk_budget_history) < self.config.anomaly_detection_window:
            return False
        
        # 计算历史统计
        recent_budgets = list(self.risk_budget_history)[-self.config.anomaly_detection_window:]
        mean_budget = np.mean(recent_budgets)
        std_budget = np.std(recent_budgets)
        
        if std_budget == 0:
            return False
        
        # Z-score检测
        z_score = abs(new_budget - mean_budget) / std_budget
        
        return z_score > self.config.anomaly_threshold_std
    
    def _handle_anomaly(self, anomalous_budget: float) -> float:
        """处理异常调整"""
        self.last_anomaly_time = datetime.now()
        
        # 使用更保守的调整
        conservative_budget = (
            self.current_risk_budget * (1 - self.config.anomaly_adjustment_factor) +
            anomalous_budget * self.config.anomaly_adjustment_factor
        )
        
        self.logger.warning(f"异常调整处理: {anomalous_budget:.3f} -> {conservative_budget:.3f}")
        
        return conservative_budget
    
    def _generate_adjustment_reason(self, factors: Dict[str, float]) -> str:
        """生成调整原因说明"""
        reasons = []
        
        if factors.get('performance', 1.0) > 1.1:
            reasons.append("表现优秀")
        elif factors.get('performance', 1.0) < 0.9:
            reasons.append("表现不佳")
        
        if factors.get('market', 1.0) < 0.8:
            reasons.append("市场不利")
        elif factors.get('market', 1.0) > 1.1:
            reasons.append("市场有利")
        
        if factors.get('consecutive_loss', 1.0) < 1.0:
            reasons.append("连续亏损")
        
        if factors.get('volatility', 1.0) < 0.9:
            reasons.append("高波动率")
        
        if factors.get('uncertainty', 1.0) < 0.9:
            reasons.append("高不确定性")
        
        if factors.get('recovery', 1.0) > 1.0:
            reasons.append("恢复调整")
        
        return "; ".join(reasons) if reasons else "常规调整"
    
    def _get_current_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前表现指标"""
        if self._performance_cache and self._cache_timestamp:
            # 检查缓存是否过期（1小时）
            if datetime.now() - self._cache_timestamp < timedelta(hours=1):
                return self._performance_cache
        
        if self.performance_history:
            return self.performance_history[-1]
        
        return None
    
    def _get_current_market_metrics(self) -> Optional[MarketMetrics]:
        """获取当前市场指标"""
        if self.market_history:
            return self.market_history[-1]
        
        return None
    
    def get_risk_budget_summary(self) -> Dict[str, Any]:
        """获取风险预算摘要"""
        return {
            'current_risk_budget': self.current_risk_budget,
            'base_risk_budget': self.config.base_risk_budget,
            'performance_regime': self.current_performance_regime.value,
            'market_condition': self.current_market_condition.value,
            'total_adjustments': len(self.adjustment_history),
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'budget_range': {
                'min': min(self.risk_budget_history) if self.risk_budget_history else self.current_risk_budget,
                'max': max(self.risk_budget_history) if self.risk_budget_history else self.current_risk_budget,
                'current': self.current_risk_budget
            },
            'recent_adjustments': [
                {
                    'timestamp': adj.timestamp.isoformat(),
                    'change': adj.new_budget - adj.old_budget,
                    'reason': adj.adjustment_reason
                }
                for adj in self.adjustment_history[-5:]  # 最近5次调整
            ]
        }
    
    def reset_system(self) -> None:
        """重置系统状态"""
        self.current_risk_budget = self.config.base_risk_budget
        self.smoothed_risk_budget = self.config.base_risk_budget
        self.last_update_time = None
        
        self.performance_history.clear()
        self.market_history.clear()
        self.risk_budget_history.clear()
        self.adjustment_history.clear()
        
        self.current_performance_regime = PerformanceRegime.AVERAGE
        self.current_market_condition = MarketCondition.SIDEWAYS
        self.consecutive_adjustments = 0
        self.last_anomaly_time = None
        
        self._performance_cache = None
        self._market_cache = None
        self._cache_timestamp = None
        
        self.logger.info("自适应风险预算系统已重置")