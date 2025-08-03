"""
奖励函数优化器模块

该模块实现了强化学习中的奖励函数优化，包括：
- 风险调整奖励计算
- 回撤惩罚机制
- 多样化奖励机制
- 奖励函数参数自适应优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """奖励函数配置参数"""
    # 基础奖励参数
    base_return_weight: float = 1.0              # 基础收益权重
    risk_aversion_coefficient: float = 0.5       # 风险厌恶系数
    
    # 回撤惩罚参数
    drawdown_penalty_factor: float = 2.0         # 回撤惩罚因子
    drawdown_threshold: float = 0.02             # 回撤惩罚阈值
    max_drawdown_penalty: float = 10.0           # 最大回撤惩罚
    drawdown_nonlinearity: float = 2.0           # 回撤惩罚非线性度
    
    # 动态惩罚参数
    dynamic_penalty_enabled: bool = True         # 启用动态惩罚
    penalty_escalation_factor: float = 1.5       # 惩罚递增因子
    consecutive_loss_threshold: int = 3          # 连续亏损阈值
    
    # 时间衰减惩罚参数
    time_decay_enabled: bool = True              # 启用时间衰减
    penalty_decay_rate: float = 0.1              # 惩罚衰减率
    decay_half_life: int = 10                    # 衰减半衰期（天）
    
    # 回撤阶段差异化参数
    phase_penalty_multipliers: Dict[str, float] = None  # 不同阶段的惩罚倍数
    
    # 多样化奖励参数
    diversification_bonus: float = 0.1           # 多样化奖励系数
    concentration_penalty: float = 0.5           # 集中度惩罚系数
    max_single_position: float = 0.2             # 单一持仓上限
    
    # 风险调整指标参数
    sharpe_target: float = 1.5                   # 目标夏普比率
    calmar_target: float = 1.0                   # 目标卡尔玛比率
    max_volatility: float = 0.25                 # 最大波动率
    
    # 时间衰减参数
    time_decay_factor: float = 0.95              # 时间衰减因子
    lookback_window: int = 252                   # 回看窗口期
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'base_return_weight': self.base_return_weight,
            'risk_aversion_coefficient': self.risk_aversion_coefficient,
            'drawdown_penalty_factor': self.drawdown_penalty_factor,
            'drawdown_threshold': self.drawdown_threshold,
            'max_drawdown_penalty': self.max_drawdown_penalty,
            'drawdown_nonlinearity': self.drawdown_nonlinearity,
            'diversification_bonus': self.diversification_bonus,
            'concentration_penalty': self.concentration_penalty,
            'max_single_position': self.max_single_position,
            'sharpe_target': self.sharpe_target,
            'calmar_target': self.calmar_target,
            'max_volatility': self.max_volatility,
            'time_decay_factor': self.time_decay_factor,
            'lookback_window': self.lookback_window
        }


@dataclass
class RiskAdjustedMetrics:
    """风险调整指标数据类"""
    sharpe_ratio: float                          # 夏普比率
    calmar_ratio: float                          # 卡尔玛比率
    sortino_ratio: float                         # 索提诺比率
    max_drawdown: float                          # 最大回撤
    volatility: float                            # 波动率
    downside_deviation: float                    # 下行偏差
    var_95: float                                # 95% VaR
    cvar_95: float                               # 95% CVaR
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95
        }


class RewardOptimizer:
    """
    奖励函数优化器
    
    负责计算和优化强化学习中的奖励函数，包括：
    - 风险调整奖励计算
    - 回撤惩罚机制
    - 多样化奖励
    - 参数自适应优化
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        初始化奖励函数优化器
        
        Args:
            config: 奖励函数配置参数
        """
        self.config = config or RewardConfig()
        
        # 设置默认的阶段惩罚倍数
        if self.config.phase_penalty_multipliers is None:
            self.config.phase_penalty_multipliers = {
                'NORMAL': 0.0,           # 正常状态无惩罚
                'DRAWDOWN_START': 1.0,   # 回撤开始标准惩罚
                'DRAWDOWN_CONTINUE': 1.5, # 回撤持续加重惩罚
                'RECOVERY': 0.5          # 恢复期减轻惩罚
            }
        
        # 历史数据存储
        self.return_history: List[float] = []
        self.drawdown_history: List[float] = []
        self.position_history: List[Dict[str, float]] = []
        self.reward_history: List[float] = []
        self.timestamps: List[datetime] = []
        self.drawdown_phases: List[str] = []  # 回撤阶段历史
        
        # 惩罚状态跟踪
        self.consecutive_losses = 0
        self.last_drawdown_penalty = 0.0
        self.penalty_history: List[float] = []
        self.phase_penalty_history: List[float] = []
        
        # 性能统计
        self.performance_stats = {
            'total_episodes': 0,
            'positive_rewards': 0,
            'negative_rewards': 0,
            'avg_reward': 0.0,
            'reward_volatility': 0.0,
            'avg_drawdown_penalty': 0.0,
            'max_drawdown_penalty': 0.0
        }
        
        logger.info("奖励函数优化器初始化完成")
    
    def calculate_risk_adjusted_reward(self, 
                                     returns: float,
                                     drawdown: float,
                                     positions: Dict[str, float],
                                     volatility: Optional[float] = None,
                                     drawdown_phase: Optional[str] = None,
                                     timestamp: Optional[datetime] = None) -> float:
        """
        计算风险调整后的奖励
        
        Args:
            returns: 当期收益率
            drawdown: 当前回撤水平
            positions: 当前持仓权重字典
            volatility: 当前波动率（可选）
            drawdown_phase: 当前回撤阶段（可选）
            timestamp: 时间戳（可选）
            
        Returns:
            float: 风险调整后的奖励值
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if drawdown_phase is None:
            drawdown_phase = 'NORMAL'
        
        # 更新历史数据
        self._update_history(returns, drawdown, positions, drawdown_phase, timestamp)
        
        # 1. 基础收益奖励
        base_reward = returns * self.config.base_return_weight
        
        # 2. 增强的回撤惩罚
        drawdown_penalty = self._calculate_enhanced_drawdown_penalty(drawdown, drawdown_phase)
        
        # 3. 多样化奖励
        diversification_reward = self._calculate_diversification_reward(positions)
        
        # 4. 波动率调整
        volatility_adjustment = self._calculate_volatility_adjustment(volatility, returns)
        
        # 5. 综合奖励计算
        total_reward = (base_reward 
                       - drawdown_penalty 
                       + diversification_reward 
                       + volatility_adjustment)
        
        # 更新奖励历史和惩罚统计
        self.reward_history.append(total_reward)
        self.penalty_history.append(drawdown_penalty)
        self.last_drawdown_penalty = drawdown_penalty
        self._update_performance_stats()
        
        logger.debug(f"奖励计算: 基础={base_reward:.4f}, 回撤惩罚={drawdown_penalty:.4f}, "
                    f"多样化={diversification_reward:.4f}, 波动率调整={volatility_adjustment:.4f}, "
                    f"总奖励={total_reward:.4f}")
        
        return total_reward
    
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            
        Returns:
            float: 夏普比率
        """
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate
        
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        
        if std_excess_return == 0:
            return 0.0
        
        # 年化夏普比率
        sharpe = (mean_excess_return / std_excess_return) * np.sqrt(252)
        
        return float(sharpe)
    
    def calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """
        计算卡尔玛比率
        
        Args:
            returns: 收益率序列
            max_drawdown: 最大回撤
            
        Returns:
            float: 卡尔玛比率
        """
        if len(returns) < 2 or max_drawdown == 0:
            return 0.0
        
        # 年化收益率
        annual_return = np.mean(returns) * 252
        
        # 卡尔玛比率 = 年化收益率 / 最大回撤
        calmar = annual_return / abs(max_drawdown)
        
        return float(calmar)
    
    def calculate_sortino_ratio(self, returns: List[float], target_return: float = 0.0) -> float:
        """
        计算索提诺比率
        
        Args:
            returns: 收益率序列
            target_return: 目标收益率
            
        Returns:
            float: 索提诺比率
        """
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - target_return
        
        # 只考虑负超额收益
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        mean_excess_return = np.mean(excess_returns)
        downside_deviation = np.std(downside_returns, ddof=1)
        
        if downside_deviation == 0:
            return 0.0
        
        # 年化索提诺比率
        sortino = (mean_excess_return / downside_deviation) * np.sqrt(252)
        
        return float(sortino)
    
    def calculate_risk_metrics(self, returns: List[float]) -> RiskAdjustedMetrics:
        """
        计算综合风险调整指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            RiskAdjustedMetrics: 风险调整指标
        """
        if len(returns) < 2:
            return self._create_empty_risk_metrics()
        
        returns_array = np.array(returns)
        
        # 基础统计量
        mean_return = np.mean(returns_array)
        volatility = np.std(returns_array, ddof=1) * np.sqrt(252)  # 年化波动率
        
        # 计算回撤
        cumulative_returns = np.cumprod(1 + returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # 下行偏差
        negative_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(252) if len(negative_returns) > 0 else 0.0
        
        # VaR和CVaR (95%置信度)
        var_95 = np.percentile(returns_array, 5)
        cvar_95 = np.mean(returns_array[returns_array <= var_95]) if np.any(returns_array <= var_95) else var_95
        
        # 风险调整比率
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns, max_drawdown)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        
        return RiskAdjustedMetrics(
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            downside_deviation=downside_deviation,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _calculate_drawdown_penalty(self, drawdown: float) -> float:
        """
        计算基础回撤惩罚
        
        Args:
            drawdown: 当前回撤水平
            
        Returns:
            float: 回撤惩罚值
        """
        # 只对超过阈值的回撤进行惩罚
        if abs(drawdown) <= self.config.drawdown_threshold:
            return 0.0
        
        # 超额回撤
        excess_drawdown = abs(drawdown) - self.config.drawdown_threshold
        
        # 非线性惩罚
        penalty = (self.config.drawdown_penalty_factor * 
                  (excess_drawdown ** self.config.drawdown_nonlinearity))
        
        # 限制最大惩罚
        penalty = min(penalty, self.config.max_drawdown_penalty)
        
        return penalty
    
    def _calculate_enhanced_drawdown_penalty(self, drawdown: float, drawdown_phase: str) -> float:
        """
        计算增强的回撤惩罚（包含动态调整、时间衰减和阶段差异化）
        
        Args:
            drawdown: 当前回撤水平
            drawdown_phase: 当前回撤阶段
            
        Returns:
            float: 增强的回撤惩罚值
        """
        # 1. 基础惩罚
        base_penalty = self._calculate_drawdown_penalty(drawdown)
        
        if base_penalty == 0.0:
            return 0.0
        
        # 2. 动态惩罚调整
        dynamic_multiplier = self._calculate_dynamic_penalty_multiplier(drawdown)
        
        # 3. 时间衰减调整
        time_decay_multiplier = self._calculate_time_decay_multiplier()
        
        # 4. 阶段差异化调整
        phase_multiplier = self._calculate_phase_penalty_multiplier(drawdown_phase)
        
        # 5. 综合惩罚
        enhanced_penalty = (base_penalty * 
                           dynamic_multiplier * 
                           time_decay_multiplier * 
                           phase_multiplier)
        
        # 限制最大惩罚
        enhanced_penalty = min(enhanced_penalty, self.config.max_drawdown_penalty)
        
        # 记录阶段惩罚
        self.phase_penalty_history.append(enhanced_penalty - base_penalty)
        
        return enhanced_penalty
    
    def _calculate_dynamic_penalty_multiplier(self, drawdown: float) -> float:
        """
        计算动态惩罚倍数（基于连续亏损和回撤恶化）
        
        Args:
            drawdown: 当前回撤水平
            
        Returns:
            float: 动态惩罚倍数
        """
        if not self.config.dynamic_penalty_enabled:
            return 1.0
        
        multiplier = 1.0
        
        # 1. 连续亏损惩罚
        if len(self.return_history) > 0:
            # 更新连续亏损计数
            if self.return_history[-1] < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # 连续亏损超过阈值时增加惩罚
            if self.consecutive_losses >= self.config.consecutive_loss_threshold:
                excess_losses = self.consecutive_losses - self.config.consecutive_loss_threshold
                multiplier *= (1.0 + excess_losses * 0.2)  # 每多一次连续亏损增加20%惩罚
        
        # 2. 回撤恶化惩罚
        if len(self.drawdown_history) >= 2:
            current_dd = abs(drawdown)
            previous_dd = abs(self.drawdown_history[-2])
            
            if current_dd > previous_dd:
                # 回撤恶化，增加惩罚
                deterioration_ratio = current_dd / max(previous_dd, 0.001)
                multiplier *= min(deterioration_ratio, self.config.penalty_escalation_factor)
        
        return multiplier
    
    def _calculate_time_decay_multiplier(self) -> float:
        """
        计算时间衰减倍数（历史惩罚的时间衰减效应）
        
        Returns:
            float: 时间衰减倍数
        """
        if not self.config.time_decay_enabled or len(self.penalty_history) < 2:
            return 1.0
        
        # 计算历史惩罚的加权平均（时间衰减）
        decay_weights = []
        for i in range(len(self.penalty_history)):
            days_ago = len(self.penalty_history) - 1 - i
            weight = np.exp(-days_ago * self.config.penalty_decay_rate)
            decay_weights.append(weight)
        
        if sum(decay_weights) == 0:
            return 1.0
        
        # 归一化权重
        decay_weights = np.array(decay_weights) / sum(decay_weights)
        
        # 计算加权历史惩罚
        weighted_historical_penalty = np.sum(np.array(self.penalty_history) * decay_weights)
        
        # 如果历史惩罚较高，当前惩罚应该有所缓解（避免过度惩罚）
        if weighted_historical_penalty > 0:
            decay_factor = 1.0 / (1.0 + weighted_historical_penalty * 0.1)
            return max(decay_factor, 0.5)  # 最多减少50%的惩罚
        
        return 1.0
    
    def _calculate_phase_penalty_multiplier(self, drawdown_phase: str) -> float:
        """
        计算阶段差异化惩罚倍数
        
        Args:
            drawdown_phase: 当前回撤阶段
            
        Returns:
            float: 阶段惩罚倍数
        """
        return self.config.phase_penalty_multipliers.get(drawdown_phase, 1.0)
    
    def get_penalty_analysis(self) -> Dict[str, Any]:
        """
        获取惩罚机制分析报告
        
        Returns:
            Dict[str, Any]: 惩罚分析报告
        """
        if not self.penalty_history:
            return {'status': '无惩罚历史数据'}
        
        penalty_array = np.array(self.penalty_history)
        phase_penalty_array = np.array(self.phase_penalty_history) if self.phase_penalty_history else np.array([])
        
        analysis = {
            'total_penalties': len(self.penalty_history),
            'avg_penalty': np.mean(penalty_array),
            'max_penalty': np.max(penalty_array),
            'min_penalty': np.min(penalty_array),
            'penalty_volatility': np.std(penalty_array),
            'consecutive_losses': self.consecutive_losses,
            'last_penalty': self.last_drawdown_penalty,
            'penalty_trend': self._calculate_penalty_trend(),
            'phase_penalty_contribution': {
                'avg_phase_penalty': np.mean(phase_penalty_array) if len(phase_penalty_array) > 0 else 0.0,
                'max_phase_penalty': np.max(phase_penalty_array) if len(phase_penalty_array) > 0 else 0.0,
                'phase_penalty_ratio': (np.mean(phase_penalty_array) / np.mean(penalty_array) 
                                      if len(phase_penalty_array) > 0 and np.mean(penalty_array) > 0 else 0.0)
            }
        }
        
        return analysis
    
    def _calculate_penalty_trend(self) -> str:
        """计算惩罚趋势"""
        if len(self.penalty_history) < 5:
            return '数据不足'
        
        recent_penalties = self.penalty_history[-5:]
        early_penalties = self.penalty_history[-10:-5] if len(self.penalty_history) >= 10 else self.penalty_history[:-5]
        
        if not early_penalties:
            return '数据不足'
        
        recent_avg = np.mean(recent_penalties)
        early_avg = np.mean(early_penalties)
        
        if recent_avg > early_avg * 1.1:
            return '上升'
        elif recent_avg < early_avg * 0.9:
            return '下降'
        else:
            return '稳定'
    
    def _calculate_diversification_reward(self, positions: Dict[str, float]) -> float:
        """
        计算多样化奖励
        
        Args:
            positions: 持仓权重字典
            
        Returns:
            float: 多样化奖励值
        """
        if not positions:
            return 0.0
        
        # 计算集中度��标（赫芬达尔指数）
        weights = np.array(list(positions.values()))
        weights = np.abs(weights)  # 取绝对值处理空头持仓
        
        if np.sum(weights) == 0:
            return 0.0
        
        # 归一化权重
        normalized_weights = weights / np.sum(weights)
        
        # 赫芬达尔指数（越小越分散）
        herfindahl_index = np.sum(normalized_weights ** 2)
        
        # 多样化程度（1表示完全分散，接近1/n表示高度集中）
        n_assets = len(positions)
        max_diversification = 1.0 / n_assets if n_assets > 0 else 0.0
        diversification_score = (1.0 - herfindahl_index) / (1.0 - max_diversification) if max_diversification < 1.0 else 0.0
        
        # 多样化奖励
        diversification_reward = self.config.diversification_bonus * diversification_score
        
        # 集中度惩罚（对单一持仓过大的惩罚）
        concentration_penalty = 0.0
        for weight in normalized_weights:
            if weight > self.config.max_single_position:
                excess_concentration = weight - self.config.max_single_position
                concentration_penalty += self.config.concentration_penalty * (excess_concentration ** 2)
        
        # Use enhanced diversification calculation
        # 1. 基础多样化奖励
        basic_diversification_reward = self._calculate_basic_diversification_reward(positions)
        
        # 2. 动态多样化奖励
        dynamic_diversification_reward = self._calculate_dynamic_diversification_reward(positions)
        
        # 3. 集中度惩罚
        enhanced_concentration_penalty = self._calculate_enhanced_concentration_penalty(positions)
        
        # 4. 相关性调整奖励
        correlation_adjustment = self._calculate_correlation_adjustment_reward(positions)
        
        # 5. 综合多样化奖励
        total_diversification_reward = (basic_diversification_reward + 
                                      dynamic_diversification_reward + 
                                      correlation_adjustment - 
                                      enhanced_concentration_penalty)
        
        return total_diversification_reward
    
    def _calculate_basic_diversification_reward(self, positions: Dict[str, float]) -> float:
        """
        计算基础多样化奖励（基于赫芬达尔指数）
        
        Args:
            positions: 持仓权重字典
            
        Returns:
            float: 基础多样化奖励
        """
        weights = np.array(list(positions.values()))
        weights = np.abs(weights)  # 取绝对值处理空头持仓
        
        if np.sum(weights) == 0:
            return 0.0
        
        # 归一化权重
        normalized_weights = weights / np.sum(weights)
        
        # 赫芬达尔指数（越小越分散）
        herfindahl_index = np.sum(normalized_weights ** 2)
        
        # 多样化程度
        n_assets = len(positions)
        max_diversification = 1.0 / n_assets if n_assets > 0 else 0.0
        diversification_score = (1.0 - herfindahl_index) / (1.0 - max_diversification) if max_diversification < 1.0 else 0.0
        
        # 基础多样化奖励
        return self.config.diversification_bonus * diversification_score
    
    def _calculate_dynamic_diversification_reward(self, positions: Dict[str, float]) -> float:
        """
        计算动态多样化奖励（基于历史集中度变化）
        
        Args:
            positions: 持仓权重字典
            
        Returns:
            float: 动态多样化奖励
        """
        if len(self.position_history) < 2:
            return 0.0
        
        # 计算当前集中度
        current_concentration = self._calculate_concentration_score(positions)
        
        # 计算历史平均集中度
        historical_concentrations = []
        for hist_positions in self.position_history[-10:]:  # 最近10期
            if hist_positions:
                hist_concentration = self._calculate_concentration_score(hist_positions)
                historical_concentrations.append(hist_concentration)
        
        if not historical_concentrations:
            return 0.0
        
        avg_historical_concentration = np.mean(historical_concentrations)
        
        # 如果当前集中度低于历史平均，给予奖励
        if current_concentration < avg_historical_concentration:
            improvement = avg_historical_concentration - current_concentration
            return self.config.diversification_bonus * 0.5 * improvement
        
        return 0.0
    
    def _calculate_enhanced_concentration_penalty(self, positions: Dict[str, float]) -> float:
        """
        计算增强的集中度惩罚
        
        Args:
            positions: 持仓权重字典
            
        Returns:
            float: 集中度惩罚值
        """
        if not positions:
            return 0.0
        
        weights = np.array(list(positions.values()))
        weights = np.abs(weights)
        
        if np.sum(weights) == 0:
            return 0.0
        
        # 归一化权重
        normalized_weights = weights / np.sum(weights)
        
        # 单一持仓过大的惩罚
        single_position_penalty = 0.0
        for weight in normalized_weights:
            if weight > self.config.max_single_position:
                excess_concentration = weight - self.config.max_single_position
                single_position_penalty += self.config.concentration_penalty * (excess_concentration ** 2)
        
        # 整体集中度惩罚
        concentration_score = self._calculate_concentration_score(positions)
        overall_concentration_penalty = 0.0
        
        # 如果集中度过高（前3大持仓占比超过80%），额外惩罚
        if len(normalized_weights) >= 3:
            top3_concentration = np.sum(np.sort(normalized_weights)[-3:])
            if top3_concentration > 0.8:
                excess_top3 = top3_concentration - 0.8
                overall_concentration_penalty = self.config.concentration_penalty * 0.5 * (excess_top3 ** 2)
        
        return single_position_penalty + overall_concentration_penalty
    
    def _calculate_correlation_adjustment_reward(self, positions: Dict[str, float]) -> float:
        """
        计算相关性调整奖励（基于资产间相关性的多样化效果）
        
        Args:
            positions: 持仓权重字典
            
        Returns:
            float: 相关性调整奖励
        """
        # 这里简化处理，实际应用中需要资产间的相关性矩阵
        # 假设资产数量越多，相关性越低，多样化效果越好
        n_assets = len(positions)
        
        if n_assets <= 1:
            return 0.0
        
        # 基于资产数量的相关性调整
        # 资产数量越多，假设相关性越低，给予更多奖励
        correlation_adjustment = self.config.diversification_bonus * 0.2 * np.log(n_assets)
        
        return correlation_adjustment
    
    def _calculate_concentration_score(self, positions: Dict[str, float]) -> float:
        """
        计算集中度评分（赫芬达尔指数）
        
        Args:
            positions: 持仓权重字典
            
        Returns:
            float: 集中度评分（0-1，越高越集中）
        """
        if not positions:
            return 0.0
        
        weights = np.array(list(positions.values()))
        weights = np.abs(weights)
        
        if np.sum(weights) == 0:
            return 0.0
        
        # 归一化权重
        normalized_weights = weights / np.sum(weights)
        
        # 赫芬达尔指数
        herfindahl_index = np.sum(normalized_weights ** 2)
        
        return herfindahl_index
    
    def calculate_portfolio_diversification_metrics(self, positions: Dict[str, float]) -> Dict[str, float]:
        """
        计算投资组合多样化指标
        
        Args:
            positions: 持仓权重字典
            
        Returns:
            Dict[str, float]: 多样化指标字典
        """
        if not positions:
            return {
                'herfindahl_index': 0.0,
                'effective_number_of_assets': 0.0,
                'max_weight': 0.0,
                'weight_entropy': 0.0,
                'diversification_ratio': 0.0,
                'concentration_score': 0.0
            }
        
        weights = np.array(list(positions.values()))
        weights = np.abs(weights)
        
        if np.sum(weights) == 0:
            return {
                'herfindahl_index': 0.0,
                'effective_number_of_assets': 0.0,
                'max_weight': 0.0,
                'weight_entropy': 0.0,
                'diversification_ratio': 0.0,
                'concentration_score': 0.0
            }
        
        # 归一化权重
        normalized_weights = weights / np.sum(weights)
        
        # 1. 赫芬达尔指数
        herfindahl_index = np.sum(normalized_weights ** 2)
        
        # 2. 有效资产数量
        effective_number_of_assets = 1.0 / herfindahl_index if herfindahl_index > 0 else 0.0
        
        # 3. 最大权重
        max_weight = np.max(normalized_weights)
        
        # 4. 权重熵
        # 避免log(0)的问题
        non_zero_weights = normalized_weights[normalized_weights > 1e-10]
        weight_entropy = -np.sum(non_zero_weights * np.log(non_zero_weights)) if len(non_zero_weights) > 0 else 0.0
        
        # 5. 多样化比率
        n_assets = len(positions)
        max_entropy = np.log(n_assets) if n_assets > 1 else 0.0
        diversification_ratio = weight_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 6. 集中度评分
        concentration_score = herfindahl_index
        
        return {
            'herfindahl_index': herfindahl_index,
            'effective_number_of_assets': effective_number_of_assets,
            'max_weight': max_weight,
            'weight_entropy': weight_entropy,
            'diversification_ratio': diversification_ratio,
            'concentration_score': concentration_score
        }
    
    def optimize_diversification_parameters(self, target_diversification: float = 0.8) -> Dict[str, float]:
        """
        优化多样化参数
        
        Args:
            target_diversification: 目标多样化水平（0-1）
            
        Returns:
            Dict[str, float]: 优化建议
        """
        if not self.position_history:
            return {'status': '无历史数据'}
        
        # 计算历史多样化指标
        diversification_scores = []
        concentration_scores = []
        
        for positions in self.position_history[-50:]:  # 最近50期
            if positions:
                metrics = self.calculate_portfolio_diversification_metrics(positions)
                diversification_scores.append(metrics['diversification_ratio'])
                concentration_scores.append(metrics['concentration_score'])
        
        if not diversification_scores:
            return {'status': '无有效历史数据'}
        
        avg_diversification = np.mean(diversification_scores)
        avg_concentration = np.mean(concentration_scores)
        
        # 生成优化建议
        suggestions = {
            'current_diversification': avg_diversification,
            'target_diversification': target_diversification,
            'current_concentration': avg_concentration,
            'diversification_gap': target_diversification - avg_diversification
        }
        
        # 参数调整建议
        if avg_diversification < target_diversification:
            # 需要提高多样化
            suggestions['diversification_bonus_adjustment'] = min(1.5, 1.0 + (target_diversification - avg_diversification))
            suggestions['concentration_penalty_adjustment'] = min(2.0, 1.0 + (target_diversification - avg_diversification) * 2)
            suggestions['max_position_adjustment'] = max(0.1, self.config.max_single_position * 0.8)
        else:
            # 多样化已足够
            suggestions['diversification_bonus_adjustment'] = 1.0
            suggestions['concentration_penalty_adjustment'] = 1.0
            suggestions['max_position_adjustment'] = self.config.max_single_position
        
        return suggestions
    
    def _calculate_volatility_adjustment(self, volatility: Optional[float], returns: float) -> float:
        """
        计算波动率调整
        
        Args:
            volatility: 当前波动率
            returns: 当期收益率
            
        Returns:
            float: 波动率调整值
        """
        if volatility is None:
            # 如果没有提供波动率，使用历史数据估算
            if len(self.return_history) >= 10:
                recent_returns = self.return_history[-10:]
                volatility = np.std(recent_returns) * np.sqrt(252)
            else:
                return 0.0
        
        # 波动率过高的惩罚
        if volatility > self.config.max_volatility:
            excess_volatility = volatility - self.config.max_volatility
            volatility_penalty = self.config.risk_aversion_coefficient * (excess_volatility ** 2)
            return -volatility_penalty
        
        # 适度波动率的奖励（风险调整收益）
        if volatility > 0:
            risk_adjusted_return = returns / volatility
            return self.config.risk_aversion_coefficient * risk_adjusted_return * 0.1
        
        return 0.0
    
    def _update_history(self, returns: float, drawdown: float, 
                       positions: Dict[str, float], drawdown_phase: str, timestamp: datetime):
        """更新历史数据"""
        self.return_history.append(returns)
        self.drawdown_history.append(drawdown)
        self.position_history.append(positions.copy())
        self.drawdown_phases.append(drawdown_phase)
        self.timestamps.append(timestamp)
        
        # 限制历史数据长度
        max_history = self.config.lookback_window
        if len(self.return_history) > max_history:
            self.return_history.pop(0)
            self.drawdown_history.pop(0)
            self.position_history.pop(0)
            self.drawdown_phases.pop(0)
            self.timestamps.pop(0)
            if self.reward_history:
                self.reward_history.pop(0)
            if self.penalty_history:
                self.penalty_history.pop(0)
            if self.phase_penalty_history:
                self.phase_penalty_history.pop(0)
    
    def _update_performance_stats(self):
        """更新性能统计"""
        if not self.reward_history:
            return
        
        self.performance_stats['total_episodes'] = len(self.reward_history)
        self.performance_stats['positive_rewards'] = sum(1 for r in self.reward_history if r > 0)
        self.performance_stats['negative_rewards'] = sum(1 for r in self.reward_history if r < 0)
        self.performance_stats['avg_reward'] = np.mean(self.reward_history)
        self.performance_stats['reward_volatility'] = np.std(self.reward_history) if len(self.reward_history) > 1 else 0.0
        
        # 更新惩罚统计
        if self.penalty_history:
            self.performance_stats['avg_drawdown_penalty'] = np.mean(self.penalty_history)
            self.performance_stats['max_drawdown_penalty'] = np.max(self.penalty_history)
    
    def _create_empty_risk_metrics(self) -> RiskAdjustedMetrics:
        """创建空的风险指标"""
        return RiskAdjustedMetrics(
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            downside_deviation=0.0,
            var_95=0.0,
            cvar_95=0.0
        )
    
    def optimize_reward_parameters(self, target_metrics: Dict[str, float]) -> RewardConfig:
        """
        基于目标指标优化奖励函数参数
        
        Args:
            target_metrics: 目标指标字典
                - 'target_sharpe': 目标夏普比率
                - 'target_calmar': 目标卡尔玛比率
                - 'max_drawdown': 最大可接受回撤
                - 'target_volatility': 目标波动率
                
        Returns:
            RewardConfig: 优化后的配置参数
        """
        if len(self.return_history) < 50:  # 需要足够的历史数据
            logger.warning("历史数据不足，无法进行参数优化")
            return self.config
        
        # 计算当前性能指标
        current_metrics = self.calculate_risk_metrics(self.return_history)
        
        # 创建新的配置
        new_config = RewardConfig(
            base_return_weight=self.config.base_return_weight,
            risk_aversion_coefficient=self.config.risk_aversion_coefficient,
            drawdown_penalty_factor=self.config.drawdown_penalty_factor,
            drawdown_threshold=self.config.drawdown_threshold,
            max_drawdown_penalty=self.config.max_drawdown_penalty,
            drawdown_nonlinearity=self.config.drawdown_nonlinearity,
            diversification_bonus=self.config.diversification_bonus,
            concentration_penalty=self.config.concentration_penalty,
            max_single_position=self.config.max_single_position,
            sharpe_target=target_metrics.get('target_sharpe', self.config.sharpe_target),
            calmar_target=target_metrics.get('target_calmar', self.config.calmar_target),
            max_volatility=target_metrics.get('target_volatility', self.config.max_volatility),
            time_decay_factor=self.config.time_decay_factor,
            lookback_window=self.config.lookback_window
        )
        
        # 调整回撤惩罚参数
        target_max_dd = target_metrics.get('max_drawdown', 0.1)
        if abs(current_metrics.max_drawdown) > target_max_dd:
            # 增加回撤惩罚
            adjustment_factor = abs(current_metrics.max_drawdown) / target_max_dd
            new_config.drawdown_penalty_factor *= min(adjustment_factor, 2.0)
        
        # 调整风险厌恶系数
        target_sharpe = target_metrics.get('target_sharpe', 1.5)
        if current_metrics.sharpe_ratio < target_sharpe:
            # 增加风险厌恶
            new_config.risk_aversion_coefficient *= 1.2
        elif current_metrics.sharpe_ratio > target_sharpe * 1.2:
            # 减少风险厌恶
            new_config.risk_aversion_coefficient *= 0.9
        
        # 调整多样化奖励
        if self.position_history:
            avg_concentration = self._calculate_average_concentration()
            if avg_concentration > 0.5:  # 过度集中
                new_config.diversification_bonus *= 1.3
                new_config.concentration_penalty *= 1.2
        
        logger.info(f"奖励参数优化完成: 回撤惩罚因子 {self.config.drawdown_penalty_factor:.2f} -> {new_config.drawdown_penalty_factor:.2f}")
        
        return new_config
    
    def _calculate_average_concentration(self) -> float:
        """计算平均集中度"""
        if not self.position_history:
            return 0.0
        
        concentrations = []
        for positions in self.position_history[-50:]:  # 最近50期
            if positions:
                weights = np.array(list(positions.values()))
                weights = np.abs(weights)
                if np.sum(weights) > 0:
                    normalized_weights = weights / np.sum(weights)
                    herfindahl_index = np.sum(normalized_weights ** 2)
                    concentrations.append(herfindahl_index)
        
        return np.mean(concentrations) if concentrations else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.return_history:
            return {'status': '无历史数据'}
        
        risk_metrics = self.calculate_risk_metrics(self.return_history)
        
        return {
            'episodes': len(self.return_history),
            'avg_return': np.mean(self.return_history),
            'total_return': np.sum(self.return_history),
            'avg_drawdown': np.mean(self.drawdown_history) if self.drawdown_history else 0.0,
            'max_drawdown': np.min(self.drawdown_history) if self.drawdown_history else 0.0,
            'risk_metrics': risk_metrics.to_dict(),
            'reward_stats': self.performance_stats.copy(),
            'config': self.config.to_dict()
        }
    
    def reset(self):
        """重置优化器状态"""
        self.return_history.clear()
        self.drawdown_history.clear()
        self.position_history.clear()
        self.reward_history.clear()
        self.timestamps.clear()
        self.drawdown_phases.clear()
        self.penalty_history.clear()
        self.phase_penalty_history.clear()
        
        # 重置惩罚状态
        self.consecutive_losses = 0
        self.last_drawdown_penalty = 0.0
        
        self.performance_stats = {
            'total_episodes': 0,
            'positive_rewards': 0,
            'negative_rewards': 0,
            'avg_reward': 0.0,
            'reward_volatility': 0.0,
            'avg_drawdown_penalty': 0.0,
            'max_drawdown_penalty': 0.0
        }
        
        logger.info("奖励函数优化器已重置")