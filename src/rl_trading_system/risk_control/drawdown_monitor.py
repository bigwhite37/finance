"""
回撤监控模块

该模块实现了实时回撤监控、回撤阶段识别和市场状态检测功能。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DrawdownPhase(Enum):
    """回撤阶段枚举"""
    NORMAL = "正常"           # 正常状态，无显著回撤
    DRAWDOWN_START = "回撤开始"  # 回撤刚开始
    DRAWDOWN_CONTINUE = "回撤持续"  # 回撤持续中
    RECOVERY = "恢复中"        # 从回撤中恢复


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL_MARKET = "牛市"       # 牛市
    BEAR_MARKET = "熊市"       # 熊市
    SIDEWAYS_MARKET = "震荡市"  # 震荡市
    HIGH_VOLATILITY = "高波动"  # 高波动期
    LOW_VOLATILITY = "低波动"   # 低波动期
    CRISIS = "危机"           # 危机期


@dataclass
class DrawdownMetrics:
    """回撤指标数据类"""
    current_drawdown: float                    # 当前回撤
    max_drawdown: float                        # 最大回撤
    drawdown_duration: int                     # 回撤持续天数
    recovery_time: Optional[int]               # 恢复时间
    peak_value: float                          # 峰值净值
    trough_value: float                        # 谷值净值
    underwater_curve: List[float]              # 水下曲线
    drawdown_frequency: float                  # 回撤频率
    average_drawdown: float                    # 平均回撤
    current_phase: DrawdownPhase               # 当前回撤阶段
    days_since_peak: int                       # 距离峰值天数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'recovery_time': self.recovery_time,
            'peak_value': self.peak_value,
            'trough_value': self.trough_value,
            'drawdown_frequency': self.drawdown_frequency,
            'average_drawdown': self.average_drawdown,
            'current_phase': self.current_phase.value,
            'days_since_peak': self.days_since_peak
        }


@dataclass
class MarketStateMetrics:
    """市场状态指标数据类"""
    regime: MarketRegime                       # 市场状态
    volatility: float                          # 波动率
    trend_strength: float                      # 趋势强度
    correlation_level: float                   # 相关性水平
    liquidity_score: float                     # 流动性评分
    confidence_score: float                    # 状态识别置信度

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'regime': self.regime.value,
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'correlation_level': self.correlation_level,
            'liquidity_score': self.liquidity_score,
            'confidence_score': self.confidence_score
        }


class DrawdownMonitor:
    """
    回撤监控器

    负责实时监控投资组合的回撤情况，包括：
    - 实时回撤计算
    - 回撤阶段识别
    - 市场状态检测
    - 回撤归因分析
    """

    def __init__(self,
                 drawdown_threshold: float = 0.05,
                 recovery_threshold: float = 0.02,
                 lookback_window: int = 252,
                 volatility_window: int = 20):
        """
        初始化回撤监控器

        Args:
            drawdown_threshold: 回撤阈值，超过此值认为进入回撤状态
            recovery_threshold: 恢复阈值，回撤减少超过此值认为开始恢复
            lookback_window: 回看窗口期（交易日）
            volatility_window: 波动率计算窗口期
        """
        self.drawdown_threshold = drawdown_threshold
        self.recovery_threshold = recovery_threshold
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window

        # 历史数据存储
        self.portfolio_values: List[float] = []
        self.timestamps: List[datetime] = []
        self.drawdown_history: List[float] = []
        self.peak_history: List[float] = []

        # 当前状态
        self.current_peak = 0.0
        self.current_trough = float('inf')
        self.peak_timestamp = None
        self.trough_timestamp = None
        self.current_phase = DrawdownPhase.NORMAL
        self.phase_start_time = None

        # 日志优化：阶段变化去抖动机制
        self.last_logged_phase = DrawdownPhase.NORMAL
        self.phase_stability_count = 0
        self.phase_stability_threshold = 2  # 需要连续稳定2次才记录日志

        logger.info("回撤监控器初始化完成")

    def update_portfolio_value(self, value: float, timestamp: datetime = None) -> DrawdownMetrics:
        """
        更新投资组合净值并计算回撤指标

        Args:
            value: 当前投资组合净值
            timestamp: 时间戳，默认为当前时间

        Returns:
            DrawdownMetrics: 回撤指标
        """
        if timestamp is None:
            timestamp = datetime.now()

        # 更新历史数据
        self.portfolio_values.append(value)
        self.timestamps.append(timestamp)

        # 限制历史数据长度
        if len(self.portfolio_values) > self.lookback_window:
            self.portfolio_values.pop(0)
            self.timestamps.pop(0)
            if self.drawdown_history:
                self.drawdown_history.pop(0)
            if self.peak_history:
                self.peak_history.pop(0)

        # 计算回撤指标
        metrics = self._calculate_drawdown_metrics()

        # 更新回撤阶段
        self._update_drawdown_phase(metrics)

        logger.debug(f"更新投资组合净值: {value:.4f}, 当前回撤: {metrics.current_drawdown:.4f}")

        return metrics

    def _calculate_drawdown_metrics(self) -> DrawdownMetrics:
        """计算回撤指标"""
        if len(self.portfolio_values) < 2:
            return self._create_empty_metrics()

        values = np.array(self.portfolio_values)

        # 计算滚动最大值
        running_max = np.maximum.accumulate(values)
        self.peak_history = running_max.tolist()

        # 计算回撤
        drawdowns = (values - running_max) / running_max
        self.drawdown_history = drawdowns.tolist()

        # 当前回撤
        current_drawdown = drawdowns[-1]

        # 最大回撤
        max_drawdown = np.min(drawdowns)

        # 更新峰值和谷值
        current_peak = running_max[-1]
        if current_peak > self.current_peak:
            self.current_peak = current_peak
            self.peak_timestamp = self.timestamps[-1]

        current_value = values[-1]
        if current_value < self.current_trough:
            self.current_trough = current_value
            self.trough_timestamp = self.timestamps[-1]

        # 计算回撤持续时间
        drawdown_duration = self._calculate_drawdown_duration(drawdowns)

        # 计算恢复时间
        recovery_time = self._calculate_recovery_time(drawdowns)

        # 计算水下曲线
        underwater_curve = drawdowns.tolist()

        # 计算回撤频率
        drawdown_frequency = self._calculate_drawdown_frequency(drawdowns)

        # 计算平均回撤
        negative_drawdowns = drawdowns[drawdowns < -0.001]  # 过滤掉微小的负值
        average_drawdown = np.mean(negative_drawdowns) if len(negative_drawdowns) > 0 else 0.0

        # 距离峰值天数
        days_since_peak = self._calculate_days_since_peak()

        return DrawdownMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            drawdown_duration=drawdown_duration,
            recovery_time=recovery_time,
            peak_value=self.current_peak,
            trough_value=self.current_trough,
            underwater_curve=underwater_curve,
            drawdown_frequency=drawdown_frequency,
            average_drawdown=average_drawdown,
            current_phase=self.current_phase,
            days_since_peak=days_since_peak
        )

    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """计算当前回撤持续时间"""
        if len(drawdowns) == 0:
            return 0

        # 从后往前找到第一个非负值的位置
        duration = 0
        for i in range(len(drawdowns) - 1, -1, -1):
            if drawdowns[i] < -0.001:  # 考虑数值精度
                duration += 1
            else:
                break

        return duration

    def _calculate_recovery_time(self, drawdowns: np.ndarray) -> Optional[int]:
        """计算从最大回撤恢复的时间"""
        if len(drawdowns) == 0:
            return None

        # 找到最大回撤的位置
        max_dd_idx = np.argmin(drawdowns)

        # 从最大回撤位置往后找恢复点
        for i in range(max_dd_idx + 1, len(drawdowns)):
            if drawdowns[i] >= -0.001:  # 基本恢复到峰值
                return i - max_dd_idx

        return None

    def _calculate_drawdown_frequency(self, drawdowns: np.ndarray) -> float:
        """计算回撤频率（每年回撤次数）"""
        if len(drawdowns) < 2:
            return 0.0

        # 识别回撤事件（从正值变为负值）
        drawdown_events = 0
        in_drawdown = False

        for dd in drawdowns:
            if dd < -0.001 and not in_drawdown:
                drawdown_events += 1
                in_drawdown = True
            elif dd >= -0.001:
                in_drawdown = False

        # 计算年化频率
        days = len(drawdowns)
        years = days / 252.0  # 假设一年252个交易日

        return drawdown_events / years if years > 0 else 0.0

    def _calculate_days_since_peak(self) -> int:
        """计算距离峰值的天数"""
        if self.peak_timestamp is None or len(self.timestamps) == 0:
            return 0

        # 找到峰值在历史数据中的位置
        for i, ts in enumerate(reversed(self.timestamps)):
            if abs((ts - self.peak_timestamp).total_seconds()) < 60:  # 1分钟内认为是同一时间
                return i

        return len(self.timestamps) - 1

    def _update_drawdown_phase(self, metrics: DrawdownMetrics):
        """更新回撤阶段"""
        current_dd = abs(metrics.current_drawdown)

        if current_dd < 0.001:  # 基本无回撤
            new_phase = DrawdownPhase.NORMAL
        elif current_dd >= self.drawdown_threshold:
            if self.current_phase == DrawdownPhase.NORMAL:
                new_phase = DrawdownPhase.DRAWDOWN_START
            elif len(self.drawdown_history) >= 2:
                # 比较最近两个回撤值判断趋势
                recent_dd = abs(self.drawdown_history[-2])
                if current_dd > recent_dd + 0.005:  # 回撤加深
                    new_phase = DrawdownPhase.DRAWDOWN_CONTINUE
                elif current_dd < recent_dd - self.recovery_threshold:  # 回撤减少
                    new_phase = DrawdownPhase.RECOVERY
                else:
                    new_phase = self.current_phase  # 保持当前阶段
            else:
                new_phase = DrawdownPhase.DRAWDOWN_CONTINUE
        else:
            # 轻微回撤，判断是否在恢复
            if self.current_phase in [DrawdownPhase.DRAWDOWN_START, DrawdownPhase.DRAWDOWN_CONTINUE]:
                new_phase = DrawdownPhase.RECOVERY
            else:
                new_phase = DrawdownPhase.NORMAL

        # 更新阶段（使用去抖动的智能日志记录）
        if new_phase != self.current_phase:
            old_phase = self.current_phase
            self.current_phase = new_phase
            self.phase_start_time = self.timestamps[-1] if self.timestamps else datetime.now()

            # 智能日志记录：使用去抖动机制避免频繁日志
            self._log_phase_change_with_debounce(old_phase, new_phase)

        # 更新metrics中的阶段信息
        metrics.current_phase = self.current_phase

    def _log_phase_change_with_debounce(self, old_phase: DrawdownPhase, new_phase: DrawdownPhase):
        """
        使用去抖动机制的智能阶段变化日志记录

        只在以下情况记录日志：
        1. 重要的阶段转换（如正常->回撤开始、恢复->正常）
        2. 状态稳定一定次数后的变化
        3. 避免恢复中和回撤持续之间的频繁震荡

        Args:
            old_phase: 旧阶段
            new_phase: 新阶段
        """
        # 定义重要的阶段转换，总是记录
        critical_transitions = {
            (DrawdownPhase.NORMAL, DrawdownPhase.DRAWDOWN_START),
            (DrawdownPhase.RECOVERY, DrawdownPhase.NORMAL),
            (DrawdownPhase.DRAWDOWN_START, DrawdownPhase.DRAWDOWN_CONTINUE),
        }

        # 定义噪音转换，需要去抖动
        noisy_transitions = {
            (DrawdownPhase.RECOVERY, DrawdownPhase.DRAWDOWN_CONTINUE),
            (DrawdownPhase.DRAWDOWN_CONTINUE, DrawdownPhase.RECOVERY),
        }

        transition = (old_phase, new_phase)

        if transition in critical_transitions:
            # 重要转换，立即记录
            logger.debug(f"回撤阶段变化: {old_phase.value} -> {new_phase.value}")
            self.last_logged_phase = new_phase
            self.phase_stability_count = 0

        elif transition in noisy_transitions:
            # 噪音转换，使用去抖动机制 - 暂时不记录，等待稳定
            if new_phase != self.last_logged_phase:
                self.phase_stability_count = 1
            else:
                self.phase_stability_count += 1

            # 只有当状态稳定足够次数且与上次记录不同时才记录
            if (self.phase_stability_count >= self.phase_stability_threshold and
                new_phase != self.last_logged_phase):
                logger.debug(f"回撤阶段变化: {self.last_logged_phase.value} -> {new_phase.value}")
                self.last_logged_phase = new_phase
                self.phase_stability_count = 0

        else:
            # 其他转换，使用默认逻辑
            if new_phase != self.last_logged_phase:
                logger.debug(f"回撤阶段变化: {old_phase.value} -> {new_phase.value}")
                self.last_logged_phase = new_phase
                self.phase_stability_count = 0

    def _create_empty_metrics(self) -> DrawdownMetrics:
        """创建空的回撤指标"""
        return DrawdownMetrics(
            current_drawdown=0.0,
            max_drawdown=0.0,
            drawdown_duration=0,
            recovery_time=None,
            peak_value=0.0,
            trough_value=0.0,
            underwater_curve=[],
            drawdown_frequency=0.0,
            average_drawdown=0.0,
            current_phase=DrawdownPhase.NORMAL,
            days_since_peak=0
        )

    def detect_market_regime(self, market_data: Dict[str, np.ndarray]) -> MarketStateMetrics:
        """
        检测市场状态

        Args:
            market_data: 市场数据字典，包含价格、成交量等信息
                - 'prices': 价格序列
                - 'volumes': 成交量序列（可选）
                - 'returns': 收益率序列（可选）

        Returns:
            MarketStateMetrics: 市场状态指标
        """
        if 'prices' not in market_data or len(market_data['prices']) < self.volatility_window:
            return self._create_default_market_state()

        prices = market_data['prices']
        returns = market_data.get('returns')

        # 如果没有提供收益率，则计算
        if returns is None:
            returns = np.diff(prices) / prices[:-1]

        # 计算波动率
        volatility = self._calculate_volatility(returns)

        # 计算趋势强度
        trend_strength = self._calculate_trend_strength(prices)

        # 计算相关性水平（如果有多个资产）
        correlation_level = self._calculate_correlation_level(market_data)

        # 计算流动性评分
        liquidity_score = self._calculate_liquidity_score(market_data)

        # 识别市场状态
        regime, confidence = self._identify_market_regime(
            volatility, trend_strength, correlation_level, returns
        )

        return MarketStateMetrics(
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            correlation_level=correlation_level,
            liquidity_score=liquidity_score,
            confidence_score=confidence
        )

    def _calculate_volatility(self, returns: np.ndarray) -> float:
        """计算年化波动率"""
        if len(returns) < 2:
            return 0.0

        # 使用最近的数据计算波动率
        recent_returns = returns[-self.volatility_window:] if len(returns) > self.volatility_window else returns
        volatility = np.std(recent_returns) * np.sqrt(252)  # 年化

        return float(volatility)

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """
        计算趋势强度
        使用线性回归的R²值来衡量趋势强度
        """
        if len(prices) < 10:
            return 0.0

        # 使用最近的价格数据
        recent_prices = prices[-min(60, len(prices)):]  # 最近60个数据点
        x = np.arange(len(recent_prices))

        # 线性回归
        coeffs = np.polyfit(x, recent_prices, 1)
        trend_line = np.polyval(coeffs, x)

        # 计算R²值
        ss_res = np.sum((recent_prices - trend_line) ** 2)
        ss_tot = np.sum((recent_prices - np.mean(recent_prices)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 趋势方向（正为上涨，负为下跌）
        trend_direction = 1 if coeffs[0] > 0 else -1

        return float(r_squared * trend_direction)

    def _calculate_correlation_level(self, market_data: Dict[str, np.ndarray]) -> float:
        """
        计算相关性水平
        如果只有单一资产，返回与历史自身的相关性
        """
        if 'prices' not in market_data:
            return 0.0

        prices = market_data['prices']

        if len(prices) < 20:
            return 0.0

        # 计算滞后相关性（与自身的滞后相关性）
        returns = np.diff(prices) / prices[:-1]

        if len(returns) < 10:
            return 0.0

        # 计算1日滞后相关性
        if len(returns) > 1:
            corr = np.corrcoef(returns[1:], returns[:-1])[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0

        return 0.0

    def _calculate_liquidity_score(self, market_data: Dict[str, np.ndarray]) -> float:
        """
        计算流动性评分
        基于成交量和价格波动的关系
        """
        if 'volumes' not in market_data or 'prices' not in market_data:
            return 0.5  # 默认中等流动性

        volumes = market_data['volumes']
        prices = market_data['prices']

        if len(volumes) < 10 or len(prices) < 10:
            return 0.5

        # 计算成交量的变异系数（CV）
        recent_volumes = volumes[-min(20, len(volumes)):]
        volume_cv = np.std(recent_volumes) / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 1.0

        # 计算价格波动与成交量的关系
        returns = np.diff(prices) / prices[:-1]
        recent_returns = returns[-min(20, len(returns)):]

        if len(recent_returns) == len(recent_volumes):
            # 流动性评分：成交量越稳定，价格波动越小，流动性越好
            price_volatility = np.std(recent_returns)
            liquidity_score = 1.0 / (1.0 + volume_cv + price_volatility * 10)
        else:
            liquidity_score = 1.0 / (1.0 + volume_cv)

        return float(np.clip(liquidity_score, 0.0, 1.0))

    def _identify_market_regime(self, volatility: float, trend_strength: float,
                              correlation_level: float, returns: np.ndarray) -> Tuple[MarketRegime, float]:
        """
        识别市场状态

        Returns:
            Tuple[MarketRegime, float]: (市场状态, 置信度)
        """
        confidence_scores = {}

        # 波动率阈值
        high_vol_threshold = 0.25  # 25%年化波动率
        low_vol_threshold = 0.10   # 10%年化波动率

        # 趋势强度阈值
        strong_trend_threshold = 0.6
        weak_trend_threshold = 0.2

        # 基于波动率判断
        if volatility > high_vol_threshold:
            confidence_scores[MarketRegime.HIGH_VOLATILITY] = min(volatility / high_vol_threshold, 2.0)

            # 高波动期间，进一步判断是否为危机
            recent_returns = returns[-min(10, len(returns)):]
            if len(recent_returns) > 0 and np.mean(recent_returns) < -0.02:  # 连续大幅下跌
                confidence_scores[MarketRegime.CRISIS] = confidence_scores[MarketRegime.HIGH_VOLATILITY] * 1.5

        elif volatility < low_vol_threshold:
            confidence_scores[MarketRegime.LOW_VOLATILITY] = low_vol_threshold / max(volatility, 0.01)

        # 基于趋势强度判断
        abs_trend = abs(trend_strength)
        if abs_trend > strong_trend_threshold:
            if trend_strength > 0:
                confidence_scores[MarketRegime.BULL_MARKET] = abs_trend
            else:
                confidence_scores[MarketRegime.BEAR_MARKET] = abs_trend
        elif abs_trend < weak_trend_threshold:
            confidence_scores[MarketRegime.SIDEWAYS_MARKET] = 1.0 - abs_trend

        # 如果没有明确的状态，默认为震荡市
        if not confidence_scores:
            confidence_scores[MarketRegime.SIDEWAYS_MARKET] = 0.5

        # 选择置信度最高的状态
        best_regime = max(confidence_scores.keys(), key=lambda x: confidence_scores[x])
        best_confidence = confidence_scores[best_regime]

        # 标准化置信度到0-1范围
        normalized_confidence = min(best_confidence, 1.0)

        return best_regime, normalized_confidence

    def _create_default_market_state(self) -> MarketStateMetrics:
        """创建默认的市场状态"""
        return MarketStateMetrics(
            regime=MarketRegime.SIDEWAYS_MARKET,
            volatility=0.15,
            trend_strength=0.0,
            correlation_level=0.0,
            liquidity_score=0.5,
            confidence_score=0.3
        )

    def analyze_drawdown_attribution(self, positions: Dict[str, float],
                                   position_returns: Dict[str, float]) -> Dict[str, float]:
        """
        分析回撤归因

        Args:
            positions: 各资产的持仓权重
            position_returns: 各资产的收益率

        Returns:
            Dict[str, float]: 各资产对回撤的贡献（归一化到总和为1）
        """
        if not positions or not position_returns:
            return {}

        # 计算各资产的负贡献
        negative_contributions = {}
        total_negative_contribution = 0.0

        for asset in positions:
            if asset in position_returns:
                contribution = positions[asset] * position_returns[asset]
                # 只考虑负贡献（亏损）
                if contribution < 0:
                    negative_contributions[asset] = abs(contribution)
                    total_negative_contribution += abs(contribution)
                else:
                    negative_contributions[asset] = 0.0

        # 归一化贡献度
        if total_negative_contribution > 0:
            contributions = {asset: contrib / total_negative_contribution
                           for asset, contrib in negative_contributions.items()}
        else:
            contributions = {asset: 0.0 for asset in positions}

        return contributions

    def get_current_status(self) -> Dict[str, Any]:
        """获取当前监控状态的完整信息"""
        if not self.portfolio_values:
            return {
                'status': '无数据',
                'portfolio_value': 0.0,
                'drawdown_metrics': self._create_empty_metrics().to_dict(),
                'monitoring_duration': 0
            }

        current_metrics = self._calculate_drawdown_metrics()
        monitoring_duration = len(self.portfolio_values)

        return {
            'status': '正常监控',
            'portfolio_value': self.portfolio_values[-1],
            'drawdown_metrics': current_metrics.to_dict(),
            'monitoring_duration': monitoring_duration,
            'last_update': self.timestamps[-1].isoformat() if self.timestamps else None
        }

    def reset(self):
        """重置监控器状态"""
        self.portfolio_values.clear()
        self.timestamps.clear()
        self.drawdown_history.clear()
        self.peak_history.clear()

        self.current_peak = 0.0
        self.current_trough = float('inf')
        self.peak_timestamp = None
        self.trough_timestamp = None
        self.current_phase = DrawdownPhase.NORMAL
        self.phase_start_time = None

        logger.info("回撤监控器已重置")