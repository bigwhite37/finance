"""
市场状态感知系统
实现基于技术指标、波动率和相关性的市场状态识别和分类
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

from ..data.data_models import MarketData


class MarketRegime(Enum):
    """市场状态枚举"""
    BULL_MARKET = "bull"                       # 牛市
    BEAR_MARKET = "bear"                       # 熊市
    SIDEWAYS_MARKET = "sideways"               # 震荡市
    HIGH_VOLATILITY = "high_vol"               # 高波动期
    LOW_VOLATILITY = "low_vol"                 # 低波动期
    CRISIS = "crisis"                          # 危机期


@dataclass
class MarketRegimeConfig:
    """市场状态检测配置"""
    # 技术指标参数
    ma_short_period: int = 20                  # 短期均线周期
    ma_long_period: int = 60                   # 长期均线周期
    rsi_period: int = 14                       # RSI周期
    bollinger_period: int = 20                 # 布林带周期
    bollinger_std: float = 2.0                 # 布林带标准差倍数
    
    # 波动率参数
    volatility_window: int = 20                # 波动率计算窗口
    high_vol_threshold: float = 0.02           # 高波动率阈值
    low_vol_threshold: float = 0.01            # 低波动率阈值
    
    # 趋势判断参数
    trend_strength_threshold: float = 0.6      # 趋势强度阈值
    sideways_threshold: float = 0.3            # 震荡市阈值
    
    # 相关性参数
    correlation_window: int = 30               # 相关性计算窗口
    high_correlation_threshold: float = 0.7    # 高相关性阈值
    
    # 危机检测参数
    crisis_drawdown_threshold: float = 0.15    # 危机回撤阈值
    crisis_volatility_multiplier: float = 3.0  # 危机波动率倍数
    
    # 状态切换参数
    regime_persistence: int = 5                # 状态持续性要求
    confidence_threshold: float = 0.7          # 置信度阈值


@dataclass
class MarketIndicators:
    """市场指标结构"""
    timestamp: datetime
    
    # 价格指标
    price: float
    ma_short: float
    ma_long: float
    price_position: float                      # 价格在布林带中的位置
    
    # 技术指标
    rsi: float
    bollinger_upper: float
    bollinger_lower: float
    bollinger_width: float
    
    # 波动率指标
    volatility: float
    volatility_percentile: float
    
    # 趋势指标
    trend_strength: float
    trend_direction: int                       # 1: 上涨, -1: 下跌, 0: 震荡
    
    # 市场结构指标
    correlation_level: float
    market_stress: float


@dataclass
class RegimeDetectionResult:
    """市场状态检测结果"""
    timestamp: datetime
    regime: MarketRegime
    confidence: float
    indicators: MarketIndicators
    regime_probabilities: Dict[MarketRegime, float]
    risk_adjustment_factor: float              # 风险调整因子
    recommended_actions: List[str]             # 推荐行动


class MarketRegimeDetector:
    """市场状态感知器"""
    
    def __init__(self, config: Optional[MarketRegimeConfig] = None):
        """初始化市场状态检测器"""
        self.config = config or MarketRegimeConfig()
        self.logger = logging.getLogger(__name__)
        
        # 历史数据存储
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        self.timestamp_history: List[datetime] = []
        
        # 指标历史
        self.indicators_history: List[MarketIndicators] = []
        
        # 当前状态
        self.current_regime: Optional[MarketRegime] = None
        self.regime_start_time: Optional[datetime] = None
        self.regime_confidence: float = 0.0
        
        # 状态持续性计数
        self.regime_persistence_count: Dict[MarketRegime, int] = {
            regime: 0 for regime in MarketRegime
        }
        
        # 风险参数调整历史
        self.risk_adjustments: List[Dict[str, Any]] = []
        
        self.logger.info("市场状态检测器初始化完成")
    
    def update_market_data(self, market_data: MarketData) -> RegimeDetectionResult:
        """更新市场数据并检测状态"""
        try:
            # 更新历史数据
            self._update_history(market_data)
            
            # 计算市场指标
            indicators = self._calculate_indicators()
            
            if indicators is None:
                # 数据不足，返回默认状态
                default_indicators = self._get_default_indicators(market_data)
                self.indicators_history.append(default_indicators)
                return RegimeDetectionResult(
                    timestamp=market_data.timestamp,
                    regime=MarketRegime.SIDEWAYS_MARKET,
                    confidence=0.0,
                    indicators=default_indicators,
                    regime_probabilities={regime: 1.0/len(MarketRegime) for regime in MarketRegime},
                    risk_adjustment_factor=1.0,
                    recommended_actions=[]
                )
            
            # 检测市场状态
            regime_probs = self._detect_regime(indicators)
            
            # 确定最可能的状态
            best_regime = max(regime_probs.items(), key=lambda x: x[1])
            regime, confidence = best_regime
            
            # 检查状态持续性
            regime = self._check_regime_persistence(regime, confidence)
            
            # 计算风险调整因子
            risk_factor = self._calculate_risk_adjustment_factor(regime, indicators)
            
            # 生成推荐行动
            actions = self._generate_recommended_actions(regime, indicators)
            
            # 更新当前状态
            self._update_current_regime(regime, market_data.timestamp, confidence)
            
            # 添加指标到历史记录
            self.indicators_history.append(indicators)
            
            # 保持历史记录长度
            if len(self.indicators_history) > 200:
                self.indicators_history = self.indicators_history[-200:]
            
            result = RegimeDetectionResult(
                timestamp=market_data.timestamp,
                regime=regime,
                confidence=confidence,
                indicators=indicators,
                regime_probabilities=regime_probs,
                risk_adjustment_factor=risk_factor,
                recommended_actions=actions
            )
            
            self.logger.debug(f"检测到市场状态: {regime.value}, 置信度: {confidence:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"市场状态检测失败: {e}")
            raise
    
    def _update_history(self, market_data: MarketData):
        """更新历史数据"""
        self.price_history.append(market_data.close_price)
        self.volume_history.append(market_data.volume)
        self.timestamp_history.append(market_data.timestamp)
        
        # 保持历史数据长度
        max_history = max(self.config.ma_long_period * 2, 100)
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.timestamp_history = self.timestamp_history[-max_history:]
    
    def _calculate_indicators(self) -> Optional[MarketIndicators]:
        """计算市场指标"""
        if len(self.price_history) < self.config.ma_long_period:
            return None
        
        prices = np.array(self.price_history)
        current_price = prices[-1]
        current_time = self.timestamp_history[-1]
        
        # 计算移动平均线
        ma_short = np.mean(prices[-self.config.ma_short_period:])
        ma_long = np.mean(prices[-self.config.ma_long_period:])
        
        # 计算RSI
        rsi = self._calculate_rsi(prices, self.config.rsi_period)
        
        # 计算布林带
        bollinger_upper, bollinger_lower, bollinger_width = self._calculate_bollinger_bands(
            prices, self.config.bollinger_period, self.config.bollinger_std
        )
        
        # 价格在布林带中的位置
        if bollinger_width > 0:
            price_position = (current_price - bollinger_lower) / bollinger_width
        else:
            price_position = 0.5
        
        # 计算波动率
        volatility = self._calculate_volatility(prices, self.config.volatility_window)
        volatility_percentile = self._calculate_volatility_percentile(volatility)
        
        # 计算趋势指标
        trend_strength, trend_direction = self._calculate_trend_indicators(prices)
        
        # 计算相关性水平（简化版，使用价格自相关）
        correlation_level = self._calculate_correlation_level(prices)
        
        # 计算市场压力指标
        market_stress = self._calculate_market_stress(prices, volatility)
        
        return MarketIndicators(
            timestamp=current_time,
            price=current_price,
            ma_short=ma_short,
            ma_long=ma_long,
            price_position=price_position,
            rsi=rsi,
            bollinger_upper=bollinger_upper,
            bollinger_lower=bollinger_lower,
            bollinger_width=bollinger_width,
            volatility=volatility,
            volatility_percentile=volatility_percentile,
            trend_strength=trend_strength,
            trend_direction=trend_direction,
            correlation_level=correlation_level,
            market_stress=market_stress
        )
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """计算RSI指标"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int, std_mult: float) -> Tuple[float, float, float]:
        """计算布林带"""
        if len(prices) < period:
            current_price = prices[-1]
            return current_price, current_price, 0.0
        
        ma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = ma + std_mult * std
        lower = ma - std_mult * std
        width = upper - lower
        
        return upper, lower, width
    
    def _calculate_volatility(self, prices: np.ndarray, window: int) -> float:
        """计算波动率"""
        if len(prices) < window + 1:
            return 0.01
        
        returns = np.diff(np.log(prices[-window-1:]))
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        return volatility
    
    def _calculate_volatility_percentile(self, current_vol: float) -> float:
        """计算波动率百分位"""
        if len(self.indicators_history) < 20:
            return 0.5
        
        historical_vols = [ind.volatility for ind in self.indicators_history[-60:]]
        percentile = stats.percentileofscore(historical_vols, current_vol) / 100.0
        
        return percentile
    
    def _calculate_trend_indicators(self, prices: np.ndarray) -> Tuple[float, int]:
        """计算趋势指标"""
        if len(prices) < 10:  # 降低最小数据要求
            return 0.0, 0
        
        # 使用线性回归计算趋势强度
        window_size = min(20, len(prices))
        x = np.arange(window_size)
        y = prices[-window_size:]
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # 趋势强度为R²
        trend_strength = r_value ** 2
        
        # 降低趋势判断阈值，使其更容易检测到趋势
        trend_threshold = max(0.3, self.config.trend_strength_threshold * 0.5)
        
        # 趋势方向 - 同时考虑斜率的绝对值
        if slope > 0 and (trend_strength > trend_threshold or abs(slope) > 0.5):
            trend_direction = 1  # 上涨趋势
        elif slope < 0 and (trend_strength > trend_threshold or abs(slope) > 0.5):
            trend_direction = -1  # 下跌趋势
        else:
            trend_direction = 0  # 震荡
        
        return trend_strength, trend_direction
    
    def _calculate_correlation_level(self, prices: np.ndarray) -> float:
        """计算相关性水平（简化版）"""
        if len(prices) < self.config.correlation_window:
            return 0.5
        
        # 使用价格序列的自相关作为相关性代理
        returns = np.diff(np.log(prices[-self.config.correlation_window:]))
        
        if len(returns) < 2:
            return 0.5
        
        # 计算1阶自相关
        correlation = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        if np.isnan(correlation):
            correlation = 0.0
        
        return abs(correlation)
    
    def _calculate_market_stress(self, prices: np.ndarray, volatility: float) -> float:
        """计算市场压力指标"""
        if len(prices) < 5:  # 降低最小数据要求
            return 0.0
        
        # 基于价格下跌幅度和波动率计算压力
        window_size = min(10, len(prices))
        recent_returns = np.diff(np.log(prices[-window_size:]))
        negative_returns = recent_returns[recent_returns < 0]
        
        if len(negative_returns) == 0:
            return 0.0
        
        # 计算累计下跌幅度
        cumulative_decline = abs(np.sum(negative_returns))
        
        # 计算连续下跌天数比例
        consecutive_declines = 0
        for ret in recent_returns[::-1]:  # 从最近开始倒序
            if ret < 0:
                consecutive_declines += 1
            else:
                break
        decline_ratio = consecutive_declines / len(recent_returns)
        
        # 综合压力指标
        stress = (abs(np.mean(negative_returns)) * volatility * 5 + 
                 cumulative_decline * 2 + 
                 decline_ratio * 0.5)
        
        return min(stress, 1.0)  # 限制在[0, 1]范围内   
 
    def _detect_regime(self, indicators: MarketIndicators) -> Dict[MarketRegime, float]:
        """检测市场状态概率"""
        probabilities = {}
        
        # 牛市概率
        bull_score = 0.0
        if indicators.trend_direction == 1:
            bull_score += 0.5 * indicators.trend_strength  # 基于趋势强度调整
        if indicators.ma_short > indicators.ma_long:
            bull_score += 0.3
        if indicators.rsi > 55:  # 稍微提高阈值
            bull_score += 0.2
        if indicators.price_position > 0.6:  # 降低阈值使其更容易触发
            bull_score += 0.2
        
        probabilities[MarketRegime.BULL_MARKET] = min(bull_score, 1.0)
        
        # 熊市概率
        bear_score = 0.0
        if indicators.trend_direction == -1:
            bear_score += 0.5 * indicators.trend_strength  # 基于趋势强度调整
        if indicators.ma_short < indicators.ma_long:
            bear_score += 0.3
        if indicators.rsi < 45:  # 稍微降低阈值
            bear_score += 0.2
        if indicators.price_position < 0.4:  # 提高阈值使其更容易触发
            bear_score += 0.2
        
        probabilities[MarketRegime.BEAR_MARKET] = min(bear_score, 1.0)
        
        # 震荡市概率
        sideways_score = 0.0
        if indicators.trend_direction == 0:
            sideways_score += 0.5
        if 0.3 <= indicators.price_position <= 0.7:
            sideways_score += 0.3
        if 40 <= indicators.rsi <= 60:
            sideways_score += 0.2
        
        probabilities[MarketRegime.SIDEWAYS_MARKET] = min(sideways_score, 1.0)
        
        # 高波动率概率
        high_vol_score = 0.0
        if indicators.volatility > self.config.high_vol_threshold:
            high_vol_score += 0.6
        if indicators.volatility_percentile > 0.8:
            high_vol_score += 0.4
        
        probabilities[MarketRegime.HIGH_VOLATILITY] = min(high_vol_score, 1.0)
        
        # 低波动率概率
        low_vol_score = 0.0
        if indicators.volatility < self.config.low_vol_threshold:
            low_vol_score += 0.6
        if indicators.volatility_percentile < 0.2:
            low_vol_score += 0.4
        
        probabilities[MarketRegime.LOW_VOLATILITY] = min(low_vol_score, 1.0)
        
        # 危机概率 - 降低阈值使其更容易触发
        crisis_score = 0.0
        if indicators.market_stress > 0.4:  # 降低阈值
            crisis_score += 0.4
        if indicators.volatility > self.config.high_vol_threshold * 1.5:  # 降低倍数
            crisis_score += 0.3
        if indicators.trend_direction == -1 and indicators.trend_strength > 0.4:  # 降低阈值
            crisis_score += 0.4
        # 如果价格大幅下跌也算危机
        if indicators.price_position < 0.2:
            crisis_score += 0.3
        
        probabilities[MarketRegime.CRISIS] = min(crisis_score, 1.0)
        
        # 标准化概率
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v / total_prob for k, v in probabilities.items()}
        else:
            # 如果所有概率都为0，设置为均匀分布
            probabilities = {regime: 1.0 / len(MarketRegime) for regime in MarketRegime}
        
        return probabilities
    
    def _check_regime_persistence(self, detected_regime: MarketRegime, confidence: float) -> MarketRegime:
        """检查状态持续性"""
        # 降低置信度要求
        confidence_threshold = max(0.3, self.config.confidence_threshold * 0.5)
        
        if confidence < confidence_threshold:
            # 置信度不足，保持当前状态
            if self.current_regime is not None:
                return self.current_regime
        
        # 更新持续性计数
        for regime in MarketRegime:
            if regime == detected_regime:
                self.regime_persistence_count[regime] += 1
            else:
                self.regime_persistence_count[regime] = 0
        
        # 降低持续性要求
        persistence_requirement = max(1, self.config.regime_persistence // 2)
        
        # 检查是否满足持续性要求
        if self.regime_persistence_count[detected_regime] >= persistence_requirement:
            return detected_regime
        elif self.current_regime is not None:
            return self.current_regime
        else:
            return detected_regime
    
    def _calculate_risk_adjustment_factor(self, regime: MarketRegime, indicators: MarketIndicators) -> float:
        """计算风险调整因子"""
        base_factor = 1.0
        
        # 根据市场状态调整
        regime_adjustments = {
            MarketRegime.BULL_MARKET: 1.1,      # 牛市适度增加风险
            MarketRegime.BEAR_MARKET: 0.7,      # 熊市降低风险
            MarketRegime.SIDEWAYS_MARKET: 0.9,  # 震荡市略微降低风险
            MarketRegime.HIGH_VOLATILITY: 0.6,  # 高波动大幅降低风险
            MarketRegime.LOW_VOLATILITY: 1.2,   # 低波动增加风险
            MarketRegime.CRISIS: 0.3,           # 危机期大幅降低风险
        }
        
        regime_factor = regime_adjustments.get(regime, 1.0)
        
        # 根据波动率微调
        vol_adjustment = 1.0 - (indicators.volatility - 0.01) * 2.0
        vol_adjustment = max(0.3, min(1.5, vol_adjustment))
        
        # 根据趋势强度微调
        trend_adjustment = 1.0
        if indicators.trend_direction == -1:  # 下跌趋势
            trend_adjustment = 1.0 - indicators.trend_strength * 0.3
        elif indicators.trend_direction == 1:  # 上涨趋势
            trend_adjustment = 1.0 + indicators.trend_strength * 0.1
        
        # 综合调整因子
        final_factor = base_factor * regime_factor * vol_adjustment * trend_adjustment
        
        # 限制在合理范围内
        return max(0.1, min(2.0, final_factor))
    
    def _generate_recommended_actions(self, regime: MarketRegime, indicators: MarketIndicators) -> List[str]:
        """生成推荐行动"""
        actions = []
        
        if regime == MarketRegime.BULL_MARKET:
            actions.extend([
                "适度增加仓位",
                "关注成长股机会",
                "设置追踪止损"
            ])
        elif regime == MarketRegime.BEAR_MARKET:
            actions.extend([
                "降低整体仓位",
                "加强止损管理",
                "关注防御性资产"
            ])
        elif regime == MarketRegime.SIDEWAYS_MARKET:
            actions.extend([
                "保持中性仓位",
                "增加交易频率",
                "关注区间操作机会"
            ])
        elif regime == MarketRegime.HIGH_VOLATILITY:
            actions.extend([
                "大幅降低仓位",
                "缩短持仓周期",
                "加强风险监控"
            ])
        elif regime == MarketRegime.LOW_VOLATILITY:
            actions.extend([
                "可适度增加杠杆",
                "延长持仓周期",
                "关注套利机会"
            ])
        elif regime == MarketRegime.CRISIS:
            actions.extend([
                "紧急降低仓位",
                "激活危机应对预案",
                "暂停新增投资",
                "加强流动性管理"
            ])
        
        # 根据具体指标添加额外建议
        if indicators.rsi > 80:
            actions.append("市场超买，考虑减仓")
        elif indicators.rsi < 20:
            actions.append("市场超卖，关注反弹机会")
        
        if indicators.volatility_percentile > 0.9:
            actions.append("波动率极高，暂停交易")
        
        if indicators.market_stress > 0.8:
            actions.append("市场压力极大，执行应急预案")
        
        return actions
    
    def _update_current_regime(self, regime: MarketRegime, timestamp: datetime, confidence: float):
        """更新当前状态"""
        if self.current_regime != regime:
            self.logger.info(f"市场状态切换: {self.current_regime} -> {regime}")
            self.current_regime = regime
            self.regime_start_time = timestamp
        
        self.regime_confidence = confidence
    
    def _get_default_indicators(self, market_data: MarketData) -> MarketIndicators:
        """获取默认指标（数据不足时使用）"""
        return MarketIndicators(
            timestamp=market_data.timestamp,
            price=market_data.close_price,
            ma_short=market_data.close_price,
            ma_long=market_data.close_price,
            price_position=0.5,
            rsi=50.0,
            bollinger_upper=market_data.close_price * 1.02,
            bollinger_lower=market_data.close_price * 0.98,
            bollinger_width=market_data.close_price * 0.04,
            volatility=0.01,
            volatility_percentile=0.5,
            trend_strength=0.0,
            trend_direction=0,
            correlation_level=0.5,
            market_stress=0.0
        )
    
    def get_current_regime(self) -> Optional[MarketRegime]:
        """获取当前市场状态"""
        return self.current_regime
    
    def get_regime_duration(self) -> Optional[timedelta]:
        """获取当前状态持续时间"""
        if self.regime_start_time is None:
            return None
        
        if self.timestamp_history:
            current_time = self.timestamp_history[-1]
            return current_time - self.regime_start_time
        
        return None
    
    def get_regime_history(self, days: int = 30) -> List[Tuple[datetime, MarketRegime, float]]:
        """获取历史状态记录"""
        if not self.indicators_history:
            return []
        
        cutoff_date = self.timestamp_history[-1] - timedelta(days=days)
        
        history = []
        for i, indicators in enumerate(self.indicators_history):
            if indicators.timestamp >= cutoff_date:
                # 重新计算该时点的状态（简化版）
                regime_probs = self._detect_regime(indicators)
                best_regime = max(regime_probs.items(), key=lambda x: x[1])
                regime, confidence = best_regime
                
                history.append((indicators.timestamp, regime, confidence))
        
        return history
    
    def adjust_risk_parameters(self, current_regime: MarketRegime, 
                             base_params: Dict[str, float]) -> Dict[str, float]:
        """根据市场状态调整风险参数"""
        adjusted_params = base_params.copy()
        
        # 获取风险调整因子
        if self.indicators_history:
            latest_indicators = self.indicators_history[-1]
            risk_factor = self._calculate_risk_adjustment_factor(current_regime, latest_indicators)
        else:
            risk_factor = 1.0
        
        # 调整各项风险参数
        if 'max_position_size' in adjusted_params:
            adjusted_params['max_position_size'] *= risk_factor
        
        if 'stop_loss_threshold' in adjusted_params:
            # 高风险时收紧止损
            if risk_factor < 0.8:
                adjusted_params['stop_loss_threshold'] *= 0.8
            elif risk_factor > 1.2:
                adjusted_params['stop_loss_threshold'] *= 1.2
        
        if 'volatility_target' in adjusted_params:
            adjusted_params['volatility_target'] *= (2.0 - risk_factor)
        
        # 记录调整历史
        adjustment_record = {
            'timestamp': datetime.now(),
            'regime': current_regime,
            'risk_factor': risk_factor,
            'original_params': base_params,
            'adjusted_params': adjusted_params
        }
        self.risk_adjustments.append(adjustment_record)
        
        # 保持历史记录长度
        if len(self.risk_adjustments) > 100:
            self.risk_adjustments = self.risk_adjustments[-100:]
        
        self.logger.info(f"风险参数已调整，调整因子: {risk_factor:.3f}")
        
        return adjusted_params
    
    def get_market_stress_level(self) -> float:
        """获取当前市场压力水平"""
        if not self.indicators_history:
            return 0.0
        
        return self.indicators_history[-1].market_stress
    
    def is_crisis_mode(self) -> bool:
        """判断是否处于危机模式"""
        return self.current_regime == MarketRegime.CRISIS
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """获取状态统计信息"""
        if not self.indicators_history:
            return {}
        
        # 计算各状态的出现频率
        regime_counts = {regime: 0 for regime in MarketRegime}
        
        for indicators in self.indicators_history[-100:]:  # 最近100个观测
            regime_probs = self._detect_regime(indicators)
            best_regime = max(regime_probs.items(), key=lambda x: x[1])[0]
            regime_counts[best_regime] += 1
        
        total_count = sum(regime_counts.values())
        regime_frequencies = {
            regime.value: count / total_count if total_count > 0 else 0
            for regime, count in regime_counts.items()
        }
        
        # 计算平均指标
        recent_indicators = self.indicators_history[-20:] if len(self.indicators_history) >= 20 else self.indicators_history
        if recent_indicators:
            avg_volatility = np.mean([ind.volatility for ind in recent_indicators])
            avg_trend_strength = np.mean([ind.trend_strength for ind in recent_indicators])
            avg_market_stress = np.mean([ind.market_stress for ind in recent_indicators])
        else:
            avg_volatility = 0.0
            avg_trend_strength = 0.0
            avg_market_stress = 0.0
        
        return {
            'current_regime': self.current_regime.value if self.current_regime else None,
            'regime_confidence': self.regime_confidence,
            'regime_duration_days': self.get_regime_duration().days if self.get_regime_duration() else 0,
            'regime_frequencies': regime_frequencies,
            'avg_volatility': avg_volatility,
            'avg_trend_strength': avg_trend_strength,
            'avg_market_stress': avg_market_stress,
            'total_observations': len(self.indicators_history)
        }
    
    def reset(self):
        """重置检测器状态"""
        self.price_history.clear()
        self.volume_history.clear()
        self.timestamp_history.clear()
        self.indicators_history.clear()
        
        self.current_regime = None
        self.regime_start_time = None
        self.regime_confidence = 0.0
        
        self.regime_persistence_count = {regime: 0 for regime in MarketRegime}
        self.risk_adjustments.clear()
        
        self.logger.info("市场状态检测器已重置")


class MarketRegimeAnalyzer:
    """市场状态分析器 - 用于回测和分析"""
    
    def __init__(self, detector: MarketRegimeDetector):
        """初始化分析器"""
        self.detector = detector
        self.logger = logging.getLogger(__name__)
    
    def analyze_historical_regimes(self, market_data_list: List[MarketData]) -> pd.DataFrame:
        """分析历史市场状态"""
        results = []
        
        for market_data in market_data_list:
            try:
                result = self.detector.update_market_data(market_data)
                
                results.append({
                    'timestamp': result.timestamp,
                    'regime': result.regime.value,
                    'confidence': result.confidence,
                    'risk_adjustment_factor': result.risk_adjustment_factor,
                    'price': result.indicators.price,
                    'volatility': result.indicators.volatility,
                    'trend_strength': result.indicators.trend_strength,
                    'trend_direction': result.indicators.trend_direction,
                    'market_stress': result.indicators.market_stress,
                    'rsi': result.indicators.rsi
                })
                
            except Exception as e:
                self.logger.error(f"分析历史数据失败: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def evaluate_regime_accuracy(self, historical_df: pd.DataFrame, 
                                actual_regimes: List[MarketRegime]) -> Dict[str, float]:
        """评估状态识别准确性"""
        if len(historical_df) != len(actual_regimes):
            raise ValueError("历史数据长度与实际状态长度不匹配")
        
        predicted_regimes = [MarketRegime(regime) for regime in historical_df['regime']]
        
        # 计算准确率
        correct_predictions = sum(1 for pred, actual in zip(predicted_regimes, actual_regimes) 
                                if pred == actual)
        accuracy = correct_predictions / len(actual_regimes)
        
        # 计算各状态的精确率和召回率
        regime_metrics = {}
        for regime in MarketRegime:
            true_positives = sum(1 for pred, actual in zip(predicted_regimes, actual_regimes)
                               if pred == regime and actual == regime)
            false_positives = sum(1 for pred, actual in zip(predicted_regimes, actual_regimes)
                                if pred == regime and actual != regime)
            false_negatives = sum(1 for pred, actual in zip(predicted_regimes, actual_regimes)
                                if pred != regime and actual == regime)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            regime_metrics[regime.value] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
        
        return {
            'overall_accuracy': accuracy,
            'regime_metrics': regime_metrics
        }
    
    def generate_regime_report(self, historical_df: pd.DataFrame) -> str:
        """生成市场状态分析报告"""
        if historical_df.empty:
            return "无历史数据可供分析"
        
        report_lines = []
        report_lines.append("=== 市场状态分析报告 ===\n")
        
        # 基本统计
        total_days = len(historical_df)
        regime_counts = historical_df['regime'].value_counts()
        
        report_lines.append(f"分析期间: {historical_df['timestamp'].min()} 至 {historical_df['timestamp'].max()}")
        report_lines.append(f"总天数: {total_days}")
        report_lines.append("")
        
        # 各状态占比
        report_lines.append("各市场状态占比:")
        for regime, count in regime_counts.items():
            percentage = count / total_days * 100
            report_lines.append(f"  {regime}: {count}天 ({percentage:.1f}%)")
        report_lines.append("")
        
        # 平均指标
        avg_volatility = historical_df['volatility'].mean()
        avg_confidence = historical_df['confidence'].mean()
        avg_risk_factor = historical_df['risk_adjustment_factor'].mean()
        
        report_lines.append("平均指标:")
        report_lines.append(f"  平均波动率: {avg_volatility:.4f}")
        report_lines.append(f"  平均置信度: {avg_confidence:.3f}")
        report_lines.append(f"  平均风险调整因子: {avg_risk_factor:.3f}")
        report_lines.append("")
        
        # 状态转换分析
        regime_transitions = self._analyze_regime_transitions(historical_df)
        report_lines.append("状态转换分析:")
        for transition, count in regime_transitions.items():
            report_lines.append(f"  {transition}: {count}次")
        
        return "\n".join(report_lines)
    
    def _analyze_regime_transitions(self, df: pd.DataFrame) -> Dict[str, int]:
        """分析状态转换"""
        transitions = {}
        
        for i in range(1, len(df)):
            prev_regime = df.iloc[i-1]['regime']
            curr_regime = df.iloc[i]['regime']
            
            if prev_regime != curr_regime:
                transition = f"{prev_regime} -> {curr_regime}"
                transitions[transition] = transitions.get(transition, 0) + 1
        
        return transitions