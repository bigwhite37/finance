"""
状态感知阈值调节器模块

根据市场状态动态调整筛选阈值，实现不同波动状态下的参数配置逻辑。
在高波动状态下收紧阈值，在低波动状态下放宽阈值，同时提供平滑处理
和置信度调整功能，确保阈值变化的合理性和稳定性。
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np

from ..data_structures import DynamicLowVolConfig
from ..exceptions import ConfigurationException, DataQualityException


class RegimeAwareThresholdAdjuster:
    """状态感知阈值调节器
    
    根据市场状态动态调整筛选阈值，实现不同波动状态下的参数配置逻辑。
    在高波动状态下收紧阈值，在低波动状态下放宽阈值。
    
    主要功能：
    - 基于市场状态的阈值配置管理
    - 根据市场波动率的阈值微调
    - 基于状态置信度的阈值调整
    - 阈值变化的平滑处理
    - 自适应分位数阈值计算
    - 阈值调整历史记录和统计
    
    Attributes:
        config: 筛选器配置对象
        default_thresholds: 默认阈值配置
        regime_thresholds: 各状态下的阈值配置
        adjustment_history: 阈值调整历史记录
    """
    
    def __init__(self, config: DynamicLowVolConfig):
        """初始化状态感知阈值调节器
        
        Args:
            config: 筛选器配置
            
        Raises:
            ConfigurationException: 配置参数错误
        """
        self.config = config
        
        # 默认阈值配置
        self.default_thresholds = {
            "percentile_cut": config.percentile_thresholds.get("中", 0.3),
            "target_vol": 0.70,  # 提高默认目标波动率
            "ivol_bad_threshold": config.ivol_bad_threshold,
            "ivol_good_threshold": config.ivol_good_threshold,
            "garch_confidence": 0.95
        }
        
        # 状态特定阈值配置
        self.regime_thresholds = {
            "高": {
                "percentile_cut": config.percentile_thresholds.get("高", 0.2),
                "target_vol": 0.65,  # 提高到65%以适应实际数据
                "ivol_bad_threshold": config.ivol_bad_threshold * 0.8,  # 更严格
                "ivol_good_threshold": config.ivol_good_threshold * 0.9,  # 更严格
                "garch_confidence": 0.99  # 更高置信度
            },
            "中": {
                "percentile_cut": config.percentile_thresholds.get("中", 0.3),
                "target_vol": 0.70,  # 提高到70%以适应实际数据
                "ivol_bad_threshold": config.ivol_bad_threshold,
                "ivol_good_threshold": config.ivol_good_threshold,
                "garch_confidence": 0.95
            },
            "低": {
                "percentile_cut": config.percentile_thresholds.get("低", 0.4),
                "target_vol": 0.75,  # 提高到75%以适应实际数据
                "ivol_bad_threshold": config.ivol_bad_threshold * 1.2,  # 更宽松
                "ivol_good_threshold": config.ivol_good_threshold * 1.1,  # 更宽松
                "garch_confidence": 0.90  # 较低置信度
            }
        }
        
        # 验证配置
        self._validate_threshold_config()
        
        # 阈值调整历史记录
        self.adjustment_history = []
        
        # 平滑参数
        self.smoothing_factor = 0.3  # 用于平滑阈值调整
        self.previous_thresholds = None
    
    def adjust_thresholds(self, 
                         current_regime: str,
                         market_volatility: Optional[float] = None,
                         regime_confidence: Optional[float] = None) -> Dict[str, float]:
        """根据市场状态动态调整筛选阈值
        
        综合考虑市场状态、波动率水平和状态置信度，
        动态调整各项筛选阈值，并应用平滑处理。
        
        Args:
            current_regime: 当前市场状态 ("低", "中", "高")
            market_volatility: 当前市场波动率，用于微调
            regime_confidence: 状态检测置信度 (0-1)
            
        Returns:
            调整后的阈值配置字典
            
        Raises:
            ConfigurationException: 状态参数无效
            DataQualityException: 输入数据异常
        """
        # 验证输入参数
        self._validate_adjustment_inputs(current_regime, market_volatility, regime_confidence)
        
        # 获取基础阈值
        base_thresholds = self._get_base_thresholds(current_regime)
        
        # 根据市场波动率微调
        if market_volatility is not None:
            base_thresholds = self._apply_volatility_adjustment(
                base_thresholds, market_volatility, current_regime
            )
        
        # 根据状态置信度调整
        if regime_confidence is not None:
            base_thresholds = self._apply_confidence_adjustment(
                base_thresholds, regime_confidence, current_regime
            )
        
        # 应用平滑处理
        smoothed_thresholds = self._apply_smoothing(base_thresholds)
        
        # 验证调整结果
        self._validate_adjusted_thresholds(smoothed_thresholds)
        
        # 记录调整历史
        self._record_adjustment(current_regime, smoothed_thresholds, 
                              market_volatility, regime_confidence)
        
        # 更新前一次阈值
        self.previous_thresholds = smoothed_thresholds.copy()
        
        return smoothed_thresholds
    
    def get_regime_specific_config(self, regime: str) -> Dict[str, float]:
        """获取特定状态下的完整配置
        
        Args:
            regime: 市场状态 ("低", "中", "高")
            
        Returns:
            该状态下的完整阈值配置
            
        Raises:
            ConfigurationException: 状态参数无效
        """
        if regime not in self.regime_thresholds:
            raise ConfigurationException(f"不支持的市场状态: {regime}")
        
        return self.regime_thresholds[regime].copy()
    
    def calculate_adaptive_percentile_threshold(self, 
                                              current_regime: str,
                                              market_stress_level: float = 0.0) -> float:
        """计算自适应分位数阈值
        
        根据市场状态和压力水平计算动态的分位数阈值，
        实现更精细的阈值调整策略。
        
        Args:
            current_regime: 当前市场状态
            market_stress_level: 市场压力水平 (-1到1，负值表示低压力，正值表示高压力)
            
        Returns:
            自适应分位数阈值
            
        Raises:
            ConfigurationException: 参数无效
        """
        if current_regime not in self.regime_thresholds:
            raise ConfigurationException(f"不支持的市场状态: {current_regime}")
        
        if not (-1 <= market_stress_level <= 1):
            raise ConfigurationException(f"市场压力水平必须在[-1,1]范围内，当前为{market_stress_level}")
        
        # 获取基础阈值
        base_threshold = self.regime_thresholds[current_regime]["percentile_cut"]
        
        # 根据压力水平调整
        # 高压力时收紧阈值，低压力时放宽阈值
        stress_adjustment = -market_stress_level * 0.1  # 最大调整10%
        adaptive_threshold = base_threshold + stress_adjustment
        
        # 确保阈值在合理范围内
        adaptive_threshold = np.clip(adaptive_threshold, 0.1, 0.5)
        
        return adaptive_threshold
    
    def get_threshold_adjustment_statistics(self) -> Dict:
        """获取阈值调整统计信息
        
        Returns:
            调整统计信息字典
        """
        if not self.adjustment_history:
            return {
                'total_adjustments': 0,
                'regime_distribution': {},
                'average_thresholds': {},
                'threshold_volatility': {}
            }
        
        # 统计调整次数
        total_adjustments = len(self.adjustment_history)
        
        # 统计状态分布
        regimes = [record['regime'] for record in self.adjustment_history]
        regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
        regime_distribution = {
            regime: count / total_adjustments 
            for regime, count in regime_counts.items()
        }
        
        # 计算平均阈值
        threshold_keys = ['percentile_cut', 'target_vol', 'ivol_bad_threshold', 'ivol_good_threshold']
        average_thresholds = {}
        threshold_volatility = {}
        
        for key in threshold_keys:
            values = [record['thresholds'][key] for record in self.adjustment_history]
            average_thresholds[key] = np.mean(values)
            threshold_volatility[key] = np.std(values)
        
        return {
            'total_adjustments': total_adjustments,
            'regime_distribution': regime_distribution,
            'average_thresholds': average_thresholds,
            'threshold_volatility': threshold_volatility,
            'latest_regime': regimes[-1] if regimes else None,
            'latest_thresholds': self.adjustment_history[-1]['thresholds'] if self.adjustment_history else {}
        }
    
    def reset_adjustment_history(self) -> None:
        """重置调整历史记录"""
        self.adjustment_history.clear()
        self.previous_thresholds = None
    
    def _validate_threshold_config(self) -> None:
        """验证阈值配置的合理性
        
        Raises:
            ConfigurationException: 配置参数不合理
        """
        for regime, thresholds in self.regime_thresholds.items():
            # 验证分位数阈值
            if not (0 < thresholds["percentile_cut"] < 1):
                raise ConfigurationException(
                    f"{regime}状态的分位数阈值{thresholds['percentile_cut']}必须在(0,1)范围内"
                )
            
            # 验证目标波动率
            if not (0 < thresholds["target_vol"] < 2):
                raise ConfigurationException(
                    f"{regime}状态的目标波动率{thresholds['target_vol']}必须在(0,2)范围内"
                )
            
            # 验证IVOL阈值
            if not (0 < thresholds["ivol_bad_threshold"] < 1):
                raise ConfigurationException(
                    f"{regime}状态的IVOL坏波动阈值{thresholds['ivol_bad_threshold']}必须在(0,1)范围内"
                )
            
            if not (0 < thresholds["ivol_good_threshold"] < 1):
                raise ConfigurationException(
                    f"{regime}状态的IVOL好波动阈值{thresholds['ivol_good_threshold']}必须在(0,1)范围内"
                )
            
            # 验证置信度
            if not (0 < thresholds["garch_confidence"] < 1):
                raise ConfigurationException(
                    f"{regime}状态的GARCH置信度{thresholds['garch_confidence']}必须在(0,1)范围内"
                )
        
        # 验证状态间阈值的合理性
        high_cut = self.regime_thresholds["高"]["percentile_cut"]
        mid_cut = self.regime_thresholds["中"]["percentile_cut"]
        low_cut = self.regime_thresholds["低"]["percentile_cut"]
        
        if not (high_cut <= mid_cut <= low_cut):
            raise ConfigurationException(
                f"分位数阈值应满足：高波动({high_cut}) <= 中波动({mid_cut}) <= 低波动({low_cut})"
            )
    
    def _validate_adjustment_inputs(self, 
                                  current_regime: str,
                                  market_volatility: Optional[float],
                                  regime_confidence: Optional[float]) -> None:
        """验证调整输入参数
        
        Args:
            current_regime: 当前市场状态
            market_volatility: 市场波动率
            regime_confidence: 状态置信度
            
        Raises:
            ConfigurationException: 参数无效
            DataQualityException: 数据异常
        """
        if current_regime not in self.regime_thresholds:
            raise ConfigurationException(f"不支持的市场状态: {current_regime}")
        
        if market_volatility is not None:
            if not isinstance(market_volatility, (int, float)):
                raise DataQualityException("市场波动率必须为数值类型")
            
            if market_volatility < 0:
                raise DataQualityException(f"市场波动率不能为负数: {market_volatility}")
            
            if market_volatility > 2.0:  # 200%的年化波动率
                raise DataQualityException(f"市场波动率{market_volatility:.2%}异常过高")
        
        if regime_confidence is not None:
            if not isinstance(regime_confidence, (int, float)):
                raise DataQualityException("状态置信度必须为数值类型")
            
            if not (0 <= regime_confidence <= 1):
                raise DataQualityException(f"状态置信度必须在[0,1]范围内: {regime_confidence}")
    
    def _get_base_thresholds(self, current_regime: str) -> Dict[str, float]:
        """获取基础阈值配置
        
        Args:
            current_regime: 当前市场状态
            
        Returns:
            基础阈值配置
        """
        return self.regime_thresholds[current_regime].copy()
    
    def _apply_volatility_adjustment(self, 
                                   base_thresholds: Dict[str, float],
                                   market_volatility: float,
                                   current_regime: str) -> Dict[str, float]:
        """根据市场波动率微调阈值
        
        Args:
            base_thresholds: 基础阈值配置
            market_volatility: 当前市场波动率
            current_regime: 当前市场状态
            
        Returns:
            调整后的阈值配置
        """
        adjusted_thresholds = base_thresholds.copy()
        
        # 定义各状态的正常波动率范围
        normal_volatility_ranges = {
            "低": (0.10, 0.25),   # 10%-25%
            "中": (0.20, 0.40),   # 20%-40%
            "高": (0.35, 0.70)    # 35%-70%
        }
        
        normal_min, normal_max = normal_volatility_ranges[current_regime]
        normal_mid = (normal_min + normal_max) / 2
        
        # 计算波动率偏离度
        deviation = (market_volatility - normal_mid) / normal_mid
        
        # 调整因子：波动率越高，调整因子越小，阈值越严格
        adjustment_factor = 1 - deviation * 0.5 # 增大调整斜率
        
        # 确保调整因子在合理范围内
        adjustment_factor = np.clip(adjustment_factor, 0.5, 1.5)
        
        # 应用调整
        adjusted_thresholds["percentile_cut"] *= adjustment_factor
        adjusted_thresholds["ivol_bad_threshold"] *= adjustment_factor
        
        # 确保调整后的阈值在合理范围内
        adjusted_thresholds["percentile_cut"] = np.clip(
            adjusted_thresholds["percentile_cut"], 0.1, 0.5
        )
        adjusted_thresholds["ivol_bad_threshold"] = np.clip(
            adjusted_thresholds["ivol_bad_threshold"], 0.1, 0.8
        )
        
        return adjusted_thresholds
    
    def _apply_confidence_adjustment(self, 
                                   base_thresholds: Dict[str, float],
                                   regime_confidence: float,
                                   current_regime: str) -> Dict[str, float]:
        """根据状态检测置信度调整阈值
        
        Args:
            base_thresholds: 基础阈值配置
            regime_confidence: 状态检测置信度
            current_regime: 当前市场状态
            
        Returns:
            调整后的阈值配置
        """
        adjusted_thresholds = base_thresholds.copy()
        
        # 当置信度较低时，向中性状态的阈值靠拢
        if regime_confidence < 0.7:
            # 获取中性状态（"中"）的阈值
            neutral_thresholds = self.regime_thresholds["中"]
            
            # 计算向中性状态靠拢的程度
            confidence_factor = regime_confidence / 0.7  # 归一化到[0,1]
            blend_factor = 1 - confidence_factor  # 置信度越低，越向中性靠拢
            
            # 混合当前状态和中性状态的阈值
            for key in ["percentile_cut", "ivol_bad_threshold", "ivol_good_threshold"]:
                current_value = adjusted_thresholds[key]
                neutral_value = neutral_thresholds[key]
                adjusted_thresholds[key] = (
                    current_value * confidence_factor + 
                    neutral_value * blend_factor
                )
        
        return adjusted_thresholds
    
    def _apply_smoothing(self, base_thresholds: Dict[str, float]) -> Dict[str, float]:
        """应用平滑处理，避免阈值剧烈变化
        
        Args:
            base_thresholds: 基础阈值配置
            
        Returns:
            平滑后的阈值配置
        """
        if self.previous_thresholds is None:
            return base_thresholds
        
        smoothed_thresholds = {}
        
        for key, current_value in base_thresholds.items():
            if key in self.previous_thresholds:
                previous_value = self.previous_thresholds[key]
                # 指数移动平均平滑
                smoothed_value = (
                    self.smoothing_factor * current_value + 
                    (1 - self.smoothing_factor) * previous_value
                )
                smoothed_thresholds[key] = smoothed_value
            else:
                smoothed_thresholds[key] = current_value
        
        return smoothed_thresholds
    
    def _validate_adjusted_thresholds(self, thresholds: Dict[str, float]) -> None:
        """验证调整后阈值的合理性
        
        Args:
            thresholds: 调整后的阈值配置
            
        Raises:
            ConfigurationException: 阈值不合理
        """
        # 验证分位数阈值
        if not (0.05 <= thresholds["percentile_cut"] <= 0.6):
            raise ConfigurationException(
                f"调整后的分位数阈值{thresholds['percentile_cut']:.3f}超出合理范围[0.05, 0.6]"
            )
        
        # 验证目标波动率
        if not (0.2 <= thresholds["target_vol"] <= 0.8):
            raise ConfigurationException(
                f"调整后的目标波动率{thresholds['target_vol']:.3f}超出合理范围[0.2, 0.8]"
            )
        
        # 验证IVOL阈值
        if not (0.05 <= thresholds["ivol_bad_threshold"] <= 0.9):
            raise ConfigurationException(
                f"调整后的IVOL坏波动阈值{thresholds['ivol_bad_threshold']:.3f}超出合理范围[0.05, 0.9]"
            )
        
        if not (0.1 <= thresholds["ivol_good_threshold"] <= 0.95):
            raise ConfigurationException(
                f"调整后的IVOL好波动阈值{thresholds['ivol_good_threshold']:.3f}超出合理范围[0.1, 0.95]"
            )
        
        # 验证GARCH置信度
        if not (0.8 <= thresholds["garch_confidence"] <= 0.999):
            raise ConfigurationException(
                f"调整后的GARCH置信度{thresholds['garch_confidence']:.3f}超出合理范围[0.8, 0.999]"
            )
    
    def _record_adjustment(self, 
                         regime: str,
                         thresholds: Dict[str, float],
                         market_volatility: Optional[float],
                         regime_confidence: Optional[float]) -> None:
        """记录阈值调整历史
        
        Args:
            regime: 市场状态
            thresholds: 调整后的阈值
            market_volatility: 市场波动率
            regime_confidence: 状态置信度
        """
        record = {
            'timestamp': pd.Timestamp.now(),
            'regime': regime,
            'thresholds': thresholds.copy(),
            'market_volatility': market_volatility,
            'regime_confidence': regime_confidence
        }
        
        self.adjustment_history.append(record)
        
        # 限制历史记录长度，避免内存占用过多
        max_history_length = 1000
        if len(self.adjustment_history) > max_history_length:
            self.adjustment_history = self.adjustment_history[-max_history_length:]