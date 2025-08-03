"""集中度控制器实现"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import logging
from abc import ABC, abstractmethod


class ConcentrationType(Enum):
    """集中度类型枚举"""
    SINGLE_ASSET = "single_asset"
    SECTOR = "sector"
    FACTOR = "factor"
    GEOGRAPHIC = "geographic"
    MARKET_CAP = "market_cap"


class ViolationSeverity(Enum):
    """违规严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConcentrationLimit:
    """集中度限制配置"""
    concentration_type: ConcentrationType
    max_weight: float                                 # 最大权重限制
    warning_threshold: float                          # 警告阈值
    target_weight: Optional[float] = None            # 目标权重
    enabled: bool = True                             # 是否启用
    description: str = ""                            # 描述


@dataclass
class ConcentrationConfig:
    """集中度控制配置"""
    # 单一资产集中度限制
    max_single_asset_weight: float = 0.15           # 单一资产最大权重 (15%)
    single_asset_warning_threshold: float = 0.12    # 单一资产警告阈值 (12%)
    
    # 行业集中度限制
    max_sector_weight: float = 0.30                 # 单一行业最大权重 (30%)
    sector_warning_threshold: float = 0.25          # 行业警告阈值 (25%)
    max_sectors_above_threshold: int = 3            # 超过阈值的最大行业数
    
    # 因子集中度限制
    max_factor_exposure: float = 0.40               # 单一因子最大暴露 (40%)
    factor_warning_threshold: float = 0.30          # 因子警告阈值 (30%)
    
    # 地理集中度限制
    max_geographic_weight: float = 0.60             # 单一地区最大权重 (60%)
    geographic_warning_threshold: float = 0.50     # 地理警告阈值 (50%)
    
    # 市值集中度限制
    max_market_cap_weight: float = 0.70             # 单一市值类别最大权重 (70%)
    market_cap_warning_threshold: float = 0.60     # 市值警告阈值 (60%)
    
    # 调整参数
    adjustment_speed: float = 0.1                   # 调整速度
    min_adjustment_threshold: float = 0.01          # 最小调整阈值
    max_adjustment_per_step: float = 0.05           # 单步最大调整幅度
    
    # 赫芬达尔指数限制
    max_herfindahl_index: float = 0.20              # 最大赫芬达尔指数
    herfindahl_warning_threshold: float = 0.15      # 赫芬达尔警告阈值


@dataclass
class AssetInfo:
    """资产信息"""
    symbol: str
    sector: str
    geographic_region: str
    market_cap_category: str                        # Large/Mid/Small Cap
    factor_exposures: Dict[str, float]              # 因子暴露
    current_weight: float
    target_weight: float = 0.0
    liquidity_score: float = 1.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ConcentrationViolation:
    """集中度违规"""
    violation_type: ConcentrationType
    severity: ViolationSeverity
    current_value: float
    limit_value: float
    excess_amount: float
    affected_items: List[str]                       # 违规的资产/行业/因子等
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class ConcentrationMetrics:
    """集中度指标"""
    herfindahl_index: float                         # 赫芬达尔指数
    max_single_weight: float                        # 最大单一权重
    top_5_concentration: float                      # 前5大持仓集中度
    top_10_concentration: float                     # 前10大持仓集中度
    sector_concentration: Dict[str, float]          # 行业集中度
    factor_concentration: Dict[str, float]          # 因子集中度
    geographic_concentration: Dict[str, float]      # 地理集中度
    effective_number_of_assets: float               # 有效资产数量
    diversification_ratio: float                    # 多样化比率
    timestamp: datetime


class ConcentrationController:
    """集中度控制器"""
    
    def __init__(self, config: ConcentrationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 违规历史
        self.violation_history: List[ConcentrationViolation] = []
        self.metrics_history: List[ConcentrationMetrics] = []
        
        # 当前状态
        self.current_violations: List[ConcentrationViolation] = []
        self.last_check_time: Optional[datetime] = None
        
        # 集中度限制配置
        self.concentration_limits = self._initialize_concentration_limits()
        
        # 调整历史
        self.adjustment_history: List[Dict[str, Any]] = []
    
    def _initialize_concentration_limits(self) -> Dict[ConcentrationType, ConcentrationLimit]:
        """初始化集中度限制"""
        limits = {
            ConcentrationType.SINGLE_ASSET: ConcentrationLimit(
                concentration_type=ConcentrationType.SINGLE_ASSET,
                max_weight=self.config.max_single_asset_weight,
                warning_threshold=self.config.single_asset_warning_threshold,
                description="单一资产权重限制"
            ),
            ConcentrationType.SECTOR: ConcentrationLimit(
                concentration_type=ConcentrationType.SECTOR,
                max_weight=self.config.max_sector_weight,
                warning_threshold=self.config.sector_warning_threshold,
                description="行业集中度限制"
            ),
            ConcentrationType.FACTOR: ConcentrationLimit(
                concentration_type=ConcentrationType.FACTOR,
                max_weight=self.config.max_factor_exposure,
                warning_threshold=self.config.factor_warning_threshold,
                description="因子暴露限制"
            ),
            ConcentrationType.GEOGRAPHIC: ConcentrationLimit(
                concentration_type=ConcentrationType.GEOGRAPHIC,
                max_weight=self.config.max_geographic_weight,
                warning_threshold=self.config.geographic_warning_threshold,
                description="地理集中度限制"
            ),
            ConcentrationType.MARKET_CAP: ConcentrationLimit(
                concentration_type=ConcentrationType.MARKET_CAP,
                max_weight=self.config.max_market_cap_weight,
                warning_threshold=self.config.market_cap_warning_threshold,
                description="市值集中度限制"
            )
        }
        return limits
    
    def check_concentration_violations(self, 
                                     portfolio_weights: Dict[str, float],
                                     asset_info: Dict[str, AssetInfo]) -> List[ConcentrationViolation]:
        """
        检查集中度违规
        
        Args:
            portfolio_weights: 投资组合权重
            asset_info: 资产信息
            
        Returns:
            集中度违规列表
        """
        violations = []
        self.last_check_time = datetime.now()
        
        # 检查单一资产集中度
        single_asset_violations = self._check_single_asset_concentration(
            portfolio_weights, asset_info
        )
        violations.extend(single_asset_violations)
        
        # 检查行业集中度
        sector_violations = self._check_sector_concentration(
            portfolio_weights, asset_info
        )
        violations.extend(sector_violations)
        
        # 检查因子集中度
        factor_violations = self._check_factor_concentration(
            portfolio_weights, asset_info
        )
        violations.extend(factor_violations)
        
        # 检查地理集中度
        geographic_violations = self._check_geographic_concentration(
            portfolio_weights, asset_info
        )
        violations.extend(geographic_violations)
        
        # 检查市值集中度
        market_cap_violations = self._check_market_cap_concentration(
            portfolio_weights, asset_info
        )
        violations.extend(market_cap_violations)
        
        # 检查赫芬达尔指数
        herfindahl_violations = self._check_herfindahl_index(portfolio_weights)
        violations.extend(herfindahl_violations)
        
        # 更新当前违规状态
        self.current_violations = violations
        self.violation_history.extend(violations)
        
        # 记录日志
        if violations:
            self.logger.warning(f"发现{len(violations)}个集中度违规")
            for violation in violations:
                self.logger.warning(f"违规类型: {violation.violation_type.value}, "
                                  f"严重程度: {violation.severity.value}, "
                                  f"消息: {violation.message}")
        
        return violations
    
    def _check_single_asset_concentration(self, 
                                        portfolio_weights: Dict[str, float],
                                        asset_info: Dict[str, AssetInfo]) -> List[ConcentrationViolation]:
        """检查单一资产集中度"""
        violations = []
        limit = self.concentration_limits[ConcentrationType.SINGLE_ASSET]
        
        if not limit.enabled:
            return violations
        
        for symbol, weight in portfolio_weights.items():
            if weight > limit.max_weight:
                severity = self._determine_severity(
                    weight, limit.max_weight, limit.warning_threshold
                )
                
                violation = ConcentrationViolation(
                    violation_type=ConcentrationType.SINGLE_ASSET,
                    severity=severity,
                    current_value=weight,
                    limit_value=limit.max_weight,
                    excess_amount=weight - limit.max_weight,
                    affected_items=[symbol],
                    message=f"资产{symbol}权重{weight:.2%}超过限制{limit.max_weight:.2%}",
                    timestamp=datetime.now(),
                    metadata={'symbol': symbol}
                )
                violations.append(violation)
        
        return violations
    
    def _check_sector_concentration(self, 
                                  portfolio_weights: Dict[str, float],
                                  asset_info: Dict[str, AssetInfo]) -> List[ConcentrationViolation]:
        """检查行业集中度"""
        violations = []
        limit = self.concentration_limits[ConcentrationType.SECTOR]
        
        if not limit.enabled:
            return violations
        
        # 计算行业权重
        sector_weights = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in asset_info:
                sector = asset_info[symbol].sector
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        # 检查每个行业
        sectors_above_threshold = 0
        for sector, weight in sector_weights.items():
            if weight > limit.warning_threshold:
                sectors_above_threshold += 1
            
            if weight > limit.max_weight:
                severity = self._determine_severity(
                    weight, limit.max_weight, limit.warning_threshold
                )
                
                # 找出该行业的所有资产
                affected_assets = [
                    symbol for symbol, info in asset_info.items()
                    if info.sector == sector and symbol in portfolio_weights
                ]
                
                violation = ConcentrationViolation(
                    violation_type=ConcentrationType.SECTOR,
                    severity=severity,
                    current_value=weight,
                    limit_value=limit.max_weight,
                    excess_amount=weight - limit.max_weight,
                    affected_items=affected_assets,
                    message=f"行业{sector}权重{weight:.2%}超过限制{limit.max_weight:.2%}",
                    timestamp=datetime.now(),
                    metadata={'sector': sector, 'asset_count': len(affected_assets)}
                )
                violations.append(violation)
        
        # 检查超过阈值的行业数量
        if sectors_above_threshold > self.config.max_sectors_above_threshold:
            violation = ConcentrationViolation(
                violation_type=ConcentrationType.SECTOR,
                severity=ViolationSeverity.MEDIUM,
                current_value=sectors_above_threshold,
                limit_value=self.config.max_sectors_above_threshold,
                excess_amount=sectors_above_threshold - self.config.max_sectors_above_threshold,
                affected_items=list(sector_weights.keys()),
                message=f"超过阈值的行业数量{sectors_above_threshold}超过限制{self.config.max_sectors_above_threshold}",
                timestamp=datetime.now(),
                metadata={'sector_weights': sector_weights}
            )
            violations.append(violation)
        
        return violations
    
    def _check_factor_concentration(self, 
                                  portfolio_weights: Dict[str, float],
                                  asset_info: Dict[str, AssetInfo]) -> List[ConcentrationViolation]:
        """检查因子集中度"""
        violations = []
        limit = self.concentration_limits[ConcentrationType.FACTOR]
        
        if not limit.enabled:
            return violations
        
        # 计算因子暴露
        factor_exposures = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in asset_info and asset_info[symbol].factor_exposures:
                for factor, exposure in asset_info[symbol].factor_exposures.items():
                    if factor not in factor_exposures:
                        factor_exposures[factor] = 0
                    factor_exposures[factor] += weight * abs(exposure)
        
        # 检查每个因子
        for factor, exposure in factor_exposures.items():
            if exposure > limit.max_weight:
                severity = self._determine_severity(
                    exposure, limit.max_weight, limit.warning_threshold
                )
                
                # 找出对该因子有显著暴露的资产
                affected_assets = [
                    symbol for symbol, info in asset_info.items()
                    if (symbol in portfolio_weights and 
                        factor in info.factor_exposures and
                        abs(info.factor_exposures[factor]) > 0.1)
                ]
                
                violation = ConcentrationViolation(
                    violation_type=ConcentrationType.FACTOR,
                    severity=severity,
                    current_value=exposure,
                    limit_value=limit.max_weight,
                    excess_amount=exposure - limit.max_weight,
                    affected_items=affected_assets,
                    message=f"因子{factor}暴露{exposure:.2%}超过限制{limit.max_weight:.2%}",
                    timestamp=datetime.now(),
                    metadata={'factor': factor, 'asset_count': len(affected_assets)}
                )
                violations.append(violation)
        
        return violations
    
    def _check_geographic_concentration(self, 
                                      portfolio_weights: Dict[str, float],
                                      asset_info: Dict[str, AssetInfo]) -> List[ConcentrationViolation]:
        """检查地理集中度"""
        violations = []
        limit = self.concentration_limits[ConcentrationType.GEOGRAPHIC]
        
        if not limit.enabled:
            return violations
        
        # 计算地理权重
        geographic_weights = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in asset_info:
                region = asset_info[symbol].geographic_region
                geographic_weights[region] = geographic_weights.get(region, 0) + weight
        
        # 检查每个地区
        for region, weight in geographic_weights.items():
            if weight > limit.max_weight:
                severity = self._determine_severity(
                    weight, limit.max_weight, limit.warning_threshold
                )
                
                # 找出该地区的所有资产
                affected_assets = [
                    symbol for symbol, info in asset_info.items()
                    if info.geographic_region == region and symbol in portfolio_weights
                ]
                
                violation = ConcentrationViolation(
                    violation_type=ConcentrationType.GEOGRAPHIC,
                    severity=severity,
                    current_value=weight,
                    limit_value=limit.max_weight,
                    excess_amount=weight - limit.max_weight,
                    affected_items=affected_assets,
                    message=f"地区{region}权重{weight:.2%}超过限制{limit.max_weight:.2%}",
                    timestamp=datetime.now(),
                    metadata={'region': region, 'asset_count': len(affected_assets)}
                )
                violations.append(violation)
        
        return violations
    
    def _check_market_cap_concentration(self, 
                                      portfolio_weights: Dict[str, float],
                                      asset_info: Dict[str, AssetInfo]) -> List[ConcentrationViolation]:
        """检查市值集中度"""
        violations = []
        limit = self.concentration_limits[ConcentrationType.MARKET_CAP]
        
        if not limit.enabled:
            return violations
        
        # 计算市值类别权重
        market_cap_weights = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in asset_info:
                market_cap = asset_info[symbol].market_cap_category
                market_cap_weights[market_cap] = market_cap_weights.get(market_cap, 0) + weight
        
        # 检查每个市值类别
        for market_cap, weight in market_cap_weights.items():
            if weight > limit.max_weight:
                severity = self._determine_severity(
                    weight, limit.max_weight, limit.warning_threshold
                )
                
                # 找出该市值类别的所有资产
                affected_assets = [
                    symbol for symbol, info in asset_info.items()
                    if info.market_cap_category == market_cap and symbol in portfolio_weights
                ]
                
                violation = ConcentrationViolation(
                    violation_type=ConcentrationType.MARKET_CAP,
                    severity=severity,
                    current_value=weight,
                    limit_value=limit.max_weight,
                    excess_amount=weight - limit.max_weight,
                    affected_items=affected_assets,
                    message=f"市值类别{market_cap}权重{weight:.2%}超过限制{limit.max_weight:.2%}",
                    timestamp=datetime.now(),
                    metadata={'market_cap': market_cap, 'asset_count': len(affected_assets)}
                )
                violations.append(violation)
        
        return violations
    
    def _check_herfindahl_index(self, 
                              portfolio_weights: Dict[str, float]) -> List[ConcentrationViolation]:
        """检查赫芬达尔指数"""
        violations = []
        
        # 计算赫芬达尔指数
        herfindahl_index = sum(weight ** 2 for weight in portfolio_weights.values())
        
        if herfindahl_index > self.config.max_herfindahl_index:
            severity = self._determine_severity(
                herfindahl_index, 
                self.config.max_herfindahl_index, 
                self.config.herfindahl_warning_threshold
            )
            
            # 找出权重最大的资产
            sorted_assets = sorted(
                portfolio_weights.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            top_assets = [asset for asset, _ in sorted_assets[:5]]
            
            violation = ConcentrationViolation(
                violation_type=ConcentrationType.SINGLE_ASSET,
                severity=severity,
                current_value=herfindahl_index,
                limit_value=self.config.max_herfindahl_index,
                excess_amount=herfindahl_index - self.config.max_herfindahl_index,
                affected_items=top_assets,
                message=f"赫芬达尔指数{herfindahl_index:.3f}超过限制{self.config.max_herfindahl_index:.3f}",
                timestamp=datetime.now(),
                metadata={'top_5_assets': top_assets}
            )
            violations.append(violation)
        
        return violations
    
    def _determine_severity(self, 
                          current_value: float, 
                          max_limit: float, 
                          warning_threshold: float) -> ViolationSeverity:
        """确定违规严重程度"""
        if current_value <= warning_threshold:
            return ViolationSeverity.LOW
        elif current_value <= max_limit:
            return ViolationSeverity.MEDIUM
        elif current_value <= max_limit * 1.2:
            return ViolationSeverity.HIGH
        else:
            return ViolationSeverity.CRITICAL
    
    def adjust_portfolio_for_concentration(self, 
                                         portfolio_weights: Dict[str, float],
                                         asset_info: Dict[str, AssetInfo],
                                         violations: List[ConcentrationViolation] = None) -> Dict[str, float]:
        """
        调整投资组合以满足集中度限制
        
        Args:
            portfolio_weights: 当前投资组合权重
            asset_info: 资产信息
            violations: 集中度违规（可选，如果不提供则自动检查）
            
        Returns:
            调整后的投资组合权重
        """
        if violations is None:
            violations = self.check_concentration_violations(portfolio_weights, asset_info)
        
        if not violations:
            return portfolio_weights.copy()
        
        adjusted_weights = portfolio_weights.copy()
        adjustment_log = {
            'timestamp': datetime.now(),
            'original_weights': portfolio_weights.copy(),
            'violations': len(violations),
            'adjustments': []
        }
        
        # 按严重程度排序违规
        violations_sorted = sorted(
            violations, 
            key=lambda x: self._get_severity_score(x.severity), 
            reverse=True
        )
        
        for violation in violations_sorted:
            if violation.violation_type == ConcentrationType.SINGLE_ASSET:
                adjusted_weights = self._adjust_single_asset_concentration(
                    adjusted_weights, violation, asset_info, adjustment_log
                )
            elif violation.violation_type == ConcentrationType.SECTOR:
                adjusted_weights = self._adjust_sector_concentration(
                    adjusted_weights, violation, asset_info, adjustment_log
                )
            elif violation.violation_type == ConcentrationType.FACTOR:
                adjusted_weights = self._adjust_factor_concentration(
                    adjusted_weights, violation, asset_info, adjustment_log
                )
            elif violation.violation_type == ConcentrationType.GEOGRAPHIC:
                adjusted_weights = self._adjust_geographic_concentration(
                    adjusted_weights, violation, asset_info, adjustment_log
                )
            elif violation.violation_type == ConcentrationType.MARKET_CAP:
                adjusted_weights = self._adjust_market_cap_concentration(
                    adjusted_weights, violation, asset_info, adjustment_log
                )
        
        # 重新归一化权重
        adjusted_weights = self._normalize_weights(adjusted_weights)
        
        # 记录调整历史
        adjustment_log['final_weights'] = adjusted_weights.copy()
        adjustment_log['total_adjustment'] = sum(
            abs(adjusted_weights.get(symbol, 0) - portfolio_weights.get(symbol, 0))
            for symbol in set(list(adjusted_weights.keys()) + list(portfolio_weights.keys()))
        )
        self.adjustment_history.append(adjustment_log)
        
        self.logger.info(f"集中度调整完成: 处理{len(violations)}个违规, "
                        f"总调整幅度{adjustment_log['total_adjustment']:.2%}")
        
        return adjusted_weights
    
    def _adjust_single_asset_concentration(self, 
                                         weights: Dict[str, float],
                                         violation: ConcentrationViolation,
                                         asset_info: Dict[str, AssetInfo],
                                         adjustment_log: Dict[str, Any]) -> Dict[str, float]:
        """调整单一资产集中度"""
        symbol = violation.affected_items[0]
        current_weight = weights.get(symbol, 0)
        target_weight = violation.limit_value
        
        # 计算调整幅度
        adjustment = min(
            current_weight - target_weight,
            self.config.max_adjustment_per_step
        )
        
        if adjustment > self.config.min_adjustment_threshold:
            weights[symbol] = current_weight - adjustment
            
            # 将减少的权重分配给其他资产
            self._redistribute_weight(weights, symbol, adjustment, asset_info)
            
            adjustment_log['adjustments'].append({
                'type': 'single_asset',
                'symbol': symbol,
                'original_weight': current_weight,
                'adjusted_weight': weights[symbol],
                'adjustment': -adjustment
            })
        
        return weights
    
    def _adjust_sector_concentration(self, 
                                   weights: Dict[str, float],
                                   violation: ConcentrationViolation,
                                   asset_info: Dict[str, AssetInfo],
                                   adjustment_log: Dict[str, Any]) -> Dict[str, float]:
        """调整行业集中度"""
        sector = violation.metadata.get('sector')
        if not sector:
            return weights
        
        # 找出该行业的所有资产
        sector_assets = [
            symbol for symbol in violation.affected_items
            if symbol in weights
        ]
        
        # 计算当前行业总权重
        current_sector_weight = sum(weights.get(symbol, 0) for symbol in sector_assets)
        target_sector_weight = violation.limit_value
        
        # 计算需要减少的权重
        total_reduction = min(
            current_sector_weight - target_sector_weight,
            self.config.max_adjustment_per_step
        )
        
        if total_reduction > self.config.min_adjustment_threshold:
            # 按当前权重比例减少
            for symbol in sector_assets:
                if symbol in weights and weights[symbol] > 0:
                    reduction_ratio = weights[symbol] / current_sector_weight
                    reduction = total_reduction * reduction_ratio
                    weights[symbol] = max(0, weights[symbol] - reduction)
            
            # 重新分配减少的权重
            self._redistribute_weight(weights, None, total_reduction, asset_info, exclude_sector=sector)
            
            adjustment_log['adjustments'].append({
                'type': 'sector',
                'sector': sector,
                'assets': sector_assets,
                'total_reduction': total_reduction
            })
        
        return weights
    
    def _adjust_factor_concentration(self, 
                                   weights: Dict[str, float],
                                   violation: ConcentrationViolation,
                                   asset_info: Dict[str, AssetInfo],
                                   adjustment_log: Dict[str, Any]) -> Dict[str, float]:
        """调整因子集中度"""
        factor = violation.metadata.get('factor')
        if not factor:
            return weights
        
        # 找出对该因子有显著暴露的资产
        factor_assets = []
        for symbol in violation.affected_items:
            if (symbol in weights and symbol in asset_info and
                factor in asset_info[symbol].factor_exposures):
                factor_assets.append((symbol, asset_info[symbol].factor_exposures[factor]))
        
        # 按因子暴露排序
        factor_assets.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 减少因子暴露最大的资产权重
        total_reduction = 0
        target_reduction = min(
            violation.excess_amount,
            self.config.max_adjustment_per_step
        )
        
        for symbol, exposure in factor_assets:
            if total_reduction >= target_reduction:
                break
            
            current_weight = weights.get(symbol, 0)
            if current_weight > 0:
                # 根据因子暴露计算减少幅度
                reduction = min(
                    current_weight * 0.1,  # 最多减少10%
                    target_reduction - total_reduction
                )
                weights[symbol] = max(0, current_weight - reduction)
                total_reduction += reduction
        
        # 重新分配权重
        if total_reduction > 0:
            self._redistribute_weight(weights, None, total_reduction, asset_info)
            
            adjustment_log['adjustments'].append({
                'type': 'factor',
                'factor': factor,
                'total_reduction': total_reduction,
                'affected_assets': [symbol for symbol, _ in factor_assets]
            })
        
        return weights
    
    def _adjust_geographic_concentration(self, 
                                       weights: Dict[str, float],
                                       violation: ConcentrationViolation,
                                       asset_info: Dict[str, AssetInfo],
                                       adjustment_log: Dict[str, Any]) -> Dict[str, float]:
        """调整地理集中度"""
        region = violation.metadata.get('region')
        if not region:
            return weights
        
        # 类似于行业集中度调整
        region_assets = [
            symbol for symbol in violation.affected_items
            if symbol in weights
        ]
        
        current_region_weight = sum(weights.get(symbol, 0) for symbol in region_assets)
        target_region_weight = violation.limit_value
        
        total_reduction = min(
            current_region_weight - target_region_weight,
            self.config.max_adjustment_per_step
        )
        
        if total_reduction > self.config.min_adjustment_threshold:
            for symbol in region_assets:
                if symbol in weights and weights[symbol] > 0:
                    reduction_ratio = weights[symbol] / current_region_weight
                    reduction = total_reduction * reduction_ratio
                    weights[symbol] = max(0, weights[symbol] - reduction)
            
            self._redistribute_weight(weights, None, total_reduction, asset_info, exclude_region=region)
            
            adjustment_log['adjustments'].append({
                'type': 'geographic',
                'region': region,
                'assets': region_assets,
                'total_reduction': total_reduction
            })
        
        return weights
    
    def _adjust_market_cap_concentration(self, 
                                       weights: Dict[str, float],
                                       violation: ConcentrationViolation,
                                       asset_info: Dict[str, AssetInfo],
                                       adjustment_log: Dict[str, Any]) -> Dict[str, float]:
        """调整市值集中度"""
        market_cap = violation.metadata.get('market_cap')
        if not market_cap:
            return weights
        
        # 类似于行业集中度调整
        market_cap_assets = [
            symbol for symbol in violation.affected_items
            if symbol in weights
        ]
        
        current_market_cap_weight = sum(weights.get(symbol, 0) for symbol in market_cap_assets)
        target_market_cap_weight = violation.limit_value
        
        total_reduction = min(
            current_market_cap_weight - target_market_cap_weight,
            self.config.max_adjustment_per_step
        )
        
        if total_reduction > self.config.min_adjustment_threshold:
            for symbol in market_cap_assets:
                if symbol in weights and weights[symbol] > 0:
                    reduction_ratio = weights[symbol] / current_market_cap_weight
                    reduction = total_reduction * reduction_ratio
                    weights[symbol] = max(0, weights[symbol] - reduction)
            
            self._redistribute_weight(weights, None, total_reduction, asset_info, exclude_market_cap=market_cap)
            
            adjustment_log['adjustments'].append({
                'type': 'market_cap',
                'market_cap': market_cap,
                'assets': market_cap_assets,
                'total_reduction': total_reduction
            })
        
        return weights
    
    def _redistribute_weight(self, 
                           weights: Dict[str, float],
                           exclude_symbol: Optional[str],
                           amount: float,
                           asset_info: Dict[str, AssetInfo],
                           exclude_sector: Optional[str] = None,
                           exclude_region: Optional[str] = None,
                           exclude_market_cap: Optional[str] = None) -> None:
        """重新分配权重"""
        if amount <= 0:
            return
        
        # 找出可以增加权重的资产
        eligible_assets = []
        for symbol, current_weight in weights.items():
            if symbol == exclude_symbol:
                continue
            
            if symbol not in asset_info:
                continue
            
            info = asset_info[symbol]
            
            # 检查排除条件
            if exclude_sector and info.sector == exclude_sector:
                continue
            if exclude_region and info.geographic_region == exclude_region:
                continue
            if exclude_market_cap and info.market_cap_category == exclude_market_cap:
                continue
            
            # 检查是否还有增加空间
            if current_weight < self.config.max_single_asset_weight:
                eligible_assets.append(symbol)
        
        if not eligible_assets:
            return
        
        # 按流动性评分分配权重
        total_liquidity = sum(
            asset_info[symbol].liquidity_score 
            for symbol in eligible_assets
        )
        
        for symbol in eligible_assets:
            if total_liquidity > 0:
                liquidity_ratio = asset_info[symbol].liquidity_score / total_liquidity
                additional_weight = amount * liquidity_ratio
                
                # 确保不超过单一资产限制
                max_additional = self.config.max_single_asset_weight - weights.get(symbol, 0)
                additional_weight = min(additional_weight, max_additional)
                
                weights[symbol] = weights.get(symbol, 0) + additional_weight
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """归一化权重"""
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return weights
        
        return {symbol: weight / total_weight for symbol, weight in weights.items()}
    
    def _get_severity_score(self, severity: ViolationSeverity) -> int:
        """获取严重程度评分"""
        severity_scores = {
            ViolationSeverity.LOW: 1,
            ViolationSeverity.MEDIUM: 2,
            ViolationSeverity.HIGH: 3,
            ViolationSeverity.CRITICAL: 4
        }
        return severity_scores.get(severity, 0)
    
    def calculate_concentration_metrics(self, 
                                      portfolio_weights: Dict[str, float],
                                      asset_info: Dict[str, AssetInfo]) -> ConcentrationMetrics:
        """计算集中度指标"""
        if not portfolio_weights:
            return ConcentrationMetrics(
                herfindahl_index=0.0,
                max_single_weight=0.0,
                top_5_concentration=0.0,
                top_10_concentration=0.0,
                sector_concentration={},
                factor_concentration={},
                geographic_concentration={},
                effective_number_of_assets=0.0,
                diversification_ratio=0.0,
                timestamp=datetime.now()
            )
        
        # 赫芬达尔指数
        herfindahl_index = sum(weight ** 2 for weight in portfolio_weights.values())
        
        # 最大单一权重
        max_single_weight = max(portfolio_weights.values())
        
        # 排序权重
        sorted_weights = sorted(portfolio_weights.values(), reverse=True)
        
        # 前5大和前10大集中度
        top_5_concentration = sum(sorted_weights[:5])
        top_10_concentration = sum(sorted_weights[:10])
        
        # 行业集中度
        sector_concentration = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in asset_info:
                sector = asset_info[symbol].sector
                sector_concentration[sector] = sector_concentration.get(sector, 0) + weight
        
        # 因子集中度
        factor_concentration = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in asset_info and asset_info[symbol].factor_exposures:
                for factor, exposure in asset_info[symbol].factor_exposures.items():
                    if factor not in factor_concentration:
                        factor_concentration[factor] = 0
                    factor_concentration[factor] += weight * abs(exposure)
        
        # 地理集中度
        geographic_concentration = {}
        for symbol, weight in portfolio_weights.items():
            if symbol in asset_info:
                region = asset_info[symbol].geographic_region
                geographic_concentration[region] = geographic_concentration.get(region, 0) + weight
        
        # 有效资产数量
        effective_number_of_assets = 1.0 / herfindahl_index if herfindahl_index > 0 else 0
        
        # 多样化比率
        diversification_ratio = 1.0 - max_single_weight
        
        metrics = ConcentrationMetrics(
            herfindahl_index=herfindahl_index,
            max_single_weight=max_single_weight,
            top_5_concentration=top_5_concentration,
            top_10_concentration=top_10_concentration,
            sector_concentration=sector_concentration,
            factor_concentration=factor_concentration,
            geographic_concentration=geographic_concentration,
            effective_number_of_assets=effective_number_of_assets,
            diversification_ratio=diversification_ratio,
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def get_concentration_summary(self) -> Dict[str, Any]:
        """获取集中度摘要"""
        return {
            'current_violations': len(self.current_violations),
            'total_violations_history': len(self.violation_history),
            'last_check_time': self.last_check_time,
            'adjustment_count': len(self.adjustment_history),
            'concentration_limits': {
                limit_type.value: {
                    'max_weight': limit.max_weight,
                    'warning_threshold': limit.warning_threshold,
                    'enabled': limit.enabled
                }
                for limit_type, limit in self.concentration_limits.items()
            }
        }
    
    def update_concentration_limit(self, 
                                 concentration_type: ConcentrationType,
                                 max_weight: Optional[float] = None,
                                 warning_threshold: Optional[float] = None,
                                 enabled: Optional[bool] = None) -> None:
        """更新集中度限制"""
        if concentration_type in self.concentration_limits:
            limit = self.concentration_limits[concentration_type]
            
            if max_weight is not None:
                limit.max_weight = max_weight
            if warning_threshold is not None:
                limit.warning_threshold = warning_threshold
            if enabled is not None:
                limit.enabled = enabled
            
            self.logger.info(f"更新集中度限制: {concentration_type.value}, "
                           f"最大权重={limit.max_weight:.2%}, "
                           f"警告阈值={limit.warning_threshold:.2%}, "
                           f"启用={limit.enabled}")
    
    def reset_history(self) -> None:
        """重置历史记录"""
        self.violation_history.clear()
        self.metrics_history.clear()
        self.adjustment_history.clear()
        self.current_violations.clear()
        self.logger.info("集中度控制历史记录已重置")