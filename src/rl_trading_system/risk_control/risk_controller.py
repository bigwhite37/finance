"""风险控制模块实现"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import logging
from abc import ABC, abstractmethod


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskViolationType(Enum):
    """风险违规类型枚举"""
    POSITION_CONCENTRATION = "position_concentration"
    SECTOR_EXPOSURE = "sector_exposure"
    STOP_LOSS = "stop_loss"
    ANOMALOUS_TRADE = "anomalous_trade"
    LEVERAGE = "leverage"
    LIQUIDITY = "liquidity"


@dataclass
class TradeDecision:
    """交易决策数据类"""
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    target_price: float
    sector: str
    timestamp: datetime
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class Trade:
    """交易数据类"""
    symbol: str
    quantity: float
    price: float
    timestamp: datetime
    action: str
    sector: str
    
    def __init__(self, symbol: str, quantity: float, price: float, timestamp: datetime, 
                 action: str = "BUY", sector: str = "Unknown"):
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        self.timestamp = timestamp
        self.action = action
        self.sector = sector


@dataclass
class Position:
    """持仓数据类"""
    symbol: str
    quantity: int
    current_price: float
    sector: str
    timestamp: datetime
    cost_basis: float = None
    
    def __post_init__(self):
        if self.cost_basis is None:
            self.cost_basis = self.current_price
        
        self.market_value = self.quantity * self.current_price
        self.unrealized_pnl = (self.current_price - self.cost_basis) * self.quantity
        self.weight = 0.0  # Will be calculated by portfolio


@dataclass
class Portfolio:
    """投资组合数据类"""
    positions: List[Position]
    cash: float
    total_value: float
    timestamp: datetime
    
    def __post_init__(self):
        # Convert list to dict for easier access
        self._position_dict = {pos.symbol: pos for pos in self.positions}
        
        # Calculate weights
        if self.total_value > 0:
            for position in self.positions:
                position.weight = position.market_value / self.total_value
    
    @property
    def position_count(self) -> int:
        return len([p for p in self.positions if p.quantity > 0])
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定股票的持仓"""
        return self._position_dict.get(symbol)


@dataclass
class RiskViolation:
    """风险违规数据类"""
    violation_type: RiskViolationType
    severity: RiskLevel
    message: str
    details: Dict[str, Any]
    timestamp: datetime


@dataclass
class RiskRule:
    """风险规则数据类"""
    rule_id: str
    rule_name: str
    risk_level: RiskLevel
    check_function: Callable
    description: str
    enabled: bool = True


@dataclass
class RiskControlConfig:
    """风险控制配置"""
    # 持仓集中度限制
    max_position_weight: float = 0.1         # 单个持仓最大权重
    max_portfolio_concentration: float = 0.6  # 投资组合最大集中度
    max_sector_exposure: float = 0.3         # 单个行业最大暴露
    
    # 损失控制
    stop_loss_threshold: float = 0.05        # 止损阈值
    max_daily_loss: float = 0.02             # 最大日损失
    max_drawdown: float = 0.1                # 最大回撤
    
    # 流动性和杠杆
    min_cash_ratio: float = 0.05             # 最小现金比例
    max_leverage: float = 2.0                # 最大杠杆
    
    # 市场风险
    volatility_threshold: float = 0.3        # 波动率阈值
    liquidity_threshold: float = 1000000     # 流动性阈值
    
    # 异常交易检测 (保持原有配置)
    volume_anomaly_threshold: float = 3.0    # 交易量异常阈值（标准差倍数）
    price_anomaly_threshold: float = 2.5     # 价格异常阈值
    frequency_limit: int = 100               # 频率限制（每分钟）
    trade_size_limit: float = 0.05           # 单笔交易规模限制
    
    # 内部兼容性属性
    @property
    def herfindahl_threshold(self) -> float:
        return 0.2
    
    @property
    def max_single_position_weight(self) -> float:
        return self.max_position_weight
    
    @property
    def portfolio_stop_loss(self) -> float:
        return -self.max_drawdown
    
    @property
    def trailing_stop_distance(self) -> float:
        return 0.03
    
    @property
    def min_liquidity_ratio(self) -> float:
        return self.min_cash_ratio
    
    @property
    def sector_concentration_limit(self) -> int:
        return 3


class RiskController:
    """主风险控制器"""
    
    def __init__(self, config: RiskControlConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化子控制器
        self.position_controller = PositionConcentrationController(config)
        self.sector_controller = SectorExposureController(config)
        self.stop_loss_controller = StopLossController(config)
        self.anomaly_detector = AnomalousTradeDetector(config)
        
        # 风险状态跟踪
        self.risk_violations: List[RiskViolation] = []
        self.violations_history: List[RiskViolation] = []
        self.last_assessment_time = None
        
        # 自定义风险规则
        self.custom_rules: List[RiskRule] = []
        self.risk_rules: List[RiskRule] = self._initialize_default_rules()
    
    def add_risk_rule(self, rule: RiskRule):
        """添加自定义风险规则"""
        self.custom_rules.append(rule)
    
    def remove_risk_rule(self, rule_id: str):
        """移除风险规则"""
        self.custom_rules = [rule for rule in self.custom_rules if rule.rule_id != rule_id]
        self.risk_rules = [rule for rule in self.risk_rules if rule.rule_id != rule_id]
    
    def _initialize_default_rules(self) -> List[RiskRule]:
        """初始化默认风险规则"""
        default_rules = [
            RiskRule(
                rule_id="max_position_weight",
                rule_name="最大持仓权重限制",
                risk_level=RiskLevel.HIGH,
                check_function=lambda portfolio, trade: True,  # 简化实现
                description="限制单个持仓的最大权重"
            ),
            RiskRule(
                rule_id="sector_exposure_limit",
                rule_name="行业暴露限制",
                risk_level=RiskLevel.MEDIUM,
                check_function=lambda portfolio, trade: True,
                description="限制单个行业的最大暴露"
            ),
            RiskRule(
                rule_id="stop_loss_check",
                rule_name="止损检查",
                risk_level=RiskLevel.HIGH,
                check_function=lambda portfolio, trade: True,
                description="检查是否触发止损条件"
            )
        ]
        return default_rules
    
    def check_trade_risk(self, trade_decision: TradeDecision, 
                        current_portfolio: Portfolio) -> Dict[str, Any]:
        """检查交易风险"""
        violations = []
        risk_score = 0.0
        recommendations = []
        
        # 检查持仓集中度风险
        concentration_result = self.position_controller.check_concentration_risk(
            trade_decision, current_portfolio
        )
        if not concentration_result['approved']:
            violations.extend(concentration_result['violations'])
            risk_score += concentration_result['risk_score']
        
        # 检查行业暴露风险
        sector_result = self.sector_controller.check_sector_risk(
            trade_decision, current_portfolio
        )
        if not sector_result['approved']:
            violations.extend(sector_result['violations'])
            risk_score += sector_result['risk_score']
        
        # 检查止损风险
        stop_loss_result = self.stop_loss_controller.check_stop_loss(
            trade_decision, current_portfolio
        )
        if not stop_loss_result['approved']:
            violations.extend(stop_loss_result['violations'])
            risk_score += stop_loss_result['risk_score']
        
        # 检查异常交易
        anomaly_result = self.anomaly_detector.detect_anomaly(
            trade_decision, current_portfolio
        )
        if not anomaly_result['approved']:
            violations.extend(anomaly_result['violations'])
            risk_score += anomaly_result['risk_score']
        
        # 生成建议
        if violations:
            recommendations = self._generate_recommendations(violations, trade_decision)
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'risk_score': min(risk_score, 10.0),  # 限制在10分以内
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
    
    def assess_portfolio_risk(self, portfolio: Portfolio) -> Dict[str, Any]:
        """评估投资组合整体风险"""
        self.last_assessment_time = datetime.now()
        
        # 计算各类风险指标
        concentration_risk = self._calculate_concentration_risk(portfolio)
        sector_risk = self._calculate_sector_risk(portfolio)
        volatility_risk = self._calculate_volatility_risk(portfolio)
        liquidity_risk = self._calculate_liquidity_risk(portfolio)
        
        # 综合风险评分 (0-10分)
        total_risk_score = (
            concentration_risk * 0.3 +
            sector_risk * 0.25 +
            volatility_risk * 0.25 +
            liquidity_risk * 0.2
        )
        
        # 确定风险等级
        if total_risk_score < 3:
            risk_level = RiskLevel.LOW
        elif total_risk_score < 6:
            risk_level = RiskLevel.MEDIUM
        elif total_risk_score < 8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        return {
            'total_risk_score': total_risk_score,
            'risk_level': risk_level,
            'concentration_risk': concentration_risk,
            'sector_risk': sector_risk,
            'volatility_risk': volatility_risk,
            'liquidity_risk': liquidity_risk,
            'position_count': portfolio.position_count,
            'timestamp': self.last_assessment_time
        }
    
    def _generate_recommendations(self, violations: List[RiskViolation], 
                                trade_decision: TradeDecision) -> List[str]:
        """生成风险控制建议"""
        recommendations = []
        
        for violation in violations:
            if violation.violation_type == RiskViolationType.POSITION_CONCENTRATION:
                recommendations.append(f"建议减少{trade_decision.symbol}的持仓规模")
            elif violation.violation_type == RiskViolationType.SECTOR_EXPOSURE:
                recommendations.append(f"建议减少{trade_decision.sector}行业的暴露")
            elif violation.violation_type == RiskViolationType.STOP_LOSS:
                recommendations.append("建议执行止损操作")
            elif violation.violation_type == RiskViolationType.ANOMALOUS_TRADE:
                recommendations.append("建议暂停交易，检查系统状态")
        
        return recommendations
    
    def _calculate_concentration_risk(self, portfolio: Portfolio) -> float:
        """计算集中度风险"""
        if not portfolio.positions:
            return 0.0
        
        weights = [pos.weight for pos in portfolio.positions]
        max_weight = max(weights) if weights else 0
        herfindahl_index = sum(w**2 for w in weights)
        
        # 基于最大权重和赫芬达尔指数评分
        max_weight_score = min(max_weight / self.config.max_single_position_weight * 5, 10)
        herfindahl_score = min(herfindahl_index / self.config.herfindahl_threshold * 5, 10)
        
        return (max_weight_score + herfindahl_score) / 2
    
    def _calculate_sector_risk(self, portfolio: Portfolio) -> float:
        """计算行业风险"""
        if not portfolio.positions:
            return 0.0
        
        sector_exposure = {}
        for position in portfolio.positions:
            sector = position.sector
            sector_exposure[sector] = sector_exposure.get(sector, 0) + position.weight
        
        max_sector_exposure = max(sector_exposure.values()) if sector_exposure else 0
        return min(max_sector_exposure / self.config.max_sector_exposure * 10, 10)
    
    def _calculate_volatility_risk(self, portfolio: Portfolio) -> float:
        """计算波动率风险"""
        if not portfolio.positions:
            return 0.0
        
        # 简化的波动率计算 - 基于未实现损益
        total_pnl = sum(pos.unrealized_pnl for pos in portfolio.positions)
        if portfolio.total_value > 0:
            pnl_ratio = abs(total_pnl) / portfolio.total_value
            return min(pnl_ratio * 20, 10)  # 5%损益对应满分
        return 0.0
    
    def _calculate_liquidity_risk(self, portfolio: Portfolio) -> float:
        """计算流动性风险"""
        if not portfolio.positions:
            return 0.0
        
        # 基于现金比例评估流动性风险
        cash_ratio = portfolio.cash / portfolio.total_value if portfolio.total_value > 0 else 0
        
        if cash_ratio >= self.config.min_liquidity_ratio:
            return 0.0
        else:
            # 现金比例越低，流动性风险越高
            return min((self.config.min_liquidity_ratio - cash_ratio) * 50, 10)


class PositionConcentrationController:
    """持仓集中度控制器"""
    
    def __init__(self, config: RiskControlConfig):
        self.config = config
    
    def check_concentration_risk(self, trade_decision: TradeDecision,
                               portfolio: Portfolio) -> Dict[str, Any]:
        """检查持仓集中度风险"""
        violations = []
        risk_score = 0.0
        
        # 模拟交易后的持仓权重
        new_portfolio = self._simulate_trade(trade_decision, portfolio)
        
        # 检查单个持仓权重
        position = new_portfolio.get_position(trade_decision.symbol)
        if position:
            new_weight = position.weight
            if new_weight > self.config.max_single_position_weight:
                violation = RiskViolation(
                    violation_type=RiskViolationType.POSITION_CONCENTRATION,
                    severity=RiskLevel.HIGH,
                    message=f"持仓{trade_decision.symbol}权重{new_weight:.2%}超过限制{self.config.max_single_position_weight:.2%}",
                    details={'symbol': trade_decision.symbol, 'weight': new_weight},
                    timestamp=datetime.now()
                )
                violations.append(violation)
                risk_score += 3.0
        
        # 检查赫芬达尔指数
        herfindahl_index = self.calculate_herfindahl_index(new_portfolio)
        if herfindahl_index > self.config.herfindahl_threshold:
            violation = RiskViolation(
                violation_type=RiskViolationType.POSITION_CONCENTRATION,
                severity=RiskLevel.MEDIUM,
                message=f"投资组合集中度指数{herfindahl_index:.3f}超过阈值{self.config.herfindahl_threshold:.3f}",
                details={'herfindahl_index': herfindahl_index},
                timestamp=datetime.now()
            )
            violations.append(violation)
            risk_score += 2.0
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'risk_score': risk_score,
            'herfindahl_index': herfindahl_index
        }
    
    def calculate_herfindahl_index(self, portfolio: Portfolio) -> float:
        """计算赫芬达尔指数"""
        if not portfolio.positions:
            return 0.0
        
        weights = [pos.weight for pos in portfolio.positions if pos.quantity > 0]
        return sum(w**2 for w in weights)
    
    def check_concentration_limits(self, portfolio: Portfolio) -> Dict[str, Any]:
        """检查集中度限制"""
        if not portfolio.positions:
            return {'within_limits': True, 'max_weight': 0.0, 'herfindahl_index': 0.0}
        
        weights = [pos.weight for pos in portfolio.positions if pos.quantity > 0]
        max_weight = max(weights) if weights else 0.0
        herfindahl_index = self.calculate_herfindahl_index(portfolio)
        
        within_limits = (
            max_weight <= self.config.max_single_position_weight and
            herfindahl_index <= self.config.herfindahl_threshold
        )
        
        return {
            'within_limits': within_limits,
            'max_weight': max_weight,
            'herfindahl_index': herfindahl_index,
            'violations': []
        }
    
    def _simulate_trade(self, trade_decision: TradeDecision, 
                      portfolio: Portfolio) -> Portfolio:
        """模拟交易后的投资组合"""
        # 创建新的持仓列表副本
        new_positions = []
        new_cash = portfolio.cash
        trade_value = trade_decision.quantity * trade_decision.target_price
        
        # 查找是否已有该股票的持仓
        existing_position = None
        for pos in portfolio.positions:
            if pos.symbol == trade_decision.symbol:
                existing_position = pos
                break
        
        if trade_decision.action == "BUY":
            new_cash -= trade_value
            if existing_position:
                # 更新现有持仓
                total_quantity = existing_position.quantity + trade_decision.quantity
                new_cost_basis = (existing_position.cost_basis * existing_position.quantity + trade_value) / total_quantity
                
                updated_position = Position(
                    symbol=existing_position.symbol,
                    quantity=total_quantity,
                    current_price=existing_position.current_price,
                    sector=existing_position.sector,
                    timestamp=existing_position.timestamp,
                    cost_basis=new_cost_basis
                )
                
                # 添加所有持仓（包括更新的持仓）
                for pos in portfolio.positions:
                    if pos.symbol == trade_decision.symbol:
                        new_positions.append(updated_position)
                    else:
                        new_positions.append(pos)
            else:
                # 添加新持仓
                new_position = Position(
                    symbol=trade_decision.symbol,
                    quantity=trade_decision.quantity,
                    current_price=trade_decision.target_price,
                    sector=trade_decision.sector,
                    timestamp=datetime.now(),
                    cost_basis=trade_decision.target_price
                )
                new_positions = portfolio.positions + [new_position]
        else:
            # 复制现有持仓
            new_positions = portfolio.positions.copy()
        
        # 重新计算总价值
        new_total_value = new_cash + sum(pos.market_value for pos in new_positions)
        
        return Portfolio(
            positions=new_positions,
            cash=new_cash,
            total_value=new_total_value,
            timestamp=datetime.now()
        )


class SectorExposureController:
    """行业暴露控制器"""
    
    def __init__(self, config: RiskControlConfig):
        self.config = config
    
    def check_sector_risk(self, trade_decision: TradeDecision,
                         portfolio: Portfolio) -> Dict[str, Any]:
        """检查行业暴露风险"""
        violations = []
        risk_score = 0.0
        
        # 计算当前行业暴露
        current_exposure = self.calculate_sector_exposure(portfolio)
        
        # 模拟交易后的行业暴露
        new_portfolio = self._simulate_trade(trade_decision, portfolio)
        new_exposure = self.calculate_sector_exposure(new_portfolio)
        
        # 检查目标行业暴露是否超限
        target_sector = trade_decision.sector
        if target_sector in new_exposure:
            sector_exposure = new_exposure[target_sector]
            if sector_exposure > self.config.max_sector_exposure:
                violation = RiskViolation(
                    violation_type=RiskViolationType.SECTOR_EXPOSURE,
                    severity=RiskLevel.HIGH,
                    message=f"行业{target_sector}暴露{sector_exposure:.2%}超过限制{self.config.max_sector_exposure:.2%}",
                    details={'sector': target_sector, 'exposure': sector_exposure},
                    timestamp=datetime.now()
                )
                violations.append(violation)
                risk_score += 3.0
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'risk_score': risk_score,
            'current_exposure': current_exposure,
            'new_exposure': new_exposure
        }
    
    def calculate_sector_exposure(self, portfolio: Portfolio) -> Dict[str, float]:
        """计算行业暴露度"""
        if not portfolio.positions:
            return {}
        
        sector_exposure = {}
        for position in portfolio.positions:
            if position.quantity > 0:
                sector = position.sector
                sector_exposure[sector] = sector_exposure.get(sector, 0) + position.weight
        
        return sector_exposure
    
    def check_diversification(self, portfolio: Portfolio) -> Dict[str, Any]:
        """检查多元化程度"""
        sector_exposure = self.calculate_sector_exposure(portfolio)
        
        if not sector_exposure:
            return {
                'diversified': True,
                'sector_count': 0,
                'max_exposure': 0.0,
                'concentration_score': 0.0
            }
        
        sector_count = len(sector_exposure)
        max_exposure = max(sector_exposure.values())
        
        # 计算行业集中度评分 (0-10分)
        concentration_score = 0.0
        if max_exposure > self.config.max_sector_exposure:
            concentration_score += (max_exposure - self.config.max_sector_exposure) * 20
        
        # 行业数量过少也会增加集中度风险
        if sector_count < self.config.sector_concentration_limit:
            concentration_score += (self.config.sector_concentration_limit - sector_count) * 2
        
        diversified = (
            max_exposure <= self.config.max_sector_exposure and
            sector_count >= self.config.sector_concentration_limit
        )
        
        return {
            'diversified': diversified,
            'sector_count': sector_count,
            'max_exposure': max_exposure,
            'concentration_score': min(concentration_score, 10.0),
            'sector_exposure': sector_exposure
        }
    
    def _simulate_trade(self, trade_decision: TradeDecision, 
                      portfolio: Portfolio) -> Portfolio:
        """模拟交易后的投资组合 (复用PositionConcentrationController的逻辑)"""
        # 这里可以复用PositionConcentrationController的_simulate_trade方法
        # 为了简化，这里使用相同的逻辑
        controller = PositionConcentrationController(self.config)
        return controller._simulate_trade(trade_decision, portfolio)


class StopLossController:
    """止损控制器"""
    
    def __init__(self, config: RiskControlConfig):
        self.config = config
        self.trailing_stops: Dict[str, float] = {}  # 追踪止损价格
    
    def check_stop_loss(self, trade_decision: TradeDecision,
                       portfolio: Portfolio) -> Dict[str, Any]:
        """检查止损条件"""
        violations = []
        risk_score = 0.0
        
        # 检查单个持仓止损
        position = None
        for pos in portfolio.positions:
            if pos.symbol == trade_decision.symbol:
                position = pos
                break
        
        if position:
            pnl_ratio = (position.current_price - position.cost_basis) / position.cost_basis
            
            if pnl_ratio <= -self.config.stop_loss_threshold:
                violation = RiskViolation(
                    violation_type=RiskViolationType.STOP_LOSS,
                    severity=RiskLevel.HIGH,
                    message=f"持仓{trade_decision.symbol}亏损{pnl_ratio:.2%}触发止损",
                    details={'symbol': trade_decision.symbol, 'pnl_ratio': pnl_ratio},
                    timestamp=datetime.now()
                )
                violations.append(violation)
                risk_score += 4.0
        
        # 检查组合级止损
        portfolio_pnl = sum(pos.unrealized_pnl for pos in portfolio.positions)
        portfolio_pnl_ratio = portfolio_pnl / portfolio.total_value if portfolio.total_value > 0 else 0
        
        if portfolio_pnl_ratio <= self.config.portfolio_stop_loss:
            violation = RiskViolation(
                violation_type=RiskViolationType.STOP_LOSS,
                severity=RiskLevel.CRITICAL,
                message=f"投资组合亏损{portfolio_pnl_ratio:.2%}触发组合止损",
                details={'portfolio_pnl_ratio': portfolio_pnl_ratio},
                timestamp=datetime.now()
            )
            violations.append(violation)
            risk_score += 5.0
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'risk_score': risk_score,
            'portfolio_pnl_ratio': portfolio_pnl_ratio
        }
    
    def update_trailing_stops(self, portfolio: Portfolio) -> Dict[str, float]:
        """更新追踪止损价格"""
        updated_stops = {}
        
        for position in portfolio.positions:
            if position.quantity <= 0:
                continue
                
            symbol = position.symbol
            current_price = position.current_price
            
            # 计算新的追踪止损价格
            if position.unrealized_pnl > 0:  # 只有盈利时才设置追踪止损
                new_stop_price = current_price * (1 - self.config.trailing_stop_distance)
                
                # 更新追踪止损价格（只能向上调整）
                if symbol not in self.trailing_stops or new_stop_price > self.trailing_stops[symbol]:
                    self.trailing_stops[symbol] = new_stop_price
                    updated_stops[symbol] = new_stop_price
        
        return updated_stops
    
    def check_trailing_stops(self, portfolio: Portfolio) -> Dict[str, Any]:
        """检查追踪止损触发"""
        triggered_stops = {}
        
        for symbol, stop_price in self.trailing_stops.items():
            position = portfolio.get_position(symbol)
            if position and position.current_price <= stop_price:
                triggered_stops[symbol] = {
                    'stop_price': stop_price,
                    'current_price': position.current_price,
                    'trigger_time': datetime.now()
                }
        
        return {
            'triggered_stops': triggered_stops,
            'symbols_to_sell': list(triggered_stops.keys())
        }
    
    def calculate_stop_loss_levels(self, portfolio: Portfolio) -> Dict[str, Dict[str, float]]:
        """计算各持仓的止损水平"""
        stop_levels = {}
        
        for position in portfolio.positions:
            if position.quantity <= 0:
                continue
            
            symbol = position.symbol
            # 固定止损
            fixed_stop = position.cost_basis * (1 - self.config.stop_loss_threshold)
            
            # 追踪止损
            trailing_stop = self.trailing_stops.get(symbol, 0.0)
            
            # 波动率调整止损 (简化版)
            volatility_adjusted_stop = position.cost_basis * (1 - self.config.stop_loss_threshold * 1.5)
            
            stop_levels[symbol] = {
                'fixed_stop': fixed_stop,
                'trailing_stop': trailing_stop,
                'volatility_adjusted_stop': volatility_adjusted_stop,
                'current_price': position.current_price
            }
        
        return stop_levels


class AnomalousTradeDetector:
    """异常交易检测器"""
    
    def __init__(self, config: RiskControlConfig):
        self.config = config
        self.trade_history: List[Dict] = []
        self.volume_stats: Dict[str, List[float]] = {}
        self.price_stats: Dict[str, List[float]] = {}
    
    def detect_anomaly(self, trade_decision: TradeDecision,
                      portfolio: Portfolio) -> Dict[str, Any]:
        """检测异常交易"""
        violations = []
        risk_score = 0.0
        
        # 检查交易量异常
        volume_anomaly = self._check_volume_anomaly(trade_decision)
        if volume_anomaly['is_anomaly']:
            violation = RiskViolation(
                violation_type=RiskViolationType.ANOMALOUS_TRADE,
                severity=RiskLevel.MEDIUM,
                message=f"交易量异常: {trade_decision.symbol}交易量超过{self.config.volume_anomaly_threshold}倍标准差",
                details=volume_anomaly,
                timestamp=datetime.now()
            )
            violations.append(violation)
            risk_score += 2.0
        
        # 检查价格异常
        price_anomaly = self._check_price_anomaly(trade_decision)
        if price_anomaly['is_anomaly']:
            violation = RiskViolation(
                violation_type=RiskViolationType.ANOMALOUS_TRADE,
                severity=RiskLevel.MEDIUM,
                message=f"价格异常: {trade_decision.symbol}价格偏离{self.config.price_anomaly_threshold}倍标准差",
                details=price_anomaly,
                timestamp=datetime.now()
            )
            violations.append(violation)
            risk_score += 2.0
        
        # 检查交易频率
        frequency_anomaly = self._check_frequency_anomaly(trade_decision)
        if frequency_anomaly['is_anomaly']:
            violation = RiskViolation(
                violation_type=RiskViolationType.ANOMALOUS_TRADE,
                severity=RiskLevel.HIGH,
                message=f"交易频率异常: 每分钟交易次数{frequency_anomaly['current_frequency']}超过限制{self.config.frequency_limit}",
                details=frequency_anomaly,
                timestamp=datetime.now()
            )
            violations.append(violation)
            risk_score += 3.0
        
        # 检查单笔交易规模
        size_anomaly = self._check_trade_size_anomaly(trade_decision, portfolio)
        if size_anomaly['is_anomaly']:
            violation = RiskViolation(
                violation_type=RiskViolationType.ANOMALOUS_TRADE,
                severity=RiskLevel.HIGH,
                message=f"交易规模异常: 单笔交易规模{size_anomaly['trade_ratio']:.2%}超过限制{self.config.trade_size_limit:.2%}",
                details=size_anomaly,
                timestamp=datetime.now()
            )
            violations.append(violation)
            risk_score += 3.0
        
        # 记录交易历史
        self._record_trade(trade_decision)
        
        return {
            'approved': len(violations) == 0,
            'violations': violations,
            'risk_score': risk_score,
            'anomaly_details': {
                'volume': volume_anomaly,
                'price': price_anomaly,
                'frequency': frequency_anomaly,
                'size': size_anomaly
            }
        }
    
    def _check_volume_anomaly(self, trade_decision: TradeDecision) -> Dict[str, Any]:
        """检查交易量异常"""
        symbol = trade_decision.symbol
        volume = trade_decision.quantity
        
        if symbol not in self.volume_stats or len(self.volume_stats[symbol]) < 10:
            return {'is_anomaly': False, 'z_score': 0.0}
        
        volumes = self.volume_stats[symbol]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        if std_volume == 0:
            z_score = 0.0
        else:
            z_score = abs(volume - mean_volume) / std_volume
        
        is_anomaly = z_score > self.config.volume_anomaly_threshold
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'current_volume': volume,
            'mean_volume': mean_volume,
            'std_volume': std_volume
        }
    
    def _check_price_anomaly(self, trade_decision: TradeDecision) -> Dict[str, Any]:
        """检查价格异常"""
        symbol = trade_decision.symbol
        price = trade_decision.target_price
        
        if symbol not in self.price_stats or len(self.price_stats[symbol]) < 10:
            return {'is_anomaly': False, 'z_score': 0.0}
        
        prices = self.price_stats[symbol]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            z_score = 0.0
        else:
            z_score = abs(price - mean_price) / std_price
        
        is_anomaly = z_score > self.config.price_anomaly_threshold
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'current_price': price,
            'mean_price': mean_price,
            'std_price': std_price
        }
    
    def _check_frequency_anomaly(self, trade_decision: TradeDecision) -> Dict[str, Any]:
        """检查交易频率异常"""
        current_time = trade_decision.timestamp
        one_minute_ago = current_time - timedelta(minutes=1)
        
        # 统计过去一分钟的交易次数
        recent_trades = [
            trade for trade in self.trade_history
            if trade['timestamp'] > one_minute_ago
        ]
        
        current_frequency = len(recent_trades) + 1  # 包括当前交易
        is_anomaly = current_frequency > self.config.frequency_limit
        
        return {
            'is_anomaly': is_anomaly,
            'current_frequency': current_frequency,
            'limit': self.config.frequency_limit,
            'time_window': '1分钟'
        }
    
    def _check_trade_size_anomaly(self, trade_decision: TradeDecision,
                                 portfolio: Portfolio) -> Dict[str, Any]:
        """检查单笔交易规模异常"""
        trade_value = trade_decision.quantity * trade_decision.target_price
        
        if portfolio.total_value <= 0:
            return {'is_anomaly': False, 'trade_ratio': 0.0}
        
        trade_ratio = trade_value / portfolio.total_value
        is_anomaly = trade_ratio > self.config.trade_size_limit
        
        return {
            'is_anomaly': is_anomaly,
            'trade_ratio': trade_ratio,
            'trade_value': trade_value,
            'portfolio_value': portfolio.total_value,
            'limit': self.config.trade_size_limit
        }
    
    def _record_trade(self, trade_decision: TradeDecision):
        """记录交易历史"""
        trade_record = {
            'symbol': trade_decision.symbol,
            'action': trade_decision.action,
            'quantity': trade_decision.quantity,
            'price': trade_decision.target_price,
            'timestamp': trade_decision.timestamp
        }
        
        self.trade_history.append(trade_record)
        
        # 更新统计数据
        symbol = trade_decision.symbol
        
        if symbol not in self.volume_stats:
            self.volume_stats[symbol] = []
        if symbol not in self.price_stats:
            self.price_stats[symbol] = []
        
        self.volume_stats[symbol].append(trade_decision.quantity)
        self.price_stats[symbol].append(trade_decision.target_price)
        
        # 保持历史数据在合理范围内
        max_history = 1000
        if len(self.volume_stats[symbol]) > max_history:
            self.volume_stats[symbol] = self.volume_stats[symbol][-max_history:]
        if len(self.price_stats[symbol]) > max_history:
            self.price_stats[symbol] = self.price_stats[symbol][-max_history:]
        
        # 清理过期的交易历史 (保留24小时)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.trade_history = [
            trade for trade in self.trade_history
            if trade['timestamp'] > cutoff_time
        ]
    
    def detect_ml_anomaly(self, trade_decision: TradeDecision,
                         portfolio: Portfolio) -> Dict[str, Any]:
        """基于机器学习的异常检测"""
        # 提取特征
        features = self._extract_features(trade_decision, portfolio)
        
        # 简化的异常检测逻辑 (实际应该使用训练好的ML模型)
        anomaly_score = self._calculate_anomaly_score(features)
        is_anomaly = anomaly_score > 0.7  # 阈值
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'features': features,
            'method': 'isolation_forest'
        }
    
    def _extract_features(self, trade_decision: TradeDecision,
                         portfolio: Portfolio) -> Dict[str, float]:
        """提取异常检测特征"""
        # 计算各种特征
        trade_value = trade_decision.quantity * trade_decision.target_price
        portfolio_ratio = trade_value / portfolio.total_value if portfolio.total_value > 0 else 0
        
        # 时间特征
        hour = trade_decision.timestamp.hour
        is_trading_hours = 9 <= hour <= 15
        
        # 持仓特征
        position_count = portfolio.position_count
        
        return {
            'trade_ratio': portfolio_ratio,
            'quantity_log': np.log(trade_decision.quantity + 1),
            'price_log': np.log(trade_decision.target_price),
            'hour': hour,
            'is_trading_hours': float(is_trading_hours),
            'position_count': position_count,
            'confidence': trade_decision.confidence
        }
    
    def _calculate_anomaly_score(self, features: Dict[str, float]) -> float:
        """计算异常得分 (简化版)"""
        # 简化的异常得分计算
        score = 0.0
        
        # 交易规模异常
        if features['trade_ratio'] > 0.1:
            score += 0.3
        
        # 非交易时间异常
        if not features['is_trading_hours']:
            score += 0.2
        
        # 交易量异常
        if features['quantity_log'] > 10:  # 约22000股
            score += 0.2
        
        # 价格异常
        if features['price_log'] > 8:  # 约3000元
            score += 0.1
        
        # 置信度异常
        if features['confidence'] < 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def detect_volume_anomaly(self, trade: Trade) -> Dict[str, Any]:
        """检测交易量异常"""
        symbol = trade.symbol
        volume = trade.quantity
        
        if symbol not in self.volume_stats or len(self.volume_stats[symbol]) < 10:
            return {'is_anomaly': False, 'z_score': 0.0, 'reason': 'insufficient_data'}
        
        volumes = self.volume_stats[symbol]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        if std_volume == 0:
            z_score = 0.0
        else:
            z_score = abs(volume - mean_volume) / std_volume
        
        is_anomaly = z_score > self.config.volume_anomaly_threshold
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'current_volume': volume,
            'mean_volume': mean_volume,
            'std_volume': std_volume,
            'threshold': self.config.volume_anomaly_threshold,
            'reason': 'volume_spike' if is_anomaly else 'normal'
        }
    
    def detect_price_anomaly(self, trade: Trade) -> Dict[str, Any]:
        """检测价格异常"""
        symbol = trade.symbol
        price = trade.price
        
        if symbol not in self.price_stats or len(self.price_stats[symbol]) < 10:
            return {'is_anomaly': False, 'z_score': 0.0, 'reason': 'insufficient_data'}
        
        prices = self.price_stats[symbol]
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            z_score = 0.0
        else:
            z_score = abs(price - mean_price) / std_price
        
        is_anomaly = z_score > self.config.price_anomaly_threshold
        
        return {
            'is_anomaly': is_anomaly,
            'z_score': z_score,
            'current_price': price,
            'mean_price': mean_price,
            'std_price': std_price,
            'threshold': self.config.price_anomaly_threshold,
            'reason': 'price_deviation' if is_anomaly else 'normal'
        }
    
    def detect_frequency_anomaly(self, trades: List[Trade], time_window_hours: int = 1) -> Dict[str, Any]:
        """检测交易频率异常"""
        if not trades:
            return {'is_anomaly': False, 'frequency': 0, 'reason': 'no_trades'}
        
        # 计算指定时间窗口内的交易频率
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=time_window_hours)
        
        recent_trades = [
            trade for trade in trades
            if hasattr(trade, 'timestamp') and trade.timestamp > cutoff_time
        ]
        
        frequency = len(recent_trades)
        frequency_per_minute = frequency / (time_window_hours * 60)
        
        is_anomaly = frequency_per_minute > self.config.frequency_limit / 60  # 转换为每分钟
        
        return {
            'is_anomaly': is_anomaly,
            'frequency': frequency,
            'frequency_per_minute': frequency_per_minute,
            'time_window_hours': time_window_hours,
            'threshold_per_minute': self.config.frequency_limit / 60,
            'reason': 'high_frequency' if is_anomaly else 'normal'
        }
    
    def detect_size_anomaly(self, trade: Trade) -> Dict[str, Any]:
        """检测交易规模异常"""
        trade_value = trade.quantity * trade.price
        
        # 简化版：基于固定阈值检测
        large_trade_threshold = 1000000  # 100万
        
        is_anomaly = trade_value > large_trade_threshold
        
        return {
            'is_anomaly': is_anomaly,
            'trade_value': trade_value,
            'threshold': large_trade_threshold,
            'ratio_to_threshold': trade_value / large_trade_threshold if large_trade_threshold > 0 else 0,
            'reason': 'large_trade' if is_anomaly else 'normal'
        }
    
    def detect_pattern_anomaly(self, trades: List[Trade]) -> Dict[str, Any]:
        """检测交易模式异常"""
        if len(trades) < 3:
            return {'is_anomaly': False, 'pattern': 'insufficient_data'}
        
        # 检查是否存在可疑的交易模式
        patterns_detected = []
        
        # 1. 检查连续同向交易
        same_direction_count = 0
        prev_action = None
        for trade in trades:
            if trade.action == prev_action:
                same_direction_count += 1
            else:
                same_direction_count = 1
            prev_action = trade.action
            
            if same_direction_count > 10:  # 连续10次同向交易
                patterns_detected.append('consecutive_same_direction')
                break
        
        # 2. 检查价格操纵模式（快速上下波动）
        if len(trades) >= 5:
            prices = [trade.price for trade in trades[-5:]]
            price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            if all(change > 0.02 for change in price_changes):  # 连续2%以上波动
                patterns_detected.append('price_manipulation')
        
        # 3. 检查wash trading（自买自卖）
        symbols = [trade.symbol for trade in trades]
        if len(set(symbols)) == 1 and len(trades) > 20:  # 同一股票大量交易
            buy_sell_ratio = sum(1 for t in trades if t.action == 'BUY') / len(trades)
            if 0.4 < buy_sell_ratio < 0.6:  # 买卖比例接近
                patterns_detected.append('wash_trading')
        
        is_anomaly = len(patterns_detected) > 0
        
        return {
            'is_anomaly': is_anomaly,
            'patterns_detected': patterns_detected,
            'pattern_count': len(patterns_detected),
            'reason': 'suspicious_pattern' if is_anomaly else 'normal'
        }
    
    def detect_anomaly_with_context(self, trade: Trade, context_info: Dict[str, Any]) -> Dict[str, Any]:
        """基于上下文信息的异常检测"""
        base_result = self.detect_ml_anomaly(None, None)  # 基础ML检测
        
        # 根据上下文调整异常评分
        adjusted_score = base_result.get('anomaly_score', 0.5)
        
        # 考虑市场条件
        if context_info.get('market_volatility', 0) > 0.3:
            adjusted_score *= 0.8  # 高波动市场中降低敏感度
        
        # 考虑新闻事件
        if context_info.get('news_impact', False):
            adjusted_score *= 0.7  # 新闻事件期间降低敏感度
        
        # 考虑交易时间
        if context_info.get('after_hours', False):
            adjusted_score *= 1.2  # 盘后交易提高敏感度
        
        # 考虑历史表现
        if context_info.get('trader_reputation', 1.0) > 0.8:
            adjusted_score *= 0.9  # 高信誉交易者降低敏感度
        
        is_anomaly = adjusted_score > 0.7
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': adjusted_score,
            'base_score': base_result.get('anomaly_score', 0.5),
            'context_adjustments': {
                'market_volatility': context_info.get('market_volatility', 0),
                'news_impact': context_info.get('news_impact', False),
                'after_hours': context_info.get('after_hours', False),
                'trader_reputation': context_info.get('trader_reputation', 1.0)
            },
            'reason': 'contextual_anomaly' if is_anomaly else 'contextually_normal'
        }