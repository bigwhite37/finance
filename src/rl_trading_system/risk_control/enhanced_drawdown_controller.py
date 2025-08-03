"""
增强的回撤控制器

在原有功能基础上添加详细的决策日志和分析功能。
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from collections import Counter, defaultdict

from .drawdown_controller import (
    DrawdownController,
    ControlSignal,
    ControlSignalType,
    MarketState,
    PortfolioState
)
from .enhanced_adaptive_risk_budget import (
    EnhancedAdaptiveRiskBudget,
    EnhancedAdaptiveRiskBudgetConfig
)
from ..backtest.drawdown_control_config import DrawdownControlConfig

logger = logging.getLogger(__name__)


class EnhancedDrawdownController(DrawdownController):
    """
    增强的回撤控制器
    
    在原有功能基础上添加：
    1. 详细的控制决策日志
    2. 市场状态判断的详细记录
    3. 风险预算使用情况的分析
    """
    
    def __init__(self, config: DrawdownControlConfig):
        """
        初始化增强回撤控制器
        
        Args:
            config: 回撤控制配置
        """
        super().__init__(config)
        
        # 增强日志配置
        self.detailed_logging_enabled = True
        
        # 决策历史记录
        self.control_decision_history: List[Dict[str, Any]] = []
        self.market_state_history: List[Dict[str, Any]] = []
        self.risk_budget_decision_history: List[Dict[str, Any]] = []
        
        # 统计信息
        self.decision_type_counter = Counter()
        self.market_regime_changes = 0
        self.last_market_regime = None
        
        # 替换原有的自适应风险预算为增强版本
        if hasattr(self, 'adaptive_risk_budget'):
            enhanced_config = EnhancedAdaptiveRiskBudgetConfig(
                base_risk_budget=config.base_risk_budget,
                enable_detailed_logging=True,
                log_budget_changes=True,
                log_usage_analysis=True
            )
            self.adaptive_risk_budget = EnhancedAdaptiveRiskBudget(enhanced_config)
        
        logger.info("增强回撤控制器初始化完成")
        logger.info("详细决策日志记录已启用")
    
    def execute_control_step(self, 
                           market_state: MarketState,
                           portfolio_state: PortfolioState) -> List[ControlSignal]:
        """
        执行控制步骤（增强版）
        
        Args:
            market_state: 市场状态
            portfolio_state: 投资组合状态
            
        Returns:
            控制信号列表
        """
        # 记录市场状态
        self._record_market_state(market_state)
        
        # 调用父类方法执行控制
        control_signals = super().execute_control_step(market_state, portfolio_state)
        
        # 增强日志记录
        if self.detailed_logging_enabled:
            self._log_control_step_details(market_state, portfolio_state, control_signals)
        
        # 记录决策历史
        self._record_control_decisions(control_signals, market_state, portfolio_state)
        
        return control_signals
    
    def _record_market_state(self, market_state: MarketState):
        """
        记录市场状态历史
        
        Args:
            market_state: 市场状态
        """
        state_record = {
            'timestamp': market_state.timestamp,
            'price_count': len(market_state.prices),
            'avg_price_change': self._calculate_avg_price_change(market_state),
            'total_volume': sum(market_state.volumes.values()) if market_state.volumes else 0,
            'market_indicators': market_state.market_indicators.copy()
        }
        
        self.market_state_history.append(state_record)
        
        # 限制历史记录长度
        if len(self.market_state_history) > 1000:
            self.market_state_history = self.market_state_history[-500:]
    
    def _calculate_avg_price_change(self, market_state: MarketState) -> float:
        """
        计算平均价格变化（简化实现）
        
        Args:
            market_state: 市场状态
            
        Returns:
            平均价格变化
        """
        if not market_state.prices or len(self.market_state_history) == 0:
            return 0.0
        
        # 简化：假设价格变化为随机值
        return np.random.normal(0, 0.01)
    
    def _log_control_step_details(self, 
                                market_state: MarketState,
                                portfolio_state: PortfolioState,
                                control_signals: List[ControlSignal]):
        """
        记录控制步骤详细信息
        
        Args:
            market_state: 市场状态
            portfolio_state: 投资组合状态
            control_signals: 控制信号列表
        """
        # 记录市场状态分析
        if len(control_signals) > 0:  # 只在有控制信号时记录详细信息
            self._log_market_state_analysis(market_state)
            self._log_portfolio_state_analysis(portfolio_state)
            self._log_control_decision_details(control_signals)
    
    def _log_market_state_analysis(self, market_state: MarketState):
        """
        记录市场状态分析
        
        Args:
            market_state: 市场状态
        """
        logger.info("🌍 市场状态分析:")
        logger.info(f"  • 交易品种数量: {len(market_state.prices)}")
        logger.info(f"  • 总交易量: {sum(market_state.volumes.values()):,.0f}")
        
        if market_state.market_indicators:
            logger.info("  • 市场指标:")
            for indicator, value in market_state.market_indicators.items():
                logger.info(f"    - {indicator}: {value:.4f}")
        
        # 检测市场状态变化
        current_regime = self._detect_market_regime(market_state)
        if current_regime != self.last_market_regime:
            logger.info(f"  • 市场状态变化: {self.last_market_regime} → {current_regime}")
            self.market_regime_changes += 1
            self.last_market_regime = current_regime
    
    def _detect_market_regime(self, market_state: MarketState) -> str:
        """
        检测市场状态（简化实现）
        
        Args:
            market_state: 市场状态
            
        Returns:
            市场状态描述
        """
        if not market_state.market_indicators:
            return "unknown"
        
        volatility = market_state.market_indicators.get('volatility', 0.15)
        trend = market_state.market_indicators.get('trend', 0.0)
        
        if volatility > 0.25:
            return "high_volatility"
        elif abs(trend) > 0.02:
            return "trending"
        else:
            return "stable"
    
    def _log_portfolio_state_analysis(self, portfolio_state: PortfolioState):
        """
        记录投资组合状态分析
        
        Args:
            portfolio_state: 投资组合状态
        """
        logger.info("💼 投资组合状态分析:")
        logger.info(f"  • 总价值: {portfolio_state.portfolio_value:,.2f}")
        logger.info(f"  • 现金: {portfolio_state.cash:,.2f}")
        logger.info(f"  • 持仓数量: {len(portfolio_state.positions)}")
        logger.info(f"  • 未实现盈亏: {portfolio_state.unrealized_pnl:,.2f}")
        logger.info(f"  • 已实现盈亏: {portfolio_state.realized_pnl:,.2f}")
        
        # 计算持仓集中度
        if portfolio_state.positions:
            position_values = list(portfolio_state.positions.values())
            total_position_value = sum(abs(v) for v in position_values)
            if total_position_value > 0:
                max_position_ratio = max(abs(v) for v in position_values) / total_position_value
                logger.info(f"  • 最大持仓占比: {max_position_ratio:.2%}")
    
    def _log_control_decision_details(self, control_signals: List[ControlSignal]):
        """
        记录控制决策详细信息
        
        Args:
            control_signals: 控制信号列表
        """
        if not control_signals:
            return
        
        logger.info("🎯 控制决策详情:")
        logger.info(f"  • 生成信号数量: {len(control_signals)}")
        
        # 按优先级分组
        priority_groups = defaultdict(list)
        for signal in control_signals:
            priority_groups[signal.priority].append(signal)
        
        for priority, signals in priority_groups.items():
            logger.info(f"  • {priority.name}优先级信号 ({len(signals)}个):")
            for signal in signals:
                self._log_individual_signal_details(signal)
    
    def _log_individual_signal_details(self, signal: ControlSignal):
        """
        记录单个信号详细信息
        
        Args:
            signal: 控制信号
        """
        logger.info(f"    - 类型: {signal.signal_type.value}")
        logger.info(f"      来源: {signal.source_component}")
        logger.info(f"      时间: {signal.timestamp.strftime('%H:%M:%S')}")
        
        # 记录信号内容的关键信息
        if signal.content:
            key_info = []
            for key, value in signal.content.items():
                if key in ['action', 'recommended_action', 'current_drawdown', 'threshold', 'risk_reduction_factor']:
                    if isinstance(value, float):
                        key_info.append(f"{key}={value:.4f}")
                    else:
                        key_info.append(f"{key}={value}")
            
            if key_info:
                logger.info(f"      详情: {', '.join(key_info)}")
    
    def _record_control_decisions(self, 
                                control_signals: List[ControlSignal],
                                market_state: MarketState,
                                portfolio_state: PortfolioState):
        """
        记录控制决策历史
        
        Args:
            control_signals: 控制信号列表
            market_state: 市场状态
            portfolio_state: 投资组合状态
        """
        # 更新决策类型统计
        for signal in control_signals:
            self.decision_type_counter[signal.signal_type.value] += 1
        
        # 记录决策历史
        decision_record = {
            'timestamp': datetime.now(),
            'signal_count': len(control_signals),
            'signal_types': [s.signal_type.value for s in control_signals],
            'portfolio_value': portfolio_state.portfolio_value,
            'market_regime': self._detect_market_regime(market_state),
            'has_critical_signals': any(s.priority.value == 1 for s in control_signals)
        }
        
        self.control_decision_history.append(decision_record)
        
        # 限制历史记录长度
        if len(self.control_decision_history) > 1000:
            self.control_decision_history = self.control_decision_history[-500:]
    
    def _update_risk_budget(self, 
                           portfolio_state: PortfolioState,
                           drawdown_metrics,
                           timestamp: datetime):
        """
        更新风险预算（增强版）
        
        Args:
            portfolio_state: 投资组合状态
            drawdown_metrics: 回撤指标
            timestamp: 时间戳
        """
        # 调用父类方法
        super()._update_risk_budget(portfolio_state, drawdown_metrics, timestamp)
        
        # 增强日志记录
        if hasattr(self, '_last_risk_budget') and self.detailed_logging_enabled:
            current_budget = self._last_risk_budget
            
            # 计算使用率（简化计算）
            usage_rate = min(abs(drawdown_metrics.current_drawdown) / current_budget, 1.0) if current_budget > 0 else 0
            
            # 计算效率（简化计算）
            efficiency = 1.0 / usage_rate if usage_rate > 0 else 1.0
            
            # 记录风险预算详细信息
            self._log_risk_budget_details(current_budget, usage_rate, efficiency)
    
    def _log_risk_budget_details(self, current_budget: float, usage_rate: float, efficiency: float):
        """
        记录风险预算详细信息
        
        Args:
            current_budget: 当前预算
            usage_rate: 使用率
            efficiency: 效率
        """
        logger.info("💰 风险预算使用情况:")
        logger.info(f"  • 当前预算: {current_budget:.4f}")
        logger.info(f"  • 使用率: {usage_rate:.2%}")
        logger.info(f"  • 效率评分: {efficiency:.2f}")
        
        # 预算使用建议
        if usage_rate > 0.9:
            logger.info("  • 建议: 预算使用率过高，考虑降低风险暴露")
        elif usage_rate < 0.3:
            logger.info("  • 建议: 预算使用率较低，可适当增加风险暴露")
        else:
            logger.info("  • 建议: 预算使用率适中，保持当前策略")
    
    def get_control_summary(self) -> Dict[str, Any]:
        """
        获取控制摘要信息
        
        Returns:
            控制摘要字典
        """
        summary = {
            'total_decisions': len(self.control_decision_history),
            'decision_types': dict(self.decision_type_counter),
            'market_state_changes': self.market_regime_changes,
            'recent_decisions': self.control_decision_history[-10:] if self.control_decision_history else []
        }
        
        # 添加风险预算信息
        if hasattr(self.adaptive_risk_budget, 'get_detailed_budget_info'):
            summary['risk_budget_info'] = self.adaptive_risk_budget.get_detailed_budget_info()
        
        # 添加市场状态统计
        if self.market_state_history:
            recent_states = self.market_state_history[-20:]
            summary['market_analysis'] = {
                'avg_volume': np.mean([s['total_volume'] for s in recent_states]),
                'price_volatility': np.std([s['avg_price_change'] for s in recent_states]),
                'state_count': len(recent_states)
            }
        
        return summary
    
    def reset_enhanced_state(self):
        """重置增强状态"""
        self.control_decision_history.clear()
        self.market_state_history.clear()
        self.risk_budget_decision_history.clear()
        self.decision_type_counter.clear()
        self.market_regime_changes = 0
        self.last_market_regime = None
        
        # 重置增强风险预算状态
        if hasattr(self.adaptive_risk_budget, 'reset_enhanced_state'):
            self.adaptive_risk_budget.reset_enhanced_state()
        
        logger.info("增强回撤控制器状态已重置")