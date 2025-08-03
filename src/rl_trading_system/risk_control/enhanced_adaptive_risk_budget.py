"""
增强的自适应风险预算分配器

在原有功能基础上添加详细的日志记录和分析功能。
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .adaptive_risk_budget import (
    AdaptiveRiskBudget,
    AdaptiveRiskBudgetConfig,
    PerformanceMetrics,
    MarketMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedAdaptiveRiskBudgetConfig(AdaptiveRiskBudgetConfig):
    """增强的自适应风险预算配置"""
    # 详细日志开关
    enable_detailed_logging: bool = True           # 启用详细日志
    log_budget_changes: bool = True                # 记录预算变化
    log_usage_analysis: bool = True                # 记录使用分析
    log_performance_impact: bool = True            # 记录性能影响
    
    # 日志阈值
    budget_change_threshold: float = 0.01          # 预算变化日志阈值（1%）
    usage_analysis_frequency: int = 10             # 使用分析日志频率
    
    # 分析参数
    enable_efficiency_analysis: bool = True        # 启用效率分析
    enable_trend_analysis: bool = True             # 启用趋势分析


class EnhancedAdaptiveRiskBudget(AdaptiveRiskBudget):
    """
    增强的自适应风险预算分配器
    
    在原有功能基础上添加：
    1. 风险预算使用情况的详细日志
    2. 预算变化的原因分析
    3. 使用效率的统计分析
    """
    
    def __init__(self, config: EnhancedAdaptiveRiskBudgetConfig):
        """
        初始化增强的自适应风险预算分配器
        
        Args:
            config: 增强配置
        """
        super().__init__(config)
        self.enhanced_config = config
        
        # 详细日志状态
        self.detailed_logging_enabled = config.enable_detailed_logging
        self.last_logged_budget = None
        self.usage_analysis_counter = 0
        
        # 分析数据
        self.budget_change_reasons: List[str] = []
        self.efficiency_history: List[float] = []
        self.decision_history: List[Dict[str, Any]] = []
        
        logger.info("增强自适应风险预算分配器初始化完成")
        if self.detailed_logging_enabled:
            logger.info("详细日志记录已启用")
    
    def calculate_adaptive_risk_budget(self) -> float:
        """
        计算自适应风险预算（增强版）
        
        Returns:
            调整后的风险预算
        """
        # 调用父类方法计算预算
        new_budget = super().calculate_adaptive_risk_budget()
        
        # 增强日志记录
        if self.detailed_logging_enabled:
            self._log_budget_calculation_details(new_budget)
        
        # 记录决策历史
        self._record_budget_decision(new_budget)
        
        return new_budget
    
    def _log_budget_calculation_details(self, new_budget: float):
        """
        记录预算计算详细信息
        
        Args:
            new_budget: 新计算的预算
        """
        # 检查是否需要记录预算变化
        if self._should_log_budget_change(new_budget):
            self._log_budget_change_details(new_budget)
        
        # 定期记录使用分析
        self.usage_analysis_counter += 1
        if (self.enhanced_config.log_usage_analysis and 
            self.usage_analysis_counter % self.enhanced_config.usage_analysis_frequency == 0):
            self._log_budget_usage_analysis()
    
    def _should_log_budget_change(self, new_budget: float) -> bool:
        """
        判断是否应该记录预算变化
        
        Args:
            new_budget: 新预算
            
        Returns:
            是否应该记录
        """
        if not self.enhanced_config.log_budget_changes:
            return False
        
        if self.last_logged_budget is None:
            self.last_logged_budget = new_budget
            return True
        
        change_ratio = abs(new_budget - self.last_logged_budget) / self.last_logged_budget
        return change_ratio >= self.enhanced_config.budget_change_threshold
    
    def _log_budget_change_details(self, new_budget: float):
        """
        记录预算变化详细信息
        
        Args:
            new_budget: 新预算
        """
        old_budget = self.last_logged_budget or self.config.base_risk_budget
        change_ratio = (new_budget - old_budget) / old_budget
        change_direction = "增加" if change_ratio > 0 else "减少"
        
        # 分析变化原因
        change_reasons = self._analyze_budget_change_reasons(new_budget, old_budget)
        
        logger.info(f"🔄 风险预算调整:")
        logger.info(f"  • 原预算: {old_budget:.4f}")
        logger.info(f"  • 新预算: {new_budget:.4f}")
        logger.info(f"  • 变化幅度: {change_direction} {abs(change_ratio):.2%}")
        logger.info(f"  • 主要原因: {', '.join(change_reasons)}")
        
        self.last_logged_budget = new_budget
        self.budget_change_reasons.extend(change_reasons)
    
    def _analyze_budget_change_reasons(self, new_budget: float, old_budget: float) -> List[str]:
        """
        分析预算变化原因
        
        Args:
            new_budget: 新预算
            old_budget: 旧预算
            
        Returns:
            变化原因列表
        """
        reasons = []
        
        # 检查表现指标影响
        if hasattr(self, 'latest_performance_metrics') and self.latest_performance_metrics:
            perf = self.latest_performance_metrics
            
            if perf.sharpe_ratio < 0.5:
                reasons.append("夏普比率偏低")
            elif perf.sharpe_ratio > 1.5:
                reasons.append("夏普比率良好")
            
            if perf.max_drawdown > 0.1:
                reasons.append("回撤过大")
            elif perf.max_drawdown < 0.05:
                reasons.append("回撤控制良好")
            
            if perf.consecutive_losses > 3:
                reasons.append("连续亏损过多")
            elif perf.win_rate > 0.6:
                reasons.append("胜率较高")
        
        # 检查市场指标影响
        if hasattr(self, 'latest_market_metrics') and self.latest_market_metrics:
            market = self.latest_market_metrics
            
            if market.market_volatility > 0.3:
                reasons.append("市场波动率高")
            elif market.market_volatility < 0.1:
                reasons.append("市场波动率低")
            
            if market.uncertainty_index > 0.7:
                reasons.append("市场不确定性高")
            elif market.regime_stability > 0.8:
                reasons.append("市场状态稳定")
        
        # 如果没有找到具体原因，使用通用描述
        if not reasons:
            if new_budget > old_budget:
                reasons.append("综合条件改善")
            else:
                reasons.append("风险控制需要")
        
        return reasons
    
    def _log_budget_usage_analysis(self):
        """记录预算使用分析"""
        if len(self.risk_budget_history) < 2 or len(self.risk_usage_history) < 2:
            logger.debug("历史数据不足，跳过使用分析")
            return
        
        # 计算使用统计
        recent_budgets = self.risk_budget_history[-10:]
        recent_usage = self.risk_usage_history[-10:]
        
        avg_budget = np.mean(recent_budgets)
        avg_usage = np.mean(recent_usage)
        avg_utilization = avg_usage / avg_budget if avg_budget > 0 else 0
        
        # 计算效率指标
        efficiency_score = self._calculate_efficiency_score(recent_budgets, recent_usage)
        
        # 计算趋势
        budget_trend = self._calculate_trend(recent_budgets)
        usage_trend = self._calculate_trend(recent_usage)
        
        logger.info(f"📊 风险预算使用分析 (最近{len(recent_budgets)}期):")
        logger.info(f"  • 平均预算: {avg_budget:.4f}")
        logger.info(f"  • 平均使用: {avg_usage:.4f}")
        logger.info(f"  • 平均使用率: {avg_utilization:.2%}")
        logger.info(f"  • 效率评分: {efficiency_score:.2f}")
        logger.info(f"  • 预算趋势: {'上升' if budget_trend > 0 else '下降' if budget_trend < 0 else '平稳'}")
        logger.info(f"  • 使用趋势: {'上升' if usage_trend > 0 else '下降' if usage_trend < 0 else '平稳'}")
        
        # 记录效率历史
        self.efficiency_history.append(efficiency_score)
    
    def _calculate_efficiency_score(self, budgets: List[float], usage: List[float]) -> float:
        """
        计算效率评分
        
        Args:
            budgets: 预算历史
            usage: 使用历史
            
        Returns:
            效率评分
        """
        if len(budgets) != len(usage) or len(budgets) == 0:
            return 0.0
        
        # 计算使用率的稳定性（使用率越接近80%越好）
        utilizations = [u / b if b > 0 else 0 for u, b in zip(usage, budgets)]
        target_utilization = 0.8
        
        # 计算与目标使用率的偏差
        deviations = [abs(u - target_utilization) for u in utilizations]
        avg_deviation = np.mean(deviations)
        
        # 效率评分：偏差越小评分越高
        efficiency_score = max(0, 1 - avg_deviation * 2)
        
        return efficiency_score
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        计算趋势（线性回归斜率）
        
        Args:
            values: 数值序列
            
        Returns:
            趋势值（正值表示上升，负值表示下降）
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        return float(trend)
    
    def _record_budget_decision(self, new_budget: float):
        """
        记录预算决策历史
        
        Args:
            new_budget: 新预算
        """
        decision_record = {
            'timestamp': datetime.now(),
            'budget': new_budget,
            'performance_metrics': getattr(self, 'latest_performance_metrics', None),
            'market_metrics': getattr(self, 'latest_market_metrics', None),
            'change_reasons': self.budget_change_reasons[-3:] if self.budget_change_reasons else []
        }
        
        self.decision_history.append(decision_record)
        
        # 限制历史记录长度
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
    
    def get_detailed_budget_info(self) -> Dict[str, Any]:
        """
        获取详细预算信息
        
        Returns:
            详细预算信息字典
        """
        info = {
            'current_budget': self.current_risk_budget,
            'base_budget': self.config.base_risk_budget,
            'total_decisions': len(self.decision_history),
            'recent_change_reasons': self.budget_change_reasons[-5:] if self.budget_change_reasons else []
        }
        
        # 添加使用统计
        if hasattr(self, 'risk_budget_history') and len(self.risk_budget_history) > 0:
            # risk_budget_history是deque，需要转换为list
            recent_budgets = list(self.risk_budget_history)[-10:]
            # 如果没有使用历史，使用预算的80%作为估计使用量
            recent_usage = [b * 0.8 for b in recent_budgets]
            
            utilizations = [u / b if b > 0 else 0 for u, b in zip(recent_usage, recent_budgets)]
            
            info.update({
                'average_utilization': np.mean(utilizations) if utilizations else 0,
                'utilization_std': np.std(utilizations) if len(utilizations) > 1 else 0,
                'budget_trend': self._calculate_trend(recent_budgets),
                'usage_trend': self._calculate_trend(recent_usage)
            })
        
        # 添加效率信息
        if len(self.efficiency_history) > 0:
            info.update({
                'efficiency_score': self.efficiency_history[-1],
                'avg_efficiency': np.mean(self.efficiency_history),
                'efficiency_trend': self._calculate_trend(self.efficiency_history)
            })
        
        return info
    
    def reset_enhanced_state(self):
        """重置增强状态"""
        self.budget_change_reasons.clear()
        self.efficiency_history.clear()
        self.decision_history.clear()
        self.last_logged_budget = None
        self.usage_analysis_counter = 0
        
        logger.info("增强自适应风险预算状态已重置")