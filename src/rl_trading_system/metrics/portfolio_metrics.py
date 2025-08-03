"""
投资组合指标计算模块

实现投资组合表现指标、智能体行为指标和风险控制指标的计算。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """投资组合表现指标"""
    sharpe_ratio: float                    # 夏普比率
    max_drawdown: float                    # 最大回撤
    alpha: float                          # Alpha（相对基准的超额收益）
    beta: float                           # Beta（相对基准的系统性风险）
    annualized_return: float              # 年化收益率
    timestamp: datetime                   # 计算时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'alpha': self.alpha,
            'beta': self.beta,
            'annualized_return': self.annualized_return,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentBehaviorMetrics:
    """智能体行为分析指标"""
    mean_entropy: float                   # 平均熵值
    entropy_trend: float                  # 熵值趋势（正值表示增加，负值表示减少）
    mean_position_concentration: float    # 平均持仓集中度
    turnover_rate: float                  # 换手率
    timestamp: datetime                   # 计算时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'mean_entropy': self.mean_entropy,
            'entropy_trend': self.entropy_trend,
            'mean_position_concentration': self.mean_position_concentration,
            'turnover_rate': self.turnover_rate,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RiskControlMetrics:
    """风险控制指标"""
    avg_risk_budget_utilization: float   # 平均风险预算使用率
    risk_budget_efficiency: float        # 风险预算效率（收益/风险预算使用）
    control_signal_frequency: float      # 控制信号频率
    market_regime_stability: float       # 市场状态稳定性
    timestamp: datetime                   # 计算时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'avg_risk_budget_utilization': self.avg_risk_budget_utilization,
            'risk_budget_efficiency': self.risk_budget_efficiency,
            'control_signal_frequency': self.control_signal_frequency,
            'market_regime_stability': self.market_regime_stability,
            'timestamp': self.timestamp.isoformat()
        }


class PortfolioMetricsCalculator:
    """投资组合指标计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率（年化）
            
        Returns:
            夏普比率
        """
        if len(returns) == 0:
            return 0.0
        
        # 计算超额收益
        mean_return = np.mean(returns)
        excess_return = mean_return - risk_free_rate / 252  # 日化无风险利率
        
        # 计算收益率标准差
        return_std = np.std(returns, ddof=1)
        
        if return_std == 0:
            return 0.0
        
        # 年化夏普比率
        sharpe_ratio = (excess_return / return_std) * np.sqrt(252)
        
        return float(sharpe_ratio)
    
    def calculate_max_drawdown(self, values: List[float]) -> float:
        """
        计算最大回撤
        
        Args:
            values: 投资组合价值序列
            
        Returns:
            最大回撤（正值）
        """
        if len(values) <= 1:
            return 0.0
        
        values = np.array(values)
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        
        return float(max_drawdown)
    
    def calculate_alpha_beta(self, portfolio_returns: np.ndarray, 
                           benchmark_returns: np.ndarray,
                           risk_free_rate: float = 0.03) -> Tuple[float, float]:
        """
        计算Alpha和Beta
        
        Args:
            portfolio_returns: 投资组合收益率序列
            benchmark_returns: 基准收益率序列
            risk_free_rate: 无风险利率（年化）
            
        Returns:
            (alpha, beta)
            
        Raises:
            RuntimeError: 当数据无效时抛出异常
        """
        # 严格的数据验证
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            raise RuntimeError("无法计算Alpha和Beta：输入数据为空")
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise RuntimeError(f"投资组合收益率和基准收益率长度不匹配：{len(portfolio_returns)} vs {len(benchmark_returns)}")
        
        # 检查数据有效性
        if np.any(np.isnan(portfolio_returns)) or np.any(np.isnan(benchmark_returns)):
            raise RuntimeError("无法计算Alpha和Beta：输入数据包含NaN值")
        
        if np.any(np.isinf(portfolio_returns)) or np.any(np.isinf(benchmark_returns)):
            raise RuntimeError("无法计算Alpha和Beta：输入数据包含无穷大值")
        
        # 计算超额收益
        daily_rf_rate = risk_free_rate / 252
        portfolio_excess = portfolio_returns - daily_rf_rate
        benchmark_excess = benchmark_returns - daily_rf_rate
        
        # 计算Beta和Alpha，增加数值稳定性检查
        benchmark_variance = np.var(benchmark_excess, ddof=1)
        
        # 处理基准方差为零或接近零的情况
        if abs(benchmark_variance) < 1e-10:
            beta = 0.0
            alpha = np.mean(portfolio_excess) * 252  # 年化Alpha
        else:
            # 使用更稳定的协方差计算
            covariance = np.cov(portfolio_excess, benchmark_excess, ddof=1)[0, 1]
            
            # 检查协方差计算结果的有效性
            if np.isnan(covariance) or np.isinf(covariance):
                raise RuntimeError("协方差计算结果无效，可能存在数值稳定性问题")
            
            beta = covariance / benchmark_variance
            
            # 检查Beta的合理性
            if np.isnan(beta) or np.isinf(beta):
                raise RuntimeError("Beta计算结果无效，可能存在数值稳定性问题")
            
            # 对Beta进行合理性检查和限制
            if abs(beta) > 1e6:
                logger.warning(f"Beta值异常大 ({beta})，可能存在数值稳定性问题，将其限制在合理范围内")
                beta = np.sign(beta) * min(abs(beta), 10.0)  # 限制Beta在[-10, 10]范围内
            
            # 计算Alpha
            alpha = (np.mean(portfolio_excess) - beta * np.mean(benchmark_excess)) * 252
            
            # 检查Alpha的合理性
            if np.isnan(alpha) or np.isinf(alpha):
                raise RuntimeError("Alpha计算结果无效，可能存在数值稳定性问题")
            
            # 对Alpha进行合理性检查和限制
            if abs(alpha) > 1e6:
                logger.warning(f"Alpha值异常大 ({alpha})，可能存在数值稳定性问题，将其限制在合理范围内")
                alpha = np.sign(alpha) * min(abs(alpha), 5.0)  # 限制Alpha在[-5, 5]范围内
        
        return float(alpha), float(beta)
    
    def calculate_annualized_return(self, start_value: float, end_value: float, 
                                  days: int) -> float:
        """
        计算年化收益率
        
        Args:
            start_value: 起始价值
            end_value: 结束价值
            days: 天数
            
        Returns:
            年化收益率
        """
        if start_value <= 0 or days <= 0:
            return 0.0
        
        if days >= 252:
            # 多年期：使用复合年化收益率
            years = days / 252
            annual_return = (end_value / start_value) ** (1 / years) - 1
        else:
            # 不足一年：简单年化
            annual_return = (end_value / start_value - 1) * (252 / days)
        
        return float(annual_return)
    
    def calculate_portfolio_metrics(self, portfolio_values: List[float],
                                  benchmark_values: List[float],
                                  dates: List[datetime],
                                  risk_free_rate: float = 0.03) -> PortfolioMetrics:
        """
        计算完整的投资组合指标
        
        Args:
            portfolio_values: 投资组合价值序列
            benchmark_values: 基准价值序列
            dates: 日期序列
            risk_free_rate: 无风险利率
            
        Returns:
            投资组合指标
        """
        if len(portfolio_values) <= 1:
            raise RuntimeError("投资组合价值序列长度不足，无法计算指标")
        
        # 计算收益率
        portfolio_returns = np.diff(portfolio_values) / np.array(portfolio_values[:-1])
        benchmark_returns = np.diff(benchmark_values) / np.array(benchmark_values[:-1])
        
        # 计算各项指标
        sharpe_ratio = self.calculate_sharpe_ratio(portfolio_returns, risk_free_rate)
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        alpha, beta = self.calculate_alpha_beta(portfolio_returns, benchmark_returns, risk_free_rate)
        
        # 计算年化收益率
        days = len(portfolio_values) - 1
        annualized_return = self.calculate_annualized_return(
            portfolio_values[0], portfolio_values[-1], days
        )
        
        return PortfolioMetrics(
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            alpha=alpha,
            beta=beta,
            annualized_return=annualized_return,
            timestamp=datetime.now()
        )
    
    def calculate_agent_behavior_metrics(self, entropy_values: List[float],
                                       position_weights_history: List[np.ndarray]) -> AgentBehaviorMetrics:
        """
        计算智能体行为指标
        
        Args:
            entropy_values: 熵值序列
            position_weights_history: 持仓权重历史
            
        Returns:
            智能体行为指标
        """
        if len(entropy_values) == 0:
            raise RuntimeError("熵值序列为空，无法计算智能体行为指标")
        
        # 计算平均熵
        mean_entropy = float(np.mean(entropy_values))
        
        # 计算熵趋势（线性回归斜率）
        if len(entropy_values) > 1:
            x = np.arange(len(entropy_values))
            entropy_trend = float(np.polyfit(x, entropy_values, 1)[0])
        else:
            entropy_trend = 0.0
        
        # 计算平均持仓集中度
        if len(position_weights_history) == 0:
            mean_position_concentration = 0.0
            turnover_rate = 0.0
        else:
            # 使用赫芬达尔指数衡量集中度
            concentrations = []
            for weights in position_weights_history:
                if len(weights) > 0:
                    concentration = np.sum(weights ** 2)
                    concentrations.append(concentration)
            
            mean_position_concentration = float(np.mean(concentrations)) if concentrations else 0.0
            
            # 计算换手率
            turnover_rate = self._calculate_turnover_rate(position_weights_history)
        
        return AgentBehaviorMetrics(
            mean_entropy=mean_entropy,
            entropy_trend=entropy_trend,
            mean_position_concentration=mean_position_concentration,
            turnover_rate=turnover_rate,
            timestamp=datetime.now()
        )
    
    def _calculate_turnover_rate(self, position_weights_history: List[np.ndarray]) -> float:
        """
        计算换手率
        
        Args:
            position_weights_history: 持仓权重历史
            
        Returns:
            换手率
        """
        if len(position_weights_history) <= 1:
            return 0.0
        
        total_turnover = 0.0
        valid_periods = 0
        
        for i in range(1, len(position_weights_history)):
            prev_weights = position_weights_history[i-1]
            curr_weights = position_weights_history[i]
            
            if len(prev_weights) == len(curr_weights):
                # 计算权重变化的绝对值之和
                weight_changes = np.abs(curr_weights - prev_weights)
                turnover = np.sum(weight_changes) / 2  # 除以2因为买入和卖出是成对的
                total_turnover += turnover
                valid_periods += 1
        
        return float(total_turnover / valid_periods) if valid_periods > 0 else 0.0
    
    def calculate_risk_control_metrics(self, risk_budget_history: List[float],
                                     risk_usage_history: List[float],
                                     control_signals: List[Dict[str, Any]],
                                     market_regime_history: List[str]) -> RiskControlMetrics:
        """
        计算风险控制指标
        
        Args:
            risk_budget_history: 风险预算历史
            risk_usage_history: 风险使用历史
            control_signals: 控制信号历史
            market_regime_history: 市场状态历史
            
        Returns:
            风险控制指标
        """
        # 计算平均风险预算使用率
        if len(risk_budget_history) == 0 or len(risk_usage_history) == 0:
            avg_risk_budget_utilization = 0.0
            risk_budget_efficiency = 0.0
        else:
            utilizations = []
            for budget, usage in zip(risk_budget_history, risk_usage_history):
                if budget > 0:
                    utilization = min(usage / budget, 1.0)
                    utilizations.append(utilization)
            
            avg_risk_budget_utilization = float(np.mean(utilizations)) if utilizations else 0.0
            
            # 风险预算效率（简化计算：使用率的倒数）
            if avg_risk_budget_utilization > 0:
                risk_budget_efficiency = 1.0 / avg_risk_budget_utilization
            else:
                risk_budget_efficiency = 0.0
        
        # 计算控制信号频率
        if len(control_signals) == 0:
            control_signal_frequency = 0.0
        else:
            # 假设信号是按时间顺序的，计算每日平均信号数
            total_days = max(len(risk_budget_history), 1)
            control_signal_frequency = len(control_signals) / total_days
        
        # 计算市场状态稳定性
        if len(market_regime_history) <= 1:
            market_regime_stability = 1.0
        else:
            # 计算状态变化频率
            changes = 0
            for i in range(1, len(market_regime_history)):
                if market_regime_history[i] != market_regime_history[i-1]:
                    changes += 1
            
            # 稳定性 = 1 - 变化频率
            change_rate = changes / (len(market_regime_history) - 1)
            market_regime_stability = 1.0 - change_rate
        
        return RiskControlMetrics(
            avg_risk_budget_utilization=float(avg_risk_budget_utilization),
            risk_budget_efficiency=float(risk_budget_efficiency),
            control_signal_frequency=float(control_signal_frequency),
            market_regime_stability=float(market_regime_stability),
            timestamp=datetime.now()
        )