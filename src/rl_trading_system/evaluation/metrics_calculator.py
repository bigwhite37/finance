"""
多维度评估指标计算器

提供全面的性能评估指标计算功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
import warnings

from ..backtest.enhanced_backtest_engine import EnhancedBacktestResult

logger = logging.getLogger(__name__)


@dataclass
class MultiDimensionalMetrics:
    """多维度评估指标数据类"""
    # 基础收益指标
    total_return: float
    annual_return: float
    monthly_returns: List[float]
    
    # 风险指标
    volatility: float
    downside_deviation: float
    max_drawdown: float
    var_95: float  # 95% VaR
    cvar_95: float  # 95% CVaR
    
    # 风险调整收益指标
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # 回撤相关指标
    drawdown_improvement: float
    average_drawdown: float
    drawdown_frequency: float
    recovery_factor: float
    
    # 交易统计指标
    total_trades: int
    win_rate: float
    profit_factor: float
    average_trade_return: float
    best_trade: float
    worst_trade: float
    
    # 稳定性指标
    return_stability: float
    rolling_sharpe_std: float
    hit_ratio: float  # 超额收益频率
    
    # 尾部风险指标
    skewness: float
    kurtosis: float
    tail_ratio: float
    
    # 市场适应性指标
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'omega_ratio': self.omega_ratio,
            'drawdown_improvement': self.drawdown_improvement,
            'average_drawdown': self.average_drawdown,
            'drawdown_frequency': self.drawdown_frequency,
            'recovery_factor': self.recovery_factor,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_trade_return': self.average_trade_return,
            'best_trade': self.best_trade,
            'worst_trade': self.worst_trade,
            'return_stability': self.return_stability,
            'rolling_sharpe_std': self.rolling_sharpe_std,
            'hit_ratio': self.hit_ratio,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'tail_ratio': self.tail_ratio,
            'beta': self.beta,
            'alpha': self.alpha,
            'information_ratio': self.information_ratio,
            'tracking_error': self.tracking_error
        }


class MetricsCalculator:
    """
    多维度指标计算器
    
    提供全面的性能评估指标计算功能。
    """
    
    def __init__(self, risk_free_rate: float = 0.03):
        """
        初始化指标计算器
        
        Args:
            risk_free_rate: 无风险利率，默认3%
        """
        self.risk_free_rate = risk_free_rate
        
        logger.info(f"初始化多维度指标计算器，无风险利率: {risk_free_rate}")
    
    def calculate_comprehensive_metrics(self, 
                                       result: EnhancedBacktestResult,
                                       benchmark_returns: Optional[pd.Series] = None) -> MultiDimensionalMetrics:
        """
        计算综合性能指标
        
        Args:
            result: 增强回测结果
            benchmark_returns: 基准收益序列
            
        Returns:
            多维度评估指标
        """
        if not hasattr(result, 'portfolio_values') or result.portfolio_values is None:
            raise ValueError("回测结果缺少投资组合价值序列")
        
        # 计算收益序列
        returns = result.portfolio_values.pct_change().dropna()
        
        if len(returns) < 2:
            raise ValueError("收益序列长度不足，无法计算指标")
        
        # 基础收益指标
        total_return = result.total_return
        annual_return = result.annual_return
        monthly_returns = self._calculate_monthly_returns(result.portfolio_values)
        
        # 风险指标
        volatility = result.volatility
        downside_deviation = self._calculate_downside_deviation(returns)
        max_drawdown = result.max_drawdown
        var_95 = self._calculate_var(returns, confidence_level=0.95)
        cvar_95 = self._calculate_cvar(returns, confidence_level=0.95)
        
        # 风险调整收益指标
        sharpe_ratio = result.sharpe_ratio
        sortino_ratio = self._calculate_sortino_ratio(returns, annual_return)
        calmar_ratio = result.calmar_ratio
        omega_ratio = self._calculate_omega_ratio(returns)
        
        # 回撤相关指标
        drawdown_improvement = result.drawdown_improvement
        average_drawdown = result.drawdown_metrics.average_drawdown
        drawdown_frequency = result.drawdown_metrics.drawdown_frequency
        recovery_factor = self._calculate_recovery_factor(result.portfolio_values)
        
        # 交易统计指标
        total_trades = result.total_trades
        win_rate = result.win_rate
        profit_factor = result.profit_factor
        average_trade_return = result.average_trade_return
        best_trade, worst_trade = self._calculate_trade_extremes(returns)
        
        # 稳定性指标
        return_stability = self._calculate_return_stability(returns)
        rolling_sharpe_std = self._calculate_rolling_sharpe_std(returns)
        hit_ratio = self._calculate_hit_ratio(returns, benchmark_returns)
        
        # 尾部风险指标
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        tail_ratio = self._calculate_tail_ratio(returns)
        
        # 市场适应性指标
        beta, alpha, information_ratio, tracking_error = self._calculate_market_metrics(
            returns, benchmark_returns
        )
        
        return MultiDimensionalMetrics(
            total_return=total_return,
            annual_return=annual_return,
            monthly_returns=monthly_returns,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            drawdown_improvement=drawdown_improvement,
            average_drawdown=average_drawdown,
            drawdown_frequency=drawdown_frequency,
            recovery_factor=recovery_factor,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_trade_return=average_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            return_stability=return_stability,
            rolling_sharpe_std=rolling_sharpe_std,
            hit_ratio=hit_ratio,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            tracking_error=tracking_error
        )
    
    def _calculate_monthly_returns(self, portfolio_values: pd.Series) -> List[float]:
        """计算月度收益"""
        if len(portfolio_values) < 2:
            return []
        
        # 假设数据是日度的，按月聚合
        try:
            # 重采样到月末
            monthly_values = portfolio_values.resample('M').last()
            monthly_returns = monthly_values.pct_change().dropna()
            return monthly_returns.tolist()
        except (ValueError, KeyError, AttributeError) as e:
            # 重采样相关的预期异常，使用简单分组作为备选方案
            logger.debug(f"重采样失败，使用简单分组: {e}")
        except Exception as e:
            # 其他未预期异常，记录并使用备选方案
            logger.warning(f"计算月度收益时发生未预期错误: {e}")
            # 使用简单分组
            n_days = len(portfolio_values)
            days_per_month = 21  # 大约每月21个交易日
            
            monthly_returns = []
            for i in range(0, n_days, days_per_month):
                end_idx = min(i + days_per_month, n_days)
                if i == 0:
                    continue
                
                month_return = (portfolio_values.iloc[end_idx-1] / portfolio_values.iloc[i-1]) - 1
                monthly_returns.append(month_return)
            
            return monthly_returns
    
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """计算下行偏差"""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算风险价值（VaR）"""
        if len(returns) == 0:
            return 0.0
        
        return -np.percentile(returns, (1 - confidence_level) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """计算条件风险价值（CVaR）"""
        var = self._calculate_var(returns, confidence_level)
        
        # CVaR是超过VaR的平均损失
        tail_losses = returns[returns < -var]
        
        if len(tail_losses) == 0:
            return var
        
        return -tail_losses.mean()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, annual_return: float) -> float:
        """计算Sortino比率"""
        downside_deviation = self._calculate_downside_deviation(returns)
        
        if downside_deviation == 0:
            return float('inf') if annual_return > self.risk_free_rate else 0.0
        
        return (annual_return - self.risk_free_rate) / downside_deviation
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """计算Omega比率"""
        if len(returns) == 0:
            return 0.0
        
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        
        if losses == 0:
            return float('inf') if gains > 0 else 1.0
        
        return gains / losses
    
    def _calculate_recovery_factor(self, portfolio_values: pd.Series) -> float:
        """计算恢复因子"""
        if len(portfolio_values) < 2:
            return 0.0
        
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        # 计算最大回撤
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = abs(drawdowns.min())
        
        if max_drawdown == 0:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / max_drawdown
    
    def _calculate_trade_extremes(self, returns: pd.Series) -> Tuple[float, float]:
        """计算最好和最差的单日收益"""
        if len(returns) == 0:
            return 0.0, 0.0
        
        return returns.max(), returns.min()
    
    def _calculate_return_stability(self, returns: pd.Series) -> float:
        """计算收益稳定性（收益的变异系数倒数）"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        mean_return = returns.mean()
        if mean_return == 0:
            return 0.0
        
        cv = returns.std() / abs(mean_return)  # 变异系数
        return 1 / (1 + cv)  # 稳定性指标
    
    def _calculate_rolling_sharpe_std(self, returns: pd.Series, window: int = 63) -> float:
        """计算滚动夏普比率的标准差"""
        if len(returns) < window * 2:
            return 0.0
        
        rolling_sharpe = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            if window_returns.std() > 0:
                sharpe = window_returns.mean() / window_returns.std() * np.sqrt(252)
                rolling_sharpe.append(sharpe)
        
        if len(rolling_sharpe) == 0:
            return 0.0
        
        return np.std(rolling_sharpe)
    
    def _calculate_hit_ratio(self, returns: pd.Series, benchmark_returns: Optional[pd.Series]) -> float:
        """计算击中比率（超越基准的频率）"""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            # 如果没有基准，计算正收益频率
            return (returns > 0).mean()
        
        # 对齐时间序列
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return 0.0
        
        excess_returns = aligned_returns - aligned_benchmark
        return (excess_returns > 0).mean()
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """计算尾部比率（95%分位数 / 5%分位数的绝对值）"""
        if len(returns) == 0:
            return 0.0
        
        q95 = np.percentile(returns, 95)
        q05 = np.percentile(returns, 5)
        
        if q05 == 0:
            return float('inf') if q95 > 0 else 0.0
        
        return abs(q95 / q05)
    
    def _calculate_market_metrics(self, 
                                returns: pd.Series, 
                                benchmark_returns: Optional[pd.Series]) -> Tuple[float, float, float, float]:
        """计算市场相关指标"""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # 对齐时间序列
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) < 2:
            return 0.0, 0.0, 0.0, 0.0
        
        try:
            # 计算Beta
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = aligned_benchmark.var()
            
            if benchmark_variance == 0:
                beta = 0.0
            else:
                beta = covariance / benchmark_variance
            
            # 计算Alpha
            alpha = aligned_returns.mean() - beta * aligned_benchmark.mean()
            
            # 计算跟踪误差
            excess_returns = aligned_returns - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            # 计算信息比率
            if tracking_error == 0:
                information_ratio = 0.0
            else:
                information_ratio = excess_returns.mean() * 252 / tracking_error
            
            return beta, alpha * 252, information_ratio, tracking_error
            
        except Exception as e:
            logger.warning(f"计算市场指标失败: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def create_metrics_summary(self, metrics: MultiDimensionalMetrics) -> pd.DataFrame:
        """创建指标摘要表"""
        summary_data = [
            {'分类': '收益指标', '指标名称': '总收益率', '数值': f"{metrics.total_return:.2%}", '说明': '策略总收益率'},
            {'分类': '收益指标', '指标名称': '年化收益率', '数值': f"{metrics.annual_return:.2%}", '说明': '年化收益率'},
            {'分类': '风险指标', '指标名称': '波动率', '数值': f"{metrics.volatility:.2%}", '说明': '年化波动率'},
            {'分类': '风险指标', '指标名称': '最大回撤', '数值': f"{metrics.max_drawdown:.2%}", '说明': '历史最大回撤'},
            {'分类': '风险指标', '指标名称': '下行偏差', '数值': f"{metrics.downside_deviation:.2%}", '说明': '负收益的标准差'},
            {'分类': '风险指标', '指标名称': '95% VaR', '数值': f"{metrics.var_95:.2%}", '说明': '95%置信度风险价值'},
            {'分类': '风险指标', '指标名称': '95% CVaR', '数值': f"{metrics.cvar_95:.2%}", '说明': '95%置信度条件风险价值'},
            {'分类': '风险调整收益', '指标名称': 'Sharpe比率', '数值': f"{metrics.sharpe_ratio:.3f}", '说明': '风险调整收益指标'},
            {'分类': '风险调整收益', '指标名称': 'Sortino比率', '数值': f"{metrics.sortino_ratio:.3f}", '说明': '下行风险调整收益'},
            {'分类': '风险调整收益', '指标名称': 'Calmar比率', '数值': f"{metrics.calmar_ratio:.3f}", '说明': '回撤调整收益'},
            {'分类': '风险调整收益', '指标名称': 'Omega比率', '数值': f"{metrics.omega_ratio:.3f}", '说明': '收益概率比率'},
            {'分类': '交易统计', '指标名称': '总交易次数', '数值': f"{metrics.total_trades}", '说明': '总交易次数'},
            {'分类': '交易统计', '指标名称': '胜率', '数值': f"{metrics.win_rate:.2%}", '说明': '盈利交易占比'},
            {'分类': '交易统计', '指标名称': '盈亏比', '数值': f"{metrics.profit_factor:.3f}", '说明': '总盈利/总亏损'},
            {'分类': '稳定性', '指标名称': '收益稳定性', '数值': f"{metrics.return_stability:.3f}", '说明': '收益稳定性指标'},
            {'分类': '稳定性', '指标名称': '滚动Sharpe标准差', '数值': f"{metrics.rolling_sharpe_std:.3f}", '说明': '滚动夏普比率波动性'},
            {'分类': '尾部风险', '指标名称': '偏度', '数值': f"{metrics.skewness:.3f}", '说明': '收益分布偏度'},
            {'分类': '尾部风险', '指标名称': '峰度', '数值': f"{metrics.kurtosis:.3f}", '说明': '收益分布峰度'},
            {'分类': '市场适应性', '指标名称': 'Beta', '数值': f"{metrics.beta:.3f}", '说明': '市场敏感性'},
            {'分类': '市场适应性', '指标名称': 'Alpha', '数值': f"{metrics.alpha:.2%}", '说明': '超额收益'},
            {'分类': '市场适应性', '指标名称': '信息比率', '数值': f"{metrics.information_ratio:.3f}", '说明': '主动管理效率'}
        ]
        
        return pd.DataFrame(summary_data)