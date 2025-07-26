"""
绩效分析器
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """策略绩效分析"""
    
    def __init__(self, config: Dict):
        """
        初始化绩效分析器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.risk_free_rate = config.get('risk_free_rate', 0.03)  # 无风险利率
        
    def generate_report(self, 
                       returns: pd.Series,
                       benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        生成完整的策略分析报告
        
        Args:
            returns: 策略收益率序列
            benchmark_returns: 基准收益率序列
            
        Returns:
            绩效分析报告
        """
        if len(returns) < 2:
            return {}
        
        report = {}
        
        # 基础收益指标
        report.update(self._calculate_return_metrics(returns))
        
        # 风险调整指标
        report.update(self._calculate_risk_adjusted_metrics(returns))
        
        # 下行风险指标
        report.update(self._calculate_downside_metrics(returns))
        
        # 基准对比指标
        if benchmark_returns is not None:
            report.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        return report
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """计算收益指标"""
        total_periods = len(returns)
        
        # 累计收益率
        cumulative_return = (1 + returns).prod() - 1
        
        # 年化收益率
        years = total_periods / 252
        annual_return = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else 0
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        return {
            '累计收益率': cumulative_return,
            '年化收益率': annual_return,
            '年化波动率': annual_volatility,
            '总交易天数': total_periods
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """计算风险调整指标"""
        annual_return = (1 + returns.mean()) ** 252 - 1
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # 卡玛比率 (年化收益率 / 最大回撤)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        return {
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '卡玛比率': calmar_ratio
        }
    
    def _calculate_downside_metrics(self, returns: pd.Series, target_return: float = 0.0) -> Dict:
        """计算下行风险指标"""
        # 下行偏差
        downside_returns = returns[returns < target_return] - target_return
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
        
        # 索提诺比率
        annual_return = (1 + returns.mean()) ** 252 - 1
        sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else float('inf')
        
        # 上行捕获率和下行捕获率（需要基准数据，这里简化处理）
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        upside_capture = positive_returns.mean() if len(positive_returns) > 0 else 0
        downside_capture = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0
        
        return {
            '下行偏差': downside_deviation,
            '索提诺比率': sortino_ratio,
            '上行捕获': upside_capture,
            '下行捕获': downside_capture
        }
    
    def _calculate_benchmark_metrics(self, 
                                   strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict:
        """计算相对基准指标"""
        # 对齐数据
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strat_ret = strategy_returns.iloc[:min_len]
        bench_ret = benchmark_returns.iloc[:min_len]
        
        # 超额收益
        excess_returns = strat_ret - bench_ret
        
        # 信息比率
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # Alpha和Beta
        alpha, beta = self.calculate_alpha_beta(strat_ret, bench_ret)
        
        # 相对最大回撤
        relative_returns = (1 + excess_returns).cumprod()
        relative_max_dd = self._calculate_drawdown_series(relative_returns).min()
        
        return {
            '信息比率': information_ratio,
            '跟踪误差': tracking_error,
            '阿尔法': alpha,
            '贝塔': beta,
            '相对最大回撤': relative_max_dd,
            '超额收益年化': excess_returns.mean() * 252
        }
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()
    
    def _calculate_drawdown_series(self, cumulative_returns: pd.Series) -> pd.Series:
        """计算回撤序列"""
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns
    
    def calculate_alpha_beta(self, 
                           strategy_returns: pd.Series,
                           benchmark_returns: pd.Series) -> tuple:
        """计算Alpha和Beta"""
        # 简单线性回归
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        # Beta
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha (年化)
        strategy_annual = strategy_returns.mean() * 252
        benchmark_annual = benchmark_returns.mean() * 252
        alpha = strategy_annual - beta * benchmark_annual
        
        return alpha, beta
    
    def calculate_win_rate(self, returns: pd.Series) -> Dict:
        """计算胜率相关指标"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        return {
            '胜率': win_rate,
            '平均盈利': avg_win,
            '平均亏损': avg_loss,
            '盈亏比': profit_loss_ratio
        }
    
    def calculate_rolling_metrics(self, 
                                returns: pd.Series,
                                window: int = 252) -> Dict[str, pd.Series]:
        """计算滚动指标"""
        rolling_return = returns.rolling(window).mean() * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_return - self.risk_free_rate) / rolling_vol
        
        # 滚动最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.rolling(window).max()
        rolling_drawdown = (cumulative - rolling_max) / rolling_max
        
        return {
            '滚动年化收益': rolling_return,
            '滚动年化波动': rolling_vol,
            '滚动夏普比率': rolling_sharpe,
            '滚动回撤': rolling_drawdown
        }
    
    def performance_attribution(self, 
                              returns: pd.Series,
                              factors: Optional[pd.DataFrame] = None) -> Dict:
        """绩效归因分析"""
        if factors is None:
            return {}
        
        # 简单的多因子归因模型
        attribution = {}
        
        # 对每个因子进行回归
        for factor_name in factors.columns:
            factor_returns = factors[factor_name]
            
            # 对齐数据
            aligned_returns = returns.reindex(factor_returns.index).dropna()
            aligned_factors = factor_returns.reindex(aligned_returns.index).dropna()
            
            if len(aligned_returns) > 10:
                # 计算因子暴露和贡献
                covariance = np.cov(aligned_returns, aligned_factors)[0, 1]
                factor_variance = np.var(aligned_factors)
                
                factor_beta = covariance / factor_variance if factor_variance > 0 else 0
                factor_contribution = factor_beta * aligned_factors.mean() * 252
                
                attribution[f'{factor_name}_beta'] = factor_beta
                attribution[f'{factor_name}_contribution'] = factor_contribution
        
        return attribution