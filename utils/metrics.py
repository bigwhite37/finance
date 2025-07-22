"""
指标计算工具
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_metrics(returns: pd.Series, 
                     benchmark: Optional[pd.Series] = None,
                     risk_free_rate: float = 0.03) -> Dict:
    """
    计算投资组合关键指标
    
    Args:
        returns: 收益率序列
        benchmark: 基准收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        指标字典
    """
    if len(returns) < 2:
        return {}
    
    # 基础统计
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    # 风险指标
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    max_drawdown = calculate_max_drawdown(returns)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean()
    
    # 其他指标
    win_rate = (returns > 0).mean()
    loss_rate = (returns < 0).mean()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'skewness': skewness,
        'kurtosis': kurtosis
    }
    
    # 相对基准指标
    if benchmark is not None:
        metrics.update(calculate_relative_metrics(returns, benchmark, risk_free_rate))
    
    return metrics


def calculate_max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    return drawdowns.min()


def calculate_relative_metrics(returns: pd.Series, 
                             benchmark: pd.Series,
                             risk_free_rate: float) -> Dict:
    """计算相对基准指标"""
    # 对齐数据
    aligned_returns = returns.reindex(benchmark.index).dropna()
    aligned_benchmark = benchmark.reindex(returns.index).dropna()
    
    min_len = min(len(aligned_returns), len(aligned_benchmark))
    if min_len < 10:
        return {}
    
    returns_subset = aligned_returns.iloc[:min_len]
    benchmark_subset = aligned_benchmark.iloc[:min_len]
    
    # 超额收益
    excess_returns = returns_subset - benchmark_subset
    
    # 跟踪误差
    tracking_error = excess_returns.std() * np.sqrt(252)
    
    # 信息比率
    information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
    
    # Beta和Alpha
    covariance = np.cov(returns_subset, benchmark_subset)[0, 1]
    benchmark_variance = benchmark_subset.var()
    
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    alpha = (returns_subset.mean() - beta * benchmark_subset.mean()) * 252
    
    return {
        'beta': beta,
        'alpha': alpha,
        'information_ratio': information_ratio,
        'tracking_error': tracking_error,
        'excess_return': excess_returns.mean() * 252
    }


def rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """计算滚动指标"""
    rolling_return = returns.rolling(window).mean() * 252
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_return / rolling_vol
    
    # 滚动最大回撤
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window).max()
    rolling_drawdown = (cumulative - rolling_max) / rolling_max
    
    result = pd.DataFrame({
        'rolling_return': rolling_return,
        'rolling_volatility': rolling_vol,
        'rolling_sharpe': rolling_sharpe,
        'rolling_drawdown': rolling_drawdown
    })
    
    return result.dropna()


def risk_adjusted_return(returns: pd.Series, risk_free_rate: float = 0.03) -> Dict:
    """计算风险调整收益指标"""
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    
    # 索提诺比率
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_vol
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
    # 卡玛比率
    max_dd = calculate_max_drawdown(returns)
    calmar = annual_return / abs(max_dd) if max_dd != 0 else float('inf')
    
    return {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar
    }