"""
结果可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_results(results: Dict, save_path: Optional[str] = None):
    """
    绘制回测结果图表
    
    Args:
        results: 回测结果字典
        save_path: 保存路径
    """
    # 提取数据
    portfolio_history = results.get('portfolio_history', pd.DataFrame())
    returns_series = results.get('returns_series', pd.Series())
    
    if portfolio_history.empty or returns_series.empty:
        print("缺少必要的数据用于绘图")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('A股强化学习量化交易回测结果', fontsize=16, fontweight='bold')
    
    # 1. 净值曲线
    ax1 = axes[0, 0]
    cumulative_nav = (1 + returns_series).cumprod()
    ax1.plot(cumulative_nav.index, cumulative_nav.values, linewidth=2, color='blue')
    ax1.set_title('投资组合净值曲线')
    ax1.set_ylabel('净值')
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤曲线
    ax2 = axes[0, 1]
    running_max = cumulative_nav.expanding().max()
    drawdowns = (cumulative_nav - running_max) / running_max
    ax2.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.3, color='red')
    ax2.plot(drawdowns.index, drawdowns.values, color='red', linewidth=1)
    ax2.set_title('回撤曲线')
    ax2.set_ylabel('回撤比例')
    ax2.grid(True, alpha=0.3)
    
    # 3. 收益分布
    ax3 = axes[1, 0]
    ax3.hist(returns_series, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(returns_series.mean(), color='red', linestyle='--', label=f'均值: {returns_series.mean():.4f}')
    ax3.set_title('日收益率分布')
    ax3.set_xlabel('日收益率')
    ax3.set_ylabel('频次')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 滚动夏普比率
    ax4 = axes[1, 1]
    window = min(60, len(returns_series) // 4)
    rolling_sharpe = calculate_rolling_sharpe(returns_series, window)
    if not rolling_sharpe.empty:
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, color='purple', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title(f'{window}日滚动夏普比率')
        ax4.set_ylabel('夏普比率')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()


def plot_performance_comparison(strategy_returns: pd.Series,
                              benchmark_returns: Optional[pd.Series] = None,
                              save_path: Optional[str] = None):
    """
    绘制策略与基准对比图
    
    Args:
        strategy_returns: 策略收益率
        benchmark_returns: 基准收益率
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 净值对比
    strategy_nav = (1 + strategy_returns).cumprod()
    axes[0].plot(strategy_nav.index, strategy_nav.values, label='策略', linewidth=2)
    
    if benchmark_returns is not None:
        benchmark_nav = (1 + benchmark_returns).cumprod()
        axes[0].plot(benchmark_nav.index, benchmark_nav.values, label='基准', linewidth=2)
    
    axes[0].set_title('策略与基准净值对比')
    axes[0].set_ylabel('累计净值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 超额收益
    if benchmark_returns is not None:
        excess_returns = strategy_returns - benchmark_returns
        axes[1].plot(excess_returns.index, excess_returns.values, color='green', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_title('超额收益')
        axes[1].set_ylabel('超额收益率')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_risk_metrics(returns: pd.Series, save_path: Optional[str] = None):
    """
    绘制风险指标图表
    
    Args:
        returns: 收益率序列
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('风险指标分析', fontsize=16)
    
    # 1. VaR分析
    ax1 = axes[0, 0]
    var_levels = np.arange(1, 10) / 100
    var_values = [returns.quantile(level) for level in var_levels]
    ax1.plot(var_levels * 100, var_values, 'o-', color='red')
    ax1.set_title('VaR分析')
    ax1.set_xlabel('置信度 (%)')
    ax1.set_ylabel('VaR值')
    ax1.grid(True, alpha=0.3)
    
    # 2. 收益率QQ图
    ax2 = axes[0, 1]
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('收益率Q-Q图 (正态性检验)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 滚动波动率
    ax3 = axes[1, 0]
    window = min(20, len(returns) // 10)
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    ax3.plot(rolling_vol.index, rolling_vol.values, color='blue')
    ax3.set_title(f'{window}日滚动波动率')
    ax3.set_ylabel('年化波动率')
    ax3.grid(True, alpha=0.3)
    
    # 4. 月度收益热图
    ax4 = axes[1, 1]
    monthly_returns = returns.groupby([returns.index.year, returns.index.month]).sum()
    if len(monthly_returns) > 12:
        # 重构为矩阵形式
        years = sorted(monthly_returns.index.get_level_values(0).unique())
        months = range(1, 13)
        
        heatmap_data = pd.DataFrame(index=years, columns=months)
        for (year, month), ret in monthly_returns.items():
            heatmap_data.loc[year, month] = ret
        
        sns.heatmap(heatmap_data.astype(float), annot=True, fmt='.2%', 
                   cmap='RdYlGn', center=0, ax=ax4)
        ax4.set_title('月度收益热图')
        ax4.set_xlabel('月份')
        ax4.set_ylabel('年份')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    """计算滚动夏普比率"""
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std
    return rolling_sharpe.dropna()


def plot_factor_analysis(factor_data: pd.DataFrame, 
                        returns: pd.Series,
                        save_path: Optional[str] = None):
    """
    绘制因子分析图
    
    Args:
        factor_data: 因子数据
        returns: 收益率数据
        save_path: 保存路径
    """
    if factor_data.empty:
        return
    
    n_factors = min(6, len(factor_data.columns))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, factor in enumerate(factor_data.columns[:n_factors]):
        factor_values = factor_data[factor]
        
        # 因子与收益率散点图
        axes[i].scatter(factor_values, returns, alpha=0.6)
        axes[i].set_title(f'{factor}')
        axes[i].set_xlabel('因子值')
        axes[i].set_ylabel('收益率')
        axes[i].grid(True, alpha=0.3)
        
        # 计算相关系数
        correlation = factor_values.corr(returns)
        axes[i].text(0.05, 0.95, f'相关系数: {correlation:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()