"""
回撤监控器使用示例

演示如何使用DrawdownMonitor进行实时回撤监控、市场状态检测和回撤归因分析
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl_trading_system.risk_control.drawdown_monitor import (
    DrawdownMonitor, DrawdownPhase, MarketRegime
)


def simulate_portfolio_data(days: int = 252, initial_value: float = 100000.0):
    """
    模拟投资组合数据
    
    Args:
        days: 模拟天数
        initial_value: 初始净值
        
    Returns:
        tuple: (时间序列, 净值序列, 价格数据)
    """
    np.random.seed(42)  # 确保结果可重复
    
    # 生成时间序列
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(days=i) for i in range(days)]
    
    # 模拟市场价格（带有趋势和波动）
    trend = 0.0002  # 日均涨幅0.02%
    volatility = 0.015  # 日波动率1.5%
    
    # 生成价格序列
    returns = np.random.normal(trend, volatility, days)
    
    # 添加一些回撤事件
    # 在第60-80天添加一个回撤
    returns[60:80] = np.random.normal(-0.01, 0.02, 20)
    # 在第150-170天添加另一个回撤
    returns[150:170] = np.random.normal(-0.008, 0.025, 20)
    
    # 计算累积净值
    portfolio_values = [initial_value]
    for ret in returns:
        new_value = portfolio_values[-1] * (1 + ret)
        portfolio_values.append(new_value)
    
    portfolio_values = portfolio_values[1:]  # 移除初始值
    
    # 生成价格数据用于市场状态检测
    prices = np.cumsum(returns) * 100 + 1000  # 模拟价格从1000开始
    
    return timestamps, portfolio_values, prices


def demonstrate_basic_monitoring():
    """演示基本的回撤监控功能"""
    print("=== 基本回撤监控演示 ===")
    
    # 创建回撤监控器
    monitor = DrawdownMonitor(
        drawdown_threshold=0.05,  # 5%回撤阈值
        recovery_threshold=0.02,  # 2%恢复阈值
        lookback_window=252,      # 一年的交易日
        volatility_window=20      # 20日波动率窗口
    )
    
    # 模拟数据
    timestamps, portfolio_values, prices = simulate_portfolio_data()
    
    print(f"模拟 {len(portfolio_values)} 天的投资组合数据")
    print(f"初始净值: {portfolio_values[0]:,.2f}")
    print(f"最终净值: {portfolio_values[-1]:,.2f}")
    
    # 逐日更新监控器
    metrics_history = []
    for i, (timestamp, value) in enumerate(zip(timestamps, portfolio_values)):
        metrics = monitor.update_portfolio_value(value, timestamp)
        metrics_history.append(metrics)
        
        # 每50天打印一次状态
        if (i + 1) % 50 == 0:
            print(f"\n第 {i+1} 天状态:")
            print(f"  净值: {value:,.2f}")
            print(f"  当前回撤: {metrics.current_drawdown:.4f} ({metrics.current_drawdown*100:.2f}%)")
            print(f"  最大回撤: {metrics.max_drawdown:.4f} ({metrics.max_drawdown*100:.2f}%)")
            print(f"  回撤阶段: {metrics.current_phase.value}")
            print(f"  回撤持续天数: {metrics.drawdown_duration}")
    
    # 最终统计
    final_metrics = metrics_history[-1]
    print(f"\n=== 最终统计 ===")
    print(f"最大回撤: {final_metrics.max_drawdown:.4f} ({final_metrics.max_drawdown*100:.2f}%)")
    print(f"平均回撤: {final_metrics.average_drawdown:.4f} ({final_metrics.average_drawdown*100:.2f}%)")
    print(f"回撤频率: {final_metrics.drawdown_frequency:.2f} 次/年")
    print(f"当前阶段: {final_metrics.current_phase.value}")
    
    return monitor, metrics_history, timestamps, portfolio_values, prices


def demonstrate_market_regime_detection(monitor, prices):
    """演示市场状态检测功能"""
    print("\n=== 市场状态检测演示 ===")
    
    # 准备市场数据
    market_data = {
        'prices': np.array(prices),
        'volumes': np.random.normal(1000000, 200000, len(prices))  # 模拟成交量
    }
    
    # 检测市场状态
    market_metrics = monitor.detect_market_regime(market_data)
    
    print(f"市场状态: {market_metrics.regime.value}")
    print(f"年化波动率: {market_metrics.volatility:.4f} ({market_metrics.volatility*100:.2f}%)")
    print(f"趋势强度: {market_metrics.trend_strength:.4f}")
    print(f"相关性水平: {market_metrics.correlation_level:.4f}")
    print(f"流动性评分: {market_metrics.liquidity_score:.4f}")
    print(f"识别置信度: {market_metrics.confidence_score:.4f}")
    
    return market_metrics


def demonstrate_attribution_analysis(monitor):
    """演示回撤归因分析功能"""
    print("\n=== 回撤归因分析演示 ===")
    
    # 模拟持仓和收益数据
    positions = {
        '股票A': 0.25,
        '股票B': 0.30,
        '股票C': 0.20,
        '股票D': 0.15,
        '现金': 0.10
    }
    
    position_returns = {
        '股票A': -0.08,  # 亏损8%
        '股票B': -0.12,  # 亏损12%
        '股票C': 0.02,   # 盈利2%
        '股票D': -0.05,  # 亏损5%
        '现金': 0.001    # 现金收益0.1%
    }
    
    # 分析回撤归因
    contributions = monitor.analyze_drawdown_attribution(positions, position_returns)
    
    print("各资产对回撤的贡献度:")
    for asset, contribution in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
        if contribution > 0:
            print(f"  {asset}: {contribution:.4f} ({contribution*100:.2f}%)")
    
    # 计算组合总收益
    total_return = sum(positions[asset] * position_returns[asset] 
                      for asset in positions if asset in position_returns)
    print(f"\n组合总收益: {total_return:.4f} ({total_return*100:.2f}%)")
    
    return contributions


def demonstrate_monitoring_status(monitor):
    """演示监控状态查询功能"""
    print("\n=== 监控状态查询演示 ===")
    
    status = monitor.get_current_status()
    
    print("当前监控状态:")
    for key, value in status.items():
        if key == 'drawdown_metrics':
            print(f"  {key}:")
            for metric_key, metric_value in value.items():
                if isinstance(metric_value, float):
                    print(f"    {metric_key}: {metric_value:.4f}")
                else:
                    print(f"    {metric_key}: {metric_value}")
        else:
            print(f"  {key}: {value}")


def plot_drawdown_analysis(timestamps, portfolio_values, metrics_history):
    """绘制回撤分析图表"""
    print("\n=== 生成回撤分析图表 ===")
    
    try:
        # 提取数据
        drawdowns = [m.current_drawdown for m in metrics_history]
        phases = [m.current_phase.value for m in metrics_history]
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # 净值曲线
        ax1.plot(timestamps, portfolio_values, 'b-', linewidth=1.5, label='投资组合净值')
        ax1.set_title('投资组合净值曲线')
        ax1.set_ylabel('净值')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 回撤曲线
        ax2.fill_between(timestamps, drawdowns, 0, alpha=0.3, color='red', label='回撤')
        ax2.plot(timestamps, drawdowns, 'r-', linewidth=1, label='当前回撤')
        ax2.set_title('回撤曲线（水下曲线）')
        ax2.set_ylabel('回撤比例')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 回撤阶段
        phase_colors = {
            '正常': 'green',
            '回撤开始': 'orange', 
            '回撤持续': 'red',
            '恢复中': 'blue'
        }
        
        for i, phase in enumerate(phases):
            color = phase_colors.get(phase, 'gray')
            ax3.scatter(timestamps[i], 1, c=color, s=10, alpha=0.7)
        
        ax3.set_title('回撤阶段变化')
        ax3.set_ylabel('阶段')
        ax3.set_ylim(0.5, 1.5)
        ax3.grid(True, alpha=0.3)
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, markersize=8, label=phase)
                          for phase, color in phase_colors.items()]
        ax3.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('drawdown_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'drawdown_analysis.png'")
        
    except ImportError:
        print("matplotlib未安装，跳过图表生成")


def main():
    """主函数"""
    print("回撤监控器使用示例")
    print("=" * 50)
    
    # 基本监控演示
    monitor, metrics_history, timestamps, portfolio_values, prices = demonstrate_basic_monitoring()
    
    # 市场状态检测演示
    market_metrics = demonstrate_market_regime_detection(monitor, prices)
    
    # 回撤归因分析演示
    contributions = demonstrate_attribution_analysis(monitor)
    
    # 监控状态查询演示
    demonstrate_monitoring_status(monitor)
    
    # 生成图表（如果可能）
    plot_drawdown_analysis(timestamps, portfolio_values, metrics_history)
    
    print("\n=== 演示完成 ===")
    print("回撤监控器已成功演示以下功能:")
    print("1. 实时回撤计算和监控")
    print("2. 回撤阶段识别")
    print("3. 市场状态检测")
    print("4. 回撤归因分析")
    print("5. 监控状态查询")


if __name__ == "__main__":
    main()