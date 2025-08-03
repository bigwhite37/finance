"""自适应风险预算系统使用示例"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.colors import qualitative
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from src.rl_trading_system.risk_control.adaptive_risk_budget import (
    AdaptiveRiskBudget,
    AdaptiveRiskBudgetConfig,
    PerformanceMetrics,
    MarketMetrics,
    MarketCondition,
    PerformanceRegime
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_config() -> AdaptiveRiskBudgetConfig:
    """创建示例配置"""
    return AdaptiveRiskBudgetConfig(
        # 基础参数
        base_risk_budget=0.12,                    # 12%基础风险预算
        min_risk_budget=0.03,                     # 3%最小风险预算
        max_risk_budget=0.25,                     # 25%最大风险预算
        
        # 表现调整参数
        performance_lookback_days=45,             # 45天表现回看
        sharpe_threshold_excellent=1.8,           # 优秀表现阈值
        sharpe_threshold_good=1.0,                # 良好表现阈值
        sharpe_threshold_poor=0.0,                # 较差表现阈值
        performance_adjustment_factor=0.25,       # 25%表现调整因子
        
        # 市场条件调整参数
        market_lookback_days=30,                  # 30天市场回看
        volatility_threshold_high=0.22,           # 22%高波动阈值
        volatility_threshold_low=0.08,            # 8%低波动阈值
        uncertainty_threshold=0.65,               # 65%不确定性阈值
        market_adjustment_factor=0.35,            # 35%市场调整因子
        
        # 连续亏损调整参数
        consecutive_loss_threshold=4,             # 4次连续亏损阈值
        loss_penalty_factor=0.15,                # 15%亏损惩罚因子
        max_loss_penalty=0.4,                    # 40%最大亏损惩罚
        
        # 平滑机制参数
        smoothing_factor=0.15,                   # 15%平滑因子
        max_daily_change=0.08,                   # 8%最大日变化率
        adjustment_delay_days=1,                 # 1天调整延迟
        
        # 异常检测参数
        anomaly_detection_window=25,             # 25天异常检测窗口
        anomaly_threshold_std=2.5,               # 2.5倍标准差异常阈值
        anomaly_adjustment_factor=0.4,           # 40%异常调整因子
        
        # 恢复机制参数
        recovery_speed_fast=0.06,                # 6%快速恢复速度
        recovery_speed_slow=0.03,                # 3%慢速恢复速度
        recovery_threshold=0.6                   # 60%恢复阈值
    )


def simulate_market_scenarios() -> List[Dict[str, Any]]:
    """模拟不同的市场情景"""
    scenarios = [
        # 情景1: 牛市开始
        {
            'name': '牛市启动',
            'duration': 30,
            'performance': PerformanceMetrics(
                sharpe_ratio=1.2,
                calmar_ratio=1.8,
                max_drawdown=0.05,
                volatility=0.12,
                win_rate=0.65,
                consecutive_losses=0,
                total_return=0.15
            ),
            'market': MarketMetrics(
                market_volatility=0.10,
                market_trend=0.12,
                uncertainty_index=0.25,
                correlation_with_market=0.75,
                liquidity_score=0.9
            )
        },
        
        # 情景2: 市场震荡
        {
            'name': '震荡整理',
            'duration': 20,
            'performance': PerformanceMetrics(
                sharpe_ratio=0.6,
                calmar_ratio=1.0,
                max_drawdown=0.08,
                volatility=0.15,
                win_rate=0.52,
                consecutive_losses=2,
                total_return=0.05
            ),
            'market': MarketMetrics(
                market_volatility=0.18,
                market_trend=0.02,
                uncertainty_index=0.45,
                correlation_with_market=0.65,
                liquidity_score=0.8
            )
        },
        
        # 情景3: 市场调整
        {
            'name': '市场调整',
            'duration': 25,
            'performance': PerformanceMetrics(
                sharpe_ratio=-0.2,
                calmar_ratio=0.3,
                max_drawdown=0.15,
                volatility=0.22,
                win_rate=0.42,
                consecutive_losses=5,
                total_return=-0.08
            ),
            'market': MarketMetrics(
                market_volatility=0.28,
                market_trend=-0.10,
                uncertainty_index=0.75,
                correlation_with_market=0.85,
                liquidity_score=0.7
            )
        },
        
        # 情景4: 危机爆发
        {
            'name': '市场危机',
            'duration': 15,
            'performance': PerformanceMetrics(
                sharpe_ratio=-1.2,
                calmar_ratio=-0.5,
                max_drawdown=0.25,
                volatility=0.35,
                win_rate=0.25,
                consecutive_losses=8,
                total_return=-0.20
            ),
            'market': MarketMetrics(
                market_volatility=0.45,
                market_trend=-0.25,
                uncertainty_index=0.95,
                correlation_with_market=0.95,
                liquidity_score=0.4
            )
        },
        
        # 情景5: 恢复阶段
        {
            'name': '市场恢复',
            'duration': 35,
            'performance': PerformanceMetrics(
                sharpe_ratio=0.8,
                calmar_ratio=1.2,
                max_drawdown=0.10,
                volatility=0.18,
                win_rate=0.58,
                consecutive_losses=1,
                total_return=0.12
            ),
            'market': MarketMetrics(
                market_volatility=0.20,
                market_trend=0.08,
                uncertainty_index=0.35,
                correlation_with_market=0.70,
                liquidity_score=0.85
            )
        },
        
        # 情景6: 强势上涨
        {
            'name': '强势上涨',
            'duration': 40,
            'performance': PerformanceMetrics(
                sharpe_ratio=2.2,
                calmar_ratio=3.5,
                max_drawdown=0.03,
                volatility=0.14,
                win_rate=0.72,
                consecutive_losses=0,
                total_return=0.25
            ),
            'market': MarketMetrics(
                market_volatility=0.12,
                market_trend=0.18,
                uncertainty_index=0.20,
                correlation_with_market=0.80,
                liquidity_score=0.95
            )
        }
    ]
    
    return scenarios


def run_adaptive_risk_budget_simulation():
    """运行自适应风险预算模拟"""
    print("=" * 60)
    print("自适应风险预算系统模拟")
    print("=" * 60)
    
    # 创建配置和系统
    config = create_sample_config()
    adaptive_budget = AdaptiveRiskBudget(config)
    
    print(f"初始配置:")
    print(f"  基础风险预算: {config.base_risk_budget:.1%}")
    print(f"  风险预算范围: {config.min_risk_budget:.1%} - {config.max_risk_budget:.1%}")
    print(f"  平滑因子: {config.smoothing_factor:.1%}")
    print(f"  最大日变化: {config.max_daily_change:.1%}")
    print()
    
    # 获取市场情景
    scenarios = simulate_market_scenarios()
    
    # 存储结果
    results = []
    dates = []
    risk_budgets = []
    performance_regimes = []
    market_conditions = []
    adjustments = []
    
    current_date = datetime(2023, 1, 1)
    
    # 模拟每个情景
    for scenario in scenarios:
        print(f"情景: {scenario['name']} (持续 {scenario['duration']} 天)")
        print(f"  表现指标: Sharpe={scenario['performance'].sharpe_ratio:.2f}, "
              f"回撤={scenario['performance'].max_drawdown:.1%}")
        print(f"  市场指标: 波动率={scenario['market'].market_volatility:.1%}, "
              f"趋势={scenario['market'].market_trend:.1%}")
        
        # 模拟该情景的每一天
        for day in range(scenario['duration']):
            # 更新指标
            adaptive_budget.update_performance_metrics(scenario['performance'])
            adaptive_budget.update_market_metrics(scenario['market'])
            
            # 计算风险预算
            risk_budget = adaptive_budget.calculate_adaptive_risk_budget()
            
            # 记录结果
            dates.append(current_date)
            risk_budgets.append(risk_budget)
            performance_regimes.append(adaptive_budget.current_performance_regime.value)
            market_conditions.append(adaptive_budget.current_market_condition.value)
            
            # 检查是否有新的调整
            if len(adaptive_budget.adjustment_history) > len(adjustments):
                latest_adjustment = adaptive_budget.adjustment_history[-1]
                adjustments.append({
                    'date': current_date,
                    'old_budget': latest_adjustment.old_budget,
                    'new_budget': latest_adjustment.new_budget,
                    'reason': latest_adjustment.adjustment_reason,
                    'scenario': scenario['name']
                })
            
            current_date += timedelta(days=1)
        
        print(f"  情景结束时风险预算: {risk_budget:.1%}")
        print()
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'date': dates,
        'risk_budget': risk_budgets,
        'performance_regime': performance_regimes,
        'market_condition': market_conditions
    })
    
    # 打印统计摘要
    print("=" * 60)
    print("模拟结果摘要")
    print("=" * 60)
    
    summary = adaptive_budget.get_risk_budget_summary()
    print(f"当前风险预算: {summary['current_risk_budget']:.1%}")
    print(f"基础风险预算: {summary['base_risk_budget']:.1%}")
    print(f"当前表现状态: {summary['performance_regime']}")
    print(f"当前市场状态: {summary['market_condition']}")
    print(f"总调整次数: {summary['total_adjustments']}")
    print()
    
    print(f"风险预算统计:")
    print(f"  最小值: {summary['budget_range']['min']:.1%}")
    print(f"  最大值: {summary['budget_range']['max']:.1%}")
    print(f"  当前值: {summary['budget_range']['current']:.1%}")
    print(f"  平均值: {np.mean(risk_budgets):.1%}")
    print(f"  标准差: {np.std(risk_budgets):.1%}")
    print()
    
    # 打印主要调整记录
    print("主要调整记录:")
    for i, adj in enumerate(adjustments[-5:], 1):  # 显示最后5次调整
        change = adj['new_budget'] - adj['old_budget']
        print(f"  {i}. {adj['date'].strftime('%Y-%m-%d')} ({adj['scenario']}): "
              f"{adj['old_budget']:.1%} -> {adj['new_budget']:.1%} "
              f"({change:+.1%}) - {adj['reason']}")
    print()
    
    # 分析不同状态下的风险预算
    regime_analysis = results_df.groupby('performance_regime')['risk_budget'].agg(['mean', 'std', 'min', 'max'])
    print("不同表现状态下的风险预算:")
    for regime in regime_analysis.index:
        stats = regime_analysis.loc[regime]
        print(f"  {regime}: 均值={stats['mean']:.1%}, "
              f"标准差={stats['std']:.1%}, "
              f"范围=[{stats['min']:.1%}, {stats['max']:.1%}]")
    print()
    
    condition_analysis = results_df.groupby('market_condition')['risk_budget'].agg(['mean', 'std', 'min', 'max'])
    print("不同市场状态下的风险预算:")
    for condition in condition_analysis.index:
        stats = condition_analysis.loc[condition]
        print(f"  {condition}: 均值={stats['mean']:.1%}, "
              f"标准差={stats['std']:.1%}, "
              f"范围=[{stats['min']:.1%}, {stats['max']:.1%}]")
    print()
    
    return results_df, adjustments, adaptive_budget


def plot_simulation_results(results_df: pd.DataFrame, adjustments: List[Dict], config: AdaptiveRiskBudgetConfig):
    """绘制模拟结果图表"""
    try:
        # 创建子图
        fig = sp.make_subplots(
            rows=3, cols=1,
            subplot_titles=('自适应风险预算时间序列', '表现状态分布', '市场状态分布'),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # 图1: 风险预算时间序列
        fig.add_trace(
            go.Scatter(
                x=results_df['date'],
                y=results_df['risk_budget'],
                mode='lines',
                name='自适应风险预算',
                line=dict(color='blue', width=2),
                hovertemplate='日期: %{x}<br>风险预算: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 添加基准线
        fig.add_hline(
            y=config.base_risk_budget,
            line_dash="dash",
            line_color="green",
            annotation_text="基础风险预算",
            annotation_position="bottom right",
            row=1, col=1
        )
        
        fig.add_hline(
            y=config.min_risk_budget,
            line_dash="dot",
            line_color="red",
            annotation_text="最小风险预算",
            annotation_position="top right",
            row=1, col=1
        )
        
        fig.add_hline(
            y=config.max_risk_budget,
            line_dash="dot",
            line_color="red",
            annotation_text="最大风险预算",
            annotation_position="bottom right",
            row=1, col=1
        )
        
        # 标记重要调整点
        adjustment_dates = [adj['date'] for adj in adjustments[::3]]
        adjustment_budgets = [adj['new_budget'] for adj in adjustments[::3]]
        adjustment_reasons = [adj['reason'] for adj in adjustments[::3]]
        
        if adjustment_dates:
            fig.add_trace(
                go.Scatter(
                    x=adjustment_dates,
                    y=adjustment_budgets,
                    mode='markers',
                    name='重要调整点',
                    marker=dict(color='red', size=8, opacity=0.7),
                    text=adjustment_reasons,
                    hovertemplate='日期: %{x}<br>调整后预算: %{y:.1%}<br>原因: %{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 图2: 表现状态分布
        regime_counts = results_df['performance_regime'].value_counts()
        regime_colors = {
            'excellent': '#2E8B57',    # 深绿色
            'good': '#90EE90',         # 浅绿色
            'average': '#FFD700',      # 金色
            'poor': '#FFA500',         # 橙色
            'terrible': '#FF6347'      # 红色
        }
        
        colors = [regime_colors.get(regime, '#808080') for regime in regime_counts.index]
        
        fig.add_trace(
            go.Bar(
                x=regime_counts.index,
                y=regime_counts.values,
                name='表现状态',
                marker_color=colors,
                hovertemplate='状态: %{x}<br>天数: %{y}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 图3: 市场状态分布
        condition_counts = results_df['market_condition'].value_counts()
        condition_colors = {
            'bull': '#4169E1',         # 蓝色
            'bear': '#DC143C',         # 深红色
            'sideways': '#808080',     # 灰色
            'high_vol': '#FF8C00',     # 橙色
            'low_vol': '#87CEEB',      # 浅蓝色
            'crisis': '#8B0000'        # 暗红色
        }
        
        colors = [condition_colors.get(condition, '#808080') for condition in condition_counts.index]
        
        fig.add_trace(
            go.Bar(
                x=condition_counts.index,
                y=condition_counts.values,
                name='市场状态',
                marker_color=colors,
                hovertemplate='状态: %{x}<br>天数: %{y}<extra></extra>',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # 更新布局
        fig.update_layout(
            height=900,
            title_text="自适应风险预算系统模拟结果",
            title_x=0.5,
            title_font_size=16,
            showlegend=True,
            hovermode='x unified'
        )
        
        # 更新y轴格式
        fig.update_yaxes(
            tickformat='.1%',
            title_text="风险预算",
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="天数",
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="天数",
            row=3, col=1
        )
        
        # 更新x轴
        fig.update_xaxes(
            title_text="日期",
            row=1, col=1
        )
        
        fig.update_xaxes(
            title_text="表现状态",
            row=2, col=1
        )
        
        fig.update_xaxes(
            title_text="市场状态",
            row=3, col=1
        )
        
        # 保存和显示图表
        fig.write_html("adaptive_risk_budget_simulation.html")
        print("交互式图表已保存为 'adaptive_risk_budget_simulation.html'")
        
        # 也保存为静态图片
        try:
            fig.write_image("adaptive_risk_budget_simulation.png", width=1200, height=900)
            print("静态图表已保存为 'adaptive_risk_budget_simulation.png'")
        except Exception as e:
            print(f"保存静态图片失败 (需要安装kaleido): {e}")
        
        fig.show()
        
    except ImportError:
        print("plotly未安装，跳过图表绘制")
    except Exception as e:
        print(f"绘制图表时出错: {e}")


def plot_detailed_analysis(results_df: pd.DataFrame, adjustments: List[Dict], adaptive_budget: AdaptiveRiskBudget):
    """绘制详细分析图表"""
    try:
        # 创建更复杂的分析图表
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '风险预算变化与调整原因',
                '不同状态下的风险预算分布',
                '风险预算调整频率分析',
                '风险预算统计指标'
            ),
            specs=[[{"secondary_y": True}, {"type": "box"}],
                   [{"type": "histogram"}, {"type": "indicator"}]]
        )
        
        # 图1: 风险预算变化与调整原因
        fig.add_trace(
            go.Scatter(
                x=results_df['date'],
                y=results_df['risk_budget'],
                mode='lines',
                name='风险预算',
                line=dict(color='blue', width=2),
                hovertemplate='日期: %{x}<br>风险预算: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 添加调整点的颜色编码
        if adjustments:
            reason_colors = {
                '表现优秀': '#2E8B57',
                '表现不佳': '#DC143C',
                '市场不利': '#FF8C00',
                '市场有利': '#4169E1',
                '连续亏损': '#8B0000',
                '高波动率': '#FF6347',
                '高不确定性': '#B22222',
                '恢复调整': '#32CD32',
                '常规调整': '#808080'
            }
            
            for reason in set(adj['reason'] for adj in adjustments):
                reason_adjustments = [adj for adj in adjustments if adj['reason'] == reason]
                if reason_adjustments:
                    fig.add_trace(
                        go.Scatter(
                            x=[adj['date'] for adj in reason_adjustments],
                            y=[adj['new_budget'] for adj in reason_adjustments],
                            mode='markers',
                            name=f'调整-{reason}',
                            marker=dict(
                                color=reason_colors.get(reason, '#808080'),
                                size=10,
                                opacity=0.8
                            ),
                            hovertemplate=f'原因: {reason}<br>日期: %{{x}}<br>新预算: %{{y:.1%}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # 图2: 不同状态下的风险预算分布箱线图
        for regime in results_df['performance_regime'].unique():
            regime_data = results_df[results_df['performance_regime'] == regime]['risk_budget']
            fig.add_trace(
                go.Box(
                    y=regime_data,
                    name=regime,
                    boxpoints='outliers',
                    hovertemplate=f'状态: {regime}<br>风险预算: %{{y:.1%}}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 图3: 风险预算调整频率分析
        if adjustments:
            # 按月统计调整次数
            adjustment_df = pd.DataFrame(adjustments)
            adjustment_df['month'] = pd.to_datetime(adjustment_df['date']).dt.to_period('M')
            monthly_adjustments = adjustment_df.groupby('month').size()
            
            fig.add_trace(
                go.Histogram(
                    x=monthly_adjustments.values,
                    nbinsx=10,
                    name='调整频率分布',
                    marker_color='lightblue',
                    opacity=0.7,
                    hovertemplate='调整次数: %{x}<br>月份数: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 图4: 关键统计指标
        risk_budget_stats = {
            '当前风险预算': results_df['risk_budget'].iloc[-1],
            '平均风险预算': results_df['risk_budget'].mean(),
            '风险预算标准差': results_df['risk_budget'].std(),
            '最大风险预算': results_df['risk_budget'].max(),
            '最小风险预算': results_df['risk_budget'].min()
        }
        
        # 创建指标卡片
        indicator_text = "<br>".join([
            f"<b>{key}:</b> {value:.1%}" 
            for key, value in risk_budget_stats.items()
        ])
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=results_df['risk_budget'].iloc[-1],
                delta={'reference': adaptive_budget.config.base_risk_budget},
                title={"text": f"当前风险预算<br><span style='font-size:0.8em;color:gray'>{indicator_text}</span>"},
                number={'suffix': "%", 'valueformat': ".1%"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            height=800,
            title_text="自适应风险预算详细分析",
            title_x=0.5,
            title_font_size=16,
            showlegend=True
        )
        
        # 更新坐标轴
        fig.update_yaxes(tickformat='.1%', title_text="风险预算", row=1, col=1)
        fig.update_yaxes(tickformat='.1%', title_text="风险预算", row=1, col=2)
        fig.update_yaxes(title_text="月份数", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_xaxes(title_text="表现状态", row=1, col=2)
        fig.update_xaxes(title_text="月调整次数", row=2, col=1)
        
        # 保存和显示
        fig.write_html("adaptive_risk_budget_detailed_analysis.html")
        print("详细分析图表已保存为 'adaptive_risk_budget_detailed_analysis.html'")
        
        fig.show()
        
    except ImportError:
        print("plotly未安装，跳过详细分析图表绘制")
    except Exception as e:
        print(f"绘制详细分析图表时出错: {e}")


def demonstrate_parameter_sensitivity():
    """演示参数敏感性分析"""
    print("=" * 60)
    print("参数敏感性分析")
    print("=" * 60)
    
    base_config = create_sample_config()
    
    # 测试不同的平滑因子
    smoothing_factors = [0.05, 0.1, 0.2, 0.3]
    print("平滑因子敏感性分析:")
    
    for factor in smoothing_factors:
        config = AdaptiveRiskBudgetConfig(
            base_risk_budget=base_config.base_risk_budget,
            smoothing_factor=factor,
            performance_adjustment_factor=base_config.performance_adjustment_factor
        )
        
        adaptive_budget = AdaptiveRiskBudget(config)
        
        # 模拟一个简单的情景
        performance = PerformanceMetrics(sharpe_ratio=1.5, consecutive_losses=0)
        market = MarketMetrics(market_volatility=0.15, uncertainty_index=0.3)
        
        adaptive_budget.update_performance_metrics(performance)
        adaptive_budget.update_market_metrics(market)
        
        # 多次更新以观察平滑效果
        budgets = []
        for _ in range(10):
            budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
            budgets.append(budget)
        
        budget_std = np.std(budgets)
        print(f"  平滑因子 {factor:.2f}: 风险预算标准差 = {budget_std:.4f}")
    
    print()
    
    # 测试不同的表现调整因子
    adjustment_factors = [0.1, 0.2, 0.3, 0.4]
    print("表现调整因子敏感性分析:")
    
    for factor in adjustment_factors:
        config = AdaptiveRiskBudgetConfig(
            base_risk_budget=base_config.base_risk_budget,
            performance_adjustment_factor=factor,
            smoothing_factor=0.1
        )
        
        adaptive_budget = AdaptiveRiskBudget(config)
        
        # 测试优秀表现
        excellent_performance = PerformanceMetrics(sharpe_ratio=2.5, consecutive_losses=0)
        market = MarketMetrics(market_volatility=0.12, uncertainty_index=0.2)
        
        adaptive_budget.update_performance_metrics(excellent_performance)
        adaptive_budget.update_market_metrics(market)
        
        budget = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        increase = (budget - base_config.base_risk_budget) / base_config.base_risk_budget
        
        print(f"  调整因子 {factor:.1f}: 优秀表现时风险预算增加 {increase:.1%}")
    
    print()


def main():
    """主函数"""
    print("自适应风险预算系统使用示例")
    print("=" * 60)
    
    try:
        # 运行主要模拟
        results_df, adjustments, adaptive_budget = run_adaptive_risk_budget_simulation()
        
        # 绘制结果图表
        config = adaptive_budget.config
        plot_simulation_results(results_df, adjustments, config)
        
        # 绘制详细分析图表
        plot_detailed_analysis(results_df, adjustments, adaptive_budget)
        
        # 参数敏感性分析
        demonstrate_parameter_sensitivity()
        
        print("=" * 60)
        print("示例运行完成！")
        print("=" * 60)
        
        # 提供使用建议
        print("\n使用建议:")
        print("1. 根据实际交易策略的历史表现调整表现阈值")
        print("2. 根据交易的资产类别调整市场波动率阈值")
        print("3. 在实盘使用前进行充分的回测验证")
        print("4. 定期监控系统的调整频率和效果")
        print("5. 根据风险承受能力调整最大和最小风险预算")
        
    except Exception as e:
        logger.error(f"运行示例时出错: {e}")
        raise


if __name__ == "__main__":
    main()