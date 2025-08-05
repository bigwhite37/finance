"""
Qlib集成可视化模块
提供专业的金融数据可视化功能，集成Qlib原生可视化组件
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Qlib可视化相关导入
import qlib
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D
from qlib.utils import flatten_dict

logger = logging.getLogger(__name__)

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')


class QlibVisualizer:
    """Qlib集成可视化器"""

    def __init__(self, figsize=(15, 10)):
        """
        初始化可视化器

        Args:
            figsize: 默认图片尺寸
        """
        self.figsize = figsize
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))

    def plot_portfolio_analysis(self,
                               results: Dict[str, Any],
                               save_path: Optional[str] = None) -> None:
        """
        绘制组合分析图表

        Args:
            results: 回测结果字典
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(20, 15))

        # 创建网格布局
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. 净值曲线对比 (大图)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_net_value_comparison(ax1, results)

        # 2. 回撤分析 (大图)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_drawdown_analysis(ax2, results)

        # 3. 收益率分布对比
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_returns_distribution(ax3, results)

        # 4. 月度收益热力图
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_monthly_returns_heatmap(ax4, results)

        # 5. 风险收益散点图
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_risk_return_scatter(ax5, results)

        # 6. 滚动指标分析
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_rolling_metrics(ax6, results)

        # 7. 持仓权重分析
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_position_weights(ax7, results)

        # 8. 收益归因分析
        ax8 = fig.add_subplot(gs[2, 2:])
        self._plot_return_attribution(ax8, results)

        # 9. 交易频率分析
        ax9 = fig.add_subplot(gs[3, 0])
        self._plot_trading_frequency(ax9, results)

        # 10. 市场状态分析
        ax10 = fig.add_subplot(gs[3, 1])
        self._plot_market_regime_analysis(ax10, results)

        # 11. 波动率分析
        ax11 = fig.add_subplot(gs[3, 2])
        self._plot_volatility_analysis(ax11, results)

        # 12. 业绩统计表
        ax12 = fig.add_subplot(gs[3, 3])
        self._plot_performance_table(ax12, results)

        plt.suptitle('强化学习投资策略全面分析报告', fontsize=20, fontweight='bold', y=0.98)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Qlib可视化图表已保存: {save_path}")

        plt.show()

    def _plot_net_value_comparison(self, ax, results):
        """绘制净值曲线对比"""
        rl_values = results['rl_strategy']['values']
        bench_values = results['benchmark']['values']

        # 归一化处理
        min_len = min(len(rl_values), len(bench_values))
        dates = pd.date_range(start=results['period']['start'],
                             periods=min_len, freq='D')

        rl_norm = rl_values[:min_len] / rl_values[0]
        bench_norm = bench_values[:min_len] / bench_values[0]

        ax.plot(dates, rl_norm, label='RL策略', linewidth=2.5, color='red', alpha=0.8)
        ax.plot(dates, bench_norm, label='基准策略', linewidth=2.5, color='blue', alpha=0.8)

        # 填充区域显示相对表现
        ax.fill_between(dates, rl_norm, bench_norm,
                       where=(rl_norm > bench_norm),
                       color='green', alpha=0.2, label='RL超额收益')
        ax.fill_between(dates, rl_norm, bench_norm,
                       where=(rl_norm <= bench_norm),
                       color='red', alpha=0.2, label='RL落后表现')

        ax.set_title('净值曲线对比分析', fontsize=14, fontweight='bold')
        ax.set_ylabel('累计净值')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # 添加关键统计信息
        final_rl = rl_norm.iloc[-1] if len(rl_norm) > 0 else 1
        final_bench = bench_norm.iloc[-1] if len(bench_norm) > 0 else 1
        ax.text(0.02, 0.98, f'RL最终收益: {(final_rl-1):.2%}\n基准最终收益: {(final_bench-1):.2%}',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _plot_drawdown_analysis(self, ax, results):
        """绘制回撤分析"""
        rl_values = results['rl_strategy']['values']
        bench_values = results['benchmark']['values']

        min_len = min(len(rl_values), len(bench_values))
        dates = pd.date_range(start=results['period']['start'],
                             periods=min_len, freq='D')

        # 计算回撤
        rl_dd = self._calculate_drawdown_series(rl_values[:min_len])
        bench_dd = self._calculate_drawdown_series(bench_values[:min_len])

        ax.fill_between(dates, rl_dd, 0, alpha=0.6, color='red', label='RL策略回撤')
        ax.fill_between(dates, bench_dd, 0, alpha=0.6, color='blue', label='基准回撤')

        ax.set_title('回撤分析', fontsize=14, fontweight='bold')
        ax.set_ylabel('回撤幅度')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 标记最大回撤点
        max_dd_rl = np.min(rl_dd)
        max_dd_bench = np.min(bench_dd)
        ax.axhline(y=max_dd_rl, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=max_dd_bench, color='blue', linestyle='--', alpha=0.7)

        ax.text(0.02, 0.02, f'RL最大回撤: {abs(max_dd_rl):.2%}\n基准最大回撤: {abs(max_dd_bench):.2%}',
                transform=ax.transAxes, va='bottom', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def _plot_returns_distribution(self, ax, results):
        """绘制收益率分布"""
        rl_returns = results['rl_strategy']['returns']
        bench_returns = results['benchmark']['returns']

        min_len = min(len(rl_returns), len(bench_returns))

        ax.hist(rl_returns[:min_len], bins=30, alpha=0.7,
               label='RL策略', color='red', density=True)
        ax.hist(bench_returns[:min_len], bins=30, alpha=0.7,
               label='基准', color='blue', density=True)

        # 添加正态分布拟合
        from scipy import stats
        rl_mean, rl_std = np.mean(rl_returns[:min_len]), np.std(rl_returns[:min_len])
        bench_mean, bench_std = np.mean(bench_returns[:min_len]), np.std(bench_returns[:min_len])

        x = np.linspace(min(rl_returns[:min_len].min(), bench_returns[:min_len].min()),
                       max(rl_returns[:min_len].max(), bench_returns[:min_len].max()), 100)

        ax.plot(x, stats.norm.pdf(x, rl_mean, rl_std), 'r-', linewidth=2, alpha=0.8)
        ax.plot(x, stats.norm.pdf(x, bench_mean, bench_std), 'b-', linewidth=2, alpha=0.8)

        ax.set_title('日收益率分布', fontsize=12, fontweight='bold')
        ax.set_xlabel('日收益率')
        ax.set_ylabel('密度')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_monthly_returns_heatmap(self, ax, results):
        """绘制月度收益热力图"""
        rl_returns = results['rl_strategy']['returns']

        if len(rl_returns) < 21:  # 少于一个月的数据
            ax.text(0.5, 0.5, '数据不足\n无法生成月度分析',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('月度收益热力图', fontsize=12, fontweight='bold')
            return

        # 简化的月度分析
        chunk_size = 21  # 假设21个交易日为一个月
        monthly_returns = []

        for i in range(0, len(rl_returns), chunk_size):
            chunk = rl_returns[i:i+chunk_size]
            if len(chunk) > 0:
                monthly_ret = np.prod(1 + chunk) - 1
                monthly_returns.append(monthly_ret)

        if len(monthly_returns) > 0:
            # 创建简单的热力图数据
            months = len(monthly_returns)
            data = np.array(monthly_returns).reshape(-1, 1)

            im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
            ax.set_title('月度收益热力图', fontsize=12, fontweight='bold')
            ax.set_ylabel('月份')

            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, '无月度数据', ha='center', va='center',
                   transform=ax.transAxes)

    def _plot_risk_return_scatter(self, ax, results):
        """绘制风险收益散点图"""
        metrics = results['performance_metrics']

        strategies = ['rl_strategy', 'benchmark']
        names = ['RL策略', '基准策略']
        colors = ['red', 'blue']

        for strategy, name, color in zip(strategies, names, colors):
            vol = metrics[strategy]['volatility']
            ret = metrics[strategy]['annualized_return']
            sharpe = metrics[strategy]['sharpe_ratio']

            ax.scatter(vol, ret, s=200, c=color, label=f'{name}\n(Sharpe: {sharpe:.3f})',
                      alpha=0.7, edgecolors='black', linewidth=1)

        ax.set_title('风险收益特征', fontsize=12, fontweight='bold')
        ax.set_xlabel('年化波动率')
        ax.set_ylabel('年化收益率')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加有效前沿参考线
        x_range = np.linspace(0, ax.get_xlim()[1], 100)
        for sharpe_level in [0, 0.5, 1.0, 1.5]:
            y_line = sharpe_level * x_range
            ax.plot(x_range, y_line, '--', alpha=0.3,
                   label=f'Sharpe={sharpe_level}' if sharpe_level > 0 else None)

    def _plot_rolling_metrics(self, ax, results):
        """绘制滚动指标"""
        rl_returns = results['rl_strategy']['returns']

        if len(rl_returns) < 60:
            ax.text(0.5, 0.5, '数据不足\n无法计算滚动指标',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('滚动夏普比率', fontsize=12, fontweight='bold')
            return

        window = min(60, len(rl_returns) // 3)
        rolling_sharpe = self._calculate_rolling_sharpe(rl_returns, window)

        dates = pd.date_range(start=results['period']['start'],
                             periods=len(rolling_sharpe), freq='D')

        ax.plot(dates, rolling_sharpe, color='purple', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe=1')
        ax.axhline(y=-1, color='red', linestyle='--', alpha=0.5, label='Sharpe=-1')

        ax.set_title(f'{window}日滚动夏普比率', fontsize=12, fontweight='bold')
        ax.set_ylabel('夏普比率')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_position_weights(self, ax, results):
        """绘制持仓权重变化"""
        if 'weights' not in results['rl_strategy'] or not results['rl_strategy']['weights']:
            ax.text(0.5, 0.5, '无持仓权重数据', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('持仓权重变化', fontsize=14, fontweight='bold')
            return

        weights_data = np.array(results['rl_strategy']['weights'])
        if weights_data.ndim != 2 or weights_data.shape[1] == 0:
            ax.text(0.5, 0.5, '权重数据格式错误', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('持仓权重变化', fontsize=14, fontweight='bold')
            return

        # 选择权重最大的前10只股票
        avg_weights = np.mean(weights_data, axis=0)
        top_indices = np.argsort(avg_weights)[-10:]

        dates = pd.date_range(start=results['period']['start'],
                             periods=len(weights_data), freq='D')

        # 堆叠面积图
        bottom = np.zeros(len(weights_data))
        for i, idx in enumerate(top_indices):
            weights = weights_data[:, idx]
            ax.fill_between(dates, bottom, bottom + weights,
                           alpha=0.7, label=f'股票{idx+1}')
            bottom += weights

        ax.set_title('主要持仓权重变化', fontsize=14, fontweight='bold')
        ax.set_ylabel('权重')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_return_attribution(self, ax, results):
        """绘制收益归因分析"""
        rl_returns = results['rl_strategy']['returns']
        bench_returns = results['benchmark']['returns']

        min_len = min(len(rl_returns), len(bench_returns))
        excess_returns = rl_returns[:min_len] - bench_returns[:min_len]

        # 计算累计超额收益
        cum_excess = np.cumprod(1 + excess_returns) - 1

        dates = pd.date_range(start=results['period']['start'],
                             periods=len(cum_excess), freq='D')

        ax.plot(dates, cum_excess, color='green', linewidth=2.5, label='累计超额收益')
        ax.fill_between(dates, cum_excess, 0,
                       where=(cum_excess > 0), color='green', alpha=0.3)
        ax.fill_between(dates, cum_excess, 0,
                       where=(cum_excess <= 0), color='red', alpha=0.3)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title('超额收益归因分析', fontsize=14, fontweight='bold')
        ax.set_ylabel('累计超额收益')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        if 'relative' in results['performance_metrics']:
            info_ratio = results['performance_metrics']['relative']['information_ratio']
            ax.text(0.02, 0.98, f'信息比率: {info_ratio:.3f}',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    def _plot_trading_frequency(self, ax, results):
        """绘制交易频率分析"""
        if 'actions' not in results['rl_strategy'] or not results['rl_strategy']['actions']:
            ax.text(0.5, 0.5, '无交易动作数据', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('交易频率分析', fontsize=12, fontweight='bold')
            return

        actions = np.array(results['rl_strategy']['actions'])

        # 计算每日交易活跃度（权重变化幅度）
        if len(actions) > 1:
            trading_activity = np.sum(np.abs(np.diff(actions, axis=0)), axis=1)

            ax.hist(trading_activity, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax.set_title('每日交易活跃度分布', fontsize=12, fontweight='bold')
            ax.set_xlabel('权重变化幅度')
            ax.set_ylabel('频次')
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            mean_activity = np.mean(trading_activity)
            ax.axvline(x=mean_activity, color='red', linestyle='--',
                      label=f'平均活跃度: {mean_activity:.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, '交易数据不足', ha='center', va='center',
                   transform=ax.transAxes)

    def _plot_market_regime_analysis(self, ax, results):
        """绘制市场状态分析"""
        bench_returns = results['benchmark']['returns']
        rl_returns = results['rl_strategy']['returns']

        min_len = min(len(rl_returns), len(bench_returns))

        # 简单的市场状态划分：根据基准收益率
        bull_mask = bench_returns[:min_len] > np.percentile(bench_returns[:min_len], 75)
        bear_mask = bench_returns[:min_len] < np.percentile(bench_returns[:min_len], 25)
        neutral_mask = ~(bull_mask | bear_mask)

        regimes = ['牛市', '震荡', '熊市']
        rl_perf = [
            np.mean(rl_returns[:min_len][bull_mask]) if np.any(bull_mask) else 0,
            np.mean(rl_returns[:min_len][neutral_mask]) if np.any(neutral_mask) else 0,
            np.mean(rl_returns[:min_len][bear_mask]) if np.any(bear_mask) else 0
        ]

        colors = ['green', 'yellow', 'red']
        bars = ax.bar(regimes, rl_perf, color=colors, alpha=0.7, edgecolor='black')

        ax.set_title('不同市场状态下表现', fontsize=12, fontweight='bold')
        ax.set_ylabel('平均日收益率')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, perf in zip(bars, rl_perf):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{perf:.4f}', ha='center', va='bottom' if height >= 0 else 'top')

    def _plot_volatility_analysis(self, ax, results):
        """绘制波动率分析"""
        rl_returns = results['rl_strategy']['returns']

        if len(rl_returns) < 30:
            ax.text(0.5, 0.5, '数据不足\n无法分析波动率',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('波动率分析', fontsize=12, fontweight='bold')
            return

        # 计算滚动波动率
        window = min(30, len(rl_returns) // 3)
        rolling_vol = []

        for i in range(window, len(rl_returns)):
            vol = np.std(rl_returns[i-window:i]) * np.sqrt(252)
            rolling_vol.append(vol)

        if rolling_vol:
            dates = pd.date_range(start=results['period']['start'],
                                 periods=len(rolling_vol), freq='D')

            ax.plot(dates, rolling_vol, color='purple', linewidth=2)
            ax.fill_between(dates, rolling_vol, alpha=0.3, color='purple')

            avg_vol = np.mean(rolling_vol)
            ax.axhline(y=avg_vol, color='red', linestyle='--',
                      label=f'平均波动率: {avg_vol:.2%}')

            ax.set_title(f'{window}日滚动年化波动率', fontsize=12, fontweight='bold')
            ax.set_ylabel('年化波动率')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_performance_table(self, ax, results):
        """绘制业绩统计表"""
        ax.axis('off')

        metrics = results['performance_metrics']

        # 准备表格数据
        table_data = []
        metrics_names = [
            ('总收益率', 'total_return', '{:.2%}'),
            ('年化收益率', 'annualized_return', '{:.2%}'),
            ('年化波动率', 'volatility', '{:.2%}'),
            ('夏普比率', 'sharpe_ratio', '{:.3f}'),
            ('最大回撤', 'max_drawdown', '{:.2%}'),
            ('Calmar比率', 'calmar_ratio', '{:.3f}'),
            ('Sortino比率', 'sortino_ratio', '{:.3f}')
        ]

        for name, key, fmt in metrics_names:
            rl_val = metrics['rl_strategy'].get(key, 0)
            bench_val = metrics['benchmark'].get(key, 0)

            table_data.append([
                name,
                fmt.format(rl_val),
                fmt.format(bench_val)
            ])

        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=['指标', 'RL策略', '基准'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.4, 0.3, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 设置表格样式
        for i in range(len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j == 0:  # 指标名称列
                        cell.set_facecolor('#E8F5E8')
                    elif j == 1:  # RL策略列
                        cell.set_facecolor('#FFF3E0')
                    else:  # 基准列
                        cell.set_facecolor('#E3F2FD')

        ax.set_title('关键指标统计', fontsize=12, fontweight='bold', pad=20)

    def _calculate_drawdown_series(self, values: np.ndarray) -> np.ndarray:
        """计算回撤序列"""
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return -drawdown

    def _calculate_rolling_sharpe(self, returns: np.ndarray, window: int) -> np.ndarray:
        """计算滚动夏普比率"""
        rolling_sharpe = []
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            if np.std(window_returns) > 0:
                sharpe = np.mean(window_returns) / np.std(window_returns) * np.sqrt(252)
            else:
                sharpe = 0
            rolling_sharpe.append(sharpe)

        return np.array(rolling_sharpe)


def create_qlib_report(results: Dict[str, Any],
                      output_dir: str = "results",
                      timestamp: str = None) -> str:
    """
    创建Qlib风格的完整分析报告

    Args:
        results: 回测结果
        output_dir: 输出目录
        timestamp: 时间戳

    Returns:
        报告文件路径
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建可视化器
    visualizer = QlibVisualizer()

    # 生成可视化图表
    chart_path = os.path.join(output_dir, f"qlib_analysis_{timestamp}.png")
    visualizer.plot_portfolio_analysis(results, chart_path)

    logger.info(f"Qlib风格分析报告已生成: {chart_path}")
    return chart_path