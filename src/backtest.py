"""
回测和可视化模块
集成Qlib回测功能，提供完整的策略分析和可视化
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

import qlib
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.strategy.base import BaseStrategy
from qlib.data.data import D

from model import PortfolioMetrics

logger = logging.getLogger(__name__)

# 设置中文字体
from font_config import setup_chinese_font
setup_chinese_font()


class RLStrategy(BaseStrategy):
    """基于强化学习的投资策略"""

    def __init__(self, model, env, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.env = env
        self.positions = {}

    def generate_trade_decision(self, execute_result=None):
        """生成交易决策"""
        # 获取当前观察
        obs = self.env._get_observation()

        # 模型预测
        action, _ = self.model.predict(obs, deterministic=True)

        # 转换为交易指令
        trade_decisions = []
        for i, weight in enumerate(action):
            stock = self.env.stock_list[i]
            trade_decisions.append({
                'instrument': stock,
                'amount': weight,
                'direction': 1 if weight > 0 else 0
            })

        return trade_decisions


class BacktestAnalyzer:
    """回测分析器"""

    def __init__(self):
        self.results = {}
        self.benchmark_results = {}

    def run_backtest(self,
                    model,
                    test_data: pd.DataFrame,
                    initial_cash: float = 1000000,
                    benchmark: str = "SH000300",
                    env_config: dict = None) -> Dict[str, Any]:
        """
        运行回测

        Args:
            model: 训练好的RL模型
            test_data: 测试数据
            initial_cash: 初始资金
            benchmark: 基准指数
            env_config: 环境配置字典

        Returns:
            回测结果字典
        """
        logger.info("开始回测分析...")

        # 提取时间范围和股票列表
        # Qlib数据索引是(instrument, datetime)
        start_time = test_data.index.get_level_values(1).min()
        end_time = test_data.index.get_level_values(1).max()
        stock_list = list(test_data.index.get_level_values(0).unique())

        logger.info(f"回测期间: {start_time} - {end_time}")
        logger.info(f"股票数量: {len(stock_list)}")

        # 运行RL策略回测
        rl_results = self._run_rl_backtest(model, test_data, initial_cash, env_config)

        # 运行基准回测
        benchmark_results = self._run_benchmark_backtest(
            stock_list, start_time, end_time, initial_cash, benchmark, test_data
        )

        # 计算性能指标
        performance_metrics = self._calculate_performance_metrics(
            rl_results, benchmark_results
        )

        # 整理结果
        backtest_results = {
            'rl_strategy': rl_results,
            'benchmark': benchmark_results,
            'performance_metrics': performance_metrics,
            'period': {'start': str(start_time), 'end': str(end_time)},
            'universe': stock_list
        }

        self.results = backtest_results
        logger.info("回测分析完成")

        return backtest_results

    def _run_rl_backtest(self, model, test_data: pd.DataFrame, initial_cash: float, env_config: dict = None) -> Dict[str, Any]:
        """运行RL策略回测"""
        from env import PortfolioEnv

        # 使用配置文件中的环境参数，如果没有则使用默认值
        if env_config is None:
            env_config = {}
        
        lookback_window = env_config.get('lookback_window', 30)
        transaction_cost = env_config.get('transaction_cost', 0.003)
        features = env_config.get('features', ['$close', '$open', '$high', '$low', '$volume', '$change', '$factor'])
        
        logger.info(f"环境参数: lookback_window={lookback_window}, transaction_cost={transaction_cost}")
        logger.info(f"特征列表: {features}")
        
        # 创建测试环境（参数与训练时一致）
        env = PortfolioEnv(
            data=test_data,
            initial_cash=initial_cash,
            lookback_window=lookback_window,
            transaction_cost=transaction_cost,
            features=features
        )

        # 重置环境
        obs, _ = env.reset()

        # 记录结果
        value_history = [initial_cash]
        weight_history = []
        action_history = []
        date_history = []

        step = 0
        while True:
            # 模型预测
            action, _ = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)

            # 记录数据
            value_history.append(info['total_value'])
            weight_history.append(info['weights'].copy())
            action_history.append(action.copy())

            if step < len(env.time_index):
                date_history.append(env.time_index[step])

            step += 1

            if terminated or truncated:
                break

        # 计算收益率序列
        values = np.array(value_history)
        returns = np.diff(values) / values[:-1]

        return {
            'values': values,
            'returns': returns,
            'weights': weight_history,
            'actions': action_history,
            'dates': date_history,
            'final_value': values[-1],
            'total_return': (values[-1] / values[0]) - 1
        }

    def _run_benchmark_backtest(self,
                               stock_list: List[str],
                               start_time,
                               end_time,
                               initial_cash: float,
                               benchmark: str,
                               test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """运行基准策略回测"""
        # 直接使用等权重策略作为基准，避免复杂的基准数据获取
        logger.info(f"使用等权重策略作为基准")
        return self._run_equal_weight_benchmark(stock_list, start_time, end_time, initial_cash, test_data)

    def _run_equal_weight_benchmark(self,
                                   stock_list: List[str],
                                   start_time,
                                   end_time,
                                   initial_cash: float,
                                   test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """等权重基准策略"""
        if test_data is None or test_data.empty:
            raise RuntimeError("等权重基准计算失败: 无可用测试数据")

        # 直接使用传入的测试数据计算等权重组合收益
        # 按时间分组，计算每个时间点的平均收益率
        close_data = test_data['$close'].unstack(level=0)  # 转换为时间×股票的矩阵
        
        # 计算日收益率
        daily_returns = close_data.pct_change().mean(axis=1, skipna=True)  # 等权重平均
        daily_returns = daily_returns.fillna(0)  # 填充NaN值

        # 计算组合价值
        portfolio_values = initial_cash * np.cumprod(1 + np.concatenate([[0], daily_returns.values]))

        return {
            'values': portfolio_values,
            'returns': daily_returns.values,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / portfolio_values[0]) - 1,
            'name': '等权重组合'
        }

    def _calculate_performance_metrics(self,
                                     rl_results: Dict[str, Any],
                                     benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算性能指标"""
        metrics = {}

        # RL策略指标
        rl_returns = rl_results['returns']
        rl_values = rl_results['values']

        metrics['rl_strategy'] = {
            'total_return': rl_results['total_return'],
            'annualized_return': (1 + rl_results['total_return']) ** (252 / len(rl_returns)) - 1,
            'volatility': np.std(rl_returns) * np.sqrt(252),
            'sharpe_ratio': PortfolioMetrics.calculate_sharpe_ratio(rl_returns),
            'max_drawdown': PortfolioMetrics.calculate_max_drawdown(rl_values),
            'calmar_ratio': PortfolioMetrics.calculate_calmar_ratio(rl_returns,
                           PortfolioMetrics.calculate_max_drawdown(rl_values)),
            'sortino_ratio': PortfolioMetrics.calculate_sortino_ratio(rl_returns)
        }

        # 基准指标
        bench_returns = benchmark_results['returns']
        bench_values = benchmark_results['values']

        metrics['benchmark'] = {
            'total_return': benchmark_results['total_return'],
            'annualized_return': (1 + benchmark_results['total_return']) ** (252 / len(bench_returns)) - 1,
            'volatility': np.std(bench_returns) * np.sqrt(252),
            'sharpe_ratio': PortfolioMetrics.calculate_sharpe_ratio(bench_returns),
            'max_drawdown': PortfolioMetrics.calculate_max_drawdown(bench_values),
            'calmar_ratio': PortfolioMetrics.calculate_calmar_ratio(bench_returns,
                           PortfolioMetrics.calculate_max_drawdown(bench_values)),
            'sortino_ratio': PortfolioMetrics.calculate_sortino_ratio(bench_returns)
        }

        # 相对指标
        if len(rl_returns) == len(bench_returns):
            metrics['relative'] = {
                'excess_return': metrics['rl_strategy']['total_return'] - metrics['benchmark']['total_return'],
                'information_ratio': PortfolioMetrics.calculate_information_ratio(rl_returns, bench_returns),
                'tracking_error': np.std(rl_returns - bench_returns) * np.sqrt(252),
                'beta': np.cov(rl_returns, bench_returns)[0, 1] / np.var(bench_returns) if np.var(bench_returns) > 0 else 0,
                'alpha': metrics['rl_strategy']['annualized_return'] -
                        metrics['benchmark']['annualized_return'] * metrics.get('relative', {}).get('beta', 1)
            }

        return metrics

    def create_performance_report(self, save_path: str = None) -> str:
        """
        创建性能报告

        Args:
            save_path: 保存路径

        Returns:
            报告内容
        """
        if not self.results:
            raise ValueError("请先运行回测")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("强化学习投资策略回测报告")
        report_lines.append("=" * 80)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"回测期间: {self.results['period']['start']} - {self.results['period']['end']}")
        report_lines.append(f"股票池大小: {len(self.results['universe'])}")
        report_lines.append("")

        # 策略表现
        rl_metrics = self.results['performance_metrics']['rl_strategy']
        bench_metrics = self.results['performance_metrics']['benchmark']

        report_lines.append("策略表现对比")
        report_lines.append("-" * 50)
        report_lines.append(f"{'指标':<20} {'RL策略':<15} {'基准':<15} {'超额':<15}")
        report_lines.append("-" * 65)

        comparisons = [
            ('总收益率', 'total_return', '{:.2%}'),
            ('年化收益率', 'annualized_return', '{:.2%}'),
            ('年化波动率', 'volatility', '{:.2%}'),
            ('夏普比率', 'sharpe_ratio', '{:.3f}'),
            ('最大回撤', 'max_drawdown', '{:.2%}'),
            ('Calmar比率', 'calmar_ratio', '{:.3f}'),
            ('Sortino比率', 'sortino_ratio', '{:.3f}')
        ]

        for name, key, fmt in comparisons:
            rl_val = rl_metrics.get(key, 0)
            bench_val = bench_metrics.get(key, 0)
            excess = rl_val - bench_val

            report_lines.append(f"{name:<20} {fmt.format(rl_val):<15} {fmt.format(bench_val):<15} {fmt.format(excess):<15}")

        # 相对指标
        if 'relative' in self.results['performance_metrics']:
            rel_metrics = self.results['performance_metrics']['relative']
            report_lines.append("")
            report_lines.append("相对指标")
            report_lines.append("-" * 30)
            report_lines.append(f"信息比率: {rel_metrics.get('information_ratio', 0):.3f}")
            report_lines.append(f"跟踪误差: {rel_metrics.get('tracking_error', 0):.2%}")
            report_lines.append(f"Beta: {rel_metrics.get('beta', 0):.3f}")
            report_lines.append(f"Alpha: {rel_metrics.get('alpha', 0):.2%}")

        # 月度表现分析
        report_lines.append("")
        report_lines.append("月度收益分析")
        report_lines.append("-" * 30)

        rl_returns = self.results['rl_strategy']['returns']
        if len(rl_returns) > 0:
            # 简化的月度分析
            monthly_return = np.mean(rl_returns) * 21  # 假设21个交易日一个月
            win_rate = np.sum(rl_returns > 0) / len(rl_returns)

            report_lines.append(f"平均月度收益: {monthly_return:.2%}")
            report_lines.append(f"胜率: {win_rate:.2%}")

        report_content = "\n".join(report_lines)

        # 保存报告
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"报告已保存: {save_path}")

        return report_content

    def plot_performance(self, save_path: str = None, figsize: Tuple[int, int] = (15, 12)):
        """
        绘制性能分析图表

        Args:
            save_path: 保存路径
            figsize: 图片尺寸
        """
        if not self.results:
            raise ValueError("请先运行回测")

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('强化学习投资策略性能分析', fontsize=16, fontweight='bold')

        # 1. 净值曲线对比
        rl_values = self.results['rl_strategy']['values']
        bench_values = self.results['benchmark']['values']

        # 归一化到相同长度
        min_len = min(len(rl_values), len(bench_values))
        rl_values = rl_values[:min_len]
        bench_values = bench_values[:min_len]

        axes[0, 0].plot(rl_values / rl_values[0], label='RL策略', linewidth=2)
        axes[0, 0].plot(bench_values / bench_values[0], label=self.results['benchmark']['name'], linewidth=2)
        axes[0, 0].set_title('净值曲线对比')
        axes[0, 0].set_ylabel('累计收益')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 回撤分析
        rl_drawdown = self._calculate_drawdown_series(rl_values)
        bench_drawdown = self._calculate_drawdown_series(bench_values)

        axes[0, 1].fill_between(range(len(rl_drawdown)), rl_drawdown, 0,
                               alpha=0.6, label='RL策略回撤')
        axes[0, 1].fill_between(range(len(bench_drawdown)), bench_drawdown, 0,
                               alpha=0.6, label='基准回撤')
        axes[0, 1].set_title('回撤分析')
        axes[0, 1].set_ylabel('回撤')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 收益率分布
        rl_returns = self.results['rl_strategy']['returns']
        bench_returns = self.results['benchmark']['returns']

        axes[0, 2].hist(rl_returns, bins=50, alpha=0.7, label='RL策略', density=True)
        axes[0, 2].hist(bench_returns[:len(rl_returns)], bins=50, alpha=0.7,
                       label='基准', density=True)
        axes[0, 2].set_title('日收益率分布')
        axes[0, 2].set_xlabel('日收益率')
        axes[0, 2].set_ylabel('密度')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. 滚动夏普比率
        window = 60  # 60天滚动窗口
        rl_rolling_sharpe = self._calculate_rolling_sharpe(rl_returns, window)
        bench_rolling_sharpe = self._calculate_rolling_sharpe(bench_returns[:len(rl_returns)], window)

        axes[1, 0].plot(rl_rolling_sharpe, label='RL策略', linewidth=2)
        axes[1, 0].plot(bench_rolling_sharpe, label='基准', linewidth=2)
        axes[1, 0].set_title(f'{window}日滚动夏普比率')
        axes[1, 0].set_ylabel('夏普比率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. 持仓权重变化（如果有）
        if 'weights' in self.results['rl_strategy'] and self.results['rl_strategy']['weights']:
            weights_data = np.array(self.results['rl_strategy']['weights'])
            if weights_data.ndim == 2 and weights_data.shape[1] > 0:
                # 只显示前5个权重最大的股票
                avg_weights = np.mean(weights_data, axis=0)
                top_indices = np.argsort(avg_weights)[-5:]

                for i, idx in enumerate(top_indices):
                    axes[1, 1].plot(weights_data[:, idx], label=f'股票{idx+1}')

                axes[1, 1].set_title('主要持仓权重变化')
                axes[1, 1].set_ylabel('权重')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, '无权重数据', ha='center', va='center',
                               transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, '无权重数据', ha='center', va='center',
                           transform=axes[1, 1].transAxes)

        # 6. 风险收益散点图
        metrics = self.results['performance_metrics']

        strategies = ['rl_strategy', 'benchmark']
        names = ['RL策略', self.results['benchmark']['name']]
        colors = ['red', 'blue']

        for i, (strategy, name, color) in enumerate(zip(strategies, names, colors)):
            vol = metrics[strategy]['volatility']
            ret = metrics[strategy]['annualized_return']
            axes[1, 2].scatter(vol, ret, s=100, c=color, label=name, alpha=0.7)

        axes[1, 2].set_title('风险收益特征')
        axes[1, 2].set_xlabel('年化波动率')
        axes[1, 2].set_ylabel('年化收益率')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"图表已保存: {save_path}")

        plt.show()

    def _calculate_drawdown_series(self, values: np.ndarray) -> np.ndarray:
        """计算回撤序列"""
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return -drawdown  # 负数表示回撤

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

    def export_results(self, export_path: str):
        """
        导出回测结果到Excel

        Args:
            export_path: 导出路径
        """
        if not self.results:
            raise ValueError("请先运行回测")

        try:
            with pd.ExcelWriter(export_path, engine='openpyxl') as writer:
                # 性能指标对比
                metrics_df = self._create_metrics_dataframe()
                metrics_df.to_excel(writer, sheet_name='性能指标', index=True)

                # 净值序列
                values_df = self._create_values_dataframe()
                values_df.to_excel(writer, sheet_name='净值序列', index=False)

                # 收益率序列
                returns_df = self._create_returns_dataframe()
                returns_df.to_excel(writer, sheet_name='收益率序列', index=False)

                # 持仓权重（如果有）
                if 'weights' in self.results['rl_strategy'] and self.results['rl_strategy']['weights']:
                    weights_df = self._create_weights_dataframe()
                    weights_df.to_excel(writer, sheet_name='持仓权重', index=False)

            logger.info(f"结果已导出到: {export_path}")

        except Exception as e:
            logger.error(f"导出失败: {e}")
            raise RuntimeError(f"导出失败: {e}")

    def _create_metrics_dataframe(self) -> pd.DataFrame:
        """创建指标对比DataFrame"""
        metrics = self.results['performance_metrics']

        data = {
            'RL策略': [
                metrics['rl_strategy']['total_return'],
                metrics['rl_strategy']['annualized_return'],
                metrics['rl_strategy']['volatility'],
                metrics['rl_strategy']['sharpe_ratio'],
                metrics['rl_strategy']['max_drawdown'],
                metrics['rl_strategy']['calmar_ratio'],
                metrics['rl_strategy']['sortino_ratio']
            ],
            '基准': [
                metrics['benchmark']['total_return'],
                metrics['benchmark']['annualized_return'],
                metrics['benchmark']['volatility'],
                metrics['benchmark']['sharpe_ratio'],
                metrics['benchmark']['max_drawdown'],
                metrics['benchmark']['calmar_ratio'],
                metrics['benchmark']['sortino_ratio']
            ]
        }

        if 'relative' in metrics:
            data['超额'] = [
                metrics['relative']['excess_return'],
                data['RL策略'][1] - data['基准'][1],  # 年化超额收益
                data['RL策略'][2] - data['基准'][2],  # 波动率差异
                data['RL策略'][3] - data['基准'][3],  # 夏普比率差异
                data['RL策略'][4] - data['基准'][4],  # 最大回撤差异
                data['RL策略'][5] - data['基准'][5],  # Calmar比率差异
                data['RL策略'][6] - data['基准'][6]   # Sortino比率差异
            ]

        index = ['总收益率', '年化收益率', '年化波动率', '夏普比率',
                '最大回撤', 'Calmar比率', 'Sortino比率']

        return pd.DataFrame(data, index=index)

    def _create_values_dataframe(self) -> pd.DataFrame:
        """创建净值序列DataFrame"""
        rl_values = self.results['rl_strategy']['values']
        bench_values = self.results['benchmark']['values']

        # 归一化长度
        min_len = min(len(rl_values), len(bench_values))

        return pd.DataFrame({
            'RL策略净值': rl_values[:min_len],
            '基准净值': bench_values[:min_len],
            'RL策略累计收益': (rl_values[:min_len] / rl_values[0]) - 1,
            '基准累计收益': (bench_values[:min_len] / bench_values[0]) - 1
        })

    def _create_returns_dataframe(self) -> pd.DataFrame:
        """创建收益率序列DataFrame"""
        rl_returns = self.results['rl_strategy']['returns']
        bench_returns = self.results['benchmark']['returns']

        min_len = min(len(rl_returns), len(bench_returns))

        return pd.DataFrame({
            'RL策略日收益': rl_returns[:min_len],
            '基准日收益': bench_returns[:min_len],
            '超额收益': rl_returns[:min_len] - bench_returns[:min_len]
        })

    def _create_weights_dataframe(self) -> pd.DataFrame:
        """创建持仓权重DataFrame"""
        weights = self.results['rl_strategy']['weights']

        if not weights:
            return pd.DataFrame()

        weights_array = np.array(weights)
        columns = [f'股票{i+1}' for i in range(weights_array.shape[1])]

        return pd.DataFrame(weights_array, columns=columns)


if __name__ == "__main__":
    # 示例用法
    print("回测分析器已实现，使用方法:")
    print("analyzer = BacktestAnalyzer()")
    print("results = analyzer.run_backtest(model, test_data)")
    print("report = analyzer.create_performance_report()")
    print("analyzer.plot_performance()")
    print("analyzer.export_results('results.xlsx')")