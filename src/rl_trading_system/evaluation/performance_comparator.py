"""
性能比较器模块

提供策略性能的对比分析和可视化功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from .statistical_tests import SignificanceTest, StatisticalTestResult
from ..backtest.enhanced_backtest_engine import EnhancedBacktestResult

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """比较结果数据类"""
    strategy_names: List[str]
    metrics_comparison: pd.DataFrame
    statistical_tests: Dict[str, Dict[str, StatisticalTestResult]]
    performance_ranking: List[Tuple[str, float]]
    improvement_summary: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'strategy_names': self.strategy_names,
            'metrics_comparison': self.metrics_comparison.to_dict(),
            'statistical_tests_summary': {
                comparison: {
                    test: result.to_dict() 
                    for test, result in tests.items()
                }
                for comparison, tests in self.statistical_tests.items()
            },
            'performance_ranking': self.performance_ranking,
            'improvement_summary': self.improvement_summary
        }


class PerformanceComparator:
    """
    性能比较器
    
    提供多策略性能对比分析功能。
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化性能比较器
        
        Args:
            confidence_level: 统计检验置信水平
        """
        self.confidence_level = confidence_level
        self.significance_test = SignificanceTest(confidence_level)
        
        logger.info(f"初始化性能比较器，置信水平: {confidence_level}")
    
    def compare_strategies(self, 
                          backtest_results: Dict[str, EnhancedBacktestResult],
                          baseline_strategy: Optional[str] = None) -> ComparisonResult:
        """
        比较多个策略的性能
        
        Args:
            backtest_results: 策略名称到回测结果的映射
            baseline_strategy: 基准策略名称
            
        Returns:
            比较结果
        """
        if len(backtest_results) < 2:
            raise ValueError("至少需要2个策略进行比较")
        
        strategy_names = list(backtest_results.keys())
        logger.info(f"开始比较策略: {strategy_names}")
        
        # 构建指标比较表
        metrics_comparison = self._build_metrics_comparison(backtest_results)
        
        # 执行统计检验
        statistical_tests = self._perform_statistical_tests(backtest_results)
        
        # 计算性能排名
        performance_ranking = self._calculate_performance_ranking(backtest_results)
        
        # 计算改进摘要
        improvement_summary = self._calculate_improvement_summary(
            backtest_results, baseline_strategy
        )
        
        return ComparisonResult(
            strategy_names=strategy_names,
            metrics_comparison=metrics_comparison,
            statistical_tests=statistical_tests,
            performance_ranking=performance_ranking,
            improvement_summary=improvement_summary
        )
    
    def _build_metrics_comparison(self, 
                                 backtest_results: Dict[str, EnhancedBacktestResult]) -> pd.DataFrame:
        """构建指标比较表"""
        metrics_data = []
        
        for strategy_name, result in backtest_results.items():
            metrics_data.append({
                '策略名称': strategy_name,
                '总收益率': result.total_return,
                '年化收益率': result.annual_return,
                '波动率': result.volatility,
                '夏普比率': result.sharpe_ratio,
                '最大回撤': result.max_drawdown,
                '卡尔玛比率': result.calmar_ratio,
                '回撤改善度': result.drawdown_improvement,
                '止损触发次数': result.stop_loss_trigger_count,
                '仓位调整次数': result.position_adjustment_count,
                '风险预算使用率': result.risk_budget_utilization,
                '胜率': result.win_rate,
                '盈亏比': result.profit_factor,
                '平均交易收益': result.average_trade_return
            })
        
        return pd.DataFrame(metrics_data).set_index('策略名称')
    
    def _perform_statistical_tests(self, 
                                  backtest_results: Dict[str, EnhancedBacktestResult]) -> Dict[str, Dict[str, StatisticalTestResult]]:
        """执行统计检验"""
        statistical_tests = {}
        strategy_names = list(backtest_results.keys())
        
        # 获取收益序列
        returns_data = {}
        for name, result in backtest_results.items():
            if hasattr(result, 'portfolio_values') and result.portfolio_values is not None:
                returns_data[name] = result.portfolio_values.pct_change().dropna()
            else:
                logger.warning(f"策略 {name} 缺少收益序列数据，跳过统计检验")
                continue
        
        # 两两比较
        for i, strategy_a in enumerate(strategy_names):
            for j, strategy_b in enumerate(strategy_names[i+1:], i+1):
                if strategy_a not in returns_data or strategy_b not in returns_data:
                    continue
                
                comparison_key = f"{strategy_a} vs {strategy_b}"
                
                try:
                    # 执行综合统计检验
                    test_results = self.significance_test.comprehensive_comparison(
                        returns_data[strategy_a], 
                        returns_data[strategy_b],
                        (strategy_a, strategy_b)
                    )
                    
                    statistical_tests[comparison_key] = test_results
                    
                except Exception as e:
                    logger.warning(f"策略比较 {comparison_key} 的统计检验失败: {e}")
        
        return statistical_tests
    
    def _calculate_performance_ranking(self, 
                                     backtest_results: Dict[str, EnhancedBacktestResult]) -> List[Tuple[str, float]]:
        """计算性能排名"""
        # 使用综合评分进行排名
        scores = []
        
        for strategy_name, result in backtest_results.items():
            # 综合评分：夏普比率 40% + 回撤改善 30% + 卡尔玛比率 30%
            score = (0.4 * result.sharpe_ratio + 
                    0.3 * (-result.max_drawdown) +  # 负号因为回撤越小越好
                    0.3 * result.calmar_ratio)
            
            scores.append((strategy_name, score))
        
        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def _calculate_improvement_summary(self, 
                                     backtest_results: Dict[str, EnhancedBacktestResult],
                                     baseline_strategy: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """计算改进摘要"""
        if baseline_strategy is None:
            # 选择第一个策略作为基准
            baseline_strategy = list(backtest_results.keys())[0]
        
        if baseline_strategy not in backtest_results:
            raise ValueError(f"基准策略 {baseline_strategy} 不存在")
        
        baseline_result = backtest_results[baseline_strategy]
        improvement_summary = {}
        
        for strategy_name, result in backtest_results.items():
            if strategy_name == baseline_strategy:
                continue
            
            improvements = {
                '年化收益率改进': result.annual_return - baseline_result.annual_return,
                '波动率变化': result.volatility - baseline_result.volatility,
                '夏普比率改进': result.sharpe_ratio - baseline_result.sharpe_ratio,
                '最大回撤改进': baseline_result.max_drawdown - result.max_drawdown,  # 正值表示改进
                '卡尔玛比率改进': result.calmar_ratio - baseline_result.calmar_ratio,
                '胜率改进': result.win_rate - baseline_result.win_rate,
                '盈亏比改进': result.profit_factor - baseline_result.profit_factor,
                '相对改进百分比': {
                    '年化收益率': (result.annual_return / baseline_result.annual_return - 1) * 100 if baseline_result.annual_return != 0 else 0,
                    '夏普比率': (result.sharpe_ratio / baseline_result.sharpe_ratio - 1) * 100 if baseline_result.sharpe_ratio != 0 else 0,
                    '最大回撤': (baseline_result.max_drawdown / result.max_drawdown - 1) * 100 if result.max_drawdown != 0 else 0
                }
            }
            
            improvement_summary[strategy_name] = improvements
        
        return improvement_summary
    
    def create_performance_dashboard(self, 
                                   comparison_result: ComparisonResult,
                                   output_path: str):
        """创建性能对比仪表板"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 创建综合图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('策略性能对比分析', fontsize=16, y=0.95)
        
        metrics_df = comparison_result.metrics_comparison
        
        # 1. 收益率对比
        ax1 = axes[0, 0]
        metrics_df['年化收益率'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('年化收益率对比')
        ax1.set_ylabel('年化收益率')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 风险指标对比
        ax2 = axes[0, 1]
        risk_metrics = metrics_df[['波动率', '最大回撤']].abs()
        risk_metrics.plot(kind='bar', ax=ax2)
        ax2.set_title('风险指标对比')
        ax2.set_ylabel('风险指标值')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        
        # 3. 风险调整收益对比
        ax3 = axes[0, 2]
        risk_adj_metrics = metrics_df[['夏普比率', '卡尔玛比率']]
        risk_adj_metrics.plot(kind='bar', ax=ax3)
        ax3.set_title('风险调整收益对比')
        ax3.set_ylabel('比率值')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        
        # 4. 回撤控制效果
        ax4 = axes[1, 0]
        drawdown_metrics = metrics_df[['最大回撤', '回撤改善度']].abs()
        drawdown_metrics.plot(kind='bar', ax=ax4)
        ax4.set_title('回撤控制效果')
        ax4.set_ylabel('回撤指标')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        
        # 5. 交易统计
        ax5 = axes[1, 1]
        trading_metrics = metrics_df[['胜率', '风险预算使用率']]
        trading_metrics.plot(kind='bar', ax=ax5)
        ax5.set_title('交易统计指标')
        ax5.set_ylabel('比率值')
        ax5.tick_params(axis='x', rotation=45)
        ax5.legend()
        
        # 6. 综合评分排名
        ax6 = axes[1, 2]
        ranking_df = pd.DataFrame(comparison_result.performance_ranking, 
                                 columns=['策略', '综合评分'])
        ranking_df.set_index('策略')['综合评分'].plot(kind='bar', ax=ax6, color='lightgreen')
        ax6.set_title('综合评分排名')
        ax6.set_ylabel('综合评分')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"性能对比仪表板已保存到: {output_path / 'performance_comparison_dashboard.png'}")
    
    def create_statistical_significance_report(self, 
                                             comparison_result: ComparisonResult,
                                             output_path: str):
        """创建统计显著性报告"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成HTML报告
        html_content = self._generate_html_report(comparison_result)
        
        with open(output_path / 'statistical_significance_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 生成文本报告
        text_report = self._generate_text_report(comparison_result)
        
        with open(output_path / 'statistical_significance_report.txt', 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        logger.info(f"统计显著性报告已保存到: {output_path}")
    
    def _generate_html_report(self, comparison_result: ComparisonResult) -> str:
        """生成HTML报告"""
        html_template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>策略性能对比分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .significant {{ color: #d32f2f; font-weight: bold; }}
                .not-significant {{ color: #388e3c; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>策略性能对比分析报告</h1>
            
            <div class="summary">
                <h2>执行摘要</h2>
                <p>本报告对比分析了 {strategy_count} 个交易策略的性能表现。</p>
                <p>分析期间包含了回撤控制效果、统计显著性检验和综合性能评估。</p>
            </div>
            
            <h2>1. 性能指标对比</h2>
            {metrics_table}
            
            <h2>2. 性能排名</h2>
            {ranking_table}
            
            <h2>3. 统计显著性检验结果</h2>
            {statistical_tests}
            
            <h2>4. 改进效果摘要</h2>
            {improvement_summary}
            
            <div class="summary">
                <h2>结论与建议</h2>
                <p>基于以上分析，建议选择综合评分最高且统计显著优于基准的策略。</p>
                <p>同时需要关注回撤控制效果和风险调整收益的平衡。</p>
            </div>
        </body>
        </html>
        """
        
        # 填充模板内容
        strategy_count = len(comparison_result.strategy_names)
        metrics_table = comparison_result.metrics_comparison.to_html(classes='metrics-table')
        
        ranking_df = pd.DataFrame(comparison_result.performance_ranking, 
                                 columns=['策略名称', '综合评分'])
        ranking_table = ranking_df.to_html(index=False, classes='ranking-table')
        
        # 统计检验结果
        statistical_tests_html = ""
        for comparison, tests in comparison_result.statistical_tests.items():
            statistical_tests_html += f"<h3>{comparison}</h3>"
            for test_name, result in tests.items():
                significance_class = "significant" if result.is_significant else "not-significant"
                statistical_tests_html += f"""
                <p class="{significance_class}">
                    <strong>{result.test_name}</strong>: 
                    统计量={result.statistic:.4f}, p值={result.p_value:.4f}, 
                    {'显著' if result.is_significant else '不显著'}
                    <br><em>{result.interpretation}</em>
                </p>
                """
        
        # 改进摘要
        improvement_html = ""
        for strategy, improvements in comparison_result.improvement_summary.items():
            improvement_html += f"<h3>{strategy}</h3><ul>"
            for metric, value in improvements.items():
                if metric != '相对改进百分比':
                    improvement_html += f"<li>{metric}: {value:.4f}</li>"
            improvement_html += "</ul>"
        
        return html_template.format(
            strategy_count=strategy_count,
            metrics_table=metrics_table,
            ranking_table=ranking_table,
            statistical_tests=statistical_tests_html,
            improvement_summary=improvement_html
        )
    
    def _generate_text_report(self, comparison_result: ComparisonResult) -> str:
        """生成文本报告"""
        report_lines = [
            "=" * 60,
            "策略性能对比分析报告",
            "=" * 60,
            "",
            f"分析策略数量: {len(comparison_result.strategy_names)}",
            f"策略列表: {', '.join(comparison_result.strategy_names)}",
            "",
            "1. 性能排名",
            "-" * 30
        ]
        
        for rank, (strategy, score) in enumerate(comparison_result.performance_ranking, 1):
            report_lines.append(f"{rank}. {strategy}: {score:.4f}")
        
        report_lines.extend([
            "",
            "2. 统计显著性检验摘要",
            "-" * 40
        ])
        
        for comparison, tests in comparison_result.statistical_tests.items():
            report_lines.append(f"\n{comparison}:")
            for test_name, result in tests.items():
                significance = "显著" if result.is_significant else "不显著"
                report_lines.append(f"  - {result.test_name}: p={result.p_value:.4f} ({significance})")
        
        return "\n".join(report_lines)
    
    def save_comparison_results(self, 
                               comparison_result: ComparisonResult,
                               output_path: str):
        """保存比较结果"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式的完整结果
        with open(output_path / 'comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式的指标对比
        comparison_result.metrics_comparison.to_csv(output_path / 'metrics_comparison.csv')
        
        # 保存排名结果
        ranking_df = pd.DataFrame(comparison_result.performance_ranking, 
                                 columns=['策略名称', '综合评分'])
        ranking_df.to_csv(output_path / 'performance_ranking.csv', index=False)
        
        logger.info(f"比较结果已保存到: {output_path}")