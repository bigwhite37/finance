"""
回测报告生成模块
实现自动生成HTML报告和可视化图表，收益曲线、持仓分析和风险分解可视化，因子暴露分析和绩效归因报告
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Tuple, Any
from decimal import Decimal
from dataclasses import dataclass, field
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .performance_metrics import PortfolioMetrics
from ..backtest.multi_frequency_backtest import Trade


@dataclass
class ReportData:
    """报告数据类"""
    returns: pd.Series
    portfolio_values: pd.Series
    benchmark_returns: pd.Series
    trades: List[Trade]
    positions: Dict[str, Dict[str, float]]
    start_date: date
    end_date: date
    initial_capital: float
    
    def __post_init__(self):
        """数据验证"""
        if self.returns.empty:
            raise ValueError("收益率序列不能为空")
        if len(self.returns) != len(self.portfolio_values):
            raise ValueError("收益率序列和组合价值序列长度不匹配")
        if self.initial_capital <= 0:
            raise ValueError("初始资本必须为正数")
    
    def calculate_metrics(self) -> Dict[str, Dict[str, float]]:
        """计算绩效指标"""
        portfolio_metrics = PortfolioMetrics(
            returns=self.returns,
            portfolio_values=self.portfolio_values,
            trades=self.trades
        )
        return portfolio_metrics.calculate_comprehensive_metrics()


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self, figure_size: Tuple[int, int] = (12, 8), style: str = 'seaborn-v0_8'):
        """
        初始化图表生成器
        
        Args:
            figure_size: 图表尺寸
            style: 图表样式
        """
        self.figure_size = figure_size
        self.style = style
        
        # 配置Plotly默认样式
        self.default_layout = {
            'width': figure_size[0] * 80,
            'height': figure_size[1] * 80,
            'font': {'size': 12},
            'showlegend': True,
            'hovermode': 'x unified'
        }
    
    def generate_returns_chart(self, portfolio_values: pd.Series, 
                             benchmark_values: Optional[pd.Series] = None) -> str:
        """生成收益率图表"""
        if portfolio_values.empty:
            raise ValueError("数据不能为空")
        
        if benchmark_values is not None and len(portfolio_values) != len(benchmark_values):
            raise ValueError("组合价值和基准价值长度不匹配")
        
        # 创建图表
        fig = go.Figure()
        
        # 添加组合收益率曲线
        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode='lines',
            name='投资组合',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # 添加基准收益率曲线
        if benchmark_values is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_values.index,
                y=benchmark_values.values,
                mode='lines',
                name='基准',
                line=dict(color='#ff7f0e', width=2)
            ))
        
        # 设置布局
        fig.update_layout(
            title='投资组合收益率走势',
            xaxis_title='日期',
            yaxis_title='组合价值',
            **self.default_layout
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="returns_chart")
    
    def generate_drawdown_chart(self, portfolio_values: pd.Series) -> str:
        """生成回撤图表"""
        if portfolio_values.empty:
            raise ValueError("数据不能为空")
        
        # 计算回撤
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak * 100
        
        # 创建图表
        fig = go.Figure()
        
        # 添加回撤曲线
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='回撤',
            fill='tonexty',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        # 设置布局
        fig.update_layout(
            title='投资组合回撤分析',
            xaxis_title='日期',
            yaxis_title='回撤 (%)',
            **self.default_layout
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="drawdown_chart")
    
    def generate_rolling_metrics_chart(self, rolling_data: pd.Series, 
                                     metric_name: str) -> str:
        """生成滚动指标图表"""
        if rolling_data.empty:
            raise ValueError("数据不能为空")
        
        # 创建图表
        fig = go.Figure()
        
        # 添加滚动指标曲线
        fig.add_trace(go.Scatter(
            x=rolling_data.index,
            y=rolling_data.values,
            mode='lines',
            name=f'滚动{metric_name}',
            line=dict(color='#2ca02c', width=2)
        ))
        
        # 设置布局
        fig.update_layout(
            title=f'滚动{metric_name}分析',
            xaxis_title='日期',
            yaxis_title=metric_name,
            **self.default_layout
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="rolling_metrics_chart")
    
    def generate_position_analysis_chart(self, positions_data: Dict[str, Dict[str, float]]) -> str:
        """生成持仓分析图表"""
        if not positions_data:
            raise ValueError("持仓数据不能为空")
        
        # 提取数据
        symbols = list(positions_data.keys())
        weights = [positions_data[symbol]['weight'] for symbol in symbols]
        
        # 创建饼图
        fig = go.Figure(data=[go.Pie(
            labels=symbols,
            values=weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        # 设置布局
        fig.update_layout(
            title='持仓分布分析',
            **self.default_layout
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="position_analysis_chart")
    
    def generate_monthly_returns_heatmap(self, monthly_returns: pd.DataFrame) -> str:
        """生成月度收益率热力图"""
        if monthly_returns.empty:
            raise ValueError("月度收益率数据不能为空")
        
        # 创建透视表
        pivot_table = monthly_returns.pivot(index='year', columns='month', values='return')
        
        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=[f'{i}月' for i in pivot_table.columns],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(pivot_table.values * 100, 2),
            texttemplate='%{text}%',
            textfont={'size': 10},
            colorbar=dict(title="收益率 (%)")
        ))
        
        # 设置布局
        fig.update_layout(
            title='月度收益率热力图',
            xaxis_title='月份',
            yaxis_title='年份',
            **self.default_layout
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="monthly_returns_heatmap")
    
    def generate_risk_metrics_radar_chart(self, risk_metrics: Dict[str, float]) -> str:
        """生成风险指标雷达图"""
        if not risk_metrics:
            raise ValueError("风险指标数据不能为空")
        
        # 标准化指标（转换为0-1范围）
        normalized_metrics = {}
        for key, value in risk_metrics.items():
            if key == 'volatility':
                normalized_metrics['波动率'] = min(abs(value) * 10, 1)
            elif key == 'max_drawdown':
                normalized_metrics['最大回撤'] = min(abs(value) * 10, 1)
            elif key == 'var_95':
                normalized_metrics['VaR(95%)'] = min(abs(value) * 20, 1)
            elif key == 'skewness':
                normalized_metrics['偏度'] = abs(value) / 2
            elif key == 'kurtosis':
                normalized_metrics['峰度'] = min(abs(value) / 5, 1)
        
        # 创建雷达图
        categories = list(normalized_metrics.keys())
        values = list(normalized_metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 闭合图形
            theta=categories + [categories[0]],
            fill='toself',
            name='风险指标',
            line=dict(color='rgba(255, 0, 0, 0.8)')
        ))
        
        # 设置布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title='风险指标雷达图',
            **self.default_layout
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="risk_metrics_radar")
    
    def generate_trading_analysis_chart(self, trading_metrics: Dict[str, float]) -> str:
        """生成交易分析图表"""
        if not trading_metrics:
            raise ValueError("交易指标数据不能为空")
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('胜率', '盈亏比', '平均盈利', '平均亏损'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # 胜率指标
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=trading_metrics.get('win_rate', 0) * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "胜率 (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ), row=1, col=1)
        
        # 盈亏比指标
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=trading_metrics.get('profit_loss_ratio', 0),
            title={'text': "盈亏比"},
            delta={'reference': 1, 'relative': True}
        ), row=1, col=2)
        
        # 平均盈利
        fig.add_trace(go.Indicator(
            mode="number",
            value=trading_metrics.get('average_win', 0) * 100,
            title={'text': "平均盈利 (%)"},
            number={'suffix': "%"}
        ), row=2, col=1)
        
        # 平均亏损
        fig.add_trace(go.Indicator(
            mode="number",
            value=abs(trading_metrics.get('average_loss', 0)) * 100,
            title={'text': "平均亏损 (%)"},
            number={'suffix': "%"}
        ), row=2, col=2)
        
        # 设置布局
        fig.update_layout(
            title='交易分析指标',
            **self.default_layout
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="trading_analysis_chart")


class HTMLReportGenerator:
    """HTML报告生成器"""
    
    def __init__(self, template_dir: Optional[str] = None):
        """
        初始化HTML报告生成器
        
        Args:
            template_dir: 模板目录路径
        """
        if template_dir is None:
            # 使用默认模板目录
            current_dir = Path(__file__).parent
            template_dir = current_dir / "templates"
        
        self.template_dir = Path(template_dir)
        self.chart_generator = ChartGenerator()
        
        # 初始化Jinja2环境
        if self.template_dir.exists():
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(['html', 'xml'])
            )
        else:
            self.jinja_env = None
    
    def generate_report(self, report_data: ReportData, output_path: str,
                       template_name: str = "default_report.html",
                       include_benchmark: bool = False) -> None:
        """
        生成HTML报告
        
        Args:
            report_data: 报告数据
            output_path: 输出路径
            template_name: 模板名称
            include_benchmark: 是否包含基准比较
        """
        # 创建输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算指标
        metrics = report_data.calculate_metrics()
        
        # 生成报告内容
        if template_name != "default_report.html":
            # 只有在指定非默认模板时才进行模板检查
            if self.jinja_env is None:
                raise FileNotFoundError(f"模板目录不存在: {self.template_dir}")
                
            template_path = self.template_dir / template_name
            if not template_path.exists():
                raise FileNotFoundError(f"模板文件不存在: {template_path}")
        
        if self.jinja_env is not None and (self.template_dir / template_name).exists():
            # 使用模板生成
            template = self.jinja_env.get_template(template_name)
            
            # 准备模板变量
            template_vars = {
                'report_data': report_data,
                'metrics': metrics,
                'summary_section': self._generate_summary_section(report_data, metrics),
                'performance_section': self._generate_performance_section(report_data, metrics),
                'risk_section': self._generate_risk_section(metrics),
                'trading_section': self._generate_trading_section(report_data, metrics),
                'positions_section': self._generate_positions_section(report_data),
                'benchmark_comparison': self._generate_benchmark_section(report_data) if include_benchmark else "",
                'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            html_content = template.render(**template_vars)
        else:
            # 使用默认模板
            html_content = self._generate_default_report(report_data, metrics, include_benchmark)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_default_report(self, report_data: ReportData, 
                                metrics: Dict[str, Dict[str, float]],
                                include_benchmark: bool = False) -> str:
        """生成默认报告"""
        # 生成各个部分
        summary_section = self._generate_summary_section(report_data, metrics)
        performance_section = self._generate_performance_section(report_data, metrics)
        risk_section = self._generate_risk_section(metrics)
        trading_section = self._generate_trading_section(report_data, metrics)
        positions_section = self._generate_positions_section(report_data)
        benchmark_section = self._generate_benchmark_section(report_data) if include_benchmark else ""
        
        # 生成完整HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>投资组合分析报告</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background: #ecf0f1;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    margin-top: 5px;
                }}
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: #7f8c8d;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>投资组合分析报告</h1>
                <p class="text-center">报告期间: {report_data.start_date} 至 {report_data.end_date}</p>
                
                {summary_section}
                {performance_section}
                {risk_section}
                {trading_section}
                {positions_section}
                {benchmark_section}
                
                <div class="footer">
                    <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>由智能量化交易系统自动生成</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_summary_section(self, report_data: ReportData, 
                                 metrics: Dict[str, Dict[str, float]]) -> str:
        """生成摘要部分"""
        return_metrics = metrics.get('return_metrics', {})
        risk_metrics = metrics.get('risk_metrics', {})
        risk_adjusted = metrics.get('risk_adjusted_metrics', {})
        
        summary_html = f"""
        <section>
            <h2>绩效摘要</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{return_metrics.get('total_return', 0):.2%}</div>
                    <div class="metric-label">总收益率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{return_metrics.get('annualized_return', 0):.2%}</div>
                    <div class="metric-label">年化收益率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('volatility', 0):.2%}</div>
                    <div class="metric-label">年化波动率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('max_drawdown', 0):.2%}</div>
                    <div class="metric-label">最大回撤</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_adjusted.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">夏普比率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_adjusted.get('sortino_ratio', 0):.2f}</div>
                    <div class="metric-label">索提诺比率</div>
                </div>
            </div>
        </section>
        """
        
        return summary_html
    
    def _generate_performance_section(self, report_data: ReportData, 
                                    metrics: Dict[str, Dict[str, float]]) -> str:
        """生成绩效分析部分"""
        # 生成收益率图表
        returns_chart = self.chart_generator.generate_returns_chart(
            report_data.portfolio_values,
            None  # 暂时不包含基准
        )
        
        # 生成回撤图表
        drawdown_chart = self.chart_generator.generate_drawdown_chart(
            report_data.portfolio_values
        )
        
        performance_html = f"""
        <section>
            <h2>绩效分析</h2>
            <div class="chart-container">
                {returns_chart}
            </div>
            <div class="chart-container">
                {drawdown_chart}
            </div>
        </section>
        """
        
        return performance_html
    
    def _generate_risk_section(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """生成风险分析部分"""
        risk_metrics = metrics.get('risk_metrics', {})
        
        # 生成风险指标雷达图
        radar_chart = self.chart_generator.generate_risk_metrics_radar_chart(risk_metrics)
        
        risk_html = f"""
        <section>
            <h2>风险分析</h2>
            <div class="chart-container">
                {radar_chart}
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('var_95', 0):.2%}</div>
                    <div class="metric-label">VaR (95%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('cvar_95', 0):.2%}</div>
                    <div class="metric-label">CVaR (95%)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('skewness', 0):.3f}</div>
                    <div class="metric-label">偏度</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{risk_metrics.get('kurtosis', 0):.3f}</div>
                    <div class="metric-label">峰度</div>
                </div>
            </div>
        </section>
        """
        
        return risk_html
    
    def _generate_trading_section(self, report_data: ReportData, 
                                 metrics: Dict[str, Dict[str, float]]) -> str:
        """生成交易分析部分"""
        trading_metrics = metrics.get('trading_metrics', {})
        
        # 生成交易分析图表
        trading_chart = self.chart_generator.generate_trading_analysis_chart(trading_metrics)
        
        trading_html = f"""
        <section>
            <h2>交易分析</h2>
            <div class="chart-container">
                {trading_chart}
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(report_data.trades)}</div>
                    <div class="metric-label">总交易次数</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{trading_metrics.get('annual_turnover', 0):.2f}</div>
                    <div class="metric-label">年化换手率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{trading_metrics.get('commission_rate', 0):.4%}</div>
                    <div class="metric-label">平均佣金率</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{trading_metrics.get('cost_ratio_to_portfolio', 0):.4%}</div>
                    <div class="metric-label">成本占比</div>
                </div>
            </div>
        </section>
        """
        
        return trading_html
    
    def _generate_positions_section(self, report_data: ReportData) -> str:
        """生成持仓分析部分"""
        if not report_data.positions:
            return "<section><h2>持仓分析</h2><p>暂无持仓数据</p></section>"
        
        # 生成持仓分析图表
        position_chart = self.chart_generator.generate_position_analysis_chart(
            report_data.positions
        )
        
        # 生成持仓明细表
        positions_table = "<table style='width: 100%; border-collapse: collapse; margin-top: 20px;'>"
        positions_table += "<tr style='background-color: #f8f9fa;'><th style='padding: 10px; border: 1px solid #dee2e6;'>股票代码</th><th style='padding: 10px; border: 1px solid #dee2e6;'>持仓数量</th><th style='padding: 10px; border: 1px solid #dee2e6;'>市值</th><th style='padding: 10px; border: 1px solid #dee2e6;'>权重</th></tr>"
        
        for symbol, pos_data in report_data.positions.items():
            positions_table += f"""
            <tr>
                <td style='padding: 10px; border: 1px solid #dee2e6;'>{symbol}</td>
                <td style='padding: 10px; border: 1px solid #dee2e6;'>{pos_data.get('quantity', 0):,}</td>
                <td style='padding: 10px; border: 1px solid #dee2e6;'>￥{pos_data.get('market_value', 0):,.2f}</td>
                <td style='padding: 10px; border: 1px solid #dee2e6;'>{pos_data.get('weight', 0):.2%}</td>
            </tr>
            """
        
        positions_table += "</table>"
        
        positions_html = f"""
        <section>
            <h2>持仓分析</h2>
            <div class="chart-container">
                {position_chart}
            </div>
            <h3>持仓明细</h3>
            {positions_table}
        </section>
        """
        
        return positions_html
    
    def _generate_benchmark_section(self, report_data: ReportData) -> str:
        """生成基准比较部分"""
        if report_data.benchmark_returns.empty:
            return ""
        
        # 计算基准组合价值
        initial_value = report_data.portfolio_values.iloc[0]
        benchmark_values = initial_value * (1 + report_data.benchmark_returns.cumsum())
        
        # 生成对比图表
        comparison_chart = self.chart_generator.generate_returns_chart(
            report_data.portfolio_values,
            benchmark_values
        )
        
        # 计算超额收益
        excess_returns = report_data.returns - report_data.benchmark_returns
        excess_return_total = excess_returns.sum()
        
        benchmark_html = f"""
        <section>
            <h2>基准比较</h2>
            <div class="chart-container">
                {comparison_chart}
            </div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{excess_return_total:.2%}</div>
                    <div class="metric-label">累计超额收益</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{excess_returns.std() * np.sqrt(252):.2%}</div>
                    <div class="metric-label">跟踪误差</div>
                </div>
            </div>
        </section>
        """
        
        return benchmark_html


class ReportGenerator:
    """报告生成器主类"""
    
    def __init__(self):
        """初始化报告生成器"""
        self.html_generator = HTMLReportGenerator()
    
    def generate_comprehensive_report(self, report_data: ReportData, 
                                    output_path: str,
                                    include_benchmark: bool = True) -> None:
        """生成综合报告"""
        self.html_generator.generate_report(
            report_data, 
            output_path,
            include_benchmark=include_benchmark
        )
    
    def generate_batch_reports(self, report_data_list: List[Tuple[str, ReportData]], 
                             output_dir: str) -> None:
        """批量生成报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for report_name, report_data in report_data_list:
            output_file = output_path / f"{report_name}.html"
            self.generate_comprehensive_report(report_data, str(output_file))
    
    def generate_comparison_report(self, report_data_dict: Dict[str, ReportData], 
                                 output_path: str) -> None:
        """生成对比报告"""
        # 创建对比数据
        comparison_data = {}
        for name, data in report_data_dict.items():
            metrics = data.calculate_metrics()
            comparison_data[name] = {
                'data': data,
                'metrics': metrics
            }
        
        # 生成对比HTML
        html_content = self._generate_comparison_html(comparison_data)
        
        # 写入文件
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_comparison_html(self, comparison_data: Dict[str, Dict]) -> str:
        """生成对比HTML"""
        # 创建对比表格
        strategies = list(comparison_data.keys())
        
        # 收集指标
        metrics_to_compare = [
            ('总收益率', 'return_metrics', 'total_return', '{:.2%}'),
            ('年化收益率', 'return_metrics', 'annualized_return', '{:.2%}'),
            ('年化波动率', 'risk_metrics', 'volatility', '{:.2%}'),
            ('最大回撤', 'risk_metrics', 'max_drawdown', '{:.2%}'),
            ('夏普比率', 'risk_adjusted_metrics', 'sharpe_ratio', '{:.3f}'),
            ('索提诺比率', 'risk_adjusted_metrics', 'sortino_ratio', '{:.3f}'),
        ]
        
        # 生成对比表格
        table_html = "<table style='width: 100%; border-collapse: collapse; margin: 20px 0;'>"
        table_html += "<tr style='background-color: #f8f9fa;'><th style='padding: 10px; border: 1px solid #dee2e6;'>指标</th>"
        
        for strategy in strategies:
            table_html += f"<th style='padding: 10px; border: 1px solid #dee2e6;'>{strategy}</th>"
        table_html += "</tr>"
        
        for metric_name, category, key, format_str in metrics_to_compare:
            table_html += f"<tr><td style='padding: 10px; border: 1px solid #dee2e6;'>{metric_name}</td>"
            
            for strategy in strategies:
                value = comparison_data[strategy]['metrics'].get(category, {}).get(key, 0)
                formatted_value = format_str.format(value)
                table_html += f"<td style='padding: 10px; border: 1px solid #dee2e6;'>{formatted_value}</td>"
            
            table_html += "</tr>"
        
        table_html += "</table>"
        
        # 生成完整HTML
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>策略对比分析报告</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #34495e;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                table {{
                    font-size: 14px;
                }}
                th {{
                    background-color: #3498db !important;
                    color: white !important;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: #7f8c8d;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>策略对比分析报告</h1>
                
                <section>
                    <h2>对比分析</h2>
                    {table_html}
                </section>
                
                <div class="footer">
                    <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>由智能量化交易系统自动生成</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_content