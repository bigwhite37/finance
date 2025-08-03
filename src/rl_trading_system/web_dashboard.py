"""
实时回撤监控Web仪表板

提供实时回撤监控的Web界面，包括图表组件、告警通知和交互功能。
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np

# Optional dependencies
try:
    import plotly.graph_objs as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

from .monitoring.drawdown_monitoring_service import DrawdownMonitoringService


class DrawdownDashboard:
    """回撤监控仪表板"""
    
    def __init__(self, monitoring_service: DrawdownMonitoringService, 
                 template_folder: str = None, static_folder: str = None):
        self.monitoring_service = monitoring_service
        self.logger = logging.getLogger(__name__)
        
        # 设置模板和静态文件路径
        if template_folder is None:
            template_folder = os.path.join(os.path.dirname(__file__), 'templates')
        if static_folder is None:
            static_folder = os.path.join(os.path.dirname(__file__), 'static')
        
        # 创建Flask应用
        self.app = Flask(__name__, 
                        template_folder=template_folder,
                        static_folder=static_folder)
        
        # 设置路由
        self._setup_routes()
        
        self.logger.info("回撤监控仪表板初始化完成")
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def dashboard():
            """主仪表板页面"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/dashboard/overview')
        def get_dashboard_overview():
            """获取仪表板概览数据"""
            try:
                current_metrics = self.monitoring_service.last_metrics
                if not current_metrics:
                    return jsonify({'error': '暂无数据'}), 404
                
                # 获取历史数据用于趋势分析
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=24)
                historical_data = self.monitoring_service.historical_data_manager.get_historical_metrics(
                    start_time, end_time
                )
                
                # 计算趋势
                drawdown_trend = self._calculate_trend([d['current_drawdown'] for d in historical_data[-10:]])
                volatility_trend = self._calculate_trend([d['volatility'] for d in historical_data[-10:]])
                
                overview = {
                    'current_metrics': current_metrics.to_dict(),
                    'trends': {
                        'drawdown_trend': drawdown_trend,
                        'volatility_trend': volatility_trend
                    },
                    'summary_stats': self._calculate_summary_stats(historical_data),
                    'last_updated': current_metrics.timestamp.isoformat()
                }
                
                return jsonify(overview)
                
            except Exception as e:
                self.logger.error(f"获取仪表板概览失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/charts/drawdown_curve')
        def get_drawdown_curve_chart():
            """获取回撤曲线图表"""
            try:
                hours = request.args.get('hours', 24, type=int)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours)
                
                historical_data = self.monitoring_service.historical_data_manager.get_historical_metrics(
                    start_time, end_time
                )
                
                if not historical_data:
                    return jsonify({'error': '暂无历史数据'}), 404
                
                # 创建回撤曲线图
                chart_data = self._create_drawdown_curve_chart(historical_data)
                return jsonify(chart_data)
                
            except Exception as e:
                self.logger.error(f"获取回撤曲线图失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/charts/portfolio_value')
        def get_portfolio_value_chart():
            """获取投资组合价值图表"""
            try:
                hours = request.args.get('hours', 24, type=int)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours)
                
                historical_data = self.monitoring_service.historical_data_manager.get_historical_metrics(
                    start_time, end_time
                )
                
                if not historical_data:
                    return jsonify({'error': '暂无历史数据'}), 404
                
                # 创建投资组合价值图
                chart_data = self._create_portfolio_value_chart(historical_data)
                return jsonify(chart_data)
                
            except Exception as e:
                self.logger.error(f"获取投资组合价值图失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/charts/risk_metrics')
        def get_risk_metrics_chart():
            """获取风险指标图表"""
            try:
                hours = request.args.get('hours', 24, type=int)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours)
                
                historical_data = self.monitoring_service.historical_data_manager.get_historical_metrics(
                    start_time, end_time
                )
                
                if not historical_data:
                    return jsonify({'error': '暂无历史数据'}), 404
                
                # 创建风险指标图
                chart_data = self._create_risk_metrics_chart(historical_data)
                return jsonify(chart_data)
                
            except Exception as e:
                self.logger.error(f"获取风险指标图失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/alerts')
        def get_alerts():
            """获取告警信息"""
            try:
                active_alerts = self.monitoring_service.historical_data_manager.get_active_alerts()
                
                # 按严重程度排序
                severity_order = {'CRITICAL': 0, 'WARNING': 1, 'INFO': 2}
                active_alerts.sort(key=lambda x: severity_order.get(x['severity'], 3))
                
                return jsonify({
                    'active_alerts': active_alerts,
                    'alert_count': len(active_alerts),
                    'critical_count': len([a for a in active_alerts if a['severity'] == 'CRITICAL']),
                    'warning_count': len([a for a in active_alerts if a['severity'] == 'WARNING'])
                })
                
            except Exception as e:
                self.logger.error(f"获取告警信息失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/performance_stats')
        def get_performance_stats():
            """获取性能统计"""
            try:
                hours = request.args.get('hours', 24, type=int)
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours)
                
                historical_data = self.monitoring_service.historical_data_manager.get_historical_metrics(
                    start_time, end_time
                )
                
                if not historical_data:
                    return jsonify({'error': '暂无历史数据'}), 404
                
                stats = self._calculate_performance_stats(historical_data)
                return jsonify(stats)
                
            except Exception as e:
                self.logger.error(f"获取性能统计失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            """静态文件服务"""
            return send_from_directory(self.app.static_folder, filename)
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return 'stable'
        
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        earlier_avg = np.mean(values[:3]) if len(values) >= 3 else values[0]
        
        change_pct = (recent_avg - earlier_avg) / abs(earlier_avg) if earlier_avg != 0 else 0
        
        if change_pct > 0.05:
            return 'up'
        elif change_pct < -0.05:
            return 'down'
        else:
            return 'stable'
    
    def _calculate_summary_stats(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """计算汇总统计"""
        if not historical_data:
            return {}
        
        drawdowns = [d['current_drawdown'] for d in historical_data]
        volatilities = [d['volatility'] for d in historical_data]
        portfolio_values = [d['portfolio_value'] for d in historical_data]
        
        return {
            'avg_drawdown': np.mean(drawdowns),
            'max_drawdown': min(drawdowns),  # 最大回撤是最小值
            'avg_volatility': np.mean(volatilities),
            'portfolio_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] if len(portfolio_values) > 1 else 0,
            'data_points': len(historical_data)
        }
    
    def _create_drawdown_curve_chart(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """创建回撤曲线图表"""
        if not PLOTLY_AVAILABLE:
            # 返回简化的数据格式
            timestamps = [d['timestamp'] for d in historical_data]
            drawdowns = [d['current_drawdown'] * 100 for d in historical_data]
            return {
                'data': [{'x': timestamps, 'y': drawdowns, 'type': 'scatter', 'name': '当前回撤'}],
                'layout': {'title': '回撤曲线', 'xaxis': {'title': '时间'}, 'yaxis': {'title': '回撤 (%)'}}
            }
        
        timestamps = [datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')) for d in historical_data]
        drawdowns = [d['current_drawdown'] * 100 for d in historical_data]  # 转换为百分比
        
        # 创建Plotly图表
        fig = go.Figure()
        
        # 添加回撤曲线
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=drawdowns,
            mode='lines',
            name='当前回撤',
            line=dict(color='red', width=2),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
        
        # 添加零线
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="零线")
        
        # 添加警告线
        fig.add_hline(y=-8, line_dash="dot", line_color="orange", annotation_text="警告线(-8%)")
        fig.add_hline(y=-15, line_dash="dot", line_color="red", annotation_text="危险线(-15%)")
        
        fig.update_layout(
            title='回撤曲线',
            xaxis_title='时间',
            yaxis_title='回撤 (%)',
            hovermode='x unified',
            showlegend=True
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def _create_portfolio_value_chart(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """创建投资组合价值图表"""
        if not PLOTLY_AVAILABLE:
            # 返回简化的数据格式
            timestamps = [d['timestamp'] for d in historical_data]
            portfolio_values = [d['portfolio_value'] for d in historical_data]
            return {
                'data': [
                    {'x': timestamps, 'y': portfolio_values, 'type': 'scatter', 'name': '投资组合价值'},
                ],
                'layout': {'title': '投资组合价值变化', 'xaxis': {'title': '时间'}, 'yaxis': {'title': '价值'}}
            }
        
        timestamps = [datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')) for d in historical_data]
        portfolio_values = [d['portfolio_value'] for d in historical_data]
        
        # 计算峰值线
        running_max = []
        current_max = portfolio_values[0] if portfolio_values else 0
        for value in portfolio_values:
            current_max = max(current_max, value)
            running_max.append(current_max)
        
        fig = go.Figure()
        
        # 添加投资组合价值
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=portfolio_values,
            mode='lines',
            name='投资组合价值',
            line=dict(color='blue', width=2)
        ))
        
        # 添加峰值线
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=running_max,
            mode='lines',
            name='历史峰值',
            line=dict(color='green', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='投资组合价值变化',
            xaxis_title='时间',
            yaxis_title='价值',
            hovermode='x unified',
            showlegend=True
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def _create_risk_metrics_chart(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """创建风险指标图表"""
        timestamps = [datetime.fromisoformat(d['timestamp'].replace('Z', '+00:00')) for d in historical_data]
        volatilities = [d['volatility'] * 100 for d in historical_data]  # 转换为百分比
        risk_budget_usage = [d['risk_budget_usage'] * 100 for d in historical_data]  # 转换为百分比
        concentration_scores = [d['concentration_score'] * 100 for d in historical_data]  # 转换为百分比
        
        fig = go.Figure()
        
        # 添加波动率
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=volatilities,
            mode='lines',
            name='波动率 (%)',
            line=dict(color='orange', width=2),
            yaxis='y'
        ))
        
        # 添加风险预算使用率
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=risk_budget_usage,
            mode='lines',
            name='风险预算使用率 (%)',
            line=dict(color='purple', width=2),
            yaxis='y2'
        ))
        
        # 添加集中度分数
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=concentration_scores,
            mode='lines',
            name='集中度分数 (%)',
            line=dict(color='brown', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='风险指标变化',
            xaxis_title='时间',
            yaxis=dict(title='波动率 (%)', side='left'),
            yaxis2=dict(title='使用率/集中度 (%)', side='right', overlaying='y'),
            hovermode='x unified',
            showlegend=True
        )
        
        return json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
    
    def _calculate_performance_stats(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """计算性能统计"""
        if not historical_data:
            return {}
        
        # 提取数据
        portfolio_values = [d['portfolio_value'] for d in historical_data]
        drawdowns = [d['current_drawdown'] for d in historical_data]
        volatilities = [d['volatility'] for d in historical_data]
        sharpe_ratios = [d['sharpe_ratio'] for d in historical_data if d['sharpe_ratio'] is not None]
        
        # 计算收益率
        if len(portfolio_values) > 1:
            returns = [(portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] 
                      for i in range(1, len(portfolio_values))]
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        else:
            returns = []
            total_return = 0
        
        # 计算统计指标
        stats = {
            'total_return': total_return * 100,  # 转换为百分比
            'annualized_return': total_return * 252 * 100 if returns else 0,  # 年化收益率
            'max_drawdown': min(drawdowns) * 100 if drawdowns else 0,
            'current_drawdown': drawdowns[-1] * 100 if drawdowns else 0,
            'avg_volatility': np.mean(volatilities) * 100 if volatilities else 0,
            'current_volatility': volatilities[-1] * 100 if volatilities else 0,
            'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'current_sharpe_ratio': sharpe_ratios[-1] if sharpe_ratios else 0,
            'win_rate': len([r for r in returns if r > 0]) / len(returns) * 100 if returns else 0,
            'avg_win': np.mean([r for r in returns if r > 0]) * 100 if returns else 0,
            'avg_loss': np.mean([r for r in returns if r < 0]) * 100 if returns else 0,
            'profit_factor': abs(sum([r for r in returns if r > 0]) / sum([r for r in returns if r < 0])) if returns and any(r < 0 for r in returns) else 0
        }
        
        return stats
    
    def run(self, host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
        """运行仪表板服务器"""
        self.logger.info(f"启动回撤监控仪表板: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


def create_dashboard_templates():
    """创建仪表板模板文件"""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    # 创建目录
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # 创建主模板
    dashboard_html = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>回撤监控仪表板</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        .trend-up { color: #28a745; }
        .trend-down { color: #dc3545; }
        .trend-stable { color: #6c757d; }
        .alert-critical { border-left: 4px solid #dc3545; }
        .alert-warning { border-left: 4px solid #ffc107; }
        .alert-info { border-left: 4px solid #17a2b8; }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-normal { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-critical { background-color: #dc3545; }
    </style>
</head>
<body class="bg-light">
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">
                <i class="fas fa-chart-line"></i> 回撤监控仪表板
            </span>
            <span class="navbar-text" id="last-updated">
                最后更新: --
            </span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- 概览指标 -->
        <div class="row" id="overview-metrics">
            <!-- 动态生成的指标卡片 -->
        </div>

        <!-- 告警信息 -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-exclamation-triangle"></i> 告警信息</h5>
                    </div>
                    <div class="card-body" id="alerts-container">
                        <!-- 动态生成的告警信息 -->
                    </div>
                </div>
            </div>
        </div>

        <!-- 图表区域 -->
        <div class="row mt-4">
            <div class="col-lg-6">
                <div class="chart-container">
                    <h5>回撤曲线</h5>
                    <div id="drawdown-chart"></div>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="chart-container">
                    <h5>投资组合价值</h5>
                    <div id="portfolio-chart"></div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-lg-8">
                <div class="chart-container">
                    <h5>风险指标</h5>
                    <div id="risk-chart"></div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="chart-container">
                    <h5>性能统计</h5>
                    <div id="performance-stats">
                        <!-- 动态生成的性能统计 -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 全局变量
        let refreshInterval;
        const REFRESH_RATE = 5000; // 5秒刷新一次

        // 初始化仪表板
        document.addEventListener('DOMContentLoaded', function() {
            loadDashboard();
            startAutoRefresh();
        });

        // 加载仪表板数据
        async function loadDashboard() {
            try {
                await Promise.all([
                    loadOverview(),
                    loadAlerts(),
                    loadCharts(),
                    loadPerformanceStats()
                ]);
            } catch (error) {
                console.error('加载仪表板失败:', error);
                showError('加载数据失败，请检查服务器连接');
            }
        }

        // 加载概览数据
        async function loadOverview() {
            const response = await fetch('/api/dashboard/overview');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            updateOverviewMetrics(data);
            updateLastUpdated(data.last_updated);
        }

        // 更新概览指标
        function updateOverviewMetrics(data) {
            const metrics = data.current_metrics;
            const trends = data.trends;
            
            const metricsHtml = `
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <div class="metric-value">${(metrics.current_drawdown * 100).toFixed(2)}%</div>
                                <div class="metric-label">当前回撤</div>
                            </div>
                            <div>
                                <i class="fas fa-arrow-${getTrendIcon(trends.drawdown_trend)} trend-${trends.drawdown_trend}"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <div class="metric-value">${(metrics.max_drawdown * 100).toFixed(2)}%</div>
                                <div class="metric-label">最大回撤</div>
                            </div>
                            <div>
                                <i class="fas fa-chart-line"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <div class="metric-value">${(metrics.volatility * 100).toFixed(2)}%</div>
                                <div class="metric-label">当前波动率</div>
                            </div>
                            <div>
                                <i class="fas fa-arrow-${getTrendIcon(trends.volatility_trend)} trend-${trends.volatility_trend}"></i>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <div class="metric-value">
                                    <span class="status-indicator status-${getAlertStatus(metrics.alert_level)}"></span>
                                    ${metrics.alert_level}
                                </div>
                                <div class="metric-label">系统状态</div>
                            </div>
                            <div>
                                <i class="fas fa-shield-alt"></i>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('overview-metrics').innerHTML = metricsHtml;
        }

        // 加载告警信息
        async function loadAlerts() {
            const response = await fetch('/api/dashboard/alerts');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            updateAlerts(data);
        }

        // 更新告警信息
        function updateAlerts(data) {
            const alertsContainer = document.getElementById('alerts-container');
            
            if (data.active_alerts.length === 0) {
                alertsContainer.innerHTML = '<div class="text-muted">暂无活跃告警</div>';
                return;
            }

            const alertsHtml = data.active_alerts.map(alert => `
                <div class="alert alert-${getSeverityClass(alert.severity)} alert-${alert.severity.toLowerCase()}" role="alert">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong>${alert.alert_type}</strong>: ${alert.message}
                        </div>
                        <small class="text-muted">${formatDateTime(alert.timestamp)}</small>
                    </div>
                </div>
            `).join('');
            
            alertsContainer.innerHTML = alertsHtml;
        }

        // 加载图表
        async function loadCharts() {
            const [drawdownChart, portfolioChart, riskChart] = await Promise.all([
                fetch('/api/dashboard/charts/drawdown_curve').then(r => r.json()),
                fetch('/api/dashboard/charts/portfolio_value').then(r => r.json()),
                fetch('/api/dashboard/charts/risk_metrics').then(r => r.json())
            ]);

            Plotly.newPlot('drawdown-chart', drawdownChart.data, drawdownChart.layout, {responsive: true});
            Plotly.newPlot('portfolio-chart', portfolioChart.data, portfolioChart.layout, {responsive: true});
            Plotly.newPlot('risk-chart', riskChart.data, riskChart.layout, {responsive: true});
        }

        // 加载性能统计
        async function loadPerformanceStats() {
            const response = await fetch('/api/dashboard/performance_stats');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            updatePerformanceStats(data);
        }

        // 更新性能统计
        function updatePerformanceStats(stats) {
            const statsHtml = `
                <div class="row">
                    <div class="col-6">
                        <div class="text-center">
                            <div class="h4 text-primary">${stats.total_return.toFixed(2)}%</div>
                            <div class="small text-muted">总收益率</div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="text-center">
                            <div class="h4 text-info">${stats.current_sharpe_ratio.toFixed(2)}</div>
                            <div class="small text-muted">夏普比率</div>
                        </div>
                    </div>
                </div>
                <hr>
                <div class="row">
                    <div class="col-6">
                        <div class="text-center">
                            <div class="h5 text-success">${stats.win_rate.toFixed(1)}%</div>
                            <div class="small text-muted">胜率</div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="text-center">
                            <div class="h5 text-warning">${stats.profit_factor.toFixed(2)}</div>
                            <div class="small text-muted">盈亏比</div>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('performance-stats').innerHTML = statsHtml;
        }

        // 工具函数
        function getTrendIcon(trend) {
            switch(trend) {
                case 'up': return 'up';
                case 'down': return 'down';
                default: return 'right';
            }
        }

        function getAlertStatus(level) {
            switch(level) {
                case 'CRITICAL': return 'critical';
                case 'WARNING': return 'warning';
                default: return 'normal';
            }
        }

        function getSeverityClass(severity) {
            switch(severity) {
                case 'CRITICAL': return 'danger';
                case 'WARNING': return 'warning';
                default: return 'info';
            }
        }

        function formatDateTime(timestamp) {
            return new Date(timestamp).toLocaleString('zh-CN');
        }

        function updateLastUpdated(timestamp) {
            document.getElementById('last-updated').textContent = 
                `最后更新: ${formatDateTime(timestamp)}`;
        }

        function showError(message) {
            const alertHtml = `
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    <strong>错误:</strong> ${message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            `;
            document.body.insertAdjacentHTML('afterbegin', alertHtml);
        }

        // 自动刷新
        function startAutoRefresh() {
            refreshInterval = setInterval(loadDashboard, REFRESH_RATE);
        }

        function stopAutoRefresh() {
            if (refreshInterval) {
                clearInterval(refreshInterval);
            }
        }

        // 页面可见性变化时控制刷新
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) {
                stopAutoRefresh();
            } else {
                startAutoRefresh();
            }
        });
    </script>
</body>
</html>
    """
    
    with open(os.path.join(template_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(dashboard_html)
    
    print(f"仪表板模板已创建: {template_dir}")


if __name__ == "__main__":
    # 创建模板文件
    create_dashboard_templates()
    
    # 示例用法
    from .monitoring.drawdown_monitoring_service import DrawdownMonitoringService
    
    # 创建监控服务
    monitoring_service = DrawdownMonitoringService()
    monitoring_service.start_monitoring()
    
    # 创建仪表板
    dashboard = DrawdownDashboard(monitoring_service)
    
    try:
        # 运行仪表板
        dashboard.run(port=8080, debug=True)
    except KeyboardInterrupt:
        monitoring_service.stop_monitoring()
        print("仪表板已停止")