"""
交易系统监控模块
实现TradingSystemMonitor类和指标收集，定义性能、风险和系统监控指标，实现指标采集、导出和Grafana仪表板配置
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import time
import threading
import psutil
import requests
from threading import Lock
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from collections import deque
import json
import re

from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, start_http_server, generate_latest

from ..backtest.multi_frequency_backtest import Trade, OrderType


class MetricsCollector:
    """指标收集器类"""
    
    def __init__(self):
        """初始化指标收集器"""
        self.metrics_registry = {}
        self.performance_metrics = {}
        self.risk_metrics = {}
        self.system_metrics = {}
        self.trading_metrics = {}
        self._lock = Lock()
    
    def collect_performance_metrics(self, portfolio_value: float, daily_return: float,
                                  total_return: float, sharpe_ratio: float) -> None:
        """收集性能指标"""
        if portfolio_value < 0:
            raise ValueError("投资组合价值不能为负数")
        
        with self._lock:
            self.performance_metrics = {
                'portfolio_value': portfolio_value,
                'daily_return': daily_return,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'timestamp': datetime.now()
            }
    
    def collect_risk_metrics(self, volatility: float, max_drawdown: float,
                           var_95: float, beta: float) -> None:
        """收集风险指标"""
        with self._lock:
            self.risk_metrics = {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'beta': beta,
                'timestamp': datetime.now()
            }
    
    def collect_system_metrics(self, cpu_usage: float, memory_usage: float,
                             disk_usage: float, model_inference_time: float) -> None:
        """收集系统指标"""
        with self._lock:
            self.system_metrics = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'model_inference_time': model_inference_time,
                'timestamp': datetime.now()
            }
    
    def collect_trading_metrics(self, total_trades: int, successful_trades: int,
                              win_rate: float, average_trade_size: float,
                              turnover_rate: float) -> None:
        """收集交易指标"""
        with self._lock:
            self.trading_metrics = {
                'total_trades': total_trades,
                'successful_trades': successful_trades,
                'win_rate': win_rate,
                'average_trade_size': average_trade_size,
                'turnover_rate': turnover_rate,
                'timestamp': datetime.now()
            }
    
    def register_metric(self, name: str, description: str, metric_type: str) -> None:
        """注册自定义指标"""
        if not name:
            raise ValueError("指标名称不能为空")
        
        if metric_type not in ['gauge', 'counter', 'histogram']:
            raise ValueError(f"不支持的指标类型: {metric_type}")
        
        with self._lock:
            self.metrics_registry[name] = {
                'description': description,
                'type': metric_type,
                'value': 0.0,
                'timestamp': datetime.now()
            }
    
    def update_metric(self, name: str, value: float) -> None:
        """更新指标值"""
        if name not in self.metrics_registry:
            raise ValueError(f"指标 {name} 不存在")
        
        with self._lock:
            self.metrics_registry[name]['value'] = value
            self.metrics_registry[name]['timestamp'] = datetime.now()
    
    def reset_metrics(self) -> None:
        """重置所有指标"""
        with self._lock:
            self.performance_metrics.clear()
            self.risk_metrics.clear()
            self.system_metrics.clear()
            self.trading_metrics.clear()
    
    def export_metrics(self) -> Dict[str, Dict[str, Any]]:
        """导出所有指标"""
        with self._lock:
            return {
                'performance_metrics': self.performance_metrics.copy(),
                'risk_metrics': self.risk_metrics.copy(),
                'system_metrics': self.system_metrics.copy(),
                'trading_metrics': self.trading_metrics.copy(),
                'custom_metrics': self.metrics_registry.copy()
            }


class PrometheusExporter:
    """Prometheus导出器类"""
    
    def __init__(self, metrics_collector: MetricsCollector, port: int = 8000):
        """初始化Prometheus导出器"""
        if not (1024 <= port <= 65535):
            raise ValueError("端口号必须在1024-65535范围内")
        
        self.metrics_collector = metrics_collector
        self.port = port
        self.registry = CollectorRegistry()
        self.is_running = False
        self._server_thread = None
        self._prometheus_metrics = {}
        self._lock = Lock()
    
    def register_prometheus_metrics(self) -> None:
        """注册Prometheus指标"""
        with self._lock:
            # 性能指标
            self._prometheus_metrics['portfolio_value'] = Gauge(
                'portfolio_value', 'Portfolio total value', registry=self.registry
            )
            self._prometheus_metrics['daily_return'] = Gauge(
                'daily_return', 'Daily return rate', registry=self.registry
            )
            self._prometheus_metrics['total_return'] = Gauge(
                'total_return', 'Total return rate', registry=self.registry
            )
            self._prometheus_metrics['sharpe_ratio'] = Gauge(
                'sharpe_ratio', 'Sharpe ratio', registry=self.registry
            )
            
            # 风险指标
            self._prometheus_metrics['volatility'] = Gauge(
                'volatility', 'Portfolio volatility', registry=self.registry
            )
            self._prometheus_metrics['max_drawdown'] = Gauge(
                'max_drawdown', 'Maximum drawdown', registry=self.registry
            )
            self._prometheus_metrics['var_95'] = Gauge(
                'var_95', 'Value at Risk 95%', registry=self.registry
            )
            self._prometheus_metrics['beta'] = Gauge(
                'beta', 'Portfolio beta', registry=self.registry
            )
            
            # 系统指标
            self._prometheus_metrics['cpu_usage'] = Gauge(
                'cpu_usage', 'CPU usage percentage', registry=self.registry
            )
            self._prometheus_metrics['memory_usage'] = Gauge(
                'memory_usage', 'Memory usage percentage', registry=self.registry
            )
            self._prometheus_metrics['disk_usage'] = Gauge(
                'disk_usage', 'Disk usage percentage', registry=self.registry
            )
            self._prometheus_metrics['model_inference_time'] = Gauge(
                'model_inference_time', 'Model inference time in seconds', registry=self.registry
            )
            
            # 交易指标
            self._prometheus_metrics['total_trades'] = Counter(
                'total_trades', 'Total number of trades', registry=self.registry
            )
            self._prometheus_metrics['successful_trades'] = Counter(
                'successful_trades', 'Number of successful trades', registry=self.registry
            )
            self._prometheus_metrics['win_rate'] = Gauge(
                'win_rate', 'Trading win rate', registry=self.registry
            )
            self._prometheus_metrics['average_trade_size'] = Gauge(
                'average_trade_size', 'Average trade size', registry=self.registry
            )
            self._prometheus_metrics['turnover_rate'] = Gauge(
                'turnover_rate', 'Portfolio turnover rate', registry=self.registry
            )
            
            # 注册自定义指标
            for name, metric_info in self.metrics_collector.metrics_registry.items():
                if metric_info['type'] == 'gauge':
                    self._prometheus_metrics[name] = Gauge(
                        name, metric_info['description'], registry=self.registry
                    )
                elif metric_info['type'] == 'counter':
                    self._prometheus_metrics[name] = Counter(
                        name, metric_info['description'], registry=self.registry
                    )
    
    def get_registered_metrics(self) -> Dict[str, Any]:
        """获取已注册的指标"""
        with self._lock:
            return list(self._prometheus_metrics.keys())
    
    def generate_metrics_output(self) -> str:
        """生成指标输出"""
        # 更新Prometheus指标值
        self._update_prometheus_metrics()
        
        # 生成Prometheus格式的输出
        return generate_latest(self.registry).decode('utf-8')
    
    def _update_prometheus_metrics(self) -> None:
        """更新Prometheus指标值"""
        metrics_data = self.metrics_collector.export_metrics()
        
        with self._lock:
            # 更新性能指标
            perf_metrics = metrics_data.get('performance_metrics', {})
            for key, value in perf_metrics.items():
                if key != 'timestamp' and key in self._prometheus_metrics:
                    self._prometheus_metrics[key].set(value)
            
            # 更新风险指标
            risk_metrics = metrics_data.get('risk_metrics', {})
            for key, value in risk_metrics.items():
                if key != 'timestamp' and key in self._prometheus_metrics:
                    self._prometheus_metrics[key].set(value)
            
            # 更新系统指标
            sys_metrics = metrics_data.get('system_metrics', {})
            for key, value in sys_metrics.items():
                if key != 'timestamp' and key in self._prometheus_metrics:
                    self._prometheus_metrics[key].set(value)
            
            # 更新交易指标
            trade_metrics = metrics_data.get('trading_metrics', {})
            for key, value in trade_metrics.items():
                if key != 'timestamp' and key in self._prometheus_metrics:
                    if key in ['total_trades', 'successful_trades']:
                        # Counter类型需要特殊处理
                        self._prometheus_metrics[key]._value._value = value
                    else:
                        self._prometheus_metrics[key].set(value)
            
            # 更新自定义指标
            custom_metrics = metrics_data.get('custom_metrics', {})
            for name, metric_info in custom_metrics.items():
                if name in self._prometheus_metrics:
                    if metric_info['type'] == 'counter':
                        self._prometheus_metrics[name]._value._value = metric_info['value']
                    else:
                        self._prometheus_metrics[name].set(metric_info['value'])
    
    def start(self) -> None:
        """启动Prometheus导出器"""
        if self.is_running:
            return
        
        self.register_prometheus_metrics()
        
        def run_server():
            start_http_server(self.port, registry=self.registry)
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        self.is_running = True
    
    def stop(self) -> None:
        """停止Prometheus导出器"""
        self.is_running = False
        # Note: prometheus_client doesn't provide a direct way to stop the server
        # In practice, this would require more complex server management
    
    def health_check(self) -> bool:
        """健康检查"""
        if not self.is_running:
            return False
        
        # 简单的健康检查：尝试访问metrics端点
        try:
            import urllib.request
            req = urllib.request.Request(f"http://localhost:{self.port}/metrics")
            with urllib.request.urlopen(req, timeout=1) as response:
                return response.getcode() == 200
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            logger.debug(f"健康检查失败: {e}")
            return False


class GrafanaDashboardManager:
    """Grafana仪表板管理器类"""
    
    def __init__(self, grafana_url: str, api_key: str):
        """初始化Grafana仪表板管理器"""
        if not grafana_url:
            raise ValueError("Grafana URL不能为空")
        
        # 简单的URL格式验证
        if not re.match(r'^https?://.+', grafana_url):
            raise ValueError("无效的Grafana URL格式")
        
        self.grafana_url = grafana_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def generate_dashboard_config(self) -> Dict[str, Any]:
        """生成仪表板配置"""
        return {
            "dashboard": {
                "id": None,
                "title": "交易系统监控",
                "tags": ["trading", "monitoring"],
                "timezone": "browser",
                "panels": [
                    self.create_panel("投资组合价值", "portfolio_value", "graph", 0, 0, 12, 8),
                    self.create_panel("日收益率", "daily_return", "graph", 0, 8, 12, 8),
                    self.create_panel("风险指标", "max_drawdown", "singlestat", 12, 0, 6, 8),
                    self.create_panel("系统性能", "cpu_usage", "graph", 12, 8, 6, 8),
                    self.create_panel("交易统计", "total_trades", "singlestat", 18, 0, 6, 8),
                    self.create_panel("夏普比率", "sharpe_ratio", "singlestat", 18, 8, 6, 8),
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "5s"
            },
            "overwrite": True
        }
    
    def create_panel(self, title: str, metric_query: str, panel_type: str,
                    x_pos: int, y_pos: int, width: int, height: int) -> Dict[str, Any]:
        """创建仪表板面板"""
        panel_config = {
            "id": None,
            "title": title,
            "type": panel_type,
            "gridPos": {
                "h": height,
                "w": width,
                "x": x_pos,
                "y": y_pos
            },
            "targets": [
                {
                    "expr": metric_query,
                    "format": "time_series",
                    "legendFormat": title,
                    "refId": "A"
                }
            ],
            "datasource": "prometheus"
        }
        
        if panel_type == "graph":
            panel_config.update({
                "xAxis": {"show": True},
                "yAxes": [
                    {"show": True, "label": title},
                    {"show": True}
                ],
                "lines": True,
                "fill": 1,
                "linewidth": 2,
                "points": False,
                "pointradius": 2
            })
        elif panel_type == "singlestat":
            panel_config.update({
                "valueName": "current",
                "format": "short",
                "prefix": "",
                "postfix": "",
                "nullText": None,
                "valueMaps": [],
                "mappingTypes": [],
                "rangeMaps": [],
                "colorBackground": False,
                "colorValue": False,
                "colors": ["#299c46", "rgba(237, 129, 40, 0.89)", "#d44a3a"],
                "sparkline": {
                    "show": False,
                    "full": False,
                    "lineColor": "rgb(31, 120, 193)",
                    "fillColor": "rgba(31, 118, 189, 0.18)"
                },
                "gauge": {
                    "show": False,
                    "minValue": 0,
                    "maxValue": 100,
                    "thresholdMarkers": True,
                    "thresholdLabels": False
                }
            })
        
        return panel_config
    
    def deploy_dashboard(self) -> Dict[str, Any]:
        """部署仪表板"""
        dashboard_config = self.generate_dashboard_config()
        
        response = self.session.post(
            f"{self.grafana_url}/api/dashboards/db",
            json=dashboard_config
        )
        
        if response.status_code == 401:
            raise Exception("Grafana API认证失败")
        
        return response.json()
    
    def update_dashboard(self, dashboard_uid: str) -> Dict[str, Any]:
        """更新仪表板"""
        dashboard_config = self.generate_dashboard_config()
        dashboard_config['dashboard']['uid'] = dashboard_uid
        
        response = self.session.post(
            f"{self.grafana_url}/api/dashboards/db",
            json=dashboard_config
        )
        
        return response.json()
    
    def delete_dashboard(self, dashboard_uid: str) -> Dict[str, Any]:
        """删除仪表板"""
        response = self.session.delete(
            f"{self.grafana_url}/api/dashboards/uid/{dashboard_uid}"
        )
        
        return response.json()
    
    def create_alert_rule(self, rule_name: str, metric_query: str, threshold: float,
                         condition: str, evaluation_interval: str) -> Dict[str, Any]:
        """创建告警规则"""
        return {
            "name": rule_name,
            "condition": {
                "query": metric_query,
                "threshold": threshold,
                "type": condition
            },
            "frequency": evaluation_interval,
            "handler": 1,
            "severity": "critical",
            "state": "ok",
            "executionErrorState": "alerting",
            "noDataState": "no_data",
            "for": "5m"
        }
    
    def configure_prometheus_datasource(self, prometheus_url: str,
                                      datasource_name: str) -> Dict[str, Any]:
        """配置Prometheus数据源"""
        return {
            "name": datasource_name,
            "type": "prometheus",
            "url": prometheus_url,
            "access": "proxy",
            "basicAuth": False,
            "isDefault": True,
            "jsonData": {
                "httpMethod": "POST",
                "prometheusType": "Prometheus",
                "prometheusVersion": "2.x"
            }
        }


class TradingSystemMonitor:
    """交易系统监控器主类"""
    
    def __init__(self, prometheus_port: int = 8000, grafana_url: str = None,
                 grafana_api_key: str = None):
        """初始化交易系统监控器"""
        self.metrics_collector = MetricsCollector()
        self.prometheus_exporter = PrometheusExporter(self.metrics_collector, prometheus_port)
        
        if grafana_url and grafana_api_key:
            self.dashboard_manager = GrafanaDashboardManager(grafana_url, grafana_api_key)
        else:
            self.dashboard_manager = None
        
        self.is_monitoring = False
        self._metrics_history = deque(maxlen=1000)  # 保留最近1000个数据点
        self._history_lock = Lock()
    
    def start_monitoring(self) -> None:
        """启动监控"""
        if self.is_monitoring:
            return
        
        self.prometheus_exporter.start()
        self.is_monitoring = True
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        if not self.is_monitoring:
            return
        
        self.prometheus_exporter.stop()
        self.is_monitoring = False
    
    def update_portfolio_metrics(self, portfolio_data: Dict[str, float]) -> None:
        """更新投资组合指标"""
        self.metrics_collector.collect_performance_metrics(
            portfolio_value=portfolio_data['value'],
            daily_return=portfolio_data['daily_return'],
            total_return=portfolio_data['total_return'],
            sharpe_ratio=portfolio_data.get('sharpe_ratio', 0.0)
        )
        
        # 如果包含风险指标，也更新风险指标
        if 'max_drawdown' in portfolio_data:
            self.metrics_collector.collect_risk_metrics(
                volatility=portfolio_data.get('volatility', 0.0),
                max_drawdown=portfolio_data['max_drawdown'],
                var_95=portfolio_data.get('var_95', 0.0),
                beta=portfolio_data.get('beta', 1.0)
            )
        
        # 保存历史记录
        self._save_metrics_history()
    
    def update_trading_metrics(self, trades: List[Trade]) -> None:
        """更新交易指标"""
        if not trades:
            return
        
        buy_trades = [t for t in trades if t.trade_type == OrderType.BUY]
        sell_trades = [t for t in trades if t.trade_type == OrderType.SELL]
        
        # 计算基本交易统计
        total_trades = len(trades)
        total_volume = sum(float(t.quantity * t.price) for t in trades)
        average_trade_size = total_volume / total_trades if total_trades > 0 else 0.0
        
        self.metrics_collector.collect_trading_metrics(
            total_trades=total_trades,
            successful_trades=len(sell_trades),  # 简化：认为卖出交易是成功的
            win_rate=0.7,  # 这里需要实际计算胜率
            average_trade_size=average_trade_size,
            turnover_rate=2.0  # 这里需要实际计算换手率
        )
        
        # 更新交易统计到trading_metrics
        self.metrics_collector.trading_metrics.update({
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades)
        })
    
    def update_system_metrics(self) -> None:
        """更新系统指标"""
        # 获取系统资源使用情况
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        self.metrics_collector.collect_system_metrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_info.percent,
            disk_usage=disk_info.percent,
            model_inference_time=0.1  # 这里需要实际测量模型推理时间
        )
    
    def get_latest_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取最新指标"""
        return self.metrics_collector.export_metrics()
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取指标历史"""
        with self._history_lock:
            history_list = list(self._metrics_history)
            return history_list[-limit:] if limit < len(history_list) else history_list
    
    def setup_dashboard(self) -> Dict[str, Any]:
        """设置监控仪表板"""
        if not self.dashboard_manager:
            raise ValueError("未配置Grafana仪表板管理器")
        
        return self.dashboard_manager.deploy_dashboard()
    
    def _save_metrics_history(self) -> None:
        """保存指标历史"""
        current_metrics = self.metrics_collector.export_metrics()
        with self._history_lock:
            self._metrics_history.append(current_metrics)