"""
交易系统监控模块的单元测试
测试监控指标的定义和收集功能，指标导出和Grafana仪表板集成，监控系统的实时性和准确性
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

from src.rl_trading_system.monitoring.trading_system_monitor import (
    TradingSystemMonitor,
    MetricsCollector,
    PrometheusExporter,
    GrafanaDashboardManager
)
from src.rl_trading_system.backtest.multi_frequency_backtest import Trade, OrderType


class TestMetricsCollector:
    """指标收集器测试类"""

    @pytest.fixture
    def metrics_collector(self):
        """创建指标收集器"""
        return MetricsCollector()

    def test_metrics_collector_initialization(self, metrics_collector):
        """测试指标收集器初始化"""
        assert metrics_collector.metrics_registry is not None
        assert isinstance(metrics_collector.performance_metrics, dict)
        assert isinstance(metrics_collector.risk_metrics, dict)
        assert isinstance(metrics_collector.system_metrics, dict)
        assert isinstance(metrics_collector.trading_metrics, dict)

    def test_performance_metrics_collection(self, metrics_collector):
        """测试性能指标收集"""
        # 收集性能指标
        portfolio_value = 1050000.0
        daily_return = 0.02
        total_return = 0.05
        sharpe_ratio = 1.5

        metrics_collector.collect_performance_metrics(
            portfolio_value=portfolio_value,
            daily_return=daily_return,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio
        )

        # 验证指标被正确收集
        assert metrics_collector.performance_metrics['portfolio_value'] == portfolio_value
        assert metrics_collector.performance_metrics['daily_return'] == daily_return
        assert metrics_collector.performance_metrics['total_return'] == total_return
        assert metrics_collector.performance_metrics['sharpe_ratio'] == sharpe_ratio
        assert 'timestamp' in metrics_collector.performance_metrics

    def test_risk_metrics_collection(self, metrics_collector):
        """测试风险指标收集"""
        # 收集风险指标
        volatility = 0.15
        max_drawdown = 0.08
        var_95 = 0.03
        beta = 1.2

        metrics_collector.collect_risk_metrics(
            volatility=volatility,
            max_drawdown=max_drawdown,
            var_95=var_95,
            beta=beta
        )

        # 验证指标被正确收集
        assert metrics_collector.risk_metrics['volatility'] == volatility
        assert metrics_collector.risk_metrics['max_drawdown'] == max_drawdown
        assert metrics_collector.risk_metrics['var_95'] == var_95
        assert metrics_collector.risk_metrics['beta'] == beta
        assert 'timestamp' in metrics_collector.risk_metrics

    def test_system_metrics_collection(self, metrics_collector):
        """测试系统指标收集"""
        # 收集系统指标
        cpu_usage = 45.5
        memory_usage = 78.2
        disk_usage = 60.0
        model_inference_time = 0.125

        metrics_collector.collect_system_metrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            model_inference_time=model_inference_time
        )

        # 验证指标被正确收集
        assert metrics_collector.system_metrics['cpu_usage'] == cpu_usage
        assert metrics_collector.system_metrics['memory_usage'] == memory_usage
        assert metrics_collector.system_metrics['disk_usage'] == disk_usage
        assert metrics_collector.system_metrics['model_inference_time'] == model_inference_time
        assert 'timestamp' in metrics_collector.system_metrics

    def test_trading_metrics_collection(self, metrics_collector):
        """测试交易指标收集"""
        # 收集交易指标
        total_trades = 150
        successful_trades = 135
        win_rate = 0.72
        average_trade_size = 10000.0
        turnover_rate = 2.5

        metrics_collector.collect_trading_metrics(
            total_trades=total_trades,
            successful_trades=successful_trades,
            win_rate=win_rate,
            average_trade_size=average_trade_size,
            turnover_rate=turnover_rate
        )

        # 验证指标被正确收集
        assert metrics_collector.trading_metrics['total_trades'] == total_trades
        assert metrics_collector.trading_metrics['successful_trades'] == successful_trades
        assert metrics_collector.trading_metrics['win_rate'] == win_rate
        assert metrics_collector.trading_metrics['average_trade_size'] == average_trade_size
        assert metrics_collector.trading_metrics['turnover_rate'] == turnover_rate
        assert 'timestamp' in metrics_collector.trading_metrics

    def test_metrics_registry_management(self, metrics_collector):
        """测试指标注册表管理"""
        # 注册自定义指标
        metric_name = "custom_metric"
        metric_description = "A custom monitoring metric"
        metric_type = "gauge"

        metrics_collector.register_metric(
            name=metric_name,
            description=metric_description,
            metric_type=metric_type
        )

        # 验证指标被注册
        assert metric_name in metrics_collector.metrics_registry
        assert metrics_collector.metrics_registry[metric_name]['description'] == metric_description
        assert metrics_collector.metrics_registry[metric_name]['type'] == metric_type

        # 更新指标值
        metric_value = 42.0
        metrics_collector.update_metric(metric_name, metric_value)

        # 验证指标值被更新
        assert metrics_collector.metrics_registry[metric_name]['value'] == metric_value

    def test_metrics_reset(self, metrics_collector):
        """测试指标重置"""
        # 先收集一些指标
        metrics_collector.collect_performance_metrics(
            portfolio_value=1000000.0,
            daily_return=0.01,
            total_return=0.1,
            sharpe_ratio=1.0
        )

        # 验证指标存在
        assert len(metrics_collector.performance_metrics) > 1

        # 重置指标
        metrics_collector.reset_metrics()

        # 验证指标被重置
        assert len(metrics_collector.performance_metrics) == 0
        assert len(metrics_collector.risk_metrics) == 0
        assert len(metrics_collector.system_metrics) == 0
        assert len(metrics_collector.trading_metrics) == 0

    def test_metrics_export_format(self, metrics_collector):
        """测试指标导出格式"""
        # 收集各类指标
        metrics_collector.collect_performance_metrics(
            portfolio_value=1000000.0,
            daily_return=0.01,
            total_return=0.1,
            sharpe_ratio=1.0
        )
        
        metrics_collector.collect_risk_metrics(
            volatility=0.15,
            max_drawdown=0.05,
            var_95=0.02,
            beta=1.1
        )

        # 导出指标
        exported_metrics = metrics_collector.export_metrics()

        # 验证导出格式
        assert isinstance(exported_metrics, dict)
        assert 'performance_metrics' in exported_metrics
        assert 'risk_metrics' in exported_metrics
        assert 'system_metrics' in exported_metrics
        assert 'trading_metrics' in exported_metrics

        # 验证每个类别包含预期的指标
        performance = exported_metrics['performance_metrics']
        assert 'portfolio_value' in performance
        assert 'daily_return' in performance
        assert 'timestamp' in performance

        risk = exported_metrics['risk_metrics']
        assert 'volatility' in risk
        assert 'max_drawdown' in risk

    def test_invalid_metric_name_error(self, metrics_collector):
        """测试无效指标名称错误"""
        # 测试空指标名称
        with pytest.raises(ValueError, match="指标名称不能为空"):
            metrics_collector.register_metric("", "description", "gauge")

        # 测试更新不存在的指标
        with pytest.raises(ValueError, match="指标.*不存在"):
            metrics_collector.update_metric("nonexistent_metric", 42.0)

    def test_invalid_metric_type_error(self, metrics_collector):
        """测试无效指标类型错误"""
        # 测试不支持的指标类型
        with pytest.raises(ValueError, match="不支持的指标类型"):
            metrics_collector.register_metric("test_metric", "description", "invalid_type")


class TestPrometheusExporter:
    """Prometheus导出器测试类"""

    @pytest.fixture
    def metrics_collector(self):
        """创建指标收集器"""
        collector = MetricsCollector()
        # 预填充一些测试数据
        collector.collect_performance_metrics(
            portfolio_value=1000000.0,
            daily_return=0.01,
            total_return=0.1,
            sharpe_ratio=1.5
        )
        collector.collect_risk_metrics(
            volatility=0.15,
            max_drawdown=0.05,
            var_95=0.02,
            beta=1.1
        )
        return collector

    @pytest.fixture
    def prometheus_exporter(self, metrics_collector):
        """创建Prometheus导出器"""
        return PrometheusExporter(metrics_collector, port=8001)

    def test_prometheus_exporter_initialization(self, prometheus_exporter):
        """测试Prometheus导出器初始化"""
        assert prometheus_exporter.metrics_collector is not None
        assert prometheus_exporter.port == 8001
        assert prometheus_exporter.registry is not None
        assert prometheus_exporter.is_running is False

    def test_prometheus_metrics_registration(self, prometheus_exporter):
        """测试Prometheus指标注册"""
        # 注册指标到Prometheus
        prometheus_exporter.register_prometheus_metrics()

        # 验证指标被注册到Prometheus注册表
        registered_metrics = prometheus_exporter.get_registered_metrics()
        
        # 检查基本指标类型
        assert 'portfolio_value' in registered_metrics
        assert 'daily_return' in registered_metrics
        assert 'volatility' in registered_metrics
        assert 'max_drawdown' in registered_metrics

    def test_metrics_export_format(self, prometheus_exporter):
        """测试指标导出格式"""
        # 注册并导出指标
        prometheus_exporter.register_prometheus_metrics()
        exported_data = prometheus_exporter.generate_metrics_output()

        # 验证Prometheus格式
        assert isinstance(exported_data, str)
        assert "portfolio_value" in exported_data
        assert "daily_return" in exported_data
        assert "volatility" in exported_data
        
        # 验证Prometheus格式规范
        lines = exported_data.strip().split('\n')
        for line in lines:
            if line.startswith('#'):
                # 注释行应该包含HELP或TYPE
                assert 'HELP' in line or 'TYPE' in line
            elif line:
                # 指标行应该包含指标名和值
                assert ' ' in line
                parts = line.split(' ')
                assert len(parts) >= 2

    def test_exporter_start_stop(self, prometheus_exporter):
        """测试导出器启动停止"""
        # 启动导出器
        prometheus_exporter.start()
        assert prometheus_exporter.is_running is True
        
        # 等待一小段时间确保服务器启动
        time.sleep(0.1)
        
        # 停止导出器
        prometheus_exporter.stop()
        assert prometheus_exporter.is_running is False

    def test_concurrent_metrics_update(self, prometheus_exporter):
        """测试并发指标更新"""
        prometheus_exporter.register_prometheus_metrics()
        
        def update_metrics():
            for i in range(10):
                prometheus_exporter.metrics_collector.collect_performance_metrics(
                    portfolio_value=1000000.0 + i * 1000,
                    daily_return=0.01 + i * 0.001,
                    total_return=0.1 + i * 0.01,
                    sharpe_ratio=1.5 + i * 0.1
                )
                time.sleep(0.01)

        # 启动多个线程同时更新指标
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=update_metrics)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证指标仍然可以正常导出
        exported_data = prometheus_exporter.generate_metrics_output()
        assert "portfolio_value" in exported_data

    def test_custom_metrics_export(self, prometheus_exporter):
        """测试自定义指标导出"""
        # 注册自定义指标
        prometheus_exporter.metrics_collector.register_metric(
            "custom_strategy_score",
            "Custom strategy performance score",
            "gauge"
        )
        
        # 更新自定义指标
        prometheus_exporter.metrics_collector.update_metric("custom_strategy_score", 85.5)
        
        # 重新注册Prometheus指标以包含新的自定义指标
        prometheus_exporter.register_prometheus_metrics()
        
        # 验证自定义指标被导出
        exported_data = prometheus_exporter.generate_metrics_output()
        assert "custom_strategy_score" in exported_data
        assert "85.5" in exported_data

    def test_invalid_port_error(self, metrics_collector):
        """测试无效端口错误"""
        # 测试无效端口范围
        with pytest.raises(ValueError, match="端口号必须在1024-65535范围内"):
            PrometheusExporter(metrics_collector, port=999)

        with pytest.raises(ValueError, match="端口号必须在1024-65535范围内"):
            PrometheusExporter(metrics_collector, port=70000)

    def test_exporter_health_check(self, prometheus_exporter):
        """测试导出器健康检查"""
        # 健康检查应该在启动前失败
        assert prometheus_exporter.health_check() is False
        
        # 启动导出器
        prometheus_exporter.start()
        
        # 等待启动
        time.sleep(0.1)
        
        # 健康检查应该成功
        assert prometheus_exporter.health_check() is True
        
        # 停止导出器
        prometheus_exporter.stop()
        
        # 健康检查应该再次失败
        assert prometheus_exporter.health_check() is False


class TestGrafanaDashboardManager:
    """Grafana仪表板管理器测试类"""

    @pytest.fixture
    def dashboard_manager(self):
        """创建Grafana仪表板管理器"""
        return GrafanaDashboardManager(
            grafana_url="http://localhost:3000",
            api_key="test_api_key"
        )

    def test_dashboard_manager_initialization(self, dashboard_manager):
        """测试仪表板管理器初始化"""
        assert dashboard_manager.grafana_url == "http://localhost:3000"
        assert dashboard_manager.api_key == "test_api_key"
        assert dashboard_manager.session is not None

    def test_dashboard_config_generation(self, dashboard_manager):
        """测试仪表板配置生成"""
        # 生成仪表板配置
        dashboard_config = dashboard_manager.generate_dashboard_config()

        # 验证配置结构
        assert isinstance(dashboard_config, dict)
        assert 'dashboard' in dashboard_config
        assert 'title' in dashboard_config['dashboard']
        assert 'panels' in dashboard_config['dashboard']

        # 验证面板配置
        panels = dashboard_config['dashboard']['panels']
        assert len(panels) > 0

        # 检查基本面板
        panel_titles = [panel['title'] for panel in panels]
        assert '投资组合价值' in panel_titles
        assert '日收益率' in panel_titles
        assert '风险指标' in panel_titles
        assert '系统性能' in panel_titles

    def test_dashboard_panel_creation(self, dashboard_manager):
        """测试仪表板面板创建"""
        # 创建单个面板
        panel_config = dashboard_manager.create_panel(
            title="测试面板",
            metric_query="portfolio_value",
            panel_type="graph",
            x_pos=0,
            y_pos=0,
            width=12,
            height=8
        )

        # 验证面板配置
        assert panel_config['title'] == "测试面板"
        assert panel_config['type'] == "graph"
        assert panel_config['gridPos']['x'] == 0
        assert panel_config['gridPos']['y'] == 0
        assert panel_config['gridPos']['w'] == 12
        assert panel_config['gridPos']['h'] == 8

        # 验证查询配置
        assert 'targets' in panel_config
        assert len(panel_config['targets']) > 0
        assert panel_config['targets'][0]['expr'] == "portfolio_value"

    def test_dashboard_deployment(self, dashboard_manager):
        """测试仪表板部署"""
        with patch.object(dashboard_manager.session, 'post') as mock_post:
            # 模拟成功的API响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'status': 'success',
                'id': 1,
                'uid': 'test-dashboard',
                'url': '/d/test-dashboard/trading-system-monitor'
            }
            mock_post.return_value = mock_response

            # 部署仪表板
            result = dashboard_manager.deploy_dashboard()

            # 验证部署结果
            assert result['status'] == 'success'
            assert result['uid'] == 'test-dashboard'

            # 验证API调用
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0].endswith('/api/dashboards/db')

    def test_dashboard_update(self, dashboard_manager):
        """测试仪表板更新"""
        with patch.object(dashboard_manager.session, 'post') as mock_post:
            # 模拟更新响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'status': 'success',
                'version': 2
            }
            mock_post.return_value = mock_response

            # 更新仪表板
            result = dashboard_manager.update_dashboard(dashboard_uid="test-dashboard")

            # 验证更新结果
            assert result['status'] == 'success'
            assert result['version'] == 2

    def test_dashboard_deletion(self, dashboard_manager):
        """测试仪表板删除"""
        with patch.object(dashboard_manager.session, 'delete') as mock_delete:
            # 模拟删除响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'status': 'success'}
            mock_delete.return_value = mock_response

            # 删除仪表板
            result = dashboard_manager.delete_dashboard(dashboard_uid="test-dashboard")

            # 验证删除结果
            assert result['status'] == 'success'

            # 验证API调用
            mock_delete.assert_called_once()
            call_args = mock_delete.call_args
            assert 'test-dashboard' in call_args[0][0]

    def test_alert_rules_creation(self, dashboard_manager):
        """测试告警规则创建"""
        # 创建告警规则
        alert_rule = dashboard_manager.create_alert_rule(
            rule_name="高风险告警",
            metric_query="max_drawdown",
            threshold=0.1,
            condition="gt",  # greater than
            evaluation_interval="1m"
        )

        # 验证告警规则配置
        assert alert_rule['name'] == "高风险告警"
        assert alert_rule['condition']['query'] == "max_drawdown"
        assert alert_rule['condition']['threshold'] == 0.1
        assert alert_rule['condition']['type'] == "gt"
        assert alert_rule['frequency'] == "1m"

    def test_datasource_configuration(self, dashboard_manager):
        """测试数据源配置"""
        # 配置Prometheus数据源
        datasource_config = dashboard_manager.configure_prometheus_datasource(
            prometheus_url="http://localhost:9090",
            datasource_name="TradingSystem"
        )

        # 验证数据源配置
        assert datasource_config['name'] == "TradingSystem"
        assert datasource_config['type'] == "prometheus"
        assert datasource_config['url'] == "http://localhost:9090"
        assert datasource_config['access'] == "proxy"

    def test_invalid_grafana_url_error(self):
        """测试无效Grafana URL错误"""
        with pytest.raises(ValueError, match="Grafana URL不能为空"):
            GrafanaDashboardManager("", "api_key")

        with pytest.raises(ValueError, match="无效的Grafana URL格式"):
            GrafanaDashboardManager("invalid_url", "api_key")

    def test_api_authentication_error(self, dashboard_manager):
        """测试API认证错误"""
        with patch.object(dashboard_manager.session, 'post') as mock_post:
            # 模拟认证失败响应
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.json.return_value = {'error': 'Unauthorized'}
            mock_post.return_value = mock_response

            # 部署仪表板应该引发认证错误
            with pytest.raises(Exception, match="Grafana API认证失败"):
                dashboard_manager.deploy_dashboard()


class TestTradingSystemMonitor:
    """交易系统监控器测试类"""

    @pytest.fixture
    def sample_trades(self):
        """创建样本交易数据"""
        return [
            Trade("000001.SZ", OrderType.BUY, 1000, Decimal("10.00"), datetime(2023, 1, 2), Decimal("10.00")),
            Trade("000001.SZ", OrderType.SELL, 500, Decimal("11.00"), datetime(2023, 1, 15), Decimal("5.50")),
            Trade("000002.SZ", OrderType.BUY, 2000, Decimal("5.00"), datetime(2023, 1, 20), Decimal("10.00")),
        ]

    @pytest.fixture
    def trading_monitor(self):
        """创建交易系统监控器"""
        return TradingSystemMonitor(
            prometheus_port=8002,
            grafana_url="http://localhost:3000",
            grafana_api_key="test_key"
        )

    def test_trading_monitor_initialization(self, trading_monitor):
        """测试交易监控器初始化"""
        assert trading_monitor.metrics_collector is not None
        assert trading_monitor.prometheus_exporter is not None
        assert trading_monitor.dashboard_manager is not None
        assert trading_monitor.is_monitoring is False

    def test_monitoring_start_stop(self, trading_monitor):
        """测试监控启动停止"""
        # 启动监控
        trading_monitor.start_monitoring()
        assert trading_monitor.is_monitoring is True
        assert trading_monitor.prometheus_exporter.is_running is True

        # 停止监控
        trading_monitor.stop_monitoring()
        assert trading_monitor.is_monitoring is False
        assert trading_monitor.prometheus_exporter.is_running is False

    def test_portfolio_monitoring(self, trading_monitor):
        """测试投资组合监控"""
        # 模拟投资组合数据
        portfolio_data = {
            'value': 1050000.0,
            'daily_return': 0.02,
            'total_return': 0.05,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.03
        }

        # 更新投资组合指标
        trading_monitor.update_portfolio_metrics(portfolio_data)

        # 验证指标被收集
        performance_metrics = trading_monitor.metrics_collector.performance_metrics
        assert performance_metrics['portfolio_value'] == 1050000.0
        assert performance_metrics['daily_return'] == 0.02

        risk_metrics = trading_monitor.metrics_collector.risk_metrics
        assert risk_metrics['max_drawdown'] == 0.03

    def test_trading_activity_monitoring(self, trading_monitor, sample_trades):
        """测试交易活动监控"""
        # 更新交易指标
        trading_monitor.update_trading_metrics(sample_trades)

        # 验证交易指标被收集
        trading_metrics = trading_monitor.metrics_collector.trading_metrics
        assert trading_metrics['total_trades'] == len(sample_trades)
        
        # 验证交易统计
        buy_trades = [t for t in sample_trades if t.trade_type == OrderType.BUY]
        sell_trades = [t for t in sample_trades if t.trade_type == OrderType.SELL]
        assert trading_metrics['buy_trades'] == len(buy_trades)
        assert trading_metrics['sell_trades'] == len(sell_trades)

    def test_system_resource_monitoring(self, trading_monitor):
        """测试系统资源监控"""
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory:
            
            # 模拟系统资源数据
            mock_cpu.return_value = 45.5
            mock_memory.return_value = Mock(percent=78.2)

            # 更新系统指标
            trading_monitor.update_system_metrics()

            # 验证系统指标被收集
            system_metrics = trading_monitor.metrics_collector.system_metrics
            assert system_metrics['cpu_usage'] == 45.5
            assert system_metrics['memory_usage'] == 78.2

    def test_real_time_monitoring(self, trading_monitor):
        """测试实时监控"""
        # 启动实时监控
        trading_monitor.start_monitoring()

        # 模拟数据更新
        portfolio_data = {
            'value': 1000000.0,
            'daily_return': 0.01,
            'total_return': 0.1,
            'sharpe_ratio': 1.0,
            'max_drawdown': 0.05
        }

        # 多次更新数据模拟实时监控
        for i in range(5):
            portfolio_data['value'] += i * 1000
            portfolio_data['daily_return'] += i * 0.001
            
            trading_monitor.update_portfolio_metrics(portfolio_data)
            time.sleep(0.1)

        # 验证最新数据被收集
        latest_metrics = trading_monitor.get_latest_metrics()
        assert latest_metrics['performance_metrics']['portfolio_value'] == 1004000.0

        # 停止监控
        trading_monitor.stop_monitoring()

    def test_monitoring_accuracy(self, trading_monitor):
        """测试监控准确性"""
        # 启动监控
        trading_monitor.start_monitoring()

        # 设置基准数据
        expected_portfolio_value = 1234567.89
        expected_daily_return = 0.0123
        expected_sharpe_ratio = 1.234

        # 更新指标
        portfolio_data = {
            'value': expected_portfolio_value,
            'daily_return': expected_daily_return,
            'total_return': 0.1,
            'sharpe_ratio': expected_sharpe_ratio,
            'max_drawdown': 0.05
        }
        trading_monitor.update_portfolio_metrics(portfolio_data)

        # 获取收集的指标
        collected_metrics = trading_monitor.get_latest_metrics()

        # 验证数据准确性（精度测试）
        assert abs(collected_metrics['performance_metrics']['portfolio_value'] - expected_portfolio_value) < 0.01
        assert abs(collected_metrics['performance_metrics']['daily_return'] - expected_daily_return) < 1e-6
        assert abs(collected_metrics['performance_metrics']['sharpe_ratio'] - expected_sharpe_ratio) < 1e-6

        # 停止监控
        trading_monitor.stop_monitoring()

    def test_monitoring_dashboard_integration(self, trading_monitor):
        """测试监控与仪表板集成"""
        with patch.object(trading_monitor.dashboard_manager, 'deploy_dashboard') as mock_deploy:
            # 模拟仪表板部署成功
            mock_deploy.return_value = {
                'status': 'success',
                'uid': 'trading-monitor',
                'url': '/d/trading-monitor/trading-system'
            }

            # 部署监控仪表板
            result = trading_monitor.setup_dashboard()

            # 验证仪表板设置成功
            assert result['status'] == 'success'
            assert 'trading-monitor' in result['uid']

            # 验证部署被调用
            mock_deploy.assert_called_once()

    def test_monitoring_error_handling(self, trading_monitor):
        """测试监控错误处理"""
        # 测试无效的投资组合数据
        invalid_portfolio_data = {
            'value': -1000000.0,  # 负值
            'daily_return': float('inf'),  # 无穷大
            'total_return': float('nan'),  # NaN
        }

        # 应该能处理无效数据而不崩溃
        with pytest.raises(ValueError, match="投资组合价值不能为负数"):
            trading_monitor.update_portfolio_metrics(invalid_portfolio_data)

    def test_metrics_history_tracking(self, trading_monitor):
        """测试指标历史跟踪"""
        # 启动监控
        trading_monitor.start_monitoring()

        # 添加多个历史数据点
        historical_data = [
            {'value': 1000000.0, 'daily_return': 0.01},
            {'value': 1010000.0, 'daily_return': 0.015},
            {'value': 1025000.0, 'daily_return': 0.008},
            {'value': 1030000.0, 'daily_return': 0.012},
        ]

        for data in historical_data:
            portfolio_data = {
                'value': data['value'],
                'daily_return': data['daily_return'],
                'total_return': 0.1,
                'sharpe_ratio': 1.0,
                'max_drawdown': 0.05
            }
            trading_monitor.update_portfolio_metrics(portfolio_data)
            time.sleep(0.05)  # 确保时间戳不同

        # 获取历史指标
        history = trading_monitor.get_metrics_history(limit=4)

        # 验证历史记录
        assert len(history) == 4
        assert history[0]['performance_metrics']['portfolio_value'] == 1000000.0
        assert history[-1]['performance_metrics']['portfolio_value'] == 1030000.0

        # 停止监控
        trading_monitor.stop_monitoring()

    def test_concurrent_monitoring_operations(self, trading_monitor):
        """测试并发监控操作"""
        # 启动监控
        trading_monitor.start_monitoring()

        def update_portfolio_metrics():
            for i in range(10):
                portfolio_data = {
                    'value': 1000000.0 + i * 1000,
                    'daily_return': 0.01 + i * 0.001,
                    'total_return': 0.1,
                    'sharpe_ratio': 1.0,
                    'max_drawdown': 0.05
                }
                trading_monitor.update_portfolio_metrics(portfolio_data)
                time.sleep(0.01)

        def update_system_metrics():
            with patch('psutil.cpu_percent', return_value=50.0), \
                 patch('psutil.virtual_memory', return_value=Mock(percent=60.0)):
                for _ in range(10):
                    trading_monitor.update_system_metrics()
                    time.sleep(0.01)

        # 启动并发操作
        threads = []
        for func in [update_portfolio_metrics, update_system_metrics]:
            thread = threading.Thread(target=func)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证监控仍然正常工作
        latest_metrics = trading_monitor.get_latest_metrics()
        assert 'performance_metrics' in latest_metrics
        assert 'system_metrics' in latest_metrics

        # 停止监控
        trading_monitor.stop_monitoring()