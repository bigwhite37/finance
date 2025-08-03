"""
Web仪表板测试模块

测试DrawdownDashboard的功能和用户体验。
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.rl_trading_system.web_dashboard import DrawdownDashboard, create_dashboard_templates
from src.rl_trading_system.monitoring.drawdown_monitoring_service import (
    DrawdownMonitoringService, MonitoringMetrics
)


class TestDrawdownDashboard:
    """回撤监控仪表板测试"""
    
    def setup_method(self):
        """测试设置"""
        # 创建临时数据库
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # 创建监控服务
        config = {'db_path': self.temp_db.name}
        self.monitoring_service = DrawdownMonitoringService(config)
        
        # Mock监控服务的组件
        self.monitoring_service.drawdown_monitor = Mock()
        self.monitoring_service.attribution_analyzer = Mock()
        self.monitoring_service.market_regime_detector = Mock()
        
        # 创建仪表板
        self.dashboard = DrawdownDashboard(self.monitoring_service)
    
    def teardown_method(self):
        """测试清理"""
        if hasattr(self, 'monitoring_service'):
            self.monitoring_service.stop_monitoring()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_dashboard_initialization(self):
        """测试仪表板初始化"""
        assert self.dashboard.monitoring_service is not None
        assert self.dashboard.app is not None
        assert self.dashboard.logger is not None
    
    def test_dashboard_routes_setup(self):
        """测试路由设置"""
        with self.dashboard.app.test_client() as client:
            # 测试主页路由
            response = client.get('/')
            assert response.status_code == 200
            assert b'dashboard.html' in response.data or '回撤监控仪表板'.encode('utf-8') in response.data
    
    def test_overview_api_no_data(self):
        """测试概览API无数据情况"""
        with self.dashboard.app.test_client() as client:
            response = client.get('/api/dashboard/overview')
            assert response.status_code == 404
            data = json.loads(response.data)
            assert 'error' in data
    
    def test_overview_api_with_data(self):
        """测试概览API有数据情况"""
        # 设置模拟数据
        mock_metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            portfolio_value=1000000.0,
            current_drawdown=-0.05,
            max_drawdown=-0.10,
            drawdown_duration=5,
            risk_budget_usage=0.5,
            position_count=10,
            concentration_score=0.2,
            volatility=0.15,
            sharpe_ratio=1.2,
            market_regime="bull",
            alert_level="NORMAL"
        )
        
        self.monitoring_service.last_metrics = mock_metrics
        
        # 添加历史数据
        for i in range(10):
            historical_metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=1000000.0 + i * 1000,
                current_drawdown=-0.01 * i,
                max_drawdown=-0.05,
                drawdown_duration=i,
                risk_budget_usage=0.5,
                position_count=10,
                concentration_score=0.2,
                volatility=0.15,
                sharpe_ratio=1.2,
                market_regime="bull",
                alert_level="NORMAL"
            )
            self.monitoring_service.historical_data_manager.save_metrics(historical_metrics)
        
        with self.dashboard.app.test_client() as client:
            response = client.get('/api/dashboard/overview')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'current_metrics' in data
            assert 'trends' in data
            assert 'summary_stats' in data
            assert 'last_updated' in data
            
            # 验证指标数据
            current_metrics = data['current_metrics']
            assert current_metrics['portfolio_value'] == 1000000.0
            assert current_metrics['current_drawdown'] == -0.05
            assert current_metrics['alert_level'] == "NORMAL"
    
    def test_alerts_api(self):
        """测试告警API"""
        # 添加测试告警
        self.monitoring_service.historical_data_manager.save_alert(
            "DRAWDOWN_WARNING", "WARNING", "测试告警消息"
        )
        
        with self.dashboard.app.test_client() as client:
            response = client.get('/api/dashboard/alerts')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'active_alerts' in data
            assert 'alert_count' in data
            assert 'critical_count' in data
            assert 'warning_count' in data
            
            assert data['alert_count'] == 1
            assert data['warning_count'] == 1
            assert len(data['active_alerts']) == 1
            assert data['active_alerts'][0]['alert_type'] == "DRAWDOWN_WARNING"
    
    def test_drawdown_curve_chart_api(self):
        """测试回撤曲线图表API"""
        # 添加历史数据
        for i in range(20):
            metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=1000000.0 - i * 1000,
                current_drawdown=-0.001 * i,
                max_drawdown=-0.02,
                drawdown_duration=i,
                risk_budget_usage=0.3,
                position_count=10,
                concentration_score=0.2,
                volatility=0.15,
                sharpe_ratio=1.0,
                market_regime="bull",
                alert_level="NORMAL"
            )
            self.monitoring_service.historical_data_manager.save_metrics(metrics)
        
        with self.dashboard.app.test_client() as client:
            response = client.get('/api/dashboard/charts/drawdown_curve')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'data' in data
            assert 'layout' in data
            
            # 验证图表数据结构
            assert isinstance(data['data'], list)
            assert len(data['data']) > 0
            assert 'x' in data['data'][0]
            assert 'y' in data['data'][0]
    
    def test_portfolio_value_chart_api(self):
        """测试投资组合价值图表API"""
        # 添加历史数据
        for i in range(20):
            metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=1000000.0 + i * 5000,
                current_drawdown=-0.001 * i,
                max_drawdown=-0.02,
                drawdown_duration=i,
                risk_budget_usage=0.3,
                position_count=10,
                concentration_score=0.2,
                volatility=0.15,
                sharpe_ratio=1.0,
                market_regime="bull",
                alert_level="NORMAL"
            )
            self.monitoring_service.historical_data_manager.save_metrics(metrics)
        
        with self.dashboard.app.test_client() as client:
            response = client.get('/api/dashboard/charts/portfolio_value')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'data' in data
            assert 'layout' in data
            
            # 验证图表包含投资组合价值和峰值线
            assert len(data['data']) >= 2  # 至少包含价值线和峰值线
    
    def test_risk_metrics_chart_api(self):
        """测试风险指标图表API"""
        # 添加历史数据
        for i in range(20):
            metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=1000000.0,
                current_drawdown=-0.001 * i,
                max_drawdown=-0.02,
                drawdown_duration=i,
                risk_budget_usage=0.3 + i * 0.01,
                position_count=10,
                concentration_score=0.2 + i * 0.005,
                volatility=0.15 + i * 0.002,
                sharpe_ratio=1.0,
                market_regime="bull",
                alert_level="NORMAL"
            )
            self.monitoring_service.historical_data_manager.save_metrics(metrics)
        
        with self.dashboard.app.test_client() as client:
            response = client.get('/api/dashboard/charts/risk_metrics')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'data' in data
            assert 'layout' in data
            
            # 验证图表包含多个风险指标
            assert len(data['data']) >= 3  # 波动率、风险预算使用率、集中度分数
    
    def test_performance_stats_api(self):
        """测试性能统计API"""
        # 添加历史数据
        portfolio_values = []
        for i in range(50):
            value = 1000000.0 + i * 2000 + np.random.normal(0, 1000)
            portfolio_values.append(value)
            
            metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=value,
                current_drawdown=-0.001 * i,
                max_drawdown=-0.02,
                drawdown_duration=i,
                risk_budget_usage=0.3,
                position_count=10,
                concentration_score=0.2,
                volatility=0.15,
                sharpe_ratio=1.2,
                market_regime="bull",
                alert_level="NORMAL"
            )
            self.monitoring_service.historical_data_manager.save_metrics(metrics)
        
        with self.dashboard.app.test_client() as client:
            response = client.get('/api/dashboard/performance_stats')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            
            # 验证性能统计指标
            required_stats = [
                'total_return', 'annualized_return', 'max_drawdown', 'current_drawdown',
                'avg_volatility', 'current_volatility', 'avg_sharpe_ratio', 'current_sharpe_ratio',
                'win_rate', 'avg_win', 'avg_loss', 'profit_factor'
            ]
            
            for stat in required_stats:
                assert stat in data
                assert isinstance(data[stat], (int, float))
    
    def test_chart_api_no_data(self):
        """测试图表API无数据情况"""
        with self.dashboard.app.test_client() as client:
            # 测试回撤曲线
            response = client.get('/api/dashboard/charts/drawdown_curve')
            assert response.status_code == 404
            
            # 测试投资组合价值
            response = client.get('/api/dashboard/charts/portfolio_value')
            assert response.status_code == 404
            
            # 测试风险指标
            response = client.get('/api/dashboard/charts/risk_metrics')
            assert response.status_code == 404
            
            # 测试性能统计
            response = client.get('/api/dashboard/performance_stats')
            assert response.status_code == 404
    
    def test_trend_calculation(self):
        """测试趋势计算"""
        # 测试上升趋势
        up_values = [1.0, 1.1, 1.2, 1.3, 1.4]
        trend = self.dashboard._calculate_trend(up_values)
        assert trend == 'up'
        
        # 测试下降趋势
        down_values = [1.4, 1.3, 1.2, 1.1, 1.0]
        trend = self.dashboard._calculate_trend(down_values)
        assert trend == 'down'
        
        # 测试稳定趋势
        stable_values = [1.0, 1.01, 0.99, 1.02, 1.0]
        trend = self.dashboard._calculate_trend(stable_values)
        assert trend == 'stable'
        
        # 测试数据不足情况
        insufficient_values = [1.0]
        trend = self.dashboard._calculate_trend(insufficient_values)
        assert trend == 'stable'
    
    def test_summary_stats_calculation(self):
        """测试汇总统计计算"""
        # 创建测试数据
        historical_data = []
        for i in range(10):
            data = {
                'current_drawdown': -0.01 * i,
                'volatility': 0.15 + i * 0.01,
                'portfolio_value': 1000000 + i * 10000
            }
            historical_data.append(data)
        
        stats = self.dashboard._calculate_summary_stats(historical_data)
        
        assert 'avg_drawdown' in stats
        assert 'max_drawdown' in stats
        assert 'avg_volatility' in stats
        assert 'portfolio_return' in stats
        assert 'data_points' in stats
        
        assert stats['data_points'] == 10
        assert stats['avg_drawdown'] == -0.045  # 平均值
        assert stats['max_drawdown'] == -0.09   # 最大回撤（最小值）
        assert abs(stats['portfolio_return'] - 0.09) < 0.001  # 9%收益率
    
    def test_performance_stats_calculation(self):
        """测试性能统计计算"""
        # 创建测试数据
        historical_data = []
        base_value = 1000000
        for i in range(20):
            value = base_value + i * 5000 + (i % 3 - 1) * 2000  # 模拟波动
            data = {
                'portfolio_value': value,
                'current_drawdown': -0.01 * (i % 5),
                'volatility': 0.15 + i * 0.001,
                'sharpe_ratio': 1.0 + i * 0.05
            }
            historical_data.append(data)
        
        stats = self.dashboard._calculate_performance_stats(historical_data)
        
        # 验证所有必需的统计指标都存在
        required_stats = [
            'total_return', 'annualized_return', 'max_drawdown', 'current_drawdown',
            'avg_volatility', 'current_volatility', 'avg_sharpe_ratio', 'current_sharpe_ratio',
            'win_rate', 'avg_win', 'avg_loss', 'profit_factor'
        ]
        
        for stat in required_stats:
            assert stat in stats
            assert isinstance(stats[stat], (int, float))
        
        # 验证一些基本的数值合理性
        assert stats['total_return'] > 0  # 应该有正收益
        assert 0 <= stats['win_rate'] <= 100  # 胜率应该在0-100%之间
        assert stats['current_volatility'] > 0  # 波动率应该为正
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        # 测试空历史数据
        empty_stats = self.dashboard._calculate_summary_stats([])
        assert empty_stats == {}
        
        empty_performance = self.dashboard._calculate_performance_stats([])
        assert empty_performance == {}
        
        # 测试空趋势数据
        empty_trend = self.dashboard._calculate_trend([])
        assert empty_trend == 'stable'


class TestDashboardTemplates:
    """仪表板模板测试"""
    
    def test_create_dashboard_templates(self):
        """测试创建仪表板模板"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 临时修改模块路径
            import src.rl_trading_system.web_dashboard as dashboard_module
            original_file = dashboard_module.__file__
            
            try:
                # 修改__file__路径以使用临时目录
                dashboard_module.__file__ = os.path.join(temp_dir, 'web_dashboard.py')
                
                # 创建模板
                create_dashboard_templates()
                
                # 验证模板文件是否创建
                template_path = os.path.join(temp_dir, 'templates', 'dashboard.html')
                assert os.path.exists(template_path)
                
                # 验证模板内容
                with open(template_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert '回撤监控仪表板' in content
                    assert 'drawdown-chart' in content
                    assert 'portfolio-chart' in content
                    assert 'risk-chart' in content
                    assert 'Bootstrap' in content or 'bootstrap' in content
                    assert 'Plotly' in content or 'plotly' in content
                
            finally:
                # 恢复原始路径
                dashboard_module.__file__ = original_file


class TestDashboardIntegration:
    """仪表板集成测试"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        config = {'db_path': self.temp_db.name}
        self.monitoring_service = DrawdownMonitoringService(config)
        self.dashboard = DrawdownDashboard(self.monitoring_service)
    
    def teardown_method(self):
        """测试清理"""
        if hasattr(self, 'monitoring_service'):
            self.monitoring_service.stop_monitoring()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_full_dashboard_workflow(self):
        """测试完整的仪表板工作流程"""
        # 1. 添加监控数据
        for i in range(30):
            metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=1000000.0 + i * 3000 + np.random.normal(0, 5000),
                current_drawdown=-0.002 * i + np.random.normal(0, 0.001),
                max_drawdown=-0.05,
                drawdown_duration=i % 10,
                risk_budget_usage=0.3 + i * 0.01,
                position_count=10 + i % 5,
                concentration_score=0.2 + i * 0.003,
                volatility=0.15 + i * 0.001,
                sharpe_ratio=1.0 + i * 0.02,
                market_regime=["bull", "bear", "sideways"][i % 3],
                alert_level=["NORMAL", "WARNING", "CRITICAL"][i % 3]
            )
            self.monitoring_service.historical_data_manager.save_metrics(metrics)
        
        # 设置当前指标
        self.monitoring_service.last_metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            portfolio_value=1100000.0,
            current_drawdown=-0.03,
            max_drawdown=-0.08,
            drawdown_duration=5,
            risk_budget_usage=0.6,
            position_count=12,
            concentration_score=0.25,
            volatility=0.18,
            sharpe_ratio=1.5,
            market_regime="bull",
            alert_level="WARNING"
        )
        
        # 添加告警
        self.monitoring_service.historical_data_manager.save_alert(
            "DRAWDOWN_WARNING", "WARNING", "回撤超过警告阈值"
        )
        self.monitoring_service.historical_data_manager.save_alert(
            "HIGH_VOLATILITY", "WARNING", "波动率过高"
        )
        
        # 2. 测试所有API端点
        with self.dashboard.app.test_client() as client:
            # 测试主页
            response = client.get('/')
            assert response.status_code == 200
            
            # 测试概览API
            response = client.get('/api/dashboard/overview')
            assert response.status_code == 200
            overview_data = json.loads(response.data)
            assert 'current_metrics' in overview_data
            assert 'trends' in overview_data
            
            # 测试告警API
            response = client.get('/api/dashboard/alerts')
            assert response.status_code == 200
            alerts_data = json.loads(response.data)
            assert alerts_data['alert_count'] == 2
            assert alerts_data['warning_count'] == 2
            
            # 测试图表API
            chart_endpoints = [
                '/api/dashboard/charts/drawdown_curve',
                '/api/dashboard/charts/portfolio_value',
                '/api/dashboard/charts/risk_metrics'
            ]
            
            for endpoint in chart_endpoints:
                response = client.get(endpoint)
                assert response.status_code == 200
                chart_data = json.loads(response.data)
                assert 'data' in chart_data
                assert 'layout' in chart_data
            
            # 测试性能统计API
            response = client.get('/api/dashboard/performance_stats')
            assert response.status_code == 200
            stats_data = json.loads(response.data)
            assert 'total_return' in stats_data
            assert 'win_rate' in stats_data
    
    def test_dashboard_error_handling(self):
        """测试仪表板错误处理"""
        # 测试数据库连接错误的情况
        with patch.object(self.monitoring_service.historical_data_manager, 'get_historical_metrics', 
                         side_effect=Exception("数据库连接失败")):
            
            with self.dashboard.app.test_client() as client:
                response = client.get('/api/dashboard/overview')
                assert response.status_code == 500
                error_data = json.loads(response.data)
                assert 'error' in error_data
    
    def test_dashboard_performance(self):
        """测试仪表板性能"""
        import time
        
        # 添加大量数据
        for i in range(1000):
            metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=1000000.0 + i * 100,
                current_drawdown=-0.001 * i,
                max_drawdown=-0.05,
                drawdown_duration=i % 20,
                risk_budget_usage=0.3,
                position_count=10,
                concentration_score=0.2,
                volatility=0.15,
                sharpe_ratio=1.0,
                market_regime="bull",
                alert_level="NORMAL"
            )
            self.monitoring_service.historical_data_manager.save_metrics(metrics)
        
        # 设置当前指标
        self.monitoring_service.last_metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            portfolio_value=1100000.0,
            current_drawdown=-0.02,
            max_drawdown=-0.05,
            drawdown_duration=3,
            risk_budget_usage=0.4,
            position_count=10,
            concentration_score=0.2,
            volatility=0.15,
            sharpe_ratio=1.2,
            market_regime="bull",
            alert_level="NORMAL"
        )
        
        with self.dashboard.app.test_client() as client:
            # 测试API响应时间
            start_time = time.time()
            response = client.get('/api/dashboard/overview')
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 2.0  # 响应时间应该小于2秒
            
            # 测试图表生成时间
            start_time = time.time()
            response = client.get('/api/dashboard/charts/drawdown_curve?hours=24')
            chart_time = time.time() - start_time
            
            assert response.status_code == 200
            assert chart_time < 3.0  # 图表生成时间应该小于3秒


if __name__ == "__main__":
    pytest.main([__file__, "-v"])