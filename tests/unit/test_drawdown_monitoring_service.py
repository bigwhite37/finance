"""
回撤监控服务测试模块

测试DrawdownMonitoringService的功能、性能和可靠性。
"""

import pytest
import time
import threading
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.rl_trading_system.monitoring.drawdown_monitoring_service import (
    DrawdownMonitoringService,
    MonitoringMetrics,
    DataCache,
    HistoricalDataManager,
    AlertConfig
)


class TestMonitoringMetrics:
    """监控指标测试"""
    
    def test_monitoring_metrics_creation(self):
        """测试监控指标创建"""
        metrics = MonitoringMetrics(
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
        
        assert metrics.portfolio_value == 1000000.0
        assert metrics.current_drawdown == -0.05
        assert metrics.alert_level == "NORMAL"
    
    def test_metrics_to_dict(self):
        """测试指标转换为字典"""
        timestamp = datetime.now()
        metrics = MonitoringMetrics(
            timestamp=timestamp,
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
        
        data = metrics.to_dict()
        assert data['portfolio_value'] == 1000000.0
        assert data['timestamp'] == timestamp.isoformat()
        assert isinstance(data, dict)


class TestDataCache:
    """数据缓存测试"""
    
    def test_memory_cache_operations(self):
        """测试内存缓存操作"""
        cache = DataCache(max_memory_items=100)
        
        # 测试存储和获取
        test_data = {'value': 123, 'timestamp': time.time()}
        cache.put('test_key', test_data)
        
        retrieved_data = cache.get('test_key')
        assert retrieved_data == test_data
    
    def test_cache_size_limit(self):
        """测试缓存大小限制"""
        cache = DataCache(max_memory_items=3)
        
        # 添加超过限制的数据
        for i in range(5):
            cache.put(f'key_{i}', {'value': i})
        
        # 检查只保留最新的3个
        assert len(cache.memory_cache) == 3
        assert cache.get('key_0') is None  # 最早的应该被移除
        assert cache.get('key_4') is not None  # 最新的应该存在
    
    def test_get_recent_data(self):
        """测试获取最近数据"""
        cache = DataCache(max_memory_items=100)
        
        # 添加测试数据
        for i in range(10):
            cache.put(f'metrics_{i}', {'value': i, 'timestamp': time.time()})
        
        recent_data = cache.get_recent_data('metrics', limit=5)
        assert len(recent_data) == 5
        # 应该按时间倒序返回
        assert recent_data[0]['value'] == 9


class TestHistoricalDataManager:
    """历史数据管理器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.manager = HistoricalDataManager(self.db_path)
    
    def teardown_method(self):
        """测试清理"""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_database_initialization(self):
        """测试数据库初始化"""
        # 数据库应该已经创建并包含必要的表
        with self.manager._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            assert 'monitoring_metrics' in tables
            assert 'alerts' in tables
    
    def test_save_and_retrieve_metrics(self):
        """测试保存和检索指标"""
        metrics = MonitoringMetrics(
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
        
        # 保存指标
        self.manager.save_metrics(metrics)
        
        # 检索指标
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        retrieved_metrics = self.manager.get_historical_metrics(start_time, end_time)
        
        assert len(retrieved_metrics) == 1
        assert retrieved_metrics[0]['portfolio_value'] == 1000000.0
        assert retrieved_metrics[0]['current_drawdown'] == -0.05
    
    def test_save_and_retrieve_alerts(self):
        """测试保存和检索告警"""
        # 保存告警
        self.manager.save_alert("DRAWDOWN_WARNING", "WARNING", "测试告警")
        
        # 检索活跃告警
        active_alerts = self.manager.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0]['alert_type'] == "DRAWDOWN_WARNING"
        assert active_alerts[0]['severity'] == "WARNING"
        assert active_alerts[0]['message'] == "测试告警"


class TestDrawdownMonitoringService:
    """回撤监控服务测试"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # 创建服务实例
        config = {'db_path': self.temp_db.name}
        self.service = DrawdownMonitoringService(config)
        
        # Mock外部依赖
        self.service.drawdown_monitor = Mock()
        self.service.attribution_analyzer = Mock()
        self.service.market_regime_detector = Mock()
    
    def teardown_method(self):
        """测试清理"""
        if hasattr(self, 'service'):
            self.service.stop_monitoring()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_service_initialization(self):
        """测试服务初始化"""
        assert self.service.drawdown_monitor is not None
        assert self.service.data_cache is not None
        assert self.service.historical_data_manager is not None
        assert not self.service.is_running
    
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        # 启动监控
        self.service.start_monitoring(update_interval=0.1)
        assert self.service.is_running
        assert self.service.monitoring_thread is not None
        
        # 等待一小段时间确保线程启动
        time.sleep(0.2)
        
        # 停止监控
        self.service.stop_monitoring()
        assert not self.service.is_running
    
    @patch('src.rl_trading_system.monitoring.drawdown_monitoring_service.DrawdownMonitoringService._collect_portfolio_data')
    @patch('src.rl_trading_system.monitoring.drawdown_monitoring_service.DrawdownMonitoringService._collect_market_data')
    def test_monitoring_loop_execution(self, mock_market_data, mock_portfolio_data):
        """测试监控循环执行"""
        # Mock数据收集方法
        mock_portfolio_data.return_value = {
            'portfolio_value': 1000000.0,
            'positions': {
                'AAPL': {'quantity': 100, 'price': 150.0, 'value': 15000}
            },
            'cash': 50000,
            'timestamp': datetime.now()
        }
        
        mock_market_data.return_value = {
            'market_volatility': 0.2,
            'market_trend': 'up',
            'correlation_matrix': np.eye(3),
            'timestamp': datetime.now()
        }
        
        # Mock drawdown monitor
        from src.rl_trading_system.risk_control.drawdown_monitor import DrawdownMetrics
        mock_drawdown_metrics = DrawdownMetrics(
            current_drawdown=-0.02,
            max_drawdown=-0.05,
            drawdown_duration=2,
            recovery_time=None,
            peak_value=1000000.0,
            trough_value=980000.0,
            underwater_curve=[],
            drawdown_frequency=0.1,
            average_drawdown=-0.03
        )
        self.service.drawdown_monitor.calculate_drawdown.return_value = mock_drawdown_metrics
        self.service.market_regime_detector.detect_regime.return_value = "bull"
        
        # 启动监控
        self.service.start_monitoring(update_interval=0.1)
        
        # 等待几个监控周期
        time.sleep(0.5)
        
        # 检查是否有数据被收集
        assert len(self.service.portfolio_values) > 0
        assert self.service.last_metrics is not None
        
        # 停止监控
        self.service.stop_monitoring()
    
    def test_alert_level_determination(self):
        """测试告警级别确定"""
        from src.rl_trading_system.risk_control.drawdown_monitor import DrawdownMetrics
        
        # 正常情况
        normal_drawdown = DrawdownMetrics(
            current_drawdown=-0.02, max_drawdown=-0.02, drawdown_duration=1,
            recovery_time=None, peak_value=1000000, trough_value=980000,
            underwater_curve=[], drawdown_frequency=0.1, average_drawdown=-0.02
        )
        alert_level = self.service._determine_alert_level(normal_drawdown, 0.15, 0.2)
        assert alert_level == "NORMAL"
        
        # 警告情况
        warning_drawdown = DrawdownMetrics(
            current_drawdown=-0.10, max_drawdown=-0.10, drawdown_duration=5,
            recovery_time=None, peak_value=1000000, trough_value=900000,
            underwater_curve=[], drawdown_frequency=0.2, average_drawdown=-0.05
        )
        alert_level = self.service._determine_alert_level(warning_drawdown, 0.15, 0.2)
        assert alert_level == "WARNING"
        
        # 严重情况
        critical_drawdown = DrawdownMetrics(
            current_drawdown=-0.20, max_drawdown=-0.20, drawdown_duration=10,
            recovery_time=None, peak_value=1000000, trough_value=800000,
            underwater_curve=[], drawdown_frequency=0.3, average_drawdown=-0.10
        )
        alert_level = self.service._determine_alert_level(critical_drawdown, 0.15, 0.2)
        assert alert_level == "CRITICAL"
    
    def test_api_routes(self):
        """测试API路由"""
        # 检查Flask是否可用
        from src.rl_trading_system.monitoring.drawdown_monitoring_service import FLASK_AVAILABLE
        if not FLASK_AVAILABLE or not self.service.app:
            pytest.skip("Flask不可用，跳过API测试")
        
        with self.service.app.test_client() as client:
            # 测试健康检查
            response = client.get('/api/health')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'status' in data
            assert data['status'] == 'healthy'
            
            # 测试获取当前指标（无数据时）
            response = client.get('/api/metrics/current')
            assert response.status_code == 404
            
            # 测试获取历史指标
            response = client.get('/api/metrics/history')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)
            
            # 测试获取活跃告警
            response = client.get('/api/alerts/active')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)
            
            # 测试获取回撤曲线
            response = client.get('/api/drawdown/curve')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)


class TestPerformanceAndReliability:
    """性能和可靠性测试"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        config = {'db_path': self.temp_db.name}
        self.service = DrawdownMonitoringService(config)
    
    def teardown_method(self):
        """测试清理"""
        if hasattr(self, 'service'):
            self.service.stop_monitoring()
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_concurrent_data_access(self):
        """测试并发数据访问"""
        cache = DataCache(max_memory_items=1000)
        
        def write_data(thread_id):
            for i in range(100):
                cache.put(f'thread_{thread_id}_key_{i}', {'value': i, 'thread': thread_id})
        
        def read_data(thread_id):
            results = []
            for i in range(100):
                data = cache.get(f'thread_{thread_id}_key_{i}')
                if data:
                    results.append(data)
            return results
        
        # 创建多个写入线程
        write_threads = []
        for i in range(5):
            thread = threading.Thread(target=write_data, args=(i,))
            write_threads.append(thread)
            thread.start()
        
        # 等待写入完成
        for thread in write_threads:
            thread.join()
        
        # 验证数据完整性
        for thread_id in range(5):
            for i in range(100):
                data = cache.get(f'thread_{thread_id}_key_{i}')
                assert data is not None
                assert data['thread'] == thread_id
    
    def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        cache = DataCache(max_memory_items=1000)
        
        # 添加大量数据
        for i in range(2000):
            large_data = {'data': [j for j in range(100)], 'index': i}
            cache.put(f'large_key_{i}', large_data)
        
        # 验证缓存大小限制
        assert len(cache.memory_cache) <= 1000
        
        # 验证最新数据仍然可访问
        recent_data = cache.get('large_key_1999')
        assert recent_data is not None
        assert recent_data['index'] == 1999
    
    def test_database_performance(self):
        """测试数据库性能"""
        manager = HistoricalDataManager(self.temp_db.name)
        
        # 批量插入测试
        start_time = time.time()
        for i in range(1000):
            metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=1000000.0 + i,
                current_drawdown=-0.01 * (i % 10),
                max_drawdown=-0.05,
                drawdown_duration=i % 20,
                risk_budget_usage=0.5,
                position_count=10,
                concentration_score=0.2,
                volatility=0.15,
                sharpe_ratio=1.2,
                market_regime="bull",
                alert_level="NORMAL"
            )
            manager.save_metrics(metrics)
        
        insert_time = time.time() - start_time
        assert insert_time < 10.0  # 应该在10秒内完成1000次插入
        
        # 查询性能测试
        start_time = time.time()
        end_time = datetime.now()
        start_query_time = end_time - timedelta(hours=1)
        
        results = manager.get_historical_metrics(start_query_time, end_time)
        query_time = time.time() - start_time
        
        assert query_time < 1.0  # 查询应该在1秒内完成
        assert len(results) > 0
    
    def test_service_reliability_under_errors(self):
        """测试错误情况下的服务可靠性"""
        # Mock数据收集方法抛出异常
        with patch.object(self.service, '_collect_portfolio_data', side_effect=Exception("数据收集失败")):
            with patch.object(self.service, '_collect_market_data', return_value={}):
                
                # 启动监控
                self.service.start_monitoring(update_interval=0.1)
                
                # 等待几个监控周期
                time.sleep(0.5)
                
                # 服务应该仍在运行，不会因为异常而崩溃
                assert self.service.is_running
                
                # 停止监控
                self.service.stop_monitoring()
    
    def test_api_response_time(self):
        """测试API响应时间"""
        # 检查Flask是否可用
        from src.rl_trading_system.monitoring.drawdown_monitoring_service import FLASK_AVAILABLE
        if not FLASK_AVAILABLE or not self.service.app:
            pytest.skip("Flask不可用，跳过API性能测试")
        
        # 添加一些测试数据
        for i in range(100):
            metrics = MonitoringMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                portfolio_value=1000000.0,
                current_drawdown=-0.01,
                max_drawdown=-0.05,
                drawdown_duration=1,
                risk_budget_usage=0.5,
                position_count=10,
                concentration_score=0.2,
                volatility=0.15,
                sharpe_ratio=1.2,
                market_regime="bull",
                alert_level="NORMAL"
            )
            self.service.historical_data_manager.save_metrics(metrics)
        
        with self.service.app.test_client() as client:
            # 测试历史数据API响应时间
            start_time = time.time()
            response = client.get('/api/metrics/history?hours=24')
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 1.0  # 响应时间应该小于1秒
            
            data = json.loads(response.data)
            assert len(data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])