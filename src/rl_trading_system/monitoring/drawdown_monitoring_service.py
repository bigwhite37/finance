"""
回撤监控服务模块

实现实时回撤监控、数据收集、指标计算和API接口功能。
"""

import asyncio
import json
import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from threading import Lock, Thread
import numpy as np
import pandas as pd
import sqlite3
from contextlib import contextmanager

# Optional dependencies
try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    jsonify = None
    request = None
    CORS = None

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from ..risk_control.drawdown_monitor import DrawdownMonitor, DrawdownMetrics
from ..risk_control.drawdown_attribution_analyzer import DrawdownAttributionAnalyzer
from ..risk_control.market_regime_detector import MarketRegimeDetector
import logging


@dataclass
class MonitoringMetrics:
    """监控指标数据结构"""
    timestamp: datetime
    portfolio_value: float
    current_drawdown: float
    max_drawdown: float
    drawdown_duration: int
    risk_budget_usage: float
    position_count: int
    concentration_score: float
    volatility: float
    sharpe_ratio: float
    market_regime: str
    alert_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AlertConfig:
    """告警配置"""
    drawdown_warning_threshold: float = 0.08
    drawdown_critical_threshold: float = 0.15
    volatility_threshold: float = 0.25
    concentration_threshold: float = 0.30
    risk_budget_threshold: float = 0.90


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0, max_memory_items: int = 10000):
        self.logger = logging.getLogger(__name__)
        self.max_memory_items = max_memory_items
        self.memory_cache = deque(maxlen=max_memory_items)
        self.cache_lock = Lock()
        
        # Redis缓存（可选）
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, port=redis_port, db=redis_db,
                    decode_responses=True, socket_timeout=1
                )
                self.redis_client.ping()
                self.use_redis = True
                self.logger.info("Redis缓存连接成功")
            except Exception as e:
                self.logger.warning(f"Redis连接失败，使用内存缓存: {e}")
                self.redis_client = None
                self.use_redis = False
        else:
            self.logger.info("Redis不可用，使用内存缓存")
            self.redis_client = None
            self.use_redis = False
    
    def put(self, key: str, data: Dict[str, Any], expire_seconds: int = 3600):
        """存储数据到缓存"""
        with self.cache_lock:
            # 内存缓存
            self.memory_cache.append((key, data, time.time()))
            
            # Redis缓存
            if self.use_redis:
                try:
                    self.redis_client.setex(
                        key, expire_seconds, json.dumps(data, default=str)
                    )
                except Exception as e:
                    self.logger.error(f"Redis存储失败: {e}")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """从缓存获取数据"""
        # 先尝试Redis
        if self.use_redis:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                self.logger.error(f"Redis读取失败: {e}")
        
        # 内存缓存
        with self.cache_lock:
            for cached_key, cached_data, timestamp in reversed(self.memory_cache):
                if cached_key == key:
                    return cached_data
        
        return None
    
    def get_recent_data(self, pattern: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近的数据"""
        results = []
        
        with self.cache_lock:
            for cached_key, cached_data, timestamp in reversed(self.memory_cache):
                if pattern in cached_key and len(results) < limit:
                    results.append(cached_data)
        
        return results


class HistoricalDataManager:
    """历史数据管理器"""
    
    def __init__(self, db_path: str = "monitoring_data.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    portfolio_value REAL NOT NULL,
                    current_drawdown REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    drawdown_duration INTEGER NOT NULL,
                    risk_budget_usage REAL NOT NULL,
                    position_count INTEGER NOT NULL,
                    concentration_score REAL NOT NULL,
                    volatility REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    market_regime TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    raw_data TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON monitoring_metrics(timestamp)
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at DATETIME
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def save_metrics(self, metrics: MonitoringMetrics):
        """保存监控指标"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO monitoring_metrics (
                    timestamp, portfolio_value, current_drawdown, max_drawdown,
                    drawdown_duration, risk_budget_usage, position_count,
                    concentration_score, volatility, sharpe_ratio,
                    market_regime, alert_level, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp, metrics.portfolio_value, metrics.current_drawdown,
                metrics.max_drawdown, metrics.drawdown_duration, metrics.risk_budget_usage,
                metrics.position_count, metrics.concentration_score, metrics.volatility,
                metrics.sharpe_ratio, metrics.market_regime, metrics.alert_level,
                json.dumps(metrics.to_dict())
            ))
    
    def get_historical_metrics(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """获取历史监控指标"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM monitoring_metrics 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (start_time, end_time))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def save_alert(self, alert_type: str, severity: str, message: str):
        """保存告警信息"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO alerts (timestamp, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            """, (datetime.now(), alert_type, severity, message))
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM alerts 
                WHERE resolved = FALSE
                ORDER BY timestamp DESC
            """)
            
            return [dict(row) for row in cursor.fetchall()]


class DrawdownMonitoringService:
    """回撤监控服务"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # 初始化组件
        self.drawdown_monitor = DrawdownMonitor()
        self.attribution_analyzer = DrawdownAttributionAnalyzer()
        self.market_regime_detector = MarketRegimeDetector()
        
        # 初始化数据管理
        self.data_cache = DataCache()
        self.historical_data_manager = HistoricalDataManager()
        self.alert_config = AlertConfig()
        
        # 监控状态
        self.is_running = False
        self.monitoring_thread = None
        self.portfolio_values = deque(maxlen=1000)
        self.last_metrics = None
        
        # Flask应用（可选）
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            CORS(self.app)
            self._setup_routes()
        else:
            self.app = None
            self.logger.warning("Flask不可用，API服务器功能将被禁用")
        
        self.logger.info("回撤监控服务初始化完成")
    
    def start_monitoring(self, update_interval: float = 1.0):
        """启动监控服务"""
        if self.is_running:
            self.logger.warning("监控服务已在运行")
            return
        
        self.is_running = True
        self.monitoring_thread = Thread(
            target=self._monitoring_loop,
            args=(update_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"监控服务已启动，更新间隔: {update_interval}秒")
    
    def stop_monitoring(self):
        """停止监控服务"""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("监控服务已停止")
    
    def _monitoring_loop(self, update_interval: float):
        """监控循环"""
        while self.is_running:
            try:
                # 收集实时数据（这里需要从实际的交易系统获取数据）
                portfolio_data = self._collect_portfolio_data()
                market_data = self._collect_market_data()
                
                if portfolio_data and market_data:
                    # 计算监控指标
                    metrics = self._calculate_monitoring_metrics(portfolio_data, market_data)
                    
                    # 缓存数据
                    cache_key = f"metrics:{int(time.time())}"
                    self.data_cache.put(cache_key, metrics.to_dict())
                    
                    # 保存历史数据
                    self.historical_data_manager.save_metrics(metrics)
                    
                    # 检查告警
                    self._check_alerts(metrics)
                    
                    self.last_metrics = metrics
                
                time.sleep(update_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                time.sleep(update_interval)
    
    def _collect_portfolio_data(self) -> Optional[Dict[str, Any]]:
        """收集投资组合数据"""
        # 这里应该从实际的交易系统获取数据
        # 为了演示，返回模拟数据
        return {
            'portfolio_value': 1000000 + np.random.normal(0, 10000),
            'positions': {
                'AAPL': {'quantity': 100, 'price': 150.0, 'value': 15000},
                'GOOGL': {'quantity': 50, 'price': 2800.0, 'value': 140000},
                'MSFT': {'quantity': 200, 'price': 300.0, 'value': 60000}
            },
            'cash': 50000,
            'timestamp': datetime.now()
        }
    
    def _collect_market_data(self) -> Optional[Dict[str, Any]]:
        """收集市场数据"""
        # 这里应该从实际的市场数据源获取数据
        # 为了演示，返回模拟数据
        return {
            'market_volatility': np.random.uniform(0.15, 0.35),
            'market_trend': np.random.choice(['up', 'down', 'sideways']),
            'correlation_matrix': np.random.rand(3, 3),
            'timestamp': datetime.now()
        }
    
    def _calculate_monitoring_metrics(self, portfolio_data: Dict[str, Any], 
                                    market_data: Dict[str, Any]) -> MonitoringMetrics:
        """计算监控指标"""
        portfolio_value = portfolio_data['portfolio_value']
        self.portfolio_values.append(portfolio_value)
        
        # 计算回撤指标
        if len(self.portfolio_values) > 1:
            values_array = np.array(self.portfolio_values)
            drawdown_metrics = self.drawdown_monitor.calculate_drawdown(values_array)
        else:
            drawdown_metrics = DrawdownMetrics(
                current_drawdown=0.0, max_drawdown=0.0, drawdown_duration=0,
                recovery_time=None, peak_value=portfolio_value, trough_value=portfolio_value,
                underwater_curve=[], drawdown_frequency=0.0, average_drawdown=0.0
            )
        
        # 计算其他指标
        positions = portfolio_data['positions']
        position_values = [pos['value'] for pos in positions.values()]
        total_position_value = sum(position_values)
        
        # 集中度分数
        if total_position_value > 0:
            weights = [value / total_position_value for value in position_values]
            concentration_score = sum(w**2 for w in weights)  # Herfindahl指数
        else:
            concentration_score = 0.0
        
        # 风险预算使用率（模拟）
        risk_budget_usage = min(abs(drawdown_metrics.current_drawdown) / 0.15, 1.0)
        
        # 波动率计算
        if len(self.portfolio_values) > 20:
            returns = np.diff(list(self.portfolio_values)[-21:]) / np.array(list(self.portfolio_values)[-21:-1])
            volatility = np.std(returns) * np.sqrt(252)
        else:
            volatility = market_data.get('market_volatility', 0.2)
        
        # 夏普比率计算
        if len(self.portfolio_values) > 20 and volatility > 0:
            returns = np.diff(list(self.portfolio_values)[-21:]) / np.array(list(self.portfolio_values)[-21:-1])
            mean_return = np.mean(returns) * 252
            sharpe_ratio = mean_return / volatility
        else:
            sharpe_ratio = 0.0
        
        # 市场状态检测
        market_regime = self.market_regime_detector.detect_regime(
            np.array(self.portfolio_values), market_data
        )
        
        # 告警级别
        alert_level = self._determine_alert_level(drawdown_metrics, volatility, concentration_score)
        
        return MonitoringMetrics(
            timestamp=datetime.now(),
            portfolio_value=portfolio_value,
            current_drawdown=drawdown_metrics.current_drawdown,
            max_drawdown=drawdown_metrics.max_drawdown,
            drawdown_duration=drawdown_metrics.drawdown_duration,
            risk_budget_usage=risk_budget_usage,
            position_count=len(positions),
            concentration_score=concentration_score,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            market_regime=market_regime,
            alert_level=alert_level
        )
    
    def _determine_alert_level(self, drawdown_metrics: DrawdownMetrics, 
                             volatility: float, concentration_score: float) -> str:
        """确定告警级别"""
        current_drawdown = abs(drawdown_metrics.current_drawdown)
        
        if (current_drawdown >= self.alert_config.drawdown_critical_threshold or
            volatility >= self.alert_config.volatility_threshold or
            concentration_score >= self.alert_config.concentration_threshold):
            return "CRITICAL"
        elif current_drawdown >= self.alert_config.drawdown_warning_threshold:
            return "WARNING"
        else:
            return "NORMAL"
    
    def _check_alerts(self, metrics: MonitoringMetrics):
        """检查告警条件"""
        alerts = []
        
        # 回撤告警
        if abs(metrics.current_drawdown) >= self.alert_config.drawdown_critical_threshold:
            alerts.append(("DRAWDOWN_CRITICAL", "CRITICAL", 
                          f"当前回撤{metrics.current_drawdown:.2%}超过临界阈值"))
        elif abs(metrics.current_drawdown) >= self.alert_config.drawdown_warning_threshold:
            alerts.append(("DRAWDOWN_WARNING", "WARNING", 
                          f"当前回撤{metrics.current_drawdown:.2%}超过警告阈值"))
        
        # 波动率告警
        if metrics.volatility >= self.alert_config.volatility_threshold:
            alerts.append(("HIGH_VOLATILITY", "WARNING", 
                          f"当前波动率{metrics.volatility:.2%}过高"))
        
        # 集中度告警
        if metrics.concentration_score >= self.alert_config.concentration_threshold:
            alerts.append(("HIGH_CONCENTRATION", "WARNING", 
                          f"投资组合集中度{metrics.concentration_score:.2f}过高"))
        
        # 风险预算告警
        if metrics.risk_budget_usage >= self.alert_config.risk_budget_threshold:
            alerts.append(("RISK_BUDGET_HIGH", "WARNING", 
                          f"风险预算使用率{metrics.risk_budget_usage:.2%}过高"))
        
        # 保存告警
        for alert_type, severity, message in alerts:
            self.historical_data_manager.save_alert(alert_type, severity, message)
            self.logger.warning(f"告警: {message}")
    
    def _setup_routes(self):
        """设置API路由"""
        if not FLASK_AVAILABLE or not self.app:
            return
        
        @self.app.route('/api/metrics/current', methods=['GET'])
        def get_current_metrics():
            """获取当前监控指标"""
            if self.last_metrics:
                return jsonify(self.last_metrics.to_dict())
            else:
                return jsonify({'error': '暂无数据'}), 404
        
        @self.app.route('/api/metrics/history', methods=['GET'])
        def get_historical_metrics():
            """获取历史监控指标"""
            hours = request.args.get('hours', 24, type=int)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            metrics = self.historical_data_manager.get_historical_metrics(start_time, end_time)
            return jsonify(metrics)
        
        @self.app.route('/api/alerts/active', methods=['GET'])
        def get_active_alerts():
            """获取活跃告警"""
            alerts = self.historical_data_manager.get_active_alerts()
            return jsonify(alerts)
        
        @self.app.route('/api/drawdown/curve', methods=['GET'])
        def get_drawdown_curve():
            """获取回撤曲线数据"""
            if len(self.portfolio_values) > 1:
                values = list(self.portfolio_values)
                drawdown_curve = []
                running_max = values[0]
                
                for i, value in enumerate(values):
                    running_max = max(running_max, value)
                    drawdown = (value - running_max) / running_max
                    drawdown_curve.append({
                        'index': i,
                        'portfolio_value': value,
                        'drawdown': drawdown,
                        'running_max': running_max
                    })
                
                return jsonify(drawdown_curve)
            else:
                return jsonify([])
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'is_monitoring': self.is_running,
                'last_update': self.last_metrics.timestamp.isoformat() if self.last_metrics else None
            })
    
    def run_api_server(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """运行API服务器"""
        if not FLASK_AVAILABLE or not self.app:
            self.logger.error("Flask不可用，无法启动API服务器")
            return
        
        self.logger.info(f"启动API服务器: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    # 示例用法
    service = DrawdownMonitoringService()
    
    # 启动监控
    service.start_monitoring(update_interval=2.0)
    
    try:
        # 运行API服务器
        service.run_api_server(port=5000, debug=True)
    except KeyboardInterrupt:
        service.stop_monitoring()
        print("监控服务已停止")