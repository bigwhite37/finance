"""
告警系统模块
实现DynamicThresholdManager类和阈值计算，多级别告警和告警规则配置，多渠道通知和告警聚合、静默和日志记录
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import json
import smtplib
import requests
import time
import threading
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from scipy import stats


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """告警渠道枚举"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class AlertRule:
    """告警规则类"""
    rule_id: str
    metric_name: str
    threshold_value: float
    comparison_operator: str
    alert_level: AlertLevel
    description: str = ""
    is_active: bool = True
    
    def __post_init__(self):
        """初始化后验证"""
        if not self.rule_id:
            raise ValueError("规则ID不能为空")
        
        valid_operators = [">", ">=", "<", "<=", "==", "!="]
        if self.comparison_operator not in valid_operators:
            raise ValueError(f"不支持的比较操作符: {self.comparison_operator}")
    
    def evaluate(self, value: float) -> bool:
        """评估规则是否触发"""
        if not self.is_active:
            return False
        
        if self.comparison_operator == ">":
            return value > self.threshold_value
        elif self.comparison_operator == ">=":
            return value >= self.threshold_value
        elif self.comparison_operator == "<":
            return value < self.threshold_value
        elif self.comparison_operator == "<=":
            return value <= self.threshold_value
        elif self.comparison_operator == "==":
            return abs(value - self.threshold_value) < 1e-10
        elif self.comparison_operator == "!=":
            return abs(value - self.threshold_value) >= 1e-10
        
        return False
    
    def activate(self):
        """激活规则"""
        self.is_active = True
    
    def deactivate(self):
        """停用规则"""
        self.is_active = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rule_id': self.rule_id,
            'metric_name': self.metric_name,
            'threshold_value': self.threshold_value,
            'comparison_operator': self.comparison_operator,
            'alert_level': self.alert_level.value,
            'description': self.description,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """从字典创建"""
        return cls(
            rule_id=data['rule_id'],
            metric_name=data['metric_name'],
            threshold_value=data['threshold_value'],
            comparison_operator=data['comparison_operator'],
            alert_level=AlertLevel(data['alert_level']),
            description=data.get('description', ''),
            is_active=data.get('is_active', True)
        )


class DynamicThresholdManager:
    """动态阈值管理器"""
    
    def __init__(self, historical_data: pd.DataFrame, lookback_window: int = 60, 
                 update_frequency: str = 'daily'):
        """
        初始化动态阈值管理器
        
        Args:
            historical_data: 历史数据
            lookback_window: 回看窗口大小
            update_frequency: 更新频率
        """
        if historical_data.empty:
            raise ValueError("历史数据不能为空")
        
        if lookback_window <= 0:
            raise ValueError("回看窗口大小必须为正数")
        
        self.historical_data = historical_data.copy()
        self.lookback_window = lookback_window
        self.update_frequency = update_frequency
        self.thresholds = {}
        self._lock = threading.Lock()
    
    def calculate_percentile_threshold(self, metric_name: str, percentile: float, 
                                     threshold_type: str = 'upper') -> float:
        """计算基于分位数的阈值"""
        if metric_name not in self.historical_data.columns:
            raise ValueError(f"指标 {metric_name} 不存在于历史数据中")
        
        if not (0 < percentile < 100):
            raise ValueError("分位数必须在0-100之间")
        
        data = self.historical_data[metric_name]
        
        if threshold_type.lower() == 'upper':
            threshold = data.quantile(percentile / 100.0)
        else:  # lower
            threshold = data.quantile(percentile / 100.0)
        
        return float(threshold)
    
    def calculate_rolling_threshold(self, metric_name: str, window_size: int, 
                                  percentile: float) -> pd.Series:
        """计算滚动窗口阈值"""
        if metric_name not in self.historical_data.columns:
            raise ValueError(f"指标 {metric_name} 不存在于历史数据中")
        
        data = self.historical_data[metric_name]
        rolling_thresholds = data.rolling(window=window_size).quantile(percentile / 100.0)
        
        return rolling_thresholds
    
    def update_threshold_with_new_data(self, metric_name: str, new_data: pd.DataFrame, 
                                     adaptation_factor: float = 0.1) -> float:
        """使用新数据更新阈值"""
        if metric_name not in new_data.columns:
            raise ValueError(f"新数据中不包含指标 {metric_name}")
        
        with self._lock:
            # 添加新数据到历史数据
            self.historical_data = pd.concat([self.historical_data, new_data], ignore_index=True)
            
            # 保持数据量在合理范围内
            if len(self.historical_data) > self.lookback_window * 5:
                self.historical_data = self.historical_data.tail(self.lookback_window * 3)
            
            # 重新计算阈值
            new_threshold = self.calculate_percentile_threshold(metric_name, 90, 'upper')
            
            return new_threshold
    
    def calculate_multiple_thresholds(self, threshold_configs: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """批量计算多个指标的阈值"""
        thresholds = {}
        
        for metric_name, config in threshold_configs.items():
            if metric_name in self.historical_data.columns:
                threshold = self.calculate_percentile_threshold(
                    metric_name=metric_name,
                    percentile=config['percentile'],
                    threshold_type=config['type']
                )
                thresholds[metric_name] = threshold
        
        return thresholds
    
    def validate_threshold_config(self, config: Dict[str, Any]) -> bool:
        """验证阈值配置有效性"""
        required_fields = ['metric_name', 'percentile', 'threshold_type']
        
        for field in required_fields:
            if field not in config:
                return False
        
        # 检查指标是否存在
        if config['metric_name'] not in self.historical_data.columns:
            return False
        
        # 检查分位数范围
        percentile = config['percentile']
        if not (0 < percentile < 100):
            return False
        
        # 检查阈值类型
        if config['threshold_type'] not in ['upper', 'lower']:
            return False
        
        return True
    
    def save_thresholds(self, file_path: str):
        """保存阈值到文件"""
        with self._lock:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.thresholds, f, indent=2, ensure_ascii=False)
    
    def load_thresholds(self, file_path: str):
        """从文件加载阈值"""
        with self._lock:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.thresholds = json.load(f)
    
    def is_statistical_outlier(self, metric_name: str, value: float, 
                             method: str = 'zscore', threshold: float = 3.0) -> bool:
        """检测统计异常值"""
        if metric_name not in self.historical_data.columns:
            raise ValueError(f"指标 {metric_name} 不存在")
        
        data = self.historical_data[metric_name]
        
        if method == 'zscore':
            # 手动计算z-score
            mean = data.mean()
            std = data.std()
            if std == 0:
                return False
            z_score = abs((value - mean) / std)
            return z_score > threshold
        elif method == 'iqr':
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return value < lower_bound or value > upper_bound
        
        return False
    
    def analyze_threshold_sensitivity(self, metric_name: str, 
                                    percentiles: List[float]) -> Dict[float, float]:
        """分析阈值敏感性"""
        sensitivity_results = {}
        
        for percentile in percentiles:
            threshold = self.calculate_percentile_threshold(
                metric_name=metric_name,
                percentile=percentile,
                threshold_type='upper'
            )
            sensitivity_results[percentile] = threshold
        
        return sensitivity_results


class AlertAggregator:
    """告警聚合器"""
    
    def __init__(self, aggregation_window: int = 300, max_alerts_per_rule: int = 5,
                 similarity_threshold: float = 0.8):
        """
        初始化告警聚合器
        
        Args:
            aggregation_window: 聚合时间窗口（秒）
            max_alerts_per_rule: 每个规则最大告警数
            similarity_threshold: 相似度阈值
        """
        self.aggregation_window = aggregation_window
        self.max_alerts_per_rule = max_alerts_per_rule
        self.similarity_threshold = similarity_threshold
        self.pending_alerts = []
        self._lock = threading.Lock()
    
    def add_alert(self, alert: Dict[str, Any]):
        """添加告警到聚合队列"""
        with self._lock:
            alert['timestamp'] = alert.get('timestamp', datetime.now())
            self.pending_alerts.append(alert)
    
    def aggregate_alerts(self) -> List[Dict[str, Any]]:
        """执行告警聚合"""
        with self._lock:
            current_time = datetime.now()
            
            # 清理过期告警
            self.pending_alerts = [
                alert for alert in self.pending_alerts
                if abs((current_time - alert['timestamp']).total_seconds()) < self.aggregation_window
            ]
            
            # 按规则分组
            alerts_by_rule = defaultdict(list)
            for alert in self.pending_alerts:
                alerts_by_rule[alert['rule_id']].append(alert)
            
            aggregated_alerts = []
            
            for rule_id, rule_alerts in alerts_by_rule.items():
                # 应用频率限制
                if len(rule_alerts) > self.max_alerts_per_rule:
                    rule_alerts = rule_alerts[:self.max_alerts_per_rule]
                
                # 聚合相似告警
                if len(rule_alerts) > 1:
                    # 检查是否可以聚合
                    similar_alerts = self._group_similar_alerts(rule_alerts)
                    for group in similar_alerts:
                        if len(group) > 1:
                            # 创建聚合告警
                            aggregated_alert = group[0].copy()
                            aggregated_alert['count'] = len(group)
                            aggregated_alert['message'] += f" (聚合了{len(group)}条相似告警)"
                            aggregated_alerts.append(aggregated_alert)
                        else:
                            aggregated_alerts.extend(group)
                else:
                    aggregated_alerts.extend(rule_alerts)
            
            # 清空已处理的告警
            self.pending_alerts.clear()
            
            return aggregated_alerts
    
    def calculate_similarity(self, alert1: Dict[str, Any], alert2: Dict[str, Any]) -> float:
        """计算两个告警的相似度"""
        similarity_score = 0.0
        total_weight = 0.0
        
        # 规则ID相似度权重: 40%
        if alert1.get('rule_id') == alert2.get('rule_id'):
            similarity_score += 0.4
        total_weight += 0.4
        
        # 指标名称相似度权重: 30%
        if alert1.get('metric_name') == alert2.get('metric_name'):
            similarity_score += 0.3
        total_weight += 0.3
        
        # 告警级别相似度权重: 20%
        if alert1.get('level') == alert2.get('level'):
            similarity_score += 0.2
        total_weight += 0.2
        
        # 消息相似度权重: 10%
        if alert1.get('message') == alert2.get('message'):
            similarity_score += 0.1
        total_weight += 0.1
        
        return similarity_score / total_weight if total_weight > 0 else 0.0
    
    def _group_similar_alerts(self, alerts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """将相似的告警分组"""
        groups = []
        processed = set()
        
        for i, alert1 in enumerate(alerts):
            if i in processed:
                continue
            
            group = [alert1]
            processed.add(i)
            
            for j, alert2 in enumerate(alerts):
                if j <= i or j in processed:
                    continue
                
                similarity = self.calculate_similarity(alert1, alert2)
                if similarity >= self.similarity_threshold:
                    group.append(alert2)
                    processed.add(j)
            
            groups.append(group)
        
        return groups


class NotificationManager:
    """通知管理器"""
    
    def __init__(self, notification_config: Dict[str, Dict[str, Any]]):
        """
        初始化通知管理器
        
        Args:
            notification_config: 通知配置
        """
        self.channels = notification_config
        self.rate_limits = {}
        self.notification_history = defaultdict(deque)
        self._lock = threading.Lock()
    
    def send_email_notification(self, alert_data: Dict[str, Any], 
                              max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """发送邮件通知"""
        if not self.channels.get('email', {}).get('enabled', False):
            return False
        
        email_config = self.channels['email']
        
        # 检查频率限制
        if not self._check_rate_limit('email'):
            return False
        
        # 格式化邮件内容
        content = self.format_email_content(alert_data)
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = email_config['username']
        msg['Subject'] = content['subject']
        
        body = MIMEText(content['body'], 'html', 'utf-8')
        msg.attach(body)
        
        # 发送邮件
        for attempt in range(max_retries):
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            
            for recipient in email_config['recipients']:
                msg['To'] = recipient
                server.send_message(msg)
                del msg['To']
            
            server.quit()
            
            # 记录发送历史
            self._record_notification('email')
            return True
        
        return False
    
    def send_webhook_notification(self, alert_data: Dict[str, Any],
                                max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """发送Webhook通知"""
        if not self.channels.get('webhook', {}).get('enabled', False):
            return False
        
        webhook_config = self.channels['webhook']
        
        # 检查频率限制
        if not self._check_rate_limit('webhook'):
            return False
        
        # 格式化Webhook内容
        content = self.format_webhook_content(alert_data)
        
        # 发送Webhook
        for attempt in range(max_retries):
            response = requests.post(
                webhook_config['url'],
                json=content,
                timeout=webhook_config.get('timeout', 10)
            )
            
            if response.status_code == 200:
                self._record_notification('webhook')
                return True
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        
        return False
    
    def send_notification(self, alert_data: Dict[str, Any]) -> bool:
        """发送通知到所有激活的渠道"""
        success = False
        
        for channel_name in self.get_active_channels():
            if channel_name == 'email':
                if self.send_email_notification(alert_data):
                    success = True
            elif channel_name == 'webhook':
                if self.send_webhook_notification(alert_data):
                    success = True
        
        return success
    
    def format_email_content(self, alert_data: Dict[str, Any]) -> Dict[str, str]:
        """格式化邮件内容"""
        subject = f"[{alert_data.get('level', AlertLevel.INFO).value.upper()}] 交易系统告警 - {alert_data.get('rule_id', '')}"
        
        body = f"""
        <html>
        <body>
        <h2>交易系统告警通知</h2>
        <p><strong>规则ID:</strong> {alert_data.get('rule_id', 'N/A')}</p>
        <p><strong>指标名称:</strong> {alert_data.get('metric_name', 'N/A')}</p>
        <p><strong>当前值:</strong> {alert_data.get('value', 'N/A')}</p>
        <p><strong>阈值:</strong> {alert_data.get('threshold', 'N/A')}</p>
        <p><strong>告警级别:</strong> {alert_data.get('level', AlertLevel.INFO).value}</p>
        <p><strong>告警消息:</strong> {alert_data.get('message', '')}</p>
        <p><strong>时间:</strong> {alert_data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body>
        </html>
        """
        
        return {'subject': subject, 'body': body}
    
    def format_webhook_content(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化Webhook内容"""
        return {
            'text': f"🚨 交易系统告警",
            'attachments': [
                {
                    'color': self._get_color_for_level(alert_data.get('level', AlertLevel.INFO)),
                    'fields': [
                        {'title': '规则ID', 'value': alert_data.get('rule_id', 'N/A'), 'short': True},
                        {'title': '指标', 'value': alert_data.get('metric_name', 'N/A'), 'short': True},
                        {'title': '当前值', 'value': str(alert_data.get('value', 'N/A')), 'short': True},
                        {'title': '阈值', 'value': str(alert_data.get('threshold', 'N/A')), 'short': True},
                        {'title': '消息', 'value': alert_data.get('message', ''), 'short': False}
                    ],
                    'timestamp': alert_data.get('timestamp', datetime.now()).isoformat()
                }
            ]
        }
    
    def _get_color_for_level(self, level: AlertLevel) -> str:
        """获取告警级别对应的颜色"""
        color_map = {
            AlertLevel.INFO: 'good',
            AlertLevel.WARNING: 'warning',
            AlertLevel.ERROR: 'danger',
            AlertLevel.CRITICAL: 'danger'
        }
        return color_map.get(level, 'good')
    
    def enable_channel(self, channel_name: str):
        """启用通知渠道"""
        if channel_name in self.channels:
            self.channels[channel_name]['enabled'] = True
    
    def disable_channel(self, channel_name: str):
        """禁用通知渠道"""
        if channel_name in self.channels:
            self.channels[channel_name]['enabled'] = False
    
    def get_active_channels(self) -> List[str]:
        """获取活跃的通知渠道"""
        return [
            name for name, config in self.channels.items()
            if config.get('enabled', False)
        ]
    
    def set_rate_limit(self, channel: str, max_notifications: int, time_window: int):
        """设置频率限制"""
        self.rate_limits[channel] = {
            'max_notifications': max_notifications,
            'time_window': time_window
        }
    
    def _check_rate_limit(self, channel: str) -> bool:
        """检查频率限制"""
        if channel not in self.rate_limits:
            return True
        
        limit_config = self.rate_limits[channel]
        current_time = datetime.now()
        
        with self._lock:
            # 清理过期记录
            cutoff_time = current_time - timedelta(seconds=limit_config['time_window'])
            history = self.notification_history[channel]
            
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # 检查是否超过限制
            if len(history) >= limit_config['max_notifications']:
                return False
            
            return True
    
    def _record_notification(self, channel: str):
        """记录通知发送"""
        with self._lock:
            self.notification_history[channel].append(datetime.now())


class AlertLogger:
    """告警日志记录器"""
    
    def __init__(self, log_file: str = None):
        """初始化告警日志记录器"""
        self.log_file = log_file
        self.logs = []
        self._lock = threading.Lock()
    
    def log_alert(self, alert: Dict[str, Any]):
        """记录告警"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'rule_id': alert.get('rule_id'),
            'metric_name': alert.get('metric_name'),
            'value': alert.get('value'),
            'threshold': alert.get('threshold'),
            'level': alert.get('level', AlertLevel.INFO).value,
            'message': alert.get('message'),
            'status': 'triggered'
        }
        
        with self._lock:
            self.logs.append(log_entry)
            
            if self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def get_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警日志"""
        with self._lock:
            return self.logs[-limit:] if limit < len(self.logs) else self.logs.copy()


class AlertSystem:
    """完整的告警系统"""
    
    def __init__(self, historical_data: pd.DataFrame, 
                 notification_config: Dict[str, Dict[str, Any]]):
        """
        初始化告警系统
        
        Args:
            historical_data: 历史数据
            notification_config: 通知配置
        """
        self.threshold_manager = DynamicThresholdManager(historical_data)
        self.notification_manager = NotificationManager(notification_config)
        self.aggregator = AlertAggregator()
        self.logger = AlertLogger()
        
        self.rules = {}
        self.silenced_rules = {}
        self._lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        with self._lock:
            self.rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str):
        """删除告警规则"""
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """获取告警规则"""
        return self.rules.get(rule_id)
    
    def check_metrics(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """检查指标并生成告警"""
        alerts = []
        
        with self._lock:
            for rule_id, rule in self.rules.items():
                if not rule.is_active:
                    continue
                
                if rule_id in self.silenced_rules:
                    silence_end = self.silenced_rules[rule_id]
                    if datetime.now() < silence_end:
                        continue
                    else:
                        # 静默期结束，删除静默记录
                        del self.silenced_rules[rule_id]
                
                if rule.metric_name in metrics:
                    value = metrics[rule.metric_name]
                    
                    if rule.evaluate(value):
                        alert = {
                            'rule_id': rule_id,
                            'metric_name': rule.metric_name,
                            'value': value,
                            'threshold': rule.threshold_value,
                            'level': rule.alert_level,
                            'message': rule.description or f"{rule.metric_name} 触发告警阈值",
                            'timestamp': datetime.now()
                        }
                        alerts.append(alert)
        
        return alerts
    
    def process_metrics(self, metrics: Dict[str, float]):
        """处理指标（检查、聚合、通知）"""
        # 检查告警
        alerts = self.check_metrics(metrics)
        
        # 添加到聚合器
        for alert in alerts:
            self.aggregator.add_alert(alert)
        
        # 执行聚合
        aggregated_alerts = self.aggregator.aggregate_alerts()
        
        # 发送通知和记录日志
        for alert in aggregated_alerts:
            self.notification_manager.send_notification(alert)
            self.logger.log_alert(alert)
    
    def silence_rule(self, rule_id: str, duration: int):
        """静默指定规则"""
        with self._lock:
            silence_end = datetime.now() + timedelta(seconds=duration)
            self.silenced_rules[rule_id] = silence_end