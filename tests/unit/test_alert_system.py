"""
告警系统的单元测试
测试基于历史分位数的动态阈值计算，阈值调整的合理性和告警准确性，告警规则配置和管理功能
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, MagicMock
import json
import threading
import time

from src.rl_trading_system.monitoring.alert_system import (
    DynamicThresholdManager,
    AlertRule,
    AlertLevel,
    AlertChannel,
    AlertAggregator,
    AlertLogger,
    AlertSystem,
    NotificationManager
)


class TestDynamicThresholdManager:
    """动态阈值管理器测试类"""

    @pytest.fixture
    def sample_historical_data(self):
        """创建样本历史数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # 创建不同类型的指标数据
        portfolio_values = np.random.normal(1000000, 50000, 252)
        daily_returns = np.random.normal(0.001, 0.02, 252)
        volatility = np.random.normal(0.15, 0.03, 252)
        max_drawdown = np.random.exponential(0.02, 252)  # 总是正值
        
        return pd.DataFrame({
            'portfolio_value': portfolio_values,
            'daily_return': daily_returns,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'timestamp': dates
        })

    @pytest.fixture
    def threshold_manager(self, sample_historical_data):
        """创建动态阈值管理器"""
        return DynamicThresholdManager(
            historical_data=sample_historical_data,
            lookback_window=60,
            update_frequency='daily'
        )

    def test_threshold_manager_initialization(self, threshold_manager, sample_historical_data):
        """测试阈值管理器初始化"""
        assert threshold_manager.historical_data is not None
        assert len(threshold_manager.historical_data) == len(sample_historical_data)
        assert threshold_manager.lookback_window == 60
        assert threshold_manager.update_frequency == 'daily'
        assert isinstance(threshold_manager.thresholds, dict)

    def test_percentile_threshold_calculation(self, threshold_manager):
        """测试基于分位数的阈值计算"""
        # 计算不同分位数的阈值
        metric_name = 'daily_return'
        
        # 95%分位数 - 上限
        upper_threshold = threshold_manager.calculate_percentile_threshold(
            metric_name=metric_name,
            percentile=95,
            threshold_type='upper'
        )
        
        # 5%分位数 - 下限
        lower_threshold = threshold_manager.calculate_percentile_threshold(
            metric_name=metric_name,
            percentile=5,
            threshold_type='lower'
        )
        
        # 验证阈值合理性
        assert isinstance(upper_threshold, float)
        assert isinstance(lower_threshold, float)
        assert upper_threshold > lower_threshold
        
        # 验证与实际分位数的一致性
        actual_data = threshold_manager.historical_data[metric_name]
        expected_upper = actual_data.quantile(0.95)
        expected_lower = actual_data.quantile(0.05)
        
        assert abs(upper_threshold - expected_upper) < 1e-10
        assert abs(lower_threshold - expected_lower) < 1e-10

    def test_rolling_threshold_calculation(self, threshold_manager):
        """测试滚动窗口阈值计算"""
        metric_name = 'volatility'
        window_size = 30
        
        rolling_thresholds = threshold_manager.calculate_rolling_threshold(
            metric_name=metric_name,
            window_size=window_size,
            percentile=90
        )
        
        # 验证滚动阈值结果
        assert isinstance(rolling_thresholds, pd.Series)
        assert len(rolling_thresholds) == len(threshold_manager.historical_data)
        
        # 前几个值应该是NaN（窗口不够）
        assert rolling_thresholds.iloc[:window_size-1].isna().all()
        
        # 后面的值应该是有效的
        valid_thresholds = rolling_thresholds.iloc[window_size-1:]
        assert valid_thresholds.notna().all()
        assert (valid_thresholds > 0).all()  # 波动率应该为正

    def test_adaptive_threshold_adjustment(self, threshold_manager):
        """测试自适应阈值调整"""
        metric_name = 'max_drawdown'
        
        # 初始阈值
        initial_threshold = threshold_manager.calculate_percentile_threshold(
            metric_name=metric_name,
            percentile=90,
            threshold_type='upper'
        )
        
        # 模拟新数据点（异常高的回撤）
        new_data = pd.DataFrame({
            'max_drawdown': [0.15, 0.12, 0.18, 0.10, 0.08],
            'timestamp': pd.date_range('2023-09-10', periods=5, freq='D')
        })
        
        # 更新阈值
        updated_threshold = threshold_manager.update_threshold_with_new_data(
            metric_name=metric_name,
            new_data=new_data,
            adaptation_factor=0.1
        )
        
        # 验证阈值调整
        assert isinstance(updated_threshold, float)
        assert updated_threshold != initial_threshold
        # 由于包含了更高的回撤值，阈值应该有所提高
        assert updated_threshold >= initial_threshold

    def test_multi_metric_threshold_management(self, threshold_manager):
        """测试多指标阈值管理"""
        metrics = ['portfolio_value', 'daily_return', 'volatility', 'max_drawdown']
        threshold_configs = {
            'portfolio_value': {'percentile': 5, 'type': 'lower'},
            'daily_return': {'percentile': 95, 'type': 'upper'},
            'volatility': {'percentile': 90, 'type': 'upper'},
            'max_drawdown': {'percentile': 95, 'type': 'upper'}
        }
        
        # 批量计算阈值
        all_thresholds = threshold_manager.calculate_multiple_thresholds(threshold_configs)
        
        # 验证结果
        assert isinstance(all_thresholds, dict)
        assert len(all_thresholds) == len(metrics)
        
        for metric in metrics:
            assert metric in all_thresholds
            assert isinstance(all_thresholds[metric], float)
        
        # 验证特定关系
        assert all_thresholds['daily_return'] > 0  # 95%分位数应该为正
        assert all_thresholds['volatility'] > 0    # 波动率应该为正
        assert all_thresholds['max_drawdown'] > 0  # 最大回撤应该为正

    def test_threshold_validation(self, threshold_manager):
        """测试阈值有效性验证"""
        # 测试有效阈值
        valid_config = {
            'metric_name': 'daily_return',
            'percentile': 95,
            'threshold_type': 'upper',
            'min_samples': 30
        }
        
        is_valid = threshold_manager.validate_threshold_config(valid_config)
        assert is_valid is True
        
        # 测试无效分位数
        invalid_percentile_config = {
            'metric_name': 'daily_return',
            'percentile': 105,  # 无效
            'threshold_type': 'upper',
            'min_samples': 30
        }
        
        is_valid = threshold_manager.validate_threshold_config(invalid_percentile_config)
        assert is_valid is False
        
        # 测试不存在的指标
        invalid_metric_config = {
            'metric_name': 'nonexistent_metric',
            'percentile': 95,
            'threshold_type': 'upper',
            'min_samples': 30
        }
        
        is_valid = threshold_manager.validate_threshold_config(invalid_metric_config)
        assert is_valid is False

    def test_threshold_persistence(self, threshold_manager, tmp_path):
        """测试阈值持久化"""
        # 计算一些阈值
        thresholds = {
            'daily_return_upper': 0.05,
            'daily_return_lower': -0.03,
            'volatility_upper': 0.25,
            'max_drawdown_upper': 0.08
        }
        
        threshold_manager.thresholds = thresholds
        
        # 保存阈值
        save_path = tmp_path / "thresholds.json"
        threshold_manager.save_thresholds(str(save_path))
        
        # 验证文件存在
        assert save_path.exists()
        
        # 加载阈值
        new_manager = DynamicThresholdManager(
            historical_data=threshold_manager.historical_data,
            lookback_window=60
        )
        new_manager.load_thresholds(str(save_path))
        
        # 验证加载的阈值
        assert new_manager.thresholds == thresholds

    def test_statistical_outlier_detection(self, threshold_manager):
        """测试统计异常值检测"""
        metric_name = 'daily_return'
        
        # 正常值
        normal_value = 0.01
        is_outlier_normal = threshold_manager.is_statistical_outlier(
            metric_name=metric_name,
            value=normal_value,
            method='zscore',
            threshold=3.0
        )
        assert is_outlier_normal == False
        
        # 异常值
        outlier_value = 0.15  # 极高的日收益率
        is_outlier_extreme = threshold_manager.is_statistical_outlier(
            metric_name=metric_name,
            value=outlier_value,
            method='zscore',
            threshold=3.0
        )
        assert is_outlier_extreme == True

    def test_threshold_sensitivity_analysis(self, threshold_manager):
        """测试阈值敏感性分析"""
        metric_name = 'volatility'
        percentiles = [80, 85, 90, 95, 99]
        
        sensitivity_results = threshold_manager.analyze_threshold_sensitivity(
            metric_name=metric_name,
            percentiles=percentiles
        )
        
        # 验证敏感性分析结果
        assert isinstance(sensitivity_results, dict)
        assert len(sensitivity_results) == len(percentiles)
        
        # 验证阈值递增性
        threshold_values = [sensitivity_results[p] for p in percentiles]
        assert all(threshold_values[i] <= threshold_values[i+1] for i in range(len(threshold_values)-1))

    def test_invalid_data_handling(self):
        """测试无效数据处理"""
        # 空数据集
        with pytest.raises(ValueError, match="历史数据不能为空"):
            DynamicThresholdManager(
                historical_data=pd.DataFrame(),
                lookback_window=60
            )
        
        # 无效窗口大小
        valid_data = pd.DataFrame({
            'metric': [1, 2, 3],
            'timestamp': pd.date_range('2023-01-01', periods=3, freq='D')
        })
        
        with pytest.raises(ValueError, match="回看窗口大小必须为正数"):
            DynamicThresholdManager(
                historical_data=valid_data,
                lookback_window=0
            )


class TestAlertRule:
    """告警规则测试类"""

    @pytest.fixture
    def sample_alert_rule(self):
        """创建样本告警规则"""
        return AlertRule(
            rule_id="test_rule_001",
            metric_name="max_drawdown",
            threshold_value=0.1,
            comparison_operator=">",
            alert_level=AlertLevel.WARNING,
            description="最大回撤过高告警"
        )

    def test_alert_rule_initialization(self, sample_alert_rule):
        """测试告警规则初始化"""
        assert sample_alert_rule.rule_id == "test_rule_001"
        assert sample_alert_rule.metric_name == "max_drawdown"
        assert sample_alert_rule.threshold_value == 0.1
        assert sample_alert_rule.comparison_operator == ">"
        assert sample_alert_rule.alert_level == AlertLevel.WARNING
        assert sample_alert_rule.is_active is True

    def test_rule_evaluation(self, sample_alert_rule):
        """测试规则评估"""
        # 触发告警的值
        trigger_value = 0.15
        should_trigger = sample_alert_rule.evaluate(trigger_value)
        assert should_trigger is True
        
        # 不触发告警的值
        normal_value = 0.05
        should_not_trigger = sample_alert_rule.evaluate(normal_value)
        assert should_not_trigger is False

    def test_different_comparison_operators(self):
        """测试不同比较操作符"""
        test_cases = [
            (">", 0.1, 0.15, True),   # 大于
            (">", 0.1, 0.05, False),
            (">=", 0.1, 0.1, True),   # 大于等于
            (">=", 0.1, 0.05, False),
            ("<", 0.1, 0.05, True),   # 小于
            ("<", 0.1, 0.15, False),
            ("<=", 0.1, 0.1, True),   # 小于等于
            ("<=", 0.1, 0.15, False),
            ("==", 0.1, 0.1, True),   # 等于
            ("==", 0.1, 0.15, False),
            ("!=", 0.1, 0.15, True),  # 不等于
            ("!=", 0.1, 0.1, False)
        ]
        
        for operator, threshold, value, expected in test_cases:
            rule = AlertRule(
                rule_id=f"test_{operator}",
                metric_name="test_metric",
                threshold_value=threshold,
                comparison_operator=operator,
                alert_level=AlertLevel.INFO
            )
            
            result = rule.evaluate(value)
            assert result == expected, f"Failed for {operator}: {value} {operator} {threshold}"

    def test_rule_serialization(self, sample_alert_rule):
        """测试规则序列化"""
        # 转换为字典
        rule_dict = sample_alert_rule.to_dict()
        
        assert isinstance(rule_dict, dict)
        assert rule_dict['rule_id'] == "test_rule_001"
        assert rule_dict['metric_name'] == "max_drawdown"
        assert rule_dict['threshold_value'] == 0.1
        assert rule_dict['comparison_operator'] == ">"
        assert rule_dict['alert_level'] == AlertLevel.WARNING.value
        
        # 从字典恢复
        restored_rule = AlertRule.from_dict(rule_dict)
        
        assert restored_rule.rule_id == sample_alert_rule.rule_id
        assert restored_rule.metric_name == sample_alert_rule.metric_name
        assert restored_rule.threshold_value == sample_alert_rule.threshold_value
        assert restored_rule.comparison_operator == sample_alert_rule.comparison_operator
        assert restored_rule.alert_level == sample_alert_rule.alert_level

    def test_rule_activation_deactivation(self, sample_alert_rule):
        """测试规则激活和停用"""
        # 初始状态应该是激活的
        assert sample_alert_rule.is_active is True
        
        # 停用规则
        sample_alert_rule.deactivate()
        assert sample_alert_rule.is_active is False
        
        # 停用状态下不应该触发告警
        trigger_value = 0.15
        should_not_trigger = sample_alert_rule.evaluate(trigger_value)
        assert should_not_trigger is False
        
        # 重新激活
        sample_alert_rule.activate()
        assert sample_alert_rule.is_active is True
        
        # 激活后应该能触发告警
        should_trigger = sample_alert_rule.evaluate(trigger_value)
        assert should_trigger is True

    def test_invalid_rule_parameters(self):
        """测试无效规则参数"""
        # 无效比较操作符
        with pytest.raises(ValueError, match="不支持的比较操作符"):
            AlertRule(
                rule_id="invalid_op",
                metric_name="test",
                threshold_value=0.1,
                comparison_operator="invalid",
                alert_level=AlertLevel.ERROR
            )
        
        # 空规则ID
        with pytest.raises(ValueError, match="规则ID不能为空"):
            AlertRule(
                rule_id="",
                metric_name="test",
                threshold_value=0.1,
                comparison_operator=">",
                alert_level=AlertLevel.ERROR
            )


class TestAlertAggregator:
    """告警聚合器测试类"""

    @pytest.fixture
    def alert_aggregator(self):
        """创建告警聚合器"""
        return AlertAggregator(
            aggregation_window=300,  # 5分钟
            max_alerts_per_rule=3,
            similarity_threshold=0.8
        )

    @pytest.fixture
    def sample_alerts(self):
        """创建样本告警"""
        base_time = datetime.now()
        return [
            {
                'rule_id': 'rule_001',
                'metric_name': 'max_drawdown',
                'value': 0.12,
                'threshold': 0.1,
                'level': AlertLevel.WARNING,
                'timestamp': base_time,
                'message': '最大回撤过高'
            },
            {
                'rule_id': 'rule_001',
                'metric_name': 'max_drawdown',
                'value': 0.13,
                'threshold': 0.1,
                'level': AlertLevel.WARNING,
                'timestamp': base_time + timedelta(seconds=30),  # 30秒内
                'message': '最大回撤过高'
            },
            {
                'rule_id': 'rule_002',
                'metric_name': 'volatility',
                'value': 0.35,
                'threshold': 0.3,
                'level': AlertLevel.ERROR,
                'timestamp': base_time + timedelta(seconds=15),  # 15秒内
                'message': '波动率异常'
            }
        ]

    def test_aggregator_initialization(self, alert_aggregator):
        """测试聚合器初始化"""
        assert alert_aggregator.aggregation_window == 300
        assert alert_aggregator.max_alerts_per_rule == 3
        assert alert_aggregator.similarity_threshold == 0.8
        assert len(alert_aggregator.pending_alerts) == 0

    def test_alert_aggregation(self, alert_aggregator, sample_alerts):
        """测试告警聚合"""
        # 添加告警
        for alert in sample_alerts:
            alert_aggregator.add_alert(alert)
        
        # 执行聚合
        aggregated_alerts = alert_aggregator.aggregate_alerts()
        
        # 验证聚合结果
        assert isinstance(aggregated_alerts, list)
        assert len(aggregated_alerts) <= len(sample_alerts)
        
        # 相同规则的告警应该被聚合
        rule_001_alerts = [a for a in aggregated_alerts if a['rule_id'] == 'rule_001']
        assert len(rule_001_alerts) == 1  # 两个相似的告警被聚合为一个
        
        # 聚合后的告警应该包含计数信息
        aggregated_alert = rule_001_alerts[0]
        assert 'count' in aggregated_alert
        assert aggregated_alert['count'] == 2

    def test_similarity_calculation(self, alert_aggregator):
        """测试相似度计算"""
        alert1 = {
            'rule_id': 'rule_001',
            'metric_name': 'max_drawdown',
            'level': AlertLevel.WARNING,
            'message': '最大回撤过高'
        }
        
        alert2 = {
            'rule_id': 'rule_001',
            'metric_name': 'max_drawdown',
            'level': AlertLevel.WARNING,
            'message': '最大回撤过高'
        }
        
        alert3 = {
            'rule_id': 'rule_002',
            'metric_name': 'volatility',
            'level': AlertLevel.ERROR,
            'message': '波动率异常'
        }
        
        # 相同规则的告警相似度应该高
        similarity_12 = alert_aggregator.calculate_similarity(alert1, alert2)
        assert similarity_12 >= 0.8
        
        # 不同规则的告警相似度应该低
        similarity_13 = alert_aggregator.calculate_similarity(alert1, alert3)
        assert similarity_13 < 0.8

    def test_rate_limiting(self, alert_aggregator):
        """测试频率限制"""
        # 创建大量相同的告警
        base_time = datetime.now()
        excessive_alerts = []
        
        for i in range(10):  # 超过max_alerts_per_rule的数量
            excessive_alerts.append({
                'rule_id': 'rule_spam',
                'metric_name': 'test_metric',
                'value': 0.1 + i * 0.01,
                'level': AlertLevel.INFO,
                'timestamp': base_time + timedelta(seconds=i * 10),
                'message': f'测试告警 {i}'
            })
        
        # 添加所有告警
        for alert in excessive_alerts:
            alert_aggregator.add_alert(alert)
        
        # 执行聚合
        aggregated_alerts = alert_aggregator.aggregate_alerts()
        
        # 验证频率限制生效
        rule_spam_alerts = [a for a in aggregated_alerts if a['rule_id'] == 'rule_spam']
        assert len(rule_spam_alerts) <= alert_aggregator.max_alerts_per_rule

    def test_time_window_expiry(self, alert_aggregator):
        """测试时间窗口过期"""
        old_time = datetime.now() - timedelta(minutes=10)  # 超出聚合窗口
        current_time = datetime.now()
        
        old_alert = {
            'rule_id': 'rule_old',
            'metric_name': 'test',
            'timestamp': old_time,
            'level': AlertLevel.INFO,
            'message': '过期告警'
        }
        
        current_alert = {
            'rule_id': 'rule_current',
            'metric_name': 'test',
            'timestamp': current_time,
            'level': AlertLevel.INFO,
            'message': '当前告警'
        }
        
        alert_aggregator.add_alert(old_alert)
        alert_aggregator.add_alert(current_alert)
        
        # 执行聚合
        aggregated_alerts = alert_aggregator.aggregate_alerts()
        
        # 过期的告警应该被清理
        rule_ids = [a['rule_id'] for a in aggregated_alerts]
        assert 'rule_old' not in rule_ids
        assert 'rule_current' in rule_ids


class TestNotificationManager:
    """通知管理器测试类"""

    @pytest.fixture
    def notification_manager(self):
        """创建通知管理器"""
        return NotificationManager({
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.test.com',
                'smtp_port': 587,
                'username': 'test@example.com',
                'password': 'test_password',
                'recipients': ['admin@example.com']
            },
            'webhook': {
                'enabled': True,
                'url': 'https://hooks.slack.com/test',
                'timeout': 10
            }
        })

    def test_notification_manager_initialization(self, notification_manager):
        """测试通知管理器初始化"""
        assert notification_manager.channels is not None
        assert 'email' in notification_manager.channels
        assert 'webhook' in notification_manager.channels
        assert notification_manager.channels['email']['enabled'] is True
        assert notification_manager.channels['webhook']['enabled'] is True

    @patch('smtplib.SMTP')
    def test_email_notification(self, mock_smtp, notification_manager):
        """测试邮件通知"""
        # 配置mock
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        alert_data = {
            'rule_id': 'test_rule',
            'metric_name': 'max_drawdown',
            'value': 0.15,
            'threshold': 0.1,
            'level': AlertLevel.ERROR,
            'message': '最大回撤严重超标'
        }
        
        # 发送邮件通知
        result = notification_manager.send_email_notification(alert_data)
        
        # 验证邮件发送
        assert result is True
        mock_smtp.assert_called_once()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()

    @patch('requests.post')
    def test_webhook_notification(self, mock_post, notification_manager):
        """测试Webhook通知"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        alert_data = {
            'rule_id': 'test_rule',
            'metric_name': 'volatility',
            'value': 0.35,
            'threshold': 0.3,
            'level': AlertLevel.WARNING,
            'message': '波动率偏高'
        }
        
        # 发送Webhook通知
        result = notification_manager.send_webhook_notification(alert_data)
        
        # 验证Webhook调用
        assert result is True
        mock_post.assert_called_once()
        
        # 验证请求参数
        call_args = mock_post.call_args
        assert call_args[0][0] == 'https://hooks.slack.com/test'
        assert 'json' in call_args[1]

    def test_notification_formatting(self, notification_manager):
        """测试通知格式化"""
        alert_data = {
            'rule_id': 'format_test',
            'metric_name': 'sharpe_ratio',
            'value': 0.5,
            'threshold': 1.0,
            'level': AlertLevel.WARNING,
            'message': '夏普比率偏低',
            'timestamp': datetime(2023, 10, 1, 10, 30, 0)
        }
        
        # 格式化邮件内容
        email_content = notification_manager.format_email_content(alert_data)
        
        assert isinstance(email_content, dict)
        assert 'subject' in email_content
        assert 'body' in email_content
        assert 'format_test' in email_content['subject']
        assert 'sharpe_ratio' in email_content['body']
        assert '0.5' in email_content['body']
        
        # 格式化Webhook内容
        webhook_content = notification_manager.format_webhook_content(alert_data)
        
        assert isinstance(webhook_content, dict)
        assert 'text' in webhook_content or 'content' in webhook_content

    def test_notification_retry_mechanism(self, notification_manager):
        """测试通知重试机制"""
        with patch('requests.post') as mock_post:
            # 模拟前两次失败，第三次成功
            mock_responses = [
                Mock(status_code=500),  # 第一次失败
                Mock(status_code=503),  # 第二次失败
                Mock(status_code=200)   # 第三次成功
            ]
            mock_post.side_effect = mock_responses
            
            alert_data = {
                'rule_id': 'retry_test',
                'level': AlertLevel.ERROR,
                'message': '重试测试'
            }
            
            # 发送通知（应该重试）
            result = notification_manager.send_webhook_notification(
                alert_data, 
                max_retries=3,
                retry_delay=0.1
            )
            
            # 验证重试成功
            assert result is True
            assert mock_post.call_count == 3

    def test_notification_channel_management(self, notification_manager):
        """测试通知渠道管理"""
        # 禁用邮件渠道
        notification_manager.disable_channel('email')
        assert notification_manager.channels['email']['enabled'] is False
        
        # 启用邮件渠道
        notification_manager.enable_channel('email')
        assert notification_manager.channels['email']['enabled'] is True
        
        # 获取活跃渠道
        active_channels = notification_manager.get_active_channels()
        assert 'email' in active_channels
        assert 'webhook' in active_channels

    def test_notification_rate_limiting(self, notification_manager):
        """测试通知频率限制"""
        # 设置频率限制（每分钟最多2条）
        notification_manager.set_rate_limit('email', max_notifications=2, time_window=60)
        
        alert_data = {
            'rule_id': 'rate_limit_test',
            'level': AlertLevel.INFO,
            'message': '频率限制测试'
        }
        
        with patch('smtplib.SMTP'):
            # 发送前两条应该成功
            assert notification_manager.send_email_notification(alert_data) is True
            assert notification_manager.send_email_notification(alert_data) is True
            
            # 第三条应该被限制
            assert notification_manager.send_email_notification(alert_data) is False


class TestAlertSystem:
    """告警系统集成测试类"""

    @pytest.fixture
    def alert_system(self):
        """创建完整的告警系统"""
        # 创建历史数据
        np.random.seed(42)
        historical_data = pd.DataFrame({
            'portfolio_value': np.random.normal(1000000, 50000, 100),
            'daily_return': np.random.normal(0.001, 0.02, 100),
            'max_drawdown': np.random.exponential(0.02, 100),
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='D')
        })
        
        # 通知配置
        notification_config = {
            'email': {
                'enabled': True,
                'recipients': ['admin@test.com']
            }
        }
        
        return AlertSystem(
            historical_data=historical_data,
            notification_config=notification_config
        )

    def test_alert_system_initialization(self, alert_system):
        """测试告警系统初始化"""
        assert alert_system.threshold_manager is not None
        assert alert_system.notification_manager is not None
        assert alert_system.aggregator is not None
        assert alert_system.logger is not None
        assert len(alert_system.rules) == 0

    def test_rule_management(self, alert_system):
        """测试规则管理"""
        rule = AlertRule(
            rule_id="system_test_rule",
            metric_name="max_drawdown",
            threshold_value=0.1,
            comparison_operator=">",
            alert_level=AlertLevel.ERROR
        )
        
        # 添加规则
        alert_system.add_rule(rule)
        assert len(alert_system.rules) == 1
        assert "system_test_rule" in alert_system.rules
        
        # 获取规则
        retrieved_rule = alert_system.get_rule("system_test_rule")
        assert retrieved_rule is not None
        assert retrieved_rule.rule_id == "system_test_rule"
        
        # 删除规则
        alert_system.remove_rule("system_test_rule")
        assert len(alert_system.rules) == 0

    def test_metric_monitoring(self, alert_system):
        """测试指标监控"""
        # 添加告警规则
        rule = AlertRule(
            rule_id="monitoring_test",
            metric_name="max_drawdown",
            threshold_value=0.08,
            comparison_operator=">",
            alert_level=AlertLevel.WARNING
        )
        alert_system.add_rule(rule)
        
        # 监控正常值
        normal_metrics = {
            'max_drawdown': 0.05,
            'daily_return': 0.01,
            'portfolio_value': 1000000
        }
        
        alerts = alert_system.check_metrics(normal_metrics)
        assert len(alerts) == 0
        
        # 监控异常值
        abnormal_metrics = {
            'max_drawdown': 0.12,  # 超过阈值
            'daily_return': 0.01,
            'portfolio_value': 1000000
        }
        
        alerts = alert_system.check_metrics(abnormal_metrics)
        assert len(alerts) == 1
        assert alerts[0]['rule_id'] == "monitoring_test"

    def test_end_to_end_alerting(self, alert_system):
        """测试端到端告警流程"""
        # 设置告警规则
        rules = [
            AlertRule("dd_warning", "max_drawdown", 0.1, ">", AlertLevel.WARNING),
            AlertRule("vol_error", "volatility", 0.3, ">", AlertLevel.ERROR)
        ]
        
        for rule in rules:
            alert_system.add_rule(rule)
        
        # 模拟异常指标
        abnormal_metrics = {
            'max_drawdown': 0.15,
            'volatility': 0.35,
            'daily_return': -0.05
        }
        
        with patch.object(alert_system.notification_manager, 'send_notification') as mock_notify:
            mock_notify.return_value = True
            
            # 处理指标（应该触发告警）
            alert_system.process_metrics(abnormal_metrics)
            
            # 验证通知被发送
            assert mock_notify.call_count >= 1

    def test_alert_silencing(self, alert_system):
        """测试告警静默"""
        rule = AlertRule(
            rule_id="silence_test",
            metric_name="daily_return",
            threshold_value=-0.05,
            comparison_operator="<",
            alert_level=AlertLevel.WARNING
        )
        alert_system.add_rule(rule)
        
        # 静默特定规则
        alert_system.silence_rule("silence_test", duration=300)  # 5分钟
        
        # 触发告警条件
        metrics = {'daily_return': -0.08}
        alerts = alert_system.check_metrics(metrics)
        
        # 应该没有告警（被静默）
        assert len(alerts) == 0

    def test_concurrent_monitoring(self, alert_system):
        """测试并发监控"""
        rule = AlertRule(
            rule_id="concurrent_test",
            metric_name="portfolio_value",
            threshold_value=900000,
            comparison_operator="<",
            alert_level=AlertLevel.ERROR
        )
        alert_system.add_rule(rule)
        
        def monitor_metrics():
            for i in range(10):
                metrics = {
                    'portfolio_value': 800000 + i * 10000,
                    'timestamp': datetime.now()
                }
                alert_system.process_metrics(metrics)
                time.sleep(0.01)
        
        # 启动多个监控线程
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=monitor_metrics)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证系统稳定运行
        assert len(alert_system.rules) == 1
        assert alert_system.rules["concurrent_test"].is_active