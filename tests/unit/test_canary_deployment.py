"""
金丝雀部署系统的单元测试
测试新模型的渐进式部署和评估机制，A/B测试框架和性能对比，部署过程的安全性和回滚能力
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
import uuid

from src.rl_trading_system.deployment.canary_deployment import (
    CanaryDeployment,
    ABTestFramework,
    ModelPerformanceComparator,
    DeploymentSafetyController,
    DeploymentStatus,
    RollbackManager,
    TrafficRouter,
    PerformanceMetrics,
    DeploymentConfig
)


# 共享的测试fixtures
@pytest.fixture
def mock_model():
    """创建模拟模型"""
    model = Mock()
    model.predict = Mock(return_value=np.array([0.1, 0.2, 0.7]))
    model.version = "v1.2.0"
    model.created_at = datetime.now()
    return model


@pytest.fixture
def mock_baseline_model():
    """创建基线模型"""
    model = Mock()
    model.predict = Mock(return_value=np.array([0.2, 0.3, 0.5]))
    model.version = "v1.1.0"
    model.created_at = datetime.now() - timedelta(days=7)
    return model


class TestCanaryDeployment:
    """金丝雀部署测试类"""

    @pytest.fixture
    def deployment_config(self):
        """创建部署配置"""
        return DeploymentConfig(
            canary_percentage=10.0,
            evaluation_period=3600,  # 1小时
            success_threshold=0.95,
            error_threshold=0.05,
            performance_threshold=0.02,
            rollback_threshold=0.1,
            max_canary_duration=7200  # 2小时
        )



    @pytest.fixture
    def canary_deployment(self, deployment_config, mock_model, mock_baseline_model):
        """创建金丝雀部署实例"""
        return CanaryDeployment(
            canary_model=mock_model,
            baseline_model=mock_baseline_model,
            config=deployment_config
        )

    def test_canary_deployment_initialization(self, canary_deployment, deployment_config):
        """测试金丝雀部署初始化"""
        assert canary_deployment.canary_model is not None
        assert canary_deployment.baseline_model is not None
        assert canary_deployment.config == deployment_config
        assert canary_deployment.status == DeploymentStatus.PENDING
        assert canary_deployment.start_time is None
        assert canary_deployment.traffic_percentage == 0.0

    def test_deployment_start(self, canary_deployment):
        """测试部署启动"""
        # 启动部署
        canary_deployment.start_deployment()
        
        # 验证状态变化
        assert canary_deployment.status == DeploymentStatus.ACTIVE
        assert canary_deployment.start_time is not None
        assert canary_deployment.traffic_percentage == canary_deployment.config.canary_percentage
        
        # 验证无法重复启动
        with pytest.raises(ValueError, match="部署已经在运行中"):
            canary_deployment.start_deployment()

    def test_gradual_traffic_increase(self, canary_deployment):
        """测试渐进式流量增长"""
        canary_deployment.start_deployment()
        
        # 模拟成功的评估结果
        canary_deployment.performance_comparator.latest_comparison = {
            'success_rate_diff': -0.01,  # 金丝雀模型更好
            'error_rate_diff': -0.02,
            'performance_improvement': 0.03,
            'statistical_significance': True
        }
        
        # 增加流量
        initial_traffic = canary_deployment.traffic_percentage
        canary_deployment.increase_traffic(step_size=10.0)
        
        assert canary_deployment.traffic_percentage == initial_traffic + 10.0

    def test_deployment_success_criteria(self, canary_deployment):
        """测试部署成功标准"""
        canary_deployment.start_deployment()
        
        # 模拟满足成功条件的指标
        metrics = PerformanceMetrics(
            success_rate=0.98,
            error_rate=0.01,
            avg_response_time=0.15,
            throughput=1000.0,
            accuracy=0.92,
            precision=0.88,
            recall=0.85,
            f1_score=0.86
        )
        
        canary_deployment.update_canary_metrics(metrics)
        
        # 评估是否满足成功标准
        meets_criteria = canary_deployment.evaluate_success_criteria()
        assert meets_criteria == True

    def test_deployment_failure_detection(self, canary_deployment):
        """测试部署失败检测"""
        canary_deployment.start_deployment()
        
        # 模拟不满足成功条件的指标
        metrics = PerformanceMetrics(
            success_rate=0.85,  # 低于阈值
            error_rate=0.15,    # 高于阈值
            avg_response_time=0.5,
            throughput=500.0,
            accuracy=0.75,
            precision=0.70,
            recall=0.68,
            f1_score=0.69
        )
        
        canary_deployment.update_canary_metrics(metrics)
        
        # 评估是否满足成功标准
        meets_criteria = canary_deployment.evaluate_success_criteria()
        assert meets_criteria == False

    def test_automatic_rollback_trigger(self, canary_deployment):
        """测试自动回滚触发"""
        canary_deployment.start_deployment()
        
        # 模拟严重的性能下降
        canary_deployment.performance_comparator.latest_comparison = {
            'success_rate_diff': 0.15,  # 金丝雀模型更差
            'error_rate_diff': 0.12,
            'performance_improvement': -0.2,
            'statistical_significance': True
        }
        
        # 检查是否触发回滚
        should_rollback = canary_deployment.should_trigger_rollback()
        assert should_rollback is True

    def test_deployment_completion_successful(self, canary_deployment):
        """测试成功的部署完成"""
        canary_deployment.start_deployment()
        
        # 模拟持续成功的表现
        for i in range(5):
            metrics = PerformanceMetrics(
                success_rate=0.98,
                error_rate=0.01,
                avg_response_time=0.1,
                throughput=1200.0,
                accuracy=0.93,
                precision=0.90,
                recall=0.88,
                f1_score=0.89
            )
            canary_deployment.update_canary_metrics(metrics)
            canary_deployment.increase_traffic(step_size=20.0)
        
        # 完成部署
        canary_deployment.complete_deployment()
        
        assert canary_deployment.status == DeploymentStatus.COMPLETED
        assert canary_deployment.traffic_percentage == 100.0

    def test_deployment_rollback(self, canary_deployment):
        """测试部署回滚"""
        canary_deployment.start_deployment()
        
        # 触发回滚
        rollback_reason = "性能严重下降"
        canary_deployment.rollback_deployment(reason=rollback_reason)
        
        assert canary_deployment.status == DeploymentStatus.ROLLED_BACK
        assert canary_deployment.traffic_percentage == 0.0
        assert rollback_reason in canary_deployment.deployment_history[-1]['reason']

    def test_deployment_timeout_handling(self, canary_deployment):
        """测试部署超时处理"""
        canary_deployment.start_deployment()
        
        # 模拟超时
        canary_deployment.start_time = datetime.now() - timedelta(seconds=canary_deployment.config.max_canary_duration + 1)
        
        is_timeout = canary_deployment.is_deployment_timeout()
        assert is_timeout is True

    def test_concurrent_deployment_prevention(self, canary_deployment):
        """测试并发部署防护"""
        canary_deployment.start_deployment()
        
        # 尝试再次启动同一个部署实例
        with pytest.raises(ValueError, match="部署已经在运行中"):
            canary_deployment.start_deployment()

    def test_deployment_metrics_collection(self, canary_deployment):
        """测试部署指标收集"""
        canary_deployment.start_deployment()
        
        # 添加多个指标数据点
        for i in range(10):
            metrics = PerformanceMetrics(
                success_rate=0.95 + i * 0.001,
                error_rate=0.02 - i * 0.001,
                avg_response_time=0.1 + i * 0.01,
                throughput=1000 + i * 10,
                accuracy=0.90 + i * 0.001,
                precision=0.85 + i * 0.001,
                recall=0.82 + i * 0.001,
                f1_score=0.83 + i * 0.001
            )
            canary_deployment.update_canary_metrics(metrics)
        
        # 验证指标历史记录
        metrics_history = canary_deployment.get_metrics_history()
        assert len(metrics_history) == 10
        assert metrics_history[-1].success_rate > metrics_history[0].success_rate

    def test_deployment_configuration_validation(self):
        """测试部署配置验证"""
        # 测试无效的金丝雀百分比
        with pytest.raises(ValueError, match="金丝雀流量百分比必须在0-100之间"):
            DeploymentConfig(
                canary_percentage=150.0,  # 无效
                evaluation_period=3600,
                success_threshold=0.95,
                error_threshold=0.05,
                performance_threshold=0.02,
                rollback_threshold=0.1,
                max_canary_duration=7200
            )
        
        # 测试无效的阈值
        with pytest.raises(ValueError, match="成功率阈值必须在0-1之间"):
            DeploymentConfig(
                canary_percentage=10.0,
                evaluation_period=3600,
                success_threshold=1.5,  # 无效
                error_threshold=0.05,
                performance_threshold=0.02,
                rollback_threshold=0.1,
                max_canary_duration=7200
            )


class TestABTestFramework:
    """A/B测试框架测试类"""

    @pytest.fixture
    def ab_test_framework(self, mock_model, mock_baseline_model):
        """创建A/B测试框架"""
        return ABTestFramework(
            model_a=mock_baseline_model,  # 基线模型
            model_b=mock_model,           # 新模型
            traffic_split=0.5,            # 50/50分流
            minimum_sample_size=100,      # 降低最小样本量以便测试
            confidence_level=0.95,
            test_duration=86400           # 24小时
        )

    def test_ab_framework_initialization(self, ab_test_framework):
        """测试A/B测试框架初始化"""
        assert ab_test_framework.model_a is not None
        assert ab_test_framework.model_b is not None
        assert ab_test_framework.traffic_split == 0.5
        assert ab_test_framework.minimum_sample_size == 100
        assert ab_test_framework.confidence_level == 0.95
        assert ab_test_framework.test_status == "pending"

    def test_traffic_routing(self, ab_test_framework):
        """测试流量路由"""
        # 模拟1000次请求
        model_a_count = 0
        model_b_count = 0
        
        for i in range(1000):
            user_id = f"user_{i}"
            selected_model = ab_test_framework.route_traffic(user_id)
            
            if selected_model == ab_test_framework.model_a:
                model_a_count += 1
            else:
                model_b_count += 1
        
        # 验证流量分配接近50/50
        total_requests = model_a_count + model_b_count
        model_a_ratio = model_a_count / total_requests
        model_b_ratio = model_b_count / total_requests
        
        assert abs(model_a_ratio - 0.5) < 0.1  # 允许10%的偏差
        assert abs(model_b_ratio - 0.5) < 0.1

    def test_consistent_user_routing(self, ab_test_framework):
        """测试用户路由一致性"""
        user_id = "test_user_123"
        
        # 同一用户多次请求应该路由到同一模型
        first_model = ab_test_framework.route_traffic(user_id)
        for _ in range(10):
            current_model = ab_test_framework.route_traffic(user_id)
            assert current_model == first_model

    def test_experiment_data_collection(self, ab_test_framework):
        """测试实验数据收集"""
        # 模拟A/B测试数据收集
        for i in range(100):
            user_id = f"user_{i}"
            selected_model = ab_test_framework.route_traffic(user_id)
            
            # 模拟预测结果和实际结果
            prediction = 0.8 if selected_model == ab_test_framework.model_a else 0.9
            actual = 1.0 if i % 3 == 0 else 0.0  # 模拟实际结果
            
            ab_test_framework.record_result(user_id, selected_model, prediction, actual)
        
        # 验证数据收集
        experiment_data = ab_test_framework.get_experiment_data()
        assert len(experiment_data) == 100
        assert all('user_id' in record for record in experiment_data)
        assert all('model' in record for record in experiment_data)
        assert all('prediction' in record for record in experiment_data)
        assert all('actual' in record for record in experiment_data)

    def test_statistical_significance_testing(self, ab_test_framework):
        """测试统计显著性检验"""
        # 添加模拟数据
        np.random.seed(42)
        
        # 模型A的数据（较差性能）
        for i in range(500):
            ab_test_framework.record_result(
                f"user_a_{i}", 
                ab_test_framework.model_a,
                np.random.normal(0.7, 0.1),  # 预测
                1.0 if np.random.random() > 0.4 else 0.0  # 60%成功率
            )
        
        # 模型B的数据（较好性能）
        for i in range(500):
            ab_test_framework.record_result(
                f"user_b_{i}",
                ab_test_framework.model_b,
                np.random.normal(0.8, 0.1),  # 预测
                1.0 if np.random.random() > 0.3 else 0.0  # 70%成功率
            )
        
        # 进行统计显著性检验
        significance_result = ab_test_framework.calculate_statistical_significance()
        
        assert isinstance(significance_result, dict)
        assert 'p_value' in significance_result
        assert 'is_significant' in significance_result
        assert 'confidence_interval' in significance_result
        assert 'effect_size' in significance_result

    def test_minimum_sample_size_check(self, ab_test_framework):
        """测试最小样本量检查"""
        # 样本量不足时 - 只添加少量数据到每个模型
        for i in range(50):  # 少于minimum_sample_size
            ab_test_framework.record_result(f"user_a_{i}", ab_test_framework.model_a, 0.8, 1.0)
            ab_test_framework.record_result(f"user_b_{i}", ab_test_framework.model_b, 0.8, 1.0)
        
        has_sufficient_data = ab_test_framework.has_sufficient_sample_size()
        assert has_sufficient_data == False
        
        # 添加更多数据达到最小样本量
        for i in range(50, 100):
            ab_test_framework.record_result(f"user_a_{i}", ab_test_framework.model_a, 0.8, 1.0)
            ab_test_framework.record_result(f"user_b_{i}", ab_test_framework.model_b, 0.8, 1.0)
        
        has_sufficient_data = ab_test_framework.has_sufficient_sample_size()
        assert has_sufficient_data == True

    def test_ab_test_completion(self, ab_test_framework):
        """测试A/B测试完成"""
        # 添加足够的数据
        for i in range(1000):
            model = ab_test_framework.model_a if i < 500 else ab_test_framework.model_b
            ab_test_framework.record_result(f"user_{i}", model, 0.8, 1.0)
        
        # 设置测试开始时间为24小时前
        ab_test_framework.start_time = datetime.now() - timedelta(hours=25)
        
        # 检查测试是否完成
        is_complete = ab_test_framework.is_test_complete()
        assert is_complete is True

    def test_winner_determination(self, ab_test_framework):
        """测试获胜者确定"""
        np.random.seed(42)
        
        # 开始测试并设置为已完成状态
        ab_test_framework.start_test()
        ab_test_framework.start_time = datetime.now() - timedelta(hours=25)  # 模拟测试已运行超过24小时
        
        # 模型A数据（较差性能）
        for i in range(150):
            success = 1.0 if np.random.random() > 0.4 else 0.0
            ab_test_framework.record_result(f"user_a_{i}", ab_test_framework.model_a, 0.7, success)
        
        # 模型B数据（较好性能）
        for i in range(150):
            success = 1.0 if np.random.random() > 0.2 else 0.0
            ab_test_framework.record_result(f"user_b_{i}", ab_test_framework.model_b, 0.8, success)
        
        winner = ab_test_framework.determine_winner()
        
        assert winner is not None
        assert winner in [ab_test_framework.model_a, ab_test_framework.model_b]


class TestModelPerformanceComparator:
    """模型性能比较器测试类"""

    @pytest.fixture
    def performance_comparator(self):
        """创建性能比较器"""
        return ModelPerformanceComparator(
            comparison_window=3600,  # 1小时
            min_samples_for_comparison=100,
            significance_threshold=0.05
        )

    @pytest.fixture
    def sample_metrics_a(self):
        """创建样本指标A"""
        return [
            PerformanceMetrics(0.95, 0.02, 0.1, 1000, 0.90, 0.88, 0.85, 0.86),
            PerformanceMetrics(0.94, 0.03, 0.12, 980, 0.89, 0.87, 0.84, 0.85),
            PerformanceMetrics(0.96, 0.01, 0.09, 1020, 0.91, 0.89, 0.86, 0.87),
        ]

    @pytest.fixture
    def sample_metrics_b(self):
        """创建样本指标B"""
        return [
            PerformanceMetrics(0.97, 0.01, 0.08, 1100, 0.92, 0.90, 0.88, 0.89),
            PerformanceMetrics(0.98, 0.01, 0.07, 1120, 0.93, 0.91, 0.89, 0.90),
            PerformanceMetrics(0.96, 0.02, 0.09, 1080, 0.91, 0.89, 0.87, 0.88),
        ]

    def test_comparator_initialization(self, performance_comparator):
        """测试性能比较器初始化"""
        assert performance_comparator.comparison_window == 3600
        assert performance_comparator.min_samples_for_comparison == 100
        assert performance_comparator.significance_threshold == 0.05
        assert len(performance_comparator.model_a_metrics) == 0
        assert len(performance_comparator.model_b_metrics) == 0

    def test_metrics_collection(self, performance_comparator, sample_metrics_a, sample_metrics_b):
        """测试指标收集"""
        # 添加模型A的指标
        for metrics in sample_metrics_a:
            performance_comparator.add_model_a_metrics(metrics)
        
        # 添加模型B的指标
        for metrics in sample_metrics_b:
            performance_comparator.add_model_b_metrics(metrics)
        
        assert len(performance_comparator.model_a_metrics) == 3
        assert len(performance_comparator.model_b_metrics) == 3

    def test_performance_comparison(self, performance_comparator, sample_metrics_a, sample_metrics_b):
        """测试性能比较"""
        # 添加指标数据
        for metrics in sample_metrics_a:
            performance_comparator.add_model_a_metrics(metrics)
        for metrics in sample_metrics_b:
            performance_comparator.add_model_b_metrics(metrics)
        
        # 进行性能比较
        comparison_result = performance_comparator.compare_performance()
        
        assert isinstance(comparison_result, dict)
        assert 'success_rate_diff' in comparison_result
        assert 'error_rate_diff' in comparison_result
        assert 'response_time_diff' in comparison_result
        assert 'throughput_diff' in comparison_result
        assert 'overall_performance_score' in comparison_result

    def test_statistical_significance(self, performance_comparator):
        """测试统计显著性检验"""
        # 创建有显著差异的数据
        np.random.seed(42)
        
        # 模型A数据（较差）
        for _ in range(100):
            metrics = PerformanceMetrics(
                success_rate=np.clip(np.random.normal(0.85, 0.05), 0, 1),
                error_rate=np.clip(np.random.normal(0.05, 0.02), 0, 1),
                avg_response_time=max(0, np.random.normal(0.15, 0.03)),
                throughput=max(0, np.random.normal(900, 100)),
                accuracy=np.clip(np.random.normal(0.85, 0.05), 0, 1),
                precision=np.clip(np.random.normal(0.83, 0.05), 0, 1),
                recall=np.clip(np.random.normal(0.82, 0.05), 0, 1),
                f1_score=np.clip(np.random.normal(0.82, 0.05), 0, 1)
            )
            performance_comparator.add_model_a_metrics(metrics)
        
        # 模型B数据（较好）
        for _ in range(100):
            metrics = PerformanceMetrics(
                success_rate=np.clip(np.random.normal(0.95, 0.03), 0, 1),
                error_rate=np.clip(np.random.normal(0.02, 0.01), 0, 1),
                avg_response_time=max(0, np.random.normal(0.10, 0.02)),
                throughput=max(0, np.random.normal(1100, 80)),
                accuracy=np.clip(np.random.normal(0.93, 0.03), 0, 1),
                precision=np.clip(np.random.normal(0.91, 0.03), 0, 1),
                recall=np.clip(np.random.normal(0.90, 0.03), 0, 1),
                f1_score=np.clip(np.random.normal(0.90, 0.03), 0, 1)
            )
            performance_comparator.add_model_b_metrics(metrics)
        
        # 计算统计显著性
        significance_test = performance_comparator.test_statistical_significance()
        
        assert isinstance(significance_test, dict)
        assert 'success_rate_significant' in significance_test
        assert 'error_rate_significant' in significance_test
        assert 'response_time_significant' in significance_test

    def test_performance_trend_analysis(self, performance_comparator):
        """测试性能趋势分析"""
        # 创建趋势数据
        base_time = datetime.now()
        
        for i in range(50):
            # 模型A性能逐渐下降
            metrics_a = PerformanceMetrics(
                success_rate=0.95 - i * 0.001,
                error_rate=0.02 + i * 0.0005,
                avg_response_time=0.1 + i * 0.001,
                throughput=1000 - i * 2,
                accuracy=0.90 - i * 0.001,
                precision=0.88 - i * 0.001,
                recall=0.85 - i * 0.001,
                f1_score=0.86 - i * 0.001
            )
            metrics_a.timestamp = base_time + timedelta(minutes=i)
            performance_comparator.add_model_a_metrics(metrics_a)
            
            # 模型B性能稳定
            metrics_b = PerformanceMetrics(
                success_rate=0.97,
                error_rate=0.01,
                avg_response_time=0.08,
                throughput=1100,
                accuracy=0.93,
                precision=0.91,
                recall=0.89,
                f1_score=0.90
            )
            metrics_b.timestamp = base_time + timedelta(minutes=i)
            performance_comparator.add_model_b_metrics(metrics_b)
        
        # 分析性能趋势
        trend_analysis = performance_comparator.analyze_performance_trends()
        
        assert isinstance(trend_analysis, dict)
        assert 'model_a_trend' in trend_analysis
        assert 'model_b_trend' in trend_analysis
        assert 'trend_significance' in trend_analysis

    def test_performance_degradation_detection(self, performance_comparator):
        """测试性能退化检测"""
        # 添加正常基线数据 - 需要至少10个作为基线比较
        for _ in range(10):
            metrics = PerformanceMetrics(0.95, 0.02, 0.1, 1000, 0.90, 0.88, 0.85, 0.86)
            performance_comparator.add_model_a_metrics(metrics)
        
        # 添加性能退化的数据 - 最近10个数据点
        for _ in range(10):
            metrics = PerformanceMetrics(0.80, 0.10, 0.2, 500, 0.75, 0.73, 0.70, 0.71)
            performance_comparator.add_model_a_metrics(metrics)
        
        # 检测性能退化
        degradation_detected = performance_comparator.detect_performance_degradation('model_a')
        assert degradation_detected == True

    def test_confidence_interval_calculation(self, performance_comparator, sample_metrics_a):
        """测试置信区间计算"""
        # 添加指标数据
        for metrics in sample_metrics_a * 30:  # 重复以获得足够样本
            performance_comparator.add_model_a_metrics(metrics)
        
        # 计算置信区间
        confidence_intervals = performance_comparator.calculate_confidence_intervals('model_a')
        
        assert isinstance(confidence_intervals, dict)
        assert 'success_rate' in confidence_intervals
        assert 'error_rate' in confidence_intervals
        assert all('lower' in ci and 'upper' in ci for ci in confidence_intervals.values())


class TestDeploymentSafetyController:
    """部署安全控制器测试类"""

    @pytest.fixture
    def safety_controller(self):
        """创建部署安全控制器"""
        return DeploymentSafetyController(
            max_error_rate=0.05,
            max_response_time=0.5,
            min_success_rate=0.90,
            circuit_breaker_threshold=10,
            recovery_check_interval=300
        )

    @pytest.fixture
    def canary_deployment_mock(self, safety_controller):
        """创建金丝雀部署模拟"""
        deployment = Mock()
        deployment.canary_model = Mock()
        deployment.baseline_model = Mock()
        deployment.status = DeploymentStatus.ACTIVE
        deployment.traffic_percentage = 10.0
        return deployment

    def test_safety_controller_initialization(self, safety_controller):
        """测试安全控制器初始化"""
        assert safety_controller.max_error_rate == 0.05
        assert safety_controller.max_response_time == 0.5
        assert safety_controller.min_success_rate == 0.90
        assert safety_controller.circuit_breaker_threshold == 10
        assert safety_controller.is_circuit_breaker_open is False

    def test_safety_check_passing(self, safety_controller):
        """测试安全检查通过"""
        # 正常的指标
        metrics = PerformanceMetrics(
            success_rate=0.95,
            error_rate=0.02,
            avg_response_time=0.1,
            throughput=1000,
            accuracy=0.90,
            precision=0.88,
            recall=0.85,
            f1_score=0.86
        )
        
        safety_check_result = safety_controller.perform_safety_check(metrics)
        
        assert safety_check_result['passed'] is True
        assert len(safety_check_result['violations']) == 0

    def test_safety_check_failure(self, safety_controller):
        """测试安全检查失败"""
        # 异常的指标
        metrics = PerformanceMetrics(
            success_rate=0.80,  # 低于阈值
            error_rate=0.10,    # 高于阈值
            avg_response_time=0.8,  # 高于阈值
            throughput=500,
            accuracy=0.75,
            precision=0.70,
            recall=0.68,
            f1_score=0.69
        )
        
        safety_check_result = safety_controller.perform_safety_check(metrics)
        
        assert safety_check_result['passed'] is False
        assert len(safety_check_result['violations']) > 0
        assert 'success_rate' in safety_check_result['violations']
        assert 'error_rate' in safety_check_result['violations']
        assert 'response_time' in safety_check_result['violations']

    def test_circuit_breaker_activation(self, safety_controller):
        """测试熔断器激活"""
        # 连续失败多次触发熔断器
        bad_metrics = PerformanceMetrics(0.70, 0.15, 1.0, 300, 0.65, 0.60, 0.58, 0.59)
        
        for _ in range(15):  # 超过circuit_breaker_threshold
            safety_controller.perform_safety_check(bad_metrics)
        
        assert safety_controller.is_circuit_breaker_open is True

    def test_circuit_breaker_recovery(self, safety_controller):
        """测试熔断器恢复"""
        # 先触发熔断器
        bad_metrics = PerformanceMetrics(0.70, 0.15, 1.0, 300, 0.65, 0.60, 0.58, 0.59)
        for _ in range(15):
            safety_controller.perform_safety_check(bad_metrics)
        
        assert safety_controller.is_circuit_breaker_open is True
        
        # 等待恢复检查间隔
        safety_controller.last_circuit_check = datetime.now() - timedelta(seconds=400)
        
        # 提供好的指标尝试恢复
        good_metrics = PerformanceMetrics(0.95, 0.02, 0.1, 1000, 0.90, 0.88, 0.85, 0.86)
        safety_controller.attempt_circuit_recovery(good_metrics)
        
        assert safety_controller.is_circuit_breaker_open is False

    def test_rollback_decision(self, safety_controller, canary_deployment_mock):
        """测试回滚决策"""
        # 模拟严重的安全违规
        bad_metrics = PerformanceMetrics(0.60, 0.25, 2.0, 200, 0.55, 0.50, 0.48, 0.49)
        
        rollback_decision = safety_controller.should_rollback_deployment(
            canary_deployment_mock, bad_metrics
        )
        
        assert rollback_decision['should_rollback'] is True
        assert 'reason' in rollback_decision
        assert len(rollback_decision['safety_violations']) > 0

    def test_gradual_traffic_control(self, safety_controller, canary_deployment_mock):
        """测试渐进式流量控制"""
        # 轻微的性能问题，建议减少流量而不是完全回滚
        moderate_metrics = PerformanceMetrics(0.88, 0.06, 0.3, 800, 0.82, 0.80, 0.78, 0.79)
        
        traffic_decision = safety_controller.evaluate_traffic_adjustment(
            canary_deployment_mock, moderate_metrics
        )
        
        assert traffic_decision['action'] in ['maintain', 'reduce', 'increase']
        assert 'recommended_percentage' in traffic_decision

    def test_safety_metrics_aggregation(self, safety_controller):
        """测试安全指标聚合"""
        # 添加多个指标数据点
        metrics_list = []
        for i in range(10):
            metrics = PerformanceMetrics(
                success_rate=0.90 + i * 0.01,
                error_rate=0.05 - i * 0.003,
                avg_response_time=0.1 + i * 0.01,
                throughput=1000 + i * 10,
                accuracy=0.85 + i * 0.01,
                precision=0.83 + i * 0.01,
                recall=0.80 + i * 0.01,
                f1_score=0.81 + i * 0.01
            )
            metrics_list.append(metrics)
            safety_controller.add_metrics_for_analysis(metrics)
        
        # 聚合安全指标
        aggregated_metrics = safety_controller.get_aggregated_safety_metrics()
        
        assert isinstance(aggregated_metrics, dict)
        assert 'avg_success_rate' in aggregated_metrics
        assert 'avg_error_rate' in aggregated_metrics
        assert 'avg_response_time' in aggregated_metrics
        assert 'percentile_95_response_time' in aggregated_metrics

    def test_emergency_stop_mechanism(self, safety_controller, canary_deployment_mock):
        """测试紧急停止机制"""
        # 极其严重的指标触发紧急停止
        critical_metrics = PerformanceMetrics(0.30, 0.50, 5.0, 50, 0.25, 0.20, 0.18, 0.19)
        
        emergency_decision = safety_controller.evaluate_emergency_stop(
            canary_deployment_mock, critical_metrics
        )
        
        assert emergency_decision['emergency_stop'] is True
        assert 'critical_violations' in emergency_decision
        assert len(emergency_decision['critical_violations']) > 0


class TestRollbackManager:
    """回滚管理器测试类"""

    @pytest.fixture
    def rollback_manager(self):
        """创建回滚管理器"""
        return RollbackManager(
            rollback_timeout=300,  # 5分钟
            verification_checks=5,
            health_check_interval=30
        )

    @pytest.fixture
    def deployment_mock(self):
        """创建部署模拟"""
        deployment = Mock()
        deployment.canary_model = Mock()
        deployment.baseline_model = Mock()
        deployment.status = DeploymentStatus.ACTIVE
        deployment.traffic_percentage = 50.0
        deployment.deployment_id = str(uuid.uuid4())
        return deployment

    def test_rollback_manager_initialization(self, rollback_manager):
        """测试回滚管理器初始化"""
        assert rollback_manager.rollback_timeout == 300
        assert rollback_manager.verification_checks == 5
        assert rollback_manager.health_check_interval == 30
        assert len(rollback_manager.rollback_history) == 0

    def test_rollback_execution(self, rollback_manager, deployment_mock):
        """测试回滚执行"""
        rollback_reason = "性能严重下降"
        
        rollback_result = rollback_manager.execute_rollback(deployment_mock, rollback_reason)
        
        assert rollback_result['success'] is True
        assert rollback_result['rollback_id'] is not None
        assert rollback_result['timestamp'] is not None
        assert len(rollback_manager.rollback_history) == 1

    def test_rollback_verification(self, rollback_manager, deployment_mock):
        """测试回滚验证"""
        # 执行回滚
        rollback_result = rollback_manager.execute_rollback(deployment_mock, "测试回滚")
        rollback_id = rollback_result['rollback_id']
        
        # 模拟验证检查
        with patch.object(rollback_manager, '_perform_health_check', return_value=True):
            verification_result = rollback_manager.verify_rollback(rollback_id)
        
        assert verification_result['verified'] is True
        assert verification_result['checks_passed'] == rollback_manager.verification_checks

    def test_rollback_failure_handling(self, rollback_manager, deployment_mock):
        """测试回滚失败处理"""
        # 模拟回滚失败
        with patch.object(rollback_manager, '_execute_traffic_rollback', side_effect=Exception("回滚失败")):
            rollback_result = rollback_manager.execute_rollback(deployment_mock, "测试失败")
        
        assert rollback_result['success'] is False
        assert 'error' in rollback_result

    def test_partial_rollback(self, rollback_manager, deployment_mock):
        """测试部分回滚"""
        # 执行部分回滚（减少流量而不是完全回滚）
        target_percentage = 10.0
        
        partial_rollback_result = rollback_manager.execute_partial_rollback(
            deployment_mock, target_percentage, "性能轻微下降"
        )
        
        assert partial_rollback_result['success'] is True
        assert partial_rollback_result['new_traffic_percentage'] == target_percentage

    def test_rollback_history_management(self, rollback_manager, deployment_mock):
        """测试回滚历史管理"""
        # 执行多次回滚
        for i in range(3):
            rollback_manager.execute_rollback(deployment_mock, f"回滚原因 {i+1}")
        
        # 获取回滚历史
        history = rollback_manager.get_rollback_history(deployment_mock.deployment_id)
        
        assert len(history) == 3
        assert all('rollback_id' in record for record in history)
        assert all('reason' in record for record in history)
        assert all('timestamp' in record for record in history)

    def test_automated_rollback_trigger(self, rollback_manager, deployment_mock):
        """测试自动回滚触发"""
        # 设置自动回滚条件
        rollback_conditions = {
            'max_error_rate': 0.10,
            'min_success_rate': 0.85,
            'max_response_time': 1.0
        }
        
        rollback_manager.set_auto_rollback_conditions(rollback_conditions)
        
        # 模拟触发自动回滚的指标
        bad_metrics = PerformanceMetrics(0.80, 0.15, 1.5, 400, 0.75, 0.70, 0.68, 0.69)
        
        should_auto_rollback = rollback_manager.should_trigger_auto_rollback(bad_metrics)
        assert should_auto_rollback is True

    def test_rollback_timeout_handling(self, rollback_manager, deployment_mock):
        """测试回滚超时处理"""
        # 模拟超时的回滚
        with patch.object(rollback_manager, '_execute_traffic_rollback') as mock_rollback:
            mock_rollback.side_effect = lambda *args: time.sleep(400)  # 超过timeout
            
            rollback_result = rollback_manager.execute_rollback(deployment_mock, "超时测试")
        
        # 注意：这个测试需要实际的超时机制实现
        # 这里我们验证rollback_manager能处理超时情况
        assert 'timeout' in str(rollback_result).lower() or rollback_result.get('success') is False

class TestTrafficRouter:
    """流量路由器测试类"""

    @pytest.fixture
    def traffic_router(self):
        """创建流量路由器"""
        return TrafficRouter(
            canary_percentage=20.0,
            routing_strategy='weighted_random',
            sticky_sessions=True
        )

    @pytest.fixture
    def models_mock(self):
        """创建模型模拟"""
        canary_model = Mock()
        canary_model.version = "v2.0.0"
        baseline_model = Mock()
        baseline_model.version = "v1.0.0"
        return canary_model, baseline_model

    def test_traffic_router_initialization(self, traffic_router):
        """测试流量路由器初始化"""
        assert traffic_router.canary_percentage == 20.0
        assert traffic_router.routing_strategy == 'weighted_random'
        assert traffic_router.sticky_sessions is True
        assert len(traffic_router.user_assignments) == 0

    def test_weighted_random_routing(self, traffic_router, models_mock):
        """测试加权随机路由"""
        canary_model, baseline_model = models_mock
        
        canary_count = 0
        baseline_count = 0
        total_requests = 1000
        
        for i in range(total_requests):
            user_id = f"user_{i}"
            selected_model = traffic_router.route_request(user_id, canary_model, baseline_model)
            
            if selected_model == canary_model:
                canary_count += 1
            else:
                baseline_count += 1
        
        # 验证流量分配接近配置的百分比
        canary_ratio = canary_count / total_requests
        expected_ratio = traffic_router.canary_percentage / 100.0
        
        assert abs(canary_ratio - expected_ratio) < 0.05  # 允许5%的偏差

    def test_sticky_session_consistency(self, traffic_router, models_mock):
        """测试粘性会话一致性"""
        canary_model, baseline_model = models_mock
        user_id = "consistent_user"
        
        # 同一用户的多次请求应该路由到同一模型
        first_model = traffic_router.route_request(user_id, canary_model, baseline_model)
        
        for _ in range(10):
            current_model = traffic_router.route_request(user_id, canary_model, baseline_model)
            assert current_model == first_model

    def test_traffic_percentage_adjustment(self, traffic_router, models_mock):
        """测试流量百分比调整"""
        canary_model, baseline_model = models_mock
        
        # 调整金丝雀流量百分比
        new_percentage = 50.0
        traffic_router.update_canary_percentage(new_percentage)
        
        assert traffic_router.canary_percentage == new_percentage
        
        # 验证新的流量分配
        canary_count = 0
        total_requests = 1000
        
        for i in range(total_requests):
            user_id = f"user_{i}"
            selected_model = traffic_router.route_request(user_id, canary_model, baseline_model)
            
            if selected_model == canary_model:
                canary_count += 1
        
        canary_ratio = canary_count / total_requests
        expected_ratio = new_percentage / 100.0
        
        assert abs(canary_ratio - expected_ratio) < 0.05

    def test_geographic_routing(self, traffic_router, models_mock):
        """测试地理位置路由"""
        canary_model, baseline_model = models_mock
        
        # 设置地理路由策略
        traffic_router.set_geographic_routing({
            'us-east': 30.0,    # 美国东部30%金丝雀流量
            'us-west': 20.0,    # 美国西部20%金丝雀流量
            'europe': 10.0,     # 欧洲10%金丝雀流量
            'asia': 5.0         # 亚洲5%金丝雀流量
        })
        
        # 测试不同地区的流量分配
        regions = ['us-east', 'us-west', 'europe', 'asia']
        
        for region in regions:
            canary_count = 0
            total_requests = 200
            
            for i in range(total_requests):
                user_id = f"{region}_user_{i}"
                selected_model = traffic_router.route_request(
                    user_id, canary_model, baseline_model, region=region
                )
                
                if selected_model == canary_model:
                    canary_count += 1
            
            canary_ratio = canary_count / total_requests
            expected_ratio = traffic_router.geographic_config[region] / 100.0
            
            assert abs(canary_ratio - expected_ratio) < 0.1  # 允许10%的偏差

    def test_traffic_routing_metrics(self, traffic_router, models_mock):
        """测试流量路由指标"""
        canary_model, baseline_model = models_mock
        
        # 生成一些流量
        for i in range(100):
            user_id = f"user_{i}"
            traffic_router.route_request(user_id, canary_model, baseline_model)
        
        # 获取路由指标
        routing_metrics = traffic_router.get_routing_metrics()
        
        assert isinstance(routing_metrics, dict)
        assert 'total_requests' in routing_metrics
        assert 'canary_requests' in routing_metrics
        assert 'baseline_requests' in routing_metrics
        assert 'canary_percentage_actual' in routing_metrics
        assert routing_metrics['total_requests'] == 100

    def test_load_balancing_fairness(self, traffic_router, models_mock):
        """测试负载均衡公平性"""
        canary_model, baseline_model = models_mock
        
        # 设置多个canary模型实例（模拟负载均衡）
        canary_instances = [Mock() for _ in range(3)]
        for i, instance in enumerate(canary_instances):
            instance.version = f"v2.0.0-instance-{i}"
        
        traffic_router.set_canary_instances(canary_instances)
        
        instance_counts = {instance.version: 0 for instance in canary_instances}
        
        # 路由到金丝雀模型的请求
        for i in range(300):
            user_id = f"user_{i}"
            selected_model = traffic_router.route_request(user_id, canary_instances, baseline_model)
            
            if selected_model in canary_instances:
                instance_counts[selected_model.version] += 1
        
        # 验证负载在实例间的分布相对均匀
        counts = list(instance_counts.values())
        if sum(counts) > 0:  # 只有当有金丝雀流量时才检查
            avg_count = sum(counts) / len(counts)
            for count in counts:
                assert abs(count - avg_count) / avg_count < 0.3  # 允许30%的偏差

    def test_emergency_traffic_cutoff(self, traffic_router, models_mock):
        """测试紧急流量切断"""
        canary_model, baseline_model = models_mock
        
        # 正常路由
        normal_user = "normal_user"
        selected_model = traffic_router.route_request(normal_user, canary_model, baseline_model)
        
        # 触发紧急切断
        traffic_router.emergency_cutoff_canary()
        
        # 所有流量应该路由到基线模型
        for i in range(50):
            user_id = f"emergency_user_{i}"
            selected_model = traffic_router.route_request(user_id, canary_model, baseline_model)
            assert selected_model == baseline_model
        
        # 恢复正常路由
        traffic_router.restore_normal_routing()
        
        # 验证恢复后的路由
        canary_count = 0
        for i in range(100):
            user_id = f"recovery_user_{i}"
            selected_model = traffic_router.route_request(user_id, canary_model, baseline_model)
            if selected_model == canary_model:
                canary_count += 1
        
        # 应该恢复到配置的金丝雀百分比
        canary_ratio = canary_count / 100
        expected_ratio = traffic_router.canary_percentage / 100.0
        assert abs(canary_ratio - expected_ratio) < 0.1