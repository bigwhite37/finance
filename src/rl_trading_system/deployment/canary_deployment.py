"""
金丝雀部署系统实现
实现CanaryDeployment类和灰度发布流程，A/B测试框架和模型性能对比，自动回滚机制和部署安全控制
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import hashlib
import time
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import numpy as np
import pandas as pd
from scipy import stats
import requests


class DeploymentStatus(Enum):
    """部署状态枚举"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    success_rate: float
    error_rate: float
    avg_response_time: float
    throughput: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """初始化后验证"""
        if not (0 <= self.success_rate <= 1):
            raise ValueError("成功率必须在0-1之间")
        if not (0 <= self.error_rate <= 1):
            raise ValueError("错误率必须在0-1之间")
        if self.avg_response_time < 0:
            raise ValueError("响应时间不能为负数")
        if self.throughput < 0:
            raise ValueError("吞吐量不能为负数")


@dataclass
class DeploymentConfig:
    """部署配置数据类"""
    canary_percentage: float
    evaluation_period: int  # 秒
    success_threshold: float
    error_threshold: float
    performance_threshold: float
    rollback_threshold: float
    max_canary_duration: int  # 秒
    
    def __post_init__(self):
        """初始化后验证"""
        if not (0 <= self.canary_percentage <= 100):
            raise ValueError("金丝雀流量百分比必须在0-100之间")
        if not (0 <= self.success_threshold <= 1):
            raise ValueError("成功率阈值必须在0-1之间")
        if not (0 <= self.error_threshold <= 1):
            raise ValueError("错误率阈值必须在0-1之间")
        if self.evaluation_period <= 0:
            raise ValueError("评估周期必须为正数")
        if self.max_canary_duration <= 0:
            raise ValueError("最大金丝雀持续时间必须为正数")


class TrafficRouter:
    """流量路由器类"""
    
    def __init__(self, canary_percentage: float = 10.0, routing_strategy: str = 'weighted_random',
                 sticky_sessions: bool = True):
        """
        初始化流量路由器
        
        Args:
            canary_percentage: 金丝雀流量百分比
            routing_strategy: 路由策略
            sticky_sessions: 是否启用粘性会话
        """
        self.canary_percentage = canary_percentage
        self.routing_strategy = routing_strategy
        self.sticky_sessions = sticky_sessions
        self.user_assignments = {}
        self.routing_metrics = {
            'total_requests': 0,
            'canary_requests': 0,
            'baseline_requests': 0
        }
        self.geographic_config = {}
        self.canary_instances = []
        self.emergency_cutoff = False
        self._lock = threading.Lock()
    
    def route_request(self, user_id: str, canary_model: Any, baseline_model: Any, 
                     region: str = None) -> Any:
        """路由请求到合适的模型"""
        with self._lock:
            self.routing_metrics['total_requests'] += 1
            
            # 紧急切断检查
            if self.emergency_cutoff:
                self.routing_metrics['baseline_requests'] += 1
                return baseline_model
            
            # 粘性会话检查
            if self.sticky_sessions and user_id in self.user_assignments:
                assigned_model = self.user_assignments[user_id]
                if assigned_model == 'canary':
                    self.routing_metrics['canary_requests'] += 1
                    return canary_model
                else:
                    self.routing_metrics['baseline_requests'] += 1
                    return baseline_model
            
            # 确定金丝雀百分比
            effective_percentage = self.canary_percentage
            if region and region in self.geographic_config:
                effective_percentage = self.geographic_config[region]
            
            # 路由决策
            if self.routing_strategy == 'weighted_random':
                route_to_canary = np.random.random() < (effective_percentage / 100.0)
            elif self.routing_strategy == 'hash_based':
                user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
                route_to_canary = (user_hash % 100) < effective_percentage
            else:
                route_to_canary = np.random.random() < (effective_percentage / 100.0)
            
            # 记录分配并更新指标
            if route_to_canary:
                if self.sticky_sessions:
                    self.user_assignments[user_id] = 'canary'
                self.routing_metrics['canary_requests'] += 1
                
                # 如果有多个金丝雀实例，进行负载均衡
                if self.canary_instances:
                    instance_index = hash(user_id) % len(self.canary_instances)
                    return self.canary_instances[instance_index]
                return canary_model
            else:
                if self.sticky_sessions:
                    self.user_assignments[user_id] = 'baseline'
                self.routing_metrics['baseline_requests'] += 1
                return baseline_model
    
    def update_canary_percentage(self, new_percentage: float):
        """更新金丝雀流量百分比"""
        if not (0 <= new_percentage <= 100):
            raise ValueError("金丝雀百分比必须在0-100之间")
        
        with self._lock:
            self.canary_percentage = new_percentage
    
    def set_geographic_routing(self, geographic_config: Dict[str, float]):
        """设置地理位置路由配置"""
        for region, percentage in geographic_config.items():
            if not (0 <= percentage <= 100):
                raise ValueError(f"地区 {region} 的百分比必须在0-100之间")
        
        self.geographic_config = geographic_config.copy()
    
    def set_canary_instances(self, instances: List[Any]):
        """设置多个金丝雀模型实例"""
        self.canary_instances = instances.copy()
    
    def emergency_cutoff_canary(self):
        """紧急切断金丝雀流量"""
        with self._lock:
            self.emergency_cutoff = True
    
    def restore_normal_routing(self):
        """恢复正常路由"""
        with self._lock:
            self.emergency_cutoff = False
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """获取路由指标"""
        with self._lock:
            metrics = self.routing_metrics.copy()
            if metrics['total_requests'] > 0:
                metrics['canary_percentage_actual'] = (
                    metrics['canary_requests'] / metrics['total_requests'] * 100
                )
            else:
                metrics['canary_percentage_actual'] = 0.0
            return metrics


class ModelPerformanceComparator:
    """模型性能比较器类"""
    
    def __init__(self, comparison_window: int = 3600, min_samples_for_comparison: int = 100,
                 significance_threshold: float = 0.05):
        """
        初始化性能比较器
        
        Args:
            comparison_window: 比较时间窗口（秒）
            min_samples_for_comparison: 最小比较样本数
            significance_threshold: 显著性阈值
        """
        self.comparison_window = comparison_window
        self.min_samples_for_comparison = min_samples_for_comparison
        self.significance_threshold = significance_threshold
        self.model_a_metrics = deque(maxlen=10000)
        self.model_b_metrics = deque(maxlen=10000)
        self.latest_comparison = {}
        self._lock = threading.Lock()
    
    def add_model_a_metrics(self, metrics: PerformanceMetrics):
        """添加模型A的指标"""
        with self._lock:
            self.model_a_metrics.append(metrics)
    
    def add_model_b_metrics(self, metrics: PerformanceMetrics):
        """添加模型B的指标"""
        with self._lock:
            self.model_b_metrics.append(metrics)
    
    def compare_performance(self) -> Dict[str, Any]:
        """比较两个模型的性能"""
        with self._lock:
            if len(self.model_a_metrics) == 0 or len(self.model_b_metrics) == 0:
                return {'error': '缺少指标数据进行比较'}
            
            # 获取最近时间窗口内的指标
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(seconds=self.comparison_window)
            
            recent_a = [m for m in self.model_a_metrics if m.timestamp >= cutoff_time]
            recent_b = [m for m in self.model_b_metrics if m.timestamp >= cutoff_time]
            
            if len(recent_a) == 0 or len(recent_b) == 0:
                return {'error': '时间窗口内缺少指标数据'}
            
            # 计算平均指标
            avg_a = self._calculate_average_metrics(recent_a)
            avg_b = self._calculate_average_metrics(recent_b)
            
            # 计算差异
            comparison_result = {
                'success_rate_diff': avg_b['success_rate'] - avg_a['success_rate'],
                'error_rate_diff': avg_b['error_rate'] - avg_a['error_rate'],
                'response_time_diff': avg_b['avg_response_time'] - avg_a['avg_response_time'],
                'throughput_diff': avg_b['throughput'] - avg_a['throughput'],
                'accuracy_diff': avg_b['accuracy'] - avg_a['accuracy'],
                'model_a_samples': len(recent_a),
                'model_b_samples': len(recent_b),
                'comparison_timestamp': current_time
            }
            
            # 计算整体性能分数
            performance_score_a = self._calculate_performance_score(avg_a)
            performance_score_b = self._calculate_performance_score(avg_b)
            comparison_result['overall_performance_score'] = performance_score_b - performance_score_a
            
            # 统计显著性检验
            if len(recent_a) >= self.min_samples_for_comparison and len(recent_b) >= self.min_samples_for_comparison:
                significance_test = self.test_statistical_significance()
                comparison_result.update(significance_test)
            
            self.latest_comparison = comparison_result
            return comparison_result
    
    def _calculate_average_metrics(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, float]:
        """计算平均指标"""
        if not metrics_list:
            return {}
        
        return {
            'success_rate': np.mean([m.success_rate for m in metrics_list]),
            'error_rate': np.mean([m.error_rate for m in metrics_list]),
            'avg_response_time': np.mean([m.avg_response_time for m in metrics_list]),
            'throughput': np.mean([m.throughput for m in metrics_list]),
            'accuracy': np.mean([m.accuracy for m in metrics_list]),
            'precision': np.mean([m.precision for m in metrics_list]),
            'recall': np.mean([m.recall for m in metrics_list]),
            'f1_score': np.mean([m.f1_score for m in metrics_list])
        }
    
    def _calculate_performance_score(self, avg_metrics: Dict[str, float]) -> float:
        """计算性能分数"""
        # 加权性能分数
        score = (
            avg_metrics.get('success_rate', 0) * 0.3 +
            (1 - avg_metrics.get('error_rate', 1)) * 0.2 +
            (1 / max(avg_metrics.get('avg_response_time', 1), 0.001)) * 0.2 +
            avg_metrics.get('throughput', 0) / 1000.0 * 0.1 +
            avg_metrics.get('accuracy', 0) * 0.2
        )
        return score
    
    def test_statistical_significance(self) -> Dict[str, Any]:
        """测试统计显著性"""
        recent_a = list(self.model_a_metrics)[-self.min_samples_for_comparison:]
        recent_b = list(self.model_b_metrics)[-self.min_samples_for_comparison:]
        
        results = {}
        
        # 成功率比较
        success_rates_a = [m.success_rate for m in recent_a]
        success_rates_b = [m.success_rate for m in recent_b]
        stat, p_val = stats.ttest_ind(success_rates_a, success_rates_b)
        results['success_rate_significant'] = p_val < self.significance_threshold
        results['success_rate_p_value'] = p_val
        
        # 错误率比较
        error_rates_a = [m.error_rate for m in recent_a]
        error_rates_b = [m.error_rate for m in recent_b]
        stat, p_val = stats.ttest_ind(error_rates_a, error_rates_b)
        results['error_rate_significant'] = p_val < self.significance_threshold
        results['error_rate_p_value'] = p_val
        
        # 响应时间比较
        response_times_a = [m.avg_response_time for m in recent_a]
        response_times_b = [m.avg_response_time for m in recent_b]
        stat, p_val = stats.ttest_ind(response_times_a, response_times_b)
        results['response_time_significant'] = p_val < self.significance_threshold
        results['response_time_p_value'] = p_val
        
        return results
    
    def analyze_performance_trends(self) -> Dict[str, Any]:
        """分析性能趋势"""
        if len(self.model_a_metrics) < 10 or len(self.model_b_metrics) < 10:
            return {'error': '数据不足以进行趋势分析'}
        
        # 计算最近10个数据点的趋势
        recent_a = list(self.model_a_metrics)[-10:]
        recent_b = list(self.model_b_metrics)[-10:]
        
        # 模型A趋势
        a_success_rates = [m.success_rate for m in recent_a]
        a_trend_slope, _, _, p_val_a, _ = stats.linregress(range(len(a_success_rates)), a_success_rates)
        
        # 模型B趋势
        b_success_rates = [m.success_rate for m in recent_b]
        b_trend_slope, _, _, p_val_b, _ = stats.linregress(range(len(b_success_rates)), b_success_rates)
        
        return {
            'model_a_trend': {
                'slope': a_trend_slope,
                'direction': 'improving' if a_trend_slope > 0 else 'declining',
                'significance': p_val_a < self.significance_threshold
            },
            'model_b_trend': {
                'slope': b_trend_slope,
                'direction': 'improving' if b_trend_slope > 0 else 'declining',
                'significance': p_val_b < self.significance_threshold
            },
            'trend_significance': p_val_a < self.significance_threshold or p_val_b < self.significance_threshold
        }
    
    def detect_performance_degradation(self, model_name: str, degradation_threshold: float = 0.1) -> bool:
        """检测性能退化"""
        metrics_deque = self.model_a_metrics if model_name == 'model_a' else self.model_b_metrics
        
        if len(metrics_deque) < 20:
            return False
        
        # 比较最近10个数据点与之前10个数据点
        recent_metrics = list(metrics_deque)[-10:]
        baseline_metrics = list(metrics_deque)[-20:-10]
        
        recent_avg = np.mean([m.success_rate for m in recent_metrics])
        baseline_avg = np.mean([m.success_rate for m in baseline_metrics])
        
        # 如果最近的性能比基线差超过阈值，认为存在性能退化
        performance_drop = baseline_avg - recent_avg
        return performance_drop > degradation_threshold
    
    def calculate_confidence_intervals(self, model_name: str, confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """计算置信区间"""
        metrics_deque = self.model_a_metrics if model_name == 'model_a' else self.model_b_metrics
        
        if len(metrics_deque) < 30:
            return {'error': '样本量不足以计算置信区间'}
        
        metrics_list = list(metrics_deque)
        alpha = 1 - confidence_level
        
        results = {}
        
        # 成功率置信区间
        success_rates = [m.success_rate for m in metrics_list]
        mean_sr = np.mean(success_rates)
        sem_sr = stats.sem(success_rates)
        ci_sr = stats.t.interval(confidence_level, len(success_rates)-1, loc=mean_sr, scale=sem_sr)
        results['success_rate'] = {'lower': ci_sr[0], 'upper': ci_sr[1]}
        
        # 错误率置信区间
        error_rates = [m.error_rate for m in metrics_list]
        mean_er = np.mean(error_rates)
        sem_er = stats.sem(error_rates)
        ci_er = stats.t.interval(confidence_level, len(error_rates)-1, loc=mean_er, scale=sem_er)
        results['error_rate'] = {'lower': ci_er[0], 'upper': ci_er[1]}
        
        return results


class DeploymentSafetyController:
    """部署安全控制器类"""
    
    def __init__(self, max_error_rate: float = 0.05, max_response_time: float = 1.0,
                 min_success_rate: float = 0.90, circuit_breaker_threshold: int = 10,
                 recovery_check_interval: int = 300):
        """
        初始化安全控制器
        
        Args:
            max_error_rate: 最大错误率
            max_response_time: 最大响应时间
            min_success_rate: 最小成功率
            circuit_breaker_threshold: 熔断器阈值
            recovery_check_interval: 恢复检查间隔（秒）
        """
        self.max_error_rate = max_error_rate
        self.max_response_time = max_response_time
        self.min_success_rate = min_success_rate
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.recovery_check_interval = recovery_check_interval
        
        self.is_circuit_breaker_open = False
        self.failure_count = 0
        self.last_circuit_check = datetime.now()
        self.metrics_for_analysis = deque(maxlen=1000)
        self._lock = threading.Lock()
    
    def perform_safety_check(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """执行安全检查"""
        violations = []
        
        # 检查成功率
        if metrics.success_rate < self.min_success_rate:
            violations.append('success_rate')
        
        # 检查错误率
        if metrics.error_rate > self.max_error_rate:
            violations.append('error_rate')
        
        # 检查响应时间
        if metrics.avg_response_time > self.max_response_time:
            violations.append('response_time')
        
        # 更新失败计数和熔断器状态
        with self._lock:
            if violations:
                self.failure_count += 1
                if self.failure_count >= self.circuit_breaker_threshold:
                    self.is_circuit_breaker_open = True
                    self.last_circuit_check = datetime.now()
            else:
                self.failure_count = max(0, self.failure_count - 1)
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'failure_count': self.failure_count,
            'circuit_breaker_open': self.is_circuit_breaker_open
        }
    
    def should_rollback_deployment(self, deployment: Any, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """判断是否应该回滚部署"""
        safety_check = self.perform_safety_check(metrics)
        
        # 严重安全违规判断
        critical_violations = []
        if metrics.success_rate < 0.7:  # 成功率极低
            critical_violations.append('critical_success_rate')
        if metrics.error_rate > 0.2:    # 错误率极高
            critical_violations.append('critical_error_rate')
        if metrics.avg_response_time > 2.0:  # 响应时间极慢
            critical_violations.append('critical_response_time')
        
        should_rollback = (
            self.is_circuit_breaker_open or
            len(critical_violations) > 0 or
            len(safety_check['violations']) >= 2  # 多个指标同时违规
        )
        
        reason = ""
        if self.is_circuit_breaker_open:
            reason = "熔断器已开启，系统保护性回滚"
        elif critical_violations:
            reason = f"严重安全违规: {', '.join(critical_violations)}"
        elif len(safety_check['violations']) >= 2:
            reason = f"多项指标违规: {', '.join(safety_check['violations'])}"
        
        return {
            'should_rollback': should_rollback,
            'reason': reason,
            'safety_violations': safety_check['violations'],
            'critical_violations': critical_violations
        }
    
    def evaluate_traffic_adjustment(self, deployment: Any, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """评估流量调整建议"""
        safety_check = self.perform_safety_check(metrics)
        
        current_percentage = deployment.traffic_percentage
        
        if not safety_check['passed']:
            # 有安全违规，建议减少流量
            if len(safety_check['violations']) == 1:
                # 轻微违规，减少一半流量
                recommended_percentage = max(5.0, current_percentage * 0.5)
                action = 'reduce'
            else:
                # 严重违规，大幅减少流量
                recommended_percentage = max(1.0, current_percentage * 0.2)
                action = 'reduce'
        else:
            # 没有违规，可以考虑增加流量
            if current_percentage < 50.0:
                recommended_percentage = min(100.0, current_percentage * 1.2)
                action = 'increase'
            else:
                recommended_percentage = current_percentage
                action = 'maintain'
        
        return {
            'action': action,
            'recommended_percentage': recommended_percentage,
            'current_percentage': current_percentage,
            'safety_violations': safety_check['violations']
        }
    
    def attempt_circuit_recovery(self, metrics: PerformanceMetrics):
        """尝试熔断器恢复"""
        if not self.is_circuit_breaker_open:
            return
        
        current_time = datetime.now()
        if (current_time - self.last_circuit_check).total_seconds() < self.recovery_check_interval:
            return
        
        # 检查指标是否恢复正常
        safety_check = self.perform_safety_check(metrics)
        if safety_check['passed']:
            with self._lock:
                self.is_circuit_breaker_open = False
                self.failure_count = 0
                self.last_circuit_check = current_time
    
    def add_metrics_for_analysis(self, metrics: PerformanceMetrics):
        """添加指标用于分析"""
        with self._lock:
            self.metrics_for_analysis.append(metrics)
    
    def get_aggregated_safety_metrics(self) -> Dict[str, Any]:
        """获取聚合的安全指标"""
        if not self.metrics_for_analysis:
            return {}
        
        with self._lock:
            metrics_list = list(self.metrics_for_analysis)
        
        return {
            'avg_success_rate': np.mean([m.success_rate for m in metrics_list]),
            'avg_error_rate': np.mean([m.error_rate for m in metrics_list]),
            'avg_response_time': np.mean([m.avg_response_time for m in metrics_list]),
            'percentile_95_response_time': np.percentile([m.avg_response_time for m in metrics_list], 95),
            'min_success_rate': np.min([m.success_rate for m in metrics_list]),
            'max_error_rate': np.max([m.error_rate for m in metrics_list]),
            'sample_count': len(metrics_list)
        }
    
    def evaluate_emergency_stop(self, deployment: Any, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """评估紧急停止"""
        critical_violations = []
        
        # 极端情况检查
        if metrics.success_rate < 0.5:
            critical_violations.append('极低成功率')
        if metrics.error_rate > 0.3:
            critical_violations.append('极高错误率')
        if metrics.avg_response_time > 3.0:
            critical_violations.append('极慢响应时间')
        if metrics.throughput < 100:
            critical_violations.append('极低吞吐量')
        
        emergency_stop = len(critical_violations) >= 2
        
        return {
            'emergency_stop': emergency_stop,
            'critical_violations': critical_violations,
            'timestamp': datetime.now()
        }


class RollbackManager:
    """回滚管理器类"""
    
    def __init__(self, rollback_timeout: int = 300, verification_checks: int = 5,
                 health_check_interval: int = 30):
        """
        初始化回滚管理器
        
        Args:
            rollback_timeout: 回滚超时时间（秒）
            verification_checks: 验证检查次数
            health_check_interval: 健康检查间隔（秒）
        """
        self.rollback_timeout = rollback_timeout
        self.verification_checks = verification_checks
        self.health_check_interval = health_check_interval
        self.rollback_history = []
        self.auto_rollback_conditions = {}
        self._lock = threading.Lock()
    
    def execute_rollback(self, deployment: Any, reason: str) -> Dict[str, Any]:
        """执行回滚"""
        rollback_id = str(uuid.uuid4())
        rollback_start = datetime.now()
        
        rollback_record = {
            'rollback_id': rollback_id,
            'deployment_id': getattr(deployment, 'deployment_id', 'unknown'),
            'reason': reason,
            'timestamp': rollback_start,
            'status': 'in_progress'
        }
        
        try:
            # 执行流量回滚
            self._execute_traffic_rollback(deployment)
            
            # 更新部署状态
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.traffic_percentage = 0.0
            
            rollback_record['status'] = 'success'
            rollback_record['completion_time'] = datetime.now()
            rollback_record['duration'] = (rollback_record['completion_time'] - rollback_start).total_seconds()
            
            with self._lock:
                self.rollback_history.append(rollback_record)
            
            return {
                'success': True,
                'rollback_id': rollback_id,
                'timestamp': rollback_start,
                'duration': rollback_record['duration']
            }
        
        except Exception as e:
            rollback_record['status'] = 'failed'
            rollback_record['error'] = str(e)
            rollback_record['completion_time'] = datetime.now()
            
            with self._lock:
                self.rollback_history.append(rollback_record)
            
            return {
                'success': False,
                'rollback_id': rollback_id,
                'error': str(e),
                'timestamp': rollback_start
            }
    
    def _execute_traffic_rollback(self, deployment: Any):
        """执行流量回滚"""
        # 模拟流量回滚过程
        if hasattr(deployment, 'traffic_router'):
            deployment.traffic_router.emergency_cutoff_canary()
        
        # 等待流量切换完成
        time.sleep(1)
    
    def execute_partial_rollback(self, deployment: Any, target_percentage: float, reason: str) -> Dict[str, Any]:
        """执行部分回滚"""
        if not (0 <= target_percentage <= 100):
            raise ValueError("目标百分比必须在0-100之间")
        
        rollback_id = str(uuid.uuid4())
        rollback_start = datetime.now()
        
        rollback_record = {
            'rollback_id': rollback_id,
            'deployment_id': getattr(deployment, 'deployment_id', 'unknown'),
            'reason': reason,
            'type': 'partial',
            'original_percentage': deployment.traffic_percentage,
            'target_percentage': target_percentage,
            'timestamp': rollback_start,
            'status': 'success'
        }
        
        # 更新流量百分比
        deployment.traffic_percentage = target_percentage
        if hasattr(deployment, 'traffic_router'):
            deployment.traffic_router.update_canary_percentage(target_percentage)
        
        rollback_record['completion_time'] = datetime.now()
        
        with self._lock:
            self.rollback_history.append(rollback_record)
        
        return {
            'success': True,
            'rollback_id': rollback_id,
            'new_traffic_percentage': target_percentage,
            'timestamp': rollback_start
        }
    
    def verify_rollback(self, rollback_id: str) -> Dict[str, Any]:
        """验证回滚是否成功"""
        rollback_record = None
        with self._lock:
            for record in self.rollback_history:
                if record['rollback_id'] == rollback_id:
                    rollback_record = record
                    break
        
        if not rollback_record:
            return {'error': f'未找到回滚记录: {rollback_id}'}
        
        # 执行验证检查
        checks_passed = 0
        for i in range(self.verification_checks):
            if self._perform_health_check():
                checks_passed += 1
            time.sleep(self.health_check_interval)
        
        verification_success = checks_passed >= (self.verification_checks * 0.8)  # 80%的检查通过
        
        return {
            'verified': verification_success,
            'checks_passed': checks_passed,
            'total_checks': self.verification_checks,
            'rollback_id': rollback_id
        }
    
    def _perform_health_check(self) -> bool:
        """执行健康检查"""
        # 模拟健康检查
        # 在实际实现中，这里会检查系统各项指标
        return True
    
    def get_rollback_history(self, deployment_id: str = None) -> List[Dict[str, Any]]:
        """获取回滚历史"""
        with self._lock:
            if deployment_id:
                return [record for record in self.rollback_history 
                       if record.get('deployment_id') == deployment_id]
            return self.rollback_history.copy()
    
    def set_auto_rollback_conditions(self, conditions: Dict[str, float]):
        """设置自动回滚条件"""
        self.auto_rollback_conditions = conditions.copy()
    
    def should_trigger_auto_rollback(self, metrics: PerformanceMetrics) -> bool:
        """判断是否应该触发自动回滚"""
        if not self.auto_rollback_conditions:
            return False
        
        # 检查各项条件
        if 'max_error_rate' in self.auto_rollback_conditions:
            if metrics.error_rate > self.auto_rollback_conditions['max_error_rate']:
                return True
        
        if 'min_success_rate' in self.auto_rollback_conditions:
            if metrics.success_rate < self.auto_rollback_conditions['min_success_rate']:
                return True
        
        if 'max_response_time' in self.auto_rollback_conditions:
            if metrics.avg_response_time > self.auto_rollback_conditions['max_response_time']:
                return True
        
        return False


class ABTestFramework:
    """A/B测试框架类"""
    
    def __init__(self, model_a: Any, model_b: Any, traffic_split: float = 0.5,
                 minimum_sample_size: int = 1000, confidence_level: float = 0.95,
                 test_duration: int = 86400):
        """
        初始化A/B测试框架
        
        Args:
            model_a: 模型A（通常是基线模型）
            model_b: 模型B（通常是新模型）
            traffic_split: 流量分割比例
            minimum_sample_size: 最小样本量
            confidence_level: 置信水平
            test_duration: 测试持续时间（秒）
        """
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        self.minimum_sample_size = minimum_sample_size
        self.confidence_level = confidence_level
        self.test_duration = test_duration
        
        self.experiment_data = []
        self.user_assignments = {}
        self.test_status = "pending"
        self.start_time = None
        self._lock = threading.Lock()
    
    def route_traffic(self, user_id: str) -> Any:
        """路由流量到A或B模型"""
        with self._lock:
            # 检查用户是否已有分配
            if user_id in self.user_assignments:
                return self.user_assignments[user_id]
            
            # 基于用户ID的哈希进行一致性分配
            user_hash = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
            if (user_hash % 100) / 100.0 < self.traffic_split:
                assigned_model = self.model_a
            else:
                assigned_model = self.model_b
            
            self.user_assignments[user_id] = assigned_model
            return assigned_model
    
    def record_result(self, user_id: str, model: Any, prediction: float, actual: float):
        """记录实验结果"""
        with self._lock:
            self.experiment_data.append({
                'user_id': user_id,
                'model': model,
                'prediction': prediction,
                'actual': actual,
                'timestamp': datetime.now()
            })
    
    def get_experiment_data(self) -> List[Dict[str, Any]]:
        """获取实验数据"""
        with self._lock:
            return self.experiment_data.copy()
    
    def has_sufficient_sample_size(self) -> bool:
        """检查是否有足够的样本量"""
        with self._lock:
            model_a_samples = sum(1 for record in self.experiment_data if record['model'] == self.model_a)
            model_b_samples = sum(1 for record in self.experiment_data if record['model'] == self.model_b)
            
            return (model_a_samples >= self.minimum_sample_size and 
                   model_b_samples >= self.minimum_sample_size)
    
    def calculate_statistical_significance(self) -> Dict[str, Any]:
        """计算统计显著性"""
        if not self.has_sufficient_sample_size():
            return {'error': '样本量不足'}
        
        with self._lock:
            # 分离A和B组的数据
            a_results = [record['actual'] for record in self.experiment_data if record['model'] == self.model_a]
            b_results = [record['actual'] for record in self.experiment_data if record['model'] == self.model_b]
        
        # 执行t检验
        stat, p_value = stats.ttest_ind(a_results, b_results)
        
        # 计算效应大小（Cohen's d）
        pooled_std = np.sqrt(((len(a_results) - 1) * np.var(a_results, ddof=1) + 
                             (len(b_results) - 1) * np.var(b_results, ddof=1)) / 
                            (len(a_results) + len(b_results) - 2))
        cohens_d = (np.mean(b_results) - np.mean(a_results)) / pooled_std
        
        # 计算置信区间
        alpha = 1 - self.confidence_level
        degrees_freedom = len(a_results) + len(b_results) - 2
        t_critical = stats.t.ppf(1 - alpha/2, degrees_freedom)
        
        mean_diff = np.mean(b_results) - np.mean(a_results)
        se_diff = pooled_std * np.sqrt(1/len(a_results) + 1/len(b_results))
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        return {
            'p_value': p_value,
            'is_significant': p_value < (1 - self.confidence_level),
            'effect_size': cohens_d,
            'confidence_interval': {'lower': ci_lower, 'upper': ci_upper},
            'mean_difference': mean_diff,
            'model_a_mean': np.mean(a_results),
            'model_b_mean': np.mean(b_results),
            'model_a_samples': len(a_results),
            'model_b_samples': len(b_results)
        }
    
    def is_test_complete(self) -> bool:
        """检查测试是否完成"""
        if not self.start_time:
            return False
        
        time_elapsed = (datetime.now() - self.start_time).total_seconds()
        return time_elapsed >= self.test_duration and self.has_sufficient_sample_size()
    
    def determine_winner(self) -> Optional[Any]:
        """确定获胜模型"""
        if not self.is_test_complete():
            return None
        
        significance_result = self.calculate_statistical_significance()
        if 'error' in significance_result:
            return None
        
        if significance_result['is_significant']:
            if significance_result['mean_difference'] > 0:
                return self.model_b  # B模型更好
            else:
                return self.model_a  # A模型更好
        else:
            # 无显著差异，返回当前基线模型
            return self.model_a
    
    def start_test(self):
        """开始测试"""
        self.start_time = datetime.now()
        self.test_status = "running"
    
    def stop_test(self):
        """停止测试"""
        self.test_status = "completed"


class CanaryDeployment:
    """金丝雀部署主类"""
    
    def __init__(self, canary_model: Any, baseline_model: Any, config: DeploymentConfig):
        """
        初始化金丝雀部署
        
        Args:
            canary_model: 金丝雀模型
            baseline_model: 基线模型
            config: 部署配置
        """
        self.canary_model = canary_model
        self.baseline_model = baseline_model
        self.config = config
        
        self.deployment_id = str(uuid.uuid4())
        self.status = DeploymentStatus.PENDING
        self.start_time = None
        self.traffic_percentage = 0.0
        self.deployment_history = []
        
        # 初始化组件
        self.traffic_router = TrafficRouter(canary_percentage=config.canary_percentage)
        self.performance_comparator = ModelPerformanceComparator()
        self.safety_controller = DeploymentSafetyController()
        self.rollback_manager = RollbackManager()
        
        # 指标收集
        self.canary_metrics_history = deque(maxlen=1000)
        self.baseline_metrics_history = deque(maxlen=1000)
        
        self._lock = threading.Lock()
    
    def start_deployment(self):
        """启动金丝雀部署"""
        if self.status != DeploymentStatus.PENDING:
            raise ValueError("部署已经在运行中")
        
        with self._lock:
            self.status = DeploymentStatus.ACTIVE
            self.start_time = datetime.now()
            self.traffic_percentage = self.config.canary_percentage
            
            # 更新流量路由器
            self.traffic_router.update_canary_percentage(self.config.canary_percentage)
            
            # 记录部署历史
            self.deployment_history.append({
                'action': 'start_deployment',
                'timestamp': self.start_time,
                'traffic_percentage': self.traffic_percentage
            })
    
    def update_canary_metrics(self, metrics: PerformanceMetrics):
        """更新金丝雀模型指标"""
        with self._lock:
            self.canary_metrics_history.append(metrics)
            self.performance_comparator.add_model_b_metrics(metrics)
            self.safety_controller.add_metrics_for_analysis(metrics)
    
    def update_baseline_metrics(self, metrics: PerformanceMetrics):
        """更新基线模型指标"""
        with self._lock:
            self.baseline_metrics_history.append(metrics)
            self.performance_comparator.add_model_a_metrics(metrics)
    
    def evaluate_success_criteria(self) -> bool:
        """评估成功标准"""
        if not self.canary_metrics_history:
            return False
        
        # 获取最近的指标
        recent_metrics = list(self.canary_metrics_history)[-10:]  # 最近10个数据点
        
        # 计算平均指标
        avg_success_rate = np.mean([m.success_rate for m in recent_metrics])
        avg_error_rate = np.mean([m.error_rate for m in recent_metrics])
        
        # 检查是否满足成功标准
        success_criteria_met = (
            avg_success_rate >= self.config.success_threshold and
            avg_error_rate <= self.config.error_threshold
        )
        
        return success_criteria_met
    
    def should_trigger_rollback(self) -> bool:
        """判断是否应该触发回滚"""
        # 性能比较检查（即使没有指标历史也可以进行）
        if self.performance_comparator.latest_comparison:
            comparison = self.performance_comparator.latest_comparison
            # 检查多个性能指标是否表明需要回滚
            performance_degradation = (
                comparison.get('overall_performance_score', 0) < -self.config.rollback_threshold or
                comparison.get('performance_improvement', 0) < -self.config.rollback_threshold or
                (comparison.get('success_rate_diff', 0) > self.config.rollback_threshold and
                 comparison.get('error_rate_diff', 0) > self.config.rollback_threshold)
            )
            if performance_degradation:
                return True
        
        # 如果没有指标历史，只能基于性能比较判断
        if not self.canary_metrics_history:
            return False
        
        # 获取最新指标
        latest_metrics = self.canary_metrics_history[-1]
        
        # 安全控制器检查
        rollback_decision = self.safety_controller.should_rollback_deployment(self, latest_metrics)
        if rollback_decision['should_rollback']:
            return True
        
        return False
    
    def increase_traffic(self, step_size: float = 10.0):
        """增加流量百分比"""
        if self.status != DeploymentStatus.ACTIVE:
            raise ValueError("部署未处于活跃状态")
        
        new_percentage = min(100.0, self.traffic_percentage + step_size)
        
        with self._lock:
            self.traffic_percentage = new_percentage
            self.traffic_router.update_canary_percentage(new_percentage)
            
            self.deployment_history.append({
                'action': 'increase_traffic',
                'timestamp': datetime.now(),
                'traffic_percentage': new_percentage,
                'step_size': step_size
            })
    
    def complete_deployment(self):
        """完成部署"""
        if self.status != DeploymentStatus.ACTIVE:
            raise ValueError("部署未处于活跃状态")
        
        with self._lock:
            self.status = DeploymentStatus.COMPLETED
            self.traffic_percentage = 100.0
            
            self.deployment_history.append({
                'action': 'complete_deployment',
                'timestamp': datetime.now(),
                'traffic_percentage': 100.0
            })
    
    def rollback_deployment(self, reason: str):
        """回滚部署"""
        rollback_result = self.rollback_manager.execute_rollback(self, reason)
        
        with self._lock:
            self.status = DeploymentStatus.ROLLED_BACK
            self.traffic_percentage = 0.0
            
            self.deployment_history.append({
                'action': 'rollback_deployment',
                'timestamp': datetime.now(),
                'reason': reason,
                'rollback_id': rollback_result.get('rollback_id'),
                'traffic_percentage': 0.0
            })
    
    def is_deployment_timeout(self) -> bool:
        """检查部署是否超时"""
        if not self.start_time:
            return False
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        return elapsed_time > self.config.max_canary_duration
    
    def get_metrics_history(self) -> List[PerformanceMetrics]:
        """获取指标历史"""
        with self._lock:
            return list(self.canary_metrics_history)
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        return {
            'deployment_id': self.deployment_id,
            'status': self.status.value,
            'start_time': self.start_time,
            'traffic_percentage': self.traffic_percentage,
            'canary_model_version': getattr(self.canary_model, 'version', 'unknown'),
            'baseline_model_version': getattr(self.baseline_model, 'version', 'unknown'),
            'deployment_duration': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'metrics_count': len(self.canary_metrics_history),
            'latest_comparison': self.performance_comparator.latest_comparison
        }