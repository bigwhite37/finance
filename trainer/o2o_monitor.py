"""
O2O训练监控和日志系统

扩展现有日志系统，支持O2O特定的监控需求，
实现训练阶段转换日志，记录关键决策点，
添加采样比例ρ(t)演化追踪，创建性能诊断报告。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class PhaseTransitionEvent:
    """阶段转换事件"""
    timestamp: datetime
    from_phase: str
    to_phase: str
    trigger: str
    metrics: Dict[str, Any]
    decision_factors: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


@dataclass
class SamplingRatioEvent:
    """采样比例事件"""
    timestamp: datetime
    phase: str
    iteration: int
    rho_value: float
    offline_samples: int
    online_samples: int
    total_samples: int
    importance_weights_stats: Dict[str, float]


@dataclass
class PerformanceDiagnostic:
    """性能诊断"""
    timestamp: datetime
    phase: str
    iteration: int
    value_function_analysis: Dict[str, Any]
    policy_divergence_metrics: Dict[str, Any]
    training_stability: Dict[str, Any]
    resource_usage: Dict[str, Any]
    recommendations: List[str]


@dataclass
class O2OMonitorConfig:
    """O2O监控配置"""
    log_dir: str = "logs/o2o_training"
    enable_phase_transition_logging: bool = True
    enable_sampling_ratio_tracking: bool = True
    enable_performance_diagnostics: bool = True
    enable_real_time_monitoring: bool = True
    
    # 日志级别和格式
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 监控频率
    diagnostic_interval: int = 100  # 每100次迭代进行一次诊断
    sampling_ratio_log_interval: int = 10  # 每10次迭代记录采样比例
    resource_monitor_interval: int = 60  # 每60秒监控资源使用
    
    # 数据保留
    max_events_per_type: int = 1000
    data_retention_days: int = 30
    
    # 可视化
    enable_plots: bool = True
    plot_update_interval: int = 300  # 每5分钟更新图表
    
    # 告警
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'policy_divergence': 0.1,
                'value_function_instability': 0.2,
                'memory_usage_percent': 90.0,
                'training_loss_spike': 2.0
            }


class O2OTrainingMonitor:
    """
    O2O训练监控和日志系统
    
    提供全面的O2O训练过程监控，包括：
    - 阶段转换日志
    - 采样比例演化追踪
    - 性能诊断报告
    - 实时监控和告警
    """
    
    def __init__(self, config: O2OMonitorConfig):
        """
        初始化O2O训练监控器
        
        Args:
            config: 监控配置
        """
        self.config = config
        
        # 创建日志目录
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 事件存储
        self.phase_transitions: deque = deque(maxlen=config.max_events_per_type)
        self.sampling_ratio_events: deque = deque(maxlen=config.max_events_per_type)
        self.performance_diagnostics: deque = deque(maxlen=config.max_events_per_type)
        
        # 实时数据
        self.current_metrics = {}
        self.resource_usage_history = deque(maxlen=1000)
        
        # 设置日志记录器
        self._setup_loggers()
        
        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 统计数据
        self.statistics = {
            'total_phase_transitions': 0,
            'successful_transitions': 0,
            'failed_transitions': 0,
            'total_diagnostics': 0,
            'alerts_triggered': 0
        }
        
        logger.info(f"O2O训练监控器初始化完成 - 日志目录: {self.log_dir}")
        
    def _setup_loggers(self):
        """设置专用日志记录器"""
        # 阶段转换日志器
        self.phase_logger = logging.getLogger('o2o.phase_transitions')
        phase_handler = logging.FileHandler(self.log_dir / 'phase_transitions.log')
        phase_handler.setFormatter(logging.Formatter(self.config.log_format))
        self.phase_logger.addHandler(phase_handler)
        self.phase_logger.setLevel(getattr(logging, self.config.log_level))
        
        # 采样比例日志器
        self.sampling_logger = logging.getLogger('o2o.sampling_ratio')
        sampling_handler = logging.FileHandler(self.log_dir / 'sampling_ratio.log')
        sampling_handler.setFormatter(logging.Formatter(self.config.log_format))
        self.sampling_logger.addHandler(sampling_handler)
        self.sampling_logger.setLevel(getattr(logging, self.config.log_level))
        
        # 性能诊断日志器
        self.diagnostic_logger = logging.getLogger('o2o.diagnostics')
        diagnostic_handler = logging.FileHandler(self.log_dir / 'performance_diagnostics.log')
        diagnostic_handler.setFormatter(logging.Formatter(self.config.log_format))
        self.diagnostic_logger.addHandler(diagnostic_handler)
        self.diagnostic_logger.setLevel(getattr(logging, self.config.log_level))
        
        # 告警日志器
        self.alert_logger = logging.getLogger('o2o.alerts')
        alert_handler = logging.FileHandler(self.log_dir / 'alerts.log')
        alert_handler.setFormatter(logging.Formatter(self.config.log_format))
        self.alert_logger.addHandler(alert_handler)
        self.alert_logger.setLevel(logging.WARNING)
        
    def log_phase_transition(self,
                           from_phase: str,
                           to_phase: str,
                           trigger: str,
                           metrics: Dict[str, Any],
                           decision_factors: Dict[str, Any],
                           success: bool = True,
                           error_message: Optional[str] = None):
        """
        记录阶段转换事件
        
        Args:
            from_phase: 源阶段
            to_phase: 目标阶段
            trigger: 触发器类型
            metrics: 相关指标
            decision_factors: 决策因素
            success: 是否成功
            error_message: 错误信息
        """
        if not self.config.enable_phase_transition_logging:
            return
        
        event = PhaseTransitionEvent(
            timestamp=datetime.now(),
            from_phase=from_phase,
            to_phase=to_phase,
            trigger=trigger,
            metrics=metrics,
            decision_factors=decision_factors,
            success=success,
            error_message=error_message
        )
        
        self.phase_transitions.append(event)
        
        # 更新统计
        self.statistics['total_phase_transitions'] += 1
        if success:
            self.statistics['successful_transitions'] += 1
        else:
            self.statistics['failed_transitions'] += 1
        
        # 记录日志
        log_message = (
            f"阶段转换: {from_phase} -> {to_phase} "
            f"(触发器: {trigger}, 成功: {success})"
        )
        
        if success:
            self.phase_logger.info(log_message)
        else:
            self.phase_logger.error(f"{log_message} - 错误: {error_message}")
        
        # 保存详细数据
        self._save_event_data('phase_transition', event)
        
        logger.info(log_message)
        
    def log_sampling_ratio(self,
                          phase: str,
                          iteration: int,
                          rho_value: float,
                          offline_samples: int,
                          online_samples: int,
                          importance_weights: Optional[np.ndarray] = None):
        """
        记录采样比例事件
        
        Args:
            phase: 训练阶段
            iteration: 迭代次数
            rho_value: 采样比例值
            offline_samples: 离线样本数
            online_samples: 在线样本数
            importance_weights: 重要性权重
        """
        if not self.config.enable_sampling_ratio_tracking:
            return
        
        if iteration % self.config.sampling_ratio_log_interval != 0:
            return
        
        # 计算重要性权重统计
        weights_stats = {}
        if importance_weights is not None:
            weights_stats = {
                'mean': float(np.mean(importance_weights)),
                'std': float(np.std(importance_weights)),
                'min': float(np.min(importance_weights)),
                'max': float(np.max(importance_weights)),
                'median': float(np.median(importance_weights))
            }
        
        event = SamplingRatioEvent(
            timestamp=datetime.now(),
            phase=phase,
            iteration=iteration,
            rho_value=rho_value,
            offline_samples=offline_samples,
            online_samples=online_samples,
            total_samples=offline_samples + online_samples,
            importance_weights_stats=weights_stats
        )
        
        self.sampling_ratio_events.append(event)
        
        # 记录日志
        log_message = (
            f"采样比例 ρ={rho_value:.4f} "
            f"(离线: {offline_samples}, 在线: {online_samples}, "
            f"迭代: {iteration})"
        )
        
        self.sampling_logger.info(log_message)
        
        # 保存详细数据
        self._save_event_data('sampling_ratio', event)
        
    def create_performance_diagnostic(self,
                                    phase: str,
                                    iteration: int,
                                    agent,
                                    training_metrics: Dict[str, Any],
                                    resource_info: Optional[Dict[str, Any]] = None):
        """
        创建性能诊断报告
        
        Args:
            phase: 训练阶段
            iteration: 迭代次数
            agent: 智能体实例
            training_metrics: 训练指标
            resource_info: 资源信息
        """
        if not self.config.enable_performance_diagnostics:
            return
        
        if iteration % self.config.diagnostic_interval != 0:
            return
        
        # 价值函数分析
        value_analysis = self._analyze_value_function(agent, training_metrics)
        
        # 策略发散指标
        policy_divergence = self._analyze_policy_divergence(agent, training_metrics)
        
        # 训练稳定性
        training_stability = self._analyze_training_stability(training_metrics)
        
        # 资源使用情况
        resource_usage = resource_info or self._get_resource_usage()
        
        # 生成建议
        recommendations = self._generate_recommendations(
            value_analysis, policy_divergence, training_stability, resource_usage
        )
        
        diagnostic = PerformanceDiagnostic(
            timestamp=datetime.now(),
            phase=phase,
            iteration=iteration,
            value_function_analysis=value_analysis,
            policy_divergence_metrics=policy_divergence,
            training_stability=training_stability,
            resource_usage=resource_usage,
            recommendations=recommendations
        )
        
        self.performance_diagnostics.append(diagnostic)
        self.statistics['total_diagnostics'] += 1
        
        # 记录日志
        self.diagnostic_logger.info(
            f"性能诊断 (迭代 {iteration}): "
            f"价值稳定性={value_analysis.get('stability_score', 0):.3f}, "
            f"策略发散={policy_divergence.get('kl_divergence', 0):.6f}"
        )
        
        # 检查告警条件
        self._check_alerts(diagnostic)
        
        # 保存详细数据
        self._save_event_data('performance_diagnostic', diagnostic)
        
    def _analyze_value_function(self, agent, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析价值函数"""
        analysis = {
            'stability_score': 0.0,
            'convergence_rate': 0.0,
            'value_range': {'min': 0.0, 'max': 0.0},
            'gradient_norm': 0.0
        }
        
        try:
            # 获取价值函数相关指标
            if 'value_loss' in training_metrics:
                recent_losses = training_metrics.get('value_loss_history', [training_metrics['value_loss']])
                if len(recent_losses) > 1:
                    # 计算稳定性分数（基于损失变化）
                    loss_changes = np.diff(recent_losses)
                    analysis['stability_score'] = 1.0 / (1.0 + np.std(loss_changes))
                    
                    # 收敛率
                    if len(recent_losses) >= 10:
                        analysis['convergence_rate'] = (recent_losses[0] - recent_losses[-1]) / len(recent_losses)
            
            # 梯度范数
            if hasattr(agent.network, 'critic_parameters'):
                total_norm = 0.0
                param_count = 0
                for param in agent.network.critic_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    analysis['gradient_norm'] = (total_norm / param_count) ** 0.5
            
        except Exception as e:
            logger.warning(f"价值函数分析失败: {e}")
        
        return analysis
        
    def _analyze_policy_divergence(self, agent, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析策略发散"""
        divergence = {
            'kl_divergence': 0.0,
            'action_entropy': 0.0,
            'policy_stability': 0.0,
            'parameter_change_norm': 0.0
        }
        
        try:
            # KL散度
            if 'kl_divergence' in training_metrics:
                divergence['kl_divergence'] = training_metrics['kl_divergence']
            
            # 动作熵
            if 'action_entropy' in training_metrics:
                divergence['action_entropy'] = training_metrics['action_entropy']
            
            # 策略稳定性（基于参数变化）
            if hasattr(agent, '_previous_policy_params'):
                current_params = {name: param.clone() for name, param in agent.network.actor_parameters()}
                param_changes = []
                
                for name, current_param in current_params.items():
                    if name in agent._previous_policy_params:
                        prev_param = agent._previous_policy_params[name]
                        change = torch.norm(current_param - prev_param).item()
                        param_changes.append(change)
                
                if param_changes:
                    divergence['parameter_change_norm'] = np.mean(param_changes)
                    divergence['policy_stability'] = 1.0 / (1.0 + divergence['parameter_change_norm'])
                
                # 更新参数历史
                agent._previous_policy_params = current_params
            else:
                # 初始化参数历史
                agent._previous_policy_params = {name: param.clone() for name, param in agent.network.actor_parameters()}
            
        except Exception as e:
            logger.warning(f"策略发散分析失败: {e}")
        
        return divergence
        
    def _analyze_training_stability(self, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析训练稳定性"""
        stability = {
            'loss_stability': 0.0,
            'gradient_stability': 0.0,
            'learning_rate_adaptation': 0.0,
            'convergence_trend': 'unknown'
        }
        
        try:
            # 损失稳定性
            if 'total_loss' in training_metrics:
                recent_losses = training_metrics.get('loss_history', [training_metrics['total_loss']])
                if len(recent_losses) > 5:
                    loss_std = np.std(recent_losses[-10:])
                    loss_mean = np.mean(recent_losses[-10:])
                    stability['loss_stability'] = 1.0 / (1.0 + loss_std / (loss_mean + 1e-8))
                    
                    # 收敛趋势
                    if len(recent_losses) >= 20:
                        early_avg = np.mean(recent_losses[-20:-10])
                        recent_avg = np.mean(recent_losses[-10:])
                        
                        if recent_avg < early_avg * 0.95:
                            stability['convergence_trend'] = 'improving'
                        elif recent_avg > early_avg * 1.05:
                            stability['convergence_trend'] = 'degrading'
                        else:
                            stability['convergence_trend'] = 'stable'
            
            # 梯度稳定性
            if 'gradient_norm' in training_metrics:
                grad_norms = training_metrics.get('gradient_norm_history', [training_metrics['gradient_norm']])
                if len(grad_norms) > 1:
                    grad_std = np.std(grad_norms[-10:])
                    grad_mean = np.mean(grad_norms[-10:])
                    stability['gradient_stability'] = 1.0 / (1.0 + grad_std / (grad_mean + 1e-8))
            
        except Exception as e:
            logger.warning(f"训练稳定性分析失败: {e}")
        
        return stability
        
    def _get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况"""
        import psutil
        
        try:
            # CPU和内存使用
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            resource_info = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'timestamp': datetime.now().isoformat()
            }
            
            # GPU使用（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_stats()
                    resource_info.update({
                        'gpu_memory_allocated_gb': gpu_memory.get('allocated_bytes.all.current', 0) / (1024**3),
                        'gpu_memory_reserved_gb': gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3),
                        'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                    })
            except:
                pass
            
            return resource_info
            
        except Exception as e:
            logger.warning(f"获取资源使用情况失败: {e}")
            return {'error': str(e)}
            
    def _generate_recommendations(self,
                                value_analysis: Dict[str, Any],
                                policy_divergence: Dict[str, Any],
                                training_stability: Dict[str, Any],
                                resource_usage: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 价值函数相关建议
        if value_analysis.get('stability_score', 1.0) < 0.5:
            recommendations.append("价值函数不稳定，建议降低学习率或增加正则化")
        
        if value_analysis.get('gradient_norm', 0.0) > 10.0:
            recommendations.append("梯度范数过大，建议启用梯度裁剪")
        
        # 策略发散相关建议
        if policy_divergence.get('kl_divergence', 0.0) > 0.1:
            recommendations.append("策略发散过大，建议增强信任域约束")
        
        if policy_divergence.get('action_entropy', 1.0) < 0.1:
            recommendations.append("动作熵过低，策略可能过于确定，建议增加探索")
        
        # 训练稳定性相关建议
        if training_stability.get('convergence_trend') == 'degrading':
            recommendations.append("训练性能下降，建议检查数据质量或调整超参数")
        
        if training_stability.get('loss_stability', 1.0) < 0.3:
            recommendations.append("损失不稳定，建议使用更小的批次大小或学习率")
        
        # 资源使用相关建议
        if resource_usage.get('memory_percent', 0) > 90:
            recommendations.append("内存使用率过高，建议减少批次大小或启用梯度累积")
        
        if resource_usage.get('cpu_percent', 0) > 95:
            recommendations.append("CPU使用率过高，建议优化数据加载或减少并行度")
        
        return recommendations
        
    def _check_alerts(self, diagnostic: PerformanceDiagnostic):
        """检查告警条件"""
        if not self.config.enable_alerts:
            return
        
        alerts = []
        thresholds = self.config.alert_thresholds
        
        # 策略发散告警
        kl_div = diagnostic.policy_divergence_metrics.get('kl_divergence', 0.0)
        if kl_div > thresholds['policy_divergence']:
            alerts.append(f"策略发散告警: KL散度 {kl_div:.6f} > {thresholds['policy_divergence']}")
        
        # 价值函数不稳定告警
        stability = diagnostic.value_function_analysis.get('stability_score', 1.0)
        if stability < (1.0 - thresholds['value_function_instability']):
            alerts.append(f"价值函数不稳定告警: 稳定性分数 {stability:.3f}")
        
        # 内存使用告警
        memory_percent = diagnostic.resource_usage.get('memory_percent', 0)
        if memory_percent > thresholds['memory_usage_percent']:
            alerts.append(f"内存使用告警: {memory_percent:.1f}% > {thresholds['memory_usage_percent']}%")
        
        # 记录告警
        for alert in alerts:
            self.alert_logger.warning(alert)
            self.statistics['alerts_triggered'] += 1
            
    def _save_event_data(self, event_type: str, event_data):
        """保存事件数据"""
        try:
            event_file = self.log_dir / f"{event_type}_events.jsonl"
            
            with open(event_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(event_data), f, default=str, ensure_ascii=False)
                f.write('\n')
                
        except Exception as e:
            logger.warning(f"保存事件数据失败: {e}")
            
    def start_real_time_monitoring(self):
        """启动实时监控"""
        if not self.config.enable_real_time_monitoring:
            return
        
        if self.monitoring_active:
            logger.warning("实时监控已经在运行")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("实时监控已启动")
        
    def stop_real_time_monitoring(self):
        """停止实时监控"""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("实时监控已停止")
        
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 收集资源使用情况
                resource_info = self._get_resource_usage()
                self.resource_usage_history.append(resource_info)
                
                # 更新当前指标
                self.current_metrics.update({
                    'last_update': datetime.now().isoformat(),
                    'resource_usage': resource_info
                })
                
                # 生成实时图表
                if self.config.enable_plots:
                    self._update_plots()
                
                time.sleep(self.config.resource_monitor_interval)
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(10)  # 错误后等待更长时间
                
    def _update_plots(self):
        """更新监控图表"""
        try:
            # 采样比例演化图
            if self.sampling_ratio_events:
                self._plot_sampling_ratio_evolution()
            
            # 性能指标图
            if self.performance_diagnostics:
                self._plot_performance_metrics()
            
            # 资源使用图
            if self.resource_usage_history:
                self._plot_resource_usage()
                
        except Exception as e:
            logger.warning(f"更新图表失败: {e}")
            
    def _plot_sampling_ratio_evolution(self):
        """绘制采样比例演化图"""
        events = list(self.sampling_ratio_events)
        if len(events) < 2:
            return
        
        iterations = [e.iteration for e in events]
        rho_values = [e.rho_value for e in events]
        offline_samples = [e.offline_samples for e in events]
        online_samples = [e.online_samples for e in events]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 采样比例演化
        ax1.plot(iterations, rho_values, 'b-', linewidth=2, label='ρ(t)')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('采样比例 ρ')
        ax1.set_title('采样比例演化')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 样本数量分布
        ax2.bar(iterations, offline_samples, alpha=0.7, label='离线样本', color='orange')
        ax2.bar(iterations, online_samples, bottom=offline_samples, alpha=0.7, label='在线样本', color='green')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('样本数量')
        ax2.set_title('样本数量分布')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'sampling_ratio_evolution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_performance_metrics(self):
        """绘制性能指标图"""
        diagnostics = list(self.performance_diagnostics)
        if len(diagnostics) < 2:
            return
        
        iterations = [d.iteration for d in diagnostics]
        stability_scores = [d.value_function_analysis.get('stability_score', 0) for d in diagnostics]
        kl_divergences = [d.policy_divergence_metrics.get('kl_divergence', 0) for d in diagnostics]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 价值函数稳定性
        ax1.plot(iterations, stability_scores, 'g-', linewidth=2, label='稳定性分数')
        ax1.set_xlabel('迭代次数')
        ax1.set_ylabel('稳定性分数')
        ax1.set_title('价值函数稳定性')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 策略发散
        ax2.plot(iterations, kl_divergences, 'r-', linewidth=2, label='KL散度')
        ax2.axhline(y=self.config.alert_thresholds['policy_divergence'], 
                   color='r', linestyle='--', alpha=0.7, label='告警阈值')
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('KL散度')
        ax2.set_title('策略发散指标')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'performance_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_resource_usage(self):
        """绘制资源使用图"""
        if len(self.resource_usage_history) < 2:
            return
        
        timestamps = [datetime.fromisoformat(r['timestamp']) for r in self.resource_usage_history]
        cpu_usage = [r.get('cpu_percent', 0) for r in self.resource_usage_history]
        memory_usage = [r.get('memory_percent', 0) for r in self.resource_usage_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # CPU使用率
        ax1.plot(timestamps, cpu_usage, 'b-', linewidth=2, label='CPU使用率')
        ax1.set_ylabel('CPU使用率 (%)')
        ax1.set_title('资源使用监控')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 内存使用率
        ax2.plot(timestamps, memory_usage, 'r-', linewidth=2, label='内存使用率')
        ax2.axhline(y=self.config.alert_thresholds['memory_usage_percent'], 
                   color='r', linestyle='--', alpha=0.7, label='告警阈值')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('内存使用率 (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'resource_usage.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def generate_summary_report(self) -> Dict[str, Any]:
        """生成监控摘要报告"""
        report = {
            'generation_time': datetime.now().isoformat(),
            'monitoring_period': {
                'start': min([e.timestamp for e in self.phase_transitions], default=datetime.now()).isoformat(),
                'end': max([e.timestamp for e in self.phase_transitions], default=datetime.now()).isoformat()
            },
            'statistics': self.statistics.copy(),
            'phase_transitions': {
                'total': len(self.phase_transitions),
                'success_rate': (self.statistics['successful_transitions'] / 
                               max(self.statistics['total_phase_transitions'], 1)) * 100,
                'recent_transitions': [
                    {
                        'from': e.from_phase,
                        'to': e.to_phase,
                        'trigger': e.trigger,
                        'success': e.success,
                        'timestamp': e.timestamp.isoformat()
                    }
                    for e in list(self.phase_transitions)[-5:]
                ]
            },
            'sampling_ratio': {
                'total_events': len(self.sampling_ratio_events),
                'current_rho': self.sampling_ratio_events[-1].rho_value if self.sampling_ratio_events else 0.0,
                'rho_evolution': [e.rho_value for e in list(self.sampling_ratio_events)[-20:]]
            },
            'performance': {
                'total_diagnostics': len(self.performance_diagnostics),
                'recent_stability': [
                    d.value_function_analysis.get('stability_score', 0)
                    for d in list(self.performance_diagnostics)[-10:]
                ],
                'recent_divergence': [
                    d.policy_divergence_metrics.get('kl_divergence', 0)
                    for d in list(self.performance_diagnostics)[-10:]
                ]
            },
            'alerts': {
                'total_triggered': self.statistics['alerts_triggered'],
                'alert_rate': (self.statistics['alerts_triggered'] / 
                              max(self.statistics['total_diagnostics'], 1)) * 100
            }
        }
        
        return report
        
    def export_monitoring_data(self, filepath: str):
        """导出监控数据"""
        export_data = {
            'config': asdict(self.config),
            'statistics': self.statistics,
            'phase_transitions': [asdict(e) for e in self.phase_transitions],
            'sampling_ratio_events': [asdict(e) for e in self.sampling_ratio_events],
            'performance_diagnostics': [asdict(e) for e in self.performance_diagnostics],
            'resource_usage_history': list(self.resource_usage_history),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"监控数据已导出: {filepath}")
        
    def cleanup_old_data(self):
        """清理旧数据"""
        cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
        
        # 清理阶段转换事件
        self.phase_transitions = deque(
            [e for e in self.phase_transitions if e.timestamp > cutoff_date],
            maxlen=self.config.max_events_per_type
        )
        
        # 清理采样比例事件
        self.sampling_ratio_events = deque(
            [e for e in self.sampling_ratio_events if e.timestamp > cutoff_date],
            maxlen=self.config.max_events_per_type
        )
        
        # 清理性能诊断
        self.performance_diagnostics = deque(
            [e for e in self.performance_diagnostics if e.timestamp > cutoff_date],
            maxlen=self.config.max_events_per_type
        )
        
        logger.info(f"已清理 {cutoff_date} 之前的监控数据")
        
    def __del__(self):
        """析构函数"""
        self.stop_real_time_monitoring()