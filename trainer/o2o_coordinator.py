"""
O2O训练流程协调器

管理完整的离线到在线(O2O)强化学习训练流程，
协调离线预训练、热身微调和在线学习三个阶段，
实现阶段转换逻辑和异常处理机制。
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

from trainer.offline_pretrainer import OfflinePretrainer, OfflinePretrainerConfig
from trainer.warmup_finetuner import WarmUpFinetuner, WarmUpFinetunerConfig
from trainer.online_learner import OnlineLearner, OnlineLearnerConfig
from trainer.checkpoint_manager import CheckpointManager, CheckpointConfig
from trainer.o2o_monitor import O2OTrainingMonitor, O2OMonitorConfig
from trainer.adaptive_hyperparameter_tuner import (
    AdaptiveHyperparameterTuner, PerformanceMetrics
)
from data.offline_dataset import OfflineDataset
from sampler.mixture_sampler import MixtureSampler, MixtureSamplerConfig
from buffers.online_replay_buffer import OnlineReplayBuffer
from sampler.mixture_sampler import MixtureSampler
from rl_agent.cvar_ppo_agent import CVaRPPOAgent
from rl_agent.trading_environment import TradingEnvironment
from monitoring.value_drift_monitor import ValueDriftMonitor

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """训练阶段枚举"""
    OFFLINE = "offline"
    WARMUP = "warmup"
    ONLINE = "online"
    COMPLETED = "completed"
    FAILED = "failed"


class TransitionTrigger(Enum):
    """阶段转换触发器"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    TIME_BASED = "time_based"


@dataclass
class O2OCoordinatorConfig:
    """O2O协调器配置"""
    # 阶段配置
    offline_config: OfflinePretrainerConfig
    warmup_config: WarmUpFinetunerConfig
    online_config: OnlineLearnerConfig
    
    # 转换条件
    auto_transition: bool = True
    offline_completion_threshold: float = 1e-3
    warmup_completion_threshold: float = 1e-4
    max_phase_duration: Dict[str, int] = None  # 每个阶段的最大持续时间（秒）
    
    # 检查点和恢复
    checkpoint_config: CheckpointConfig = None
    auto_save_interval: int = 300  # 自动保存间隔（秒）
    enable_recovery: bool = True
    max_recovery_attempts: int = 3
    
    # 监控和日志
    enable_monitoring: bool = True
    monitor_config: O2OMonitorConfig = None
    log_level: str = "INFO"
    performance_tracking: bool = True
    
    # 自适应超参数调优
    enable_adaptive_tuning: bool = True
    adaptive_tuning_config: Dict[str, Any] = None
    
    # 失败处理
    failure_recovery_strategy: str = "rollback"  # "rollback", "restart", "skip"
    max_failures_per_phase: int = 2
    
    def __post_init__(self):
        if self.max_phase_duration is None:
            self.max_phase_duration = {
                "offline": 3600,  # 1小时
                "warmup": 1800,   # 30分钟
                "online": 7200    # 2小时
            }
        if self.checkpoint_config is None:
            self.checkpoint_config = CheckpointConfig(
                base_dir="checkpoints/o2o_training"
            )
        if self.monitor_config is None:
            self.monitor_config = O2OMonitorConfig(
                log_dir="logs/o2o_training",
                log_level=self.log_level
            )
        if self.adaptive_tuning_config is None:
            self.adaptive_tuning_config = {
                'initial_lr': 3e-4,
                'min_lr': 1e-6,
                'max_lr': 1e-2,
                'initial_rho': 0.2,
                'min_rho': 0.1,
                'max_rho': 0.9,
                'initial_beta': 1.0,
                'target_kl': 0.01,
                'enable_hyperparameter_search': False
            }


@dataclass
class PhaseResult:
    """阶段执行结果"""
    phase: TrainingPhase
    status: str
    start_time: datetime
    end_time: datetime
    duration: float
    metrics: Dict[str, Any]
    checkpoint_path: Optional[str] = None
    error_message: Optional[str] = None
    recovery_attempts: int = 0


@dataclass
class TrainingState:
    """训练状态"""
    current_phase: TrainingPhase
    phase_start_time: datetime
    total_start_time: datetime
    phase_results: List[PhaseResult]
    failure_counts: Dict[str, int]
    recovery_attempts: int
    last_checkpoint: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.failure_counts is None:
            self.failure_counts = {phase.value: 0 for phase in TrainingPhase}
        if self.metadata is None:
            self.metadata = {}


class O2OTrainingCoordinator:
    """
    O2O训练流程协调器
    
    管理完整的离线到在线强化学习训练流程，
    协调三个训练阶段，处理阶段转换和异常恢复。
    """
    
    def __init__(self, 
                 agent: CVaRPPOAgent,
                 environment: TradingEnvironment,
                 config: O2OCoordinatorConfig):
        """
        初始化O2O训练协调器
        
        Args:
            agent: CVaR-PPO智能体
            environment: 交易环境
            config: 协调器配置
        """
        self.agent = agent
        self.environment = environment
        self.config = config
        
        # 训练组件
        self.offline_pretrainer = OfflinePretrainer(agent, config.offline_config)
        self.warmup_finetuner = WarmUpFinetuner(agent, config.warmup_config)
        self.online_learner = OnlineLearner(agent, config.online_config)
        
        # 数据组件
        self.offline_dataset: Optional[OfflineDataset] = None
        self.online_buffer: Optional[OnlineReplayBuffer] = None
        self.mixture_sampler: Optional[MixtureSampler] = None
        
        # 监控组件
        self.drift_monitor: Optional[ValueDriftMonitor] = None
        self.o2o_monitor: Optional[O2OTrainingMonitor] = None
        
        if config.enable_monitoring:
            self.drift_monitor = ValueDriftMonitor({})
            self.o2o_monitor = O2OTrainingMonitor(config.monitor_config)
        
        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(config.checkpoint_config)
        
        # 自适应超参数调优器
        self.adaptive_tuner: Optional[AdaptiveHyperparameterTuner] = None
        if config.enable_adaptive_tuning:
            self.adaptive_tuner = AdaptiveHyperparameterTuner(config.adaptive_tuning_config)
            logger.info("启用自适应超参数调优")
        
        # 训练状态
        self.training_state = TrainingState(
            current_phase=TrainingPhase.OFFLINE,
            phase_start_time=datetime.now(),
            total_start_time=datetime.now(),
            phase_results=[],
            failure_counts={phase.value: 0 for phase in TrainingPhase},
            recovery_attempts=0
        )
        
        # 设置日志级别
        logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
        
        logger.info(f"O2O训练协调器初始化完成 - 配置: {config}")
        
    def run_full_training(self,
                         offline_dataset: OfflineDataset,
                         online_buffer: OnlineReplayBuffer,
                         mixture_sampler: Optional[MixtureSampler] = None) -> Dict[str, Any]:
        """
        运行完整的O2O训练流程
        
        Args:
            offline_dataset: 离线数据集
            online_buffer: 在线数据缓冲区
            mixture_sampler: 混合采样器（可选）
            
        Returns:
            训练结果统计
        """
        logger.info("开始完整的O2O训练流程...")
        
        # 设置数据组件
        self.offline_dataset = offline_dataset
        self.online_buffer = online_buffer
        self.mixture_sampler = mixture_sampler or MixtureSampler(
            offline_dataset, 
            online_buffer, 
            MixtureSamplerConfig()
        )
        
        # 重置训练状态
        self._reset_training_state()
        
        try:
            # 启动监控
            if self.o2o_monitor:
                self.o2o_monitor.start_real_time_monitoring()
            
            # 尝试从检查点恢复
            if self.config.enable_recovery:
                self._attempt_recovery()
            
            # 执行训练阶段
            while self.training_state.current_phase != TrainingPhase.COMPLETED:
                phase_result = self._execute_current_phase()
                
                # 处理阶段结果
                if phase_result.status == "success":
                    self._handle_phase_success(phase_result)
                else:
                    if not self._handle_phase_failure(phase_result):
                        break  # 无法恢复，终止训练
                
                # 自动保存
                self._auto_save_checkpoint()
            
            # 生成最终结果
            final_result = self._generate_final_result()
            
        except Exception as e:
            logger.error(f"O2O训练过程中发生未处理的异常: {e}")
            final_result = self._handle_critical_failure(str(e))
        
        finally:
            # 停止监控
            if self.o2o_monitor:
                self.o2o_monitor.stop_real_time_monitoring()
                
                # 生成最终监控报告
                if final_result.get('status') != 'critical_failure':
                    final_result['monitoring_report'] = self.o2o_monitor.generate_summary_report()
        
        logger.info(f"O2O训练流程完成: {final_result['status']}")
        return final_result
        
    def _reset_training_state(self):
        """重置训练状态"""
        self.training_state = TrainingState(
            current_phase=TrainingPhase.OFFLINE,
            phase_start_time=datetime.now(),
            total_start_time=datetime.now(),
            phase_results=[],
            failure_counts={phase.value: 0 for phase in TrainingPhase},
            recovery_attempts=0
        )
        
    def _execute_current_phase(self) -> PhaseResult:
        """执行当前训练阶段"""
        phase = self.training_state.current_phase
        start_time = datetime.now()
        
        logger.info(f"开始执行训练阶段: {phase.value}")
        
        try:
            # 检查阶段超时
            if self._check_phase_timeout():
                return PhaseResult(
                    phase=phase,
                    status="timeout",
                    start_time=start_time,
                    end_time=datetime.now(),
                    duration=0,
                    metrics={},
                    error_message="阶段执行超时"
                )
            
            # 执行具体阶段
            if phase == TrainingPhase.OFFLINE:
                result = self._execute_offline_phase()
            elif phase == TrainingPhase.WARMUP:
                result = self._execute_warmup_phase()
            elif phase == TrainingPhase.ONLINE:
                result = self._execute_online_phase()
            else:
                raise ValueError(f"未知的训练阶段: {phase}")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return PhaseResult(
                phase=phase,
                status="success" if result.get('status') in ['completed', 'converged'] else "failed",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                metrics=result,
                checkpoint_path=result.get('checkpoint_path')
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"阶段 {phase.value} 执行失败: {e}")
            
            return PhaseResult(
                phase=phase,
                status="error",
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                metrics={},
                error_message=str(e)
            )
            
    def _execute_offline_phase(self) -> Dict[str, Any]:
        """执行离线预训练阶段"""
        logger.info("执行离线预训练阶段...")
        
        # 设置环境为离线模式
        self.environment.set_mode('offline')
        
        # 执行离线预训练
        result = self.offline_pretrainer.pretrain(
            self.offline_dataset,
            checkpoint_dir=os.path.join(self.config.checkpoint_dir, "offline")
        )
        
        # 保存检查点路径
        if result.get('final_checkpoint'):
            result['checkpoint_path'] = result['final_checkpoint']
        
        return result
        
    def _execute_warmup_phase(self) -> Dict[str, Any]:
        """执行热身微调阶段"""
        logger.info("执行热身微调阶段...")
        
        # 设置环境为在线模式以收集最近数据
        self.environment.set_mode('online')
        
        # 执行热身微调
        result = self.warmup_finetuner.finetune(
            self.online_buffer,
            days=self.config.warmup_config.warmup_days,
            checkpoint_dir=os.path.join(self.config.checkpoint_dir, "warmup")
        )
        
        return result
        
    def _execute_online_phase(self) -> Dict[str, Any]:
        """执行在线学习阶段"""
        logger.info("执行在线学习阶段...")
        
        # 确保环境为在线模式
        self.environment.set_mode('online')
        
        # 执行在线学习
        result = self.online_learner.learn(
            self.mixture_sampler,
            self.online_buffer,
            drift_monitor=self.drift_monitor,
            checkpoint_dir=os.path.join(self.config.checkpoint_dir, "online")
        )
        
        return result
        
    def _handle_phase_success(self, phase_result: PhaseResult):
        """处理阶段成功"""
        # 记录阶段结果
        self.training_state.phase_results.append(phase_result)
        
        # 重置失败计数
        self.training_state.failure_counts[phase_result.phase.value] = 0
        
        # 更新检查点
        if phase_result.checkpoint_path:
            self.training_state.last_checkpoint = phase_result.checkpoint_path
        
        # 决定下一阶段
        next_phase = self._determine_next_phase(phase_result)
        
        if next_phase != self.training_state.current_phase:
            # 记录阶段转换
            if self.o2o_monitor:
                self.o2o_monitor.log_phase_transition(
                    from_phase=self.training_state.current_phase.value,
                    to_phase=next_phase.value,
                    trigger="automatic" if self.config.auto_transition else "manual",
                    metrics=phase_result.metrics,
                    decision_factors=self._get_transition_decision_factors(phase_result),
                    success=True
                )
            
            logger.info(f"阶段转换: {self.training_state.current_phase.value} -> {next_phase.value}")
            self.training_state.current_phase = next_phase
            self.training_state.phase_start_time = datetime.now()
        
    def _handle_phase_failure(self, phase_result: PhaseResult) -> bool:
        """
        处理阶段失败
        
        Returns:
            是否可以继续训练
        """
        # 记录失败结果
        self.training_state.phase_results.append(phase_result)
        
        # 增加失败计数
        phase_name = phase_result.phase.value
        self.training_state.failure_counts[phase_name] += 1
        
        logger.warning(
            f"阶段 {phase_name} 失败 "
            f"({self.training_state.failure_counts[phase_name]}/{self.config.max_failures_per_phase}): "
            f"{phase_result.error_message}"
        )
        
        # 检查是否超过最大失败次数
        if self.training_state.failure_counts[phase_name] >= self.config.max_failures_per_phase:
            logger.error(f"阶段 {phase_name} 失败次数过多，终止训练")
            self.training_state.current_phase = TrainingPhase.FAILED
            return False
        
        # 尝试恢复
        return self._attempt_phase_recovery(phase_result)
        
    def _attempt_phase_recovery(self, phase_result: PhaseResult) -> bool:
        """尝试阶段恢复"""
        strategy = self.config.failure_recovery_strategy
        
        logger.info(f"尝试使用策略 '{strategy}' 恢复阶段 {phase_result.phase.value}")
        
        if strategy == "rollback":
            return self._rollback_recovery(phase_result)
        elif strategy == "restart":
            return self._restart_recovery(phase_result)
        elif strategy == "skip":
            return self._skip_recovery(phase_result)
        else:
            logger.error(f"未知的恢复策略: {strategy}")
            return False
            
    def _rollback_recovery(self, phase_result: PhaseResult) -> bool:
        """回滚恢复策略"""
        # 尝试加载最近的检查点
        if self.training_state.last_checkpoint and os.path.exists(self.training_state.last_checkpoint):
            try:
                self._load_checkpoint(self.training_state.last_checkpoint)
                logger.info(f"成功回滚到检查点: {self.training_state.last_checkpoint}")
                return True
            except Exception as e:
                logger.error(f"检查点回滚失败: {e}")
        
        # 如果没有检查点，尝试重新开始当前阶段
        logger.info("没有可用检查点，重新开始当前阶段")
        self.training_state.phase_start_time = datetime.now()
        return True
        
    def _restart_recovery(self, phase_result: PhaseResult) -> bool:
        """重启恢复策略"""
        # 重新开始当前阶段
        logger.info(f"重新开始阶段: {phase_result.phase.value}")
        self.training_state.phase_start_time = datetime.now()
        return True
        
    def _skip_recovery(self, phase_result: PhaseResult) -> bool:
        """跳过恢复策略"""
        # 跳到下一阶段
        current_phase = phase_result.phase
        
        if current_phase == TrainingPhase.OFFLINE:
            next_phase = TrainingPhase.WARMUP
        elif current_phase == TrainingPhase.WARMUP:
            next_phase = TrainingPhase.ONLINE
        elif current_phase == TrainingPhase.ONLINE:
            next_phase = TrainingPhase.COMPLETED
        else:
            return False
        
        logger.warning(f"跳过失败阶段 {current_phase.value}，转到 {next_phase.value}")
        self.training_state.current_phase = next_phase
        self.training_state.phase_start_time = datetime.now()
        return True
        
    def _determine_next_phase(self, phase_result: PhaseResult) -> TrainingPhase:
        """确定下一个训练阶段"""
        current_phase = phase_result.phase
        
        if not self.config.auto_transition:
            return current_phase  # 手动模式，保持当前阶段
        
        # 自动转换逻辑
        if current_phase == TrainingPhase.OFFLINE:
            if self._check_offline_completion(phase_result):
                return TrainingPhase.WARMUP
        elif current_phase == TrainingPhase.WARMUP:
            if self._check_warmup_completion(phase_result):
                return TrainingPhase.ONLINE
        elif current_phase == TrainingPhase.ONLINE:
            if self._check_online_completion(phase_result):
                return TrainingPhase.COMPLETED
        
        return current_phase
        
    def _check_offline_completion(self, phase_result: PhaseResult) -> bool:
        """检查离线预训练完成条件"""
        metrics = phase_result.metrics
        
        # 检查损失阈值
        final_loss = metrics.get('best_loss', float('inf'))
        if final_loss <= self.config.offline_completion_threshold:
            return True
        
        # 检查训练状态
        if metrics.get('status') == 'completed':
            return True
        
        return False
        
    def _check_warmup_completion(self, phase_result: PhaseResult) -> bool:
        """检查热身微调完成条件"""
        metrics = phase_result.metrics
        
        # 检查收敛状态
        if metrics.get('status') == 'converged':
            return True
        
        # 检查损失阈值
        final_loss = metrics.get('final_loss', float('inf'))
        if final_loss <= self.config.warmup_completion_threshold:
            return True
        
        return False
        
    def _check_online_completion(self, phase_result: PhaseResult) -> bool:
        """检查在线学习完成条件"""
        metrics = phase_result.metrics
        
        # 检查学习状态
        if metrics.get('status') in ['completed', 'converged']:
            return True
        
        # 检查性能指标
        if self.config.performance_tracking:
            performance_metrics = metrics.get('performance_metrics', {})
            if performance_metrics.get('stable_performance', False):
                return True
        
        return False
        
    def _check_phase_timeout(self) -> bool:
        """检查阶段是否超时"""
        current_phase = self.training_state.current_phase.value
        max_duration = self.config.max_phase_duration.get(current_phase, float('inf'))
        
        elapsed = (datetime.now() - self.training_state.phase_start_time).total_seconds()
        
        if elapsed > max_duration:
            logger.warning(f"阶段 {current_phase} 超时: {elapsed:.1f}s > {max_duration}s")
            return True
        
        return False
        
    def _auto_save_checkpoint(self):
        """自动保存检查点"""
        if not hasattr(self, '_last_save_time'):
            self._last_save_time = time.time()
        
        current_time = time.time()
        if current_time - self._last_save_time >= self.config.auto_save_interval:
            self._save_training_state()
            self._last_save_time = current_time
            
    def _save_training_state(self):
        """保存训练状态"""
        # 使用检查点管理器保存训练状态
        try:
            # 获取当前模型状态
            model_state = self.agent.network.state_dict()
            
            # 获取优化器状态
            optimizer_states = {}
            if hasattr(self.agent, 'optimizer') and self.agent.optimizer:
                optimizer_states['main'] = self.agent.optimizer.state_dict()
            if hasattr(self.agent, 'actor_optimizer') and self.agent.actor_optimizer:
                optimizer_states['actor'] = self.agent.actor_optimizer.state_dict()
            if hasattr(self.agent, 'critic_optimizer') and self.agent.critic_optimizer:
                optimizer_states['critic'] = self.agent.critic_optimizer.state_dict()
            
            # 准备训练指标
            training_metrics = {
                'current_phase': self.training_state.current_phase.value,
                'phase_results': [asdict(result) for result in self.training_state.phase_results],
                'failure_counts': self.training_state.failure_counts,
                'recovery_attempts': self.training_state.recovery_attempts
            }
            
            # 准备配置信息
            config_data = {
                'coordinator_config': asdict(self.config),
                'agent_config': self.agent.config if hasattr(self.agent, 'config') else {}
            }
            
            # 创建训练快照
            snapshot_id = self.checkpoint_manager.create_training_snapshot(
                phase=self.training_state.current_phase.value,
                iteration=len(self.training_state.phase_results),
                model_state=model_state,
                optimizer_states=optimizer_states,
                training_metrics=training_metrics,
                config=config_data,
                metadata={
                    'phase_start_time': self.training_state.phase_start_time.isoformat(),
                    'total_start_time': self.training_state.total_start_time.isoformat(),
                    'last_checkpoint': self.training_state.last_checkpoint
                }
            )
            
            # 更新最后检查点
            self.training_state.last_checkpoint = snapshot_id
            
            logger.debug(f"训练状态已保存: {snapshot_id}")
            
        except Exception as e:
            logger.error(f"保存训练状态失败: {e}")
            # 回退到原始方法
            self._save_training_state_fallback()
            
    def _save_training_state_fallback(self):
        """保存训练状态的回退方法"""
        state_file = os.path.join(self.config.checkpoint_config.base_dir, "training_state.json")
        
        # 准备可序列化的状态数据
        state_data = {
            'current_phase': self.training_state.current_phase.value,
            'phase_start_time': self.training_state.phase_start_time.isoformat(),
            'total_start_time': self.training_state.total_start_time.isoformat(),
            'failure_counts': self.training_state.failure_counts,
            'recovery_attempts': self.training_state.recovery_attempts,
            'last_checkpoint': self.training_state.last_checkpoint,
            'metadata': self.training_state.metadata,
            'phase_results': [
                {
                    'phase': result.phase.value,
                    'status': result.status,
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'duration': result.duration,
                    'metrics': result.metrics,
                    'checkpoint_path': result.checkpoint_path,
                    'error_message': result.error_message,
                    'recovery_attempts': result.recovery_attempts
                }
                for result in self.training_state.phase_results
            ]
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"训练状态已保存（回退方法）: {state_file}")
        
    def _attempt_recovery(self):
        """尝试从检查点恢复"""
        try:
            # 尝试从检查点管理器恢复最新快照
            latest_snapshot = self.checkpoint_manager.get_latest_snapshot()
            
            if latest_snapshot is None:
                logger.info("没有找到训练快照，从头开始训练")
                return
            
            # 加载完整快照数据
            snapshot_data = self.checkpoint_manager.load_training_snapshot(latest_snapshot.snapshot_id)
            
            # 恢复模型状态
            self.agent.network.load_state_dict(snapshot_data.model_state)
            
            # 恢复优化器状态
            if snapshot_data.optimizer_states:
                if 'main' in snapshot_data.optimizer_states and hasattr(self.agent, 'optimizer'):
                    self.agent.optimizer.load_state_dict(snapshot_data.optimizer_states['main'])
                if 'actor' in snapshot_data.optimizer_states and hasattr(self.agent, 'actor_optimizer'):
                    self.agent.actor_optimizer.load_state_dict(snapshot_data.optimizer_states['actor'])
                if 'critic' in snapshot_data.optimizer_states and hasattr(self.agent, 'critic_optimizer'):
                    self.agent.critic_optimizer.load_state_dict(snapshot_data.optimizer_states['critic'])
            
            # 恢复训练状态
            training_metrics = snapshot_data.training_metrics
            self.training_state.current_phase = TrainingPhase(training_metrics['current_phase'])
            self.training_state.failure_counts = training_metrics.get('failure_counts', {})
            self.training_state.recovery_attempts = training_metrics.get('recovery_attempts', 0)
            
            # 恢复元数据
            metadata = snapshot_data.metadata
            if 'phase_start_time' in metadata:
                self.training_state.phase_start_time = datetime.fromisoformat(metadata['phase_start_time'])
            if 'total_start_time' in metadata:
                self.training_state.total_start_time = datetime.fromisoformat(metadata['total_start_time'])
            
            self.training_state.last_checkpoint = latest_snapshot.snapshot_id
            
            # 恢复阶段结果
            self.training_state.phase_results = []
            for result_data in training_metrics.get('phase_results', []):
                phase_result = PhaseResult(
                    phase=TrainingPhase(result_data['phase']),
                    status=result_data['status'],
                    start_time=datetime.fromisoformat(result_data['start_time']),
                    end_time=datetime.fromisoformat(result_data['end_time']),
                    duration=result_data['duration'],
                    metrics=result_data['metrics'],
                    checkpoint_path=result_data.get('checkpoint_path'),
                    error_message=result_data.get('error_message'),
                    recovery_attempts=result_data.get('recovery_attempts', 0)
                )
                self.training_state.phase_results.append(phase_result)
            
            logger.info(f"成功从快照恢复训练状态: {latest_snapshot.snapshot_id}, 当前阶段: {self.training_state.current_phase.value}")
            
        except Exception as e:
            logger.error(f"从快照恢复失败: {e}")
            # 尝试回退方法
            self._attempt_recovery_fallback()
            
    def _attempt_recovery_fallback(self):
        """恢复的回退方法"""
        state_file = os.path.join(self.config.checkpoint_config.base_dir, "training_state.json")
        
        if not os.path.exists(state_file):
            logger.info("没有找到训练状态文件，从头开始训练")
            return
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 恢复训练状态
            self.training_state.current_phase = TrainingPhase(state_data['current_phase'])
            self.training_state.phase_start_time = datetime.fromisoformat(state_data['phase_start_time'])
            self.training_state.total_start_time = datetime.fromisoformat(state_data['total_start_time'])
            self.training_state.failure_counts = state_data['failure_counts']
            self.training_state.recovery_attempts = state_data['recovery_attempts']
            self.training_state.last_checkpoint = state_data.get('last_checkpoint')
            self.training_state.metadata = state_data.get('metadata', {})
            
            # 恢复阶段结果
            self.training_state.phase_results = []
            for result_data in state_data.get('phase_results', []):
                phase_result = PhaseResult(
                    phase=TrainingPhase(result_data['phase']),
                    status=result_data['status'],
                    start_time=datetime.fromisoformat(result_data['start_time']),
                    end_time=datetime.fromisoformat(result_data['end_time']),
                    duration=result_data['duration'],
                    metrics=result_data['metrics'],
                    checkpoint_path=result_data.get('checkpoint_path'),
                    error_message=result_data.get('error_message'),
                    recovery_attempts=result_data.get('recovery_attempts', 0)
                )
                self.training_state.phase_results.append(phase_result)
            
            logger.info(f"成功恢复训练状态（回退方法），当前阶段: {self.training_state.current_phase.value}")
            
        except Exception as e:
            logger.error(f"恢复训练状态失败: {e}")
            logger.info("将从头开始训练")
            
    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.agent.device)
        
        # 加载网络参数
        self.agent.network.load_state_dict(checkpoint['network_state_dict'])
        
        # 加载优化器状态（如果存在）
        if 'optimizer_state_dict' in checkpoint and hasattr(self.agent, 'optimizer'):
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"模型检查点已加载: {checkpoint_path}")
        
    def _generate_final_result(self) -> Dict[str, Any]:
        """生成最终训练结果"""
        total_duration = (datetime.now() - self.training_state.total_start_time).total_seconds()
        
        # 统计各阶段结果
        phase_summary = {}
        for phase in TrainingPhase:
            if phase in [TrainingPhase.COMPLETED, TrainingPhase.FAILED]:
                continue
            
            phase_results = [r for r in self.training_state.phase_results if r.phase == phase]
            if phase_results:
                successful_results = [r for r in phase_results if r.status == "success"]
                phase_summary[phase.value] = {
                    'attempts': len(phase_results),
                    'successes': len(successful_results),
                    'total_duration': sum(r.duration for r in phase_results),
                    'final_status': phase_results[-1].status if phase_results else 'not_executed'
                }
            else:
                phase_summary[phase.value] = {
                    'attempts': 0,
                    'successes': 0,
                    'total_duration': 0,
                    'final_status': 'not_executed'
                }
        
        # 确定最终状态
        if self.training_state.current_phase == TrainingPhase.COMPLETED:
            final_status = "completed"
        elif self.training_state.current_phase == TrainingPhase.FAILED:
            final_status = "failed"
        else:
            final_status = "interrupted"
        
        return {
            'status': final_status,
            'total_duration': total_duration,
            'final_phase': self.training_state.current_phase.value,
            'phase_summary': phase_summary,
            'total_failures': sum(self.training_state.failure_counts.values()),
            'recovery_attempts': self.training_state.recovery_attempts,
            'checkpoint_dir': self.config.checkpoint_dir,
            'training_history': [asdict(result) for result in self.training_state.phase_results]
        }
        
    def _handle_critical_failure(self, error_message: str) -> Dict[str, Any]:
        """处理关键失败"""
        self.training_state.current_phase = TrainingPhase.FAILED
        
        return {
            'status': 'critical_failure',
            'error_message': error_message,
            'total_duration': (datetime.now() - self.training_state.total_start_time).total_seconds(),
            'final_phase': TrainingPhase.FAILED.value,
            'training_history': [asdict(result) for result in self.training_state.phase_results]
        }
        
    def manual_phase_transition(self, target_phase: TrainingPhase) -> bool:
        """
        手动阶段转换
        
        Args:
            target_phase: 目标阶段
            
        Returns:
            转换是否成功
        """
        if self.training_state.current_phase == target_phase:
            logger.info(f"已经处于目标阶段: {target_phase.value}")
            return True
        
        # 验证转换的合理性
        valid_transitions = {
            TrainingPhase.OFFLINE: [TrainingPhase.WARMUP],
            TrainingPhase.WARMUP: [TrainingPhase.ONLINE, TrainingPhase.OFFLINE],
            TrainingPhase.ONLINE: [TrainingPhase.WARMUP, TrainingPhase.COMPLETED]
        }
        
        current_phase = self.training_state.current_phase
        if target_phase not in valid_transitions.get(current_phase, []):
            logger.error(f"无效的阶段转换: {current_phase.value} -> {target_phase.value}")
            return False
        
        # 执行转换
        logger.info(f"手动阶段转换: {current_phase.value} -> {target_phase.value}")
        self.training_state.current_phase = target_phase
        self.training_state.phase_start_time = datetime.now()
        
        return True
        
    def get_training_status(self) -> Dict[str, Any]:
        """获取当前训练状态"""
        current_time = datetime.now()
        total_elapsed = (current_time - self.training_state.total_start_time).total_seconds()
        phase_elapsed = (current_time - self.training_state.phase_start_time).total_seconds()
        
        return {
            'current_phase': self.training_state.current_phase.value,
            'total_elapsed_time': total_elapsed,
            'phase_elapsed_time': phase_elapsed,
            'completed_phases': len([r for r in self.training_state.phase_results if r.status == "success"]),
            'total_failures': sum(self.training_state.failure_counts.values()),
            'phase_failures': self.training_state.failure_counts,
            'last_checkpoint': self.training_state.last_checkpoint,
            'recovery_attempts': self.training_state.recovery_attempts
        }
        
    def stop_training(self):
        """停止训练"""
        logger.info("收到停止训练请求")
        self.training_state.current_phase = TrainingPhase.COMPLETED
        self._save_training_state()
        
    def pause_training(self):
        """暂停训练"""
        logger.info("暂停训练")
        self._save_training_state()
        
    def resume_training(self):
        """恢复训练"""
        logger.info("恢复训练")
        self._attempt_recovery()
        
    def save_model_version(self, 
                          phase: str, 
                          performance_metrics: Dict[str, float],
                          tags: Optional[List[str]] = None) -> str:
        """
        保存模型版本
        
        Args:
            phase: 训练阶段
            performance_metrics: 性能指标
            tags: 版本标签
            
        Returns:
            版本ID
        """
        model_state = self.agent.network.state_dict()
        config_data = {
            'coordinator_config': asdict(self.config),
            'agent_config': self.agent.config if hasattr(self.agent, 'config') else {}
        }
        
        version_id = self.checkpoint_manager.save_model_version(
            model_state=model_state,
            phase=phase,
            performance_metrics=performance_metrics,
            config=config_data,
            tags=tags,
            parent_version=self.training_state.last_checkpoint
        )
        
        logger.info(f"模型版本已保存: {version_id}")
        return version_id
        
    def load_model_version(self, version_id: str):
        """
        加载模型版本
        
        Args:
            version_id: 版本ID
        """
        model_data = self.checkpoint_manager.load_model_version(version_id)
        
        # 加载模型状态
        self.agent.network.load_state_dict(model_data['model_state_dict'])
        
        logger.info(f"模型版本已加载: {version_id}")
        
    def list_model_versions(self, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出模型版本
        
        Args:
            phase: 阶段过滤
            
        Returns:
            版本列表
        """
        versions = self.checkpoint_manager.list_versions(phase=phase)
        
        return [
            {
                'version_id': v.version_id,
                'phase': v.phase,
                'timestamp': v.timestamp.isoformat(),
                'performance_metrics': v.performance_metrics,
                'tags': v.tags,
                'model_hash': v.model_hash
            }
            for v in versions
        ]
        
    def create_backup(self) -> str:
        """
        创建完整备份
        
        Returns:
            备份路径
        """
        backup_path = self.checkpoint_manager.create_backup()
        logger.info(f"完整备份已创建: {backup_path}")
        return backup_path
        
    def restore_from_backup(self, backup_path: str):
        """
        从备份恢复
        
        Args:
            backup_path: 备份路径
        """
        self.checkpoint_manager.restore_from_backup(backup_path)
        logger.info(f"已从备份恢复: {backup_path}")
        
        # 重新加载训练状态
        self._attempt_recovery()
        
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """
        获取检查点统计信息
        
        Returns:
            统计信息
        """
        storage_stats = self.checkpoint_manager.get_storage_statistics()
        training_stats = self.get_training_status()
        
        return {
            'storage': storage_stats,
            'training': training_stats,
            'checkpoint_config': asdict(self.config.checkpoint_config)
        }
        
    def cleanup_old_checkpoints(self):
        """清理旧检查点"""
        # 触发检查点管理器的清理
        self.checkpoint_manager._cleanup_old_versions()
        self.checkpoint_manager._cleanup_old_snapshots()
        logger.info("旧检查点清理完成")
        
    def export_training_history(self, filepath: str):
        """
        导出训练历史
        
        Args:
            filepath: 导出文件路径
        """
        history_data = {
            'training_state': {
                'current_phase': self.training_state.current_phase.value,
                'total_start_time': self.training_state.total_start_time.isoformat(),
                'phase_start_time': self.training_state.phase_start_time.isoformat(),
                'failure_counts': self.training_state.failure_counts,
                'recovery_attempts': self.training_state.recovery_attempts
            },
            'phase_results': [asdict(result) for result in self.training_state.phase_results],
            'checkpoint_statistics': self.get_checkpoint_statistics(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"训练历史已导出: {filepath}")
        
    def _get_transition_decision_factors(self, phase_result: PhaseResult) -> Dict[str, Any]:
        """获取阶段转换决策因素"""
        factors = {
            'phase_duration': phase_result.duration,
            'final_metrics': phase_result.metrics,
            'completion_status': phase_result.status,
            'auto_transition_enabled': self.config.auto_transition
        }
        
        # 添加阶段特定的决策因素
        if phase_result.phase == TrainingPhase.OFFLINE:
            factors.update({
                'loss_threshold_met': phase_result.metrics.get('best_loss', float('inf')) <= self.config.offline_completion_threshold,
                'training_completed': phase_result.metrics.get('status') == 'completed'
            })
        elif phase_result.phase == TrainingPhase.WARMUP:
            factors.update({
                'convergence_achieved': phase_result.metrics.get('status') == 'converged',
                'loss_threshold_met': phase_result.metrics.get('final_loss', float('inf')) <= self.config.warmup_completion_threshold
            })
        elif phase_result.phase == TrainingPhase.ONLINE:
            factors.update({
                'performance_stable': phase_result.metrics.get('performance_metrics', {}).get('stable_performance', False),
                'learning_completed': phase_result.metrics.get('status') in ['completed', 'converged']
            })
        
        return factors
        
    def log_sampling_ratio_update(self, 
                                 phase: str, 
                                 iteration: int, 
                                 rho_value: float,
                                 offline_samples: int,
                                 online_samples: int,
                                 importance_weights: Optional[np.ndarray] = None):
        """记录采样比例更新"""
        if self.o2o_monitor:
            self.o2o_monitor.log_sampling_ratio(
                phase=phase,
                iteration=iteration,
                rho_value=rho_value,
                offline_samples=offline_samples,
                online_samples=online_samples,
                importance_weights=importance_weights
            )
            
    def create_performance_diagnostic(self,
                                    phase: str,
                                    iteration: int,
                                    training_metrics: Dict[str, Any]):
        """创建性能诊断报告"""
        if self.o2o_monitor:
            self.o2o_monitor.create_performance_diagnostic(
                phase=phase,
                iteration=iteration,
                agent=self.agent,
                training_metrics=training_metrics
            )
            
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.o2o_monitor:
            return {'monitoring_enabled': False}
        
        return self.o2o_monitor.generate_summary_report()
        
    def export_monitoring_data(self, filepath: str):
        """导出监控数据"""
        if self.o2o_monitor:
            self.o2o_monitor.export_monitoring_data(filepath)
            logger.info(f"监控数据已导出: {filepath}")
        else:
            logger.warning("监控未启用，无法导出监控数据")
            
    def cleanup_monitoring_data(self):
        """清理监控数据"""
        if self.o2o_monitor:
            self.o2o_monitor.cleanup_old_data()
            logger.info("监控数据清理完成")
            
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """获取实时指标"""
        if not self.o2o_monitor:
            return {'monitoring_enabled': False}
        
        return {
            'monitoring_enabled': True,
            'current_metrics': self.o2o_monitor.current_metrics,
            'training_status': self.get_training_status(),
            'recent_alerts': self.o2o_monitor.statistics.get('alerts_triggered', 0),
            'monitoring_statistics': self.o2o_monitor.statistics
        }    

    def update_adaptive_hyperparameters(self, 
                                       performance_metrics: Dict[str, float],
                                       kl_divergence: float = 0.0) -> Dict[str, float]:
        """
        更新自适应超参数
        
        Args:
            performance_metrics: 性能指标
            kl_divergence: KL散度
            
        Returns:
            Dict: 更新后的超参数
        """
        if not self.adaptive_tuner:
            return {}
            
        # 构建性能指标对象
        metrics = PerformanceMetrics(
            loss=performance_metrics.get('loss', 0.0),
            reward=performance_metrics.get('reward', 0.0),
            cvar_estimate=performance_metrics.get('cvar_estimate', 0.0),
            kl_divergence=kl_divergence,
            memory_usage=performance_metrics.get('memory_usage', 0.0),
            training_time=performance_metrics.get('training_time', 0.0)
        )
        
        # 更新超参数
        updated_params = self.adaptive_tuner.update(metrics, kl_divergence)
        
        # 应用更新到智能体
        self._apply_hyperparameter_updates(updated_params)
        
        # 记录更新
        if self.o2o_monitor:
            self.o2o_monitor.log_hyperparameter_update(updated_params)
            
        logger.debug(f"自适应超参数更新: {updated_params}")
        
        return updated_params
    
    def _apply_hyperparameter_updates(self, params: Dict[str, float]):
        """应用超参数更新到智能体和训练器"""
        
        # 更新学习率
        if 'learning_rate' in params:
            new_lr = params['learning_rate']
            
            if self.agent.use_split_optimizers:
                if self.agent.actor_optimizer:
                    for param_group in self.agent.actor_optimizer.param_groups:
                        param_group['lr'] = new_lr
                if self.agent.critic_optimizer:
                    for param_group in self.agent.critic_optimizer.param_groups:
                        param_group['lr'] = new_lr
            else:
                for param_group in self.agent.optimizer.param_groups:
                    param_group['lr'] = new_lr
                    
        # 更新采样比例
        if 'sampling_ratio' in params and self.mixture_sampler:
            self.mixture_sampler.rho = params['sampling_ratio']
            
        # 更新信任域参数
        if 'trust_region_beta' in params:
            self.agent.trust_region_beta = params['trust_region_beta']
            
    def get_adaptive_tuning_statistics(self) -> Dict[str, Any]:
        """获取自适应调优统计信息"""
        if not self.adaptive_tuner:
            return {'adaptive_tuning_enabled': False}
            
        return {
            'adaptive_tuning_enabled': True,
            'statistics': self.adaptive_tuner.get_statistics(),
            'current_phase': self.training_state.current_phase.value
        }
    
    def save_adaptive_tuning_history(self, filepath: str):
        """保存自适应调优历史"""
        if self.adaptive_tuner:
            self.adaptive_tuner.save_adaptation_history(filepath)
            logger.info(f"自适应调优历史已保存: {filepath}")
        else:
            logger.warning("自适应调优未启用")
            
    def suggest_hyperparameter_config(self) -> Optional[Dict[str, float]]:
        """建议超参数配置（用于全局搜索）"""
        if self.adaptive_tuner:
            return self.adaptive_tuner.suggest_hyperparameter_config()
        return None
    
    def report_hyperparameter_result(self, config: Dict[str, float], performance: float):
        """报告超参数配置结果"""
        if self.adaptive_tuner:
            self.adaptive_tuner.report_hyperparameter_result(config, performance)
            
    def run_hyperparameter_search(self, 
                                 n_trials: int = 50,
                                 trial_duration: int = 1800) -> Dict[str, Any]:
        """
        运行超参数搜索
        
        Args:
            n_trials: 试验次数
            trial_duration: 每次试验持续时间（秒）
            
        Returns:
            Dict: 搜索结果
        """
        if not self.adaptive_tuner or not self.adaptive_tuner.searcher:
            logger.error("超参数搜索未启用")
            return {}
            
        logger.info(f"开始超参数搜索，试验次数: {n_trials}")
        
        search_results = []
        
        for trial_idx in range(n_trials):
            # 获取建议配置
            suggested_config = self.adaptive_tuner.suggest_hyperparameter_config()
            if not suggested_config:
                continue
                
            logger.info(f"试验 {trial_idx + 1}/{n_trials}: {suggested_config}")
            
            # 应用配置
            self._apply_hyperparameter_updates(suggested_config)
            
            # 运行短期训练
            start_time = time.time()
            trial_performance = self._run_trial_training(trial_duration)
            trial_time = time.time() - start_time
            
            # 报告结果
            self.adaptive_tuner.report_hyperparameter_result(
                suggested_config, trial_performance
            )
            
            search_results.append({
                'trial_idx': trial_idx,
                'config': suggested_config,
                'performance': trial_performance,
                'duration': trial_time
            })
            
            logger.info(f"试验 {trial_idx + 1} 完成，性能: {trial_performance:.4f}")
            
        # 获取最佳配置
        best_config = self.adaptive_tuner.searcher.get_best_config()
        
        logger.info(f"超参数搜索完成，最佳配置: {best_config}")
        
        return {
            'n_trials': n_trials,
            'search_results': search_results,
            'best_config': best_config,
            'best_performance': self.adaptive_tuner.searcher.best_performance
        }
    
    def _run_trial_training(self, duration: int) -> float:
        """运行试验训练并返回性能指标"""
        # 这是一个简化的试验训练实现
        # 实际应用中需要根据具体需求实现
        
        start_time = time.time()
        total_reward = 0.0
        episode_count = 0
        
        while time.time() - start_time < duration:
            # 模拟训练episode
            episode_reward = np.random.normal(0.1, 0.05)  # 占位符
            total_reward += episode_reward
            episode_count += 1
            
            # 简单的早停机制
            if episode_count > 10 and total_reward / episode_count < -0.1:
                break
                
        return total_reward / max(episode_count, 1)