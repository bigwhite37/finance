"""
强化学习训练器实现
实现RLTrainer类和强化学习训练循环，包括早停机制、学习率调度和梯度裁剪
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import pickle
from pathlib import Path
import time
import multiprocessing

from .data_split_strategy import SplitResult
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold, BaseCallback
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """训练指标收集器"""
    
    def __init__(self):
        """初始化训练指标"""
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.validation_scores: List[float] = []
        
    def add_episode_metrics(self, reward: float, length: int, actor_loss: float, critic_loss: float):
        """添加episode指标"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
    def add_validation_score(self, score: float):
        """添加验证分数"""
        self.validation_scores.append(score)
        
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.episode_rewards:
            return {}
            
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'mean_actor_loss': np.mean(self.actor_losses),
            'mean_critic_loss': np.mean(self.critic_losses),
            'total_episodes': len(self.episode_rewards)
        }
        
    def get_recent_statistics(self, window: int = 100) -> Dict[str, float]:
        """获取最近window个episode的统计信息"""
        if not self.episode_rewards:
            return {}
            
        start_idx = max(0, len(self.episode_rewards) - window)
        recent_rewards = self.episode_rewards[start_idx:]
        recent_lengths = self.episode_lengths[start_idx:]
        recent_actor_losses = self.actor_losses[start_idx:]
        recent_critic_losses = self.critic_losses[start_idx:]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_length': np.mean(recent_lengths),
            'mean_actor_loss': np.mean(recent_actor_losses),
            'mean_critic_loss': np.mean(recent_critic_losses),
            'episodes_count': len(recent_rewards)
        }
        
    def reset(self):
        """重置所有指标"""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.actor_losses.clear()
        self.critic_losses.clear()
        self.validation_scores.clear()


class TrainingProgressCallback(BaseCallback):
    """
    改进的训练进度回调函数
    
    修复问题：
    1. 使用model.num_timesteps替代n_calls（如sb3.md建议）
    2. 添加多进程rank判断避免重复日志
    3. 考虑多环境下的步数调整
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._custom_logger = logging.getLogger(__name__)
        self.last_logged_timestep = 0
        self._n_envs = None  # 将在第一次调用时获取
        
        # 获取当前进程rank（用于多进程环境）
        self.rank = os.environ.get('RANK', '0')
        self.is_main_process = self.rank == '0' or self.rank == 0
        
    def _on_step(self) -> bool:
        """每步调用的回调"""
        # 只在主进程中记录日志
        if not self.is_main_process:
            return True
            
        # 获取环境数量（第一次调用时）
        if self._n_envs is None:
            if hasattr(self.training_env, 'num_envs'):
                self._n_envs = self.training_env.num_envs
            else:
                self._n_envs = 1
            
            # 根据环境数量调整日志频率
            if self._n_envs > 1:
                self._custom_logger.info(f"检测到{self._n_envs}个并行环境，日志频率已自动调整")
            
        # 使用model.num_timesteps而不是n_calls（如sb3.md建议）
        current_timesteps = self.model.num_timesteps
        
        # 计算实际的日志间隔（考虑多环境）
        effective_log_freq = self.log_freq * self._n_envs
        
        # 每隔effective_log_freq个timesteps打印一次统计信息  
        if current_timesteps - self.last_logged_timestep >= effective_log_freq:
            self.last_logged_timestep = current_timesteps
            
            # 获取训练统计信息
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # 从SB3的logger中获取统计信息
                record = self.model.logger.name_to_value
                
                # 提取关键指标
                stats = {}
                if 'train/actor_loss' in record:
                    stats['actor_loss'] = record['train/actor_loss']
                if 'train/critic_loss' in record:
                    stats['critic_loss'] = record['train/critic_loss']
                if 'train/ent_coef' in record:
                    stats['entropy_coef'] = record['train/ent_coef']
                if 'train/ent_coef_loss' in record:
                    stats['entropy_loss'] = record['train/ent_coef_loss']
                if 'train/learning_rate' in record:
                    stats['learning_rate'] = record['train/learning_rate']
                
                # 打印统计信息
                if stats:
                    stats_str = " | ".join([f"{k}: {v:.4f}" for k, v in stats.items()])
                    self.logger.info(f"Timestep {current_timesteps}: {stats_str}")
        
        return True

# 尝试导入增强指标模块，如果不存在则跳过
try:
    from ..metrics.portfolio_metrics import (
        PortfolioMetricsCalculator,
        PortfolioMetrics,
        AgentBehaviorMetrics,
        RiskControlMetrics
    )
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    logger.warning("增强指标模块不可用，将跳过相关功能")
    ENHANCED_METRICS_AVAILABLE = False






@dataclass
class TrainingConfig:
    """
    训练配置 - 作为配置优先级的主配置类
    
    配置优先级：
    1. TrainingConfig中的参数（最高优先级）
    2. SACConfig中的参数（如果TrainingConfig中没有对应参数）
    3. 默认值（最低优先级）
    
    注意：重复字段将从SACConfig中移除，由此类统一管理
    """
    # === SB3核心参数（高优先级统一管理） ===
    total_timesteps: int = 1000000  # 总训练步数
    n_envs: int = 1  # 并行环境数量
    batch_size: int = 256           # 批次大小
    learning_rate: float = 3e-4     # 学习率
    buffer_size: int = 1000000      # 经验回放缓冲区大小
    
    # === SAC算法参数 ===
    gamma: float = 0.99  # 折扣因子
    tau: float = 0.005   # 软更新参数

    # === SB3回调频率（timesteps单位） ===
    eval_freq: int = 10000          # 评估频率（以timesteps计）
    save_freq: int = 50000          # 保存频率（以timesteps计）
    n_eval_episodes: int = 5        # 评估时运行的episode数

    # === 向后兼容的episode参数（用于计算total_timesteps） ===
    n_episodes: int = 5000
    max_steps_per_episode: int = 180  # 降低以匹配实际数据长度
    
    # 向后兼容的episode频率（将被转换为timestep频率）
    validation_frequency: int = 50
    save_frequency: int = 100

    # === 早停参数（timesteps单位） ===
    early_stopping_patience: int = 20000    # 早停耐心（从episode转为timesteps）
    early_stopping_min_delta: float = 0.001
    early_stopping_mode: str = 'max'  # 'max' or 'min'

    # === 学习率调度 ===
    lr_scheduler_step_size: int = 1000
    lr_scheduler_gamma: float = 0.95

    # === 梯度裁剪 ===
    gradient_clip_norm: Optional[float] = 1.0

    # === 已废弃字段（由SB3接管，保留仅为向后兼容） ===
    warmup_episodes: int = 1  # DEPRECATED: SB3使用learning_starts
    update_frequency: int = 1  # DEPRECATED: SB3使用train_freq
    target_update_frequency: int = 1  # DEPRECATED: SB3使用target_update_interval

    # 保存路径
    save_dir: str = "./checkpoints"

    # 随机种子
    random_seed: Optional[int] = None

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 回撤控制和奖励优化参数
    enable_drawdown_monitoring: bool = False    # 启用训练过程中的回撤监控
    drawdown_early_stopping: bool = False      # 基于回撤的早停
    max_training_drawdown: float = 0.3         # 训练过程最大允许回撤
    reward_enhancement_progress: float = 0.5   # 奖励增强进度（0-1）

    # 自适应训练参数
    enable_adaptive_learning: bool = False     # 启用自适应学习参数调整
    lr_adaptation_factor: float = 0.8          # 学习率自适应因子
    min_lr_factor: float = 0.01                # 最小学习率因子
    max_lr_factor: float = 1.0                 # 最大学习率因子
    lr_recovery_factor: float = 1.25           # 学习率恢复因子
    performance_threshold_down: float = 0.85   # 性能下降阈值（更严格）
    performance_threshold_up: float = 1.15     # 性能提升阈值（更严格）
    batch_size_adaptation: bool = False        # 批次大小自适应
    exploration_decay_by_performance: bool = False  # 基于性能的探索衰减

    # 多核并行优化参数（保留以向后兼容）
    enable_multiprocessing: bool = True            # 启用多进程优化
    num_workers: int = field(default_factory=lambda: min(8, multiprocessing.cpu_count()))  # 数据加载工作进程数
    parallel_environments: int = field(default_factory=lambda: min(4, multiprocessing.cpu_count() // 2))  # 并行环境数量（已迁移到n_envs）
    data_loader_workers: int = field(default_factory=lambda: min(4, multiprocessing.cpu_count() // 2))    # DataLoader工作线程数
    pin_memory: bool = True                        # GPU内存固定
    persistent_workers: bool = True                # 持久化工作进程
    prefetch_factor: int = 2                       # 预取因子

    # 奖励阈值（用于早停）
    reward_threshold: Optional[float] = None       # 奖励阈值，达到时停止训练

    # GPU优化参数
    enable_mixed_precision: bool = False         # DEPRECATED: SB3 >= 2.2会自动处理混合精度，手动启用可能冲突
    enable_cudnn_benchmark: bool = True          # 启用cuDNN基准测试
    non_blocking_transfer: bool = True           # 非阻塞数据传输

    # 增强指标配置
    enable_portfolio_metrics: bool = True          # 启用投资组合指标计算
    enable_agent_behavior_metrics: bool = True     # 启用智能体行为指标计算
    enable_risk_control_metrics: bool = True       # 启用风险控制指标计算

    # 指标计算频率（timesteps单位）
    metrics_calculation_frequency: int = 3600      # 每N个timesteps计算一次指标（从episode换算）

    # 基准数据配置
    benchmark_data_path: Optional[str] = None      # 基准数据路径
    risk_free_rate: float = 0.03                   # 无风险利率

    # 环境配置（用于指标计算的默认值）
    initial_cash: float = 1000000.0                # 初始资金（用于指标计算默认值）

    # 日志配置
    detailed_metrics_logging: bool = True          # 详细指标日志
    metrics_log_level: str = 'INFO'                # 指标日志级别

    def __post_init__(self):
        """配置验证"""
        if self.n_episodes <= 0:
            raise ValueError("n_episodes必须为正数")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate必须为正数")

        if self.batch_size <= 0:
            raise ValueError("batch_size必须为正数")

        if self.max_steps_per_episode <= 0:
            raise ValueError("max_steps_per_episode必须为正数")

        if self.early_stopping_mode not in ['max', 'min']:
            raise ValueError("early_stopping_mode必须是'max'或'min'")

        # 自动计算total_timesteps（如果未设置或为0）
        if self.total_timesteps == 1000000 or self.total_timesteps == 0:  # 使用默认值或占位符
            self.total_timesteps = self.n_episodes * self.max_steps_per_episode * self.n_envs

        # 自动计算SB3回调频率（从episode转换为timesteps）
        if self.eval_freq == 10000:
            self.eval_freq = self.validation_frequency * self.max_steps_per_episode * self.n_envs
        if self.save_freq == 50000:
            self.save_freq = self.save_frequency * self.max_steps_per_episode * self.n_envs
            
        # 转换早停耐心从episode到timesteps（如果使用默认值）
        if self.early_stopping_patience == 20000:  # 新的默认值
            # 基于原始episode的early_stopping_patience计算
            episode_patience = 20  # 原始的episode耐心
            self.early_stopping_patience = episode_patience * self.max_steps_per_episode * self.n_envs
            
        # 转换指标计算频率从episode到timesteps（如果使用默认值）  
        if self.metrics_calculation_frequency == 3600:  # 新的默认值
            # 基于原始episode的metrics_calculation_frequency计算
            episode_freq = 20  # 原始的episode频率
            self.metrics_calculation_frequency = episode_freq * self.max_steps_per_episode * self.n_envs
            
        # 检查废弃字段的使用并发出警告
        self._check_deprecated_fields()

        if self.min_lr_factor <= 0 or self.min_lr_factor >= self.max_lr_factor:
            raise ValueError("min_lr_factor必须大于0且小于max_lr_factor")

        if self.lr_recovery_factor <= 1.0:
            raise ValueError("lr_recovery_factor必须大于1.0")

        if self.performance_threshold_down >= 1.0 or self.performance_threshold_up <= 1.0:
            raise ValueError("performance_threshold_down必须小于1.0，performance_threshold_up必须大于1.0")

        # 多核配置验证
        if self.num_workers < 0:
            raise ValueError("num_workers必须为非负数")

        if self.parallel_environments < 1:
            raise ValueError("parallel_environments必须为正数")

        if self.data_loader_workers < 0:
            raise ValueError("data_loader_workers必须为非负数")

        # 自动调整多核配置以避免资源过度使用
        max_workers = multiprocessing.cpu_count()
        if self.num_workers > max_workers:
            self.num_workers = max_workers
        if self.parallel_environments > max_workers:
            self.parallel_environments = max_workers

        # 同步n_envs与parallel_environments（确保单一数据源）
        if self.n_envs == 1 and self.parallel_environments > 1:
            self.n_envs = self.parallel_environments
            logger.info(f"已将n_envs同步为parallel_environments: {self.n_envs}")
        elif self.n_envs > 1 and self.parallel_environments != self.n_envs:
            # 如果两者都不是默认值且不相等，以n_envs为准
            self.parallel_environments = self.n_envs
            logger.info(f"已将parallel_environments同步为n_envs: {self.parallel_environments}")

        # 增强指标配置验证
        if self.metrics_calculation_frequency <= 0:
            raise ValueError("metrics_calculation_frequency必须为正数")

        if self.enable_portfolio_metrics and self.benchmark_data_path == "":
            raise ValueError("启用投资组合指标时，benchmark_data_path不能为空")

        if self.risk_free_rate < 0:
            raise ValueError("risk_free_rate不能为负数")
            
    def _check_deprecated_fields(self):
        """检查废弃字段的使用并发出警告"""
        import warnings
        
        # 检查warmup_episodes（现在使用learning_starts）
        if self.warmup_episodes != 1:  # 非默认值
            warnings.warn(
                f"warmup_episodes已废弃，请使用SACConfig.learning_starts。"
                f"当前值：{self.warmup_episodes}",
                DeprecationWarning,
                stacklevel=3
            )
            
        # 检查update_frequency（现在使用train_freq）
        if self.update_frequency != 1:  # 非默认值
            warnings.warn(
                f"update_frequency已废弃，请使用SACConfig.train_freq。"
                f"当前值：{self.update_frequency}",
                DeprecationWarning,
                stacklevel=3
            )
            
        # 检查target_update_frequency（现在使用target_update_interval）
        if self.target_update_frequency != 1:  # 非默认值
            warnings.warn(
                f"target_update_frequency已废弃，请使用SACConfig.target_update_interval。"
                f"当前值：{self.target_update_frequency}",
                DeprecationWarning,
                stacklevel=3
            )
            
        # 检查enable_mixed_precision（现在由SB3自动处理）
        if self.enable_mixed_precision:
            warnings.warn(
                "enable_mixed_precision已废弃，SB3 >= 2.2会自动处理混合精度训练。"
                "手动启用可能导致冲突，建议设置为False。",
                DeprecationWarning,
                stacklevel=3
            )


class EarlyStoppingCallback(BaseCallback):
    """
    基于SB3的早停回调
    
    监控评估指标，当指标在指定步数内没有改进时停止训练
    """

    def __init__(self, patience: int = 20000, min_delta: float = 0.001, mode: str = 'max', 
                 check_freq: int = 1000, verbose: int = 0):
        """
        初始化早停回调

        Args:
            patience: 耐心值（以timesteps计）
            min_delta: 最小改进幅度
            mode: 'max'表示分数越高越好，'min'表示分数越低越好
            check_freq: 检查频率（以timesteps计）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.check_freq = check_freq
        self.best_score = None
        self.last_check_timestep = 0
        self.no_improvement_timesteps = 0
        self.early_stop = False

        if mode == 'max':
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta

    def _on_step(self) -> bool:
        """每步调用的回调"""
        current_timesteps = self.model.num_timesteps
        
        # 按指定频率检查
        if current_timesteps - self.last_check_timestep >= self.check_freq:
            self.last_check_timestep = current_timesteps
            
            # 从模型logger获取评估分数
            score = self._get_evaluation_score()
            if score is not None:
                should_stop = self._update_early_stopping(score, current_timesteps)
                if should_stop:
                    if self.verbose > 0:
                        print(f"\n早停触发: 在 {self.no_improvement_timesteps} 个timesteps内无改进")
                    return False
        
        return True
    
    def _get_evaluation_score(self) -> Optional[float]:
        """从模型logger获取评估分数"""
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            record = self.model.logger.name_to_value
            
            # 尝试获取评估奖励
            if 'eval/mean_reward' in record:
                return float(record['eval/mean_reward'])
            elif 'rollout/ep_rew_mean' in record:
                return float(record['rollout/ep_rew_mean'])
        
        return None
    
    def _update_early_stopping(self, score: float, current_timesteps: int) -> bool:
        """
        更新早停状态

        Args:
            score: 当前分数
            current_timesteps: 当前时间步

        Returns:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self.no_improvement_timesteps = 0
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.no_improvement_timesteps = 0
            if self.verbose > 0:
                print(f"新的最佳分数: {score:.4f} (timestep {current_timesteps})")
        else:
            self.no_improvement_timesteps += self.check_freq

        if self.no_improvement_timesteps >= self.patience:
            self.early_stop = True
            return True

        return False

    def reset(self):
        """重置早停状态"""
        self.best_score = None
        self.no_improvement_timesteps = 0
        self.early_stop = False


class EarlyStopping:
    """早停机制（向后兼容类）"""

    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = 'max'):
        """
        初始化早停机制

        Args:
            patience: 耐心值，即允许的无改进epoch数
            min_delta: 最小改进幅度
            mode: 'max'表示分数越高越好，'min'表示分数越低越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if mode == 'max':
            self.is_better = lambda score, best: score > best + min_delta
        else:
            self.is_better = lambda score, best: score < best - min_delta

    def step(self, score: float) -> bool:
        """
        更新早停状态

        Args:
            score: 当前分数

        Returns:
            bool: 是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter > self.patience:
            self.early_stop = True
            return True

        return False

    def reset(self):
        """重置早停状态"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False


class DrawdownEarlyStoppingCallback(BaseCallback):
    """
    基于回撤的早停回调
    
    监控训练过程中的投资组合回撤，当回撤超过阈值时停止训练
    """

    def __init__(self, max_drawdown: float = 0.3, patience: int = 10, check_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.max_drawdown = max_drawdown
        self.patience = patience
        self.check_freq = check_freq
        self.peak_value = None
        self.counter = 0
        self.drawdown_history = []
        self.last_check_timestep = 0

    def _on_step(self) -> bool:
        """每步检查回撤"""
        current_timesteps = self.model.num_timesteps
        
        # 按指定频率检查
        if current_timesteps - self.last_check_timestep >= self.check_freq:
            self.last_check_timestep = current_timesteps
            
            # 从环境获取投资组合价值
            current_value = self._get_portfolio_value()
            if current_value is not None:
                should_stop = self._update_drawdown_monitoring(current_value)
                if should_stop:
                    return False
        
        return True
    
    def _get_portfolio_value(self) -> Optional[float]:
        """获取当前投资组合价值"""
        try:
            if hasattr(self.training_env, 'get_attr'):
                portfolio_values = self.training_env.get_attr('total_value')
                if portfolio_values and len(portfolio_values) > 0:
                    return float(portfolio_values[0])  # 第一个环境的值
            elif hasattr(self.training_env, 'total_value'):
                return float(self.training_env.total_value)
        except AttributeError:
            # 环境不支持获取投资组合价值
            if self.verbose > 1:
                print("警告: 环境不支持投资组合价值获取，跳过回撤监控")
        except Exception as e:
            # 其他异常需要记录
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"获取投资组合价值时发生错误: {e}")
        
        return None
    
    def _update_drawdown_monitoring(self, current_value: float) -> bool:
        """
        更新回撤监控状态

        Args:
            current_value: 当前投资组合价值

        Returns:
            bool: 是否应该早停
        """
        if self.peak_value is None:
            self.peak_value = current_value
            self.drawdown_history.append(0.0)
            return False

        # 更新峰值
        if current_value > self.peak_value:
            self.peak_value = current_value

        # 计算当前回撤
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - current_value) / self.peak_value
            current_drawdown = max(current_drawdown, 0.0)
        else:
            current_drawdown = 0.0

        self.drawdown_history.append(current_drawdown)

        # 检查是否超过回撤阈值
        if current_drawdown > self.max_drawdown:
            self.counter += 1
            if self.verbose > 0:
                print(f"回撤警告: {current_drawdown:.4f} > {self.max_drawdown:.4f} (连续 {self.counter} 次)")
        else:
            self.counter = 0

        # 判断是否需要早停
        if self.counter >= self.patience:
            if self.verbose > 0:
                print(f"\n触发回撤早停: 当前回撤 {current_drawdown:.4f} > 阈值 {self.max_drawdown:.4f} "
                      f"(连续 {self.counter} 次检查)")
            return True

        return False

    def get_current_drawdown(self) -> float:
        """获取当前回撤"""
        return self.drawdown_history[-1] if self.drawdown_history else 0.0
    
    def get_max_drawdown(self) -> float:
        """获取历史最大回撤"""
        return max(self.drawdown_history) if self.drawdown_history else 0.0


class DrawdownEarlyStopping:
    """基于回撤的早停机制（兼容性类）"""

    def __init__(self, max_drawdown: float = 0.3, patience: int = 10):
        """
        初始化基于回撤的早停机制

        Args:
            max_drawdown: 最大允许回撤阈值
            patience: 超过阈值后的耐心值
        """
        self.max_drawdown = max_drawdown
        self.patience = patience
        self.peak_value = None
        self.counter = 0
        self.early_stop = False
        self.drawdown_history = []

    def step(self, current_value: float) -> bool:
        """
        更新回撤早停状态

        Args:
            current_value: 当前投资组合价值

        Returns:
            bool: 是否应该早停
        """
        if self.peak_value is None:
            # 初始化峰值为当前值
            self.peak_value = current_value
            self.drawdown_history.append(0.0)
            return False

        # 更新峰值：当前值大于历史峰值时更新
        if current_value > self.peak_value:
            self.peak_value = current_value

        # 标准回撤计算：相对于历史最高点的损失百分比
        if self.peak_value > 0:
            current_drawdown = (self.peak_value - current_value) / self.peak_value
            # 确保回撤为非负数
            current_drawdown = max(current_drawdown, 0.0)
        else:
            # 如果峰值为0或负数，无法计算有意义的回撤
            current_drawdown = 0.0

        self.drawdown_history.append(current_drawdown)

        # 检查是否超过回撤阈值
        if current_drawdown > self.max_drawdown:
            self.counter += 1
        else:
            self.counter = 0

        # 判断是否需要早停
        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False

    def get_current_drawdown(self) -> float:
        """获取当前回撤"""
        return self.drawdown_history[-1] if self.drawdown_history else 0.0

    def reset(self):
        """重置早停状态"""
        self.peak_value = None
        self.counter = 0
        self.early_stop = False
        self.drawdown_history = []




class RLTrainer:
    """强化学习训练器"""

    def __init__(self, config: TrainingConfig, environment, agent, data_split: SplitResult, env_factory=None):
        """
        初始化训练器

        Args:
            config: 训练配置
            environment: 交易环境
            agent: 强化学习智能体
            data_split: 数据划分结果
            env_factory: 环境工厂函数（用于创建多个环境实例）
        """
        self.config = config
        self.environment = environment
        self.agent = agent
        self.data_split = data_split
        self.env_factory = env_factory

        # 初始化训练组件（SB3自带指标收集）
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode=config.early_stopping_mode
        )

        # 初始化回撤控制相关组件（保留以向后兼容）
        if config.enable_drawdown_monitoring:
            self.drawdown_early_stopping = DrawdownEarlyStopping(
                max_drawdown=config.max_training_drawdown,
                patience=config.early_stopping_patience // 2  # 回撤早停耐心值更小
            )
            self.drawdown_metrics = []
            logger.info("回撤监控已启用（将通过SB3回调实现）")
        else:
            self.drawdown_early_stopping = None
            self.drawdown_metrics = []

        # 自适应训练参数
        self.adaptive_learning_enabled = config.enable_adaptive_learning
        self.current_lr_factor = 1.0
        self.performance_history = []

        # 初始化增强指标组件
        if ENHANCED_METRICS_AVAILABLE:
            self.metrics_calculator = PortfolioMetricsCalculator()

            # 历史数据存储
            self.portfolio_values_history: List[float] = []
            self.benchmark_values_history: List[float] = []
            self.dates_history: List[datetime] = []

            # 智能体行为数据
            self.entropy_history: List[float] = []
            self.position_weights_history: List[np.ndarray] = []

            # 风险控制数据
            self.risk_budget_history: List[float] = []
            self.risk_usage_history: List[float] = []
            self.control_signals_history: List[Dict[str, Any]] = []
            self.market_regime_history: List[str] = []

            logger.info(f"指标计算配置: 投资组合指标={config.enable_portfolio_metrics}, "
                       f"智能体行为指标={config.enable_agent_behavior_metrics}, "
                       f"风险控制指标={config.enable_risk_control_metrics}")
        else:
            self.metrics_calculator = None
            self.portfolio_values_history = []
            self.benchmark_values_history = []
            self.dates_history = []
            self.entropy_history = []
            self.position_weights_history = []
            self.risk_budget_history = []
            self.risk_usage_history = []
            self.control_signals_history = []
            self.market_regime_history = []

        # 设置随机种子
        if config.random_seed is not None:
            self._set_random_seed(config.random_seed)

        # 创建保存目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化学习率调度器（如果智能体支持）
        self._setup_lr_scheduler()

        # 初始化多核优化组件
        self._setup_multicore_optimization()

        logger.info("训练器初始化完成")
        logger.debug(f"训练配置: {config}")

    def _set_random_seed(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _setup_lr_scheduler(self):
        """设置学习率调度器"""
        # 在实际实现中，这里会设置PyTorch的学习率调度器
        # 由于使用模拟智能体，这里只是占位符
        self.lr_scheduler = None

    def _setup_multicore_optimization(self):
        """设置多核优化"""
        if not self.config.enable_multiprocessing:
            logger.info("多进程优化已禁用")
            self.parallel_env_manager = None
            self.data_loader = None
            return

        logger.info(f"配置多核优化: {self.config.num_workers} 个数据工作进程, "
                   f"{self.config.parallel_environments} 个并行环境")

        # 设置PyTorch多进程上下文（只在主进程中设置）
        try:
            # 检查是否已经设置了start method
            current_method = mp.get_start_method(allow_none=True)
            if current_method is None:
                mp.set_start_method('spawn')
                logger.info("已设置多进程启动方法为 'spawn'")
            else:
                logger.info(f"多进程启动方法已设置为: {current_method}")
        except RuntimeError as e:
            # 如果在子进程中调用或已经设置，记录警告但不中断
            logger.warning(f"无法设置多进程启动方法: {e}")
        except Exception as e:
            logger.error(f"设置多进程启动方法时发生未预期错误: {e}")
            raise RuntimeError(f"多进程配置失败: {e}") from e

        # 设置cuDNN基准测试（如果启用GPU优化）
        if self.config.enable_cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN基准测试已启用")

        # 混合精度现在由SB3自动处理，无需手动初始化
        if self.config.enable_mixed_precision:
            logger.warning("enable_mixed_precision已启用但将被忽略，SB3会自动处理混合精度")

        # SB3的VecEnv会自动处理并行环境
        logger.info(f"将使用SB3 VecEnv处理{self.config.parallel_environments}个并行环境")

    def _get_current_learning_rate(self, episode: int) -> float:
        """获取当前学习率"""
        # 简单的指数衰减
        decay_rate = self.config.lr_scheduler_gamma
        decay_steps = self.config.lr_scheduler_step_size

        decay_factor = decay_rate ** (episode // decay_steps)
        return self.config.learning_rate * decay_factor


    def save_checkpoint(self, filepath: str, timesteps: int):
        """保存检查点（SB3兼容版本）"""
        checkpoint = {
            'timesteps': timesteps,
            'config': self.config,
            'early_stopping_state': {
                'best_score': self.early_stopping.best_score,
                'counter': self.early_stopping.counter,
                'early_stop': self.early_stopping.early_stop
            }
        }

        # 保存智能体状态（如果支持）
        if hasattr(self.agent, 'save'):
            agent_path = filepath.replace('.pth', '_agent.pth')
            self.agent.save(agent_path)
            checkpoint['agent_path'] = agent_path

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        logger.debug(f"检查点已保存到: {filepath}")

    def load_checkpoint(self, filepath: str) -> int:
        """加载检查点"""
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        timesteps = checkpoint.get('timesteps', checkpoint.get('episode', 0))  # 向后兼容

        # 恢复早停状态
        early_stopping_state = checkpoint['early_stopping_state']
        self.early_stopping.best_score = early_stopping_state['best_score']
        self.early_stopping.counter = early_stopping_state['counter']
        self.early_stopping.early_stop = early_stopping_state['early_stop']

        # 加载智能体状态（如果存在）
        if 'agent_path' in checkpoint and hasattr(self.agent, 'load'):
            self.agent.load(checkpoint['agent_path'])

        logger.debug(f"检查点已从 {filepath} 加载，timesteps: {timesteps}")
        return timesteps



    def train(self):
        """执行训练 - 使用SB3的.learn()方法"""
        logger.info(f"开始SB3训练，总步数: {self.config.total_timesteps}")
        logger.info(f"并行环境数: {self.config.n_envs}")

        start_time = time.time()

        # 设置智能体的环境（如果还没设置）
        if hasattr(self.agent, 'set_env'):
            if self.env_factory and self.config.n_envs > 1:
                # 使用环境工厂创建多环境
                self.agent.set_env(None, env_factory=self.env_factory, n_envs=self.config.n_envs)
            else:
                # 使用单环境
                self.agent.set_env(self.environment)

        # 创建回调列表
        callbacks = self._create_callbacks()

        # 执行SB3训练
        try:
            self.agent.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=100,  # 每100次更新打印一次日志
                reset_num_timesteps=True
            )
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        except Exception as e:
            logger.error(f"训练过程中出现错误: {e}")
            raise RuntimeError(f"SB3训练失败: {e}") from e

        training_time = time.time() - start_time
        logger.info(f"SB3训练完成，总用时: {training_time:.2f}秒")

        # 保存最终模型
        final_model_path = self.save_dir / "final_model"
        self.agent.save(final_model_path)
        logger.info(f"最终模型已保存到: {final_model_path}")

        # 获取训练统计
        stats = self.agent.get_training_stats() if hasattr(self.agent, 'get_training_stats') else {}
        stats['training_time'] = training_time
        stats['total_timesteps'] = self.config.total_timesteps

        return stats

    def _create_callbacks(self):
        """创建SB3回调列表"""
        callbacks = []

        # 1. 添加进度回调（本地实现，集中管理）
        # 计算基础日志频率（不考虑多环境，回调内部会自动调整）
        base_log_freq = max(1000, self.config.total_timesteps // 50)
        progress_callback = TrainingProgressCallback(
            log_freq=base_log_freq
        )
        callbacks.append(progress_callback)

        # 2. 添加评估回调
        if self.config.eval_freq > 0:
            # 创建评估环境
            eval_env = self.environment
            if self.env_factory:
                # 如果有环境工厂，创建单独的评估环境
                eval_env = self.env_factory()

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.save_dir),
                log_path=str(self.save_dir),
                eval_freq=self.config.eval_freq,
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=1
            )
            callbacks.append(eval_callback)

        # 3. 添加检查点回调
        if self.config.save_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=self.config.save_freq,
                save_path=str(self.save_dir),
                name_prefix="sac_checkpoint",
                verbose=1
            )
            callbacks.append(checkpoint_callback)

        # 4. 添加基于评估指标的早停回调
        if self.config.early_stopping_patience > 0:
            early_stop_callback = EarlyStoppingCallback(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode=self.config.early_stopping_mode,
                check_freq=max(1000, self.config.eval_freq // 2),  # 检查频率为评估频率的一半
                verbose=1
            )
            callbacks.append(early_stop_callback)

        # 5. 添加奖励阈值早停回调
        if hasattr(self.config, 'reward_threshold') and self.config.reward_threshold is not None:
            reward_threshold_callback = StopTrainingOnRewardThreshold(
                reward_threshold=self.config.reward_threshold,
                verbose=1
            )
            callbacks.append(reward_threshold_callback)

        # 6. 添加回撤早停回调（如果启用）
        if self.config.enable_drawdown_monitoring and self.config.drawdown_early_stopping:
            drawdown_callback = DrawdownEarlyStoppingCallback(
                max_drawdown=self.config.max_training_drawdown,
                patience=max(5, self.config.early_stopping_patience // (self.config.max_steps_per_episode * 10)),  # 转换为合理的检查次数
                check_freq=max(100, self.config.max_steps_per_episode),  # 每个episode检查一次
                verbose=1
            )
            callbacks.append(drawdown_callback)

        # 7. 添加增强指标记录回调（如果启用）
        if (ENHANCED_METRICS_AVAILABLE and
            (self.config.enable_portfolio_metrics or
             self.config.enable_agent_behavior_metrics or
             self.config.enable_risk_control_metrics)):
            metrics_callback = EnhancedMetricsCallback(
                trainer=self,
                log_freq=self.config.metrics_calculation_frequency * self.config.max_steps_per_episode
            )
            callbacks.append(metrics_callback)

        logger.info(f"创建了{len(callbacks)}个回调: {[type(cb).__name__ for cb in callbacks]}")
        return callbacks


class EnhancedMetricsCallback(BaseCallback):
    """增强指标记录回调"""

    def __init__(self, trainer: 'RLTrainer', log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.trainer = trainer
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """每步记录增强指标"""
        if self.n_calls % self.log_freq == 0:
            try:
                # 从 trainer 中计算增强指标
                if hasattr(self.trainer, '_calculate_and_log_enhanced_metrics'):
                    # 模拟 episode 编号（基于 timesteps）
                    episode_num = self.n_calls // self.trainer.config.max_steps_per_episode
                    self.trainer._calculate_and_log_enhanced_metrics(episode_num)
            except Exception as e:
                if self.verbose > 0:
                    print(f"计算增强指标时发生错误: {e}")
        return True

    def _cleanup_multicore_resources(self):
        """清理多核资源（SB3自动处理）"""
        # SB3 VecEnv 会自动清理资源
        logger.debug("SB3会自动清理多核资源")

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """评估智能体性能（使用SB3内置评估）"""
        logger.info(f"开始评估，episodes: {n_episodes}")

        try:
            from stable_baselines3.common.evaluation import evaluate_policy
            
            # 创建评估环境
            eval_env = self.environment
            if self.env_factory:
                # 如果有环境工厂，创建单独的评估环境
                eval_env = self.env_factory()
            
            # 使用SB3的evaluate_policy进行评估
            mean_reward, std_reward = evaluate_policy(
                self.agent.model if hasattr(self.agent, 'model') else self.agent,
                eval_env,
                n_eval_episodes=n_episodes,
                deterministic=True,
                render=False,
                return_episode_rewards=False
            )
            
            logger.info(f"评估完成: 平均奖励={mean_reward:.4f}, 标准差={std_reward:.4f}")
            
            # 返回评估结果
            results = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'n_episodes': n_episodes
            }
            
            # 如果智能体有额外的统计信息，也包含进来
            if hasattr(self.agent, 'get_training_stats'):
                agent_stats = self.agent.get_training_stats()
                results.update(agent_stats)
                
            return results
            
        except ImportError as e:
            logger.error("无法导入stable_baselines3.common.evaluation.evaluate_policy")
            raise RuntimeError("SB3评估功能不可用，请检查stable_baselines3安装") from e
        except Exception as e:
            logger.error(f"评估过程中出现错误: {e}")
            raise RuntimeError(f"智能体评估失败: {e}") from e

    def _monitor_drawdown(self, episode_reward: float, episode: int):
        """监控训练过程中的回撤"""
        try:
            # 从环境获取真实的投资组合价值来计算回撤
            if hasattr(self.environment, 'total_value'):
                portfolio_value = self.environment.total_value
            else:
                # 如果环境不支持，记录警告并跳过回撤监控
                logger.warning("环境不支持投资组合价值获取，跳过回撤监控")
                return

            # 更新回撤早停状态（基于投资组合价值而非累积奖励）
            self.drawdown_early_stopping.step(portfolio_value)

            # 获取更新后的当前回撤
            current_drawdown = self.drawdown_early_stopping.get_current_drawdown()

            # 从环境获取准确的回撤值（如果支持）
            env_drawdown = 0.0
            if hasattr(self.environment, '_calculate_current_drawdown'):
                env_drawdown = self.environment._calculate_current_drawdown()

            self.drawdown_metrics.append({
                'episode': episode,
                'portfolio_value': portfolio_value,
                'episode_reward': episode_reward,
                'training_drawdown': current_drawdown,  # 基于投资组合价值的训练回撤
                'env_drawdown': env_drawdown  # 环境内部计算的回撤
            })

            # 从环境获取其他回撤指标（如果环境支持）
            if hasattr(self.environment, 'get_drawdown_metrics'):
                try:
                    env_metrics = self.environment.get_drawdown_metrics()
                    if env_metrics and isinstance(env_metrics, dict):
                        self.drawdown_metrics[-1].update(env_metrics)
                except Exception as e:
                    logger.debug(f"获取环境回撤指标失败: {e}")

            # 定期记录回撤信息和异常检测
            if episode % 50 == 0:
                logger.info(f"Episode {episode}: 投资组合价值 {portfolio_value:.4f}, "
                          f"训练回撤 {current_drawdown:.4f}, 环境回撤 {env_drawdown:.4f}")

                # 回撤异常检测
                if current_drawdown == 0.0 and episode > 50:
                    logger.warning(f"回撤计算异常: 连续{episode}个episode回撤为0，可能存在计算缺陷")
                elif current_drawdown > self.config.max_training_drawdown * 0.8:
                    logger.warning(f"回撤风险较高: {current_drawdown:.4f}, 接近阈值 {self.config.max_training_drawdown:.4f}")

        except Exception as e:
            logger.error(f"回撤监控失败: {e}")
            raise RuntimeError(f"训练回撤监控出现错误: {e}") from e

    def _adapt_training_parameters(self, episode_reward: float, episode: int):
        """基于性能自适应调整训练参数"""
        try:
            self.performance_history.append(episode_reward)

            # 只在有足够历史数据时进行调整
            if len(self.performance_history) < 50:
                return

            # 计算最近性能
            recent_performance = np.mean(self.performance_history[-20:])
            long_term_performance = np.mean(self.performance_history[-50:])

            # 改进的自适应学习率调整（修复负奖励环境下的逻辑错误）
            if self.config.enable_adaptive_learning:
                old_lr_factor = self.current_lr_factor

                # 计算性能变化的绝对值和相对值，适应正负奖励环境
                performance_diff = recent_performance - long_term_performance
                performance_change_ratio = abs(performance_diff) / max(abs(long_term_performance), 1.0)

                # 性能显著恶化的判断（适应负奖励）
                is_performance_worse = False
                if long_term_performance >= 0:
                    # 正奖励环境：最近表现低于长期表现的阈值
                    is_performance_worse = recent_performance < long_term_performance * self.config.performance_threshold_down
                else:
                    # 负奖励环境：最近表现更负（绝对值更大）
                    is_performance_worse = recent_performance < long_term_performance / self.config.performance_threshold_down

                # 性能显著改善的判断
                is_performance_better = False
                if long_term_performance >= 0:
                    # 正奖励环境：最近表现超过长期表现的阈值
                    is_performance_better = recent_performance > long_term_performance * self.config.performance_threshold_up
                else:
                    # 负奖励环境：最近表现更好（绝对值更小）
                    is_performance_better = recent_performance > long_term_performance / self.config.performance_threshold_up

                # 避免过于频繁的学习率调整
                significant_change = performance_change_ratio > 0.05  # 至少5%的变化

                if is_performance_worse and significant_change:
                    # 性能下降，降低学习率但有最小值限制
                    self.current_lr_factor = max(
                        self.config.min_lr_factor,
                        self.current_lr_factor * self.config.lr_adaptation_factor
                    )
                    if self.current_lr_factor != old_lr_factor:
                        logger.info(f"Episode {episode}: 检测到性能下降，降低学习率因子到 {self.current_lr_factor:.4f}")

                elif is_performance_better and significant_change:
                    # 性能提升，快速恢复学习率
                    self.current_lr_factor = min(
                        self.config.max_lr_factor,
                        self.current_lr_factor * self.config.lr_recovery_factor
                    )
                    if self.current_lr_factor != old_lr_factor:
                        logger.info(f"Episode {episode}: 检测到性能提升，调整学习率因子到 {self.current_lr_factor:.4f}")

            # 基于回撤的探索调整
            if (self.config.exploration_decay_by_performance and
                hasattr(self.agent, 'exploration_rate')):
                if self.drawdown_early_stopping:
                    current_drawdown = self.drawdown_early_stopping.get_current_drawdown()
                    if current_drawdown > 0.2:  # 回撤较大时减少探索
                        exploration_decay = 0.95
                        self.agent.exploration_rate *= exploration_decay
                        logger.info(f"Episode {episode}: 大回撤，减少探索率到 {self.agent.exploration_rate:.4f}")

        except Exception as e:
            logger.error(f"自适应参数调整失败: {e}")
            raise RuntimeError(f"训练参数自适应调整出现错误: {e}") from e

    def collect_drawdown_metrics(self) -> Dict[str, Any]:
        """收集回撤指标"""
        if not self.drawdown_metrics:
            return {'drawdown_monitoring_enabled': False}

        try:
            training_drawdowns = [m['training_drawdown'] for m in self.drawdown_metrics]
            env_drawdowns = [m['env_drawdown'] for m in self.drawdown_metrics]
            portfolio_values = [m['portfolio_value'] for m in self.drawdown_metrics]

            return {
                'drawdown_monitoring_enabled': True,
                'max_training_drawdown': max(training_drawdowns) if training_drawdowns else 0.0,
                'avg_training_drawdown': np.mean(training_drawdowns) if training_drawdowns else 0.0,
                'max_env_drawdown': max(env_drawdowns) if env_drawdowns else 0.0,
                'avg_env_drawdown': np.mean(env_drawdowns) if env_drawdowns else 0.0,
                'final_portfolio_value': portfolio_values[-1] if portfolio_values else 0.0,
                'peak_portfolio_value': max(portfolio_values) if portfolio_values else 0.0,
                'significant_drawdown_episodes': len([d for d in training_drawdowns if d > 0.1]),
                'total_monitored_episodes': len(self.drawdown_metrics)
            }

        except Exception as e:
            logger.error(f"收集回撤指标失败: {e}")
            raise RuntimeError(f"回撤指标收集出现错误: {e}") from e

    # ==================== 增强指标相关方法 ====================

    def _should_calculate_metrics(self, timesteps: int) -> bool:
        """
        判断是否应该计算指标（基于timesteps）

        Args:
            timesteps: 当前timesteps数

        Returns:
            是否应该计算指标
        """
        return timesteps % self.config.metrics_calculation_frequency == 0

    def _calculate_and_log_enhanced_metrics(self, episode_num: int):
        """
        计算并记录增强指标

        Args:
            episode_num: episode编号
        """
        if not ENHANCED_METRICS_AVAILABLE:
            return

        try:
            # 计算投资组合指标
            portfolio_metrics = None
            if self.config.enable_portfolio_metrics:
                portfolio_metrics = self._calculate_portfolio_metrics()

            # 计算智能体行为指标
            agent_metrics = None
            if self.config.enable_agent_behavior_metrics:
                agent_metrics = self._calculate_agent_behavior_metrics()

            # 计算风险控制指标
            risk_metrics = None
            if self.config.enable_risk_control_metrics:
                risk_metrics = self._calculate_risk_control_metrics()

            # 记录指标日志
            if self.config.detailed_metrics_logging:
                self._log_enhanced_metrics(episode_num, portfolio_metrics, agent_metrics, risk_metrics)

        except Exception as e:
            logger.error(f"计算增强指标时发生错误: {e}")

    def _calculate_portfolio_metrics(self) -> Optional['PortfolioMetrics']:
        """
        计算投资组合指标

        Returns:
            投资组合指标或None（如果数据不足）
        """
        if not ENHANCED_METRICS_AVAILABLE or len(self.portfolio_values_history) <= 1:
            logger.debug("投资组合价值历史数据不足，跳过指标计算")
            return None

        try:
            # 确保基准数据长度匹配
            if len(self.benchmark_values_history) != len(self.portfolio_values_history):
                logger.warning(f"基准数据长度({len(self.benchmark_values_history)})与投资组合数据长度"
                             f"({len(self.portfolio_values_history)})不匹配")
                # 截取到较短的长度
                min_len = min(len(self.benchmark_values_history), len(self.portfolio_values_history))
                portfolio_values = self.portfolio_values_history[:min_len]
                benchmark_values = self.benchmark_values_history[:min_len]
                dates = self.dates_history[:min_len] if len(self.dates_history) >= min_len else [datetime.now()] * min_len
            else:
                portfolio_values = self.portfolio_values_history
                benchmark_values = self.benchmark_values_history
                dates = self.dates_history if len(self.dates_history) == len(portfolio_values) else [datetime.now()] * len(portfolio_values)

            metrics = self.metrics_calculator.calculate_portfolio_metrics(
                portfolio_values=portfolio_values,
                benchmark_values=benchmark_values,
                dates=dates,
                risk_free_rate=self.config.risk_free_rate
            )

            return metrics

        except Exception as e:
            logger.error(f"计算投资组合指标失败: {e}")
            return None

    def _calculate_agent_behavior_metrics(self) -> Optional['AgentBehaviorMetrics']:
        """
        计算智能体行为指标

        Returns:
            智能体行为指标或None（如果数据不足）
        """
        if not ENHANCED_METRICS_AVAILABLE or len(self.entropy_history) == 0:
            logger.debug("熵值历史数据为空，跳过智能体行为指标计算")
            return None

        try:
            metrics = self.metrics_calculator.calculate_agent_behavior_metrics(
                entropy_values=self.entropy_history,
                position_weights_history=self.position_weights_history
            )

            return metrics

        except Exception as e:
            logger.error(f"计算智能体行为指标失败: {e}")
            return None

    def _calculate_risk_control_metrics(self) -> Optional['RiskControlMetrics']:
        """
        计算风险控制指标

        Returns:
            风险控制指标或None（如果数据不足）
        """
        if not ENHANCED_METRICS_AVAILABLE:
            return None

        # 检查环境是否有回撤控制器
        if not hasattr(self.environment, 'drawdown_controller') or self.environment.drawdown_controller is None:
            logger.debug("环境中没有回撤控制器，跳过风险控制指标计算")
            return None

        try:
            drawdown_controller = self.environment.drawdown_controller

            # 从回撤控制器获取数据
            risk_budget_history = getattr(drawdown_controller.adaptive_risk_budget, 'risk_budget_history', [])
            risk_usage_history = getattr(drawdown_controller.adaptive_risk_budget, 'risk_usage_history', [])
            control_signals = getattr(drawdown_controller, 'control_signal_queue', [])

            # 获取市场状态历史
            market_regime_history = []
            if hasattr(drawdown_controller, 'market_regime_detector') and drawdown_controller.market_regime_detector:
                market_regime_history = getattr(drawdown_controller.market_regime_detector, 'regime_history', [])

            # 转换控制信号为字典格式
            control_signals_dict = []
            for signal in control_signals:
                if hasattr(signal, 'to_dict'):
                    control_signals_dict.append(signal.to_dict())
                elif isinstance(signal, dict):
                    control_signals_dict.append(signal)

            metrics = self.metrics_calculator.calculate_risk_control_metrics(
                risk_budget_history=risk_budget_history,
                risk_usage_history=risk_usage_history,
                control_signals=control_signals_dict,
                market_regime_history=market_regime_history
            )

            return metrics

        except Exception as e:
            logger.error(f"计算风险控制指标失败: {e}")
            return None

    def _log_enhanced_metrics(self, episode: int,
                            portfolio_metrics: Optional['PortfolioMetrics'],
                            agent_metrics: Optional['AgentBehaviorMetrics'],
                            risk_metrics: Optional['RiskControlMetrics']):
        """
        记录增强指标日志

        Args:
            episode: episode编号
            portfolio_metrics: 投资组合指标
            agent_metrics: 智能体行为指标
            risk_metrics: 风险控制指标
        """
        log_lines = [f"=== Episode {episode} 增强指标报告 ==="]

        # 投资组合指标
        if portfolio_metrics:
            log_lines.append("📊 投资组合与市场表现对比指标:")
            log_lines.append(f"  • 夏普比率 (Sharpe Ratio): {portfolio_metrics.sharpe_ratio:.4f}")
            log_lines.append(f"  • 最大回撤 (Max Drawdown): {portfolio_metrics.max_drawdown:.4f}")
            log_lines.append(f"  • Alpha (相对基准超额收益): {portfolio_metrics.alpha:.4f}")
            log_lines.append(f"  • Beta (系统性风险): {portfolio_metrics.beta:.4f}")
            log_lines.append(f"  • 年化收益率 (Annualized Return): {portfolio_metrics.annualized_return:.4f}")
        else:
            log_lines.append("📊 投资组合指标: 数据不足，跳过计算")

        # 智能体行为指标
        if agent_metrics:
            log_lines.append("🤖 智能体行为分析指标:")
            log_lines.append(f"  • 平均熵值 (Mean Entropy): {agent_metrics.mean_entropy:.4f}")
            log_lines.append(f"  • 熵值趋势 (Entropy Trend): {agent_metrics.entropy_trend:.4f}")
            log_lines.append(f"  • 平均持仓集中度 (Position Concentration): {agent_metrics.mean_position_concentration:.4f}")
            log_lines.append(f"  • 换手率 (Turnover Rate): {agent_metrics.turnover_rate:.4f}")
        else:
            log_lines.append("🤖 智能体行为指标: 数据不足，跳过计算")

        # 风险控制指标
        if risk_metrics:
            log_lines.append("🛡️ 风险与回撤控制指标:")
            log_lines.append(f"  • 平均风险预算使用率: {risk_metrics.avg_risk_budget_utilization:.4f}")
            log_lines.append(f"  • 风险预算效率: {risk_metrics.risk_budget_efficiency:.4f}")
            log_lines.append(f"  • 控制信号频率: {risk_metrics.control_signal_frequency:.4f}")
            log_lines.append(f"  • 市场状态稳定性: {risk_metrics.market_regime_stability:.4f}")
        else:
            log_lines.append("🛡️ 风险控制指标: 回撤控制器未启用或数据不足")

        log_lines.append("=" * 50)

        # 输出日志
        for line in log_lines:
            logger.info(line)

    def _update_metrics_histories(self, episode_info: Dict[str, Any], update_info: Dict[str, Any]):
        """
        更新指标历史数据

        Args:
            episode_info: episode信息
            update_info: 智能体更新信息
        """
        if not ENHANCED_METRICS_AVAILABLE:
            return

        # 更新投资组合价值历史
        if 'portfolio_value' in episode_info:
            self.portfolio_values_history.append(episode_info['portfolio_value'])

        # 更新基准价值历史（如果有）
        if 'benchmark_value' in episode_info:
            self.benchmark_values_history.append(episode_info['benchmark_value'])
        elif len(self.portfolio_values_history) > len(self.benchmark_values_history):
            # 如果没有基准数据，使用默认增长率
            if len(self.benchmark_values_history) == 0:
                self.benchmark_values_history.append(self.config.initial_cash)
            else:
                # 假设基准年化收益率为8%
                daily_return = 0.08 / 252
                last_value = self.benchmark_values_history[-1]
                self.benchmark_values_history.append(last_value * (1 + daily_return))

        # 更新日期历史
        self.dates_history.append(datetime.now())

        # 更新智能体行为数据
        if 'policy_entropy' in update_info:
            self.entropy_history.append(update_info['policy_entropy'])

        if 'positions' in episode_info:
            positions = episode_info['positions']
            if isinstance(positions, np.ndarray):
                self.position_weights_history.append(positions.copy())

    def get_enhanced_training_stats(self) -> Dict[str, Any]:
        """
        获取增强训练统计信息

        Returns:
            增强训练统计信息
        """
        # 获取基础统计（从agent或创建空字典）
        base_stats = self.agent.get_training_stats() if hasattr(self.agent, 'get_training_stats') else {}

        if not ENHANCED_METRICS_AVAILABLE:
            return base_stats

        # 添加增强统计
        enhanced_stats = {
            'portfolio_values_count': len(self.portfolio_values_history),
            'entropy_values_count': len(self.entropy_history),
            'position_weights_count': len(self.position_weights_history),
            'latest_portfolio_value': self.portfolio_values_history[-1] if self.portfolio_values_history else 0,
            'latest_entropy': self.entropy_history[-1] if self.entropy_history else 0,
        }

        # 如果有足够数据，计算最新指标
        if len(self.portfolio_values_history) > 1:
            try:
                latest_portfolio_metrics = self._calculate_portfolio_metrics()
                if latest_portfolio_metrics:
                    enhanced_stats.update({
                        'latest_sharpe_ratio': latest_portfolio_metrics.sharpe_ratio,
                        'latest_max_drawdown': latest_portfolio_metrics.max_drawdown,
                        'latest_alpha': latest_portfolio_metrics.alpha,
                        'latest_beta': latest_portfolio_metrics.beta,
                        'latest_annualized_return': latest_portfolio_metrics.annualized_return
                    })
            except Exception as e:
                logger.debug(f"计算最新投资组合指标失败: {e}")

        if len(self.entropy_history) > 0:
            try:
                latest_agent_metrics = self._calculate_agent_behavior_metrics()
                if latest_agent_metrics:
                    enhanced_stats.update({
                        'latest_mean_entropy': latest_agent_metrics.mean_entropy,
                        'latest_entropy_trend': latest_agent_metrics.entropy_trend,
                        'latest_position_concentration': latest_agent_metrics.mean_position_concentration,
                        'latest_turnover_rate': latest_agent_metrics.turnover_rate
                    })
            except Exception as e:
                logger.debug(f"计算最新智能体行为指标失败: {e}")

        # 合并统计信息
        base_stats.update(enhanced_stats)
        return base_stats

    def reset_enhanced_histories(self):
        """重置增强历史数据"""
        if not ENHANCED_METRICS_AVAILABLE:
            return

        self.portfolio_values_history.clear()
        self.benchmark_values_history.clear()
        self.dates_history.clear()
        self.entropy_history.clear()
        self.position_weights_history.clear()
        self.risk_budget_history.clear()
        self.risk_usage_history.clear()
        self.control_signals_history.clear()
        self.market_regime_history.clear()

        logger.info("增强历史数据已重置")


# 为了向后兼容，创建别名
EnhancedRLTrainer = RLTrainer
EnhancedTrainingConfig = TrainingConfig