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
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import pickle
from pathlib import Path
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .data_split_strategy import SplitResult

logger = logging.getLogger(__name__)


class ExperienceDataset(Dataset):
    """经验回放数据集，支持多进程加载"""
    
    def __init__(self, experiences: List[Tuple], transform=None):
        """
        初始化数据集
        
        Args:
            experiences: 经验数据列表 [(state, action, reward, next_state, done), ...]
            transform: 可选的数据转换函数
        """
        self.experiences = experiences
        self.transform = transform
    
    def __len__(self):
        return len(self.experiences)
    
    def __getitem__(self, idx):
        experience = self.experiences[idx]
        
        if self.transform:
            experience = self.transform(experience)
        
        return experience


class ParallelEnvironmentManager:
    """并行环境管理器"""
    
    def __init__(self, env_factory, num_envs: int = 4):
        """
        初始化并行环境管理器
        
        Args:
            env_factory: 环境工厂函数
            num_envs: 并行环境数量
        """
        self.env_factory = env_factory
        self.num_envs = num_envs
        self.envs = []
        self.executor = None
    
    def _create_env(self):
        """创建单个环境实例"""
        return self.env_factory()
    
    def initialize(self):
        """初始化并行环境"""
        logger.info(f"初始化 {self.num_envs} 个并行环境...")
        
        # 使用进程池创建环境
        with ProcessPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [executor.submit(self._create_env) for _ in range(self.num_envs)]
            self.envs = [future.result() for future in futures]
        
        logger.info(f"并行环境初始化完成")
    
    def reset_all(self):
        """重置所有环境"""
        if not self.envs:
            raise RuntimeError("环境未初始化")
        
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [executor.submit(env.reset) for env in self.envs]
            states = [future.result() for future in futures]
        
        return states
    
    def step_all(self, actions):
        """并行执行所有环境的step操作"""
        if not self.envs:
            raise RuntimeError("环境未初始化")
        
        if len(actions) != len(self.envs):
            raise ValueError(f"actions数量 {len(actions)} 与环境数量 {len(self.envs)} 不匹配")
        
        with ThreadPoolExecutor(max_workers=self.num_envs) as executor:
            futures = [executor.submit(env.step, action) for env, action in zip(self.envs, actions)]
            results = [future.result() for future in futures]
        
        return results
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
        self.envs.clear()


@dataclass
class TrainingConfig:
    """训练配置"""
    n_episodes: int = 5000
    max_steps_per_episode: int = 180  # 降低以匹配实际数据长度
    batch_size: int = 256
    learning_rate: float = 3e-4
    buffer_size: int = 1000000
    gamma: float = 0.99  # 折扣因子
    tau: float = 0.005  # 软更新参数

    # 验证和保存频率
    validation_frequency: int = 50
    save_frequency: int = 100

    # 早停参数
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    early_stopping_mode: str = 'max'  # 'max' or 'min'

    # 学习率调度
    lr_scheduler_step_size: int = 1000
    lr_scheduler_gamma: float = 0.95

    # 梯度裁剪
    gradient_clip_norm: Optional[float] = 1.0

    # 训练稳定性
    warmup_episodes: int = 1  # 降低以便测试时模型能立即开始学习
    update_frequency: int = 1
    target_update_frequency: int = 1

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
    
    # 多核并行优化参数
    enable_multiprocessing: bool = True            # 启用多进程优化
    num_workers: int = field(default_factory=lambda: min(8, multiprocessing.cpu_count()))  # 数据加载工作进程数
    parallel_environments: int = field(default_factory=lambda: min(4, multiprocessing.cpu_count() // 2))  # 并行环境数量
    data_loader_workers: int = field(default_factory=lambda: min(4, multiprocessing.cpu_count() // 2))    # DataLoader工作线程数
    pin_memory: bool = True                        # GPU内存固定
    persistent_workers: bool = True                # 持久化工作进程
    prefetch_factor: int = 2                       # 预取因子
    
    # GPU优化参数
    enable_mixed_precision: bool = True           # 启用混合精度训练
    enable_cudnn_benchmark: bool = True           # 启用cuDNN基准测试
    non_blocking_transfer: bool = True            # 非阻塞数据传输

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


class EarlyStopping:
    """早停机制"""

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


class DrawdownEarlyStopping:
    """基于回撤的早停机制"""

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


class TrainingMetrics:
    """训练指标收集器"""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.temperature_losses: List[float] = []
        self.validation_scores: List[float] = []
        self.timestamps: List[datetime] = []

    def add_episode_metrics(self, reward: float, length: int,
                          actor_loss: float = 0.0, critic_loss: float = 0.0,
                          temperature_loss: float = 0.0):
        """添加episode指标"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        self.temperature_losses.append(temperature_loss)
        self.timestamps.append(datetime.now())

    def add_validation_score(self, score: float):
        """添加验证分数"""
        self.validation_scores.append(score)

    def get_statistics(self, window: Optional[int] = None) -> Dict[str, float]:
        """获取统计信息"""
        if len(self.episode_rewards) == 0:
            return {}

        if window is not None:
            rewards = self.episode_rewards[-window:]
            lengths = self.episode_lengths[-window:]
            actor_losses = self.actor_losses[-window:]
            critic_losses = self.critic_losses[-window:]
        else:
            rewards = self.episode_rewards
            lengths = self.episode_lengths
            actor_losses = self.actor_losses
            critic_losses = self.critic_losses

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths),
            'mean_actor_loss': np.mean(actor_losses),
            'mean_critic_loss': np.mean(critic_losses),
            'total_episodes': len(self.episode_rewards)
        }

    def get_recent_statistics(self, window: int = 100) -> Dict[str, float]:
        """获取最近window个episode的统计信息"""
        return self.get_statistics(window=window)


class RLTrainer:
    """强化学习训练器"""

    def __init__(self, config: TrainingConfig, environment, agent, data_split: SplitResult):
        """
        初始化训练器

        Args:
            config: 训练配置
            environment: 交易环境
            agent: 强化学习智能体
            data_split: 数据划分结果
        """
        self.config = config
        self.environment = environment
        self.agent = agent
        self.data_split = data_split

        # 初始化训练组件
        self.metrics = TrainingMetrics()
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode=config.early_stopping_mode
        )

        # 初始化回撤控制相关组件
        if config.enable_drawdown_monitoring:
            self.drawdown_early_stopping = DrawdownEarlyStopping(
                max_drawdown=config.max_training_drawdown,
                patience=config.early_stopping_patience // 2  # 回撤早停耐心值更小
            )
            self.drawdown_metrics = []
            logger.info("回撤监控已启用")
        else:
            self.drawdown_early_stopping = None
            self.drawdown_metrics = []

        # 自适应训练参数
        self.adaptive_learning_enabled = config.enable_adaptive_learning
        self.current_lr_factor = 1.0
        self.performance_history = []

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
        
        # 设置PyTorch多进程上下文
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果已经设置了start method，忽略错误
            pass
        
        # 设置cuDNN基准测试（如果启用GPU优化）
        if self.config.enable_cudnn_benchmark and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN基准测试已启用")
        
        # 初始化混合精度训练
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("混合精度训练已启用")
        else:
            self.scaler = None
        
        # 初始化并行环境管理器（如果配置了多个环境）
        if self.config.parallel_environments > 1:
            # 创建环境工厂函数
            def env_factory():
                # 这里需要根据实际环境类型创建副本
                # 暂时使用占位符，在实际使用时需要提供合适的环境复制机制
                return self.environment
            
            self.parallel_env_manager = ParallelEnvironmentManager(
                env_factory, 
                self.config.parallel_environments
            )
            logger.info(f"并行环境管理器已初始化，{self.config.parallel_environments} 个环境")
        else:
            self.parallel_env_manager = None
        
        # 预分配经验数据集和DataLoader（在实际训练中使用）
        self.experience_dataset = None
        self.data_loader = None

    def _get_current_learning_rate(self, episode: int) -> float:
        """获取当前学习率"""
        # 简单的指数衰减
        decay_rate = self.config.lr_scheduler_gamma
        decay_steps = self.config.lr_scheduler_step_size

        decay_factor = decay_rate ** (episode // decay_steps)
        return self.config.learning_rate * decay_factor

    def _run_episode(self, episode_num: int, training: bool = True) -> Tuple[float, int]:
        """
        运行单个episode

        Args:
            episode_num: episode编号
            training: 是否为训练模式

        Returns:
            Tuple[float, int]: episode奖励和长度
        """
        if training:
            self.agent.train()
        else:
            self.agent.eval()

        obs = self.environment.reset()
        episode_reward = 0.0
        episode_length = 0

        for step in range(self.config.max_steps_per_episode):
            # 选择动作
            action_tensor = self.agent.get_action(obs, deterministic=not training)

            # 将PyTorch张量转换为numpy数组传递给环境
            if isinstance(action_tensor, torch.Tensor):
                action = action_tensor.detach().cpu().numpy()
            else:
                action = action_tensor

            # 执行动作
            next_obs, reward, done, info = self.environment.step(action)

            episode_reward += reward
            episode_length += 1

            # 如果是训练模式，存储经验并更新智能体
            if training and hasattr(self.agent, 'replay_buffer'):
                # 创建经验对象并存储到回放缓冲区
                from ..models.replay_buffer import Experience

                # 将字典观察转换为编码张量（使用Transformer编码，推理模式）
                if isinstance(obs, dict):
                    state_tensor = self.agent.encode_observation(obs, training=False)
                else:
                    state_tensor = obs

                if isinstance(next_obs, dict):
                    next_state_tensor = self.agent.encode_observation(next_obs, training=False)
                else:
                    next_state_tensor = next_obs

                experience = Experience(
                    state=state_tensor,
                    action=action_tensor,  # 使用原始张量而不是numpy数组
                    reward=reward,
                    next_state=next_state_tensor,
                    done=done
                )
                self.agent.add_experience(experience)

                # 调试日志：验证total_env_steps是否正确更新（降低频率）
                if step % 100 == 0:  # 每100步记录一次
                    logger.debug(f"Episode {episode_num}, Step {step}: total_env_steps = {self.agent.total_env_steps}, can_update = {self.agent.can_update()}")


            obs = next_obs

            if done:
                break

        return episode_reward, episode_length

    def _update_agent(self) -> Dict[str, float]:
        """更新智能体参数"""
        if hasattr(self.agent, 'update'):
            return self.agent.update(update_actor=True)
        return {}

    def _validate(self) -> float:
        """运行验证"""
        logger.debug("开始验证...")

        validation_rewards = []
        n_validation_episodes = 5  # 运行5个验证episode

        for _ in range(n_validation_episodes):
            reward, _ = self._run_episode(episode_num=-1, training=False)
            validation_rewards.append(reward)

        validation_score = np.mean(validation_rewards)
        validation_std = np.std(validation_rewards)
        
        # 添加验证稳定性检查
        if validation_std > validation_score * 0.5:  # 标准差超过均值50%
            logger.warning(f"验证结果不稳定: 标准差 {validation_std:.4f} 过大，模型泛化能力可能不足")
        
        logger.info(f"验证完成，平均奖励: {validation_score:.4f}")

        return validation_score

    def save_checkpoint(self, filepath: str, episode: int):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'config': self.config,
            'metrics': self.metrics,
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

        episode = checkpoint['episode']
        self.metrics = checkpoint['metrics']

        # 恢复早停状态
        early_stopping_state = checkpoint['early_stopping_state']
        self.early_stopping.best_score = early_stopping_state['best_score']
        self.early_stopping.counter = early_stopping_state['counter']
        self.early_stopping.early_stop = early_stopping_state['early_stop']

        # 加载智能体状态（如果存在）
        if 'agent_path' in checkpoint and hasattr(self.agent, 'load'):
            self.agent.load(checkpoint['agent_path'])

        logger.debug(f"检查点已从 {filepath} 加载，episode: {episode}")
        return episode

    def _log_episode_stats(self, episode: int, reward: float, length: int,
                          update_info: Dict[str, float]):
        """记录episode统计信息"""
        # 根据训练进度调整日志频率
        if self.config.n_episodes <= 10:
            # 少量episode时每个都记录
            log_frequency = 1
        elif self.config.n_episodes <= 100:
            # 中等数量episode时每5个记录一次
            log_frequency = 5
        else:
            # 大量episode时每20个记录一次
            log_frequency = 20

        if episode % log_frequency == 0:
            recent_stats = self.metrics.get_recent_statistics(window=min(10, episode))

            log_msg = (
                f"Episode {episode:4d} | "
                f"Reward: {reward:7.2f} | "
                f"Length: {length:3d} | "
                f"Avg Reward ({min(10, episode)}): {recent_stats.get('mean_reward', 0):7.2f}"
            )

            if update_info:
                log_msg += (
                    f" | Actor Loss: {update_info.get('actor_loss', 0):.4f}"
                    f" | Critic Loss: {update_info.get('critic_loss', 0):.4f}"
                )
                
                # 添加损失异常检测
                actor_loss = update_info.get('actor_loss', 0)
                critic_loss = update_info.get('critic_loss', 0)
                
                if abs(actor_loss) > 10.0:
                    logger.warning(f"Actor损失异常: {actor_loss:.4f}, 可能存在梯度爆炸")
                if critic_loss > 50.0:
                    logger.warning(f"Critic损失过高: {critic_loss:.4f}, 学习可能不稳定")

            logger.info(log_msg)
            
            # 添加训练稳定性检查
            if episode >= 20:
                reward_std = recent_stats.get('std_reward', 0)
                mean_reward = recent_stats.get('mean_reward', 0)
                if mean_reward != 0 and reward_std / abs(mean_reward) > 1.0:
                    logger.warning(f"训练不稳定: 奖励方差过大 {reward_std:.2f}, 可能需要调整学习率")

    def _create_data_loader(self, experiences: List[Tuple]) -> DataLoader:
        """创建支持多进程的数据加载器"""
        if not experiences:
            return None
        
        dataset = ExperienceDataset(experiences)
        
        # 配置DataLoader参数
        dataloader_kwargs = {
            'batch_size': self.config.batch_size,
            'shuffle': True,
            'pin_memory': self.config.pin_memory and torch.cuda.is_available(),
            'prefetch_factor': self.config.prefetch_factor
        }
        
        # 只有在多进程启用时才使用num_workers
        if self.config.enable_multiprocessing and self.config.data_loader_workers > 0:
            dataloader_kwargs.update({
                'num_workers': self.config.data_loader_workers,
                'persistent_workers': self.config.persistent_workers
            })
        
        return DataLoader(dataset, **dataloader_kwargs)
    
    def _run_parallel_episodes(self, episode_nums: List[int]) -> List[Tuple[float, int]]:
        """并行运行多个episodes"""
        if not self.parallel_env_manager:
            # 如果没有并行环境管理器，退回到顺序执行
            return [self._run_episode(ep, training=True) for ep in episode_nums]
        
        try:
            # 重置所有环境
            states = self.parallel_env_manager.reset_all()
            
            results = []
            for i, episode_num in enumerate(episode_nums):
                # 这里需要根据实际的并行执行逻辑来实现
                # 暂时使用简化版本
                episode_reward, episode_length = self._run_episode(episode_num, training=True)
                results.append((episode_reward, episode_length))
            
            return results
        except Exception as e:
            logger.warning(f"并行episode执行失败，退回到顺序执行: {e}")
            return [self._run_episode(ep, training=True) for ep in episode_nums]
    
    def _update_agent_with_dataloader(self, dataloader: DataLoader) -> Dict[str, float]:
        """使用DataLoader进行批量更新"""
        if not dataloader:
            return self._update_agent()
        
        total_losses = {'actor_loss': 0.0, 'critic_loss': 0.0, 'temperature_loss': 0.0}
        batch_count = 0
        
        for batch_experiences in dataloader:
            # 这里需要根据实际的智能体更新逻辑来实现
            # 暂时使用简化版本
            update_info = self._update_agent()
            
            for key in total_losses:
                total_losses[key] += update_info.get(key, 0.0)
            batch_count += 1
        
        # 计算平均损失
        if batch_count > 0:
            for key in total_losses:
                total_losses[key] /= batch_count
        
        return total_losses

    def train(self):
        """执行训练"""
        logger.info(f"开始训练，总episodes: {self.config.n_episodes}")
        logger.info(f"多核优化: {'启用' if self.config.enable_multiprocessing else '禁用'}")
        if self.config.enable_multiprocessing:
            logger.info(f"  - 数据工作进程: {self.config.num_workers}")
            logger.info(f"  - 并行环境: {self.config.parallel_environments}")
            logger.info(f"  - 混合精度: {'启用' if self.config.enable_mixed_precision else '禁用'}")

        # 系统状态检查
        if self.config.enable_drawdown_monitoring:
            logger.info(f"回撤监控已启用: 最大回撤阈值 {self.config.max_training_drawdown:.3f}")
        else:
            logger.warning("回撤监控未启用，无法进行风险控制")

        start_time = time.time()
        best_validation_score = float('-inf') if self.config.early_stopping_mode == 'max' else float('inf')

        for episode in range(1, self.config.n_episodes + 1):
            # 运行训练episode
            episode_reward, episode_length = self._run_episode(episode, training=True)

            # 更新智能体（获取损失信息）
            update_info = {}
            if episode > self.config.warmup_episodes:
                update_info = self._update_agent()

            # 记录指标
            self.metrics.add_episode_metrics(
                reward=episode_reward,
                length=episode_length,
                actor_loss=update_info.get('actor_loss', 0.0),
                critic_loss=update_info.get('critic_loss', 0.0),
                temperature_loss=update_info.get('temperature_loss', 0.0)
            )

            # 回撤监控和自适应参数调整
            if self.config.enable_drawdown_monitoring:
                self._monitor_drawdown(episode_reward, episode)

            if self.adaptive_learning_enabled:
                self._adapt_training_parameters(episode_reward, episode)

            # 记录日志
            self._log_episode_stats(episode, episode_reward, episode_length, update_info)

            # 验证
            if episode % self.config.validation_frequency == 0:
                validation_score = self._validate()
                self.metrics.add_validation_score(validation_score)
                
                # 添加训练-验证性能对比分析
                recent_train_reward = np.mean(self.metrics.episode_rewards[-10:]) if len(self.metrics.episode_rewards) >= 10 else 0
                if recent_train_reward > 0:
                    performance_gap = abs(validation_score - recent_train_reward) / recent_train_reward
                    if performance_gap > 0.2:  # 性能差异超过20%
                        logger.warning(f"训练-验证性能差异过大: {performance_gap:.3f}, 可能存在过拟合")
                    else:
                        logger.info(f"模型泛化良好: 训练-验证性能差异 {performance_gap:.3f}")

                # 早停检查
                if self.early_stopping.step(validation_score):
                    logger.info(f"触发性能早停，episode: {episode}")
                    break

                # 回撤早停检查（检查是否已经触发早停，不重复调用step）
                if (self.drawdown_early_stopping and
                    self.drawdown_early_stopping.early_stop):
                    logger.info(f"触发回撤早停，episode: {episode}, "
                              f"当前回撤: {self.drawdown_early_stopping.get_current_drawdown():.4f}")
                    break

                # 更新最佳验证分数
                if ((self.config.early_stopping_mode == 'max' and validation_score > best_validation_score) or
                    (self.config.early_stopping_mode == 'min' and validation_score < best_validation_score)):
                    best_validation_score = validation_score
                    # 保存最佳模型
                    best_model_path = self.save_dir / "best_model.pth"
                    self.save_checkpoint(str(best_model_path), episode)

            # 定期保存检查点
            if episode % self.config.save_frequency == 0:
                checkpoint_path = self.save_dir / f"checkpoint_episode_{episode}.pth"
                self.save_checkpoint(str(checkpoint_path), episode)

        training_time = time.time() - start_time
        logger.info(f"训练完成，总用时: {training_time:.2f}秒")

        # 训练结果分析
        final_stats = self.metrics.get_statistics()
        if final_stats:
            logger.info(f"训练总结: 平均奖励 {final_stats.get('mean_reward', 0):.4f}, "
                       f"奖励标准差 {final_stats.get('std_reward', 0):.4f}")
            
            # 检查训练是否收敛
            if len(self.metrics.validation_scores) >= 2:
                recent_improvement = self.metrics.validation_scores[-1] - self.metrics.validation_scores[-2]
                if abs(recent_improvement) < 0.01:
                    logger.info("模型已收敛: 验证分数变化很小")
                else:
                    logger.info(f"模型仍在学习: 最近验证改进 {recent_improvement:.4f}")

        # 保存最终模型
        final_model_path = self.save_dir / "final_model.pth"
        self.save_checkpoint(str(final_model_path), episode)

        # 清理多核资源
        self._cleanup_multicore_resources()
        
        # 返回训练统计
        return self.metrics.get_statistics()
    
    def _cleanup_multicore_resources(self):
        """清理多核资源"""
        if self.parallel_env_manager:
            self.parallel_env_manager.close()
            logger.info("并行环境资源已清理")
        
        if hasattr(self, 'data_loader') and self.data_loader:
            # DataLoader会自动清理worker进程
            del self.data_loader
            logger.debug("数据加载器资源已清理")

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """评估智能体性能"""
        logger.info(f"开始评估，episodes: {n_episodes}")

        self.agent.eval()
        evaluation_rewards = []
        evaluation_lengths = []

        for episode in range(n_episodes):
            reward, length = self._run_episode(episode, training=False)
            evaluation_rewards.append(reward)
            evaluation_lengths.append(length)

        evaluation_stats = {
            'mean_reward': np.mean(evaluation_rewards),
            'std_reward': np.std(evaluation_rewards),
            'min_reward': np.min(evaluation_rewards),
            'max_reward': np.max(evaluation_rewards),
            'mean_length': np.mean(evaluation_lengths),
            'total_episodes': n_episodes
        }

        logger.info(f"评估完成，平均奖励: {evaluation_stats['mean_reward']:.4f}")
        logger.debug(f"详细评估结果: {evaluation_stats}")
        return evaluation_stats

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