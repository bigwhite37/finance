"""
强化学习训练器实现
实现RLTrainer类和强化学习训练循环，包括早停机制、学习率调度和梯度裁剪
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import pickle
from pathlib import Path
import time

from .data_split_strategy import SplitResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    n_episodes: int = 5000
    max_steps_per_episode: int = 252
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
    warmup_episodes: int = 100
    update_frequency: int = 1
    target_update_frequency: int = 1
    
    # 保存路径
    save_dir: str = "./checkpoints"
    
    # 随机种子
    random_seed: Optional[int] = None
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        
        # 设置随机种子
        if config.random_seed is not None:
            self._set_random_seed(config.random_seed)
        
        # 创建保存目录
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化学习率调度器（如果智能体支持）
        self._setup_lr_scheduler()
        
        logger.info(f"训练器初始化完成，配置: {config}")
    
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
            action = self.agent.get_action(obs, deterministic=not training)
            
            # 执行动作
            next_obs, reward, done, info = self.environment.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # 如果是训练模式，存储经验并更新智能体
            if training and hasattr(self.agent, 'replay_buffer'):
                # 创建经验对象并存储到回放缓冲区
                from ..models.replay_buffer import Experience
                
                # 将字典观察转换为张量（如果需要）
                if isinstance(obs, dict):
                    state_tensor = self.agent._flatten_dict_observation(obs)
                else:
                    state_tensor = obs
                    
                if isinstance(next_obs, dict):
                    next_state_tensor = self.agent._flatten_dict_observation(next_obs)
                else:
                    next_state_tensor = next_obs
                
                experience = Experience(
                    state=state_tensor,
                    action=action,
                    reward=reward,
                    next_state=next_state_tensor,
                    done=done
                )
                self.agent.replay_buffer.add(experience)
                
                # 定期更新智能体
                if (episode_num > self.config.warmup_episodes and 
                    step % self.config.update_frequency == 0):
                    self._update_agent()
            
            obs = next_obs
            
            if done:
                break
        
        return episode_reward, episode_length
    
    def _update_agent(self) -> Dict[str, float]:
        """更新智能体参数"""
        if hasattr(self.agent, 'update'):
            return self.agent.update(
                replay_buffer=getattr(self.agent, 'replay_buffer', None),
                batch_size=self.config.batch_size
            )
        return {}
    
    def _validate(self) -> float:
        """运行验证"""
        logger.info("开始验证...")
        
        validation_rewards = []
        n_validation_episodes = 5  # 运行5个验证episode
        
        for _ in range(n_validation_episodes):
            reward, _ = self._run_episode(episode_num=-1, training=False)
            validation_rewards.append(reward)
        
        validation_score = np.mean(validation_rewards)
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
        
        logger.info(f"检查点已保存到: {filepath}")
    
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
        
        logger.info(f"检查点已从 {filepath} 加载，episode: {episode}")
        return episode
    
    def _log_episode_stats(self, episode: int, reward: float, length: int, 
                          update_info: Dict[str, float]):
        """记录episode统计信息"""
        if episode % 10 == 0:  # 每10个episode记录一次
            recent_stats = self.metrics.get_recent_statistics(window=10)
            
            log_msg = (
                f"Episode {episode:4d} | "
                f"Reward: {reward:7.2f} | "
                f"Length: {length:3d} | "
                f"Avg Reward (10): {recent_stats.get('mean_reward', 0):7.2f}"
            )
            
            if update_info:
                log_msg += (
                    f" | Actor Loss: {update_info.get('actor_loss', 0):.4f}"
                    f" | Critic Loss: {update_info.get('critic_loss', 0):.4f}"
                )
            
            logger.info(log_msg)
    
    def train(self):
        """执行训练"""
        logger.info(f"开始训练，总episodes: {self.config.n_episodes}")
        
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
            
            # 记录日志
            self._log_episode_stats(episode, episode_reward, episode_length, update_info)
            
            # 验证
            if episode % self.config.validation_frequency == 0:
                validation_score = self._validate()
                self.metrics.add_validation_score(validation_score)
                
                # 早停检查
                if self.early_stopping.step(validation_score):
                    logger.info(f"触发早停，episode: {episode}")
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
        
        # 保存最终模型
        final_model_path = self.save_dir / "final_model.pth"
        self.save_checkpoint(str(final_model_path), episode)
        
        # 返回训练统计
        return self.metrics.get_statistics()
    
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
        
        logger.info(f"评估完成: {evaluation_stats}")
        return evaluation_stats