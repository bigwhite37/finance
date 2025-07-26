"""
混合采样器

实现离线/在线数据的智能混合，支持动态采样比例调整、重要性权重计算和最小离线样本比例保护。
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from threading import Lock

from data.offline_dataset import OfflineDataset
from buffers.online_replay_buffer import OnlineReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class MixtureSamplerConfig:
    """混合采样器配置"""
    initial_rho: float = 0.2  # 初始在线采样比例
    rho_increment: float = 0.01  # 采样比例增长率
    max_rho: float = 1.0  # 最大在线采样比例
    min_offline_ratio: float = 0.1  # 最小离线样本比例保护
    batch_size: int = 256
    importance_weight_clip: float = 10.0  # 重要性权重裁剪
    enable_importance_weighting: bool = True
    adaptive_rho: bool = True  # 是否启用自适应采样比例
    performance_threshold: float = 0.05  # 性能阈值，用于自适应调整


class MixtureSampler:
    """离线/在线混合采样器
    
    实现动态采样比例调整 ρ(t) = min(1, ρ₀ + α·t)，
    开发重要性权重计算，纠正离线数据的分布偏差，
    添加最小离线样本比例保护，确保训练稳定性。
    """
    
    def __init__(self, 
                 offline_dataset: OfflineDataset,
                 online_buffer: OnlineReplayBuffer,
                 config: MixtureSamplerConfig):
        """
        初始化混合采样器
        
        Args:
            offline_dataset: 离线数据集
            online_buffer: 在线回放缓冲区
            config: 采样器配置
        """
        self.offline_dataset = offline_dataset
        self.online_buffer = online_buffer
        self.config = config
        
        # 采样比例相关
        self.current_rho = config.initial_rho
        self.episode_count = 0
        self.total_episodes = 1000  # 默认总episode数
        
        # 重要性权重相关
        self.offline_policy_cache = {}
        self.current_policy = None
        
        # 性能监控
        self.performance_history = []
        self.last_performance = 0.0
        
        # 线程安全
        self._lock = Lock()
        
        logger.info(f"初始化混合采样器，初始ρ={self.current_rho}")
        
    def sample_mixed_batch(self, 
                          batch_size: Optional[int] = None,
                          rho: Optional[float] = None) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        混合采样批次数据
        
        Args:
            batch_size: 批次大小
            rho: 在线采样比例，如果为None则使用当前比例
            
        Returns:
            (混合批次数据, 重要性权重)
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if rho is None:
            rho = self.current_rho
            
        # 应用最小离线样本比例保护
        effective_rho = min(rho, 1.0 - self.config.min_offline_ratio)
        
        with self._lock:
            # 计算在线和离线样本数量
            online_samples = int(batch_size * effective_rho)
            offline_samples = batch_size - online_samples
            
            logger.debug(f"采样批次: 在线={online_samples}, 离线={offline_samples}, ρ={effective_rho:.3f}")
            
            # 采样在线数据
            online_batch = self._sample_online_data(online_samples)
            
            # 采样离线数据
            offline_batch = self._sample_offline_data(offline_samples)
            
            # 合并批次数据
            mixed_batch = self._merge_batches(online_batch, offline_batch)
            
            # 计算重要性权重
            importance_weights = self._calculate_importance_weights(
                mixed_batch, online_samples, offline_samples, effective_rho
            )
            
            return mixed_batch, importance_weights
            
    def _sample_online_data(self, num_samples: int) -> Dict[str, Any]:
        """采样在线数据"""
        if num_samples == 0 or len(self.online_buffer) == 0:
            return self._empty_batch()
            
        # 从在线缓冲区采样
        online_batch = self.online_buffer.sample_batch(num_samples)
        online_batch['data_source'] = 'online'
        
        return online_batch
        
    def _sample_offline_data(self, num_samples: int) -> Dict[str, Any]:
        """采样离线数据"""
        if num_samples == 0 or len(self.offline_dataset) == 0:
            return self._empty_batch()
            
        # 从离线数据集随机采样
        indices = np.random.choice(len(self.offline_dataset), num_samples, replace=True)
        
        states_list = []
        targets_list = []
        
        for idx in indices:
            state, target = self.offline_dataset[idx]
            states_list.append(state.numpy())
            targets_list.append(target.numpy())
            
        # 构造批次格式
        offline_batch = {
            'states': np.array(states_list),
            'actions': np.array(targets_list),  # 离线数据中target作为action
            'rewards': np.zeros(num_samples),   # 离线数据没有即时奖励
            'next_states': np.array(states_list),  # 简化处理
            'dones': np.zeros(num_samples, dtype=bool),
            'data_source': 'offline',
            'indices': indices,
            'weights': np.ones(num_samples)
        }
        
        return offline_batch
        
    def _merge_batches(self, 
                      online_batch: Dict[str, Any], 
                      offline_batch: Dict[str, Any]) -> Dict[str, Any]:
        """合并在线和离线批次数据"""
        merged_batch = {}
        
        # 获取所有可能的键
        all_keys = set(online_batch.keys()) | set(offline_batch.keys())
        
        for key in all_keys:
            if key in ['data_source', 'indices', 'weights']:
                continue  # 这些键需要特殊处理
                
            online_data = online_batch.get(key, np.array([]))
            offline_data = offline_batch.get(key, np.array([]))
            
            # 确保数据类型一致
            if len(online_data) > 0 and len(offline_data) > 0:
                # 都有数据，合并
                if isinstance(online_data, list) and isinstance(offline_data, list):
                    merged_batch[key] = online_data + offline_data
                else:
                    merged_batch[key] = np.concatenate([online_data, offline_data])
            elif len(online_data) > 0:
                # 只有在线数据
                merged_batch[key] = online_data
            elif len(offline_data) > 0:
                # 只有离线数据
                merged_batch[key] = offline_data
            else:
                # 都没有数据
                merged_batch[key] = np.array([])
                
        # 特殊处理数据源标记
        online_sources = ['online'] * len(online_batch.get('states', []))
        offline_sources = ['offline'] * len(offline_batch.get('states', []))
        merged_batch['data_sources'] = online_sources + offline_sources
        
        return merged_batch
        
    def _calculate_importance_weights(self, 
                                    mixed_batch: Dict[str, Any],
                                    online_samples: int,
                                    offline_samples: int,
                                    rho: float) -> np.ndarray:
        """
        计算重要性权重
        
        重要性权重公式: w_i = (1-ρ) * π_θ(a_i|s_i) / β(a_i|s_i)
        其中 π_θ 是当前策略，β 是行为策略（离线数据的策略）
        """
        if not self.config.enable_importance_weighting:
            # 如果不启用重要性权重，返回均匀权重
            total_samples = online_samples + offline_samples
            return np.ones(total_samples)
            
        weights = []
        
        # 在线样本权重为1（无需纠正）
        online_weights = np.ones(online_samples)
        weights.extend(online_weights)
        
        # 离线样本需要计算重要性权重
        if offline_samples > 0 and self.current_policy is not None:
            offline_weights = self._compute_offline_importance_weights(
                mixed_batch, online_samples, offline_samples, rho
            )
            weights.extend(offline_weights)
        else:
            # 如果没有当前策略信息，使用简单的权重
            offline_weights = np.full(offline_samples, 1.0 - rho)
            weights.extend(offline_weights)
            
        weights = np.array(weights)
        
        # 权重裁剪，防止极端值
        weights = np.clip(weights, 1.0 / self.config.importance_weight_clip, 
                         self.config.importance_weight_clip)
        
        # 权重归一化
        weights = weights / np.mean(weights)
        
        return weights
        
    def _compute_offline_importance_weights(self, 
                                          mixed_batch: Dict[str, Any],
                                          online_samples: int,
                                          offline_samples: int,
                                          rho: float) -> np.ndarray:
        """计算离线样本的重要性权重"""
        # 获取离线样本的状态和动作
        offline_states = mixed_batch['states'][online_samples:]
        offline_actions = mixed_batch['actions'][online_samples:]
        
        # 计算当前策略的动作概率
        current_log_probs = self._evaluate_current_policy(offline_states, offline_actions)
        
        # 计算行为策略的动作概率（从缓存或估计）
        behavior_log_probs = self._evaluate_behavior_policy(offline_states, offline_actions)
        
        # 重要性权重: (1-ρ) * π_θ(a|s) / β(a|s)
        log_importance_ratios = current_log_probs - behavior_log_probs
        importance_ratios = np.exp(np.clip(log_importance_ratios, -5, 5))  # 防止数值溢出
        
        weights = (1.0 - rho) * importance_ratios
        
        return weights
        
    def _evaluate_current_policy(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """评估当前策略的动作概率"""
        if self.current_policy is None:
            # 如果没有当前策略，返回均匀分布的对数概率
            return np.zeros(len(states))
            
        try:
            # 这里需要根据实际的策略接口来实现
            # 假设策略有evaluate_actions方法
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.FloatTensor(actions)
                log_probs = self.current_policy.evaluate_actions(states_tensor, actions_tensor)
                return log_probs.cpu().numpy()
        except Exception as e:
            logger.warning(f"评估当前策略失败: {e}")
            return np.zeros(len(states))
            
    def _evaluate_behavior_policy(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """评估行为策略的动作概率"""
        # 简化实现：假设行为策略是均匀分布或使用缓存的估计
        # 在实际应用中，这可能需要从离线数据中学习或估计
        
        # 使用简单的高斯分布假设
        # 假设动作在[-1, 1]范围内，标准差为0.5
        log_probs_per_dim = -0.5 * ((actions / 0.5) ** 2) - np.log(0.5 * np.sqrt(2 * np.pi))
        
        # 对所有动作维度求和，得到总的log_prob
        if len(log_probs_per_dim.shape) > 1:
            log_probs = np.sum(log_probs_per_dim, axis=1)
        else:
            log_probs = log_probs_per_dim
        
        return log_probs
        
    def update_sampling_ratio(self, episode: int, total_episodes: Optional[int] = None):
        """
        更新采样比例 ρ(t) = min(1, ρ₀ + α·t)
        
        Args:
            episode: 当前episode
            total_episodes: 总episode数
        """
        if total_episodes is not None:
            self.total_episodes = total_episodes
            
        self.episode_count = episode
        
        if self.config.adaptive_rho:
            # 自适应调整
            self._adaptive_rho_update()
        else:
            # 线性增长
            progress = episode / self.total_episodes
            self.current_rho = min(
                self.config.max_rho,
                self.config.initial_rho + self.config.rho_increment * progress
            )
            
        logger.debug(f"Episode {episode}: ρ = {self.current_rho:.3f}")
        
    def _adaptive_rho_update(self):
        """自适应采样比例更新"""
        if len(self.performance_history) < 2:
            # 性能历史不足，使用线性增长
            progress = self.episode_count / self.total_episodes
            self.current_rho = min(
                self.config.max_rho,
                self.config.initial_rho + self.config.rho_increment * progress
            )
            return
            
        # 计算性能变化
        recent_performance = np.mean(self.performance_history[-5:])  # 最近5次的平均性能
        performance_change = recent_performance - self.last_performance
        
        if performance_change > self.config.performance_threshold:
            # 性能提升，可以增加在线比例
            self.current_rho = min(
                self.config.max_rho,
                self.current_rho + self.config.rho_increment * 2
            )
        elif performance_change < -self.config.performance_threshold:
            # 性能下降，减少在线比例
            self.current_rho = max(
                self.config.initial_rho,
                self.current_rho - self.config.rho_increment
            )
        else:
            # 性能稳定，正常增长
            progress = self.episode_count / self.total_episodes
            target_rho = min(
                self.config.max_rho,
                self.config.initial_rho + self.config.rho_increment * progress
            )
            self.current_rho = min(target_rho, self.current_rho + self.config.rho_increment)
            
        self.last_performance = recent_performance
        
    def update_performance(self, performance: float):
        """
        更新性能指标，用于自适应采样比例调整
        
        Args:
            performance: 当前性能指标（如奖励、夏普率等）
        """
        self.performance_history.append(performance)
        
        # 保持历史长度在合理范围内
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
            
    def set_current_policy(self, policy):
        """
        设置当前策略，用于重要性权重计算
        
        Args:
            policy: 当前策略对象
        """
        self.current_policy = policy
        logger.info("更新当前策略用于重要性权重计算")
        
    def _empty_batch(self) -> Dict[str, Any]:
        """返回空批次"""
        return {
            'states': np.array([]),
            'actions': np.array([]),
            'rewards': np.array([]),
            'next_states': np.array([]),
            'dones': np.array([]),
            'data_source': 'empty',
            'indices': np.array([]),
            'weights': np.array([])
        }
        
    def get_sampling_statistics(self) -> Dict[str, Any]:
        """获取采样统计信息"""
        return {
            'current_rho': self.current_rho,
            'episode_count': self.episode_count,
            'total_episodes': self.total_episodes,
            'offline_dataset_size': len(self.offline_dataset),
            'online_buffer_size': len(self.online_buffer),
            'performance_history_length': len(self.performance_history),
            'last_performance': self.last_performance,
            'config': {
                'initial_rho': self.config.initial_rho,
                'rho_increment': self.config.rho_increment,
                'max_rho': self.config.max_rho,
                'min_offline_ratio': self.config.min_offline_ratio,
                'adaptive_rho': self.config.adaptive_rho
            }
        }
        
    def reset_sampling_ratio(self):
        """重置采样比例到初始值"""
        self.current_rho = self.config.initial_rho
        self.episode_count = 0
        self.performance_history.clear()
        self.last_performance = 0.0
        logger.info(f"采样比例已重置到初始值: {self.current_rho}")
        
    def get_current_rho(self) -> float:
        """获取当前采样比例"""
        return self.current_rho
        
    def set_rho(self, rho: float):
        """
        手动设置采样比例
        
        Args:
            rho: 新的采样比例
        """
        self.current_rho = np.clip(rho, 0.0, self.config.max_rho)
        logger.info(f"手动设置采样比例: {self.current_rho}")


# 工具函数
def create_mixture_sampler(offline_dataset: OfflineDataset,
                          online_buffer: OnlineReplayBuffer,
                          config: Optional[MixtureSamplerConfig] = None) -> MixtureSampler:
    """
    创建混合采样器的便捷函数
    
    Args:
        offline_dataset: 离线数据集
        online_buffer: 在线回放缓冲区
        config: 采样器配置
        
    Returns:
        混合采样器实例
    """
    if config is None:
        config = MixtureSamplerConfig()
        
    return MixtureSampler(offline_dataset, online_buffer, config)


def calculate_sampling_schedule(total_episodes: int,
                              initial_rho: float = 0.2,
                              final_rho: float = 1.0,
                              schedule_type: str = 'linear') -> List[float]:
    """
    计算采样比例调度
    
    Args:
        total_episodes: 总episode数
        initial_rho: 初始采样比例
        final_rho: 最终采样比例
        schedule_type: 调度类型 ('linear', 'exponential', 'cosine')
        
    Returns:
        采样比例序列
    """
    episodes = np.arange(total_episodes)
    
    if schedule_type == 'linear':
        rho_schedule = initial_rho + (final_rho - initial_rho) * episodes / (total_episodes - 1)
    elif schedule_type == 'exponential':
        decay_rate = np.log(final_rho / initial_rho) / (total_episodes - 1)
        rho_schedule = initial_rho * np.exp(decay_rate * episodes)
    elif schedule_type == 'cosine':
        rho_schedule = initial_rho + (final_rho - initial_rho) * (1 - np.cos(np.pi * episodes / (total_episodes - 1))) / 2
    else:
        raise ValueError(f"不支持的调度类型: {schedule_type}")
        
    return np.clip(rho_schedule, 0.0, 1.0).tolist()