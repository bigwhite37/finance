"""
在线经验回放缓冲区

实现高效的轨迹存储和检索，支持优先级采样机制、时间加权衰减功能和FIFO淘汰策略。
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryData:
    """轨迹数据结构"""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    log_probs: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    advantages: Optional[np.ndarray] = None
    timestamp: float = None
    market_regime: str = "normal"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class OnlineReplayConfig:
    """在线回放缓冲区配置"""
    capacity: int = 10000
    priority_alpha: float = 0.6  # 优先级采样参数
    priority_beta: float = 0.4   # 重要性权重参数
    time_decay_factor: float = 0.99  # 时间衰减因子
    min_priority: float = 1e-6   # 最小优先级
    batch_size: int = 256
    enable_time_decay: bool = True
    enable_priority_sampling: bool = True


class OnlineReplayBuffer:
    """在线经验回放缓冲区
    
    支持优先级采样机制，基于TD误差动态调整样本权重，
    添加时间加权衰减功能，实现FIFO淘汰策略。
    """
    
    def __init__(self, config: OnlineReplayConfig):
        """
        初始化在线回放缓冲区
        
        Args:
            config: 缓冲区配置
        """
        self.config = config
        self.capacity = config.capacity
        
        # 数据存储
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.timestamps = deque(maxlen=self.capacity)
        
        # 优先级采样相关
        self.priority_alpha = config.priority_alpha
        self.priority_beta = config.priority_beta
        self.min_priority = config.min_priority
        
        # 时间衰减相关
        self.time_decay_factor = config.time_decay_factor
        self.enable_time_decay = config.enable_time_decay
        self.enable_priority_sampling = config.enable_priority_sampling
        
        # 统计信息
        self.total_added = 0
        self.total_sampled = 0
        
        # 线程安全
        self._lock = Lock()
        
        logger.info(f"初始化在线回放缓冲区，容量: {self.capacity}")
        
    def add_trajectory(self, 
                      trajectory: TrajectoryData, 
                      priority: Optional[float] = None):
        """
        添加交易轨迹
        
        Args:
            trajectory: 轨迹数据
            priority: 初始优先级，如果为None则使用默认值
        """
        with self._lock:
            # 设置默认优先级
            if priority is None:
                priority = 1.0  # 新样本默认高优先级
                
            # 确保优先级不小于最小值
            priority = max(priority, self.min_priority)
            
            # 添加到缓冲区
            self.buffer.append(trajectory)
            self.priorities.append(priority)
            self.timestamps.append(trajectory.timestamp)
            
            self.total_added += 1
            
            if len(self.buffer) % 1000 == 0:
                logger.debug(f"缓冲区大小: {len(self.buffer)}/{self.capacity}")
                
    def add_batch_trajectories(self, trajectories: List[TrajectoryData]):
        """
        批量添加轨迹
        
        Args:
            trajectories: 轨迹数据列表
        """
        for trajectory in trajectories:
            self.add_trajectory(trajectory)
            
    def sample_batch(self, 
                    batch_size: Optional[int] = None, 
                    beta: Optional[float] = None) -> Dict[str, Any]:
        """
        优先级采样批次数据
        
        Args:
            batch_size: 批次大小
            beta: 重要性权重参数
            
        Returns:
            采样的批次数据
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if beta is None:
            beta = self.priority_beta
            
        if len(self.buffer) < batch_size:
            logger.warning(f"缓冲区样本不足: {len(self.buffer)} < {batch_size}")
            batch_size = len(self.buffer)
            
        if batch_size == 0:
            return self._empty_batch()
            
        with self._lock:
            if self.enable_priority_sampling:
                indices, weights = self._priority_sample(batch_size, beta)
            else:
                indices = np.random.choice(len(self.buffer), batch_size, replace=False)
                weights = np.ones(batch_size)
                
            # 提取样本
            batch_data = self._extract_batch(indices)
            batch_data['indices'] = indices
            batch_data['weights'] = weights
            
            self.total_sampled += batch_size
            
            return batch_data
            
    def _priority_sample(self, batch_size: int, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """优先级采样"""
        # 应用时间衰减
        if self.enable_time_decay:
            self._apply_time_decay()
            
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.priority_alpha
        probs = probs / probs.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # 计算重要性权重
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # 归一化
        
        return indices, weights
        
    def _extract_batch(self, indices: np.ndarray) -> Dict[str, Any]:
        """提取批次数据"""
        batch_trajectories = [self.buffer[i] for i in indices]
        
        # 合并所有轨迹数据
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        log_probs_list = []
        values_list = []
        advantages_list = []
        timestamps_list = []
        market_regimes_list = []
        
        for traj in batch_trajectories:
            # 处理不同长度的轨迹
            traj_len = len(traj.states)
            
            states_list.extend(traj.states)
            actions_list.extend(traj.actions)
            rewards_list.extend(traj.rewards)
            next_states_list.extend(traj.next_states)
            dones_list.extend(traj.dones)
            
            if traj.log_probs is not None:
                log_probs_list.extend(traj.log_probs)
            if traj.values is not None:
                values_list.extend(traj.values)
            if traj.advantages is not None:
                advantages_list.extend(traj.advantages)
                
            timestamps_list.extend([traj.timestamp] * traj_len)
            market_regimes_list.extend([traj.market_regime] * traj_len)
        
        # 转换为numpy数组
        batch_data = {
            'states': np.array(states_list),
            'actions': np.array(actions_list),
            'rewards': np.array(rewards_list),
            'next_states': np.array(next_states_list),
            'dones': np.array(dones_list),
            'timestamps': np.array(timestamps_list),
            'market_regimes': market_regimes_list
        }
        
        # 可选数据
        if log_probs_list:
            batch_data['log_probs'] = np.array(log_probs_list)
        if values_list:
            batch_data['values'] = np.array(values_list)
        if advantages_list:
            batch_data['advantages'] = np.array(advantages_list)
            
        return batch_data
        
    def _empty_batch(self) -> Dict[str, Any]:
        """返回空批次"""
        return {
            'states': np.array([]),
            'actions': np.array([]),
            'rewards': np.array([]),
            'next_states': np.array([]),
            'dones': np.array([]),
            'timestamps': np.array([]),
            'market_regimes': [],
            'indices': np.array([]),
            'weights': np.array([])
        }
        
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        更新样本优先级
        
        Args:
            indices: 样本索引列表
            priorities: 新的优先级列表
        """
        with self._lock:
            for idx, priority in zip(indices, priorities):
                if 0 <= idx < len(self.priorities):
                    # 确保优先级在合理范围内
                    priority = max(priority, self.min_priority)
                    self.priorities[idx] = priority
                    
    def _apply_time_decay(self):
        """应用时间衰减权重"""
        if not self.enable_time_decay or len(self.timestamps) == 0:
            return
            
        current_time = time.time()
        
        # 计算时间衰减权重
        for i in range(len(self.priorities)):
            time_diff = current_time - self.timestamps[i]
            # 时间衰减：越老的数据优先级越低
            decay_weight = self.time_decay_factor ** (time_diff / 3600)  # 按小时衰减
            self.priorities[i] *= decay_weight
            
            # 确保不低于最小优先级
            self.priorities[i] = max(self.priorities[i], self.min_priority)
            
    def get_recent_trajectory(self, window: int = 60) -> List[TrajectoryData]:
        """
        获取最近的交易轨迹
        
        Args:
            window: 时间窗口（天数）
            
        Returns:
            最近的轨迹数据列表
        """
        if len(self.buffer) == 0:
            return []
            
        current_time = time.time()
        window_seconds = window * 24 * 3600  # 转换为秒
        
        recent_trajectories = []
        
        with self._lock:
            for i, trajectory in enumerate(self.buffer):
                if current_time - trajectory.timestamp <= window_seconds:
                    recent_trajectories.append(trajectory)
                    
        logger.info(f"获取最近{window}天的轨迹: {len(recent_trajectories)}条")
        return recent_trajectories
        
    def get_recent_data(self, window: int = 60) -> Dict[str, Any]:
        """
        获取最近的数据（批次格式）
        
        Args:
            window: 时间窗口（天数）
            
        Returns:
            最近的数据批次
        """
        recent_trajectories = self.get_recent_trajectory(window)
        
        if not recent_trajectories:
            return self._empty_batch()
            
        # 创建临时索引
        indices = list(range(len(recent_trajectories)))
        
        # 临时存储原始缓冲区
        original_buffer = list(self.buffer)
        
        # 替换缓冲区内容进行提取
        self.buffer.clear()
        self.buffer.extend(recent_trajectories)
        
        try:
            batch_data = self._extract_batch(np.array(indices))
            batch_data['indices'] = np.array(indices)
            batch_data['weights'] = np.ones(len(indices))
        finally:
            # 恢复原始缓冲区
            self.buffer.clear()
            self.buffer.extend(original_buffer)
            
        return batch_data
        
    def clear(self):
        """清空缓冲区"""
        with self._lock:
            self.buffer.clear()
            self.priorities.clear()
            self.timestamps.clear()
            logger.info("缓冲区已清空")
            
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        with self._lock:
            if len(self.buffer) == 0:
                return {
                    'size': 0,
                    'capacity': self.capacity,
                    'total_added': self.total_added,
                    'total_sampled': self.total_sampled,
                    'utilization': 0.0
                }
                
            priorities_array = np.array(self.priorities)
            timestamps_array = np.array(self.timestamps)
            current_time = time.time()
            
            return {
                'size': len(self.buffer),
                'capacity': self.capacity,
                'utilization': len(self.buffer) / self.capacity,
                'total_added': self.total_added,
                'total_sampled': self.total_sampled,
                'priority_stats': {
                    'mean': float(np.mean(priorities_array)),
                    'std': float(np.std(priorities_array)),
                    'min': float(np.min(priorities_array)),
                    'max': float(np.max(priorities_array))
                },
                'time_stats': {
                    'oldest': current_time - np.max(timestamps_array),
                    'newest': current_time - np.min(timestamps_array),
                    'avg_age': current_time - np.mean(timestamps_array)
                }
            }
            
    def __len__(self) -> int:
        """返回缓冲区大小"""
        return len(self.buffer)
        
    def __bool__(self) -> bool:
        """检查缓冲区是否为空"""
        return len(self.buffer) > 0


class PrioritizedReplayBuffer(OnlineReplayBuffer):
    """优先级回放缓冲区（OnlineReplayBuffer的别名）"""
    pass


# 工具函数
def create_trajectory_from_episode(states: np.ndarray,
                                 actions: np.ndarray,
                                 rewards: np.ndarray,
                                 next_states: np.ndarray,
                                 dones: np.ndarray,
                                 log_probs: Optional[np.ndarray] = None,
                                 values: Optional[np.ndarray] = None,
                                 market_regime: str = "normal") -> TrajectoryData:
    """
    从episode数据创建轨迹
    
    Args:
        states: 状态序列
        actions: 动作序列
        rewards: 奖励序列
        next_states: 下一状态序列
        dones: 结束标志序列
        log_probs: 动作对数概率
        values: 价值估计
        market_regime: 市场制度
        
    Returns:
        轨迹数据
    """
    return TrajectoryData(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        log_probs=log_probs,
        values=values,
        market_regime=market_regime
    )


def calculate_td_errors(values: np.ndarray, 
                       rewards: np.ndarray, 
                       next_values: np.ndarray,
                       dones: np.ndarray,
                       gamma: float = 0.99) -> np.ndarray:
    """
    计算TD误差，用于优先级更新
    
    Args:
        values: 当前状态价值
        rewards: 奖励
        next_values: 下一状态价值
        dones: 结束标志
        gamma: 折扣因子
        
    Returns:
        TD误差数组
    """
    targets = rewards + gamma * next_values * (1 - dones)
    td_errors = np.abs(targets - values)
    return td_errors