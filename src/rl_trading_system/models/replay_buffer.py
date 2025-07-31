"""
经验回放缓冲区实现
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union
import torch
import numpy as np
import random
from collections import namedtuple, deque
import threading
import multiprocessing as mp
from abc import ABC, abstractmethod


@dataclass
class Experience:
    """经验数据结构"""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    info: Optional[Dict[str, Any]] = None


@dataclass
class ReplayBufferConfig:
    """回放缓冲区配置"""
    capacity: int = 100000
    batch_size: int = 256
    state_dim: int = 256
    action_dim: int = 100
    device: str = 'cpu'
    
    # 优先级回放参数
    alpha: float = 0.6  # 优先级指数
    beta: float = 0.4   # 重要性采样指数
    beta_increment: float = 0.001  # beta增长率
    epsilon: float = 1e-6  # 防止零优先级的小值
    
    # 多进程参数
    n_workers: int = 1
    shared_memory: bool = False


class BaseReplayBuffer(ABC):
    """回放缓冲区基类"""
    
    def __init__(self, config: ReplayBufferConfig):
        self.config = config
        self.capacity = config.capacity
        self.batch_size = config.batch_size
        self.device = torch.device(config.device)
        
    @abstractmethod
    def add(self, experience: Experience, **kwargs) -> None:
        """添加经验"""
        pass
    
    @abstractmethod
    def sample(self) -> Union[List[Experience], Tuple[List[Experience], torch.Tensor, torch.Tensor]]:
        """采样批次"""
        pass
    
    @abstractmethod
    def can_sample(self) -> bool:
        """是否可以采样"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓冲区"""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """当前大小"""
        pass


class ReplayBuffer(BaseReplayBuffer):
    """
    标准经验回放缓冲区
    
    使用循环缓冲区存储经验，支持随机采样
    """
    
    def __init__(self, config: ReplayBufferConfig):
        super().__init__(config)
        
        # 初始化缓冲区
        self.buffer: List[Optional[Experience]] = [None] * self.capacity
        self.position = 0
        self._size = 0
        
        # 线程锁（用于多线程安全）
        self._lock = threading.RLock()
        
    def add(self, experience: Experience, **kwargs) -> None:
        """
        添加经验到缓冲区
        
        Args:
            experience: 经验数据
        """
        with self._lock:
            self.buffer[self.position] = experience
            self.position = (self.position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
    
    def sample(self) -> List[Experience]:
        """
        随机采样批次
        
        Returns:
            batch: 经验批次
        """
        if not self.can_sample():
            raise ValueError(f"缓冲区中的经验不足，需要至少{self.batch_size}个，当前有{self.size}个")
        
        with self._lock:
            # 随机采样索引
            indices = random.sample(range(self.size), self.batch_size)
            batch = [self.buffer[i] for i in indices]
            
        return batch
    
    def can_sample(self) -> bool:
        """检查是否可以采样"""
        return self.size >= self.batch_size
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self.buffer = [None] * self.capacity
            self.position = 0
            self._size = 0
    
    @property
    def size(self) -> int:
        """当前缓冲区大小"""
        return self._size
    
    def get_all_experiences(self) -> List[Experience]:
        """
        获取所有经验（用于调试和分析）
        
        Returns:
            experiences: 所有经验列表
        """
        with self._lock:
            if self._size < self.capacity:
                return [exp for exp in self.buffer[:self._size] if exp is not None]
            else:
                # 缓冲区已满，需要按正确顺序返回
                return [exp for exp in (self.buffer[self.position:] + self.buffer[:self.position]) if exp is not None]
    
    def state_dict(self) -> Dict[str, Any]:
        """
        获取缓冲区状态（用于保存）
        
        Returns:
            state: 状态字典
        """
        with self._lock:
            return {
                'buffer': self.buffer.copy(),
                'position': self.position,
                'size': self._size,
                'capacity': self.capacity,
                'batch_size': self.batch_size
            }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        加载缓冲区状态
        
        Args:
            state_dict: 状态字典
        """
        with self._lock:
            self.buffer = state_dict['buffer']
            self.position = state_dict['position']
            self._size = state_dict['size']
            self.capacity = state_dict['capacity']
            self.batch_size = state_dict['batch_size']


class PrioritizedReplayBuffer(BaseReplayBuffer):
    """
    优先级经验回放缓冲区
    
    基于TD误差的优先级采样，使用SumTree数据结构
    """
    
    def __init__(self, config: ReplayBufferConfig):
        super().__init__(config)
        
        self.alpha = config.alpha
        self.beta = config.beta
        self.beta_increment = config.beta_increment
        self.epsilon = config.epsilon
        
        # 初始化缓冲区和优先级
        self.buffer: List[Optional[Experience]] = [None] * self.capacity
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self._size = 0
        self.max_priority = 1.0
        
        # 构建SumTree
        self._build_sum_tree()
        
        # 线程锁
        self._lock = threading.RLock()
    
    def _build_sum_tree(self):
        """构建SumTree数据结构"""
        # SumTree的大小是2*capacity-1
        tree_size = 2 * self.capacity - 1
        self.tree = np.zeros(tree_size, dtype=np.float32)
        
    def _update_tree(self, idx: int, priority: float):
        """更新SumTree中的优先级"""
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        
        # 向上更新父节点
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change
    
    def _get_leaf(self, value: float) -> int:
        """根据值获取叶子节点索引"""
        parent_idx = 0
        
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return data_idx
    
    def add(self, experience: Experience, priority: Optional[float] = None, **kwargs) -> None:
        """
        添加经验到缓冲区
        
        Args:
            experience: 经验数据
            priority: 优先级（如果为None，使用最大优先级）
        """
        if priority is None:
            priority = self.max_priority
        
        with self._lock:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.alpha
            self._update_tree(self.position, self.priorities[self.position])
            
            self.position = (self.position + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
    
    def sample(self) -> Tuple[List[Experience], np.ndarray, torch.Tensor]:
        """
        基于优先级采样批次
        
        Returns:
            batch: 经验批次
            indices: 采样索引
            weights: 重要性采样权重
        """
        if not self.can_sample():
            raise ValueError(f"缓冲区中的经验不足，需要至少{self.batch_size}个，当前有{self.size}个")
        
        with self._lock:
            indices = []
            priorities = []
            
            # 计算优先级区间
            priority_segment = self.tree[0] / self.batch_size
            
            for i in range(self.batch_size):
                a = priority_segment * i
                b = priority_segment * (i + 1)
                
                value = np.random.uniform(a, b)
                idx = self._get_leaf(value)
                
                # 确保索引有效
                idx = max(0, min(idx, self.size - 1))
                indices.append(idx)
                priorities.append(self.priorities[idx])
            
            # 计算重要性采样权重
            sampling_probabilities = np.array(priorities) / self.tree[0]
            weights = (self.size * sampling_probabilities) ** (-self.beta)
            weights = weights / weights.max()  # 归一化
            
            # 获取经验批次
            batch = [self.buffer[idx] for idx in indices]
            
            # 更新beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
        return batch, np.array(indices), torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def update_priorities(self, indices: np.ndarray, priorities: torch.Tensor) -> None:
        """
        更新经验的优先级
        
        Args:
            indices: 经验索引
            priorities: 新的优先级
        """
        with self._lock:
            for idx, priority in zip(indices, priorities):
                priority = float(priority)
                priority = max(priority, self.epsilon)  # 防止零优先级
                
                # 存储原始优先级（不加alpha指数）用于比较
                raw_priority = priority
                self.priorities[idx] = priority ** self.alpha
                self._update_tree(idx, self.priorities[idx])
                
                # 更新最大优先级
                self.max_priority = max(self.max_priority, raw_priority)
    
    def can_sample(self) -> bool:
        """检查是否可以采样"""
        return self.size >= self.batch_size
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self.buffer = [None] * self.capacity
            self.priorities = np.zeros(self.capacity, dtype=np.float32)
            self.position = 0
            self._size = 0
            self.max_priority = 1.0
            self._build_sum_tree()
    
    @property
    def size(self) -> int:
        """当前缓冲区大小"""
        return self._size


class MultiProcessReplayBuffer(BaseReplayBuffer):
    """
    多进程经验回放缓冲区
    
    支持多个进程并发添加和采样经验
    """
    
    def __init__(self, config: ReplayBufferConfig):
        super().__init__(config)
        
        self.n_workers = config.n_workers
        self.shared_memory = config.shared_memory
        
        if self.shared_memory:
            # 使用共享内存
            self._init_shared_memory()
        else:
            # 使用进程间通信
            self._init_ipc()
    
    def _init_shared_memory(self):
        """初始化共享内存"""
        # 这里需要实现共享内存版本
        # 由于PyTorch张量的共享内存比较复杂，这里提供框架
        raise NotImplementedError("共享内存版本待实现")
    
    def _init_ipc(self):
        """初始化进程间通信"""
        # 使用队列进行进程间通信
        self.experience_queue = mp.Queue(maxsize=self.capacity)
        self.sample_queue = mp.Queue()
        
        # 启动后台进程管理缓冲区
        self.manager_process = mp.Process(target=self._buffer_manager)
        self.manager_process.start()
    
    def _buffer_manager(self):
        """缓冲区管理进程"""
        buffer = ReplayBuffer(self.config)
        
        while True:
            try:
                # 处理添加请求
                if not self.experience_queue.empty():
                    experience = self.experience_queue.get_nowait()
                    if experience is None:  # 终止信号
                        break
                    buffer.add(experience)
                
                # 处理采样请求
                if not self.sample_queue.empty():
                    request = self.sample_queue.get_nowait()
                    if request == 'sample' and buffer.can_sample():
                        batch = buffer.sample()
                        # 这里需要将批次发送回请求进程
                        # 实际实现需要更复杂的通信机制
                        pass
                
            except Exception as e:
                print(f"缓冲区管理进程错误: {e}")
                break
    
    def add(self, experience: Experience, **kwargs) -> None:
        """添加经验（多进程版本）"""
        try:
            self.experience_queue.put_nowait(experience)
        except:
            # 队列满时的处理
            pass
    
    def sample(self) -> List[Experience]:
        """采样批次（多进程版本）"""
        # 发送采样请求
        self.sample_queue.put('sample')
        
        # 等待结果（这里需要实现结果接收机制）
        # 实际实现需要更复杂的通信协议
        raise NotImplementedError("多进程采样待完善")
    
    def can_sample(self) -> bool:
        """检查是否可以采样（多进程版本）"""
        # 需要查询管理进程的状态
        return True  # 简化实现
    
    def clear(self) -> None:
        """清空缓冲区（多进程版本）"""
        # 发送清空信号
        pass
    
    @property
    def size(self) -> int:
        """当前缓冲区大小（多进程版本）"""
        # 需要查询管理进程的状态
        return 0  # 简化实现
    
    def close(self):
        """关闭多进程缓冲区"""
        if hasattr(self, 'manager_process'):
            # 发送终止信号
            self.experience_queue.put(None)
            self.manager_process.join(timeout=5)
            if self.manager_process.is_alive():
                self.manager_process.terminate()


def create_replay_buffer(config: ReplayBufferConfig) -> BaseReplayBuffer:
    """
    工厂函数：创建回放缓冲区
    
    Args:
        config: 缓冲区配置
        
    Returns:
        buffer: 回放缓冲区实例
    """
    if config.alpha > 0:
        # 使用优先级回放
        return PrioritizedReplayBuffer(config)
    elif config.n_workers > 1:
        # 使用多进程回放
        return MultiProcessReplayBuffer(config)
    else:
        # 使用标准回放
        return ReplayBuffer(config)