"""
共享经验池与V-Trace校正
解决离策略漂移的关键组件
参考claude.md中的SharedReplayBuffer实现
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import random
import time
from collections import deque, namedtuple
import threading
import pickle
import gzip


# 经验数据结构
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'expert_id', 'action_logits', 'timestamp', 'episode_id', 'step_id'
])


class VTraceCalculator:
    """
    V-Trace重要性采样校正计算器
    实现DeepMind的V-Trace算法用于离策略学习
    """
    
    def __init__(self, c_bar: float = 1.0, rho_bar: float = 1.0):
        """
        Args:
            c_bar: V-trace截断参数c̄
            rho_bar: V-trace截断参数ρ̄
        """
        self.c_bar = c_bar
        self.rho_bar = rho_bar
    
    def compute_importance_weights(self, 
                                 target_log_probs: torch.Tensor,
                                 behavior_log_probs: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算V-Trace重要性权重
        
        Args:
            target_log_probs: 目标策略的对数概率 [batch_size, seq_len]
            behavior_log_probs: 行为策略的对数概率 [batch_size, seq_len]
            mask: 有效步骤掩码 [batch_size, seq_len]
        
        Returns:
            c_weights: c权重 [batch_size, seq_len]
            rho_weights: ρ权重 [batch_size, seq_len]
        """
        # 计算重要性比率
        log_rhos = target_log_probs - behavior_log_probs
        rhos = torch.exp(log_rhos)
        
        # 应用截断
        c_weights = torch.clamp(rhos, max=self.c_bar)
        rho_weights = torch.clamp(rhos, max=self.rho_bar)
        
        # 应用掩码
        if mask is not None:
            c_weights = c_weights * mask
            rho_weights = rho_weights * mask
        
        return c_weights, rho_weights
    
    def compute_vtrace_targets(self,
                             rewards: torch.Tensor,
                             values: torch.Tensor,
                             next_values: torch.Tensor,
                             dones: torch.Tensor,
                             c_weights: torch.Tensor,
                             rho_weights: torch.Tensor,
                             gamma: float = 0.99) -> torch.Tensor:
        """
        计算V-Trace目标值
        
        Args:
            rewards: 奖励 [batch_size, seq_len]
            values: 当前状态价值 [batch_size, seq_len]
            next_values: 下一状态价值 [batch_size, seq_len]
            dones: 结束标志 [batch_size, seq_len]
            c_weights: c权重 [batch_size, seq_len]
            rho_weights: ρ权重 [batch_size, seq_len]
            gamma: 折扣因子
        
        Returns:
            vtrace_targets: V-Trace目标值 [batch_size, seq_len]
        """
        batch_size, seq_len = rewards.shape
        
        # 计算时序差分误差
        td_errors = rewards + gamma * next_values * (1 - dones) - values
        
        # V-Trace校正的时序差分误差
        corrected_td_errors = rho_weights * td_errors
        
        # 反向累积计算V-Trace目标
        vtrace_targets = torch.zeros_like(values)
        
        # 从后往前计算
        vs_minus_v_xs = torch.zeros(batch_size, device=rewards.device)
        
        for t in reversed(range(seq_len)):
            vs_minus_v_xs = corrected_td_errors[:, t] + gamma * c_weights[:, t] * vs_minus_v_xs
            vtrace_targets[:, t] = values[:, t] + vs_minus_v_xs
        
        return vtrace_targets


class SharedReplayBuffer:
    """
    共享经验池
    支持多专家经验存储和V-Trace重要性采样
    """
    
    def __init__(self, 
                 capacity: int = int(2e6),
                 n_experts: int = 5,
                 sequence_length: int = 32,
                 compress_data: bool = True,
                 device: str = "cpu"):
        
        self.capacity = capacity
        self.n_experts = n_experts
        self.sequence_length = sequence_length
        self.compress_data = compress_data
        self.device = device
        
        # 主缓冲区
        self.buffer = []
        self.position = 0
        self.size = 0
        
        # 专家特定的索引（加速采样）
        self.expert_indices = {i: [] for i in range(n_experts)}
        
        # V-Trace计算器
        self.vtrace_calculator = VTraceCalculator()
        
        # 统计信息
        self.total_adds = 0
        self.total_samples = 0
        self.expert_add_counts = np.zeros(n_experts)
        self.expert_sample_counts = np.zeros(n_experts)
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 性能监控
        self.add_times = deque(maxlen=1000)
        self.sample_times = deque(maxlen=1000)
        
    def add(self, 
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool,
            expert_id: int,
            action_logits: np.ndarray,
            episode_id: Optional[int] = None,
            step_id: Optional[int] = None) -> None:
        """
        添加经验到缓冲区
        """
        start_time = time.time()
        
        with self.lock:
            # 创建经验对象
            experience = Experience(
                state=self._maybe_compress(state),
                action=self._maybe_compress(action),
                reward=reward,
                next_state=self._maybe_compress(next_state),
                done=done,
                expert_id=expert_id,
                action_logits=self._maybe_compress(action_logits),
                timestamp=time.time(),
                episode_id=episode_id or 0,
                step_id=step_id or 0
            )
            
            # 添加到主缓冲区
            if self.size < self.capacity:
                self.buffer.append(experience)
                self.size += 1
            else:
                # 替换最旧的经验
                old_experience = self.buffer[self.position]
                self.buffer[self.position] = experience
                
                # 更新专家索引（移除旧的）
                old_expert_id = old_experience.expert_id
                if old_expert_id in self.expert_indices:
                    try:
                        self.expert_indices[old_expert_id].remove(self.position)
                    except ValueError:
                        # 索引已经不在列表中，这是正常情况
                        logger.debug(f"专家 {old_expert_id} 的索引 {self.position} 已不在专家索引列表中")
            
            # 更新专家索引（添加新的）
            if expert_id in self.expert_indices:
                self.expert_indices[expert_id].append(self.position)
            
            # 更新位置指针
            self.position = (self.position + 1) % self.capacity
            
            # 更新统计
            self.total_adds += 1
            self.expert_add_counts[expert_id] += 1
        
        # 记录性能
        self.add_times.append(time.time() - start_time)
    
    def sample(self, 
               batch_size: int,
               current_expert_id: Optional[int] = None,
               expert_sampling_ratio: float = 0.5) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        采样经验批次，返回数据和V-Trace权重
        
        Args:
            batch_size: 批次大小
            current_expert_id: 当前专家ID（用于V-Trace计算）
            expert_sampling_ratio: 专家特定采样比例
        
        Returns:
            batch_data: 批次数据字典
            importance_weights: V-Trace重要性权重
        """
        start_time = time.time()
        
        with self.lock:
            if self.size < batch_size:
                raise ValueError(f"缓冲区大小 {self.size} 小于批次大小 {batch_size}")
            
            # 采样策略：混合随机采样和专家特定采样
            batch_indices = []
            
            if current_expert_id is not None and expert_sampling_ratio > 0:
                # 部分从当前专家采样
                expert_batch_size = int(batch_size * expert_sampling_ratio)
                expert_indices = self.expert_indices.get(current_expert_id, [])
                
                if len(expert_indices) >= expert_batch_size:
                    expert_sample_indices = random.sample(expert_indices, expert_batch_size)
                    batch_indices.extend(expert_sample_indices)
                
                # 剩余从全局采样
                remaining_size = batch_size - len(batch_indices)
            else:
                remaining_size = batch_size
            
            # 全局随机采样
            if remaining_size > 0:
                available_indices = list(range(self.size))
                # 排除已选择的索引
                available_indices = [i for i in available_indices if i not in batch_indices]
                
                if len(available_indices) >= remaining_size:
                    global_sample_indices = random.sample(available_indices, remaining_size)
                    batch_indices.extend(global_sample_indices)
                else:
                    # 如果不够，从全部索引中采样
                    batch_indices = random.sample(range(self.size), batch_size)
            
            # 获取经验数据
            batch_experiences = [self.buffer[i] for i in batch_indices]
            
            # 更新统计
            self.total_samples += 1
            for exp in batch_experiences:
                self.expert_sample_counts[exp.expert_id] += 1
        
        # 转换为张量
        batch_data = self._experiences_to_tensors(batch_experiences)
        
        # 计算V-Trace权重
        importance_weights = self._compute_importance_weights(
            batch_experiences, current_expert_id
        )
        
        # 记录性能
        self.sample_times.append(time.time() - start_time)
        
        return batch_data, importance_weights
    
    def sample_sequences(self,
                        batch_size: int,
                        sequence_length: Optional[int] = None,
                        current_expert_id: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        采样序列批次（用于RNN训练）
        """
        seq_len = sequence_length or self.sequence_length
        
        with self.lock:
            if self.size < batch_size * seq_len:
                raise ValueError("缓冲区数据不足以采样序列")
            
            # 找到有效的序列起始点
            valid_starts = []
            for i in range(self.size - seq_len + 1):
                # 检查序列的连续性（相同episode）
                start_exp = self.buffer[i]
                end_exp = self.buffer[i + seq_len - 1]
                
                if (start_exp.episode_id == end_exp.episode_id and
                    end_exp.step_id - start_exp.step_id == seq_len - 1):
                    valid_starts.append(i)
            
            if len(valid_starts) < batch_size:
                # 如果有效序列不够，放松约束
                valid_starts = list(range(self.size - seq_len + 1))
            
            # 采样序列起始点
            start_indices = random.sample(valid_starts, min(batch_size, len(valid_starts)))
            
            # 收集序列数据
            sequences = []
            for start_idx in start_indices:
                sequence = [self.buffer[start_idx + j] for j in range(seq_len)]
                sequences.append(sequence)
        
        # 转换为序列张量
        batch_data = self._sequences_to_tensors(sequences)
        
        # 计算序列权重
        importance_weights = torch.ones(batch_size, seq_len, device=self.device)
        
        return batch_data, importance_weights
    
    def _experiences_to_tensors(self, experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """将经验列表转换为张量字典"""
        batch_size = len(experiences)
        
        # 获取第一个经验来确定形状
        first_exp = experiences[0] 
        state_shape = self._maybe_decompress(first_exp.state).shape
        action_shape = self._maybe_decompress(first_exp.action).shape
        logits_shape = self._maybe_decompress(first_exp.action_logits).shape
        
        # 预分配张量
        states = torch.zeros((batch_size,) + state_shape, dtype=torch.float32, device=self.device)
        next_states = torch.zeros((batch_size,) + state_shape, dtype=torch.float32, device=self.device)
        actions = torch.zeros((batch_size,) + action_shape, dtype=torch.float32, device=self.device)
        rewards = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        dones = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        expert_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        action_logits = torch.zeros((batch_size,) + logits_shape, dtype=torch.float32, device=self.device)
        
        # 填充数据
        for i, exp in enumerate(experiences):
            states[i] = torch.from_numpy(self._maybe_decompress(exp.state))
            next_states[i] = torch.from_numpy(self._maybe_decompress(exp.next_state))
            actions[i] = torch.from_numpy(self._maybe_decompress(exp.action))
            rewards[i] = exp.reward
            dones[i] = exp.done
            expert_ids[i] = exp.expert_id
            action_logits[i] = torch.from_numpy(self._maybe_decompress(exp.action_logits))
        
        return {
            'states': states,
            'next_states': next_states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'expert_ids': expert_ids,
            'action_logits': action_logits
        }
    
    def _sequences_to_tensors(self, sequences: List[List[Experience]]) -> Dict[str, torch.Tensor]:
        """将序列列表转换为张量字典"""
        batch_size = len(sequences)
        seq_len = len(sequences[0])
        
        # 获取形状信息
        first_exp = sequences[0][0]
        state_shape = self._maybe_decompress(first_exp.state).shape
        action_shape = self._maybe_decompress(first_exp.action).shape
        logits_shape = self._maybe_decompress(first_exp.action_logits).shape
        
        # 预分配张量
        states = torch.zeros((batch_size, seq_len) + state_shape, dtype=torch.float32, device=self.device)
        next_states = torch.zeros((batch_size, seq_len) + state_shape, dtype=torch.float32, device=self.device)
        actions = torch.zeros((batch_size, seq_len) + action_shape, dtype=torch.float32, device=self.device)
        rewards = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=self.device)
        dones = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
        expert_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        action_logits = torch.zeros((batch_size, seq_len) + logits_shape, dtype=torch.float32, device=self.device)
        
        # 填充数据
        for i, sequence in enumerate(sequences):
            for j, exp in enumerate(sequence):
                states[i, j] = torch.from_numpy(self._maybe_decompress(exp.state))
                next_states[i, j] = torch.from_numpy(self._maybe_decompress(exp.next_state))
                actions[i, j] = torch.from_numpy(self._maybe_decompress(exp.action))
                rewards[i, j] = exp.reward
                dones[i, j] = exp.done
                expert_ids[i, j] = exp.expert_id
                action_logits[i, j] = torch.from_numpy(self._maybe_decompress(exp.action_logits))
        
        return {
            'states': states,
            'next_states': next_states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'expert_ids': expert_ids,
            'action_logits': action_logits
        }
    
    def _compute_importance_weights(self, experiences: List[Experience], 
                                  current_expert_id: Optional[int]) -> torch.Tensor:
        """计算V-Trace重要性权重"""
        batch_size = len(experiences)
        weights = torch.ones(batch_size, dtype=torch.float32, device=self.device)
        
        if current_expert_id is None:
            return weights
        
        for i, exp in enumerate(experiences):
            if exp.expert_id == current_expert_id:
                weights[i] = 1.0
            else:
                # 计算重要性比率
                # 这里简化处理，实际应该基于策略的概率比率
                rho = self._compute_importance_ratio(exp, current_expert_id)
                weights[i] = min(1.0, rho)  # V-Trace截断
        
        return weights
    
    def _compute_importance_ratio(self, experience: Experience, current_expert_id: int) -> float:
        """
        计算重要性比率 π(a|s) / μ(a|s)
        这里需要访问当前策略来计算真实的比率
        """
        # 简化实现：基于专家ID的启发式
        if experience.expert_id == current_expert_id:
            return 1.0
        else:
            # 不同专家之间的重要性比率可以基于性能差异调整
            return 0.8  # 简化为常数
    
    def _maybe_compress(self, data: np.ndarray) -> bytes:
        """可选的数据压缩"""
        if self.compress_data:
            return gzip.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        else:
            return data
    
    def _maybe_decompress(self, data) -> np.ndarray:
        """可选的数据解压缩"""
        if self.compress_data and isinstance(data, bytes):
            return pickle.loads(gzip.decompress(data))
        else:
            return data
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        with self.lock:
            expert_ratios = self.expert_add_counts / max(self.total_adds, 1)
            sample_ratios = self.expert_sample_counts / max(self.total_samples, 1)
            
            return {
                'capacity': self.capacity,
                'size': self.size,
                'total_adds': self.total_adds,
                'total_samples': self.total_samples,
                'expert_add_counts': self.expert_add_counts.tolist(),
                'expert_sample_counts': self.expert_sample_counts.tolist(),
                'expert_add_ratios': expert_ratios.tolist(),
                'expert_sample_ratios': sample_ratios.tolist(),
                'avg_add_time': np.mean(self.add_times) if self.add_times else 0.0,
                'avg_sample_time': np.mean(self.sample_times) if self.sample_times else 0.0,
                'memory_usage_mb': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量(MB)"""
        if self.size == 0:
            return 0.0
        
        # 估算单个经验的大小
        sample_exp = self.buffer[0]
        exp_size = 0
        
        for field in sample_exp._fields:
            value = getattr(sample_exp, field)
            if isinstance(value, (bytes, np.ndarray)):
                exp_size += len(value) if isinstance(value, bytes) else value.nbytes
            else:
                exp_size += 8  # 估算其他类型
        
        total_size = exp_size * self.size
        return total_size / (1024 * 1024)
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
            self.position = 0
            self.size = 0
            self.expert_indices = {i: [] for i in range(self.n_experts)}
            
            # 重置统计
            self.total_adds = 0
            self.total_samples = 0
            self.expert_add_counts.fill(0)
            self.expert_sample_counts.fill(0)
    
    def save(self, filepath: str):
        """保存缓冲区到文件"""
        with self.lock:
            save_data = {
                'buffer': self.buffer,
                'position': self.position,
                'size': self.size,
                'expert_indices': self.expert_indices,
                'total_adds': self.total_adds,
                'total_samples': self.total_samples,
                'expert_add_counts': self.expert_add_counts,
                'expert_sample_counts': self.expert_sample_counts,
                'capacity': self.capacity,
                'n_experts': self.n_experts
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, filepath: str):
        """从文件加载缓冲区"""
        with self.lock:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            self.buffer = save_data['buffer']
            self.position = save_data['position']
            self.size = save_data['size']
            self.expert_indices = save_data['expert_indices']
            self.total_adds = save_data['total_adds']
            self.total_samples = save_data['total_samples']
            self.expert_add_counts = save_data['expert_add_counts']
            self.expert_sample_counts = save_data['expert_sample_counts']


if __name__ == "__main__":
    # 测试共享经验池
    print("测试共享经验池...")
    
    # 创建缓冲区
    buffer = SharedReplayBuffer(
        capacity=1000,
        n_experts=3,
        compress_data=False
    )
    
    print(f"创建缓冲区，容量: {buffer.capacity}")
    
    # 添加一些测试数据
    for i in range(100):
        state = np.random.randn(10)
        action = np.random.randn(3)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = np.random.rand() < 0.1
        expert_id = np.random.randint(0, 3)
        action_logits = np.random.randn(3)
        
        buffer.add(state, action, reward, next_state, done, 
                  expert_id, action_logits, episode_id=i//10, step_id=i%10)
    
    print(f"添加了100个经验，当前大小: {buffer.size}")
    
    # 测试采样
    try:
        batch_data, weights = buffer.sample(batch_size=32, current_expert_id=0)
        print(f"采样成功，批次形状:")
        for key, value in batch_data.items():
            print(f"  {key}: {value.shape}")
        print(f"重要性权重形状: {weights.shape}")
        
        # 测试序列采样
        seq_data, seq_weights = buffer.sample_sequences(batch_size=8, sequence_length=10)
        print(f"序列采样成功，序列形状:")
        for key, value in seq_data.items():
            print(f"  {key}: {value.shape}")
            
    except (ValueError, RuntimeError, IndexError) as e:
        print(f"采样出错: {e}")
    
    # 显示统计信息
    stats = buffer.get_stats()
    print("缓冲区统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("共享经验池测试完成")