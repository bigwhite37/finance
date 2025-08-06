"""
专家策略实现
基于SAC的低层专家，带DIAYN互信息奖励机制
参考claude.md中的ExpertPolicy实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import random
from collections import deque
from stable_baselines3 import SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.noise import NormalActionNoise
import gym

from .trans_encoder import MutualInfoEstimator


class CustomSACPolicy(BasePolicy):
    """
    自定义SAC策略，集成TimesNet编码器和互信息奖励
    """
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 lr_schedule, net_arch: Optional[Dict[str, Any]] = None,
                 activation_fn: type = nn.ReLU, features_extractor_class=None,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True, optimizer_class=torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None):
        
        super().__init__(observation_space, action_space, 
                        features_extractor_class, features_extractor_kwargs, 
                        optimizer_class, optimizer_kwargs, normalize_images)
        
        self.net_arch = net_arch or {}
        self.activation_fn = activation_fn
        self.lr_schedule = lr_schedule
        
        # 网络架构
        self.actor_net = self._build_actor()
        self.critic_net = self._build_critic()
        
        # 优化器
        self.actor_optimizer = optimizer_class(
            self.actor_net.parameters(), 
            lr=lr_schedule(1), 
            **(optimizer_kwargs or {})
        )
        self.critic_optimizer = optimizer_class(
            self.critic_net.parameters(), 
            lr=lr_schedule(1), 
            **(optimizer_kwargs or {})
        )
    
    def _build_actor(self):
        """构建actor网络"""
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        
        return nn.Sequential(
            nn.Linear(obs_dim, 256),
            self.activation_fn(),
            nn.Linear(256, 256),
            self.activation_fn(),
            nn.Linear(256, action_dim * 2)  # mean和log_std
        )
    
    def _build_critic(self):
        """构建critic网络"""
        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]
        
        return nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            self.activation_fn(),
            nn.Linear(256, 256), 
            self.activation_fn(),
            nn.Linear(256, 1)
        )
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """前向传播"""
        actor_output = self.actor_net(obs)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            return action


class ExpertPolicy:
    """
    专家策略类
    基于SAC实现，集成DIAYN互信息奖励机制
    """
    
    def __init__(self, 
                 expert_id: int,
                 env: GymEnv,
                 skill_dim: int = 10,
                 buffer_size: int = int(1e6),
                 learning_rate: float = 3e-4,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 mi_reward_weight: float = 0.5,  # β parameter
                 device: str = "auto"):
        
        self.expert_id = expert_id
        self.env = env
        self.skill_dim = skill_dim
        self.mi_reward_weight = mi_reward_weight
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 生成专家的技能向量（固定的，定义专家特征）
        np.random.seed(expert_id)  # 确保每个专家有独特但固定的技能
        self.skill_z = torch.tensor(
            np.random.randn(skill_dim), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 动作噪声
        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[0]),
            sigma=0.1 * np.ones(env.action_space.shape[0])
        )
        
        # SAC策略
        self.policy = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            tau=tau,
            gamma=gamma,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            action_noise=action_noise,
            device=self.device,
            verbose=0
        )
        
        # 互信息估计器
        self.mi_estimator = MutualInfoEstimator(
            skill_dim=skill_dim,
            action_dim=env.action_space.shape[0],
            hidden_dim=64
        ).to(self.device)
        
        self.mi_optimizer = torch.optim.Adam(
            self.mi_estimator.parameters(), 
            lr=learning_rate
        )
        
        # 性能跟踪
        self.performance_history = deque(maxlen=100)
        self.cumulative_reward = 0.0
        self.episode_count = 0
        
        # 互信息训练历史
        self.mi_loss_history = deque(maxlen=1000)
        
    def compute_intrinsic_reward(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        计算DIAYN内在奖励：β * I(a; z)
        使用InfoNCE估计互信息
        """
        with torch.no_grad():
            # 转换为tensor
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
            skill_tensor = self.skill_z.unsqueeze(0)  # 添加batch维度
            
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.unsqueeze(0)
            
            # 估计互信息
            mi_estimate = self.mi_estimator(action_tensor, skill_tensor)
            mi_reward = mi_estimate.item() * self.mi_reward_weight
            
            return mi_reward
    
    def update_mi_estimator(self, batch_actions: torch.Tensor, batch_skills: torch.Tensor,
                           negative_samples: int = 10) -> float:
        """
        使用InfoNCE损失更新互信息估计器
        
        Args:
            batch_actions: 动作批次 [batch_size, action_dim]
            batch_skills: 技能批次 [batch_size, skill_dim]
            negative_samples: 负样本数量
            
        Returns:
            互信息损失值
        """
        batch_size = batch_actions.size(0)
        
        # 正样本：真实的(action, skill)对
        positive_scores = self.mi_estimator(batch_actions, batch_skills)
        
        # 负样本：随机配对的(action, skill)
        negative_scores = []
        for _ in range(negative_samples):
            # 打乱技能向量创建负样本
            shuffled_skills = batch_skills[torch.randperm(batch_size)]
            neg_scores = self.mi_estimator(batch_actions, shuffled_skills)
            negative_scores.append(neg_scores)
        
        negative_scores = torch.cat(negative_scores, dim=1)  # [batch_size, negative_samples]
        
        # InfoNCE损失
        # log(exp(positive) / (exp(positive) + sum(exp(negatives))))
        all_scores = torch.cat([positive_scores, negative_scores], dim=1)  # [batch_size, 1 + negative_samples]
        
        # 使用log-sum-exp技巧提高数值稳定性
        log_prob = positive_scores.squeeze(1) - torch.logsumexp(all_scores, dim=1)
        mi_loss = -log_prob.mean()  # 最大化互信息等价于最小化负对数似然
        
        # 反向传播
        self.mi_optimizer.zero_grad()
        mi_loss.backward()
        self.mi_optimizer.step()
        
        # 记录损失
        loss_value = mi_loss.item()
        self.mi_loss_history.append(loss_value)
        
        return loss_value
    
    def train_step(self, total_timesteps: int = 5000) -> Dict[str, float]:
        """
        训练一个步骤
        """
        # 训练SAC策略
        self.policy.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        
        # 如果回放缓冲区有足够数据，训练互信息估计器
        if len(self.policy.replay_buffer) > 1000:
            self._train_mi_estimator_batch()
        
        # 更新性能指标
        info = {
            'expert_id': self.expert_id,
            'episodes_trained': self.episode_count,
            'avg_performance': np.mean(self.performance_history) if self.performance_history else 0.0,
            'mi_loss': np.mean(self.mi_loss_history) if self.mi_loss_history else 0.0,
            'skill_norm': torch.norm(self.skill_z).item()
        }
        
        return info
    
    def _train_mi_estimator_batch(self, batch_size: int = 256):
        """从回放缓冲区采样训练互信息估计器"""
        if len(self.policy.replay_buffer) < batch_size:
            return
        
        # 从回放缓冲区采样
        batch = self.policy.replay_buffer.sample(batch_size, env=self.env)
        
        # 转换为tensor
        actions = torch.tensor(batch.actions, dtype=torch.float32, device=self.device)
        if actions.dim() == 3:  # 如果有额外维度，压缩
            actions = actions.squeeze(1)
        
        # 为每个动作分配当前专家的技能向量
        skills = self.skill_z.unsqueeze(0).repeat(batch_size, 1)
        
        # 更新互信息估计器
        self.update_mi_estimator(actions, skills)
    
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        预测动作
        """
        action, _states = self.policy.predict(observation, deterministic=deterministic)
        
        # 计算内在奖励（用于记录）
        intrinsic_reward = self.compute_intrinsic_reward(observation, action)
        
        info = {
            'expert_id': self.expert_id,
            'intrinsic_reward': intrinsic_reward,
            'skill_z': self.skill_z.cpu().numpy()
        }
        
        return action, info
    
    def update_performance(self, episode_reward: float):
        """更新性能记录"""
        self.performance_history.append(episode_reward)
        self.cumulative_reward = episode_reward
        self.episode_count += 1
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return {
            'expert_id': self.expert_id,
            'avg_reward': np.mean(self.performance_history) if self.performance_history else 0.0,
            'std_reward': np.std(self.performance_history) if len(self.performance_history) > 1 else 0.0,
            'recent_reward': self.performance_history[-1] if self.performance_history else 0.0,
            'episode_count': self.episode_count,
            'mi_loss': np.mean(self.mi_loss_history) if self.mi_loss_history else 0.0,
            'skill_vector': self.skill_z.cpu().numpy().tolist()
        }
    
    def reset_with_random_skill(self):
        """
        重置专家技能（用于多样性注入）
        """
        self.skill_z = torch.tensor(
            np.random.randn(self.skill_dim),
            dtype=torch.float32,
            device=self.device
        )
        
        # 清理性能历史，重新开始
        self.performance_history.clear()
        self.mi_loss_history.clear()
        self.episode_count = 0
        self.cumulative_reward = 0.0
        
        print(f"Expert {self.expert_id} reset with new skill vector")
    
    def save_model(self, path: str):
        """保存模型"""
        save_dict = {
            'policy': self.policy,
            'mi_estimator': self.mi_estimator.state_dict(),
            'skill_z': self.skill_z,
            'expert_id': self.expert_id,
            'performance_history': list(self.performance_history)
        }
        torch.save(save_dict, path)
    
    def load_model(self, path: str):
        """加载模型"""
        save_dict = torch.load(path, map_location=self.device)
        self.policy = save_dict['policy']
        self.mi_estimator.load_state_dict(save_dict['mi_estimator'])
        self.skill_z = save_dict['skill_z'].to(self.device)
        self.expert_id = save_dict['expert_id']
        self.performance_history = deque(save_dict['performance_history'], maxlen=100)


class ExpertPopulation:
    """
    专家种群管理器
    管理多个专家策略，实现种群级别的操作
    """
    
    def __init__(self, 
                 n_experts: int,
                 env_factory,  # 环境工厂函数
                 skill_dim: int = 10,
                 **expert_kwargs):
        
        self.n_experts = n_experts
        self.skill_dim = skill_dim
        self.env_factory = env_factory
        
        # 创建专家
        self.experts = []
        for i in range(n_experts):
            env = env_factory()
            expert = ExpertPolicy(
                expert_id=i,
                env=env,
                skill_dim=skill_dim,
                **expert_kwargs
            )
            self.experts.append(expert)
        
        print(f"创建了 {n_experts} 个专家策略")
    
    def get_expert(self, expert_id: int) -> ExpertPolicy:
        """获取指定专家"""
        return self.experts[expert_id]
    
    def train_all_experts(self, total_timesteps: int = 5000) -> Dict[str, Any]:
        """训练所有专家"""
        results = {}
        for expert in self.experts:
            expert_info = expert.train_step(total_timesteps)
            results[f'expert_{expert.expert_id}'] = expert_info
        
        # 汇总统计
        all_performances = [expert.get_performance_metrics()['avg_reward'] 
                          for expert in self.experts]
        
        results['population_stats'] = {
            'avg_performance': np.mean(all_performances),
            'std_performance': np.std(all_performances),
            'diversity_score': self.compute_diversity_score()
        }
        
        return results
    
    def compute_diversity_score(self) -> float:
        """
        计算专家多样性得分
        基于技能向量的平均欧式距离
        """
        if len(self.experts) < 2:
            return 0.0
        
        skill_vectors = [expert.skill_z.cpu().numpy() for expert in self.experts]
        total_distance = 0.0
        count = 0
        
        for i in range(len(skill_vectors)):
            for j in range(i + 1, len(skill_vectors)):
                distance = np.linalg.norm(skill_vectors[i] - skill_vectors[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def get_population_metrics(self) -> Dict[str, Any]:
        """获取种群级别的指标"""
        expert_metrics = [expert.get_performance_metrics() for expert in self.experts]
        
        return {
            'n_experts': self.n_experts,
            'experts': expert_metrics,
            'diversity_score': self.compute_diversity_score(),
            'avg_population_reward': np.mean([m['avg_reward'] for m in expert_metrics]),
            'total_episodes': sum([m['episode_count'] for m in expert_metrics])
        }
    
    def reset_worst_expert(self):
        """重置表现最差的专家（多样性注入）"""
        performances = [expert.get_performance_metrics()['avg_reward'] 
                       for expert in self.experts]
        worst_idx = np.argmin(performances)
        self.experts[worst_idx].reset_with_random_skill()
        
        return worst_idx


if __name__ == "__main__":
    # 测试专家策略
    import gym
    
    # 创建测试环境
    def make_env():
        return gym.make('Pendulum-v1')
    
    # 创建专家种群
    expert_population = ExpertPopulation(
        n_experts=3,
        env_factory=make_env,
        skill_dim=10,
        buffer_size=10000,
        learning_rate=3e-4
    )
    
    print("专家种群创建完成")
    
    # 测试训练
    results = expert_population.train_all_experts(total_timesteps=1000)
    print("训练结果:", results)
    
    # 测试多样性计算
    diversity = expert_population.compute_diversity_score()
    print(f"种群多样性得分: {diversity:.4f}")
    
    # 获取种群指标
    metrics = expert_population.get_population_metrics()
    print("种群指标:", metrics)