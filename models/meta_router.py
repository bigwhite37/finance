"""
元路由器实现
基于MaskablePPO的高层决策器，带KL散度多样性奖励
参考claude.md中的MetaRouter实现
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import gym
from gym import spaces
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import GymEnv


class MetaEnvironment(gym.Env):
    """
    元环境：高层路由器的环境包装
    状态：市场状态特征
    动作：选择专家ID
    奖励：外部收益 + KL多样性奖励
    """
    
    def __init__(self, 
                 experts: List,
                 base_env: GymEnv,
                 kl_lambda: float = 0.1,
                 episode_length: int = 252,  # 一年交易日
                 history_length: int = 1000):
        
        super().__init__()
        
        self.experts = experts
        self.base_env = base_env
        self.n_experts = len(experts)
        self.kl_lambda = kl_lambda
        self.episode_length = episode_length
        self.history_length = history_length
        
        # 动作空间：选择专家ID
        self.action_space = spaces.Discrete(self.n_experts)
        
        # 观测空间：基础环境观测 + 专家性能特征
        base_obs_dim = base_env.observation_space.shape[0]
        expert_features_dim = self.n_experts * 3  # 每个专家的：平均回报、最近回报、使用频率
        market_features_dim = 10  # 市场状态特征
        
        total_obs_dim = base_obs_dim + expert_features_dim + market_features_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )
        
        # 专家使用历史（用于计算KL散度）
        self.expert_usage_history = deque(maxlen=history_length)
        self.current_step = 0
        self.current_episode = 0
        
        # 性能跟踪
        self.episode_rewards = []
        self.expert_selection_counts = np.zeros(self.n_experts)
        
        # 状态缓存
        self._last_base_obs = None
        self._last_expert_metrics = None
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.current_episode += 1
        
        # 重置基础环境
        base_obs = self.base_env.reset()
        self._last_base_obs = base_obs
        
        # 获取专家特征
        expert_features = self._get_expert_features()
        
        # 获取市场特征
        market_features = self._get_market_features()
        
        # 合并观测
        full_obs = np.concatenate([base_obs, expert_features, market_features])
        
        return full_obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步"""
        selected_expert_id = action
        self.current_step += 1
        
        # 记录专家选择
        self.expert_usage_history.append(selected_expert_id)
        self.expert_selection_counts[selected_expert_id] += 1
        
        # 使用选定专家执行动作
        expert = self.experts[selected_expert_id]
        expert_action, expert_info = expert.predict(self._last_base_obs, deterministic=False)
        
        # 在基础环境中执行动作
        base_obs, base_reward, base_done, base_info = self.base_env.step(expert_action)
        self._last_base_obs = base_obs
        
        # 计算KL多样性奖励
        kl_reward = self._compute_kl_reward(selected_expert_id)
        
        # 总奖励 = 基础奖励 + KL多样性奖励
        total_reward = base_reward + kl_reward
        
        # 检查是否结束
        done = base_done or (self.current_step >= self.episode_length)
        
        # 获取新的观测
        expert_features = self._get_expert_features()
        market_features = self._get_market_features()
        full_obs = np.concatenate([base_obs, expert_features, market_features])
        
        # 信息字典
        info = {
            'selected_expert': selected_expert_id,
            'base_reward': base_reward,
            'kl_reward': kl_reward,
            'total_reward': total_reward,
            'expert_info': expert_info,
            'base_info': base_info,
            'expert_usage_dist': self._get_usage_distribution(),
            'step': self.current_step
        }
        
        # 如果回合结束，记录性能
        if done:
            self.episode_rewards.append(total_reward)
        
        return full_obs.astype(np.float32), total_reward, done, info
    
    def _compute_kl_reward(self, selected_expert_id: int) -> float:
        """
        计算KL散度奖励
        KL(π_selected || π_average) 鼓励选择不常用的专家
        """
        if len(self.expert_usage_history) < 10:  # 需要足够的历史数据
            return 0.0
        
        # 计算历史使用分布
        usage_counts = np.bincount(list(self.expert_usage_history), minlength=self.n_experts)
        usage_dist = usage_counts / usage_counts.sum()
        
        # 当前选择的one-hot分布
        selected_dist = np.zeros(self.n_experts)
        selected_dist[selected_expert_id] = 1.0
        
        # 计算KL散度：KL(selected || average)
        # 添加小常数防止log(0)
        eps = 1e-8
        usage_dist = usage_dist + eps
        selected_dist = selected_dist + eps
        
        kl_div = np.sum(selected_dist * np.log(selected_dist / usage_dist))
        
        # KL奖励：鼓励多样性
        kl_reward = self.kl_lambda * kl_div
        
        return kl_reward
    
    def _get_expert_features(self) -> np.ndarray:
        """获取专家特征"""
        features = []
        
        for expert in self.experts:
            metrics = expert.get_performance_metrics()
            
            # 专家特征：平均回报、最近回报、使用频率
            avg_reward = metrics.get('avg_reward', 0.0)
            recent_reward = metrics.get('recent_reward', 0.0)
            
            # 使用频率（近期历史）
            recent_usage = list(self.expert_usage_history)[-100:] if len(self.expert_usage_history) >= 100 else list(self.expert_usage_history)
            usage_freq = recent_usage.count(expert.expert_id) / max(len(recent_usage), 1)
            
            features.extend([avg_reward, recent_reward, usage_freq])
        
        return np.array(features, dtype=np.float32)
    
    def _get_market_features(self) -> np.ndarray:
        """获取市场状态特征"""
        # 这里可以添加更多市场特征，如波动率、趋势指标等
        # 目前使用简单的特征
        features = [
            self.current_step / self.episode_length,  # 进度
            len(self.expert_usage_history) / self.history_length,  # 历史长度比例
            np.std(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0.0,  # 近期回报波动
            np.mean(self.episode_rewards[-5:]) if len(self.episode_rewards) >= 5 else 0.0,  # 近期平均回报
            self._get_diversity_score(),  # 当前多样性得分
            self.kl_lambda,  # KL权重（作为策略参数）
            float(self.current_episode % 252) / 252.0,  # 年内进度
            float(self.current_episode // 252),  # 年数
            self._get_entropy_score(),  # 选择熵
            self._get_expert_performance_spread()  # 专家性能差异
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _get_usage_distribution(self) -> np.ndarray:
        """获取专家使用分布"""
        if not self.expert_usage_history:
            return np.ones(self.n_experts) / self.n_experts
        
        usage_counts = np.bincount(list(self.expert_usage_history), minlength=self.n_experts)
        return usage_counts / usage_counts.sum()
    
    def _get_diversity_score(self) -> float:
        """计算当前选择的多样性得分（熵）"""
        usage_dist = self._get_usage_distribution()
        eps = 1e-8
        entropy = -np.sum(usage_dist * np.log(usage_dist + eps))
        max_entropy = np.log(self.n_experts)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _get_entropy_score(self) -> float:
        """计算选择熵"""
        if len(self.expert_usage_history) < 2:
            return 1.0
        
        recent_choices = list(self.expert_usage_history)[-50:]  # 最近50步
        counts = np.bincount(recent_choices, minlength=self.n_experts)
        probs = counts / counts.sum()
        
        eps = 1e-8
        entropy = -np.sum(probs * np.log(probs + eps))
        max_entropy = np.log(self.n_experts)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _get_expert_performance_spread(self) -> float:
        """计算专家性能差异"""
        performances = [expert.get_performance_metrics()['avg_reward'] for expert in self.experts]
        return np.std(performances) if len(performances) > 1 else 0.0
    
    def get_action_mask(self) -> np.ndarray:
        """
        获取动作掩码（可选）
        可以用于禁用表现过差或风险过高的专家
        """
        mask = np.ones(self.n_experts, dtype=bool)
        
        # 示例：禁用性能过差的专家
        for i, expert in enumerate(self.experts):
            metrics = expert.get_performance_metrics()
            avg_reward = metrics.get('avg_reward', 0.0)
            
            # 如果专家平均回报过低，可以考虑禁用
            # 这里设置一个动态阈值
            if len(self.episode_rewards) > 10:
                threshold = np.percentile(self.episode_rewards, 10)  # 最低10%
                if avg_reward < threshold:
                    mask[i] = False
        
        # 确保至少有一个专家可用
        if not mask.any():
            mask[:] = True
        
        return mask
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'episode_count': self.current_episode,
            'current_step': self.current_step,
            'expert_usage_counts': self.expert_selection_counts.tolist(),
            'expert_usage_distribution': self._get_usage_distribution().tolist(),
            'diversity_score': self._get_diversity_score(),
            'entropy_score': self._get_entropy_score(),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'recent_episode_rewards': self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards,
            'kl_lambda': self.kl_lambda
        }


class MetaRouterFeaturesExtractor(BaseFeaturesExtractor):
    """
    自定义特征提取器，用于处理元环境的复杂观测
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class MetaRouter:
    """
    元路由器：基于MaskablePPO的高层决策器
    负责在每个时间步选择最适合的专家
    """
    
    def __init__(self, 
                 experts: List,
                 base_env: GymEnv,
                 kl_lambda: float = 0.1,
                 n_steps: int = 1024,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 ent_coef: float = 0.01,
                 learning_rate: float = 3e-4,
                 device: str = "auto"):
        
        self.experts = experts
        self.n_experts = len(experts)
        self.kl_lambda = kl_lambda
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建元环境
        self.meta_env = MetaEnvironment(
            experts=experts,
            base_env=base_env,
            kl_lambda=kl_lambda
        )
        
        # 创建MaskablePPO策略
        policy_kwargs = {
            "features_extractor_class": MetaRouterFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": [dict(pi=[128, 64], vf=[128, 64])]
        }
        
        self.policy = MaskablePPO(
            policy="MlpPolicy",
            env=self.meta_env,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=1
        )
        
        # 性能跟踪
        self.training_history = []
        self.expert_selection_history = deque(maxlen=10000)
        
    def train(self, total_timesteps: int, callback=None) -> Dict[str, Any]:
        """训练元路由器"""
        print(f"开始训练元路由器，总步数: {total_timesteps}")
        
        # 训练
        self.policy.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False
        )
        
        # 收集训练信息
        training_info = {
            'total_timesteps': total_timesteps,
            'meta_env_stats': self.meta_env.get_stats(),
            'expert_performances': [expert.get_performance_metrics() for expert in self.experts]
        }
        
        self.training_history.append(training_info)
        
        return training_info
    
    def predict(self, observation: np.ndarray, 
                action_mask: Optional[np.ndarray] = None,
                deterministic: bool = False) -> Tuple[int, Dict]:
        """预测选择哪个专家"""
        
        if action_mask is None:
            action_mask = self.meta_env.get_action_mask()
        
        # 使用MaskablePPO预测
        action, _states = self.policy.predict(
            observation, 
            action_masks=action_mask,
            deterministic=deterministic
        )
        
        selected_expert_id = int(action)
        
        # 记录选择历史
        self.expert_selection_history.append(selected_expert_id)
        
        # 返回信息
        info = {
            'selected_expert_id': selected_expert_id,
            'action_mask': action_mask,
            'usage_distribution': self.meta_env._get_usage_distribution(),
            'kl_lambda': self.kl_lambda
        }
        
        return selected_expert_id, info
    
    def update_kl_lambda(self, new_lambda: float):
        """动态调整KL权重"""
        self.kl_lambda = new_lambda
        self.meta_env.kl_lambda = new_lambda
        print(f"KL lambda更新为: {new_lambda}")
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """获取专家选择统计"""
        if not self.expert_selection_history:
            return {}
        
        selections = list(self.expert_selection_history)
        selection_counts = np.bincount(selections, minlength=self.n_experts)
        selection_probs = selection_counts / len(selections)
        
        # 计算选择熵
        eps = 1e-8
        entropy = -np.sum(selection_probs * np.log(selection_probs + eps))
        max_entropy = np.log(self.n_experts)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return {
            'selection_counts': selection_counts.tolist(),
            'selection_probabilities': selection_probs.tolist(),
            'selection_entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'total_selections': len(selections),
            'most_selected_expert': int(np.argmax(selection_counts)),
            'least_selected_expert': int(np.argmin(selection_counts))
        }
    
    def save_model(self, path: str):
        """保存模型"""
        self.policy.save(path)
        
        # 保存额外信息
        extra_info = {
            'kl_lambda': self.kl_lambda,
            'n_experts': self.n_experts,
            'training_history': self.training_history,
            'selection_history': list(self.expert_selection_history)
        }
        
        import pickle
        with open(f"{path}_extra.pkl", 'wb') as f:
            pickle.dump(extra_info, f)
    
    def load_model(self, path: str):
        """加载模型"""
        self.policy.load(path)
        
        # 加载额外信息
        import pickle
        try:
            with open(f"{path}_extra.pkl", 'rb') as f:
                extra_info = pickle.load(f)
                self.kl_lambda = extra_info['kl_lambda']
                self.training_history = extra_info['training_history']
                self.expert_selection_history = deque(extra_info['selection_history'], maxlen=10000)
        except FileNotFoundError as e:
            logger.warning(f"未找到额外信息文件: {e}，使用默认设置")
            # 不抛出异常，因为这是可选的额外信息


if __name__ == "__main__":
    # 测试元路由器
    import gym
    from models.expert_policy import ExpertPopulation
    
    print("创建测试环境...")
    
    # 创建基础环境
    def make_env():
        return gym.make('Pendulum-v1')
    
    base_env = make_env()
    
    # 创建专家种群
    expert_population = ExpertPopulation(
        n_experts=3,
        env_factory=make_env,
        skill_dim=5,
        buffer_size=5000
    )
    
    # 创建元路由器
    meta_router = MetaRouter(
        experts=expert_population.experts,
        base_env=base_env,
        kl_lambda=0.1,
        n_steps=512,
        batch_size=32
    )
    
    print("元路由器创建完成")
    
    # 测试预测
    obs = base_env.reset()
    meta_obs = meta_router.meta_env.reset()
    
    selected_expert, info = meta_router.predict(meta_obs)
    print(f"选择的专家: {selected_expert}")
    print(f"选择信息: {info}")
    
    # 测试统计
    stats = meta_router.get_selection_statistics()
    print(f"选择统计: {stats}")
    
    print("元路由器测试完成")