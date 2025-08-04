"""
SAC (Soft Actor-Critic) 智能体实现
基于 Stable Baselines3 的 SAC 算法包装
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import json
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecTransposeDict

from .transformer import TimeSeriesTransformer, TransformerConfig



@dataclass
class SACConfig:
    """
    基于 Stable Baselines3 的 SAC 配置
    
    注意：核心参数（learning_rate, batch_size, buffer_size, gamma, tau）
    现在由TrainingConfig统一管理，此配置类主要包含SAC特有参数
    """
    # === SAC特有算法参数 ===
    ent_coef: str = 'auto'       # 熵系数，'auto' 表示自动调整
    target_entropy: str = 'auto' # 目标熵，'auto' 表示 -dim(action_space)
    
    # === SB3训练行为参数 ===
    learning_starts: int = 1000    # 开始学习的最小步数
    train_freq: int = 1           # 训练频率
    gradient_steps: int = 1       # 每次更新的梯度步数
    target_update_interval: int = 1 # 目标网络更新间隔
    
    # === 网络架构 ===
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "relu"   # 激活函数类型
    
    # === 设备和其他 ===
    device: str = 'auto'
    seed: Optional[int] = None
    verbose: int = 1
    
    # === 自定义特征提取器参数 ===
    use_transformer: bool = True
    transformer_config: Optional[TransformerConfig] = None
    
    # === 从TrainingConfig注入的参数（运行时设置，不设默认值） ===
    # 这些参数将在创建时从TrainingConfig注入：
    # - learning_rate
    # - batch_size  
    # - buffer_size
    # - gamma
    # - tau
    # - total_timesteps
    # - eval_freq
    # - n_eval_episodes
    
    def get_activation_fn(self):
        """获取激活函数"""
        if self.activation_fn == "relu":
            return nn.ReLU
        elif self.activation_fn == "tanh":
            return nn.Tanh
        elif self.activation_fn == "elu":
            return nn.ELU
        else:
            raise ValueError(f"不支持的激活函数: {self.activation_fn}")


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    基于 Transformer 的特征提取器，用于 Stable Baselines3
    """
    
    def __init__(self, observation_space: gym.Space, transformer_config: TransformerConfig, **kwargs):
        # 计算特征维度
        features_dim = transformer_config.d_model
        super().__init__(observation_space, features_dim)
        
        self.transformer = TimeSeriesTransformer(transformer_config)
        
        # 记录调试信息
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"初始化TransformerFeaturesExtractor: d_model={features_dim}, "
                   f"观察空间={observation_space}")
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            observations: 输入观察，可能是字典或张量
            
        Returns:
            torch.Tensor: 编码后的特征
        """
        # 处理字典观察
        if isinstance(observations, dict):
            features = observations['features']
        else:
            features = observations
            
        # 确保输入是正确的维度格式
        if features.dim() == 3:
            features = features.unsqueeze(0)  # 添加批次维度
        elif features.dim() == 2:
            features = features.unsqueeze(0).unsqueeze(0)  # 添加批次和序列维度
            
        # 通过 Transformer 编码
        encoded = self.transformer(features)
        
        # 如果输出是 3D 的，对股票维度进行平均池化
        if encoded.dim() == 3:
            encoded = encoded.mean(dim=1)
            
        # 移除批次维度（如果只有一个样本）
        if encoded.size(0) == 1:
            encoded = encoded.squeeze(0)
            
        return encoded


class SACAgent:
    """
    基于 Stable Baselines3 的 SAC 智能体包装器
    
    提供与原有接口兼容的 SAC 实现，内部使用 SB3 的 SAC 算法
    """
    
    def __init__(self, config: SACConfig, env: Union[gym.Env, 'VecEnv'] = None, env_factory=None, n_envs: int = 1, 
                 training_config=None):
        """
        初始化SAC智能体
        
        Args:
            config: SACConfig配置
            env: 环境实例
            env_factory: 环境工厂函数
            n_envs: 并行环境数量
            training_config: TrainingConfig配置（用于注入统一参数）
        """
        self.config = config
        self.env = env
        self.env_factory = env_factory
        self.n_envs = n_envs
        self.logger = logging.getLogger(__name__)
        
        # 从TrainingConfig注入参数（如果提供）
        self.training_config = training_config
        if training_config:
            self._inject_training_params(training_config)
        
        # 存储训练统计信息（SB3兼容）
        self.training_step = 0
        self.total_env_steps = 0
        
        # SB3 模型（将在需要时创建）
        self._model = None
        self._vec_env = None
        
        # 设置设备
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
            
    def _inject_training_params(self, training_config):
        """
        从TrainingConfig注入统一管理的参数到SACConfig
        
        Args:
            training_config: TrainingConfig实例
        """
        # 注入核心参数（以TrainingConfig为准）
        self.learning_rate = training_config.learning_rate
        self.batch_size = training_config.batch_size
        self.buffer_size = training_config.buffer_size
        self.gamma = training_config.gamma
        self.tau = training_config.tau
        self.total_timesteps = training_config.total_timesteps
        self.eval_freq = training_config.eval_freq
        self.n_eval_episodes = training_config.n_eval_episodes
        
        self.logger.info(f"已从TrainingConfig注入参数: lr={self.learning_rate}, "
                        f"batch_size={self.batch_size}, buffer_size={self.buffer_size}")
        self.logger.info(f"训练参数: total_timesteps={self.total_timesteps}, "
                        f"eval_freq={self.eval_freq}, n_eval_episodes={self.n_eval_episodes}")
            
    def _create_vec_env(self):
        """创建向量化环境"""
        if self._vec_env is not None:
            return self._vec_env
            
        if self.env is not None:
            # 如果已有环境，检查是否为VecEnv
            if hasattr(self.env, 'num_envs'):
                # 已经是VecEnv
                self._vec_env = self.env
                self.n_envs = self.env.num_envs
            else:
                # 单环境，包装为DummyVecEnv
                from stable_baselines3.common.vec_env import DummyVecEnv
                self._vec_env = DummyVecEnv([lambda: self.env])
                self.n_envs = 1
        elif self.env_factory is not None:
            # 首先检查env_factory是否返回VecEnv（带异常处理）
            try:
                test_env = self.env_factory()
                if hasattr(test_env, 'num_envs'):
                    # env_factory返回的已经是VecEnv，直接使用
                    self.logger.info("env_factory返回VecEnv，直接使用而不重复包装")
                    self._vec_env = test_env
                    self.n_envs = test_env.num_envs
                else:
                    # env_factory返回普通环境，需要向量化包装
                    if self.n_envs > 1:
                        # 包装环境工厂以增强异常追踪
                        def safe_env_factory():
                            try:
                                return self.env_factory()
                            except Exception as e:
                                import traceback
                                self.logger.error(f"环境工厂创建环境失败: {e}")
                                self.logger.error(f"完整traceback: {traceback.format_exc()}")
                                raise RuntimeError(f"环境工厂创建环境失败: {e}") from e
                        
                        self._vec_env = make_vec_env(
                            safe_env_factory,
                            n_envs=self.n_envs,
                            vec_env_cls=SubprocVecEnv,
                            vec_env_kwargs={'start_method': 'spawn'}  # 兼容性设置
                        )
                    else:
                        self._vec_env = make_vec_env(
                            self.env_factory,
                            n_envs=1,
                            vec_env_cls=DummyVecEnv
                        )
            except Exception as e:
                import traceback
                self.logger.error(f"环境工厂测试失败: {e}")
                self.logger.error(f"完整traceback: {traceback.format_exc()}")
                raise RuntimeError(f"环境工厂测试失败，无法创建向量化环境: {e}") from e
        else:
            raise ValueError("需要提供env或env_factory参数")
            
        # 检查是否需要VecTransposeDict包装
        if hasattr(self._vec_env.observation_space, 'spaces'):
            self._vec_env = VecTransposeDict(self._vec_env)
            self.logger.info("已应用VecTransposeDict包装器")
            
        self.logger.info(f"创建了{self.n_envs}个并行环境")
        return self._vec_env
        
    def _create_model(self, vec_env) -> SAC:
        """创建 SB3 SAC 模型"""
        # 准备策略参数
        policy_kwargs = {
            "net_arch": self.config.net_arch,
            "activation_fn": self.config.get_activation_fn(),
        }
        
        # 如果使用 Transformer，添加自定义特征提取器
        if self.config.use_transformer and self.config.transformer_config is not None:
            policy_kwargs["features_extractor_class"] = TransformerFeaturesExtractor
            policy_kwargs["features_extractor_kwargs"] = {
                "transformer_config": self.config.transformer_config
            }
        
        # 检查观察空间类型，选择合适的策略
        if hasattr(vec_env.observation_space, 'spaces'):
            # 字典观察空间
            policy_type = "MultiInputPolicy"
        else:
            # 普通观察空间
            policy_type = "MlpPolicy"
        
        # 使用注入的参数或默认值创建 SAC 模型
        learning_rate = getattr(self, 'learning_rate', 3e-4)
        buffer_size = getattr(self, 'buffer_size', 1000000)
        batch_size = getattr(self, 'batch_size', 256)
        gamma = getattr(self, 'gamma', 0.99)
        tau = getattr(self, 'tau', 0.005)
        
        model = SAC(
            policy=policy_type,
            env=vec_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            target_update_interval=self.config.target_update_interval,
            ent_coef=self.config.ent_coef,
            target_entropy=self.config.target_entropy,
            policy_kwargs=policy_kwargs,
            verbose=self.config.verbose,
            seed=self.config.seed,
            device=self.device
        )
        
        return model
        
    @property
    def model(self) -> SAC:
        """获取 SB3 模型实例"""
        if self._model is None:
            vec_env = self._create_vec_env()
            self._model = self._create_model(vec_env)
        return self._model
        
    @property
    def vec_env(self):
        """获取向量化环境"""
        if self._vec_env is None:
            self._create_vec_env()
        return self._vec_env
        
    def set_env(self, env: Union[gym.Env, 'VecEnv'], env_factory=None, n_envs: int = None):
        """设置环境"""
        self.env = env
        if env_factory is not None:
            self.env_factory = env_factory
        if n_envs is not None:
            self.n_envs = n_envs
            
        # 重置向量化环境和模型
        self._vec_env = None
        if self._model is not None:
            vec_env = self._create_vec_env()
            self._model.set_env(vec_env)
        else:
            # 如果模型还没创建，现在创建它
            vec_env = self._create_vec_env()
            self._model = self._create_model(vec_env)
            
    def get_action(self, state, deterministic: bool = False, return_log_prob: bool = False):
        """
        获取动作（兼容原有接口）
        
        Args:
            state: 状态张量或字典观察
            deterministic: 是否使用确定性策略
            return_log_prob: 是否返回对数概率（SB3不直接支持，返回None）
            
        Returns:
            action: 动作张量
            log_prob: 对数概率（如果return_log_prob=True，但SB3不支持，返回None）  
        """
        if self._model is None:
            raise RuntimeError("模型尚未初始化，请先调用 set_env() 或进行训练")
            
        # 转换状态格式
        if isinstance(state, dict):
            obs = state
        else:
            # 如果是张量，转换为numpy数组
            if isinstance(state, torch.Tensor):
                obs = state.detach().cpu().numpy()
            else:
                obs = state
        
        # 使用SB3预测动作
        action, _ = self.model.predict(obs, deterministic=deterministic)
        
        # 转换为torch张量
        action_tensor = torch.from_numpy(action).float()
        
        if return_log_prob:
            # SB3不直接提供log_prob，返回None
            return action_tensor, None
        else:
            return action_tensor
    
    def learn(self, total_timesteps: int = None, **kwargs):
        """
        训练模型
        
        Args:
            total_timesteps: 训练步数
            **kwargs: 其他参数
        """
        if total_timesteps is None:
            # 使用注入的参数或默认值
            total_timesteps = getattr(self, 'total_timesteps', 100000)
            
        self.logger.info(f"开始训练 SAC 模型，总步数: {total_timesteps}")
        
        # 训练模型（回调现在由trainer统一管理）
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        
        # 更新统计信息
        self.total_env_steps = self.model.num_timesteps
        self.training_step = self.model.num_timesteps
        
        # 打印最终训练统计
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            record = self.model.logger.name_to_value
            final_stats = {}
            if 'train/actor_loss' in record:
                final_stats['最终Actor损失'] = record['train/actor_loss']
            if 'train/critic_loss' in record:
                final_stats['最终Critic损失'] = record['train/critic_loss']
            if 'train/ent_coef' in record:
                final_stats['最终熵系数'] = record['train/ent_coef']
            
            if final_stats:
                stats_str = " | ".join([f"{k}: {v:.4f}" for k, v in final_stats.items()])
                self.logger.info(f"训练完成 - {stats_str}")
        
    def save(self, path: Union[str, Path]) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self._model is not None:
            # 保存SB3模型
            self.model.save(path / "sac_model")
            
            # 保存配置
            with open(path / 'config.json', 'w') as f:
                # 将配置转换为可序列化的字典
                config_dict = {
                    k: v for k, v in self.config.__dict__.items() 
                    if isinstance(v, (int, float, str, bool, list, type(None)))
                }
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"模型已保存到 {path}")
        else:
            raise RuntimeError("模型尚未初始化，无法保存")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        path = Path(path)
        
        # 加载SB3模型
        model_path = path / "sac_model"
        if model_path.with_suffix('.zip').exists():
            self._model = SAC.load(model_path, env=self.env)
            self.logger.info(f"模型已从 {path} 加载")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    def can_update(self) -> bool:
        """检查是否可以更新（兼容性方法）"""
        if self._model is None:
            return False
        return self.model.num_timesteps >= self.config.learning_starts
    
    def update(self, **kwargs) -> Dict[str, float]:
        """
        更新网络参数（兼容性方法，推荐直接使用.learn()）
        
        注意：SB3的训练应该通过learn()方法进行，这里保留以向后兼容
        """
        if self._model is None:
            return {}
        
        # 检查是否可以进行训练
        if not self.can_update():
            return {}
        
        # 简化版本：返回基本统计信息
        self.total_env_steps = self.model.num_timesteps
        self.training_step = self.model.num_timesteps
        
        # 从SB3 logger获取最新统计（如果可用）
        stats = {}
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            record = self.model.logger.name_to_value
            if 'train/actor_loss' in record:
                stats['actor_loss'] = float(record['train/actor_loss'])
            if 'train/critic_loss' in record:
                stats['critic_loss'] = float(record['train/critic_loss'])
            if 'train/ent_coef' in record:
                stats['entropy_coef'] = float(record['train/ent_coef'])
        
        return stats
    
    def get_training_stats(self) -> Dict[str, float]:
        """获取训练统计（兼容性方法）"""
        stats = {
            'training_step': self.training_step,
            'total_env_steps': self.total_env_steps,
        }
        
        if self._model is not None:
            stats['buffer_size'] = len(self.model.replay_buffer) if hasattr(self.model, 'replay_buffer') else 0
        
        return stats
    
    def train(self):
        """设置模型为训练模式（兼容PyTorch接口）"""
        if self._model is not None:
            # SB3模型内部的网络设置为训练模式
            if hasattr(self.model.policy, 'train'):
                self.model.policy.train()
            if hasattr(self.model, 'critic') and hasattr(self.model.critic, 'train'):
                self.model.critic.train()
            if hasattr(self.model, 'critic_target') and hasattr(self.model.critic_target, 'train'):
                self.model.critic_target.train()
        return self
    
    def eval(self):
        """设置模型为评估模式（兼容PyTorch接口）"""
        if self._model is not None:
            # SB3模型内部的网络设置为评估模式
            if hasattr(self.model.policy, 'eval'):
                self.model.policy.eval()
            if hasattr(self.model, 'critic') and hasattr(self.model.critic, 'eval'):
                self.model.critic.eval()
            if hasattr(self.model, 'critic_target') and hasattr(self.model.critic_target, 'eval'):
                self.model.critic_target.eval()
        return self
    
    def encode_observation(self, obs, training: bool = False):
        """
        编码观察（兼容性方法，简化版本）
        
        Args:
            obs: 观察数据，可能是字典或张量
            training: 是否为训练模式
            
        Returns:
            torch.Tensor: 编码后的观察
        """
        # 简化处理：SB3会自动通过特征提取器处理观察
        if isinstance(obs, dict):
            features = obs.get('features', obs)
        else:
            features = obs
            
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features).float()
            
        return features
    
    def add_experience(self, experience):
        """
        添加经验到回放缓冲区（兼容性方法）
        
        注意：SB3会自动管理经验回放，这个方法主要用于向后兼容
        """
        # SB3的经验回放是在.learn()过程中自动处理的
        # 这里只是为了向后兼容，实际不需要手动添加经验
        self.logger.debug("SB3会自动管理经验回放，无需手动添加")