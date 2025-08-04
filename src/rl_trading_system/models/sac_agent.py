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

from .transformer import TimeSeriesTransformer, TransformerConfig


@dataclass
class SACConfig:
    """基于 Stable Baselines3 的 SAC 配置"""
    # 学习率
    learning_rate: float = 3e-4
    
    # SAC算法参数
    gamma: float = 0.99          # 折扣因子
    tau: float = 0.005           # 软更新系数
    ent_coef: str = 'auto'       # 熵系数，'auto' 表示自动调整
    target_entropy: str = 'auto' # 目标熵，'auto' 表示 -dim(action_space)
    
    # 训练参数
    batch_size: int = 256
    buffer_size: int = 1000000
    learning_starts: int = 1000    # 开始学习的最小步数
    train_freq: int = 1           # 训练频率
    gradient_steps: int = 1       # 每次更新的梯度步数
    target_update_interval: int = 1 # 目标网络更新间隔
    
    # 网络架构
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation_fn: str = "relu"   # 激活函数类型
    
    # 设备和其他
    device: str = 'auto'
    seed: Optional[int] = None
    verbose: int = 1
    
    # 自定义特征提取器参数
    use_transformer: bool = True
    transformer_config: Optional[TransformerConfig] = None
    
    # 训练和评估相关
    total_timesteps: int = 100000
    eval_freq: int = 5000
    n_eval_episodes: int = 10
    
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
    
    def __init__(self, observation_space: gym.Space, transformer_config: TransformerConfig):
        # 计算特征维度
        features_dim = transformer_config.d_model
        super().__init__(observation_space, features_dim)
        
        self.transformer = TimeSeriesTransformer(transformer_config)
        
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
    
    def __init__(self, config: SACConfig, env: gym.Env = None):
        self.config = config
        self.env = env
        self.logger = logging.getLogger(__name__)
        
        # 存储训练统计信息
        self.training_step = 0
        self.episode_count = 0
        self.total_env_steps = 0
        
        # SB3 模型（将在需要时创建）
        self._model = None
        
        # 设置设备
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
            
    def _create_model(self, env: gym.Env) -> SAC:
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
        
        # 创建 SAC 模型
        model = SAC(
            policy="MlpPolicy",  # 如果有自定义特征提取器，SB3会自动使用
            env=env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
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
            if self.env is None:
                raise RuntimeError("需要先设置环境或创建模型后才能访问")
            self._model = self._create_model(self.env)
        return self._model
        
    def set_env(self, env: gym.Env):
        """设置环境"""
        self.env = env
        if self._model is not None:
            self._model.set_env(env)
            
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
            total_timesteps = self.config.total_timesteps
            
        self.logger.info(f"开始训练 SAC 模型，总步数: {total_timesteps}")
        
        # 训练模型
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
        
        # 更新统计信息
        self.total_env_steps = self.model.num_timesteps
        self.training_step = self.model.num_timesteps
        
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
        更新网络参数（兼容性包装）
        注意：SB3的训练是通过learn()方法进行的，这里只是为了兼容性
        """
        # SB3不支持单步更新，返回空字典
        return {}
    
    def get_training_stats(self) -> Dict[str, float]:
        """获取训练统计（兼容性方法）"""
        stats = {
            'training_step': self.training_step,
            'total_env_steps': self.total_env_steps,
        }
        
        if self._model is not None:
            stats['buffer_size'] = len(self.model.replay_buffer) if hasattr(self.model, 'replay_buffer') else 0
        
        return stats