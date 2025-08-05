"""
强化学习训练流水线
集成Qlib数据加载、环境创建、模型训练、评估和回测
"""
import os
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnRewardThreshold,
    CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
from tqdm import tqdm

from data_loader import QlibDataLoader, split_data
from env import PortfolioEnv, DrawdownEarlyStoppingCallback
from model import TradingPolicy, RiskAwareRewardWrapper, PortfolioMetrics

logger = logging.getLogger(__name__)


class DrawdownStoppingCallback(BaseCallback):
    """回撤早停回调，用于Stable-Baselines3"""

    def __init__(self, max_drawdown: float = 0.15, patience: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.max_drawdown = max_drawdown
        self.patience = patience
        self.violation_count = 0
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        """每步检查回撤"""
        # 获取环境信息
        if hasattr(self.training_env, 'get_attr'):
            try:
                current_drawdowns = self.training_env.get_attr('current_drawdown')
                max_drawdown = max(current_drawdowns)

                if max_drawdown > self.max_drawdown:
                    self.violation_count += 1
                    if self.violation_count >= self.patience:
                        if self.verbose > 0:
                            print(f"\n回撤{max_drawdown:.2%}超过阈值{self.max_drawdown:.2%}，"
                                 f"连续{self.patience}步，触发早停")
                        return False
                else:
                    self.violation_count = 0

            except Exception as e:
                if self.verbose > 1:
                    print(f"获取回撤信息失败: {e}")

        return True


class TensorBoardCallback(BaseCallback):
    """自定义TensorBoard回调，记录额外指标"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """记录自定义指标"""
        if hasattr(self.training_env, 'get_attr'):
            try:
                # 获取环境指标
                total_values = self.training_env.get_attr('total_value')
                drawdowns = self.training_env.get_attr('current_drawdown')

                # 记录到TensorBoard
                if len(total_values) > 0:
                    self.logger.record('env/mean_total_value', np.mean(total_values))
                    self.logger.record('env/max_total_value', np.max(total_values))
                    self.logger.record('env/mean_drawdown', np.mean(drawdowns))
                    self.logger.record('env/max_drawdown', np.max(drawdowns))

            except Exception as e:
                if self.verbose > 1:
                    print(f"记录指标失败: {e}")

        return True


class RLTrainer:
    """强化学习训练器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 训练配置字典
        """
        self.config = config
        self.setup_logging()
        self.setup_directories()

        # 初始化组件
        self.data_loader = None
        self.train_env = None
        self.eval_env = None
        self.model = None

        # 训练状态
        self.training_start_time = None
        self.best_reward = -np.inf
        self.training_history = []

    def setup_logging(self):
        """设置日志"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )

    def setup_directories(self):
        """创建必要目录"""
        dirs = ['models', 'logs', 'tensorboard', 'results']
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)

    def initialize_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        初始化数据加载器并获取训练数据

        Returns:
            训练集、验证集、测试集
        """
        logger.info("初始化数据加载器...")

        data_config = self.config['data']
        self.data_loader = QlibDataLoader(data_config.get('data_root'))

        # 初始化Qlib
        provider_uri = data_config.get('provider_uri')
        self.data_loader.initialize_qlib(provider_uri)

        # 获取股票列表
        market = data_config.get('market', 'csi300')
        stock_limit = data_config.get('stock_limit', 50)
        stock_list = self.data_loader.get_stock_list(market, stock_limit)

        logger.info(f"获取{len(stock_list)}只股票用于训练")

        # 加载数据
        data = self.data_loader.load_data(
            instruments=stock_list,
            start_time=data_config['start_time'],
            end_time=data_config['end_time'],
            freq=data_config.get('freq', 'day'),
            fields=data_config.get('fields')
        )

        # 分割数据
        train_data, valid_data, test_data = split_data(
            data,
            data_config['train_start'], data_config['train_end'],
            data_config['valid_start'], data_config['valid_end'],
            data_config['test_start'], data_config['test_end']
        )

        logger.info(f"数据分割完成 - 训练: {train_data.shape}, 验证: {valid_data.shape}, 测试: {test_data.shape}")

        return train_data, valid_data, test_data

    def create_environments(self, train_data: pd.DataFrame, valid_data: pd.DataFrame) -> None:
        """
        创建训练和评估环境

        Args:
            train_data: 训练数据
            valid_data: 验证数据
        """
        logger.info("创建训练和评估环境...")

        env_config = self.config['environment']

        def make_env(data: pd.DataFrame, rank: int = 0) -> gym.Env:
            """创建环境工厂函数"""
            def _init():
                env = PortfolioEnv(
                    data=data,
                    initial_cash=env_config.get('initial_cash', 1000000),
                    lookback_window=env_config.get('lookback_window', 30),
                    transaction_cost=env_config.get('transaction_cost', 0.003),
                    max_drawdown_threshold=env_config.get('max_drawdown_threshold', 0.15),
                    reward_penalty=env_config.get('reward_penalty', 2.0),
                    features=env_config.get('features')
                )

                # 添加Monitor包装器用于记录
                log_dir = f"logs/env_logs/rank_{rank}"
                os.makedirs(log_dir, exist_ok=True)
                env = Monitor(env, log_dir)

                # 设置随机种子
                env.reset(seed=rank)
                return env

            return _init

        # 创建训练环境
        train_config = self.config['training']
        n_envs = train_config.get('n_envs', 4)

        if n_envs > 1:
            self.train_env = SubprocVecEnv([
                make_env(train_data, i) for i in range(n_envs)
            ])
        else:
            self.train_env = DummyVecEnv([make_env(train_data, 0)])

        # 创建评估环境
        self.eval_env = DummyVecEnv([make_env(valid_data, 999)])

        logger.info(f"环境创建完成 - 训练环境数量: {n_envs}")

    def create_model(self) -> None:
        """创建强化学习模型"""
        logger.info("创建强化学习模型...")

        model_config = self.config['model']
        algorithm = model_config.get('algorithm', 'SAC').upper()

        # 获取环境信息
        sample_env = self.train_env.envs[0] if hasattr(self.train_env, 'envs') else self.train_env.unwrapped

        # 模型参数
        model_kwargs = {
            'learning_rate': model_config.get('learning_rate', 3e-4),
            'batch_size': model_config.get('batch_size', 256),
            'gamma': model_config.get('gamma', 0.99),
            'verbose': 1,
            'tensorboard_log': "tensorboard/",
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # 策略参数
        net_arch = model_config.get('net_arch', [256, 256])
        
        # 如果使用自定义策略
        if model_config.get('use_custom_policy', False):
            env_config = self.config['environment']
            policy_kwargs = {
                'lookback_window': env_config.get('lookback_window', 30),
                'num_stocks': len(sample_env.stock_list) if hasattr(sample_env, 'stock_list') else 50,
                'num_features': len(env_config.get('features', [])) if env_config.get('features') else 5
            }
            policy = TradingPolicy
        else:
            # 标准策略需要正确的net_arch格式
            if algorithm == 'PPO' and isinstance(net_arch, dict):
                policy_kwargs = {'net_arch': net_arch}
            else:
                policy_kwargs = {'net_arch': net_arch if isinstance(net_arch, list) else [256, 256]}
            policy = 'MlpPolicy'

        # 创建模型
        if algorithm == 'SAC':
            model_kwargs.update({
                'buffer_size': model_config.get('buffer_size', 1000000),
                'tau': model_config.get('tau', 0.005),
                'target_update_interval': model_config.get('target_update_interval', 1),
                'learning_starts': model_config.get('learning_starts', 100)
            })
            self.model = SAC(policy, self.train_env, policy_kwargs=policy_kwargs, **model_kwargs)

        elif algorithm == 'PPO':
            model_kwargs.update({
                'n_steps': model_config.get('n_steps', 2048),
                'n_epochs': model_config.get('n_epochs', 10),
                'clip_range': model_config.get('clip_range', 0.2),
                'ent_coef': model_config.get('ent_coef', 0.0)
            })
            self.model = PPO(policy, self.train_env, policy_kwargs=policy_kwargs, **model_kwargs)

        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        logger.info(f"模型创建完成 - 算法: {algorithm}, 设备: {model_kwargs['device']}")

    def setup_callbacks(self) -> List[BaseCallback]:
        """设置训练回调"""
        callbacks = []

        callback_config = self.config.get('callbacks', {})

        # 评估回调
        if callback_config.get('enable_eval', True):
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=f"models/best_model",
                log_path="logs/eval_logs",
                eval_freq=callback_config.get('eval_freq', 10000),
                n_eval_episodes=callback_config.get('n_eval_episodes', 5),
                deterministic=True,
                render=False,
                verbose=1
            )
            callbacks.append(eval_callback)

        # 检查点回调
        if callback_config.get('enable_checkpoint', True):
            checkpoint_callback = CheckpointCallback(
                save_freq=callback_config.get('save_freq', 50000),
                save_path="models/checkpoints",
                name_prefix="rl_model"
            )
            callbacks.append(checkpoint_callback)

        # 回撤早停回调
        if callback_config.get('enable_drawdown_stopping', True):
            drawdown_callback = DrawdownStoppingCallback(
                max_drawdown=callback_config.get('max_training_drawdown', 0.15),
                patience=callback_config.get('drawdown_patience', 100),
                verbose=1
            )
            callbacks.append(drawdown_callback)

        # TensorBoard回调
        if callback_config.get('enable_tensorboard', True):
            tb_callback = TensorBoardCallback(verbose=1)
            callbacks.append(tb_callback)

        return callbacks

    def train(self) -> None:
        """执行训练"""
        logger.info("开始训练...")
        self.training_start_time = datetime.now()

        # 训练配置
        train_config = self.config['training']
        total_timesteps = train_config.get('total_timesteps', 1000000)

        # 设置随机种子
        if 'seed' in train_config:
            set_random_seed(train_config['seed'])

        # 设置回调
        callbacks = self.setup_callbacks()

        try:
            # 开始训练
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=train_config.get('log_interval', 10),
                progress_bar=True
            )

            # 保存最终模型
            final_model_path = f"models/final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.model.save(final_model_path)
            logger.info(f"最终模型已保存: {final_model_path}")

        except KeyboardInterrupt:
            logger.info("训练被用户中断")
            # 保存当前模型
            interrupt_model_path = f"models/interrupted_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.model.save(interrupt_model_path)
            logger.info(f"中断模型已保存: {interrupt_model_path}")

        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            raise RuntimeError(f"训练失败: {e}")

        training_time = datetime.now() - self.training_start_time
        logger.info(f"训练完成，耗时: {training_time}")

    def evaluate(self, model_path: str = None, test_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        评估模型性能

        Args:
            model_path: 模型路径，如果None则使用当前模型
            test_data: 测试数据，如果None则使用验证环境

        Returns:
            评估结果字典
        """
        logger.info("开始模型评估...")

        # 加载模型
        if model_path:
            model_config = self.config['model']
            algorithm = model_config.get('algorithm', 'SAC').upper()

            if algorithm == 'SAC':
                eval_model = SAC.load(model_path)
            elif algorithm == 'PPO':
                eval_model = PPO.load(model_path)
            else:
                raise ValueError(f"不支持的算法: {algorithm}")
        else:
            eval_model = self.model

        # 创建评估环境
        if test_data is not None:
            env_config = self.config['environment']
            eval_env = PortfolioEnv(
                data=test_data,
                initial_cash=env_config.get('initial_cash', 1000000),
                lookback_window=env_config.get('lookback_window', 30),
                transaction_cost=env_config.get('transaction_cost', 0.003),
                max_drawdown_threshold=env_config.get('max_drawdown_threshold', 0.15),
                reward_penalty=env_config.get('reward_penalty', 2.0),
                features=env_config.get('features')
            )
        else:
            eval_env = self.eval_env.envs[0]

        # 执行评估
        n_eval_episodes = self.config.get('evaluation', {}).get('n_episodes', 5)
        episode_rewards = []
        episode_lengths = []

        for episode in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                action, _ = eval_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            logger.info(f"评估回合 {episode + 1}: 奖励 = {episode_reward:.4f}, 长度 = {episode_length}")

        # 获取组合性能
        portfolio_performance = eval_env.get_portfolio_performance()

        # 整理评估结果
        evaluation_results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'portfolio_performance': portfolio_performance
        }

        # 保存评估结果
        results_path = f"results/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        with open(results_path, 'w', encoding='utf-8') as f:
            yaml.dump(evaluation_results, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"评估完成，结果已保存: {results_path}")

        return evaluation_results

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整的训练-评估流水线

        Returns:
            训练和评估结果
        """
        logger.info("开始运行完整训练流水线...")

        try:
            # 1. 初始化数据
            train_data, valid_data, test_data = self.initialize_data()

            # 2. 创建环境
            self.create_environments(train_data, valid_data)

            # 3. 创建模型
            self.create_model()

            # 4. 执行训练
            self.train()

            # 5. 评估模型
            evaluation_results = self.evaluate(test_data=test_data)

            # 6. 生成综合报告
            pipeline_results = {
                'config': self.config,
                'training_time': str(datetime.now() - self.training_start_time),
                'evaluation_results': evaluation_results,
                'data_info': {
                    'train_shape': train_data.shape,
                    'valid_shape': valid_data.shape,
                    'test_shape': test_data.shape
                }
            }

            # 保存完整结果
            results_path = f"results/pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            with open(results_path, 'w', encoding='utf-8') as f:
                yaml.dump(pipeline_results, f, allow_unicode=True, default_flow_style=False)

            logger.info(f"完整流水线执行完成，结果已保存: {results_path}")

            return pipeline_results

        except Exception as e:
            logger.error(f"流水线执行失败: {e}")
            raise RuntimeError(f"流水线执行失败: {e}")

        finally:
            # 清理资源
            if self.train_env:
                self.train_env.close()
            if self.eval_env:
                self.eval_env.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败 {config_path}: {e}")


if __name__ == "__main__":
    # 示例配置
    example_config = {
        'data': {
            'data_root': os.path.expanduser("/Users/shuzhenyi/.qlib/qlib_data"),
            'provider_uri': {
                '1min': os.path.expanduser("/Users/shuzhenyi/.qlib/qlib_data/cn_data_1min"),
                'day': os.path.expanduser("/Users/shuzhenyi/.qlib/qlib_data/cn_data")
            },
            'market': 'csi300',
            'stock_limit': 30,
            'start_time': '2020-01-01',
            'end_time': '2023-12-31',
            'train_start': '2020-01-01',
            'train_end': '2022-12-31',
            'valid_start': '2023-01-01',
            'valid_end': '2023-06-30',
            'test_start': '2023-07-01',
            'test_end': '2023-12-31',
            'freq': 'day',
            'fields': ['$close', '$open', '$high', '$low', '$volume']
        },
        'environment': {
            'initial_cash': 1000000,
            'lookback_window': 30,
            'transaction_cost': 0.003,
            'max_drawdown_threshold': 0.15,
            'reward_penalty': 2.0,
            'features': ['$close', '$open', '$high', '$low', '$volume']
        },
        'model': {
            'algorithm': 'SAC',
            'learning_rate': 3e-4,
            'batch_size': 256,
            'gamma': 0.99,
            'buffer_size': 1000000,
            'tau': 0.005,
            'use_custom_policy': False,
            'net_arch': [256, 256]
        },
        'training': {
            'total_timesteps': 500000,
            'n_envs': 4,
            'seed': 42,
            'log_interval': 10
        },
        'callbacks': {
            'enable_eval': True,
            'eval_freq': 10000,
            'n_eval_episodes': 3,
            'enable_checkpoint': True,
            'save_freq': 50000,
            'enable_drawdown_stopping': True,
            'max_training_drawdown': 0.15,
            'drawdown_patience': 100,
            'enable_tensorboard': True
        },
        'evaluation': {
            'n_episodes': 5
        },
        'log_level': 'INFO'
    }

    # 保存示例配置
    os.makedirs('configs', exist_ok=True)
    with open('configs/example_sac.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(example_config, f, allow_unicode=True, default_flow_style=False)

    print("示例配置已保存到 configs/example_sac.yaml")
    print("训练器类已实现，可通过以下方式使用:")
    print("trainer = RLTrainer(config)")
    print("results = trainer.run_full_pipeline()")