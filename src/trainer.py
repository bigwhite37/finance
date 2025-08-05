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


class TrainingQualityAnalyzer:
    """训练质量分析器，用于评估训练参数、代码、模型质量"""
    
    def __init__(self):
        self.metrics_history = []
        self.quality_scores = {}
        
    def analyze_training_stability(self, rewards: List[float], window_size: int = 100) -> Dict[str, float]:
        """
        分析训练稳定性
        
        Args:
            rewards: 奖励历史
            window_size: 滑动窗口大小
            
        Returns:
            稳定性指标字典
        """
        if len(rewards) < window_size:
            return {
                'stability_score': 0.0, 
                'trend_score': 0.0, 
                'variance_score': 0.0,
                'slope': 0.0,
                'variance': 0.0
            }
        
        # 计算滑动平均
        moving_avg = []
        for i in range(window_size, len(rewards)):
            window_rewards = rewards[i-window_size:i]
            moving_avg.append(np.mean(window_rewards))
        
        if len(moving_avg) < 2:
            return {
                'stability_score': 0.0, 
                'trend_score': 0.0, 
                'variance_score': 0.0,
                'slope': 0.0,
                'variance': 0.0
            }
        
        # 趋势分析 - 线性回归斜率
        x = np.arange(len(moving_avg))
        slope = np.polyfit(x, moving_avg, 1)[0]
        trend_score = max(0, min(1, (slope + 0.01) / 0.02))  # 归一化到[0,1]
        
        # 方差分析
        variance = np.var(moving_avg)
        variance_score = max(0, min(1, 1 - variance / (np.mean(moving_avg) ** 2 + 1e-8)))
        
        # 综合稳定性得分
        stability_score = 0.6 * trend_score + 0.4 * variance_score
        
        return {
            'stability_score': stability_score,
            'trend_score': trend_score,
            'variance_score': variance_score,
            'slope': slope,
            'variance': variance
        }
    
    def analyze_convergence_quality(self, losses: List[float]) -> Dict[str, float]:
        """
        分析收敛质量
        
        Args:
            losses: 损失历史
            
        Returns:
            收敛质量指标
        """
        if len(losses) < 100:
            return {
                'convergence_score': 0.0, 
                'oscillation_score': 0.0,
                'improvement_ratio': 0.0,
                'oscillation_ratio': 0.0
            }
        
        # 计算损失下降趋势
        recent_losses = losses[-100:]
        early_losses = losses[:100] if len(losses) >= 200 else losses[:len(losses)//2]
        
        improvement = (np.mean(early_losses) - np.mean(recent_losses)) / (np.mean(early_losses) + 1e-8)
        convergence_score = max(0, min(1, improvement))
        
        # 分析震荡程度
        loss_diff = np.diff(recent_losses)
        oscillation = np.std(loss_diff) / (np.mean(np.abs(loss_diff)) + 1e-8)
        oscillation_score = max(0, min(1, 1 - oscillation / 10))
        
        return {
            'convergence_score': convergence_score,
            'oscillation_score': oscillation_score,
            'improvement_ratio': improvement,
            'oscillation_ratio': oscillation
        }
    
    def analyze_hyperparameter_quality(self, config: Dict[str, Any]) -> Dict[str, float]:
        """
        分析超参数质量
        
        Args:
            config: 配置字典
            
        Returns:
            超参数质量评分
        """
        scores = {}
        
        # 学习率评估
        lr = config.get('model', {}).get('learning_rate', 3e-4)
        if 1e-5 <= lr <= 1e-2:
            scores['learning_rate_score'] = 1.0
        elif 1e-6 <= lr <= 1e-1:
            scores['learning_rate_score'] = 0.7
        else:
            scores['learning_rate_score'] = 0.3
        
        # 批次大小评估
        batch_size = config.get('model', {}).get('batch_size', 256)
        if 64 <= batch_size <= 512:
            scores['batch_size_score'] = 1.0
        elif 32 <= batch_size <= 1024:
            scores['batch_size_score'] = 0.8
        else:
            scores['batch_size_score'] = 0.5
        
        # 网络架构评估
        net_arch = config.get('model', {}).get('net_arch', [256, 256])
        if isinstance(net_arch, list) and 2 <= len(net_arch) <= 4:
            if all(64 <= size <= 512 for size in net_arch):
                scores['network_arch_score'] = 1.0
            else:
                scores['network_arch_score'] = 0.7
        else:
            scores['network_arch_score'] = 0.5
        
        # 环境参数评估
        env_config = config.get('environment', {})
        transaction_cost = env_config.get('transaction_cost', 0.003)
        if 0.001 <= transaction_cost <= 0.01:
            scores['transaction_cost_score'] = 1.0
        else:
            scores['transaction_cost_score'] = 0.6
        
        # 综合评分
        scores['overall_hyperparameter_score'] = np.mean(list(scores.values()))
        
        return scores
    
    def generate_quality_report(self, 
                              rewards: List[float],
                              losses: List[float],
                              config: Dict[str, Any],
                              portfolio_values: List[float]) -> str:
        """
        生成训练质量报告
        
        Args:
            rewards: 奖励历史
            losses: 损失历史  
            config: 配置
            portfolio_values: 组合价值历史
            
        Returns:
            质量报告字符串
        """
        report = []
        report.append("=" * 80)
        report.append("训练质量分析报告")
        report.append("=" * 80)
        
        # 稳定性分析
        if rewards:
            stability = self.analyze_training_stability(rewards)
            report.append(f"\n📊 训练稳定性分析:")
            report.append(f"  综合稳定性得分: {stability['stability_score']:.3f}")
            report.append(f"  趋势得分: {stability['trend_score']:.3f}")
            report.append(f"  方差得分: {stability['variance_score']:.3f}")
            report.append(f"  奖励趋势斜率: {stability['slope']:.6f}")
            
            if stability['stability_score'] > 0.8:
                report.append("  ✅ 训练稳定性良好")
            elif stability['stability_score'] > 0.6:
                report.append("  ⚠️  训练稳定性一般，建议调整学习率")
            else:
                report.append("  ❌ 训练不稳定，建议检查超参数设置")
        
        # 收敛质量分析
        if losses:
            convergence = self.analyze_convergence_quality(losses)
            report.append(f"\n🎯 收敛质量分析:")
            report.append(f"  收敛得分: {convergence['convergence_score']:.3f}")
            report.append(f"  震荡得分: {convergence['oscillation_score']:.3f}")
            report.append(f"  改进比率: {convergence['improvement_ratio']:.3f}")
            
            if convergence['convergence_score'] > 0.7:
                report.append("  ✅ 模型收敛良好")
            elif convergence['convergence_score'] > 0.4:
                report.append("  ⚠️  收敛速度较慢，可考虑调整学习率")
            else:
                report.append("  ❌ 收敛困难，建议检查模型架构或数据质量")
        
        # 超参数质量分析
        hyperparams = self.analyze_hyperparameter_quality(config)
        report.append(f"\n⚙️  超参数质量分析:")
        report.append(f"  学习率得分: {hyperparams['learning_rate_score']:.3f}")
        report.append(f"  批次大小得分: {hyperparams['batch_size_score']:.3f}")
        report.append(f"  网络架构得分: {hyperparams['network_arch_score']:.3f}")
        report.append(f"  交易成本得分: {hyperparams['transaction_cost_score']:.3f}")
        report.append(f"  综合超参数得分: {hyperparams['overall_hyperparameter_score']:.3f}")
        
        # 组合表现分析
        if portfolio_values and len(portfolio_values) > 1:
            initial_value = portfolio_values[0]
            final_value = portfolio_values[-1]
            total_return = (final_value / initial_value - 1) * 100
            
            # 计算最大回撤
            peak = portfolio_values[0]
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            report.append(f"\n💰 组合表现分析:")
            report.append(f"  总收益率: {total_return:+.2f}%")
            report.append(f"  最大回撤: {max_drawdown:.2%}")
            
            # 风险调整收益
            if max_drawdown > 0:
                risk_adjusted_return = total_return / (max_drawdown * 100)
                report.append(f"  风险调整收益: {risk_adjusted_return:.3f}")
                
                if risk_adjusted_return > 2.0:
                    report.append("  ✅ 风险调整收益优秀")
                elif risk_adjusted_return > 1.0:
                    report.append("  ⚠️  风险调整收益一般")
                else:
                    report.append("  ❌ 风险调整收益较差，需要优化风险控制")
        
        # 总体建议
        report.append(f"\n🔍 总体建议:")
        
        if rewards and len(rewards) > 100:
            recent_performance = np.mean(rewards[-100:])
            early_performance = np.mean(rewards[:100])
            
            if recent_performance > early_performance * 1.1:
                report.append("  ✅ 模型持续改进，训练效果良好")
            elif recent_performance > early_performance * 0.9:
                report.append("  ⚠️  模型性能趋于稳定，可考虑调整探索策略")
            else:
                report.append("  ❌ 模型性能下降，建议检查过拟合或数据泄露问题")
        
        report.append("=" * 80)
        
        return "\n".join(report)


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


class TrainingMetricsCallback(BaseCallback):
    """训练指标监控回调，定期输出训练相关指标"""

    def __init__(self, 
                 log_interval: int = 1000,
                 eval_interval: int = 10000,
                 verbose: int = 1):
        """
        初始化训练指标回调
        
        Args:
            log_interval: 指标输出间隔（timesteps）
            eval_interval: 详细评估间隔（timesteps）
            verbose: 详细程度
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # 指标历史记录
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values = []
        self.drawdowns = []
        self.actions_history = []
        
        # 统计信息
        self.last_log_step = 0
        self.last_eval_step = 0
        self.episode_count = 0
        
        # 性能指标
        self.best_mean_reward = -np.inf
        self.best_portfolio_value = 0
        self.worst_drawdown = 0
        
        # 质量分析器
        self.quality_analyzer = TrainingQualityAnalyzer()
        self.losses_history = []
        
        logger.info(f"训练指标监控已启用 - 日志间隔: {log_interval}, 评估间隔: {eval_interval}")

    def _on_step(self) -> bool:
        """每步执行的监控逻辑"""
        current_step = self.num_timesteps
        
        # 收集环境指标
        self._collect_env_metrics()
        
        # 定期输出基础指标
        if current_step - self.last_log_step >= self.log_interval:
            self._log_basic_metrics(current_step)
            self.last_log_step = current_step
        
        # 定期进行详细评估
        if current_step - self.last_eval_step >= self.eval_interval:
            self._log_detailed_metrics(current_step)
            self._log_quality_analysis(current_step)
            self.last_eval_step = current_step
        
        return True

    def _collect_env_metrics(self):
        """收集环境指标"""
        if not hasattr(self.training_env, 'get_attr'):
            return
            
        try:
            # 获取环境状态
            total_values = self.training_env.get_attr('total_value')
            drawdowns = self.training_env.get_attr('current_drawdown')
            
            if len(total_values) > 0:
                self.portfolio_values.extend(total_values)
                self.drawdowns.extend(drawdowns)
                
                # 记录到TensorBoard
                self.logger.record('env/mean_total_value', np.mean(total_values))
                self.logger.record('env/max_total_value', np.max(total_values))
                self.logger.record('env/mean_drawdown', np.mean(drawdowns))
                self.logger.record('env/max_drawdown', np.max(drawdowns))
                
                # 更新最佳指标
                current_max_value = np.max(total_values)
                current_worst_drawdown = np.max(drawdowns)
                
                if current_max_value > self.best_portfolio_value:
                    self.best_portfolio_value = current_max_value
                    
                if current_worst_drawdown > self.worst_drawdown:
                    self.worst_drawdown = current_worst_drawdown
                    
        except Exception as e:
            if self.verbose > 1:
                logger.warning(f"收集环境指标失败: {e}")

    def _log_basic_metrics(self, current_step: int):
        """输出基础训练指标"""
        if self.verbose < 1:
            return
            
        try:
            # 获取最近的奖励信息
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                recent_episodes = list(self.model.ep_info_buffer)[-10:]  # 最近10个回合
                recent_rewards = [ep['r'] for ep in recent_episodes]
                recent_lengths = [ep['l'] for ep in recent_episodes]
                
                mean_reward = np.mean(recent_rewards)
                mean_length = np.mean(recent_lengths)
                
                # 更新最佳奖励
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                
                # 输出基础指标
                logger.info(f"[步骤 {current_step:,}] "
                           f"平均奖励: {mean_reward:.4f} | "
                           f"平均回合长度: {mean_length:.1f} | "
                           f"最佳奖励: {self.best_mean_reward:.4f}")
                
                # 如果有组合价值数据
                if self.portfolio_values:
                    recent_values = self.portfolio_values[-100:]  # 最近100个值
                    current_value = recent_values[-1] if recent_values else 0
                    value_change = ((current_value / recent_values[0]) - 1) * 100 if len(recent_values) > 1 else 0
                    
                    logger.info(f"[步骤 {current_step:,}] "
                               f"当前组合价值: {current_value:,.0f} | "
                               f"价值变化: {value_change:+.2f}% | "
                               f"最大回撤: {self.worst_drawdown:.2%}")
                
        except Exception as e:
            logger.warning(f"输出基础指标失败: {e}")

    def _log_detailed_metrics(self, current_step: int):
        """输出详细训练指标"""
        if self.verbose < 1:
            return
            
        try:
            logger.info("=" * 80)
            logger.info(f"详细训练报告 - 步骤 {current_step:,}")
            logger.info("-" * 80)
            
            # 模型学习率等参数
            if hasattr(self.model, 'learning_rate'):
                current_lr = self.model.learning_rate
                if callable(current_lr):
                    current_lr = current_lr(1.0)  # 获取当前学习率
                logger.info(f"当前学习率: {current_lr:.2e}")
            
            # 回合统计
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                episodes = list(self.model.ep_info_buffer)
                if len(episodes) >= 10:
                    rewards = [ep['r'] for ep in episodes[-50:]]  # 最近50个回合
                    lengths = [ep['l'] for ep in episodes[-50:]]
                    
                    logger.info(f"回合统计 (最近50回合):")
                    logger.info(f"  平均奖励: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
                    logger.info(f"  奖励范围: [{np.min(rewards):.4f}, {np.max(rewards):.4f}]")
                    logger.info(f"  平均长度: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            
            # 组合性能统计
            if self.portfolio_values:
                recent_values = self.portfolio_values[-1000:]  # 最近1000个值
                if len(recent_values) > 1:
                    initial_value = recent_values[0]
                    current_value = recent_values[-1]
                    total_return = (current_value / initial_value - 1) * 100
                    
                    # 计算波动率
                    returns = np.diff(recent_values) / recent_values[:-1]
                    volatility = np.std(returns) * np.sqrt(252) * 100  # 年化波动率
                    
                    logger.info(f"组合性能统计:")
                    logger.info(f"  初始价值: {initial_value:,.0f}")
                    logger.info(f"  当前价值: {current_value:,.0f}")
                    logger.info(f"  总收益率: {total_return:+.2f}%")
                    logger.info(f"  年化波动率: {volatility:.2f}%")
                    logger.info(f"  最大价值: {self.best_portfolio_value:,.0f}")
            
            # 回撤统计
            if self.drawdowns:
                recent_drawdowns = self.drawdowns[-1000:]
                logger.info(f"回撤统计:")
                logger.info(f"  当前回撤: {recent_drawdowns[-1]:.2%}")
                logger.info(f"  平均回撤: {np.mean(recent_drawdowns):.2%}")
                logger.info(f"  最大回撤: {np.max(recent_drawdowns):.2%}")
            
            # 训练稳定性指标
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                metrics = self.model.logger.name_to_value
                if 'train/loss' in metrics:
                    logger.info(f"训练损失: {metrics['train/loss']:.6f}")
                if 'train/policy_gradient_loss' in metrics:
                    logger.info(f"策略梯度损失: {metrics['train/policy_gradient_loss']:.6f}")
                if 'train/value_loss' in metrics:
                    logger.info(f"价值函数损失: {metrics['train/value_loss']:.6f}")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.warning(f"输出详细指标失败: {e}")

    def _log_quality_analysis(self, current_step: int):
        """输出训练质量分析"""
        try:
            # 收集训练损失
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                metrics = self.model.logger.name_to_value
                if 'train/loss' in metrics:
                    self.losses_history.append(metrics['train/loss'])
            
            # 收集奖励历史
            rewards_history = []
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                episodes = list(self.model.ep_info_buffer)
                rewards_history = [ep['r'] for ep in episodes]
            
            # 生成质量报告
            if len(rewards_history) > 50 and len(self.portfolio_values) > 50:
                # 获取配置信息（需要从父类传递）
                config = getattr(self, 'config', {})
                
                quality_report = self.quality_analyzer.generate_quality_report(
                    rewards=rewards_history,
                    losses=self.losses_history,
                    config=config,
                    portfolio_values=self.portfolio_values
                )
                
                logger.info("\n" + quality_report)
                
                # 保存质量报告到文件
                report_path = f"logs/quality_report_{current_step}.txt"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(quality_report)
                    
        except Exception as e:
            logger.warning(f"生成质量分析失败: {e}")

    def set_config(self, config: Dict[str, Any]):
        """设置配置信息供质量分析使用"""
        self.config = config

    def _on_training_end(self) -> None:
        """训练结束时的总结"""
        logger.info("=" * 80)
        logger.info("训练完成 - 最终统计")
        logger.info("-" * 80)
        
        try:
            if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
                all_episodes = list(self.model.ep_info_buffer)
                all_rewards = [ep['r'] for ep in all_episodes]
                all_lengths = [ep['l'] for ep in all_episodes]
                
                logger.info(f"总回合数: {len(all_episodes)}")
                logger.info(f"平均奖励: {np.mean(all_rewards):.4f}")
                logger.info(f"最佳奖励: {np.max(all_rewards):.4f}")
                logger.info(f"最差奖励: {np.min(all_rewards):.4f}")
                logger.info(f"奖励标准差: {np.std(all_rewards):.4f}")
                
            if self.portfolio_values:
                logger.info(f"最终组合价值: {self.portfolio_values[-1]:,.0f}")
                logger.info(f"最佳组合价值: {self.best_portfolio_value:,.0f}")
                
            if self.drawdowns:
                logger.info(f"最大回撤: {self.worst_drawdown:.2%}")
                
        except Exception as e:
            logger.warning(f"输出最终统计失败: {e}")
            
        logger.info("=" * 80)


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

        # 训练指标监控回调 - 放在最前面以确保及时监控
        if callback_config.get('enable_training_metrics', True):
            metrics_callback = TrainingMetricsCallback(
                log_interval=callback_config.get('metrics_log_interval', 1000),
                eval_interval=callback_config.get('metrics_eval_interval', 10000),
                verbose=1
            )
            # 传递配置信息给回调
            metrics_callback.set_config(self.config)
            callbacks.append(metrics_callback)

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
            'enable_training_metrics': True,
            'metrics_log_interval': 1000,      # 每1000步输出基础指标
            'metrics_eval_interval': 10000,    # 每10000步输出详细指标
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