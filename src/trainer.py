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
from env import PortfolioEnv
from model import TradingPolicy, RiskAwareRewardWrapper, PortfolioMetrics

logger = logging.getLogger(__name__)


class TrainingQualityAnalyzer:
    """训练质量分析器，用于评估训练参数、代码、模型质量"""

    def __init__(self):
        self.metrics_history = []
        self.quality_scores = {}

    def analyze_training_stability(self, rewards: List[float], window_size: int = 50) -> Dict[str, float]:
        """
        分析训练稳定性

        Args:
            rewards: 奖励历史
            window_size: 滑动窗口大小（默认50而非100）

        Returns:
            稳定性指标字典
        """
        if len(rewards) < 10:  # 最少需要10个奖励值
            return {
                'stability_score': 0.0,
                'trend_score': 0.0,
                'variance_score': 0.0,
                'slope': 0.0,
                'variance': 0.0
            }

        # 动态调整窗口大小避免过大
        actual_window_size = min(window_size, max(10, len(rewards) // 2))

        # 如果数据不足窗口大小，直接用所有数据
        if len(rewards) < actual_window_size:
            moving_avg = [np.mean(rewards)]
        else:
            # 计算滑动平均
            moving_avg = []
            for i in range(actual_window_size, len(rewards) + 1):
                window_rewards = rewards[i-actual_window_size:i]
                moving_avg.append(np.mean(window_rewards))

        if len(moving_avg) < 2:
            # 如果滑动平均不足2个点，直接分析原始奖励
            x = np.arange(len(rewards))
            slope = np.polyfit(x, rewards, 1)[0] if len(rewards) > 1 else 0.0
            variance = np.var(rewards)
            mean_reward = np.mean(rewards)
        else:
            # 趋势分析 - 线性回归斜率
            x = np.arange(len(moving_avg))
            slope = np.polyfit(x, moving_avg, 1)[0]
            variance = np.var(moving_avg)
            mean_reward = np.mean(moving_avg)

        # 改进的趋势评分：考虑负奖励情况
        if mean_reward < 0:
            # 对于负奖励，斜率为正（趋向0或正值）得高分
            trend_score = max(0, min(1, (slope + 0.1) / 0.2))
        else:
            # 对于正奖励，斜率为正得高分
            trend_score = max(0, min(1, (slope + 0.01) / 0.02))

        # 针对百分点尺度的方差评分：直接用标准差评估
        std_dev = np.sqrt(variance)
        # 对于百分点尺度奖励，标准差超过2.0认为是高方差
        variance_score = max(0, min(1, 1 - std_dev / 3.0))  # 标准差超过3.0得分为0

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
                              portfolio_values: List[float],
                              env_data=None) -> str:
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

        # 组合表现分析 - 需要调试数据来源问题
        if portfolio_values and len(portfolio_values) > 1:
            report.append(f"\n💰 组合表现分析 (DEBUG):")
            report.append(f"  数据点总数: {len(portfolio_values)}")
            report.append(f"  前10个值: {portfolio_values[:10]}")
            report.append(f"  后10个值: {portfolio_values[-10:]}")
            report.append(f"  最小值: {min(portfolio_values):,.0f}")
            report.append(f"  最大值: {max(portfolio_values):,.0f}")
            
            # 等权投资baseline计算
            # 添加调试信息
            report.append(f"\n🔍 环境数据调试信息:")
            if env_data is None:
                report.append(f"  环境数据: None")
            else:
                report.append(f"  环境数据类型: {type(env_data)}")
                report.append(f"  环境属性: {[attr for attr in dir(env_data) if not attr.startswith('_')][:10]}")
                if hasattr(env_data, 'data'):
                    report.append(f"  数据形状: {env_data.data.shape}")
                    report.append(f"  数据列: {list(env_data.data.columns)[:10]}")
                if hasattr(env_data, 'time_index'):
                    report.append(f"  时间索引长度: {len(env_data.time_index)}")
                if hasattr(env_data, 'stock_list'):
                    report.append(f"  股票列表: {env_data.stock_list}")
            
            if not config or 'data' not in config or 'custom_stocks' not in config['data']:
                raise RuntimeError("配置信息不足，无法计算baseline")
                
            stocks = config['data']['custom_stocks']
            num_stocks = len(stocks)
            
            if env_data is not None:
                # 使用环境数据计算更精确的等权baseline
                report.append(f"\n📊 等权投资Baseline:")
                report.append(f"  股票池: {stocks}")
                report.append(f"  股票数量: {num_stocks}")
                
                # 获取环境的历史数据
                has_data = (hasattr(env_data, 'data') and 
                           hasattr(env_data, 'time_index') and 
                           hasattr(env_data, 'stock_list'))
                
                report.append(f"  数据可用性检查: {has_data}")
                if hasattr(env_data, 'data'):
                    report.append(f"  有data属性: True, 形状: {getattr(env_data.data, 'shape', 'N/A')}")
                else:
                    report.append(f"  有data属性: False")
                if hasattr(env_data, 'time_index'):
                    report.append(f"  有time_index属性: True, 长度: {len(getattr(env_data, 'time_index', []))}")
                else:
                    report.append(f"  有time_index属性: False")
                if hasattr(env_data, 'stock_list'):
                    report.append(f"  有stock_list属性: True, 内容: {getattr(env_data, 'stock_list', [])}")
                else:
                    report.append(f"  有stock_list属性: False")
                
                if has_data:
                    # 计算等权组合在相同时间段的表现
                    equal_weight = 1.0 / num_stocks
                    initial_value = portfolio_values[0]
                    
                    # 使用实际股票价格数据计算等权收益
                    start_idx = max(0, len(env_data.time_index) - len(portfolio_values))
                    end_idx = len(env_data.time_index) - 1
                    
                    report.append(f"  时间范围: 索引 {start_idx} -> {end_idx}")
                    
                    if start_idx < end_idx and hasattr(env_data, 'data'):
                        # 检查数据列
                        close_column = None
                        for col in ['$close', 'close', 'Close', '$Close']:
                            if col in env_data.data.columns:
                                close_column = col
                                break
                        
                        if not close_column:
                            raise RuntimeError(f"数据中找不到价格列，可用列: {list(env_data.data.columns)}")
                        
                        # 获取起始和结束价格
                        start_time = env_data.time_index[start_idx]
                        end_time = env_data.time_index[end_idx]
                        
                        report.append(f"  使用价格列: {close_column}")
                        report.append(f"  时间范围: {start_time} -> {end_time}")
                        
                        total_return = 0.0
                        valid_stocks = 0
                        stock_details = []
                        
                        for stock in stocks:
                            if (stock, start_time) not in env_data.data.index:
                                stock_details.append(f"{stock}: 起始时间数据缺失")
                                continue
                            if (stock, end_time) not in env_data.data.index:
                                stock_details.append(f"{stock}: 结束时间数据缺失")
                                continue
                                
                            start_price = env_data.data.loc[(stock, start_time), close_column]
                            end_price = env_data.data.loc[(stock, end_time), close_column]
                            
                            if start_price <= 0 or end_price <= 0:
                                stock_details.append(f"{stock}: 价格无效 ({start_price}->{end_price})")
                                continue
                                
                            stock_return = (end_price / start_price - 1)
                            total_return += stock_return * equal_weight
                            valid_stocks += 1
                            stock_details.append(f"{stock}: {start_price:.2f}->{end_price:.2f} ({stock_return:+.2%})")
                        
                        # 显示股票详情
                        for detail in stock_details:
                            report.append(f"  {detail}")
                        
                        if valid_stocks == 0:
                            raise RuntimeError("没有有效的股票数据")
                        
                        # 重新归一化权重
                        total_return = total_return * num_stocks / valid_stocks
                        baseline_final_value = initial_value * (1 + total_return)
                        baseline_return = total_return * 100
                        
                        report.append(f"  有效股票数: {valid_stocks}/{num_stocks}")
                        report.append(f"  等权策略期末价值: {baseline_final_value:,.0f}")
                        report.append(f"  等权策略总收益率: {baseline_return:+.2f}%")
                    else:
                        raise RuntimeError(f"时间索引范围无效: {start_idx} -> {end_idx}")
                        
                else:
                    # 如果环境数据不可用，使用收益率方法计算baseline
                    if hasattr(env_data, 'return_history') and len(env_data.return_history) > 0:
                        avg_market_return = np.mean(list(env_data.return_history))
                        baseline_final_value = portfolio_values[0] * (1 + avg_market_return * len(portfolio_values))
                        baseline_return = (baseline_final_value / portfolio_values[0] - 1) * 100
                        
                        report.append(f"  等权策略期末价值 (估算): {baseline_final_value:,.0f}")
                        report.append(f"  等权策略总收益率 (估算): {baseline_return:+.2f}%")
                        report.append(f"  (使用环境平均收益估算)")
                    else:
                        # 如果环境数据不可用，使用收益率方法计算baseline
                        report.append(f"\n  使用收益率方法计算等权baseline...")
                        
                        import sys
                        import os
                        sys.path.append('src')
                        import qlib
                        from qlib.data import D
                        
                        # 重新初始化qlib和获取数据
                        if 'data' not in config or 'provider_uri' not in config['data']:
                            raise RuntimeError("配置中缺少数据路径信息")
                            
                        qlib.init(provider_uri=config['data']['provider_uri']['day'], region='cn')
                        
                        # 使用正确的训练时间范围
                        start_time = config['data']['train_start']
                        end_time = config['data']['train_end']
                        
                        # 获取原始价格数据（不使用标准化）
                        data = D.features(
                            instruments=stocks,
                            fields=['$close'],
                            start_time=start_time,
                            end_time=end_time,
                            freq='day'
                        )
                        
                        if data.empty:
                            raise RuntimeError("无法获取股票价格数据")
                            
                        report.append(f"  原始数据形状: {data.shape}")
                        report.append(f"  时间范围: {start_time} -> {end_time}")
                        
                        # 获取时间索引
                        time_index = data.index.get_level_values(1).unique().sort_values()
                        
                        # 根据portfolio_values的长度计算对应的时间段
                        portfolio_steps = len(portfolio_values)
                        total_days = len(time_index)
                        
                        # portfolio_steps是训练步数，不等于交易天数
                        # 需要根据环境的max_steps来计算实际对应的天数
                        if 'environment' in config and 'max_steps' in config['environment']:
                            max_episode_steps = config['environment']['max_steps']
                            # 计算实际使用的交易天数（取较小值）
                            actual_trading_days = min(portfolio_steps, max_episode_steps, total_days)
                        else:
                            # 如果没有配置信息，使用可用天数
                            actual_trading_days = min(portfolio_steps, total_days)
                        
                        if actual_trading_days <= 1:
                            raise RuntimeError(f"计算的交易天数太少: {actual_trading_days}")
                        
                        # 使用最近的actual_trading_days天计算baseline
                        baseline_start_idx = total_days - actual_trading_days
                        baseline_end_idx = total_days - 1
                        
                        baseline_start_time = time_index[baseline_start_idx]
                        baseline_end_time = time_index[baseline_end_idx]
                        
                        report.append(f"  总训练步数: {portfolio_steps}")
                        report.append(f"  可用交易天数: {total_days}")
                        report.append(f"  实际计算天数: {actual_trading_days}")
                        report.append(f"  baseline计算时间段: {baseline_start_time} -> {baseline_end_time}")
                        
                        # 计算每只股票的收益率
                        equal_weight = 1.0 / num_stocks
                        total_return = 0.0
                        valid_stocks = 0
                        
                        for stock in stocks:
                            if (stock, baseline_start_time) not in data.index or (stock, baseline_end_time) not in data.index:
                                report.append(f"  {stock}: 缺少时间点数据")
                                continue
                                
                            start_price = data.loc[(stock, baseline_start_time), '$close']
                            end_price = data.loc[(stock, baseline_end_time), '$close']
                            
                            if start_price <= 0 or end_price <= 0:
                                report.append(f"  {stock}: 价格无效 ({start_price:.2f}->{end_price:.2f})")
                                continue
                            
                            stock_return = (end_price / start_price - 1)
                            total_return += stock_return * equal_weight
                            valid_stocks += 1
                            
                            report.append(f"  {stock}: {start_price:.2f}->{end_price:.2f} ({stock_return:+.2%})")
                        
                        if valid_stocks == 0:
                            raise RuntimeError("没有有效的股票数据用于计算baseline")
                        
                        # 重新归一化权重（如果有股票数据缺失）
                        if valid_stocks < num_stocks:
                            total_return = total_return * num_stocks / valid_stocks
                            report.append(f"  权重归一化: {valid_stocks}/{num_stocks}只股票有效")
                        
                        baseline_final_value = portfolio_values[0] * (1 + total_return)
                        baseline_return = total_return * 100
                        
                        report.append(f"  ✅ 成功计算等权baseline:")
                        report.append(f"  有效股票数: {valid_stocks}/{num_stocks}")
                        report.append(f"  等权策略期末价值: {baseline_final_value:,.0f}")
                        report.append(f"  等权策略总收益率: {baseline_return:+.2f}%")
            else:
                raise RuntimeError("没有可用的环境数据计算baseline")
            
            # 暂时用简单计算查看问题
            if len(portfolio_values) > 1:
                initial_value = portfolio_values[0]
                final_value = portfolio_values[-1]
                total_return = (final_value / initial_value - 1) * 100
                
                report.append(f"\n  === 策略 vs Baseline 对比 ===")
                report.append(f"  初始值: {initial_value:,.0f}")
                report.append(f"  最终值: {final_value:,.0f}")
                report.append(f"  策略总收益率: {total_return:+.2f}%")
                
                # 与baseline对比
                if 'baseline_return' in locals():
                    excess_return = total_return - baseline_return
                    report.append(f"  超额收益 (vs 等权): {excess_return:+.2f}%")
                    if excess_return > 0:
                        report.append(f"  ✅ 策略跑赢等权baseline")
                    else:
                        report.append(f"  ❌ 策略跑输等权baseline")
                
                # 简单回撤计算来找问题
                peak = portfolio_values[0]
                max_drawdown = 0
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                        
                report.append(f"  最大回撤: {max_drawdown:.2%}")
                report.append(f"  峰值: {peak:,.0f}")
                
                # 找出导致最大回撤的值
                for i, value in enumerate(portfolio_values):
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    if abs(drawdown - max_drawdown) < 0.001:
                        report.append(f"  最大回撤发生在索引{i}: 峰值{peak:,.0f} -> 当前{value:,.0f}")
                        break

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
    """多尺度回撤早停回调，支持梯度递增耐心和模型快照"""

    def __init__(self, max_drawdown: float = 0.20, base_patience: int = 100, 
                 warmup_steps: int = 50000, verbose: int = 0):
        super().__init__(verbose)
        self.max_drawdown = max_drawdown
        self.base_patience = base_patience
        self.warmup_steps = warmup_steps
        
        # 多尺度检测窗口
        self.short_window = 20   # 短期检测
        self.medium_window = 100 # 中期检测
        self.long_window = 200   # 长期检测
        
        # 多尺度历史记录
        self.short_history = []
        self.medium_history = []
        self.long_history = []
        
        # 违规计数器
        self.short_violations = 0
        self.medium_violations = 0
        self.long_violations = 0
        
        # 梯度递增耐心机制
        self.current_patience = base_patience
        self.consecutive_violations = 0
        
        # 最佳状态快照
        self.best_portfolio_value = 0
        self.best_model_step = 0

    def _on_step(self) -> bool:
        """多尺度回撤检测和梯度递增早停"""
        # 预热期间不执行早停
        if self.num_timesteps < self.warmup_steps:
            return True
            
        # 获取环境信息
        if not hasattr(self.training_env, 'get_attr'):
            return True
            
        current_drawdowns = self.training_env.get_attr('current_drawdown')
        portfolio_values = self.training_env.get_attr('total_value')
        
        if not current_drawdowns or not portfolio_values:
            return True
            
        # 计算平均指标
        avg_drawdown = np.mean(current_drawdowns)
        avg_portfolio_value = np.mean(portfolio_values)
        
        # 更新最佳状态
        if avg_portfolio_value > self.best_portfolio_value:
            self.best_portfolio_value = avg_portfolio_value
            self.best_model_step = self.num_timesteps
        
        # 多尺度历史更新
        self._update_histories(avg_drawdown)
        
        # 多尺度检测
        violation_detected = self._multi_scale_detection()
        
        if violation_detected:
            self.consecutive_violations += 1
            # 梯度递增耐心：连续违规时指数减少耐心
            patience_factor = max(0.1, 0.8 ** (self.consecutive_violations // 10))
            current_patience = int(self.base_patience * patience_factor)
            
            if self.consecutive_violations >= current_patience:
                if self.verbose > 0:
                    short_dd = np.mean(self.short_history) if self.short_history else 0
                    medium_dd = np.mean(self.medium_history) if self.medium_history else 0
                    long_dd = np.mean(self.long_history) if self.long_history else 0
                    
                    logger.warning(f"多尺度回撤超限触发早停:")
                    logger.warning(f"  短期回撤({self.short_window}步): {short_dd:.2%}")
                    logger.warning(f"  中期回撤({self.medium_window}步): {medium_dd:.2%}")
                    logger.warning(f"  长期回撤({self.long_window}步): {long_dd:.2%}")
                    logger.warning(f"  连续违规: {self.consecutive_violations}步")
                    logger.warning(f"  当前耐心: {current_patience}")
                
                return False
        else:
            # 违规缓解时重置计数器
            self.consecutive_violations = max(0, self.consecutive_violations - 2)
        
        return True
    
    def _update_histories(self, avg_drawdown: float):
        """更新多尺度历史记录"""
        self.short_history.append(avg_drawdown)
        self.medium_history.append(avg_drawdown)
        self.long_history.append(avg_drawdown)
        
        # 维持窗口大小
        if len(self.short_history) > self.short_window:
            self.short_history.pop(0)
        if len(self.medium_history) > self.medium_window:
            self.medium_history.pop(0)
        if len(self.long_history) > self.long_window:
            self.long_history.pop(0)
    
    def _multi_scale_detection(self) -> bool:
        """多尺度违规检测"""
        violation = False
        
        # 短期检测：快速响应
        if len(self.short_history) >= 10:
            short_avg = np.mean(self.short_history)
            if short_avg > self.max_drawdown * 1.2:  # 短期阈值更严格
                self.short_violations += 1
                violation = True
            else:
                self.short_violations = max(0, self.short_violations - 1)
        
        # 中期检测：平衡检测
        if len(self.medium_history) >= 30:
            medium_avg = np.mean(self.medium_history)
            if medium_avg > self.max_drawdown:
                self.medium_violations += 1
                violation = True
            else:
                self.medium_violations = max(0, self.medium_violations - 1)
        
        # 长期检测：趋势确认
        if len(self.long_history) >= 50:
            long_avg = np.mean(self.long_history)
            if long_avg > self.max_drawdown * 0.8:  # 长期阈值稍微宽松
                self.long_violations += 1
                violation = True
            else:
                self.long_violations = max(0, self.long_violations - 1)
        
        # 综合判断：任一尺度持续违规即为违规
        return (self.short_violations >= 5 or 
                self.medium_violations >= 10 or 
                self.long_violations >= 20)


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

        # 指标历史记录 - 统一的portfolio_values_history用于一致的质量报告
        self.episode_rewards = []
        self.episode_lengths = []
        self.portfolio_values_history = []  # 统一的组合价值历史
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
        """收集环境指标，包括CVaR等新风险指标"""
        if not hasattr(self.training_env, 'get_attr'):
            return

        # 获取基础环境状态
        total_values = self.training_env.get_attr('total_value')
        drawdowns = self.training_env.get_attr('current_drawdown')
        
        # 获取新增的风险指标
        cvar_values = self.training_env.get_attr('cvar_value')
        rolling_peaks = self.training_env.get_attr('rolling_peak')

        if len(total_values) > 0:
            # 改进：只记录第一个环境的数据，避免多环境混淆
            # 这样可以得到一个连续的组合价值序列
            self.portfolio_values_history.append(total_values[0])  # 只取第一个环境
            self.drawdowns.append(drawdowns[0])  # 只取第一个环境

            # 基础指标记录到TensorBoard
            self.logger.record('env/mean_total_value', np.mean(total_values))
            self.logger.record('env/max_total_value', np.max(total_values))
            self.logger.record('env/mean_drawdown', np.mean(drawdowns))
            self.logger.record('env/max_drawdown', np.max(drawdowns))
            
            # 新增风险指标
            if cvar_values:
                self.logger.record('risk/mean_cvar', np.mean(cvar_values))
                self.logger.record('risk/max_cvar', np.max(cvar_values))
            
            if rolling_peaks:
                self.logger.record('env/mean_rolling_peak', np.mean(rolling_peaks))
                
            # 计算Calmar比率（年化收益/最大回撤）
            if len(self.portfolio_values_history) > 252:  # 至少一年数据
                recent_values = self.portfolio_values_history[-252:]
                annual_return = (recent_values[-1] / recent_values[0]) - 1
                max_dd_period = np.max(self.drawdowns[-252:]) if len(self.drawdowns) >= 252 else np.max(self.drawdowns)
                if max_dd_period > 0:
                    calmar_ratio = annual_return / max_dd_period
                    self.logger.record('performance/calmar_ratio', calmar_ratio)

            # 更新最佳指标
            current_max_value = np.max(total_values)
            current_worst_drawdown = np.max(drawdowns)

            if current_max_value > self.best_portfolio_value:
                self.best_portfolio_value = current_max_value

            if current_worst_drawdown > self.worst_drawdown:
                self.worst_drawdown = current_worst_drawdown

    def _log_basic_metrics(self, current_step: int):
        """输出基础训练指标"""
        if self.verbose < 1:
            return

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
            if self.portfolio_values_history:
                recent_values = self.portfolio_values_history[-100:]  # 最近100个值
                current_value = recent_values[-1] if recent_values else 0
                value_change = ((current_value / recent_values[0]) - 1) * 100 if len(recent_values) > 1 else 0

                logger.info(f"[步骤 {current_step:,}] "
                           f"当前组合价值: {current_value:,.0f} | "
                           f"价值变化: {value_change:+.2f}% | "
                           f"最大回撤: {self.worst_drawdown:.2%}")

    def _log_detailed_metrics(self, current_step: int):
        """输出详细训练指标"""
        if self.verbose < 1:
            return

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
        if self.portfolio_values_history:
            recent_values = self.portfolio_values_history[-1000:]  # 最近1000个值
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

    def _log_quality_analysis(self, current_step: int):
        """输出训练质量分析"""
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
        if len(rewards_history) > 50 and len(self.portfolio_values_history) > 50:
            # 获取配置信息（需要从父类传递）
            config = getattr(self, 'config', {})

            # 尝试获取环境数据进行调试
            env_data = None
            debug_info = []
            
            debug_info.append(f"training_env类型: {type(self.training_env)}")
            debug_info.append(f"training_env属性: {[attr for attr in dir(self.training_env) if not attr.startswith('_')][:15]}")
            
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                # VecEnv情况
                env_data = self.training_env.envs[0]
                debug_info.append(f"从VecEnv获取: {type(env_data)}")
            elif hasattr(self.training_env, 'env'):
                # Monitor包装的情况
                env_data = self.training_env.env
                debug_info.append(f"从Monitor获取: {type(env_data)}")
            else:
                # 直接环境
                env_data = self.training_env
                debug_info.append(f"直接环境: {type(env_data)}")
                
            # 进一步解包Monitor
            if hasattr(env_data, 'env'):
                env_data = env_data.env
                debug_info.append(f"进一步解包: {type(env_data)}")
                
            # 尝试通过get_attr获取环境数据
            if env_data is None and hasattr(self.training_env, 'get_attr'):
                env_attrs = self.training_env.get_attr('data')
                if len(env_attrs) > 0:
                    # 创建一个临时对象来存储数据
                    class TempEnv:
                        def __init__(self):
                            self.data = env_attrs[0]
                            self.time_index = self.training_env.get_attr('time_index')[0]
                            self.stock_list = self.training_env.get_attr('stock_list')[0]
                    env_data = TempEnv()
                    debug_info.append(f"通过get_attr创建临时环境")

            # 将调试信息加入报告
            for info in debug_info:
                logger.info(f"环境调试: {info}")

            quality_report = self.quality_analyzer.generate_quality_report(
                rewards=rewards_history,
                losses=self.losses_history,
                config=config,
                portfolio_values=self.portfolio_values_history,  # 使用统一历史记录
                env_data=env_data
            )

            logger.info("\n" + quality_report)

            # 保存质量报告到文件
            report_path = f"logs/quality_report_{current_step}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(quality_report)

    def set_config(self, config: Dict[str, Any]):
        """设置配置信息供质量分析使用"""
        self.config = config

    def _on_training_end(self) -> None:
        """训练结束时的总结"""
        logger.info("=" * 80)
        logger.info("训练完成 - 最终统计")
        logger.info("-" * 80)

        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            all_episodes = list(self.model.ep_info_buffer)
            all_rewards = [ep['r'] for ep in all_episodes]
            all_lengths = [ep['l'] for ep in all_episodes]

            logger.info(f"总回合数: {len(all_episodes)}")
            logger.info(f"平均奖励: {np.mean(all_rewards):.4f}")
            logger.info(f"最佳奖励: {np.max(all_rewards):.4f}")
            logger.info(f"最差奖励: {np.min(all_rewards):.4f}")
            logger.info(f"奖励标准差: {np.std(all_rewards):.4f}")

        if self.portfolio_values_history:
            logger.info(f"最终组合价值: {self.portfolio_values_history[-1]:,.0f}")
            logger.info(f"最佳组合价值: {self.best_portfolio_value:,.0f}")

        if self.drawdowns:
            logger.info(f"最大回撤: {self.worst_drawdown:.2%}")

        logger.info("=" * 80)


class TensorBoardCallback(BaseCallback):
    """自定义TensorBoard回调，记录额外指标"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """记录自定义指标"""
        if hasattr(self.training_env, 'get_attr'):
            # 获取环境指标
            total_values = self.training_env.get_attr('total_value')
            drawdowns = self.training_env.get_attr('current_drawdown')

            # 记录到TensorBoard
            if len(total_values) > 0:
                self.logger.record('env/mean_total_value', np.mean(total_values))
                self.logger.record('env/max_total_value', np.max(total_values))
                self.logger.record('env/mean_drawdown', np.mean(drawdowns))
                self.logger.record('env/max_drawdown', np.max(drawdowns))

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
        custom_stocks = data_config.get('custom_stocks', None)
        stock_list = self.data_loader.get_stock_list(market, stock_limit, custom_stocks)

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
                    features=env_config.get('features'),
                    max_steps=env_config.get('max_steps')
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

        # 模型参数 - 移除GPU学习率放大逻辑
        learning_rate = model_config.get('learning_rate', 3e-4)
        logger.info(f"使用学习率: {learning_rate}")

        model_kwargs = {
            'learning_rate': learning_rate,  # 不再人为放大学习率
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
                'learning_starts': model_config.get('learning_starts', 100),
                'ent_coef': model_config.get('ent_coef', 'auto')  # 添加熵系数支持
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
                max_drawdown=callback_config.get('max_training_drawdown', 0.20),
                base_patience=callback_config.get('drawdown_base_patience', 100),
                warmup_steps=callback_config.get('drawdown_warmup_steps', 50000),
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
                features=env_config.get('features'),
                max_steps=env_config.get('max_steps')
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
