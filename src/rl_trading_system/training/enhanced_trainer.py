"""
增强的强化学习训练器

在原有训练器基础上添加详细的投资组合指标、智能体行为分析和风险控制指标。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .trainer import RLTrainer, TrainingConfig
from ..metrics.portfolio_metrics import (
    PortfolioMetricsCalculator,
    PortfolioMetrics,
    AgentBehaviorMetrics,
    RiskControlMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainingConfig(TrainingConfig):
    """增强训练配置"""
    # 指标计算开关
    enable_portfolio_metrics: bool = True          # 启用投资组合指标计算
    enable_agent_behavior_metrics: bool = True     # 启用智能体行为指标计算
    enable_risk_control_metrics: bool = True       # 启用风险控制指标计算
    
    # 指标计算频率
    metrics_calculation_frequency: int = 10        # 每N个episode计算一次指标
    
    # 基准数据配置
    benchmark_data_path: Optional[str] = None      # 基准数据路径
    risk_free_rate: float = 0.03                   # 无风险利率
    
    # 环境配置（用于指标计算的默认值）
    initial_cash: float = 1000000.0                # 初始资金（用于指标计算默认值）
    
    # 日志配置
    detailed_metrics_logging: bool = True          # 详细指标日志
    metrics_log_level: str = 'INFO'                # 指标日志级别
    
    def __post_init__(self):
        """配置验证"""
        super().__post_init__()
        
        if self.metrics_calculation_frequency <= 0:
            raise ValueError("metrics_calculation_frequency必须为正数")
        
        if self.enable_portfolio_metrics and self.benchmark_data_path == "":
            raise ValueError("启用投资组合指标时，benchmark_data_path不能为空")
        
        if self.risk_free_rate < 0:
            raise ValueError("risk_free_rate不能为负数")


class EnhancedRLTrainer(RLTrainer):
    """
    增强的强化学习训练器
    
    在原有训练器基础上添加：
    1. 投资组合与市场表现对比指标（夏普比率、最大回撤、Alpha、Beta、年化收益率）
    2. 智能体行为分析指标（熵、平均持仓权重、换手率）
    3. 风险与回撤控制模块的详细日志
    """
    
    def __init__(self, config: EnhancedTrainingConfig, environment, agent, data_split):
        """
        初始化增强训练器
        
        Args:
            config: 增强训练配置
            environment: 交易环境
            agent: 强化学习智能体
            data_split: 数据划分结果
        """
        super().__init__(config, environment, agent, data_split)
        
        # 指标计算器
        self.metrics_calculator = PortfolioMetricsCalculator()
        
        # 历史数据存储
        self.portfolio_values_history: List[float] = []
        self.benchmark_values_history: List[float] = []
        self.dates_history: List[datetime] = []
        
        # 智能体行为数据
        self.entropy_history: List[float] = []
        self.position_weights_history: List[np.ndarray] = []
        
        # 风险控制数据
        self.risk_budget_history: List[float] = []
        self.risk_usage_history: List[float] = []
        self.control_signals_history: List[Dict[str, Any]] = []
        self.market_regime_history: List[str] = []
        
        logger.info("增强训练器初始化完成")
        logger.info(f"指标计算配置: 投资组合指标={config.enable_portfolio_metrics}, "
                   f"智能体行为指标={config.enable_agent_behavior_metrics}, "
                   f"风险控制指标={config.enable_risk_control_metrics}")
    
    def _run_episode(self, episode_num: int, training: bool = True) -> Tuple[float, int]:
        """
        运行单个episode（增强版）
        
        Args:
            episode_num: episode编号
            training: 是否为训练模式
            
        Returns:
            Tuple[float, int]: episode奖励和长度
        """
        # 调用父类方法运行episode
        episode_reward, episode_length = super()._run_episode(episode_num, training)
        
        # 如果是训练模式且到了指标计算频率，计算并记录增强指标
        if training and self._should_calculate_metrics(episode_num):
            self._calculate_and_log_enhanced_metrics(episode_num)
        
        return episode_reward, episode_length
    
    def _should_calculate_metrics(self, episode_num: int) -> bool:
        """
        判断是否应该计算指标
        
        Args:
            episode_num: episode编号
            
        Returns:
            是否应该计算指标
        """
        return episode_num % self.config.metrics_calculation_frequency == 0
    
    def _calculate_and_log_enhanced_metrics(self, episode_num: int):
        """
        计算并记录增强指标
        
        Args:
            episode_num: episode编号
        """
        try:
            # 计算投资组合指标
            portfolio_metrics = None
            if self.config.enable_portfolio_metrics:
                portfolio_metrics = self._calculate_portfolio_metrics()
            
            # 计算智能体行为指标
            agent_metrics = None
            if self.config.enable_agent_behavior_metrics:
                agent_metrics = self._calculate_agent_behavior_metrics()
            
            # 计算风险控制指标
            risk_metrics = None
            if self.config.enable_risk_control_metrics:
                risk_metrics = self._calculate_risk_control_metrics()
            
            # 记录指标日志
            if self.config.detailed_metrics_logging:
                self._log_enhanced_metrics(episode_num, portfolio_metrics, agent_metrics, risk_metrics)
            
        except Exception as e:
            logger.error(f"计算增强指标时发生错误: {e}")
    
    def _calculate_portfolio_metrics(self) -> Optional[PortfolioMetrics]:
        """
        计算投资组合指标
        
        Returns:
            投资组合指标或None（如果数据不足）
        """
        if len(self.portfolio_values_history) <= 1:
            logger.debug("投资组合价值历史数据不足，跳过指标计算")
            return None
        
        try:
            # 确保基准数据长度匹配
            if len(self.benchmark_values_history) != len(self.portfolio_values_history):
                logger.warning(f"基准数据长度({len(self.benchmark_values_history)})与投资组合数据长度"
                             f"({len(self.portfolio_values_history)})不匹配")
                # 截取到较短的长度
                min_len = min(len(self.benchmark_values_history), len(self.portfolio_values_history))
                portfolio_values = self.portfolio_values_history[:min_len]
                benchmark_values = self.benchmark_values_history[:min_len]
                dates = self.dates_history[:min_len] if len(self.dates_history) >= min_len else [datetime.now()] * min_len
            else:
                portfolio_values = self.portfolio_values_history
                benchmark_values = self.benchmark_values_history
                dates = self.dates_history if len(self.dates_history) == len(portfolio_values) else [datetime.now()] * len(portfolio_values)
            
            metrics = self.metrics_calculator.calculate_portfolio_metrics(
                portfolio_values=portfolio_values,
                benchmark_values=benchmark_values,
                dates=dates,
                risk_free_rate=self.config.risk_free_rate
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算投资组合指标失败: {e}")
            return None
    
    def _calculate_agent_behavior_metrics(self) -> Optional[AgentBehaviorMetrics]:
        """
        计算智能体行为指标
        
        Returns:
            智能体行为指标或None（如果数据不足）
        """
        if len(self.entropy_history) == 0:
            logger.debug("熵值历史数据为空，跳过智能体行为指标计算")
            return None
        
        try:
            metrics = self.metrics_calculator.calculate_agent_behavior_metrics(
                entropy_values=self.entropy_history,
                position_weights_history=self.position_weights_history
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算智能体行为指标失败: {e}")
            return None
    
    def _calculate_risk_control_metrics(self) -> Optional[RiskControlMetrics]:
        """
        计算风险控制指标
        
        Returns:
            风险控制指标或None（如果数据不足）
        """
        # 检查环境是否有回撤控制器
        if not hasattr(self.environment, 'drawdown_controller') or self.environment.drawdown_controller is None:
            logger.debug("环境中没有回撤控制器，跳过风险控制指标计算")
            return None
        
        try:
            drawdown_controller = self.environment.drawdown_controller
            
            # 从回撤控制器获取数据
            risk_budget_history = getattr(drawdown_controller.adaptive_risk_budget, 'risk_budget_history', [])
            risk_usage_history = getattr(drawdown_controller.adaptive_risk_budget, 'risk_usage_history', [])
            control_signals = getattr(drawdown_controller, 'control_signal_queue', [])
            
            # 获取市场状态历史
            market_regime_history = []
            if hasattr(drawdown_controller, 'market_regime_detector') and drawdown_controller.market_regime_detector:
                market_regime_history = getattr(drawdown_controller.market_regime_detector, 'regime_history', [])
            
            # 转换控制信号为字典格式
            control_signals_dict = []
            for signal in control_signals:
                if hasattr(signal, 'to_dict'):
                    control_signals_dict.append(signal.to_dict())
                elif isinstance(signal, dict):
                    control_signals_dict.append(signal)
            
            metrics = self.metrics_calculator.calculate_risk_control_metrics(
                risk_budget_history=risk_budget_history,
                risk_usage_history=risk_usage_history,
                control_signals=control_signals_dict,
                market_regime_history=market_regime_history
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"计算风险控制指标失败: {e}")
            return None
    
    def _log_enhanced_metrics(self, episode: int,
                            portfolio_metrics: Optional[PortfolioMetrics],
                            agent_metrics: Optional[AgentBehaviorMetrics],
                            risk_metrics: Optional[RiskControlMetrics]):
        """
        记录增强指标日志
        
        Args:
            episode: episode编号
            portfolio_metrics: 投资组合指标
            agent_metrics: 智能体行为指标
            risk_metrics: 风险控制指标
        """
        log_lines = [f"=== Episode {episode} 增强指标报告 ==="]
        
        # 投资组合指标
        if portfolio_metrics:
            log_lines.append("📊 投资组合与市场表现对比指标:")
            log_lines.append(f"  • 夏普比率 (Sharpe Ratio): {portfolio_metrics.sharpe_ratio:.4f}")
            log_lines.append(f"  • 最大回撤 (Max Drawdown): {portfolio_metrics.max_drawdown:.4f}")
            log_lines.append(f"  • Alpha (相对基准超额收益): {portfolio_metrics.alpha:.4f}")
            log_lines.append(f"  • Beta (系统性风险): {portfolio_metrics.beta:.4f}")
            log_lines.append(f"  • 年化收益率 (Annualized Return): {portfolio_metrics.annualized_return:.4f}")
        else:
            log_lines.append("📊 投资组合指标: 数据不足，跳过计算")
        
        # 智能体行为指标
        if agent_metrics:
            log_lines.append("🤖 智能体行为分析指标:")
            log_lines.append(f"  • 平均熵值 (Mean Entropy): {agent_metrics.mean_entropy:.4f}")
            log_lines.append(f"  • 熵值趋势 (Entropy Trend): {agent_metrics.entropy_trend:.4f}")
            log_lines.append(f"  • 平均持仓集中度 (Position Concentration): {agent_metrics.mean_position_concentration:.4f}")
            log_lines.append(f"  • 换手率 (Turnover Rate): {agent_metrics.turnover_rate:.4f}")
        else:
            log_lines.append("🤖 智能体行为指标: 数据不足，跳过计算")
        
        # 风险控制指标
        if risk_metrics:
            log_lines.append("🛡️ 风险与回撤控制指标:")
            log_lines.append(f"  • 平均风险预算使用率: {risk_metrics.avg_risk_budget_utilization:.4f}")
            log_lines.append(f"  • 风险预算效率: {risk_metrics.risk_budget_efficiency:.4f}")
            log_lines.append(f"  • 控制信号频率: {risk_metrics.control_signal_frequency:.4f}")
            log_lines.append(f"  • 市场状态稳定性: {risk_metrics.market_regime_stability:.4f}")
        else:
            log_lines.append("🛡️ 风险控制指标: 回撤控制器未启用或数据不足")
        
        log_lines.append("=" * 50)
        
        # 输出日志
        for line in log_lines:
            logger.info(line)
    
    def _update_metrics_histories(self, episode_info: Dict[str, Any], update_info: Dict[str, Any]):
        """
        更新指标历史数据
        
        Args:
            episode_info: episode信息
            update_info: 智能体更新信息
        """
        # 更新投资组合价值历史
        if 'portfolio_value' in episode_info:
            self.portfolio_values_history.append(episode_info['portfolio_value'])
        
        # 更新基准价值历史（如果有）
        if 'benchmark_value' in episode_info:
            self.benchmark_values_history.append(episode_info['benchmark_value'])
        elif len(self.portfolio_values_history) > len(self.benchmark_values_history):
            # 如果没有基准数据，使用默认增长率
            if len(self.benchmark_values_history) == 0:
                self.benchmark_values_history.append(self.config.initial_cash)
            else:
                # 假设基准年化收益率为8%
                daily_return = 0.08 / 252
                last_value = self.benchmark_values_history[-1]
                self.benchmark_values_history.append(last_value * (1 + daily_return))
        
        # 更新日期历史
        self.dates_history.append(datetime.now())
        
        # 更新智能体行为数据
        if 'policy_entropy' in update_info:
            self.entropy_history.append(update_info['policy_entropy'])
        
        if 'positions' in episode_info:
            positions = episode_info['positions']
            if isinstance(positions, np.ndarray):
                self.position_weights_history.append(positions.copy())
    
    def _log_episode_stats(self, episode: int, reward: float, length: int,
                          update_info: Dict[str, float]):
        """
        记录episode统计信息（增强版）
        
        Args:
            episode: episode编号
            reward: episode奖励
            length: episode长度
            update_info: 更新信息
        """
        # 调用父类方法记录基础统计
        super()._log_episode_stats(episode, reward, length, update_info)
        
        # 更新指标历史数据
        episode_info = {
            'portfolio_value': getattr(self.environment, 'total_value', self.config.initial_cash),
            'positions': getattr(self.environment, 'current_positions', np.zeros(5))
        }
        
        self._update_metrics_histories(episode_info, update_info)
    
    def get_enhanced_training_stats(self) -> Dict[str, Any]:
        """
        获取增强训练统计信息
        
        Returns:
            增强训练统计信息
        """
        # 获取基础统计
        base_stats = self.get_training_stats() if hasattr(super(), 'get_training_stats') else {}
        
        # 添加增强统计
        enhanced_stats = {
            'portfolio_values_count': len(self.portfolio_values_history),
            'entropy_values_count': len(self.entropy_history),
            'position_weights_count': len(self.position_weights_history),
            'latest_portfolio_value': self.portfolio_values_history[-1] if self.portfolio_values_history else 0,
            'latest_entropy': self.entropy_history[-1] if self.entropy_history else 0,
        }
        
        # 如果有足够数据，计算最新指标
        if len(self.portfolio_values_history) > 1:
            try:
                latest_portfolio_metrics = self._calculate_portfolio_metrics()
                if latest_portfolio_metrics:
                    enhanced_stats.update({
                        'latest_sharpe_ratio': latest_portfolio_metrics.sharpe_ratio,
                        'latest_max_drawdown': latest_portfolio_metrics.max_drawdown,
                        'latest_alpha': latest_portfolio_metrics.alpha,
                        'latest_beta': latest_portfolio_metrics.beta,
                        'latest_annualized_return': latest_portfolio_metrics.annualized_return
                    })
            except Exception as e:
                logger.debug(f"计算最新投资组合指标失败: {e}")
        
        if len(self.entropy_history) > 0:
            try:
                latest_agent_metrics = self._calculate_agent_behavior_metrics()
                if latest_agent_metrics:
                    enhanced_stats.update({
                        'latest_mean_entropy': latest_agent_metrics.mean_entropy,
                        'latest_entropy_trend': latest_agent_metrics.entropy_trend,
                        'latest_position_concentration': latest_agent_metrics.mean_position_concentration,
                        'latest_turnover_rate': latest_agent_metrics.turnover_rate
                    })
            except Exception as e:
                logger.debug(f"计算最新智能体行为指标失败: {e}")
        
        # 合并统计信息
        base_stats.update(enhanced_stats)
        return base_stats
    
    def reset_enhanced_histories(self):
        """重置增强历史数据"""
        self.portfolio_values_history.clear()
        self.benchmark_values_history.clear()
        self.dates_history.clear()
        self.entropy_history.clear()
        self.position_weights_history.clear()
        self.risk_budget_history.clear()
        self.risk_usage_history.clear()
        self.control_signals_history.clear()
        self.market_regime_history.clear()
        
        logger.info("增强历史数据已重置")