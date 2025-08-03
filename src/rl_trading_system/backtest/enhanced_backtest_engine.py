"""
增强回测引擎

集成回撤控制策略的回测引擎，支持参数化配置和多维度评估。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
from datetime import datetime
import warnings

from .drawdown_control_config import DrawdownControlConfig, BacktestComparisonConfig
from ..risk_control.drawdown_monitor import DrawdownMonitor, DrawdownMetrics
from ..risk_control.dynamic_stop_loss import DynamicStopLoss
from ..risk_control.reward_optimizer import RewardOptimizer
from ..risk_control.adaptive_risk_budget import AdaptiveRiskBudget
from ..trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from ..models.sac_agent import SACAgent, SACConfig

logger = logging.getLogger(__name__)


@dataclass
class EnhancedBacktestResult:
    """增强回测结果数据类"""
    # 基础性能指标
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # 回撤控制效果指标
    drawdown_improvement: float          # 回撤改善度
    stop_loss_trigger_count: int         # 止损触发次数
    position_adjustment_count: int       # 仓位调整次数
    risk_budget_utilization: float      # 风险预算使用率
    
    # 详细回撤分析
    drawdown_metrics: DrawdownMetrics
    drawdown_periods: List[Dict[str, Any]]
    recovery_times: List[int]
    
    # 交易统计
    total_trades: int
    win_rate: float
    profit_factor: float
    average_trade_return: float
    
    # 时间序列数据
    portfolio_values: pd.Series
    drawdown_series: pd.Series
    position_series: pd.DataFrame
    
    # 配置信息
    config: DrawdownControlConfig
    backtest_period: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'drawdown_improvement': self.drawdown_improvement,
            'stop_loss_trigger_count': self.stop_loss_trigger_count,
            'position_adjustment_count': self.position_adjustment_count,
            'risk_budget_utilization': self.risk_budget_utilization,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_trade_return': self.average_trade_return,
            'drawdown_metrics': self.drawdown_metrics.to_dict(),
            'backtest_period': self.backtest_period,
            'config': self.config.to_dict()
        }


class EnhancedBacktestEngine:
    """
    增强回测引擎
    
    集成回撤控制策略的回测引擎，提供：
    - 回撤控制组件集成
    - 参数化配置测试
    - 多维度性能评估
    - 统计显著性检验
    """
    
    def __init__(self, 
                 portfolio_config: PortfolioConfig,
                 data_interface,
                 drawdown_config: Optional[DrawdownControlConfig] = None):
        """
        初始化增强回测引擎
        
        Args:
            portfolio_config: 投资组合配置
            data_interface: 数据接口
            drawdown_config: 回撤控制配置
        """
        self.portfolio_config = portfolio_config
        self.data_interface = data_interface
        self.drawdown_config = drawdown_config or DrawdownControlConfig()
        
        # 初始化回撤控制组件
        self._initialize_drawdown_control_components()
        
        # 回测历史记录
        self.backtest_history: List[EnhancedBacktestResult] = []
        
        logger.info("初始化增强回测引擎成功")
    
    def _initialize_drawdown_control_components(self):
        """初始化回撤控制组件"""
        try:
            # 回撤监控器
            self.drawdown_monitor = DrawdownMonitor(
                lookback_window=self.drawdown_config.drawdown_calculation_window
            )
            
            # 动态止损控制器配置
            from ..risk_control.dynamic_stop_loss import StopLossConfig
            stop_loss_config = StopLossConfig(
                base_stop_loss=self.drawdown_config.base_stop_loss,
                volatility_multiplier=self.drawdown_config.volatility_multiplier,
                trailing_stop_distance=self.drawdown_config.trailing_stop_distance
            )
            self.stop_loss_controller = DynamicStopLoss(config=stop_loss_config)
            
            # 奖励函数优化器配置
            from ..risk_control.reward_optimizer import RewardConfig
            reward_config = RewardConfig(
                drawdown_penalty_factor=self.drawdown_config.drawdown_penalty_factor,
                risk_aversion_coefficient=self.drawdown_config.risk_aversion_coefficient
            )
            self.reward_optimizer = RewardOptimizer(config=reward_config)
            
            # 自适应风险预算配置
            from ..risk_control.adaptive_risk_budget import AdaptiveRiskBudgetConfig
            risk_budget_config = AdaptiveRiskBudgetConfig(
                base_risk_budget=self.drawdown_config.base_risk_budget
            )
            self.risk_budget_manager = AdaptiveRiskBudget(config=risk_budget_config)
            
            logger.info("回撤控制组件初始化成功")
            
        except Exception as e:
            logger.error(f"初始化回撤控制组件失败: {e}")
            raise RuntimeError(f"初始化回撤控制组件失败: {e}")
    
    def run_backtest(self, 
                     agent: SACAgent,
                     start_date: str,
                     end_date: str,
                     baseline_name: str = "enhanced") -> EnhancedBacktestResult:
        """
        运行增强回测
        
        Args:
            agent: 训练好的SAC智能体
            start_date: 开始日期
            end_date: 结束日期
            baseline_name: 基线名称
            
        Returns:
            增强回测结果
        """
        logger.info(f"开始运行增强回测: {start_date} 到 {end_date}")
        
        try:
            # 创建增强的投资组合环境
            enhanced_env = self._create_enhanced_environment()
            
            # 运行回测
            portfolio_values, positions_history, trades_history = self._execute_backtest(
                enhanced_env, agent, start_date, end_date
            )
            
            # 计算性能指标
            result = self._calculate_enhanced_metrics(
                portfolio_values, positions_history, trades_history,
                start_date, end_date, baseline_name
            )
            
            # 保存回测历史
            self.backtest_history.append(result)
            
            logger.info(f"增强回测完成，最大回撤: {result.max_drawdown:.4f}, "
                       f"年化收益: {result.annual_return:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"增强回测执行失败: {e}")
            raise RuntimeError(f"增强回测执行失败: {e}")
    
    def _create_enhanced_environment(self) -> PortfolioEnvironment:
        """创建集成回撤控制的增强环境"""
        # 基于原始配置创建环境
        env = PortfolioEnvironment(self.portfolio_config, self.data_interface)
        
        # 集成回撤控制组件
        env.drawdown_monitor = self.drawdown_monitor
        env.stop_loss_controller = self.stop_loss_controller
        env.reward_optimizer = self.reward_optimizer
        env.risk_budget_manager = self.risk_budget_manager
        
        # 增强step方法以集成回撤控制逻辑
        original_step = env.step
        
        def enhanced_step(action):
            # 执行原始step
            obs, reward, done, info = original_step(action)
            
            # 应用回撤控制逻辑
            portfolio_value = env.portfolio_value
            positions = env.current_positions
            
            # 更新回撤监控
            drawdown_metrics = self.drawdown_monitor.update(portfolio_value)
            
            # 检查止损触发
            stop_loss_signals = self.stop_loss_controller.check_stop_loss_triggers(
                positions
            )
            
            # 调整奖励函数
            enhanced_reward = self.reward_optimizer.calculate_risk_adjusted_reward(
                reward, drawdown_metrics.current_drawdown, positions
            )
            
            # 更新风险预算
            current_risk_budget = self.risk_budget_manager.update(
                portfolio_value, drawdown_metrics.current_drawdown
            )
            
            # 更新info
            info.update({
                'drawdown_metrics': drawdown_metrics,
                'stop_loss_signals': stop_loss_signals,
                'risk_budget': current_risk_budget,
                'original_reward': reward,
                'enhanced_reward': enhanced_reward
            })
            
            return obs, enhanced_reward, done, info
        
        env.step = enhanced_step
        return env
    
    def _execute_backtest(self, 
                         env: PortfolioEnvironment,
                         agent: SACAgent,
                         start_date: str,
                         end_date: str) -> Tuple[pd.Series, pd.DataFrame, List[Dict]]:
        """执行回测逻辑"""
        # 重置环境
        obs = env.reset()
        
        # 记录数据
        portfolio_values = []
        positions_history = []
        trades_history = []
        
        done = False
        step_count = 0
        stop_loss_trigger_count = 0
        position_adjustment_count = 0
        
        while not done:
            # 获取动作
            action = agent.get_action(obs, deterministic=True)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 记录投资组合价值
            portfolio_values.append(env.portfolio_value)
            
            # 记录持仓
            positions_history.append({
                'step': step_count,
                'date': env.current_date if hasattr(env, 'current_date') else None,
                'positions': env.current_positions.copy(),
                'portfolio_value': env.portfolio_value
            })
            
            # 检查止损触发
            if 'stop_loss_signals' in info and any(info['stop_loss_signals'].values()):
                stop_loss_trigger_count += 1
                
            # 检查仓位调整
            if step_count > 0 and 'risk_budget' in info:
                prev_positions = positions_history[-2]['positions'] if len(positions_history) > 1 else {}
                current_positions = env.current_positions
                
                position_changed = not np.allclose(
                    list(prev_positions.values()), 
                    list(current_positions.values()),
                    atol=0.01
                )
                
                if position_changed:
                    position_adjustment_count += 1
            
            # 记录交易
            if hasattr(env, 'last_trades'):
                trades_history.extend(env.last_trades)
            
            obs = next_obs
            step_count += 1
        
        # 转换为时间序列
        portfolio_values_series = pd.Series(portfolio_values)
        positions_df = pd.DataFrame(positions_history)
        
        # 存储统计信息
        self._store_backtest_stats(stop_loss_trigger_count, position_adjustment_count)
        
        return portfolio_values_series, positions_df, trades_history
    
    def _store_backtest_stats(self, stop_loss_count: int, position_adj_count: int):
        """存储回测统计信息"""
        self._stop_loss_trigger_count = stop_loss_count
        self._position_adjustment_count = position_adj_count
    
    def _calculate_enhanced_metrics(self,
                                   portfolio_values: pd.Series,
                                   positions_history: pd.DataFrame,
                                   trades_history: List[Dict],
                                   start_date: str,
                                   end_date: str,
                                   baseline_name: str) -> EnhancedBacktestResult:
        """计算增强性能指标"""
        
        # 基础性能计算
        returns = portfolio_values.pct_change().dropna()
        
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # 回撤计算
        running_max = portfolio_values.expanding().max()
        drawdown_series = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown_series.min()
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 回撤详细分析
        drawdown_metrics = self._analyze_drawdown_periods(portfolio_values, drawdown_series)
        
        # 交易统计
        trade_stats = self._calculate_trade_statistics(trades_history, returns)
        
        # 获取存储的统计信息
        stop_loss_count = getattr(self, '_stop_loss_trigger_count', 0)
        position_adj_count = getattr(self, '_position_adjustment_count', 0)
        
        # 风险预算使用率（模拟）
        risk_budget_utilization = min(abs(max_drawdown) / self.drawdown_config.max_drawdown_threshold, 1.0)
        
        # 回撤改善度（与基准比较，这里先设为0，实际需要基准数据）
        drawdown_improvement = 0.0
        
        return EnhancedBacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            drawdown_improvement=drawdown_improvement,
            stop_loss_trigger_count=stop_loss_count,
            position_adjustment_count=position_adj_count,
            risk_budget_utilization=risk_budget_utilization,
            drawdown_metrics=drawdown_metrics,
            drawdown_periods=self._extract_drawdown_periods(drawdown_series),
            recovery_times=self._calculate_recovery_times(drawdown_series),
            total_trades=trade_stats['total_trades'],
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            average_trade_return=trade_stats['average_trade_return'],
            portfolio_values=portfolio_values,
            drawdown_series=drawdown_series,
            position_series=positions_history,
            config=self.drawdown_config,
            backtest_period={'start_date': start_date, 'end_date': end_date}
        )
    
    def _analyze_drawdown_periods(self, portfolio_values: pd.Series, drawdown_series: pd.Series) -> DrawdownMetrics:
        """分析回撤周期"""
        max_drawdown = drawdown_series.min()
        current_drawdown = drawdown_series.iloc[-1]
        
        # 计算回撤频率和平均回撤
        drawdown_periods = self._extract_drawdown_periods(drawdown_series)
        drawdown_frequency = len(drawdown_periods) / len(portfolio_values) * 252
        average_drawdown = np.mean([p['max_drawdown'] for p in drawdown_periods]) if drawdown_periods else 0
        
        # 找到峰值和谷值
        peak_idx = portfolio_values.idxmax()
        trough_idx = portfolio_values.idxmin()
        peak_value = portfolio_values.max()
        trough_value = portfolio_values.min()
        
        # 计算距离峰值天数
        days_since_peak = len(portfolio_values) - peak_idx - 1 if peak_idx < len(portfolio_values) - 1 else 0
        
        return DrawdownMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown,
            drawdown_duration=self._calculate_max_drawdown_duration(drawdown_series),
            recovery_time=None,  # 需要更复杂的计算
            peak_value=peak_value,
            trough_value=trough_value,
            underwater_curve=drawdown_series.tolist(),
            drawdown_frequency=drawdown_frequency,
            average_drawdown=average_drawdown,
            current_phase=self._determine_drawdown_phase(drawdown_series),
            days_since_peak=days_since_peak
        )
    
    def _extract_drawdown_periods(self, drawdown_series: pd.Series) -> List[Dict[str, Any]]:
        """提取回撤周期"""
        periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown_series):
            if dd < -0.01 and not in_drawdown:  # 开始回撤
                in_drawdown = True
                start_idx = i
            elif dd >= -0.01 and in_drawdown:  # 结束回撤
                in_drawdown = False
                end_idx = i
                
                if start_idx is not None:
                    period_dd = drawdown_series.iloc[start_idx:end_idx]
                    periods.append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'duration': end_idx - start_idx,
                        'max_drawdown': period_dd.min(),
                        'recovery_time': None  # 简化处理
                    })
        
        return periods
    
    def _calculate_recovery_times(self, drawdown_series: pd.Series) -> List[int]:
        """计算恢复时间"""
        # 简化实现，返回空列表
        return []
    
    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """计算最大回撤持续期"""
        max_duration = 0
        current_duration = 0
        
        for dd in drawdown_series:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _determine_drawdown_phase(self, drawdown_series: pd.Series):
        """确定当前回撤阶段"""
        from ..risk_control.drawdown_monitor import DrawdownPhase
        
        current_dd = drawdown_series.iloc[-1]
        
        if current_dd >= -0.01:
            return DrawdownPhase.NORMAL
        elif len(drawdown_series) >= 5:
            recent_trend = drawdown_series.iloc[-5:].diff().mean()
            if recent_trend < -0.001:
                return DrawdownPhase.DRAWDOWN_CONTINUE
            elif recent_trend > 0.001:
                return DrawdownPhase.RECOVERY
            else:
                return DrawdownPhase.DRAWDOWN_START
        else:
            return DrawdownPhase.DRAWDOWN_START
    
    def _calculate_trade_statistics(self, trades_history: List[Dict], returns: pd.Series) -> Dict[str, float]:
        """计算交易统计"""
        if not trades_history:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'average_trade_return': 0.0
            }
        
        # 简化实现
        total_trades = len(trades_history)
        
        # 基于收益序列估算胜率
        positive_returns = (returns > 0).sum()
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
        
        # 计算盈亏比
        positive_sum = returns[returns > 0].sum()
        negative_sum = abs(returns[returns < 0].sum())
        profit_factor = positive_sum / negative_sum if negative_sum > 0 else float('inf')
        
        average_trade_return = returns.mean()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_trade_return': average_trade_return
        }
    
    def compare_with_baseline(self, 
                             results: List[EnhancedBacktestResult],
                             baseline_result: EnhancedBacktestResult) -> Dict[str, Any]:
        """与基线策略比较"""
        if not results:
            raise ValueError("结果列表不能为空")
        
        comparison = {}
        
        for i, result in enumerate(results):
            comparison[f'strategy_{i}'] = {
                'return_improvement': result.annual_return - baseline_result.annual_return,
                'volatility_change': result.volatility - baseline_result.volatility,
                'sharpe_improvement': result.sharpe_ratio - baseline_result.sharpe_ratio,
                'drawdown_improvement': baseline_result.max_drawdown - result.max_drawdown,
                'calmar_improvement': result.calmar_ratio - baseline_result.calmar_ratio
            }
        
        return comparison
    
    def save_results(self, result: EnhancedBacktestResult, output_path: str):
        """保存回测结果"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存JSON格式结果
        result_dict = result.to_dict()
        
        # 处理不能序列化的对象
        if 'portfolio_values' in result_dict:
            del result_dict['portfolio_values']
        if 'drawdown_series' in result_dict:
            del result_dict['drawdown_series']
        if 'position_series' in result_dict:
            del result_dict['position_series']
        
        with open(output_path / 'enhanced_backtest_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # 保存时间序列数据
        result.portfolio_values.to_csv(output_path / 'portfolio_values.csv')
        result.drawdown_series.to_csv(output_path / 'drawdown_series.csv')
        
        if isinstance(result.position_series, pd.DataFrame):
            result.position_series.to_csv(output_path / 'position_series.csv', index=False)
        
        logger.info(f"增强回测结果已保存到: {output_path}")