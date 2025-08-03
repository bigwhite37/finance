"""
回撤控制主控制器

统一协调和管理所有回撤控制组件，实现决策优先级排序和冲突解决。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque, defaultdict
import threading
from pathlib import Path

from .drawdown_monitor import DrawdownMonitor, DrawdownMetrics, DrawdownPhase
from .dynamic_stop_loss import DynamicStopLoss, StopLossConfig
from .reward_optimizer import RewardOptimizer, RewardConfig
from .adaptive_risk_budget import AdaptiveRiskBudget, AdaptiveRiskBudgetConfig
from .market_regime_detector import MarketRegimeDetector
from .stress_test_engine import StressTestEngine
from ..backtest.drawdown_control_config import DrawdownControlConfig

logger = logging.getLogger(__name__)


class ControlSignalType(Enum):
    """控制信号类型"""
    STOP_LOSS = "stop_loss"                    # 止损信号
    POSITION_ADJUSTMENT = "position_adjustment" # 仓位调整信号
    RISK_BUDGET_CHANGE = "risk_budget_change"  # 风险预算变更信号
    REWARD_OPTIMIZATION = "reward_optimization" # 奖励优化信号
    MARKET_REGIME_CHANGE = "market_regime_change" # 市场状态变化信号
    EMERGENCY_HALT = "emergency_halt"          # 紧急停止信号


class ControlSignalPriority(Enum):
    """控制信号优先级"""
    CRITICAL = 1    # 紧急关键
    HIGH = 2        # 高优先级
    MEDIUM = 3      # 中等优先级
    LOW = 4         # 低优先级
    INFO = 5        # 信息类


@dataclass
class ControlSignal:
    """控制信号数据类"""
    signal_type: ControlSignalType
    priority: ControlSignalPriority
    timestamp: datetime
    source_component: str
    content: Dict[str, Any]
    expiry_time: Optional[datetime] = None
    processed: bool = False
    
    def __post_init__(self):
        if self.expiry_time is None:
            # 默认5分钟过期
            self.expiry_time = self.timestamp + timedelta(minutes=5)
    
    def is_expired(self) -> bool:
        """检查信号是否过期"""
        return datetime.now() > self.expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'signal_type': self.signal_type.value,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'source_component': self.source_component,
            'content': self.content,
            'expiry_time': self.expiry_time.isoformat() if self.expiry_time else None,
            'processed': self.processed
        }


@dataclass
class MarketState:
    """市场状态数据类"""
    prices: Dict[str, float]
    volumes: Dict[str, float]
    timestamp: datetime
    market_indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'prices': self.prices,
            'volumes': self.volumes,
            'timestamp': self.timestamp.isoformat(),
            'market_indicators': self.market_indicators
        }


@dataclass
class PortfolioState:
    """投资组合状态数据类"""
    positions: Dict[str, float]
    portfolio_value: float
    cash: float
    timestamp: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'positions': self.positions,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'timestamp': self.timestamp.isoformat(),
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.ConflictResolver')
    
    def resolve_conflicts(self, signals: List[ControlSignal]) -> List[ControlSignal]:
        """
        解决控制信号冲突
        
        Args:
            signals: 控制信号列表
            
        Returns:
            解决冲突后的信号列表
        """
        if not signals:
            return []
        
        # 按优先级排序
        sorted_signals = sorted(signals, key=lambda s: s.priority.value)
        
        # 移除过期信号
        valid_signals = [s for s in sorted_signals if not s.is_expired()]
        
        # 解决具体冲突
        resolved_signals = self._resolve_specific_conflicts(valid_signals)
        
        # 只在有真正冲突时记录日志
        has_conflicts = self._detect_conflicts(signals, valid_signals, resolved_signals)
        if has_conflicts:
            self.logger.info(f"冲突解决完成：输入{len(signals)}个信号，输出{len(resolved_signals)}个信号")
        
        return resolved_signals
    
    def _detect_conflicts(self, 
                         original_signals: List[ControlSignal],
                         valid_signals: List[ControlSignal],
                         resolved_signals: List[ControlSignal]) -> bool:
        """
        检测是否有真正的冲突
        
        Args:
            original_signals: 原始信号列表
            valid_signals: 移除过期信号后的列表
            resolved_signals: 最终解决后的信号列表
            
        Returns:
            是否存在真正的冲突
        """
        # 1. 检查是否有过期信号被移除
        if len(original_signals) != len(valid_signals):
            return True
        
        # 2. 检查是否有同类型信号冲突（信号数量减少）
        if len(valid_signals) != len(resolved_signals):
            return True
        
        # 3. 检查是否有跨类型冲突（信号内容被修改）
        # 简化：如果信号数量没变，但有多个同类型信号，说明有潜在冲突被解决
        signal_types = set()
        duplicate_types = set()
        for signal in valid_signals:
            if signal.signal_type in signal_types:
                duplicate_types.add(signal.signal_type)
            signal_types.add(signal.signal_type)
        
        if duplicate_types:
            return True
        
        # 4. 没有检测到冲突
        return False
    
    def _resolve_specific_conflicts(self, signals: List[ControlSignal]) -> List[ControlSignal]:
        """解决具体类型冲突"""
        if len(signals) <= 1:
            return signals
        
        # 按信号类型分组
        signal_groups = defaultdict(list)
        for signal in signals:
            signal_groups[signal.signal_type].append(signal)
        
        resolved = []
        
        for signal_type, group_signals in signal_groups.items():
            if len(group_signals) == 1:
                resolved.extend(group_signals)
            else:
                # 处理同类型信号冲突
                resolved_group = self._resolve_same_type_conflicts(group_signals)
                resolved.extend(resolved_group)
        
        # 处理跨类型冲突
        return self._resolve_cross_type_conflicts(resolved)
    
    def _resolve_same_type_conflicts(self, signals: List[ControlSignal]) -> List[ControlSignal]:
        """解决同类型信号冲突"""
        if not signals:
            return []
        
        # 选择优先级最高的信号
        return [min(signals, key=lambda s: s.priority.value)]
    
    def _resolve_cross_type_conflicts(self, signals: List[ControlSignal]) -> List[ControlSignal]:
        """解决跨类型信号冲突"""
        # 检查紧急停止信号
        emergency_signals = [s for s in signals if s.signal_type == ControlSignalType.EMERGENCY_HALT]
        if emergency_signals:
            return emergency_signals  # 紧急停止信号优先级最高
        
        # 检查止损与仓位调整冲突
        stop_loss_signals = [s for s in signals if s.signal_type == ControlSignalType.STOP_LOSS]
        position_signals = [s for s in signals if s.signal_type == ControlSignalType.POSITION_ADJUSTMENT]
        
        if stop_loss_signals and position_signals:
            # 止损信号优先级更高
            self.logger.info("检测到止损与仓位调整冲突，优先执行止损")
            return stop_loss_signals + [s for s in signals if s not in position_signals]
        
        return signals


class StateManager:
    """状态管理器"""
    
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.market_state_history: deque = deque(maxlen=max_history_size)
        self.portfolio_state_history: deque = deque(maxlen=max_history_size)
        self.control_signal_history: deque = deque(maxlen=max_history_size)
        self.current_market_state: Optional[MarketState] = None
        self.current_portfolio_state: Optional[PortfolioState] = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__ + '.StateManager')
    
    def update_market_state(self, market_state: MarketState):
        """更新市场状态"""
        with self.lock:
            self.current_market_state = market_state
            self.market_state_history.append(market_state)
            self.logger.debug(f"更新市场状态: {len(market_state.prices)}只股票")
    
    def update_portfolio_state(self, portfolio_state: PortfolioState):
        """更新投资组合状态"""
        with self.lock:
            self.current_portfolio_state = portfolio_state
            self.portfolio_state_history.append(portfolio_state)
            self.logger.debug(f"更新投资组合状态: 总价值 {portfolio_state.portfolio_value:.2f}")
    
    def record_control_signal(self, signal: ControlSignal):
        """记录控制信号"""
        with self.lock:
            self.control_signal_history.append(signal)
            self.logger.debug(f"记录控制信号: {signal.signal_type.value}")
    
    def get_state_consistency_check(self) -> Dict[str, Any]:
        """获取状态一致性检查结果"""
        with self.lock:
            if not self.current_market_state or not self.current_portfolio_state:
                return {'consistent': False, 'reason': '缺少当前状态数据'}
            
            # 检查时间戳一致性（允许1秒误差）
            time_diff = abs((self.current_market_state.timestamp - 
                           self.current_portfolio_state.timestamp).total_seconds())
            
            if time_diff > 1.0:
                return {
                    'consistent': False, 
                    'reason': f'市场状态和投资组合状态时间戳差异过大: {time_diff:.2f}秒'
                }
            
            return {'consistent': True, 'time_diff': time_diff}


class DataFlowManager:
    """数据流管理器"""
    
    def __init__(self):
        self.data_flow_graph: Dict[str, List[str]] = {
            'market_data': ['drawdown_monitor', 'market_regime_detector'],
            'portfolio_state': ['drawdown_monitor', 'dynamic_stop_loss', 'adaptive_risk_budget'],
            'drawdown_metrics': ['dynamic_stop_loss', 'reward_optimizer', 'adaptive_risk_budget'],
            'risk_signals': ['reward_optimizer', 'decision_engine'],
            'market_regime': ['adaptive_risk_budget', 'dynamic_stop_loss', 'decision_engine']
        }
        self.logger = logging.getLogger(__name__ + '.DataFlowManager')
    
    def validate_data_flow(self, source: str, target: str) -> bool:
        """验证数据流是否有效"""
        if source not in self.data_flow_graph:
            self.logger.warning(f"未知的数据源: {source}")
            return False
        
        if target not in self.data_flow_graph[source]:
            self.logger.warning(f"无效的数据流: {source} -> {target}")
            return False
        
        return True
    
    def get_data_dependencies(self, component: str) -> List[str]:
        """获取组件的数据依赖"""
        dependencies = []
        for source, targets in self.data_flow_graph.items():
            if component in targets:
                dependencies.append(source)
        return dependencies


class DrawdownController:
    """
    回撤控制主控制器
    
    统一协调所有回撤控制组件，实现：
    - 组件协调和数据流管理
    - 控制决策的优先级排序和冲突解决
    - 统一的控制接口和状态管理
    """
    
    def __init__(self, config: DrawdownControlConfig):
        """
        初始化回撤控制器
        
        Args:
            config: 回撤控制配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化管理器
        self.state_manager = StateManager()
        self.conflict_resolver = ConflictResolver()
        self.data_flow_manager = DataFlowManager()
        
        # 初始化控制组件
        self._initialize_control_components()
        
        # 控制状态
        self.is_active = False
        self.last_update_time: Optional[datetime] = None
        self.control_signal_queue: List[ControlSignal] = []
        
        self.logger.info("回撤控制器初始化完成")
    
    def _initialize_control_components(self):
        """初始化控制组件"""
        try:
            # 回撤监控器
            self.drawdown_monitor = DrawdownMonitor(
                lookback_window=self.config.drawdown_calculation_window
            )
            
            # 动态止损控制器
            stop_loss_config = StopLossConfig(
                base_stop_loss=self.config.base_stop_loss,
                volatility_multiplier=self.config.volatility_multiplier,
                trailing_stop_distance=self.config.trailing_stop_distance
            )
            self.dynamic_stop_loss = DynamicStopLoss(config=stop_loss_config)
            
            # 奖励优化器
            reward_config = RewardConfig(
                drawdown_penalty_factor=self.config.drawdown_penalty_factor,
                risk_aversion_coefficient=self.config.risk_aversion_coefficient
            )
            self.reward_optimizer = RewardOptimizer(config=reward_config)
            
            # 自适应风险预算
            risk_budget_config = AdaptiveRiskBudgetConfig(
                base_risk_budget=self.config.base_risk_budget
            )
            self.adaptive_risk_budget = AdaptiveRiskBudget(config=risk_budget_config)
            
            # 市场状态检测器
            if self.config.enable_market_regime_detection:
                self.market_regime_detector = MarketRegimeDetector()
            else:
                self.market_regime_detector = None
            
            # 压力测试引擎（可选）
            self.stress_test_engine = None  # 根据需要初始化
            
            self.logger.info("所有控制组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"控制组件初始化失败: {e}")
            raise RuntimeError(f"控制组件初始化失败: {e}")
    
    def execute_control_step(self, 
                           market_state: MarketState,
                           portfolio_state: PortfolioState) -> List[ControlSignal]:
        """
        执行控制步骤
        
        Args:
            market_state: 市场状态
            portfolio_state: 投资组合状态
            
        Returns:
            控制信号列表
        """
        try:
            # 更新状态
            self.state_manager.update_market_state(market_state)
            self.state_manager.update_portfolio_state(portfolio_state)
            
            # 检查状态一致性
            consistency_check = self.state_manager.get_state_consistency_check()
            if not consistency_check['consistent']:
                self.logger.warning(f"状态一致性检查失败: {consistency_check['reason']}")
            
            # 清空信号队列
            self.control_signal_queue.clear()
            
            # 执行各组件控制逻辑
            self._execute_component_controls(market_state, portfolio_state)
            
            # 解决信号冲突
            resolved_signals = self.conflict_resolver.resolve_conflicts(self.control_signal_queue)
            
            # 记录信号历史
            for signal in resolved_signals:
                self.state_manager.record_control_signal(signal)
            
            self.last_update_time = datetime.now()
            
            self.logger.debug(f"控制步骤完成，生成{len(resolved_signals)}个控制信号")
            
            return resolved_signals
            
        except Exception as e:
            self.logger.error(f"执行控制步骤失败: {e}")
            raise RuntimeError(f"执行控制步骤失败: {e}")
    
    def _execute_component_controls(self, 
                                  market_state: MarketState,
                                  portfolio_state: PortfolioState):
        """执行各组件控制逻辑"""
        current_time = datetime.now()
        
        # 1. 更新回撤监控
        drawdown_metrics = self.drawdown_monitor.update_portfolio_value(portfolio_state.portfolio_value)
        
        # 2. 检查动态止损
        self._check_dynamic_stop_loss(market_state, portfolio_state, drawdown_metrics, current_time)
        
        # 3. 更新风险预算
        self._update_risk_budget(portfolio_state, drawdown_metrics, current_time)
        
        # 4. 优化奖励函数
        self._optimize_reward_function(portfolio_state, drawdown_metrics, current_time)
        
        # 5. 检查市场状态变化
        if self.market_regime_detector:
            self._check_market_regime_change(market_state, current_time)
    
    def _check_dynamic_stop_loss(self, 
                                market_state: MarketState,
                                portfolio_state: PortfolioState,
                                drawdown_metrics: DrawdownMetrics,
                                timestamp: datetime):
        """检查动态止损"""
        # 检查组合级止损
        if abs(drawdown_metrics.current_drawdown) > self.config.portfolio_stop_loss:
            # 根据回撤程度计算风险减少因子
            drawdown_severity = abs(drawdown_metrics.current_drawdown) / self.config.portfolio_stop_loss
            risk_reduction_factor = max(0.1, min(0.8, 1.0 - drawdown_severity * 0.5))
            
            signal = ControlSignal(
                signal_type=ControlSignalType.STOP_LOSS,
                priority=ControlSignalPriority.CRITICAL,
                timestamp=timestamp,
                source_component='dynamic_stop_loss',
                content={
                    'action': 'portfolio_stop_loss',
                    'current_drawdown': drawdown_metrics.current_drawdown,
                    'threshold': self.config.portfolio_stop_loss,
                    'recommended_action': 'reduce_positions',
                    'risk_reduction_factor': risk_reduction_factor
                }
            )
            self.control_signal_queue.append(signal)
    
    def _update_risk_budget(self, 
                           portfolio_state: PortfolioState,
                           drawdown_metrics: DrawdownMetrics,
                           timestamp: datetime):
        """更新风险预算"""
        # 计算并提供表现指标数据给AdaptiveRiskBudget
        performance_metrics = self._calculate_performance_metrics(portfolio_state, drawdown_metrics, timestamp)
        if performance_metrics:
            self.adaptive_risk_budget.update_performance_metrics(performance_metrics)
        
        # 计算并提供市场指标数据给AdaptiveRiskBudget  
        market_metrics = self._calculate_market_metrics(timestamp)
        if market_metrics:
            self.adaptive_risk_budget.update_market_metrics(market_metrics)
        
        try:
            current_budget = self.adaptive_risk_budget.calculate_adaptive_risk_budget()
        except RuntimeError as e:
            # 在训练初期没有足够历史数据时，使用基础风险预算
            if "无法获取必要的" in str(e) and "数据" in str(e):
                self.logger.debug(f"自适应风险预算初期缺少数据，使用基础预算: {e}")
                current_budget = self.adaptive_risk_budget.config.base_risk_budget
            else:
                # 其他类型的错误需要重新抛出
                raise
        
        # 如果风险预算发生显著变化，生成信号
        if hasattr(self, '_last_risk_budget'):
            budget_change = abs(current_budget - self._last_risk_budget)
            if budget_change > 0.01:  # 1%的变化阈值
                signal = ControlSignal(
                    signal_type=ControlSignalType.RISK_BUDGET_CHANGE,
                    priority=ControlSignalPriority.MEDIUM,
                    timestamp=timestamp,
                    source_component='adaptive_risk_budget',
                    content={
                        'new_budget': current_budget,
                        'old_budget': self._last_risk_budget,
                        'change': budget_change
                    }
                )
                self.control_signal_queue.append(signal)
        
        self._last_risk_budget = current_budget
    
    def _calculate_performance_metrics(self, 
                                     portfolio_state: PortfolioState,
                                     drawdown_metrics: DrawdownMetrics,
                                     timestamp: datetime) -> Optional['PerformanceMetrics']:
        """计算表现指标数据"""
        try:
            from .adaptive_risk_budget import PerformanceMetrics
            
            # 从回撤监控器获取历史数据来计算表现指标
            portfolio_history = getattr(self.drawdown_monitor, 'portfolio_values', [])
            
            if len(portfolio_history) < 2:
                # 数据不足，无法计算表现指标
                return None
            
            # 计算收益率序列
            returns = []
            for i in range(1, len(portfolio_history)):
                ret = (portfolio_history[i] - portfolio_history[i-1]) / portfolio_history[i-1]
                returns.append(ret)
            
            if len(returns) == 0:
                return None
            
            returns = np.array(returns)
            
            # 计算表现指标
            total_return = (portfolio_state.portfolio_value / portfolio_history[0]) - 1 if portfolio_history[0] > 0 else 0
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0  # 年化波动率
            mean_return = np.mean(returns) * 252  # 年化收益率
            
            # 计算夏普比率 (假设无风险利率为3%)
            risk_free_rate = 0.03
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # 计算卡尔玛比率
            calmar_ratio = mean_return / abs(drawdown_metrics.max_drawdown) if abs(drawdown_metrics.max_drawdown) > 1e-6 else 0
            
            # 计算胜率
            positive_returns = returns[returns > 0]
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            
            # 计算盈亏比
            profit_factor = abs(np.sum(positive_returns)) / abs(np.sum(returns[returns < 0])) if np.sum(returns[returns < 0]) != 0 else 1
            
            # 计算连续亏损和连续盈利
            consecutive_losses = 0
            consecutive_wins = 0
            current_loss_streak = 0
            current_win_streak = 0
            
            for ret in returns:
                if ret < 0:
                    current_loss_streak += 1
                    current_win_streak = 0
                    consecutive_losses = max(consecutive_losses, current_loss_streak)
                elif ret > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    consecutive_wins = max(consecutive_wins, current_win_streak)
            
            # 计算下行标准差和Sortino比率
            negative_returns = returns[returns < 0]
            downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = (mean_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # 计算VaR 95%
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            
            # 计算期望损失 (CVaR)
            var_threshold = np.percentile(returns, 5) if len(returns) > 0 else 0
            expected_shortfall = np.mean(returns[returns <= var_threshold]) if len(returns[returns <= var_threshold]) > 0 else 0
            
            return PerformanceMetrics(
                sharpe_ratio=sharpe_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=drawdown_metrics.max_drawdown,
                volatility=volatility,
                win_rate=win_rate,
                profit_factor=profit_factor,
                consecutive_losses=consecutive_losses,
                consecutive_wins=consecutive_wins,
                total_return=total_return,
                downside_deviation=downside_deviation,
                sortino_ratio=sortino_ratio,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                timestamp=timestamp
            )
        except Exception as e:
            self.logger.warning(f"计算表现指标失败: {e}")
            return None
    
    def _calculate_market_metrics(self, timestamp: datetime) -> Optional['MarketMetrics']:
        """计算市场指标数据"""
        try:
            from .adaptive_risk_budget import MarketMetrics
            
            # 从市场状态检测器获取市场指标
            if self.market_regime_detector is None:
                # 如果没有市场状态检测器，使用默认值
                return MarketMetrics(
                    market_volatility=0.15,  # 默认市场波动率
                    market_trend=0.0,        # 默认趋势中性
                    correlation_with_market=0.5,  # 默认中等相关性
                    liquidity_score=1.0,     # 默认流动性良好
                    uncertainty_index=0.3,   # 默认中等不确定性
                    regime_stability=0.8,    # 默认较高稳定性
                    timestamp=timestamp
                )
            
            # 从市场状态检测器获取实际指标
            market_data = getattr(self.market_regime_detector, 'market_data', {})
            
            market_volatility = market_data.get('volatility', 0.15)
            market_trend = market_data.get('trend', 0.0)
            correlation_with_market = market_data.get('correlation', 0.5)
            
            # 计算其他指标
            liquidity_score = 1.0 - min(market_volatility / 0.5, 1.0)  # 高波动率对应低流动性
            uncertainty_index = min(market_volatility * 2, 1.0)  # 波动率越高不确定性越大
            regime_stability = max(0.1, 1.0 - market_volatility)  # 波动率越低状态越稳定
            
            return MarketMetrics(
                market_volatility=market_volatility,
                market_trend=market_trend,
                correlation_with_market=correlation_with_market,
                liquidity_score=liquidity_score,
                uncertainty_index=uncertainty_index,
                regime_stability=regime_stability,
                timestamp=timestamp
            )
        except Exception as e:
            self.logger.warning(f"计算市场指标失败: {e}")
            return None
    
    def _optimize_reward_function(self, 
                                 portfolio_state: PortfolioState,
                                 drawdown_metrics: DrawdownMetrics,
                                 timestamp: datetime):
        """优化奖励函数"""
        # 计算风险调整奖励
        risk_adjusted_reward = self.reward_optimizer.calculate_risk_adjusted_reward(
            0.0,  # 当前收益（这里简化处理）
            drawdown_metrics.current_drawdown,
            portfolio_state.positions
        )
        
        # 生成奖励优化信号
        signal = ControlSignal(
            signal_type=ControlSignalType.REWARD_OPTIMIZATION,
            priority=ControlSignalPriority.LOW,
            timestamp=timestamp,
            source_component='reward_optimizer',
            content={
                'risk_adjusted_reward': risk_adjusted_reward,
                'drawdown_penalty': drawdown_metrics.current_drawdown * self.config.drawdown_penalty_factor
            }
        )
        self.control_signal_queue.append(signal)
    
    def _check_market_regime_change(self, market_state: MarketState, timestamp: datetime):
        """检查市场状态变化"""
        if not self.market_regime_detector:
            return
        
        # 这里简化处理，实际应该调用市场状态检测器
        signal = ControlSignal(
            signal_type=ControlSignalType.MARKET_REGIME_CHANGE,
            priority=ControlSignalPriority.MEDIUM,
            timestamp=timestamp,
            source_component='market_regime_detector',
            content={
                'regime': 'normal',  # 简化处理
                'confidence': 0.8
            }
        )
        self.control_signal_queue.append(signal)
    
    def update_market_data(self, market_data: Dict[str, Any]):
        """更新市场数据"""
        market_state = MarketState(
            prices=market_data.get('prices', {}),
            volumes=market_data.get('volumes', {}),
            timestamp=datetime.now(),
            market_indicators=market_data.get('indicators', {})
        )
        self.state_manager.update_market_state(market_state)
    
    def update_portfolio_state(self, portfolio_data: Dict[str, Any]):
        """更新投资组合状态"""
        portfolio_state = PortfolioState(
            positions=portfolio_data.get('positions', {}),
            portfolio_value=portfolio_data.get('portfolio_value', 0.0),
            cash=portfolio_data.get('cash', 0.0),
            timestamp=datetime.now(),
            unrealized_pnl=portfolio_data.get('unrealized_pnl', 0.0),
            realized_pnl=portfolio_data.get('realized_pnl', 0.0)
        )
        self.state_manager.update_portfolio_state(portfolio_state)
    
    def get_control_signals(self) -> List[Dict[str, Any]]:
        """获取控制信号"""
        return [signal.to_dict() for signal in self.control_signal_queue]
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        if not self.state_manager.current_portfolio_state:
            return {}
        
        # 获取最新的回撤指标
        portfolio_value = self.state_manager.current_portfolio_state.portfolio_value
        drawdown_metrics = self.drawdown_monitor.update_portfolio_value(portfolio_value)
        
        return {
            'current_drawdown': drawdown_metrics.current_drawdown,
            'max_drawdown': drawdown_metrics.max_drawdown,
            'drawdown_duration': drawdown_metrics.drawdown_duration,
            'current_phase': drawdown_metrics.current_phase.value,
            'risk_budget': getattr(self, '_last_risk_budget', self.config.base_risk_budget),
            'portfolio_value': portfolio_value,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_control_state(self):
        """重置控制状态"""
        self.control_signal_queue.clear()
        self.is_active = False
        self.last_update_time = None
        
        # 重置各组件状态
        if hasattr(self.drawdown_monitor, 'reset'):
            self.drawdown_monitor.reset()
        
        self.logger.info("控制状态已重置")
    
    def activate(self):
        """激活控制器"""
        self.is_active = True
        self.logger.info("回撤控制器已激活")
    
    def deactivate(self):
        """停用控制器"""
        self.is_active = False
        self.logger.info("回撤控制器已停用")
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """获取组件状态"""
        return {
            'drawdown_monitor': {
                'active': self.drawdown_monitor is not None,
                'last_update': getattr(self.drawdown_monitor, 'last_update_time', None)
            },
            'dynamic_stop_loss': {
                'active': self.dynamic_stop_loss is not None,
                'stop_loss_levels': getattr(self.dynamic_stop_loss, 'stop_loss_levels', {})
            },
            'reward_optimizer': {
                'active': self.reward_optimizer is not None
            },
            'adaptive_risk_budget': {
                'active': self.adaptive_risk_budget is not None,
                'current_budget': getattr(self, '_last_risk_budget', self.config.base_risk_budget)
            },
            'market_regime_detector': {
                'active': self.market_regime_detector is not None
            }
        }