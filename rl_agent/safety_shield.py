"""
安全保护层 - Shielded RL实现
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from utils.logging_utils import throttled_warning, statistical_warning

logger = logging.getLogger(__name__)


class SafetyShield:
    """安全保护层"""
    
    def __init__(self, config: Dict):
        """
        初始化安全保护层
        
        Args:
            config: 保护层配置
        """
        self.config = config
        
        # 风险约束参数 - 调整为更宽松的设置
        self.max_position = config.get('max_position', 0.15)
        self.max_leverage = config.get('max_leverage', 1.5)
        self.var_threshold = config.get('var_threshold', 0.025)
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.08)
        self.volatility_threshold = config.get('volatility_threshold', 0.20)
        
        # 风险预测器参数
        self.lookback_window = config.get('lookback_window', 20)
        
        # 历史数据用于风险预测
        self.price_history = []
        self.return_history = []
        self.portfolio_history = []
        
        # 收益率缓存 - 避免重复计算
        self._cached_returns = None
        self._cache_timestamp = None
        
        # 风险事件统计
        self.risk_event_counts = {
            'leverage_constraint': 0,
            'var_constraint': 0,
            'volatility_constraint': 0,
            'drawdown_constraint': 0,
            'market_regime': 0
        }
        
    def shield_action(self, 
                     action: np.ndarray, 
                     state: Dict,
                     price_data: Optional[pd.DataFrame] = None,
                     current_portfolio: Optional[np.ndarray] = None) -> np.ndarray:
        """
        安全检查与动作修正
        
        Args:
            action: 原始动作
            state: 当前状态信息
            price_data: 价格数据
            current_portfolio: 当前组合权重
            
        Returns:
            修正后的安全动作
        """
        # 复制原始动作
        safe_action = action.copy()
        
        # 1. 基础约束检查
        safe_action = self._apply_basic_constraints(safe_action)
        
        # 2. VaR风险检查
        if price_data is not None:
            safe_action = self._check_var_constraint(safe_action, price_data, state)
        
        # 3. 波动率检查
        if current_portfolio is not None:
            safe_action = self._check_volatility_constraint(safe_action, current_portfolio, state)
        
        # 4. 回撤风险检查
        safe_action = self._check_drawdown_constraint(safe_action, state)
        
        # 5. 极端市场条件检查
        safe_action = self._check_market_regime(safe_action, state)
        
        return safe_action
    
    def _apply_basic_constraints(self, action: np.ndarray) -> np.ndarray:
        """应用基础约束"""
        # 单股票仓位限制
        action = np.clip(action, -self.max_position, self.max_position)
        
        # 总杠杆限制
        total_leverage = np.sum(np.abs(action))
        if total_leverage > self.max_leverage:
            scaling_factor = self.max_leverage / total_leverage
            action = action * scaling_factor
            self.risk_event_counts['leverage_constraint'] += 1
            
            # 使用限制器控制日志频率
            throttled_warning(
                logger,
                f"杠杆约束触发，缩放因子: {scaling_factor:.3f}",
                "leverage_constraint",
                min_interval=10.0,  # 10秒间隔
                max_per_minute=2    # 每分钟最多2次
            )
        
        return action
    
    def _check_var_constraint(self, 
                            action: np.ndarray, 
                            price_data: pd.DataFrame,
                            state: Dict) -> np.ndarray:
        """检查VaR约束"""
        if len(price_data) < self.lookback_window:
            return action
        
        # 使用缓存避免重复计算收益率
        current_timestamp = id(price_data)
        if (self._cached_returns is None or 
            self._cache_timestamp != current_timestamp or 
            len(self._cached_returns) != len(price_data)):
            
            # 只计算必要的收益率（最近的lookback_window期间）
            recent_prices = price_data.iloc[-min(self.lookback_window + 1, len(price_data)):]
            self._cached_returns = recent_prices.pct_change(fill_method=None).dropna()
            self._cache_timestamp = current_timestamp
        
        if len(self._cached_returns) < self.lookback_window:
            return action
            
        # 使用最近的数据计算协方差矩阵
        recent_returns = self._cached_returns.iloc[-self.lookback_window:]
        
        # 简化协方差计算 - 只使用对角线（独立资产假设）
        if len(recent_returns.columns) != len(action):
            return action
            
        # 使用简化的风险模型：只考虑波动率，不考虑相关性
        volatilities = recent_returns.std().values
        portfolio_volatility = np.sqrt(np.sum((action * volatilities) ** 2))
        predicted_var = -1.645 * portfolio_volatility  # 95% VaR
        
        # 如果VaR超出阈值，减少仓位
        if abs(predicted_var) > self.var_threshold:
            reduction_factor = self.var_threshold / abs(predicted_var)
            action = action * reduction_factor
            self.risk_event_counts['var_constraint'] += 1
            
            # 使用限制器控制日志频率
            throttled_warning(
                logger,
                f"VaR约束触发，预测VaR: {predicted_var:.4f}, 减少因子: {reduction_factor:.3f}",
                "var_constraint",
                min_interval=15.0,  # 15秒间隔
                max_per_minute=2    # 每分钟最多2次
            )
        
        return action
    
    def _check_volatility_constraint(self, 
                                   action: np.ndarray,
                                   current_portfolio: np.ndarray, 
                                   state: Dict) -> np.ndarray:
        """检查波动率约束"""
        # 获取当前波动率
        current_vol = state.get('portfolio_volatility', 0.0)
        
        # 如果当前波动率超过阈值，倾向于减少仓位变化
        if current_vol > self.volatility_threshold:
            # 计算仓位变化幅度
            position_changes = np.abs(action - current_portfolio)
            max_change = np.max(position_changes)
            
            if max_change > 0.05:  # 单次最大调整5%
                scaling_factor = 0.05 / max_change
                action = current_portfolio + (action - current_portfolio) * scaling_factor
                self.risk_event_counts['volatility_constraint'] += 1
                
                # 使用限制器控制日志频率
                throttled_warning(
                    logger,
                    f"波动率约束触发，当前波动率: {current_vol:.4f}, 缩放因子: {scaling_factor:.3f}",
                    "volatility_constraint",
                    min_interval=15.0,  # 15秒间隔
                    max_per_minute=2    # 每分钟最多2次
                )
        
        return action
    
    def _check_drawdown_constraint(self, action: np.ndarray, state: Dict) -> np.ndarray:
        """检查回撤约束"""
        current_drawdown = state.get('max_drawdown', 0.0)
        
        # 如果当前回撤接近阈值，强制减仓
        if current_drawdown > self.max_drawdown_threshold * 0.8:  # 80%阈值预警
            # 减少总仓位
            reduction_factor = 0.5  # 减仓50%
            action = action * reduction_factor
            self.risk_event_counts['drawdown_constraint'] += 1
            
            # 使用限制器控制日志频率
            throttled_warning(
                logger,
                f"回撤约束触发，当前回撤: {current_drawdown:.4f}, 强制减仓至: {reduction_factor}",
                "drawdown_constraint",
                min_interval=5.0,   # 5秒间隔（回撤比较紧急）
                max_per_minute=3    # 每分钟最多3次
            )
        
        return action
    
    def _check_market_regime(self, action: np.ndarray, state: Dict) -> np.ndarray:
        """检查市场环境约束"""
        # 模拟市场状态检测
        market_volatility = state.get('market_volatility', 0.0)
        
        # 在高波动市场环境下，降低整体仓位
        if market_volatility > 0.3:  # 假设30%为高波动阈值
            defensive_factor = 0.7
            action = action * defensive_factor
            self.risk_event_counts['market_regime'] += 1
            
            # 使用限制器控制日志频率
            throttled_warning(
                logger,
                f"高波动市场环境，市场波动率: {market_volatility:.4f}, 防守因子: {defensive_factor}",
                "market_regime",
                min_interval=20.0,  # 20秒间隔
                max_per_minute=1    # 每分钟最多1次
            )
        
        return action
    
    def update_history(self, 
                      portfolio_weights: np.ndarray,
                      portfolio_return: float,
                      price_data: Optional[pd.DataFrame] = None):
        """更新历史数据"""
        self.portfolio_history.append(portfolio_weights.copy())
        self.return_history.append(portfolio_return)
        
        if price_data is not None:
            self.price_history.append(price_data.iloc[-1].values.copy())
        
        # 保持历史数据长度
        max_history = self.lookback_window * 2
        if len(self.portfolio_history) > max_history:
            self.portfolio_history = self.portfolio_history[-max_history:]
            self.return_history = self.return_history[-max_history:]
            if self.price_history:
                self.price_history = self.price_history[-max_history:]
    
    def get_risk_metrics(self) -> Dict:
        """获取风险指标统计"""
        if len(self.return_history) < 2:
            return {}
        
        returns = np.array(self.return_history)
        
        return {
            'portfolio_volatility': np.std(returns) * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
            'max_drawdown': self._calculate_max_drawdown(returns),
            'total_positions': len(self.portfolio_history),
            'avg_leverage': np.mean([np.sum(np.abs(w)) for w in self.portfolio_history[-10:]])
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return np.min(drawdowns)
    
    def is_action_safe(self, action: np.ndarray, state: Dict) -> Tuple[bool, str]:
        """
        判断动作是否安全
        
        Returns:
            (is_safe, reason)
        """
        # 杠杆检查
        total_leverage = np.sum(np.abs(action))
        if total_leverage > self.max_leverage:
            return False, f"总杠杆超限: {total_leverage:.3f} > {self.max_leverage}"
        
        # 单股票仓位检查
        max_position = np.max(np.abs(action))
        if max_position > self.max_position:
            return False, f"单股票仓位超限: {max_position:.3f} > {self.max_position}"
        
        # 回撤检查
        current_drawdown = state.get('max_drawdown', 0.0)
        if current_drawdown > self.max_drawdown_threshold:
            return False, f"回撤超限: {current_drawdown:.4f} > {self.max_drawdown_threshold}"
        
        return True, "安全"
    
    def get_risk_event_summary(self) -> str:
        """获取风险事件统计摘要"""
        total_events = sum(self.risk_event_counts.values())
        if total_events == 0:
            return "无风险约束触发事件"
        
        summary_lines = [f"风险约束触发统计 (总计: {total_events})"]
        for event_type, count in self.risk_event_counts.items():
            if count > 0:
                percentage = (count / total_events) * 100
                event_name = {
                    'leverage_constraint': '杠杆约束',
                    'var_constraint': 'VaR约束', 
                    'volatility_constraint': '波动率约束',
                    'drawdown_constraint': '回撤约束',
                    'market_regime': '市场环境调整'
                }.get(event_type, event_type)
                summary_lines.append(f"  - {event_name}: {count}次 ({percentage:.1f}%)")
        
        return '\n'.join(summary_lines)
    
    def reset_risk_event_counts(self):
        """重置风险事件计数"""
        for key in self.risk_event_counts:
            self.risk_event_counts[key] = 0