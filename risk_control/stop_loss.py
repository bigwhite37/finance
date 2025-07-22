"""
动态止损机制
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class DynamicStopLoss:
    """动态止损与再平衡"""
    
    def __init__(self, config: Dict):
        """
        初始化动态止损管理器
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 止损参数
        self.stop_loss_pct = config.get('stop_loss_pct', 0.03)  # 3%止损
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.05)  # 5%移动止损
        self.max_drawdown_stop = config.get('max_drawdown_stop', 0.08)  # 8%最大回撤止损
        
        # 止损状态
        self.trailing_high = None
        self.initial_value = None
        self.stop_loss_triggered = False
        self.trigger_history = []
        
        # 再平衡参数
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # 5%偏离触发再平衡
        self.rebalance_frequency = config.get('rebalance_frequency', 20)  # 20个交易日强制再平衡
        self.last_rebalance = 0
        
    def check_stop_loss(self, current_nav: float) -> bool:
        """
        检查是否触发止损
        
        Args:
            current_nav: 当前净值
            
        Returns:
            是否触发止损
        """
        if self.initial_value is None:
            self.initial_value = current_nav
            
        if self.trailing_high is None:
            self.trailing_high = current_nav
        
        # 更新追踪高点
        if current_nav > self.trailing_high:
            self.trailing_high = current_nav
            
        # 计算回撤
        drawdown_from_high = (self.trailing_high - current_nav) / self.trailing_high
        drawdown_from_initial = (self.initial_value - current_nav) / self.initial_value
        
        # 检查各种止损条件
        stop_triggers = []
        
        # 1. 移动止损
        if drawdown_from_high > self.trailing_stop_pct:
            stop_triggers.append(f'移动止损触发: {drawdown_from_high:.2%}')
            
        # 2. 固定止损
        if drawdown_from_initial > self.stop_loss_pct:
            stop_triggers.append(f'固定止损触发: {drawdown_from_initial:.2%}')
            
        # 3. 最大回撤止损
        if drawdown_from_high > self.max_drawdown_stop:
            stop_triggers.append(f'最大回撤止损触发: {drawdown_from_high:.2%}')
        
        # 记录触发
        if stop_triggers:
            self.stop_loss_triggered = True
            trigger_info = {
                'timestamp': pd.Timestamp.now(),
                'nav': current_nav,
                'drawdown_from_high': drawdown_from_high,
                'drawdown_from_initial': drawdown_from_initial,
                'triggers': stop_triggers
            }
            self.trigger_history.append(trigger_info)
            
            for trigger in stop_triggers:
                logger.warning(trigger)
            
            return True
        
        return False
    
    def check_rebalance_signal(self, 
                             current_weights: np.ndarray,
                             target_weights: np.ndarray,
                             current_step: int) -> bool:
        """
        检查是否需要再平衡
        
        Args:
            current_weights: 当前权重
            target_weights: 目标权重
            current_step: 当前步数
            
        Returns:
            是否需要再平衡
        """
        # 1. 强制再平衡检查
        if current_step - self.last_rebalance >= self.rebalance_frequency:
            logger.info(f"强制再平衡触发: {current_step - self.last_rebalance}天")
            self.last_rebalance = current_step
            return True
        
        # 2. 偏离阈值检查
        weight_deviation = np.sum(np.abs(current_weights - target_weights))
        if weight_deviation > self.rebalance_threshold:
            logger.info(f"权重偏离再平衡触发: {weight_deviation:.3f}")
            self.last_rebalance = current_step
            return True
        
        return False
    
    def apply_stop_loss_adjustment(self, 
                                 portfolio_weights: np.ndarray,
                                 reduction_factor: float = 0.5) -> np.ndarray:
        """
        应用止损调整
        
        Args:
            portfolio_weights: 当前权重
            reduction_factor: 减仓因子
            
        Returns:
            调整后的权重
        """
        if not self.stop_loss_triggered:
            return portfolio_weights
            
        # 按比例减仓
        adjusted_weights = portfolio_weights * reduction_factor
        
        logger.info(f"应用止损减仓: {reduction_factor:.1%}")
        
        return adjusted_weights
    
    def get_protective_position_size(self, 
                                   base_position: float,
                                   volatility: float,
                                   confidence_level: float = 0.95) -> float:
        """
        计算保护性仓位大小
        
        Args:
            base_position: 基础仓位
            volatility: 波动率
            confidence_level: 置信度
            
        Returns:
            调整后的仓位大小
        """
        # Kelly公式的简化版本
        # 考虑风险调整后的仓位
        
        if volatility <= 0:
            return base_position
        
        # 基于VaR的仓位调整
        z_score = 1.645 if confidence_level == 0.95 else 2.33  # 95%或99%
        max_loss_per_position = z_score * volatility
        
        # 限制单日最大损失在1%以内
        max_daily_loss = 0.01
        
        if max_loss_per_position > max_daily_loss:
            position_scaling = max_daily_loss / max_loss_per_position
            adjusted_position = base_position * position_scaling
        else:
            adjusted_position = base_position
        
        return adjusted_position
    
    def implement_circuit_breaker(self, 
                                 returns: List[float],
                                 threshold: float = -0.05) -> bool:
        """
        实施熔断机制
        
        Args:
            returns: 最近收益率列表
            threshold: 熔断阈值
            
        Returns:
            是否触发熔断
        """
        if len(returns) < 1:
            return False
        
        # 单日跌幅熔断
        if returns[-1] < threshold:
            logger.error(f"单日跌幅熔断触发: {returns[-1]:.2%}")
            return True
        
        # 连续下跌熔断
        if len(returns) >= 3:
            recent_returns = returns[-3:]
            if all(r < -0.01 for r in recent_returns):  # 连续3日跌超1%
                logger.error("连续下跌熔断触发")
                return True
        
        return False
    
    def calculate_stop_loss_levels(self, current_price: float) -> Dict[str, float]:
        """
        计算各类止损点位
        
        Args:
            current_price: 当前价格
            
        Returns:
            止损点位字典
        """
        levels = {
            'fixed_stop': current_price * (1 - self.stop_loss_pct),
            'trailing_stop': self.trailing_high * (1 - self.trailing_stop_pct) if self.trailing_high else current_price,
            'max_drawdown_stop': self.trailing_high * (1 - self.max_drawdown_stop) if self.trailing_high else current_price
        }
        
        return levels
    
    def get_stop_loss_stats(self) -> Dict:
        """获取止损统计信息"""
        return {
            'trigger_count': len(self.trigger_history),
            'current_trailing_high': self.trailing_high,
            'initial_value': self.initial_value,
            'stop_loss_active': self.stop_loss_triggered,
            'last_rebalance_step': self.last_rebalance,
            'trigger_history': self.trigger_history[-5:]  # 最近5次触发
        }
    
    def get_trigger_count(self) -> int:
        """获取触发次数"""
        return len(self.trigger_history)
    
    def reset_stop_loss(self, new_nav: Optional[float] = None):
        """
        重置止损状态
        
        Args:
            new_nav: 新的基准净值
        """
        if new_nav is not None:
            self.initial_value = new_nav
            self.trailing_high = new_nav
        else:
            self.initial_value = None
            self.trailing_high = None
            
        self.stop_loss_triggered = False
        logger.info("止损状态已重置")
    
    def reset(self):
        """完全重置"""
        self.trailing_high = None
        self.initial_value = None
        self.stop_loss_triggered = False
        self.trigger_history.clear()
        self.last_rebalance = 0