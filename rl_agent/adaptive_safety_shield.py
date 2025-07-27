"""
自适应安全保护层
动态风险管理，平衡收益与安全
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class AdaptiveSafetyShield:
    """自适应安全保护层 - 智能风险管理"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 基础风险参数 - 更宽松的设置
        self.base_max_position = config.get('max_position', 0.25)  # 提升至25%
        self.base_max_leverage = config.get('max_leverage', 2.0)   # 提升至2倍
        self.base_var_threshold = config.get('var_threshold', 0.05) # 提升至5%
        self.base_drawdown_threshold = config.get('max_drawdown_threshold', 0.12) # 12%
        self.base_volatility_threshold = config.get('volatility_threshold', 0.30) # 30%
        
        # 动态调整参数
        self.risk_adaptation_factor = config.get('risk_adaptation_factor', 1.0)
        self.market_regime_sensitivity = config.get('market_regime_sensitivity', 0.3)
        self.performance_feedback_weight = config.get('performance_feedback_weight', 0.2)
        
        # 市场状态检测参数
        self.lookback_window = config.get('lookback_window', 60)
        self.volatility_window = config.get('volatility_window', 20)
        self.trend_window = config.get('trend_window', 30)
        
        # 历史数据存储
        self.return_history = []
        self.portfolio_history = []
        self.performance_history = []
        self.market_state_history = []
        
        # 动态风险阈值（会根据市场状态调整）
        self.current_max_position = self.base_max_position
        self.current_max_leverage = self.base_max_leverage
        self.current_var_threshold = self.base_var_threshold
        self.current_drawdown_threshold = self.base_drawdown_threshold
        self.current_volatility_threshold = self.base_volatility_threshold
        
        # 风险事件统计
        self.risk_stats = {
            'leverage_violations': 0,
            'position_violations': 0,
            'var_violations': 0,
            'drawdown_violations': 0,
            'volatility_violations': 0,
            'total_actions': 0,
            'constraint_rate': 0.0
        }
        
        # 性能追踪
        self.performance_tracker = {
            'recent_returns': [],
            'recent_sharpe': 0.0,
            'recent_max_dd': 0.0,
            'win_rate': 0.0,
            'risk_adjusted_return': 0.0
        }
        
        logger.info("初始化自适应安全保护层")
        
    def shield_action(self, action: np.ndarray, state: Dict, 
                     price_data: Optional[pd.DataFrame] = None,
                     current_portfolio: Optional[np.ndarray] = None,
                     recent_performance: Optional[float] = None) -> np.ndarray:
        """
        智能安全检查与动作修正
        
        Args:
            action: 原始动作
            state: 当前状态
            price_data: 价格数据  
            current_portfolio: 当前组合
            recent_performance: 近期表现
            
        Returns:
            修正后的安全动作
        """
        self.risk_stats['total_actions'] += 1
        
        # 1. 更新市场状态和动态阈值
        self._update_market_state(price_data, recent_performance)
        self._update_dynamic_thresholds()
        
        # 2. 复制并开始处理动作
        safe_action = action.copy()
        original_action = action.copy()
        
        # 3. 应用智能约束
        safe_action = self._apply_adaptive_position_constraints(safe_action)
        safe_action = self._apply_adaptive_leverage_constraints(safe_action, current_portfolio)
        
        if price_data is not None:
            safe_action = self._apply_adaptive_risk_constraints(safe_action, price_data, state)
        
        # 4. 渐进式约束调整
        safe_action = self._apply_progressive_constraints(safe_action, original_action)
        
        # 5. 更新统计信息
        self._update_constraint_statistics(original_action, safe_action)
        
        return safe_action
    
    def _update_market_state(self, price_data: Optional[pd.DataFrame], 
                           recent_performance: Optional[float]):
        """更新市场状态分析"""
        if price_data is None or len(price_data) < self.volatility_window:
            return
        
        # 计算市场指标
        recent_returns = price_data.pct_change().dropna()
        
        if len(recent_returns) < self.volatility_window:
            return
            
        # 1. 波动率状态
        recent_vol = recent_returns.iloc[-self.volatility_window:].std().mean()
        long_term_vol = recent_returns.std().mean() if len(recent_returns) > 60 else recent_vol
        vol_regime = recent_vol / (long_term_vol + 1e-8)
        
        # 2. 趋势状态  
        if len(recent_returns) >= self.trend_window:
            recent_trend = recent_returns.iloc[-self.trend_window:].mean().mean()
            trend_strength = abs(recent_trend) / (recent_vol + 1e-8)
        else:
            trend_strength = 0.0
        
        # 3. 市场稳定性
        if len(recent_returns) >= 5:
            stability = 1.0 / (1.0 + recent_returns.iloc[-5:].std().mean())
        else:
            stability = 0.5
        
        # 4. 综合市场状态评分
        market_state = {
            'volatility_regime': min(vol_regime, 3.0),  # 限制最大值
            'trend_strength': min(trend_strength, 2.0),
            'stability': stability,
            'composite_score': 0.4 * stability + 0.3 * (2.0 - vol_regime) + 0.3 * trend_strength
        }
        
        self.market_state_history.append(market_state)
        if len(self.market_state_history) > 100:
            self.market_state_history.pop(0)
        
        # 5. 更新性能追踪
        if recent_performance is not None:
            self.performance_tracker['recent_returns'].append(recent_performance)
            if len(self.performance_tracker['recent_returns']) > 30:
                self.performance_tracker['recent_returns'].pop(0)
            
            # 计算风险调整收益
            if len(self.performance_tracker['recent_returns']) >= 10:
                returns_array = np.array(self.performance_tracker['recent_returns'])
                self.performance_tracker['recent_sharpe'] = (
                    np.mean(returns_array) / (np.std(returns_array) + 1e-8) * np.sqrt(252)
                )
                self.performance_tracker['win_rate'] = np.mean(returns_array > 0)
                self.performance_tracker['risk_adjusted_return'] = np.mean(returns_array) * 252
    
    def _update_dynamic_thresholds(self):
        """根据市场状态动态调整风险阈值"""
        if not self.market_state_history:
            return
        
        current_state = self.market_state_history[-1]
        composite_score = current_state['composite_score']
        
        # 基于市场状态的调整因子
        # composite_score 越高表示市场越稳定，可以承受更多风险
        market_adjustment = 0.7 + 0.6 * min(max(composite_score, 0.0), 1.0)
        
        # 基于近期表现的调整因子  
        performance_adjustment = 1.0
        if len(self.performance_tracker['recent_returns']) >= 10:
            recent_sharpe = self.performance_tracker['recent_sharpe']
            win_rate = self.performance_tracker['win_rate']
            
            # 表现好时略微放宽限制，表现差时收紧
            performance_adjustment = 0.8 + 0.4 * min(max(win_rate, 0.0), 1.0)
            if recent_sharpe > 0.5:
                performance_adjustment *= 1.1
            elif recent_sharpe < -0.5:
                performance_adjustment *= 0.9
        
        # 综合调整因子
        total_adjustment = market_adjustment * performance_adjustment
        total_adjustment = min(max(total_adjustment, 0.6), 1.4)  # 限制调整范围
        
        # 更新动态阈值
        self.current_max_position = min(self.base_max_position * total_adjustment, 0.30)
        self.current_max_leverage = min(self.base_max_leverage * total_adjustment, 2.5)
        self.current_var_threshold = min(self.base_var_threshold * total_adjustment, 0.08)
        self.current_drawdown_threshold = min(self.base_drawdown_threshold * total_adjustment, 0.15)
        self.current_volatility_threshold = min(self.base_volatility_threshold * total_adjustment, 0.35)
        
        # 记录调整信息
        if self.risk_stats['total_actions'] % 1000 == 0:
            logger.info(f"动态风险阈值更新 - 调整因子: {total_adjustment:.3f}, "
                       f"仓位限制: {self.current_max_position:.3f}, "
                       f"杠杆限制: {self.current_max_leverage:.3f}")
    
    def _apply_adaptive_position_constraints(self, action: np.ndarray) -> np.ndarray:
        """应用自适应仓位约束"""
        # 计算各个仓位
        position_sizes = np.abs(action)
        
        # 检查是否违反仓位限制
        violations = position_sizes > self.current_max_position
        
        if violations.any():
            # 渐进式调整而非硬性限制
            scale_factor = self.current_max_position / np.max(position_sizes[violations])
            scale_factor = max(scale_factor, 0.8)  # 最小保留80%的原始信号
            
            action[violations] *= scale_factor
            self.risk_stats['position_violations'] += np.sum(violations)
        
        return action
    
    def _apply_adaptive_leverage_constraints(self, action: np.ndarray, 
                                           current_portfolio: Optional[np.ndarray]) -> np.ndarray:
        """应用自适应杠杆约束"""
        total_leverage = np.sum(np.abs(action))
        
        if total_leverage > self.current_max_leverage:
            # 使用平滑的缩放而非硬截断
            scale_factor = self.current_max_leverage / total_leverage
            scale_factor = max(scale_factor, 0.7)  # 保留至少70%的信号强度
            
            action *= scale_factor
            self.risk_stats['leverage_violations'] += 1
        
        return action
    
    def _apply_adaptive_risk_constraints(self, action: np.ndarray, 
                                       price_data: pd.DataFrame, state: Dict) -> np.ndarray:
        """应用自适应风险约束"""
        if len(price_data) < self.volatility_window:
            return action
        
        # 计算预期风险
        recent_returns = price_data.pct_change().dropna()
        if len(recent_returns) < 10:
            return action
        
        # 估算组合风险
        returns_sample = recent_returns.iloc[-self.volatility_window:]
        cov_matrix = returns_sample.cov().values
        
        # 确保维度匹配
        min_dim = min(len(action), cov_matrix.shape[0])
        action_truncated = action[:min_dim]
        cov_truncated = cov_matrix[:min_dim, :min_dim]
        
        # 计算组合方差
        try:
            portfolio_var = np.dot(action_truncated, np.dot(cov_truncated, action_truncated))
            portfolio_vol = np.sqrt(max(portfolio_var, 0.0))
            
            # VaR估计 (95%置信度)
            portfolio_var_95 = 1.645 * portfolio_vol
            
            # 如果风险过高，进行调整
            if portfolio_var_95 > self.current_var_threshold:
                risk_scale = self.current_var_threshold / portfolio_var_95
                risk_scale = max(risk_scale, 0.6)  # 保留至少60%的信号
                
                action[:min_dim] *= risk_scale
                self.risk_stats['var_violations'] += 1
                
        except Exception as e:
            logger.warning(f"风险计算错误: {e}")
        
        return action
    
    def _apply_progressive_constraints(self, safe_action: np.ndarray, 
                                     original_action: np.ndarray) -> np.ndarray:
        """应用渐进式约束，保持信号强度"""
        # 计算调整幅度
        adjustment_ratio = np.linalg.norm(safe_action) / (np.linalg.norm(original_action) + 1e-8)
        
        # 如果调整过于激烈，使用渐进式方法
        if adjustment_ratio < 0.5:
            # 保留更多原始信号，但确保不违反核心约束
            progressive_action = 0.7 * original_action + 0.3 * safe_action
            
            # 重新检查核心约束
            total_leverage = np.sum(np.abs(progressive_action))
            if total_leverage > self.current_max_leverage * 1.1:  # 允许10%的缓冲
                progressive_action *= (self.current_max_leverage * 1.1) / total_leverage
            
            return progressive_action
        
        return safe_action
    
    def _update_constraint_statistics(self, original_action: np.ndarray, 
                                    safe_action: np.ndarray):
        """更新约束统计信息"""
        # 计算约束率
        total_change = np.linalg.norm(original_action - safe_action)
        original_norm = np.linalg.norm(original_action)
        
        if original_norm > 1e-8:
            constraint_ratio = total_change / original_norm
            self.risk_stats['constraint_rate'] = (
                0.9 * self.risk_stats['constraint_rate'] + 0.1 * constraint_ratio
            )
    
    def get_constraint_statistics(self) -> Dict:
        """获取约束统计信息"""
        total_actions = max(self.risk_stats['total_actions'], 1)
        
        return {
            'total_actions': total_actions,
            'leverage_violation_rate': self.risk_stats['leverage_violations'] / total_actions,
            'position_violation_rate': self.risk_stats['position_violations'] / total_actions,
            'var_violation_rate': self.risk_stats['var_violations'] / total_actions,
            'average_constraint_rate': self.risk_stats['constraint_rate'],
            'current_thresholds': {
                'max_position': self.current_max_position,
                'max_leverage': self.current_max_leverage,
                'var_threshold': self.current_var_threshold,
                'drawdown_threshold': self.current_drawdown_threshold
            },
            'performance_metrics': self.performance_tracker
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        for key in self.risk_stats:
            if key != 'constraint_rate':
                self.risk_stats[key] = 0
        
        logger.info("风险约束统计信息已重置")
    
    def get_risk_budget_utilization(self) -> Dict:
        """获取风险预算利用率"""
        return {
            'position_utilization': np.mean([
                min(1.0, pos / self.current_max_position) 
                for pos in [0.15, 0.20]  # 示例仓位
            ]),
            'leverage_utilization': min(1.0, 1.5 / self.current_max_leverage),
            'var_utilization': min(1.0, 0.03 / self.current_var_threshold),
            'overall_risk_efficiency': self.performance_tracker.get('risk_adjusted_return', 0.0)
        }