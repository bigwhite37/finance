"""
增强奖励系统
稳定、高质量的奖励信号设计，专注于8%年化收益目标
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque

logger = logging.getLogger(__name__)


class EnhancedRewardSystem:
    """增强奖励系统 - 稳定高效的奖励计算"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 奖励组件权重配置
        self.weights = {
            'return_base': config.get('return_weight', 1.0),
            'sharpe_bonus': config.get('sharpe_weight', 0.3),
            'consistency_bonus': config.get('consistency_weight', 0.2),
            'momentum_bonus': config.get('momentum_weight', 0.15),
            'efficiency_bonus': config.get('efficiency_weight', 0.1),
            'risk_penalty': config.get('risk_penalty_weight', 0.1),
            'cost_penalty': config.get('cost_penalty_weight', 0.05)
        }
        
        # 目标设定
        self.target_annual_return = config.get('target_annual_return', 0.08)  # 8%目标
        self.target_sharpe_ratio = config.get('target_sharpe_ratio', 1.0)
        self.max_acceptable_drawdown = config.get('max_drawdown', 0.12)
        
        # 历史数据窗口
        self.short_window = config.get('short_window', 10)
        self.medium_window = config.get('medium_window', 30) 
        self.long_window = config.get('long_window', 60)
        
        # 奖励历史记录
        self.reward_history = deque(maxlen=1000)
        self.component_history = {
            'return_base': deque(maxlen=100),
            'sharpe_bonus': deque(maxlen=100),
            'consistency_bonus': deque(maxlen=100),
            'momentum_bonus': deque(maxlen=100),
            'efficiency_bonus': deque(maxlen=100),
            'risk_penalty': deque(maxlen=100),
            'cost_penalty': deque(maxlen=100)
        }
        
        # 奖励标准化参数
        self.reward_normalizer = {
            'mean': 0.0,
            'std': 1.0,
            'update_rate': 0.01
        }
        
        # 性能追踪
        self.performance_tracker = {
            'cumulative_return': 0.0,
            'recent_sharpe': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 1.0
        }
        
        logger.info("初始化增强奖励系统")
    
    def calculate_reward(self, 
                        current_return: float,
                        portfolio_returns: List[float],
                        transaction_costs: float,
                        portfolio_value: float,
                        peak_value: float,
                        market_return: Optional[float] = None) -> Tuple[float, Dict]:
        """
        计算综合奖励
        
        Args:
            current_return: 当期收益率
            portfolio_returns: 历史收益序列
            transaction_costs: 交易成本
            portfolio_value: 当前组合价值
            peak_value: 历史最高价值
            market_return: 市场基准收益
            
        Returns:
            (总奖励, 奖励组件详情)
        """
        # 转换为numpy数组便于计算
        returns_array = np.array(portfolio_returns) if portfolio_returns else np.array([current_return])
        
        # 1. 基础收益奖励 - 核心组件
        return_reward = self._calculate_return_reward(current_return, returns_array)
        
        # 2. 夏普比率奖励 - 风险调整收益
        sharpe_bonus = self._calculate_sharpe_bonus(returns_array)
        
        # 3. 一致性奖励 - 稳定盈利能力
        consistency_bonus = self._calculate_consistency_bonus(returns_array)
        
        # 4. 动量奖励 - 趋势跟踪能力
        momentum_bonus = self._calculate_momentum_bonus(returns_array)
        
        # 5. 效率奖励 - 相对基准表现
        efficiency_bonus = self._calculate_efficiency_bonus(current_return, market_return)
        
        # 6. 风险惩罚 - 回撤控制
        risk_penalty = self._calculate_risk_penalty(portfolio_value, peak_value)
        
        # 7. 成本惩罚 - 交易效率
        cost_penalty = self._calculate_cost_penalty(transaction_costs, current_return)
        
        # 组合奖励组件
        reward_components = {
            'return_base': return_reward,
            'sharpe_bonus': sharpe_bonus,
            'consistency_bonus': consistency_bonus,
            'momentum_bonus': momentum_bonus,
            'efficiency_bonus': efficiency_bonus,
            'risk_penalty': risk_penalty,
            'cost_penalty': cost_penalty
        }
        
        # 计算加权总奖励
        total_reward = (
            self.weights['return_base'] * return_reward +
            self.weights['sharpe_bonus'] * sharpe_bonus +
            self.weights['consistency_bonus'] * consistency_bonus +
            self.weights['momentum_bonus'] * momentum_bonus +
            self.weights['efficiency_bonus'] * efficiency_bonus -
            self.weights['risk_penalty'] * risk_penalty -
            self.weights['cost_penalty'] * cost_penalty
        )
        
        # 应用奖励标准化
        normalized_reward = self._normalize_reward(total_reward)
        
        # 更新历史记录
        self._update_history(reward_components, normalized_reward)
        
        # 更新性能追踪
        self._update_performance_tracking(current_return, portfolio_value, peak_value)
        
        return normalized_reward, reward_components
    
    def _calculate_return_reward(self, current_return: float, returns_array: np.ndarray) -> float:
        """计算基础收益奖励"""
        # 使用对数收益形式，更加稳定
        if current_return > 0:
            # 正收益：对数缩放 + 目标导向
            log_return = np.log(1 + current_return)
            target_ratio = current_return * 252 / self.target_annual_return  # 年化后与目标比较
            return log_return * (1 + min(target_ratio, 2.0))  # 限制最大倍数
        else:
            # 负收益：轻微惩罚，避免过度规避风险
            return current_return * 0.5
    
    def _calculate_sharpe_bonus(self, returns_array: np.ndarray) -> float:
        """计算夏普比率奖励"""
        if len(returns_array) < self.short_window:
            return 0.0
        
        # 使用不同时间窗口的夏普比率
        windows = [min(self.short_window, len(returns_array)), 
                  min(self.medium_window, len(returns_array))]
        
        sharpe_scores = []
        for window in windows:
            recent_returns = returns_array[-window:]
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return > 1e-8:
                sharpe = mean_return / std_return * np.sqrt(252)
                sharpe_scores.append(sharpe)
        
        if not sharpe_scores:
            return 0.0
        
        # 平均夏普比率，并与目标比较
        avg_sharpe = np.mean(sharpe_scores)
        sharpe_bonus = max(0, avg_sharpe / self.target_sharpe_ratio - 0.5)  # 超过一半目标开始奖励
        
        return min(sharpe_bonus, 2.0)  # 限制最大奖励
    
    def _calculate_consistency_bonus(self, returns_array: np.ndarray) -> float:
        """计算一致性奖励"""
        if len(returns_array) < self.short_window:
            return 0.0
        
        recent_returns = returns_array[-self.short_window:]
        
        # 胜率计算
        win_rate = np.mean(recent_returns > 0)
        
        # 收益稳定性（变异系数的倒数）
        mean_return = np.mean(recent_returns)
        if abs(mean_return) > 1e-8:
            cv = np.std(recent_returns) / abs(mean_return)
            stability = 1.0 / (1.0 + cv)
        else:
            stability = 0.0
        
        # 综合一致性得分
        consistency_score = 0.6 * win_rate + 0.4 * stability
        
        # 只有在正收益时才给予一致性奖励
        if mean_return > 0:
            return consistency_score
        else:
            return 0.0
    
    def _calculate_momentum_bonus(self, returns_array: np.ndarray) -> float:
        """计算动量奖励"""
        if len(returns_array) < 5:
            return 0.0
        
        # 短期动量
        short_momentum = np.mean(returns_array[-5:])
        
        # 中期动量（如果数据足够）
        if len(returns_array) >= 15:
            medium_momentum = np.mean(returns_array[-15:]) 
            # 组合动量信号
            combined_momentum = 0.7 * short_momentum + 0.3 * medium_momentum
        else:
            combined_momentum = short_momentum
        
        # 只奖励正动量
        if combined_momentum > 0:
            return min(combined_momentum * 5.0, 1.0)  # 放大但限制范围
        else:
            return 0.0
    
    def _calculate_efficiency_bonus(self, current_return: float, 
                                  market_return: Optional[float]) -> float:
        """计算效率奖励（相对基准表现）"""
        if market_return is None:
            return 0.0
        
        # 超额收益
        excess_return = current_return - market_return
        
        # 信息比率导向的奖励
        if excess_return > 0:
            return min(excess_return * 2.0, 0.5)  # 奖励超额收益
        else:
            return max(excess_return * 0.5, -0.2)  # 轻微惩罚负超额收益
    
    def _calculate_risk_penalty(self, portfolio_value: float, peak_value: float) -> float:
        """计算风险惩罚"""
        if peak_value <= 0:
            return 0.0
        
        current_drawdown = (peak_value - portfolio_value) / peak_value
        
        # 分级惩罚系统
        if current_drawdown <= 0.02:  # 2%以内无惩罚
            return 0.0
        elif current_drawdown <= 0.05:  # 2-5%轻微惩罚
            return (current_drawdown - 0.02) * 2.0
        elif current_drawdown <= self.max_acceptable_drawdown:  # 5-12%渐进惩罚
            return 0.06 + (current_drawdown - 0.05) * 5.0
        else:  # 超过12%重度惩罚
            return 0.41 + (current_drawdown - self.max_acceptable_drawdown) * 10.0
    
    def _calculate_cost_penalty(self, transaction_costs: float, current_return: float) -> float:
        """计算成本惩罚"""
        if abs(current_return) < 1e-8:
            return transaction_costs * 2.0  # 无收益时成本惩罚加倍
        
        # 成本收益比
        cost_ratio = transaction_costs / abs(current_return)
        
        # 分级成本惩罚
        if cost_ratio <= 0.1:  # 成本低于收益10%
            return transaction_costs * 0.5
        elif cost_ratio <= 0.2:  # 成本在收益10-20%
            return transaction_costs * 1.0
        else:  # 成本超过收益20%
            return transaction_costs * 2.0
    
    def _normalize_reward(self, reward: float) -> float:
        """奖励标准化"""
        # 更新标准化参数
        self.reward_normalizer['mean'] = (
            (1 - self.reward_normalizer['update_rate']) * self.reward_normalizer['mean'] +
            self.reward_normalizer['update_rate'] * reward
        )
        
        if len(self.reward_history) > 10:
            recent_rewards = list(self.reward_history)[-50:]  # 使用最近50个奖励
            current_std = np.std(recent_rewards)
            self.reward_normalizer['std'] = (
                (1 - self.reward_normalizer['update_rate']) * self.reward_normalizer['std'] +
                self.reward_normalizer['update_rate'] * max(current_std, 0.1)
            )
        
        # 标准化
        normalized = (reward - self.reward_normalizer['mean']) / self.reward_normalizer['std']
        
        # 限制范围防止极值
        return np.clip(normalized, -5.0, 5.0)
    
    def _update_history(self, components: Dict, total_reward: float):
        """更新历史记录"""
        self.reward_history.append(total_reward)
        
        for key, value in components.items():
            if key in self.component_history:
                self.component_history[key].append(value)
    
    def _update_performance_tracking(self, current_return: float, 
                                   portfolio_value: float, peak_value: float):
        """更新性能追踪"""
        # 累积收益
        self.performance_tracker['cumulative_return'] = (portfolio_value - 1.0) if portfolio_value > 0 else 0.0
        
        # 最大回撤
        if peak_value > 0:
            self.performance_tracker['max_drawdown'] = max(
                self.performance_tracker['max_drawdown'],
                (peak_value - portfolio_value) / peak_value
            )
        
        # 胜率（基于最近收益）
        if len(self.reward_history) >= 10:
            recent_returns = [r for r in list(self.reward_history)[-10:] if abs(r) > 1e-8]
            if recent_returns:
                self.performance_tracker['win_rate'] = np.mean([r > 0 for r in recent_returns])
    
    def get_reward_analysis(self) -> Dict:
        """获取奖励分析报告"""
        if not self.reward_history:
            return {'status': 'No reward history available'}
        
        recent_rewards = list(self.reward_history)[-50:]
        
        analysis = {
            'reward_statistics': {
                'mean': np.mean(recent_rewards),
                'std': np.std(recent_rewards),
                'min': np.min(recent_rewards),
                'max': np.max(recent_rewards)
            },
            'component_contributions': {},
            'performance_tracking': self.performance_tracker.copy(),
            'reward_stability': {
                'recent_volatility': np.std(recent_rewards[-10:]) if len(recent_rewards) >= 10 else 0.0,
                'trend': np.mean(recent_rewards[-10:]) - np.mean(recent_rewards[-20:-10]) if len(recent_rewards) >= 20 else 0.0
            }
        }
        
        # 组件贡献分析
        for component, history in self.component_history.items():
            if history:
                analysis['component_contributions'][component] = {
                    'mean': np.mean(list(history)[-20:]),
                    'contribution_ratio': np.mean(list(history)[-20:]) / (np.mean(recent_rewards) + 1e-8)
                }
        
        return analysis
    
    def adjust_weights(self, performance_feedback: Dict):
        """根据性能反馈调整权重"""
        current_annual_return = performance_feedback.get('annual_return', 0.0)
        current_sharpe = performance_feedback.get('sharpe_ratio', 0.0)
        
        # 如果收益不足，增加收益组件权重
        if current_annual_return < self.target_annual_return * 0.5:
            self.weights['return_base'] = min(self.weights['return_base'] * 1.1, 2.0)
            self.weights['momentum_bonus'] = min(self.weights['momentum_bonus'] * 1.2, 0.3)
        
        # 如果夏普比率不足，增加一致性权重
        if current_sharpe < self.target_sharpe_ratio * 0.5:
            self.weights['consistency_bonus'] = min(self.weights['consistency_bonus'] * 1.1, 0.4)
        
        logger.info(f"奖励权重已调整: {self.weights}")
    
    def reset_system(self):
        """重置奖励系统"""
        self.reward_history.clear()
        for history in self.component_history.values():
            history.clear()
        
        self.reward_normalizer = {'mean': 0.0, 'std': 1.0, 'update_rate': 0.01}
        self.performance_tracker = {
            'cumulative_return': 0.0, 'recent_sharpe': 0.0, 
            'max_drawdown': 0.0, 'win_rate': 0.0, 'profit_factor': 1.0
        }
        
        logger.info("奖励系统已重置")