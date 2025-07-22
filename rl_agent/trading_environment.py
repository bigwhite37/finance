"""
强化学习交易环境
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """A股交易强化学习环境"""
    
    def __init__(self, 
                 factor_data: pd.DataFrame,
                 price_data: pd.DataFrame,
                 config: Dict):
        """
        初始化交易环境
        
        Args:
            factor_data: 因子数据
            price_data: 价格数据
            config: 环境配置
        """
        super().__init__()
        
        self.factor_data = factor_data
        self.price_data = price_data
        self.config = config
        
        # 环境参数
        self.n_stocks = len(price_data.columns)
        self.n_factors = len(factor_data.columns)
        self.lookback_window = config.get('lookback_window', 20)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.max_position = config.get('max_position', 0.1)
        self.max_leverage = config.get('max_leverage', 1.2)
        
        # 状态和动作空间
        self._setup_spaces()
        
        # 交易状态
        self.current_step = 0
        self.portfolio_weights = np.zeros(self.n_stocks)
        self.cash = 1.0  # 初始现金比例
        self.portfolio_value = 1.0
        self.portfolio_returns = []
        self.max_drawdown = 0.0
        self.peak_value = 1.0
        
        # 奖励参数
        self.lambda1 = config.get('lambda1', 2.0)  # 回撤惩罚
        self.lambda2 = config.get('lambda2', 1.0)  # CVaR惩罚
        self.max_dd_threshold = config.get('max_dd_threshold', 0.05)
        
    def _setup_spaces(self):
        """设置状态和动作空间"""
        # 状态空间: 因子 + 宏观因子 + 组合状态  
        state_dim = (
            self.n_factors +  # 因子数据（已经是拉平的）
            5 +  # 宏观状态: VIX, 北向资金等
            3    # 组合状态: 当前权重均值, 波动率, 回撤
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # 动作空间: 连续权重向量
        self.action_space = spaces.Box(
            low=-self.max_position,
            high=self.max_position,
            shape=(self.n_stocks,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置交易状态
        self.current_step = self.lookback_window
        self.portfolio_weights = np.zeros(self.n_stocks)
        self.cash = 1.0
        self.portfolio_value = 1.0
        self.portfolio_returns = []
        self.max_drawdown = 0.0
        self.peak_value = 1.0
        
        # 获取初始状态
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        # 约束动作
        action = self._constrain_action(action)
        
        # 计算交易成本
        trade_amounts = np.abs(action - self.portfolio_weights)
        transaction_costs = np.sum(trade_amounts) * self.transaction_cost
        
        # 更新仓位
        self.portfolio_weights = action
        
        # 前进一步
        self.current_step += 1
        
        # 计算收益
        if self.current_step < len(self.price_data):
            returns = self._calculate_portfolio_return()
            
            # 保护组合价值计算
            new_value = self.portfolio_value * (1 + returns - transaction_costs)
            if np.isfinite(new_value) and new_value > 0:
                self.portfolio_value = new_value
            else:
                # 如果计算出现问题，保持当前价值不变
                logger.warning(f"组合价值计算异常: returns={returns}, transaction_costs={transaction_costs}")
                
            self.portfolio_returns.append(returns)
            
            # 更新回撤统计
            self._update_drawdown_stats()
            
            # 计算奖励
            reward = self._calculate_reward(returns, transaction_costs)
            
            # 检查终止条件
            terminated = self._check_termination()
            truncated = self.current_step >= len(self.price_data) - 1
            
            # 获取新状态
            observation = self._get_observation()
            info = self._get_info()
            
            return observation, reward, terminated, truncated, info
        else:
            # 回合结束
            observation = self._get_observation()
            info = self._get_info()
            return observation, 0.0, True, True, info
    
    def _constrain_action(self, action: np.ndarray) -> np.ndarray:
        """约束动作空间"""
        # 单股票仓位限制
        action = np.clip(action, -self.max_position, self.max_position)
        
        # 总杠杆限制
        total_leverage = np.sum(np.abs(action))
        if total_leverage > self.max_leverage:
            action = action * (self.max_leverage / total_leverage)
            
        return action
    
    def _get_observation(self) -> np.ndarray:
        """获取当前状态观测"""
        if self.current_step >= len(self.factor_data):
            # 使用最后一个可用的状态
            step = len(self.factor_data) - 1
        else:
            step = self.current_step
            
        # 因子观测 (取当前行的所有因子)
        factor_obs = self.factor_data.iloc[step].values
        
        # 宏观状态 (模拟)
        macro_obs = np.array([
            np.random.normal(0, 0.1),  # VIX代理
            np.random.normal(0, 0.05), # 北向资金代理
            np.random.normal(0, 0.02), # 十年期国债代理
            len(self.portfolio_returns) / 252,  # 时间进度
            self.portfolio_value - 1.0  # 累计收益
        ])
        
        # 组合状态
        portfolio_obs = np.array([
            np.mean(np.abs(self.portfolio_weights)),  # 平均仓位
            np.std(self.portfolio_returns[-20:]) if len(self.portfolio_returns) >= 20 else 0.0,  # 近期波动
            self.max_drawdown  # 当前最大回撤
        ])
        
        # 合并观测
        observation = np.concatenate([factor_obs, macro_obs, portfolio_obs])
        
        # 数值稳定性检查
        observation = np.where(np.isfinite(observation), observation, 0.0)
        observation = np.clip(observation, -1000, 1000)  # 限制极值
        
        return observation.astype(np.float32)
    
    def _calculate_portfolio_return(self) -> float:
        """计算组合收益率"""
        if self.current_step == 0 or self.current_step >= len(self.price_data):
            return 0.0
            
        # 获取价格变化
        current_prices = self.price_data.iloc[self.current_step]
        previous_prices = self.price_data.iloc[self.current_step - 1]
        
        # 安全除法：避免除零和负价格
        mask = (previous_prices > 0.001)  # 避免零或极小值
        stock_returns = np.zeros_like(current_prices)
        
        # 只对有效价格计算收益率
        stock_returns[mask] = (current_prices[mask] - previous_prices[mask]) / previous_prices[mask]
        
        # 限制极端收益率
        stock_returns = np.clip(stock_returns, -0.5, 0.5)  # 限制在±50%
        
        # 计算组合收益率
        portfolio_return = np.dot(self.portfolio_weights, stock_returns)
        
        # 确保返回值数值稳定
        if not np.isfinite(portfolio_return):
            portfolio_return = 0.0
            
        return portfolio_return
    
    def _update_drawdown_stats(self):
        """更新回撤统计"""
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
            
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _calculate_reward(self, returns: float, transaction_costs: float) -> float:
        """计算CVaR增强奖励函数"""
        # 基础收益
        base_reward = returns
        
        # 回撤惩罚
        drawdown_penalty = 0.0
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if current_drawdown > self.max_dd_threshold:
            drawdown_penalty = self.lambda1 * (current_drawdown - self.max_dd_threshold)
        
        # CVaR惩罚
        cvar_penalty = 0.0
        if len(self.portfolio_returns) >= 20:
            recent_returns = np.array(self.portfolio_returns[-20:])
            var_95 = np.percentile(recent_returns, 5)
            cvar_95 = np.mean(recent_returns[recent_returns <= var_95])
            cvar_penalty = self.lambda2 * max(0, -cvar_95 - 0.02)  # CVaR超过2%则惩罚
        
        # 交易成本惩罚 - 降低影响避免过度保守
        cost_penalty = transaction_costs * 3
        
        reward = base_reward - drawdown_penalty - cvar_penalty - cost_penalty
        
        return reward
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        # 破产检查 - 放宽至30%亏损
        if self.portfolio_value < 0.7:
            return True
            
        # 极端回撤检查 - 放宽至25%回撤
        if self.max_drawdown > 0.25:
            return True
            
        return False
    
    def _get_info(self) -> Dict:
        """获取环境信息"""
        return {
            'current_step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'max_drawdown': self.max_drawdown,
            'portfolio_weights': self.portfolio_weights.copy(),
            'cash': self.cash,
            'total_return': self.portfolio_value - 1.0,
            'annual_return': (self.portfolio_value ** (252 / max(len(self.portfolio_returns), 1))) - 1 if self.portfolio_returns else 0,
            'volatility': np.std(self.portfolio_returns) * np.sqrt(252) if len(self.portfolio_returns) > 1 else 0,
            'sharpe_ratio': (np.mean(self.portfolio_returns) / np.std(self.portfolio_returns) * np.sqrt(252)) if len(self.portfolio_returns) > 1 and np.std(self.portfolio_returns) > 0 else 0
        }
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['current_step']}")
            print(f"Portfolio Value: {info['portfolio_value']:.4f}")
            print(f"Total Return: {info['total_return']:.2%}")
            print(f"Max Drawdown: {info['max_drawdown']:.2%}")
            print(f"Sharpe Ratio: {info['sharpe_ratio']:.2f}")
            print("-" * 40)