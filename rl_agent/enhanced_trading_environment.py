"""
增强交易环境
集成所有改进，专注于8%年化收益目标
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

# 导入增强组件
try:
    from .stable_cvar_ppo_agent import StableCVaRPPOAgent, StableActorCriticNetwork
    from .adaptive_safety_shield import AdaptiveSafetyShield
    from .enhanced_reward_system import EnhancedRewardSystem
    from ..factors.advanced_alpha_factors import AdvancedAlphaFactors
except ImportError:
    # 绝对导入作为备选
    from rl_agent.stable_cvar_ppo_agent import StableCVaRPPOAgent, StableActorCriticNetwork
    from rl_agent.adaptive_safety_shield import AdaptiveSafetyShield
    from rl_agent.enhanced_reward_system import EnhancedRewardSystem
    from factors.advanced_alpha_factors import AdvancedAlphaFactors

logger = logging.getLogger(__name__)


class EnhancedTradingEnvironment(gym.Env):
    """增强交易环境 - 集成所有改进"""
    
    def __init__(self,
                 factor_data: pd.DataFrame,
                 price_data: pd.DataFrame,
                 config: Dict,
                 train_universe: List[str] = None):
        """
        初始化增强交易环境
        
        Args:
            factor_data: 因子数据
            price_data: 价格数据
            config: 环境配置
            train_universe: 训练股票池
        """
        super().__init__()
        
        self.factor_data = factor_data
        self.price_data = price_data
        self.config = config
        self.train_universe = train_universe if train_universe is not None else list(price_data.columns)
        
        # 环境配置
        self.lookback_window = config.get('lookback_window', 30)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.initial_capital = config.get('initial_capital', 1000000)
        
        # 增强组件初始化
        self._initialize_enhanced_components()
        
        # 环境状态
        self.current_step = 0
        self.current_date = None
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = np.zeros(len(self.train_universe))
        self.portfolio_returns = []
        self.peak_value = self.initial_capital
        
        # 创建动作和观察空间
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(self.train_universe),), 
            dtype=np.float32
        )
        
        # 观察空间：因子 + 组合状态 + 市场状态
        factor_dim = len(self.enhanced_factors.alpha_factors)
        portfolio_state_dim = 10  # 组合相关状态
        market_state_dim = 5     # 市场状态
        total_obs_dim = factor_dim + portfolio_state_dim + market_state_dim
        
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
        
        # 性能追踪
        self.performance_metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 1.0,
            'calmar_ratio': 0.0
        }
        
        # 训练统计
        self.episode_stats = {
            'episodes_completed': 0,
            'best_annual_return': -np.inf,
            'best_sharpe_ratio': -np.inf,
            'average_performance': 0.0,
            'convergence_indicator': 0.0
        }
        
        logger.info(f"增强交易环境初始化完成 - 股票数: {len(self.train_universe)}, 因子数: {factor_dim}")
    
    def _initialize_enhanced_components(self):
        """初始化增强组件"""
        # 1. 高级Alpha因子计算器
        self.enhanced_factors = AdvancedAlphaFactors(self.config.get('factors', {}))
        
        # 2. 自适应安全保护层
        safety_config = self.config.get('safety_shield', {})
        safety_config.update({
            'max_position': 0.25,
            'max_leverage': 2.0,
            'var_threshold': 0.05,
            'max_drawdown_threshold': 0.12,
            'volatility_threshold': 0.30
        })
        self.safety_shield = AdaptiveSafetyShield(safety_config)
        
        # 3. 增强奖励系统
        reward_config = self.config.get('reward_system', {})
        reward_config.update({
            'target_annual_return': 0.08,
            'target_sharpe_ratio': 1.0,
            'return_weight': 1.0,
            'sharpe_weight': 0.3,
            'consistency_weight': 0.2,
            'momentum_weight': 0.15
        })
        self.reward_system = EnhancedRewardSystem(reward_config)
        
        logger.info("增强组件初始化完成")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置环境状态
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = np.zeros(len(self.train_universe))
        self.portfolio_returns = []
        self.peak_value = self.initial_capital
        
        # 找到有效的起始日期
        self.current_date = self._get_valid_start_date()
        
        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步交易"""
        # 1. 应用安全保护层
        current_portfolio_weights = self.holdings / (self.portfolio_value + 1e-8)
        safe_action = self.safety_shield.shield_action(
            action, 
            self._get_state_dict(),
            self._get_current_price_data(),
            current_portfolio_weights,
            self._get_recent_performance()
        )
        
        # 2. 执行交易
        transaction_costs = self._execute_trades(safe_action)
        
        # 3. 更新组合价值
        new_portfolio_value = self._update_portfolio_value()
        
        # 4. 计算收益
        period_return = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        self.portfolio_returns.append(period_return)
        
        # 5. 计算增强奖励
        reward, reward_components = self.reward_system.calculate_reward(
            current_return=period_return,
            portfolio_returns=self.portfolio_returns,
            transaction_costs=transaction_costs,
            portfolio_value=new_portfolio_value,
            peak_value=self.peak_value,
            market_return=self._get_market_return()
        )
        
        # 6. 更新状态
        self.portfolio_value = new_portfolio_value
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.current_step += 1
        
        # 7. 检查终止条件
        terminated = self._check_termination()
        truncated = self.current_step >= len(self.price_data) - self.lookback_window - 1
        
        # 8. 获取下一个观察
        if not terminated and not truncated:
            self.current_date = self._get_next_date()
        
        observation = self._get_observation()
        info = self._get_info()
        info['reward_components'] = reward_components
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """获取增强观察状态"""
        if self.current_date is None:
            # 返回零观察
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        observations = []
        
        # 1. 因子特征
        factor_features = self._get_factor_features()
        observations.extend(factor_features)
        
        # 2. 组合状态特征
        portfolio_features = self._get_portfolio_features()
        observations.extend(portfolio_features)
        
        # 3. 市场状态特征
        market_features = self._get_market_features()
        observations.extend(market_features)
        
        # 确保观察维度正确
        observations = np.array(observations, dtype=np.float32)
        expected_dim = self.observation_space.shape[0]
        
        if len(observations) < expected_dim:
            # 填充零值
            observations = np.pad(observations, (0, expected_dim - len(observations)))
        elif len(observations) > expected_dim:
            # 截断
            observations = observations[:expected_dim]
        
        # 数值稳定性检查
        observations = np.nan_to_num(observations, nan=0.0, posinf=5.0, neginf=-5.0)
        observations = np.clip(observations, -10.0, 10.0)
        
        return observations
    
    def _get_factor_features(self) -> List[float]:
        """获取因子特征"""
        try:
            # 获取当前日期的因子数据
            if self.current_date not in self.factor_data.index:
                return [0.0] * len(self.enhanced_factors.alpha_factors)
            
            current_factors = self.factor_data.loc[self.current_date]
            
            # 计算因子的横截面统计特征
            factor_means = []
            for factor_name in self.enhanced_factors.alpha_factors:
                if factor_name in current_factors.columns:
                    factor_values = current_factors[factor_name][self.train_universe]
                    factor_mean = factor_values.mean()
                    factor_means.append(factor_mean if not np.isnan(factor_mean) else 0.0)
                else:
                    factor_means.append(0.0)
            
            return factor_means
            
        except Exception as e:
            logger.warning(f"获取因子特征时出错: {e}")
            return [0.0] * len(self.enhanced_factors.alpha_factors)
    
    def _get_portfolio_features(self) -> List[float]:
        """获取组合状态特征"""
        try:
            portfolio_features = []
            
            # 1. 当前组合权重统计
            weights = self.holdings / (self.portfolio_value + 1e-8)
            portfolio_features.extend([
                np.sum(np.abs(weights)),  # 总杠杆
                np.max(np.abs(weights)),  # 最大单股仓位
                np.mean(weights),         # 平均权重
                np.std(weights),          # 权重标准差
            ])
            
            # 2. 收益序列特征
            if len(self.portfolio_returns) >= 5:
                recent_returns = np.array(self.portfolio_returns[-5:])
                portfolio_features.extend([
                    np.mean(recent_returns),      # 平均收益
                    np.std(recent_returns),       # 收益波动率
                    np.sum(recent_returns > 0) / len(recent_returns),  # 胜率
                ])
            else:
                portfolio_features.extend([0.0, 0.0, 0.5])
            
            # 3. 风险指标
            current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            portfolio_features.extend([
                current_drawdown,             # 当前回撤
                self.portfolio_value / self.initial_capital - 1.0,  # 累积收益
                len(self.portfolio_returns) / 252.0,  # 时间进度
            ])
            
            return portfolio_features
            
        except Exception as e:
            logger.warning(f"获取组合特征时出错: {e}")
            return [0.0] * 10
    
    def _get_market_features(self) -> List[float]:
        """获取市场状态特征"""
        try:
            market_features = []
            
            # 获取市场收益和波动率
            if self.current_date is not None and len(self.portfolio_returns) >= 5:
                # 市场平均收益（使用当前股票池）
                current_prices = self._get_current_price_data()
                if current_prices is not None and len(current_prices) > 0:
                    market_return = current_prices.pct_change().mean().mean()
                    market_volatility = current_prices.pct_change().std().mean()
                else:
                    market_return = 0.0
                    market_volatility = 0.01
            else:
                market_return = 0.0
                market_volatility = 0.01
            
            market_features.extend([
                market_return,                    # 市场收益
                market_volatility,                # 市场波动率
                market_return / (market_volatility + 1e-8),  # 市场夏普比率
                self.current_step / 252.0,        # 时间进度
                1.0,                             # 占位符
            ])
            
            return market_features
            
        except Exception as e:
            logger.warning(f"获取市场特征时出错: {e}")
            return [0.0] * 5
    
    def _execute_trades(self, action: np.ndarray) -> float:
        """执行交易并返回交易成本"""
        # 将动作转换为目标权重
        target_weights = np.array(action)
        target_weights = target_weights / (np.sum(np.abs(target_weights)) + 1e-8)
        
        # 计算当前权重
        current_weights = self.holdings / (self.portfolio_value + 1e-8)
        
        # 计算交易量
        weight_changes = target_weights - current_weights
        turnover = np.sum(np.abs(weight_changes))
        
        # 计算交易成本
        transaction_costs = turnover * self.transaction_cost * self.portfolio_value
        
        # 更新持仓
        self.holdings = target_weights * self.portfolio_value
        self.cash = self.portfolio_value - np.sum(self.holdings)
        
        return transaction_costs
    
    def _update_portfolio_value(self) -> float:
        """更新组合价值"""
        try:
            current_prices = self._get_current_price_data()
            if current_prices is None:
                return self.portfolio_value
            
            # 获取价格变化
            prev_date = self._get_previous_date()
            if prev_date is not None and prev_date in self.price_data.index:
                prev_prices = self.price_data.loc[prev_date][self.train_universe]
                current_prices_aligned = current_prices[self.train_universe]
                
                # 计算收益率
                price_changes = (current_prices_aligned / prev_prices) - 1.0
                price_changes = price_changes.fillna(0.0)
                
                # 更新持仓价值
                holdings_array = np.array(self.holdings)
                price_changes_array = np.array(price_changes)
                
                # 确保维度匹配
                min_len = min(len(holdings_array), len(price_changes_array))
                if min_len > 0:
                    portfolio_change = np.sum(holdings_array[:min_len] * price_changes_array[:min_len])
                    new_portfolio_value = self.portfolio_value + portfolio_change
                else:
                    new_portfolio_value = self.portfolio_value
            else:
                new_portfolio_value = self.portfolio_value
            
            return max(new_portfolio_value, 0.01)  # 防止负值
            
        except Exception as e:
            logger.warning(f"更新组合价值时出错: {e}")
            return self.portfolio_value
    
    def _get_current_price_data(self) -> Optional[pd.Series]:
        """获取当前价格数据"""
        if self.current_date is not None and self.current_date in self.price_data.index:
            return self.price_data.loc[self.current_date]
        return None
    
    def _get_valid_start_date(self) -> Optional[pd.Timestamp]:
        """获取有效的起始日期"""
        available_dates = list(self.price_data.index)
        if len(available_dates) > self.lookback_window:
            return available_dates[self.lookback_window]
        return None
    
    def _get_next_date(self) -> Optional[pd.Timestamp]:
        """获取下一个交易日期"""
        try:
            available_dates = list(self.price_data.index)
            current_idx = available_dates.index(self.current_date)
            if current_idx + 1 < len(available_dates):
                return available_dates[current_idx + 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def _get_previous_date(self) -> Optional[pd.Timestamp]:
        """获取上一个交易日期"""
        try:
            available_dates = list(self.price_data.index)
            current_idx = available_dates.index(self.current_date)
            if current_idx > 0:
                return available_dates[current_idx - 1]
        except (ValueError, IndexError):
            pass
        return None
    
    def _get_state_dict(self) -> Dict:
        """获取状态字典"""
        return {
            'current_step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'holdings': self.holdings.copy(),
            'peak_value': self.peak_value,
            'portfolio_returns': self.portfolio_returns.copy()
        }
    
    def _get_recent_performance(self) -> Optional[float]:
        """获取近期表现"""
        if len(self.portfolio_returns) >= 5:
            return np.mean(self.portfolio_returns[-5:])
        return None
    
    def _get_market_return(self) -> Optional[float]:
        """获取市场基准收益"""
        current_prices = self._get_current_price_data()
        prev_date = self._get_previous_date()
        
        if current_prices is not None and prev_date is not None:
            prev_prices = self.price_data.loc[prev_date][self.train_universe]
            market_return = (current_prices[self.train_universe] / prev_prices - 1.0).mean()
            return market_return if not np.isnan(market_return) else None
        
        return None
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        # 1. 严重亏损
        if self.portfolio_value < self.initial_capital * 0.5:
            logger.info("因严重亏损终止交易")
            return True
        
        # 2. 极端回撤
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if current_drawdown > 0.25:
            logger.info(f"因极端回撤终止交易: {current_drawdown:.2%}")
            return True
        
        return False
    
    def _get_info(self) -> Dict:
        """获取环境信息"""
        # 更新性能指标
        self._update_performance_metrics()
        
        info = {
            'current_step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'peak_value': self.peak_value,
            'current_drawdown': (self.peak_value - self.portfolio_value) / self.peak_value,
            'cumulative_return': (self.portfolio_value / self.initial_capital) - 1.0,
            'performance_metrics': self.performance_metrics.copy(),
            'safety_stats': self.safety_shield.get_constraint_statistics(),
            'reward_analysis': self.reward_system.get_reward_analysis()
        }
        
        return info
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        if len(self.portfolio_returns) < 10:
            return
        
        returns_array = np.array(self.portfolio_returns)
        
        # 累积收益
        self.performance_metrics['total_return'] = (self.portfolio_value / self.initial_capital) - 1.0
        
        # 年化收益
        trading_days = len(returns_array)
        if trading_days > 0:
            total_return = self.performance_metrics['total_return']
            self.performance_metrics['annualized_return'] = ((1 + total_return) ** (252 / trading_days)) - 1
        
        # 夏普比率
        if np.std(returns_array) > 1e-8:
            self.performance_metrics['sharpe_ratio'] = (
                np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            )
        
        # 最大回撤
        self.performance_metrics['max_drawdown'] = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # 胜率
        self.performance_metrics['win_rate'] = np.mean(returns_array > 0)
        
        # 卡尔玛比率
        if self.performance_metrics['max_drawdown'] > 1e-8:
            self.performance_metrics['calmar_ratio'] = (
                self.performance_metrics['annualized_return'] / self.performance_metrics['max_drawdown']
            )
    
    def get_final_report(self) -> Dict:
        """获取最终报告"""
        self._update_performance_metrics()
        
        report = {
            'episode_stats': self.episode_stats.copy(),
            'final_performance': self.performance_metrics.copy(),
            'safety_analysis': self.safety_shield.get_constraint_statistics(),
            'reward_analysis': self.reward_system.get_reward_analysis(),
            'training_summary': {
                'total_steps': self.current_step,
                'final_portfolio_value': self.portfolio_value,
                'target_achievement': {
                    'annual_return_target': 0.08,
                    'actual_annual_return': self.performance_metrics['annualized_return'],
                    'target_achieved': self.performance_metrics['annualized_return'] >= 0.08,
                    'improvement_needed': max(0, 0.08 - self.performance_metrics['annualized_return'])
                }
            }
        }
        
        return report