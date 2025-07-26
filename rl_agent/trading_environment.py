"""
强化学习交易环境
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# 导入动态低波筛选器
try:
    from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter
    from data.data_manager import DataManager
except ImportError as e:
    logger.warning(f"无法导入动态低波筛选器: {e}")
    DynamicLowVolFilter = None
    DataManager = None


class TradingEnvironment(gym.Env):
    """A股交易强化学习环境"""

    def __init__(self,
                 factor_data: pd.DataFrame,
                 price_data: pd.DataFrame,
                 config: Dict,
                 train_universe: List[str] = None):
        """
        初始化交易环境

        Args:
            factor_data: 因子数据
            price_data: 价格数据
            config: 环境配置
            train_universe: 训练时的股票池
        """
        super().__init__()

        self.factor_data = factor_data
        self.price_data = price_data
        self.config = config
        self.train_universe = train_universe if train_universe is not None else list(price_data.columns)
        self.backtest_universe = list(price_data.columns)

        # 创建动作空间映射
        self._create_action_space_map()

        # 环境参数
        self.n_stocks = len(self.train_universe) # 动作空间基于训练股票池
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
        self.lambda1 = config.get('lambda1', 1.0)  # 回撤惩罚
        self.lambda2 = config.get('lambda2', 0.5)  # CVaR惩罚
        self.max_dd_threshold = config.get('max_dd_threshold', 0.05)

        # 初始化动态低波筛选器
        self.lowvol_filter = None
        if (DynamicLowVolFilter is not None and DataManager is not None and 
            'dynamic_lowvol' in config):
            try:
                # 创建简化的数据管理器实例
                data_manager = self._create_data_manager()
                
                # 获取动态低波筛选器配置
                lowvol_config = config.get('dynamic_lowvol', {})
                
                # 初始化动态低波筛选器
                self.lowvol_filter = DynamicLowVolFilter(
                    config=lowvol_config,
                    data_manager=data_manager
                )
                logger.info("动态低波筛选器初始化成功")
            except Exception as e:
                logger.warning(f"动态低波筛选器初始化失败: {e}")
                self.lowvol_filter = None
        else:
            if 'dynamic_lowvol' not in config:
                logger.info("未配置动态低波筛选器，将使用默认筛选逻辑")
            else:
                logger.warning("动态低波筛选器模块未可用，将使用默认筛选逻辑")

        # 当前可交易掩码
        self._current_tradable_mask = None
        
        # O2O模式支持
        self.mode = 'offline'  # 'offline' | 'online'
        self.trajectory_buffer = []  # 轨迹收集缓冲区

    def _create_data_manager(self):
        """创建简化的数据管理器实例"""
        class SimplifiedDataManager:
            """简化的数据管理器，使用交易环境中的数据"""
            def __init__(self, price_data, factor_data):
                self.price_data = price_data
                self.factor_data = factor_data
            
            def get_price_data(self, end_date=None, lookback_days=250):
                """获取价格数据"""
                if end_date is None:
                    return self.price_data
                
                # 找到end_date在价格数据中的位置
                try:
                    end_idx = self.price_data.index.get_loc(end_date)
                except KeyError:
                    # 如果找不到确切日期，使用最近的日期
                    available_dates = self.price_data.index[self.price_data.index <= end_date]
                    if len(available_dates) == 0:
                        return self.price_data.head(lookback_days)
                    end_date = available_dates[-1]
                    end_idx = self.price_data.index.get_loc(end_date)
                
                start_idx = max(0, end_idx - lookback_days)
                return self.price_data.iloc[start_idx:end_idx + 1]
            
            def get_volume_data(self, end_date=None, lookback_days=250):
                """获取成交量数据（模拟）"""
                price_data = self.get_price_data(end_date, lookback_days)
                # 创建模拟的成交量数据
                volume_data = pd.DataFrame(
                    np.random.lognormal(10, 1, price_data.shape),
                    index=price_data.index,
                    columns=price_data.columns
                )
                return volume_data
            
            def get_factor_data(self, end_date=None, lookback_days=250):
                """获取因子数据"""
                if end_date is None:
                    return self.factor_data
                
                try:
                    end_idx = self.factor_data.index.get_loc(end_date)
                except KeyError:
                    available_dates = self.factor_data.index[self.factor_data.index <= end_date]
                    if len(available_dates) == 0:
                        return self.factor_data.head(lookback_days)
                    end_date = available_dates[-1]
                    end_idx = self.factor_data.index.get_loc(end_date)
                
                start_idx = max(0, end_idx - lookback_days)
                return self.factor_data.iloc[start_idx:end_idx + 1]
            
            def get_market_data(self, end_date=None, lookback_days=250):
                """获取市场数据（模拟）"""
                price_data = self.get_price_data(end_date, lookback_days)
                # 创建模拟的市场指数数据
                market_data = pd.DataFrame({
                    'market_index': price_data.mean(axis=1),
                    'returns': price_data.mean(axis=1).pct_change()
                }, index=price_data.index)
                return market_data
        
        return SimplifiedDataManager(self.price_data, self.factor_data)

    def _setup_spaces(self):
        """设置状态和动作空间"""
        # 扩展状态空间: 因子 + 宏观因子 + 组合状态 + 市场状态信号 + 时间信息 + 制度信息
        state_dim = (
            self.n_factors +  # 因子数据（已经是拉平的）
            5 +  # 宏观状态: VIX, 北向资金等
            3 +  # 组合状态: 当前权重均值, 波动率, 回撤
            3 +  # 市场制度信号: 低波动制度, 中等波动制度, 高波动制度 (one-hot编码)
            4 +  # 时间信息: 当前日期(年内天数/365), 交易日进度, 月份信息, 季度信息
            2    # 动态低波筛选器状态: 当前筛选强度, 可交易股票比例
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

    def _create_action_space_map(self):
        """创建从训练动作空间到回测动作空间的映射"""
        self.action_map = [self.train_universe.index(stock) for stock in self.backtest_universe if stock in self.train_universe]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置交易状态
        self.current_step = self.lookback_window
        self.portfolio_weights = np.zeros(len(self.backtest_universe))
        self.cash = 1.0
        self.portfolio_value = 1.0
        self.portfolio_returns = []
        self.max_drawdown = 0.0
        self.peak_value = 1.0
        
        # 模式特定的重置行为
        if self.mode == 'online':
            # 在线模式下，保留轨迹缓冲区用于持续学习
            logger.info(f"在线模式重置，保留 {len(self.trajectory_buffer)} 条历史轨迹")
        else:
            # 离线模式下，清空轨迹缓冲区
            self.trajectory_buffer = []

        # 获取初始状态
        observation = self._get_observation()
        info = self._get_info()
        
        # 添加模式信息到info中
        info['mode'] = self.mode

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行动作"""
        # 保存当前状态用于轨迹收集
        if self.mode == 'online':
            current_state = self._get_observation()
        
        # 更新可交易掩码（在约束动作之前）
        if self.lowvol_filter is not None and self.current_step < len(self.price_data):
            # 使用虚拟日期
            base_date = pd.Timestamp('2020-01-01')
            current_date = base_date + pd.Timedelta(days=self.current_step)
            self._current_tradable_mask = self.lowvol_filter.update_tradable_mask(current_date)

        # 约束动作
        action = self._constrain_action(action)

        # 映射动作到回测空间
        backtest_action = np.zeros(len(self.backtest_universe))
        backtest_action = action[self.action_map]

        # 计算交易成本
        trade_amounts = np.abs(backtest_action - self.portfolio_weights)
        transaction_costs = np.sum(trade_amounts) * self.transaction_cost

        # 更新仓位
        old_weights = self.portfolio_weights.copy()
        self.portfolio_weights = backtest_action

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
            
            # 在线模式下收集轨迹数据
            if self.mode == 'online':
                self.collect_trajectory(
                    state=current_state,
                    action=action,
                    reward=reward,
                    next_state=observation,
                    done=terminated or truncated,
                    info=info
                )

            return observation, reward, terminated, truncated, info
        else:
            # 回合结束
            observation = self._get_observation()
            info = self._get_info()
            
            # 在线模式下收集最后的轨迹数据
            if self.mode == 'online':
                self.collect_trajectory(
                    state=current_state,
                    action=action,
                    reward=0.0,
                    next_state=observation,
                    done=True,
                    info=info
                )
            
            return observation, 0.0, True, True, info

    def _constrain_action(self, action: np.ndarray) -> np.ndarray:
        """约束动作空间"""
        # 单股票仓位限制
        action = np.clip(action, -self.max_position, self.max_position)

        # 应用可交易掩码约束
        if self._current_tradable_mask is not None:
            # 确保掩码长度与动作长度匹配
            if len(self._current_tradable_mask) == len(action):
                # 将不可交易股票的权重设为0
                action = action * self._current_tradable_mask.astype(float)
            else:
                logger.warning(f"可交易掩码长度{len(self._current_tradable_mask)}与动作长度{len(action)}不匹配")

        # 总杠杆限制
        total_leverage = np.sum(np.abs(action))
        if total_leverage > self.max_leverage:
            action = action * (self.max_leverage / total_leverage)

        return action

    def _get_observation(self) -> np.ndarray:
        """获取当前状态观测"""
        if self.current_step >= len(self.price_data):
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

        # 扩展市场制度信号 (one-hot编码)
        regime_obs = np.array([0.0, 1.0, 0.0])  # 默认中等波动制度
        if self.lowvol_filter is not None and hasattr(self.lowvol_filter, 'get_current_regime'):
            current_regime = self.lowvol_filter.get_current_regime()
            if current_regime == "低":
                regime_obs = np.array([1.0, 0.0, 0.0])
            elif current_regime == "中":
                regime_obs = np.array([0.0, 1.0, 0.0])
            elif current_regime == "高":
                regime_obs = np.array([0.0, 0.0, 1.0])

        # 时间信息 - 简化为使用虚拟时间
        # 由于数据结构复杂性，直接使用步数生成虚拟时间特征
        base_date = pd.Timestamp('2020-01-01')
        current_date = base_date + pd.Timedelta(days=step)
        
        day_of_year = current_date.dayofyear / 365.0  # 年内天数进度
        trading_progress = self.current_step / len(self.price_data)  # 交易日进度
        month_info = current_date.month / 12.0  # 月份信息
        quarter_info = ((current_date.month - 1) // 3) / 4.0  # 季度信息
        
        time_obs = np.array([day_of_year, trading_progress, month_info, quarter_info])

        # 动态低波筛选器状态信息
        filter_strength = 0.5  # 默认筛选强度
        tradable_ratio = 1.0   # 默认全部可交易
        
        if self.lowvol_filter is not None:
            # 获取筛选强度（如果有相关方法）
            if hasattr(self.lowvol_filter, 'get_filter_strength'):
                filter_strength = self.lowvol_filter.get_filter_strength()
            
            # 计算可交易股票比例
            if self._current_tradable_mask is not None:
                tradable_ratio = np.mean(self._current_tradable_mask.astype(float))
        
        filter_obs = np.array([filter_strength, tradable_ratio])

        # 合并所有观测
        observation = np.concatenate([
            factor_obs,      # 因子数据
            macro_obs,       # 宏观状态
            portfolio_obs,   # 组合状态
            regime_obs,      # 市场制度信号 (one-hot)
            time_obs,        # 时间信息
            filter_obs       # 筛选器状态
        ])

        # 数值稳定性检查
        observation = np.nan_to_num(observation.astype(np.float32))
        observation = np.clip(observation, -1000, 1000)  # 限制极值

        return observation.astype(np.float32)
    
    def set_mode(self, mode: str):
        """
        设置环境模式
        
        Args:
            mode: 环境模式，'offline' 或 'online'
        """
        if mode not in ['offline', 'online']:
            raise ValueError(f"无效的环境模式: {mode}. 必须是 'offline' 或 'online'")
        
        old_mode = self.mode
        self.mode = mode
        
        logger.info(f"环境模式从 '{old_mode}' 切换到 '{mode}'")
        
        # 模式切换时的特殊处理
        if mode == 'online' and old_mode == 'offline':
            # 切换到在线模式时，清空轨迹缓冲区
            self.trajectory_buffer = []
            logger.info("切换到在线模式，轨迹缓冲区已清空")
        elif mode == 'offline' and old_mode == 'online':
            # 切换到离线模式时，保留轨迹数据用于分析
            logger.info(f"切换到离线模式，保留 {len(self.trajectory_buffer)} 条轨迹记录")
    
    def get_mode(self) -> str:
        """获取当前环境模式"""
        return self.mode
    
    def collect_trajectory(self, state: np.ndarray, action: np.ndarray, reward: float, 
                          next_state: np.ndarray, done: bool, info: Dict):
        """
        收集交易轨迹数据
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            info: 额外信息
        """
        # 获取当前时间戳 - 使用虚拟时间
        base_date = pd.Timestamp('2020-01-01')
        timestamp = base_date + pd.Timedelta(days=self.current_step)
        
        # 获取市场制度信息
        market_regime = "中"  # 默认值
        if self.lowvol_filter is not None:
            try:
                market_regime = self.lowvol_filter.get_current_regime()
            except Exception as e:
                logger.warning(f"获取市场制度失败: {e}")
        
        # 创建轨迹记录
        trajectory_record = {
            'timestamp': timestamp,
            'step': self.current_step,
            'state': state.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': self.portfolio_weights.copy(),
            'market_regime': market_regime,
            'max_drawdown': self.max_drawdown,
            'transaction_cost': np.sum(np.abs(action - info.get('previous_weights', np.zeros_like(action)))) * self.transaction_cost,
            'info': info.copy()
        }
        
        # 添加到轨迹缓冲区
        self.trajectory_buffer.append(trajectory_record)
        
        # 限制缓冲区大小，保留最近的轨迹
        max_buffer_size = self.config.get('trajectory_buffer_size', 10000)
        if len(self.trajectory_buffer) > max_buffer_size:
            self.trajectory_buffer = self.trajectory_buffer[-max_buffer_size:]
            
        logger.debug(f"收集轨迹: step={self.current_step}, reward={reward:.4f}, regime={market_regime}")
    
    def get_recent_trajectory(self, window: int = 60) -> List[Dict]:
        """
        获取最近的交易轨迹数据
        
        Args:
            window: 获取最近多少条轨迹记录
            
        Returns:
            最近的轨迹记录列表
        """
        if not self.trajectory_buffer:
            logger.warning("轨迹缓冲区为空")
            return []
        
        # 返回最近window条记录
        recent_trajectories = self.trajectory_buffer[-window:] if len(self.trajectory_buffer) >= window else self.trajectory_buffer
        
        logger.info(f"获取最近 {len(recent_trajectories)} 条轨迹记录")
        return recent_trajectories
    
    def get_trajectory_by_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Dict]:
        """
        根据日期范围获取轨迹数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            指定日期范围内的轨迹记录列表
        """
        filtered_trajectories = []
        
        for trajectory in self.trajectory_buffer:
            if start_date <= trajectory['timestamp'] <= end_date:
                filtered_trajectories.append(trajectory)
        
        logger.info(f"日期范围 {start_date} 到 {end_date} 内找到 {len(filtered_trajectories)} 条轨迹记录")
        return filtered_trajectories
    
    def get_trajectory_statistics(self) -> Dict:
        """
        获取轨迹缓冲区统计信息
        
        Returns:
            轨迹统计信息字典
        """
        if not self.trajectory_buffer:
            return {
                'total_trajectories': 0,
                'date_range': None,
                'avg_reward': 0.0,
                'regime_distribution': {}
            }
        
        # 计算统计信息
        rewards = [t['reward'] for t in self.trajectory_buffer]
        regimes = [t['market_regime'] for t in self.trajectory_buffer]
        timestamps = [t['timestamp'] for t in self.trajectory_buffer]
        
        # 制度分布统计
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'total_trajectories': len(self.trajectory_buffer),
            'date_range': (min(timestamps), max(timestamps)),
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'regime_distribution': regime_counts,
            'avg_portfolio_value': np.mean([t['portfolio_value'] for t in self.trajectory_buffer]),
            'max_drawdown_encountered': max([t['max_drawdown'] for t in self.trajectory_buffer])
        }
    
    def clear_trajectory_buffer(self):
        """清空轨迹缓冲区"""
        buffer_size = len(self.trajectory_buffer)
        self.trajectory_buffer = []
        logger.info(f"已清空轨迹缓冲区，删除了 {buffer_size} 条记录")
    
    def get_market_regime_info(self) -> Dict:
        """
        获取详细的市场制度信息
        
        Returns:
            包含市场制度详细信息的字典
        """
        regime_info = {
            'current_regime': '中',  # 默认制度
            'regime_confidence': 0.5,  # 制度置信度
            'regime_duration': 0,  # 当前制度持续天数
            'volatility_level': 0.0,  # 波动率水平
            'filter_active': False,  # 筛选器是否激活
            'tradable_stocks_count': self.n_stocks,  # 可交易股票数量
            'filter_strength': 0.5  # 筛选强度
        }
        
        if self.lowvol_filter is not None:
            try:
                regime_info['current_regime'] = self.lowvol_filter.get_current_regime()
                regime_info['filter_active'] = True
                
                # 如果有相关方法，获取更多信息
                if hasattr(self.lowvol_filter, 'get_regime_confidence'):
                    regime_info['regime_confidence'] = self.lowvol_filter.get_regime_confidence()
                
                if hasattr(self.lowvol_filter, 'get_regime_duration'):
                    regime_info['regime_duration'] = self.lowvol_filter.get_regime_duration()
                
                if hasattr(self.lowvol_filter, 'get_volatility_level'):
                    regime_info['volatility_level'] = self.lowvol_filter.get_volatility_level()
                
                if hasattr(self.lowvol_filter, 'get_filter_strength'):
                    regime_info['filter_strength'] = self.lowvol_filter.get_filter_strength()
                
                # 计算可交易股票数量
                if self._current_tradable_mask is not None:
                    regime_info['tradable_stocks_count'] = int(np.sum(self._current_tradable_mask))
                    
            except Exception as e:
                logger.warning(f"获取市场制度详细信息失败: {e}")
        
        return regime_info
    
    def get_extended_state_info(self) -> Dict:
        """
        获取扩展的状态信息，用于O2O决策
        
        Returns:
            包含扩展状态信息的字典
        """
        current_date = self.price_data.index[self.current_step] if self.current_step < len(self.price_data) else self.price_data.index[-1]
        
        extended_info = {
            'current_date': current_date,
            'trading_day_progress': self.current_step / len(self.price_data),
            'market_regime': self.get_market_regime_info(),
            'portfolio_state': {
                'weights': self.portfolio_weights.copy(),
                'value': self.portfolio_value,
                'cash': self.cash,
                'max_drawdown': self.max_drawdown,
                'recent_returns': self.portfolio_returns[-20:] if len(self.portfolio_returns) >= 20 else self.portfolio_returns
            },
            'time_features': {
                'day_of_year': current_date.dayofyear,
                'month': current_date.month,
                'quarter': (current_date.month - 1) // 3 + 1,
                'weekday': current_date.weekday(),
                'is_month_end': current_date.day >= 25,  # 简单的月末判断
                'is_quarter_end': current_date.month in [3, 6, 9, 12] and current_date.day >= 25
            },
            'risk_metrics': {
                'volatility': np.std(self.portfolio_returns) * np.sqrt(252) if len(self.portfolio_returns) > 1 else 0,
                'sharpe_ratio': (np.mean(self.portfolio_returns) / np.std(self.portfolio_returns) * np.sqrt(252)) if len(self.portfolio_returns) > 1 and np.std(self.portfolio_returns) > 0 else 0,
                'max_leverage': np.sum(np.abs(self.portfolio_weights)),
                'concentration': np.sum(self.portfolio_weights ** 2)  # 集中度指标
            }
        }
        
        return extended_info

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
        """计算优化的奖励函数 - 注重收益最大化"""
        
        # 基础收益奖励 - 智能动态调整策略增强
        base_reward = returns * 18.0  # 18倍放大收益信号
        
        # 动量奖励 - 奖励持续盈利
        momentum_bonus = 0.0
        if len(self.portfolio_returns) >= 5:
            recent_returns = np.array(self.portfolio_returns[-5:])
            if np.mean(recent_returns) > 0:
                momentum_bonus = np.mean(recent_returns) * 5.0  # 奖励近期盈利趋势
        
        # 夏普比率奖励 - 大幅增强
        sharpe_bonus = 0.0
        if len(self.portfolio_returns) >= 20:
            returns_array = np.array(self.portfolio_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            if std_return > 1e-8 and mean_return > 0:
                sharpe_ratio = mean_return / std_return * np.sqrt(252)
                sharpe_bonus = sharpe_ratio * 2.0  # 大幅提升夏普比率奖励
        
        # 适度的风险控制 - 仅在极端情况下惩罚
        risk_penalty = 0.0
        
        # 极端回撤惩罚 - 只在超过20%时惩罚
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        if current_drawdown > 0.20:
            risk_penalty += 0.5 * (current_drawdown - 0.20)  # 减轻惩罚强度
        
        # 轻微的交易成本约束
        cost_penalty = transaction_costs * 0.1  # 大幅减轻交易成本惩罚
        
        # 收益稳定性奖励
        stability_bonus = 0.0
        if len(self.portfolio_returns) >= 10:
            recent_returns = np.array(self.portfolio_returns[-10:])
            if np.mean(recent_returns) > 0:
                # 奖励稳定的正收益
                consistency = 1.0 - (np.std(recent_returns) / (np.abs(np.mean(recent_returns)) + 1e-8))
                stability_bonus = max(0, consistency) * np.mean(recent_returns) * 3.0
        
        # 组合最终奖励 - 重点激励收益
        total_reward = (base_reward + momentum_bonus + sharpe_bonus + 
                       stability_bonus - risk_penalty - cost_penalty)
        
        return total_reward

    def _check_termination(self) -> bool:
        """检查终止条件"""
        # 破产检查 - 放宽至50%亏损
        if self.portfolio_value < 0.5:
            return True

        # 极端回撤检查 - 放宽至35%回撤
        if self.max_drawdown > 0.35:
            return True

        return False

    def _get_info(self) -> Dict:
        """获取环境信息"""
        base_info = {
            'current_step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'max_drawdown': self.max_drawdown,
            'portfolio_weights': self.portfolio_weights.copy(),
            'cash': self.cash,
            'total_return': self.portfolio_value - 1.0,
            'annual_return': (self.portfolio_value ** (252 / max(len(self.portfolio_returns), 1))) - 1 if self.portfolio_returns else 0,
            'volatility': np.std(self.portfolio_returns) * np.sqrt(252) if len(self.portfolio_returns) > 1 else 0,
            'sharpe_ratio': (np.mean(self.portfolio_returns) / np.std(self.portfolio_returns) * np.sqrt(252)) if len(self.portfolio_returns) > 1 and np.std(self.portfolio_returns) > 0 else 0,
            'mode': self.mode,  # 添加模式信息
            'trajectory_buffer_size': len(self.trajectory_buffer)  # 添加轨迹缓冲区大小信息
        }
        
        # 在在线模式下添加扩展状态信息
        if self.mode == 'online':
            base_info['extended_state'] = self.get_extended_state_info()
        
        return base_info

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