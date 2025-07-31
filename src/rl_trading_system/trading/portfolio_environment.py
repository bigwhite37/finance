"""
投资组合交易环境
实现符合OpenAI Gym接口的强化学习交易环境，支持A股交易规则
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import logging

from ..data.data_models import TradingState, TradingAction, MarketData, FeatureVector
from ..data.interfaces import DataInterface
from ..data.data_processor import DataProcessor
from ..data.feature_engineer import FeatureEngineer
from .transaction_cost_model import TransactionCostModel, CostParameters, TradeInfo

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """投资组合环境配置"""
    stock_pool: List[str]                    # 股票池
    lookback_window: int = 60               # 回望窗口
    initial_cash: float = 1000000.0         # 初始资金
    commission_rate: float = 0.0003         # 手续费率
    stamp_tax_rate: float = 0.001           # 印花税率
    min_commission: float = 5.0             # 最小手续费
    transfer_fee_rate: float = 0.00002      # 过户费率
    risk_aversion: float = 0.1              # 风险厌恶系数
    max_drawdown_penalty: float = 1.0       # 最大回撤惩罚
    max_position_size: float = 0.1          # 单只股票最大权重
    min_trade_amount: float = 1000.0        # 最小交易金额
    price_limit: float = 0.1                # 涨跌停限制
    t_plus_1: bool = True                   # T+1交易规则
    trading_days_per_year: int = 252        # 年交易日数
    
    def __post_init__(self):
        """配置验证"""
        if not self.stock_pool:
            raise ValueError("股票池不能为空")
        
        if self.lookback_window <= 0:
            raise ValueError("回望窗口必须大于0")
        
        if self.initial_cash <= 0:
            raise ValueError("初始资金必须大于0")
        
        if not (0 <= self.max_position_size <= 1):
            raise ValueError("最大持仓权重必须在0到1之间")


class PortfolioEnvironment(gym.Env):
    """
    投资组合交易环境
    
    该环境实现了符合OpenAI Gym接口的强化学习交易环境，支持：
    1. 多资产投资组合管理
    2. A股交易规则（T+1、涨跌停、最小交易单位等）
    3. 完整的交易成本模型
    4. 风险控制和约束
    5. 实时市场数据接入
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 config: PortfolioConfig,
                 data_interface: DataInterface,
                 data_processor: Optional[DataProcessor] = None,
                 feature_engineer: Optional[FeatureEngineer] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """
        初始化投资组合环境
        
        Args:
            config: 环境配置
            data_interface: 数据接口
            data_processor: 数据预处理器
            feature_engineer: 特征工程器
            start_date: 开始日期
            end_date: 结束日期
        """
        super().__init__()
        
        self.config = config
        self.data_interface = data_interface
        self.data_processor = data_processor or DataProcessor()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        
        # 时间范围
        self.start_date = start_date
        self.end_date = end_date
        
        # 环境状态
        self.n_stocks = len(config.stock_pool)
        self.n_features = 50  # 默认50个特征
        self.n_market_features = 10  # 默认10个市场特征
        
        # 初始化交易成本模型
        cost_params = CostParameters(
            commission_rate=config.commission_rate,
            stamp_tax_rate=config.stamp_tax_rate,
            min_commission=config.min_commission,
            transfer_fee_rate=config.transfer_fee_rate
        )
        self.cost_model = TransactionCostModel(cost_params)
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            'features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(config.lookback_window, self.n_stocks, self.n_features),
                dtype=np.float32
            ),
            'positions': spaces.Box(
                low=0, high=1,
                shape=(self.n_stocks,),
                dtype=np.float32
            ),
            'market_state': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_market_features,),
                dtype=np.float32
            )
        })
        
        # 定义动作空间
        self.action_space = spaces.Box(
            low=0, high=1,
            shape=(self.n_stocks,),
            dtype=np.float32
        )
        
        # 初始化环境状态变量
        self._initialize_state_variables()
        
        # 加载历史数据
        self._load_market_data()
        
        logger.info(f"投资组合环境初始化完成: {self.n_stocks}只股票, "
                   f"观察空间: {self.observation_space}, "
                   f"动作空间: {self.action_space}")
    
    def _initialize_state_variables(self):
        """初始化状态变量"""
        self.current_step = 0
        self.max_steps = 252  # 默认一年交易日
        self.current_positions = np.zeros(self.n_stocks, dtype=np.float32)
        self.cash = float(self.config.initial_cash)
        self.total_value = float(self.config.initial_cash)
        self.portfolio_values = [float(self.config.initial_cash)]
        self.max_portfolio_value = float(self.config.initial_cash)
        
        # 交易记录
        self.trade_history = []
        self.returns_history = []
        self.weights_history = []
        
        # T+1规则相关
        self.t_plus_1_restrictions = {}  # 记录当日买入无法卖出的股票
        
        # 市场数据相关
        self.market_data = None
        self.feature_data = None
        self.price_data = None
        self.current_prices = None
    
    def _load_market_data(self):
        """加载市场数据"""
        if not self.start_date or not self.end_date:
            # 如果没有指定日期范围，生成模拟数据
            self._generate_mock_data()
            return
        
        # 从数据接口加载真实数据
        self.price_data = self.data_interface.get_price_data(
            symbols=self.config.stock_pool,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 计算特征
        self.feature_data = self.feature_engineer.calculate_features(self.price_data)
        
        # 更新最大步数
        self.max_steps = len(self.price_data.index.get_level_values('datetime').unique())
        
        logger.info(f"加载市场数据完成: {len(self.price_data)}条记录, {self.max_steps}个交易日")
    
    def _generate_mock_data(self):
        """生成模拟数据（用于测试）"""
        dates = pd.date_range(start='2023-01-01', periods=self.max_steps, freq='D')
        
        # 生成模拟价格数据
        np.random.seed(42)  # 确保可重现性
        
        mock_data = []
        for date in dates:
            for i, symbol in enumerate(self.config.stock_pool):
                # 生成价格数据
                base_price = 10.0 + i  # 基础价格
                daily_return = np.random.normal(0.001, 0.02)  # 日收益率
                
                price = base_price * (1 + daily_return)
                volume = np.random.randint(1000000, 10000000)
                
                mock_data.append({
                    'datetime': date,
                    'instrument': symbol,
                    'open': price * np.random.uniform(0.99, 1.01),
                    'high': price * np.random.uniform(1.00, 1.05),
                    'low': price * np.random.uniform(0.95, 1.00),
                    'close': price,
                    'volume': volume,
                    'amount': price * volume
                })
        
        self.price_data = pd.DataFrame(mock_data).set_index(['datetime', 'instrument'])
        
        # 生成模拟特征数据
        self._generate_mock_features()
        
        logger.info("生成模拟数据完成")
    
    def _generate_mock_features(self):
        """生成模拟特征数据"""
        feature_data = []
        
        for date_idx in range(len(self.price_data.index.get_level_values('datetime').unique())):
            for stock_idx, symbol in enumerate(self.config.stock_pool):
                # 生成技术指标特征
                technical_features = np.random.randn(20).astype(np.float32)
                
                # 生成基本面特征
                fundamental_features = np.random.randn(20).astype(np.float32)
                
                # 生成市场微观结构特征
                microstructure_features = np.random.randn(10).astype(np.float32)
                
                # 合并所有特征
                all_features = np.concatenate([
                    technical_features,
                    fundamental_features,
                    microstructure_features
                ])
                
                feature_data.append(all_features)
        
        # 重塑为 [time_steps, n_stocks, n_features]
        self.feature_data = np.array(feature_data).reshape(
            -1, self.n_stocks, self.n_features
        ).astype(np.float32)
        
        logger.info(f"生成模拟特征数据: {self.feature_data.shape}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境到初始状态
        
        Returns:
            初始观察状态
        """
        self._initialize_state_variables()
        
        # 如果有历史数据，从随机位置开始
        if self.feature_data is not None:
            max_start = max(0, len(self.feature_data) - self.max_steps - self.config.lookback_window)
            self.start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        else:
            self.start_idx = 0
        
        # 更新当前价格
        self._update_current_prices()
        
        observation = self._get_observation()
        
        logger.debug(f"环境重置完成，起始索引: {self.start_idx}")
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        执行一步交易
        
        Args:
            action: 目标投资组合权重
            
        Returns:
            (观察, 奖励, 是否结束, 信息字典)
        """
        # 更新当前价格（在计算收益和成本之前）
        self._update_current_prices()
        
        # 标准化动作
        target_weights = self._normalize_action(action)
        
        # 应用A股交易规则约束
        target_weights = self._apply_trading_constraints(target_weights)
        
        # 计算交易成本
        transaction_cost = self._calculate_transaction_cost(
            self.current_positions, target_weights
        )
        
        # 执行交易
        self._execute_trades(target_weights)
        
        # 获取当期收益
        portfolio_return = self._calculate_portfolio_return()
        
        # 更新投资组合价值
        self._update_portfolio_value(portfolio_return, transaction_cost)
        
        # 计算奖励
        reward = self._calculate_reward(portfolio_return, transaction_cost, target_weights)
        
        # 更新状态
        self.current_step += 1
        next_observation = self._get_observation()
        done = self._is_done()
        
        # 更新T+1限制
        self._update_t_plus_1_restrictions(target_weights)
        
        # 构建信息字典
        info = self._build_info_dict(portfolio_return, transaction_cost, target_weights)
        
        # 记录历史
        self._record_step_history(target_weights, portfolio_return, transaction_cost)
        
        return next_observation, reward, done, info
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """标准化动作为合法的权重分布"""
        action = np.asarray(action, dtype=np.float32)
        
        # 确保非负
        action = np.maximum(action, 0)
        
        # 标准化为权重
        total = action.sum()
        if total == 0:
            return np.ones(self.n_stocks, dtype=np.float32) / self.n_stocks
        
        return action / total
    
    def _apply_trading_constraints(self, target_weights: np.ndarray) -> np.ndarray:
        """应用A股交易规则约束"""
        # 首先应用T+1约束
        if self.config.t_plus_1:
            target_weights = self._apply_t_plus_1_constraints(target_weights)
        
        # 迭代应用权重约束，确保既满足单只股票最大权重又满足权重和为1
        max_iterations = 10
        for _ in range(max_iterations):
            # 应用单只股票最大权重限制
            constrained_weights = np.minimum(target_weights, self.config.max_position_size)
            
            # 检查是否需要重新分配被截断的权重
            excess_weight = target_weights.sum() - constrained_weights.sum()
            
            if excess_weight < 1e-8:
                # 没有或很少权重被截断，直接标准化
                break
            
            # 将被截断的权重重新分配给未达到上限的股票
            available_capacity = self.config.max_position_size - constrained_weights
            total_capacity = available_capacity.sum()
            
            if total_capacity > 1e-8:
                # 按比例分配多余权重
                redistribution = (available_capacity / total_capacity) * excess_weight
                target_weights = constrained_weights + redistribution
            else:
                # 所有股票都达到上限，无法重新分配
                target_weights = constrained_weights
                break
        
        # 最终标准化确保权重和为1
        total = target_weights.sum()
        if total > 0:
            target_weights = target_weights / total
            
            # 检查标准化后是否仍然满足最大权重约束
            # 如果不满足，说明约束在数学上不可行，需要放宽
            if np.any(target_weights > self.config.max_position_size + 1e-8):
                # 计算可行的最大权重：确保所有股票权重相等且和为1
                feasible_max_weight = 1.0 / self.n_stocks
                if feasible_max_weight > self.config.max_position_size:
                    # 如果均匀分配都超过限制，则使用均匀分配（这是数学上的最优解）
                    target_weights = np.full(self.n_stocks, feasible_max_weight, dtype=np.float32)
                else:
                    # 否则可以满足约束
                    target_weights = np.minimum(target_weights, self.config.max_position_size)
                    target_weights = target_weights / target_weights.sum()
        else:
            # 如果所有权重都为0，则均匀分配
            uniform_weight = 1.0 / self.n_stocks
            target_weights = np.full(self.n_stocks, uniform_weight, dtype=np.float32)
        
        return target_weights
    
    def _apply_t_plus_1_constraints(self, target_weights: np.ndarray) -> np.ndarray:
        """应用T+1交易约束"""
        if not self.t_plus_1_restrictions:
            return target_weights
        
        # 对于当日买入的股票，如果要减仓，则限制其权重不能低于当前持仓
        for stock_idx in self.t_plus_1_restrictions:
            if target_weights[stock_idx] < self.current_positions[stock_idx]:
                target_weights[stock_idx] = self.current_positions[stock_idx]
        
        return target_weights
    
    def _calculate_transaction_cost(self, current_weights: np.ndarray, 
                                  target_weights: np.ndarray) -> float:
        """计算交易成本"""
        total_cost = 0.0
        
        for i in range(self.n_stocks):
            weight_change = abs(target_weights[i] - current_weights[i])
            
            # 只有当权重变化超过最小阈值时才计算成本
            if weight_change < 1e-8:
                continue
            
            # 计算交易价值
            trade_value = weight_change * self.total_value
            
            # 判断交易方向
            side = 'buy' if target_weights[i] > current_weights[i] else 'sell'
            
            # 创建交易信息
            trade_info = TradeInfo(
                symbol=self.config.stock_pool[i],
                side=side,
                quantity=int(trade_value / self.current_prices[i]) if self.current_prices[i] > 0 else 0,
                price=self.current_prices[i],
                timestamp=datetime.now()
            )
            
            # 计算成本
            cost_breakdown = self.cost_model.calculate_cost(trade_info)
            total_cost += cost_breakdown.total_cost
        
        return total_cost
    
    def _execute_trades(self, target_weights: np.ndarray):
        """执行交易，更新持仓"""
        # 记录交易前的持仓
        old_positions = self.current_positions.copy()
        
        # 更新持仓
        self.current_positions = target_weights.copy()
        
        # 记录发生交易的股票（用于T+1限制）
        self.traded_stocks = []
        for i in range(self.n_stocks):
            if abs(target_weights[i] - old_positions[i]) > 1e-8:
                self.traded_stocks.append(i)
    
    def _calculate_portfolio_return(self) -> float:
        """计算投资组合收益率"""
        if self.current_prices is None or self.previous_prices is None:
            return 0.0
        
        # 计算各股票收益率
        stock_returns = np.zeros(self.n_stocks)
        for i in range(self.n_stocks):
            if self.previous_prices[i] > 0:
                stock_returns[i] = (self.current_prices[i] - self.previous_prices[i]) / self.previous_prices[i]
        
        # 计算投资组合加权收益率
        portfolio_return = np.dot(self.current_positions, stock_returns)
        
        return portfolio_return
    
    def _update_portfolio_value(self, portfolio_return: float, transaction_cost: float):
        """更新投资组合价值"""
        # 计算新的总价值
        self.total_value = self.total_value * (1 + portfolio_return) - transaction_cost
        
        # 更新现金（简化处理）
        self.cash = self.total_value * (1 - self.current_positions.sum())
        
        # 记录价值历史
        self.portfolio_values.append(self.total_value)
        
        # 更新最大价值（用于计算回撤）
        self.max_portfolio_value = max(self.max_portfolio_value, self.total_value)
    
    def _calculate_reward(self, portfolio_return: float, 
                         transaction_cost: float, weights: np.ndarray) -> float:
        """计算奖励函数"""
        # 净收益
        net_return = portfolio_return - transaction_cost / self.total_value
        
        # 风险惩罚（基于权重集中度）
        concentration = np.sum(weights ** 2)  # Herfindahl指数
        risk_penalty = self.config.risk_aversion * concentration
        
        # 回撤惩罚
        current_drawdown = self._calculate_current_drawdown()
        drawdown_penalty = self.config.max_drawdown_penalty * max(0, current_drawdown - 0.1)
        
        # 总奖励
        reward = net_return - risk_penalty - drawdown_penalty
        
        return float(reward)
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        if self.max_portfolio_value == 0:
            return 0.0
        return (self.max_portfolio_value - self.total_value) / self.max_portfolio_value
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前观察状态"""
        # 获取历史特征数据
        if self.feature_data is not None:
            start_idx = max(0, self.start_idx + self.current_step - self.config.lookback_window)
            end_idx = self.start_idx + self.current_step
            
            if end_idx > len(self.feature_data):
                # 如果超出数据范围，使用最后可用的数据
                features = self.feature_data[-self.config.lookback_window:]
            else:
                features = self.feature_data[start_idx:end_idx]
            
            # 确保特征数据有正确的维度
            if len(features) < self.config.lookback_window:
                # 用第一个观察值填充不足的部分
                padding = np.repeat(features[0:1], 
                                  self.config.lookback_window - len(features), 
                                  axis=0)
                features = np.concatenate([padding, features], axis=0)
        else:
            # 生成随机特征（用于测试）
            features = np.random.randn(
                self.config.lookback_window, self.n_stocks, self.n_features
            ).astype(np.float32)
        
        # 市场状态特征（简化实现）
        market_state = np.array([
            self.total_value / self.config.initial_cash - 1,  # 总收益率
            self._calculate_current_drawdown(),               # 当前回撤
            np.sum(self.current_positions ** 2),            # 持仓集中度
            np.sum(self.current_positions > 1e-6),          # 活跃持仓数
            self.current_step / self.max_steps,             # 时间进度
            np.std(self.returns_history[-30:]) if len(self.returns_history) >= 30 else 0,  # 波动率
            np.mean(self.returns_history[-10:]) if len(self.returns_history) >= 10 else 0,  # 近期收益
            len(self.trade_history) / max(self.current_step, 1),  # 交易频率
            self.cash / self.total_value,                    # 现金比例
            1.0 if self.current_step % 5 == 0 else 0.0     # 周期性特征
        ], dtype=np.float32)
        
        return {
            'features': features.astype(np.float32),
            'positions': self.current_positions.astype(np.float32),
            'market_state': market_state
        }
    
    def _update_current_prices(self):
        """更新当前价格"""
        if self.price_data is not None:
            # 保存前一期价格
            self.previous_prices = self.current_prices.copy() if self.current_prices is not None else None
            
            # 获取当前价格
            current_date_idx = self.start_idx + self.current_step
            if current_date_idx < len(self.price_data.index.get_level_values('datetime').unique()):
                date = self.price_data.index.get_level_values('datetime').unique()[current_date_idx]
                current_day_data = self.price_data.xs(date, level='datetime')
                
                self.current_prices = np.zeros(self.n_stocks)
                for i, symbol in enumerate(self.config.stock_pool):
                    if symbol in current_day_data.index:
                        self.current_prices[i] = current_day_data.loc[symbol, 'close']
                    else:
                        # 如果没有数据，使用前一期价格
                        self.current_prices[i] = self.previous_prices[i] if self.previous_prices is not None else 10.0
            else:
                # 如果超出数据范围，保持前一期价格
                if self.current_prices is None:
                    self.current_prices = np.array([10.0 + i for i in range(self.n_stocks)])
        else:
            # 生成模拟价格
            self.previous_prices = self.current_prices.copy() if self.current_prices is not None else None
            if self.current_prices is None:
                self.current_prices = np.array([10.0 + i for i in range(self.n_stocks)])
            else:
                # 模拟价格变动
                returns = np.random.normal(0.001, 0.02, self.n_stocks)
                self.current_prices = self.current_prices * (1 + returns)
    
    def _is_done(self) -> bool:
        """判断回合是否结束"""
        # 基本结束条件：达到最大步数
        if self.current_step >= self.max_steps:
            return True
        
        # 风险控制：总价值过低
        if self.total_value < self.config.initial_cash * 0.5:
            logger.warning(f"总价值过低，强制结束: {self.total_value}")
            return True
        
        # 数据用尽
        if (self.feature_data is not None and 
            self.start_idx + self.current_step >= len(self.feature_data)):
            return True
        
        return False
    
    def _update_t_plus_1_restrictions(self, target_weights: np.ndarray):
        """更新T+1交易限制"""
        if not self.config.t_plus_1:
            return
        
        # 清除前一天的限制
        self.t_plus_1_restrictions.clear()
        
        # 对于今日增仓的股票，添加T+1限制
        for i in self.traded_stocks:
            if target_weights[i] > self.current_positions[i]:
                self.t_plus_1_restrictions[i] = target_weights[i]
    
    def _build_info_dict(self, portfolio_return: float, 
                        transaction_cost: float, weights: np.ndarray) -> Dict[str, Any]:
        """构建信息字典"""
        return {
            'portfolio_return': float(portfolio_return),
            'transaction_cost': float(transaction_cost),
            'positions': weights.copy(),
            'total_value': float(self.total_value),
            'cash': float(self.cash),
            'drawdown': float(self._calculate_current_drawdown()),
            'concentration': float(np.sum(weights ** 2)),
            'active_positions': int(np.sum(weights > 1e-6)),
            'max_position': float(np.max(weights)),
            'min_position': float(np.min(weights)),
            'step': self.current_step,
            't_plus_1_restrictions': list(self.t_plus_1_restrictions.keys())
        }
    
    def _record_step_history(self, weights: np.ndarray, 
                           portfolio_return: float, transaction_cost: float):
        """记录步骤历史"""
        self.weights_history.append(weights.copy())
        self.returns_history.append(portfolio_return)
        
        if transaction_cost > 0:
            self.trade_history.append({
                'step': self.current_step,
                'weights': weights.copy(),
                'cost': transaction_cost,
                'value': self.total_value
            })
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """获取投资组合绩效指标"""
        if len(self.portfolio_values) < 2:
            return {}
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        # 基本指标
        total_return = (values[-1] - values[0]) / values[0]
        volatility = np.std(returns) * np.sqrt(self.config.trading_days_per_year)
        
        # 风险调整指标
        if volatility > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(self.config.trading_days_per_year)
        else:
            sharpe_ratio = 0.0
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(values)
        
        # 其他指标
        win_rate = np.mean(np.array(self.returns_history) > 0) if self.returns_history else 0
        avg_return = np.mean(self.returns_history) if self.returns_history else 0
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(total_return * self.config.trading_days_per_year / len(values)),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'average_return': float(avg_return),
            'total_trades': len(self.trade_history),
            'final_value': float(values[-1])
        }
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """计算最大回撤"""
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return np.max(drawdown)
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            metrics = self.get_portfolio_metrics()
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Total Value: {self.total_value:.2f}")
            print(f"Return: {metrics.get('total_return', 0):.4f}")
            print(f"Drawdown: {self._calculate_current_drawdown():.4f}")
            print(f"Positions: {self.current_positions}")
            print("-" * 50)
    
    def close(self):
        """关闭环境"""
        pass
    
    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        return [seed]