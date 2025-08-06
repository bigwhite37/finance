"""
交易环境实现
处理真实市场约束的强化学习环境
参考claude.md中的TradingEnv实现
"""
import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from collections import deque
import warnings

from src.data_loader import QlibDataLoader

logger = logging.getLogger(__name__)


class MarketMicrostructure:
    """
    市场微观结构模型
    处理交易成本、滑点、流动性约束等
    """

    def __init__(self,
                 commission_rate: float = 0.0003,  # 双边3bp
                 min_commission: float = 5.0,
                 slippage_model: str = 'linear',
                 impact_coefficient: float = 0.01):

        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage_model = slippage_model
        self.impact_coefficient = impact_coefficient

    def compute_transaction_cost(self,
                               trade_value: float,
                               volume_ratio: float = 0.01) -> float:
        """
        计算交易成本

        Args:
            trade_value: 交易金额
            volume_ratio: 交易量占成交量比例

        Returns:
            总交易成本
        """
        # 基础佣金
        commission = max(abs(trade_value) * self.commission_rate, self.min_commission)

        # 滑点成本
        if self.slippage_model == 'linear':
            slippage = abs(trade_value) * volume_ratio * self.impact_coefficient
        elif self.slippage_model == 'sqrt':
            slippage = abs(trade_value) * np.sqrt(volume_ratio) * self.impact_coefficient
        else:
            slippage = 0.0

        return commission + slippage

    def compute_slippage(self,
                        trade_size: float,
                        current_price: float,
                        market_depth: float = 1e6) -> float:
        """
        计算价格滑点

        Args:
            trade_size: 交易数量（正数买入，负数卖出）
            current_price: 当前价格
            market_depth: 市场深度

        Returns:
            滑点比例（相对于当前价格）
        """
        trade_value = abs(trade_size * current_price)
        impact_ratio = trade_value / market_depth

        if self.slippage_model == 'linear':
            slippage_pct = impact_ratio * self.impact_coefficient
        elif self.slippage_model == 'sqrt':
            slippage_pct = np.sqrt(impact_ratio) * self.impact_coefficient
        else:
            slippage_pct = 0.0

        # 买入正滑点，卖出负滑点
        return slippage_pct * np.sign(trade_size)


class RiskManager:
    """
    风险管理模块
    """

    def __init__(self,
                 max_position_pct: float = 0.1,  # 单只股票最大仓位10%
                 max_drawdown_pct: float = 0.15,  # 最大回撤15%
                 var_confidence: float = 0.95,
                 lookback_days: int = 30):

        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.var_confidence = var_confidence
        self.lookback_days = lookback_days

        # 历史数据
        self.return_history = deque(maxlen=lookback_days)
        self.portfolio_values = deque(maxlen=lookback_days * 2)

    def check_position_limit(self,
                           target_position: float,
                           current_price: float,
                           portfolio_value: float) -> float:
        """检查并调整仓位限制"""
        position_value = abs(target_position * current_price)
        max_position_value = portfolio_value * self.max_position_pct

        if position_value > max_position_value:
            # 调整仓位到最大允许值
            adjusted_position = np.sign(target_position) * max_position_value / current_price
            return adjusted_position

        return target_position

    def compute_var(self, confidence: Optional[float] = None) -> float:
        """计算VaR"""
        if len(self.return_history) < 10:
            return 0.0

        conf = confidence or self.var_confidence
        returns = np.array(self.return_history)
        return np.percentile(returns, (1 - conf) * 100)

    def compute_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.portfolio_values) < 2:
            return 0.0

        values = np.array(self.portfolio_values)
        peak = np.expand_dims(np.maximum.accumulate(values), -1)
        drawdown = (values - peak) / peak
        return np.min(drawdown)

    def update_metrics(self, portfolio_value: float, returns: float):
        """更新风险指标"""
        self.portfolio_values.append(portfolio_value)
        self.return_history.append(returns)

    def get_risk_metrics(self) -> Dict[str, float]:
        """获取风险指标"""
        return {
            'var_95': self.compute_var(0.95),
            'var_99': self.compute_var(0.99),
            'max_drawdown': self.compute_max_drawdown(),
            'volatility': np.std(self.return_history) if len(self.return_history) > 1 else 0.0,
            'sharpe_ratio': self._compute_sharpe_ratio()
        }

    def _compute_sharpe_ratio(self, risk_free_rate: float = 0.03) -> float:
        """计算夏普比率"""
        if len(self.return_history) < 2:
            return 0.0

        returns = np.array(self.return_history)
        excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)


class TradingEnvironment(gym.Env):
    """
    交易环境
    集成qlib数据、市场微观结构、风险管理的完整交易环境
    """

    def __init__(self,
                 instruments: List[str],
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 1000000.0,  # 100万初始资金
                 data_loader: Optional[QlibDataLoader] = None,
                 lookback_window: int = 60,  # 观测窗口60天
                 max_stocks: int = 50,  # 最大持仓股票数
                 transaction_cost_model: Optional[MarketMicrostructure] = None,
                 risk_manager: Optional[RiskManager] = None,
                 normalize_observations: bool = True,
                 random_start: bool = True):

        super().__init__()

        self.instruments = instruments[:max_stocks]  # 限制股票数量
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.max_stocks = max_stocks
        self.normalize_observations = normalize_observations
        self.random_start = random_start

        # 数据加载器
        self.data_loader = data_loader or QlibDataLoader()
        if not self.data_loader.is_initialized:
            self.data_loader.initialize_qlib()

        # 市场微观结构
        self.market_microstructure = transaction_cost_model or MarketMicrostructure()

        # 风险管理
        self.risk_manager = risk_manager or RiskManager()

        # 加载数据
        self._load_market_data()

        # 定义动作空间：连续权重分配 [-1, 1] 对应做空到做多
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(len(self.instruments),),
            dtype=np.float32
        )

        # 定义观测空间
        self._define_observation_space()

        # 状态变量
        self.current_step = 0
        self.current_date_idx = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = np.zeros(len(self.instruments))  # 持仓数量
        self.position_values = np.zeros(len(self.instruments))  # 持仓价值

        # 历史记录
        self.portfolio_history = []
        self.return_history = []
        self.transaction_history = []
        self.action_history = []

        # 性能指标
        self.total_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0

        logger.info(f"交易环境初始化完成: {len(self.instruments)}只股票, "
                   f"数据期间: {start_date} - {end_date}")

    def _load_market_data(self):
        """加载市场数据"""
        try:
            # 使用稳健的数据加载器
            from utils.data_utils import load_robust_stock_data
            
            fields = ["$open", "$high", "$low", "$close", "$volume", "$change", "$factor"]

            self.market_data, valid_instruments = load_robust_stock_data(
                instruments=self.instruments,
                start_time=self.start_date,
                end_time=self.end_date,
                freq="day",
                fields=fields,
                max_missing_ratio=0.1
            )
            
            # 更新股票列表为有效股票
            original_count = len(self.instruments)
            self.instruments = valid_instruments
            
            if len(self.instruments) != original_count:
                logger.info(f"股票池从 {original_count} 只调整为 {len(self.instruments)} 只")

            # 处理数据格式
            if isinstance(self.market_data.index, pd.MultiIndex):
                # 重新排列索引为 (datetime, instrument)
                self.market_data = self.market_data.swaplevel().sort_index()

            # 获取交易日历
            self.trading_calendar = sorted(self.market_data.index.get_level_values(0).unique())

            # 计算技术指标
            self._compute_technical_indicators()

            logger.info(f"数据加载完成: {self.market_data.shape}, "
                       f"交易日数量: {len(self.trading_calendar)}")

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise RuntimeError(f"无法加载市场数据: {e}")

    def _compute_technical_indicators(self):
        """计算技术指标"""
        try:
            # 为每只股票计算技术指标
            technical_data = []

            for instrument in self.instruments:
                try:
                    # 获取单只股票数据
                    stock_data = self.market_data.xs(instrument, level=1)

                    if stock_data.empty:
                        logger.warning(f"股票 {instrument} 数据为空，跳过")
                        continue

                    # 计算技术指标
                    indicators = pd.DataFrame(index=stock_data.index)

                    # 价格相关
                    indicators['returns'] = stock_data['$close'].pct_change()
                    indicators['log_returns'] = np.log(stock_data['$close'] / stock_data['$close'].shift(1))

                    # 移动平均
                    for window in [5, 10, 20, 60]:
                        indicators[f'ma_{window}'] = stock_data['$close'].rolling(window).mean()
                        indicators[f'ma_ratio_{window}'] = stock_data['$close'] / indicators[f'ma_{window}'] - 1

                    # 波动率
                    indicators['volatility_20'] = indicators['returns'].rolling(20).std()
                    indicators['volatility_60'] = indicators['returns'].rolling(60).std()

                    # RSI
                    delta = stock_data['$close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    indicators['rsi'] = 100 - (100 / (1 + rs))

                    # MACD
                    exp1 = stock_data['$close'].ewm(span=12).mean()
                    exp2 = stock_data['$close'].ewm(span=26).mean()
                    indicators['macd'] = exp1 - exp2
                    indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
                    indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']

                    # 成交量指标
                    indicators['volume_ma_20'] = stock_data['$volume'].rolling(20).mean()
                    indicators['volume_ratio'] = stock_data['$volume'] / indicators['volume_ma_20']

                    # 价格位置
                    indicators['price_position'] = (stock_data['$close'] - stock_data['$low'].rolling(20).min()) / \
                                                 (stock_data['$high'].rolling(20).max() - stock_data['$low'].rolling(20).min())

                    # 添加股票标识
                    indicators['instrument'] = instrument
                    technical_data.append(indicators)

                except Exception as e:
                    logger.warning(f"计算 {instrument} 技术指标失败: {e}")
                    continue

            if technical_data:
                # 合并所有股票的技术指标
                self.technical_indicators = pd.concat(technical_data)
                self.technical_indicators = self.technical_indicators.reset_index().set_index(['datetime', 'instrument'])

                # 填充缺失值
                self.technical_indicators = self.technical_indicators.fillna(method='ffill').fillna(0)

                logger.info(f"技术指标计算完成: {self.technical_indicators.shape}")
            else:
                # 如果没有成功计算任何指标，创建空的DataFrame
                self.technical_indicators = pd.DataFrame()
                logger.warning("未能计算任何技术指标")

        except Exception as e:
            logger.error(f"技术指标计算失败: {e}")
            self.technical_indicators = pd.DataFrame()

    def _define_observation_space(self):
        """定义观测空间"""
        # 基础特征维度
        basic_features = 7  # OHLCV + change + factor
        technical_features = 20  # 技术指标数量
        position_features = 3  # 当前仓位、仓位价值、权重
        portfolio_features = 5  # 组合价值、现金、总收益、回撤、夏普比率

        features_per_stock = basic_features + technical_features + position_features
        total_features = features_per_stock * len(self.instruments) + portfolio_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback_window, total_features),
            dtype=np.float32
        )

        logger.info(f"观测空间定义完成: {self.observation_space.shape}")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple:
        """重置环境"""
        super().reset(seed=seed)

        # 重置状态
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = np.zeros(len(self.instruments))
        self.position_values = np.zeros(len(self.instruments))

        # 选择起始日期
        if self.random_start and len(self.trading_calendar) > self.lookback_window + 60:
            # 随机选择起始点，留出足够的数据用于观测
            max_start_idx = len(self.trading_calendar) - self.lookback_window - 60
            self.current_date_idx = np.random.randint(self.lookback_window, max_start_idx)
        else:
            self.current_date_idx = self.lookback_window

        self.current_step = 0

        # 清空历史记录
        self.portfolio_history = []
        self.return_history = []
        self.transaction_history = []
        self.action_history = []

        # 重置风险管理器
        self.risk_manager = RiskManager()

        observation = self._get_observation()
        info = {
            'current_step': self.current_step,
            'current_date_idx': self.current_date_idx,
            'portfolio_value': self.portfolio_value
        }
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步交易"""
        # 确保动作在有效范围内
        action = np.clip(action, -1.0, 1.0)

        # 记录动作
        self.action_history.append(action.copy())

        # 获取当前市场数据
        current_date = self.trading_calendar[self.current_date_idx]

        try:
            current_prices = self._get_current_prices(current_date)

            # 检查价格数据有效性
            if np.any(np.isnan(current_prices)) or np.any(current_prices <= 0):
                logger.warning(f"日期 {current_date} 存在无效价格数据")
                # 使用前一日价格
                if self.current_date_idx > 0:
                    prev_date = self.trading_calendar[self.current_date_idx - 1]
                    current_prices = self._get_current_prices(prev_date)
                else:
                    # 如果是第一天，跳过交易
                    self.current_step += 1
                    self.current_date_idx += 1
                    done = self._is_done()
                    return self._get_observation(), 0.0, done, False, {}

        except Exception as e:
            logger.error(f"获取价格数据失败: {e}")
            self.current_step += 1
            self.current_date_idx += 1
            return self._get_observation(), 0.0, True, False, {'error': str(e)}

        # 计算目标仓位
        portfolio_value_before = self._calculate_portfolio_value(current_prices)
        target_weights = action  # 动作直接表示权重
        target_positions = self._weights_to_positions(target_weights, current_prices, portfolio_value_before)

        # 应用风险管理约束
        target_positions = self._apply_risk_constraints(target_positions, current_prices, portfolio_value_before)

        # 执行交易
        transaction_cost = self._execute_trades(target_positions, current_prices)

        # 更新组合价值
        portfolio_value_after = self._calculate_portfolio_value(current_prices)

        # 计算收益
        returns = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        reward = self._calculate_reward(returns, transaction_cost, portfolio_value_before)

        # 更新状态
        self.portfolio_value = portfolio_value_after
        self.portfolio_history.append(portfolio_value_after)
        self.return_history.append(returns)

        # 更新风险指标
        self.risk_manager.update_metrics(portfolio_value_after, returns)

        # 准备下一步
        self.current_step += 1
        self.current_date_idx += 1

        # 检查是否结束
        done = self._is_done()

        # 构造信息字典
        info = self._get_info_dict(current_prices, returns, transaction_cost, reward)

        return self._get_observation(), reward, done, False, info

    def _get_current_prices(self, date: pd.Timestamp) -> np.ndarray:
        """获取当前日期的价格数据"""
        prices = np.zeros(len(self.instruments))

        for i, instrument in enumerate(self.instruments):
            try:
                # 数据索引格式为 (datetime, instrument)，在swaplevel()后
                if (date, instrument) in self.market_data.index:
                    price = self.market_data.loc[(date, instrument), '$close']
                    if pd.isna(price) or price <= 0:
                        # 使用前值填充
                        price = self._get_last_valid_price(instrument, date)
                    prices[i] = price
                else:
                    # 如果没有数据，使用最近的有效价格
                    prices[i] = self._get_last_valid_price(instrument, date)

            except Exception as e:
                logger.warning(f"获取 {instrument} 在 {date} 的价格失败: {e}")
                prices[i] = self._get_last_valid_price(instrument, date)

        return prices

    def _get_last_valid_price(self, instrument: str, current_date: pd.Timestamp) -> float:
        """获取最近的有效价格"""
        try:
            # 获取该股票的所有历史数据
            stock_data = self.market_data.xs(instrument, level=1)
            # 获取当前日期之前的数据（包括当前日期）
            valid_data = stock_data[stock_data.index <= current_date]
            
            if not valid_data.empty:
                # 从后往前查找最近的有效价格
                for i in range(len(valid_data) - 1, -1, -1):
                    price = valid_data['$close'].iloc[i]
                    if pd.notna(price) and price > 0:
                        return float(price)
                
                # 如果当前日期之前都没有有效价格，向后查找
                future_data = stock_data[stock_data.index > current_date]
                if not future_data.empty:
                    for i in range(len(future_data)):
                        price = future_data['$close'].iloc[i]
                        if pd.notna(price) and price > 0:
                            logger.debug(f"使用股票 {instrument} 在 {current_date} 后的价格: {price}")
                            return float(price)

            # 如果还是没有有效价格，抛出错误
            raise RuntimeError(f"无法获取股票 {instrument} 在 {current_date} 的任何有效价格数据")

        except (KeyError, ValueError, IndexError, pd.errors.EmptyDataError) as e:
            raise RuntimeError(f"获取股票 {instrument} 历史价格失败: {e}")

    def _weights_to_positions(self, weights: np.ndarray, prices: np.ndarray, portfolio_value: float) -> np.ndarray:
        """将权重转换为持仓数量"""
        target_values = weights * portfolio_value
        # 转换为股数（向下取整到100股的倍数）
        target_positions = np.floor(target_values / prices / 100) * 100
        return target_positions

    def _apply_risk_constraints(self, target_positions: np.ndarray, prices: np.ndarray, portfolio_value: float) -> np.ndarray:
        """应用风险管理约束"""
        constrained_positions = target_positions.copy()

        for i, (pos, price) in enumerate(zip(target_positions, prices)):
            # 单只股票仓位限制
            constrained_pos = self.risk_manager.check_position_limit(pos, price, portfolio_value)
            constrained_positions[i] = constrained_pos

        return constrained_positions

    def _execute_trades(self, target_positions: np.ndarray, current_prices: np.ndarray) -> float:
        """执行交易并计算交易成本"""
        total_transaction_cost = 0.0
        transactions = []

        for i, (target_pos, current_pos, price) in enumerate(zip(target_positions, self.positions, current_prices)):
            trade_size = target_pos - current_pos

            if abs(trade_size) > 0:  # 有交易发生
                trade_value = trade_size * price

                # 计算交易成本
                transaction_cost = self.market_microstructure.compute_transaction_cost(trade_value)

                # 计算滑点
                slippage_pct = self.market_microstructure.compute_slippage(trade_size, price)
                execution_price = price * (1 + slippage_pct)

                # 检查现金是否足够
                required_cash = trade_size * execution_price + transaction_cost
                if trade_size > 0 and required_cash > self.cash:
                    # 资金不足，调整交易量
                    affordable_shares = (self.cash - transaction_cost) / execution_price
                    trade_size = max(0, affordable_shares)
                    target_pos = current_pos + trade_size
                    required_cash = trade_size * execution_price + transaction_cost

                # 执行交易
                if abs(trade_size) > 0:
                    self.positions[i] = target_pos
                    self.cash -= required_cash
                    total_transaction_cost += transaction_cost

                    # 记录交易
                    transactions.append({
                        'instrument': self.instruments[i],
                        'trade_size': trade_size,
                        'price': execution_price,
                        'cost': transaction_cost,
                        'slippage_pct': slippage_pct
                    })

        # 更新持仓价值
        self.position_values = self.positions * current_prices

        # 记录交易历史
        if transactions:
            self.transaction_history.append({
                'date': self.trading_calendar[self.current_date_idx],
                'transactions': transactions,
                'total_cost': total_transaction_cost
            })

        return total_transaction_cost

    def _calculate_portfolio_value(self, current_prices: np.ndarray) -> float:
        """计算组合价值"""
        position_value = np.sum(self.positions * current_prices)
        return self.cash + position_value

    def _calculate_reward(self, returns: float, transaction_cost: float, portfolio_value: float) -> float:
        """计算奖励"""
        # 基础收益奖励
        base_reward = returns

        # 交易成本惩罚
        cost_penalty = transaction_cost / portfolio_value

        # 风险调整
        risk_metrics = self.risk_manager.get_risk_metrics()

        # 风险惩罚
        risk_penalty = 0.0
        if risk_metrics['max_drawdown'] < -self.risk_manager.max_drawdown_pct:
            risk_penalty += abs(risk_metrics['max_drawdown']) * 0.5

        # 夏普比率奖励
        sharpe_bonus = 0.0
        if len(self.return_history) > 30:
            sharpe_bonus = max(0, risk_metrics['sharpe_ratio'] - 1.0) * 0.1

        # 总奖励
        total_reward = base_reward - cost_penalty - risk_penalty + sharpe_bonus

        return total_reward

    def _get_observation(self) -> np.ndarray:
        """获取当前观测"""
        # 获取历史数据窗口
        start_idx = max(0, self.current_date_idx - self.lookback_window)
        end_idx = self.current_date_idx

        window_dates = self.trading_calendar[start_idx:end_idx]

        # 如果数据不足，填充
        if len(window_dates) < self.lookback_window:
            padding_size = self.lookback_window - len(window_dates)
            window_dates = [window_dates[0]] * padding_size + list(window_dates)

        observations = []

        for date in window_dates:
            obs_features = []

            # 为每只股票收集特征
            for i, instrument in enumerate(self.instruments):
                # 基础行情特征
                basic_features = self._get_basic_features(instrument, date)

                # 技术指标特征
                technical_features = self._get_technical_features(instrument, date)

                # 持仓特征
                position_features = self._get_position_features(i)

                # 合并股票特征
                stock_features = np.concatenate([basic_features, technical_features, position_features])
                obs_features.extend(stock_features)

            # 组合级别特征
            portfolio_features = self._get_portfolio_features()
            obs_features.extend(portfolio_features)

            observations.append(obs_features)

        # 转换为numpy数组
        obs_array = np.array(observations, dtype=np.float32)

        # 归一化（可选）
        if self.normalize_observations:
            obs_array = self._normalize_observation(obs_array)

        # 处理缺失值
        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)

        return obs_array

    def _get_basic_features(self, instrument: str, date: pd.Timestamp) -> np.ndarray:
        """获取基础行情特征"""
        try:
            if (date, instrument) in self.market_data.index:
                row = self.market_data.loc[(date, instrument)]
                features = [
                    row.get('$open', 0),
                    row.get('$high', 0),
                    row.get('$low', 0),
                    row.get('$close', 0),
                    row.get('$volume', 0),
                    row.get('$change', 0),
                    row.get('$factor', 1)
                ]
                return np.array(features, dtype=np.float32)
        except (KeyError, ValueError, IndexError) as e:
            raise RuntimeError(f"获取基础特征失败，股票: {instrument}, 日期: {date}, 错误: {e}")

        # 如果没有找到数据，抛出错误而不是返回默认值
        raise RuntimeError(f"股票 {instrument} 在日期 {date} 没有基础行情数据")

    def _get_technical_features(self, instrument: str, date: pd.Timestamp) -> np.ndarray:
        """获取技术指标特征"""
        try:
            if not self.technical_indicators.empty and (date, instrument) in self.technical_indicators.index:
                row = self.technical_indicators.loc[(date, instrument)]
                features = [
                    row.get('returns', 0),
                    row.get('log_returns', 0),
                    row.get('ma_ratio_5', 0),
                    row.get('ma_ratio_10', 0),
                    row.get('ma_ratio_20', 0),
                    row.get('ma_ratio_60', 0),
                    row.get('volatility_20', 0),
                    row.get('volatility_60', 0),
                    row.get('rsi', 50),
                    row.get('macd', 0),
                    row.get('macd_signal', 0),
                    row.get('macd_hist', 0),
                    row.get('volume_ratio', 1),
                    row.get('price_position', 0.5),
                    0, 0, 0, 0, 0, 0  # 预留6个特征位
                ]
                return np.array(features[:20], dtype=np.float32)
        except (KeyError, ValueError, IndexError) as e:
            raise RuntimeError(f"获取技术特征失败，股票: {instrument}, 日期: {date}, 错误: {e}")

        # 如果没有技术指标数据，抛出错误
        raise RuntimeError(f"股票 {instrument} 在日期 {date} 没有技术指标数据")

    def _get_position_features(self, stock_idx: int) -> np.ndarray:
        """获取持仓特征"""
        position = self.positions[stock_idx]
        position_value = self.position_values[stock_idx]
        weight = position_value / max(self.portfolio_value, 1.0)

        return np.array([position, position_value, weight], dtype=np.float32)

    def _get_portfolio_features(self) -> np.ndarray:
        """获取组合特征"""
        risk_metrics = self.risk_manager.get_risk_metrics()

        features = [
            self.portfolio_value / self.initial_capital - 1,  # 总收益率
            self.cash / self.portfolio_value,  # 现金比例
            risk_metrics.get('max_drawdown', 0),  # 最大回撤
            risk_metrics.get('sharpe_ratio', 0),  # 夏普比率
            risk_metrics.get('volatility', 0)  # 波动率
        ]

        return np.array(features, dtype=np.float32)

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测数据"""
        # 简单的标准化
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = np.nanmean(obs, axis=0, keepdims=True)
            std = np.nanstd(obs, axis=0, keepdims=True)
            std = np.where(std == 0, 1, std)  # 避免除零
            normalized = (obs - mean) / std

        return normalized

    def _is_done(self) -> bool:
        """检查是否结束"""
        # 时间结束
        if self.current_date_idx >= len(self.trading_calendar):
            return True

        # 资金耗尽
        if self.portfolio_value <= self.initial_capital * 0.1:  # 损失90%
            return True

        # 最大回撤超限
        if self.risk_manager.compute_max_drawdown() < -0.5:  # 最大回撤50%
            return True

        return False

    def _get_info_dict(self, current_prices: np.ndarray, returns: float,
                      transaction_cost: float, reward: float) -> Dict:
        """构造信息字典"""
        risk_metrics = self.risk_manager.get_risk_metrics()

        return {
            'date': self.trading_calendar[self.current_date_idx-1].strftime('%Y-%m-%d'),
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'returns': returns,
            'transaction_cost': transaction_cost,
            'reward': reward,
            'positions': self.positions.tolist(),
            'position_values': self.position_values.tolist(),
            'weights': (self.position_values / max(self.portfolio_value, 1.0)).tolist(),
            'risk_metrics': risk_metrics,
            'total_return': (self.portfolio_value / self.initial_capital - 1),
            'step': self.current_step
        }

    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Date: {self.trading_calendar[self.current_date_idx-1].strftime('%Y-%m-%d')}")
            print(f"Portfolio Value: {self.portfolio_value:,.2f}")
            print(f"Total Return: {(self.portfolio_value/self.initial_capital-1)*100:.2f}%")
            print(f"Cash: {self.cash:,.2f}")
            print(f"Positions: {len(np.nonzero(self.positions)[0])} stocks")

            risk_metrics = self.risk_manager.get_risk_metrics()
            print(f"Max Drawdown: {risk_metrics['max_drawdown']*100:.2f}%")
            print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.3f}")
            print("-" * 50)

    def get_portfolio_stats(self) -> Dict[str, Any]:
        """获取组合统计信息"""
        if len(self.return_history) < 2:
            return {}

        returns = np.array(self.return_history)

        stats = {
            'total_return': (self.portfolio_value / self.initial_capital - 1),
            'annualized_return': np.mean(returns) * 252,
            'volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': self.risk_manager.get_risk_metrics()['sharpe_ratio'],
            'max_drawdown': self.risk_manager.get_risk_metrics()['max_drawdown'],
            'win_rate': np.mean(returns > 0),
            'profit_factor': np.sum(returns[returns > 0]) / abs(np.sum(returns[returns < 0])) if np.any(returns < 0) else np.inf,
            'total_trades': len(self.transaction_history),
            'trading_days': len(self.return_history),
            'portfolio_value': self.portfolio_value,
            'cash_ratio': self.cash / self.portfolio_value
        }

        return stats


if __name__ == "__main__":
    # 测试交易环境
    print("测试交易环境...")

    # 创建数据加载器
    data_loader = QlibDataLoader()
    data_loader.initialize_qlib()

    # 获取股票列表
    instruments = data_loader.get_stock_list(market="csi300", limit=10)
    print(f"测试股票: {instruments[:5]}...")

    # 创建交易环境
    env = TradingEnvironment(
        instruments=instruments[:5],  # 只用5只股票测试
        start_date="2020-01-01",
        end_date="2020-12-31",
        initial_capital=1000000,
        data_loader=data_loader,
        lookback_window=20,
        random_start=False
    )

    print(f"环境创建成功")
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")

    # 测试环境
    obs = env.reset()
    print(f"初始观测形状: {obs.shape}")

    # 随机交易几步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print(f"Step {i+1}:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Portfolio Value: {info['portfolio_value']:,.2f}")
        print(f"  Total Return: {info['total_return']*100:.2f}%")

        if done:
            print("环境结束")
            break

    # 显示最终统计
    stats = env.get_portfolio_stats()
    print("最终统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("交易环境测试完成")