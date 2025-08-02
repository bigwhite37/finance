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
from ..risk_control.risk_controller import (
    RiskController, RiskControlConfig, Portfolio, Position, Trade, TradeDecision,
    RiskViolationType, RiskLevel
)

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
    
    # 风险控制配置
    enable_risk_control: bool = True        # 启用风险控制
    risk_config: Optional[RiskControlConfig] = None  # 风险控制配置
    
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
        self.n_features_per_stock = 12  # 每只股票12个特征
        self.n_market_features = 10  # 默认10个市场特征
        
        # 初始化交易成本模型
        cost_params = CostParameters(
            commission_rate=config.commission_rate,
            stamp_tax_rate=config.stamp_tax_rate,
            min_commission=config.min_commission,
            transfer_fee_rate=config.transfer_fee_rate
        )
        self.cost_model = TransactionCostModel(cost_params)
        
        # 初始化风险控制器
        if config.enable_risk_control:
            risk_config = config.risk_config or RiskControlConfig(
                max_position_weight=config.max_position_size,
                stop_loss_threshold=0.05,
                max_daily_loss=0.02,
                max_drawdown=0.1
            )
            self.risk_controller = RiskController(risk_config)
            logger.info("风险控制器已启用")
        else:
            self.risk_controller = None
            logger.info("风险控制器已禁用")
        
        # 定义观察空间
        self.observation_space = spaces.Dict({
            'features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(config.lookback_window, self.n_stocks, self.n_features_per_stock),
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
        
        # 初始化市场数据变量（在_initialize_state_variables之前）
        self.market_data = None
        self.feature_data = None
        self.price_data = None
        self.current_prices = None
        
        # 初始化环境状态变量
        self._initialize_state_variables()
        
        # 加载历史数据
        self._load_market_data()
        
        logger.info(f"投资组合环境初始化完成: {self.n_stocks}只股票")
        logger.debug(f"观察空间: {self.observation_space}")
        logger.debug(f"动作空间: {self.action_space}")
    
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
        
        # 价格相关（重置时需要清除）
        self.current_prices = None
    
    def _load_market_data(self):
        """加载市场数据"""
        if not self.start_date or not self.end_date:
            raise ValueError("必须指定开始日期和结束日期才能加载市场数据")
        
        # 从数据接口加载真实数据
        raw_price_data = self.data_interface.get_price_data(
            symbols=self.config.stock_pool,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 如果没有获取到数据，抛出异常
        if raw_price_data.empty:
            raise ValueError(f"无法获取股票数据: symbols={self.config.stock_pool}, "
                           f"start_date={self.start_date}, end_date={self.end_date}")
        
        # 数据清洗：处理缺失值
        self.price_data = self._clean_price_data(raw_price_data)
        logger.info(f"数据清洗完成: 原始数据{len(raw_price_data)}条记录，清洗后{len(self.price_data)}条记录")
        
        # 提取可用的交易日期
        datetime_level = 'datetime' if 'datetime' in self.price_data.index.names else 1
        self.dates = self.price_data.index.get_level_values(datetime_level).unique().sort_values()
        
        # 加载基准指数数据（沪深300）
        benchmark_symbol = "000300.SH"  # 沪深300指数
        logger.debug(f"加载基准指数数据: {benchmark_symbol}")
        
        self.benchmark_data = self.data_interface.get_price_data(
            symbols=[benchmark_symbol],
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if not self.benchmark_data.empty:
            logger.debug(f"原始基准数据结构: index={self.benchmark_data.index.names}, columns={list(self.benchmark_data.columns)}")
            logger.debug(f"基准数据前5行:\n{self.benchmark_data.head()}")
            
            # 如果基准数据是多层索引，需要展平为单一时间序列
            if isinstance(self.benchmark_data.index, pd.MultiIndex):
                # 对于基准指数，我们只需要时间序列数据，不需要多层索引
                # 重置索引，保留datetime作为索引
                self.benchmark_data = self.benchmark_data.reset_index()
                
                # 找到时间列
                time_col = None
                for col in ['datetime', 'date']:
                    if col in self.benchmark_data.columns:
                        time_col = col
                        break
                
                if time_col:
                    self.benchmark_data = self.benchmark_data.set_index(time_col)
                    # 如果有多个股票（虽然基准应该只有一个），取第一个
                    if 'instrument' in self.benchmark_data.columns:
                        # 只保留基准指数的数据
                        benchmark_instruments = self.benchmark_data['instrument'].unique()
                        if len(benchmark_instruments) > 0:
                            self.benchmark_data = self.benchmark_data[
                                self.benchmark_data['instrument'] == benchmark_instruments[0]
                            ].drop('instrument', axis=1)
                else:
                    raise RuntimeError("无法找到时间列，基准数据格式不正确")
            
            if self.benchmark_data is not None:
                # 确保数据按时间排序
                self.benchmark_data = self.benchmark_data.sort_index()
                logger.debug(f"处理后基准数据结构: index={self.benchmark_data.index.name}, columns={list(self.benchmark_data.columns)}")
                logger.debug(f"基准数据加载成功: {len(self.benchmark_data)}条记录")
            else:
                raise RuntimeError("基准数据处理失败")
        else:
            raise RuntimeError("基准数据为空，无法加载基准指数数据")
        
        # 计算特征
        self.feature_data = self.feature_engineer.calculate_features(self.price_data)
        
        # 动态计算每只股票的特征数
        if not self.feature_data.empty:
            self.n_features_per_stock = self.feature_data.shape[1]  # 每只股票都有相同的特征数
            logger.info(f"动态计算特征维度: 每只股票特征数={self.n_features_per_stock}, 股票数={self.n_stocks}")
            
            # 重新定义观察空间
            self.observation_space = spaces.Dict({
                'features': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.config.lookback_window, self.n_stocks, self.n_features_per_stock),
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
        
        # 更新最大步数
        if isinstance(self.price_data.index, pd.MultiIndex):
            # 找到日期层级
            datetime_level = None
            for level_name in self.price_data.index.names:
                if level_name and ('time' in level_name.lower() or 'date' in level_name.lower()):
                    datetime_level = level_name
                    break
            
            if datetime_level:
                unique_dates = self.price_data.index.get_level_values(datetime_level).unique()
                self.max_steps = len(unique_dates)
                self._max_steps = self.max_steps  # 保存实际数据长度
            else:
                # 如果没有找到日期层级，使用数据长度除以股票数量
                self.max_steps = len(self.price_data) // len(self.config.stock_pool) if len(self.config.stock_pool) > 0 else len(self.price_data)
                self._max_steps = self.max_steps
        else:
            # 如果没有多层索引，使用数据长度
            self.max_steps = len(self.price_data)
            self._max_steps = self.max_steps
        
        logger.info(f"加载市场数据完成: {len(self.price_data)}条记录, {self.max_steps}个交易日")
    
    def _clean_price_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        清洗价格数据，处理缺失值和节假日数据
        
        Args:
            raw_data: 原始价格数据
            
        Returns:
            清洗后的价格数据
            
        Raises:
            RuntimeError: 当数据质量问题无法修复时
        """
        if raw_data.empty:
            raise RuntimeError("原始价格数据为空，无法进行清洗")
        
        # 检查数据结构
        if not isinstance(raw_data.index, pd.MultiIndex):
            raise RuntimeError("价格数据必须是多层索引格式 (datetime, instrument)")
        
        # 找到日期层级
        datetime_level = None
        instrument_level = None
        for level_name in raw_data.index.names:
            if level_name and ('time' in level_name.lower() or 'date' in level_name.lower()):
                datetime_level = level_name
            elif level_name and ('instrument' in level_name.lower() or 'symbol' in level_name.lower()):
                instrument_level = level_name
        
        if not datetime_level or not instrument_level:
            raise RuntimeError(f"无法识别数据索引层级: {raw_data.index.names}")
        
        # 复制数据进行清洗
        cleaned_data = raw_data.copy()
        
        # 第一步：找到所有股票都有有效数据的第一个交易日
        valid_start_date = self._find_first_valid_trading_date(cleaned_data, datetime_level, instrument_level)
        
        # 第二步：从有效开始日期截取数据
        all_dates = cleaned_data.index.get_level_values(datetime_level).unique().sort_values()
        valid_dates = all_dates[all_dates >= valid_start_date]
        
        if len(valid_dates) == 0:
            raise RuntimeError("没有找到任何有效的交易日数据")
        
        # 过滤数据，只保留有效日期范围内的数据
        cleaned_data = cleaned_data.loc[cleaned_data.index.get_level_values(datetime_level).isin(valid_dates)]
        
        # 第三步：按股票分组进行数据清洗
        for symbol in self.config.stock_pool:
            try:
                # 提取单只股票的数据
                symbol_data = cleaned_data.xs(symbol, level=instrument_level)
                
                # 检查是否有任何有效数据
                if symbol_data.isnull().all().all():
                    raise RuntimeError(f"股票{symbol}没有任何有效的价格数据")
                
                # 现在第一个交易日应该是有效的，但为了安全起见再次检查
                first_day_data = symbol_data.iloc[0]
                if first_day_data.isnull().any():
                    raise RuntimeError(f"数据清洗后第一个交易日仍然无效，股票: {symbol}, 日期: {symbol_data.index[0]}")
                
                # 前向填充缺失值（处理中间的节假日）
                filled_data = symbol_data.ffill()
                
                # 检查填充后是否还有NaN
                if filled_data.isnull().any().any():
                    # 如果还有NaN，说明数据有问题，抛出异常
                    nan_info = filled_data.isnull().sum()
                    raise RuntimeError(f"股票{symbol}前向填充后仍有缺失值: {nan_info.to_dict()}")
                
                # 将填充后的数据放回原DataFrame
                for col in filled_data.columns:
                    cleaned_data.loc[(symbol, slice(None)), col] = filled_data[col].values
                    
            except KeyError:
                raise RuntimeError(f"股票{symbol}在价格数据中不存在")
        
        # 最终验证：确保没有NaN值
        if cleaned_data.isnull().any().any():
            nan_summary = cleaned_data.isnull().sum()
            raise RuntimeError(f"数据清洗失败，仍存在缺失值: {nan_summary.to_dict()}")
        
        logger.info(f"价格数据清洗成功: {len(self.config.stock_pool)}只股票, "
                   f"{len(cleaned_data.index.get_level_values(datetime_level).unique())}个交易日, "
                   f"有效开始日期: {valid_start_date}")
        
        return cleaned_data
    
    def _find_first_valid_trading_date(self, data: pd.DataFrame, datetime_level: str, instrument_level: str) -> pd.Timestamp:
        """
        找到所有股票都有有效数据的第一个交易日
        
        Args:
            data: 原始数据
            datetime_level: 日期层级名称
            instrument_level: 股票层级名称
            
        Returns:
            第一个有效交易日的日期
            
        Raises:
            RuntimeError: 如果找不到有效的交易日
        """
        all_dates = data.index.get_level_values(datetime_level).unique().sort_values()
        
        for date in all_dates:
            # 检查这个日期是否所有股票都有有效数据
            date_data = data.xs(date, level=datetime_level)
            
            # 检查是否所有股票在这个日期都有数据
            missing_symbols = set(self.config.stock_pool) - set(date_data.index)
            if missing_symbols:
                logger.debug(f"日期{date}缺少股票数据: {missing_symbols}")
                continue
            
            # 检查是否所有股票的数据都是有效的（非NaN）
            all_valid = True
            for symbol in self.config.stock_pool:
                if symbol not in date_data.index:
                    logger.debug(f"日期{date}股票{symbol}不在数据中")
                    all_valid = False
                    break
                
                symbol_row = date_data.loc[symbol]
                # 检查关键价格字段是否有效
                key_fields = ['open', 'high', 'low', 'close', 'volume']
                for field in key_fields:
                    if field in symbol_row.index:
                        value = symbol_row[field]
                        if pd.isna(value):
                            logger.debug(f"日期{date}股票{symbol}字段{field}为NaN")
                            all_valid = False
                            break
                        elif value <= 0:
                            logger.debug(f"日期{date}股票{symbol}字段{field}值无效: {value}")
                            all_valid = False
                            break
                    else:
                        logger.debug(f"日期{date}股票{symbol}缺少字段{field}")
                        all_valid = False
                        break
                
                if not all_valid:
                    break
            
            if all_valid:
                logger.info(f"找到第一个有效交易日: {date}")
                return date
        
        # 如果没有找到任何有效的交易日，抛出异常
        raise RuntimeError("没有找到任何所有股票都有有效数据的交易日")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境到初始状态
        
        Returns:
            初始观察状态
        """
        self._initialize_state_variables()
        
        # 如果有历史数据，从随机位置开始
        # 使用价格数据的实际日期数量来计算起始索引
        if self.price_data is not None and isinstance(self.price_data.index, pd.MultiIndex):
            # 找到日期层级
            datetime_level = None
            for level_name in self.price_data.index.names:
                if level_name and ('time' in level_name.lower() or 'date' in level_name.lower()):
                    datetime_level = level_name
                    break
            
            if datetime_level:
                unique_dates = self.price_data.index.get_level_values(datetime_level).unique()
                available_dates = len(unique_dates)
                # 确保有足够的数据进行一个完整的episode
                max_start = max(0, available_dates - self.max_steps)
                self.start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            else:
                self.start_idx = 0
        else:
            self.start_idx = 0
        
        # 初始化价格：在reset时设置初始价格，但不设置previous_prices
        # 这样第一步的收益率会是0，这是正确的
        self._update_current_prices()
        # 在reset时，previous_prices应该为None，这样第一步收益率为0
        self.previous_prices = None
        
        observation = self._get_observation()
        
        logger.debug(f"环境重置完成，起始索引: {self.start_idx}, 初始价格: {self.current_prices}")
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        执行一步交易
        
        Args:
            action: 目标投资组合权重
            
        Returns:
            (观察, 奖励, 是否结束, 信息字典)
        """
        # 先更新步数
        self.current_step += 1
        
        # 立即检查是否应该结束 - 避免在无效状态下执行后续操作
        if self._is_done():
            # 如果已经结束，返回最后的观察状态和信息
            try:
                final_observation = self._get_observation()
            except (IndexError, RuntimeError, KeyError) as e:
                # 如果无法获取观察，使用上一次的观察或创建空观察
                logger.warning(f"无法获取最终观察状态: {e}")
                final_observation = {
                    'features': np.zeros((self.config.lookback_window, self.n_stocks, self.n_features_per_stock), dtype=np.float32),
                    'positions': self.current_positions.astype(np.float32),
                    'market_state': np.zeros(self.n_market_features, dtype=np.float32)
                }
            
            # 构建最终信息字典
            final_info = {
                'portfolio_return': 0.0,
                'transaction_cost': 0.0,
                'positions': self.current_positions.copy(),
                'total_value': float(self.total_value),
                'cash': float(self.cash),
                'drawdown': float(self._calculate_current_drawdown()),
                'step': self.current_step,
                'termination_reason': 'episode_completed'
            }
            
            return final_observation, 0.0, True, final_info
        
        # 更新当前价格（使用新的时间步）
        self._update_current_prices()
        
        # 标准化动作
        target_weights = self._normalize_action(action)
        
        # 应用A股交易规则约束
        target_weights = self._apply_trading_constraints(target_weights)
        
        # 风险检查：在执行交易前进行风险评估
        risk_adjusted_weights = self._check_trading_risks(target_weights)
        
        # 计算交易成本
        transaction_cost = self._calculate_transaction_cost(
            self.current_positions, risk_adjusted_weights
        )
        
        # 执行交易
        self._execute_trades(risk_adjusted_weights)
        
        # 获取当期收益（基于价格变化和持仓）
        portfolio_return = self._calculate_portfolio_return()
        
        # 更新投资组合价值
        self._update_portfolio_value(portfolio_return, transaction_cost)
        
        # 计算奖励
        reward = self._calculate_reward(portfolio_return, transaction_cost, target_weights)
        
        # 获取下一个观察
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
        if self.current_prices is None:
            logger.warning("当前价格为None，无法计算收益率")
            return 0.0
            
        if self.previous_prices is None:
            logger.debug("前期价格为None，第一步收益率为0")
            return 0.0
        
        # 计算各股票收益率
        stock_returns = np.zeros(self.n_stocks)
        for i in range(self.n_stocks):
            if self.previous_prices[i] > 1e-8:  # 避免除零
                return_val = (self.current_prices[i] - self.previous_prices[i]) / self.previous_prices[i]
                # 确保收益率不是NaN或无穷值
                if np.isnan(return_val) or np.isinf(return_val):
                    logger.warning(f"股票{i}收益率异常: current={self.current_prices[i]}, previous={self.previous_prices[i]}")
                    stock_returns[i] = 0.0
                else:
                    stock_returns[i] = return_val
            else:
                logger.warning(f"股票{i}前期价格过小: {self.previous_prices[i]}")
                stock_returns[i] = 0.0
        
        # 计算投资组合加权收益率
        portfolio_return = np.dot(self.current_positions, stock_returns)
        
        # 确保投资组合收益率不是NaN或无穷值
        if np.isnan(portfolio_return) or np.isinf(portfolio_return):
            logger.warning(f"投资组合收益率异常: {portfolio_return}")
            portfolio_return = 0.0
        
        # 调试日志：记录收益计算详情（降低频率并设为debug级别）
        if self.current_step % 200 == 0:
            logger.debug(f"Step {self.current_step} 收益计算详情:")
            logger.debug(f"  当前价格: {self.current_prices}")
            logger.debug(f"  前期价格: {self.previous_prices}")
            logger.debug(f"  个股收益率: {stock_returns}")
            logger.debug(f"  持仓权重: {self.current_positions}")
            logger.debug(f"  投资组合收益率: {portfolio_return}")
        
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
        """
        计算增强的Alpha奖励函数（超额收益 + 早期学习引导）
        
        核心理念：
        1. 主要奖励：超额收益（跑赢基准）
        2. 辅助奖励：鼓励模型探索不同权重分配
        3. 避免过度惩罚，给模型更多学习机会
        """
        # 计算市场基准收益率（等权重组合作为简化基准）
        market_return = self._calculate_market_benchmark_return()
        
        # 计算超额收益（Alpha）
        excess_return = portfolio_return - market_return
        
        # 计算交易成本比率
        if self.total_value > 1e-8:
            transaction_cost_ratio = transaction_cost / self.total_value
        else:
            transaction_cost_ratio = 0.0
        
        # Alpha奖励：超额收益减去交易成本，显著放大以鼓励探索
        alpha_reward = (excess_return - transaction_cost_ratio) * 1000
        
        # 确保Alpha奖励不是NaN或无穷值
        if np.isnan(alpha_reward) or np.isinf(alpha_reward):
            alpha_reward = 0.0
        
        # 早期学习引导奖励：鼓励权重差异化
        exploration_bonus = 0.0
        weight_variance = np.var(weights)  # 权重方差，衡量差异化程度
        if weight_variance > 0.01:  # 当权重有一定差异时给予奖励
            exploration_bonus = min(weight_variance * 20.0, 2.0)  # 最多奖励2.0
        
        # 增量学习奖励：基于个股表现差异
        if self.current_prices is not None and self.previous_prices is not None:
            # 计算各股票单独收益率
            individual_returns = []
            for i in range(self.n_stocks):
                if self.previous_prices[i] > 1e-8:
                    ret = (self.current_prices[i] - self.previous_prices[i]) / self.previous_prices[i]
                    if not (np.isnan(ret) or np.isinf(ret)):
                        individual_returns.append(ret)
            
            # 如果股票表现有差异，奖励选择较好股票的行为
            if len(individual_returns) >= 2:
                returns_array = np.array(individual_returns[:self.n_stocks])
                if np.std(returns_array) > 1e-6:  # 有表现差异
                    # 计算权重与收益的相关性（简化）
                    best_stock_idx = np.argmax(returns_array)
                    worst_stock_idx = np.argmin(returns_array)
                    
                    # 奖励更多配置在表现好的股票上
                    selection_bonus = (weights[best_stock_idx] - weights[worst_stock_idx]) * 10.0
                    # 限制奖励范围
                    selection_bonus = np.clip(selection_bonus, -1.0, 3.0)
                    exploration_bonus += selection_bonus
        
        # 风险控制器风险惩罚
        risk_penalty = self._calculate_risk_penalty(weights)
        
        # 轻微的风险控制（避免过度集中）
        concentration_penalty = 0.0
        if np.max(weights) > 0.9:  # 提高阈值，单股票权重超过90%时才惩罚
            concentration_penalty = (np.max(weights) - 0.9) * 10.0
        
        # 轻微的回撤控制
        drawdown_penalty = 0.0
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown > 0.5:  # 提高阈值，回撤超过50%时才惩罚
            drawdown_penalty = (current_drawdown - 0.5) * 5.0
        
        # 最终奖励：Alpha + 探索奖励 - 所有风险惩罚
        reward = alpha_reward + exploration_bonus - concentration_penalty - drawdown_penalty - risk_penalty
        
        # 确保最终奖励不是NaN或无穷值
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        # 调试日志（降低频率并设为debug级别）
        if self.current_step % 200 == 0:  # 降低日志频率
            logger.debug(f"Step {self.current_step}: portfolio_return={portfolio_return:.6f}, "
                        f"market_return={market_return:.6f}, excess_return={excess_return:.6f}, "
                        f"transaction_cost_ratio={transaction_cost_ratio:.6f}")
            logger.debug(f"Alpha_reward={alpha_reward:.4f}, exploration_bonus={exploration_bonus:.4f}, "
                       f"concentration_penalty={concentration_penalty:.4f}, "
                       f"drawdown_penalty={drawdown_penalty:.4f}, final_reward={reward:.4f}")
            logger.debug(f"Weights: {[f'{w:.3f}' for w in weights]}, "
                       f"Weight_variance: {np.var(weights):.4f}")
        
        # 记录Alpha收益历史，用于后续分析
        if not hasattr(self, 'alpha_history'):
            self.alpha_history = []
        self.alpha_history.append(excess_return)
        
        return float(reward)
    
    def _calculate_market_benchmark_return(self) -> float:
        """
        计算市场基准收益率
        
        使用真实的市场基准指数（如沪深300）作为基准
        这样智能体需要真正跑赢市场才能获得正奖励
        """
        if not hasattr(self, 'benchmark_data') or self.benchmark_data is None or self.benchmark_data.empty:
            # 如果没有基准数据，抛出异常
            raise RuntimeError("基准数据不可用，无法计算市场基准收益率")
        
        if self.current_step <= 1:
            return 0.0  # 第一步没有前一期数据
        
        # 计算日期索引位置
        current_date_idx = self.start_idx + self.current_step - 1
        previous_date_idx = self.start_idx + self.current_step - 2
        
        # 严格的边界检查 - 如果索引无效，说明环境状态有问题，应该抛出异常
        if hasattr(self, 'dates') and self.dates is not None:
            if (current_date_idx >= len(self.dates) or 
                previous_date_idx < 0 or 
                current_date_idx < 0):
                raise RuntimeError(f"无法计算基准收益率：日期索引超出边界。"
                                 f"current_idx={current_date_idx}, previous_idx={previous_date_idx}, "
                                 f"dates_len={len(self.dates)}, current_step={self.current_step}, "
                                 f"start_idx={self.start_idx}。这表明环境边界检查失效。")
            
            # 获取对应的日期
            current_date = self.dates[current_date_idx]
            previous_date = self.dates[previous_date_idx]
            
            # 在基准数据中找到对应日期的索引
            try:
                current_data_idx = self.benchmark_data.index.get_loc(current_date)
                previous_data_idx = self.benchmark_data.index.get_loc(previous_date)
            except KeyError as e:
                raise RuntimeError(f"基准数据中找不到日期：current={current_date}, previous={previous_date}。"
                                 f"基准数据日期范围：{self.benchmark_data.index.min()} 到 {self.benchmark_data.index.max()}。"
                                 f"原始错误：{e}")
        else:
            # 如果没有dates属性，使用原始索引计算
            current_data_idx = current_date_idx
            previous_data_idx = previous_date_idx
        
        # 基准数据边界检查
        if (current_data_idx >= len(self.benchmark_data) or 
            previous_data_idx < 0 or 
            previous_data_idx >= len(self.benchmark_data)):
            raise RuntimeError(f"基准数据索引超出边界：current_idx={current_data_idx}, "
                             f"previous_idx={previous_data_idx}, benchmark_data_len={len(self.benchmark_data)}。"
                             f"这表明基准数据与主数据不匹配。")
        
        # 处理多层索引的情况
        if isinstance(self.benchmark_data.index, pd.MultiIndex):
            # 如果是多层索引，需要找到对应的数据
            try:
                # 获取当前和前一期的基准价格
                current_benchmark_price = self.benchmark_data.iloc[current_data_idx]['close']
                previous_benchmark_price = self.benchmark_data.iloc[previous_data_idx]['close']
            except (IndexError, KeyError) as e:
                logger.error(f"无法获取基准数据: {e}")
                raise RuntimeError(f"基准数据访问失败: {e}")
        else:
            # 单层索引的情况
            if 'close' not in self.benchmark_data.columns:
                logger.warning(f"基准数据缺少'close'列，可用列: {list(self.benchmark_data.columns)}")
                return 0.0001
            
            try:
                current_benchmark_price = self.benchmark_data.iloc[current_data_idx]['close']
                previous_benchmark_price = self.benchmark_data.iloc[previous_data_idx]['close']
            except IndexError as e:
                logger.error(f"基准数据索引错误: {e}")
                raise RuntimeError(f"基准数据索引访问失败: {e}")
        
        # 检查价格有效性
        if pd.isna(current_benchmark_price) or pd.isna(previous_benchmark_price):
            raise RuntimeError(f"基准价格包含NaN: current={current_benchmark_price}, previous={previous_benchmark_price}")
            
        if previous_benchmark_price <= 1e-8:
            raise RuntimeError(f"基准数据前期价格无效: {previous_benchmark_price}")
        
        # 计算基准收益率
        market_return = (current_benchmark_price - previous_benchmark_price) / previous_benchmark_price
        
        # 确保市场收益率不是NaN或无穷值
        if np.isnan(market_return) or np.isinf(market_return):
            raise RuntimeError(f"计算出的市场收益率无效: {market_return}")
            
        return market_return
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        if self.max_portfolio_value == 0:
            return 0.0
        return (self.max_portfolio_value - self.total_value) / self.max_portfolio_value
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前观察状态"""
        # 获取历史特征数据
        if self.feature_data is not None:
            # 计算特征数据的索引范围，确保至少返回一些数据
            end_idx = self.start_idx + self.current_step + 1  # +1 确保至少有一行数据
            start_idx = max(0, end_idx - self.config.lookback_window)
            
            if end_idx > len(self.feature_data):
                # 如果超出数据范围，使用最后可用的数据
                features = self.feature_data[-self.config.lookback_window:].values
            else:
                features = self.feature_data[start_idx:end_idx].values
            
            # 确保特征数据有正确的维度
            if len(features) < self.config.lookback_window:
                # 用第一个观察值填充不足的部分
                padding = np.repeat(features[0:1], 
                                  self.config.lookback_window - len(features), 
                                  axis=0)
                features = np.concatenate([padding, features], axis=0)
            
            # 从MultiIndex DataFrame中为每只股票提取特征
            # feature_data的结构是 (instrument, datetime) -> features
            # 需要为每只股票、每个时间步提取特征
            stock_features_list = []
            
            # 获取当前时间窗口的日期
            end_idx = self.start_idx + self.current_step + 1
            start_idx = max(0, end_idx - self.config.lookback_window)
            
            if end_idx > len(self.dates):
                current_dates = self.dates[-self.config.lookback_window:]
            else:
                current_dates = self.dates[start_idx:end_idx]
            
            for symbol in self.config.stock_pool:
                stock_features = []
                for date in current_dates:
                    try:
                        # 从MultiIndex DataFrame中提取特定股票和日期的特征
                        if isinstance(self.feature_data.index, pd.MultiIndex):
                            stock_date_features = self.feature_data.loc[(symbol, date)].values
                        else:
                            # 如果不是MultiIndex，尝试按日期索引
                            stock_date_features = self.feature_data.loc[date].values
                        stock_features.append(stock_date_features)
                    except KeyError:
                        # 如果找不到数据，使用零填充
                        stock_features.append(np.zeros(self.n_features_per_stock))
                
                # 确保有正确的时间步数
                while len(stock_features) < self.config.lookback_window:
                    stock_features.insert(0, stock_features[0] if stock_features else np.zeros(self.n_features_per_stock))
                
                stock_features_list.append(np.array(stock_features))
            
            # 堆叠为 (lookback_window, n_stocks, n_features_per_stock)
            features = np.stack(stock_features_list, axis=1)
        else:
            # 如果没有特征数据，抛出异常
            raise ValueError("没有可用的特征数据，无法生成观察")
        
        # 市场状态特征（简化实现）
        # 安全地计算总收益率，避免除零
        total_return = (self.total_value / self.config.initial_cash - 1) if self.config.initial_cash > 0 else 0.0
        
        # 安全地计算波动率，避免NaN
        volatility = 0.0
        if len(self.returns_history) >= 30:
            recent_returns = np.array(self.returns_history[-30:])
            if not np.any(np.isnan(recent_returns)) and len(recent_returns) > 1:
                volatility = np.std(recent_returns)
                if np.isnan(volatility) or np.isinf(volatility):
                    volatility = 0.0
        
        # 安全地计算近期平均收益，避免NaN
        recent_mean_return = 0.0
        if len(self.returns_history) >= 10:
            recent_returns = np.array(self.returns_history[-10:])
            if not np.any(np.isnan(recent_returns)):
                recent_mean_return = np.mean(recent_returns)
                if np.isnan(recent_mean_return) or np.isinf(recent_mean_return):
                    recent_mean_return = 0.0
        
        # 安全地计算现金比例，避免除零
        cash_ratio = 0.0
        if self.total_value > 1e-8:
            cash_ratio = self.cash / self.total_value
            if np.isnan(cash_ratio) or np.isinf(cash_ratio):
                cash_ratio = 0.0
        
        market_state = np.array([
            total_return,                                   # 总收益率
            self._calculate_current_drawdown(),             # 当前回撤
            np.sum(self.current_positions ** 2),          # 持仓集中度
            np.sum(self.current_positions > 1e-6),        # 活跃持仓数
            self.current_step / self.max_steps,           # 时间进度
            volatility,                                    # 波动率
            recent_mean_return,                           # 近期收益
            len(self.trade_history) / max(self.current_step, 1),  # 交易频率
            cash_ratio,                                   # 现金比例
            1.0 if self.current_step % 5 == 0 else 0.0   # 周期性特征
        ], dtype=np.float32)
        
        # 确保所有市场状态特征都是有限值
        market_state = np.nan_to_num(market_state, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 确保特征数据也没有NaN或无穷值
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return {
            'features': features.astype(np.float32),
            'positions': self.current_positions.astype(np.float32),
            'market_state': market_state
        }
    
    def _update_current_prices(self):
        """更新当前价格"""
        if self.price_data is None:
            raise RuntimeError("没有可用的价格数据，无法更新当前价格")
        
        # 保存前一期价格
        self.previous_prices = self.current_prices.copy() if self.current_prices is not None else None
        
        # 获取当前价格
        current_date_idx = self.start_idx + self.current_step
        
        # 找到正确的日期层级名称
        datetime_level = None
        for level_name in self.price_data.index.names:
            if level_name and ('time' in level_name.lower() or 'date' in level_name.lower()):
                datetime_level = level_name
                break
        
        if not datetime_level:
            raise RuntimeError(f"价格数据中未找到日期层级，可用层级: {self.price_data.index.names}")
        
        unique_dates = self.price_data.index.get_level_values(datetime_level).unique()
        
        if current_date_idx >= len(unique_dates):
            # 如果超出数据范围，使用最后一个可用日期的数据
            current_date_idx = len(unique_dates) - 1
            
        date = unique_dates[current_date_idx]
        
        # 检查日期是否存在于价格数据中
        if date not in self.price_data.index.get_level_values(datetime_level):
            raise RuntimeError(f"价格数据中不存在日期: {date}")
        
        current_day_data = self.price_data.xs(date, level=datetime_level)
        
        # 初始化当前价格数组
        new_current_prices = np.zeros(self.n_stocks)
        
        for i, symbol in enumerate(self.config.stock_pool):
            if symbol in current_day_data.index:
                if 'close' not in current_day_data.columns:
                    raise RuntimeError(f"价格数据缺少'close'列，可用列: {list(current_day_data.columns)}")
                
                price = current_day_data.loc[symbol, 'close']
                # 数据已经在加载时清洗过，这里应该不会有无效价格
                if pd.isna(price) or price <= 0:
                    raise RuntimeError(f"股票{symbol}在{date}的价格无效: {price}，数据清洗失败")
                
                new_current_prices[i] = float(price)
            else:
                raise RuntimeError(f"股票{symbol}在{date}没有价格数据")
        
        self.current_prices = new_current_prices
        
        # 调试日志：记录价格更新详情
        if self.current_step % 50 == 0:
            logger.debug(f"Step {self.current_step}: 价格更新 - 日期: {date}")
            logger.debug(f"当前价格: {self.current_prices}")
            if self.previous_prices is not None:
                logger.debug(f"前期价格: {self.previous_prices}")
                price_changes = (self.current_prices - self.previous_prices) / self.previous_prices
                logger.debug(f"价格变化率: {price_changes}")
    
    def _is_done(self) -> bool:
        """判断回合是否结束"""
        # 日期索引边界检查（优先级最高）
        if (hasattr(self, 'dates') and self.dates is not None and 
            self.start_idx + self.current_step >= len(self.dates)):
            logger.debug(f"日期索引用尽，结束episode: step={self.current_step}, "
                        f"start_idx={self.start_idx}, dates_len={len(self.dates)}")
            return True
        
        # 数据用尽检查
        if (self.feature_data is not None and 
            self.start_idx + self.current_step >= len(self.feature_data)):
            logger.debug(f"数据用尽，结束episode: step={self.current_step}, data_len={len(self.feature_data)}")
            return True
        
        # 基本结束条件：达到实际数据的最大步数
        if self.current_step >= self._max_steps:
            logger.debug(f"达到最大步数，结束episode: step={self.current_step}, max_steps={self._max_steps}")
            return True
        
        # 风险控制：总价值过低
        if self.total_value < self.config.initial_cash * 0.5:
            logger.warning(f"总价值过低，强制结束: {self.total_value}")
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
    
    def _check_trading_risks(self, target_weights: np.ndarray) -> np.ndarray:
        """
        执行交易前的风险检查
        
        Args:
            target_weights: 目标权重
            
        Returns:
            风险调整后的权重
        """
        if self.risk_controller is None:
            return target_weights
            
        try:
            # 构建当前投资组合状态
            current_portfolio = self._build_portfolio_for_risk_check()
            
            # 构建预期交易决策列表
            proposed_trade_decisions = self._build_trade_decisions_for_risk_check(target_weights)
            
            # 执行风险检查
            risk_violations = []
            for trade_decision in proposed_trade_decisions:
                trade_risk_result = self.risk_controller.check_trade_risk(trade_decision, current_portfolio)
                if not trade_risk_result.get('approved', True):
                    risk_violations.extend(trade_risk_result.get('violations', []))
            
            # 如果有严重风险违规，调整目标权重
            if any(violation.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL] for violation in risk_violations):
                logger.warning(f"检测到严重风险违规，调整交易计划: {[v.message for v in risk_violations]}")
                return self._adjust_weights_for_risk_violations(target_weights, risk_violations)
            else:
                # 记录中低级风险警告
                if risk_violations:
                    logger.info(f"检测到风险警告: {[v.message for v in risk_violations]}")
                return target_weights
                
        except (AttributeError, ValueError, KeyError) as e:
            logger.error(f"风险检查失败: {e}")
            # 风险检查失败时，抛出异常，不掩盖错误
            raise RuntimeError(f"风险检查模块出现错误，无法继续交易: {e}") from e
    
    def _build_portfolio_for_risk_check(self) -> Portfolio:
        """构建用于风险检查的投资组合对象"""
        positions = []
        current_time = datetime.now()
        
        for i, symbol in enumerate(self.config.stock_pool):
            if self.current_positions[i] > 0:
                # 获取当前价格
                current_price = float(self.current_prices[i]) if self.current_prices is not None else 10.0
                quantity = int(self.current_positions[i] * self.total_value / current_price)
                
                if quantity > 0:
                    position = Position(
                        symbol=symbol,
                        quantity=quantity,
                        current_price=current_price,
                        sector="Unknown",  # 简化处理，实际应用中需要行业分类
                        timestamp=current_time
                    )
                    positions.append(position)
        
        return Portfolio(
            positions=positions,
            cash=float(self.cash),
            total_value=float(self.total_value),
            timestamp=current_time
        )
    
    def _build_trades_for_risk_check(self, target_weights: np.ndarray) -> List[Trade]:
        """构建用于风险检查的交易列表"""
        trades = []
        current_time = datetime.now()
        
        for i, symbol in enumerate(self.config.stock_pool):
            weight_diff = target_weights[i] - self.current_positions[i]
            
            if abs(weight_diff) > 0.001:  # 忽略微小变化
                current_price = float(self.current_prices[i]) if self.current_prices is not None else 10.0
                trade_value = abs(weight_diff) * self.total_value
                quantity = trade_value / current_price
                action = "BUY" if weight_diff > 0 else "SELL"
                
                trade = Trade(
                    symbol=symbol,
                    quantity=quantity,
                    price=current_price,
                    timestamp=current_time,
                    action=action,
                    sector="Unknown"  # 简化处理
                )
                trades.append(trade)
        
        return trades
    
    def _build_trade_decisions_for_risk_check(self, target_weights: np.ndarray) -> List[TradeDecision]:
        """构建用于风险检查的交易决策列表"""
        trade_decisions = []
        current_time = datetime.now()
        
        for i, symbol in enumerate(self.config.stock_pool):
            weight_diff = target_weights[i] - self.current_positions[i]
            
            if abs(weight_diff) > 0.001:  # 忽略微小变化
                current_price = float(self.current_prices[i]) if self.current_prices is not None else 10.0
                trade_value = abs(weight_diff) * self.total_value
                quantity = int(trade_value / current_price)
                action = "BUY" if weight_diff > 0 else "SELL"
                
                trade_decision = TradeDecision(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    target_price=current_price,
                    sector="Unknown",  # 简化处理
                    timestamp=current_time,
                    confidence=0.5,  # 默认置信度
                    metadata={"weight_change": weight_diff}
                )
                trade_decisions.append(trade_decision)
        
        return trade_decisions
    
    def _adjust_weights_for_risk_violations(self, target_weights: np.ndarray, violations) -> np.ndarray:
        """根据风险违规调整权重"""
        adjusted_weights = target_weights.copy()
        
        for violation in violations:
            if violation.violation_type == RiskViolationType.POSITION_CONCENTRATION:
                # 对于持仓集中度违规，降低最大持仓权重
                max_idx = np.argmax(adjusted_weights)
                adjusted_weights[max_idx] = min(adjusted_weights[max_idx], self.config.max_position_size)
                
            elif violation.violation_type == RiskViolationType.STOP_LOSS:
                # 对于止损违规，减少持仓
                for i in range(len(adjusted_weights)):
                    if adjusted_weights[i] > self.current_positions[i]:
                        adjusted_weights[i] = self.current_positions[i] * 0.8  # 减少20%
        
        # 重新标准化权重
        weight_sum = np.sum(adjusted_weights)
        if weight_sum > 1.0:
            adjusted_weights = adjusted_weights / weight_sum
        
        return adjusted_weights
    
    def _calculate_risk_penalty(self, weights: np.ndarray) -> float:
        """
        计算基于风险控制器的风险惩罚项
        
        Args:
            weights: 目标权重
            
        Returns:
            风险惩罚值（越高表示风险越大）
        """
        if self.risk_controller is None:
            return 0.0
            
        try:
            # 构建投资组合状态
            current_portfolio = self._build_portfolio_for_risk_check()
            
            # 构建预期交易决策列表
            proposed_trade_decisions = self._build_trade_decisions_for_risk_check(weights)
            
            # 执行投资组合风险评估
            portfolio_risk_result = self.risk_controller.assess_portfolio_risk(current_portfolio)
            
            # 执行交易风险评估
            trade_violations = []
            for trade_decision in proposed_trade_decisions:
                trade_risk_result = self.risk_controller.check_trade_risk(trade_decision, current_portfolio)
                if not trade_risk_result.get('approved', True):
                    trade_violations.extend(trade_risk_result.get('violations', []))
            
            # 计算风险惩罚
            total_penalty = 0.0
            
            # 根据投资组合风险评分计算惩罚
            portfolio_risk_score = portfolio_risk_result.get('risk_score', 0.0)
            total_penalty += portfolio_risk_score * 0.5  # 投资组合基础风险惩罚
            
            # 根据交易违规级别计算惩罚
            for violation in trade_violations:
                if violation.severity == RiskLevel.CRITICAL:
                    total_penalty += 5.0  # 严重风险
                elif violation.severity == RiskLevel.HIGH:
                    total_penalty += 2.0  # 高风险
                elif violation.severity == RiskLevel.MEDIUM:
                    total_penalty += 0.5  # 中等风险
                elif violation.severity == RiskLevel.LOW:
                    total_penalty += 0.1  # 低风险
            
            # 限制惩罚范围，避免过度惩罚影响学习
            total_penalty = min(total_penalty, 10.0)
            
            return total_penalty
            
        except (AttributeError, ValueError, KeyError) as e:
            logger.error(f"风险惩罚计算失败: {e}")
            # 风险惩罚计算失败时，抛出异常，不掩盖错误
            raise RuntimeError(f"风险惩罚计算模块出现错误，无法继续评估风险: {e}") from e
    
    def close(self):
        """关闭环境"""
        pass
    
    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        return [seed]