"""
压力测试引擎
实现历史情景重现、蒙特卡洛模拟等压力测试方法，用于评估投资组合在极端市场条件下的风险暴露
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from ..data.data_models import MarketData, TradingState
from .drawdown_monitor import DrawdownMonitor, DrawdownMetrics

logger = logging.getLogger(__name__)


class StressTestType(Enum):
    """压力测试类型"""
    HISTORICAL_SCENARIO = "historical_scenario"      # 历史情景重现
    MONTE_CARLO = "monte_carlo"                      # 蒙特卡洛模拟
    PARAMETRIC_VAR = "parametric_var"                # 参数化VaR
    EXTREME_VALUE = "extreme_value"                  # 极值理论
    CORRELATION_BREAKDOWN = "correlation_breakdown"   # 相关性崩溃
    LIQUIDITY_CRISIS = "liquidity_crisis"            # 流动性危机


class MarketScenario(Enum):
    """市场情景类型"""
    MARKET_CRASH = "market_crash"                    # 市场崩盘
    SECTOR_ROTATION = "sector_rotation"              # 板块轮动
    VOLATILITY_SPIKE = "volatility_spike"            # 波动率飙升
    LIQUIDITY_DROUGHT = "liquidity_drought"          # 流动性枯竭
    INTEREST_RATE_SHOCK = "interest_rate_shock"      # 利率冲击
    CURRENCY_CRISIS = "currency_crisis"              # 货币危机
    BLACK_SWAN = "black_swan"                        # 黑天鹅事件


@dataclass
class StressTestConfig:
    """压力测试配置"""
    # 基础配置
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99, 0.999])
    time_horizons: List[int] = field(default_factory=lambda: [1, 5, 10, 22])  # 天数
    num_simulations: int = 10000                     # 蒙特卡洛模拟次数
    random_seed: int = 42                            # 随机种子
    
    # 历史情景配置
    historical_lookback_years: int = 10              # 历史数据回望年数
    scenario_window_days: int = 22                   # 情景窗口天数
    min_scenario_severity: float = -0.05             # 最小情景严重程度
    
    # 蒙特卡洛配置
    correlation_decay_factor: float = 0.94           # 相关性衰减因子
    volatility_scaling_factor: float = 1.2           # 波动率缩放因子
    fat_tail_adjustment: bool = True                 # 厚尾调整
    
    # 极端情景配置
    extreme_percentiles: List[float] = field(default_factory=lambda: [1, 5, 10])
    correlation_shock_factor: float = 0.5            # 相关性冲击因子
    liquidity_impact_factor: float = 1.5             # 流动性影响因子
    
    # 并行计算配置
    max_workers: int = 4                             # 最大工作线程数
    chunk_size: int = 1000                           # 批处理大小


@dataclass
class StressTestResult:
    """压力测试结果"""
    test_type: StressTestType
    scenario: Optional[MarketScenario]
    timestamp: datetime
    
    # 损失分布
    portfolio_losses: np.ndarray                     # 投资组合损失分布
    var_estimates: Dict[float, float]                # VaR估计值
    cvar_estimates: Dict[float, float]               # CVaR估计值
    expected_shortfall: Dict[float, float]           # 期望损失
    
    # 风险指标
    max_loss: float                                  # 最大损失
    probability_of_loss: float                       # 损失概率
    tail_expectation: float                          # 尾部期望
    
    # 归因分析
    asset_contributions: Dict[str, float]            # 资产贡献
    factor_contributions: Dict[str, float]           # 因子贡献
    
    # 统计信息
    statistics: Dict[str, float]                     # 统计指标
    
    def __post_init__(self):
        """结果验证"""
        if len(self.portfolio_losses) == 0:
            raise ValueError("投资组合损失分布不能为空")
        
        if not all(0 <= conf <= 1 for conf in self.var_estimates.keys()):
            raise ValueError("置信水平必须在0到1之间")


@dataclass
class ScenarioDefinition:
    """情景定义"""
    name: str
    description: str
    scenario_type: MarketScenario
    
    # 市场参数
    market_shock: float                              # 市场冲击幅度
    volatility_multiplier: float                     # 波动率倍数
    correlation_adjustment: float                    # 相关性调整
    
    # 持续时间
    duration_days: int                               # 持续天数
    recovery_days: int                               # 恢复天数
    
    # 概率权重
    probability: float                               # 发生概率
    
    def __post_init__(self):
        """验证情景定义"""
        if not (0 <= self.probability <= 1):
            raise ValueError("概率必须在0到1之间")
        
        if self.duration_days <= 0:
            raise ValueError("持续天数必须大于0")


@dataclass
class ExtremeScenarioParameters:
    """极端情景参数"""
    scenario_type: MarketScenario
    shock_magnitude: float                       # 冲击幅度
    shock_duration: int                          # 冲击持续时间（天）
    recovery_time: int                           # 恢复时间（天）
    volatility_spike: float                      # 波动率飙升倍数
    correlation_increase: float                  # 相关性增加幅度
    liquidity_impact: float                      # 流动性影响
    contagion_probability: float                 # 传染概率
    fat_tail_parameter: float                    # 厚尾参数


class ExtremeScenarioSimulator:
    """
    极端情景模拟器
    
    专门用于模拟各种极端市场情景，包括：
    1. 市场崩盘
    2. 流动性危机
    3. 系统性风险传染
    4. 黑天鹅事件
    """
    
    def __init__(self, config: StressTestConfig):
        """
        初始化极端情景模拟器
        
        Args:
            config: 压力测试配置
        """
        self.config = config
        self.scenario_parameters = self._initialize_scenario_parameters()
        
        # 设置随机种子
        np.random.seed(config.random_seed)
        
        logger.info("极端情景模拟器初始化完成")
    
    def simulate_market_crash(self, 
                            asset_returns: pd.DataFrame,
                            portfolio_weights: np.ndarray,
                            crash_magnitude: float = -0.3,
                            crash_duration: int = 5) -> np.ndarray:
        """
        模拟市场崩盘情景
        
        Args:
            asset_returns: 历史资产收益率
            portfolio_weights: 投资组合权重
            crash_magnitude: 崩盘幅度
            crash_duration: 崩盘持续天数
            
        Returns:
            模拟的投资组合损失
        """
        logger.info(f"模拟市场崩盘情景: 幅度={crash_magnitude}, 持续={crash_duration}天")
        
        # 计算正常时期的统计特征
        mean_returns = asset_returns.mean()
        cov_matrix = asset_returns.cov()
        
        # 崩盘期间的参数调整
        crash_mean = mean_returns + crash_magnitude / crash_duration
        crash_cov = cov_matrix * (3.0 ** 2)  # 波动率增加3倍
        
        # 增加相关性（危机时期资产相关性上升）
        crash_corr = cov_matrix.corr()
        crash_corr = crash_corr * 0.3 + 0.7 * np.ones_like(crash_corr)
        crash_corr = crash_corr.values  # 转换为numpy数组
        np.fill_diagonal(crash_corr, 1.0)
        
        # 重构协方差矩阵
        crash_std = np.sqrt(np.diag(crash_cov))
        crash_cov = np.outer(crash_std, crash_std) * crash_corr
        
        # 模拟崩盘期间的收益率
        crash_returns = np.random.multivariate_normal(
            crash_mean, crash_cov, self.config.num_simulations
        )
        
        # 计算多日累积损失
        cumulative_losses = []
        for _ in range(self.config.num_simulations):
            daily_returns = np.random.multivariate_normal(crash_mean, crash_cov, crash_duration)
            cumulative_return = np.prod(1 + daily_returns, axis=0) - 1
            portfolio_loss = -(cumulative_return @ portfolio_weights)
            cumulative_losses.append(portfolio_loss)
        
        return np.array(cumulative_losses)
    
    def simulate_liquidity_crisis(self,
                                asset_returns: pd.DataFrame,
                                portfolio_weights: np.ndarray,
                                liquidity_shock: float = 0.05,
                                bid_ask_widening: float = 3.0) -> np.ndarray:
        """
        模拟流动性危机情景
        
        Args:
            asset_returns: 历史资产收益率
            portfolio_weights: 投资组合权重
            liquidity_shock: 流动性冲击（额外成本）
            bid_ask_widening: 买卖价差扩大倍数
            
        Returns:
            模拟的投资组合损失
        """
        logger.info(f"模拟流动性危机情景: 冲击={liquidity_shock}, 价差扩大={bid_ask_widening}倍")
        
        # 基础收益率模拟
        mean_returns = asset_returns.mean()
        cov_matrix = asset_returns.cov() * (1.5 ** 2)  # 波动率增加
        
        # 流动性成本建模
        # 大权重资产面临更高的流动性成本
        liquidity_costs = portfolio_weights * liquidity_shock * bid_ask_widening
        
        # 价格冲击建模（非线性）
        price_impact = np.sqrt(portfolio_weights) * liquidity_shock * 0.5
        
        # 模拟收益率
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, self.config.num_simulations
        )
        
        # 应用流动性成本和价格冲击
        adjusted_returns = simulated_returns - liquidity_costs - price_impact
        portfolio_losses = -(adjusted_returns @ portfolio_weights)
        
        return portfolio_losses
    
    def simulate_contagion_effect(self,
                                asset_returns: pd.DataFrame,
                                portfolio_weights: np.ndarray,
                                initial_shock_assets: List[int],
                                contagion_probability: float = 0.7,
                                contagion_magnitude: float = 0.5) -> np.ndarray:
        """
        模拟传染效应情景
        
        Args:
            asset_returns: 历史资产收益率
            portfolio_weights: 投资组合权重
            initial_shock_assets: 初始受冲击的资产索引
            contagion_probability: 传染概率
            contagion_magnitude: 传染幅度
            
        Returns:
            模拟的投资组合损失
        """
        logger.info(f"模拟传染效应情景: 初始冲击资产={len(initial_shock_assets)}, 传染概率={contagion_probability}")
        
        n_assets = len(asset_returns.columns)
        n_simulations = self.config.num_simulations
        
        portfolio_losses = []
        
        for _ in range(n_simulations):
            # 初始化资产冲击状态
            shocked_assets = set(initial_shock_assets)
            asset_shocks = np.zeros(n_assets)
            
            # 对初始冲击资产施加冲击
            for asset_idx in initial_shock_assets:
                asset_shocks[asset_idx] = -0.2  # 20%的负冲击
            
            # 模拟传染过程
            for round_num in range(3):  # 最多3轮传染
                new_shocked_assets = set()
                
                for shocked_asset in shocked_assets:
                    # 计算与其他资产的相关性
                    correlations = asset_returns.corr().iloc[shocked_asset]
                    
                    for asset_idx in range(n_assets):
                        if asset_idx not in shocked_assets and asset_idx not in new_shocked_assets:
                            # 传染概率与相关性相关
                            transmission_prob = contagion_probability * abs(correlations.iloc[asset_idx])
                            
                            if np.random.random() < transmission_prob:
                                new_shocked_assets.add(asset_idx)
                                # 传染冲击随轮次衰减
                                asset_shocks[asset_idx] = -0.2 * contagion_magnitude * (0.7 ** round_num)
                
                shocked_assets.update(new_shocked_assets)
                
                if not new_shocked_assets:  # 没有新的传染，停止
                    break
            
            # 计算投资组合损失
            portfolio_loss = -(asset_shocks @ portfolio_weights)
            portfolio_losses.append(portfolio_loss)
        
        return np.array(portfolio_losses)
    
    def simulate_black_swan_event(self,
                                asset_returns: pd.DataFrame,
                                portfolio_weights: np.ndarray,
                                event_probability: float = 0.001,
                                event_magnitude: float = -0.5) -> np.ndarray:
        """
        模拟黑天鹅事件
        
        Args:
            asset_returns: 历史资产收益率
            portfolio_weights: 投资组合权重
            event_probability: 事件发生概率
            event_magnitude: 事件冲击幅度
            
        Returns:
            模拟的投资组合损失
        """
        logger.info(f"模拟黑天鹅事件: 概率={event_probability}, 幅度={event_magnitude}")
        
        # 正常时期的收益率模拟
        mean_returns = asset_returns.mean()
        cov_matrix = asset_returns.cov()
        
        portfolio_losses = []
        
        for _ in range(self.config.num_simulations):
            # 判断是否发生黑天鹅事件
            if np.random.random() < event_probability:
                # 发生黑天鹅事件
                # 所有资产同时受到冲击，但幅度不同
                asset_shocks = np.random.uniform(
                    event_magnitude * 0.5, event_magnitude * 1.5, len(portfolio_weights)
                )
                portfolio_loss = -(asset_shocks @ portfolio_weights)
            else:
                # 正常情况
                normal_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
                portfolio_loss = -(normal_returns @ portfolio_weights)
            
            portfolio_losses.append(portfolio_loss)
        
        return np.array(portfolio_losses)
    
    def simulate_regime_shift(self,
                            asset_returns: pd.DataFrame,
                            portfolio_weights: np.ndarray,
                            shift_probability: float = 0.1,
                            new_regime_params: Optional[Dict] = None) -> np.ndarray:
        """
        模拟市场制度转换
        
        Args:
            asset_returns: 历史资产收益率
            portfolio_weights: 投资组合权重
            shift_probability: 制度转换概率
            new_regime_params: 新制度参数
            
        Returns:
            模拟的投资组合损失
        """
        logger.info(f"模拟市场制度转换: 转换概率={shift_probability}")
        
        # 默认新制度参数（熊市制度）
        if new_regime_params is None:
            new_regime_params = {
                'mean_adjustment': -0.002,  # 日均收益率下降
                'volatility_multiplier': 1.8,  # 波动率增加
                'correlation_increase': 0.3  # 相关性增加
            }
        
        # 正常制度参数
        normal_mean = asset_returns.mean()
        normal_cov = asset_returns.cov()
        
        # 新制度参数
        new_mean = normal_mean + new_regime_params['mean_adjustment']
        new_cov = normal_cov * (new_regime_params['volatility_multiplier'] ** 2)
        
        # 调整相关性
        new_corr = normal_cov.corr()
        new_corr = new_corr * (1 - new_regime_params['correlation_increase']) + \
                  new_regime_params['correlation_increase'] * np.ones_like(new_corr)
        new_corr = new_corr.values  # 转换为numpy数组
        np.fill_diagonal(new_corr, 1.0)
        
        # 重构协方差矩阵
        new_std = np.sqrt(np.diag(new_cov))
        new_cov = np.outer(new_std, new_std) * new_corr
        
        portfolio_losses = []
        current_regime = 'normal'  # 初始制度
        
        for _ in range(self.config.num_simulations):
            # 检查是否发生制度转换
            if np.random.random() < shift_probability:
                current_regime = 'new' if current_regime == 'normal' else 'normal'
            
            # 根据当前制度生成收益率
            if current_regime == 'normal':
                returns = np.random.multivariate_normal(normal_mean, normal_cov)
            else:
                returns = np.random.multivariate_normal(new_mean, new_cov)
            
            portfolio_loss = -(returns @ portfolio_weights)
            portfolio_losses.append(portfolio_loss)
        
        return np.array(portfolio_losses)
    
    def calibrate_scenario_parameters(self,
                                    asset_returns: pd.DataFrame,
                                    historical_events: Optional[List[Dict]] = None) -> Dict[MarketScenario, ExtremeScenarioParameters]:
        """
        校准情景参数
        
        Args:
            asset_returns: 历史资产收益率
            historical_events: 历史事件数据
            
        Returns:
            校准后的情景参数
        """
        logger.info("校准极端情景参数")
        
        calibrated_params = {}
        
        # 基于历史数据校准参数
        portfolio_returns = asset_returns.mean(axis=1)  # 等权重组合收益率
        
        # 识别极端事件
        extreme_losses = portfolio_returns[portfolio_returns < np.percentile(portfolio_returns, 1)]
        extreme_volatility = asset_returns.std().mean()
        
        # 市场崩盘参数校准
        crash_magnitude = extreme_losses.min() if len(extreme_losses) > 0 else -0.3
        crash_volatility = extreme_volatility * 3
        
        calibrated_params[MarketScenario.MARKET_CRASH] = ExtremeScenarioParameters(
            scenario_type=MarketScenario.MARKET_CRASH,
            shock_magnitude=crash_magnitude,
            shock_duration=5,
            recovery_time=60,
            volatility_spike=3.0,
            correlation_increase=0.7,
            liquidity_impact=0.05,
            contagion_probability=0.8,
            fat_tail_parameter=3.0
        )
        
        # 流动性危机参数校准
        calibrated_params[MarketScenario.LIQUIDITY_DROUGHT] = ExtremeScenarioParameters(
            scenario_type=MarketScenario.LIQUIDITY_DROUGHT,
            shock_magnitude=-0.15,
            shock_duration=10,
            recovery_time=30,
            volatility_spike=2.0,
            correlation_increase=0.5,
            liquidity_impact=0.08,
            contagion_probability=0.6,
            fat_tail_parameter=4.0
        )
        
        # 波动率飙升参数校准
        calibrated_params[MarketScenario.VOLATILITY_SPIKE] = ExtremeScenarioParameters(
            scenario_type=MarketScenario.VOLATILITY_SPIKE,
            shock_magnitude=-0.1,
            shock_duration=7,
            recovery_time=21,
            volatility_spike=2.5,
            correlation_increase=0.4,
            liquidity_impact=0.03,
            contagion_probability=0.5,
            fat_tail_parameter=5.0
        )
        
        return calibrated_params
    
    def estimate_scenario_probabilities(self,
                                      asset_returns: pd.DataFrame,
                                      lookback_years: int = 20) -> Dict[MarketScenario, float]:
        """
        估计情景发生概率
        
        Args:
            asset_returns: 历史资产收益率
            lookback_years: 回望年数
            
        Returns:
            各情景的发生概率
        """
        logger.info(f"估计情景发生概率，回望期={lookback_years}年")
        
        # 计算投资组合收益率（等权重）
        portfolio_returns = asset_returns.mean(axis=1)
        
        # 计算滚动统计指标
        rolling_vol = portfolio_returns.rolling(window=22).std()  # 月度波动率
        
        # 计算滚动相关性（简化版本）
        rolling_corr_values = []
        for i in range(22, len(asset_returns)):
            window_data = asset_returns.iloc[i-22:i]
            corr_matrix = window_data.corr()
            # 计算上三角矩阵的平均相关性（排除对角线）
            upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
            avg_corr = np.mean(upper_triangle)
            rolling_corr_values.append(avg_corr)
        
        rolling_corr = pd.Series(rolling_corr_values, index=asset_returns.index[22:])
        
        # 定义极端事件阈值
        crash_threshold = np.percentile(portfolio_returns, 1)  # 1%分位数
        vol_spike_threshold = np.percentile(rolling_vol, 95)  # 95%分位数
        
        # 统计历史事件频率
        crash_events = (portfolio_returns < crash_threshold).sum()
        vol_spike_events = (rolling_vol > vol_spike_threshold).sum()
        
        # 计算年化概率
        total_days = len(portfolio_returns)
        years = total_days / 252
        
        probabilities = {
            MarketScenario.MARKET_CRASH: crash_events / years / 365,  # 日概率
            MarketScenario.VOLATILITY_SPIKE: vol_spike_events / years / 365,
            MarketScenario.LIQUIDITY_DROUGHT: 0.02 / 365,  # 基于专家判断
            MarketScenario.BLACK_SWAN: 0.001 / 365,  # 极低概率事件
            MarketScenario.SECTOR_ROTATION: 0.1 / 365,  # 相对常见
            MarketScenario.CURRENCY_CRISIS: 0.005 / 365  # 低概率事件
        }
        
        return probabilities
    
    def _initialize_scenario_parameters(self) -> Dict[MarketScenario, ExtremeScenarioParameters]:
        """初始化默认情景参数"""
        return {
            MarketScenario.MARKET_CRASH: ExtremeScenarioParameters(
                scenario_type=MarketScenario.MARKET_CRASH,
                shock_magnitude=-0.3,
                shock_duration=5,
                recovery_time=60,
                volatility_spike=3.0,
                correlation_increase=0.7,
                liquidity_impact=0.05,
                contagion_probability=0.8,
                fat_tail_parameter=3.0
            ),
            MarketScenario.LIQUIDITY_DROUGHT: ExtremeScenarioParameters(
                scenario_type=MarketScenario.LIQUIDITY_DROUGHT,
                shock_magnitude=-0.15,
                shock_duration=10,
                recovery_time=30,
                volatility_spike=2.0,
                correlation_increase=0.5,
                liquidity_impact=0.08,
                contagion_probability=0.6,
                fat_tail_parameter=4.0
            ),
            MarketScenario.VOLATILITY_SPIKE: ExtremeScenarioParameters(
                scenario_type=MarketScenario.VOLATILITY_SPIKE,
                shock_magnitude=-0.1,
                shock_duration=7,
                recovery_time=21,
                volatility_spike=2.5,
                correlation_increase=0.4,
                liquidity_impact=0.03,
                contagion_probability=0.5,
                fat_tail_parameter=5.0
            ),
            MarketScenario.BLACK_SWAN: ExtremeScenarioParameters(
                scenario_type=MarketScenario.BLACK_SWAN,
                shock_magnitude=-0.5,
                shock_duration=1,
                recovery_time=90,
                volatility_spike=4.0,
                correlation_increase=0.8,
                liquidity_impact=0.15,
                contagion_probability=0.9,
                fat_tail_parameter=2.0
            ),
            MarketScenario.SECTOR_ROTATION: ExtremeScenarioParameters(
                scenario_type=MarketScenario.SECTOR_ROTATION,
                shock_magnitude=-0.05,
                shock_duration=14,
                recovery_time=45,
                volatility_spike=1.5,
                correlation_increase=0.2,
                liquidity_impact=0.02,
                contagion_probability=0.3,
                fat_tail_parameter=6.0
            ),
            MarketScenario.CURRENCY_CRISIS: ExtremeScenarioParameters(
                scenario_type=MarketScenario.CURRENCY_CRISIS,
                shock_magnitude=-0.2,
                shock_duration=21,
                recovery_time=120,
                volatility_spike=2.8,
                correlation_increase=0.6,
                liquidity_impact=0.06,
                contagion_probability=0.7,
                fat_tail_parameter=3.5
            )
        }


class StressTestEngine:
    """
    压力测试引擎
    
    实现多种压力测试方法，包括：
    1. 历史情景重现
    2. 蒙特卡洛模拟
    3. 参数化VaR
    4. 极值理论分析
    5. 相关性崩溃测试
    6. 流动性危机模拟
    7. 极端情景模拟
    """
    
    def __init__(self, config: StressTestConfig):
        """
        初始化压力测试引擎
        
        Args:
            config: 压力测试配置
        """
        self.config = config
        self.drawdown_monitor = DrawdownMonitor()
        
        # 设置随机种子
        np.random.seed(config.random_seed)
        
        # 预定义极端情景
        self.extreme_scenarios = self._define_extreme_scenarios()
        
        # 初始化极端情景模拟器
        self.scenario_simulator = ExtremeScenarioSimulator(config)
        
        logger.info(f"压力测试引擎初始化完成，配置: {config}")
    
    def run_stress_test(self, 
                       portfolio_weights: np.ndarray,
                       asset_returns: pd.DataFrame,
                       test_type: StressTestType,
                       scenario: Optional[MarketScenario] = None) -> StressTestResult:
        """
        执行压力测试
        
        Args:
            portfolio_weights: 投资组合权重
            asset_returns: 资产收益率历史数据
            test_type: 压力测试类型
            scenario: 市场情景（可选）
            
        Returns:
            压力测试结果
        """
        logger.info(f"开始执行压力测试: {test_type.value}, 情景: {scenario}")
        
        # 验证输入
        self._validate_inputs(portfolio_weights, asset_returns)
        
        # 根据测试类型执行相应的压力测试
        if test_type == StressTestType.HISTORICAL_SCENARIO:
            result = self._run_historical_scenario_test(portfolio_weights, asset_returns, scenario)
        elif test_type == StressTestType.MONTE_CARLO:
            result = self._run_monte_carlo_test(portfolio_weights, asset_returns, scenario)
        elif test_type == StressTestType.PARAMETRIC_VAR:
            result = self._run_parametric_var_test(portfolio_weights, asset_returns)
        elif test_type == StressTestType.EXTREME_VALUE:
            result = self._run_extreme_value_test(portfolio_weights, asset_returns)
        elif test_type == StressTestType.CORRELATION_BREAKDOWN:
            result = self._run_correlation_breakdown_test(portfolio_weights, asset_returns)
        elif test_type == StressTestType.LIQUIDITY_CRISIS:
            result = self._run_liquidity_crisis_test(portfolio_weights, asset_returns)
        else:
            raise ValueError(f"不支持的压力测试类型: {test_type}")
        
        logger.info(f"压力测试完成: {test_type.value}")
        return result
    
    def run_comprehensive_stress_test(self,
                                    portfolio_weights: np.ndarray,
                                    asset_returns: pd.DataFrame) -> Dict[str, StressTestResult]:
        """
        执行综合压力测试
        
        Args:
            portfolio_weights: 投资组合权重
            asset_returns: 资产收益率历史数据
            
        Returns:
            各种压力测试结果的字典
        """
        logger.info("开始执行综合压力测试")
        
        results = {}
        
        # 并行执行多种压力测试
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            # 提交各种压力测试任务
            for test_type in StressTestType:
                if test_type in [StressTestType.CORRELATION_BREAKDOWN, StressTestType.LIQUIDITY_CRISIS]:
                    # 这些测试需要特定情景
                    for scenario in [MarketScenario.MARKET_CRASH, MarketScenario.VOLATILITY_SPIKE]:
                        future = executor.submit(self.run_stress_test, 
                                               portfolio_weights, asset_returns, test_type, scenario)
                        futures[f"{test_type.value}_{scenario.value}"] = future
                else:
                    future = executor.submit(self.run_stress_test, 
                                           portfolio_weights, asset_returns, test_type)
                    futures[test_type.value] = future
            
            # 收集结果
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    logger.error(f"压力测试 {name} 执行失败: {e}")
                    continue
        
        logger.info(f"综合压力测试完成，共执行 {len(results)} 项测试")
        return results
    
    def _run_historical_scenario_test(self,
                                    portfolio_weights: np.ndarray,
                                    asset_returns: pd.DataFrame,
                                    scenario: Optional[MarketScenario] = None) -> StressTestResult:
        """执行历史情景重现测试"""
        logger.info("执行历史情景重现测试")
        
        # 识别历史极端事件
        portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
        
        # 找出历史上最严重的损失期间
        rolling_returns = portfolio_returns.rolling(window=self.config.scenario_window_days).sum()
        worst_periods = rolling_returns.nsmallest(100)  # 取最差的100个时期
        
        # 过滤满足最小严重程度的情景
        severe_scenarios = worst_periods[worst_periods <= self.config.min_scenario_severity]
        
        if len(severe_scenarios) == 0:
            logger.warning("未找到满足条件的历史极端情景")
            # 使用最差的时期
            severe_scenarios = worst_periods.head(10)
        
        # 计算每个历史情景下的投资组合损失
        scenario_losses = []
        asset_contributions = {}
        
        for date, loss in severe_scenarios.items():
            # 获取该时期的资产收益率
            start_idx = asset_returns.index.get_loc(date) - self.config.scenario_window_days + 1
            end_idx = asset_returns.index.get_loc(date) + 1
            
            if start_idx >= 0:
                period_returns = asset_returns.iloc[start_idx:end_idx]
                period_portfolio_loss = -((period_returns * portfolio_weights).sum(axis=1).sum())
                scenario_losses.append(period_portfolio_loss)
                
                # 计算资产贡献
                asset_loss_contrib = -(period_returns * portfolio_weights).sum(axis=0)
                for asset, contrib in asset_loss_contrib.items():
                    if asset not in asset_contributions:
                        asset_contributions[asset] = []
                    asset_contributions[asset].append(contrib)
        
        scenario_losses = np.array(scenario_losses)
        
        # 计算平均资产贡献
        avg_asset_contributions = {
            asset: np.mean(contribs) 
            for asset, contribs in asset_contributions.items()
        }
        
        # 计算风险指标
        var_estimates = self._calculate_var(scenario_losses, self.config.confidence_levels)
        cvar_estimates = self._calculate_cvar(scenario_losses, self.config.confidence_levels)
        expected_shortfall = self._calculate_expected_shortfall(scenario_losses, self.config.confidence_levels)
        
        # 计算统计指标
        statistics = self._calculate_statistics(scenario_losses)
        
        return StressTestResult(
            test_type=StressTestType.HISTORICAL_SCENARIO,
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_losses=scenario_losses,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            expected_shortfall=expected_shortfall,
            max_loss=scenario_losses.max(),
            probability_of_loss=np.mean(scenario_losses > 0),
            tail_expectation=np.mean(scenario_losses[scenario_losses > np.percentile(scenario_losses, 95)]),
            asset_contributions=avg_asset_contributions,
            factor_contributions={},  # 历史情景测试暂不计算因子贡献
            statistics=statistics
        )
    
    def _run_monte_carlo_test(self,
                            portfolio_weights: np.ndarray,
                            asset_returns: pd.DataFrame,
                            scenario: Optional[MarketScenario] = None) -> StressTestResult:
        """执行蒙特卡洛模拟测试"""
        logger.info("执行蒙特卡洛模拟测试")
        
        # 估计资产收益率的统计特征
        mean_returns = asset_returns.mean()
        cov_matrix = asset_returns.cov()
        
        # 如果指定了情景，调整参数
        if scenario:
            mean_returns, cov_matrix = self._adjust_parameters_for_scenario(
                mean_returns, cov_matrix, scenario
            )
        
        # 厚尾调整
        if self.config.fat_tail_adjustment:
            # 使用t分布而不是正态分布
            df = 5  # 自由度，控制厚尾程度
            simulated_returns = self._simulate_t_distribution_returns(
                mean_returns, cov_matrix, df, self.config.num_simulations
            )
        else:
            # 标准正态分布模拟
            simulated_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, self.config.num_simulations
            )
        
        # 计算投资组合损失
        portfolio_losses = -(simulated_returns @ portfolio_weights)
        
        # 计算资产贡献
        asset_contributions = {}
        for i, asset in enumerate(asset_returns.columns):
            asset_contributions[asset] = -np.mean(simulated_returns[:, i] * portfolio_weights[i])
        
        # 计算风险指标
        var_estimates = self._calculate_var(portfolio_losses, self.config.confidence_levels)
        cvar_estimates = self._calculate_cvar(portfolio_losses, self.config.confidence_levels)
        expected_shortfall = self._calculate_expected_shortfall(portfolio_losses, self.config.confidence_levels)
        
        # 计算统计指标
        statistics = self._calculate_statistics(portfolio_losses)
        
        return StressTestResult(
            test_type=StressTestType.MONTE_CARLO,
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_losses=portfolio_losses,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            expected_shortfall=expected_shortfall,
            max_loss=portfolio_losses.max(),
            probability_of_loss=np.mean(portfolio_losses > 0),
            tail_expectation=np.mean(portfolio_losses[portfolio_losses > np.percentile(portfolio_losses, 95)]),
            asset_contributions=asset_contributions,
            factor_contributions={},  # 蒙特卡洛测试暂不计算因子贡献
            statistics=statistics
        )
    
    def _run_parametric_var_test(self,
                               portfolio_weights: np.ndarray,
                               asset_returns: pd.DataFrame) -> StressTestResult:
        """执行参数化VaR测试"""
        logger.info("执行参数化VaR测试")
        
        # 计算投资组合收益率统计特征
        portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
        portfolio_mean = portfolio_returns.mean()
        portfolio_std = portfolio_returns.std()
        
        # 计算不同置信水平的VaR
        var_estimates = {}
        cvar_estimates = {}
        expected_shortfall = {}
        
        for conf_level in self.config.confidence_levels:
            # 参数化VaR（假设正态分布）
            z_score = stats.norm.ppf(1 - conf_level)
            var = -(portfolio_mean + z_score * portfolio_std)
            var_estimates[conf_level] = var
            
            # 参数化CVaR
            cvar = -(portfolio_mean - portfolio_std * stats.norm.pdf(z_score) / (1 - conf_level))
            cvar_estimates[conf_level] = cvar
            
            # 期望损失
            expected_shortfall[conf_level] = cvar
        
        # 生成损失分布用于统计分析
        simulated_losses = np.random.normal(
            -portfolio_mean, portfolio_std, self.config.num_simulations
        )
        
        # 计算资产贡献（基于边际VaR）
        asset_contributions = self._calculate_marginal_var_contributions(
            portfolio_weights, asset_returns
        )
        
        # 计算统计指标
        statistics = self._calculate_statistics(simulated_losses)
        
        return StressTestResult(
            test_type=StressTestType.PARAMETRIC_VAR,
            scenario=None,
            timestamp=datetime.now(),
            portfolio_losses=simulated_losses,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            expected_shortfall=expected_shortfall,
            max_loss=simulated_losses.max(),
            probability_of_loss=0.5,  # 正态分布假设下
            tail_expectation=np.mean(simulated_losses[simulated_losses > np.percentile(simulated_losses, 95)]),
            asset_contributions=asset_contributions,
            factor_contributions={},
            statistics=statistics
        )
    
    def _run_extreme_value_test(self,
                              portfolio_weights: np.ndarray,
                              asset_returns: pd.DataFrame) -> StressTestResult:
        """执行极值理论测试"""
        logger.info("执行极值理论测试")
        
        # 计算投资组合收益率
        portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
        portfolio_losses = -portfolio_returns
        
        # 选择阈值（使用95%分位数）
        threshold = np.percentile(portfolio_losses, 95)
        exceedances = portfolio_losses[portfolio_losses > threshold] - threshold
        
        if len(exceedances) < 10:
            logger.warning("极值样本过少，使用90%分位数作为阈值")
            threshold = np.percentile(portfolio_losses, 90)
            exceedances = portfolio_losses[portfolio_losses > threshold] - threshold
        
        # 拟合广义帕累托分布(GPD)
        try:
            # 使用scipy拟合GPD
            shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)
            
            # 计算极值VaR
            var_estimates = {}
            cvar_estimates = {}
            expected_shortfall = {}
            
            n_exceedances = len(exceedances)
            n_total = len(portfolio_losses)
            
            for conf_level in self.config.confidence_levels:
                if conf_level > (n_total - n_exceedances) / n_total:
                    # 计算极值VaR
                    p_u = n_exceedances / n_total
                    var_extreme = threshold + (scale / shape) * (
                        ((n_total / n_exceedances) * (1 - conf_level)) ** (-shape) - 1
                    )
                    var_estimates[conf_level] = var_extreme
                    
                    # 计算极值CVaR
                    if shape < 1:
                        cvar_extreme = var_extreme / (1 - shape) + (scale - shape * threshold) / (1 - shape)
                        cvar_estimates[conf_level] = cvar_extreme
                        expected_shortfall[conf_level] = cvar_extreme
                    else:
                        cvar_estimates[conf_level] = var_extreme * 1.5  # 近似值
                        expected_shortfall[conf_level] = var_extreme * 1.5
                else:
                    # 使用经验分位数
                    var_estimates[conf_level] = np.percentile(portfolio_losses, conf_level * 100)
                    cvar_estimates[conf_level] = np.mean(
                        portfolio_losses[portfolio_losses >= var_estimates[conf_level]]
                    )
                    expected_shortfall[conf_level] = cvar_estimates[conf_level]
            
        except Exception as e:
            logger.error(f"极值理论拟合失败: {e}")
            # 回退到经验方法
            var_estimates = self._calculate_var(portfolio_losses, self.config.confidence_levels)
            cvar_estimates = self._calculate_cvar(portfolio_losses, self.config.confidence_levels)
            expected_shortfall = self._calculate_expected_shortfall(portfolio_losses, self.config.confidence_levels)
        
        # 计算资产贡献
        asset_contributions = {}
        for i, asset in enumerate(asset_returns.columns):
            asset_losses = -asset_returns.iloc[:, i] * portfolio_weights[i]
            asset_contributions[asset] = np.mean(asset_losses[portfolio_losses > threshold])
        
        # 计算统计指标
        statistics = self._calculate_statistics(portfolio_losses)
        statistics['evt_shape_parameter'] = shape if 'shape' in locals() else np.nan
        statistics['evt_scale_parameter'] = scale if 'scale' in locals() else np.nan
        statistics['evt_threshold'] = threshold
        statistics['evt_exceedances'] = len(exceedances)
        
        return StressTestResult(
            test_type=StressTestType.EXTREME_VALUE,
            scenario=None,
            timestamp=datetime.now(),
            portfolio_losses=portfolio_losses,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            expected_shortfall=expected_shortfall,
            max_loss=portfolio_losses.max(),
            probability_of_loss=np.mean(portfolio_losses > 0),
            tail_expectation=np.mean(portfolio_losses[portfolio_losses > np.percentile(portfolio_losses, 95)]),
            asset_contributions=asset_contributions,
            factor_contributions={},
            statistics=statistics
        )
    
    def _run_correlation_breakdown_test(self,
                                      portfolio_weights: np.ndarray,
                                      asset_returns: pd.DataFrame) -> StressTestResult:
        """执行相关性崩溃测试"""
        logger.info("执行相关性崩溃测试")
        
        # 计算正常时期的相关性矩阵
        normal_corr = asset_returns.corr()
        
        # 模拟相关性崩溃：所有相关性向1收敛
        shocked_corr = normal_corr * self.config.correlation_shock_factor + \
                      (1 - self.config.correlation_shock_factor) * np.ones_like(normal_corr)
        shocked_corr = shocked_corr.values  # 转换为numpy数组
        np.fill_diagonal(shocked_corr, 1.0)
        
        # 确保相关性矩阵正定
        shocked_corr = self._make_correlation_matrix_positive_definite(shocked_corr)
        
        # 计算对应的协方差矩阵
        asset_std = asset_returns.std()
        shocked_cov = np.outer(asset_std, asset_std) * shocked_corr
        
        # 蒙特卡洛模拟
        mean_returns = asset_returns.mean()
        simulated_returns = np.random.multivariate_normal(
            mean_returns, shocked_cov, self.config.num_simulations
        )
        
        # 计算投资组合损失
        portfolio_losses = -(simulated_returns @ portfolio_weights)
        
        # 计算资产贡献
        asset_contributions = {}
        for i, asset in enumerate(asset_returns.columns):
            asset_contributions[asset] = -np.mean(simulated_returns[:, i] * portfolio_weights[i])
        
        # 计算风险指标
        var_estimates = self._calculate_var(portfolio_losses, self.config.confidence_levels)
        cvar_estimates = self._calculate_cvar(portfolio_losses, self.config.confidence_levels)
        expected_shortfall = self._calculate_expected_shortfall(portfolio_losses, self.config.confidence_levels)
        
        # 计算统计指标
        statistics = self._calculate_statistics(portfolio_losses)
        statistics['correlation_shock_factor'] = self.config.correlation_shock_factor
        statistics['avg_normal_correlation'] = normal_corr.values[np.triu_indices_from(normal_corr, k=1)].mean()
        statistics['avg_shocked_correlation'] = shocked_corr[np.triu_indices_from(shocked_corr, k=1)].mean()
        
        return StressTestResult(
            test_type=StressTestType.CORRELATION_BREAKDOWN,
            scenario=MarketScenario.MARKET_CRASH,
            timestamp=datetime.now(),
            portfolio_losses=portfolio_losses,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            expected_shortfall=expected_shortfall,
            max_loss=portfolio_losses.max(),
            probability_of_loss=np.mean(portfolio_losses > 0),
            tail_expectation=np.mean(portfolio_losses[portfolio_losses > np.percentile(portfolio_losses, 95)]),
            asset_contributions=asset_contributions,
            factor_contributions={},
            statistics=statistics
        )
    
    def _run_liquidity_crisis_test(self,
                                 portfolio_weights: np.ndarray,
                                 asset_returns: pd.DataFrame) -> StressTestResult:
        """执行流动性危机测试"""
        logger.info("执行流动性危机测试")
        
        # 模拟流动性危机：增加交易成本和价格冲击
        mean_returns = asset_returns.mean()
        cov_matrix = asset_returns.cov()
        
        # 增加波动率以模拟流动性不足
        liquidity_adjusted_cov = cov_matrix * (self.config.liquidity_impact_factor ** 2)
        
        # 添加负的流动性溢价
        liquidity_premium = -0.001 * portfolio_weights  # 流动性成本
        adjusted_mean_returns = mean_returns + liquidity_premium
        
        # 蒙特卡洛模拟
        simulated_returns = np.random.multivariate_normal(
            adjusted_mean_returns, liquidity_adjusted_cov, self.config.num_simulations
        )
        
        # 计算投资组合损失
        portfolio_losses = -(simulated_returns @ portfolio_weights)
        
        # 计算资产贡献
        asset_contributions = {}
        for i, asset in enumerate(asset_returns.columns):
            asset_contributions[asset] = -np.mean(simulated_returns[:, i] * portfolio_weights[i])
        
        # 计算风险指标
        var_estimates = self._calculate_var(portfolio_losses, self.config.confidence_levels)
        cvar_estimates = self._calculate_cvar(portfolio_losses, self.config.confidence_levels)
        expected_shortfall = self._calculate_expected_shortfall(portfolio_losses, self.config.confidence_levels)
        
        # 计算统计指标
        statistics = self._calculate_statistics(portfolio_losses)
        statistics['liquidity_impact_factor'] = self.config.liquidity_impact_factor
        statistics['avg_liquidity_premium'] = np.mean(liquidity_premium)
        
        return StressTestResult(
            test_type=StressTestType.LIQUIDITY_CRISIS,
            scenario=MarketScenario.LIQUIDITY_DROUGHT,
            timestamp=datetime.now(),
            portfolio_losses=portfolio_losses,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            expected_shortfall=expected_shortfall,
            max_loss=portfolio_losses.max(),
            probability_of_loss=np.mean(portfolio_losses > 0),
            tail_expectation=np.mean(portfolio_losses[portfolio_losses > np.percentile(portfolio_losses, 95)]),
            asset_contributions=asset_contributions,
            factor_contributions={},
            statistics=statistics
        )
    
    def generate_stress_test_report(self, 
                                  results: Dict[str, StressTestResult],
                                  output_path: Optional[str] = None) -> str:
        """
        生成压力测试报告
        
        Args:
            results: 压力测试结果字典
            output_path: 输出路径（可选）
            
        Returns:
            报告内容
        """
        logger.info("生成压力测试报告")
        
        report_lines = []
        report_lines.append("# 压力测试报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 执行摘要
        report_lines.append("## 执行摘要")
        report_lines.append(f"共执行 {len(results)} 项压力测试")
        
        # 汇总最严重的风险
        max_var_99 = max([r.var_estimates.get(0.99, 0) for r in results.values()])
        max_loss = max([r.max_loss for r in results.values()])
        
        report_lines.append(f"最大99%VaR: {max_var_99:.4f}")
        report_lines.append(f"最大可能损失: {max_loss:.4f}")
        report_lines.append("")
        
        # 详细结果
        for test_name, result in results.items():
            report_lines.append(f"## {test_name}")
            report_lines.append(f"测试类型: {result.test_type.value}")
            if result.scenario:
                report_lines.append(f"情景: {result.scenario.value}")
            
            report_lines.append("### 风险指标")
            for conf_level in sorted(result.var_estimates.keys()):
                var_val = result.var_estimates[conf_level]
                cvar_val = result.cvar_estimates[conf_level]
                report_lines.append(f"- {conf_level*100:.1f}% VaR: {var_val:.4f}")
                report_lines.append(f"- {conf_level*100:.1f}% CVaR: {cvar_val:.4f}")
            
            report_lines.append(f"- 最大损失: {result.max_loss:.4f}")
            report_lines.append(f"- 损失概率: {result.probability_of_loss:.4f}")
            report_lines.append("")
            
            # 资产贡献
            if result.asset_contributions:
                report_lines.append("### 资产贡献")
                sorted_contributions = sorted(
                    result.asset_contributions.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                for asset, contrib in sorted_contributions[:10]:  # 显示前10个
                    report_lines.append(f"- {asset}: {contrib:.4f}")
                report_lines.append("")
        
        # 风险建议
        report_lines.append("## 风险管理建议")
        report_lines.append("基于压力测试结果，建议:")
        
        if max_var_99 > 0.1:  # 如果99%VaR超过10%
            report_lines.append("- 当前投资组合面临较高的尾部风险，建议降低整体风险暴露")
        
        if max_loss > 0.2:  # 如果最大损失超过20%
            report_lines.append("- 极端情况下可能面临重大损失，建议增加对冲措施")
        
        report_lines.append("- 定期更新压力测试，监控风险变化")
        report_lines.append("- 建立应急预案，应对极端市场情况")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"压力测试报告已保存至: {output_path}")
        
        return report_content
    
    def run_extreme_scenario_simulation(self,
                                      portfolio_weights: np.ndarray,
                                      asset_returns: pd.DataFrame,
                                      scenario: MarketScenario,
                                      custom_parameters: Optional[ExtremeScenarioParameters] = None) -> StressTestResult:
        """
        运行极端情景模拟
        
        Args:
            portfolio_weights: 投资组合权重
            asset_returns: 资产收益率历史数据
            scenario: 市场情景类型
            custom_parameters: 自定义参数（可选）
            
        Returns:
            压力测试结果
        """
        logger.info(f"运行极端情景模拟: {scenario.value}")
        
        # 验证输入
        self._validate_inputs(portfolio_weights, asset_returns)
        
        # 获取情景参数
        if custom_parameters:
            params = custom_parameters
        else:
            params = self.scenario_simulator.scenario_parameters.get(scenario)
            if params is None:
                raise ValueError(f"未定义的情景类型: {scenario}")
        
        # 根据情景类型执行相应的模拟
        if scenario == MarketScenario.MARKET_CRASH:
            portfolio_losses = self.scenario_simulator.simulate_market_crash(
                asset_returns, portfolio_weights,
                crash_magnitude=params.shock_magnitude,
                crash_duration=params.shock_duration
            )
        elif scenario == MarketScenario.LIQUIDITY_DROUGHT:
            portfolio_losses = self.scenario_simulator.simulate_liquidity_crisis(
                asset_returns, portfolio_weights,
                liquidity_shock=params.liquidity_impact,
                bid_ask_widening=params.volatility_spike
            )
        elif scenario == MarketScenario.VOLATILITY_SPIKE:
            # 使用传染效应模拟波动率飙升
            n_assets = len(asset_returns.columns)
            initial_shock_assets = np.random.choice(n_assets, size=max(1, n_assets//4), replace=False)
            portfolio_losses = self.scenario_simulator.simulate_contagion_effect(
                asset_returns, portfolio_weights,
                initial_shock_assets=initial_shock_assets.tolist(),
                contagion_probability=params.contagion_probability,
                contagion_magnitude=0.7
            )
        elif scenario == MarketScenario.BLACK_SWAN:
            portfolio_losses = self.scenario_simulator.simulate_black_swan_event(
                asset_returns, portfolio_weights,
                event_probability=0.1,  # 在模拟中提高概率以获得足够样本
                event_magnitude=params.shock_magnitude
            )
        else:
            # 使用制度转换模拟其他情景
            portfolio_losses = self.scenario_simulator.simulate_regime_shift(
                asset_returns, portfolio_weights,
                shift_probability=0.3,
                new_regime_params={
                    'mean_adjustment': params.shock_magnitude / 10,
                    'volatility_multiplier': params.volatility_spike,
                    'correlation_increase': params.correlation_increase
                }
            )
        
        # 计算资产贡献（简化版本）
        asset_contributions = {}
        for i, asset in enumerate(asset_returns.columns):
            # 基于权重和平均损失估算贡献
            asset_contributions[asset] = np.mean(portfolio_losses) * portfolio_weights[i]
        
        # 计算风险指标
        var_estimates = self._calculate_var(portfolio_losses, self.config.confidence_levels)
        cvar_estimates = self._calculate_cvar(portfolio_losses, self.config.confidence_levels)
        expected_shortfall = self._calculate_expected_shortfall(portfolio_losses, self.config.confidence_levels)
        
        # 计算统计指标
        statistics = self._calculate_statistics(portfolio_losses)
        statistics['scenario_type'] = scenario.value
        statistics['shock_magnitude'] = params.shock_magnitude
        statistics['shock_duration'] = params.shock_duration
        statistics['volatility_spike'] = params.volatility_spike
        statistics['correlation_increase'] = params.correlation_increase
        
        return StressTestResult(
            test_type=StressTestType.EXTREME_VALUE,  # 使用极值类型
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_losses=portfolio_losses,
            var_estimates=var_estimates,
            cvar_estimates=cvar_estimates,
            expected_shortfall=expected_shortfall,
            max_loss=portfolio_losses.max(),
            probability_of_loss=np.mean(portfolio_losses > 0),
            tail_expectation=np.mean(portfolio_losses[portfolio_losses > np.percentile(portfolio_losses, 95)]),
            asset_contributions=asset_contributions,
            factor_contributions={},
            statistics=statistics
        )
    
    def calibrate_extreme_scenarios(self,
                                  asset_returns: pd.DataFrame,
                                  historical_events: Optional[List[Dict]] = None) -> Dict[MarketScenario, ExtremeScenarioParameters]:
        """
        校准极端情景参数
        
        Args:
            asset_returns: 历史资产收益率
            historical_events: 历史事件数据
            
        Returns:
            校准后的情景参数
        """
        logger.info("校准极端情景参数")
        
        calibrated_params = self.scenario_simulator.calibrate_scenario_parameters(
            asset_returns, historical_events
        )
        
        # 更新模拟器参数
        self.scenario_simulator.scenario_parameters.update(calibrated_params)
        
        return calibrated_params
    
    def estimate_scenario_probabilities(self,
                                      asset_returns: pd.DataFrame,
                                      lookback_years: int = 20) -> Dict[MarketScenario, float]:
        """
        估计各种极端情景的发生概率
        
        Args:
            asset_returns: 历史资产收益率
            lookback_years: 回望年数
            
        Returns:
            各情景的发生概率
        """
        return self.scenario_simulator.estimate_scenario_probabilities(asset_returns, lookback_years)
    
    def generate_risk_limits_recommendations(self,
                                           results: Dict[str, StressTestResult],
                                           risk_tolerance: float = 0.05) -> Dict[str, float]:
        """
        基于压力测试结果生成风险限额建议
        
        Args:
            results: 压力测试结果
            risk_tolerance: 风险容忍度
            
        Returns:
            风险限额建议
        """
        logger.info("生成风险限额建议")
        
        recommendations = {}
        
        # 基于最严重情景设置VaR限额
        max_var_99 = max([r.var_estimates.get(0.99, 0) for r in results.values()])
        recommendations['var_99_limit'] = max_var_99 * (1 - risk_tolerance)
        
        # 基于最大损失设置止损限额
        max_loss = max([r.max_loss for r in results.values()])
        recommendations['stop_loss_limit'] = max_loss * (1 - risk_tolerance)
        
        # 基于尾部期望设置风险预算
        max_tail_expectation = max([r.tail_expectation for r in results.values()])
        recommendations['risk_budget_limit'] = max_tail_expectation * 0.8
        
        # 基于损失概率设置仓位限额
        max_loss_prob = max([r.probability_of_loss for r in results.values()])
        if max_loss_prob > 0.3:  # 如果损失概率过高
            recommendations['position_size_reduction'] = 0.2  # 建议减仓20%
        
        # 流动性储备建议
        liquidity_results = [r for r in results.values() 
                           if r.scenario == MarketScenario.LIQUIDITY_DROUGHT]
        if liquidity_results:
            avg_liquidity_loss = np.mean([r.max_loss for r in liquidity_results])
            recommendations['liquidity_reserve'] = avg_liquidity_loss * 1.2
        
        logger.info(f"风险限额建议: {recommendations}")
        return recommendations
    
    def visualize_stress_test_results(self, 
                                    results: Dict[str, StressTestResult],
                                    save_path: Optional[str] = None) -> str:
        """
        可视化压力测试结果
        
        Args:
            results: 压力测试结果字典
            save_path: 保存路径（可选）
            
        Returns:
            HTML内容或文件路径
        """
        logger.info("生成压力测试可视化图表")
        
        # 创建子图
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=('99% VaR 对比', '损失分布', '最大损失对比', '损失概率对比'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        test_names = list(results.keys())
        var_99_values = [results[name].var_estimates.get(0.99, 0) for name in test_names]
        max_losses = [results[name].max_loss for name in test_names]
        loss_probs = [results[name].probability_of_loss for name in test_names]
        
        # 1. VaR对比图
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=var_99_values,
                name='99% VaR',
                text=[f'{v:.3f}' for v in var_99_values],
                textposition='auto',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. 损失分布直方图（选择一个代表性结果）
        if results:
            representative_result = list(results.values())[0]
            representative_name = list(results.keys())[0]
            
            fig.add_trace(
                go.Histogram(
                    x=representative_result.portfolio_losses,
                    nbinsx=50,
                    name=f'损失分布 ({representative_name})',
                    opacity=0.7,
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            # 添加VaR线
            var_99 = representative_result.var_estimates.get(0.99, 0)
            fig.add_vline(
                x=var_99,
                line_dash="dash",
                line_color="red",
                annotation_text="99% VaR",
                row=1, col=2
            )
        
        # 3. 最大损失对比
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=max_losses,
                name='最大损失',
                text=[f'{v:.3f}' for v in max_losses],
                textposition='auto',
                marker_color='orange'
            ),
            row=2, col=1
        )
        
        # 4. 损失概率对比
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=loss_probs,
                name='损失概率',
                text=[f'{v:.3f}' for v in loss_probs],
                textposition='auto',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title_text="压力测试结果可视化",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        # 更新x轴标签
        for i in range(1, 3):
            for j in range(1, 3):
                if i == 1 and j == 2:
                    continue  # 跳过直方图
                fig.update_xaxes(tickangle=45, row=i, col=j)
        
        # 保存或显示
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
                logger.info(f"压力测试可视化图表已保存至: {save_path}")
                return save_path
            else:
                # 保存为图片格式
                fig.write_image(save_path)
                logger.info(f"压力测试可视化图表已保存至: {save_path}")
                return save_path
        else:
            # 返回HTML内容
            html_content = fig.to_html(include_plotlyjs=True)
            return html_content
    
    # 辅助方法
    def _validate_inputs(self, portfolio_weights: np.ndarray, asset_returns: pd.DataFrame):
        """验证输入参数"""
        if asset_returns.empty:
            raise ValueError("资产收益率数据不能为空")
        
        if len(portfolio_weights) != len(asset_returns.columns):
            raise ValueError("投资组合权重数量与资产数量不匹配")
        
        if abs(portfolio_weights.sum() - 1.0) > 1e-6:
            raise ValueError("投资组合权重和必须接近1")
        
        if (portfolio_weights < 0).any():
            raise ValueError("投资组合权重不能为负数")
    
    def _define_extreme_scenarios(self) -> Dict[MarketScenario, ScenarioDefinition]:
        """定义极端市场情景"""
        scenarios = {}
        
        # 市场崩盘
        scenarios[MarketScenario.MARKET_CRASH] = ScenarioDefinition(
            name="市场崩盘",
            description="股市大幅下跌，类似2008年金融危机",
            scenario_type=MarketScenario.MARKET_CRASH,
            market_shock=-0.3,
            volatility_multiplier=3.0,
            correlation_adjustment=0.8,
            duration_days=30,
            recovery_days=180,
            probability=0.01
        )
        
        # 波动率飙升
        scenarios[MarketScenario.VOLATILITY_SPIKE] = ScenarioDefinition(
            name="波动率飙升",
            description="市场波动率急剧上升",
            scenario_type=MarketScenario.VOLATILITY_SPIKE,
            market_shock=-0.1,
            volatility_multiplier=2.5,
            correlation_adjustment=0.6,
            duration_days=14,
            recovery_days=60,
            probability=0.05
        )
        
        # 流动性枯竭
        scenarios[MarketScenario.LIQUIDITY_DROUGHT] = ScenarioDefinition(
            name="流动性枯竭",
            description="市场流动性严重不足",
            scenario_type=MarketScenario.LIQUIDITY_DROUGHT,
            market_shock=-0.15,
            volatility_multiplier=2.0,
            correlation_adjustment=0.7,
            duration_days=21,
            recovery_days=90,
            probability=0.02
        )
        
        return scenarios
    
    def _adjust_parameters_for_scenario(self, 
                                      mean_returns: pd.Series, 
                                      cov_matrix: pd.DataFrame,
                                      scenario: MarketScenario) -> Tuple[pd.Series, pd.DataFrame]:
        """根据情景调整参数"""
        if scenario not in self.extreme_scenarios:
            return mean_returns, cov_matrix
        
        scenario_def = self.extreme_scenarios[scenario]
        
        # 调整均值收益率
        adjusted_mean = mean_returns + scenario_def.market_shock / 252  # 日化
        
        # 调整协方差矩阵
        adjusted_cov = cov_matrix * (scenario_def.volatility_multiplier ** 2)
        
        # 调整相关性
        corr_matrix = cov_matrix.corr()
        adjusted_corr = corr_matrix * scenario_def.correlation_adjustment + \
                       (1 - scenario_def.correlation_adjustment) * np.eye(len(corr_matrix))
        
        # 重构协方差矩阵
        std_vector = np.sqrt(np.diag(adjusted_cov))
        adjusted_cov = np.outer(std_vector, std_vector) * adjusted_corr
        adjusted_cov = pd.DataFrame(adjusted_cov, index=cov_matrix.index, columns=cov_matrix.columns)
        
        return adjusted_mean, adjusted_cov
    
    def _simulate_t_distribution_returns(self, 
                                       mean_returns: pd.Series, 
                                       cov_matrix: pd.DataFrame, 
                                       df: int, 
                                       n_simulations: int) -> np.ndarray:
        """模拟t分布收益率"""
        # 生成多元t分布随机数
        normal_samples = np.random.multivariate_normal(
            np.zeros(len(mean_returns)), cov_matrix, n_simulations
        )
        
        # 生成卡方分布随机数
        chi2_samples = np.random.chisquare(df, n_simulations)
        
        # 构造t分布样本
        t_samples = normal_samples * np.sqrt(df / chi2_samples[:, np.newaxis])
        
        # 添加均值
        return t_samples + mean_returns.values
    
    def _make_correlation_matrix_positive_definite(self, corr_matrix: np.ndarray) -> np.ndarray:
        """确保相关性矩阵正定"""
        eigenvals, eigenvecs = np.linalg.eigh(corr_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)  # 确保特征值为正
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    def _calculate_var(self, losses: np.ndarray, confidence_levels: List[float]) -> Dict[float, float]:
        """计算VaR"""
        return {conf: np.percentile(losses, conf * 100) for conf in confidence_levels}
    
    def _calculate_cvar(self, losses: np.ndarray, confidence_levels: List[float]) -> Dict[float, float]:
        """计算CVaR（条件VaR）"""
        cvar_dict = {}
        for conf in confidence_levels:
            var_threshold = np.percentile(losses, conf * 100)
            tail_losses = losses[losses >= var_threshold]
            cvar_dict[conf] = np.mean(tail_losses) if len(tail_losses) > 0 else var_threshold
        return cvar_dict
    
    def _calculate_expected_shortfall(self, losses: np.ndarray, confidence_levels: List[float]) -> Dict[float, float]:
        """计算期望损失"""
        # 期望损失与CVaR相同
        return self._calculate_cvar(losses, confidence_levels)
    
    def _calculate_marginal_var_contributions(self, 
                                            portfolio_weights: np.ndarray, 
                                            asset_returns: pd.DataFrame) -> Dict[str, float]:
        """计算边际VaR贡献"""
        contributions = {}
        
        # 计算投资组合VaR
        portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
        portfolio_var = np.percentile(-portfolio_returns, 99)
        
        # 计算每个资产的边际贡献
        epsilon = 0.01  # 小的权重变化
        
        for i, asset in enumerate(asset_returns.columns):
            # 增加该资产权重
            perturbed_weights = portfolio_weights.copy()
            perturbed_weights[i] += epsilon
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # 重新标准化
            
            # 计算新的VaR
            perturbed_returns = (asset_returns * perturbed_weights).sum(axis=1)
            perturbed_var = np.percentile(-perturbed_returns, 99)
            
            # 边际贡献
            marginal_contrib = (perturbed_var - portfolio_var) / epsilon
            contributions[asset] = marginal_contrib * portfolio_weights[i]
        
        return contributions
    
    def _calculate_statistics(self, losses: np.ndarray) -> Dict[str, float]:
        """计算统计指标"""
        return {
            'mean': np.mean(losses),
            'std': np.std(losses),
            'skewness': stats.skew(losses),
            'kurtosis': stats.kurtosis(losses),
            'min': np.min(losses),
            'max': np.max(losses),
            'median': np.median(losses),
            'q25': np.percentile(losses, 25),
            'q75': np.percentile(losses, 75)
        }