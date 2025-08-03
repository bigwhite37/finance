"""仓位优化算法实现"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum
import logging
from abc import ABC, abstractmethod
from scipy.optimize import minimize, LinearConstraint, Bounds
import warnings


class OptimizationMethod(Enum):
    """优化方法枚举"""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    EQUAL_WEIGHT = "equal_weight"


class ObjectiveType(Enum):
    """目标函数类型枚举"""
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MAXIMIZE_UTILITY = "maximize_utility"
    MINIMIZE_TRACKING_ERROR = "minimize_tracking_error"


@dataclass
class OptimizationConfig:
    """优化配置"""
    # 基本参数
    method: OptimizationMethod = OptimizationMethod.MEAN_VARIANCE
    objective: ObjectiveType = ObjectiveType.MAXIMIZE_SHARPE
    risk_aversion: float = 1.0                       # 风险厌恶系数
    
    # 约束参数
    min_weight: float = 0.0                          # 最小权重
    max_weight: float = 1.0                          # 最大权重
    max_turnover: float = 0.5                        # 最大换手率
    target_return: Optional[float] = None            # 目标收益率
    target_risk: Optional[float] = None              # 目标风险
    
    # 交易成本参数
    transaction_cost_rate: float = 0.001             # 交易成本率
    market_impact_factor: float = 0.0001             # 市场冲击因子
    bid_ask_spread: float = 0.0005                   # 买卖价差
    
    # 优化参数
    max_iterations: int = 1000                       # 最大迭代次数
    tolerance: float = 1e-6                          # 收敛容差
    regularization: float = 1e-8                     # 正则化参数
    
    # 风险模型参数
    lookback_days: int = 252                         # 历史数据回看天数
    half_life: int = 63                              # 半衰期（天）
    shrinkage_factor: float = 0.1                    # 收缩因子
    
    # Black-Litterman参数
    tau: float = 0.025                               # 不确定性参数
    confidence_level: float = 0.95                   # 置信水平


@dataclass
class AssetData:
    """资产数据"""
    symbol: str
    expected_return: float                           # 预期收益率
    volatility: float                                # 波动率
    current_weight: float                            # 当前权重
    market_cap: float                                # 市值
    liquidity: float                                 # 流动性指标
    sector: str                                      # 行业
    beta: float = 1.0                                # 贝塔系数
    dividend_yield: float = 0.0                      # 股息率
    price: float = 100.0                             # 当前价格
    shares_outstanding: float = 1e6                  # 流通股数
    
    def __post_init__(self):
        if self.market_cap <= 0:
            self.market_cap = self.price * self.shares_outstanding


@dataclass
class OptimizationResult:
    """优化结果"""
    optimal_weights: Dict[str, float]                # 最优权重
    expected_return: float                           # 预期收益率
    expected_risk: float                             # 预期风险
    sharpe_ratio: float                              # 夏普比率
    turnover: float                                  # 换手率
    transaction_costs: float                         # 交易成本
    objective_value: float                           # 目标函数值
    optimization_status: str                         # 优化状态
    iterations: int                                  # 迭代次数
    computation_time: float                          # 计算时间
    metadata: Dict[str, Any]                        # 元数据
    timestamp: datetime


class PositionOptimizer:
    """仓位优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 优化历史
        self.optimization_history: List[OptimizationResult] = []
        
        # 缓存
        self.covariance_matrix_cache: Optional[np.ndarray] = None
        self.expected_returns_cache: Optional[np.ndarray] = None
        self.cache_timestamp: Optional[datetime] = None
        
        # 性能统计
        self.performance_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_computation_time': 0.0,
            'cache_hits': 0
        }
    
    def optimize_portfolio(self, 
                          asset_data: Dict[str, AssetData],
                          historical_returns: Optional[pd.DataFrame] = None,
                          benchmark_weights: Optional[Dict[str, float]] = None,
                          custom_constraints: Optional[List[Any]] = None) -> OptimizationResult:
        """
        优化投资组合
        
        Args:
            asset_data: 资产数据
            historical_returns: 历史收益率数据
            benchmark_weights: 基准权重
            custom_constraints: 自定义约束
            
        Returns:
            优化结果
        """
        start_time = datetime.now()
        
        try:
            # 准备数据
            symbols = list(asset_data.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                raise ValueError("资产列表不能为空")
            
            # 构建预期收益率向量
            expected_returns = self._build_expected_returns(asset_data, historical_returns)
            
            # 构建协方差矩阵
            covariance_matrix = self._build_covariance_matrix(asset_data, historical_returns)
            
            # 获取当前权重
            current_weights = np.array([asset_data[symbol].current_weight for symbol in symbols])
            
            # 执行优化
            if self.config.method == OptimizationMethod.MEAN_VARIANCE:
                result = self._mean_variance_optimization(
                    expected_returns, covariance_matrix, current_weights, symbols, custom_constraints
                )
            elif self.config.method == OptimizationMethod.RISK_PARITY:
                result = self._risk_parity_optimization(
                    covariance_matrix, current_weights, symbols
                )
            elif self.config.method == OptimizationMethod.MIN_VARIANCE:
                result = self._min_variance_optimization(
                    covariance_matrix, current_weights, symbols, custom_constraints
                )
            elif self.config.method == OptimizationMethod.MAX_SHARPE:
                result = self._max_sharpe_optimization(
                    expected_returns, covariance_matrix, current_weights, symbols, custom_constraints
                )
            elif self.config.method == OptimizationMethod.BLACK_LITTERMAN:
                result = self._black_litterman_optimization(
                    expected_returns, covariance_matrix, current_weights, symbols, benchmark_weights
                )
            elif self.config.method == OptimizationMethod.EQUAL_WEIGHT:
                result = self._equal_weight_optimization(symbols)
            else:
                raise ValueError(f"不支持的优化方法: {self.config.method}")
            
            # 计算性能指标
            computation_time = (datetime.now() - start_time).total_seconds()
            result.computation_time = computation_time
            result.timestamp = datetime.now()
            
            # 更新统计信息
            self.performance_stats['total_optimizations'] += 1
            if result.optimization_status == 'success':
                self.performance_stats['successful_optimizations'] += 1
            
            # 更新平均计算时间
            total_time = (self.performance_stats['average_computation_time'] * 
                         (self.performance_stats['total_optimizations'] - 1) + computation_time)
            self.performance_stats['average_computation_time'] = total_time / self.performance_stats['total_optimizations']
            
            # 保存历史
            self.optimization_history.append(result)
            
            self.logger.info(f"投资组合优化完成: 方法={self.config.method.value}, "
                           f"状态={result.optimization_status}, "
                           f"夏普比率={result.sharpe_ratio:.3f}, "
                           f"计算时间={computation_time:.3f}秒")
            
            return result
            
        except Exception as e:
            self.logger.error(f"投资组合优化失败: {str(e)}")
            # 返回等权重作为备选方案
            return self._equal_weight_optimization(list(asset_data.keys()), error_message=str(e))
    
    def _build_expected_returns(self, 
                              asset_data: Dict[str, AssetData],
                              historical_returns: Optional[pd.DataFrame] = None) -> np.ndarray:
        """构建预期收益率向量"""
        symbols = list(asset_data.keys())
        
        if historical_returns is not None and not historical_returns.empty:
            # 使用历史数据计算预期收益率
            returns = historical_returns[symbols].dropna()
            if len(returns) > 0:
                # 指数加权移动平均
                weights = np.exp(-np.arange(len(returns)) / self.config.half_life)
                weights = weights[::-1] / weights.sum()
                expected_returns = np.average(returns.values, axis=0, weights=weights)
            else:
                expected_returns = np.array([asset_data[symbol].expected_return for symbol in symbols])
        else:
            # 使用提供的预期收益率
            expected_returns = np.array([asset_data[symbol].expected_return for symbol in symbols])
        
        return expected_returns
    
    def _build_covariance_matrix(self, 
                               asset_data: Dict[str, AssetData],
                               historical_returns: Optional[pd.DataFrame] = None) -> np.ndarray:
        """构建协方差矩阵"""
        symbols = list(asset_data.keys())
        n_assets = len(symbols)
        
        if historical_returns is not None and not historical_returns.empty:
            # 使用历史数据计算协方差矩阵
            returns = historical_returns[symbols].dropna()
            if len(returns) > self.config.half_life:
                # 指数加权协方差矩阵
                weights = np.exp(-np.arange(len(returns)) / self.config.half_life)
                weights = weights[::-1] / weights.sum()
                
                # 计算加权协方差矩阵
                weighted_returns = returns.values * np.sqrt(weights).reshape(-1, 1)
                covariance_matrix = np.cov(weighted_returns.T)
            else:
                covariance_matrix = returns.cov().values
        else:
            # 使用对角协方差矩阵（基于波动率）
            volatilities = np.array([asset_data[symbol].volatility for symbol in symbols])
            covariance_matrix = np.diag(volatilities ** 2)
        
        # 添加正则化以确保正定性
        covariance_matrix += np.eye(n_assets) * self.config.regularization
        
        # Ledoit-Wolf收缩估计
        if self.config.shrinkage_factor > 0:
            target = np.trace(covariance_matrix) / n_assets * np.eye(n_assets)
            covariance_matrix = ((1 - self.config.shrinkage_factor) * covariance_matrix + 
                               self.config.shrinkage_factor * target)
        
        return covariance_matrix
    
    def _mean_variance_optimization(self, 
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray,
                                  current_weights: np.ndarray,
                                  symbols: List[str],
                                  custom_constraints: Optional[List[Any]] = None) -> OptimizationResult:
        """均值方差优化"""
        n_assets = len(symbols)
        
        # 目标函数：最大化效用 = 预期收益 - 0.5 * 风险厌恶 * 方差
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            
            # 交易成本
            turnover = np.sum(np.abs(weights - current_weights))
            transaction_costs = self._calculate_transaction_costs(weights, current_weights, symbols)
            
            # 效用函数
            utility = portfolio_return - 0.5 * self.config.risk_aversion * portfolio_variance - transaction_costs
            return -utility  # 最小化负效用
        
        # 约束条件
        constraints = []
        
        # 权重和为1
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        
        # 目标收益率约束
        if self.config.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - self.config.target_return
            })
        
        # 目标风险约束
        if self.config.target_risk is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sqrt(np.dot(w, np.dot(covariance_matrix, w))) - self.config.target_risk
            })
        
        # 换手率约束
        if self.config.max_turnover < float('inf'):
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.config.max_turnover - np.sum(np.abs(w - current_weights))
            })
        
        # 自定义约束
        if custom_constraints:
            constraints.extend(custom_constraints)
        
        # 边界约束
        bounds = Bounds(
            lb=np.full(n_assets, self.config.min_weight),
            ub=np.full(n_assets, self.config.max_weight)
        )
        
        # 初始猜测
        x0 = current_weights if np.sum(current_weights) > 0 else np.ones(n_assets) / n_assets
        
        # 执行优化
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': self.config.max_iterations,
                        'ftol': self.config.tolerance
                    }
                )
            
            if result.success:
                optimal_weights = result.x
                status = 'success'
                iterations = result.nit
            else:
                # 如果优化失败，使用等权重
                optimal_weights = np.ones(n_assets) / n_assets
                status = f'failed: {result.message}'
                iterations = result.nit if hasattr(result, 'nit') else 0
        
        except Exception as e:
            optimal_weights = np.ones(n_assets) / n_assets
            status = f'error: {str(e)}'
            iterations = 0
        
        # 计算结果指标
        return self._create_optimization_result(
            optimal_weights, expected_returns, covariance_matrix, current_weights,
            symbols, status, iterations
        )
    
    def _risk_parity_optimization(self, 
                                covariance_matrix: np.ndarray,
                                current_weights: np.ndarray,
                                symbols: List[str]) -> OptimizationResult:
        """风险平价优化"""
        n_assets = len(symbols)
        
        # 目标函数：最小化风险贡献的差异
        def objective(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            marginal_contrib = np.dot(covariance_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            # 目标风险贡献（等权重）
            target_contrib = np.ones(n_assets) / n_assets
            
            # 最小化风险贡献与目标的差异
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # 边界约束
        bounds = Bounds(
            lb=np.full(n_assets, self.config.min_weight),
            ub=np.full(n_assets, self.config.max_weight)
        )
        
        # 初始猜测：逆波动率权重
        volatilities = np.sqrt(np.diag(covariance_matrix))
        x0 = (1 / volatilities) / np.sum(1 / volatilities)
        
        # 执行优化
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': self.config.max_iterations,
                        'ftol': self.config.tolerance
                    }
                )
            
            if result.success:
                optimal_weights = result.x
                status = 'success'
                iterations = result.nit
            else:
                optimal_weights = x0
                status = f'failed: {result.message}'
                iterations = result.nit if hasattr(result, 'nit') else 0
        
        except Exception as e:
            optimal_weights = x0
            status = f'error: {str(e)}'
            iterations = 0
        
        # 使用零预期收益率（风险平价不依赖收益预测）
        expected_returns = np.zeros(n_assets)
        
        return self._create_optimization_result(
            optimal_weights, expected_returns, covariance_matrix, current_weights,
            symbols, status, iterations
        )
    
    def _min_variance_optimization(self, 
                                 covariance_matrix: np.ndarray,
                                 current_weights: np.ndarray,
                                 symbols: List[str],
                                 custom_constraints: Optional[List[Any]] = None) -> OptimizationResult:
        """最小方差优化"""
        n_assets = len(symbols)
        
        # 目标函数：最小化投资组合方差
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        if custom_constraints:
            constraints.extend(custom_constraints)
        
        # 边界约束
        bounds = Bounds(
            lb=np.full(n_assets, self.config.min_weight),
            ub=np.full(n_assets, self.config.max_weight)
        )
        
        # 初始猜测
        x0 = np.ones(n_assets) / n_assets
        
        # 执行优化
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': self.config.max_iterations,
                        'ftol': self.config.tolerance
                    }
                )
            
            if result.success:
                optimal_weights = result.x
                status = 'success'
                iterations = result.nit
            else:
                optimal_weights = x0
                status = f'failed: {result.message}'
                iterations = result.nit if hasattr(result, 'nit') else 0
        
        except Exception as e:
            optimal_weights = x0
            status = f'error: {str(e)}'
            iterations = 0
        
        # 使用零预期收益率
        expected_returns = np.zeros(n_assets)
        
        return self._create_optimization_result(
            optimal_weights, expected_returns, covariance_matrix, current_weights,
            symbols, status, iterations
        )
    
    def _max_sharpe_optimization(self, 
                               expected_returns: np.ndarray,
                               covariance_matrix: np.ndarray,
                               current_weights: np.ndarray,
                               symbols: List[str],
                               custom_constraints: Optional[List[Any]] = None) -> OptimizationResult:
        """最大夏普比率优化"""
        n_assets = len(symbols)
        
        # 目标函数：最大化夏普比率（等价于最小化负夏普比率）
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_volatility == 0:
                return -float('inf')
            
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        if custom_constraints:
            constraints.extend(custom_constraints)
        
        # 边界约束
        bounds = Bounds(
            lb=np.full(n_assets, self.config.min_weight),
            ub=np.full(n_assets, self.config.max_weight)
        )
        
        # 初始猜测
        x0 = np.ones(n_assets) / n_assets
        
        # 执行优化
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': self.config.max_iterations,
                        'ftol': self.config.tolerance
                    }
                )
            
            if result.success:
                optimal_weights = result.x
                status = 'success'
                iterations = result.nit
            else:
                optimal_weights = x0
                status = f'failed: {result.message}'
                iterations = result.nit if hasattr(result, 'nit') else 0
        
        except Exception as e:
            optimal_weights = x0
            status = f'error: {str(e)}'
            iterations = 0
        
        return self._create_optimization_result(
            optimal_weights, expected_returns, covariance_matrix, current_weights,
            symbols, status, iterations
        )
    
    def _black_litterman_optimization(self, 
                                    expected_returns: np.ndarray,
                                    covariance_matrix: np.ndarray,
                                    current_weights: np.ndarray,
                                    symbols: List[str],
                                    benchmark_weights: Optional[Dict[str, float]] = None) -> OptimizationResult:
        """Black-Litterman优化"""
        n_assets = len(symbols)
        
        # 如果没有基准权重，使用市值权重
        if benchmark_weights is None:
            market_weights = np.ones(n_assets) / n_assets
        else:
            market_weights = np.array([benchmark_weights.get(symbol, 0) for symbol in symbols])
            market_weights = market_weights / np.sum(market_weights)
        
        # 隐含收益率（反向优化）
        risk_aversion = self.config.risk_aversion
        implied_returns = risk_aversion * np.dot(covariance_matrix, market_weights)
        
        # Black-Litterman调整
        # 这里简化实现，实际应用中需要投资者观点
        tau = self.config.tau
        
        # 先验分布
        mu_prior = implied_returns
        sigma_prior = tau * covariance_matrix
        
        # 后验分布（无观点情况下等于先验）
        mu_bl = mu_prior
        sigma_bl = sigma_prior
        
        # 使用调整后的参数进行均值方差优化
        def objective(weights):
            portfolio_return = np.dot(weights, mu_bl)
            portfolio_variance = np.dot(weights, np.dot(sigma_bl, weights))
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
            return -utility
        
        # 约束条件
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # 边界约束
        bounds = Bounds(
            lb=np.full(n_assets, self.config.min_weight),
            ub=np.full(n_assets, self.config.max_weight)
        )
        
        # 初始猜测
        x0 = market_weights
        
        # 执行优化
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': self.config.max_iterations,
                        'ftol': self.config.tolerance
                    }
                )
            
            if result.success:
                optimal_weights = result.x
                status = 'success'
                iterations = result.nit
            else:
                optimal_weights = x0
                status = f'failed: {result.message}'
                iterations = result.nit if hasattr(result, 'nit') else 0
        
        except Exception as e:
            optimal_weights = x0
            status = f'error: {str(e)}'
            iterations = 0
        
        return self._create_optimization_result(
            optimal_weights, mu_bl, sigma_bl, current_weights,
            symbols, status, iterations
        )
    
    def _equal_weight_optimization(self, 
                                 symbols: List[str],
                                 error_message: Optional[str] = None) -> OptimizationResult:
        """等权重优化（备选方案）"""
        n_assets = len(symbols)
        optimal_weights = np.ones(n_assets) / n_assets
        
        # 创建虚拟数据
        expected_returns = np.zeros(n_assets)
        covariance_matrix = np.eye(n_assets) * 0.01
        current_weights = np.zeros(n_assets)
        
        status = 'fallback_equal_weight'
        if error_message:
            status += f': {error_message}'
        
        return self._create_optimization_result(
            optimal_weights, expected_returns, covariance_matrix, current_weights,
            symbols, status, 0
        )
    
    def _calculate_transaction_costs(self, 
                                   new_weights: np.ndarray,
                                   current_weights: np.ndarray,
                                   symbols: List[str]) -> float:
        """计算交易成本"""
        # 换手率
        turnover = np.sum(np.abs(new_weights - current_weights))
        
        # 基础交易成本
        basic_cost = turnover * self.config.transaction_cost_rate
        
        # 买卖价差成本
        spread_cost = turnover * self.config.bid_ask_spread / 2
        
        # 市场冲击成本（简化模型）
        impact_cost = turnover ** 1.5 * self.config.market_impact_factor
        
        total_cost = basic_cost + spread_cost + impact_cost
        return total_cost
    
    def _create_optimization_result(self, 
                                  optimal_weights: np.ndarray,
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray,
                                  current_weights: np.ndarray,
                                  symbols: List[str],
                                  status: str,
                                  iterations: int) -> OptimizationResult:
        """创建优化结果"""
        # 计算投资组合指标
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance = np.dot(optimal_weights, np.dot(covariance_matrix, optimal_weights))
        portfolio_risk = np.sqrt(portfolio_variance)
        
        # 夏普比率
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # 换手率
        turnover = np.sum(np.abs(optimal_weights - current_weights))
        
        # 交易成本
        transaction_costs = self._calculate_transaction_costs(optimal_weights, current_weights, symbols)
        
        # 目标函数值
        if self.config.objective == ObjectiveType.MAXIMIZE_RETURN:
            objective_value = portfolio_return
        elif self.config.objective == ObjectiveType.MINIMIZE_RISK:
            objective_value = -portfolio_risk
        elif self.config.objective == ObjectiveType.MAXIMIZE_SHARPE:
            objective_value = sharpe_ratio
        else:
            objective_value = portfolio_return - 0.5 * self.config.risk_aversion * portfolio_variance
        
        # 创建权重字典
        weights_dict = {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
        
        return OptimizationResult(
            optimal_weights=weights_dict,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            turnover=turnover,
            transaction_costs=transaction_costs,
            objective_value=objective_value,
            optimization_status=status,
            iterations=iterations,
            computation_time=0.0,  # 将在调用函数中设置
            metadata={
                'method': self.config.method.value if hasattr(self.config.method, 'value') else str(self.config.method),
                'objective': self.config.objective.value if hasattr(self.config.objective, 'value') else str(self.config.objective),
                'n_assets': len(symbols),
                'risk_aversion': self.config.risk_aversion
            },
            timestamp=datetime.now()
        )
    
    def get_efficient_frontier(self, 
                             asset_data: Dict[str, AssetData],
                             historical_returns: Optional[pd.DataFrame] = None,
                             n_points: int = 50) -> List[OptimizationResult]:
        """计算有效前沿"""
        symbols = list(asset_data.keys())
        expected_returns = self._build_expected_returns(asset_data, historical_returns)
        covariance_matrix = self._build_covariance_matrix(asset_data, historical_returns)
        current_weights = np.array([asset_data[symbol].current_weight for symbol in symbols])
        
        # 计算最小和最大预期收益率
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        
        # 生成目标收益率序列
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_frontier = []
        original_config = self.config
        
        for target_return in target_returns:
            # 临时修改配置
            self.config = OptimizationConfig(
                method=OptimizationMethod.MIN_VARIANCE,
                target_return=target_return,
                min_weight=original_config.min_weight,
                max_weight=original_config.max_weight,
                max_iterations=original_config.max_iterations,
                tolerance=original_config.tolerance
            )
            
            try:
                result = self._min_variance_optimization(
                    covariance_matrix, current_weights, symbols
                )
                result.expected_return = target_return  # 确保使用目标收益率
                efficient_frontier.append(result)
            except Exception as e:
                self.logger.warning(f"计算有效前沿点失败 (目标收益率={target_return:.4f}): {str(e)}")
        
        # 恢复原始配置
        self.config = original_config
        
        return efficient_frontier
    
    def get_performance_attribution(self, 
                                  result: OptimizationResult,
                                  asset_data: Dict[str, AssetData]) -> Dict[str, Any]:
        """获取业绩归因分析"""
        attribution = {
            'asset_contribution': {},
            'sector_contribution': {},
            'factor_contribution': {},
            'risk_contribution': {}
        }
        
        # 资产贡献
        for symbol, weight in result.optimal_weights.items():
            if symbol in asset_data:
                asset_return = asset_data[symbol].expected_return
                attribution['asset_contribution'][symbol] = weight * asset_return
        
        # 行业贡献
        sector_weights = {}
        sector_returns = {}
        for symbol, weight in result.optimal_weights.items():
            if symbol in asset_data:
                sector = asset_data[symbol].sector
                if sector not in sector_weights:
                    sector_weights[sector] = 0
                    sector_returns[sector] = 0
                sector_weights[sector] += weight
                sector_returns[sector] += weight * asset_data[symbol].expected_return
        
        for sector in sector_weights:
            if sector_weights[sector] > 0:
                attribution['sector_contribution'][sector] = sector_returns[sector]
        
        return attribution
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return {
            'total_optimizations': len(self.optimization_history),
            'performance_stats': self.performance_stats.copy(),
            'recent_results': [
                {
                    'timestamp': result.timestamp,
                    'method': result.metadata.get('method'),
                    'status': result.optimization_status,
                    'sharpe_ratio': result.sharpe_ratio,
                    'turnover': result.turnover,
                    'computation_time': result.computation_time
                }
                for result in self.optimization_history[-10:]  # 最近10次结果
            ]
        }
    
    def reset_history(self) -> None:
        """重置优化历史"""
        self.optimization_history.clear()
        self.performance_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'average_computation_time': 0.0,
            'cache_hits': 0
        }
        self.logger.info("仓位优化历史已重置")