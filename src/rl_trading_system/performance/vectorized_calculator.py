"""
向量化计算器

该模块提供高性能的向量化金融计算功能，包括：
- 向量化回撤计算
- 批量投资组合指标计算
- 并行风险计算
- 滚动窗口指标计算
- 内存优化的大数据集处理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class VectorizedCalculator:
    """
    向量化计算器
    
    提供高性能的向量化金融计算功能，通过以下技术提升性能：
    - NumPy向量化操作
    - 并行处理
    - 内存缓存
    - 算法优化
    """
    
    def __init__(self, 
                 enable_parallel: bool = True,
                 n_jobs: int = None,
                 cache_size: int = 128,
                 chunk_size: int = 1000):
        """
        初始化向量化计算器
        
        Args:
            enable_parallel: 是否启用并行计算
            n_jobs: 并行作业数，None表示使用所有CPU核心
            cache_size: 缓存大小
            chunk_size: 大数据集分块大小
        """
        self.enable_parallel = enable_parallel
        self.n_jobs = n_jobs or mp.cpu_count()
        self.cache_size = cache_size
        self.chunk_size = chunk_size
        self.enable_memory_optimization = False
        
        # 初始化缓存
        self._cache = {}
        self._cache_timestamps = {}
        self._cache_max_age = 3600  # 缓存1小时过期
        
        # 配置NumPy优化
        self._configure_numpy_optimization()
        
        logger.info(f"向量化计算器初始化完成，并行作业数: {self.n_jobs}")
    
    def _configure_numpy_optimization(self):
        """配置NumPy性能优化"""
        # 设置NumPy错误处理
        np.seterr(divide='warn', invalid='warn', over='warn')
        
        # 配置线程数
        try:
            import os
            os.environ['OMP_NUM_THREADS'] = str(self.n_jobs)
            os.environ['MKL_NUM_THREADS'] = str(self.n_jobs)
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.n_jobs)
        except Exception as e:
            logger.warning(f"配置NumPy线程数失败: {e}")
    
    def _get_cache_key(self, data: np.ndarray, operation: str, **kwargs) -> str:
        """生成缓存键"""
        # 计算数据哈希
        data_hash = hashlib.md5(data.tobytes()).hexdigest()
        
        # 包含操作和参数
        params_str = f"{operation}_{data.shape}_{kwargs}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return f"{data_hash}_{params_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """从缓存获取结果"""
        if cache_key not in self._cache:
            return None
        
        # 检查缓存是否过期
        if cache_key in self._cache_timestamps:
            age = time.time() - self._cache_timestamps[cache_key]
            if age > self._cache_max_age:
                self._cache.pop(cache_key, None)
                self._cache_timestamps.pop(cache_key, None)
                return None
        
        return self._cache.get(cache_key)
    
    def _set_cache(self, cache_key: str, result: Any):
        """设置缓存"""
        # 限制缓存大小
        if len(self._cache) >= self.cache_size:
            # 删除最旧的缓存项
            oldest_key = min(self._cache_timestamps.keys(), 
                           key=lambda k: self._cache_timestamps[k])
            self._cache.pop(oldest_key, None)
            self._cache_timestamps.pop(oldest_key, None)
        
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
    
    def calculate_vectorized_drawdown(self, 
                                    portfolio_values: Union[np.ndarray, List[float]],
                                    use_cache: bool = True) -> Dict[str, Any]:
        """
        向量化计算回撤指标
        
        Args:
            portfolio_values: 投资组合净值序列
            use_cache: 是否使用缓存
            
        Returns:
            Dict包含各种回撤指标
        """
        # 输入验证
        values = np.asarray(portfolio_values, dtype=np.float64)
        
        if len(values) == 0:
            raise ValueError("输入数据不能为空")
        
        if len(values) < 2:
            raise ValueError("至少需要2个数据点进行回撤计算")
        
        # 处理NaN值
        if np.any(np.isnan(values)):
            logger.warning("输入数据包含NaN值，将进行清理")
            valid_mask = ~np.isnan(values)
            values = values[valid_mask]
            
            if len(values) < 2:
                raise ValueError("清理NaN后数据点不足")
        
        # 检查缓存
        if use_cache:
            cache_key = self._get_cache_key(values, "drawdown")
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        # 向量化计算
        result = self._compute_drawdown_vectorized(values)
        
        # 设置缓存
        if use_cache:
            self._set_cache(cache_key, result)
        
        return result
    
    def _compute_drawdown_vectorized(self, values: np.ndarray) -> Dict[str, Any]:
        """向量化计算回撤指标的核心实现"""
        # 计算累积最大值（峰值序列）
        running_max = np.maximum.accumulate(values)
        
        # 计算回撤序列
        drawdown_series = (values - running_max) / running_max
        
        # 计算当前回撤
        current_drawdown = drawdown_series[-1]
        
        # 计算最大回撤
        max_drawdown = np.min(drawdown_series)
        
        # 计算水下曲线（累积回撤时间）
        underwater_curve = self._calculate_underwater_curve(drawdown_series)
        
        # 计算恢复期间
        recovery_periods = self._calculate_recovery_periods(drawdown_series)
        
        # 计算回撤统计
        drawdown_stats = self._calculate_drawdown_statistics(drawdown_series)
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'drawdown_series': drawdown_series,
            'underwater_curve': underwater_curve,
            'recovery_periods': recovery_periods,
            'peak_indices': np.where(values == running_max)[0],
            'trough_index': np.argmin(drawdown_series),
            **drawdown_stats
        }
    
    def _calculate_underwater_curve(self, drawdown_series: np.ndarray) -> np.ndarray:
        """计算水下曲线（连续回撤时间）"""
        underwater = np.zeros_like(drawdown_series)
        current_underwater = 0
        
        for i, dd in enumerate(drawdown_series):
            if dd < -1e-10:  # 在回撤中
                current_underwater += 1
            else:  # 恢复到峰值
                current_underwater = 0
            underwater[i] = current_underwater
        
        return underwater
    
    def _calculate_recovery_periods(self, drawdown_series: np.ndarray) -> List[int]:
        """计算各次回撤的恢复期间"""
        recovery_periods = []
        in_drawdown = False
        drawdown_start = 0
        
        for i, dd in enumerate(drawdown_series):
            if dd < -1e-10 and not in_drawdown:
                # 开始回撤
                in_drawdown = True
                drawdown_start = i
            elif dd >= -1e-10 and in_drawdown:
                # 恢复到峰值
                recovery_period = i - drawdown_start
                recovery_periods.append(recovery_period)
                in_drawdown = False
        
        # 如果结束时仍在回撤中
        if in_drawdown:
            recovery_period = len(drawdown_series) - 1 - drawdown_start
            recovery_periods.append(recovery_period)
        
        return recovery_periods
    
    def _calculate_drawdown_statistics(self, drawdown_series: np.ndarray) -> Dict[str, float]:
        """计算回撤统计指标"""
        # 过滤出显著回撤（绝对值大于0.1%）
        significant_drawdowns = drawdown_series[drawdown_series < -0.001]
        
        if len(significant_drawdowns) == 0:
            return {
                'average_drawdown': 0.0,
                'drawdown_volatility': 0.0,
                'drawdown_skewness': 0.0,
                'drawdown_frequency': 0.0,
                'max_drawdown_duration': 0,
                'average_recovery_time': 0.0
            }
        
        # 基本统计
        avg_drawdown = np.mean(significant_drawdowns)
        dd_volatility = np.std(significant_drawdowns)
        
        # 偏度计算
        if dd_volatility > 0:
            dd_skewness = np.mean(((significant_drawdowns - avg_drawdown) / dd_volatility) ** 3)
        else:
            dd_skewness = 0.0
        
        # 回撤频率（每年）
        total_periods = len(drawdown_series)
        drawdown_periods = len(significant_drawdowns)
        dd_frequency = (drawdown_periods / total_periods) * 252  # 年化
        
        # 最大回撤持续时间
        underwater_curve = self._calculate_underwater_curve(drawdown_series)
        max_dd_duration = np.max(underwater_curve)
        
        # 平均恢复时间
        recovery_periods = self._calculate_recovery_periods(drawdown_series)
        avg_recovery_time = np.mean(recovery_periods) if recovery_periods else 0.0
        
        return {
            'average_drawdown': avg_drawdown,
            'drawdown_volatility': dd_volatility,
            'drawdown_skewness': dd_skewness,
            'drawdown_frequency': dd_frequency,
            'max_drawdown_duration': int(max_dd_duration),
            'average_recovery_time': avg_recovery_time
        }
    
    def calculate_batch_portfolio_metrics(self, 
                                        returns: np.ndarray,
                                        weights: np.ndarray,
                                        risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        批量计算投资组合指标
        
        Args:
            returns: 资产收益率矩阵 (T, N)
            weights: 投资组合权重 (N,)
            risk_free_rate: 无风险利率
            
        Returns:
            Dict包含各种投资组合指标
        """
        # 输入验证
        returns = np.asarray(returns)
        weights = np.asarray(weights)
        
        if returns.ndim != 2:
            raise ValueError("收益率矩阵应为2维")
        
        if len(weights) != returns.shape[1]:
            raise ValueError("权重维度与资产数量不匹配")
        
        if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
            logger.warning("权重总和不为1，将进行标准化")
            weights = weights / np.sum(weights)
        
        # 计算投资组合收益率
        portfolio_returns = np.dot(returns, weights)
        
        # 基本指标
        portfolio_return = np.mean(portfolio_returns) * 252  # 年化收益率
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)  # 年化波动率
        
        # 风险调整指标
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        # 回撤指标
        portfolio_values = np.cumprod(1 + portfolio_returns)
        drawdown_result = self.calculate_vectorized_drawdown(portfolio_values, use_cache=False)
        max_drawdown = drawdown_result['max_drawdown']
        
        # Calmar比率
        calmar_ratio = portfolio_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        # Sortino比率
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (portfolio_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0.0
        
        # VaR和CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        # 信息比率（相对于等权重基准）
        equal_weight_returns = np.mean(returns, axis=1)
        excess_returns = portfolio_returns - equal_weight_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0.0
        
        return {
            'portfolio_return': portfolio_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'downside_volatility': downside_volatility,
            'win_rate': np.sum(portfolio_returns > 0) / len(portfolio_returns),
            'average_win': np.mean(portfolio_returns[portfolio_returns > 0]) if np.any(portfolio_returns > 0) else 0.0,
            'average_loss': np.mean(portfolio_returns[portfolio_returns < 0]) if np.any(portfolio_returns < 0) else 0.0
        }
    
    def calculate_parallel_risk_metrics(self, 
                                      returns: np.ndarray,
                                      weight_sets: List[np.ndarray],
                                      n_jobs: int = None) -> List[Dict[str, Any]]:
        """
        并行计算多个投资组合的风险指标
        
        Args:
            returns: 资产收益率矩阵
            weight_sets: 权重组合列表
            n_jobs: 并行作业数
            
        Returns:
            风险指标结果列表
        """
        if not self.enable_parallel:
            # 串行计算
            results = []
            for i, weights in enumerate(weight_sets):
                metrics = self.calculate_batch_portfolio_metrics(returns, weights)
                results.append({
                    'portfolio_id': i,
                    'weights': weights,
                    'metrics': metrics
                })
            return results
        
        n_jobs = n_jobs or self.n_jobs
        
        # 定义计算函数
        def compute_single_portfolio(args):
            """计算单个投资组合的指标"""
            portfolio_id, weights, returns_data = args
            try:
                metrics = self.calculate_batch_portfolio_metrics(returns_data, weights)
                return {
                    'portfolio_id': portfolio_id,
                    'weights': weights,
                    'metrics': metrics,
                    'success': True
                }
            except Exception as e:
                logger.error(f"计算投资组合 {portfolio_id} 失败: {e}")
                return {
                    'portfolio_id': portfolio_id,
                    'weights': weights,
                    'error': str(e),
                    'success': False
                }
        
        # 准备参数
        args_list = [(i, weights, returns) for i, weights in enumerate(weight_sets)]
        
        # 并行计算
        if len(weight_sets) < 4:
            # 数据量小时使用线程池
            with ThreadPoolExecutor(max_workers=min(n_jobs, len(weight_sets))) as executor:
                results = list(executor.map(compute_single_portfolio, args_list))
        else:
            # 数据量大时使用进程池
            with ProcessPoolExecutor(max_workers=min(n_jobs, len(weight_sets))) as executor:
                results = list(executor.map(compute_single_portfolio, args_list))
        
        # 过滤成功的结果
        successful_results = [r for r in results if r.get('success', False)]
        
        if len(successful_results) < len(weight_sets):
            logger.warning(f"部分投资组合计算失败: {len(weight_sets) - len(successful_results)} 个")
        
        return successful_results
    
    def calculate_rolling_metrics(self, 
                                portfolio_values: np.ndarray,
                                window_size: int,
                                step_size: int = 1) -> Dict[str, np.ndarray]:
        """
        计算滚动窗口指标
        
        Args:
            portfolio_values: 投资组合净值序列
            window_size: 滚动窗口大小
            step_size: 步长
            
        Returns:
            Dict包含各种滚动指标
        """
        values = np.asarray(portfolio_values)
        
        if len(values) < window_size:
            raise ValueError(f"数据长度 {len(values)} 小于窗口大小 {window_size}")
        
        # 计算滚动收益率
        returns = np.diff(values) / values[:-1]
        
        # 预分配结果数组
        n_windows = (len(values) - window_size) // step_size + 1
        rolling_returns = np.zeros(n_windows)
        rolling_volatility = np.zeros(n_windows)
        rolling_sharpe = np.zeros(n_windows)
        rolling_drawdown = np.zeros(n_windows)
        rolling_var = np.zeros(n_windows)
        
        # 向量化计算滚动指标
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            window_values = values[start_idx:end_idx]
            window_returns = returns[start_idx:end_idx-1] if start_idx < len(returns) else returns[start_idx-1:end_idx-1]
            
            # 滚动收益率
            rolling_returns[i] = np.mean(window_returns)
            
            # 滚动波动率
            rolling_volatility[i] = np.std(window_returns)
            
            # 滚动夏普比率
            if rolling_volatility[i] > 0:
                rolling_sharpe[i] = rolling_returns[i] / rolling_volatility[i]
            else:
                rolling_sharpe[i] = 0.0
            
            # 滚动最大回撤
            dd_result = self.calculate_vectorized_drawdown(window_values, use_cache=False)
            rolling_drawdown[i] = dd_result['max_drawdown']
            
            # 滚动VaR
            rolling_var[i] = np.percentile(window_returns, 5)
        
        return {
            'rolling_returns': rolling_returns,
            'rolling_volatility': rolling_volatility,
            'rolling_sharpe': rolling_sharpe,
            'rolling_drawdown': rolling_drawdown,
            'rolling_var': rolling_var,
            'window_dates': np.arange(window_size-1, len(values), step_size)[:n_windows]
        }
    
    def calculate_correlation_matrix(self, 
                                   returns: np.ndarray,
                                   method: str = 'pearson') -> Dict[str, Any]:
        """
        高效计算相关性矩阵及其特征
        
        Args:
            returns: 收益率矩阵 (T, N)
            method: 相关性方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict包含相关性矩阵和相关统计
        """
        returns = np.asarray(returns)
        
        if returns.ndim != 2:
            raise ValueError("收益率矩阵应为2维")
        
        # 处理缺失值
        if np.any(np.isnan(returns)):
            logger.warning("收益率数据包含NaN，将使用成对完整观测计算相关性")
        
        # 计算相关性矩阵
        if method == 'pearson':
            # 使用NumPy的高效实现
            corr_matrix = np.corrcoef(returns.T)
        else:
            # 对于其他方法，转换为pandas计算
            df = pd.DataFrame(returns)
            corr_matrix = df.corr(method=method).values
        
        # 处理可能的NaN
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # 确保对角线为1
        np.fill_diagonal(corr_matrix, 1.0)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues = np.real(eigenvalues)  # 确保实数
        
        # 计算矩阵特征
        condition_number = np.max(eigenvalues) / np.max(eigenvalues[eigenvalues > 1e-10])
        effective_rank = np.sum(eigenvalues > 1e-10)
        
        # 计算分散化比率
        n_assets = returns.shape[1]
        avg_correlation = (np.sum(corr_matrix) - n_assets) / (n_assets * (n_assets - 1))
        diversification_ratio = (1 + (n_assets - 1) * avg_correlation) ** 0.5
        
        return {
            'correlation_matrix': corr_matrix,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'condition_number': condition_number,
            'effective_rank': effective_rank,
            'average_correlation': avg_correlation,
            'diversification_ratio': diversification_ratio,
            'max_correlation': np.max(corr_matrix[~np.eye(n_assets, dtype=bool)]),
            'min_correlation': np.min(corr_matrix[~np.eye(n_assets, dtype=bool)])
        }
    
    def calculate_risk_attribution(self, 
                                 returns: np.ndarray,
                                 weights: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算风险归因
        
        Args:
            returns: 收益率矩阵 (T, N)
            weights: 投资组合权重 (N,)
            
        Returns:
            Dict包含各种风险归因指标
        """
        returns = np.asarray(returns)
        weights = np.asarray(weights)
        
        # 计算协方差矩阵
        cov_matrix = np.cov(returns.T)
        
        # 计算投资组合方差
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 计算边际风险贡献
        marginal_contributions = np.dot(cov_matrix, weights) / portfolio_volatility
        
        # 计算组合风险贡献
        component_contributions = weights * marginal_contributions
        
        # 计算百分比贡献
        percentage_contributions = component_contributions / portfolio_variance
        
        # 验证贡献度总和
        total_contribution = np.sum(component_contributions)
        if not np.isclose(total_contribution, portfolio_variance, rtol=1e-10):
            logger.warning(f"风险贡献总和验证失败: {total_contribution} vs {portfolio_variance}")
        
        return {
            'component_contributions': component_contributions,
            'marginal_contributions': marginal_contributions,
            'percentage_contributions': percentage_contributions,
            'portfolio_variance': portfolio_variance,
            'portfolio_volatility': portfolio_volatility,
            'covariance_matrix': cov_matrix
        }
    
    def calculate_chunk_metrics(self, chunk_data: np.ndarray) -> Dict[str, Any]:
        """
        计算数据块的指标（用于内存优化）
        
        Args:
            chunk_data: 数据块
            
        Returns:
            块指标字典
        """
        if chunk_data.size == 0:
            return {'chunk_size': 0}
        
        # 基本统计
        mean_return = np.mean(chunk_data, axis=0) if chunk_data.ndim > 1 else np.mean(chunk_data)
        volatility = np.std(chunk_data, axis=0) if chunk_data.ndim > 1 else np.std(chunk_data)
        
        # 风险指标
        if chunk_data.ndim == 1:
            var_95 = np.percentile(chunk_data, 5)
            skewness = self._calculate_skewness(chunk_data)
            kurtosis = self._calculate_kurtosis(chunk_data)
        else:
            var_95 = np.percentile(chunk_data, 5, axis=0)
            skewness = np.apply_along_axis(self._calculate_skewness, 0, chunk_data)
            kurtosis = np.apply_along_axis(self._calculate_kurtosis, 0, chunk_data)
        
        return {
            'chunk_size': chunk_data.shape[0],
            'mean_return': mean_return,
            'volatility': volatility,
            'var_95': var_95,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'min_value': np.min(chunk_data),
            'max_value': np.max(chunk_data)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # 超额峰度
        return kurtosis
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        current_time = time.time()
        
        # 计算缓存命中率（简化实现）
        valid_cache_count = 0
        expired_cache_count = 0
        
        for cache_key in self._cache_timestamps:
            age = current_time - self._cache_timestamps[cache_key]
            if age <= self._cache_max_age:
                valid_cache_count += 1
            else:
                expired_cache_count += 1
        
        return {
            'cache_size': len(self._cache),
            'valid_entries': valid_cache_count,
            'expired_entries': expired_cache_count,
            'memory_usage_mb': self._estimate_cache_memory(),
            'max_age_hours': self._cache_max_age / 3600
        }
    
    def _estimate_cache_memory(self) -> float:
        """估算缓存内存使用量（MB）"""
        try:
            import sys
            total_size = 0
            for value in self._cache.values():
                total_size += sys.getsizeof(value)
            return total_size / (1024 * 1024)
        except ImportError:
            return 0.0