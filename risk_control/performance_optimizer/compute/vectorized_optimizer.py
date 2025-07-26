"""
向量化计算优化器模块

提供高性能的向量化计算函数，替代循环操作，优化数值计算性能。
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from functools import wraps

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
    PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    PANDAS_AVAILABLE = False
    # 创建占位符类以避免导入错误
    class pd:
        DataFrame = None
        Series = None
    class np:
        sqrt = lambda x: x ** 0.5
        nan = None
        inf = float('inf')
        isnan = lambda x: x != x
        isinf = lambda x: x == float('inf') or x == float('-inf')


def performance_timer(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if execution_time > 0.1:  # 记录超过100ms的操作
            logging.debug(f"向量化操作 {func.__name__} 执行时间: {execution_time:.3f}s")
        
        return result
    return wrapper


class VectorizedOptimizer:
    """向量化计算优化器
    
    提供高性能的向量化计算函数，替代循环操作。
    """
    
    def __init__(self):
        """初始化向量化优化器"""
        self._performance_stats = {
            'operations_count': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'speedup_ratio': 0.0
        }
    
    @performance_timer
    def vectorized_rolling_volatility(self, returns: pd.DataFrame, 
                                    windows: List[int]) -> Dict[int, pd.DataFrame]:
        """向量化滚动波动率计算
        
        Args:
            returns: 收益率数据 (日期 x 股票)
            windows: 滚动窗口长度列表
            
        Returns:
            不同窗口的滚动波动率字典
        """
        if returns.empty:
            return {window: pd.DataFrame() for window in windows}
        
        results = {}
        
        # 使用pandas的向量化操作
        for window in windows:
            if window <= 0:
                logging.warning(f"无效的窗口大小: {window}")
                results[window] = pd.DataFrame(index=returns.index, columns=returns.columns)
                continue
            
            # 计算滚动标准差并年化
            min_periods = max(1, window // 2)
            rolling_std = returns.rolling(window=window, min_periods=min_periods).std()
            results[window] = rolling_std * np.sqrt(252)
        
        self._update_stats()
        return results
    
    @performance_timer
    def vectorized_percentile_ranks(self, data: pd.DataFrame, 
                                  axis: int = 1) -> pd.DataFrame:
        """向量化分位数排名计算
        
        Args:
            data: 输入数据
            axis: 计算轴 (0=按列, 1=按行)
            
        Returns:
            分位数排名数据
        """
        if data.empty:
            return data.copy()
        
        # 使用pandas的rank函数进行向量化排名
        result = data.rank(axis=axis, pct=True, method='min')
        
        self._update_stats()
        return result
    
    @performance_timer
    def vectorized_factor_regression(self, returns: pd.DataFrame,
                                   factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """向量化因子回归计算
        
        Args:
            returns: 股票收益率数据 (日期 x 股票)
            factors: 因子数据 (日期 x 因子)
            
        Returns:
            (回归系数, 残差)
        """
        if returns.empty or factors.empty:
            empty_betas = pd.DataFrame(index=returns.columns, columns=factors.columns)
            empty_residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
            return empty_betas, empty_residuals
        
        # 确保数据对齐
        common_dates = returns.index.intersection(factors.index)
        if len(common_dates) == 0:
            logging.warning("收益率和因子数据没有共同的日期")
            empty_betas = pd.DataFrame(index=returns.columns, columns=factors.columns)
            empty_residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
            return empty_betas, empty_residuals
        
        aligned_returns = returns.loc[common_dates]
        aligned_factors = factors.loc[common_dates]
        
        # 使用numpy的向量化线性代数运算
        X = aligned_factors.values
        Y = aligned_returns.values
        
        # 检查数据有效性
        if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
            logging.warning("数据包含NaN值，将进行清理")
            # 删除包含NaN的行
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
            X = X[valid_mask]
            Y = Y[valid_mask]
            common_dates = common_dates[valid_mask]
        
        if X.shape[0] < X.shape[1] + 1:  # 样本数少于参数数
            logging.warning("样本数不足以进行回归")
            empty_betas = pd.DataFrame(np.nan, index=returns.columns, columns=factors.columns)
            empty_residuals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
            return empty_betas, empty_residuals
        
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # 批量计算回归系数: beta = (X'X)^(-1)X'Y
        try:
            XTX = X_with_intercept.T @ X_with_intercept
            XTY = X_with_intercept.T @ Y
            
            # 使用更稳定的求解方法
            betas = np.linalg.solve(XTX, XTY)
            
            # 计算残差
            Y_pred = X_with_intercept @ betas
            residuals = Y - Y_pred
            
            # 转换回DataFrame格式
            beta_df = pd.DataFrame(
                betas[1:].T,  # 排除截距项
                index=aligned_returns.columns,
                columns=aligned_factors.columns
            )
            
            residuals_df = pd.DataFrame(
                residuals,
                index=common_dates,
                columns=aligned_returns.columns
            )
            
            # 如果原始数据有更多日期，用NaN填充
            if len(returns.index) > len(common_dates):
                full_residuals = pd.DataFrame(
                    np.nan,
                    index=returns.index,
                    columns=returns.columns
                )
                full_residuals.loc[common_dates] = residuals_df
                residuals_df = full_residuals
            
            self._update_stats()
            return beta_df, residuals_df
            
        except (np.linalg.LinAlgError, np.linalg.linalg.LinAlgError):
            logging.warning("矩阵奇异，无法进行回归")
            # 如果矩阵不可逆，返回NaN结果
            beta_df = pd.DataFrame(
                np.nan,
                index=aligned_returns.columns,
                columns=aligned_factors.columns
            )
            residuals_df = pd.DataFrame(
                np.nan,
                index=returns.index,
                columns=returns.columns
            )
            
            self._update_stats()
            return beta_df, residuals_df
    
    @performance_timer
    def vectorized_volatility_decomposition(self, residuals: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """向量化波动率分解
        
        Args:
            residuals: 残差数据 (日期 x 股票)
            
        Returns:
            (好波动序列, 坏波动序列)
        """
        if residuals.empty:
            empty_series = pd.Series(index=residuals.columns, dtype=float)
            return empty_series, empty_series
        
        # 分离正负残差
        positive_residuals = residuals.where(residuals > 0, np.nan)
        negative_residuals = residuals.where(residuals < 0, np.nan)
        
        # 向量化计算标准差
        good_volatility = positive_residuals.std() * np.sqrt(252)
        bad_volatility = negative_residuals.std().abs() * np.sqrt(252)
        
        # 填充缺失值
        good_volatility = good_volatility.fillna(0.0)
        bad_volatility = bad_volatility.fillna(0.0)
        
        self._update_stats()
        return good_volatility, bad_volatility
    
    @performance_timer
    def vectorized_garch_batch_prediction(self, returns_dict: Dict[str, pd.Series],
                                        horizon: int = 5) -> Dict[str, float]:
        """批量GARCH预测（简化版本）
        
        Args:
            returns_dict: 股票收益率字典 {股票代码: 收益率序列}
            horizon: 预测期限
            
        Returns:
            预测波动率字典 {股票代码: 预测波动率}
        """
        if not returns_dict:
            return {}
        
        predictions = {}
        lambda_param = 0.94  # RiskMetrics标准参数
        
        for stock_code, returns_series in returns_dict.items():
            try:
                if returns_series.empty or len(returns_series) < 10:
                    logging.warning(f"股票 {stock_code} 数据不足")
                    predictions[stock_code] = 0.0
                    continue
                
                # 简化的GARCH预测：使用EWMA模型
                # 这里使用指数加权移动平均作为GARCH的快速近似
                
                # 计算平方收益率
                squared_returns = returns_series ** 2
                
                # 去除异常值
                squared_returns = squared_returns.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(squared_returns) < 5:
                    logging.warning(f"股票 {stock_code} 有效数据不足")
                    predictions[stock_code] = returns_series.std() * np.sqrt(252)
                    continue
                
                # EWMA波动率预测
                ewma_var = squared_returns.ewm(alpha=1-lambda_param, adjust=False).mean()
                predicted_vol = np.sqrt(ewma_var.iloc[-1] * 252)  # 年化
                
                # 检查结果有效性
                if np.isnan(predicted_vol) or np.isinf(predicted_vol):
                    predicted_vol = returns_series.std() * np.sqrt(252)
                
                predictions[stock_code] = float(predicted_vol)
                
            except Exception as e:
                logging.warning(f"股票 {stock_code} GARCH预测失败: {e}")
                # 预测失败时使用历史波动率
                try:
                    historical_vol = returns_series.std() * np.sqrt(252)
                    predictions[stock_code] = float(historical_vol) if not np.isnan(historical_vol) else 0.0
                except:
                    predictions[stock_code] = 0.0
        
        self._update_stats()
        return predictions
    
    @performance_timer
    def vectorized_correlation_matrix(self, returns: pd.DataFrame, 
                                    method: str = 'pearson') -> pd.DataFrame:
        """向量化相关系数矩阵计算
        
        Args:
            returns: 收益率数据 (日期 x 股票)
            method: 相关系数方法 ('pearson', 'spearman', 'kendall')
            
        Returns:
            相关系数矩阵
        """
        if returns.empty:
            return pd.DataFrame()
        
        # 使用pandas的向量化相关系数计算
        correlation_matrix = returns.corr(method=method)
        
        # 处理NaN值
        correlation_matrix = correlation_matrix.fillna(0.0)
        
        self._update_stats()
        return correlation_matrix
    
    @performance_timer
    def vectorized_rolling_correlation(self, x: pd.Series, y: pd.Series, 
                                     window: int) -> pd.Series:
        """向量化滚动相关系数计算
        
        Args:
            x: 第一个时间序列
            y: 第二个时间序列
            window: 滚动窗口大小
            
        Returns:
            滚动相关系数序列
        """
        if x.empty or y.empty or window <= 0:
            return pd.Series(index=x.index, dtype=float)
        
        # 确保数据对齐
        aligned_data = pd.DataFrame({'x': x, 'y': y}).dropna()
        
        if len(aligned_data) < window:
            return pd.Series(index=x.index, dtype=float)
        
        # 计算滚动相关系数
        rolling_corr = aligned_data['x'].rolling(window=window).corr(aligned_data['y'])
        
        # 重新索引到原始索引
        result = pd.Series(index=x.index, dtype=float)
        result.loc[rolling_corr.index] = rolling_corr
        
        self._update_stats()
        return result
    
    @performance_timer
    def vectorized_exponential_smoothing(self, data: pd.DataFrame, 
                                       alpha: float = 0.3) -> pd.DataFrame:
        """向量化指数平滑
        
        Args:
            data: 输入数据
            alpha: 平滑参数
            
        Returns:
            平滑后的数据
        """
        if data.empty or not (0 < alpha <= 1):
            return data.copy()
        
        # 使用pandas的ewm方法进行指数加权移动平均
        smoothed_data = data.ewm(alpha=alpha, adjust=False).mean()
        
        self._update_stats()
        return smoothed_data
    
    def _update_stats(self):
        """更新性能统计"""
        self._performance_stats['operations_count'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息
        
        Returns:
            性能统计信息字典
        """
        return self._performance_stats.copy()
    
    def reset_stats(self):
        """重置性能统计"""
        self._performance_stats = {
            'operations_count': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0,
            'speedup_ratio': 0.0
        }


# 为了向后兼容，保留原有的类名
VectorizedComputeOptimizer = VectorizedOptimizer