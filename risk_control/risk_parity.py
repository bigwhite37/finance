"""
风险平价优化器
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import scipy.optimize as opt
import logging

logger = logging.getLogger(__name__)


class RiskParityOptimizer:
    """风险平价权重优化"""
    
    def __init__(self, config: Dict):
        """
        初始化风险平价优化器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.method = config.get('rp_method', 'inverse_volatility')  # 'inverse_volatility', 'equal_risk_contribution'
        self.max_weight = config.get('rp_max_weight', 0.2)
        self.min_weight = config.get('rp_min_weight', 0.0)
        
    def optimize_weights(self, 
                        expected_returns: pd.Series,
                        cov_matrix: pd.DataFrame,
                        method: Optional[str] = None) -> pd.Series:
        """
        基于风险平价的权重优化
        
        Args:
            expected_returns: 预期收益率
            cov_matrix: 协方差矩阵
            method: 优化方法
            
        Returns:
            优化后的权重
        """
        if method is None:
            method = self.method
            
        n_assets = len(expected_returns)
        
        if method == 'inverse_volatility':
            weights = self._inverse_volatility_weights(cov_matrix)
        elif method == 'equal_risk_contribution':
            weights = self._equal_risk_contribution_weights(cov_matrix)
        elif method == 'risk_budgeting':
            weights = self._risk_budgeting_weights(cov_matrix)
        else:
            # 默认使用逆波动率权重
            weights = self._inverse_volatility_weights(cov_matrix)
        
        # 确保权重为Series格式
        if isinstance(weights, np.ndarray):
            weights = pd.Series(weights, index=expected_returns.index)
        
        return weights
    
    def _inverse_volatility_weights(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        逆波动率权重
        
        Args:
            cov_matrix: 协方差矩阵
            
        Returns:
            逆波动率权重
        """
        # 计算波动率
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # 逆波动率权重
        inv_vol_weights = 1 / volatilities
        inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
        
        return pd.Series(inv_vol_weights, index=cov_matrix.index)
    
    def _equal_risk_contribution_weights(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        等风险贡献权重
        
        Args:
            cov_matrix: 协方差矩阵
            
        Returns:
            等风险贡献权重
        """
        n_assets = len(cov_matrix)
        
        # 目标函数：最小化风险贡献差异
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib) ** 2)
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 权重和为1
        ]
        
        # 边界条件
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = opt.minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
        else:
            logger.warning("等风险贡献优化失败，使用逆波动率权重")
            weights = self._inverse_volatility_weights(cov_matrix).values
        
        return pd.Series(weights, index=cov_matrix.index)
    
    def _risk_budgeting_weights(self, 
                              cov_matrix: pd.DataFrame,
                              risk_budgets: Optional[np.ndarray] = None) -> pd.Series:
        """
        风险预算权重
        
        Args:
            cov_matrix: 协方差矩阵
            risk_budgets: 风险预算分配
            
        Returns:
            风险预算权重
        """
        n_assets = len(cov_matrix)
        
        if risk_budgets is None:
            risk_budgets = np.ones(n_assets) / n_assets
        
        # 目标函数
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            return np.sum((contrib - risk_budgets) ** 2)
        
        # 约束和边界
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = opt.minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
        else:
            logger.warning("风险预算优化失败，使用等权重")
            weights = np.ones(n_assets) / n_assets
        
        return pd.Series(weights, index=cov_matrix.index)
    
    def calculate_risk_contributions(self, 
                                   weights: np.ndarray,
                                   cov_matrix: pd.DataFrame) -> pd.Series:
        """
        计算风险贡献度
        
        Args:
            weights: 组合权重
            cov_matrix: 协方差矩阵
            
        Returns:
            各资产风险贡献度
        """
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        
        return pd.Series(risk_contrib, index=cov_matrix.index)
    
    def calculate_diversification_ratio(self, 
                                      weights: np.ndarray,
                                      cov_matrix: pd.DataFrame) -> float:
        """
        计算分散化比率
        
        Args:
            weights: 组合权重
            cov_matrix: 协方差矩阵
            
        Returns:
            分散化比率
        """
        # 加权平均波动率
        weighted_avg_vol = np.dot(weights, np.sqrt(np.diag(cov_matrix)))
        
        # 组合波动率
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        return weighted_avg_vol / portfolio_vol
    
    def optimize_maximum_diversification(self, cov_matrix: pd.DataFrame) -> pd.Series:
        """
        最大分散化权重优化
        
        Args:
            cov_matrix: 协方差矩阵
            
        Returns:
            最大分散化权重
        """
        n_assets = len(cov_matrix)
        
        # 目标函数：最大化分散化比率等价于最小化其倒数
        def objective(weights):
            weighted_avg_vol = np.dot(weights, np.sqrt(np.diag(cov_matrix)))
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -weighted_avg_vol / portfolio_vol  # 负号用于最大化
        
        # 约束和边界
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # 初始权重
        x0 = np.ones(n_assets) / n_assets
        
        # 优化
        result = opt.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
        else:
            logger.warning("最大分散化优化失败，使用等权重")
            weights = np.ones(n_assets) / n_assets
        
        return pd.Series(weights, index=cov_matrix.index)
    
    def hierarchical_risk_parity(self, 
                                cov_matrix: pd.DataFrame) -> pd.Series:
        """
        分层风险平价 (HRP)
        
        Args:
            cov_matrix: 协方差矩阵
            
        Returns:
            HRP权重
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # 将协方差矩阵转换为距离矩阵
        correlations = cov_matrix.div(np.outer(np.sqrt(np.diag(cov_matrix)), 
                                              np.sqrt(np.diag(cov_matrix))))
        distances = np.sqrt(0.5 * (1 - correlations))
        
        # 层次聚类
        linkage_matrix = linkage(squareform(distances.values), method='ward')
        
        # 递归分配权重
        def _recursive_bisection(items):
            if len(items) == 1:
                return {items[0]: 1.0}
            
            # 找到最优分割点
            n = len(items)
            mid = n // 2
            
            left_items = items[:mid]
            right_items = items[mid:]
            
            # 计算子组合的逆波动率权重
            left_vol = np.sqrt(np.diag(cov_matrix.loc[left_items, left_items]).mean())
            right_vol = np.sqrt(np.diag(cov_matrix.loc[right_items, right_items]).mean())
            
            # 分配权重
            total_vol = left_vol + right_vol
            left_weight = right_vol / total_vol
            right_weight = left_vol / total_vol
            
            # 递归
            left_weights = _recursive_bisection(left_items)
            right_weights = _recursive_bisection(right_items)
            
            # 合并权重
            weights = {}
            for item, w in left_weights.items():
                weights[item] = w * left_weight
            for item, w in right_weights.items():
                weights[item] = w * right_weight
                
            return weights
        
        # 获取聚类顺序
        items = list(cov_matrix.index)
        hrp_weights = _recursive_bisection(items)
        
        return pd.Series(hrp_weights, index=cov_matrix.index)