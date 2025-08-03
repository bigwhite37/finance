"""
贝叶斯优化器

独立的贝叶斯优化实现，支持参数调优。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

# 尝试导入scikit-optimize，如果不可用则提供替代实现
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
except ImportError:
    # 严格遵守规则6：无法获取必要依赖时立即抛出RuntimeError
    raise RuntimeError("scikit-optimize不可用：贝叶斯优化需要安装scikit-optimize，请运行 'pip install scikit-optimize'")

logger = logging.getLogger(__name__)


@dataclass
class BayesianOptimizationResult:
    """贝叶斯优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    convergence_history: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_evaluations': self.total_evaluations,
            'convergence_achieved': len(self.convergence_history) > 0
        }


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    使用高斯过程或随机森林进行参数优化。
    """
    
    def __init__(self, 
                 acquisition_function: str = 'EI',
                 random_state: Optional[int] = None):
        """
        初始化贝叶斯优化器
        
        Args:
            acquisition_function: 获取函数类型 ('EI', 'PI', 'UCB')
            random_state: 随机种子
        """
        self.acquisition_function = acquisition_function
        self.random_state = random_state
        
        
        logger.info(f"初始化贝叶斯优化器，获取函数: {acquisition_function}")
    
    def optimize(self, 
                 objective_function: Callable,
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 n_calls: int = 50,
                 n_initial_points: Optional[int] = None) -> BayesianOptimizationResult:
        """
        执行贝叶斯优化
        
        Args:
            objective_function: 目标函数，接受参数字典返回分数
            parameter_bounds: 参数边界字典
            n_calls: 优化调用次数
            n_initial_points: 初始随机点数量
            
        Returns:
            优化结果
        """
        return self._optimize_with_skopt(
            objective_function, parameter_bounds, n_calls, n_initial_points
        )
    
    def _optimize_with_skopt(self, 
                           objective_function: Callable,
                           parameter_bounds: Dict[str, Tuple[float, float]],
                           n_calls: int,
                           n_initial_points: Optional[int]) -> BayesianOptimizationResult:
        """使用scikit-optimize进行优化"""
        # 创建搜索空间
        dimensions = []
        param_names = list(parameter_bounds.keys())
        
        for param_name, (low, high) in parameter_bounds.items():
            dimensions.append(Real(low, high, name=param_name))
        
        # 存储优化历史
        optimization_history = []
        convergence_history = []
        
        @use_named_args(dimensions)
        def wrapped_objective(**params):
            """包装的目标函数"""
            try:
                # 调用用户定义的目标函数
                score = objective_function(params)
                
                # 记录历史
                optimization_history.append({
                    'params': params.copy(),
                    'score': score
                })
                
                # 记录收敛历史
                if optimization_history:
                    best_so_far = max(entry['score'] for entry in optimization_history)
                    convergence_history.append(best_so_far)
                
                logger.info(f"评估参数 {params}: 分数 = {score:.4f}")
                
                # scikit-optimize进行最小化，所以返回负分数
                return -score
                
            except Exception as e:
                logger.error(f"目标函数评估失败 {params}: {e}")
                return 999999  # 返回很大的值表示失败
        
        # 设置默认初始点数量
        if n_initial_points is None:
            n_initial_points = min(10, n_calls // 2)
        
        try:
            # 执行优化
            result = gp_minimize(
                func=wrapped_objective,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func=self.acquisition_function,
                random_state=self.random_state
            )
            
            # 提取最佳参数
            best_params = dict(zip(param_names, result.x))
            best_score = -result.fun  # 转换回正分数
            
            logger.info(f"贝叶斯优化完成，最佳分数: {best_score:.4f}")
            
            return BayesianOptimizationResult(
                best_params=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                total_evaluations=len(optimization_history),
                convergence_history=convergence_history
            )
            
        except Exception as e:
            logger.error(f"贝叶斯优化执行失败: {e}")
            raise RuntimeError(f"贝叶斯优化执行失败: {e}")
    
    def create_convergence_plot(self, 
                               result: BayesianOptimizationResult,
                               output_path: Optional[str] = None):
        """创建收敛图"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(result.convergence_history, 'b-', linewidth=2)
            plt.title('贝叶斯优化收敛过程')
            plt.xlabel('评估次数')
            plt.ylabel('最佳分数')
            plt.grid(True, alpha=0.3)
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"收敛图已保存到: {output_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib不可用，无法生成收敛图")
        except Exception as e:
            logger.error(f"生成收敛图失败: {e}")
    
    def analyze_parameter_importance(self, 
                                   result: BayesianOptimizationResult) -> pd.DataFrame:
        """分析参数重要性"""
        if not result.optimization_history:
            return pd.DataFrame()
        
        # 提取参数和分数
        params_df = pd.DataFrame([entry['params'] for entry in result.optimization_history])
        scores = [entry['score'] for entry in result.optimization_history]
        
        # 计算每个参数与分数的相关性
        correlations = []
        for param_name in params_df.columns:
            correlation = np.corrcoef(params_df[param_name], scores)[0, 1]
            correlations.append({
                '参数名称': param_name,
                '相关系数': correlation,
                '重要性等级': self._classify_importance(abs(correlation))
            })
        
        # 按重要性排序
        importance_df = pd.DataFrame(correlations)
        importance_df = importance_df.sort_values('相关系数', key=abs, ascending=False)
        
        return importance_df
    
    def _classify_importance(self, abs_correlation: float) -> str:
        """分类重要性等级"""
        if abs_correlation >= 0.7:
            return '高'
        elif abs_correlation >= 0.4:
            return '中'
        elif abs_correlation >= 0.2:
            return '低'
        else:
            return '很低'
    
    def suggest_next_parameters(self, 
                               result: BayesianOptimizationResult,
                               parameter_bounds: Dict[str, Tuple[float, float]],
                               n_suggestions: int = 3) -> List[Dict[str, Any]]:
        """基于优化历史建议下一组参数"""
        if not result.optimization_history:
            return []
        
        # 简化实现：基于最佳参数附近的随机采样
        best_params = result.best_params
        suggestions = []
        
        # 设置随机种子
        if self.random_state is not None:
            np.random.seed(self.random_state + 1000)
        
        for _ in range(n_suggestions):
            suggested_params = {}
            
            for param_name, best_value in best_params.items():
                low, high = parameter_bounds[param_name]
                
                # 在最佳值附近进行高斯采样
                param_range = high - low
                std = param_range * 0.1  # 标准差为范围的10%
                
                suggested_value = np.random.normal(best_value, std)
                suggested_value = np.clip(suggested_value, low, high)
                
                suggested_params[param_name] = suggested_value
            
            suggestions.append(suggested_params)
        
        return suggestions