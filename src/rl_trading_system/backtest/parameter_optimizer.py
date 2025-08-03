"""
参数优化器模块

提供网格搜索和贝叶斯优化等参数调优方法。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
except ImportError:
    # 严格遵守规则6：无法获取必要依赖时立即抛出RuntimeError
    raise RuntimeError("scikit-optimize不可用：参数优化需要安装scikit-optimize，请运行 'pip install scikit-optimize'")

from .drawdown_control_config import DrawdownControlConfig
from .enhanced_backtest_engine import EnhancedBacktestEngine, EnhancedBacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    best_params: Dict[str, Any]
    best_score: float
    best_result: EnhancedBacktestResult
    all_results: List[Tuple[Dict[str, Any], float, EnhancedBacktestResult]]
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_evaluations': self.total_evaluations,
            'optimization_time': self.optimization_time,
            'best_result_summary': {
                'annual_return': self.best_result.annual_return,
                'max_drawdown': self.best_result.max_drawdown,
                'sharpe_ratio': self.best_result.sharpe_ratio,
                'calmar_ratio': self.best_result.calmar_ratio
            }
        }


class ParameterGridSearch:
    """
    参数网格搜索优化器
    
    通过穷举搜索找到最优参数组合。
    """
    
    def __init__(self, 
                 backtest_engine: EnhancedBacktestEngine,
                 scoring_function: Optional[Callable] = None,
                 n_jobs: int = 1):
        """
        初始化网格搜索优化器
        
        Args:
            backtest_engine: 回测引擎
            scoring_function: 评分函数，接受EnhancedBacktestResult返回float
            n_jobs: 并行作业数
        """
        self.backtest_engine = backtest_engine
        self.scoring_function = scoring_function or self._default_scoring_function
        self.n_jobs = n_jobs
        
        logger.info(f"初始化参数网格搜索优化器，并行作业数: {n_jobs}")
    
    def _default_scoring_function(self, result: EnhancedBacktestResult) -> float:
        """默认评分函数：加权综合指标"""
        # 组合多个指标：夏普比率权重0.4，回撤改善权重0.3，卡尔玛比率权重0.3
        score = (0.4 * result.sharpe_ratio + 
                0.3 * (-result.max_drawdown) +  # 回撤越小越好
                0.3 * result.calmar_ratio)
        return score
    
    def optimize(self, 
                 parameter_grid: Dict[str, List[Any]],
                 agent,
                 start_date: str,
                 end_date: str,
                 max_evaluations: Optional[int] = None) -> OptimizationResult:
        """
        执行网格搜索优化
        
        Args:
            parameter_grid: 参数网格
            agent: SAC智能体
            start_date: 回测开始日期
            end_date: 回测结束日期
            max_evaluations: 最大评估次数
            
        Returns:
            优化结果
        """
        logger.info("开始网格搜索参数优化")
        
        # 生成所有参数组合
        param_combinations = self._generate_parameter_combinations(parameter_grid)
        
        if max_evaluations and len(param_combinations) > max_evaluations:
            # 随机采样
            import random
            param_combinations = random.sample(param_combinations, max_evaluations)
        
        logger.info(f"总共需要评估 {len(param_combinations)} 个参数组合")
        
        import time
        start_time = time.time()
        
        # 执行优化
        if self.n_jobs == 1:
            results = self._sequential_optimization(param_combinations, agent, start_date, end_date)
        else:
            results = self._parallel_optimization(param_combinations, agent, start_date, end_date)
        
        optimization_time = time.time() - start_time
        
        # 找到最佳结果
        best_params, best_score, best_result = max(results, key=lambda x: x[1])
        
        # 创建优化历史
        optimization_history = [
            {
                'params': params,
                'score': score,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio
            }
            for params, score, result in results
        ]
        
        logger.info(f"网格搜索完成，最佳分数: {best_score:.4f}, 用时: {optimization_time:.2f}秒")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=results,
            optimization_history=optimization_history,
            total_evaluations=len(results),
            optimization_time=optimization_time
        )
    
    def _generate_parameter_combinations(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """生成所有参数组合"""
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _sequential_optimization(self, 
                                param_combinations: List[Dict[str, Any]],
                                agent,
                                start_date: str,
                                end_date: str) -> List[Tuple[Dict[str, Any], float, EnhancedBacktestResult]]:
        """顺序执行优化"""
        results = []
        
        for i, params in enumerate(param_combinations):
            logger.info(f"评估参数组合 {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # 更新配置
                config = DrawdownControlConfig(**params)
                self.backtest_engine.drawdown_config = config
                self.backtest_engine._initialize_drawdown_control_components()
                
                # 运行回测
                result = self.backtest_engine.run_backtest(agent, start_date, end_date)
                
                # 计算评分
                score = self.scoring_function(result)
                
                results.append((params, score, result))
                
                logger.info(f"评分: {score:.4f}, 年化收益: {result.annual_return:.4f}, "
                           f"最大回撤: {result.max_drawdown:.4f}")
                
            except Exception as e:
                logger.error(f"评估参数组合失败 {params}: {e}")
                # 给失败的组合一个很低的分数
                results.append((params, -999999, None))
        
        return results
    
    def _parallel_optimization(self, 
                              param_combinations: List[Dict[str, Any]],
                              agent,
                              start_date: str,
                              end_date: str) -> List[Tuple[Dict[str, Any], float, EnhancedBacktestResult]]:
        """并行执行优化"""
        # 注意：由于序列化问题，并行执行可能有限制
        # 这里提供框架，实际使用时可能需要调整
        logger.warning("并行优化功能暂未完全实现，使用顺序执行")
        return self._sequential_optimization(param_combinations, agent, start_date, end_date)


class BayesianOptimizer:
    """
    贝叶斯优化器
    
    使用高斯过程优化参数。
    """
    
    def __init__(self, 
                 backtest_engine: EnhancedBacktestEngine,
                 scoring_function: Optional[Callable] = None):
        """
        初始化贝叶斯优化器
        
        Args:
            backtest_engine: 回测引擎
            scoring_function: 评分函数
        """
# scikit-optimize依赖已在模块导入时检查，如果不可用会在导入时抛出RuntimeError
        
        self.backtest_engine = backtest_engine
        self.scoring_function = scoring_function or self._default_scoring_function
        
        logger.info("初始化贝叶斯优化器")
    
    def _default_scoring_function(self, result: EnhancedBacktestResult) -> float:
        """默认评分函数"""
        if result is None:
            return -999999
        
        score = (0.4 * result.sharpe_ratio + 
                0.3 * (-result.max_drawdown) +
                0.3 * result.calmar_ratio)
        return score
    
    def optimize(self, 
                 parameter_bounds: Dict[str, Tuple[float, float]],
                 agent,
                 start_date: str,
                 end_date: str,
                 n_calls: int = 50,
                 random_state: Optional[int] = None) -> OptimizationResult:
        """
        执行贝叶斯优化
        
        Args:
            parameter_bounds: 参数边界字典
            agent: SAC智能体
            start_date: 回测开始日期
            end_date: 回测结束日期
            n_calls: 优化调用次数
            random_state: 随机种子
            
        Returns:
            优化结果
        """
        logger.info(f"开始贝叶斯优化，调用次数: {n_calls}")
        
        # 创建搜索空间
        dimensions = []
        param_names = []
        
        for param_name, (low, high) in parameter_bounds.items():
            dimensions.append(Real(low, high, name=param_name))
            param_names.append(param_name)
        
        # 存储优化历史
        optimization_history = []
        all_results = []
        
        @use_named_args(dimensions)
        def objective(**params):
            """目标函数"""
            logger.info(f"评估参数: {params}")
            
            try:
                # 更新配置
                config = DrawdownControlConfig(**params)
                self.backtest_engine.drawdown_config = config
                self.backtest_engine._initialize_drawdown_control_components()
                
                # 运行回测
                result = self.backtest_engine.run_backtest(agent, start_date, end_date)
                
                # 计算评分（注意：gp_minimize是最小化，所以返回负分数）
                score = self.scoring_function(result)
                
                # 记录历史
                optimization_history.append({
                    'params': params.copy(),
                    'score': score,
                    'annual_return': result.annual_return,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio
                })
                
                all_results.append((params.copy(), score, result))
                
                logger.info(f"评分: {score:.4f}, 年化收益: {result.annual_return:.4f}, "
                           f"最大回撤: {result.max_drawdown:.4f}")
                
                return -score  # 返回负分数用于最小化
                
            except Exception as e:
                logger.error(f"评估参数失败 {params}: {e}")
                return 999999  # 返回大的正数表示很差的结果
        
        import time
        start_time = time.time()
        
        # 执行贝叶斯优化
        try:
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_calls,
                random_state=random_state,
                acq_func='EI',  # Expected Improvement
                n_initial_points=min(10, n_calls // 2)
            )
            
            optimization_time = time.time() - start_time
            
            # 提取最佳参数
            best_params = dict(zip(param_names, result.x))
            best_score = -result.fun  # 转换回正分数
            
            # 找到最佳结果
            best_result = None
            for params, score, backtest_result in all_results:
                if abs(score - best_score) < 1e-6:
                    best_result = backtest_result
                    break
            
            logger.info(f"贝叶斯优化完成，最佳分数: {best_score:.4f}, 用时: {optimization_time:.2f}秒")
            
            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                best_result=best_result,
                all_results=all_results,
                optimization_history=optimization_history,
                total_evaluations=len(all_results),
                optimization_time=optimization_time
            )
            
        except Exception as e:
            logger.error(f"贝叶斯优化执行失败: {e}")
            raise RuntimeError(f"贝叶斯优化执行失败: {e}")
    
    def _default_scoring_function(self, result: EnhancedBacktestResult) -> float:
        """默认评分函数"""
        if result is None:
            return -999999
        
        score = (0.4 * result.sharpe_ratio + 
                0.3 * (-result.max_drawdown) +
                0.3 * result.calmar_ratio)
        return score


def save_optimization_results(result: OptimizationResult, output_path: str):
    """保存优化结果"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存优化结果摘要
    with open(output_path / 'optimization_result.json', 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    # 保存详细历史
    history_df = pd.DataFrame(result.optimization_history)
    history_df.to_csv(output_path / 'optimization_history.csv', index=False)
    
    # 保存最佳回测结果
    if result.best_result:
        best_result_dict = result.best_result.to_dict()
        
        # 处理不能序列化的对象
        if 'portfolio_values' in best_result_dict:
            del best_result_dict['portfolio_values']
        if 'drawdown_series' in best_result_dict:
            del best_result_dict['drawdown_series']
        if 'position_series' in best_result_dict:
            del best_result_dict['position_series']
        
        with open(output_path / 'best_backtest_result.json', 'w', encoding='utf-8') as f:
            json.dump(best_result_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"优化结果已保存到: {output_path}")