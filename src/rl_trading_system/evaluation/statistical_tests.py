"""
统计显著性检验模块

提供各种统计检验方法来验证策略改进的显著性。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """统计检验结果数据类"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    interpretation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'confidence_level': self.confidence_level,
            'effect_size': self.effect_size,
            'interpretation': self.interpretation
        }


class SignificanceTest:
    """
    统计显著性检验类
    
    提供多种统计检验方法来验证策略改进的显著性。
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化统计检验器
        
        Args:
            confidence_level: 置信水平，默认0.95
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
        logger.info(f"初始化统计检验器，置信水平: {confidence_level}")
    
    def t_test(self, 
               returns_a: pd.Series, 
               returns_b: pd.Series,
               alternative: str = 'two-sided') -> StatisticalTestResult:
        """
        执行双样本t检验
        
        Args:
            returns_a: 策略A的收益序列
            returns_b: 策略B的收益序列
            alternative: 备择假设 ('two-sided', 'less', 'greater')
            
        Returns:
            统计检验结果
        """
        # 检查数据有效性
        if len(returns_a) == 0 or len(returns_b) == 0:
            raise ValueError("收益序列不能为空")
        
        if returns_a.isnull().all() or returns_b.isnull().all():
            raise ValueError("收益序列不能全为NaN")
        
        # 移除NaN值
        returns_a_clean = returns_a.dropna()
        returns_b_clean = returns_b.dropna()
        
        if len(returns_a_clean) < 2 or len(returns_b_clean) < 2:
            raise ValueError("每个收益序列至少需要2个有效观测值")
        
        try:
            # 执行独立样本t检验
            statistic, p_value = stats.ttest_ind(
                returns_a_clean, returns_b_clean, 
                alternative=alternative, equal_var=False  # Welch's t-test
            )
            
            is_significant = p_value < self.alpha
            
            # 计算效应大小（Cohen's d）
            pooled_std = np.sqrt(
                ((len(returns_a_clean) - 1) * returns_a_clean.var() + 
                 (len(returns_b_clean) - 1) * returns_b_clean.var()) / 
                (len(returns_a_clean) + len(returns_b_clean) - 2)
            )
            
            if pooled_std > 0:
                cohens_d = (returns_a_clean.mean() - returns_b_clean.mean()) / pooled_std
            else:
                cohens_d = 0.0
            
            # 生成解释
            interpretation = self._interpret_t_test_result(
                statistic, p_value, is_significant, cohens_d, alternative
            )
            
            return StatisticalTestResult(
                test_name="独立样本t检验",
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                effect_size=cohens_d,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"t检验执行失败: {e}")
            raise RuntimeError(f"t检验执行失败: {e}")
    
    def mann_whitney_test(self, 
                         returns_a: pd.Series, 
                         returns_b: pd.Series,
                         alternative: str = 'two-sided') -> StatisticalTestResult:
        """
        执行Mann-Whitney U检验（非参数检验）
        
        Args:
            returns_a: 策略A的收益序列
            returns_b: 策略B的收益序列
            alternative: 备择假设
            
        Returns:
            统计检验结果
        """
        # 检查数据有效性
        if len(returns_a) == 0 or len(returns_b) == 0:
            raise ValueError("收益序列不能为空")
        
        # 移除NaN值
        returns_a_clean = returns_a.dropna()
        returns_b_clean = returns_b.dropna()
        
        if len(returns_a_clean) < 1 or len(returns_b_clean) < 1:
            raise ValueError("每个收益序列至少需要1个有效观测值")
        
        try:
            # 执行Mann-Whitney U检验
            statistic, p_value = stats.mannwhitneyu(
                returns_a_clean, returns_b_clean,
                alternative=alternative
            )
            
            is_significant = p_value < self.alpha
            
            # 计算效应大小（r = Z / sqrt(N)）
            n1, n2 = len(returns_a_clean), len(returns_b_clean)
            z_score = stats.norm.ppf(1 - p_value/2) if alternative == 'two-sided' else stats.norm.ppf(1 - p_value)
            effect_size = abs(z_score) / np.sqrt(n1 + n2)
            
            # 生成解释
            interpretation = self._interpret_mann_whitney_result(
                statistic, p_value, is_significant, effect_size, alternative
            )
            
            return StatisticalTestResult(
                test_name="Mann-Whitney U检验",
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                effect_size=effect_size,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Mann-Whitney U检验执行失败: {e}")
            raise RuntimeError(f"Mann-Whitney U检验执行失败: {e}")
    
    def paired_t_test(self, 
                     returns_a: pd.Series, 
                     returns_b: pd.Series) -> StatisticalTestResult:
        """
        执行配对t检验
        
        Args:
            returns_a: 策略A的收益序列
            returns_b: 策略B的收益序列（必须同长度）
            
        Returns:
            统计检验结果
        """
        if len(returns_a) != len(returns_b):
            raise ValueError("配对t检验要求两个序列长度相同")
        
        # 创建配对数据
        paired_data = pd.DataFrame({'A': returns_a, 'B': returns_b}).dropna()
        
        if len(paired_data) < 2:
            raise ValueError("配对t检验至少需要2对有效观测值")
        
        try:
            # 计算差值
            differences = paired_data['A'] - paired_data['B']
            
            # 执行单样本t检验（检验差值是否为0）
            statistic, p_value = stats.ttest_1samp(differences, 0)
            
            is_significant = p_value < self.alpha
            
            # 计算效应大小
            cohens_d = differences.mean() / differences.std() if differences.std() > 0 else 0.0
            
            # 生成解释
            interpretation = self._interpret_paired_t_test_result(
                statistic, p_value, is_significant, cohens_d, differences.mean()
            )
            
            return StatisticalTestResult(
                test_name="配对t检验",
                statistic=statistic,
                p_value=p_value,
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                effect_size=cohens_d,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"配对t检验执行失败: {e}")
            raise RuntimeError(f"配对t检验执行失败: {e}")
    
    def bootstrap_test(self, 
                      returns_a: pd.Series, 
                      returns_b: pd.Series,
                      metric_func: callable,
                      n_bootstrap: int = 1000) -> StatisticalTestResult:
        """
        执行Bootstrap检验
        
        Args:
            returns_a: 策略A的收益序列
            returns_b: 策略B的收益序列
            metric_func: 指标计算函数
            n_bootstrap: Bootstrap采样次数
            
        Returns:
            统计检验结果
        """
        # 移除NaN值
        returns_a_clean = returns_a.dropna()
        returns_b_clean = returns_b.dropna()
        
        if len(returns_a_clean) < 2 or len(returns_b_clean) < 2:
            raise ValueError("每个收益序列至少需要2个有效观测值")
        
        try:
            # 计算原始指标差异
            original_diff = metric_func(returns_a_clean) - metric_func(returns_b_clean)
            
            # Bootstrap采样
            bootstrap_diffs = []
            rng = np.random.RandomState(42)  # 设置随机种子以保证可重复性
            
            for _ in range(n_bootstrap):
                # 重采样
                sample_a = rng.choice(returns_a_clean, size=len(returns_a_clean), replace=True)
                sample_b = rng.choice(returns_b_clean, size=len(returns_b_clean), replace=True)
                
                # 计算指标差异
                diff = metric_func(pd.Series(sample_a)) - metric_func(pd.Series(sample_b))
                bootstrap_diffs.append(diff)
            
            bootstrap_diffs = np.array(bootstrap_diffs)
            
            # 计算p值（双侧检验）
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(original_diff))
            
            is_significant = p_value < self.alpha
            
            # 计算置信区间
            ci_lower = np.percentile(bootstrap_diffs, (self.alpha/2) * 100)
            ci_upper = np.percentile(bootstrap_diffs, (1 - self.alpha/2) * 100)
            
            # 生成解释
            interpretation = (
                f"Bootstrap检验结果：原始差异为{original_diff:.6f}，"
                f"{self.confidence_level*100:.0f}%置信区间为[{ci_lower:.6f}, {ci_upper:.6f}]。"
                f"{'差异' if is_significant else '无显著差异'}（p={p_value:.4f}）"
            )
            
            return StatisticalTestResult(
                test_name="Bootstrap检验",
                statistic=original_diff,
                p_value=p_value,
                is_significant=is_significant,
                confidence_level=self.confidence_level,
                effect_size=original_diff,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Bootstrap检验执行失败: {e}")
            raise RuntimeError(f"Bootstrap检验执行失败: {e}")
    
    def multiple_comparison_correction(self, 
                                     p_values: List[float],
                                     method: str = 'bonferroni') -> List[float]:
        """
        多重比较校正
        
        Args:
            p_values: p值列表
            method: 校正方法（'bonferroni', 'holm'）
            
        Returns:
            校正后的p值
        """
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni校正
            corrected_p = p_values * n
            return np.minimum(corrected_p, 1.0).tolist()
        
        elif method == 'holm':
            # Holm-Bonferroni校正
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected_p = np.zeros_like(p_values)
            
            for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
                corrected_p[idx] = min(p * (n - i), 1.0)
            
            # 确保单调性
            for i in range(1, n):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                corrected_p[idx] = max(corrected_p[idx], corrected_p[prev_idx])
            
            return corrected_p.tolist()
        
        else:
            raise ValueError(f"不支持的校正方法: {method}")
    
    def _interpret_t_test_result(self, 
                                statistic: float, 
                                p_value: float, 
                                is_significant: bool,
                                cohens_d: float,
                                alternative: str) -> str:
        """解释t检验结果"""
        direction = ""
        if alternative == 'greater':
            direction = "策略A显著优于策略B" if is_significant else "策略A不显著优于策略B"
        elif alternative == 'less':
            direction = "策略A显著劣于策略B" if is_significant else "策略A不显著劣于策略B"
        else:
            direction = "两策略存在显著差异" if is_significant else "两策略无显著差异"
        
        effect_interpretation = ""
        if abs(cohens_d) < 0.2:
            effect_interpretation = "效应很小"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "效应中等"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "效应较大"
        else:
            effect_interpretation = "效应很大"
        
        return (f"t检验结果：{direction}（t={statistic:.4f}, p={p_value:.4f}）。"
                f"效应大小Cohen's d={cohens_d:.4f}（{effect_interpretation}）")
    
    def _interpret_mann_whitney_result(self, 
                                     statistic: float, 
                                     p_value: float, 
                                     is_significant: bool,
                                     effect_size: float,
                                     alternative: str) -> str:
        """解释Mann-Whitney检验结果"""
        direction = ""
        if alternative == 'greater':
            direction = "策略A显著优于策略B" if is_significant else "策略A不显著优于策略B"
        elif alternative == 'less':
            direction = "策略A显著劣于策略B" if is_significant else "策略A不显著劣于策略B"
        else:
            direction = "两策略存在显著差异" if is_significant else "两策略无显著差异"
        
        return (f"Mann-Whitney U检验结果：{direction}（U={statistic:.0f}, p={p_value:.4f}）。"
                f"效应大小r={effect_size:.4f}")
    
    def _interpret_paired_t_test_result(self, 
                                       statistic: float, 
                                       p_value: float, 
                                       is_significant: bool,
                                       cohens_d: float,
                                       mean_diff: float) -> str:
        """解释配对t检验结果"""
        direction = "策略A显著优于策略B" if mean_diff > 0 and is_significant else \
                   "策略A显著劣于策略B" if mean_diff < 0 and is_significant else \
                   "两策略无显著差异"
        
        return (f"配对t检验结果：{direction}（t={statistic:.4f}, p={p_value:.4f}）。"
                f"平均差异={mean_diff:.6f}，效应大小Cohen's d={cohens_d:.4f}")
    
    def comprehensive_comparison(self, 
                               returns_a: pd.Series, 
                               returns_b: pd.Series,
                               strategy_names: Tuple[str, str] = ("策略A", "策略B")) -> Dict[str, StatisticalTestResult]:
        """
        综合比较分析
        
        Args:
            returns_a: 策略A的收益序列
            returns_b: 策略B的收益序列
            strategy_names: 策略名称元组
            
        Returns:
            包含多种检验结果的字典
        """
        results = {}
        
        try:
            # t检验
            results['t_test'] = self.t_test(returns_a, returns_b)
        except Exception as e:
            logger.warning(f"t检验失败: {e}")
        
        try:
            # Mann-Whitney检验
            results['mann_whitney'] = self.mann_whitney_test(returns_a, returns_b)
        except Exception as e:
            logger.warning(f"Mann-Whitney检验失败: {e}")
        
        try:
            # Bootstrap检验（比较夏普比率）
            def sharpe_ratio(returns):
                if len(returns) < 2 or returns.std() == 0:
                    return 0
                return returns.mean() / returns.std() * np.sqrt(252)
            
            results['bootstrap_sharpe'] = self.bootstrap_test(returns_a, returns_b, sharpe_ratio)
        except Exception as e:
            logger.warning(f"Bootstrap检验失败: {e}")
        
        logger.info(f"完成{strategy_names[0]}与{strategy_names[1]}的综合比较分析")
        
        return results