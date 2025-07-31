"""
数据质量检查工具
实现数据质量检查和清洗功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        """初始化数据质量检查器"""
        self.quality_rules = {
            'price': self._get_price_quality_rules(),
            'fundamental': self._get_fundamental_quality_rules(),
            'general': self._get_general_quality_rules()
        }
    
    def _get_price_quality_rules(self) -> Dict[str, Any]:
        """获取价格数据质量规则"""
        return {
            'required_columns': ['open', 'high', 'low', 'close', 'volume', 'amount'],
            'numeric_columns': ['open', 'high', 'low', 'close', 'volume', 'amount'],
            'positive_columns': ['open', 'high', 'low', 'close', 'volume', 'amount'],
            'price_relations': [
                ('high', 'low', '>='),  # 最高价 >= 最低价
                ('high', 'open', '>='),  # 最高价 >= 开盘价
                ('high', 'close', '>='),  # 最高价 >= 收盘价
                ('low', 'open', '<='),   # 最低价 <= 开盘价
                ('low', 'close', '<=')   # 最低价 <= 收盘价
            ],
            'outlier_thresholds': {
                'price_change_ratio': 0.2,  # 单日涨跌幅超过20%视为异常
                'volume_change_ratio': 10.0,  # 成交量变化超过10倍视为异常
                'price_zscore': 3.0,  # 价格Z-score超过3视为异常
                'volume_zscore': 3.0   # 成交量Z-score超过3视为异常
            }
        }
    
    def _get_fundamental_quality_rules(self) -> Dict[str, Any]:
        """获取基本面数据质量规则"""
        return {
            'numeric_columns': ['PE', 'PB', 'PS', 'PCF', 'ROE', 'ROA'],
            'ratio_ranges': {
                'PE': (0, 1000),    # 市盈率合理范围
                'PB': (0, 100),     # 市净率合理范围
                'ROE': (-1, 1),     # ROE合理范围
                'ROA': (-1, 1)      # ROA合理范围
            }
        }
    
    def _get_general_quality_rules(self) -> Dict[str, Any]:
        """获取通用数据质量规则"""
        return {
            'max_missing_ratio': 0.1,  # 最大缺失值比例
            'min_data_points': 10,     # 最少数据点数量
            'duplicate_tolerance': 0.05  # 重复数据容忍度
        }
    
    def check_data_quality(self, df: pd.DataFrame, 
                          data_type: str = 'price') -> Dict[str, Any]:
        """
        检查数据质量
        
        Args:
            df: 待检查的数据
            data_type: 数据类型 ('price', 'fundamental', 'general')
            
        Returns:
            数据质量报告
        """
        if df.empty:
            return {
                'status': 'error',
                'score': 0.0,
                'issues': ['数据为空'],
                'warnings': [],
                'statistics': {}
            }
        
        issues = []
        warnings = []
        statistics = {}
        
        # 基本统计信息
        statistics.update(self._get_basic_statistics(df))
        
        # 通用检查
        general_issues, general_warnings = self._check_general_quality(df)
        issues.extend(general_issues)
        warnings.extend(general_warnings)
        
        # 特定类型检查
        if data_type in self.quality_rules:
            type_issues, type_warnings = self._check_type_specific_quality(df, data_type)
            issues.extend(type_issues)
            warnings.extend(type_warnings)
        
        # 计算质量分数
        score = self._calculate_quality_score(df, issues, warnings)
        
        # 确定状态
        if score >= 0.8:
            status = 'good'
        elif score >= 0.6:
            status = 'warning'
        else:
            status = 'error'
        
        return {
            'status': status,
            'score': score,
            'issues': issues,
            'warnings': warnings,
            'statistics': statistics
        }
    
    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取基本统计信息"""
        stats = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # 数值列统计
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            stats['numeric_summary'] = df[numeric_columns].describe().to_dict()
        
        return stats
    
    def _check_general_quality(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """检查通用数据质量"""
        issues = []
        warnings = []
        rules = self.quality_rules['general']
        
        # 检查数据量
        if len(df) < rules['min_data_points']:
            issues.append(f"数据点数量不足: {len(df)} < {rules['min_data_points']}")
        
        # 检查缺失值比例
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > rules['max_missing_ratio']:
            issues.append(f"缺失值比例过高: {missing_ratio:.2%} > {rules['max_missing_ratio']:.2%}")
        
        # 检查重复数据
        duplicate_count = df.duplicated().sum()
        duplicate_ratio = duplicate_count / len(df)
        if duplicate_ratio > rules['duplicate_tolerance']:
            warnings.append(f"重复数据比例: {duplicate_ratio:.2%}")
        
        return issues, warnings
    
    def _check_type_specific_quality(self, df: pd.DataFrame, 
                                   data_type: str) -> Tuple[List[str], List[str]]:
        """检查特定类型的数据质量"""
        issues = []
        warnings = []
        rules = self.quality_rules[data_type]
        
        if data_type == 'price':
            issues_p, warnings_p = self._check_price_quality(df, rules)
            issues.extend(issues_p)
            warnings.extend(warnings_p)
        elif data_type == 'fundamental':
            issues_f, warnings_f = self._check_fundamental_quality(df, rules)
            issues.extend(issues_f)
            warnings.extend(warnings_f)
        
        return issues, warnings
    
    def _check_price_quality(self, df: pd.DataFrame, 
                           rules: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """检查价格数据质量"""
        issues = []
        warnings = []
        
        # 检查必要列
        missing_columns = set(rules['required_columns']) - set(df.columns)
        if missing_columns:
            issues.append(f"缺少必要列: {missing_columns}")
            return issues, warnings
        
        # 检查数值类型
        for col in rules['numeric_columns']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"列{col}不是数值类型")
        
        # 检查正值
        for col in rules['positive_columns']:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"列{col}存在{negative_count}个负值")
        
        # 检查价格关系
        for col1, col2, operator in rules['price_relations']:
            if col1 in df.columns and col2 in df.columns:
                if operator == '>=':
                    violation_count = (df[col1] < df[col2]).sum()
                elif operator == '<=':
                    violation_count = (df[col1] > df[col2]).sum()
                else:
                    continue
                
                if violation_count > 0:
                    issues.append(f"价格关系违规: {col1} {operator} {col2}, "
                                f"违规数量: {violation_count}")
        
        # 检查异常值
        outlier_issues, outlier_warnings = self._check_price_outliers(df, rules['outlier_thresholds'])
        issues.extend(outlier_issues)
        warnings.extend(outlier_warnings)
        
        return issues, warnings
    
    def _check_fundamental_quality(self, df: pd.DataFrame, 
                                 rules: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """检查基本面数据质量"""
        issues = []
        warnings = []
        
        # 检查数值类型
        for col in rules['numeric_columns']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                warnings.append(f"基本面列{col}不是数值类型")
        
        # 检查比率范围
        for col, (min_val, max_val) in rules['ratio_ranges'].items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    warnings.append(f"列{col}存在{out_of_range}个超出合理范围的值")
        
        return issues, warnings
    
    def _check_price_outliers(self, df: pd.DataFrame, 
                            thresholds: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """检查价格异常值"""
        issues = []
        warnings = []
        
        if 'close' in df.columns:
            # 检查价格变化异常
            price_change = df['close'].pct_change().abs()
            extreme_changes = (price_change > thresholds['price_change_ratio']).sum()
            if extreme_changes > 0:
                warnings.append(f"存在{extreme_changes}个极端价格变化")
            
            # 检查价格Z-score异常
            price_zscore = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
            price_outliers = (price_zscore > thresholds['price_zscore']).sum()
            if price_outliers > 0:
                warnings.append(f"存在{price_outliers}个价格异常值")
        
        if 'volume' in df.columns:
            # 检查成交量变化异常
            volume_change = df['volume'].pct_change().abs()
            extreme_volume_changes = (volume_change > thresholds['volume_change_ratio']).sum()
            if extreme_volume_changes > 0:
                warnings.append(f"存在{extreme_volume_changes}个极端成交量变化")
            
            # 检查成交量Z-score异常
            volume_zscore = np.abs((df['volume'] - df['volume'].mean()) / df['volume'].std())
            volume_outliers = (volume_zscore > thresholds['volume_zscore']).sum()
            if volume_outliers > 0:
                warnings.append(f"存在{volume_outliers}个成交量异常值")
        
        return issues, warnings
    
    def _calculate_quality_score(self, df: pd.DataFrame, 
                               issues: List[str], warnings: List[str]) -> float:
        """计算数据质量分数"""
        base_score = 1.0
        
        # 根据问题数量扣分
        issue_penalty = len(issues) * 0.2
        warning_penalty = len(warnings) * 0.05
        
        # 根据缺失值比例扣分
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        missing_penalty = missing_ratio * 0.3
        
        # 计算最终分数
        final_score = max(0.0, base_score - issue_penalty - warning_penalty - missing_penalty)
        
        return final_score
    
    def clean_data(self, df: pd.DataFrame, 
                   data_type: str = 'price',
                   strategy: str = 'conservative') -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            df: 待清洗的数据
            data_type: 数据类型
            strategy: 清洗策略 ('conservative', 'aggressive')
            
        Returns:
            清洗后的数据
        """
        if df.empty:
            return df
        
        cleaned_df = df.copy()
        
        # 删除完全重复的行
        cleaned_df = cleaned_df.drop_duplicates()
        
        # 处理缺失值
        if strategy == 'conservative':
            # 保守策略：只删除全部为空的行
            cleaned_df = cleaned_df.dropna(how='all')
        elif strategy == 'aggressive':
            # 激进策略：删除任何包含空值的行
            cleaned_df = cleaned_df.dropna()
        
        # 特定类型的清洗
        if data_type == 'price':
            cleaned_df = self._clean_price_data(cleaned_df, strategy)
        elif data_type == 'fundamental':
            cleaned_df = self._clean_fundamental_data(cleaned_df, strategy)
        
        logger.info(f"数据清洗完成: {len(df)} -> {len(cleaned_df)} 行")
        
        return cleaned_df
    
    def _clean_price_data(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """清洗价格数据"""
        cleaned_df = df.copy()
        
        # 删除负价格
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in cleaned_df.columns:
                cleaned_df = cleaned_df[cleaned_df[col] > 0]
        
        # 删除成交量为负的记录
        if 'volume' in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df['volume'] >= 0]
        
        # 修正价格关系错误
        if all(col in cleaned_df.columns for col in ['high', 'low']):
            # 删除最高价低于最低价的记录
            cleaned_df = cleaned_df[cleaned_df['high'] >= cleaned_df['low']]
        
        # 处理极端异常值
        if strategy == 'aggressive':
            for col in price_columns:
                if col in cleaned_df.columns:
                    # 使用3σ规则删除异常值
                    mean_val = cleaned_df[col].mean()
                    std_val = cleaned_df[col].std()
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    cleaned_df = cleaned_df[
                        (cleaned_df[col] >= lower_bound) & 
                        (cleaned_df[col] <= upper_bound)
                    ]
        
        return cleaned_df
    
    def _clean_fundamental_data(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """清洗基本面数据"""
        cleaned_df = df.copy()
        
        # 处理极端比率值
        ratio_columns = ['PE', 'PB', 'PS', 'PCF']
        for col in ratio_columns:
            if col in cleaned_df.columns:
                # 删除负值和极大值
                cleaned_df = cleaned_df[
                    (cleaned_df[col] > 0) & 
                    (cleaned_df[col] < 1000)
                ]
        
        return cleaned_df


# 全局数据质量检查器实例
_global_quality_checker = None


def get_global_quality_checker() -> DataQualityChecker:
    """获取全局数据质量检查器实例"""
    global _global_quality_checker
    if _global_quality_checker is None:
        _global_quality_checker = DataQualityChecker()
    return _global_quality_checker