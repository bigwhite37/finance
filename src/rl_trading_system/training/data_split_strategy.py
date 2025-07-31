"""
数据划分策略实现
实现时序数据的训练/验证/测试划分，支持滚动窗口和数据泄露防护
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """数据划分配置"""
    train_ratio: float = 0.7
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    min_train_samples: int = 100
    min_validation_samples: int = 30
    min_test_samples: int = 10
    gap_days: int = 0  # 划分间隔天数
    rolling_window_size: Optional[int] = None  # 滚动窗口大小
    step_size: Optional[int] = None  # 滚动步长
    train_end_date: Optional[str] = None  # 训练结束日期
    validation_end_date: Optional[str] = None  # 验证结束日期
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """配置验证"""
        # 检查比例和
        if abs(self.train_ratio + self.validation_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("比例之和必须为1")
        
        # 检查比例非负
        if any(ratio < 0 for ratio in [self.train_ratio, self.validation_ratio, self.test_ratio]):
            raise ValueError("比例不能为负数")
        
        # 检查最小样本数
        if any(samples <= 0 for samples in [self.min_train_samples, self.min_validation_samples, self.min_test_samples]):
            raise ValueError("最小样本数必须为正数")
        
        # 检查间隔天数
        if self.gap_days < 0:
            raise ValueError("间隔天数不能为负数")


@dataclass
class SplitResult:
    """数据划分结果"""
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    split_dates: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """结果验证"""
        # 检查索引重叠
        if len(np.intersect1d(self.train_indices, self.validation_indices)) > 0:
            raise ValueError("训练和验证索引不能重叠")
        
        if len(np.intersect1d(self.validation_indices, self.test_indices)) > 0:
            raise ValueError("验证和测试索引不能重叠")
        
        if len(np.intersect1d(self.train_indices, self.test_indices)) > 0:
            raise ValueError("训练和测试索引不能重叠")
    
    def get_metrics(self) -> Dict[str, float]:
        """获取划分统计指标"""
        train_size = len(self.train_indices)
        val_size = len(self.validation_indices)
        test_size = len(self.test_indices)
        total_size = train_size + val_size + test_size
        
        return {
            'train_size': train_size,
            'validation_size': val_size,
            'test_size': test_size,
            'total_size': total_size,
            'train_ratio': train_size / total_size if total_size > 0 else 0,
            'validation_ratio': val_size / total_size if total_size > 0 else 0,
            'test_ratio': test_size / total_size if total_size > 0 else 0
        }


class DataSplitStrategy(ABC):
    """数据划分策略抽象基类"""
    
    def __init__(self, config: SplitConfig):
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    @abstractmethod
    def split(self, data: pd.DataFrame) -> SplitResult:
        """
        执行数据划分
        
        Args:
            data: 待划分的数据，必须有datetime索引
            
        Returns:
            SplitResult: 划分结果
        """
        pass
    
    def _validate_data(self, data: pd.DataFrame):
        """验证输入数据"""
        if data.empty:
            raise ValueError("输入数据不能为空")
        
        # 检查是否有datetime索引
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("数据必须有MultiIndex，包含datetime")
        
        if 'datetime' not in data.index.names:
            raise ValueError("数据索引必须包含datetime")
    
    def _get_datetime_index(self, data: pd.DataFrame) -> pd.DatetimeIndex:
        """获取datetime索引"""
        return data.index.get_level_values('datetime')
    
    def detect_temporal_leakage(self, result: SplitResult, data: pd.DataFrame) -> bool:
        """
        检测时间泄露
        
        Args:
            result: 划分结果
            data: 原始数据
            
        Returns:
            bool: 是否检测到泄露
        """
        datetime_index = self._get_datetime_index(data)
        
        # 检查训练和验证的时间顺序
        train_dates = datetime_index[result.train_indices]
        val_dates = datetime_index[result.validation_indices]
        test_dates = datetime_index[result.test_indices]
        
        # 训练数据的最大日期应该小于验证数据的最小日期
        if len(train_dates) > 0 and len(val_dates) > 0:
            if train_dates.max() >= val_dates.min():
                logger.warning("检测到训练和验证数据的时间泄露")
                return True
        
        # 验证数据的最大日期应该小于测试数据的最小日期
        if len(val_dates) > 0 and len(test_dates) > 0:
            if val_dates.max() >= test_dates.min():
                logger.warning("检测到验证和测试数据的时间泄露")
                return True
        
        return False
    
    def detect_feature_leakage(self, result: SplitResult, data: pd.DataFrame, 
                             feature_columns: List[str]) -> bool:
        """
        检测特征泄露（使用未来信息）
        
        Args:
            result: 划分结果
            data: 原始数据
            feature_columns: 要检查的特征列
            
        Returns:
            bool: 是否检测到泄露
        """
        for feature in feature_columns:
            if feature not in data.columns:
                continue
            
            # 检查整个数据集中的NaN模式
            full_feature = data[feature]
            
            # 检查特征名称模式，识别可能的泄露特征
            if any(pattern in feature.lower() for pattern in ['future', 'shift', 'lag', 'lead']):
                if 'future' in feature.lower() or 'shift' in feature.lower() or 'lead' in feature.lower():
                    logger.warning(f"特征 {feature} 名称暗示可能使用了未来信息")
                    return True
            
            # 检查NaN模式：如果末尾有连续NaN而开头没有，可能是前向shift
            if len(full_feature) > 20:
                # 检查末尾20%的数据
                tail_size = int(len(full_feature) * 0.2)
                tail_values = full_feature.tail(tail_size)
                head_values = full_feature.head(tail_size)
                
                tail_nan_ratio = tail_values.isna().sum() / len(tail_values)
                head_nan_ratio = head_values.isna().sum() / len(head_values)
                
                # 如果末尾NaN比例明显高于开头，可能是前向shift
                # 检查是否有明显的不对称NaN分布
                if tail_nan_ratio > 0.05 and head_nan_ratio == 0:
                    logger.warning(f"特征 {feature} 末尾NaN比例过高 ({tail_nan_ratio:.2f})，开头无NaN，疑似使用了未来信息")
                    return True
                elif tail_nan_ratio > 0.3 and head_nan_ratio < 0.1:
                    logger.warning(f"特征 {feature} 末尾NaN比例过高 ({tail_nan_ratio:.2f})，疑似使用了未来信息")
                    return True
            
            # 检查训练集中的特征值
            train_feature = data[feature].iloc[result.train_indices]
            
            # 如果整个训练集的NaN比例过高
            if len(train_feature) > 0:
                nan_ratio = train_feature.isna().sum() / len(train_feature)
                if nan_ratio > 0.5:  # 超过50%的数据为NaN
                    logger.warning(f"特征 {feature} 在训练集中有大量NaN值 ({nan_ratio:.2f})，可能使用了未来信息")
                    return True
        
        return False
    
    def detect_target_leakage(self, result: SplitResult, data: pd.DataFrame,
                            target_column: str, feature_columns: List[str]) -> bool:
        """
        检测目标变量泄露
        
        Args:
            result: 划分结果
            data: 原始数据
            target_column: 目标变量列名
            feature_columns: 特征列名
            
        Returns:
            bool: 是否检测到泄露
        """
        if target_column not in data.columns:
            return False
        
        train_data = data.iloc[result.train_indices]
        target_data = train_data[target_column]
        
        for feature in feature_columns:
            if feature not in data.columns:
                continue
            
            feature_data = train_data[feature]
            
            # 首先检查特征是否是目标的简单偏移
            # 检查特征名称是否暗示使用了未来目标（如target_lag, target_shift等）
            if 'target' in feature.lower() and ('lag' in feature.lower() or 'shift' in feature.lower()):
                logger.warning(f"特征 {feature} 名称暗示可能使用了目标变量的未来值")
                return True
            
            # 检查特征与目标的相关性
            valid_mask = ~(feature_data.isna() | target_data.isna())
            if valid_mask.sum() > 10:  # 需要足够的有效数据
                valid_feature = feature_data[valid_mask]
                valid_target = target_data[valid_mask]
                
                if len(valid_feature) > 1 and len(valid_target) > 1:
                    correlation = stats.pearsonr(valid_feature, valid_target)[0]
                    
                    if abs(correlation) > 0.95:  # 异常高的相关性
                        logger.warning(f"特征 {feature} 与目标变量相关性异常高 ({correlation:.3f})，可能存在泄露")
                        return True
        
        return False


class TimeSeriesSplitStrategy(DataSplitStrategy):
    """时序数据划分策略"""
    
    def split(self, data: pd.DataFrame) -> SplitResult:
        """
        按时间顺序划分数据
        
        Args:
            data: 待划分的时序数据
            
        Returns:
            SplitResult: 划分结果
        """
        self._validate_data(data)
        
        # 获取唯一的时间点并排序
        datetime_index = self._get_datetime_index(data)
        unique_dates = datetime_index.unique().sort_values()
        
        if len(unique_dates) < (self.config.min_train_samples + 
                               self.config.min_validation_samples + 
                               self.config.min_test_samples):
            raise ValueError("数据量不足以满足最小样本数要求")
        
        # 计算划分点
        total_dates = len(unique_dates)
        
        # 考虑间隔天数
        effective_dates = total_dates - 2 * self.config.gap_days
        if effective_dates <= 0:
            raise ValueError("间隔天数过大，无法进行有效划分")
        
        train_size = int(effective_dates * self.config.train_ratio)
        val_size = int(effective_dates * self.config.validation_ratio)
        
        # 确保满足最小样本数要求
        train_size = max(train_size, self.config.min_train_samples)
        val_size = max(val_size, self.config.min_validation_samples)
        
        # 计算日期边界
        train_end_idx = train_size
        val_start_idx = train_end_idx + self.config.gap_days
        val_end_idx = val_start_idx + val_size
        test_start_idx = val_end_idx + self.config.gap_days
        
        if test_start_idx >= total_dates:
            raise ValueError("配置参数导致无法生成测试集")
        
        # 获取对应的日期
        train_dates = unique_dates[:train_end_idx]
        val_dates = unique_dates[val_start_idx:val_end_idx]
        test_dates = unique_dates[test_start_idx:]
        
        # 转换为索引
        train_indices = self._dates_to_indices(data, train_dates)
        val_indices = self._dates_to_indices(data, val_dates)
        test_indices = self._dates_to_indices(data, test_dates)
        
        # 构建日期字典
        split_dates = {
            'train_start': train_dates[0].strftime('%Y-%m-%d') if len(train_dates) > 0 else '',
            'train_end': train_dates[-1].strftime('%Y-%m-%d') if len(train_dates) > 0 else '',
            'val_start': val_dates[0].strftime('%Y-%m-%d') if len(val_dates) > 0 else '',
            'val_end': val_dates[-1].strftime('%Y-%m-%d') if len(val_dates) > 0 else '',
            'test_start': test_dates[0].strftime('%Y-%m-%d') if len(test_dates) > 0 else '',
            'test_end': test_dates[-1].strftime('%Y-%m-%d') if len(test_dates) > 0 else ''
        }
        
        result = SplitResult(
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices,
            split_dates=split_dates,
            metadata={'strategy': 'time_series', 'gap_days': self.config.gap_days}
        )
        
        # 检测泄露
        if self.detect_temporal_leakage(result, data):
            logger.warning("检测到时间泄露，请检查数据和配置")
        
        return result
    
    def _dates_to_indices(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
        """将日期转换为数据索引"""
        datetime_index = self._get_datetime_index(data)
        mask = datetime_index.isin(dates)
        return np.where(mask)[0]


class FixedSplitStrategy(DataSplitStrategy):
    """固定划分策略"""
    
    def split(self, data: pd.DataFrame) -> SplitResult:
        """
        按固定日期或比例划分数据
        
        Args:
            data: 待划分的时序数据
            
        Returns:
            SplitResult: 划分结果
        """
        self._validate_data(data)
        
        datetime_index = self._get_datetime_index(data)
        
        if self.config.train_end_date and self.config.validation_end_date:
            # 按日期划分
            return self._split_by_dates(data, datetime_index)
        else:
            # 按比例划分
            return self._split_by_ratios(data, datetime_index)
    
    def _split_by_dates(self, data: pd.DataFrame, datetime_index: pd.DatetimeIndex) -> SplitResult:
        """按日期划分"""
        train_end = pd.Timestamp(self.config.train_end_date)
        val_end = pd.Timestamp(self.config.validation_end_date)
        
        # 创建掩码
        train_mask = datetime_index <= train_end
        val_mask = (datetime_index > train_end) & (datetime_index <= val_end)
        test_mask = datetime_index > val_end
        
        # 应用间隔
        if self.config.gap_days > 0:
            gap_delta = timedelta(days=self.config.gap_days)
            
            # 调整验证集开始时间
            val_start = train_end + gap_delta
            val_mask = (datetime_index >= val_start) & (datetime_index <= val_end)
            
            # 调整测试集开始时间
            test_start = val_end + gap_delta
            test_mask = datetime_index >= test_start
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        split_dates = {
            'train_start': datetime_index[train_indices[0]].strftime('%Y-%m-%d') if len(train_indices) > 0 else '',
            'train_end': self.config.train_end_date,
            'val_start': datetime_index[val_indices[0]].strftime('%Y-%m-%d') if len(val_indices) > 0 else '',
            'val_end': self.config.validation_end_date,
            'test_start': datetime_index[test_indices[0]].strftime('%Y-%m-%d') if len(test_indices) > 0 else '',
            'test_end': datetime_index[test_indices[-1]].strftime('%Y-%m-%d') if len(test_indices) > 0 else ''
        }
        
        return SplitResult(
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices,
            split_dates=split_dates,
            metadata={'strategy': 'fixed_dates'}
        )
    
    def _split_by_ratios(self, data: pd.DataFrame, datetime_index: pd.DatetimeIndex) -> SplitResult:
        """按比例划分"""
        total_samples = len(data)
        
        train_size = int(total_samples * self.config.train_ratio)
        val_size = int(total_samples * self.config.validation_ratio)
        
        # 考虑间隔
        gap_samples = int(self.config.gap_days * len(datetime_index.unique()) / 
                         (datetime_index.max() - datetime_index.min()).days)
        
        train_indices = np.arange(train_size)
        val_start = train_size + gap_samples
        val_indices = np.arange(val_start, val_start + val_size)
        test_start = val_start + val_size + gap_samples
        test_indices = np.arange(test_start, total_samples)
        
        # 确保索引有效
        val_indices = val_indices[val_indices < total_samples]
        test_indices = test_indices[test_indices < total_samples]
        
        split_dates = {
            'train_start': datetime_index[train_indices[0]].strftime('%Y-%m-%d') if len(train_indices) > 0 else '',
            'train_end': datetime_index[train_indices[-1]].strftime('%Y-%m-%d') if len(train_indices) > 0 else '',
            'val_start': datetime_index[val_indices[0]].strftime('%Y-%m-%d') if len(val_indices) > 0 else '',
            'val_end': datetime_index[val_indices[-1]].strftime('%Y-%m-%d') if len(val_indices) > 0 else '',
            'test_start': datetime_index[test_indices[0]].strftime('%Y-%m-%d') if len(test_indices) > 0 else '',
            'test_end': datetime_index[test_indices[-1]].strftime('%Y-%m-%d') if len(test_indices) > 0 else ''
        }
        
        return SplitResult(
            train_indices=train_indices,
            validation_indices=val_indices,
            test_indices=test_indices,
            split_dates=split_dates,
            metadata={'strategy': 'fixed_ratios'}
        )


class RollingWindowSplitStrategy(DataSplitStrategy):
    """滚动窗口划分策略"""
    
    def split(self, data: pd.DataFrame) -> SplitResult:
        """
        单次划分（返回第一个窗口）
        
        Args:
            data: 待划分的时序数据
            
        Returns:
            SplitResult: 第一个窗口的划分结果
        """
        splits = self.split_rolling(data)
        return splits[0] if splits else SplitResult(
            train_indices=np.array([]),
            validation_indices=np.array([]),
            test_indices=np.array([])
        )
    
    def split_rolling(self, data: pd.DataFrame) -> List[SplitResult]:
        """
        滚动窗口划分
        
        Args:
            data: 待划分的时序数据
            
        Returns:
            List[SplitResult]: 多个窗口的划分结果
        """
        self._validate_data(data)
        
        if self.config.rolling_window_size is None:
            raise ValueError("滚动窗口大小必须指定")
        
        datetime_index = self._get_datetime_index(data)
        unique_dates = datetime_index.unique().sort_values()
        total_dates = len(unique_dates)
        
        window_size = self.config.rolling_window_size
        step_size = self.config.step_size or window_size // 4
        
        splits = []
        start_idx = 0
        
        while start_idx + window_size <= total_dates:
            end_idx = start_idx + window_size
            window_dates = unique_dates[start_idx:end_idx]
            
            # 在窗口内按比例划分
            window_size_actual = len(window_dates)
            train_size = int(window_size_actual * self.config.train_ratio)
            val_size = int(window_size_actual * self.config.validation_ratio)
            
            # 考虑间隔
            gap_dates = self.config.gap_days
            
            train_dates = window_dates[:train_size]
            val_start = train_size + gap_dates
            val_dates = window_dates[val_start:val_start + val_size]
            test_start = val_start + val_size + gap_dates
            test_dates = window_dates[test_start:]
            
            # 检查是否有足够的数据
            if len(train_dates) < self.config.min_train_samples:
                break
            if len(val_dates) < self.config.min_validation_samples:
                break
            if len(test_dates) < self.config.min_test_samples:
                break
            
            # 转换为索引
            train_indices = self._dates_to_indices(data, train_dates)
            val_indices = self._dates_to_indices(data, val_dates)
            test_indices = self._dates_to_indices(data, test_dates)
            
            split_dates = {
                'train_start': train_dates[0].strftime('%Y-%m-%d'),
                'train_end': train_dates[-1].strftime('%Y-%m-%d'),
                'val_start': val_dates[0].strftime('%Y-%m-%d') if len(val_dates) > 0 else '',
                'val_end': val_dates[-1].strftime('%Y-%m-%d') if len(val_dates) > 0 else '',
                'test_start': test_dates[0].strftime('%Y-%m-%d') if len(test_dates) > 0 else '',
                'test_end': test_dates[-1].strftime('%Y-%m-%d') if len(test_dates) > 0 else ''
            }
            
            result = SplitResult(
                train_indices=train_indices,
                validation_indices=val_indices,
                test_indices=test_indices,
                split_dates=split_dates,
                metadata={
                    'strategy': 'rolling_window',
                    'window_id': len(splits),
                    'window_start_date': window_dates[0].strftime('%Y-%m-%d'),
                    'window_end_date': window_dates[-1].strftime('%Y-%m-%d')
                }
            )
            
            splits.append(result)
            start_idx += step_size
        
        if not splits:
            raise ValueError("无法创建任何有效的滚动窗口")
        
        logger.info(f"创建了 {len(splits)} 个滚动窗口")
        return splits
    
    def _dates_to_indices(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
        """将日期转换为数据索引"""
        datetime_index = self._get_datetime_index(data)
        mask = datetime_index.isin(dates)
        return np.where(mask)[0]


def create_split_strategy(strategy_type: str, config: SplitConfig) -> DataSplitStrategy:
    """
    工厂函数：创建数据划分策略
    
    Args:
        strategy_type: 策略类型 ('time_series', 'fixed', 'rolling_window')
        config: 划分配置
        
    Returns:
        DataSplitStrategy: 对应的策略实例
    """
    strategies = {
        'time_series': TimeSeriesSplitStrategy,
        'fixed': FixedSplitStrategy,
        'rolling_window': RollingWindowSplitStrategy
    }
    
    if strategy_type not in strategies:
        raise ValueError(f"不支持的策略类型: {strategy_type}")
    
    return strategies[strategy_type](config)


def validate_split_quality(result: SplitResult, data: pd.DataFrame, 
                          target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    验证划分质量
    
    Args:
        result: 划分结果
        data: 原始数据
        target_column: 目标变量列名（可选）
        
    Returns:
        Dict: 质量评估结果
    """
    metrics = result.get_metrics()
    
    # 基本统计
    quality_report = {
        'basic_stats': metrics,
        'temporal_order_valid': True,
        'no_overlap': True,
        'size_balance': True,
        'warnings': []
    }
    
    # 检查时间顺序
    if 'datetime' in data.index.names:
        datetime_index = data.index.get_level_values('datetime')
        
        train_dates = datetime_index[result.train_indices]
        val_dates = datetime_index[result.validation_indices]
        test_dates = datetime_index[result.test_indices]
        
        if len(train_dates) > 0 and len(val_dates) > 0:
            if train_dates.max() >= val_dates.min():
                quality_report['temporal_order_valid'] = False
                quality_report['warnings'].append("训练集和验证集时间顺序错误")
        
        if len(val_dates) > 0 and len(test_dates) > 0:
            if val_dates.max() >= test_dates.min():
                quality_report['temporal_order_valid'] = False
                quality_report['warnings'].append("验证集和测试集时间顺序错误")
    
    # 检查集合大小平衡
    sizes = [metrics['train_size'], metrics['validation_size'], metrics['test_size']]
    if min(sizes) < max(sizes) * 0.05:  # 最小集合不到最大集合的5%
        quality_report['size_balance'] = False
        quality_report['warnings'].append("数据集大小严重不平衡")
    
    # 检查目标变量分布（如果提供）
    if target_column and target_column in data.columns:
        train_target = data[target_column].iloc[result.train_indices]
        val_target = data[target_column].iloc[result.validation_indices]
        test_target = data[target_column].iloc[result.test_indices]
        
        # 检查分布相似性（使用KS检验）
        if len(train_target) > 10 and len(val_target) > 10:
            ks_stat, p_value = stats.ks_2samp(train_target.dropna(), val_target.dropna())
            quality_report['train_val_distribution_similar'] = p_value > 0.05
            if p_value <= 0.05:
                quality_report['warnings'].append("训练集和验证集目标分布差异显著")
    
    return quality_report