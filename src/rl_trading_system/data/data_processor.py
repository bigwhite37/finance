"""
数据预处理模块
实现完整的数据预处理流水线，包括数据清洗、特征计算、标准化和缓存
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

from .data_models import FeatureVector, MarketData
from .feature_engineer import FeatureEngineer
from .data_quality import DataQualityChecker, get_global_quality_checker
from .data_cache import DataCache, get_global_cache

logger = logging.getLogger(__name__)


class DataProcessor:
    """数据预处理器"""
    
    def __init__(self, 
                 feature_engineer: Optional[FeatureEngineer] = None,
                 quality_checker: Optional[DataQualityChecker] = None,
                 cache: Optional[DataCache] = None):
        """
        初始化数据预处理器
        
        Args:
            feature_engineer: 特征工程器实例
            quality_checker: 数据质量检查器实例
            cache: 数据缓存实例
        """
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.quality_checker = quality_checker or get_global_quality_checker()
        self.cache = cache or get_global_cache()
        
        # 默认配置
        self.config = {
            'clean_strategy': 'conservative',
            'missing_value_method': 'ffill',
            'outlier_treatment': 'clip',
            'normalize': False,
            'normalization_method': 'zscore',
            'calculate_features': True,
            'feature_selection': False,
            'cache_enabled': True,
            'parallel_processing': True,
            'max_workers': 4
        }
        
        # 处理统计信息
        self.stats = {
            'processed_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': []
        }
    
    def configure_pipeline(self, config: Dict[str, Any]):
        """
        配置预处理流水线
        
        Args:
            config: 配置字典
        """
        self.config.update(config)
        logger.info(f"数据预处理流水线配置已更新: {config}")
    
    def process_data(self, 
                    data: pd.DataFrame,
                    symbols: List[str],
                    data_type: str = 'price',
                    **kwargs) -> Dict[str, Any]:
        """
        处理单个数据集
        
        Args:
            data: 输入数据
            symbols: 股票代码列表
            data_type: 数据类型 ('price', 'fundamental')
            **kwargs: 额外参数，会覆盖默认配置
            
        Returns:
            处理结果字典，包含processed_data, quality_report, feature_vectors等
        """
        import time
        start_time = time.time()
        
        # 合并配置
        config = {**self.config, **kwargs}
        
        # 检查缓存
        if config.get('use_cache', config['cache_enabled']):
            cache_key = self._generate_cache_key(data, symbols, data_type, config)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.stats['cache_hits'] += 1
                logger.debug(f"缓存命中: {symbols}")
                return cached_result
            else:
                self.stats['cache_misses'] += 1
        
        try:
            # 数据验证
            if not self.validate_data(data, data_type):
                raise ValueError(f"数据验证失败: {data_type}")
            
            # 数据质量检查
            quality_report = self.check_data_quality(data, data_type)
            
            # 数据清洗
            cleaned_data = self._clean_data(data, data_type, config)
            
            # 处理缺失值
            processed_data = self._handle_missing_values(cleaned_data, config)
            
            # 异常值处理
            processed_data = self._handle_outliers(processed_data, config)
            
            # 特征工程
            feature_vectors = []
            if config.get('calculate_features', True):
                feature_vectors = self._calculate_features(
                    processed_data, symbols, data_type, config
                )
            
            # 数据标准化
            if config.get('normalize', False):
                processed_data = self._normalize_data(processed_data, config)
            
            # 特征选择
            if config.get('feature_selection', False) and feature_vectors:
                feature_vectors = self._select_features(feature_vectors, config)
            
            # 构建结果
            result = {
                'processed_data': processed_data,
                'quality_report': quality_report,
                'feature_vectors': feature_vectors,
                'processing_info': {
                    'symbols': symbols,
                    'data_type': data_type,
                    'config': config,
                    'processing_time': time.time() - start_time,
                    'original_shape': data.shape,
                    'processed_shape': processed_data.shape
                }
            }
            
            # 缓存结果
            if config.get('use_cache', config['cache_enabled']):
                self.cache.set(cache_key, result)
            
            # 更新统计信息
            self.stats['processed_count'] += 1
            self.stats['processing_times'].append(time.time() - start_time)
            
            logger.info(f"数据处理完成: {symbols}, 耗时: {time.time() - start_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"数据处理失败: {symbols}, 错误: {str(e)}")
            raise
    
    def process_batch(self, 
                     batch_data: Dict[str, pd.DataFrame],
                     data_type: str = 'price',
                     parallel: bool = None,
                     **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        批量处理多个数据集
        
        Args:
            batch_data: 批量数据字典，键为股票代码，值为数据
            data_type: 数据类型
            parallel: 是否并行处理
            **kwargs: 额外参数
            
        Returns:
            批量处理结果字典
        """
        if parallel is None:
            parallel = self.config.get('parallel_processing', True)
        
        results = {}
        
        if parallel and len(batch_data) > 1:
            # 并行处理
            max_workers = min(self.config.get('max_workers', 4), len(batch_data))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_symbol = {
                    executor.submit(
                        self.process_data, 
                        data, 
                        [symbol], 
                        data_type, 
                        **kwargs
                    ): symbol
                    for symbol, data in batch_data.items()
                }
                
                # 收集结果
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                    except Exception as e:
                        logger.error(f"批量处理失败: {symbol}, 错误: {str(e)}")
                        results[symbol] = {
                            'error': str(e),
                            'processed_data': pd.DataFrame(),
                            'quality_report': {'status': 'error', 'score': 0.0},
                            'feature_vectors': []
                        }
        else:
            # 串行处理
            for symbol, data in batch_data.items():
                try:
                    result = self.process_data(data, [symbol], data_type, **kwargs)
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"批量处理失败: {symbol}, 错误: {str(e)}")
                    results[symbol] = {
                        'error': str(e),
                        'processed_data': pd.DataFrame(),
                        'quality_report': {'status': 'error', 'score': 0.0},
                        'feature_vectors': []
                    }
        
        logger.info(f"批量处理完成: {len(batch_data)}个数据集")
        return results
    
    def validate_data(self, data: pd.DataFrame, data_type: str) -> bool:
        """
        验证数据有效性
        
        Args:
            data: 输入数据
            data_type: 数据类型
            
        Returns:
            是否有效
        """
        if data.empty:
            logger.warning("数据为空")
            return False
        
        if data_type == 'price':
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            missing_columns = set(required_columns) - set(data.columns)
            if missing_columns:
                logger.warning(f"缺少必要列: {missing_columns}")
                return False
            
            # 检查价格关系
            if 'high' in data.columns and 'low' in data.columns:
                invalid_relations = (data['high'] < data['low']).sum()
                if invalid_relations > len(data) * 0.1:  # 超过10%的数据有问题
                    logger.warning(f"价格关系错误过多: {invalid_relations}")
                    return False
        
        return True
    
    def check_data_quality(self, data: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """
        检查数据质量
        
        Args:
            data: 输入数据
            data_type: 数据类型
            
        Returns:
            质量报告
        """
        return self.quality_checker.check_data_quality(data, data_type)
    
    def _clean_data(self, data: pd.DataFrame, data_type: str, config: Dict[str, Any]) -> pd.DataFrame:
        """清洗数据"""
        strategy = config.get('clean_strategy', 'conservative')
        return self.quality_checker.clean_data(data, data_type, strategy)
    
    def _handle_missing_values(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """处理缺失值"""
        method = config.get('missing_value_method', 'ffill')
        return self.feature_engineer.handle_missing_values(data, method)
    
    def _handle_outliers(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """处理异常值"""
        treatment = config.get('outlier_treatment', 'clip')
        if treatment == 'clip':
            return self.feature_engineer.treat_outliers(data, method='clip')
        elif treatment == 'remove':
            return self.feature_engineer.treat_outliers(data, method='remove')
        else:
            return data
    
    def _calculate_features(self, 
                          data: pd.DataFrame, 
                          symbols: List[str], 
                          data_type: str,
                          config: Dict[str, Any]) -> List[FeatureVector]:
        """计算特征"""
        feature_vectors = []
        
        try:
            if data_type == 'price':
                # 计算技术指标
                technical_features = self.feature_engineer.calculate_technical_indicators(data)
                
                # 计算市场微观结构特征
                microstructure_features = self.feature_engineer.calculate_microstructure_features(data)
                
                # 合并特征
                all_features = self.feature_engineer.combine_features([
                    technical_features, microstructure_features
                ])
                
                # 为每个时间点和股票创建特征向量
                for timestamp in all_features.index:
                    if pd.isna(timestamp):
                        continue
                    
                    for symbol in symbols:
                        try:
                            feature_series = all_features.loc[timestamp]
                            if feature_series.isna().all():
                                continue
                            
                            feature_vector = self.feature_engineer.create_feature_vector(
                                timestamp=timestamp,
                                symbol=symbol,
                                normalized_features=feature_series
                            )
                            feature_vectors.append(feature_vector)
                        except Exception as e:
                            logger.warning(f"创建特征向量失败: {symbol}, {timestamp}, {str(e)}")
                            continue
            
            elif data_type == 'fundamental':
                # 计算基本面因子
                fundamental_features = self.feature_engineer.calculate_fundamental_factors(data)
                
                # 为每个时间点和股票创建特征向量
                for timestamp in fundamental_features.index:
                    if pd.isna(timestamp):
                        continue
                    
                    for symbol in symbols:
                        try:
                            feature_series = fundamental_features.loc[timestamp]
                            if feature_series.isna().all():
                                continue
                            
                            # 创建基本面特征向量
                            feature_vector = FeatureVector(
                                timestamp=timestamp,
                                symbol=symbol,
                                technical_indicators={'default_tech': 0.0},
                                fundamental_factors=feature_series.to_dict(),
                                market_microstructure={'default_micro': 0.0}
                            )
                            feature_vectors.append(feature_vector)
                        except Exception as e:
                            logger.warning(f"创建基本面特征向量失败: {symbol}, {timestamp}, {str(e)}")
                            continue
            
        except Exception as e:
            logger.error(f"特征计算失败: {str(e)}")
        
        logger.info(f"计算特征完成: {len(feature_vectors)}个特征向量")
        return feature_vectors
    
    def _normalize_data(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """标准化数据"""
        method = config.get('normalization_method', 'zscore')
        return self.feature_engineer.normalize_features(data, method)
    
    def _select_features(self, feature_vectors: List[FeatureVector], 
                        config: Dict[str, Any]) -> List[FeatureVector]:
        """特征选择"""
        # 这里可以实现特征选择逻辑
        # 暂时返回原始特征向量
        return feature_vectors
    
    def _generate_cache_key(self, 
                          data: pd.DataFrame, 
                          symbols: List[str], 
                          data_type: str,
                          config: Dict[str, Any]) -> str:
        """生成缓存键"""
        # 创建数据指纹
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(data, index=True).values
        ).hexdigest()[:16]
        
        # 创建配置指纹
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        
        # 组合缓存键
        cache_key = f"data_processor_{data_type}_{'-'.join(symbols)}_{data_hash}_{config_hash}"
        
        return cache_key
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        return {
            'processed_count': self.stats['processed_count'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses']),
            'average_processing_time': avg_time,
            'total_processing_time': sum(self.stats['processing_times'])
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("数据处理器缓存已清空")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'processed_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': []
        }
        logger.info("数据处理器统计信息已重置")


class BatchDataProcessor:
    """批量数据处理器"""
    
    def __init__(self, processor: DataProcessor):
        """
        初始化批量处理器
        
        Args:
            processor: 数据处理器实例
        """
        self.processor = processor
    
    def process_time_series_batch(self, 
                                 data: pd.DataFrame,
                                 symbols: List[str],
                                 window_size: int = 60,
                                 step_size: int = 1,
                                 data_type: str = 'price',
                                 **kwargs) -> List[Dict[str, Any]]:
        """
        处理时间序列批量数据
        
        Args:
            data: 时间序列数据
            symbols: 股票代码列表
            window_size: 窗口大小
            step_size: 步长
            data_type: 数据类型
            **kwargs: 额外参数
            
        Returns:
            批量处理结果列表
        """
        results = []
        
        # 按窗口切分数据
        for i in range(0, len(data) - window_size + 1, step_size):
            window_data = data.iloc[i:i + window_size]
            
            try:
                result = self.processor.process_data(
                    data=window_data,
                    symbols=symbols,
                    data_type=data_type,
                    **kwargs
                )
                
                # 添加窗口信息
                result['window_info'] = {
                    'start_index': i,
                    'end_index': i + window_size,
                    'start_date': window_data.index[0],
                    'end_date': window_data.index[-1]
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"窗口处理失败: {i}-{i + window_size}, 错误: {str(e)}")
                continue
        
        logger.info(f"时间序列批量处理完成: {len(results)}个窗口")
        return results
    
    def process_cross_sectional_batch(self,
                                    data_dict: Dict[str, pd.DataFrame],
                                    timestamp: datetime,
                                    data_type: str = 'price',
                                    **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        处理横截面批量数据
        
        Args:
            data_dict: 股票数据字典
            timestamp: 时间戳
            data_type: 数据类型
            **kwargs: 额外参数
            
        Returns:
            横截面处理结果字典
        """
        # 提取指定时间戳的数据
        cross_sectional_data = {}
        
        for symbol, data in data_dict.items():
            if timestamp in data.index:
                # 提取单行数据并转换为DataFrame
                row_data = data.loc[[timestamp]]
                cross_sectional_data[symbol] = row_data
        
        # 批量处理
        return self.processor.process_batch(
            batch_data=cross_sectional_data,
            data_type=data_type,
            **kwargs
        )


# 全局数据处理器实例
_global_processor = None


def get_global_processor() -> DataProcessor:
    """获取全局数据处理器实例"""
    global _global_processor
    if _global_processor is None:
        _global_processor = DataProcessor()
    return _global_processor


def set_global_processor(processor: DataProcessor):
    """设置全局数据处理器实例"""
    global _global_processor
    _global_processor = processor