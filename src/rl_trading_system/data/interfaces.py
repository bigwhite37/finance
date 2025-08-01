"""
数据接口抽象类
定义数据获取的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataInterface(ABC):
    """数据接口抽象类"""
    
    def __init__(self):
        """初始化数据接口"""
        self._cache = {}
        self._cache_enabled = True
    
    @abstractmethod
    def get_stock_list(self, market: str = 'A') -> List[str]:
        """
        获取股票列表
        
        Args:
            market: 市场代码，如'A'表示A股
            
        Returns:
            股票代码列表
        """
        pass
    
    @abstractmethod
    def get_price_data(self, symbols: List[str], 
                      start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取价格数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            
        Returns:
            价格数据DataFrame，包含open, high, low, close, volume, amount列
        """
        pass
    
    @abstractmethod
    def get_fundamental_data(self, symbols: List[str], 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取基本面数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            
        Returns:
            基本面数据DataFrame
        """
        pass
    
    def validate_date_format(self, date_str: str) -> bool:
        """
        验证日期格式
        
        Args:
            date_str: 日期字符串
            
        Returns:
            是否为有效格式
        """
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """
        验证日期范围
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            日期范围是否有效
        """
        if not (self.validate_date_format(start_date) and 
                self.validate_date_format(end_date)):
            return False
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        return start <= end
    
    def validate_symbols(self, symbols: List[str]) -> bool:
        """
        验证股票代码格式
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            股票代码是否有效
        """
        if not symbols:
            return False
        
        for symbol in symbols:
            if not isinstance(symbol, str) or len(symbol) == 0:
                return False
        
        return True
    
    def standardize_dataframe(self, df: pd.DataFrame, 
                            data_type: str = 'price') -> pd.DataFrame:
        """
        标准化DataFrame格式
        
        Args:
            df: 原始DataFrame
            data_type: 数据类型，'price'或'fundamental'
            
        Returns:
            标准化后的DataFrame
        """
        if df.empty:
            return df
        
        # 确保有datetime和instrument索引
        if data_type == 'price':
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        else:
            required_columns = []
        
        # 检查必要列是否存在
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"缺少必要列: {missing_columns}")
        
        # 确保索引包含datetime和instrument层级
        if not isinstance(df.index, pd.MultiIndex):
            logger.warning("数据没有多层索引，这可能导致后续处理失败")
        else:
            # 检查索引名称
            index_names = df.index.names
            if 'datetime' not in index_names or 'instrument' not in index_names:
                logger.warning(f"索引层级名称不标准，当前为: {index_names}")
                # 尝试重命名索引
                if len(index_names) >= 2:
                    new_names = index_names.copy()
                    if index_names[0] != 'datetime':
                        new_names[0] = 'datetime'
                    if index_names[1] != 'instrument':
                        new_names[1] = 'instrument'
                    df.index.names = new_names
                    logger.info(f"索引重命名为: {new_names}")
        
        return df
    
    def enable_cache(self, enabled: bool = True):
        """启用或禁用缓存"""
        self._cache_enabled = enabled
        if not enabled:
            self._cache.clear()
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
    
    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [method] + [str(arg) for arg in args]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)
    
    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        if not self._cache_enabled:
            return None
        return self._cache.get(cache_key)
    
    def _set_cache(self, cache_key: str, data: pd.DataFrame):
        """设置缓存数据"""
        if self._cache_enabled:
            self._cache[cache_key] = data.copy()
    
    def check_data_quality(self, df: pd.DataFrame, 
                          data_type: str = 'price') -> Dict[str, Any]:
        """
        检查数据质量
        
        Args:
            df: 数据DataFrame
            data_type: 数据类型
            
        Returns:
            数据质量报告
        """
        if df.empty:
            return {'status': 'empty', 'issues': ['数据为空']}
        
        issues = []
        
        # 检查缺失值
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            issues.append(f"存在缺失值: {missing_data.to_dict()}")
        
        # 检查价格数据的逻辑关系
        if data_type == 'price' and all(col in df.columns for col in ['high', 'low']):
            invalid_price_relation = (df['high'] < df['low']).sum()
            if invalid_price_relation > 0:
                issues.append(f"存在{invalid_price_relation}条最高价低于最低价的记录")
        
        # 检查异常值
        if data_type == 'price':
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_columns:
                if col in df.columns:
                    negative_values = (df[col] < 0).sum()
                    if negative_values > 0:
                        issues.append(f"列{col}存在{negative_values}个负值")
        
        status = 'good' if not issues else 'warning'
        return {'status': status, 'issues': issues, 'row_count': len(df)}