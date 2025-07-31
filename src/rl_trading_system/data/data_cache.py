"""
数据缓存管理模块
实现数据缓存机制，提高数据获取效率
"""

import os
import pickle
import hashlib
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache", 
                 default_ttl: int = 3600):
        """
        初始化数据缓存
        
        Args:
            cache_dir: 缓存目录
            default_ttl: 默认缓存时间（秒）
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 内存缓存
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
    
    def _get_cache_key(self, key: str) -> str:
        """生成缓存键的哈希值"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _is_expired(self, timestamp: datetime, ttl: int) -> bool:
        """检查缓存是否过期"""
        return datetime.now() - timestamp > timedelta(seconds=ttl)
    
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            ttl: 缓存时间，None使用默认值
            
        Returns:
            缓存的数据，如果不存在或过期返回None
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_key = self._get_cache_key(key)
        
        # 先检查内存缓存
        if cache_key in self._memory_cache:
            cache_item = self._memory_cache[cache_key]
            if not self._is_expired(cache_item['timestamp'], ttl):
                logger.debug(f"内存缓存命中: {key}")
                return cache_item['data']
            else:
                # 过期，删除内存缓存
                del self._memory_cache[cache_key]
        
        # 检查文件缓存
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cache_item = pickle.load(f)
                
                if not self._is_expired(cache_item['timestamp'], ttl):
                    # 加载到内存缓存
                    self._memory_cache[cache_key] = cache_item
                    logger.debug(f"文件缓存命中: {key}")
                    return cache_item['data']
                else:
                    # 过期，删除文件
                    os.remove(cache_path)
                    logger.debug(f"缓存过期，已删除: {key}")
                    
            except Exception as e:
                logger.error(f"读取缓存文件失败: {e}")
                # 删除损坏的缓存文件
                try:
                    os.remove(cache_path)
                except:
                    pass
        
        return None
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None):
        """
        设置缓存数据
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            ttl: 缓存时间，None使用默认值
        """
        if ttl is None:
            ttl = self.default_ttl
        
        cache_key = self._get_cache_key(key)
        cache_item = {
            'data': data,
            'timestamp': datetime.now(),
            'ttl': ttl
        }
        
        # 设置内存缓存
        self._memory_cache[cache_key] = cache_item
        
        # 设置文件缓存
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_item, f)
            logger.debug(f"数据已缓存: {key}")
        except Exception as e:
            logger.error(f"写入缓存文件失败: {e}")
    
    def delete(self, key: str):
        """
        删除缓存数据
        
        Args:
            key: 缓存键
        """
        cache_key = self._get_cache_key(key)
        
        # 删除内存缓存
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        # 删除文件缓存
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.debug(f"缓存已删除: {key}")
            except Exception as e:
                logger.error(f"删除缓存文件失败: {e}")
    
    def clear(self):
        """清空所有缓存"""
        # 清空内存缓存
        self._memory_cache.clear()
        
        # 清空文件缓存
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
            logger.info("所有缓存已清空")
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            缓存统计信息
        """
        memory_count = len(self._memory_cache)
        
        file_count = 0
        total_size = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_count += 1
                    file_path = os.path.join(self.cache_dir, filename)
                    total_size += os.path.getsize(file_path)
        except Exception as e:
            logger.error(f"获取缓存信息失败: {e}")
        
        return {
            'memory_cache_count': memory_count,
            'file_cache_count': file_count,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': self.cache_dir
        }
    
    def cleanup_expired(self):
        """清理过期的缓存"""
        # 清理内存缓存
        expired_keys = []
        for cache_key, cache_item in self._memory_cache.items():
            if self._is_expired(cache_item['timestamp'], cache_item['ttl']):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        # 清理文件缓存
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            cache_item = pickle.load(f)
                        
                        if self._is_expired(cache_item['timestamp'], cache_item['ttl']):
                            os.remove(file_path)
                    except:
                        # 如果文件损坏，直接删除
                        os.remove(file_path)
            
            logger.info(f"清理了{len(expired_keys)}个过期缓存")
        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")


# 全局缓存实例
_global_cache = None


def get_global_cache() -> DataCache:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCache()
    return _global_cache


def set_global_cache(cache: DataCache):
    """设置全局缓存实例"""
    global _global_cache
    _global_cache = cache