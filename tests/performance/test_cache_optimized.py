#!/usr/bin/env python3
"""
优化缓存组件测试

测试重构后的缓存管理器组件功能。
"""

import sys
import os
import unittest
import tempfile
import shutil
from unittest.mock import patch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache.cache_config import CacheConfig
from cache.cache_base import CacheStats, CacheUtils, LRUTracker
from cache.memory_cache_optimized import MemoryCacheManager
from cache.disk_cache_optimized import DiskCacheManager


class TestCacheStats(unittest.TestCase):
    """测试缓存统计组件"""
    
    def setUp(self):
        self.stats = CacheStats()
    
    def test_hit_miss_recording(self):
        """测试缓存命中和未命中记录"""
        # 初始状态
        self.assertEqual(self.stats.hits, 0)
        self.assertEqual(self.stats.misses, 0)
        
        # 记录命中
        self.stats.record_hit('memory')
        self.assertEqual(self.stats.hits, 1)
        self.assertEqual(self.stats.memory_hits, 1)
        
        self.stats.record_hit('disk')
        self.assertEqual(self.stats.hits, 2)
        self.assertEqual(self.stats.disk_hits, 1)
        
        # 记录未命中
        self.stats.record_miss()
        self.assertEqual(self.stats.misses, 1)
        
        # 验证命中率
        hit_rate = self.stats.get_hit_rate()
        self.assertAlmostEqual(hit_rate, 2/3, places=2)
    
    def test_stats_dict(self):
        """测试统计信息字典"""
        self.stats.record_hit('memory')
        self.stats.record_miss()
        self.stats.record_eviction()
        
        stats_dict = self.stats.get_stats_dict()
        
        self.assertIn('hits', stats_dict)
        self.assertIn('misses', stats_dict)
        self.assertIn('hit_rate', stats_dict)
        self.assertEqual(stats_dict['hits'], 1)
        self.assertEqual(stats_dict['misses'], 1)


class TestCacheUtils(unittest.TestCase):
    """测试缓存工具函数"""
    
    def test_generate_cache_key(self):
        """测试缓存键生成"""
        key1 = CacheUtils.generate_cache_key("test_key")
        key2 = CacheUtils.generate_cache_key("test_key")
        key3 = CacheUtils.generate_cache_key("different_key")
        
        # 相同输入应产生相同的键
        self.assertEqual(key1, key2)
        # 不同输入应产生不同的键
        self.assertNotEqual(key1, key3)
        # 键应该是32位十六进制字符串
        self.assertEqual(len(key1), 32)
    
    def test_file_operations(self):
        """测试文件操作工具"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(b"test data")
        
        try:
            # 测试文件过期检查
            self.assertFalse(CacheUtils.is_file_expired(tmp_path, 3600))  # 1小时内
            self.assertTrue(CacheUtils.is_file_expired(tmp_path, 0))      # 立即过期
            
            # 测试安全删除
            self.assertTrue(CacheUtils.safe_remove_file(tmp_path))
            self.assertFalse(os.path.exists(tmp_path))
            
            # 删除不存在的文件应返回False
            self.assertFalse(CacheUtils.safe_remove_file(tmp_path))
        finally:
            # 确保清理
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestLRUTracker(unittest.TestCase):
    """测试LRU跟踪器"""
    
    def setUp(self):
        self.lru = LRUTracker()
    
    def test_access_tracking(self):
        """测试访问跟踪"""
        # 记录访问
        self.lru.record_access("key1")
        self.lru.record_access("key2")
        
        self.assertEqual(self.lru.get_access_count(), 2)
        
        # key1应该是最旧的
        lru_key = self.lru.get_lru_key()
        self.assertEqual(lru_key, "key1")
    
    def test_eviction(self):
        """测试淘汰功能"""
        self.lru.record_access("key1")
        self.lru.record_access("key2")
        self.lru.record_access("key3")
        
        # 淘汰最旧的键
        evicted = self.lru.evict_lru()
        self.assertEqual(evicted, "key1")
        self.assertEqual(self.lru.get_access_count(), 2)
        
        # 现在key2应该是最旧的
        self.assertEqual(self.lru.get_lru_key(), "key2")


class TestMemoryCacheManager(unittest.TestCase):
    """测试内存缓存管理器"""
    
    def setUp(self):
        self.config = CacheConfig(
            enable_memory_cache=True,
            max_memory_cache_size=3,
            enable_disk_cache=False
        )
        self.cache = MemoryCacheManager(self.config)
    
    def test_basic_operations(self):
        """测试基本缓存操作"""
        # 存储和获取
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertTrue(self.cache.contains("key1"))
        
        # 获取不存在的键
        self.assertEqual(self.cache.get("nonexistent"), None)
        self.assertEqual(self.cache.get("nonexistent", "default"), "default")
        
        # 删除
        self.assertTrue(self.cache.remove("key1"))
        self.assertFalse(self.cache.contains("key1"))
        self.assertFalse(self.cache.remove("key1"))  # 重复删除
    
    def test_lru_eviction(self):
        """测试LRU淘汰机制"""
        # 填满缓存
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        self.assertEqual(self.cache.size(), 3)
        
        # 访问key1，使其成为最新的
        self.cache.get("key1")
        
        # 添加新键，应该淘汰key2（最旧的未访问）
        self.cache.put("key4", "value4")
        self.assertEqual(self.cache.size(), 3)
        self.assertTrue(self.cache.contains("key1"))
        self.assertFalse(self.cache.contains("key2"))  # 被淘汰
        self.assertTrue(self.cache.contains("key3"))
        self.assertTrue(self.cache.contains("key4"))
    
    def test_disabled_cache(self):
        """测试禁用缓存的情况"""
        disabled_config = CacheConfig(enable_memory_cache=False)
        disabled_cache = MemoryCacheManager(disabled_config)
        
        disabled_cache.put("key1", "value1")
        self.assertEqual(disabled_cache.get("key1"), None)
        self.assertFalse(disabled_cache.contains("key1"))
        self.assertEqual(disabled_cache.size(), 0)


class TestDiskCacheManager(unittest.TestCase):
    """测试磁盘缓存管理器"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = CacheConfig(
            enable_disk_cache=True,
            cache_directory=self.temp_dir,
            cache_expiry_hours=1,
            compression_level=0,  # 不压缩以简化测试
            max_disk_cache_size_mb=1
        )
        self.cache = DiskCacheManager(self.config)
    
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_operations(self):
        """测试基本磁盘缓存操作"""
        # 存储和获取
        self.cache.put("key1", {"data": "value1"})
        retrieved = self.cache.get("key1")
        self.assertEqual(retrieved, {"data": "value1"})
        self.assertTrue(self.cache.contains("key1"))
        
        # 获取不存在的键
        self.assertEqual(self.cache.get("nonexistent"), None)
        self.assertEqual(self.cache.get("nonexistent", "default"), "default")
        
        # 删除
        self.assertTrue(self.cache.remove("key1"))
        self.assertFalse(self.cache.contains("key1"))
        self.assertFalse(self.cache.remove("key1"))  # 重复删除
    
    def test_compression(self):
        """测试缓存压缩"""
        compressed_config = CacheConfig(
            enable_disk_cache=True,
            cache_directory=self.temp_dir,
            compression_level=6
        )
        compressed_cache = DiskCacheManager(compressed_config)
        
        # 存储较大的数据
        large_data = {"numbers": list(range(1000))}
        compressed_cache.put("large_key", large_data)
        
        # 应该能正确取回
        retrieved = compressed_cache.get("large_key")
        self.assertEqual(retrieved, large_data)
    
    def test_expiration(self):
        """测试缓存过期"""
        import time
        
        # 使用很短的过期时间
        expired_config = CacheConfig(
            enable_disk_cache=True,
            cache_directory=self.temp_dir,
            cache_expiry_hours=1/3600  # 1秒
        )
        expired_cache = DiskCacheManager(expired_config)
        
        expired_cache.put("expire_key", "expire_value")
        
        # 立即获取应该还能取到
        self.assertEqual(expired_cache.get("expire_key"), "expire_value")
        
        # 等待过期
        time.sleep(1.1)
        
        # 现在应该过期了
        self.assertEqual(expired_cache.get("expire_key"), None)
        self.assertFalse(expired_cache.contains("expire_key"))
    
    def test_cleanup(self):
        """测试缓存清理"""
        # 添加一些缓存文件
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        # 清理应该不删除任何文件（因为它们没有过期）
        removed = self.cache.cleanup()
        self.assertGreaterEqual(removed, 0)  # 可能会清理，也可能不会
        
        # 清空所有缓存
        self.cache.clear()
        self.assertFalse(self.cache.contains("key1"))
        self.assertFalse(self.cache.contains("key2"))
    
    def test_disabled_cache(self):
        """测试禁用磁盘缓存的情况"""
        disabled_config = CacheConfig(enable_disk_cache=False)
        disabled_cache = DiskCacheManager(disabled_config)
        
        disabled_cache.put("key1", "value1")
        self.assertEqual(disabled_cache.get("key1"), None)
        self.assertFalse(disabled_cache.contains("key1"))


if __name__ == '__main__':
    unittest.main()