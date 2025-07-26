#!/usr/bin/env python3

from risk_control.performance_optimizer import CacheConfig, AdvancedCacheManager
import tempfile

# Create test setup similar to the failing test
temp_dir = tempfile.mkdtemp()
cache_config = CacheConfig()
cache_config.memory_cache_size = 10
cache_config.disk_cache_dir = temp_dir
cache_config.disk_cache_ttl = 3600
cache_manager = AdvancedCacheManager(cache_config)

print(f"Cache max size: {cache_config.memory_cache_size}")

# Fill the cache
for i in range(cache_config.memory_cache_size):
    cache_manager.put(f"key_{i}", f"value_{i}")
    print(f"Added key_{i}, cache size: {cache_manager.memory_cache.size()}")

print(f"Cache is full, size: {cache_manager.memory_cache.size()}")

# Check what keys exist
print("Keys before adding new key:")
for i in range(cache_config.memory_cache_size):
    key = f"key_{i}"
    value = cache_manager.get(key, None)
    print(f"  {key}: {value}")

# Add one more item - this should trigger LRU eviction
print("\nAdding new_key...")
cache_manager.put("new_key", "new_value")

print(f"Cache size after adding new_key: {cache_manager.memory_cache.size()}")

# Check what keys exist after eviction
print("Keys after adding new key:")
for i in range(cache_config.memory_cache_size):
    key = f"key_{i}"
    value = cache_manager.get(key, None)
    print(f"  {key}: {value}")

new_value = cache_manager.get("new_key", None)
print(f"  new_key: {new_value}")

# Check memory vs disk cache
print(f"\nMemory cache keys: {list(cache_manager.memory_cache._memory_cache.keys())}")
if cache_manager.disk_cache:
    print(f"Disk cache enabled: Yes")
else:
    print(f"Disk cache enabled: No")

# Check the specific assertion from the test
key_0_value = cache_manager.get("key_0", None)
print(f"\nTest assertion: cache_manager.get('key_0', None) is None")
print(f"Actual result: {key_0_value} is None = {key_0_value is None}")

# Check directly from memory cache only
key_0_memory_only = cache_manager.memory_cache.get("key_0", None)
print(f"Memory cache only: {key_0_memory_only} is None = {key_0_memory_only is None}")

# Check what's actually in the memory cache dict
print(f"Raw memory cache dict: {cache_manager.memory_cache._memory_cache}")
print(f"key_0 in raw dict: {'key_0' in cache_manager.memory_cache._memory_cache}")

# Check access times
print(f"Access times keys: {list(cache_manager.memory_cache._cache_access_times.keys())}")
print(f"key_0 in access times: {'key_0' in cache_manager.memory_cache._cache_access_times}")