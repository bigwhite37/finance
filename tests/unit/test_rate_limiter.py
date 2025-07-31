"""
API限流系统的单元测试
测试请求限流和并发控制，限流算法的有效性和性能
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import asyncio

from src.rl_trading_system.api.rate_limiter import (
    RateLimiter,
    TokenBucket,
    SlidingWindow,
    FixedWindow,
    RateLimitConfig,
    RateLimitResult,
    ConcurrencyLimiter,
    DistributedRateLimiter,
    RateLimitRule,
    RateLimitStrategy,
    QuotaManager,
    ThrottleManager
)


class TestTokenBucket:
    """令牌桶算法测试类"""

    @pytest.fixture
    def token_bucket_config(self):
        """创建令牌桶配置"""
        return {
            'capacity': 100,          # 桶容量
            'refill_rate': 10,        # 每秒补充令牌数
            'initial_tokens': 100     # 初始令牌数
        }

    @pytest.fixture
    def token_bucket(self, token_bucket_config):
        """创建令牌桶"""
        return TokenBucket(
            capacity=token_bucket_config['capacity'],
            refill_rate=token_bucket_config['refill_rate'],
            initial_tokens=token_bucket_config['initial_tokens']
        )

    def test_token_bucket_initialization(self, token_bucket, token_bucket_config):
        """测试令牌桶初始化"""
        assert token_bucket.capacity == token_bucket_config['capacity']
        assert token_bucket.refill_rate == token_bucket_config['refill_rate']
        assert token_bucket.available_tokens == token_bucket_config['initial_tokens']
        assert token_bucket.last_refill_time is not None

    def test_consume_tokens_success(self, token_bucket):
        """测试成功消费令牌"""
        # 消费10个令牌
        result = token_bucket.consume(10)
        
        assert result == True
        assert token_bucket.available_tokens == 90

    def test_consume_tokens_insufficient(self, token_bucket):
        """测试令牌不足的情况"""
        # 尝试消费超过可用令牌数
        result = token_bucket.consume(150)
        
        assert result == False
        assert token_bucket.available_tokens == 100  # 令牌数不变

    def test_token_refill_mechanism(self, token_bucket):
        """测试令牌补充机制"""
        # 消费所有令牌
        token_bucket.consume(100)
        assert token_bucket.available_tokens == 0
        
        # 等待1秒，应该补充10个令牌
        time.sleep(1.1)
        
        # 尝试消费令牌，应该触发补充
        result = token_bucket.consume(5)
        assert result == True
        assert token_bucket.available_tokens >= 5  # 至少补充了10个令牌

    def test_token_bucket_capacity_limit(self, token_bucket):
        """测试令牌桶容量限制"""
        # 等待足够时间让令牌满
        time.sleep(2.0)
        
        # 强制触发补充
        token_bucket._refill()
        
        # 令牌数不应超过容量
        assert token_bucket.available_tokens <= token_bucket.capacity

    def test_concurrent_token_consumption(self, token_bucket):
        """测试并发令牌消费"""
        success_count = 0
        lock = threading.Lock()
        
        def consume_tokens():
            nonlocal success_count
            if token_bucket.consume(1):
                with lock:
                    success_count += 1
        
        # 创建多个线程并发消费令牌
        threads = []
        for _ in range(150):  # 超过桶容量的请求
            thread = threading.Thread(target=consume_tokens)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 成功消费的令牌数应该不超过桶容量
        assert success_count <= token_bucket.capacity
        assert token_bucket.available_tokens >= 0


class TestSlidingWindow:
    """滑动窗口算法测试类"""

    @pytest.fixture
    def sliding_window_config(self):
        """创建滑动窗口配置"""
        return {
            'window_size_seconds': 60,  # 60秒窗口
            'max_requests': 100,        # 最大请求数
            'precision_seconds': 1      # 1秒精度
        }

    @pytest.fixture
    def sliding_window(self, sliding_window_config):
        """创建滑动窗口"""
        return SlidingWindow(
            window_size=sliding_window_config['window_size_seconds'],
            max_requests=sliding_window_config['max_requests'],
            precision=sliding_window_config['precision_seconds']
        )

    def test_sliding_window_initialization(self, sliding_window, sliding_window_config):
        """测试滑动窗口初始化"""
        assert sliding_window.window_size == sliding_window_config['window_size_seconds']
        assert sliding_window.max_requests == sliding_window_config['max_requests']
        assert sliding_window.precision == sliding_window_config['precision_seconds']
        assert len(sliding_window.request_times) == 0

    def test_allow_request_within_limit(self, sliding_window):
        """测试限制内的请求"""
        current_time = time.time()
        
        # 发送50个请求，应该都被允许
        for i in range(50):
            result = sliding_window.allow_request(current_time + i * 0.1)
            assert result == True
        
        assert len(sliding_window.request_times) == 50

    def test_block_request_exceeding_limit(self, sliding_window):
        """测试超过限制的请求"""
        current_time = time.time()
        
        # 发送100个请求到达限制
        for i in range(100):
            sliding_window.allow_request(current_time + i * 0.1)
        
        # 第101个请求应该被拒绝
        result = sliding_window.allow_request(current_time + 10)
        assert result == False

    def test_sliding_window_cleanup(self, sliding_window):
        """测试滑动窗口清理过期请求"""
        current_time = time.time()
        
        # 添加一些旧请求
        for i in range(50):
            sliding_window.allow_request(current_time - 120 + i)  # 2分钟前的请求
        
        # 添加一些新请求
        for i in range(30):
            sliding_window.allow_request(current_time + i)
        
        # 清理后应该只保留窗口内的请求
        sliding_window._cleanup_expired_requests(current_time)
        assert len(sliding_window.request_times) == 30

    def test_get_current_request_count(self, sliding_window):
        """测试获取当前请求计数"""
        current_time = time.time()
        
        # 添加一些请求
        for i in range(25):
            sliding_window.allow_request(current_time + i)
        
        count = sliding_window.get_current_count(current_time + 30)
        assert count == 25

    def test_get_remaining_quota(self, sliding_window):
        """测试获取剩余配额"""
        current_time = time.time()
        
        # 使用30个请求
        for i in range(30):
            sliding_window.allow_request(current_time + i)
        
        remaining = sliding_window.get_remaining_quota(current_time + 35)
        assert remaining == 70  # 100 - 30

    def test_concurrent_sliding_window_requests(self, sliding_window):
        """测试并发滑动窗口请求"""
        current_time = time.time()
        allowed_count = 0
        lock = threading.Lock()
        
        def make_request():
            nonlocal allowed_count
            if sliding_window.allow_request(time.time()):
                with lock:
                    allowed_count += 1
        
        # 创建多个线程并发请求
        threads = []
        for _ in range(150):  # 超过限制的请求
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 允许的请求数不应超过限制
        assert allowed_count <= sliding_window.max_requests


class TestFixedWindow:
    """固定窗口算法测试类"""

    @pytest.fixture
    def fixed_window_config(self):
        """创建固定窗口配置"""
        return {
            'window_size_seconds': 60,  # 60秒窗口
            'max_requests': 100         # 最大请求数
        }

    @pytest.fixture
    def fixed_window(self, fixed_window_config):
        """创建固定窗口"""
        return FixedWindow(
            window_size=fixed_window_config['window_size_seconds'],
            max_requests=fixed_window_config['max_requests']
        )

    def test_fixed_window_initialization(self, fixed_window, fixed_window_config):
        """测试固定窗口初始化"""
        assert fixed_window.window_size == fixed_window_config['window_size_seconds']
        assert fixed_window.max_requests == fixed_window_config['max_requests']
        assert fixed_window.current_window_start is not None
        assert fixed_window.current_request_count == 0

    def test_allow_request_in_same_window(self, fixed_window):
        """测试同一窗口内的请求"""
        current_time = time.time()
        
        # 在同一窗口内发送50个请求
        for i in range(50):
            result = fixed_window.allow_request(current_time + i)
            assert result == True
        
        assert fixed_window.current_request_count == 50

    def test_window_reset_mechanism(self, fixed_window):
        """测试窗口重置机制"""
        current_time = time.time()
        
        # 在当前窗口发送50个请求
        for i in range(50):
            fixed_window.allow_request(current_time + i)
        
        # 移动到下一个窗口
        next_window_time = current_time + fixed_window.window_size + 1
        result = fixed_window.allow_request(next_window_time)
        
        assert result == True
        assert fixed_window.current_request_count == 1  # 重置后的第一个请求

    def test_exceed_window_limit(self, fixed_window):
        """测试超过窗口限制"""
        current_time = time.time()
        
        # 发送满限制的请求
        for i in range(fixed_window.max_requests):
            result = fixed_window.allow_request(current_time + i * 0.1)
            assert result == True
        
        # 下一个请求应该被拒绝
        result = fixed_window.allow_request(current_time + 10)
        assert result == False
        assert fixed_window.current_request_count == fixed_window.max_requests


class TestConcurrencyLimiter:
    """并发限制器测试类"""

    @pytest.fixture
    def concurrency_config(self):
        """创建并发限制配置"""
        return {
            'max_concurrent_requests': 10,
            'timeout_seconds': 5,
            'queue_size': 20
        }

    @pytest.fixture
    def concurrency_limiter(self, concurrency_config):
        """创建并发限制器"""
        return ConcurrencyLimiter(
            max_concurrent=concurrency_config['max_concurrent_requests'],
            timeout=concurrency_config['timeout_seconds'],
            queue_size=concurrency_config['queue_size']
        )

    def test_concurrency_limiter_initialization(self, concurrency_limiter, concurrency_config):
        """测试并发限制器初始化"""
        assert concurrency_limiter.max_concurrent == concurrency_config['max_concurrent_requests']
        assert concurrency_limiter.timeout == concurrency_config['timeout_seconds']
        assert concurrency_limiter.current_concurrent == 0

    def test_acquire_semaphore_success(self, concurrency_limiter):
        """测试成功获取信号量"""
        # 获取5个信号量，应该都成功
        semaphores = []
        for i in range(5):
            semaphore = concurrency_limiter.acquire()
            assert semaphore is not None
            semaphores.append(semaphore)
        
        assert concurrency_limiter.current_concurrent == 5
        
        # 释放信号量
        for semaphore in semaphores:
            concurrency_limiter.release(semaphore)

    def test_acquire_semaphore_at_limit(self, concurrency_limiter):
        """测试在限制边界获取信号量"""
        # 获取到最大限制
        semaphores = []
        for i in range(concurrency_limiter.max_concurrent):
            semaphore = concurrency_limiter.acquire()
            assert semaphore is not None
            semaphores.append(semaphore)
        
        # 下一个请求应该被阻塞或拒绝（取决于实现）
        start_time = time.time()
        semaphore = concurrency_limiter.acquire(timeout=1.0)
        end_time = time.time()
        
        # 如果返回None，说明被拒绝；如果等待时间接近超时，说明被阻塞
        assert semaphore is None or (end_time - start_time) >= 0.9
        
        # 释放信号量
        for sem in semaphores:
            concurrency_limiter.release(sem)

    def test_release_semaphore(self, concurrency_limiter):
        """测试释放信号量"""
        # 获取信号量
        semaphore = concurrency_limiter.acquire()
        assert concurrency_limiter.current_concurrent == 1
        
        # 释放信号量
        concurrency_limiter.release(semaphore)
        assert concurrency_limiter.current_concurrent == 0

    def test_concurrent_acquire_release(self, concurrency_limiter):
        """测试并发获取和释放"""
        successful_acquisitions = 0
        lock = threading.Lock()
        
        def acquire_and_release():
            nonlocal successful_acquisitions
            semaphore = concurrency_limiter.acquire(timeout=2.0)
            if semaphore:
                with lock:
                    successful_acquisitions += 1
                time.sleep(0.1)  # 模拟处理时间
                concurrency_limiter.release(semaphore)
        
        # 创建更多线程than限制
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=acquire_and_release)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 所有请求最终应该都能被处理
        assert successful_acquisitions == 20
        assert concurrency_limiter.current_concurrent == 0

    def test_timeout_handling(self, concurrency_limiter):
        """测试超时处理"""
        # 占满所有信号量
        semaphores = []
        for _ in range(concurrency_limiter.max_concurrent):
            sem = concurrency_limiter.acquire()
            semaphores.append(sem)
        
        # 尝试获取会超时
        start_time = time.time()
        semaphore = concurrency_limiter.acquire(timeout=1.0)
        end_time = time.time()
        
        assert semaphore is None
        assert (end_time - start_time) >= 0.9  # 接近超时时间
        
        # 清理
        for sem in semaphores:
            concurrency_limiter.release(sem)


class TestRateLimiter:
    """综合限流器测试类"""

    @pytest.fixture
    def rate_limit_config(self):
        """创建限流配置"""
        return RateLimitConfig(
            # 令牌桶配置
            token_bucket_capacity=100,
            token_bucket_refill_rate=10,
            
            # 滑动窗口配置
            sliding_window_size=60,
            sliding_window_max_requests=1000,
            
            # 固定窗口配置
            fixed_window_size=60,
            fixed_window_max_requests=500,
            
            # 并发限制配置
            max_concurrent_requests=50,
            
            # 策略配置
            primary_strategy=RateLimitStrategy.TOKEN_BUCKET,
            fallback_strategy=RateLimitStrategy.SLIDING_WINDOW,
            
            # 用户级别配置
            user_specific_limits=True,
            default_user_limit=100,
            premium_user_limit=500,
            
            # 惩罚配置
            violation_penalty_seconds=300,  # 5分钟
            progressive_penalty=True
        )

    @pytest.fixture
    def rate_limiter(self, rate_limit_config):
        """创建限流器"""
        return RateLimiter(config=rate_limit_config)

    def test_rate_limiter_initialization(self, rate_limiter, rate_limit_config):
        """测试限流器初始化"""
        assert rate_limiter.config == rate_limit_config
        assert rate_limiter.token_bucket is not None
        assert rate_limiter.sliding_window is not None
        assert rate_limiter.fixed_window is not None
        assert rate_limiter.concurrency_limiter is not None

    def test_allow_request_success(self, rate_limiter):
        """测试允许请求成功"""
        client_id = 'client_001'
        
        result = rate_limiter.allow_request(client_id, request_weight=1)
        
        assert result.allowed == True
        assert result.remaining_quota > 0
        assert result.reset_time is not None
        assert result.retry_after is None

    def test_allow_request_rate_limited(self, rate_limiter):
        """测试请求被限流"""
        client_id = 'client_001'
        
        # 快速发送大量请求直到被限制
        requests_sent = 0
        while requests_sent < 200:  # 避免无限循环
            result = rate_limiter.allow_request(client_id, request_weight=1)
            requests_sent += 1
            
            if not result.allowed:
                assert result.remaining_quota == 0
                assert result.retry_after > 0
                break
        
        # 至少应该处理了一些请求才被限制
        assert requests_sent > 50

    def test_user_specific_rate_limits(self, rate_limiter):
        """测试用户特定限流"""
        regular_user = 'regular_user_001'
        premium_user = 'premium_user_001'
        
        # 设置用户限制
        rate_limiter.set_user_limit(regular_user, 50)
        rate_limiter.set_user_limit(premium_user, 200)
        
        # 测试常规用户限制
        regular_requests = 0
        while regular_requests < 100:
            result = rate_limiter.allow_request(regular_user)
            regular_requests += 1
            if not result.allowed:
                break
        
        # 测试高级用户限制
        premium_requests = 0
        while premium_requests < 250:
            result = rate_limiter.allow_request(premium_user)
            premium_requests += 1
            if not result.allowed:
                break
        
        # 高级用户应该能发送更多请求
        assert premium_requests > regular_requests

    def test_request_weight_handling(self, rate_limiter):
        """测试请求权重处理"""
        client_id = 'client_001'
        
        # 发送高权重请求
        heavy_result = rate_limiter.allow_request(client_id, request_weight=10)
        assert heavy_result.allowed == True
        
        # 发送轻权重请求
        light_result = rate_limiter.allow_request(client_id, request_weight=1)
        assert light_result.allowed == True
        
        # 高权重请求应该消耗更多配额
        assert heavy_result.remaining_quota < light_result.remaining_quota + 9

    def test_rate_limit_reset(self, rate_limiter):
        """测试限流重置"""
        client_id = 'client_001'
        
        # 发送请求直到被限制
        while True:
            result = rate_limiter.allow_request(client_id)
            if not result.allowed:
                break
        
        # 重置客户端限制
        rate_limiter.reset_client_limits(client_id)
        
        # 应该能再次发送请求
        result = rate_limiter.allow_request(client_id)
        assert result.allowed == True

    def test_get_client_statistics(self, rate_limiter):
        """测试获取客户端统计"""
        client_id = 'client_001'
        
        # 发送一些请求
        for i in range(20):
            rate_limiter.allow_request(client_id)
        
        stats = rate_limiter.get_client_statistics(client_id)
        
        assert stats['total_requests'] == 20
        assert stats['allowed_requests'] <= 20
        assert stats['rejected_requests'] >= 0
        assert 'first_request_time' in stats
        assert 'last_request_time' in stats

    def test_global_rate_limiting(self, rate_limiter):
        """测试全局限流"""
        # 启用全局限流
        rate_limiter.enable_global_rate_limit(max_global_requests_per_second=50)
        
        start_time = time.time()
        allowed_count = 0
        
        # 快速发送大量请求
        for i in range(100):
            result = rate_limiter.allow_request(f'client_{i}')
            if result.allowed:
                allowed_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 验证全局限流生效
        requests_per_second = allowed_count / duration
        assert requests_per_second <= 60  # 允许一些误差


class TestQuotaManager:
    """配额管理器测试类"""

    @pytest.fixture
    def quota_manager(self):
        """创建配额管理器"""
        return QuotaManager()

    def test_set_user_quota(self, quota_manager):
        """测试设置用户配额"""
        user_id = 'user_001'
        quota = {
            'daily_requests': 10000,
            'hourly_requests': 1000,
            'concurrent_requests': 10,
            'data_transfer_mb': 1000
        }
        
        quota_manager.set_user_quota(user_id, quota)
        
        retrieved_quota = quota_manager.get_user_quota(user_id)
        assert retrieved_quota == quota

    def test_consume_quota(self, quota_manager):
        """测试消费配额"""
        user_id = 'user_001'
        quota = {
            'daily_requests': 1000,
            'hourly_requests': 100
        }
        
        quota_manager.set_user_quota(user_id, quota)
        
        # 消费配额
        result = quota_manager.consume_quota(user_id, 'daily_requests', 50)
        assert result == True
        
        # 检查剩余配额
        remaining = quota_manager.get_remaining_quota(user_id, 'daily_requests')
        assert remaining == 950

    def test_quota_exceeded(self, quota_manager):
        """测试配额超限"""
        user_id = 'user_001'
        quota = {'daily_requests': 100}
        
        quota_manager.set_user_quota(user_id, quota)
        
        # 消费配额直到超限
        result = quota_manager.consume_quota(user_id, 'daily_requests', 150)
        assert result == False
        
        # 配额应该没有变化
        remaining = quota_manager.get_remaining_quota(user_id, 'daily_requests')
        assert remaining == 100

    def test_quota_reset_schedule(self, quota_manager):
        """测试配额重置调度"""
        user_id = 'user_001'
        quota = {'hourly_requests': 100}
        
        quota_manager.set_user_quota(user_id, quota)
        
        # 消费一些配额
        quota_manager.consume_quota(user_id, 'hourly_requests', 50)
        assert quota_manager.get_remaining_quota(user_id, 'hourly_requests') == 50
        
        # 模拟时间过去（1小时后）
        with patch('time.time', return_value=time.time() + 3600):
            quota_manager.reset_expired_quotas()
            
            # 配额应该被重置
            remaining = quota_manager.get_remaining_quota(user_id, 'hourly_requests')
            assert remaining == 100


class TestThrottleManager:
    """节流管理器测试类"""

    @pytest.fixture
    def throttle_manager(self):
        """创建节流管理器"""
        return ThrottleManager()

    def test_client_throttling(self, throttle_manager):
        """测试客户端节流"""
        client_id = 'client_001'
        
        # 设置节流：每秒最多5个请求
        throttle_manager.set_client_throttle(client_id, requests_per_second=5)
        
        start_time = time.time()
        allowed_count = 0
        
        # 快速发送10个请求
        for i in range(10):
            if throttle_manager.allow_request(client_id):
                allowed_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 验证节流效果
        if duration < 1.0:
            # 如果在1秒内完成，应该只允许5个请求
            assert allowed_count <= 5
        else:
            # 如果超过1秒，可能允许更多请求
            expected_max = int(duration * 5) + 1
            assert allowed_count <= expected_max

    def test_progressive_throttling(self, throttle_manager):
        """测试渐进式节流"""
        client_id = 'client_001'
        
        # 启用渐进式节流
        throttle_manager.enable_progressive_throttling(client_id)
        
        # 发送违规请求
        violation_count = 0
        for i in range(20):
            if not throttle_manager.allow_request(client_id):
                violation_count += 1
                throttle_manager.record_violation(client_id)
        
        # 检查节流级别是否增加
        throttle_level = throttle_manager.get_throttle_level(client_id)
        assert throttle_level > 1

    def test_throttle_recovery(self, throttle_manager):
        """测试节流恢复"""
        client_id = 'client_001'
        
        # 设置严格节流
        throttle_manager.set_client_throttle(client_id, requests_per_second=1)
        
        # 记录违规
        for i in range(5):
            throttle_manager.record_violation(client_id)
        
        initial_level = throttle_manager.get_throttle_level(client_id)
        
        # 等待恢复期
        time.sleep(2)
        throttle_manager.update_throttle_levels()
        
        # 节流级别应该有所恢复
        current_level = throttle_manager.get_throttle_level(client_id)
        assert current_level <= initial_level

    def test_whitelist_bypass(self, throttle_manager):
        """测试白名单绕过"""
        client_id = 'whitelisted_client'
        
        # 将客户端加入白名单
        throttle_manager.add_to_whitelist(client_id)
        
        # 设置严格节流
        throttle_manager.set_client_throttle(client_id, requests_per_second=1)
        
        # 快速发送多个请求
        allowed_count = 0
        for i in range(10):
            if throttle_manager.allow_request(client_id):
                allowed_count += 1
        
        # 白名单客户端应该绕过节流
        assert allowed_count == 10


class TestDistributedRateLimiter:
    """分布式限流器测试类"""

    @pytest.fixture
    def distributed_limiter(self):
        """创建分布式限流器"""
        # 模拟Redis连接
        mock_redis = Mock()
        return DistributedRateLimiter(redis_client=mock_redis)

    def test_distributed_token_bucket(self, distributed_limiter):
        """测试分布式令牌桶"""
        client_id = 'client_001'
        
        # 模拟Redis操作
        distributed_limiter.redis_client.eval.return_value = 1  # 成功消费
        
        result = distributed_limiter.allow_request(client_id, algorithm='token_bucket')
        
        assert result.allowed == True
        distributed_limiter.redis_client.eval.assert_called_once()

    def test_distributed_sliding_window(self, distributed_limiter):
        """测试分布式滑动窗口"""
        client_id = 'client_001'
        
        # 模拟Redis操作
        distributed_limiter.redis_client.zcard.return_value = 50  # 当前请求数
        distributed_limiter.redis_client.zadd.return_value = 1
        
        result = distributed_limiter.allow_request(client_id, algorithm='sliding_window')
        
        assert result.allowed == True
        distributed_limiter.redis_client.zcard.assert_called_once()

    def test_cross_instance_coordination(self, distributed_limiter):
        """测试跨实例协调"""
        client_id = 'client_001'
        
        # 模拟多个实例的操作
        instance_ids = ['instance_1', 'instance_2', 'instance_3']
        
        for instance_id in instance_ids:
            # 每个实例报告其限流状态
            distributed_limiter.report_instance_status(
                instance_id=instance_id,
                client_limits={'client_001': {'requests': 50, 'capacity': 100}}
            )
        
        # 获取全局限流状态
        global_status = distributed_limiter.get_global_limits(client_id)
        
        assert global_status['total_requests'] == 150  # 3个实例 * 50请求
        assert global_status['total_capacity'] == 300   # 3个实例 * 100容量

    def test_redis_failover_handling(self, distributed_limiter):
        """测试Redis故障转移处理"""
        client_id = 'client_001'
        
        # 模拟Redis连接失败
        distributed_limiter.redis_client.eval.side_effect = ConnectionError("Redis connection failed")
        
        # 应该回退到本地限流
        result = distributed_limiter.allow_request(client_id, fallback_to_local=True)
        
        # 即使Redis失败，也应该有结果（本地限流）
        assert result is not None
        assert hasattr(result, 'allowed')


class TestRateLimitPerformance:
    """限流性能测试类"""

    def test_high_throughput_token_bucket(self):
        """测试令牌桶高吞吐量性能"""
        token_bucket = TokenBucket(capacity=10000, refill_rate=1000)
        
        start_time = time.time()
        success_count = 0
        
        # 快速消费10000个令牌
        for i in range(10000):
            if token_bucket.consume(1):
                success_count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能检查：应该在100ms内完成
        assert duration < 0.1
        assert success_count == 10000

    def test_concurrent_rate_limiting_performance(self):
        """测试并发限流性能"""
        rate_limiter = RateLimiter(RateLimitConfig(
            token_bucket_capacity=10000,
            token_bucket_refill_rate=1000,
            max_concurrent_requests=100
        ))
        
        success_count = 0
        lock = threading.Lock()
        
        def make_requests():
            nonlocal success_count
            for i in range(50):
                result = rate_limiter.allow_request(f'client_{threading.current_thread().ident}')
                if result.allowed:
                    with lock:
                        success_count += 1
        
        start_time = time.time()
        
        # 创建20个线程，每个发送50个请求
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=make_requests)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能检查
        assert duration < 2.0  # 应该在2秒内完成
        assert success_count > 0  # 应该有成功的请求
        
        # 计算吞吐量
        throughput = success_count / duration
        assert throughput > 500  # 每秒至少500个请求

    def test_memory_usage_under_load(self):
        """测试负载下的内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 创建大量客户端的限流器
        rate_limiter = RateLimiter(RateLimitConfig())
        
        # 为1000个不同客户端发送请求
        for i in range(1000):
            client_id = f'client_{i:04d}'
            for j in range(10):
                rate_limiter.allow_request(client_id)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内（小于100MB）
        assert memory_increase < 100 * 1024 * 1024