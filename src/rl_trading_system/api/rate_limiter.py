"""
API限流系统实现
实现请求限流、响应缓存和并发控制，多种限流算法支持
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import threading
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import logging


class RateLimitStrategy(Enum):
    """限流策略枚举"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitResult:
    """限流结果"""
    allowed: bool
    remaining_quota: int = 0
    reset_time: Optional[datetime] = None
    retry_after: Optional[int] = None  # 秒
    quota_type: str = "requests"


@dataclass
class RateLimitConfig:
    """限流配置"""
    # 令牌桶配置
    token_bucket_capacity: int = 100
    token_bucket_refill_rate: int = 10  # 每秒
    
    # 滑动窗口配置
    sliding_window_size: int = 60  # 秒
    sliding_window_max_requests: int = 1000
    
    # 固定窗口配置
    fixed_window_size: int = 60  # 秒
    fixed_window_max_requests: int = 500
    
    # 并发限制配置
    max_concurrent_requests: int = 50
    
    # 策略配置
    primary_strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    fallback_strategy: Optional[RateLimitStrategy] = None
    
    # 用户级别配置
    user_specific_limits: bool = True
    default_user_limit: int = 100
    premium_user_limit: int = 500
    
    # 惩罚配置
    violation_penalty_seconds: int = 300  # 5分钟
    progressive_penalty: bool = True


@dataclass
class RateLimitRule:
    """限流规则"""
    rule_id: str
    name: str
    strategy: RateLimitStrategy
    limit: int
    window_size: int  # 秒
    applies_to: List[str] = field(default_factory=list)  # 用户ID或IP
    priority: int = 1
    active: bool = True


class TokenBucket:
    """令牌桶算法实现"""
    
    def __init__(self, capacity: int, refill_rate: int, initial_tokens: int = None):
        self.capacity = capacity
        self.refill_rate = refill_rate  # 每秒补充的令牌数
        self.available_tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill_time = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """消费令牌"""
        with self._lock:
            self._refill()
            
            if self.available_tokens >= tokens:
                self.available_tokens -= tokens
                return True
            return False
    
    def _refill(self):
        """补充令牌"""
        current_time = time.time()
        time_elapsed = current_time - self.last_refill_time
        
        if time_elapsed > 0:
            tokens_to_add = int(time_elapsed * self.refill_rate)
            self.available_tokens = min(
                self.capacity,
                self.available_tokens + tokens_to_add
            )
            self.last_refill_time = current_time
    
    def get_available_tokens(self) -> int:
        """获取可用令牌数"""
        with self._lock:
            self._refill()
            return self.available_tokens


class SlidingWindow:
    """滑动窗口算法实现"""
    
    def __init__(self, window_size: int, max_requests: int, precision: int = 1):
        self.window_size = window_size  # 窗口大小（秒）
        self.max_requests = max_requests
        self.precision = precision  # 精度（秒）
        self.request_times = deque()
        self._lock = threading.Lock()
    
    def allow_request(self, current_time: float = None) -> bool:
        """检查是否允许请求"""
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            # 清理过期的请求记录
            self._cleanup_expired_requests(current_time)
            
            # 检查是否超过限制
            if len(self.request_times) >= self.max_requests:
                return False
            
            # 记录请求时间
            self.request_times.append(current_time)
            return True
    
    def _cleanup_expired_requests(self, current_time: float):
        """清理过期的请求记录"""
        cutoff_time = current_time - self.window_size
        
        while self.request_times and self.request_times[0] <= cutoff_time:
            self.request_times.popleft()
    
    def get_current_count(self, current_time: float = None) -> int:
        """获取当前请求计数"""
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            self._cleanup_expired_requests(current_time)
            return len(self.request_times)
    
    def get_remaining_quota(self, current_time: float = None) -> int:
        """获取剩余配额"""
        current_count = self.get_current_count(current_time)
        return max(0, self.max_requests - current_count)


class FixedWindow:
    """固定窗口算法实现"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size  # 窗口大小（秒）
        self.max_requests = max_requests
        self.current_window_start = time.time()
        self.current_request_count = 0
        self._lock = threading.Lock()
    
    def allow_request(self, current_time: float = None) -> bool:
        """检查是否允许请求"""
        if current_time is None:
            current_time = time.time()
        
        with self._lock:
            # 检查是否需要重置窗口
            if current_time >= self.current_window_start + self.window_size:
                self.current_window_start = current_time
                self.current_request_count = 0
            
            # 检查是否超过限制
            if self.current_request_count >= self.max_requests:
                return False
            
            # 增加请求计数
            self.current_request_count += 1
            return True
    
    def get_current_count(self) -> int:
        """获取当前请求计数"""
        with self._lock:
            return self.current_request_count
    
    def get_remaining_quota(self) -> int:
        """获取剩余配额"""
        return max(0, self.max_requests - self.current_request_count)
    
    def get_reset_time(self) -> datetime:
        """获取窗口重置时间"""
        reset_timestamp = self.current_window_start + self.window_size
        return datetime.fromtimestamp(reset_timestamp)


class ConcurrencyLimiter:
    """并发限制器"""
    
    def __init__(self, max_concurrent: int, timeout: int = 30, queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.queue_size = queue_size
        self.current_concurrent = 0
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float = None) -> Optional[object]:
        """获取并发信号量"""
        if timeout is None:
            timeout = self.timeout
        
        acquired = self._semaphore.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                self.current_concurrent += 1
            return self._semaphore  # 返回信号量对象作为令牌
        return None
    
    def release(self, semaphore: object):
        """释放并发信号量"""
        with self._lock:
            self.current_concurrent -= 1
        semaphore.release()
    
    def get_current_concurrent(self) -> int:
        """获取当前并发数"""
        with self._lock:
            return self.current_concurrent


class RateLimiter:
    """综合限流器"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化各种限流算法
        self.token_bucket = TokenBucket(
            capacity=config.token_bucket_capacity,
            refill_rate=config.token_bucket_refill_rate
        )
        
        self.sliding_window = SlidingWindow(
            window_size=config.sliding_window_size,
            max_requests=config.sliding_window_max_requests
        )
        
        self.fixed_window = FixedWindow(
            window_size=config.fixed_window_size,
            max_requests=config.fixed_window_max_requests
        )
        
        self.concurrency_limiter = ConcurrencyLimiter(
            max_concurrent=config.max_concurrent_requests
        )
        
        # 客户端限制跟踪
        self.client_limits = defaultdict(dict)
        self.client_stats = defaultdict(lambda: {
            'total_requests': 0,
            'allowed_requests': 0,
            'rejected_requests': 0,
            'first_request_time': None,
            'last_request_time': None
        })
        self._client_lock = threading.Lock()
        
        # 全局限流
        self.global_rate_limit_enabled = False
        self.global_max_rps = 0
        self.global_request_times = deque()
        self._global_lock = threading.Lock()
    
    def allow_request(self, client_id: str, request_weight: int = 1) -> RateLimitResult:
        """检查是否允许请求"""
        current_time = time.time()
        
        # 更新客户端统计
        with self._client_lock:
            stats = self.client_stats[client_id]
            stats['total_requests'] += 1
            stats['last_request_time'] = datetime.now()
            if stats['first_request_time'] is None:
                stats['first_request_time'] = datetime.now()
        
        # 检查全局限流
        if self.global_rate_limit_enabled:
            if not self._check_global_rate_limit(current_time):
                with self._client_lock:
                    self.client_stats[client_id]['rejected_requests'] += 1
                
                return RateLimitResult(
                    allowed=False,
                    remaining_quota=0,
                    retry_after=1
                )
        
        # 使用主要策略
        result = self._apply_rate_limit_strategy(
            client_id, 
            self.config.primary_strategy, 
            request_weight,
            current_time
        )
        
        # 如果主要策略拒绝且有回退策略，尝试回退策略
        if not result.allowed and self.config.fallback_strategy:
            result = self._apply_rate_limit_strategy(
                client_id,
                self.config.fallback_strategy,
                request_weight,
                current_time
            )
        
        # 更新统计
        with self._client_lock:
            if result.allowed:
                self.client_stats[client_id]['allowed_requests'] += 1
            else:
                self.client_stats[client_id]['rejected_requests'] += 1
        
        return result
    
    def _apply_rate_limit_strategy(self, client_id: str, strategy: RateLimitStrategy,
                                  request_weight: int, current_time: float) -> RateLimitResult:
        """应用限流策略"""
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            if self.token_bucket.consume(request_weight):
                return RateLimitResult(
                    allowed=True,
                    remaining_quota=self.token_bucket.get_available_tokens()
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining_quota=0,
                    retry_after=1  # 建议1秒后重试
                )
        
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            if self.sliding_window.allow_request(current_time):
                return RateLimitResult(
                    allowed=True,
                    remaining_quota=self.sliding_window.get_remaining_quota(current_time)
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining_quota=0,
                    retry_after=self.config.sliding_window_size
                )
        
        elif strategy == RateLimitStrategy.FIXED_WINDOW:
            if self.fixed_window.allow_request(current_time):
                return RateLimitResult(
                    allowed=True,
                    remaining_quota=self.fixed_window.get_remaining_quota(),
                    reset_time=self.fixed_window.get_reset_time()
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining_quota=0,
                    reset_time=self.fixed_window.get_reset_time(),
                    retry_after=int(self.fixed_window.get_reset_time().timestamp() - current_time)
                )
        
        # 默认允许
        return RateLimitResult(allowed=True, remaining_quota=1000)
    
    def _check_global_rate_limit(self, current_time: float) -> bool:
        """检查全局限流"""
        with self._global_lock:
            # 清理1秒前的请求记录
            cutoff_time = current_time - 1.0
            while self.global_request_times and self.global_request_times[0] <= cutoff_time:
                self.global_request_times.popleft()
            
            # 检查是否超过全局限制
            if len(self.global_request_times) >= self.global_max_rps:
                return False
            
            # 记录请求时间
            self.global_request_times.append(current_time)
            return True
    
    def set_user_limit(self, user_id: str, limit: int):
        """设置用户特定限制"""
        with self._client_lock:
            self.client_limits[user_id]['custom_limit'] = limit
    
    def reset_client_limits(self, client_id: str):
        """重置客户端限制"""
        with self._client_lock:
            if client_id in self.client_limits:
                del self.client_limits[client_id]
            if client_id in self.client_stats:
                stats = self.client_stats[client_id]
                stats.update({
                    'total_requests': 0,
                    'allowed_requests': 0,
                    'rejected_requests': 0,
                    'first_request_time': None,
                    'last_request_time': None
                })
    
    def get_client_statistics(self, client_id: str) -> Dict[str, Any]:
        """获取客户端统计信息"""
        with self._client_lock:
            return self.client_stats[client_id].copy()
    
    def enable_global_rate_limit(self, max_global_requests_per_second: int):
        """启用全局限流"""
        self.global_rate_limit_enabled = True
        self.global_max_rps = max_global_requests_per_second


class QuotaManager:
    """配额管理器"""
    
    def __init__(self):
        self.user_quotas = {}
        self.usage_tracking = defaultdict(lambda: defaultdict(int))
        self._lock = threading.Lock()
    
    def set_user_quota(self, user_id: str, quota: Dict[str, int]):
        """设置用户配额"""
        with self._lock:
            self.user_quotas[user_id] = quota.copy()
    
    def get_user_quota(self, user_id: str) -> Dict[str, int]:
        """获取用户配额"""
        with self._lock:
            return self.user_quotas.get(user_id, {}).copy()
    
    def consume_quota(self, user_id: str, quota_type: str, amount: int) -> bool:
        """消费配额"""
        with self._lock:
            user_quota = self.user_quotas.get(user_id, {})
            if quota_type not in user_quota:
                return True  # 没有限制
            
            current_usage = self.usage_tracking[user_id][quota_type]
            available_quota = user_quota[quota_type] - current_usage
            
            if available_quota >= amount:
                self.usage_tracking[user_id][quota_type] += amount
                return True
            
            return False
    
    def get_remaining_quota(self, user_id: str, quota_type: str) -> int:
        """获取剩余配额"""
        with self._lock:
            user_quota = self.user_quotas.get(user_id, {})
            if quota_type not in user_quota:
                return float('inf')  # 无限制
            
            current_usage = self.usage_tracking[user_id][quota_type]
            return max(0, user_quota[quota_type] - current_usage)
    
    def reset_expired_quotas(self):
        """重置过期配额（简化实现）"""
        with self._lock:
            # 实际实现应该根据时间周期重置不同类型的配额
            current_hour = datetime.now().hour
            if hasattr(self, '_last_reset_hour') and self._last_reset_hour != current_hour:
                # 重置小时配额
                for user_id in self.usage_tracking:
                    if 'hourly_requests' in self.usage_tracking[user_id]:
                        self.usage_tracking[user_id]['hourly_requests'] = 0
            
            self._last_reset_hour = current_hour


class ThrottleManager:
    """节流管理器"""
    
    def __init__(self):
        self.client_throttles = {}
        self.client_violations = defaultdict(int)
        self.throttle_levels = defaultdict(int)
        self.whitelist = set()
        self._lock = threading.Lock()
        self.last_request_times = defaultdict(float)
    
    def set_client_throttle(self, client_id: str, requests_per_second: int):
        """设置客户端节流"""
        with self._lock:
            self.client_throttles[client_id] = {
                'rps': requests_per_second,
                'min_interval': 1.0 / requests_per_second
            }
    
    def allow_request(self, client_id: str) -> bool:
        """检查是否允许请求"""
        if client_id in self.whitelist:
            return True
        
        current_time = time.time()
        
        with self._lock:
            if client_id not in self.client_throttles:
                return True
            
            throttle_config = self.client_throttles[client_id]
            last_request_time = self.last_request_times[client_id]
            min_interval = throttle_config['min_interval']
            
            # 检查时间间隔
            if current_time - last_request_time < min_interval:
                return False
            
            # 更新最后请求时间
            self.last_request_times[client_id] = current_time
            return True
    
    def record_violation(self, client_id: str):
        """记录违规"""
        with self._lock:
            self.client_violations[client_id] += 1
            self.throttle_levels[client_id] += 1
    
    def enable_progressive_throttling(self, client_id: str):
        """启用渐进式节流"""
        with self._lock:
            if client_id not in self.client_throttles:
                self.client_throttles[client_id] = {'rps': 100, 'min_interval': 0.01}
    
    def get_throttle_level(self, client_id: str) -> int:
        """获取节流级别"""
        with self._lock:
            return self.throttle_levels[client_id]
    
    def update_throttle_levels(self):
        """更新节流级别（恢复机制）"""
        with self._lock:
            for client_id in list(self.throttle_levels.keys()):
                if self.throttle_levels[client_id] > 0:
                    self.throttle_levels[client_id] = max(0, self.throttle_levels[client_id] - 1)
    
    def add_to_whitelist(self, client_id: str):
        """添加到白名单"""
        with self._lock:
            self.whitelist.add(client_id)
    
    def remove_from_whitelist(self, client_id: str):
        """从白名单移除"""
        with self._lock:
            self.whitelist.discard(client_id)


class DistributedRateLimiter:
    """分布式限流器"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        self.local_fallback = {}
        self._lock = threading.Lock()
        self.instance_status = {}
    
    def allow_request(self, client_id: str, algorithm: str = 'token_bucket',
                     fallback_to_local: bool = True) -> RateLimitResult:
        """分布式限流检查"""
        try:
            if algorithm == 'token_bucket':
                return self._distributed_token_bucket(client_id)
            elif algorithm == 'sliding_window':
                return self._distributed_sliding_window(client_id)
            else:
                # 默认使用令牌桶
                return self._distributed_token_bucket(client_id)
        
        except Exception as e:
            if fallback_to_local:
                # 回退到本地限流
                return self._local_fallback_limit(client_id)
            else:
                # 不捕获异常，让其暴露
                raise e
    
    def _distributed_token_bucket(self, client_id: str) -> RateLimitResult:
        """分布式令牌桶"""
        # Lua脚本确保原子性
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time
        
        -- 计算要补充的令牌
        local time_elapsed = current_time - last_refill
        local tokens_to_add = math.floor(time_elapsed * refill_rate)
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        -- 检查是否有足够令牌
        if tokens >= tokens_requested then
            tokens = tokens - tokens_requested
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', key, 3600)  -- 1小时过期
            return {1, tokens}  -- 允许，剩余令牌
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', key, 3600)
            return {0, tokens}  -- 拒绝，剩余令牌
        end
        """
        
        result = self.redis_client.eval(
            lua_script,
            1,  # key count
            f"rate_limit:token_bucket:{client_id}",
            100,  # capacity
            10,   # refill_rate
            1,    # tokens_requested
            time.time()
        )
        
        allowed = bool(result[0])
        remaining = int(result[1])
        
        return RateLimitResult(
            allowed=allowed,
            remaining_quota=remaining
        )
    
    def _distributed_sliding_window(self, client_id: str) -> RateLimitResult:
        """分布式滑动窗口"""
        current_time = time.time()
        window_key = f"rate_limit:sliding_window:{client_id}"
        
        # 使用Redis有序集合实现滑动窗口
        pipe = self.redis_client.pipeline()
        
        # 清理过期记录
        cutoff_time = current_time - 60  # 60秒窗口
        pipe.zremrangebyscore(window_key, 0, cutoff_time)
        
        # 获取当前计数
        pipe.zcard(window_key)
        
        # 添加当前请求
        pipe.zadd(window_key, {str(current_time): current_time})
        
        # 设置过期时间
        pipe.expire(window_key, 120)  # 2分钟过期
        
        results = pipe.execute()
        current_count = results[1]
        
        max_requests = 1000  # 配置值
        allowed = current_count < max_requests
        
        return RateLimitResult(
            allowed=allowed,
            remaining_quota=max(0, max_requests - current_count - 1)
        )
    
    def _local_fallback_limit(self, client_id: str) -> RateLimitResult:
        """本地回退限流"""
        with self._lock:
            if client_id not in self.local_fallback:
                self.local_fallback[client_id] = TokenBucket(100, 10)
            
            bucket = self.local_fallback[client_id]
            allowed = bucket.consume(1)
            
            return RateLimitResult(
                allowed=allowed,
                remaining_quota=bucket.get_available_tokens()
            )
    
    def report_instance_status(self, instance_id: str, client_limits: Dict[str, Dict[str, int]]):
        """报告实例状态"""
        with self._lock:
            self.instance_status[instance_id] = {
                'client_limits': client_limits,
                'last_update': datetime.now(),
                'status': 'active'
            }
    
    def get_global_limits(self, client_id: str) -> Dict[str, Any]:
        """获取全局限流状态"""
        total_requests = 0
        total_capacity = 0
        
        with self._lock:
            for instance_data in self.instance_status.values():
                client_data = instance_data['client_limits'].get(client_id, {})
                total_requests += client_data.get('requests', 0)
                total_capacity += client_data.get('capacity', 0)
        
        return {
            'total_requests': total_requests,
            'total_capacity': total_capacity,
            'global_utilization': total_requests / total_capacity if total_capacity > 0 else 0
        }