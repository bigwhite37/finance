"""
日志优化工具
"""

import time
from collections import defaultdict, deque
from typing import Dict, Optional
import logging


class LogThrottler:
    """日志限制器 - 避免重复日志刷屏"""
    
    def __init__(self):
        # 记录每个消息的最后记录时间
        self._last_logged = {}
        # 记录每个消息的统计计数
        self._message_counts = defaultdict(int)
        # 记录窗口内的消息（用于频率控制）
        self._message_windows = defaultdict(lambda: deque())
        
    def should_log(self, 
                   message_key: str, 
                   min_interval: float = 5.0,
                   max_per_minute: int = 3) -> bool:
        """
        检查是否应该记录日志
        
        Args:
            message_key: 消息的唯一标识
            min_interval: 同样消息的最小间隔时间（秒）
            max_per_minute: 每分钟最大记录次数
            
        Returns:
            是否应该记录日志
        """
        current_time = time.time()
        
        # 检查最小间隔限制
        if message_key in self._last_logged:
            if current_time - self._last_logged[message_key] < min_interval:
                self._message_counts[message_key] += 1
                return False
        
        # 检查频率限制
        window = self._message_windows[message_key]
        # 清理1分钟之前的记录
        while window and window[0] < current_time - 60:
            window.popleft()
            
        if len(window) >= max_per_minute:
            self._message_counts[message_key] += 1
            return False
        
        # 记录此次日志
        self._last_logged[message_key] = current_time
        window.append(current_time)
        
        return True
        
    def get_suppressed_count(self, message_key: str) -> int:
        """获取被抑制的日志数量"""
        return self._message_counts.get(message_key, 0)
        
    def reset_count(self, message_key: str):
        """重置计数"""
        self._message_counts[message_key] = 0


class StatisticalLogger:
    """统计性日志记录器 - 定期汇总而不是逐个记录"""
    
    def __init__(self, report_interval: int = 100):
        self.report_interval = report_interval  # 每N次事件报告一次
        self.stats = defaultdict(int)
        self.details = defaultdict(list)
        self.last_report = {}
        
    def log_event(self, event_type: str, details: str = None):
        """记录事件，但不立即输出日志"""
        self.stats[event_type] += 1
        
        if details and len(self.details[event_type]) < 5:  # 只保留前5个详情作为样例
            self.details[event_type].append(details)
            
    def should_report(self, event_type: str) -> bool:
        """检查是否应该生成报告"""
        return self.stats[event_type] % self.report_interval == 0
        
    def get_report(self, event_type: str) -> str:
        """生成统计报告"""
        count = self.stats[event_type]
        samples = self.details[event_type][:3]  # 显示前3个样例
        
        report = f"{event_type}: 发生 {count} 次"
        if samples:
            report += f", 样例: {'; '.join(samples)}"
            
        return report
        
    def reset_stats(self, event_type: str = None):
        """重置统计"""
        if event_type:
            self.stats[event_type] = 0
            self.details[event_type].clear()
        else:
            self.stats.clear()
            self.details.clear()


# 全局实例
throttler = LogThrottler()
stat_logger = StatisticalLogger()


def throttled_warning(logger: logging.Logger, 
                     message: str, 
                     message_key: str = None,
                     min_interval: float = 5.0,
                     max_per_minute: int = 3):
    """
    受限制的warning日志
    
    Args:
        logger: 日志器实例
        message: 日志消息
        message_key: 消息唯一标识（如果为None则使用message）
        min_interval: 最小间隔时间
        max_per_minute: 每分钟最大次数
    """
    if message_key is None:
        message_key = message
        
    if throttler.should_log(message_key, min_interval, max_per_minute):
        suppressed = throttler.get_suppressed_count(message_key)
        if suppressed > 0:
            logger.warning(f"{message} (已抑制 {suppressed} 条相似消息)")
            throttler.reset_count(message_key)
        else:
            logger.warning(message)


def statistical_warning(logger: logging.Logger,
                       event_type: str,
                       details: str = None,
                       report_interval: int = 100):
    """
    统计性warning日志
    
    Args:
        logger: 日志器实例
        event_type: 事件类型
        details: 事件详情
        report_interval: 报告间隔
    """
    stat_logger.log_event(event_type, details)
    
    if stat_logger.should_report(event_type):
        report = stat_logger.get_report(event_type)
        logger.warning(f"[统计报告] {report}")