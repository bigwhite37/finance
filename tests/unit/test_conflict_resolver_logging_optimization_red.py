#!/usr/bin/env python3
"""
冲突解决器日志优化的红色阶段TDD测试
验证只在有真正冲突时记录日志
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest.mock
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.risk_control.drawdown_controller import (
    ConflictResolver, 
    ControlSignal, 
    ControlSignalType, 
    ControlSignalPriority
)


class TestConflictResolverLoggingOptimization:
    """测试冲突解决器的日志优化"""
    
    def test_should_not_log_when_no_conflicts(self):
        """Red: 测试没有冲突时不应该记录日志"""
        print("=== Red: 验证没有冲突时不记录日志 ===")
        
        resolver = ConflictResolver()
        
        # 创建两个不同类型的信号（不冲突）
        signal1 = ControlSignal(
            signal_type=ControlSignalType.REWARD_OPTIMIZATION,
            priority=ControlSignalPriority.LOW,
            timestamp=datetime.now(),
            source_component='test1',
            content={}
        )
        
        signal2 = ControlSignal(
            signal_type=ControlSignalType.MARKET_REGIME_CHANGE,
            priority=ControlSignalPriority.MEDIUM,
            timestamp=datetime.now(),
            source_component='test2',
            content={}
        )
        
        with unittest.mock.patch.object(resolver.logger, 'info') as mock_info:
            resolved_signals = resolver.resolve_conflicts([signal1, signal2])
            
            # 当前实现会记录日志，但我们希望在没有冲突时不记录
            # 这个测试应该失败，因为当前实现总是记录日志
            mock_info.assert_not_called()
        
        assert len(resolved_signals) == 2
        print("✅ 没有冲突时不记录日志")
    
    def test_should_log_when_same_type_signals_conflict(self):
        """Red: 测试同类型信号冲突时应该记录日志"""
        print("=== Red: 验证同类型信号冲突时记录日志 ===")
        
        resolver = ConflictResolver()
        
        # 创建两个相同类型的信号（有冲突）
        signal1 = ControlSignal(
            signal_type=ControlSignalType.STOP_LOSS,
            priority=ControlSignalPriority.CRITICAL,
            timestamp=datetime.now(),
            source_component='test1',
            content={}
        )
        
        signal2 = ControlSignal(
            signal_type=ControlSignalType.STOP_LOSS,
            priority=ControlSignalPriority.HIGH,
            timestamp=datetime.now(),
            source_component='test2',
            content={}
        )
        
        with unittest.mock.patch.object(resolver.logger, 'info') as mock_info:
            resolved_signals = resolver.resolve_conflicts([signal1, signal2])
            
            # 应该记录冲突解决的日志
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "冲突解决" in call_args
            assert "输入2个信号，输出1个信号" in call_args  # 2个信号冲突，解决后剩1个
        
        assert len(resolved_signals) == 1  # 冲突解决后只保留一个
        print("✅ 同类型信号冲突时记录日志")
    
    def test_should_log_when_expired_signals_removed(self):
        """Red: 测试移除过期信号时应该记录日志"""
        print("=== Red: 验证移除过期信号时记录日志 ===")
        
        resolver = ConflictResolver()
        
        # 创建一个过期信号和一个有效信号
        past_time = datetime.now() - timedelta(minutes=10)
        expiry_time = past_time + timedelta(minutes=5)  # 过期时间是5分钟前
        expired_signal = ControlSignal(
            signal_type=ControlSignalType.STOP_LOSS,
            priority=ControlSignalPriority.HIGH,
            timestamp=past_time,
            source_component='test1',
            content={},
            expiry_time=expiry_time  # 已过期
        )
        
        valid_signal = ControlSignal(
            signal_type=ControlSignalType.REWARD_OPTIMIZATION,
            priority=ControlSignalPriority.LOW,
            timestamp=datetime.now(),
            source_component='test2',
            content={}
        )
        
        with unittest.mock.patch.object(resolver.logger, 'info') as mock_info:
            resolved_signals = resolver.resolve_conflicts([expired_signal, valid_signal])
            
            # 应该记录移除过期信号的日志
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "冲突解决" in call_args
            assert "输入2个信号，输出1个信号" in call_args  # 过期信号被移除
        
        assert len(resolved_signals) == 1  # 过期信号被移除
        print("✅ 移除过期信号时记录日志")
    
    def test_optimized_implementation_does_not_log_without_conflicts(self):
        """Green: 测试优化后的实现在没有冲突时不记录日志"""
        print("=== Green: 验证优化后没有冲突时不记录日志 ===")
        
        resolver = ConflictResolver()
        
        # 创建两个不冲突的信号
        signal1 = ControlSignal(
            signal_type=ControlSignalType.REWARD_OPTIMIZATION,
            priority=ControlSignalPriority.LOW,
            timestamp=datetime.now(),
            source_component='test1',
            content={}
        )
        
        signal2 = ControlSignal(
            signal_type=ControlSignalType.MARKET_REGIME_CHANGE,
            priority=ControlSignalPriority.MEDIUM,
            timestamp=datetime.now(),
            source_component='test2',
            content={}
        )
        
        with unittest.mock.patch.object(resolver.logger, 'info') as mock_info:
            resolved_signals = resolver.resolve_conflicts([signal1, signal2])
            
            # 优化后的实现不会在没有冲突时记录日志
            mock_info.assert_not_called()
        
        assert len(resolved_signals) == 2
        print("✅ 优化后没有冲突时不记录日志")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])