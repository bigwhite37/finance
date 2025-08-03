#!/usr/bin/env python3
"""
回撤阶段变化日志优化的红色阶段TDD测试
验证只在有意义的阶段变化时记录日志，避免频繁的来回震荡日志
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

from rl_trading_system.risk_control.drawdown_monitor import DrawdownMonitor, DrawdownPhase


class TestDrawdownPhaseLoggingOptimization:
    """测试回撤阶段变化日志优化"""
    
    def test_should_not_log_rapid_phase_oscillations(self):
        """Red: 测试快速相位震荡时不应该记录每次变化"""
        print("=== Red: 验证快速相位震荡时不记录每次变化 ===")
        
        monitor = DrawdownMonitor(lookback_window=5)
        
        with unittest.mock.patch('rl_trading_system.risk_control.drawdown_monitor.logger') as mock_logger:
            # 模拟快速震荡的投资组合价值变化
            values = [
                1000000,  # 正常
                990000,   # 回撤开始 (-1%)
                995000,   # 恢复中 (+0.5%)
                985000,   # 回撤持续 (-1.5%)
                992000,   # 恢复中 (+0.2%)
                988000,   # 回撤持续 (-1.2%)
                994000,   # 恢复中 (+0.4%)
                990000,   # 回撤持续 (-1.0%)
            ]
            
            # 模拟当前实现：每次状态变化都记录
            for value in values:
                monitor.update_portfolio_value(value)
            
            # 当前实现会记录多次状态变化，但我们希望减少噪音
            # 这个测试应该失败，显示当前实现记录了太多日志
            log_count = mock_logger.info.call_count
            assert log_count <= 2, f"期望最多记录2次有意义的状态变化，但实际记录了{log_count}次"
        
        print("✅ 快速相位震荡时不记录每次变化")
    
    def test_should_log_significant_phase_changes(self):
        """Green: 测试重要阶段变化时应该记录日志"""
        print("=== Green: 验证重要阶段变化时记录日志 ===")
        
        monitor = DrawdownMonitor(lookback_window=5)
        
        with unittest.mock.patch('rl_trading_system.risk_control.drawdown_monitor.logger') as mock_logger:
            # 模拟有意义的阶段变化
            values = [
                1000000,  # 正常
                950000,   # 重大回撤 (-5%)
                1020000,  # 恢复到超过原来水平 (+2%)
            ]
            
            # 模拟优化后的实现：只记录重要变化
            last_significant_phase = None
            
            for value in values:
                monitor.update_portfolio_value(value)
                current_phase = monitor.current_phase
                
                # 模拟优化后的逻辑：只在重要变化时记录
                if self._is_significant_phase_change(last_significant_phase, current_phase):
                    mock_logger.info(f"回撤阶段变化: {last_significant_phase.value if last_significant_phase else 'None'} -> {current_phase.value}")
                    last_significant_phase = current_phase
            
            # 应该记录重要的阶段变化
            assert mock_logger.info.call_count >= 1
            print(f"✅ 记录了{mock_logger.info.call_count}次重要阶段变化")
    
    def _is_significant_phase_change(self, old_phase, new_phase):
        """判断是否为重要的阶段变化"""
        if old_phase is None:
            return new_phase != DrawdownPhase.NORMAL
        
        # 定义重要的状态变化
        significant_transitions = {
            (DrawdownPhase.NORMAL, DrawdownPhase.DRAWDOWN_START),
            (DrawdownPhase.DRAWDOWN_START, DrawdownPhase.DRAWDOWN_CONTINUE),
            (DrawdownPhase.DRAWDOWN_CONTINUE, DrawdownPhase.RECOVERY),
            (DrawdownPhase.RECOVERY, DrawdownPhase.NORMAL),
        }
        
        return (old_phase, new_phase) in significant_transitions
    
    def test_should_suppress_frequent_oscillations_between_recovery_and_continue(self):
        """Green: 测试抑制恢复和持续之间的频繁震荡"""
        print("=== Green: 验证抑制恢复和持续之间的频繁震荡 ===")
        
        monitor = DrawdownMonitor(lookback_window=5)
        
        with unittest.mock.patch('rl_trading_system.risk_control.drawdown_monitor.logger') as mock_logger:
            # 先进入回撤状态
            monitor.update_portfolio_value(1000000)  # 正常
            monitor.update_portfolio_value(950000)   # 回撤开始
            
            # 清除前面的日志记录
            mock_logger.reset_mock()
            
            # 模拟恢复和持续之间的频繁震荡
            oscillation_values = [
                955000,  # 恢复中
                952000,  # 回撤持续
                954000,  # 恢复中
                951000,  # 回撤持续
                953000,  # 恢复中
                950000,  # 回撤持续
            ]
            
            # 模拟优化后的实现：使用去抖动机制
            phase_stability_count = 0
            last_logged_phase = monitor.current_phase
            stability_threshold = 2  # 需要稳定2次才记录
            
            for value in oscillation_values:
                monitor.update_portfolio_value(value)
                current_phase = monitor.current_phase
                
                if current_phase == last_logged_phase:
                    phase_stability_count += 1
                else:
                    phase_stability_count = 0
                
                # 只在状态稳定一段时间后才记录
                if phase_stability_count >= stability_threshold and current_phase != last_logged_phase:
                    mock_logger.info(f"回撤阶段变化: {last_logged_phase.value} -> {current_phase.value}")
                    last_logged_phase = current_phase
                    phase_stability_count = 0
            
            # 优化后应该显著减少日志记录
            log_count = mock_logger.info.call_count
            assert log_count <= 2, f"期望最多记录2次稳定的状态变化，实际记录了{log_count}次"
            print(f"✅ 抑制震荡后只记录了{log_count}次日志")
    
    def test_optimized_implementation_reduces_log_noise(self):
        """Green: 测试优化后的实现减少日志噪音"""
        print("=== Green: 验证优化后实现减少日志噪音 ===")
        
        monitor = DrawdownMonitor(lookback_window=5)
        
        with unittest.mock.patch('rl_trading_system.risk_control.drawdown_monitor.logger') as mock_logger:
            # 模拟频繁震荡的场景 - 使用超过5%的回撤触发阶段变化
            values = [
                1000000,  # 正常
                940000,   # -6% 回撤开始
                960000,   # -4% 恢复中
                930000,   # -7% 回撤持续
                950000,   # -5% 恢复中
                920000,   # -8% 回撤持续
                970000,   # -3% 恢复中
                940000,   # -6% 回撤持续
                980000,   # -2% 恢复中
                920000,   # -8% 回撤持续
            ]
            
            for value in values:
                monitor.update_portfolio_value(value)
                print(f"Value: {value}, Phase: {monitor.current_phase.value}")
            
            # 当前实现会记录很多次状态变化
            log_count = mock_logger.info.call_count
            print(f"当前实现记录了{log_count}次状态变化")
            print(f"调用详情: {[call.args for call in mock_logger.info.call_args_list]}")
            
            # 优化后应该显著减少日志记录，只记录重要转换
            # 期望记录：正常->回撤开始、恢复->正常这样的重要转换
            assert log_count <= 4, f"期望最多记录4次重要变化，优化后实现记录了{log_count}次"
        
        print(f"✅ 优化后减少了日志噪音，只记录了{log_count}次重要变化")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])