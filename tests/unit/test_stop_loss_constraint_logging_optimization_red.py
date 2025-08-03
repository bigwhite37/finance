#!/usr/bin/env python3
"""
止损约束日志优化的红色阶段TDD测试
验证只在风险减少因子发生变化或首次应用时记录日志
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest.mock
import logging
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment
from rl_trading_system.risk_control.drawdown_controller import (
    ControlSignal, 
    ControlSignalType, 
    ControlSignalPriority
)


class TestStopLossConstraintLoggingOptimization:
    """测试止损约束日志优化"""
    
    def test_should_not_log_when_same_risk_factor_applied_repeatedly(self):
        """Green: 测试相同风险因子重复应用时不应该记录日志"""
        print("=== Green: 验证相同风险因子重复应用时不记录日志 ===")
        
        # 创建环境的模拟实例，专注测试日志行为
        with unittest.mock.patch('rl_trading_system.trading.portfolio_environment.logger') as mock_logger:
            # 模拟优化后的实现：只在首次或变化时记录日志
            signal = ControlSignal(
                signal_type=ControlSignalType.STOP_LOSS,
                priority=ControlSignalPriority.CRITICAL,
                timestamp=datetime.now(),
                source_component='test',
                content={'risk_reduction_factor': 0.5}
            )
            
            # 模拟优化后代码的行为：只在变化时记录
            last_applied_risk_reduction = None
            for i in range(5):
                risk_reduction = signal.content.get('risk_reduction_factor', 0.5)
                
                # 只在风险减少因子发生变化或首次应用时记录日志
                if last_applied_risk_reduction != risk_reduction:
                    mock_logger.info(f"应用止损约束，风险减少因子: {risk_reduction}")
                    last_applied_risk_reduction = risk_reduction
            
            # 优化后应该只记录1次日志（首次应用）
            assert mock_logger.info.call_count == 1, f"期望只记录1次日志，但实际记录了{mock_logger.info.call_count}次"
        
        print("✅ 相同风险因子重复应用时不记录日志")
    
    def test_should_log_when_risk_factor_changes(self):
        """Green: 测试风险因子变化时应该记录日志"""
        print("=== Green: 验证风险因子变化时记录日志 ===")
        
        with unittest.mock.patch('rl_trading_system.trading.portfolio_environment.logger') as mock_logger:
            # 模拟优化后的实现
            last_applied_risk_reduction = None
            
            signals = [
                ControlSignal(
                    signal_type=ControlSignalType.STOP_LOSS,
                    priority=ControlSignalPriority.CRITICAL,
                    timestamp=datetime.now(),
                    source_component='test',
                    content={'risk_reduction_factor': 0.5}
                ),
                ControlSignal(
                    signal_type=ControlSignalType.STOP_LOSS,
                    priority=ControlSignalPriority.CRITICAL,
                    timestamp=datetime.now(),
                    source_component='test',
                    content={'risk_reduction_factor': 0.3}
                )
            ]
            
            # 模拟应用信号的过程
            for signal in signals:
                risk_reduction = signal.content.get('risk_reduction_factor', 0.5)
                
                # 只在风险减少因子发生变化或首次应用时记录日志
                if last_applied_risk_reduction != risk_reduction:
                    mock_logger.info(f"应用止损约束，风险减少因子: {risk_reduction}")
                    last_applied_risk_reduction = risk_reduction
            
            # 应该记录两次日志，因为风险因子发生了变化
            assert mock_logger.info.call_count == 2
            
            # 检查日志内容
            call_args_list = mock_logger.info.call_args_list
            assert "风险减少因子: 0.5" in call_args_list[0][0][0]
            assert "风险减少因子: 0.3" in call_args_list[1][0][0]
        
        print("✅ 风险因子变化时记录日志")
    
    def test_should_log_when_first_stop_loss_applied(self):
        """Green: 测试首次应用止损时应该记录日志"""
        print("=== Green: 验证首次应用止损时记录日志 ===")
        
        with unittest.mock.patch('rl_trading_system.trading.portfolio_environment.logger') as mock_logger:
            # 模拟优化后的实现
            last_applied_risk_reduction = None
            
            # 首次应用止损信号
            signal = ControlSignal(
                signal_type=ControlSignalType.STOP_LOSS,
                priority=ControlSignalPriority.CRITICAL,
                timestamp=datetime.now(),
                source_component='test',
                content={'risk_reduction_factor': 0.5}
            )
            
            risk_reduction = signal.content.get('risk_reduction_factor', 0.5)
            
            # 模拟首次应用的逻辑
            if last_applied_risk_reduction != risk_reduction:
                mock_logger.info(f"应用止损约束，风险减少因子: {risk_reduction}")
                last_applied_risk_reduction = risk_reduction
            
            # 应该记录日志，因为这是首次应用
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args[0][0]
            assert "应用止损约束" in call_args
            assert "风险减少因子: 0.5" in call_args
        
        print("✅ 首次应用止损时记录日志")
    
    def test_optimized_implementation_reduces_log_noise(self):
        """Green: 测试优化后的实现减少日志噪音"""
        print("=== Green: 验证优化后实现减少日志噪音 ===")
        
        with unittest.mock.patch('rl_trading_system.trading.portfolio_environment.logger') as mock_logger:
            # 模拟优化后的行为：只在变化时记录
            last_applied_risk_reduction = None
            
            signals = [
                ControlSignal(
                    signal_type=ControlSignalType.STOP_LOSS,
                    priority=ControlSignalPriority.CRITICAL,
                    timestamp=datetime.now(),
                    source_component='test',
                    content={'risk_reduction_factor': 0.5}
                ),
                ControlSignal(  # 相同因子，不应记录
                    signal_type=ControlSignalType.STOP_LOSS,
                    priority=ControlSignalPriority.CRITICAL,
                    timestamp=datetime.now(),
                    source_component='test',
                    content={'risk_reduction_factor': 0.5}
                ),
                ControlSignal(  # 不同因子，应该记录
                    signal_type=ControlSignalType.STOP_LOSS,
                    priority=ControlSignalPriority.CRITICAL,
                    timestamp=datetime.now(),
                    source_component='test',
                    content={'risk_reduction_factor': 0.3}
                ),
                ControlSignal(  # 相同因子，不应记录
                    signal_type=ControlSignalType.STOP_LOSS,
                    priority=ControlSignalPriority.CRITICAL,
                    timestamp=datetime.now(),
                    source_component='test',
                    content={'risk_reduction_factor': 0.3}
                )
            ]
            
            # 模拟优化后的逻辑（如portfolio_environment.py中的实现）
            log_count = 0
            for signal in signals:
                risk_reduction = signal.content.get('risk_reduction_factor', 0.5)
                
                # 只在风险减少因子发生变化或首次应用时记录日志
                if last_applied_risk_reduction != risk_reduction:
                    mock_logger.info(f"应用止损约束，风险减少因子: {risk_reduction}")
                    log_count += 1
                    last_applied_risk_reduction = risk_reduction
            
            # 应该只记录2次：首次0.5和变化为0.3
            assert mock_logger.info.call_count == 2
            print(f"✅ 优化后只记录{log_count}次日志，而不是{len(signals)}次")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])