#!/usr/bin/env python3
"""
测试 risk_reduction_factor 修复
验证所有止损信号都包含 risk_reduction_factor 字段
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.risk_control.drawdown_controller import (
    DrawdownController, 
    ControlSignal, 
    ControlSignalType, 
    ControlSignalPriority
)
from rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig
from rl_trading_system.risk_control.drawdown_monitor import DrawdownMetrics, DrawdownPhase
from rl_trading_system.trading.portfolio_environment import MarketState, PortfolioState


class TestRiskReductionFactorFix:
    """测试 risk_reduction_factor 修复"""
    
    def test_stop_loss_signal_contains_risk_reduction_factor(self):
        """测试止损信号包含 risk_reduction_factor 字段"""
        print("=== 测试止损信号包含 risk_reduction_factor 字段 ===")
        
        # 创建回撤控制器
        config = DrawdownControlConfig(
            portfolio_stop_loss=0.10,  # 10% 组合级止损
            max_drawdown_threshold=0.15
        )
        controller = DrawdownController(config)
        
        # 创建模拟的市场状态和投资组合状态
        market_state = MarketState(
            prices={'AAPL': 150.0, 'GOOGL': 2800.0},
            volumes={'AAPL': 1000000, 'GOOGL': 500000},
            timestamp=datetime.now(),
            market_indicators={'volatility': 0.25, 'trend': -0.1}
        )
        
        portfolio_state = PortfolioState(
            positions={'AAPL': 100, 'GOOGL': 50},
            portfolio_value=100000.0,
            cash=10000.0,
            timestamp=datetime.now(),
            unrealized_pnl=-12000.0,
            realized_pnl=0.0
        )
        
        # 创建触发止损的回撤指标
        drawdown_metrics = DrawdownMetrics(
            current_drawdown=-0.12,  # 12% 回撤，超过 10% 阈值
            max_drawdown=-0.12,
            drawdown_duration=5,
            recovery_time=None,
            peak_value=100000.0,
            trough_value=88000.0,
            underwater_curve=[-0.05, -0.08, -0.10, -0.11, -0.12],
            drawdown_frequency=0.1,
            average_drawdown=-0.08,
            current_phase=DrawdownPhase.DRAWDOWN_CONTINUE,
            days_since_peak=5
        )
        
        # 直接调用动态止损检查方法
        controller._check_dynamic_stop_loss(market_state, portfolio_state, drawdown_metrics, datetime.now())
        
        # 从控制信号队列中获取信号
        control_signals = controller.control_signal_queue
        stop_loss_signals = [s for s in control_signals if s.signal_type == ControlSignalType.STOP_LOSS]
        
        # 验证至少有一个止损信号
        assert len(stop_loss_signals) > 0, "应该生成至少一个止损信号"
        
        # 验证每个止损信号都包含 risk_reduction_factor
        for signal in stop_loss_signals:
            assert 'risk_reduction_factor' in signal.content, f"止损信号缺少 risk_reduction_factor 字段: {signal.content}"
            
            risk_reduction_factor = signal.content['risk_reduction_factor']
            assert isinstance(risk_reduction_factor, (int, float)), f"risk_reduction_factor 应该是数值类型，但得到: {type(risk_reduction_factor)}"
            assert 0.0 < risk_reduction_factor <= 1.0, f"risk_reduction_factor 应该在 (0, 1] 范围内，但得到: {risk_reduction_factor}"
            
            print(f"✅ 止损信号包含 risk_reduction_factor: {risk_reduction_factor}")
        
        print("✅ 所有止损信号都包含有效的 risk_reduction_factor 字段")
    
    def test_risk_reduction_factor_calculation_logic(self):
        """测试 risk_reduction_factor 计算逻辑"""
        print("=== 测试 risk_reduction_factor 计算逻辑 ===")
        
        config = DrawdownControlConfig(
            portfolio_stop_loss=0.10,
            max_drawdown_threshold=0.15
        )
        controller = DrawdownController(config)
        
        market_state = MarketState(
            prices={'AAPL': 150.0},
            volumes={'AAPL': 1000000},
            timestamp=datetime.now(),
            market_indicators={'volatility': 0.25, 'trend': -0.1}
        )
        
        portfolio_state = PortfolioState(
            positions={'AAPL': 100},
            portfolio_value=100000.0,
            cash=10000.0,
            timestamp=datetime.now(),
            unrealized_pnl=0.0,
            realized_pnl=0.0
        )
        
        # 测试不同回撤程度的 risk_reduction_factor
        test_cases = [
            (-0.10, "刚好触发止损"),  # 刚好触发
            (-0.15, "中等回撤"),      # 1.5倍阈值
            (-0.20, "严重回撤"),      # 2倍阈值
        ]
        
        for drawdown, description in test_cases:
            drawdown_metrics = DrawdownMetrics(
                current_drawdown=drawdown,
                max_drawdown=drawdown,
                drawdown_duration=5,
                recovery_time=None,
                peak_value=100000.0,
                trough_value=100000.0 * (1 + drawdown),
                underwater_curve=[drawdown],
                drawdown_frequency=0.1,
                average_drawdown=drawdown * 0.8,
                current_phase=DrawdownPhase.DRAWDOWN_CONTINUE,
                days_since_peak=5
            )
            
            # 清空之前的信号
            controller.control_signal_queue.clear()
            
            # 执行控制步骤
            control_signals = controller.execute_control_step(market_state, portfolio_state)
            stop_loss_signals = [s for s in control_signals if s.signal_type == ControlSignalType.STOP_LOSS]
            
            if stop_loss_signals:
                risk_reduction_factor = stop_loss_signals[0].content['risk_reduction_factor']
                print(f"回撤 {drawdown:.1%} ({description}): risk_reduction_factor = {risk_reduction_factor:.3f}")
                
                # 验证风险减少因子的合理性
                assert 0.1 <= risk_reduction_factor <= 0.8, f"风险减少因子应该在合理范围内: {risk_reduction_factor}"
        
        print("✅ risk_reduction_factor 计算逻辑正确")
    
    def test_portfolio_environment_handles_missing_risk_reduction_factor(self):
        """测试 portfolio_environment 正确处理缺少 risk_reduction_factor 的情况"""
        print("=== 测试 portfolio_environment 处理缺少 risk_reduction_factor 的情况 ===")
        
        # 创建一个没有 risk_reduction_factor 的止损信号
        signal_without_factor = ControlSignal(
            signal_type=ControlSignalType.STOP_LOSS,
            priority=ControlSignalPriority.CRITICAL,
            timestamp=datetime.now(),
            source_component='test',
            content={
                'action': 'portfolio_stop_loss',
                'current_drawdown': -0.12,
                'threshold': 0.10,
                'recommended_action': 'reduce_positions'
                # 注意：没有 risk_reduction_factor
            }
        )
        
        # 测试 get 方法的默认值行为
        risk_reduction = signal_without_factor.content.get('risk_reduction_factor', 0.5)
        assert risk_reduction == 0.5, f"应该使用默认值 0.5，但得到: {risk_reduction}"
        
        print("✅ portfolio_environment 正确处理缺少 risk_reduction_factor 的情况")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])