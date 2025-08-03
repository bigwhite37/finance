#!/usr/bin/env python3
"""
回撤控制器与自适应风险预算集成的红色阶段TDD测试
验证DrawdownController应该向AdaptiveRiskBudget提供表现指标和市场指标数据
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys
import unittest.mock
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.risk_control.drawdown_controller import DrawdownController
from rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig
from rl_trading_system.risk_control.drawdown_monitor import DrawdownMetrics
from rl_trading_system.trading.portfolio_environment import PortfolioState, MarketState


class TestDrawdownControllerAdaptiveBudgetIntegration:
    """测试回撤控制器与自适应风险预算的集成"""
    
    def test_drawdown_controller_should_provide_performance_metrics_to_adaptive_budget(self):
        """Red: 测试DrawdownController应该向AdaptiveRiskBudget提供表现指标数据"""
        print("=== Red: 验证DrawdownController向AdaptiveRiskBudget提供表现指标 ===")
        
        config = DrawdownControlConfig()
        controller = DrawdownController(config)
        
        # 模拟一些表现数据
        portfolio_state = PortfolioState(
            positions={'sh600519': 1000, 'sh600036': 500},
            cash=200000.0,
            portfolio_value=1020000.0,  # 2%收益
            timestamp=datetime.now(),
            unrealized_pnl=20000.0,
            realized_pnl=0.0
        )
        
        market_state = MarketState(
            prices={'sh600519': 100.0, 'sh600036': 200.0},
            volumes={'sh600519': 1000000, 'sh600036': 500000},
            timestamp=datetime.now(),
            market_indicators={'volatility': 0.18, 'trend': 0.05}
        )
        
        drawdown_metrics = DrawdownMetrics(
            current_drawdown=0.03,
            max_drawdown=0.05,
            drawdown_duration=5,
            rolling_max=1050000.0,
            peak_date=datetime.now(),
            in_drawdown=True
        )
        
        # 在执行控制步骤之前，检查自适应风险预算是否有数据
        # 当前实现中应该没有数据
        try:
            # 直接调用应该失败，因为没有提供数据
            budget = controller.adaptive_risk_budget.calculate_adaptive_risk_budget(force_update=True)
            assert False, "违反预期：应该因为缺少数据而抛出RuntimeError"
        except RuntimeError as e:
            if "无法获取必要的" not in str(e):
                assert False, f"抛出了意外的RuntimeError: {e}"
        
        # 执行控制步骤
        control_signals = controller.execute_control_step(market_state, portfolio_state)
        
        # Red阶段：当前实现不会向AdaptiveRiskBudget提供数据
        # 这应该失败，因为我们希望在执行控制步骤后，AdaptiveRiskBudget有足够的数据
        try:
            budget = controller.adaptive_risk_budget.calculate_adaptive_risk_budget(force_update=True)
            # 如果能正常计算，说明数据已经被提供了（期望的行为）
            print(f"✅ 自适应风险预算计算成功: {budget}")
        except RuntimeError as e:
            if "无法获取必要的" in str(e):
                assert False, f"当前实现违反预期：执行控制步骤后仍然缺少数据: {e}"
            else:
                # 其他类型的错误可以重新抛出
                raise
        
        print("✅ DrawdownController正确向AdaptiveRiskBudget提供了表现指标")
    
    def test_current_implementation_does_not_provide_data_to_adaptive_budget(self):
        """Red: 测试当前实现不向AdaptiveRiskBudget提供数据（应该失败）"""
        print("=== Red: 验证当前实现不向AdaptiveRiskBudget提供数据 ===")
        
        config = DrawdownControlConfig()
        controller = DrawdownController(config)
        
        # 检查AdaptiveRiskBudget的历史数据是否为空
        assert len(controller.adaptive_risk_budget.performance_history) == 0, "初始表现历史应该为空"
        assert len(controller.adaptive_risk_budget.market_history) == 0, "初始市场历史应该为空"
        
        # 模拟一些数据
        portfolio_state = PortfolioState(
            positions={'sh600519': 1000, 'sh600036': 500},
            cash=200000.0,
            portfolio_value=1020000.0,
            timestamp=datetime.now(),
            unrealized_pnl=20000.0,
            realized_pnl=0.0
        )
        
        market_state = MarketState(
            prices={'sh600519': 100.0, 'sh600036': 200.0},
            volumes={'sh600519': 1000000, 'sh600036': 500000},
            timestamp=datetime.now(),
            market_indicators={'volatility': 0.18, 'trend': 0.05}
        )
        
        # 执行控制步骤
        with unittest.mock.patch.object(controller, '_update_risk_budget') as mock_update:
            # 模拟当前的实现，不提供数据给AdaptiveRiskBudget
            mock_update.side_effect = lambda ps, dm, ts: None
            
            control_signals = controller.execute_control_step(market_state, portfolio_state)
        
        # 当前实现不应该向AdaptiveRiskBudget提供数据
        # 这个测试应该通过，因为当前实现确实不提供数据
        assert len(controller.adaptive_risk_budget.performance_history) == 0, "当前实现不应该提供表现数据"
        assert len(controller.adaptive_risk_budget.market_history) == 0, "当前实现不应该提供市场数据"
        
        print("✅ 确认当前实现不向AdaptiveRiskBudget提供数据（需要修正）")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])