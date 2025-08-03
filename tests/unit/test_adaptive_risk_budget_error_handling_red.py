#!/usr/bin/env python3
"""
自适应风险预算错误处理的红色阶段TDD测试
验证数据不足时应该抛出RuntimeError而不是返回默认值
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.risk_control.adaptive_risk_budget import (
    AdaptiveRiskBudget, 
    AdaptiveRiskBudgetConfig,
    PerformanceMetrics,
    MarketMetrics
)


class TestAdaptiveRiskBudgetErrorHandling:
    """测试自适应风险预算的错误处理"""
    
    def test_calculate_adaptive_risk_budget_should_raise_error_when_no_performance_data(self):
        """Red: 测试当缺少表现数据时应该抛出RuntimeError"""
        print("=== Red: 验证缺少表现数据时抛出RuntimeError ===")
        
        config = AdaptiveRiskBudgetConfig()
        adaptive_budget = AdaptiveRiskBudget(config)
        
        # 只提供市场数据，没有表现数据
        market_metrics = MarketMetrics(
            market_volatility=0.15,
            market_trend=0.05,
            correlation_with_market=0.8
        )
        adaptive_budget.update_market_metrics(market_metrics)
        
        # 当缺少表现数据时，应该抛出RuntimeError而不是返回默认值
        with pytest.raises(RuntimeError, match="无法获取必要的表现指标数据"):
            adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        print("✅ 缺少表现数据时正确抛出RuntimeError")
    
    def test_calculate_adaptive_risk_budget_should_raise_error_when_no_market_data(self):
        """Red: 测试当缺少市场数据时应该抛出RuntimeError"""
        print("=== Red: 验证缺少市场数据时抛出RuntimeError ===")
        
        config = AdaptiveRiskBudgetConfig()
        adaptive_budget = AdaptiveRiskBudget(config)
        
        # 只提供表现数据，没有市场数据
        performance_metrics = PerformanceMetrics(
            sharpe_ratio=1.2,
            calmar_ratio=1.5,
            max_drawdown=0.08,
            volatility=0.12
        )
        adaptive_budget.update_performance_metrics(performance_metrics)
        
        # 当缺少市场数据时，应该抛出RuntimeError而不是返回默认值
        with pytest.raises(RuntimeError, match="无法获取必要的市场指标数据"):
            adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        print("✅ 缺少市场数据时正确抛出RuntimeError")
    
    def test_calculate_adaptive_risk_budget_should_raise_error_when_no_data(self):
        """Red: 测试当完全没有数据时应该抛出RuntimeError"""
        print("=== Red: 验证完全没有数据时抛出RuntimeError ===")
        
        config = AdaptiveRiskBudgetConfig()
        adaptive_budget = AdaptiveRiskBudget(config)
        
        # 不提供任何数据
        
        # 当完全没有数据时，应该抛出RuntimeError而不是返回默认值
        with pytest.raises(RuntimeError, match="无法获取必要的.*数据"):
            adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
        
        print("✅ 完全没有数据时正确抛出RuntimeError")
    
    def test_current_implementation_violates_rules(self):
        """Red: 测试当前实现违反规则（仅返回默认值而不抛出异常）"""
        print("=== Red: 验证当前实现违反规则 ===")
        
        config = AdaptiveRiskBudgetConfig()
        adaptive_budget = AdaptiveRiskBudget(config)
        
        # 不提供任何数据，当前实现会返回默认值而不是抛出异常
        # 这违反了规则6：无法获取数据时立即 raise RuntimeError(...)
        
        # 当前的错误实现会返回current_risk_budget而不是抛出异常
        try:
            result = adaptive_budget.calculate_adaptive_risk_budget(force_update=True)
            # 如果没有抛出异常，说明违反了规则
            assert False, f"违反规则：应该抛出RuntimeError，但返回了默认值 {result}"
        except RuntimeError:
            # 这是期望的正确行为
            print("✅ 正确抛出了RuntimeError")
        except Exception as e:
            assert False, f"抛出了错误类型的异常: {type(e).__name__}: {e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])