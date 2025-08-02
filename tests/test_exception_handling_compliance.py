#!/usr/bin/env python3
"""
异常处理合规性测试
验证代码严格遵守规则1：禁止捕获异常后吞掉不处理
"""

import pytest
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from rl_trading_system.data.qlib_interface import QlibDataInterface
from rl_trading_system.data.feature_engineer import FeatureEngineer


class TestExceptionHandlingCompliance:
    """测试异常处理合规性"""
    
    def test_risk_penalty_calculation_raises_on_error(self):
        """测试风险惩罚计算在错误时正确抛出异常"""
        print("=== 测试风险惩罚计算异常处理合规性 ===")
        
        from rl_trading_system.risk_control.risk_controller import RiskController, RiskControlConfig
        
        # 直接创建一个最小化的测试场景
        portfolio_config = PortfolioConfig(
            stock_pool=['A'],
            initial_cash=1000000,
            enable_risk_control=True
        )
        
        # 直接创建环境对象的风险惩罚方法测试
        environment = Mock()
        environment.risk_controller = Mock(spec=RiskController)
        environment.current_positions = np.array([0.5])
        environment.current_prices = np.array([10.0])
        environment.total_value = 1000000
        environment.cash = 500000
        
        # 将实际的方法绑定到mock对象
        from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment
        environment._calculate_risk_penalty = PortfolioEnvironment._calculate_risk_penalty.__get__(environment)
        environment._build_portfolio_for_risk_check = PortfolioEnvironment._build_portfolio_for_risk_check.__get__(environment)
        environment._build_trades_for_risk_check = PortfolioEnvironment._build_trades_for_risk_check.__get__(environment)
        environment.config = portfolio_config
        
        # 模拟风险控制器方法失败
        environment.risk_controller.assess_portfolio_risk.side_effect = ValueError("模拟的风险评估错误")
        
        # 验证异常被正确抛出而不是被吞掉
        with pytest.raises(RuntimeError) as exc_info:
            environment._calculate_risk_penalty(np.array([0.6]))
        
        # 验证异常消息包含原始错误信息
        assert "风险惩罚计算模块出现错误" in str(exc_info.value)
        assert "模拟的风险评估错误" in str(exc_info.value)
        
        print("✅ 风险惩罚计算正确抛出异常，不吞掉错误")
    
    def test_risk_check_raises_on_error(self):
        """测试风险检查在错误时正确抛出异常"""
        print("=== 测试风险检查异常处理合规性 ===")
        
        from rl_trading_system.risk_control.risk_controller import RiskController, RiskControlConfig
        
        # 直接创建一个最小化的测试场景
        portfolio_config = PortfolioConfig(
            stock_pool=['A'],
            initial_cash=1000000,
            enable_risk_control=True
        )
        
        # 直接创建环境对象的风险检查方法测试
        environment = Mock()
        environment.risk_controller = Mock(spec=RiskController)
        environment.current_positions = np.array([0.5])
        environment.current_prices = np.array([10.0])
        environment.total_value = 1000000
        environment.cash = 500000
        
        # 将实际的方法绑定到mock对象
        from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment
        environment._check_trading_risks = PortfolioEnvironment._check_trading_risks.__get__(environment)
        environment._build_portfolio_for_risk_check = PortfolioEnvironment._build_portfolio_for_risk_check.__get__(environment)
        environment._build_trades_for_risk_check = PortfolioEnvironment._build_trades_for_risk_check.__get__(environment)
        environment.config = portfolio_config
        
        # 模拟风险控制器方法失败
        environment.risk_controller.check_trade_risks = Mock(side_effect=AttributeError("模拟的风险检查错误"))
        
        # 验证异常被正确抛出而不是被吞掉
        with pytest.raises(RuntimeError) as exc_info:
            environment._check_trading_risks(np.array([0.6]))
        
        # 验证异常消息包含原始错误信息
        assert "风险检查模块出现错误" in str(exc_info.value)
        assert "模拟的风险检查错误" in str(exc_info.value)
        
        print("✅ 风险检查正确抛出异常，不吞掉错误")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])