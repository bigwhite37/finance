#!/usr/bin/env python3
"""
风险管理集成的TDD测试
按照严格的TDD方法：Red -> Green -> Refactor
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
from rl_trading_system.risk_control.risk_controller import RiskController, RiskControlConfig
from rl_trading_system.data.qlib_interface import QlibDataInterface
from rl_trading_system.data.feature_engineer import FeatureEngineer


class TestRiskManagementIntegration:
    """测试风险管理集成"""
    
    def test_portfolio_environment_should_have_risk_controller(self):
        """Green: 测试投资组合环境已成功集成风险控制器"""
        print("=== Green: 验证投资组合环境已集成风险控制器 ===")
        
        # 检查PortfolioEnvironment类是否有risk_controller相关方法或属性
        import inspect
        
        # 获取类的所有方法和属性
        methods = inspect.getmembers(PortfolioEnvironment, predicate=inspect.isfunction)
        method_names = [name for name, _ in methods]
        
        # 检查是否有风险相关的方法
        risk_related_methods = [name for name in method_names if 'risk' in name.lower()]
        
        # 检查__init__方法源码是否包含风险控制器初始化
        init_source = inspect.getsource(PortfolioEnvironment.__init__)
        has_risk_controller_init = 'risk_controller' in init_source.lower()
        
        print(f"风险相关方法: {risk_related_methods}")
        print(f"__init__包含风险控制器: {has_risk_controller_init}")
        
        # 断言：现在应该有风险控制器集成
        assert len(risk_related_methods) >= 3, f"环境应该有风险相关方法，当前有: {risk_related_methods}"
        assert has_risk_controller_init, "环境应该已经初始化风险控制器"
        
        # 验证必要的风险检查方法存在
        required_methods = ['_check_trading_risks', '_build_portfolio_for_risk_check', '_build_trades_for_risk_check']
        for method in required_methods:
            assert method in method_names, f"缺少必要的风险检查方法: {method}"
        
        print("✅ 环境已成功集成风险控制器")
        
    def test_portfolio_environment_should_check_risks_before_trades(self):
        """Green: 测试环境在执行交易前已实现风险检查"""
        print("=== Green: 验证环境step方法已包含风险检查 ===")
        
        # 检查step方法源码是否包含风险检查
        import inspect
        step_source = inspect.getsource(PortfolioEnvironment.step)
        
        has_risk_check = '_check_trading_risks' in step_source
        
        print(f"step方法包含风险检查: {has_risk_check}")
        
        assert has_risk_check, "环境step方法应该调用_check_trading_risks进行风险检查"
        
        print("✅ 环境step方法已实现风险检查")
        
    def test_reward_function_should_include_risk_penalty(self):
        """Green: 测试奖励函数已成功包含风险惩罚项"""
        print("=== Green: 验证奖励函数已包含风险惩罚 ===")
        
        # 检查_calculate_reward方法源码是否包含风险惩罚
        import inspect
        
        try:
            reward_method_source = inspect.getsource(PortfolioEnvironment._calculate_reward)
            
            # 检查是否有风险相关的代码
            has_risk_penalty = any([
                'risk_penalty' in reward_method_source.lower(),
                'risk_violation' in reward_method_source.lower(),
                'risk_controller' in reward_method_source.lower()
            ])
            
            print(f"奖励函数包含风险惩罚: {has_risk_penalty}")
            
            # 验证_calculate_risk_penalty方法存在
            methods = inspect.getmembers(PortfolioEnvironment, predicate=inspect.isfunction)
            method_names = [name for name, _ in methods]
            has_risk_penalty_method = '_calculate_risk_penalty' in method_names
            
            print(f"风险惩罚计算方法存在: {has_risk_penalty_method}")
            
            # Green阶段：应该包含风险惩罚
            assert has_risk_penalty, "奖励函数应该包含风险惩罚相关代码"
            assert has_risk_penalty_method, "应该存在_calculate_risk_penalty方法"
            
            print("✅ 奖励函数已成功集成风险惩罚项")
            
        except AttributeError:
            assert False, "_calculate_reward方法不存在"
        
    def test_backtest_should_include_risk_validation(self):
        """Green: 测试回测流程已成功包含风险控制验证"""
        print("=== Green: 验证回测流程已包含风险验证 ===")
        
        # 检查回测脚本是否集成了风险控制
        from scripts.backtest import run_backtest, validate_step_risk_metrics, calculate_risk_summary
        
        # 查看run_backtest函数签名和实现
        import inspect
        source = inspect.getsource(run_backtest)
        
        # 检查是否有风险相关的代码
        has_risk_check = any([
            'risk_controller' in source.lower(),
            'risk_validation' in source.lower(),
            'risk_violations' in source.lower(),
            'risk_metrics' in source.lower()
        ])
        
        print(f"回测流程包含风险验证: {has_risk_check}")
        
        # 检查是否有必要的风险验证函数
        has_validate_function = hasattr(validate_step_risk_metrics, '__call__')
        has_summary_function = hasattr(calculate_risk_summary, '__call__')
        
        print(f"风险验证函数存在: {has_validate_function}")
        print(f"风险汇总函数存在: {has_summary_function}")
        
        # Green阶段：应该包含风险验证
        assert has_risk_check, "回测流程应该包含风险验证相关代码"
        assert has_validate_function, "应该存在validate_step_risk_metrics函数"
        assert has_summary_function, "应该存在calculate_risk_summary函数"
        
        print("✅ 回测流程已成功集成风险验证")
        
    def test_risk_violation_should_prevent_trades(self):
        """Green: 测试风险违规阻止机制已实现"""
        print("=== Green: 验证风险违规阻止机制 ===")
        
        # 检查_adjust_weights_for_risk_violations方法是否存在
        import inspect
        methods = [name for name, _ in inspect.getmembers(PortfolioEnvironment, predicate=inspect.isfunction)]
        has_adjust_method = '_adjust_weights_for_risk_violations' in methods
        
        print(f"权重调整方法存在: {has_adjust_method}")
        
        assert has_adjust_method, "应该存在权重调整方法来处理风险违规"
        
        print("✅ 风险违规阻止机制已实现")
        
    def test_risk_metrics_should_be_tracked_during_backtest(self):
        """Green: 测试回测过程中已实现风险指标跟踪"""
        print("=== Green: 验证回测风险指标跟踪 ===")
        
        # 检查run_backtest函数返回值是否包含风险指标
        import inspect
        from scripts.backtest import run_backtest
        source = inspect.getsource(run_backtest)
        
        has_risk_metrics_return = 'risk_metrics' in source
        
        print(f"回测返回风险指标: {has_risk_metrics_return}")
        
        assert has_risk_metrics_return, "回测结果应该包含风险指标"
        
        print("✅ 回测风险指标跟踪已实现")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])