#!/usr/bin/env python3
"""
scikit-optimize依赖错误处理的红色阶段TDD测试
验证缺少依赖时应该抛出RuntimeError而不是提供降级实现
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys
import unittest.mock

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestSkoptDependencyErrorHandling:
    """测试scikit-optimize依赖的错误处理"""
    
    def test_bayesian_optimizer_should_raise_error_when_skopt_unavailable(self):
        """Red: 测试当scikit-optimize不可用时应该抛出RuntimeError"""
        print("=== Red: 验证BayesianOptimizer在缺少依赖时抛出RuntimeError ===")
        
        # 模拟scikit-optimize不可用的情况
        with unittest.mock.patch.dict('sys.modules', {'skopt': None}):
            # 重新导入模块以触发ImportError
            import importlib
            import rl_trading_system.optimization.bayesian_optimizer
            importlib.reload(rl_trading_system.optimization.bayesian_optimizer)
            
            from rl_trading_system.optimization.bayesian_optimizer import BayesianOptimizer
            
            # 当依赖不可用时，应该在初始化或使用时抛出RuntimeError
            with pytest.raises(RuntimeError, match="scikit-optimize.*不可用"):
                optimizer = BayesianOptimizer()
                optimizer.optimize(
                    objective_function=lambda x: x**2,
                    parameter_bounds={'x': (-5.0, 5.0)},
                    n_calls=5
                )
        
        print("✅ BayesianOptimizer正确抛出RuntimeError")
    
    def test_parameter_optimizer_should_raise_error_when_skopt_unavailable(self):
        """Red: 测试当scikit-optimize不可用时ParameterOptimizer应该抛出RuntimeError"""
        print("=== Red: 验证ParameterOptimizer在缺少依赖时抛出RuntimeError ===")
        
        # 模拟scikit-optimize不可用的情况
        with unittest.mock.patch.dict('sys.modules', {'skopt': None}):
            # 重新导入模块以触发ImportError
            import importlib
            import rl_trading_system.backtest.parameter_optimizer
            importlib.reload(rl_trading_system.backtest.parameter_optimizer)
            
            from rl_trading_system.backtest.parameter_optimizer import ParameterOptimizer
            
            # 当依赖不可用时，应该在尝试贝叶斯优化时抛出RuntimeError
            with pytest.raises(RuntimeError, match="scikit-optimize.*不可用"):
                optimizer = ParameterOptimizer()
                optimizer.optimize_parameters_bayesian({}, {}, 5)
        
        print("✅ ParameterOptimizer正确抛出RuntimeError")
    
    def test_current_implementation_violates_rules(self):
        """Red: 测试当前实现违反规则（只记录警告而不抛出异常）"""
        print("=== Red: 验证当前实现违反规则 ===")
        
        # 检查当前实现是否在ImportError时只记录警告
        # 这违反了规则1和规则6
        
        # 这个测试应该失败，因为当前实现不会抛出RuntimeError
        with unittest.mock.patch.dict('sys.modules', {'skopt': None}):
            import importlib
            
            # 重新导入会触发ImportError，但当前实现只记录警告
            try:
                import rl_trading_system.optimization.bayesian_optimizer
                importlib.reload(rl_trading_system.optimization.bayesian_optimizer)
                
                # 如果能正常导入而没有抛出异常，说明违反了规则
                assert False, "违反规则：应该在缺少必要依赖时抛出RuntimeError，但只记录了警告"
            except RuntimeError:
                # 这是期望的正确行为
                print("✅ 正确抛出了RuntimeError")
            except Exception as e:
                if "ImportError" not in str(type(e)):
                    assert False, f"抛出了错误类型的异常: {type(e).__name__}: {e}"
                else:
                    # ImportError是预期的，但我们希望它被转换为RuntimeError
                    assert False, "违反规则：ImportError应该被转换为RuntimeError"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])