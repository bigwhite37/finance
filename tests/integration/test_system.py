"""
系统功能验证脚本 - 快速验证核心功能
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """测试模块导入"""
    print("1. 测试模块导入...")
    
    modules_to_test = [
        ('config', ['ConfigManager', 'get_default_config']),
        ('data', ['DataManager']),
        ('factors', ['FactorEngine', 'AlphaFactors', 'RiskFactors']),
        ('rl_agent', ['TradingEnvironment', 'CVaRPPOAgent', 'SafetyShield']),
        ('risk_control', ['RiskController']),
        ('backtest', ['BacktestEngine', 'PerformanceAnalyzer', 'ComfortabilityMetrics']),
        ('utils', ['setup_logger', 'calculate_metrics'])
    ]
    
    import_results = {}
    
    for module_name, classes in modules_to_test:
        try:
            module = __import__(module_name, fromlist=classes)
            
            for class_name in classes:
                getattr(module, class_name)
            
            import_results[module_name] = "✓ 成功"
            print(f"   {module_name}: ✓")
            
        except ImportError as e:
            import_results[module_name] = f"✗ 失败: {e}"
            print(f"   {module_name}: ✗ {e}")
        except AttributeError as e:
            import_results[module_name] = f"✗ 属性错误: {e}"
            print(f"   {module_name}: ✗ {e}")
    
    return all("成功" in result for result in import_results.values())


def test_config_manager():
    """测试配置管理器"""
    print("\n2. 测试配置管理器...")
    
    try:
        from config import ConfigManager, get_default_config
        
        # 测试默认配置
        default_config = get_default_config()
        assert isinstance(default_config, dict), "默认配置应该是字典"
        assert 'data' in default_config, "应该包含data配置"
        assert 'agent' in default_config, "应该包含agent配置"
        print("   默认配置: ✓")
        
        # 测试配置管理器
        config_manager = ConfigManager()
        assert config_manager.validate_config(), "配置验证失败"
        
        # 测试配置获取
        data_config = config_manager.get_data_config()
        assert isinstance(data_config, dict), "数据配置应该是字典"
        print("   配置管理器: ✓")
        
        # 测试配置路径访问
        lr = config_manager.get_value('agent.learning_rate')
        assert isinstance(lr, (int, float)), "学习率应该是数值"
        print("   路径访问: ✓")
        
        return True
        
    except Exception as e:
        print(f"   配置管理器测试失败: {e}")
        return False


def test_data_components():
    """测试数据组件（使用模拟数据）"""
    print("\n3. 测试数据组件...")
    
    try:
        from data import DataManager
        
        # 创建模拟配置
        config = {
            'provider': 'mock',
            'region': 'cn',
            'universe': 'test'
        }
        
        # 注意：实际的DataManager需要qlib初始化，这里只测试创建
        # 在没有qlib数据的情况下，这会抛出异常，这是预期的
        try:
            data_manager = DataManager(config)
            print("   数据管理器创建: ✓")
        except Exception as e:
            if "qlib" in str(e).lower():
                print("   数据管理器创建: ⚠ qlib未配置（这是正常的）")
            else:
                raise e
        
        return True
        
    except ImportError as e:
        print(f"   数据组件导入失败: {e}")
        return False
    except Exception as e:
        print(f"   数据组件测试失败: {e}")
        return False


def test_factor_engine():
    """测试因子引擎"""
    print("\n4. 测试因子引擎...")
    
    try:
        from factors import FactorEngine, AlphaFactors, RiskFactors
        
        # 创建测试配置
        config = {'default_factors': ['return_20d', 'volatility_60d']}
        
        # 创建因子引擎
        factor_engine = FactorEngine(config)
        print("   因子引擎创建: ✓")
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        price_data = pd.DataFrame(
            data=100 + np.cumsum(np.random.normal(0, 0.02, (100, 3)), axis=0),
            index=dates,
            columns=['stock_1', 'stock_2', 'stock_3']
        )
        
        # 测试低波动筛选
        low_vol_stocks = factor_engine.filter_low_volatility_universe(price_data)
        assert isinstance(low_vol_stocks, list), "应该返回股票列表"
        print("   低波动筛选: ✓")
        
        # 测试因子暴露度计算
        factor_data = pd.DataFrame(np.random.normal(0, 1, (100, 3)))
        exposure = factor_engine.calculate_factor_exposure(factor_data)
        assert isinstance(exposure, pd.DataFrame), "应该返回DataFrame"
        print("   因子暴露度: ✓")
        
        # 测试Alpha和风险因子
        alpha_factors = AlphaFactors(config)
        risk_factors = RiskFactors(config)
        print("   Alpha因子: ✓")
        print("   风险因子: ✓")
        
        return True
        
    except Exception as e:
        print(f"   因子引擎测试失败: {e}")
        return False


def test_rl_components():
    """测试强化学习组件"""
    print("\n5. 测试强化学习组件...")
    
    try:
        from rl_agent import TradingEnvironment, CVaRPPOAgent, SafetyShield
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        price_data = pd.DataFrame(
            data=100 + np.cumsum(np.random.normal(0, 0.02, (100, 3)), axis=0),
            index=dates,
            columns=['stock_1', 'stock_2', 'stock_3']
        )
        
        factor_data = pd.DataFrame(
            data=np.random.normal(0, 1, (100, 4)),
            index=dates,
            columns=['factor_1', 'factor_2', 'factor_3', 'factor_4']
        )
        
        # 测试交易环境
        env_config = {
            'lookback_window': 10,
            'transaction_cost': 0.001,
            'max_position': 0.1,
            'max_leverage': 1.2
        }
        
        environment = TradingEnvironment(factor_data, price_data, env_config)
        assert environment.observation_space.shape[0] > 0, "状态空间维度应该大于0"
        assert environment.action_space.shape[0] == 3, "动作空间应该等于股票数量"
        print("   交易环境: ✓")
        
        # 测试智能体
        agent_config = {
            'hidden_dim': 64,
            'learning_rate': 3e-4,
            'cvar_alpha': 0.05
        }
        
        state_dim = environment.observation_space.shape[0]
        action_dim = environment.action_space.shape[0]
        
        agent = CVaRPPOAgent(state_dim, action_dim, agent_config)
        print("   CVaR-PPO智能体: ✓")
        
        # 测试安全保护层
        shield_config = {
            'max_position': 0.1,
            'max_leverage': 1.2
        }
        
        safety_shield = SafetyShield(shield_config)
        print("   安全保护层: ✓")
        
        # 测试简单交互
        obs, info = environment.reset()
        action, log_prob, value, cvar_est = agent.get_action(obs)
        safe_action = safety_shield.shield_action(action, info)
        
        assert len(safe_action) == action_dim, "安全动作维度应该一致"
        print("   环境智能体交互: ✓")
        
        return True
        
    except Exception as e:
        print(f"   强化学习组件测试失败: {e}")
        return False


def test_risk_control():
    """测试风险控制"""
    print("\n6. 测试风险控制...")
    
    try:
        from risk_control import RiskController, TargetVolatilityController, DynamicStopLoss
        
        # 测试风险控制器
        risk_config = {
            'target_volatility': 0.12,
            'max_position': 0.1,
            'max_leverage': 1.2
        }
        
        risk_controller = RiskController(risk_config)
        print("   风险控制器: ✓")
        
        # 测试目标波动率控制
        vol_config = {'target_volatility': 0.12}
        vol_controller = TargetVolatilityController(vol_config)
        print("   目标波动率控制: ✓")
        
        # 测试动态止损
        stop_config = {'stop_loss_pct': 0.03}
        stop_loss = DynamicStopLoss(stop_config)
        
        # 测试止损逻辑
        assert not stop_loss.check_stop_loss(1.0), "初始值不应触发止损"
        assert not stop_loss.check_stop_loss(1.05), "上涨不应触发止损"
        print("   动态止损: ✓")
        
        return True
        
    except Exception as e:
        print(f"   风险控制测试失败: {e}")
        return False


def test_backtest_components():
    """测试回测组件"""
    print("\n7. 测试回测组件...")
    
    try:
        from backtest import BacktestEngine, PerformanceAnalyzer, ComfortabilityMetrics
        
        # 测试回测引擎
        backtest_config = {
            'initial_capital': 1000000,
            'transaction_cost': 0.001,
            'risk_free_rate': 0.03
        }
        
        backtest_engine = BacktestEngine(backtest_config)
        print("   回测引擎: ✓")
        
        # 测试绩效分析器
        perf_analyzer = PerformanceAnalyzer(backtest_config)
        
        # 创建模拟收益率数据
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        metrics = perf_analyzer._calculate_return_metrics(returns)
        assert isinstance(metrics, dict), "应该返回指标字典"
        assert '年化收益率' in metrics, "应该包含年化收益率"
        print("   绩效分析器: ✓")
        
        # 测试心理舒适度
        comfort_config = {
            'monthly_dd_threshold': 0.05,
            'max_consecutive_losses': 5
        }
        
        comfort_metrics = ComfortabilityMetrics(comfort_config)
        
        max_losses = comfort_metrics.max_consecutive_losses(returns)
        assert isinstance(max_losses, int), "连续亏损天数应该是整数"
        print("   心理舒适度: ✓")
        
        return True
        
    except Exception as e:
        print(f"   回测组件测试失败: {e}")
        return False


def test_utils():
    """测试工具模块"""
    print("\n8. 测试工具模块...")
    
    try:
        from utils import setup_logger, calculate_metrics
        
        # 测试日志设置
        logger = setup_logger('test_logger')
        assert logger is not None, "应该返回logger对象"
        print("   日志工具: ✓")
        
        # 测试指标计算
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))
        
        metrics = calculate_metrics(returns)
        assert isinstance(metrics, dict), "应该返回指标字典"
        assert 'sharpe_ratio' in metrics, "应该包含夏普比率"
        print("   指标计算: ✓")
        
        return True
        
    except Exception as e:
        print(f"   工具模块测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("开始系统功能验证...\n")
    
    tests = [
        test_imports,
        test_config_manager,
        test_data_components,
        test_factor_engine,
        test_rl_components,
        test_risk_control,
        test_backtest_components,
        test_utils
    ]
    
    results = []
    
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   测试异常: {e}")
            results.append(False)
    
    # 打印总结
    print("\n" + "="*50)
    print("系统功能验证总结")
    print("="*50)
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"总测试数: {total_tests}")
    print(f"通过数: {passed_tests}")
    print(f"失败数: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests / total_tests * 100:.1f}%")
    
    if all(results):
        print("\n🎉 所有核心功能验证通过！系统可以正常运行。")
        print("\n建议下一步:")
        print("1. 运行 python examples/quick_start.py 查看完整演示")
        print("2. 运行 python tests/run_tests.py 执行完整测试套件")
        print("3. 配置qlib数据源后运行完整训练流程")
    else:
        print("\n⚠️  部分功能验证失败，请检查相关模块。")
        
        failed_indices = [i for i, result in enumerate(results) if not result]
        failed_tests = [tests[i].__name__ for i in failed_indices]
        print(f"失败的测试: {', '.join(failed_tests)}")
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)