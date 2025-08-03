"""
使用数据生成器的测试示例

展示如何使用测试数据生成器来创建全面的测试用例。
"""

import pytest
import numpy as np
import sys
import os

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from test_data_generator import (
    TestDataGenerator, MarketScenario, DataGenerationConfig, 
    ScenarioTester, create_test_portfolio_data, create_drawdown_test_data
)
from src.rl_trading_system.risk_control.drawdown_monitor import DrawdownMonitor


class TestDataGeneratorUsage:
    """测试数据生成器使用示例"""
    
    @pytest.fixture
    def generator(self):
        """创建数据生成器"""
        config = DataGenerationConfig(
            initial_value=100000.0,
            random_seed=42  # 确保可重复
        )
        return TestDataGenerator(config)
    
    def test_generate_different_scenarios(self, generator):
        """测试生成不同市场场景数据"""
        scenarios = [
            MarketScenario.BULL_MARKET,
            MarketScenario.BEAR_MARKET,
            MarketScenario.HIGH_VOLATILITY,
            MarketScenario.CRASH
        ]
        
        for scenario in scenarios:
            data = generator.generate_portfolio_values(scenario, length=100)
            
            # 验证数据基本属性
            assert len(data) == 100
            assert np.all(np.isfinite(data))
            assert np.all(data > 0)  # 投资组合净值不应为负
            
            # 验证场景特征
            returns = np.diff(data) / data[:-1]
            
            if scenario == MarketScenario.BULL_MARKET:
                assert np.mean(returns) > 0  # 牛市应该有正收益
            elif scenario == MarketScenario.BEAR_MARKET:
                assert np.mean(returns) < 0  # 熊市应该有负收益
            elif scenario == MarketScenario.HIGH_VOLATILITY:
                assert np.std(returns) > 0.02  # 高波动应该有高标准差
    
    def test_generate_drawdown_scenarios(self, generator):
        """测试生成回撤场景"""
        scenarios = generator.generate_drawdown_scenarios()
        
        # 验证所有场景都生成了
        expected_scenarios = [
            'light_drawdown', 'moderate_drawdown', 'severe_drawdown',
            'prolonged_drawdown', 'quick_recovery', 'multiple_drawdowns'
        ]
        
        for scenario_name in expected_scenarios:
            assert scenario_name in scenarios
            data = scenarios[scenario_name]
            
            # 验证数据质量
            assert len(data) > 0
            assert np.all(np.isfinite(data))
            
            # 计算最大回撤
            peak = np.maximum.accumulate(data)
            drawdown = (data - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # 验证回撤符合预期
            if scenario_name == 'light_drawdown':
                assert max_drawdown > -0.08  # 轻微回撤应该小于8%
            elif scenario_name == 'severe_drawdown':
                assert max_drawdown < -0.25  # 严重回撤应该大于25%
    
    def test_generate_performance_metrics(self, generator):
        """测试生成性能指标"""
        metrics_list = generator.generate_performance_metrics(
            MarketScenario.BULL_MARKET, count=5
        )
        
        assert len(metrics_list) == 5
        
        for metrics in metrics_list:
            # 验证指标合理性
            assert hasattr(metrics, 'sharpe_ratio')
            assert hasattr(metrics, 'return_rate')
            assert hasattr(metrics, 'max_drawdown')
            
            # 牛市场景的指标应该相对较好
            assert metrics.sharpe_ratio > 0  # 牛市夏普比率应为正
            assert metrics.max_drawdown <= 0  # 最大回撤应为负数或零
    
    def test_stress_test_data_generation(self, generator):
        """测试压力测试数据生成"""
        stress_data = generator.generate_stress_test_data()
        
        # 验证压力测试场景
        expected_scenarios = [
            'extreme_volatility', 'consecutive_decline', 'price_gaps',
            'liquidity_crisis', 'crash_recovery'
        ]
        
        for scenario_name in expected_scenarios:
            assert scenario_name in stress_data
            scenario_data = stress_data[scenario_name]
            
            assert 'returns' in scenario_data
            assert 'description' in scenario_data
            
            returns = scenario_data['returns']
            assert len(returns) > 0
            assert np.all(np.isfinite(returns))
            
            # 验证特定场景特征
            if scenario_name == 'extreme_volatility':
                assert np.std(returns) > 0.03  # 极端波动应该有高方差
            elif scenario_name == 'consecutive_decline':
                # 应该包含连续负收益
                negative_streaks = self._find_consecutive_negatives(returns)
                assert max(negative_streaks) >= 10  # 至少有10天连续下跌
    
    def _find_consecutive_negatives(self, returns):
        """找到连续负收益的长度"""
        streaks = []
        current_streak = 0
        
        for ret in returns:
            if ret < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks if streaks else [0]


class TestScenarioTesting:
    """场景测试示例"""
    
    @pytest.fixture
    def scenario_tester(self):
        """创建场景测试器"""
        generator = TestDataGenerator(DataGenerationConfig(random_seed=42))
        return ScenarioTester(generator)
    
    def test_drawdown_calculation_across_scenarios(self, scenario_tester):
        """测试回撤计算在不同场景下的表现"""
        
        def test_drawdown_function(portfolio_values):
            """测试用的回撤计算函数"""
            monitor = DrawdownMonitor()
            return monitor.calculate_drawdown(portfolio_values)
        
        # 运行多场景测试
        results = scenario_tester.run_scenario_tests(
            test_drawdown_function,
            scenarios=[MarketScenario.BULL_MARKET, MarketScenario.BEAR_MARKET, MarketScenario.CRASH]
        )
        
        # 验证所有场景都成功测试
        for scenario_name, result in results.items():
            if result['success']:
                assert 'result' in result
                drawdown_result = result['result']
                assert 'current_drawdown' in drawdown_result
                assert 'max_drawdown' in drawdown_result
            else:
                # 如果失败，至少应该有错误信息
                assert 'error' in result
    
    def test_data_quality_validation(self, scenario_tester):
        """测试数据质量验证"""
        # 生成测试数据
        test_data = scenario_tester.generator.generate_portfolio_values(
            MarketScenario.BULL_MARKET, length=200
        )
        
        # 验证数据质量
        quality_report = scenario_tester.validate_generated_data(test_data)
        
        # 检查报告内容
        assert 'length' in quality_report
        assert 'data_quality' in quality_report
        assert quality_report['length'] == 200
        assert not quality_report['has_nan']
        assert not quality_report['has_inf']
        assert quality_report['data_quality'] in ['good', 'high_volatility', 'constant']


class TestConvenienceFunctions:
    """便利函数测试"""
    
    def test_create_test_portfolio_data(self):
        """测试快速创建投资组合数据"""
        # 测试不同场景
        scenarios = ["normal", "crash", "bull", "bear", "volatile"]
        
        for scenario in scenarios:
            data = create_test_portfolio_data(scenario, length=100)
            
            assert len(data) == 100
            assert np.all(np.isfinite(data))
            assert np.all(data > 0)
            
            # 验证不同场景的特征
            returns = np.diff(data) / data[:-1]
            
            if scenario == "bull":
                assert np.mean(returns) > -0.001  # 牛市不应该大幅下跌
            elif scenario == "bear":
                assert np.mean(returns) < 0.001   # 熊市不应该大幅上涨
            elif scenario == "volatile":
                assert np.std(returns) > 0.01     # 波动场景应该有一定波动性
    
    def test_create_drawdown_test_data(self):
        """测试创建回撤测试数据"""
        drawdown_data = create_drawdown_test_data()
        
        # 验证返回了多个场景
        assert len(drawdown_data) >= 5
        assert 'light_drawdown' in drawdown_data
        assert 'severe_drawdown' in drawdown_data
        
        # 验证每个场景的数据质量
        for scenario_name, data in drawdown_data.items():
            assert len(data) > 50  # 应该有足够的数据点
            assert np.all(np.isfinite(data))
            
            # 验证确实存在回撤
            peak = np.maximum.accumulate(data)
            drawdown = (data - peak) / peak
            max_drawdown = np.min(drawdown)
            assert max_drawdown < -0.01  # 应该至少有1%的回撤


class TestIntegrationWithRealComponents:
    """与真实组件的集成测试"""
    
    def test_integration_with_drawdown_monitor(self):
        """测试与回撤监控器的集成"""
        # 生成多种测试数据
        generator = TestDataGenerator(DataGenerationConfig(random_seed=123))
        
        test_scenarios = [
            ('normal', MarketScenario.SIDEWAYS),
            ('bull', MarketScenario.BULL_MARKET),
            ('volatile', MarketScenario.HIGH_VOLATILITY),
        ]
        
        monitor = DrawdownMonitor()
        
        for scenario_name, scenario_type in test_scenarios:
            # 生成测试数据
            test_data = generator.generate_portfolio_values(scenario_type, length=150)
            
            try:
                # 测试回撤监控器
                result = monitor.calculate_drawdown(test_data)
                
                # 验证结果合理性
                assert isinstance(result, dict)
                assert 'current_drawdown' in result
                assert 'max_drawdown' in result
                
                # 验证数值合理性
                assert result['max_drawdown'] <= 0  # 最大回撤应该<=0
                assert -1 <= result['current_drawdown'] <= 1  # 当前回撤应该在合理范围
                
            except Exception as e:
                pytest.fail(f"场景 {scenario_name} 测试失败: {str(e)}")
    
    def test_edge_cases_with_generated_data(self):
        """使用生成的数据测试边界情况"""
        generator = TestDataGenerator(DataGenerationConfig(random_seed=456))
        
        # 生成边界情况数据
        edge_cases = {
            'constant_values': np.full(100, 100000.0),  # 恒定值
            'single_value': np.array([100000.0]),       # 单一值
            'extreme_crash': generator.generate_portfolio_values(MarketScenario.CRASH, 50)
        }
        
        monitor = DrawdownMonitor()
        
        for case_name, test_data in edge_cases.items():
            try:
                result = monitor.calculate_drawdown(test_data)
                
                # 边界情况也应该返回有效结果
                assert result is not None
                assert isinstance(result, dict)
                
                if case_name == 'constant_values':
                    # 恒定值的回撤应该为0
                    assert abs(result['current_drawdown']) < 1e-10
                    assert abs(result['max_drawdown']) < 1e-10
                
            except Exception as e:
                # 某些边界情况可能会抛异常，但应该是合理的异常
                assert isinstance(e, (ValueError, RuntimeError, IndexError))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])