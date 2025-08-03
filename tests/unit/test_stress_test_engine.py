"""
压力测试引擎单元测试
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.rl_trading_system.risk_control.stress_test_engine import (
    StressTestEngine, StressTestConfig, StressTestType, MarketScenario,
    StressTestResult, ScenarioDefinition, ExtremeScenarioParameters
)


class TestStressTestEngine:
    """压力测试引擎测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return StressTestConfig(
            confidence_levels=[0.95, 0.99],
            time_horizons=[1, 5, 10],
            num_simulations=1000,
            random_seed=42
        )
    
    @pytest.fixture
    def engine(self, config):
        """压力测试引擎实例"""
        return StressTestEngine(config)
    
    @pytest.fixture
    def sample_data(self):
        """样本数据"""
        # 生成模拟的资产收益率数据
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        assets = ['STOCK_A', 'STOCK_B', 'STOCK_C']
        
        # 生成相关的收益率数据
        returns_data = np.random.multivariate_normal(
            mean=[0.0005, 0.0003, 0.0004],  # 日收益率均值
            cov=[[0.0004, 0.0001, 0.0002],  # 协方差矩阵
                 [0.0001, 0.0003, 0.0001],
                 [0.0002, 0.0001, 0.0005]],
            size=len(dates)
        )
        
        asset_returns = pd.DataFrame(returns_data, index=dates, columns=assets)
        portfolio_weights = np.array([0.4, 0.3, 0.3])
        
        return asset_returns, portfolio_weights
    
    def test_engine_initialization(self, config):
        """测试引擎初始化"""
        engine = StressTestEngine(config)
        
        assert engine.config == config
        assert engine.drawdown_monitor is not None
        assert len(engine.extreme_scenarios) > 0
        
        # 验证极端情景定义
        assert MarketScenario.MARKET_CRASH in engine.extreme_scenarios
        assert MarketScenario.VOLATILITY_SPIKE in engine.extreme_scenarios
        assert MarketScenario.LIQUIDITY_DROUGHT in engine.extreme_scenarios
    
    def test_input_validation(self, engine, sample_data):
        """测试输入验证"""
        asset_returns, portfolio_weights = sample_data
        
        # 测试权重数量不匹配
        with pytest.raises(ValueError, match="投资组合权重数量与资产数量不匹配"):
            wrong_weights = np.array([0.5, 0.5])  # 只有2个权重，但有3个资产
            engine.run_stress_test(wrong_weights, asset_returns, StressTestType.MONTE_CARLO)
        
        # 测试权重和不为1
        with pytest.raises(ValueError, match="投资组合权重和必须接近1"):
            wrong_weights = np.array([0.4, 0.4, 0.4])  # 权重和为1.2
            engine.run_stress_test(wrong_weights, asset_returns, StressTestType.MONTE_CARLO)
        
        # 测试负权重
        with pytest.raises(ValueError, match="投资组合权重不能为负数"):
            wrong_weights = np.array([0.6, 0.5, -0.1])  # 包含负权重
            engine.run_stress_test(wrong_weights, asset_returns, StressTestType.MONTE_CARLO)
        
        # 测试空数据
        with pytest.raises(ValueError, match="资产收益率数据不能为空"):
            empty_returns = pd.DataFrame()
            engine.run_stress_test(portfolio_weights, empty_returns, StressTestType.MONTE_CARLO)
    
    def test_historical_scenario_test(self, engine, sample_data):
        """测试历史情景重现"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.HISTORICAL_SCENARIO
        )
        
        # 验证结果结构
        assert isinstance(result, StressTestResult)
        assert result.test_type == StressTestType.HISTORICAL_SCENARIO
        assert len(result.portfolio_losses) > 0
        assert len(result.var_estimates) == len(engine.config.confidence_levels)
        assert len(result.cvar_estimates) == len(engine.config.confidence_levels)
        
        # 验证VaR单调性（高置信水平对应更高的VaR）
        confidence_levels = sorted(result.var_estimates.keys())
        var_values = [result.var_estimates[conf] for conf in confidence_levels]
        assert all(var_values[i] <= var_values[i+1] for i in range(len(var_values)-1))
        
        # 验证CVaR >= VaR
        for conf in confidence_levels:
            assert result.cvar_estimates[conf] >= result.var_estimates[conf]
        
        # 验证资产贡献
        assert len(result.asset_contributions) == len(asset_returns.columns)
        assert all(isinstance(contrib, (int, float)) for contrib in result.asset_contributions.values())
    
    def test_monte_carlo_test(self, engine, sample_data):
        """测试蒙特卡洛模拟"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.MONTE_CARLO
        )
        
        # 验证结果结构
        assert isinstance(result, StressTestResult)
        assert result.test_type == StressTestType.MONTE_CARLO
        assert len(result.portfolio_losses) == engine.config.num_simulations
        
        # 验证损失分布的统计特性
        assert np.isfinite(result.portfolio_losses).all()
        assert result.max_loss >= 0
        assert 0 <= result.probability_of_loss <= 1
        
        # 验证资产贡献和约等于投资组合损失均值
        total_contribution = sum(result.asset_contributions.values())
        portfolio_loss_mean = np.mean(result.portfolio_losses)
        assert abs(total_contribution - portfolio_loss_mean) < 0.01
    
    def test_monte_carlo_with_scenario(self, engine, sample_data):
        """测试带情景的蒙特卡洛模拟"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_stress_test(
            portfolio_weights, asset_returns, 
            StressTestType.MONTE_CARLO, MarketScenario.MARKET_CRASH
        )
        
        assert result.scenario == MarketScenario.MARKET_CRASH
        
        # 市场崩盘情景下的损失应该更大
        normal_result = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.MONTE_CARLO
        )
        
        # 崩盘情景的VaR应该更高
        assert result.var_estimates[0.99] > normal_result.var_estimates[0.99]
    
    def test_parametric_var_test(self, engine, sample_data):
        """测试参数化VaR"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.PARAMETRIC_VAR
        )
        
        # 验证结果结构
        assert isinstance(result, StressTestResult)
        assert result.test_type == StressTestType.PARAMETRIC_VAR
        assert len(result.portfolio_losses) == engine.config.num_simulations
        
        # 参数化VaR应该基于正态分布假设
        assert result.probability_of_loss == 0.5  # 正态分布的对称性
        
        # 验证资产贡献（边际VaR）
        assert len(result.asset_contributions) == len(asset_returns.columns)
    
    def test_extreme_value_test(self, engine, sample_data):
        """测试极值理论"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.EXTREME_VALUE
        )
        
        # 验证结果结构
        assert isinstance(result, StressTestResult)
        assert result.test_type == StressTestType.EXTREME_VALUE
        
        # 验证极值理论特有的统计信息
        assert 'evt_threshold' in result.statistics
        assert 'evt_exceedances' in result.statistics
        assert result.statistics['evt_exceedances'] >= 0
    
    def test_correlation_breakdown_test(self, engine, sample_data):
        """测试相关性崩溃"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.CORRELATION_BREAKDOWN
        )
        
        # 验证结果结构
        assert isinstance(result, StressTestResult)
        assert result.test_type == StressTestType.CORRELATION_BREAKDOWN
        assert result.scenario == MarketScenario.MARKET_CRASH
        
        # 验证相关性冲击统计信息
        assert 'correlation_shock_factor' in result.statistics
        assert 'avg_normal_correlation' in result.statistics
        assert 'avg_shocked_correlation' in result.statistics
        
        # 冲击后的相关性应该更高
        normal_corr = result.statistics['avg_normal_correlation']
        shocked_corr = result.statistics['avg_shocked_correlation']
        assert shocked_corr > normal_corr
    
    def test_liquidity_crisis_test(self, engine, sample_data):
        """测试流动性危机"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.LIQUIDITY_CRISIS
        )
        
        # 验证结果结构
        assert isinstance(result, StressTestResult)
        assert result.test_type == StressTestType.LIQUIDITY_CRISIS
        assert result.scenario == MarketScenario.LIQUIDITY_DROUGHT
        
        # 验证流动性影响统计信息
        assert 'liquidity_impact_factor' in result.statistics
        assert 'avg_liquidity_premium' in result.statistics
        
        # 流动性危机应该导致更高的损失
        assert result.statistics['avg_liquidity_premium'] < 0  # 负的流动性溢价
    
    def test_comprehensive_stress_test(self, engine, sample_data):
        """测试综合压力测试"""
        asset_returns, portfolio_weights = sample_data
        
        results = engine.run_comprehensive_stress_test(portfolio_weights, asset_returns)
        
        # 验证结果数量
        assert len(results) > 0
        assert isinstance(results, dict)
        
        # 验证每个结果都是有效的
        for name, result in results.items():
            assert isinstance(result, StressTestResult)
            assert len(result.portfolio_losses) > 0
            assert len(result.var_estimates) > 0
    
    def test_var_calculation(self, engine):
        """测试VaR计算"""
        losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        confidence_levels = [0.9, 0.95]
        
        var_estimates = engine._calculate_var(losses, confidence_levels)
        
        assert len(var_estimates) == len(confidence_levels)
        assert var_estimates[0.9] == np.percentile(losses, 90)
        assert var_estimates[0.95] == np.percentile(losses, 95)
        
        # 高置信水平对应更高的VaR
        assert var_estimates[0.95] >= var_estimates[0.9]
    
    def test_cvar_calculation(self, engine):
        """测试CVaR计算"""
        losses = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        confidence_levels = [0.9, 0.95]
        
        cvar_estimates = engine._calculate_cvar(losses, confidence_levels)
        var_estimates = engine._calculate_var(losses, confidence_levels)
        
        assert len(cvar_estimates) == len(confidence_levels)
        
        # CVaR应该大于等于VaR
        for conf in confidence_levels:
            assert cvar_estimates[conf] >= var_estimates[conf]
    
    def test_scenario_parameter_adjustment(self, engine, sample_data):
        """测试情景参数调整"""
        asset_returns, _ = sample_data
        
        mean_returns = asset_returns.mean()
        cov_matrix = asset_returns.cov()
        
        # 测试市场崩盘情景
        adjusted_mean, adjusted_cov = engine._adjust_parameters_for_scenario(
            mean_returns, cov_matrix, MarketScenario.MARKET_CRASH
        )
        
        # 崩盘情景下收益率应该更低
        assert (adjusted_mean < mean_returns).all()
        
        # 波动率应该更高
        adjusted_vol = np.sqrt(np.diag(adjusted_cov))
        original_vol = np.sqrt(np.diag(cov_matrix))
        assert (adjusted_vol > original_vol).all()
    
    def test_correlation_matrix_positive_definite(self, engine):
        """测试相关性矩阵正定性调整"""
        # 创建一个非正定的相关性矩阵
        corr_matrix = np.array([
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 1.0]
        ])
        
        # 人为破坏正定性
        corr_matrix[0, 1] = 0.99
        corr_matrix[1, 0] = 0.99
        corr_matrix[0, 2] = 0.99
        corr_matrix[2, 0] = 0.99
        corr_matrix[1, 2] = 0.99
        corr_matrix[2, 1] = 0.99
        
        adjusted_corr = engine._make_correlation_matrix_positive_definite(corr_matrix)
        
        # 验证调整后的矩阵是正定的
        eigenvals = np.linalg.eigvals(adjusted_corr)
        assert (eigenvals > 0).all()
        
        # 验证对角线元素为1
        assert np.allclose(np.diag(adjusted_corr), 1.0)
    
    def test_t_distribution_simulation(self, engine, sample_data):
        """测试t分布模拟"""
        asset_returns, _ = sample_data
        
        mean_returns = asset_returns.mean()
        cov_matrix = asset_returns.cov()
        df = 5
        n_simulations = 1000
        
        t_samples = engine._simulate_t_distribution_returns(
            mean_returns, cov_matrix, df, n_simulations
        )
        
        # 验证样本形状
        assert t_samples.shape == (n_simulations, len(mean_returns))
        
        # 验证样本均值接近目标均值
        sample_mean = np.mean(t_samples, axis=0)
        assert np.allclose(sample_mean, mean_returns.values, atol=0.01)
        
        # t分布应该比正态分布有更厚的尾部
        from scipy import stats
        sample_kurtosis = [stats.kurtosis(t_samples[:, i]) for i in range(t_samples.shape[1])]
        assert all(k > 0 for k in sample_kurtosis)  # 正超额峰度
    
    def test_marginal_var_contributions(self, engine, sample_data):
        """测试边际VaR贡献计算"""
        asset_returns, portfolio_weights = sample_data
        
        contributions = engine._calculate_marginal_var_contributions(
            portfolio_weights, asset_returns
        )
        
        # 验证贡献数量
        assert len(contributions) == len(asset_returns.columns)
        
        # 验证所有贡献都是数值
        assert all(isinstance(contrib, (int, float)) for contrib in contributions.values())
        
        # 验证贡献和接近投资组合VaR
        total_contribution = sum(contributions.values())
        portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
        portfolio_var = np.percentile(-portfolio_returns, 99)
        
        # 由于线性近似，可能有一些误差
        assert abs(total_contribution - portfolio_var) < max(portfolio_var * 1.5, 0.05)
    
    def test_statistics_calculation(self, engine):
        """测试统计指标计算"""
        losses = np.random.normal(0.05, 0.02, 1000)
        
        statistics = engine._calculate_statistics(losses)
        
        # 验证统计指标
        expected_keys = ['mean', 'std', 'skewness', 'kurtosis', 'min', 'max', 'median', 'q25', 'q75']
        assert all(key in statistics for key in expected_keys)
        
        # 验证统计值的合理性
        assert abs(statistics['mean'] - np.mean(losses)) < 1e-10
        assert abs(statistics['std'] - np.std(losses)) < 1e-10
        assert statistics['min'] <= statistics['q25'] <= statistics['median'] <= statistics['q75'] <= statistics['max']
    
    def test_stress_test_result_validation(self):
        """测试压力测试结果验证"""
        # 测试空损失分布
        with pytest.raises(ValueError, match="投资组合损失分布不能为空"):
            StressTestResult(
                test_type=StressTestType.MONTE_CARLO,
                scenario=None,
                timestamp=datetime.now(),
                portfolio_losses=np.array([]),
                var_estimates={0.95: 0.05},
                cvar_estimates={0.95: 0.06},
                expected_shortfall={0.95: 0.06},
                max_loss=0.1,
                probability_of_loss=0.3,
                tail_expectation=0.08,
                asset_contributions={},
                factor_contributions={},
                statistics={}
            )
        
        # 测试无效置信水平
        with pytest.raises(ValueError, match="置信水平必须在0到1之间"):
            StressTestResult(
                test_type=StressTestType.MONTE_CARLO,
                scenario=None,
                timestamp=datetime.now(),
                portfolio_losses=np.array([0.01, 0.02, 0.03]),
                var_estimates={1.5: 0.05},  # 无效置信水平
                cvar_estimates={1.5: 0.06},
                expected_shortfall={1.5: 0.06},
                max_loss=0.1,
                probability_of_loss=0.3,
                tail_expectation=0.08,
                asset_contributions={},
                factor_contributions={},
                statistics={}
            )
    
    def test_scenario_definition_validation(self):
        """测试情景定义验证"""
        # 测试无效概率
        with pytest.raises(ValueError, match="概率必须在0到1之间"):
            ScenarioDefinition(
                name="测试情景",
                description="测试描述",
                scenario_type=MarketScenario.MARKET_CRASH,
                market_shock=-0.2,
                volatility_multiplier=2.0,
                correlation_adjustment=0.8,
                duration_days=30,
                recovery_days=90,
                probability=1.5  # 无效概率
            )
        
        # 测试无效持续天数
        with pytest.raises(ValueError, match="持续天数必须大于0"):
            ScenarioDefinition(
                name="测试情景",
                description="测试描述",
                scenario_type=MarketScenario.MARKET_CRASH,
                market_shock=-0.2,
                volatility_multiplier=2.0,
                correlation_adjustment=0.8,
                duration_days=0,  # 无效持续天数
                recovery_days=90,
                probability=0.01
            )
    
    def test_visualization(self, engine, sample_data):
        """测试可视化功能"""
        asset_returns, portfolio_weights = sample_data
        
        # 运行几个压力测试
        results = {}
        results['monte_carlo'] = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.MONTE_CARLO
        )
        results['parametric_var'] = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.PARAMETRIC_VAR
        )
        
        # 测试可视化不会抛出异常
        try:
            html_content = engine.visualize_stress_test_results(results)
            assert isinstance(html_content, str)
            assert len(html_content) > 0
            assert 'plotly' in html_content.lower()
        except Exception as e:
            pytest.fail(f"可视化功能失败: {e}")
    
    def test_report_generation(self, engine, sample_data):
        """测试报告生成"""
        asset_returns, portfolio_weights = sample_data
        
        # 运行几个压力测试
        results = {}
        results['monte_carlo'] = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.MONTE_CARLO
        )
        results['historical_scenario'] = engine.run_stress_test(
            portfolio_weights, asset_returns, StressTestType.HISTORICAL_SCENARIO
        )
        
        # 生成报告
        report = engine.generate_stress_test_report(results)
        
        # 验证报告内容
        assert isinstance(report, str)
        assert len(report) > 0
        assert "压力测试报告" in report
        assert "执行摘要" in report
        assert "风险管理建议" in report
        
        # 验证包含测试结果
        for test_name in results.keys():
            assert test_name in report
    
    def test_extreme_scenario_simulation(self, engine, sample_data):
        """测试极端情景模拟"""
        asset_returns, portfolio_weights = sample_data
        
        # 测试市场崩盘情景
        result = engine.run_extreme_scenario_simulation(
            portfolio_weights, asset_returns, MarketScenario.MARKET_CRASH
        )
        
        # 验证结果结构
        assert isinstance(result, StressTestResult)
        assert result.scenario == MarketScenario.MARKET_CRASH
        assert len(result.portfolio_losses) > 0
        
        # 验证情景特有的统计信息
        assert 'scenario_type' in result.statistics
        assert 'shock_magnitude' in result.statistics
        assert 'shock_duration' in result.statistics
        assert result.statistics['scenario_type'] == MarketScenario.MARKET_CRASH.value
        
        # 市场崩盘应该产生较大的损失
        assert result.max_loss > 0
        assert result.var_estimates[0.99] > 0
    
    def test_liquidity_crisis_simulation(self, engine, sample_data):
        """测试流动性危机模拟"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_extreme_scenario_simulation(
            portfolio_weights, asset_returns, MarketScenario.LIQUIDITY_DROUGHT
        )
        
        assert result.scenario == MarketScenario.LIQUIDITY_DROUGHT
        assert len(result.portfolio_losses) == engine.config.num_simulations
        
        # 流动性危机应该产生正的损失
        assert np.mean(result.portfolio_losses) > 0
    
    def test_black_swan_simulation(self, engine, sample_data):
        """测试黑天鹅事件模拟"""
        asset_returns, portfolio_weights = sample_data
        
        result = engine.run_extreme_scenario_simulation(
            portfolio_weights, asset_returns, MarketScenario.BLACK_SWAN
        )
        
        assert result.scenario == MarketScenario.BLACK_SWAN
        
        # 黑天鹅事件应该有极端的尾部损失
        tail_losses = result.portfolio_losses[result.portfolio_losses > np.percentile(result.portfolio_losses, 95)]
        if len(tail_losses) > 0:
            assert np.mean(tail_losses) > np.mean(result.portfolio_losses)
    
    def test_scenario_parameter_calibration(self, engine, sample_data):
        """测试情景参数校准"""
        asset_returns, portfolio_weights = sample_data
        
        # 校准参数
        calibrated_params = engine.calibrate_extreme_scenarios(asset_returns)
        
        # 验证校准结果
        assert isinstance(calibrated_params, dict)
        assert len(calibrated_params) > 0
        
        # 验证包含主要情景
        assert MarketScenario.MARKET_CRASH in calibrated_params
        assert MarketScenario.LIQUIDITY_DROUGHT in calibrated_params
        
        # 验证参数结构
        for scenario, params in calibrated_params.items():
            assert isinstance(params, ExtremeScenarioParameters)
            assert params.scenario_type == scenario
            assert params.shock_magnitude < 0  # 负冲击
            assert params.shock_duration > 0
            assert params.volatility_spike > 1.0
    
    def test_scenario_probability_estimation(self, engine, sample_data):
        """测试情景概率估计"""
        asset_returns, portfolio_weights = sample_data
        
        probabilities = engine.estimate_scenario_probabilities(asset_returns)
        
        # 验证概率结构
        assert isinstance(probabilities, dict)
        assert len(probabilities) > 0
        
        # 验证概率值合理性
        for scenario, prob in probabilities.items():
            assert isinstance(prob, float)
            assert 0 <= prob <= 1
            assert prob < 0.1  # 极端事件概率应该较低
        
        # 市场崩盘概率应该比黑天鹅事件高
        if MarketScenario.MARKET_CRASH in probabilities and MarketScenario.BLACK_SWAN in probabilities:
            assert probabilities[MarketScenario.MARKET_CRASH] > probabilities[MarketScenario.BLACK_SWAN]
    
    def test_risk_limits_recommendations(self, engine, sample_data):
        """测试风险限额建议"""
        asset_returns, portfolio_weights = sample_data
        
        # 运行多个压力测试
        results = {}
        results['market_crash'] = engine.run_extreme_scenario_simulation(
            portfolio_weights, asset_returns, MarketScenario.MARKET_CRASH
        )
        results['liquidity_crisis'] = engine.run_extreme_scenario_simulation(
            portfolio_weights, asset_returns, MarketScenario.LIQUIDITY_DROUGHT
        )
        
        # 生成风险限额建议
        recommendations = engine.generate_risk_limits_recommendations(results)
        
        # 验证建议结构
        assert isinstance(recommendations, dict)
        assert len(recommendations) > 0
        
        # 验证包含关键限额
        expected_limits = ['var_99_limit', 'stop_loss_limit', 'risk_budget_limit']
        for limit in expected_limits:
            if limit in recommendations:
                assert isinstance(recommendations[limit], (int, float))
                assert recommendations[limit] > 0
        
        # 如果有流动性储备建议，应该为正值
        if 'liquidity_reserve' in recommendations:
            assert recommendations['liquidity_reserve'] > 0
    
    def test_custom_scenario_parameters(self, engine, sample_data):
        """测试自定义情景参数"""
        from src.rl_trading_system.risk_control.stress_test_engine import ExtremeScenarioParameters
        
        asset_returns, portfolio_weights = sample_data
        
        # 创建自定义参数
        custom_params = ExtremeScenarioParameters(
            scenario_type=MarketScenario.MARKET_CRASH,
            shock_magnitude=-0.4,  # 更严重的冲击
            shock_duration=10,
            recovery_time=120,
            volatility_spike=4.0,
            correlation_increase=0.8,
            liquidity_impact=0.1,
            contagion_probability=0.9,
            fat_tail_parameter=2.0
        )
        
        # 使用自定义参数运行模拟
        result = engine.run_extreme_scenario_simulation(
            portfolio_weights, asset_returns, MarketScenario.MARKET_CRASH, custom_params
        )
        
        # 验证使用了自定义参数
        assert result.statistics['shock_magnitude'] == -0.4
        assert result.statistics['shock_duration'] == 10
        assert result.statistics['volatility_spike'] == 4.0
    
    def test_extreme_scenario_simulator_initialization(self, config):
        """测试极端情景模拟器初始化"""
        from src.rl_trading_system.risk_control.stress_test_engine import ExtremeScenarioSimulator
        
        simulator = ExtremeScenarioSimulator(config)
        
        # 验证初始化
        assert simulator.config == config
        assert isinstance(simulator.scenario_parameters, dict)
        assert len(simulator.scenario_parameters) > 0
        
        # 验证默认参数
        for scenario, params in simulator.scenario_parameters.items():
            assert isinstance(params, ExtremeScenarioParameters)
            assert params.scenario_type == scenario


if __name__ == '__main__':
    pytest.main([__file__])