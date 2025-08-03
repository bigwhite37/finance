"""
压力测试引擎使用示例
演示如何使用压力测试引擎进行各种风险分析
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl_trading_system.risk_control.stress_test_engine import (
    StressTestEngine, StressTestConfig, StressTestType, MarketScenario,
    ExtremeScenarioParameters
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data():
    """生成示例数据"""
    # 生成模拟的资产收益率数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    assets = ['沪深300', '中证500', '创业板指', '科创50', '上证50']
    
    # 生成相关的收益率数据
    returns_data = np.random.multivariate_normal(
        mean=[0.0005, 0.0008, 0.0012, 0.0015, 0.0003],  # 日收益率均值
        cov=[[0.0004, 0.0002, 0.0003, 0.0002, 0.0001],  # 协方差矩阵
             [0.0002, 0.0006, 0.0004, 0.0003, 0.0002],
             [0.0003, 0.0004, 0.0009, 0.0005, 0.0002],
             [0.0002, 0.0003, 0.0005, 0.0012, 0.0001],
             [0.0001, 0.0002, 0.0002, 0.0001, 0.0003]],
        size=len(dates)
    )
    
    asset_returns = pd.DataFrame(returns_data, index=dates, columns=assets)
    portfolio_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    
    return asset_returns, portfolio_weights


def basic_stress_testing_example():
    """基础压力测试示例"""
    logger.info("=== 基础压力测试示例 ===")
    
    # 生成示例数据
    asset_returns, portfolio_weights = generate_sample_data()
    
    # 创建压力测试配置
    config = StressTestConfig(
        confidence_levels=[0.95, 0.99, 0.999],
        time_horizons=[1, 5, 10, 22],
        num_simulations=5000,
        random_seed=42
    )
    
    # 初始化压力测试引擎
    engine = StressTestEngine(config)
    
    # 执行不同类型的压力测试
    test_types = [
        StressTestType.MONTE_CARLO,
        StressTestType.HISTORICAL_SCENARIO,
        StressTestType.PARAMETRIC_VAR,
        StressTestType.EXTREME_VALUE
    ]
    
    results = {}
    for test_type in test_types:
        logger.info(f"执行 {test_type.value} 压力测试...")
        result = engine.run_stress_test(portfolio_weights, asset_returns, test_type)
        results[test_type.value] = result
        
        # 打印关键指标
        logger.info(f"  99% VaR: {result.var_estimates[0.99]:.4f}")
        logger.info(f"  99% CVaR: {result.cvar_estimates[0.99]:.4f}")
        logger.info(f"  最大损失: {result.max_loss:.4f}")
        logger.info(f"  损失概率: {result.probability_of_loss:.4f}")
    
    return results


def extreme_scenario_simulation_example():
    """极端情景模拟示例"""
    logger.info("=== 极端情景模拟示例 ===")
    
    # 生成示例数据
    asset_returns, portfolio_weights = generate_sample_data()
    
    # 创建压力测试引擎
    config = StressTestConfig(num_simulations=3000)
    engine = StressTestEngine(config)
    
    # 测试不同的极端情景
    scenarios = [
        MarketScenario.MARKET_CRASH,
        MarketScenario.LIQUIDITY_DROUGHT,
        MarketScenario.VOLATILITY_SPIKE,
        MarketScenario.BLACK_SWAN
    ]
    
    scenario_results = {}
    for scenario in scenarios:
        logger.info(f"模拟 {scenario.value} 情景...")
        result = engine.run_extreme_scenario_simulation(
            portfolio_weights, asset_returns, scenario
        )
        scenario_results[scenario.value] = result
        
        # 打印情景特定指标
        logger.info(f"  情景类型: {result.statistics['scenario_type']}")
        logger.info(f"  冲击幅度: {result.statistics['shock_magnitude']:.4f}")
        logger.info(f"  99% VaR: {result.var_estimates[0.99]:.4f}")
        logger.info(f"  最大损失: {result.max_loss:.4f}")
    
    return scenario_results


def custom_scenario_parameters_example():
    """自定义情景参数示例"""
    logger.info("=== 自定义情景参数示例 ===")
    
    # 生成示例数据
    asset_returns, portfolio_weights = generate_sample_data()
    
    # 创建压力测试引擎
    config = StressTestConfig(num_simulations=2000)
    engine = StressTestEngine(config)
    
    # 定义自定义的极端市场崩盘情景
    custom_crash_params = ExtremeScenarioParameters(
        scenario_type=MarketScenario.MARKET_CRASH,
        shock_magnitude=-0.5,  # 50%的市场下跌
        shock_duration=10,     # 持续10天
        recovery_time=180,     # 180天恢复期
        volatility_spike=5.0,  # 波动率增加5倍
        correlation_increase=0.9,  # 相关性增加到0.9
        liquidity_impact=0.1,  # 10%的流动性成本
        contagion_probability=0.95,  # 95%的传染概率
        fat_tail_parameter=2.0
    )
    
    # 使用自定义参数运行模拟
    logger.info("使用自定义参数模拟极端市场崩盘...")
    result = engine.run_extreme_scenario_simulation(
        portfolio_weights, asset_returns, 
        MarketScenario.MARKET_CRASH, custom_crash_params
    )
    
    logger.info(f"自定义情景结果:")
    logger.info(f"  冲击幅度: {result.statistics['shock_magnitude']:.4f}")
    logger.info(f"  持续时间: {result.statistics['shock_duration']} 天")
    logger.info(f"  波动率倍数: {result.statistics['volatility_spike']:.1f}")
    logger.info(f"  99% VaR: {result.var_estimates[0.99]:.4f}")
    logger.info(f"  99.9% VaR: {result.var_estimates[0.999]:.4f}")
    logger.info(f"  最大损失: {result.max_loss:.4f}")
    
    return result


def scenario_calibration_example():
    """情景校准示例"""
    logger.info("=== 情景校准示例 ===")
    
    # 生成示例数据
    asset_returns, portfolio_weights = generate_sample_data()
    
    # 创建压力测试引擎
    config = StressTestConfig()
    engine = StressTestEngine(config)
    
    # 校准情景参数
    logger.info("基于历史数据校准情景参数...")
    calibrated_params = engine.calibrate_extreme_scenarios(asset_returns)
    
    for scenario, params in calibrated_params.items():
        logger.info(f"{scenario.value} 校准参数:")
        logger.info(f"  冲击幅度: {params.shock_magnitude:.4f}")
        logger.info(f"  持续时间: {params.shock_duration} 天")
        logger.info(f"  波动率倍数: {params.volatility_spike:.2f}")
        logger.info(f"  相关性增加: {params.correlation_increase:.2f}")
    
    # 估计情景概率
    logger.info("估计情景发生概率...")
    probabilities = engine.estimate_scenario_probabilities(asset_returns)
    
    for scenario, prob in probabilities.items():
        logger.info(f"{scenario.value} 年化概率: {prob * 365:.6f}")
    
    return calibrated_params, probabilities


def comprehensive_stress_testing_example():
    """综合压力测试示例"""
    logger.info("=== 综合压力测试示例 ===")
    
    # 生成示例数据
    asset_returns, portfolio_weights = generate_sample_data()
    
    # 创建压力测试引擎
    config = StressTestConfig(
        confidence_levels=[0.95, 0.99, 0.999],
        num_simulations=3000,
        max_workers=2  # 并行执行
    )
    engine = StressTestEngine(config)
    
    # 执行综合压力测试
    logger.info("执行综合压力测试（并行执行多种测试）...")
    results = engine.run_comprehensive_stress_test(portfolio_weights, asset_returns)
    
    logger.info(f"共完成 {len(results)} 项压力测试")
    
    # 分析结果
    var_99_values = []
    max_losses = []
    
    for test_name, result in results.items():
        var_99 = result.var_estimates.get(0.99, 0)
        max_loss = result.max_loss
        var_99_values.append(var_99)
        max_losses.append(max_loss)
        
        logger.info(f"{test_name}:")
        logger.info(f"  99% VaR: {var_99:.4f}")
        logger.info(f"  最大损失: {max_loss:.4f}")
    
    # 汇总统计
    logger.info("汇总统计:")
    logger.info(f"  平均99% VaR: {np.mean(var_99_values):.4f}")
    logger.info(f"  最大99% VaR: {np.max(var_99_values):.4f}")
    logger.info(f"  平均最大损失: {np.mean(max_losses):.4f}")
    logger.info(f"  最大可能损失: {np.max(max_losses):.4f}")
    
    return results


def risk_management_recommendations_example():
    """风险管理建议示例"""
    logger.info("=== 风险管理建议示例 ===")
    
    # 生成示例数据
    asset_returns, portfolio_weights = generate_sample_data()
    
    # 创建压力测试引擎
    config = StressTestConfig(num_simulations=2000)
    engine = StressTestEngine(config)
    
    # 运行关键压力测试
    results = {}
    key_scenarios = [
        MarketScenario.MARKET_CRASH,
        MarketScenario.LIQUIDITY_DROUGHT,
        MarketScenario.VOLATILITY_SPIKE
    ]
    
    for scenario in key_scenarios:
        result = engine.run_extreme_scenario_simulation(
            portfolio_weights, asset_returns, scenario
        )
        results[scenario.value] = result
    
    # 生成风险限额建议
    logger.info("生成风险管理建议...")
    recommendations = engine.generate_risk_limits_recommendations(
        results, risk_tolerance=0.05
    )
    
    logger.info("风险限额建议:")
    for limit_type, value in recommendations.items():
        if isinstance(value, float):
            logger.info(f"  {limit_type}: {value:.4f}")
        else:
            logger.info(f"  {limit_type}: {value}")
    
    # 生成详细报告
    logger.info("生成压力测试报告...")
    report = engine.generate_stress_test_report(results)
    
    # 保存报告
    report_path = "stress_test_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已保存至: {report_path}")
    
    return recommendations, report


def visualization_example():
    """可视化示例"""
    logger.info("=== 可视化示例 ===")
    
    # 生成示例数据
    asset_returns, portfolio_weights = generate_sample_data()
    
    # 创建压力测试引擎
    config = StressTestConfig(num_simulations=2000)
    engine = StressTestEngine(config)
    
    # 运行几个压力测试
    results = {}
    results['蒙特卡洛模拟'] = engine.run_stress_test(
        portfolio_weights, asset_returns, StressTestType.MONTE_CARLO
    )
    results['历史情景重现'] = engine.run_stress_test(
        portfolio_weights, asset_returns, StressTestType.HISTORICAL_SCENARIO
    )
    results['市场崩盘情景'] = engine.run_extreme_scenario_simulation(
        portfolio_weights, asset_returns, MarketScenario.MARKET_CRASH
    )
    
    # 生成可视化图表
    logger.info("生成可视化图表...")
    html_content = engine.visualize_stress_test_results(results)
    
    # 保存可视化结果
    viz_path = "stress_test_visualization.html"
    with open(viz_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"可视化图表已保存至: {viz_path}")
    
    return html_content


def main():
    """主函数"""
    logger.info("压力测试引擎使用示例")
    logger.info("=" * 50)
    
    try:
        # 1. 基础压力测试
        basic_results = basic_stress_testing_example()
        
        # 2. 极端情景模拟
        scenario_results = extreme_scenario_simulation_example()
        
        # 3. 自定义情景参数
        custom_result = custom_scenario_parameters_example()
        
        # 4. 情景校准
        calibrated_params, probabilities = scenario_calibration_example()
        
        # 5. 综合压力测试
        comprehensive_results = comprehensive_stress_testing_example()
        
        # 6. 风险管理建议
        recommendations, report = risk_management_recommendations_example()
        
        # 7. 可视化
        html_content = visualization_example()
        
        logger.info("=" * 50)
        logger.info("所有示例执行完成！")
        logger.info("生成的文件:")
        logger.info("  - stress_test_report.md: 压力测试报告")
        logger.info("  - stress_test_visualization.html: 可视化图表")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")
        raise


if __name__ == "__main__":
    main()