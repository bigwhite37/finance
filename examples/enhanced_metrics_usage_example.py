"""
增强指标使用示例

展示如何使用新增的投资组合指标、智能体行为指标和风险控制指标。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.rl_trading_system.training.enhanced_trainer import (
    EnhancedRLTrainer,
    EnhancedTrainingConfig
)
from src.rl_trading_system.metrics.portfolio_metrics import (
    PortfolioMetricsCalculator,
    PortfolioMetrics,
    AgentBehaviorMetrics,
    RiskControlMetrics
)
from src.rl_trading_system.risk_control.enhanced_adaptive_risk_budget import (
    EnhancedAdaptiveRiskBudget,
    EnhancedAdaptiveRiskBudgetConfig
)
from src.rl_trading_system.risk_control.enhanced_drawdown_controller import (
    EnhancedDrawdownController
)
from src.rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig


def demonstrate_portfolio_metrics():
    """演示投资组合指标计算"""
    print("=" * 60)
    print("📊 投资组合指标计算演示")
    print("=" * 60)
    
    # 创建指标计算器
    calculator = PortfolioMetricsCalculator()
    
    # 模拟投资组合和基准数据
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # 投资组合：年化收益15%，波动率20%
    np.random.seed(42)
    portfolio_returns = np.random.normal(0.15/252, 0.20/np.sqrt(252), 252)
    portfolio_values = [1000000]
    for ret in portfolio_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # 基准：年化收益8%，波动率15%
    benchmark_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), 252)
    benchmark_values = [1000000]
    for ret in benchmark_returns:
        benchmark_values.append(benchmark_values[-1] * (1 + ret))
    
    # 计算指标
    metrics = calculator.calculate_portfolio_metrics(
        portfolio_values=portfolio_values,
        benchmark_values=benchmark_values,
        dates=dates.tolist(),
        risk_free_rate=0.03
    )
    
    # 显示结果
    print(f"夏普比率 (Sharpe Ratio): {metrics.sharpe_ratio:.4f}")
    print(f"最大回撤 (Max Drawdown): {metrics.max_drawdown:.4f}")
    print(f"Alpha (相对基准超额收益): {metrics.alpha:.4f}")
    print(f"Beta (系统性风险): {metrics.beta:.4f}")
    print(f"年化收益率 (Annualized Return): {metrics.annualized_return:.4f}")
    
    # 解释指标含义
    print("\n📈 指标解释:")
    if metrics.sharpe_ratio > 1.0:
        print("✅ 夏普比率 > 1.0，风险调整后收益良好")
    else:
        print("⚠️ 夏普比率 < 1.0，风险调整后收益一般")
    
    if metrics.alpha > 0:
        print(f"✅ Alpha > 0，相对基准有 {metrics.alpha:.2%} 的超额收益")
    else:
        print(f"❌ Alpha < 0，相对基准有 {abs(metrics.alpha):.2%} 的负超额收益")
    
    if metrics.max_drawdown < 0.15:
        print("✅ 最大回撤 < 15%，风险控制良好")
    else:
        print("⚠️ 最大回撤 > 15%，需要加强风险控制")


def demonstrate_agent_behavior_metrics():
    """演示智能体行为指标计算"""
    print("\n" + "=" * 60)
    print("🤖 智能体行为指标计算演示")
    print("=" * 60)
    
    calculator = PortfolioMetricsCalculator()
    
    # 模拟智能体训练过程中的熵值变化（从探索到利用）
    entropy_values = []
    for i in range(100):
        # 熵值从高到低，模拟从探索到利用的过程
        entropy = 3.0 * np.exp(-i / 50) + 0.5 + np.random.normal(0, 0.1)
        entropy_values.append(max(entropy, 0.1))  # 确保熵值为正
    
    # 模拟持仓权重历史（逐渐集中）
    position_weights_history = []
    for i in range(100):
        # 初期分散，后期集中
        concentration_factor = i / 100
        if i < 50:
            # 早期：相对分散
            weights = np.random.dirichlet([1, 1, 1, 1, 1])
        else:
            # 后期：逐渐集中
            weights = np.random.dirichlet([3, 2, 1, 0.5, 0.5])
        position_weights_history.append(weights)
    
    # 计算指标
    metrics = calculator.calculate_agent_behavior_metrics(
        entropy_values=entropy_values,
        position_weights_history=position_weights_history
    )
    
    # 显示结果
    print(f"平均熵值 (Mean Entropy): {metrics.mean_entropy:.4f}")
    print(f"熵值趋势 (Entropy Trend): {metrics.entropy_trend:.4f}")
    print(f"平均持仓集中度 (Position Concentration): {metrics.mean_position_concentration:.4f}")
    print(f"换手率 (Turnover Rate): {metrics.turnover_rate:.4f}")
    
    # 解释指标含义
    print("\n🧠 智能体行为分析:")
    if metrics.entropy_trend < 0:
        print("✅ 熵值下降趋势，智能体从探索转向利用，学习正常")
    else:
        print("⚠️ 熵值上升趋势，智能体可能过度探索")
    
    if 0.3 < metrics.mean_position_concentration < 0.7:
        print("✅ 持仓集中度适中，既有分散又有重点")
    elif metrics.mean_position_concentration > 0.7:
        print("⚠️ 持仓过度集中，风险较高")
    else:
        print("⚠️ 持仓过度分散，可能错失机会")
    
    if metrics.turnover_rate < 0.5:
        print("✅ 换手率适中，交易成本可控")
    else:
        print("⚠️ 换手率较高，注意交易成本")


def demonstrate_risk_control_metrics():
    """演示风险控制指标计算"""
    print("\n" + "=" * 60)
    print("🛡️ 风险控制指标计算演示")
    print("=" * 60)
    
    calculator = PortfolioMetricsCalculator()
    
    # 模拟风险预算和使用历史
    risk_budget_history = []
    risk_usage_history = []
    
    for i in range(100):
        # 风险预算根据市场情况调整
        if i < 30:
            # 初期：保守
            budget = 0.08 + np.random.normal(0, 0.01)
        elif i < 70:
            # 中期：积极
            budget = 0.12 + np.random.normal(0, 0.02)
        else:
            # 后期：回归保守
            budget = 0.10 + np.random.normal(0, 0.015)
        
        risk_budget_history.append(max(budget, 0.02))
        
        # 风险使用通常低于预算
        usage = budget * (0.7 + np.random.normal(0, 0.1))
        risk_usage_history.append(max(usage, 0.01))
    
    # 模拟控制信号
    control_signals = []
    for i in range(20):
        signal_types = ['position_adjustment', 'stop_loss', 'risk_budget_change']
        signal = {
            'type': np.random.choice(signal_types),
            'timestamp': datetime.now() - timedelta(days=i)
        }
        control_signals.append(signal)
    
    # 模拟市场状态历史
    market_regime_history = []
    current_regime = 'bull'
    for i in range(100):
        # 偶尔切换市场状态
        if np.random.random() < 0.05:  # 5%概率切换
            current_regime = np.random.choice(['bull', 'bear', 'neutral'])
        market_regime_history.append(current_regime)
    
    # 计算指标
    metrics = calculator.calculate_risk_control_metrics(
        risk_budget_history=risk_budget_history,
        risk_usage_history=risk_usage_history,
        control_signals=control_signals,
        market_regime_history=market_regime_history
    )
    
    # 显示结果
    print(f"平均风险预算使用率: {metrics.avg_risk_budget_utilization:.4f}")
    print(f"风险预算效率: {metrics.risk_budget_efficiency:.4f}")
    print(f"控制信号频率: {metrics.control_signal_frequency:.4f}")
    print(f"市场状态稳定性: {metrics.market_regime_stability:.4f}")
    
    # 解释指标含义
    print("\n🎯 风险控制分析:")
    if 0.6 < metrics.avg_risk_budget_utilization < 0.9:
        print("✅ 风险预算使用率适中，风险控制有效")
    elif metrics.avg_risk_budget_utilization > 0.9:
        print("⚠️ 风险预算使用率过高，可能风险暴露过大")
    else:
        print("⚠️ 风险预算使用率过低，可能过于保守")
    
    if metrics.risk_budget_efficiency > 1.0:
        print("✅ 风险预算效率良好，风险收益比合理")
    else:
        print("⚠️ 风险预算效率偏低，需要优化风险配置")
    
    if metrics.market_regime_stability > 0.8:
        print("✅ 市场状态稳定，策略执行环境良好")
    else:
        print("⚠️ 市场状态变化频繁，需要增强适应性")


def demonstrate_enhanced_risk_budget():
    """演示增强的自适应风险预算"""
    print("\n" + "=" * 60)
    print("💰 增强自适应风险预算演示")
    print("=" * 60)
    
    # 创建增强配置
    config = EnhancedAdaptiveRiskBudgetConfig(
        base_risk_budget=0.1,
        enable_detailed_logging=True,
        log_budget_changes=True,
        log_usage_analysis=True,
        budget_change_threshold=0.005  # 0.5%变化就记录日志
    )
    
    # 创建增强风险预算分配器
    risk_budget = EnhancedAdaptiveRiskBudget(config)
    
    # 首先添加初始的表现和市场指标
    from src.rl_trading_system.risk_control.adaptive_risk_budget import PerformanceMetrics, MarketMetrics
    
    # 添加初始表现指标
    initial_perf = PerformanceMetrics(
        sharpe_ratio=1.0, calmar_ratio=1.5, max_drawdown=0.1,
        volatility=0.15, win_rate=0.55, profit_factor=1.3,
        consecutive_losses=2, consecutive_wins=3, total_return=0.08,
        downside_deviation=0.1, sortino_ratio=1.2, var_95=0.02,
        expected_shortfall=0.03, timestamp=datetime.now()
    )
    risk_budget.update_performance_metrics(initial_perf)
    
    # 添加初始市场指标
    initial_market = MarketMetrics(
        market_volatility=0.2, market_trend=0.01, correlation_with_market=0.7,
        liquidity_score=0.8, uncertainty_index=0.4, regime_stability=0.7,
        timestamp=datetime.now()
    )
    risk_budget.update_market_metrics(initial_market)
    
    # 模拟几次预算计算
    print("模拟风险预算调整过程:")
    for i in range(5):
        # 模拟不同的市场和表现条件
        if i == 0:
            print(f"\n第{i+1}次计算 - 初始状态:")
        elif i == 1:
            print(f"\n第{i+1}次计算 - 表现良好:")
            # 添加良好的表现指标
            perf = PerformanceMetrics(
                sharpe_ratio=1.8, calmar_ratio=2.5, max_drawdown=0.05,
                volatility=0.12, win_rate=0.65, profit_factor=2.1,
                consecutive_losses=1, consecutive_wins=8, total_return=0.15,
                downside_deviation=0.08, sortino_ratio=2.2, var_95=0.015,
                expected_shortfall=0.02, timestamp=datetime.now()
            )
            risk_budget.update_performance_metrics(perf)
        elif i == 2:
            print(f"\n第{i+1}次计算 - 市场波动加大:")
            market = MarketMetrics(
                market_volatility=0.35, market_trend=-0.02, correlation_with_market=0.8,
                liquidity_score=0.6, uncertainty_index=0.8, regime_stability=0.4,
                timestamp=datetime.now()
            )
            risk_budget.update_market_metrics(market)
        elif i == 3:
            print(f"\n第{i+1}次计算 - 表现下滑:")
            perf = PerformanceMetrics(
                sharpe_ratio=0.3, calmar_ratio=0.8, max_drawdown=0.18,
                volatility=0.25, win_rate=0.4, profit_factor=0.9,
                consecutive_losses=5, consecutive_wins=2, total_return=-0.05,
                downside_deviation=0.18, sortino_ratio=0.2, var_95=0.04,
                expected_shortfall=0.06, timestamp=datetime.now()
            )
            risk_budget.update_performance_metrics(perf)
        else:
            print(f"\n第{i+1}次计算 - 市场恢复:")
            market = MarketMetrics(
                market_volatility=0.15, market_trend=0.03, correlation_with_market=0.6,
                liquidity_score=0.9, uncertainty_index=0.3, regime_stability=0.8,
                timestamp=datetime.now()
            )
            risk_budget.update_market_metrics(market)
        
        # 计算新的风险预算
        new_budget = risk_budget.calculate_adaptive_risk_budget()
        print(f"计算结果: 风险预算 = {new_budget:.4f}")
    
    # 显示详细预算信息
    print("\n📊 详细预算信息:")
    budget_info = risk_budget.get_detailed_budget_info()
    for key, value in budget_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def demonstrate_enhanced_training_config():
    """演示增强训练配置"""
    print("\n" + "=" * 60)
    print("⚙️ 增强训练配置演示")
    print("=" * 60)
    
    # 创建增强训练配置
    config = EnhancedTrainingConfig(
        # 基础训练参数
        n_episodes=1000,
        max_steps_per_episode=180,
        batch_size=256,
        learning_rate=3e-4,
        
        # 增强指标开关
        enable_portfolio_metrics=True,
        enable_agent_behavior_metrics=True,
        enable_risk_control_metrics=True,
        
        # 指标计算频率
        metrics_calculation_frequency=20,  # 每20个episode计算一次
        
        # 基准和风险参数
        risk_free_rate=0.03,
        
        # 日志配置
        detailed_metrics_logging=True,
        metrics_log_level='INFO'
    )
    
    print("增强训练配置:")
    print(f"  总episodes: {config.n_episodes}")
    print(f"  投资组合指标: {'启用' if config.enable_portfolio_metrics else '禁用'}")
    print(f"  智能体行为指标: {'启用' if config.enable_agent_behavior_metrics else '禁用'}")
    print(f"  风险控制指标: {'启用' if config.enable_risk_control_metrics else '禁用'}")
    print(f"  指标计算频率: 每{config.metrics_calculation_frequency}个episode")
    print(f"  详细日志: {'启用' if config.detailed_metrics_logging else '禁用'}")
    
    print("\n📋 配置建议:")
    print("  • 对于快速实验，可以将metrics_calculation_frequency设为较大值（如50）")
    print("  • 对于详细分析，可以将metrics_calculation_frequency设为较小值（如10）")
    print("  • 在生产环境中，建议禁用detailed_metrics_logging以提高性能")
    print("  • risk_free_rate应根据当前市场环境调整")


def main():
    """主函数"""
    print("🚀 增强指标系统使用示例")
    print("本示例展示了新增的投资组合指标、智能体行为指标和风险控制指标的使用方法")
    
    # 演示各种指标计算
    demonstrate_portfolio_metrics()
    demonstrate_agent_behavior_metrics()
    demonstrate_risk_control_metrics()
    demonstrate_enhanced_risk_budget()
    demonstrate_enhanced_training_config()
    
    print("\n" + "=" * 60)
    print("✅ 示例演示完成")
    print("=" * 60)
    print("\n📚 使用指南:")
    print("1. 在训练脚本中使用EnhancedRLTrainer替代原有的RLTrainer")
    print("2. 配置EnhancedTrainingConfig启用需要的指标类型")
    print("3. 在环境中启用EnhancedDrawdownController获取风险控制指标")
    print("4. 通过日志观察详细的指标分析结果")
    print("5. 使用get_enhanced_training_stats()获取编程接口的统计信息")
    
    print("\n🎯 关键指标解读:")
    print("• 夏普比率 > 1.0: 风险调整后收益良好")
    print("• 最大回撤 < 15%: 风险控制有效")
    print("• Alpha > 0: 相对基准有超额收益")
    print("• 熵值下降: 智能体学习正常（从探索到利用）")
    print("• 换手率 < 50%: 交易成本可控")
    print("• 风险预算使用率 60%-90%: 风险配置合理")


if __name__ == "__main__":
    main()