"""
市场状态感知系统使用示例
演示如何使用MarketRegimeDetector进行市场状态识别和风险参数调整
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from src.rl_trading_system.risk_control.market_regime_detector import (
    MarketRegimeDetector, MarketRegimeConfig, MarketRegime, 
    MarketRegimeAnalyzer
)
from src.rl_trading_system.data.data_models import MarketData


def create_sample_market_data(scenario: str = "mixed") -> List[MarketData]:
    """创建示例市场数据"""
    base_time = datetime(2024, 1, 1)
    data_list = []
    
    if scenario == "bull_market":
        # 牛市场景：稳定上涨
        prices = []
        base_price = 100
        for i in range(60):
            # 添加随机波动但整体上涨
            trend = i * 0.5
            noise = np.random.normal(0, 1)
            price = base_price + trend + noise
            prices.append(max(price, 50))  # 确保价格不会太低
        
    elif scenario == "bear_market":
        # 熊市场景：持续下跌
        prices = []
        base_price = 150
        for i in range(60):
            trend = -i * 0.8
            noise = np.random.normal(0, 1.5)
            price = base_price + trend + noise
            prices.append(max(price, 50))
        
    elif scenario == "high_volatility":
        # 高波动场景：剧烈波动
        prices = []
        base_price = 100
        for i in range(60):
            volatility = 5 * np.sin(i * 0.3) + np.random.normal(0, 3)
            price = base_price + volatility
            prices.append(max(price, 50))
        
    elif scenario == "crisis":
        # 危机场景：急剧下跌 + 高波动
        prices = []
        base_price = 120
        for i in range(60):
            if i < 20:
                # 正常期
                price = base_price + np.random.normal(0, 1)
            else:
                # 危机期
                crisis_trend = -(i - 20) * 1.5
                crisis_volatility = np.random.normal(0, 5)
                price = base_price + crisis_trend + crisis_volatility
            prices.append(max(price, 30))
    
    else:  # mixed scenario
        # 混合场景：包含多种市场状态
        prices = []
        base_price = 100
        
        # 第一阶段：牛市 (0-20天)
        for i in range(20):
            price = base_price + i * 0.5 + np.random.normal(0, 0.5)
            prices.append(price)
        
        # 第二阶段：震荡市 (20-40天)
        for i in range(20, 40):
            price = prices[-1] + np.random.normal(0, 2)
            prices.append(price)
        
        # 第三阶段：熊市 (40-60天)
        for i in range(40, 60):
            price = prices[-1] - 0.8 + np.random.normal(0, 1)
            prices.append(max(price, 50))
    
    # 转换为MarketData对象
    for i, price in enumerate(prices):
        timestamp = base_time + timedelta(days=i)
        
        # 生成OHLC数据
        open_price = price + np.random.normal(0, 0.2)
        high_price = max(open_price, price) + abs(np.random.normal(0, 0.5))
        low_price = min(open_price, price) - abs(np.random.normal(0, 0.5))
        close_price = price
        
        volume = int(1000000 + np.random.normal(0, 200000))
        volume = max(volume, 100000)
        
        market_data = MarketData(
            timestamp=timestamp,
            symbol="SAMPLE",
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            amount=close_price * volume
        )
        
        data_list.append(market_data)
    
    return data_list


def basic_usage_example():
    """基本使用示例"""
    print("=== 市场状态检测器基本使用示例 ===\n")
    
    # 1. 创建配置和检测器
    config = MarketRegimeConfig(
        ma_short_period=10,
        ma_long_period=20,
        volatility_window=10,
        regime_persistence=3,
        confidence_threshold=0.6
    )
    
    detector = MarketRegimeDetector(config)
    
    # 2. 创建示例数据
    market_data_list = create_sample_market_data("mixed")
    
    # 3. 逐步更新市场数据并检测状态
    results = []
    
    print("开始处理市场数据...")
    for i, market_data in enumerate(market_data_list):
        result = detector.update_market_data(market_data)
        results.append(result)
        
        # 每10天打印一次状态
        if i % 10 == 0 and i > 0:
            print(f"第{i+1}天:")
            print(f"  当前价格: {market_data.close_price:.2f}")
            print(f"  市场状态: {result.regime.value}")
            print(f"  置信度: {result.confidence:.3f}")
            print(f"  风险调整因子: {result.risk_adjustment_factor:.3f}")
            print(f"  波动率: {result.indicators.volatility:.4f}")
            print(f"  推荐行动: {', '.join(result.recommended_actions[:2])}")
            print()
    
    # 4. 显示最终统计
    final_stats = detector.get_regime_statistics()
    print("=== 最终统计信息 ===")
    print(f"当前状态: {final_stats.get('current_regime', 'N/A')}")
    print(f"状态持续天数: {final_stats.get('regime_duration_days', 0)}")
    print(f"平均波动率: {final_stats.get('avg_volatility', 0):.4f}")
    print(f"平均市场压力: {final_stats.get('avg_market_stress', 0):.3f}")
    print("\n各状态出现频率:")
    for regime, freq in final_stats.get('regime_frequencies', {}).items():
        print(f"  {regime}: {freq:.1%}")


def risk_parameter_adjustment_example():
    """风险参数调整示例"""
    print("\n=== 风险参数调整示例 ===\n")
    
    detector = MarketRegimeDetector()
    
    # 基础风险参数
    base_risk_params = {
        'max_position_size': 0.15,
        'stop_loss_threshold': 0.05,
        'volatility_target': 0.12,
        'leverage_limit': 2.0
    }
    
    print("基础风险参数:")
    for param, value in base_risk_params.items():
        print(f"  {param}: {value}")
    print()
    
    # 测试不同市场状态下的参数调整
    scenarios = ["bull_market", "bear_market", "high_volatility", "crisis"]
    
    for scenario in scenarios:
        print(f"--- {scenario.upper()} 场景 ---")
        
        # 重置检测器
        detector.reset()
        
        # 添加场景数据
        market_data_list = create_sample_market_data(scenario)
        
        # 处理数据直到有足够的历史
        for market_data in market_data_list[:30]:  # 使用前30天数据
            result = detector.update_market_data(market_data)
        
        # 调整风险参数
        adjusted_params = detector.adjust_risk_parameters(
            result.regime, base_risk_params
        )
        
        print(f"检测到的市场状态: {result.regime.value}")
        print(f"风险调整因子: {result.risk_adjustment_factor:.3f}")
        print("调整后的风险参数:")
        for param, value in adjusted_params.items():
            original = base_risk_params[param]
            change = (value - original) / original * 100
            print(f"  {param}: {value:.4f} ({change:+.1f}%)")
        
        print(f"推荐行动: {', '.join(result.recommended_actions)}")
        print()


def historical_analysis_example():
    """历史分析示例"""
    print("\n=== 历史市场状态分析示例 ===\n")
    
    # 创建检测器和分析器
    detector = MarketRegimeDetector()
    analyzer = MarketRegimeAnalyzer(detector)
    
    # 创建长期历史数据
    market_data_list = create_sample_market_data("mixed")
    
    # 分析历史状态
    print("正在分析历史市场状态...")
    historical_df = analyzer.analyze_historical_regimes(market_data_list)
    
    # 生成分析报告
    report = analyzer.generate_regime_report(historical_df)
    print(report)
    
    # 创建模拟的"真实"状态用于准确性评估
    actual_regimes = []
    for i in range(len(historical_df)):
        if i < 20:
            actual_regimes.append(MarketRegime.BULL_MARKET)
        elif i < 40:
            actual_regimes.append(MarketRegime.SIDEWAYS_MARKET)
        else:
            actual_regimes.append(MarketRegime.BEAR_MARKET)
    
    # 评估准确性
    accuracy_metrics = analyzer.evaluate_regime_accuracy(historical_df, actual_regimes)
    
    print(f"\n=== 检测准确性评估 ===")
    print(f"总体准确率: {accuracy_metrics['overall_accuracy']:.1%}")
    print("\n各状态详细指标:")
    for regime, metrics in accuracy_metrics['regime_metrics'].items():
        print(f"{regime}:")
        print(f"  精确率: {metrics['precision']:.3f}")
        print(f"  召回率: {metrics['recall']:.3f}")
        print(f"  F1分数: {metrics['f1_score']:.3f}")


def real_time_monitoring_example():
    """实时监控示例"""
    print("\n=== 实时监控示例 ===\n")
    
    detector = MarketRegimeDetector()
    
    # 模拟实时数据流
    market_data_list = create_sample_market_data("crisis")
    
    print("模拟实时市场数据流...")
    print("监控关键指标变化:\n")
    
    for i, market_data in enumerate(market_data_list):
        result = detector.update_market_data(market_data)
        
        # 检查是否需要告警
        alerts = []
        
        if result.regime == MarketRegime.CRISIS:
            alerts.append("🚨 危机模式激活!")
        elif result.regime == MarketRegime.HIGH_VOLATILITY:
            alerts.append("⚠️  高波动率警告!")
        elif result.indicators.market_stress > 0.7:
            alerts.append("📉 市场压力过高!")
        elif result.indicators.rsi > 80:
            alerts.append("📈 市场超买!")
        elif result.indicators.rsi < 20:
            alerts.append("📉 市场超卖!")
        
        # 每5天或有告警时打印状态
        if i % 5 == 0 or alerts:
            print(f"Day {i+1:2d} | 价格: {market_data.close_price:6.2f} | "
                  f"状态: {result.regime.value:12s} | "
                  f"置信度: {result.confidence:.2f} | "
                  f"风险因子: {result.risk_adjustment_factor:.2f}")
            
            if alerts:
                for alert in alerts:
                    print(f"       {alert}")
            
            if result.regime in [MarketRegime.CRISIS, MarketRegime.HIGH_VOLATILITY]:
                print(f"       推荐: {result.recommended_actions[0] if result.recommended_actions else '无'}")
            
            print()
    
    # 最终状态总结
    print("=== 监控期间总结 ===")
    stats = detector.get_regime_statistics()
    print(f"最终状态: {stats.get('current_regime', 'N/A')}")
    print(f"平均波动率: {stats.get('avg_volatility', 0):.4f}")
    print(f"平均市场压力: {stats.get('avg_market_stress', 0):.3f}")
    
    if detector.is_crisis_mode():
        print("⚠️  系统当前处于危机模式，建议采取紧急风险控制措施!")


def visualization_example():
    """可视化示例"""
    print("\n=== 市场状态可视化示例 ===\n")
    
    try:
        import matplotlib.pyplot as plt
        
        detector = MarketRegimeDetector()
        market_data_list = create_sample_market_data("mixed")
        
        # 收集数据
        timestamps = []
        prices = []
        regimes = []
        volatilities = []
        risk_factors = []
        
        for market_data in market_data_list:
            result = detector.update_market_data(market_data)
            
            timestamps.append(market_data.timestamp)
            prices.append(market_data.close_price)
            regimes.append(result.regime.value)
            volatilities.append(result.indicators.volatility)
            risk_factors.append(result.risk_adjustment_factor)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 价格和市场状态
        ax1.plot(timestamps, prices, 'b-', linewidth=1)
        ax1.set_title('价格走势')
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)
        
        # 为不同状态着色
        regime_colors = {
            'bull': 'green',
            'bear': 'red', 
            'sideways': 'orange',
            'high_vol': 'purple',
            'low_vol': 'blue',
            'crisis': 'black'
        }
        
        for i in range(1, len(timestamps)):
            regime = regimes[i]
            color = regime_colors.get(regime, 'gray')
            ax1.axvspan(timestamps[i-1], timestamps[i], alpha=0.2, color=color)
        
        # 波动率
        ax2.plot(timestamps, volatilities, 'r-', linewidth=1)
        ax2.set_title('市场波动率')
        ax2.set_ylabel('波动率')
        ax2.grid(True, alpha=0.3)
        
        # 风险调整因子
        ax3.plot(timestamps, risk_factors, 'g-', linewidth=1)
        ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('风险调整因子')
        ax3.set_ylabel('调整因子')
        ax3.grid(True, alpha=0.3)
        
        # 状态分布
        regime_counts = pd.Series(regimes).value_counts()
        ax4.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        ax4.set_title('市场状态分布')
        
        plt.tight_layout()
        plt.savefig('market_regime_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'market_regime_analysis.png'")
        
        # 显示图表（如果在交互环境中）
        # plt.show()
        
    except ImportError:
        print("matplotlib未安装，跳过可视化示例")


def main():
    """主函数"""
    print("市场状态感知系统使用示例")
    print("=" * 50)
    
    # 设置随机种子以获得可重复的结果
    np.random.seed(42)
    
    # 运行各种示例
    basic_usage_example()
    risk_parameter_adjustment_example()
    historical_analysis_example()
    real_time_monitoring_example()
    visualization_example()
    
    print("\n所有示例运行完成!")


if __name__ == "__main__":
    main()