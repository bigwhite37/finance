"""
回撤归因分析器使用示例

本示例展示如何使用DrawdownAttributionAnalyzer进行回撤归因分析，
包括数据准备、分析执行、可视化生成和报告输出。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl_trading_system.risk_control.drawdown_attribution_analyzer import (
    DrawdownAttributionAnalyzer,
    StockInfo
)


def create_sample_data():
    """创建示例数据"""
    
    # 1. 股票基本信息
    stock_info = {
        '000001.SZ': StockInfo('000001.SZ', '平安银行', '金融', '银行', 2.5e11),
        '000002.SZ': StockInfo('000002.SZ', '万科A', '房地产', '房地产开发', 1.8e11),
        '000858.SZ': StockInfo('000858.SZ', '五粮液', '消费', '白酒', 4.5e11),
        '002415.SZ': StockInfo('002415.SZ', '海康威视', '科技', '安防设备', 4.2e11),
        '600036.SH': StockInfo('600036.SH', '招商银行', '金融', '银行', 3.8e11),
        '600519.SH': StockInfo('600519.SH', '贵州茅台', '消费', '白酒', 2.2e12),
        '600887.SH': StockInfo('600887.SH', '伊利股份', '消费', '乳制品', 2.1e11),
        '000858.SZ': StockInfo('000858.SZ', '五粮液', '消费', '白酒', 4.5e11)
    }
    
    # 2. 因子载荷数据
    factor_loadings = {
        '000001.SZ': {'size_factor': 0.6, 'value_factor': 0.8, 'growth_factor': 0.2, 'momentum_factor': -0.1},
        '000002.SZ': {'size_factor': 0.7, 'value_factor': 0.6, 'growth_factor': 0.1, 'momentum_factor': -0.2},
        '000858.SZ': {'size_factor': 0.8, 'value_factor': 0.3, 'growth_factor': 0.6, 'momentum_factor': 0.4},
        '002415.SZ': {'size_factor': 0.9, 'value_factor': -0.2, 'growth_factor': 0.8, 'momentum_factor': 0.3},
        '600036.SH': {'size_factor': 0.7, 'value_factor': 0.7, 'growth_factor': 0.3, 'momentum_factor': 0.1},
        '600519.SH': {'size_factor': 1.0, 'value_factor': 0.2, 'growth_factor': 0.7, 'momentum_factor': 0.5},
        '600887.SH': {'size_factor': 0.5, 'value_factor': 0.4, 'growth_factor': 0.5, 'momentum_factor': 0.2}
    }
    
    # 3. 投资组合持仓
    portfolio_positions = {
        '000001.SZ': 0.15,  # 平安银行
        '000002.SZ': 0.10,  # 万科A
        '000858.SZ': 0.20,  # 五粮液
        '002415.SZ': 0.15,  # 海康威视
        '600036.SH': 0.15,  # 招商银行
        '600519.SH': 0.15,  # 贵州茅台
        '600887.SH': 0.10   # 伊利股份
    }
    
    # 4. 模拟回撤期间的收益率数据
    drawdown_returns = {
        '000001.SZ': -0.12,  # 银行股大跌
        '000002.SZ': -0.15,  # 房地产调控影响
        '000858.SZ': -0.08,  # 白酒相对抗跌
        '002415.SZ': -0.10,  # 科技股调整
        '600036.SH': -0.09,  # 银行股跟跌
        '600519.SH': -0.05,  # 茅台相对抗跌
        '600887.SH': -0.07   # 消费股小幅下跌
    }
    
    # 5. 因子收益率
    factor_returns = {
        'size_factor': -0.03,    # 大盘股跑输小盘股
        'value_factor': -0.02,   # 价值因子表现不佳
        'growth_factor': -0.04,  # 成长因子大幅下跌
        'momentum_factor': -0.01 # 动量因子轻微下跌
    }
    
    # 6. 基准收益率（可选）
    benchmark_returns = {stock: -0.08 for stock in portfolio_positions.keys()}  # 假设基准下跌8%
    
    return stock_info, factor_loadings, portfolio_positions, drawdown_returns, factor_returns, benchmark_returns


def demonstrate_basic_attribution_analysis():
    """演示基本的归因分析功能"""
    print("=" * 60)
    print("回撤归因分析器使用示例")
    print("=" * 60)
    
    # 准备数据
    stock_info, factor_loadings, positions, returns, factor_returns, benchmark_returns = create_sample_data()
    
    # 创建分析器
    analyzer = DrawdownAttributionAnalyzer(
        stock_info=stock_info,
        factor_loadings=factor_loadings,
        lookback_window=30,
        min_contribution_threshold=0.005
    )
    
    print("1. 执行回撤归因分析...")
    
    # 执行归因分析
    result = analyzer.analyze_drawdown_attribution(
        current_positions=positions,
        position_returns=returns,
        benchmark_returns=benchmark_returns,
        factor_returns=factor_returns
    )
    
    # 显示基本结果
    print(f"\n总回撤: {result.total_drawdown:.2%}")
    print(f"归因置信度: {result.confidence_score:.2%}")
    print(f"未解释部分: {result.unexplained_portion:.2%}")
    
    # 显示个股贡献
    print("\n2. 个股贡献分析:")
    print("-" * 40)
    for stock, contribution in sorted(result.stock_contributions.items(), 
                                    key=lambda x: x[1]):
        stock_name = stock_info.get(stock, type('obj', (object,), {'name': stock})).name
        print(f"{stock_name:12} ({stock:10}): {contribution:8.2%}")
    
    # 显示行业贡献
    print("\n3. 行业贡献分析:")
    print("-" * 40)
    for sector, contribution in sorted(result.sector_contributions.items(), 
                                     key=lambda x: x[1]):
        print(f"{sector:12}: {contribution:8.2%}")
    
    # 显示因子贡献
    print("\n4. 因子贡献分析:")
    print("-" * 40)
    for factor, contribution in sorted(result.factor_contributions.items(), 
                                     key=lambda x: x[1]):
        print(f"{factor:12}: {contribution:8.2%}")
    
    return analyzer, result


def demonstrate_visualization_and_reporting():
    """演示可视化和报告生成功能"""
    print("\n" + "=" * 60)
    print("可视化和报告生成演示")
    print("=" * 60)
    
    # 获取分析器和结果
    analyzer, result = demonstrate_basic_attribution_analysis()
    
    # 创建输出目录
    output_dir = Path("attribution_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    print("\n5. 生成可视化图表...")
    
    try:
        # 生成可视化图表
        figures = analyzer.generate_attribution_visualization(
            result=result,
            save_path=str(output_dir / "charts")
        )
        
        print(f"生成了 {len(figures)} 个图表:")
        for chart_name in figures.keys():
            print(f"  - {chart_name}")
        
        print(f"图表已保存到: {output_dir / 'charts'}")
        
    except Exception as e:
        print(f"可视化生成失败: {e}")
        print("可能需要安装 plotly 和 kaleido: pip install plotly kaleido")
    
    print("\n6. 生成归因分析报告...")
    
    # 生成报告
    report = analyzer.generate_attribution_report(
        result=result,
        save_path=str(output_dir / "attribution_report")
    )
    
    # 显示报告摘要
    summary = report['executive_summary']
    print(f"\n执行摘要:")
    print(f"  总回撤: {summary['total_drawdown_pct']}")
    print(f"  置信度等级: {summary['confidence_level']}")
    print(f"  最大个股贡献: {summary['main_contributors']['top_stock']['name']} "
          f"({summary['main_contributors']['top_stock']['contribution']})")
    print(f"  最大行业贡献: {summary['main_contributors']['top_sector']['name']} "
          f"({summary['main_contributors']['top_sector']['contribution']})")
    
    print(f"\n关键发现:")
    for finding in summary['key_findings']:
        print(f"  - {finding}")
    
    print(f"\n风险洞察:")
    for insight in report['risk_insights']:
        print(f"  - {insight}")
    
    print(f"\n改进建议:")
    for recommendation in report['recommendations']:
        print(f"  - {recommendation}")
    
    print(f"\n报告已保存到: {output_dir}")
    
    return analyzer, report


def demonstrate_historical_analysis():
    """演示历史归因分析功能"""
    print("\n" + "=" * 60)
    print("历史归因分析演示")
    print("=" * 60)
    
    # 准备数据
    stock_info, factor_loadings, positions, _, factor_returns, benchmark_returns = create_sample_data()
    
    # 创建分析器
    analyzer = DrawdownAttributionAnalyzer(
        stock_info=stock_info,
        factor_loadings=factor_loadings,
        lookback_window=20
    )
    
    print("7. 模拟多期归因分析...")
    
    # 模拟10期的归因分析
    np.random.seed(42)  # 确保结果可重现
    
    for period in range(10):
        # 生成随机收益率（模拟不同时期的市场表现）
        base_returns = np.random.normal(-0.02, 0.05, len(positions))  # 平均-2%，标准差5%
        period_returns = {
            stock: max(-0.20, min(0.10, ret))  # 限制在-20%到10%之间
            for stock, ret in zip(positions.keys(), base_returns)
        }
        
        # 执行归因分析
        result = analyzer.analyze_drawdown_attribution(
            current_positions=positions,
            position_returns=period_returns,
            factor_returns=factor_returns
        )
        
        print(f"  期间 {period+1:2d}: 回撤 {result.total_drawdown:7.2%}, "
              f"置信度 {result.confidence_score:6.2%}")
    
    print("\n8. 生成历史归因摘要...")
    
    # 获取历史摘要
    historical_summary = analyzer.get_historical_attribution_summary()
    
    print(f"\n历史分析摘要:")
    print(f"  分析期间: {historical_summary['analysis_period']['start_date'][:10]} 到 "
          f"{historical_summary['analysis_period']['end_date'][:10]}")
    print(f"  总分析次数: {historical_summary['analysis_period']['total_analyses']}")
    
    print(f"\n回撤统计:")
    stats = historical_summary['drawdown_statistics']
    print(f"  平均回撤: {stats['average_drawdown']}")
    print(f"  最大回撤: {stats['max_drawdown']}")
    print(f"  最小回撤: {stats['min_drawdown']}")
    print(f"  回撤波动率: {stats['drawdown_volatility']}")
    
    print(f"\n置信度统计:")
    conf_stats = historical_summary['confidence_statistics']
    print(f"  平均置信度: {conf_stats['average_confidence']}")
    print(f"  最低置信度: {conf_stats['min_confidence']}")
    print(f"  最高置信度: {conf_stats['max_confidence']}")
    
    print(f"\n频繁贡献者 (个股):")
    for contributor in historical_summary['frequent_contributors']['stocks'][:5]:
        print(f"  {contributor['name']:12}: 出现 {contributor['frequency']} 次, "
              f"平均贡献 {contributor['average_contribution']}")
    
    print(f"\n频繁贡献者 (行业):")
    for contributor in historical_summary['frequent_contributors']['sectors'][:5]:
        print(f"  {contributor['name']:12}: 出现 {contributor['frequency']} 次, "
              f"平均贡献 {contributor['average_contribution']}")


def demonstrate_edge_cases():
    """演示边界情况处理"""
    print("\n" + "=" * 60)
    print("边界情况处理演示")
    print("=" * 60)
    
    # 创建简单的分析器
    analyzer = DrawdownAttributionAnalyzer()
    
    print("9. 测试各种边界情况...")
    
    # 情况1: 空输入
    print("\n  情况1: 空输入处理")
    result = analyzer.analyze_drawdown_attribution({}, {})
    print(f"    空输入结果 - 总回撤: {result.total_drawdown:.2%}, "
          f"置信度: {result.confidence_score:.2%}")
    
    # 情况2: 单一持仓
    print("\n  情况2: 单一持仓")
    single_position = {'STOCK_A': 1.0}
    single_return = {'STOCK_A': -0.10}
    result = analyzer.analyze_drawdown_attribution(single_position, single_return)
    print(f"    单一持仓结果 - 总回撤: {result.total_drawdown:.2%}, "
          f"个股贡献数: {len(result.stock_contributions)}")
    
    # 情况3: 正收益（无回撤）
    print("\n  情况3: 正收益情况")
    positive_positions = {'STOCK_A': 0.5, 'STOCK_B': 0.5}
    positive_returns = {'STOCK_A': 0.05, 'STOCK_B': 0.03}
    result = analyzer.analyze_drawdown_attribution(positive_positions, positive_returns)
    print(f"    正收益结果 - 总回撤: {result.total_drawdown:.2%}, "
          f"个股贡献数: {len(result.stock_contributions)}")
    
    # 情况4: 混合收益
    print("\n  情况4: 混合收益")
    mixed_positions = {'STOCK_A': 0.3, 'STOCK_B': 0.3, 'STOCK_C': 0.4}
    mixed_returns = {'STOCK_A': 0.02, 'STOCK_B': -0.08, 'STOCK_C': -0.05}
    result = analyzer.analyze_drawdown_attribution(mixed_positions, mixed_returns)
    print(f"    混合收益结果 - 总回撤: {result.total_drawdown:.2%}, "
          f"负贡献股票数: {len(result.stock_contributions)}")
    
    # 情况5: 极小权重过滤
    print("\n  情况5: 极小权重过滤")
    tiny_positions = {'STOCK_A': 0.001, 'STOCK_B': 0.999}  # STOCK_A权重极小
    large_loss_returns = {'STOCK_A': -0.50, 'STOCK_B': -0.02}  # STOCK_A大幅下跌但权重小
    result = analyzer.analyze_drawdown_attribution(tiny_positions, large_loss_returns)
    print(f"    极小权重结果 - 总回撤: {result.total_drawdown:.2%}")
    print(f"    贡献股票: {list(result.stock_contributions.keys())}")


def main():
    """主函数"""
    try:
        # 基本归因分析演示
        demonstrate_basic_attribution_analysis()
        
        # 可视化和报告演示
        demonstrate_visualization_and_reporting()
        
        # 历史分析演示
        demonstrate_historical_analysis()
        
        # 边界情况演示
        demonstrate_edge_cases()
        
        print("\n" + "=" * 60)
        print("回撤归因分析器演示完成！")
        print("=" * 60)
        
        print("\n使用建议:")
        print("1. 确保提供完整的股票基本信息以提高归因准确性")
        print("2. 定期更新因子载荷数据以反映最新的因子暴露")
        print("3. 设置合适的最小贡献阈值以过滤噪音")
        print("4. 结合历史趋势分析识别持续性风险因素")
        print("5. 根据归因结果及时调整投资组合配置")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()