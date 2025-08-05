#!/usr/bin/env python3
"""
测试backtest.py模块功能
使用训练好的模型验证回测功能
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtest import BacktestAnalyzer
from data_loader import QlibDataLoader, split_data
from stable_baselines3 import PPO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主测试函数"""
    model_path = "models/final_model_20250805_081244.zip"
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"模型文件不存在: {model_path}")
    
    logger.info("开始测试backtest.py模块...")
    
    # 加载数据
    logger.info("加载测试数据...")
    loader = QlibDataLoader()
    loader.initialize_qlib()
    
    # 获取与训练时相同的股票数量和配置
    stocks = loader.get_stock_list(market='csi300', limit=10)
    data = loader.load_data(
        instruments=stocks,
        start_time='2023-01-01',
        end_time='2023-12-31',
        freq='day'
    )
    
    # 分割数据
    train_data, valid_data, test_data = split_data(
        data,
        '2023-01-01', '2023-08-31',
        '2023-09-01', '2023-10-31', 
        '2023-11-01', '2023-12-31'
    )
    
    logger.info(f"测试数据形状: {test_data.shape}")
    
    # 加载模型
    logger.info("加载训练好的模型...")
    model = PPO.load(model_path)
    
    # 创建回测分析器并运行回测
    logger.info("开始回测分析...")
    analyzer = BacktestAnalyzer()
    
    try:
        results = analyzer.run_backtest(model, test_data)
        
        # 显示回测结果
        logger.info("回测完成，显示结果...")
        perf = results.get('performance_metrics', {})
        
        rl_metrics = perf.get('rl_strategy', {})
        bench_metrics = perf.get('benchmark', {})
        
        print("\n" + "="*60)
        print("回测结果摘要")
        print("="*60)
        print(f"RL策略总收益率: {rl_metrics.get('total_return', 0):.2%}")
        print(f"RL策略年化收益率: {rl_metrics.get('annualized_return', 0):.2%}")
        print(f"RL策略最大回撤: {rl_metrics.get('max_drawdown', 0):.2%}")
        print(f"RL策略夏普比率: {rl_metrics.get('sharpe_ratio', 0):.3f}")
        print()
        print(f"基准总收益率: {bench_metrics.get('total_return', 0):.2%}")
        print(f"基准年化收益率: {bench_metrics.get('annualized_return', 0):.2%}")
        print(f"基准最大回撤: {bench_metrics.get('max_drawdown', 0):.2%}")
        print(f"基准夏普比率: {bench_metrics.get('sharpe_ratio', 0):.3f}")
        
        if 'relative' in perf:
            rel_metrics = perf['relative']
            print()
            print(f"超额收益: {rel_metrics.get('excess_return', 0):.2%}")
            print(f"信息比率: {rel_metrics.get('information_ratio', 0):.3f}")
        
        print("="*60)
        
        # 测试报告生成
        logger.info("生成性能报告...")
        report = analyzer.create_performance_report()
        with open('results/backtest_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("性能报告已保存到: results/backtest_report.txt")
        
        # 测试图表绘制
        logger.info("生成性能图表...")
        analyzer.plot_performance(save_path='results/backtest_analysis.png')
        
        # 测试Excel导出
        logger.info("导出Excel结果...")
        analyzer.export_results('results/backtest_results.xlsx')
        
        logger.info("所有回测功能测试完成！")
        
    except Exception as e:
        logger.error(f"回测测试失败: {e}")
        raise

if __name__ == "__main__":
    main()