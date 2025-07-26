#!/usr/bin/env python3
"""
动态低波筛选器核心回测验证测试

专注于核心回测逻辑和性能指标验证。
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import unittest
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_control.dynamic_lowvol_filter.data_structures import DynamicLowVolConfig
from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter
from tests.unit.test_backtest_utils import (
    BacktestDataGenerator, 
    BacktestMetricsCalculator,
    BacktestConfigFactory,
    BacktestExpectedMetrics
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestCoreBacktestValidator(unittest.TestCase):
    """核心回测验证器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = BacktestConfigFactory.create_backtest_config()
        self.data_manager = BacktestConfigFactory.create_benchmark_data_manager()
        self.expected_metrics = BacktestExpectedMetrics.EXPECTED_METRICS
        
    def test_core_backtest_validation(self):
        """运行核心回测验证"""
        logger.info("开始核心回测验证...")
        
        try:
            # 初始化筛选器
            filter_instance = DynamicLowVolFilter(
                config=DynamicLowVolConfig(**self.config),
                data_manager=self.data_manager
            )
            
            # 获取回测数据
            price_data = self.data_manager.get_price_data()
            returns_data = price_data.pct_change().dropna()
            
            logger.info(f"回测数据规模: {returns_data.shape}")
            
            # 运行筛选策略回测
            filtered_results = self._run_filtered_strategy_backtest(filter_instance, returns_data)
            
            # 运行基准策略回测
            benchmark_results = self._run_benchmark_strategy_backtest(returns_data)
            
            # 验证指标
            validation_results = self._validate_core_metrics(filtered_results, benchmark_results)
            
            # 整合结果
            results = {
                'filtered_strategy': filtered_results,
                'benchmark_strategy': benchmark_results,
                'validation': validation_results,
                'data_info': {
                    'start_date': returns_data.index[0].strftime('%Y-%m-%d'),
                    'end_date': returns_data.index[-1].strftime('%Y-%m-%d'),
                    'trading_days': len(returns_data),
                    'stocks_count': len(returns_data.columns)
                }
            }
            
            logger.info("核心回测验证完成")
            
            # 验证测试结果
            self.assertIsInstance(results, dict)
            self.assertIn('filtered_strategy', results)
            self.assertIn('benchmark_strategy', results)
            self.assertIn('validation', results)
            
            # 验证策略结果结构
            filtered_strategy = results['filtered_strategy']
            self.assertIn('returns', filtered_strategy)
            self.assertIn('metrics', filtered_strategy)
            
            # 验证指标计算正确
            metrics = filtered_strategy['metrics']
            self.assertIsInstance(metrics['annual_return'], (int, float))
            self.assertIsInstance(metrics['annual_volatility'], (int, float))
            self.assertIsInstance(metrics['sharpe_ratio'], (int, float))
            
            return results
            
        except Exception as e:
            logger.error(f"核心回测验证失败: {str(e)}")
            self.fail(f"核心回测验证失败: {str(e)}")
    
    def _run_filtered_strategy_backtest(self, filter_instance: DynamicLowVolFilter, 
                                       returns_data: pd.DataFrame) -> Dict:
        """运行筛选策略回测"""
        logger.info("运行筛选策略回测...")
        
        portfolio_returns = []
        filtered_stocks_count = []
        
        # 滚动回测
        lookback_days = 60
        for i in range(lookback_days, len(returns_data)):
            current_date = returns_data.index[i]
            
            try:
                # 获取当日筛选结果
                filter_result = filter_instance.filter_stocks(current_date)
                selected_stocks = filter_result.get('selected_stocks', [])
                filtered_stocks_count.append(len(selected_stocks))
                
                if len(selected_stocks) > 0:
                    # 等权重组合收益
                    daily_portfolio_return = returns_data.loc[current_date, selected_stocks].mean()
                else:
                    # 无股票被选中时，收益为0
                    daily_portfolio_return = 0.0
                
                portfolio_returns.append(daily_portfolio_return)
                
            except Exception as e:
                logger.warning(f"日期 {current_date} 筛选失败: {str(e)}")
                portfolio_returns.append(0.0)
                filtered_stocks_count.append(0)
        
        # 计算策略指标
        portfolio_returns = np.array(portfolio_returns)
        metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(portfolio_returns)
        
        # 添加筛选统计
        metrics['avg_selected_stocks'] = np.mean(filtered_stocks_count)
        metrics['selection_consistency'] = np.std(filtered_stocks_count) / np.mean(filtered_stocks_count) if np.mean(filtered_stocks_count) > 0 else 0
        
        return {
            'returns': portfolio_returns,
            'metrics': metrics,
            'selected_stocks_count': filtered_stocks_count
        }
    
    def _run_benchmark_strategy_backtest(self, returns_data: pd.DataFrame) -> Dict:
        """运行基准策略回测（等权重所有股票）"""
        logger.info("运行基准策略回测...")
        
        # 等权重所有股票的收益
        benchmark_returns = returns_data.mean(axis=1).values
        
        # 只取与筛选策略相同的时间段
        lookback_days = 60
        benchmark_returns = benchmark_returns[lookback_days:]
        
        # 计算基准指标
        metrics = BacktestMetricsCalculator.calculate_portfolio_metrics(benchmark_returns)
        
        return {
            'returns': benchmark_returns,
            'metrics': metrics
        }
    
    def _validate_core_metrics(self, filtered_results: Dict, benchmark_results: Dict) -> Dict:
        """验证核心指标"""
        logger.info("验证核心指标...")
        
        filtered_metrics = filtered_results['metrics']
        benchmark_metrics = benchmark_results['metrics']
        
        validation_results = {
            'passed': True,
            'failed_metrics': [],
            'metric_comparisons': {},
            'improvement_analysis': {}
        }
        
        # 验证绝对指标要求
        absolute_checks = [
            ('annual_return', filtered_metrics['annual_return'], self.expected_metrics['annual_return_min'], '>='),
            ('annual_volatility', filtered_metrics['annual_volatility'], self.expected_metrics['annual_volatility_max'], '<='),
            ('max_drawdown', filtered_metrics['max_drawdown'], self.expected_metrics['max_drawdown_max'], '<='),
            ('sharpe_ratio', filtered_metrics['sharpe_ratio'], self.expected_metrics['sharpe_ratio_min'], '>=')
        ]
        
        for metric_name, actual_value, expected_value, operator in absolute_checks:
            passed = (
                (operator == '>=' and actual_value >= expected_value) or
                (operator == '<=' and actual_value <= expected_value)
            )
            
            validation_results['metric_comparisons'][metric_name] = {
                'actual': actual_value,
                'expected': expected_value,
                'operator': operator,
                'passed': passed
            }
            
            if not passed:
                validation_results['passed'] = False
                validation_results['failed_metrics'].append(metric_name)
        
        # 计算相对于基准的改进
        relative_improvements = {
            'return_improvement': filtered_metrics['annual_return'] - benchmark_metrics['annual_return'],
            'volatility_reduction': benchmark_metrics['annual_volatility'] - filtered_metrics['annual_volatility'],
            'drawdown_reduction': benchmark_metrics['max_drawdown'] - filtered_metrics['max_drawdown'],
            'sharpe_improvement': filtered_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
            'tail_loss_reduction': benchmark_metrics['tail_loss_frequency'] - filtered_metrics['tail_loss_frequency']
        }
        
        validation_results['improvement_analysis'] = relative_improvements
        
        # 验证尾部亏损降低要求
        tail_loss_reduction_rate = relative_improvements['tail_loss_reduction'] / benchmark_metrics['tail_loss_frequency'] if benchmark_metrics['tail_loss_frequency'] > 0 else 0
        tail_loss_passed = tail_loss_reduction_rate >= self.expected_metrics['tail_loss_reduction_min']
        
        validation_results['metric_comparisons']['tail_loss_reduction'] = {
            'actual': tail_loss_reduction_rate,
            'expected': self.expected_metrics['tail_loss_reduction_min'],
            'operator': '>=',
            'passed': tail_loss_passed
        }
        
        if not tail_loss_passed:
            validation_results['passed'] = False
            validation_results['failed_metrics'].append('tail_loss_reduction')
        
        return validation_results
    
    def generate_core_backtest_report(self, results: Dict) -> str:
        """生成核心回测报告"""
        filtered_metrics = results['filtered_strategy']['metrics']
        benchmark_metrics = results['benchmark_strategy']['metrics']
        validation = results['validation']
        data_info = results['data_info']
        
        report = f"""
# 动态低波筛选器核心回测验证报告

## 回测概览
- 回测期间: {data_info['start_date']} ~ {data_info['end_date']}
- 交易日数: {data_info['trading_days']}
- 股票数量: {data_info['stocks_count']}
- 平均选股数量: {filtered_metrics.get('avg_selected_stocks', 0):.1f}

## 策略表现

### 筛选策略指标
- 年化收益率: {filtered_metrics['annual_return']:.2%}
- 年化波动率: {filtered_metrics['annual_volatility']:.2%}
- 最大回撤: {filtered_metrics['max_drawdown']:.2%}
- 夏普比率: {filtered_metrics['sharpe_ratio']:.3f}
- 尾部亏损频次: {filtered_metrics['tail_loss_frequency']:.2%}

### 基准策略指标
- 年化收益率: {benchmark_metrics['annual_return']:.2%}
- 年化波动率: {benchmark_metrics['annual_volatility']:.2%}
- 最大回撤: {benchmark_metrics['max_drawdown']:.2%}
- 夏普比率: {benchmark_metrics['sharpe_ratio']:.3f}
- 尾部亏损频次: {benchmark_metrics['tail_loss_frequency']:.2%}

## 验证结果

### 总体验证: {'✅ 通过' if validation['passed'] else '❌ 失败'}

### 详细指标验证
"""
        
        for metric_name, comparison in validation['metric_comparisons'].items():
            status = '✅' if comparison['passed'] else '❌'
            report += f"- {metric_name}: {status} {comparison['actual']:.3f} {comparison['operator']} {comparison['expected']:.3f}\n"
        
        if validation['failed_metrics']:
            report += f"\n### 未通过指标: {', '.join(validation['failed_metrics'])}\n"
        
        # 改进分析
        improvements = validation['improvement_analysis']
        report += f"""
## 相对基准改进分析
- 收益提升: {improvements['return_improvement']:.2%}
- 波动降低: {improvements['volatility_reduction']:.2%}
- 回撤降低: {improvements['drawdown_reduction']:.2%}
- 夏普改进: {improvements['sharpe_improvement']:.3f}
- 尾部亏损降低: {improvements['tail_loss_reduction']:.2%} ({improvements['tail_loss_reduction']/benchmark_metrics['tail_loss_frequency']:.1%})
"""
        
        return report


if __name__ == '__main__':
    unittest.main()