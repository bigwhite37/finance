#!/usr/bin/env python3
"""
动态低波筛选器性能基准测试

专门用于验证计算延迟<100ms的性能要求
"""

import sys
import os
import time
import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from unittest.mock import Mock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        self.performance_threshold_ms = 100  # 100ms性能要求
        self.config = self._create_benchmark_config()
        self.data_manager = self._create_benchmark_data_manager()
        
    def _create_benchmark_config(self) -> Dict:
        """创建基准测试配置"""
        return {
            'rolling_windows': [20, 60],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            'garch_window': 250,
            'forecast_horizon': 5,
            'enable_ml_predictor': False,
            'ivol_bad_threshold': 0.3,
            'ivol_good_threshold': 0.6,
            'regime_detection_window': 60,
            'regime_model_type': "HMM",
            'enable_caching': True,
            'cache_expiry_days': 1,
            'parallel_processing': False  # 单线程测试
        }
    
    def _create_benchmark_data_manager(self):
        """创建基准测试数据管理器"""
        mock_manager = Mock()
        
        # 生成真实规模的测试数据
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stocks = [f'stock_{i:04d}' for i in range(1000)]  # 1000只股票
        
        np.random.seed(42)
        
        # 价格数据 - 使用几何布朗运动
        returns = np.random.normal(0.0005, 0.02, (len(dates), len(stocks)))
        prices = pd.DataFrame(
            np.exp(returns.cumsum(axis=0)) * 100,
            index=dates, columns=stocks
        )
        
        # 成交量数据
        volumes = pd.DataFrame(
            np.random.lognormal(10, 1, (len(dates), len(stocks))),
            index=dates, columns=stocks
        )
        
        # 因子数据
        factors = ['market', 'size', 'value', 'profitability', 'investment']
        factor_data = pd.DataFrame(
            np.random.normal(0, 0.01, (len(dates), len(factors))),
            index=dates, columns=factors
        )
        
        # 市场数据
        market_data = pd.DataFrame({
            'returns': np.random.normal(0, 0.015, len(dates)),
            'volatility': np.random.exponential(0.2, len(dates))
        }, index=dates)
        
        mock_manager.get_price_data.return_value = prices
        mock_manager.get_volume_data.return_value = volumes
        mock_manager.get_factor_data.return_value = factor_data
        mock_manager.get_market_data.return_value = market_data
        
        return mock_manager
    
    def run_single_update_benchmark(self) -> Dict:
        """运行单次更新性能基准测试"""
        logger.info("开始单次更新性能基准测试...")
        
        filter_instance = DynamicLowVolFilter(self.config, self.data_manager)
        test_date = pd.Timestamp('2023-06-01')
        
        # 预热运行（避免初始化开销影响测试）
        logger.info("执行预热运行...")
        for _ in range(3):
            filter_instance.update_tradable_mask(test_date)
        
        # 性能测试 - 多次运行取平均
        logger.info("开始正式性能测试...")
        execution_times = []
        
        for i in range(10):  # 运行10次
            start_time = time.perf_counter()
            mask = filter_instance.update_tradable_mask(test_date)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            execution_times.append(execution_time_ms)
            
            logger.info(f"第{i+1}次运行: {execution_time_ms:.2f}ms")
            
            # 验证结果有效性
            assert isinstance(mask, np.ndarray)
            assert len(mask) == 1000
            assert np.all((mask == 0) | (mask == 1))
        
        # 计算统计指标
        avg_time = np.mean(execution_times)
        min_time = np.min(execution_times)
        max_time = np.max(execution_times)
        std_time = np.std(execution_times)
        
        results = {
            'test_type': '单次更新性能',
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'std_time_ms': std_time,
            'threshold_ms': self.performance_threshold_ms,
            'passed': avg_time < self.performance_threshold_ms,
            'stock_count': 1000,
            'execution_times': execution_times
        }
        
        logger.info(f"单次更新性能测试完成:")
        logger.info(f"  平均耗时: {avg_time:.2f}ms")
        logger.info(f"  最小耗时: {min_time:.2f}ms")
        logger.info(f"  最大耗时: {max_time:.2f}ms")
        logger.info(f"  标准差: {std_time:.2f}ms")
        logger.info(f"  性能要求: <{self.performance_threshold_ms}ms")
        logger.info(f"  测试结果: {'通过' if results['passed'] else '失败'}")
        
        return results
    
    def run_batch_update_benchmark(self) -> Dict:
        """运行批量更新性能基准测试"""
        logger.info("开始批量更新性能基准测试...")
        
        filter_instance = DynamicLowVolFilter(self.config, self.data_manager)
        test_dates = pd.date_range('2023-06-01', '2023-06-10', freq='D')  # 10个交易日
        
        # 预热
        filter_instance.update_tradable_mask(test_dates[0])
        
        # 批量性能测试
        individual_times = []
        
        start_time = time.perf_counter()
        
        for i, date in enumerate(test_dates):
            iter_start = time.perf_counter()
            mask = filter_instance.update_tradable_mask(date)
            iter_end = time.perf_counter()
            
            iter_time_ms = (iter_end - iter_start) * 1000
            individual_times.append(iter_time_ms)
            
            logger.info(f"日期 {date.strftime('%Y-%m-%d')}: {iter_time_ms:.2f}ms")
            
            # 验证结果
            assert isinstance(mask, np.ndarray)
            assert len(mask) == 1000
        
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_update = total_time_ms / len(test_dates)
        
        results = {
            'test_type': '批量更新性能',
            'total_time_ms': total_time_ms,
            'avg_time_per_update_ms': avg_time_per_update,
            'min_time_ms': np.min(individual_times),
            'max_time_ms': np.max(individual_times),
            'std_time_ms': np.std(individual_times),
            'threshold_ms': self.performance_threshold_ms,
            'passed': avg_time_per_update < self.performance_threshold_ms,
            'update_count': len(test_dates),
            'individual_times': individual_times
        }
        
        logger.info(f"批量更新性能测试完成:")
        logger.info(f"  总耗时: {total_time_ms:.2f}ms")
        logger.info(f"  平均每次更新: {avg_time_per_update:.2f}ms")
        logger.info(f"  最快更新: {results['min_time_ms']:.2f}ms")
        logger.info(f"  最慢更新: {results['max_time_ms']:.2f}ms")
        logger.info(f"  性能要求: <{self.performance_threshold_ms}ms")
        logger.info(f"  测试结果: {'通过' if results['passed'] else '失败'}")
        
        return results
    
    def run_memory_performance_benchmark(self) -> Dict:
        """运行内存性能基准测试"""
        logger.info("开始内存性能基准测试...")
        
        import psutil
        import gc
        
        process = psutil.Process()
        
        # 获取初始内存
        gc.collect()
        initial_memory_mb = process.memory_info().rss / 1024 / 1024
        
        filter_instance = DynamicLowVolFilter(self.config, self.data_manager)
        
        # 执行多次更新测试内存使用
        test_dates = pd.date_range('2023-06-01', '2023-06-30', freq='D')  # 30天
        memory_snapshots = []
        
        for i, date in enumerate(test_dates):
            mask = filter_instance.update_tradable_mask(date)
            
            # 每5次更新记录内存
            if i % 5 == 0:
                gc.collect()
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory_mb - initial_memory_mb
                memory_snapshots.append({
                    'update_count': i + 1,
                    'memory_mb': current_memory_mb,
                    'memory_increase_mb': memory_increase
                })
                
                logger.info(f"更新{i+1}次后内存: {current_memory_mb:.1f}MB (+{memory_increase:.1f}MB)")
        
        # 最终内存检查
        gc.collect()
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory_mb - initial_memory_mb
        
        results = {
            'test_type': '内存性能',
            'initial_memory_mb': initial_memory_mb,
            'final_memory_mb': final_memory_mb,
            'total_memory_increase_mb': total_memory_increase,
            'max_memory_increase_mb': max([s['memory_increase_mb'] for s in memory_snapshots]),
            'memory_threshold_mb': 500,  # 500MB阈值
            'passed': total_memory_increase < 500,
            'update_count': len(test_dates),
            'memory_snapshots': memory_snapshots
        }
        
        logger.info(f"内存性能测试完成:")
        logger.info(f"  初始内存: {initial_memory_mb:.1f}MB")
        logger.info(f"  最终内存: {final_memory_mb:.1f}MB")
        logger.info(f"  总内存增长: {total_memory_increase:.1f}MB")
        logger.info(f"  最大内存增长: {results['max_memory_increase_mb']:.1f}MB")
        logger.info(f"  内存阈值: <500MB")
        logger.info(f"  测试结果: {'通过' if results['passed'] else '失败'}")
        
        return results
    
    def run_caching_performance_benchmark(self) -> Dict:
        """运行缓存性能基准测试"""
        logger.info("开始缓存性能基准测试...")
        
        test_date = pd.Timestamp('2023-06-01')
        
        # 测试无缓存性能
        no_cache_config = self.config.copy()
        no_cache_config['enable_caching'] = False
        
        filter_no_cache = DynamicLowVolFilter(no_cache_config, self.data_manager)
        
        # 无缓存 - 连续调用
        no_cache_times = []
        for i in range(5):
            start_time = time.perf_counter()
            mask = filter_no_cache.update_tradable_mask(test_date)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            no_cache_times.append(execution_time_ms)
            logger.info(f"无缓存第{i+1}次: {execution_time_ms:.2f}ms")
        
        # 测试有缓存性能
        cache_config = self.config.copy()
        cache_config['enable_caching'] = True
        
        filter_with_cache = DynamicLowVolFilter(cache_config, self.data_manager)
        
        # 有缓存 - 连续调用
        cache_times = []
        for i in range(5):
            start_time = time.perf_counter()
            mask = filter_with_cache.update_tradable_mask(test_date)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            cache_times.append(execution_time_ms)
            logger.info(f"有缓存第{i+1}次: {execution_time_ms:.2f}ms")
        
        # 计算性能提升
        avg_no_cache_time = np.mean(no_cache_times[1:])  # 排除第一次（初始化开销）
        avg_cache_time = np.mean(cache_times[1:])        # 排除第一次
        
        improvement_ratio = avg_no_cache_time / avg_cache_time if avg_cache_time > 0 else 1
        
        results = {
            'test_type': '缓存性能',
            'avg_no_cache_time_ms': avg_no_cache_time,
            'avg_cache_time_ms': avg_cache_time,
            'improvement_ratio': improvement_ratio,
            'min_improvement_ratio': 1.2,  # 至少20%提升
            'passed': improvement_ratio >= 1.2,
            'no_cache_times': no_cache_times,
            'cache_times': cache_times
        }
        
        logger.info(f"缓存性能测试完成:")
        logger.info(f"  无缓存平均耗时: {avg_no_cache_time:.2f}ms")
        logger.info(f"  有缓存平均耗时: {avg_cache_time:.2f}ms")
        logger.info(f"  性能提升比例: {improvement_ratio:.2f}x")
        logger.info(f"  要求提升比例: ≥1.2x")
        logger.info(f"  测试结果: {'通过' if results['passed'] else '失败'}")
        
        return results
    
    def run_scalability_benchmark(self) -> Dict:
        """运行可扩展性基准测试"""
        logger.info("开始可扩展性基准测试...")
        
        test_date = pd.Timestamp('2023-06-01')
        stock_counts = [100, 300, 500, 1000, 1500]  # 不同股票数量
        
        scalability_results = []
        
        for stock_count in stock_counts:
            logger.info(f"测试{stock_count}只股票的性能...")
            
            # 创建对应规模的数据管理器
            scaled_data_manager = self._create_scaled_data_manager(stock_count)
            filter_instance = DynamicLowVolFilter(self.config, scaled_data_manager)
            
            # 预热
            filter_instance.update_tradable_mask(test_date)
            
            # 性能测试
            execution_times = []
            for _ in range(5):
                start_time = time.perf_counter()
                mask = filter_instance.update_tradable_mask(test_date)
                end_time = time.perf_counter()
                
                execution_time_ms = (end_time - start_time) * 1000
                execution_times.append(execution_time_ms)
            
            avg_time = np.mean(execution_times)
            
            scalability_results.append({
                'stock_count': stock_count,
                'avg_time_ms': avg_time,
                'time_per_stock_ms': avg_time / stock_count,
                'passed': avg_time < self.performance_threshold_ms
            })
            
            logger.info(f"  {stock_count}只股票: {avg_time:.2f}ms "
                       f"(每只股票{avg_time/stock_count:.4f}ms)")
        
        # 分析可扩展性
        times = [r['avg_time_ms'] for r in scalability_results]
        counts = [r['stock_count'] for r in scalability_results]
        
        # 计算时间复杂度（线性拟合）
        coeffs = np.polyfit(counts, times, 1)
        slope = coeffs[0]  # 每增加一只股票的时间增长
        
        results = {
            'test_type': '可扩展性',
            'scalability_results': scalability_results,
            'time_per_stock_slope': slope,
            'max_stock_count_under_threshold': max([r['stock_count'] for r in scalability_results if r['passed']]),
            'linear_complexity': slope < 0.1,  # 每只股票增长<0.1ms认为是线性
            'passed': all(r['passed'] for r in scalability_results)
        }
        
        logger.info(f"可扩展性测试完成:")
        logger.info(f"  时间复杂度斜率: {slope:.4f}ms/股票")
        logger.info(f"  最大支持股票数: {results['max_stock_count_under_threshold']}")
        logger.info(f"  线性复杂度: {'是' if results['linear_complexity'] else '否'}")
        logger.info(f"  测试结果: {'通过' if results['passed'] else '失败'}")
        
        return results
    
    def _create_scaled_data_manager(self, stock_count: int):
        """创建指定规模的数据管理器"""
        mock_manager = Mock()
        
        dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
        stocks = [f'stock_{i:04d}' for i in range(stock_count)]
        
        np.random.seed(42)
        
        # 价格数据
        returns = np.random.normal(0.0005, 0.02, (len(dates), len(stocks)))
        prices = pd.DataFrame(
            np.exp(returns.cumsum(axis=0)) * 100,
            index=dates, columns=stocks
        )
        
        # 其他数据
        volumes = pd.DataFrame(
            np.random.lognormal(10, 1, (len(dates), len(stocks))),
            index=dates, columns=stocks
        )
        
        factors = ['market', 'size', 'value', 'profitability', 'investment']
        factor_data = pd.DataFrame(
            np.random.normal(0, 0.01, (len(dates), len(factors))),
            index=dates, columns=factors
        )
        
        market_data = pd.DataFrame({
            'returns': np.random.normal(0, 0.015, len(dates)),
            'volatility': np.random.exponential(0.2, len(dates))
        }, index=dates)
        
        mock_manager.get_price_data.return_value = prices
        mock_manager.get_volume_data.return_value = volumes
        mock_manager.get_factor_data.return_value = factor_data
        mock_manager.get_market_data.return_value = market_data
        
        return mock_manager
    
    def run_all_benchmarks(self) -> Dict:
        """运行所有性能基准测试"""
        logger.info("开始运行所有性能基准测试...")
        
        all_results = {}
        
        try:
            # 1. 单次更新性能
            all_results['single_update'] = self.run_single_update_benchmark()
        except Exception as e:
            logger.error(f"单次更新性能测试失败: {e}")
            all_results['single_update'] = {'passed': False, 'error': str(e)}
        
        try:
            # 2. 批量更新性能
            all_results['batch_update'] = self.run_batch_update_benchmark()
        except Exception as e:
            logger.error(f"批量更新性能测试失败: {e}")
            all_results['batch_update'] = {'passed': False, 'error': str(e)}
        
        try:
            # 3. 内存性能
            all_results['memory_performance'] = self.run_memory_performance_benchmark()
        except Exception as e:
            logger.error(f"内存性能测试失败: {e}")
            all_results['memory_performance'] = {'passed': False, 'error': str(e)}
        
        try:
            # 4. 缓存性能
            all_results['caching_performance'] = self.run_caching_performance_benchmark()
        except Exception as e:
            logger.error(f"缓存性能测试失败: {e}")
            all_results['caching_performance'] = {'passed': False, 'error': str(e)}
        
        try:
            # 5. 可扩展性
            all_results['scalability'] = self.run_scalability_benchmark()
        except Exception as e:
            logger.error(f"可扩展性测试失败: {e}")
            all_results['scalability'] = {'passed': False, 'error': str(e)}
        
        # 汇总结果
        passed_tests = sum(1 for result in all_results.values() if result.get('passed', False))
        total_tests = len(all_results)
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests,
            'overall_passed': passed_tests == total_tests,
            'detailed_results': all_results
        }
        
        logger.info(f"性能基准测试完成:")
        logger.info(f"  总测试数: {total_tests}")
        logger.info(f"  通过测试: {passed_tests}")
        logger.info(f"  失败测试: {total_tests - passed_tests}")
        logger.info(f"  成功率: {summary['success_rate']:.1%}")
        logger.info(f"  整体结果: {'通过' if summary['overall_passed'] else '失败'}")
        
        return summary
    
    def generate_performance_report(self, results: Dict) -> str:
        """生成性能测试报告"""
        from datetime import datetime
        
        report_content = f"""# 动态低波筛选器性能基准测试报告

## 测试概要
- 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 性能要求: 单次更新延迟 < 100ms
- 总测试数: {results['total_tests']}
- 通过测试: {results['passed_tests']}
- 失败测试: {results['failed_tests']}
- 成功率: {results['success_rate']:.1%}
- 整体结果: {'通过' if results['overall_passed'] else '失败'}

## 详细测试结果

"""
        
        # 添加各项测试结果
        for test_name, test_result in results['detailed_results'].items():
            if 'error' in test_result:
                report_content += f"""### {test_result.get('test_type', test_name)}
- 状态: 失败
- 错误: {test_result['error']}

"""
            else:
                if test_name == 'single_update':
                    report_content += f"""### {test_result['test_type']}
- 状态: {'通过' if test_result['passed'] else '失败'}
- 平均耗时: {test_result['avg_time_ms']:.2f}ms
- 最小耗时: {test_result['min_time_ms']:.2f}ms
- 最大耗时: {test_result['max_time_ms']:.2f}ms
- 标准差: {test_result['std_time_ms']:.2f}ms
- 股票数量: {test_result['stock_count']}
- 性能要求: < {test_result['threshold_ms']}ms

"""
                elif test_name == 'batch_update':
                    report_content += f"""### {test_result['test_type']}
- 状态: {'通过' if test_result['passed'] else '失败'}
- 总耗时: {test_result['total_time_ms']:.2f}ms
- 平均每次更新: {test_result['avg_time_per_update_ms']:.2f}ms
- 最快更新: {test_result['min_time_ms']:.2f}ms
- 最慢更新: {test_result['max_time_ms']:.2f}ms
- 更新次数: {test_result['update_count']}
- 性能要求: < {test_result['threshold_ms']}ms

"""
                elif test_name == 'memory_performance':
                    report_content += f"""### {test_result['test_type']}
- 状态: {'通过' if test_result['passed'] else '失败'}
- 初始内存: {test_result['initial_memory_mb']:.1f}MB
- 最终内存: {test_result['final_memory_mb']:.1f}MB
- 总内存增长: {test_result['total_memory_increase_mb']:.1f}MB
- 最大内存增长: {test_result['max_memory_increase_mb']:.1f}MB
- 更新次数: {test_result['update_count']}
- 内存要求: < {test_result['memory_threshold_mb']}MB

"""
                elif test_name == 'caching_performance':
                    report_content += f"""### {test_result['test_type']}
- 状态: {'通过' if test_result['passed'] else '失败'}
- 无缓存平均耗时: {test_result['avg_no_cache_time_ms']:.2f}ms
- 有缓存平均耗时: {test_result['avg_cache_time_ms']:.2f}ms
- 性能提升比例: {test_result['improvement_ratio']:.2f}x
- 要求提升比例: ≥ {test_result['min_improvement_ratio']}x

"""
                elif test_name == 'scalability':
                    report_content += f"""### {test_result['test_type']}
- 状态: {'通过' if test_result['passed'] else '失败'}
- 时间复杂度斜率: {test_result['time_per_stock_slope']:.4f}ms/股票
- 最大支持股票数: {test_result['max_stock_count_under_threshold']}
- 线性复杂度: {'是' if test_result['linear_complexity'] else '否'}

#### 不同规模性能表现:
"""
                    for scale_result in test_result['scalability_results']:
                        report_content += f"- {scale_result['stock_count']}只股票: {scale_result['avg_time_ms']:.2f}ms ({'通过' if scale_result['passed'] else '失败'})\\n"
                    
                    report_content += "\\n"
        
        report_content += f"""## 结论

{'所有性能测试通过，系统满足<100ms延迟要求。' if results['overall_passed'] else '存在性能测试失败，需要优化系统性能。'}

## 建议

"""
        
        if not results['overall_passed']:
            report_content += """- 检查失败的测试项目，针对性优化
- 考虑启用并行处理提升性能
- 优化GARCH模型拟合算法
- 增强缓存机制
- 考虑使用更高效的数据结构

"""
        else:
            report_content += """- 系统性能表现良好，满足设计要求
- 可以考虑进一步优化以支持更大规模数据
- 建议定期运行性能基准测试监控性能变化

"""
        
        return report_content


def main():
    """主函数"""
    logger.info("开始动态低波筛选器性能基准测试...")
    
    # 创建性能基准测试实例
    benchmark = PerformanceBenchmark()
    
    # 运行所有基准测试
    results = benchmark.run_all_benchmarks()
    
    # 生成测试报告
    report_content = benchmark.generate_performance_report(results)
    
    # 保存报告
    report_path = 'reports/performance_benchmark_report.md'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"性能基准测试报告已保存到: {report_path}")
    
    # 返回测试结果
    if results['overall_passed']:
        logger.info("所有性能基准测试通过！")
        return 0
    else:
        logger.error("存在性能基准测试失败！")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)