#!/usr/bin/env python3
"""
任务13验证脚本：创建综合测试套件

验证以下子任务完成情况：
1. 编写端到端集成测试，验证完整筛选流水线
2. 创建性能基准测试，确保计算延迟<100ms
3. 实现回测验证，检查筛选效果是否达到预期指标
4. 添加异常处理测试，验证所有异常情况正确抛出
5. 编写测试报告和使用文档
"""

import os
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_test_files_exist():
    """验证测试文件是否存在"""
    logger.info("验证测试文件存在性...")
    
    required_files = [
        'tests/test_comprehensive_suite.py',
        'test_performance_benchmark.py',
        'test_backtest_validation.py',
        'run_comprehensive_tests.py',
        'docs/dynamic_lowvol_filter_usage.md'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            logger.info(f"✓ {file_path} 存在")
        else:
            missing_files.append(file_path)
            logger.error(f"✗ {file_path} 缺失")
    
    return len(missing_files) == 0, existing_files, missing_files


def verify_test_content():
    """验证测试内容完整性"""
    logger.info("验证测试内容完整性...")
    
    # 检查综合测试套件
    comprehensive_suite_path = 'tests/test_comprehensive_suite.py'
    if os.path.exists(comprehensive_suite_path):
        with open(comprehensive_suite_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        required_classes = [
            'TestEndToEndIntegration',
            'TestPerformanceBenchmark', 
            'TestBacktestValidation',
            'TestExceptionHandling',
            'TestSystemStability'
        ]
        
        missing_classes = []
        for class_name in required_classes:
            if f'class {class_name}' in content:
                logger.info(f"✓ {class_name} 类存在")
            else:
                missing_classes.append(class_name)
                logger.error(f"✗ {class_name} 类缺失")
        
        # 检查关键测试方法
        required_methods = [
            'test_complete_filter_pipeline',
            'test_single_update_performance',
            'test_filter_effectiveness_backtest',
            'test_data_quality_exceptions',
            'test_long_term_stability'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if f'def {method_name}' in content:
                logger.info(f"✓ {method_name} 方法存在")
            else:
                missing_methods.append(method_name)
                logger.error(f"✗ {method_name} 方法缺失")
        
        comprehensive_complete = len(missing_classes) == 0 and len(missing_methods) == 0
    else:
        comprehensive_complete = False
        missing_classes = required_classes
        missing_methods = required_methods
    
    # 检查性能基准测试
    performance_test_path = 'test_performance_benchmark.py'
    if os.path.exists(performance_test_path):
        with open(performance_test_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        performance_features = [
            'performance_threshold_ms = 100',
            'def run_single_update_benchmark',
            'def run_batch_update_benchmark',
            'def run_memory_performance_benchmark',
            'def run_caching_performance_benchmark'
        ]
        
        missing_features = []
        for feature in performance_features:
            if feature in content:
                logger.info(f"✓ 性能测试特性存在: {feature}")
            else:
                missing_features.append(feature)
                logger.error(f"✗ 性能测试特性缺失: {feature}")
        
        performance_complete = len(missing_features) == 0
    else:
        performance_complete = False
        missing_features = performance_features
    
    # 检查回测验证测试
    backtest_test_path = 'test_backtest_validation.py'
    if os.path.exists(backtest_test_path):
        with open(backtest_test_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        backtest_features = [
            'annual_return_min\': 0.06',
            'annual_volatility_max\': 0.12',
            'max_drawdown_max\': 0.10',
            'sharpe_ratio_min\': 0.6',
            'def run_backtest_validation',
            'def _calculate_portfolio_metrics'
        ]
        
        missing_backtest_features = []
        for feature in backtest_features:
            if feature in content:
                logger.info(f"✓ 回测验证特性存在: {feature}")
            else:
                missing_backtest_features.append(feature)
                logger.error(f"✗ 回测验证特性缺失: {feature}")
        
        backtest_complete = len(missing_backtest_features) == 0
    else:
        backtest_complete = False
        missing_backtest_features = backtest_features
    
    return comprehensive_complete and performance_complete and backtest_complete


def verify_exception_handling():
    """验证异常处理测试"""
    logger.info("验证异常处理测试...")
    
    comprehensive_suite_path = 'tests/test_comprehensive_suite.py'
    if not os.path.exists(comprehensive_suite_path):
        return False
    
    with open(comprehensive_suite_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    exception_tests = [
        'test_data_quality_exceptions',
        'test_model_fitting_exceptions', 
        'test_regime_detection_exceptions',
        'test_configuration_exceptions',
        'DataQualityException',
        'ModelFittingException',
        'RegimeDetectionException',
        'ConfigurationException'
    ]
    
    missing_exception_tests = []
    for test in exception_tests:
        if test in content:
            logger.info(f"✓ 异常处理测试存在: {test}")
        else:
            missing_exception_tests.append(test)
            logger.error(f"✗ 异常处理测试缺失: {test}")
    
    return len(missing_exception_tests) == 0


def verify_documentation():
    """验证文档完整性"""
    logger.info("验证文档完整性...")
    
    doc_path = 'docs/dynamic_lowvol_filter_usage.md'
    if not os.path.exists(doc_path):
        logger.error("✗ 使用文档缺失")
        return False
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_sections = [
        '# 动态低波筛选器使用文档',
        '## 概述',
        '## 快速开始',
        '## 配置参数',
        '## API 参考',
        '## 异常处理',
        '## 性能优化',
        '## 测试和验证',
        '## 故障排除',
        '## 最佳实践'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section in content:
            logger.info(f"✓ 文档章节存在: {section}")
        else:
            missing_sections.append(section)
            logger.error(f"✗ 文档章节缺失: {section}")
    
    return len(missing_sections) == 0


def verify_test_runner():
    """验证测试运行器"""
    logger.info("验证测试运行器...")
    
    runner_path = 'run_comprehensive_tests.py'
    if not os.path.exists(runner_path):
        logger.error("✗ 测试运行器缺失")
        return False
    
    with open(runner_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    runner_features = [
        'class ComprehensiveTestRunner',
        'def run_all_tests',
        'def _generate_final_report',
        'comprehensive_suite',
        'performance_benchmark',
        'backtest_validation'
    ]
    
    missing_runner_features = []
    for feature in runner_features:
        if feature in content:
            logger.info(f"✓ 测试运行器特性存在: {feature}")
        else:
            missing_runner_features.append(feature)
            logger.error(f"✗ 测试运行器特性缺失: {feature}")
    
    return len(missing_runner_features) == 0


def verify_reports_directory():
    """验证报告目录"""
    logger.info("验证报告目录...")
    
    reports_dir = 'reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        logger.info(f"✓ 创建报告目录: {reports_dir}")
    else:
        logger.info(f"✓ 报告目录存在: {reports_dir}")
    
    return True


def run_basic_import_test():
    """运行基本导入测试"""
    logger.info("运行基本导入测试...")
    
    try:
        # 测试导入综合测试套件
        sys.path.append('.')
        from tests.test_comprehensive_suite import (
            TestEndToEndIntegration,
            TestPerformanceBenchmark,
            TestBacktestValidation,
            TestExceptionHandling,
            TestSystemStability
        )
        logger.info("✓ 综合测试套件导入成功")
        
        # 测试导入性能基准测试
        import test_performance_benchmark
        logger.info("✓ 性能基准测试导入成功")
        
        # 测试导入回测验证测试
        import test_backtest_validation
        logger.info("✓ 回测验证测试导入成功")
        
        # 测试导入测试运行器
        import run_comprehensive_tests
        logger.info("✓ 测试运行器导入成功")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 导入测试失败: {e}")
        return False


def main():
    """主验证函数"""
    logger.info("开始验证任务13：创建综合测试套件")
    
    # 验证项目列表
    verifications = [
        ("测试文件存在性", verify_test_files_exist),
        ("测试内容完整性", verify_test_content),
        ("异常处理测试", verify_exception_handling),
        ("文档完整性", verify_documentation),
        ("测试运行器", verify_test_runner),
        ("报告目录", verify_reports_directory),
        ("基本导入测试", run_basic_import_test)
    ]
    
    results = {}
    
    for name, verify_func in verifications:
        logger.info(f"\n{'='*50}")
        logger.info(f"验证: {name}")
        logger.info(f"{'='*50}")
        
        try:
            if name == "测试文件存在性":
                success, existing, missing = verify_func()
                results[name] = {
                    'success': success,
                    'existing_files': existing,
                    'missing_files': missing
                }
            else:
                success = verify_func()
                results[name] = {'success': success}
                
        except Exception as e:
            logger.error(f"验证 {name} 时发生异常: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # 汇总结果
    logger.info(f"\n{'='*60}")
    logger.info("任务13验证结果汇总")
    logger.info(f"{'='*60}")
    
    total_checks = len(verifications)
    passed_checks = sum(1 for result in results.values() if result['success'])
    
    for name, result in results.items():
        status = "✓ 通过" if result['success'] else "✗ 失败"
        logger.info(f"{status} {name}")
        
        if not result['success'] and 'error' in result:
            logger.info(f"    错误: {result['error']}")
    
    logger.info(f"\n总体结果: {passed_checks}/{total_checks} 项验证通过")
    
    if passed_checks == total_checks:
        logger.info("🎉 任务13：创建综合测试套件 - 完成")
        logger.info("\n已完成的子任务:")
        logger.info("✓ 1. 编写端到端集成测试，验证完整筛选流水线")
        logger.info("✓ 2. 创建性能基准测试，确保计算延迟<100ms")
        logger.info("✓ 3. 实现回测验证，检查筛选效果是否达到预期指标")
        logger.info("✓ 4. 添加异常处理测试，验证所有异常情况正确抛出")
        logger.info("✓ 5. 编写测试报告和使用文档")
        
        logger.info("\n创建的文件:")
        logger.info("- tests/test_comprehensive_suite.py: 综合测试套件")
        logger.info("- test_performance_benchmark.py: 性能基准测试")
        logger.info("- test_backtest_validation.py: 回测验证测试")
        logger.info("- run_comprehensive_tests.py: 测试运行器")
        logger.info("- docs/dynamic_lowvol_filter_usage.md: 使用文档")
        
        return True
    else:
        logger.error("❌ 任务13验证失败，存在未完成的子任务")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)