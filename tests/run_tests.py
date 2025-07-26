"""
测试运行器 - 运行所有测试并生成报告
"""

import unittest
import sys
import os
import time
from io import StringIO

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def discover_tests():
    """发现所有测试"""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    return suite


def run_tests_with_report():
    """运行测试并生成报告"""
    print("=== A股强化学习量化交易系统测试套件 ===\n")
    
    # 发现测试
    suite = discover_tests()
    
    # 计算测试数量
    test_count = suite.countTestCases()
    print(f"发现 {test_count} 个测试用例\n")
    
    # 创建测试运行器
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True
    )
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行测试
    result = runner.run(suite)
    
    # 记录结束时间
    end_time = time.time()
    duration = end_time - start_time
    
    # 获取详细输出
    test_output = stream.getvalue()
    
    # 打印结果摘要
    print("=== 测试结果摘要 ===")
    print(f"总测试数量: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"执行时间: {duration:.2f} 秒")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # 打印失败详情
    if result.failures:
        print("\n=== 失败测试详情 ===")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print(f"   {traceback.split('AssertionError:')[-1].strip()}")
    
    # 打印错误详情  
    if result.errors:
        print("\n=== 错误测试详情 ===")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            # 提取错误类型
            error_lines = traceback.strip().split('\n')
            error_msg = error_lines[-1] if error_lines else "未知错误"
            print(f"   {error_msg}")
    
    # 保存详细测试报告
    save_detailed_report(test_output, result, duration)
    
    return result.wasSuccessful()


def save_detailed_report(test_output, result, duration):
    """保存详细测试报告"""
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(report_dir, f'test_report_{timestamp}.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("A股强化学习量化交易系统 - 测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"执行时长: {duration:.2f} 秒\n")
        f.write(f"总测试数: {result.testsRun}\n")
        f.write(f"成功数: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"失败数: {len(result.failures)}\n")
        f.write(f"错误数: {len(result.errors)}\n")
        f.write(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%\n\n")
        
        f.write("详细测试输出:\n")
        f.write("-" * 30 + "\n")
        f.write(test_output)
    
    print(f"\n详细测试报告已保存至: {report_file}")


def run_specific_test(test_name):
    """运行特定测试"""
    print(f"运行特定测试: {test_name}")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_test_coverage():
    """运行测试覆盖率分析"""
    try:
        import coverage
        
        print("开始测试覆盖率分析...")
        
        # 创建覆盖率对象
        cov = coverage.Coverage(source=['.'])
        cov.start()
        
        # 运行测试
        suite = discover_tests()
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        # 停止覆盖率收集
        cov.stop()
        cov.save()
        
        # 生成报告
        print("\n=== 测试覆盖率报告 ===")
        cov.report()
        
        # 保存HTML报告
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reports', 'coverage')
        os.makedirs(report_dir, exist_ok=True)
        cov.html_report(directory=report_dir)
        
        print(f"\nHTML覆盖率报告保存至: {report_dir}")
        
        return result.wasSuccessful()
        
    except ImportError:
        print("coverage包未安装，跳过覆盖率分析")
        print("安装命令: pip install coverage")
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'coverage':
            success = run_test_coverage()
        else:
            # 运行特定测试
            success = run_specific_test(sys.argv[1])
    else:
        # 运行所有测试
        success = run_tests_with_report()
    
    # 退出码
    sys.exit(0 if success else 1)