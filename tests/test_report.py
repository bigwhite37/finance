#!/usr/bin/env python3
"""
生成详细的测试报告
"""

import subprocess
import sys
from pathlib import Path

def main():
    """生成测试报告"""
    
    print("🧪 开始全量测试执行和报告生成")
    print("="*80)
    
    # 定义测试套件
    test_suites = {
        '核心功能测试': [
            'tests/dynamic_lowvol_filter/test_exceptions.py',
            'tests/dynamic_lowvol_filter/test_backtest.py', 
            'tests/dynamic_lowvol_filter/test_integration.py'
        ],
        '相关单元测试': [
            'tests/unit/test_ivol_constraint_filter.py',
            'tests/unit/test_regime_aware_threshold_adjuster.py',
            'tests/unit/test_data_preprocessor.py',
            'tests/unit/test_dynamic_lowvol_config.py',
            'tests/unit/test_rolling_percentile_filter.py'
        ]
    }
    
    overall_results = {}
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for suite_name, test_files in test_suites.items():
        print(f"\n📋 运行 {suite_name}")
        print("-" * 50)
        
        # 运行测试
        cmd = ['python', '-m', 'pytest'] + test_files + ['--tb=short', '-v']
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 解析结果
        output_lines = result.stdout.split('\n')
        suite_tests = 0
        suite_passed = 0
        suite_failed = 0
        
        for line in output_lines:
            if 'PASSED' in line:
                suite_passed += 1
                suite_tests += 1
            elif 'FAILED' in line:
                suite_failed += 1
                suite_tests += 1
        
        # 从最后的统计行获取更准确的数字
        for line in reversed(output_lines):
            if 'passed' in line and 'in' in line:
                import re
                passed_match = re.search(r'(\d+) passed', line)
                if passed_match:
                    suite_passed = int(passed_match.group(1))
                    suite_tests = suite_passed + suite_failed
                break
        
        overall_results[suite_name] = {
            'tests': suite_tests,
            'passed': suite_passed,
            'failed': suite_failed,
            'status': 'PASSED' if suite_failed == 0 else 'FAILED',
            'returncode': result.returncode
        }
        
        total_tests += suite_tests
        total_passed += suite_passed
        total_failed += suite_failed
        
        # 输出结果
        status_symbol = "✅" if suite_failed == 0 else "❌"
        print(f"{status_symbol} {suite_name}: {suite_passed}/{suite_tests} 通过")
        
        if suite_failed > 0:
            print(f"   失败的测试: {suite_failed} 个")
    
    # 生成总结
    print(f"\n{'='*80}")
    print("📊 测试执行总结")
    print(f"{'='*80}")
    
    overall_status = "PASSED" if total_failed == 0 else "FAILED"
    status_symbol = "✅" if total_failed == 0 else "❌"
    
    print(f"{status_symbol} 总体状态: {overall_status}")
    print(f"📈 总计: {total_passed}/{total_tests} 个测试通过")
    print(f"✅ 通过: {total_passed} 个")
    print(f"❌ 失败: {total_failed} 个")
    print(f"📊 成功率: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
    
    print(f"\n📋 详细结果:")
    for suite_name, result in overall_results.items():
        status_symbol = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"   {status_symbol} {suite_name}: {result['passed']}/{result['tests']} 通过")
    
    # 结论
    print(f"\n{'='*80}")
    if total_failed == 0:
        print("🎉 所有核心功能测试均通过！系统状态良好。")
    else:
        print(f"⚠️  发现 {total_failed} 个失败的测试，需要进一步调查。")
    
    print("测试完成。")
    
    return 0 if total_failed == 0 else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)