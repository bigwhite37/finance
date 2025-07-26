#!/usr/bin/env python3
"""
全量测试运行器

一键运行tests目录下的所有测试，提供详细的测试报告和统计信息。
支持不同的运行模式和输出格式。
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """测试运行器类"""
    
    def __init__(self, verbose: bool = True, fast_fail: bool = False, timeout: int = 600):
        """初始化测试运行器
        
        Args:
            verbose: 是否输出详细信息
            fast_fail: 是否在第一个失败时停止
            timeout: 测试超时时间（秒）
        """
        self.verbose = verbose
        self.fast_fail = fast_fail
        self.timeout = timeout
        self.tests_dir = Path(__file__).parent
        self.results = {}
        
    def discover_test_modules(self) -> Dict[str, List[str]]:
        """发现所有测试模块"""
        test_categories = {
            'core_functionality': [
                'dynamic_lowvol_filter/test_exceptions.py',
                'dynamic_lowvol_filter/test_backtest.py', 
                'dynamic_lowvol_filter/test_integration.py'
            ],
            'unit_tests': [],
            'integration_tests': [],
            'performance_tests': [
                'dynamic_lowvol_filter/test_performance.py',
                'dynamic_lowvol_filter/test_stability.py'
            ]
        }
        
        # 发现单元测试
        unit_dir = self.tests_dir / 'unit'
        if unit_dir.exists():
            for test_file in unit_dir.glob('test_*.py'):
                test_categories['unit_tests'].append(f'unit/{test_file.name}')
        
        # 发现集成测试
        integration_dir = self.tests_dir / 'integration'
        if integration_dir.exists():
            for test_file in integration_dir.glob('test_*.py'):
                test_categories['integration_tests'].append(f'integration/{test_file.name}')
        
        # 过滤存在的文件
        for category in test_categories:
            test_categories[category] = [
                test for test in test_categories[category]
                if (self.tests_dir / test).exists()
            ]
        
        return test_categories
    
    def run_pytest_command(self, test_files: List[str], extra_args: List[str] = None) -> Tuple[int, str, str]:
        """运行pytest命令"""
        if extra_args is None:
            extra_args = []
            
        # 构建pytest命令
        cmd = ['python', '-m', 'pytest'] + extra_args
        
        # 添加测试文件
        for test_file in test_files:
            cmd.append(str(self.tests_dir / test_file))
        
        if self.verbose:
            print(f"运行命令: {' '.join(cmd)}")
        
        # 运行命令
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(project_root)
        )
        
        stdout, stderr = process.communicate(timeout=self.timeout)
        end_time = time.time()
        
        return process.returncode, stdout, stderr, end_time - start_time
    
    def run_test_category(self, category: str, test_files: List[str]) -> Dict:
        """运行特定类别的测试"""
        if not test_files:
            return {
                'category': category,
                'status': 'skipped',
                'reason': 'No test files found',
                'duration': 0,
                'tests_run': 0,
                'failures': 0,
                'errors': 0
            }
        
        print(f"\n{'='*60}")
        print(f"运行 {category.upper()} 测试")
        print(f"{'='*60}")
        print(f"测试文件: {', '.join(test_files)}")
        
        # 设置pytest参数
        pytest_args = ['--tb=short', '-v']
        if self.fast_fail:
            pytest_args.append('-x')
        
        # 根据测试类别调整参数
        if category == 'performance_tests':
            pytest_args.extend(['--timeout=300'])  # 性能测试用较短超时
        elif category == 'unit_tests':
            pytest_args.append('-q')  # 单元测试用简洁输出
        
        returncode, stdout, stderr, duration = self.run_pytest_command(test_files, pytest_args)
        
        # 解析结果
        result = self.parse_pytest_output(stdout, stderr, returncode, duration)
        result['category'] = category
        result['test_files'] = test_files
        
        # 输出结果摘要
        status_symbol = "✅" if result['status'] == 'passed' else "❌" if result['status'] == 'failed' else "⚠️"
        print(f"\n{status_symbol} {category}: {result['tests_run']} 个测试运行，{result['failures']} 个失败，{result['errors']} 个错误")
        print(f"   耗时: {duration:.2f}秒")
        
        if result['status'] == 'failed' and stderr:
            print(f"   错误信息: {stderr[:200]}...")
        
        return result
    
    def parse_pytest_output(self, stdout: str, stderr: str, returncode: int, duration: float) -> Dict:
        """解析pytest输出"""
        result = {
            'status': 'unknown',
            'duration': duration,
            'tests_run': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'stdout': stdout,
            'stderr': stderr,
            'returncode': returncode
        }
        
        if returncode == 0:
            result['status'] = 'passed'
        elif returncode == 1:
            result['status'] = 'failed'
        elif returncode == 2:
            result['status'] = 'error'
        else:
            result['status'] = 'timeout' if 'timeout' in stderr.lower() else 'unknown'
        
        # 更好的输出解析 - 寻找最后的统计行
        lines = stdout.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
                
            # 寻找统计行，如 "12 passed, 2 warnings in 108.82s"
            if ('passed' in line or 'failed' in line or 'error' in line) and ('in' in line and 's' in line):
                import re
                # 提取数字和状态
                patterns = [
                    (r'(\d+)\s+passed', 'passed'),
                    (r'(\d+)\s+failed', 'failed'), 
                    (r'(\d+)\s+error', 'errors'),
                    (r'(\d+)\s+skipped', 'skipped')
                ]
                
                for pattern, status_type in patterns:
                    matches = re.findall(pattern, line)
                    if matches:
                        count = int(matches[0])
                        if status_type == 'passed':
                            result['tests_run'] += count
                        elif status_type == 'failed':
                            result['failures'] = count
                            result['tests_run'] += count
                        elif status_type == 'errors':
                            result['errors'] = count
                            result['tests_run'] += count
                        elif status_type == 'skipped':
                            result['skipped'] = count
                            result['tests_run'] += count
                break
        
        # 如果没有找到统计信息，尝试从输出中计算
        if result['tests_run'] == 0:
            # 计算PASSED和FAILED标记的数量
            passed_count = stdout.count('PASSED')
            failed_count = stdout.count('FAILED')
            error_count = stdout.count('ERROR')
            skipped_count = stdout.count('SKIPPED')
            
            result['tests_run'] = passed_count + failed_count + error_count + skipped_count
            result['failures'] = failed_count
            result['errors'] = error_count
            result['skipped'] = skipped_count
        
        return result
    
    def run_all_tests(self, categories: Optional[List[str]] = None) -> Dict:
        """运行所有测试"""
        print("🧪 开始全量测试运行")
        print(f"项目根目录: {project_root}")
        print(f"测试目录: {self.tests_dir}")
        
        # 发现测试模块
        test_modules = self.discover_test_modules()
        
        if categories:
            # 只运行指定类别
            test_modules = {k: v for k, v in test_modules.items() if k in categories}
        
        print(f"\n发现的测试类别: {list(test_modules.keys())}")
        
        # 运行测试
        all_results = {}
        total_start_time = time.time()
        
        for category, test_files in test_modules.items():
            result = self.run_test_category(category, test_files)
            all_results[category] = result
            
            # 如果启用快速失败且有失败
            if self.fast_fail and result['status'] == 'failed':
                print(f"\n⚠️ 快速失败模式: 在 {category} 中检测到失败，停止后续测试")
                break
        
        total_duration = time.time() - total_start_time
        
        # 生成总结报告
        summary = self.generate_summary_report(all_results, total_duration)
        
        return {
            'summary': summary,
            'results': all_results,
            'total_duration': total_duration
        }
    
    def generate_summary_report(self, results: Dict, total_duration: float) -> Dict:
        """生成总结报告"""
        summary = {
            'total_categories': len(results),
            'passed_categories': 0,
            'failed_categories': 0,
            'total_tests': 0,
            'total_failures': 0,
            'total_errors': 0,
            'total_skipped': 0,
            'overall_status': 'passed',
            'total_duration': total_duration
        }
        
        for category, result in results.items():
            if result['status'] == 'passed':
                summary['passed_categories'] += 1
            elif result['status'] in ['failed', 'error']:
                summary['failed_categories'] += 1
                summary['overall_status'] = 'failed'
            
            summary['total_tests'] += result['tests_run']
            summary['total_failures'] += result['failures']
            summary['total_errors'] += result['errors']
            summary['total_skipped'] += result['skipped']
        
        return summary
    
    def print_final_report(self, test_results: Dict):
        """打印最终报告"""
        summary = test_results['summary']
        
        print(f"\n{'='*80}")
        print("📊 测试执行总结报告")
        print(f"{'='*80}")
        
        # 总体状态
        status_symbol = "✅" if summary['overall_status'] == 'passed' else "❌"
        print(f"{status_symbol} 总体状态: {summary['overall_status'].upper()}")
        print(f"⏱️  总耗时: {summary['total_duration']:.2f}秒")
        print()
        
        # 分类统计
        print("📋 分类统计:")
        print(f"   测试类别: {summary['total_categories']} 个")
        print(f"   通过类别: {summary['passed_categories']} 个")
        print(f"   失败类别: {summary['failed_categories']} 个")
        print()
        
        # 测试统计
        print("🧮 测试统计:")
        print(f"   总测试数: {summary['total_tests']} 个")
        print(f"   失败数: {summary['total_failures']} 个")
        print(f"   错误数: {summary['total_errors']} 个")
        print(f"   跳过数: {summary['total_skipped']} 个")
        print()
        
        # 详细结果
        print("📝 详细结果:")
        for category, result in test_results['results'].items():
            status_symbol = "✅" if result['status'] == 'passed' else "❌" if result['status'] == 'failed' else "⚠️"
            print(f"   {status_symbol} {category:<20} | {result['tests_run']:>3} 测试 | {result['failures']:>2} 失败 | {result['duration']:>6.2f}s")
        
        # 失败详情
        failed_categories = [cat for cat, result in test_results['results'].items() if result['status'] in ['failed', 'error']]
        if failed_categories:
            print(f"\n❌ 失败的测试类别:")
            for category in failed_categories:
                result = test_results['results'][category]
                print(f"   • {category}: {result['failures']} 个失败, {result['errors']} 个错误")
                if result['stderr']:
                    print(f"     错误信息: {result['stderr'][:150]}...")
        
        print(f"\n{'='*80}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行全量测试")
    parser.add_argument('--categories', nargs='+', 
                       choices=['core_functionality', 'unit_tests', 'integration_tests', 'performance_tests'],
                       help='指定要运行的测试类别')
    parser.add_argument('--fast-fail', action='store_true', help='在第一个失败时停止')
    parser.add_argument('--quiet', action='store_true', help='简洁输出')
    parser.add_argument('--timeout', type=int, default=600, help='测试超时时间（秒）')
    
    args = parser.parse_args()
    
    # 创建测试运行器
    runner = TestRunner(
        verbose=not args.quiet,
        fast_fail=args.fast_fail,
        timeout=args.timeout
    )
    
    # 运行测试
    results = runner.run_all_tests(categories=args.categories)
    
    # 打印报告
    runner.print_final_report(results)
    
    # 返回适当的退出码
    exit_code = 0 if results['summary']['overall_status'] == 'passed' else 1
    sys.exit(exit_code)


if __name__ == '__main__':
    main()