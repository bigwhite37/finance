#!/usr/bin/env python3
"""
重构测试验证脚本
用于在重构过程中验证代码功能完整性和测试通过情况
"""

import os
import sys
import subprocess
import time
import unittest
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from io import StringIO


class RefactoringTestValidator:
    """重构测试验证器"""
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.test_results = {}
        
    def validate_imports(self, module_paths: List[str]) -> Dict[str, bool]:
        """验证模块导入是否正常"""
        print("🔍 验证模块导入...")
        import_results = {}
        
        # 确保项目根目录在Python路径中
        project_root = str(self.project_root.absolute())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        for module_path in module_paths:
            try:
                # 将文件路径转换为模块路径
                module_name = module_path.replace('/', '.').replace('.py', '')
                if module_name.startswith('.'):
                    module_name = module_name[1:]
                
                # 尝试导入模块
                importlib.import_module(module_name)
                import_results[module_path] = True
                print(f"  ✅ {module_path}")
                
            except ImportError as e:
                import_results[module_path] = False
                print(f"  ❌ {module_path}: {str(e)}")
            except (ImportError, SyntaxError) as e:
                import_results[module_path] = False
                print(f"  ❌ {module_path}: {str(e)}")
        
        success_count = sum(import_results.values())
        total_count = len(import_results)
        print(f"\n导入验证结果: {success_count}/{total_count} 成功")
        
        return import_results
    
    def run_specific_tests(self, test_patterns: List[str]) -> Dict[str, Dict]:
        """运行特定的测试模式"""
        print("🧪 运行特定测试...")
        test_results = {}
        
        for pattern in test_patterns:
            print(f"\n运行测试模式: {pattern}")
            
            try:
                # 发现匹配的测试
                loader = unittest.TestLoader()
                suite = loader.discover(
                    start_dir=str(self.project_root / 'tests'),
                    pattern=pattern
                )
                
                # 运行测试
                stream = StringIO()
                runner = unittest.TextTestRunner(
                    stream=stream,
                    verbosity=2,
                    buffer=True
                )
                
                start_time = time.time()
                result = runner.run(suite)
                end_time = time.time()
                
                test_results[pattern] = {
                    'success': result.wasSuccessful(),
                    'tests_run': result.testsRun,
                    'failures': len(result.failures),
                    'errors': len(result.errors),
                    'execution_time': end_time - start_time,
                    'output': stream.getvalue(),
                    'failure_details': result.failures,
                    'error_details': result.errors
                }
                
                if result.wasSuccessful():
                    print(f"  ✅ {pattern}: {result.testsRun} 测试通过")
                else:
                    print(f"  ❌ {pattern}: {len(result.failures)} 失败, {len(result.errors)} 错误")
                    
            except Exception as e:
                test_results[pattern] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': 0
                }
                print(f"  ❌ {pattern}: 运行异常 - {str(e)}")
        
        return test_results
    
    def validate_api_compatibility(self, api_modules: List[str]) -> Dict[str, bool]:
        """验证API兼容性"""
        print("🔌 验证API兼容性...")
        api_results = {}
        
        for module_name in api_modules:
            try:
                # 导入模块
                module = importlib.import_module(module_name)
                
                # 检查关键API是否存在
                expected_apis = self._get_expected_apis(module_name)
                missing_apis = []
                
                for api_name in expected_apis:
                    if not hasattr(module, api_name):
                        missing_apis.append(api_name)
                
                if not missing_apis:
                    api_results[module_name] = True
                    print(f"  ✅ {module_name}: API完整")
                else:
                    api_results[module_name] = False
                    print(f"  ❌ {module_name}: 缺少API - {', '.join(missing_apis)}")
                    
            except (ImportError, AttributeError) as e:
                api_results[module_name] = False
                print(f"  ❌ {module_name}: {str(e)}")
        
        return api_results
    
    def _get_expected_apis(self, module_name: str) -> List[str]:
        """获取模块预期的API列表"""
        # 根据模块名称返回预期的API
        api_mapping = {
            'risk_control.dynamic_lowvol_filter': [
                'DynamicLowVolFilter', 'DataPreprocessor', 'RollingPercentileFilter',
                'IVOLConstraintFilter', 'GARCHVolatilityPredictor', 'MarketRegimeDetector',
                'RegimeAwareThresholdAdjuster'
            ],
            'risk_control.performance_optimizer': [
                'PerformanceOptimizer'
            ]
        }
        
        return api_mapping.get(module_name, [])
    
    def run_integration_tests(self) -> Dict[str, Dict]:
        """运行集成测试"""
        print("🔗 运行集成测试...")
        
        integration_test_patterns = [
            'test_*integration*.py',
            'test_dynamic_lowvol_filter_integration.py',
            'test_risk_controller_integration.py'
        ]
        
        return self.run_specific_tests(integration_test_patterns)
    
    def run_performance_tests(self) -> Dict[str, Dict]:
        """运行性能测试"""
        print("⚡ 运行性能测试...")
        
        performance_test_patterns = [
            'test_*performance*.py',
            'test_*benchmark*.py'
        ]
        
        return self.run_specific_tests(performance_test_patterns)
    
    def validate_refactored_module(self, module_path: str, test_patterns: List[str] = None) -> Dict:
        """验证重构后的模块"""
        print(f"🔧 验证重构模块: {module_path}")
        
        if test_patterns is None:
            # 根据模块路径推断测试模式
            module_name = Path(module_path).stem
            test_patterns = [f'test_{module_name}*.py', f'test_*{module_name}*.py']
        
        validation_result = {
            'module_path': module_path,
            'import_success': False,
            'test_results': {},
            'api_compatibility': {},
            'overall_success': False
        }
        
        # 验证导入
        import_results = self.validate_imports([module_path])
        validation_result['import_success'] = import_results.get(module_path, False)
        
        if validation_result['import_success']:
            # 运行相关测试
            validation_result['test_results'] = self.run_specific_tests(test_patterns)
            
            # 验证API兼容性
            module_name = module_path.replace('/', '.').replace('.py', '')
            if module_name.startswith('.'):
                module_name = module_name[1:]
            validation_result['api_compatibility'] = self.validate_api_compatibility([module_name])
            
            # 判断整体成功
            test_success = all(result.get('success', False) for result in validation_result['test_results'].values())
            api_success = all(validation_result['api_compatibility'].values())
            validation_result['overall_success'] = test_success and api_success
        
        return validation_result
    
    def generate_validation_report(self, results: Dict, output_file: str = None) -> str:
        """生成验证报告"""
        report_content = f"""# 重构验证报告

生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 验证摘要

"""
        
        # 添加各项验证结果
        if 'import_results' in results:
            import_success = sum(results['import_results'].values())
            import_total = len(results['import_results'])
            report_content += f"- **导入验证**: {import_success}/{import_total} 成功\n"
        
        if 'test_results' in results:
            test_success = sum(1 for r in results['test_results'].values() if r.get('success', False))
            test_total = len(results['test_results'])
            report_content += f"- **测试验证**: {test_success}/{test_total} 成功\n"
        
        if 'api_results' in results:
            api_success = sum(results['api_results'].values())
            api_total = len(results['api_results'])
            report_content += f"- **API兼容性**: {api_success}/{api_total} 成功\n"
        
        # 添加详细结果
        report_content += "\n## 详细结果\n\n"
        
        if 'test_results' in results:
            report_content += "### 测试结果\n\n"
            for pattern, result in results['test_results'].items():
                status = "✅" if result.get('success', False) else "❌"
                report_content += f"- {status} **{pattern}**\n"
                if 'tests_run' in result:
                    report_content += f"  - 运行测试: {result['tests_run']}\n"
                    report_content += f"  - 失败: {result.get('failures', 0)}\n"
                    report_content += f"  - 错误: {result.get('errors', 0)}\n"
                    report_content += f"  - 耗时: {result.get('execution_time', 0):.2f}s\n"
                
                if not result.get('success', False) and 'failure_details' in result:
                    report_content += "  - 失败详情:\n"
                    for failure in result['failure_details'][:3]:  # 只显示前3个失败
                        report_content += f"    - {failure[0]}: {failure[1].split('AssertionError:')[-1].strip()[:100]}...\n"
                
                report_content += "\n"
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"验证报告已保存到: {output_file}")
        
        return report_content
    
    def quick_validation(self, module_path: str) -> bool:
        """快速验证模块（用于重构过程中的快速检查）"""
        print(f"⚡ 快速验证: {module_path}")
        
        # 验证导入
        import_results = self.validate_imports([module_path])
        if not import_results.get(module_path, False):
            print("❌ 导入失败")
            return False
        
        # 运行基本测试
        module_name = Path(module_path).stem
        basic_tests = [f'test_{module_name}.py']
        test_results = self.run_specific_tests(basic_tests)
        
        success = all(result.get('success', False) for result in test_results.values())
        print(f"{'✅' if success else '❌'} 快速验证{'通过' if success else '失败'}")
        
        return success


def main():
    parser = argparse.ArgumentParser(description='重构测试验证工具')
    parser.add_argument('--module', type=str, help='验证特定模块')
    parser.add_argument('--test-pattern', type=str, action='append', help='测试模式')
    parser.add_argument('--quick', action='store_true', help='快速验证模式')
    parser.add_argument('--integration', action='store_true', help='运行集成测试')
    parser.add_argument('--performance', action='store_true', help='运行性能测试')
    parser.add_argument('--output', type=str, help='输出报告文件')
    parser.add_argument('--api-check', type=str, action='append', help='检查API兼容性的模块')
    
    args = parser.parse_args()
    
    validator = RefactoringTestValidator()
    results = {}
    
    if args.module:
        if args.quick:
            success = validator.quick_validation(args.module)
            sys.exit(0 if success else 1)
        else:
            test_patterns = args.test_pattern or None
            results = validator.validate_refactored_module(args.module, test_patterns)
    
    elif args.integration:
        results['test_results'] = validator.run_integration_tests()
    
    elif args.performance:
        results['test_results'] = validator.run_performance_tests()
    
    elif args.api_check:
        results['api_results'] = validator.validate_api_compatibility(args.api_check)
    
    else:
        # 默认运行全面验证
        print("🔍 运行全面验证...")
        
        # 验证关键模块导入 (更新：使用重构后的模块路径)
        key_modules = [
            'risk_control/dynamic_lowvol_filter/__init__.py',  # 重构后为包
            'risk_control/performance_optimizer.py'
        ]
        results['import_results'] = validator.validate_imports(key_modules)
        
        # 运行集成测试
        results['test_results'] = validator.run_integration_tests()
        
        # 验证API兼容性
        api_modules = [
            'risk_control.dynamic_lowvol_filter',
            'risk_control.performance_optimizer'
        ]
        results['api_results'] = validator.validate_api_compatibility(api_modules)
    
    # 生成报告
    if results:
        output_file = args.output or 'reports/refactoring_validation_report.md'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        validator.generate_validation_report(results, output_file)
        
        # 判断整体成功
        overall_success = True
        if 'import_results' in results:
            overall_success &= all(results['import_results'].values())
        if 'test_results' in results:
            overall_success &= all(r.get('success', False) for r in results['test_results'].values())
        if 'api_results' in results:
            overall_success &= all(results['api_results'].values())
        
        print(f"\n{'✅' if overall_success else '❌'} 整体验证{'通过' if overall_success else '失败'}")
        sys.exit(0 if overall_success else 1)


if __name__ == '__main__':
    main()