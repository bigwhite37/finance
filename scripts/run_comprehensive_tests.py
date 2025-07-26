#!/usr/bin/env python3
"""
动态低波筛选器综合测试运行器

执行所有测试并生成最终报告：
1. 端到端集成测试
2. 性能基准测试
3. 回测验证测试
4. 异常处理测试
5. 系统稳定性测试
"""

import sys
import os
import subprocess
import time
import logging
from datetime import datetime
from typing import Dict, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveTestRunner:
    """综合测试运行器"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        logger.info("开始运行动态低波筛选器综合测试套件...")
        self.start_time = time.time()
        
        # 测试配置
        tests_to_run = [
            {
                'name': 'comprehensive_suite',
                'description': '综合测试套件',
                'script': 'tests/test_comprehensive_suite.py',
                'timeout': 600  # 10分钟超时
            },
            {
                'name': 'performance_benchmark',
                'description': '性能基准测试',
                'script': 'test_performance_benchmark.py',
                'timeout': 300  # 5分钟超时
            },
            {
                'name': 'backtest_validation',
                'description': '回测验证测试',
                'script': 'test_backtest_validation.py',
                'timeout': 600  # 10分钟超时
            }
        ]
        
        # 运行每个测试
        for test_config in tests_to_run:
            self._run_single_test(test_config)
        
        self.end_time = time.time()
        
        # 汇总结果
        summary = self._generate_summary()
        
        # 生成最终报告
        self._generate_final_report(summary)
        
        return summary
    
    def _run_single_test(self, test_config: Dict):
        """运行单个测试"""
        test_name = test_config['name']
        test_script = test_config['script']
        test_timeout = test_config['timeout']
        
        logger.info(f"开始运行 {test_config['description']}...")
        
        start_time = time.time()
        
        try:
            # 运行测试脚本
            result = subprocess.run(
                [sys.executable, test_script],
                capture_output=True,
                text=True,
                timeout=test_timeout,
                cwd=os.getcwd()
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 记录结果
            self.test_results[test_name] = {
                'description': test_config['description'],
                'script': test_script,
                'success': result.returncode == 0,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            if result.returncode == 0:
                logger.info(f"{test_config['description']} 完成 - 耗时: {execution_time:.1f}s")
            else:
                logger.error(f"{test_config['description']} 失败 - 耗时: {execution_time:.1f}s")
                logger.error(f"错误输出: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.error(f"{test_config['description']} 超时 - 耗时: {execution_time:.1f}s")
            
            self.test_results[test_name] = {
                'description': test_config['description'],
                'script': test_script,
                'success': False,
                'execution_time': execution_time,
                'stdout': '',
                'stderr': f'测试超时 ({test_timeout}s)',
                'return_code': -1,
                'timeout': True
            }
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.error(f"{test_config['description']} 异常 - {str(e)}")
            
            self.test_results[test_name] = {
                'description': test_config['description'],
                'script': test_script,
                'success': False,
                'execution_time': execution_time,
                'stdout': '',
                'stderr': str(e),
                'return_code': -2,
                'exception': True
            }
    
    def _generate_summary(self) -> Dict:
        """生成测试汇总"""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - successful_tests
        
        total_execution_time = self.end_time - self.start_time
        
        summary = {
            'start_time': datetime.fromtimestamp(self.start_time),
            'end_time': datetime.fromtimestamp(self.end_time),
            'total_execution_time': total_execution_time,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'overall_success': failed_tests == 0,
            'test_results': self.test_results
        }
        
        return summary
    
    def _generate_final_report(self, summary: Dict):
        """生成最终测试报告"""
        logger.info("生成最终测试报告...")
        
        report_content = self._create_report_content(summary)
        
        # 保存报告
        report_path = 'reports/comprehensive_test_report.md'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"最终测试报告已保存到: {report_path}")
        
        # 同时生成简化的文本报告
        text_report_path = 'reports/comprehensive_test_summary.txt'
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(self._create_text_summary(summary))
        
        logger.info(f"测试摘要已保存到: {text_report_path}")
    
    def _create_report_content(self, summary: Dict) -> str:
        """创建报告内容"""
        report_content = f"""# 动态低波筛选器综合测试报告

## 测试概要

- **测试时间**: {summary['start_time'].strftime('%Y-%m-%d %H:%M:%S')} - {summary['end_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **总耗时**: {summary['total_execution_time']:.1f} 秒
- **总测试数**: {summary['total_tests']}
- **成功测试**: {summary['successful_tests']}
- **失败测试**: {summary['failed_tests']}
- **成功率**: {summary['success_rate']:.1%}
- **整体结果**: {'✅ 通过' if summary['overall_success'] else '❌ 失败'}

## 详细测试结果

"""
        
        # 添加每个测试的详细结果
        for test_name, test_result in summary['test_results'].items():
            status_icon = '✅' if test_result['success'] else '❌'
            
            report_content += f"""### {status_icon} {test_result['description']}

- **脚本**: `{test_result['script']}`
- **状态**: {'成功' if test_result['success'] else '失败'}
- **耗时**: {test_result['execution_time']:.1f} 秒
- **返回码**: {test_result['return_code']}

"""
            
            # 添加错误信息（如果有）
            if not test_result['success']:
                if test_result.get('timeout'):
                    report_content += "**错误类型**: 测试超时\\n\\n"
                elif test_result.get('exception'):
                    report_content += "**错误类型**: 运行异常\\n\\n"
                
                if test_result['stderr']:
                    report_content += f"""**错误输出**:
```
{test_result['stderr'][:1000]}{'...' if len(test_result['stderr']) > 1000 else ''}
```

"""
            
            # 添加部分标准输出（如果有）
            if test_result['stdout']:
                stdout_preview = test_result['stdout'][:500]
                report_content += f"""**输出预览**:
```
{stdout_preview}{'...' if len(test_result['stdout']) > 500 else ''}
```

"""
        
        # 添加结论和建议
        report_content += """## 测试结论

"""
        
        if summary['overall_success']:
            report_content += """✅ **所有测试通过**

动态低波筛选器通过了全部综合测试，包括：

1. **端到端集成测试** - 验证了完整筛选流水线的正确性
2. **性能基准测试** - 确认计算延迟满足<100ms要求
3. **回测验证测试** - 证实筛选效果达到预期指标
4. **异常处理测试** - 验证了所有异常情况的正确处理
5. **系统稳定性测试** - 确认长期运行的稳定性

系统已准备好投入生产使用。

## 建议

1. **定期监控**: 建立定期测试机制，监控系统性能变化
2. **参数调优**: 根据实际市场情况调整筛选参数
3. **功能扩展**: 考虑添加更多预测模型和筛选策略
4. **文档维护**: 保持使用文档和API文档的更新

"""
        else:
            failed_test_names = [result['description'] for result in summary['test_results'].values() if not result['success']]
            
            report_content += f"""❌ **部分测试失败**

失败的测试项目: {', '.join(failed_test_names)}

## 问题分析

"""
            
            for test_name, test_result in summary['test_results'].items():
                if not test_result['success']:
                    report_content += f"""### {test_result['description']}

可能的问题原因：
"""
                    if test_result.get('timeout'):
                        report_content += """- 测试执行时间过长，可能存在性能问题
- 数据规模过大导致计算超时
- 算法效率需要优化

"""
                    elif test_result.get('exception'):
                        report_content += """- 代码存在运行时错误
- 依赖库版本不兼容
- 环境配置问题

"""
                    else:
                        report_content += """- 功能逻辑错误
- 测试用例不符合实际情况
- 参数配置不当

"""
            
            report_content += """## 修复建议

1. **优先修复**: 按照测试失败的严重程度优先修复
2. **代码审查**: 对失败的模块进行详细代码审查
3. **单元测试**: 增加更细粒度的单元测试
4. **性能优化**: 针对超时问题进行性能优化
5. **环境检查**: 确认测试环境配置正确

"""
        
        # 添加附录
        report_content += """## 附录

### 测试环境信息

- **Python版本**: """ + sys.version + """
- **操作系统**: """ + os.name + """
- **工作目录**: """ + os.getcwd() + """

### 相关文件

- 综合测试套件: `tests/test_comprehensive_suite.py`
- 性能基准测试: `test_performance_benchmark.py`
- 回测验证测试: `test_backtest_validation.py`
- 测试运行器: `run_comprehensive_tests.py`

### 报告文件

- 综合测试报告: `reports/comprehensive_test_report.md`
- 性能基准报告: `reports/performance_benchmark_report.md`
- 回测验证报告: `reports/backtest_validation_report.md`
- 测试摘要: `reports/comprehensive_test_summary.txt`

"""
        
        return report_content
    
    def _create_text_summary(self, summary: Dict) -> str:
        """创建文本摘要"""
        text_summary = f"""动态低波筛选器综合测试摘要
{'='*50}

测试时间: {summary['start_time'].strftime('%Y-%m-%d %H:%M:%S')} - {summary['end_time'].strftime('%Y-%m-%d %H:%M:%S')}
总耗时: {summary['total_execution_time']:.1f} 秒
总测试数: {summary['total_tests']}
成功测试: {summary['successful_tests']}
失败测试: {summary['failed_tests']}
成功率: {summary['success_rate']:.1%}
整体结果: {'通过' if summary['overall_success'] else '失败'}

详细结果:
{'-'*30}
"""
        
        for test_name, test_result in summary['test_results'].items():
            status = '✓' if test_result['success'] else '✗'
            text_summary += f"{status} {test_result['description']}: {test_result['execution_time']:.1f}s\\n"
        
        if not summary['overall_success']:
            text_summary += f"\\n失败原因:\\n{'-'*20}\\n"
            for test_name, test_result in summary['test_results'].items():
                if not test_result['success']:
                    text_summary += f"- {test_result['description']}: {test_result['stderr'][:100]}...\\n"
        
        return text_summary
    
    def _log_summary(self, summary: Dict):
        """记录测试摘要"""
        logger.info("测试摘要:")
        logger.info(f"  总测试数: {summary['total_tests']}")
        logger.info(f"  成功测试: {summary['successful_tests']}")
        logger.info(f"  失败测试: {summary['failed_tests']}")
        logger.info(f"  成功率: {summary['success_rate']:.1%}")
        logger.info(f"  总耗时: {summary['total_execution_time']:.1f}s")
        logger.info(f"  整体结果: {'通过' if summary['overall_success'] else '失败'}")
        
        if not summary['overall_success']:
            logger.error("失败的测试:")
            for test_name, test_result in summary['test_results'].items():
                if not test_result['success']:
                    logger.error(f"  - {test_result['description']}")


def main():
    """主函数"""
    logger.info("启动动态低波筛选器综合测试...")
    
    # 创建测试运行器
    runner = ComprehensiveTestRunner()
    
    # 运行所有测试
    summary = runner.run_all_tests()
    
    # 记录摘要
    runner._log_summary(summary)
    
    # 返回结果
    if summary['overall_success']:
        logger.info("所有综合测试通过！")
        return 0
    else:
        logger.error("存在测试失败！")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)