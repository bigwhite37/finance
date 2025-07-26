#!/usr/bin/env python3
"""
O2O测试套件运行器

运行O2O系统的完整测试套件，包括单元测试、集成测试和性能基准测试。
"""

import sys
import subprocess
import time
from pathlib import Path

def run_test_suite(test_type: str = "all"):
    """
    运行指定类型的测试套件
    
    Args:
        test_type: 测试类型 ("unit", "integration", "performance", "all")
    """
    print("=" * 60)
    print("O2O强化学习系统测试套件")
    print("=" * 60)
    
    test_files = []
    
    if test_type in ["unit", "all"]:
        test_files.append("tests/test_o2o_components.py")
    
    if test_type in ["integration", "all"]:
        test_files.append("tests/test_o2o_integration.py")
    
    if test_type in ["performance", "all"]:
        test_files.append("tests/test_o2o_performance.py")
    
    total_passed = 0
    total_failed = 0
    total_time = 0
    
    for test_file in test_files:
        print(f"\n运行测试文件: {test_file}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # 运行pytest
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short"
            ], capture_output=True, text=True)
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # 解析结果
            output_lines = result.stdout.split('\n')
            
            passed = 0
            failed = 0
            
            for line in output_lines:
                if " PASSED " in line:
                    passed += 1
                elif " FAILED " in line:
                    failed += 1
            
            total_passed += passed
            total_failed += failed
            
            print(f"通过: {passed}, 失败: {failed}, 耗时: {elapsed_time:.2f}s")
            
            if failed > 0:
                print("失败的测试:")
                for line in output_lines:
                    if " FAILED " in line:
                        print(f"  - {line}")
                        
                # 显示错误详情
                if result.stderr:
                    print("\n错误详情:")
                    print(result.stderr)
            
        except Exception as e:
            print(f"运行测试时发生错误: {e}")
            total_failed += 1
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"总通过: {total_passed}")
    print(f"总失败: {total_failed}")
    print(f"总耗时: {total_time:.2f}s")
    
    if total_failed == 0:
        print("🎉 所有测试通过!")
        return 0
    else:
        print(f"❌ {total_failed} 个测试失败")
        return 1


def run_specific_tests():
    """运行特定的测试用例"""
    print("\n运行关键测试用例...")
    
    key_tests = [
        "tests/test_o2o_components.py::TestOfflineDataset::test_initialization",
        "tests/test_o2o_components.py::TestOnlineReplayBuffer::test_sample_batch",
        "tests/test_o2o_components.py::TestMixtureSampler::test_mixed_batch_sampling",
        "tests/test_o2o_integration.py::TestO2OIntegration::test_coordinator_initialization",
        "tests/test_o2o_integration.py::TestModeSwitch::test_environment_mode_switching",
        "tests/test_o2o_performance.py::TestO2OPerformanceBenchmark::test_sample_efficiency_comparison"
    ]
    
    for test in key_tests:
        print(f"\n运行: {test}")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test, 
                "-v"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ 通过")
            else:
                print("❌ 失败")
                print(result.stdout)
                
        except Exception as e:
            print(f"❌ 错误: {e}")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
    else:
        test_type = "all"
    
    if test_type == "key":
        run_specific_tests()
        return 0
    
    return run_test_suite(test_type)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)