#!/usr/bin/env python3
"""
Comprehensive test runner to identify all failing tests
"""

import subprocess
import sys
import os
from datetime import datetime

def run_test_file(test_file):
    """Run a single test file and return results"""
    print(f"Testing {test_file}...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, 
            '-v', '--tb=short', '--maxfail=1000'
        ], 
        capture_output=True, text=True, timeout=1800)  # 30 minutes
        
        return {
            'file': test_file,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'file': test_file,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Test timed out after 30 minutes',
            'passed': False
        }
    except Exception as e:
        return {
            'file': test_file,
            'returncode': -2,
            'stdout': '',
            'stderr': str(e),
            'passed': False
        }

def find_test_files():
    """Find all test files"""
    test_files = []
    for root, dirs, files in os.walk('tests'):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    return sorted(test_files)

def main():
    """Main test runner"""
    print("Starting comprehensive test run...")
    print(f"Start time: {datetime.now()}")
    
    test_files = find_test_files()
    print(f"Found {len(test_files)} test files")
    
    passed_tests = []
    failed_tests = []
    timed_out_tests = []
    
    for i, test_file in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] Testing {test_file}")
        result = run_test_file(test_file)
        
        if result['passed']:
            passed_tests.append(result)
            print(f"✅ PASSED")
        elif result['returncode'] == -1:
            timed_out_tests.append(result)
            print(f"⏰ TIMED OUT")
        else:
            failed_tests.append(result)
            print(f"❌ FAILED (code: {result['returncode']})")
    
    # Write results to file
    with open('archive/test_failure_list.md', 'w') as f:
        f.write("# Test Results Report\n\n")
        f.write(f"**Generated on**: {datetime.now()}\n")
        f.write(f"**Total test files**: {len(test_files)}\n")
        f.write(f"**Passed**: {len(passed_tests)}\n")
        f.write(f"**Failed**: {len(failed_tests)}\n")
        f.write(f"**Timed out**: {len(timed_out_tests)}\n\n")
        
        if failed_tests:
            f.write("## Failed Tests\n\n")
            for result in failed_tests:
                f.write(f"### {result['file']}\n")
                f.write(f"**Return code**: {result['returncode']}\n")
                f.write(f"**Error output**:\n```\n{result['stderr']}\n```\n")
                f.write(f"**Test output**:\n```\n{result['stdout'][-2000:]}\n```\n\n")
        
        if timed_out_tests:
            f.write("## Timed Out Tests\n\n")
            for result in timed_out_tests:
                f.write(f"### {result['file']}\n")
                f.write(f"**Error**: {result['stderr']}\n\n")
        
        if passed_tests:
            f.write("## Passed Tests\n\n")
            for result in passed_tests:
                f.write(f"- ✅ {result['file']}\n")
    
    print(f"\n\nTest run completed!")
    print(f"End time: {datetime.now()}")
    print(f"Results written to archive/test_failure_list.md")
    print(f"Summary: {len(passed_tests)} passed, {len(failed_tests)} failed, {len(timed_out_tests)} timed out")

if __name__ == '__main__':
    main()