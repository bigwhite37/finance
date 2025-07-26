#!/usr/bin/env python3
import subprocess
import sys
import os

# Test files that might be problematic
test_files = [
    "tests/dynamic_lowvol_filter/test_stability.py",
    "tests/dynamic_lowvol_filter/test_performance.py",  
    "tests/integration/test_comprehensive_suite.py",
    "tests/integration/test_system.py",
    "tests/integration/test_coordination_demo.py",
    "tests/integration/test_optimization_integration.py",
    "tests/acceptance/"
]

failed_tests = []

for test_file in test_files:
    if not os.path.exists(test_file):
        continue
        
    print(f"Testing {test_file}...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', test_file, 
            '--tb=no', '-q', '--maxfail=1'
        ], 
        capture_output=True, text=True, timeout=300)  # 5 minutes per file
        
        if result.returncode == 0:
            print(f"✅ {test_file} - PASSED")
        else:
            print(f"❌ {test_file} - FAILED")
            failed_tests.append((test_file, result.stdout, result.stderr))
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file} - TIMED OUT")
        failed_tests.append((test_file, "TIMEOUT", "Test timed out after 5 minutes"))

# Write quick failure list
with open('archive/quick_failure_check.md', 'w') as f:
    f.write("# Quick Test Failure Check\n\n")
    if failed_tests:
        f.write("## Failed or Timed Out Tests\n\n")
        for test_file, stdout, stderr in failed_tests:
            f.write(f"### {test_file}\n")
            f.write(f"**Output**: {stdout[-500:]}\n")
            f.write(f"**Error**: {stderr[-500:]}\n\n")
    else:
        f.write("All checked tests passed!\n")

print(f"\nQuick check completed. {len(failed_tests)} problematic tests found.")