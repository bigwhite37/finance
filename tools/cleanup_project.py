#!/usr/bin/env python3
"""
项目清理脚本 - 整理代码文件结构
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """清理项目文件"""
    
    # 1. 创建必要的目录结构
    directories_to_create = [
        'scripts',
        'temp',
        'archive',
        'docs/examples'
    ]
    
    for dir_path in directories_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")
    
    # 2. 移动临时和测试文件
    files_to_move = {
        # 临时文件移到 temp 目录
        'final_training_test.py': 'temp/',
        'simple_training_config.py': 'temp/',
        'error.txt': 'temp/',
        'tmp.md': 'temp/',
        'performance_benchmark_report.txt': 'temp/',
        
        # 脚本文件移到 scripts 目录
        'npm-install-17084.sh': 'archive/',  # 这个文件不需要，移到archive
    }
    
    for src, dst_dir in files_to_move.items():
        if os.path.exists(src):
            dst = os.path.join(dst_dir, src)
            shutil.move(src, dst)
            print(f"📁 移动文件: {src} -> {dst}")
    
    # 3. 删除不需要的文件
    files_to_delete = [
        'npm-install-17084.sh',  # 如果还存在的话
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ 删除文件: {file_path}")
    
    # 4. 整理测试文件 - 将根目录的测试文件移到tests目录
    test_files_in_root = [
        'test_backtest_validation.py',
        'test_coordination_demo.py',
        'test_task10_verification.py',
        'test_task12_verification.py',
        'test_task13_verification.py',
        'test_task8_verification.py',
        'test_task9_verification.py',
    ]
    
    for test_file in test_files_in_root:
        if os.path.exists(test_file):
            dst = os.path.join('tests', test_file)
            if not os.path.exists(dst):  # 避免覆盖已存在的文件
                shutil.move(test_file, dst)
                print(f"🧪 移动测试文件: {test_file} -> tests/")
            else:
                os.remove(test_file)  # 如果tests目录已有同名文件，删除根目录的
                print(f"🗑️ 删除重复测试文件: {test_file}")
    
    # 5. 整理scripts目录中的文件
    script_files_in_root = [
        'run_comprehensive_tests.py',
    ]
    
    for script_file in script_files_in_root:
        if os.path.exists(script_file):
            dst = os.path.join('scripts', script_file)
            if not os.path.exists(dst):
                shutil.move(script_file, dst)
                print(f"📜 移动脚本文件: {script_file} -> scripts/")
    
    # 6. 清理空的缓存目录
    cache_dirs = ['__pycache__', 'temp_cache_test']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
            if cache_dir == 'temp_cache_test' and not os.listdir(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"🗑️ 删除空目录: {cache_dir}")
    
    print("\n🎉 项目清理完成！")
    
    # 7. 显示清理后的项目结构
    print("\n📁 清理后的项目结构:")
    print_directory_tree(".", max_depth=2)

def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """打印目录树"""
    if current_depth >= max_depth:
        return
    
    path = Path(path)
    items = sorted([item for item in path.iterdir() 
                   if not item.name.startswith('.') and 
                   item.name not in ['__pycache__', 'node_modules']])
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and current_depth < max_depth - 1:
            extension = "    " if is_last else "│   "
            print_directory_tree(item, prefix + extension, max_depth, current_depth + 1)

if __name__ == "__main__":
    cleanup_project()