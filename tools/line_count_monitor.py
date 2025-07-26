#!/usr/bin/env python3
"""
代码行数监控脚本
用于监控Python文件的行数，识别需要重构的超长文件
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


def count_lines_in_file(file_path: Path) -> int:
    """计算文件的行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except (UnicodeDecodeError, IOError):
        return 0


def scan_python_files(root_dir: Path, exclude_dirs: List[str] = None) -> List[Tuple[Path, int]]:
    """扫描目录中的所有Python文件并统计行数"""
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', '.pytest_cache', 'venv', 'env', '.venv']
    
    python_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # 排除指定目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                line_count = count_lines_in_file(file_path)
                python_files.append((file_path, line_count))
    
    return python_files


def analyze_files(files: List[Tuple[Path, int]], threshold: int = 500) -> Dict[str, List[Tuple[Path, int]]]:
    """分析文件，按行数分类"""
    result = {
        'over_threshold': [],
        'near_threshold': [],
        'normal': []
    }
    
    for file_path, line_count in files:
        if line_count > threshold:
            result['over_threshold'].append((file_path, line_count))
        elif line_count > threshold * 0.8:  # 80%阈值作为接近警告
            result['near_threshold'].append((file_path, line_count))
        else:
            result['normal'].append((file_path, line_count))
    
    # 按行数降序排序
    for category in result.values():
        category.sort(key=lambda x: x[1], reverse=True)
    
    return result


def print_report(analysis: Dict[str, List[Tuple[Path, int]]], threshold: int = 500):
    """打印分析报告"""
    print(f"代码行数监控报告 (阈值: {threshold}行)")
    print("=" * 60)
    
    # 超过阈值的文件
    if analysis['over_threshold']:
        print(f"\n🔴 超过阈值的文件 ({len(analysis['over_threshold'])}个):")
        for file_path, line_count in analysis['over_threshold']:
            try:
                relative_path = file_path.relative_to(Path.cwd())
            except ValueError:
                relative_path = file_path
            print(f"  {relative_path}: {line_count}行")
    
    # 接近阈值的文件
    if analysis['near_threshold']:
        print(f"\n🟡 接近阈值的文件 ({len(analysis['near_threshold'])}个):")
        for file_path, line_count in analysis['near_threshold']:
            try:
                relative_path = file_path.relative_to(Path.cwd())
            except ValueError:
                relative_path = file_path
            print(f"  {relative_path}: {line_count}行")
    
    # 统计信息
    total_files = sum(len(files) for files in analysis.values())
    total_lines = sum(line_count for files in analysis.values() for _, line_count in files)
    
    print(f"\n📊 统计信息:")
    print(f"  总文件数: {total_files}")
    print(f"  总行数: {total_lines}")
    print(f"  平均行数: {total_lines // total_files if total_files > 0 else 0}")
    print(f"  需要重构的文件: {len(analysis['over_threshold'])}")


def save_report(analysis: Dict[str, List[Tuple[Path, int]]], output_file: Path, threshold: int = 500):
    """保存报告到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# 代码行数监控报告\n\n")
        f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"阈值: {threshold}行\n\n")
        
        if analysis['over_threshold']:
            f.write(f"## 超过阈值的文件 ({len(analysis['over_threshold'])}个)\n\n")
            for file_path, line_count in analysis['over_threshold']:
                try:
                    relative_path = file_path.relative_to(Path.cwd())
                except ValueError:
                    relative_path = file_path
                f.write(f"- `{relative_path}`: {line_count}行\n")
            f.write("\n")
        
        if analysis['near_threshold']:
            f.write(f"## 接近阈值的文件 ({len(analysis['near_threshold'])}个)\n\n")
            for file_path, line_count in analysis['near_threshold']:
                try:
                    relative_path = file_path.relative_to(Path.cwd())
                except ValueError:
                    relative_path = file_path
                f.write(f"- `{relative_path}`: {line_count}行\n")
            f.write("\n")
        
        total_files = sum(len(files) for files in analysis.values())
        total_lines = sum(line_count for files in analysis.values() for _, line_count in files)
        
        f.write(f"## 统计信息\n\n")
        f.write(f"- 总文件数: {total_files}\n")
        f.write(f"- 总行数: {total_lines}\n")
        f.write(f"- 平均行数: {total_lines // total_files if total_files > 0 else 0}\n")
        f.write(f"- 需要重构的文件: {len(analysis['over_threshold'])}\n")


def get_refactoring_priority_files(analysis: Dict[str, List[Tuple[Path, int]]]) -> List[Tuple[Path, int]]:
    """获取按重构优先级排序的文件列表"""
    priority_files = []
    
    # 按行数降序排列超过阈值的文件
    for file_path, line_count in analysis['over_threshold']:
        priority_files.append((file_path, line_count))
    
    return priority_files


def check_refactoring_progress(before_report: str, after_report: str = None) -> Dict:
    """检查重构进度"""
    if not Path(before_report).exists():
        print(f"错误: 基准报告文件 {before_report} 不存在")
        return {}
    
    # 读取重构前的报告
    with open(before_report, 'r', encoding='utf-8') as f:
        before_content = f.read()
    
    # 如果没有指定重构后报告，则生成当前状态报告
    if after_report is None:
        files = scan_python_files(Path('.'))
        after_analysis = analyze_files(files, 500)
    else:
        # 读取重构后的报告（这里简化处理，实际应该解析markdown）
        after_analysis = analyze_files(scan_python_files(Path('.')), 500)
    
    # 计算改进情况
    before_over_threshold = len([line for line in before_content.split('\n') if line.startswith('- `') and '行' in line])
    after_over_threshold = len(after_analysis['over_threshold'])
    
    progress = {
        'before_count': before_over_threshold,
        'after_count': after_over_threshold,
        'improved_count': before_over_threshold - after_over_threshold,
        'improvement_rate': (before_over_threshold - after_over_threshold) / before_over_threshold if before_over_threshold > 0 else 0
    }
    
    return progress


def main():
    parser = argparse.ArgumentParser(description='监控Python文件行数')
    parser.add_argument('--threshold', type=int, default=500, help='行数阈值 (默认: 500)')
    parser.add_argument('--output', type=str, help='输出报告文件路径')
    parser.add_argument('--directory', type=str, default='.', help='扫描目录 (默认: 当前目录)')
    parser.add_argument('--priority', action='store_true', help='显示重构优先级列表')
    parser.add_argument('--progress', type=str, help='检查重构进度，指定基准报告文件路径')
    
    args = parser.parse_args()
    
    root_dir = Path(args.directory)
    if not root_dir.exists():
        print(f"错误: 目录 {root_dir} 不存在")
        sys.exit(1)
    
    print(f"扫描目录: {root_dir.absolute()}")
    files = scan_python_files(root_dir)
    analysis = analyze_files(files, args.threshold)
    
    if args.priority:
        # 显示重构优先级
        priority_files = get_refactoring_priority_files(analysis)
        print(f"\n🎯 重构优先级列表 (按行数降序):")
        for i, (file_path, line_count) in enumerate(priority_files, 1):
            try:
                relative_path = file_path.relative_to(Path.cwd())
            except ValueError:
                relative_path = file_path
            print(f"  {i}. {relative_path}: {line_count}行")
    
    if args.progress:
        # 检查重构进度
        progress = check_refactoring_progress(args.progress)
        if progress:
            print(f"\n📈 重构进度:")
            print(f"  重构前超标文件: {progress['before_count']}")
            print(f"  重构后超标文件: {progress['after_count']}")
            print(f"  已改进文件数: {progress['improved_count']}")
            print(f"  改进率: {progress['improvement_rate']:.1%}")
    
    print_report(analysis, args.threshold)
    
    if args.output:
        output_path = Path(args.output)
        save_report(analysis, output_path, args.threshold)
        print(f"\n报告已保存到: {output_path}")


if __name__ == '__main__':
    main()