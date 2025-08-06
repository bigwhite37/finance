#!/usr/bin/env python3
"""
代码质量检查测试
验证是否符合代码审查规则
"""
import ast
import os
import re
from pathlib import Path
from typing import List, Tuple

def find_python_files(directory: str) -> List[str]:
    """查找所有Python文件"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # 跳过测试目录和__pycache__
        dirs[:] = [d for d in dirs if not d.startswith('__pycache__') and d != '.git']
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def check_broad_exception_handling(file_path: str) -> List[Tuple[int, str]]:
    """检查是否有违规的异常处理"""
    violations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # 检查违规模式
        violation_patterns = [
            r'except\s*:',  # 裸except
            r'except\s+Exception\s*:',  # except Exception:
            r'except\s+BaseException\s*:',  # except BaseException:
        ]
        
        for i, line in enumerate(lines, 1):
            # 跳过注释和字符串字面量
            if line.strip().startswith('#') or 'print(' in line:
                continue
                
            for pattern in violation_patterns:
                if re.search(pattern, line):
                    # 检查下一行是否是pass或仅记录
                    next_lines = lines[i:i+3] if i < len(lines)-2 else lines[i:]
                    combined = ' '.join(next_lines).strip()
                    
                    if 'pass' in combined or ('logger.warning' in combined and 'raise' not in combined):
                        violations.append((i, line.strip()))
                    elif 'return' in combined and 'raise' not in combined and 'main(' not in combined:
                        violations.append((i, line.strip()))
        
    except Exception as e:
        print(f"检查文件 {file_path} 时出错: {e}")
    
    return violations

def check_hardcoded_data_generation(file_path: str) -> List[Tuple[int, str]]:
    """检查是否有硬编码数据生成"""
    violations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # 检查硬编码默认值返回
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # 检查return 硬编码数值的情况，但排除main函数的退出码
            if (re.match(r'return\s+\d+\.?\d*\s*$', line_stripped) or
                re.match(r'return\s+np\.zeros\s*\(', line_stripped) or
                re.match(r'return\s+\[\]', line_stripped) or
                re.match(r'return\s+""', line_stripped)):
                
                # 查看上下文，确认是否是在异常处理或数据缺失情况下的返回
                context_start = max(0, i-10)
                context_end = min(len(lines), i+2)
                context = '\n'.join(lines[context_start:context_end])
                
                # 排除main函数的退出码和正常的程序退出
                if ('def main(' in context or 'exit(' in context or 
                    'sys.exit(' in context or '成功完成' in context or
                    '中断信号' in context or '回测完成' in context):
                    continue
                    
                if ('except' in context or 'if.*empty' in context or 
                    '找到数据' in context or '缺失' in context or '默认' in context):
                    violations.append((i, line.strip()))
        
    except Exception as e:
        print(f"检查文件 {file_path} 时出错: {e}")
    
    return violations

def main():
    """主检查函数"""
    print("🔍 开始代码质量检查...")
    
    project_root = "/Users/shuzhenyi/code/python/finance"
    python_files = find_python_files(project_root)
    
    total_violations = 0
    
    print(f"\n📁 检查 {len(python_files)} 个Python文件")
    
    for file_path in python_files:
        relative_path = os.path.relpath(file_path, project_root)
        
        # 检查异常处理违规
        exception_violations = check_broad_exception_handling(file_path)
        if exception_violations:
            print(f"\n❌ {relative_path}")
            print("  违规的异常处理:")
            for line_no, line in exception_violations:
                print(f"    第{line_no}行: {line}")
            total_violations += len(exception_violations)
        
        # 检查硬编码数据生成违规
        data_violations = check_hardcoded_data_generation(file_path)
        if data_violations:
            if not exception_violations:  # 避免重复打印文件名
                print(f"\n❌ {relative_path}")
            print("  硬编码数据生成:")
            for line_no, line in data_violations:
                print(f"    第{line_no}行: {line}")
            total_violations += len(data_violations)
    
    print(f"\n📊 检查完成!")
    if total_violations == 0:
        print("✅ 未发现违规问题")
    else:
        print(f"❌ 发现 {total_violations} 处违规问题")
        print("\n修复建议:")
        print("1. 将 except: 或 except Exception: 改为具体的异常类型")
        print("2. 异常处理后必须重新抛出或正确处理，不能简单pass")
        print("3. 数据缺失时应抛出 RuntimeError，不能返回硬编码默认值")
    
    return total_violations == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)