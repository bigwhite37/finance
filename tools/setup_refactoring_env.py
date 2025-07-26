#!/usr/bin/env python3
"""
重构环境设置脚本
准备重构所需的环境和工具
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def setup_directories():
    """设置必要的目录结构"""
    print("📁 设置目录结构...")
    
    directories = [
        'reports',
        'backup',
        'temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")


def create_baseline_report():
    """创建基准行数报告"""
    print("📊 创建基准行数报告...")
    
    try:
        result = subprocess.run([
            sys.executable, 'tools/line_count_monitor.py',
            '--output', 'reports/baseline_line_count.md',
            '--priority'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ 基准报告创建成功")
            print(f"  📄 报告位置: reports/baseline_line_count.md")
        else:
            print(f"  ❌ 基准报告创建失败: {result.stderr}")
            
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"  ❌ 创建基准报告时出错: {str(e)}")


def run_initial_tests():
    """运行初始测试验证"""
    print("🧪 运行初始测试验证...")
    
    try:
        result = subprocess.run([
            sys.executable, 'tools/refactoring_test_validator.py',
            '--output', 'reports/initial_test_validation.md'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ 初始测试验证通过")
        else:
            print("  ⚠️  初始测试验证有问题，请检查")
            print(f"  📄 详细报告: reports/initial_test_validation.md")
            
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"  ❌ 运行初始测试时出错: {str(e)}")


def check_tools():
    """检查工具可用性"""
    print("🔧 检查工具可用性...")
    
    tools = [
        ('tools/line_count_monitor.py', '行数监控工具'),
        ('tools/refactoring_test_validator.py', '测试验证工具'),
        ('tests/run_tests.py', '测试运行器')
    ]
    
    all_available = True
    
    for tool_path, tool_name in tools:
        if Path(tool_path).exists():
            print(f"  ✅ {tool_name}")
        else:
            print(f"  ❌ {tool_name} - 文件不存在: {tool_path}")
            all_available = False
    
    return all_available


def make_tools_executable():
    """使工具脚本可执行"""
    print("⚙️  设置工具权限...")
    
    tools = [
        'tools/line_count_monitor.py',
        'tools/refactoring_test_validator.py',
        'tools/setup_refactoring_env.py'
    ]
    
    for tool in tools:
        if Path(tool).exists():
            try:
                os.chmod(tool, 0o755)
                print(f"  ✅ {tool}")
            except OSError as e:
                print(f"  ⚠️  {tool}: {str(e)}")


def create_refactoring_checklist():
    """创建重构检查清单"""
    print("📋 创建重构检查清单...")
    
    checklist_content = """# 重构检查清单

## 重构前准备
- [ ] 运行基准测试: `python tools/refactoring_test_validator.py`
- [ ] 生成基准行数报告: `python tools/line_count_monitor.py --output reports/baseline.md`
- [ ] 备份关键文件到 backup/ 目录

## 重构过程中
- [ ] 一次只重构一个文件
- [ ] 每次重构后运行快速验证: `python tools/refactoring_test_validator.py --module <module_path> --quick`
- [ ] 确保所有测试通过
- [ ] 检查导入语句正确性

## 重构后验证
- [ ] 运行完整测试套件: `python tests/run_tests.py`
- [ ] 验证API兼容性
- [ ] 检查文件行数: `python tools/line_count_monitor.py`
- [ ] 运行性能测试确保无回归

## 文档更新
- [ ] 更新模块文档
- [ ] 更新导入示例
- [ ] 更新 README.md
- [ ] 更新 PROJECT_STRUCTURE.md

## 最终检查
- [ ] 所有测试通过
- [ ] 文件行数符合要求
- [ ] API向后兼容
- [ ] 性能无回归
- [ ] 文档完整更新
"""
    
    checklist_path = Path('reports/refactoring_checklist.md')
    with open(checklist_path, 'w', encoding='utf-8') as f:
        f.write(checklist_content)
    
    print(f"  ✅ 检查清单已创建: {checklist_path}")


def main():
    """主函数"""
    print("🚀 设置重构环境...")
    print("=" * 50)
    
    # 检查当前目录
    if not Path('main.py').exists():
        print("❌ 请在项目根目录运行此脚本")
        sys.exit(1)
    
    # 设置目录
    setup_directories()
    
    # 检查工具
    if not check_tools():
        print("❌ 工具检查失败，请确保所有工具文件存在")
        sys.exit(1)
    
    # 设置权限
    make_tools_executable()
    
    # 创建基准报告
    create_baseline_report()
    
    # 运行初始测试
    run_initial_tests()
    
    # 创建检查清单
    create_refactoring_checklist()
    
    print("\n" + "=" * 50)
    print("✅ 重构环境设置完成！")
    print("\n📋 下一步:")
    print("1. 查看基准报告: reports/baseline_line_count.md")
    print("2. 查看测试验证: reports/initial_test_validation.md")
    print("3. 参考检查清单: reports/refactoring_checklist.md")
    print("4. 开始重构第一个文件")
    
    print("\n🔧 常用命令:")
    print("- 监控行数: python tools/line_count_monitor.py --priority")
    print("- 快速验证: python tools/refactoring_test_validator.py --module <path> --quick")
    print("- 完整验证: python tools/refactoring_test_validator.py")
    print("- 运行测试: python tests/run_tests.py")


if __name__ == '__main__':
    main()