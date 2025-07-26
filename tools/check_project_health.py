#!/usr/bin/env python3
"""
项目健康检查脚本 - 检查项目结构和文件状态
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def check_project_health():
    """检查项目健康状态"""
    
    print("🏥 项目健康检查")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # 1. 检查核心模块
    print("\n🧠 检查核心模块...")
    core_modules = {
        'data': ['data_manager.py', 'qlib_provider.py'],
        'factors': ['factor_engine.py', 'alpha_factors.py', 'risk_factors.py'],
        'rl_agent': ['trading_environment.py', 'cvar_ppo_agent.py', 'safety_shield.py'],
        'risk_control': ['risk_controller.py', 'target_volatility.py'],
        'backtest': ['backtest_engine.py', 'performance_analyzer.py'],
        'config': ['config_manager.py', 'default_config.py'],
        'utils': ['logger.py', 'metrics.py']
    }
    
    for module, files in core_modules.items():
        module_path = Path(module)
        if not module_path.exists():
            issues.append(f"缺少核心模块: {module}/")
            continue
        
        print(f"  ✅ {module}/")
        for file in files:
            file_path = module_path / file
            if not file_path.exists():
                warnings.append(f"缺少文件: {module}/{file}")
            else:
                # 检查文件大小
                size = file_path.stat().st_size
                if size == 0:
                    warnings.append(f"空文件: {module}/{file}")
    
    # 2. 检查主要文件
    print("\n📋 检查主要文件...")
    main_files = ['main.py', 'README.md', 'requirements.txt', 'Makefile']
    for file in main_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            issues.append(f"缺少主要文件: {file}")
    
    # 3. 检查目录结构
    print("\n📁 检查目录结构...")
    required_dirs = [
        'logs', 'models', 'results', 'tests', 'examples',
        'docs', 'tools', 'scripts', 'temp', 'cache'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
            # 检查是否有.gitkeep文件
            if dir_name in ['logs', 'models', 'results', 'cache']:
                gitkeep = dir_path / '.gitkeep'
                if not gitkeep.exists():
                    warnings.append(f"缺少.gitkeep: {dir_name}/")
        else:
            warnings.append(f"缺少目录: {dir_name}/")
    
    # 4. 检查配置文件
    print("\n⚙️ 检查配置文件...")
    config_files = [
        'config/templates/example_config.yaml',
        '.gitignore'
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"  ✅ {config_file}")
        else:
            warnings.append(f"缺少配置文件: {config_file}")
    
    # 5. 检查文件大小和数量
    print("\n📊 检查文件统计...")
    
    # 检查日志文件数量
    logs_dir = Path('logs')
    if logs_dir.exists():
        log_files = list(logs_dir.glob('*.log'))
        print(f"  📝 日志文件: {len(log_files)} 个")
        if len(log_files) > 10:
            warnings.append(f"日志文件过多: {len(log_files)} 个，建议清理")
    
    # 检查模型文件数量
    models_dir = Path('models')
    if models_dir.exists():
        model_files = list(models_dir.glob('*.pth'))
        print(f"  🤖 模型文件: {len(model_files)} 个")
        if len(model_files) > 5:
            warnings.append(f"模型文件过多: {len(model_files)} 个，建议清理")
    
    # 检查结果文件数量
    results_dir = Path('results')
    if results_dir.exists():
        result_files = list(results_dir.glob('*.pkl'))
        print(f"  📊 结果文件: {len(result_files)} 个")
        if len(result_files) > 5:
            warnings.append(f"结果文件过多: {len(result_files)} 个，建议清理")
    
    # 6. 检查Python语法
    print("\n🐍 检查Python语法...")
    python_files = []
    for root, dirs, files in os.walk('.'):
        # 跳过某些目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    syntax_errors = 0
    for py_file in python_files[:10]:  # 只检查前10个文件，避免太慢
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), py_file, 'exec')
        except (SyntaxError, UnicodeDecodeError, IOError) as e:
            issues.append(f"语法或读取错误: {py_file}:{getattr(e, 'lineno', 'N/A')}")
            syntax_errors += 1
    
    if syntax_errors == 0:
        print(f"  ✅ 检查了 {min(len(python_files), 10)} 个Python文件，无语法错误")
    
    # 7. 生成报告
    print("\n" + "=" * 50)
    print("📋 健康检查报告")
    print("=" * 50)
    
    if not issues and not warnings:
        print("🎉 项目状态良好！所有检查都通过了。")
    else:
        if issues:
            print(f"❌ 发现 {len(issues)} 个严重问题:")
            for issue in issues:
                print(f"  • {issue}")
        
        if warnings:
            print(f"\n⚠️ 发现 {len(warnings)} 个警告:")
            for warning in warnings:
                print(f"  • {warning}")
    
    # 8. 提供建议
    print("\n💡 建议:")
    if len(warnings) > 0 or len(issues) > 0:
        print("  • 运行 'python tools/organize_project.py' 来修复一些问题")
        print("  • 运行 'make clean' 来清理临时文件")
        print("  • 检查 .gitignore 文件是否正确配置")
    
    print("  • 定期运行此脚本检查项目健康状态")
    print("  • 使用 'make test' 运行测试确保代码质量")
    print("  • 查看 PROJECT_STRUCTURE.md 了解项目结构")
    
    print(f"\n📅 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = check_project_health()
    sys.exit(0 if success else 1)