#!/usr/bin/env python3
"""
项目结构优化脚本 - 进一步整理和优化代码组织
"""

import os
import shutil
from pathlib import Path

def organize_project():
    """优化项目结构"""
    
    print("🔧 开始优化项目结构...")
    
    # 1. 清理日志文件 - 只保留最新的几个
    print("\n📝 清理日志文件...")
    logs_dir = Path('logs')
    if logs_dir.exists():
        log_files = sorted(logs_dir.glob('*.log'), key=os.path.getmtime)
        if len(log_files) > 5:  # 只保留最新的5个日志文件
            for old_log in log_files[:-5]:
                old_log.unlink()
                print(f"🗑️ 删除旧日志: {old_log.name}")
    
    # 2. 清理结果文件 - 只保留最新的几个
    print("\n📊 清理结果文件...")
    results_dir = Path('results')
    if results_dir.exists():
        result_files = sorted(results_dir.glob('*.pkl'), key=os.path.getmtime)
        if len(result_files) > 3:  # 只保留最新的3个结果文件
            for old_result in result_files[:-3]:
                old_result.unlink()
                print(f"🗑️ 删除旧结果: {old_result.name}")
    
    # 3. 清理模型文件 - 只保留最佳的几个
    print("\n🤖 清理模型文件...")
    models_dir = Path('models')
    if models_dir.exists():
        model_files = list(models_dir.glob('*.pth'))
        if len(model_files) > 2:  # 只保留最新的2个模型
            model_files.sort(key=os.path.getmtime)
            for old_model in model_files[:-2]:
                old_model.unlink()
                print(f"🗑️ 删除旧模型: {old_model.name}")
    
    # 4. 创建更好的文档结构
    print("\n📚 优化文档结构...")
    docs_structure = {
        'docs/api': '存放API文档',
        'docs/tutorials': '存放教程文档', 
        'docs/design': '存放设计文档',
        'docs/research': '存放研究笔记'
    }
    
    for doc_dir, description in docs_structure.items():
        Path(doc_dir).mkdir(parents=True, exist_ok=True)
        readme_path = Path(doc_dir) / 'README.md'
        if not readme_path.exists():
            readme_path.write_text(f"# {doc_dir.split('/')[-1].title()}\n\n{description}\n")
        print(f"📁 创建文档目录: {doc_dir}")
    
    # 5. 移动现有文档到合适位置
    if Path('docs/dynamic_lowvol_filter_usage.md').exists():
        shutil.move('docs/dynamic_lowvol_filter_usage.md', 'docs/tutorials/')
        print("📄 移动使用文档到tutorials目录")
    
    # 6. 创建开发工具目录
    print("\n🛠️ 创建开发工具目录...")
    dev_dirs = {
        'tools': '开发工具脚本',
        'notebooks': 'Jupyter notebooks',
        'experiments': '实验代码'
    }
    
    for dev_dir, description in dev_dirs.items():
        Path(dev_dir).mkdir(exist_ok=True)
        readme_path = Path(dev_dir) / 'README.md'
        if not readme_path.exists():
            readme_path.write_text(f"# {dev_dir.title()}\n\n{description}\n")
        print(f"🔧 创建开发目录: {dev_dir}")
    
    # 7. 移动清理脚本到tools目录
    if Path('cleanup_project.py').exists():
        shutil.move('cleanup_project.py', 'tools/')
        print("🔧 移动清理脚本到tools目录")
    
    # 8. 创建配置文件模板
    print("\n⚙️ 创建配置模板...")
    config_templates_dir = Path('config/templates')
    config_templates_dir.mkdir(exist_ok=True)
    
    # 创建示例配置文件
    example_config = """# 示例配置文件
# 复制此文件并根据需要修改参数

data:
  start_date: '2020-01-01'
  end_date: '2023-12-31'
  universe: 'csi300'

agent:
  learning_rate: 3e-4
  hidden_dim: 64

risk_control:
  target_volatility: 0.12
  max_leverage: 1.2
"""
    
    (config_templates_dir / 'example_config.yaml').write_text(example_config)
    print("📝 创建配置模板文件")
    
    # 9. 创建项目根目录的Makefile
    print("\n🔨 创建Makefile...")
    makefile_content = """# A股强化学习量化交易系统 Makefile

.PHONY: help install test train backtest clean lint format

help:  ## 显示帮助信息
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \\033[36m%-15s\\033[0m %s\\n", $$1, $$2}'

install:  ## 安装依赖
	pip install -r requirements.txt

test:  ## 运行测试
	python -m pytest tests/ -v

train:  ## 训练模型
	python main.py --mode train

backtest:  ## 运行回测
	python main.py --mode backtest

full:  ## 完整流程（训练+回测）
	python main.py --mode full

clean:  ## 清理临时文件
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/

lint:  ## 代码检查
	flake8 --max-line-length=100 --ignore=E203,W503 .

format:  ## 代码格式化
	black --line-length=100 .
	isort .

quick-test:  ## 快速测试
	python temp/final_training_test.py

logs:  ## 查看最新日志
	tail -f logs/trading_system_*.log | head -100
"""
    
    Path('Makefile').write_text(makefile_content)
    print("🔨 创建Makefile")
    
    # 10. 更新.gitignore
    print("\n🙈 更新.gitignore...")
    gitignore_additions = """
# 临时文件
temp/
*.tmp
*.temp

# 日志文件（保留目录但忽略内容）
logs/*.log
!logs/.gitkeep

# 模型文件（太大，不提交）
models/*.pth
!models/.gitkeep

# 结果文件
results/*.pkl
!results/.gitkeep

# 缓存文件
cache/
*.cache

# IDE文件
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'a') as f:
        f.write(gitignore_additions)
    print("📝 更新.gitignore文件")
    
    # 11. 创建.gitkeep文件保持空目录
    keep_dirs = ['logs', 'models', 'results', 'cache', 'notebooks', 'experiments']
    for keep_dir in keep_dirs:
        keep_file = Path(keep_dir) / '.gitkeep'
        if Path(keep_dir).exists():
            keep_file.touch()
            print(f"📌 创建.gitkeep: {keep_file}")
    
    print("\n🎉 项目结构优化完成！")
    
    # 显示优化后的结构
    print("\n📁 优化后的项目结构:")
    print_project_summary()

def print_project_summary():
    """打印项目结构摘要"""
    structure = {
        "核心模块": ["data/", "factors/", "rl_agent/", "risk_control/", "backtest/"],
        "配置管理": ["config/", "config/templates/"],
        "测试代码": ["tests/"],
        "文档": ["docs/api/", "docs/tutorials/", "docs/design/", "docs/research/"],
        "开发工具": ["tools/", "scripts/", "notebooks/", "experiments/"],
        "示例代码": ["examples/"],
        "临时文件": ["temp/", "archive/"],
        "输出文件": ["logs/", "models/", "results/", "reports/"],
        "工具文件": ["utils/"]
    }
    
    for category, dirs in structure.items():
        print(f"\n{category}:")
        for dir_name in dirs:
            if Path(dir_name).exists():
                print(f"  ✅ {dir_name}")
            else:
                print(f"  ❌ {dir_name} (不存在)")

if __name__ == "__main__":
    organize_project()