# A股强化学习量化交易系统 Makefile

.PHONY: help install test train backtest clean lint format

help:  ## 显示帮助信息
	@echo "可用命令:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

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
