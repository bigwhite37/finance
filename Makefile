# 强化学习量化交易系统 Makefile

.PHONY: help install install-dev test test-unit test-integration test-e2e lint format type-check clean docs serve-docs build docker-build docker-run

# 默认目标
help:
	@echo "可用的命令："
	@echo "  install        - 安装项目依赖"
	@echo "  install-dev    - 安装开发依赖"
	@echo "  test           - 运行所有测试"
	@echo "  test-unit      - 运行单元测试"
	@echo "  test-integration - 运行集成测试"
	@echo "  test-e2e       - 运行端到端测试"
	@echo "  lint           - 运行代码检查"
	@echo "  format         - 格式化代码"
	@echo "  type-check     - 运行类型检查"
	@echo "  clean          - 清理临时文件"
	@echo "  docs           - 生成文档"
	@echo "  serve-docs     - 启动文档服务器"
	@echo "  build          - 构建项目"
	@echo "  docker-build   - 构建Docker镜像"
	@echo "  docker-run     - 运行Docker容器"

# 安装依赖
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# 测试
test:
	pytest tests/ --cov=src/rl_trading_system --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

# 代码质量
lint:
	flake8 src/ tests/ scripts/
	mypy src/rl_trading_system/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

type-check:
	mypy src/rl_trading_system/

# 清理
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# 文档
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8080

# 构建
build:
	python setup.py sdist bdist_wheel

# Docker
docker-build:
	docker build -t rl-trading-system:latest .

docker-run:
	docker run -it --rm -p 8000:8000 rl-trading-system:latest

# 训练和评估
train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py --model-path ./checkpoints/best_model.pth

deploy:
	python scripts/deploy.py --model-path ./checkpoints/best_model.pth

monitor:
	python scripts/monitor.py

# 开发工具
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

tensorboard:
	tensorboard --logdir=./logs --port=6006

# 数据处理
download-data:
	python -c "import qlib; qlib.init(); from qlib.data import D; print('Qlib数据初始化完成')"

# 环境检查
check-env:
	python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
	python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
	python -c "import qlib; print('Qlib导入成功')"
	python -c "import akshare; print('Akshare导入成功')"