# 强化学习量化交易系统 Docker 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
COPY setup.py .
COPY README.md .

# 安装项目
RUN pip install -e .

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/checkpoints /app/outputs

# 设置权限
RUN chmod +x scripts/*.py

# 暴露端口
EXPOSE 8000 8888 6006

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认命令
CMD ["python", "scripts/monitor.py"]