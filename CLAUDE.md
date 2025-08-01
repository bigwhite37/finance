# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于强化学习和Transformer的A股量化交易系统，使用SAC（Soft Actor-Critic）算法作为决策引擎。系统采用分层架构：数据处理层、特征工程层、时序编码层（Transformer）、强化学习决策层、执行与监控层。

## 核心架构

- **数据层**: `src/rl_trading_system/data/` - 支持Qlib和Akshare数据源，包含数据处理、特征工程、数据质量控制
- **模型层**: `src/rl_trading_system/models/` - Transformer编码器和SAC智能体，含Actor/Critic网络
- **交易环境**: `src/rl_trading_system/trading/` - Portfolio环境、交易成本模型（Almgren-Chriss）、A股规则
- **训练**: `src/rl_trading_system/training/` - 强化学习训练器，支持数据分割策略
- **系统集成**: `src/rl_trading_system/system_integration.py` - 统一的系统入口，集成所有组件

## 开发命令

### 环境设置
```bash
make install-dev  # 安装开发依赖和pre-commit钩子
```

### 测试
```bash
make test         # 运行所有测试（单元、集成、e2e）
make test-unit    # 仅运行单元测试
make test-integration  # 仅运行集成测试
make test-e2e     # 仅运行端到端测试
```

### 代码质量
```bash
make lint         # flake8 + mypy检查
make format       # black + isort格式化
make type-check   # mypy类型检查
```

### 训练和评估
```bash
make train        # 训练模型（使用scripts/train.py）
make evaluate     # 评估模型性能
make deploy       # 部署模型
make monitor      # 启动监控服务
```

### 工具
```bash
make jupyter      # 启动Jupyter Lab
make tensorboard  # 启动TensorBoard
make check-env    # 检查环境依赖（PyTorch、CUDA、Qlib、Akshare）
```

## 配置文件

- `config/model_config.yaml` - 模型参数（Transformer、SAC）和训练配置
- `config/trading_config.yaml` - 交易相关配置
- `config/system_config.yaml` - 系统级配置
- `config/audit_config.yaml` - 审计日志配置

所有配置支持环境变量覆盖，例如 `MODEL_TRANSFORMER_D_MODEL=512`。

## 关键约定

- 使用PyTorch作为深度学习框架
- 遵循OpenAI Gym规范的环境接口
- 支持A股交易规则（T+1、涨跌停限制）
- 严格的类型检查（mypy）和代码格式化（black）
- 测试覆盖率要求≥80%

## 数据依赖

项目依赖外部数据源：
- **Qlib**: Microsoft量化投资平台，需要初始化：`python -c "import qlib; qlib.init()"`
- **Akshare**: 中文财经数据接口，用于实时数据获取

## 测试标记

- `@pytest.mark.unit` - 单元测试
- `@pytest.mark.integration` - 集成测试  
- `@pytest.mark.e2e` - 端到端测试
- `@pytest.mark.slow` - 慢速测试
- `@pytest.mark.gpu` - 需要GPU的测试

运行特定标记的测试：`pytest -m unit`