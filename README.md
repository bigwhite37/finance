# 强化学习量化交易系统

基于强化学习与Transformer的A股量化交易智能体系统，采用SAC（Soft Actor-Critic）算法作为决策引擎，使用Transformer架构捕捉长期时序依赖。

## 🎯 项目目标

- **年化收益目标**: 8%-12%
- **最大回撤控制**: ≤15%
- **夏普比率**: ≥1.0
- **信息比率**: ≥0.5

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    强化学习量化交易智能体系统                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  数据处理层  │  │  特征工程层   │  │    时序编码层        │   │
│  │             │  │              │  │                      │   │
│  │ • Qlib     │─▶│ • 技术指标   │─▶│ • Transformer       │   │
│  │ • Akshare  │  │ • 基本面因子 │  │ • Multi-Head Attn   │   │
│  │ • 实时行情  │  │ • 市场微观   │  │ • Positional Enc    │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
│                                              │                  │
│  ┌──────────────────────────────────────────▼────────────────┐ │
│  │                    强化学习决策层                          │ │
│  │  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐  │ │
│  │  │Portfolio Env │  │Actor Network│  │Critic Network   │  │ │
│  │  │              │  │             │  │                 │  │ │
│  │  │State Space  │◀▶│Policy Head  │  │Value Head       │  │ │
│  │  │Action Space │  │(SAC)        │  │Q-Function       │  │ │
│  │  │Reward Func  │  │             │  │                 │  │ │
│  │  └──────────────┘  └─────────────┘  └─────────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                                │
│  ┌───────────────────────────▼──────────────────────────────┐ │
│  │                    执行与监控层                           │ │
│  │  • 交易成本模型  • 风险控制  • 实时监控  • 审计日志      │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装

```bash
# 克隆项目
git clone https://github.com/rl-trading/rl-trading-system.git
cd rl-trading-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 配置

1. 复制配置模板：
```bash
cp config/model_config.yaml.example config/model_config.yaml
cp config/trading_config.yaml.example config/trading_config.yaml
```

2. 修改配置文件中的参数

### 训练模型

```bash
# 使用默认配置训练
python scripts/train.py

# 指定配置文件和训练轮数
python scripts/train.py --config config/model_config.yaml --episodes 5000
```

### 回测评估

```bash
# 评估模型性能
python scripts/evaluate.py --model-path ./checkpoints/best_model.pth

# 指定回测时间段
python scripts/evaluate.py \
    --model-path ./checkpoints/best_model.pth \
    --start-date 2022-01-01 \
    --end-date 2023-12-31
```

### 部署模型

```bash
# 金丝雀部署（推荐）
python scripts/deploy.py --model-path ./checkpoints/best_model.pth --deployment-type canary

# 全量部署
python scripts/deploy.py --model-path ./checkpoints/best_model.pth --deployment-type full
```

### 启动监控

```bash
# 启动监控服务
python scripts/monitor.py

# 访问监控面板
# Prometheus: http://localhost:8000/metrics
# Grafana: http://localhost:3000 (需要单独配置)
```

## 📊 核心功能

### 数据处理
- **多数据源支持**: Qlib、Akshare、实时行情
- **特征工程**: 技术指标、基本面因子、市场微观结构
- **数据质量控制**: 异常检测、缺失值处理、数据验证

### 模型架构
- **Transformer编码器**: 捕捉长期时序依赖
- **SAC智能体**: 连续动作空间的强化学习
- **注意力机制**: 多头注意力和时间注意力聚合

### 交易环境
- **Portfolio Environment**: 符合OpenAI Gym规范
- **交易成本模型**: Almgren-Chriss市场冲击模型
- **A股规则**: T+1、涨跌停、交易时间限制

### 风险控制
- **持仓限制**: 单股最大持仓、行业暴露控制
- **止损机制**: 动态止损、最大回撤控制
- **实时监控**: 风险指标实时计算和告警

## 🧪 测试

```bash
# 运行所有测试
pytest

# 运行单元测试
pytest tests/unit

# 运行集成测试
pytest tests/integration

# 生成覆盖率报告
pytest --cov=src/rl_trading_system --cov-report=html
```

## 📈 性能指标

### 回测结果（示例）
- **年化收益率**: 10.5%
- **最大回撤**: 12.3%
- **夏普比率**: 1.25
- **信息比率**: 0.68
- **胜率**: 52.3%
- **平均持仓期**: 5.2天

### 系统性能
- **模型推理延迟**: <50ms
- **数据处理吞吐**: >1000 stocks/s
- **内存使用**: <4GB
- **CPU使用率**: <80%

## 🔧 开发指南

### 代码规范
- 使用 `black` 进行代码格式化
- 使用 `flake8` 进行代码检查
- 使用 `mypy` 进行类型检查
- 遵循 PEP 8 编码规范

### 测试驱动开发
1. 先编写测试用例
2. 运行测试（应该失败）
3. 编写最小实现代码
4. 运行测试（应该通过）
5. 重构和优化代码

### 提交规范
```bash
# 安装pre-commit钩子
pre-commit install

# 提交代码前会自动运行检查
git commit -m "feat: 添加新功能"
```

## 📚 文档

- [API文档](docs/api/) - 详细的API接口文档
- [用户指南](docs/user_guide/) - 用户使用手册
- [开发者指南](docs/developer_guide/) - 开发者文档
- [部署指南](docs/deployment/) - 系统部署文档

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## ⚠️ 免责声明

本系统仅供学习和研究使用，不构成投资建议。使用本系统进行实际交易的风险由用户自行承担。

## 📞 联系我们

- 项目主页: https://github.com/rl-trading/rl-trading-system
- 问题反馈: https://github.com/rl-trading/rl-trading-system/issues
- 邮箱: team@rltrading.com

## 🙏 致谢

感谢以下开源项目的支持：
- [Qlib](https://github.com/microsoft/qlib) - 量化投资平台
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [OpenAI Gym](https://gym.openai.com/) - 强化学习环境
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习算法库