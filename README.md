# A股强化学习量化交易系统

基于CVaR-PPO的A股量化交易智能体，专注于低波动、高夏普比率的稳健收益策略。

## 核心特性

### 🎯 投资目标
- **年化收益率**: ≥ 5%
- **最大回撤**: ≤ 10%  
- **年化波动率**: 10-12%
- **夏普比率**: > 0.5
- **心理舒适度**: 高度优化

### 🧠 AI驱动
- **CVaR-PPO算法**: 条件风险价值约束的策略梯度
- **安全保护层**: Shielded-RL实时风险控制
- **因子驱动**: 多维度Alpha因子 + 风险因子

### 🛡️ 风险管理
- **目标波动率控制**: 动态杠杆调整
- **风险平价优化**: 多种权重分配策略
- **动态止损**: 移动止损 + 最大回撤保护
- **实时监控**: VaR/CVaR/回撤全方位监控

## 系统架构

```
finance/
├── data/              # 数据管理模块
│   ├── data_manager.py
│   └── qlib_provider.py
├── factors/           # 因子工程模块
│   ├── factor_engine.py
│   ├── alpha_factors.py
│   └── risk_factors.py
├── rl_agent/          # 强化学习模块
│   ├── trading_environment.py
│   ├── cvar_ppo_agent.py
│   └── safety_shield.py
├── risk_control/      # 风险控制模块
│   ├── risk_controller.py
│   ├── target_volatility.py
│   ├── risk_parity.py
│   └── stop_loss.py
├── backtest/          # 回测评估模块
│   ├── backtest_engine.py
│   ├── performance_analyzer.py
│   └── comfort_metrics.py
├── config/            # 配置管理模块
├── utils/             # 工具模块
├── examples/          # 示例代码
└── main.py           # 主程序
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化qlib数据
# 按照qlib官方文档配置中国A股数据
```

### 2. 运行演示

```bash
# 快速演示
python examples/quick_start.py

# 查看系统功能
python main.py --help
```

### 3. 完整流程

```bash
# 训练智能体
python main.py --mode train

# 运行回测
python main.py --mode backtest --model ./models/best_agent.pth

# 完整流程（训练+回测）
python main.py --mode full
```

## 配置说明

### 默认配置

系统提供完整的默认配置，覆盖所有模块参数：

```python
from config import ConfigManager

config = ConfigManager()
config.print_config_summary()
```

### 自定义配置

创建YAML或JSON配置文件：

```yaml
# config.yaml
data:
  start_date: '2020-01-01'
  end_date: '2023-12-31'
  universe: 'csi300'

agent:
  learning_rate: 3e-4
  cvar_threshold: -0.02

risk_control:
  target_volatility: 0.12
  max_leverage: 1.2
```

使用自定义配置：
```bash
python main.py --config config.yaml --mode full
```

## 核心模块详解

### 1. 数据管理 (data/)
- 基于qlib的统一数据接口
- 支持多种数据源（Yahoo Finance, 本地数据等）
- 自动数据清洗和预处理

### 2. 因子工程 (factors/)
- **Alpha因子**: 动量、反转、技术指标等
- **风险因子**: 波动率、贝塔、VaR等  
- **低波动筛选**: 自动筛选低波动股票池

### 3. 强化学习 (rl_agent/)
- **CVaR-PPO**: 集成条件风险价值约束
- **交易环境**: 考虑交易成本、滑点的真实环境
- **安全保护**: 多层次风险约束和动作修正

### 4. 风险控制 (risk_control/)
- **目标波动率**: 动态调整组合杠杆
- **风险平价**: 多种权重优化策略
- **动态止损**: 智能止损和再平衡

### 5. 回测评估 (backtest/)
- **绩效分析**: 全面的风险收益指标
- **心理舒适度**: 量化投资心理感受
- **对比分析**: 与基准的详细对比

## 心理舒适度指标

系统专门设计了心理舒适度评估体系：

- **月度最大回撤** < 5%
- **连续亏损天数** < 5天
- **下跌日占比** < 40%
- **95% VaR** < 1%
- **综合舒适度得分** 0-100分

## 性能预期

根据架构设计，系统预期实现：

| 指标 | 目标值 | 备注 |
|------|--------|------|
| 年化收益率 | 6-9% | 基于历史回测 |
| 最大回撤 | ≤10% | 多层风险控制 |
| 年化波动率 | ≈11% | 目标波动率管理 |
| 夏普比率 | 0.6-0.8 | 风险调整后收益 |
| 月度最大回撤 | <5% | 心理舒适度保障 |

## 扩展指南

### 添加新因子

```python
# factors/alpha_factors.py
def calculate_my_factor(self, price_data, volume_data):
    # 实现您的因子逻辑
    return factor_values
```

### 自定义风险控制

```python
# risk_control/custom_control.py
class CustomRiskControl:
    def __init__(self, config):
        pass
    
    def process_weights(self, weights):
        # 实现自定义风险控制逻辑
        return adjusted_weights
```

### 集成新数据源

```python
# data/custom_provider.py
class CustomDataProvider:
    def get_data(self, symbols, start_date, end_date):
        # 实现数据获取逻辑
        return data
```

## 注意事项

1. **数据质量**: 确保qlib数据源配置正确
2. **计算资源**: 训练需要一定的计算资源，建议使用GPU
3. **风险提示**: 本系统仅供学习研究使用，实盘交易需谨慎
4. **回测vs实盘**: 回测结果不代表实盘表现，存在过拟合风险

## 技术特点

- **模块化设计**: 各模块独立，易于扩展和维护
- **配置驱动**: 丰富的配置选项，支持多种运行模式  
- **异常安全**: 去除try-except，直接暴露异常便于调试
- **代码约束**: 单文件≤500行，保持代码清晰
- **中文优化**: 全面支持中文数据和输出

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

---

**免责声明**: 本项目仅供学术研究和学习使用，不构成投资建议。使用者需自行承担投资风险。