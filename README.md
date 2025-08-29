# A股量化交易策略系统

## 概述

这是一个基于Python的A股量化交易策略系统，专注于风险敏感型趋势跟踪和多因子分析。系统集成了强化学习优化、实时风险管理、多因子模型和实盘交易信号生成功能。

## 核心特性

### 🎯 多因子量化策略
- **动量因子**: 多窗口期动量计算（63日、126日、252日）
- **波动率因子**: 结合总波动率和下行波动率的复合风险评估
- **趋势强度因子**: 基于线性回归的趋势强度量化
- **流动性因子**: ADV、Amihud非流动性、换手率等综合评估
- **量价背离因子**: OBV与价格走势的背离分析
- **下行风险因子**: Sortino比率和CVaR的风险评分

### 🛡️ 风险管理系统
- **动态回撤控制**: 基于指数回撤状态的仓位缩放
- **ATR止损**: 基于平均真实波动范围的止损机制
- **相关性过滤**: 避免高相关性股票的集中持仓风险
- **流动性筛选**: ADTV（平均日成交量）限制和停牌股票过滤
- **涨跌停风险**: A股特有的价格限制风险管理

### 🤖 强化学习优化
- **PPO算法**: 使用Proximal Policy Optimization进行参数优化
- **超参数调优**: 集成Optuna进行自动化超参数搜索
- **滚动窗口训练**: 时间序列交叉验证确保模型稳健性
- **性能监控**: 实时追踪训练指标和模型收敛状态

### 📊 实时分析仪表板
- **净值曲线**: 策略收益vs基准指数对比
- **风险指标**: 最大回撤、夏普比率、波动率等关键指标
- **行业分布**: 持仓的行业配置分析
- **交易统计**: 胜率、平均持仓期、换手率等交易表现

## 系统架构

```
claude.py                    # 核心策略引擎
├── RiskSensitiveTrendStrategy   # 主策略类
├── DailyTradingPlan            # 交易计划生成器
├── StrategyAnalytics           # 策略分析器
└── 多因子计算模块               # 独立因子计算函数

rl_config_optimized.yaml     # 系统配置文件
├── claude                   # 策略基础配置
├── backtest                 # 回测时间设置
├── factor_calculation       # 因子计算参数
├── risk_management          # 风险管理配置
├── stock_selection          # 选股策略配置
├── training                 # 强化学习训练配置
└── transaction_cost         # 交易成本设置
```

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install qlib pandas numpy akshare plotly pyyaml optuna stable-baselines3

# 初始化Qlib数据
python -c "import qlib; qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')"
```

### 基础用法

```python
from claude import RiskSensitiveTrendStrategy

# 1. 初始化策略
strategy = RiskSensitiveTrendStrategy(
    start_date='20250101',
    end_date='20250425',
    config_path='rl_config_optimized.yaml'
)

# 2. 运行策略分析
selected_stocks, position_sizes = strategy.run_strategy()

# 3. 生成交易计划
from claude import DailyTradingPlan
trading_plan = DailyTradingPlan(strategy)
plan = trading_plan.generate_complete_daily_plan(capital=1000000)
```

### 命令行使用

```bash
# 运行策略分析
python claude.py --mode analysis --config rl_config_optimized.yaml

# 执行回测
python claude.py --mode backtest --start-date 20250101 --end-date 20250425

# 生成交易信号
python claude.py --mode live --config rl_config_optimized.yaml
```

## 配置说明

### 核心配置项

```yaml
# 回测时间配置
backtest:
  start_date: '2025-01-01'    # 策略运行开始时间
  end_date: '2025-04-25'      # 策略运行结束时间

# 数据加载配置
data_loading:
  preload_days: 410           # 因子计算预加载天数

# 选股策略配置
stock_selection:
  max_stocks: 1200           # 打分股票池大小
  rank_top_k: 200            # 排序后保留数量
  min_adtv_shares: 1000000   # 最小平均日成交量(股)
  filter_st: true            # 是否过滤ST股票

# 风险管理配置
risk_management:
  max_drawdown: -0.2         # 最大回撤限制
  max_turnover: 0.4          # 最大换手率限制
  stop_loss_pct: -0.10       # 止损比例
  take_profit_pct: 0.3       # 止盈比例
```

### 因子权重配置

```yaml
# 在claude配置中设置因子权重
claude:
  factor_weights:
    momentum: 1.0             # 动量因子权重
    volatility: 0.4           # 波动率因子权重
    trend_strength: 0.3       # 趋势强度权重
    liquidity: 0.5            # 流动性因子权重
    downside_risk: 0.6        # 下行风险权重
    volume_price_divergence: 0.2  # 量价背离权重
```

## 主要功能模块

### 1. 多因子分析

系统实现了6个主要因子：

- **动量因子**: 基于多时间窗口的价格动量，使用对数收益率计算
- **波动率因子**: 结合历史波动率和下行波动率，支持动态MAR计算
- **趋势强度因子**: 通过线性回归评估价格趋势的强度和方向
- **流动性因子**: 综合ADV、成交量稳定性、Amihud非流动性指标
- **量价背离因子**: 基于OBV和价格相关性的技术分析
- **下行风险因子**: Sortino比率结合CVaR的风险调整评分

### 2. 风险控制

- **实时回撤监控**: 基于基准指数的市场风险状态判断
- **个股风险过滤**: ATR止损、波动率阈值、相关性控制
- **流动性管理**: ADTV限制、停牌检测、最小交易单位约束
- **交易成本控制**: 包含佣金、印花税、过户费、滑点的精确计算

### 3. 信号生成

```python
# 生成投资信号
signals_path = trading_plan.export_invest_signals(
    capital=1000000,
    max_positions=30,
    selected_stocks=selected_stocks
)

# 输出格式（Parquet文件）
{
    'code': '000001',           # 股票代码
    'target_weight': 0.0333,    # 目标权重
    'score': 85.2,              # 综合评分
    'risk_flags': {...},        # 风险标记
    'stop_loss': 12.50,         # 止损价位
    'take_profit': 16.80,       # 止盈价位
    'adtv_20d': 15000000,       # 20日平均成交量
    'board': 'Main'             # 交易板块
}
```

### 4. 性能分析

系统提供详细的策略表现分析：

- **收益指标**: 总收益率、年化收益率、超额收益
- **风险指标**: 最大回撤、波动率、下行偏差
- **风险调整收益**: 夏普比率、Sortino比率、Calmar比率
- **交易统计**: 胜率、平均持仓期、换手率

## 强化学习集成

### 训练配置

```yaml
training:
  algo: PPO                   # 算法类型
  training_steps: 150000      # 训练步数
  learning_rate: 0.0001       # 学习率
  enable_hyperopt: true       # 启用超参数优化

hyperopt_config:
  n_startup_trials: 15        # 预热试验数
  study_name: rl_trading_optimized_{timestamp}
```

### 环境配置

```yaml
environment:
  lookback_days: 10           # 观测窗口长度
  beta: 0.4                   # 风险偏好系数
  reward_scaling: 1           # 奖励缩放系数
  temperature: 1.2            # 探索温度参数
```

## 数据依赖

### 必需数据源

1. **Qlib数据**: 中国A股日线数据（价格、成交量）
2. **AkShare**: 实时数据补充和股票基础信息
3. **本地缓存**: stocks_akshare.json（股票信息缓存）

### 数据更新

```python
# 更新股票基础信息
python dump_st.py

# 更新因子数据
strategy.update_factor_data()
```

## 输出文件

系统会生成以下输出文件：

```
data/
├── signals/
│   └── 2025-04-25.parquet     # 投资信号文件
├── portfolio/
│   └── portfolio_2025-04-25.csv  # 组合持仓文件
└── logs/
    └── trading_audit_20250425.log  # 交易审计日志

# 可视化输出
portfolio_curve.html           # 净值曲线图
portfolio_analysis_enhanced.html  # 增强分析报告
risk_dashboard.html           # 风险仪表板
```

## 性能优化

### 多进程计算

```python
# CPU配置
cpu_config:
  max_cpu_cores: 5            # 最大CPU核心数
  data_fetch_max_workers: 4   # 数据获取线程数
  eval_max_workers: 4         # 评估线程数
```

### 缓存机制

- 价格数据缓存
- 因子计算结果缓存
- 股票信息本地缓存
- Qlib数据集缓存

## 注意事项

### A股特色处理

1. **T+1交易制度**: 买入后次日才能卖出
2. **涨跌停限制**: 不同板块的价格限制（10%/5%/30%）
3. **最小交易单位**: 100股整手交易
4. **ST股票处理**: 风险警示股票的特殊处理

### 风险提示

1. **历史数据局限**: 策略基于历史数据，不能保证未来表现
2. **市场风险**: 策略无法消除系统性市场风险
3. **流动性风险**: 小盘股或停牌股票可能面临流动性问题
4. **模型风险**: 量化模型可能在极端市场环境下失效

## 开发计划

- [ ] 增加更多技术指标因子
- [ ] 实现高频数据支持
- [ ] 添加期货和期权策略
- [ ] 优化强化学习算法
- [ ] 开发实时交易接口
- [ ] 增强风险预警系统

## 许可证

本项目仅供学习和研究使用，不构成投资建议。使用者需自行承担投资风险。

## 技术支持

如有技术问题或改进建议，请通过GitHub Issues提交。

---

*最后更新: 2025年8月27日*