# A股趋势跟踪策略 + 实盘交易引擎

基于Qlib的A股趋势跟踪与相对强度策略，现已升级为完整的**实盘信号&风控引擎**，支持每日交易计划生成。

## 🌟 核心特性

### 策略特点
- **趋势跟踪**: 多周期移动平均线信号
- **相对强度**: 风险调整后的动量评分
- **风险管理**: ATR止损、波动率过滤、最大回撤控制
- **A股适配**: T+1制度、涨跌停限制、ST股识别

### 实盘引擎
- **每日交易计划**: 自动生成4张表(买入/减仓/加仓/观察)
- **精确仓位管理**: 2%风险法 + ATR止损的科学sizing
- **流动性约束**: ADV20限制，避免流动性风险
- **风险标记**: 自动识别涨跌停、流动性等风险
- **可复现性**: 基于交易日期的固定随机种子

## 📦 依赖安装

```bash
# 核心依赖
pip install qlib akshare pandas numpy plotly

# Qlib数据初始化（首次使用）
python -c "import qlib; qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region='cn')"
```

## 🚀 快速开始

### 1. 每日交易模式（推荐）

```bash
# 生成今日交易计划
python claude.py --mode trading

# 指定资金和持仓数
python claude.py --mode trading --capital 1000000 --max-positions 5

# 指定特定交易日期（确保可复现）
python claude.py --mode trading --trade-date 20250812

# 包含当前持仓风控检查
python claude.py --mode trading --current-holdings holdings.json
```

### 2. 策略分析模式

```bash
# 基础策略回测分析
python claude.py --mode analysis

# 指定时间范围
python claude.py --mode analysis --start-date 20240101 --end-date 20241231

# 限制股票数量
python claude.py --mode analysis --max-stocks 100

# 使用指数成分股
python claude.py --mode analysis --pool-mode index --index-code 000300
```

## 📊 输出文件说明

### 交易计划文件

运行交易模式后生成：

- **`daily_trading_plan_YYYYMMDD.csv`** - 主交易计划
- **`watchlist_YYYYMMDD.csv`** - 观察清单

### 分析模式文件

运行分析模式后生成：

- **`risk_dashboard.html`** - 风险管理仪表板
- **`portfolio_curve.html`** - 组合净值曲线

## 📋 CSV文件格式

### 交易计划表字段

| 字段 | 说明 | 示例 |
|------|------|------|
| date | 交易日期 | 20250812 |
| code | 股票代码 | 000001 |
| name | 股票名称 | 平安银行 |
| signal | 信号来源 | RS_15.6 |
| plan_action | 操作类型 | buy/reduce/exit |
| plan_shares | 计划股数 | 1000 |
| plan_weight | 仓位权重(%) | 5.2 |
| entry_hint | 执行建议 | 开盘价 |
| stop_loss | 止损价 | 12.50 |
| atr | ATR值 | 0.85 |
| risk_used | 风险占用 | 20000 |
| notes | 风险提示 | 涨停风险;流动性风险 |

### 持仓文件格式 (holdings.json)

```json
{
    "000001": 1000,
    "600000": 500,
    "300750": 800
}
```

## ⚙️ 参数配置

### 完整命令行参数说明

#### 运行模式控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | 选择 | `analysis` | **运行模式选择**<br/>• `analysis`: 策略分析模式，进行回测和风险分析<br/>• `trading`: 每日交易引擎模式，生成实盘交易计划 |

#### 基础时间参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start-date`, `-s` | 字符串 | `20250101` | **开始日期**，格式YYYYMMDD<br/>• 分析模式：回测起始日期<br/>• 交易模式：数据获取的起始范围 |
| `--end-date`, `-e` | 字符串 | 今天 | **结束日期**，格式YYYYMMDD<br/>• 分析模式：回测截止日期<br/>• 交易模式：数据获取的截止日期 |
| `--qlib-dir` | 路径 | `~/.qlib/qlib_data/cn_data` | **Qlib数据目录路径**<br/>• 指定本地Qlib金融数据存放位置<br/>• 支持波浪号(~)展开用户目录 |

#### 交易引擎专用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--capital` | 浮点数 | `1000000` | **总资本金额**（仅交易模式有效）<br/>• 用于风险法仓位计算<br/>• 每笔交易风险 = capital × 2% |
| `--max-positions` | 整数 | `5` | **最大持仓数量**（仅交易模式有效）<br/>• 限制同时持有的股票数量<br/>• 用于风险分散和管理 |
| `--trade-date` | 字符串 | 今天 | **交易日期**，格式YYYYMMDD<br/>• 指定生成交易计划的目标日期<br/>• 用于固定随机种子确保可复现 |
| `--current-holdings` | 文件路径 | `None` | **当前持仓JSON文件**<br/>• 包含现有持仓的股票代码和数量<br/>• 用于风控检查和加仓/减仓计划 |

#### 股票池配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--pool-mode` | 选择 | `auto` | **股票池模式选择**<br/>• `auto`: 自动从Qlib获取所有可用A股<br/>• `index`: 使用指定指数的成分股<br/>• `custom`: 使用自定义股票列表 |
| `--index-code` | 字符串 | `000300` | **指数代码**（pool-mode=index时）<br/>• 沪深300: `000300`<br/>• 中证500: `000905`<br/>• 创业板指: `399006` |
| `--stocks` | 列表 | `None` | **自定义股票代码**（pool-mode=custom时）<br/>• 6位数字格式，空格分隔<br/>• 例如: `000001 600000 300750` |
| `--max-stocks` | 整数 | `200` | **股票池最大数量**（auto模式）<br/>• 限制候选股票数量，0表示不限制<br/>• 随机抽样确保多样性 |

#### 性能优化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--no-concurrent` | 开关 | `False` | **禁用并发处理**<br/>• 默认使用多线程加速数据获取<br/>• 调试或资源受限时可禁用 |
| `--max-workers` | 整数 | CPU核心数×75% | **最大并发线程数**<br/>• 控制同时运行的工作线程数量<br/>• 避免过度占用系统资源 |

#### 输出控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--no-dashboard` | 开关 | `False` | **禁用风险仪表板**<br/>• 不生成risk_dashboard.html文件<br/>• 适用于纯数据输出场景 |
| `--no-backtest` | 开关 | `False` | **禁用回测功能**<br/>• 不运行策略回测<br/>• 仅进行股票筛选和分析 |

### 策略核心参数

```python
# 风险管理参数
risk_per_trade = 0.02          # 每笔交易风险2%
atr_multiplier = 2.0           # ATR止损倍数
max_correlation = 0.7          # 最大相关性阈值
max_drawdown_threshold = 0.15  # 最大回撤阈值15%
volatility_threshold = 0.35    # 年化波动率阈值35%

# 流动性参数
min_adv_20d = 20_000_000      # 20日平均成交额阈值(2000万)
max_suspend_days_60d = 10     # 60日内最大停牌天数
max_position_pct = 0.05       # 单笔不超过ADV20的5%

# A股制度参数
t_plus_1 = True               # T+1交易制度
price_limit_pct = 0.10        # 主板涨跌停10%
st_limit_pct = 0.05           # ST股涨跌停5%
bj_limit_pct = 0.30          # 北交所涨跌停30%
transaction_cost = 0.003      # 双边交易成本0.3%
```

## 💡 高级用法示例

### 命令行参数组合使用

#### 📈 策略分析场景

```bash
# 标准回测分析 - 沪深300成分股，近一年数据
python claude.py --mode analysis --pool-mode index --index-code 000300 \
  --start-date 20240101 --end-date 20241231

# 高性能分析 - 禁用可视化，专注数据输出
python claude.py --mode analysis --max-stocks 500 --no-dashboard \
  --max-workers 8 --start-date 20230101

# 自定义股票池分析
python claude.py --mode analysis --pool-mode custom \
  --stocks 000001 600000 000002 600036 300750 --no-backtest

# 小规模快速测试
python claude.py --mode analysis --max-stocks 50 --no-concurrent \
  --start-date 20241201 --end-date 20241231
```

#### 💼 实盘交易场景

```bash
# 标准每日交易计划 - 100万资金，5只持仓
python claude.py --mode trading --capital 1000000 --max-positions 5

# 大资金多持仓策略 - 500万资金，10只分散持仓
python claude.py --mode trading --capital 5000000 --max-positions 10 \
  --max-stocks 300

# 小资金精选策略 - 20万资金，3只集中持仓
python claude.py --mode trading --capital 200000 --max-positions 3 \
  --max-stocks 100

# 指定交易日期（确保可复现）
python claude.py --mode trading --trade-date 20250812 \
  --capital 1000000 --max-positions 5

# 包含持仓检查的交易计划
python claude.py --mode trading --current-holdings holdings.json \
  --capital 1000000 --max-positions 8
```

#### 🎯 特定指数策略

```bash
# 沪深300指数增强策略
python claude.py --mode trading --pool-mode index --index-code 000300 \
  --capital 2000000 --max-positions 15

# 中证500量化选股
python claude.py --mode trading --pool-mode index --index-code 000905 \
  --capital 1000000 --max-positions 10

# 创业板指数策略（高风险）
python claude.py --mode trading --pool-mode index --index-code 399006 \
  --capital 500000 --max-positions 5
```

#### ⚡ 性能优化场景

```bash
# 高性能模式 - 最大化并发
python claude.py --mode analysis --max-workers 12 --max-stocks 1000

# 资源受限模式 - 顺序处理
python claude.py --mode analysis --no-concurrent --max-stocks 100

# 纯数据模式 - 无可视化输出
python claude.py --mode analysis --no-dashboard --no-backtest \
  --max-stocks 200
```

### 参数交互说明

#### 🔗 参数依赖关系

| 主参数 | 相关参数 | 说明 |
|--------|----------|------|
| `--mode trading` | `--capital`, `--max-positions`, `--trade-date` | 交易模式必需参数 |
| `--pool-mode index` | `--index-code` | 指数模式需要指定指数代码 |
| `--pool-mode custom` | `--stocks` | 自定义模式需要股票列表 |
| `--pool-mode auto` | `--max-stocks` | 自动模式可限制股票数量 |
| `--current-holdings` | `--mode trading` | 持仓文件仅在交易模式有效 |

#### ⚠️ 参数冲突与限制

```bash
# ❌ 错误：交易模式下使用分析参数
python claude.py --mode trading --no-backtest  # no-backtest对交易模式无效

# ❌ 错误：指数模式但未指定代码
python claude.py --pool-mode index  # 缺少 --index-code

# ❌ 错误：自定义模式但未提供股票
python claude.py --pool-mode custom  # 缺少 --stocks

# ✅ 正确：完整的自定义交易策略
python claude.py --mode trading --pool-mode custom \
  --stocks 000001 600000 300750 --capital 1000000 --max-positions 3
```

### 日期格式与时区说明

```bash
# 标准日期格式 YYYYMMDD
--start-date 20250101    # ✅ 正确
--start-date 2025-01-01  # ❌ 错误格式
--start-date 20250132    # ❌ 无效日期

# 自动日期处理
--end-date               # 默认今天
--trade-date             # 默认今天（交易模式）

# 时间范围建议
--start-date 20240101 --end-date 20241231  # 一年回测
--start-date 20240701    # 最近半年分析（推荐）
--start-date 20220101    # 长期回测（数据量大）
```

## 🔄 每日操作流程

### 1. 收盘后（15:00以后）- 生成明日计划

```bash
python claude.py --mode trading --trade-date $(date +%Y%m%d)
```

**输出**：
- 交易计划CSV文件
- 观察清单CSV文件
- 风险提示和仓位建议

### 2. 盘前（9:20-9:30）- 校验与准备

1. **核对价格**：确认前收盘价与涨跌停价格
2. **检查风险**：查看CSV中的风险标记（涨停风险/流动性风险）
3. **设置订单**：
   - 买入：建议开盘价或VWAP执行
   - 卖出：市价或限价单
   - 避免追涨停板

### 3. 盘中（9:30-15:00）- 执行与风控

- **按计划下单**：参考plan_shares数量
- **流动性控制**：单笔不超过ADV20的5%
- **风控监控**：
  - ATR止损触发：立即减仓
  - 波动率超限：考虑减仓
  - 趋势反转：关注退出
- **T+1约束**：当日买入次日才能卖出

### 4. 收盘后（15:00以后）- 复盘记录

- **记录成交**：实际价格、数量、滑点
- **更新持仓**：修改holdings.json文件
- **统计分析**：
  - 计划执行率
  - 不可成交率（涨停/流动性）
  - 滑点成本
- **准备明日**：生成新的交易计划

## 🛡️ 风险控制机制

### 仓位风险管理
- **2%风险法则**：每笔交易最大风险=总资本×2%
- **ATR止损**：动态止损价=入场价-ATR×2.0
- **相关性控制**：避免同风格股票集中持仓

### 流动性风险管理
- **ADV过滤**：20日平均成交额≥2000万
- **停牌过滤**：60日内停牌天数≤10天
- **下单限制**：单笔交易≤ADV20的5%

### A股制度风险
- **涨跌停检测**：自动标记接近限价的股票
- **ST股识别**：使用AkShare API精确识别
- **T+1约束**：权重次日生效，避免当日反手

## 📈 策略指标

### 技术指标
- **移动平均**：20日/60日均线系统
- **相对强弱**：14日RSI指标
- **波动率**：ATR真实波幅
- **布林带**：20日布林带通道

### 风险指标
- **夏普比率**：年化超额收益/波动率
- **最大回撤**：60日滚动最大回撤
- **下行偏差**：负收益标准差
- **风险评分**：综合风险评估(0-100)

## 🔍 常见问题

### 💻 命令行使用问题

#### Q: 参数太多记不住怎么办？
```bash
# 查看完整帮助
python claude.py --help

# 最简单的交易模式启动
python claude.py --mode trading

# 最简单的分析模式启动
python claude.py --mode analysis
```

#### Q: 如何处理参数报错？
```bash
# 常见错误1：模式拼写错误
python claude.py --mode trader  # ❌ 错误
python claude.py --mode trading # ✅ 正确

# 常见错误2：日期格式错误
python claude.py --start-date 2025-01-01  # ❌ 带连字符
python claude.py --start-date 20250101    # ✅ 纯数字

# 常见错误3：股票代码格式错误
python claude.py --stocks SH600000  # ❌ 带前缀
python claude.py --stocks 600000    # ✅ 6位数字

# 常见错误4：路径不存在
python claude.py --current-holdings not_exist.json  # 会给出警告但继续运行
```

#### Q: 如何调试参数配置？
```bash
# 使用小数据集快速验证
python claude.py --mode analysis --max-stocks 10 --start-date 20241201

# 禁用并发便于调试
python claude.py --mode analysis --no-concurrent --max-stocks 20

# 最小化输出专注数据
python claude.py --mode analysis --no-dashboard --no-backtest
```

### 🎯 策略配置问题

#### Q: 如何调整风险参数？
**方法1: 修改代码中的默认值**
```python
# 在claude.py中找到策略类，修改参数：
# 保守配置
risk_per_trade = 0.01          # 降低单笔风险至1%
atr_multiplier = 1.5           # 更紧的止损

# 激进配置
risk_per_trade = 0.03          # 提高单笔风险至3%
atr_multiplier = 2.5           # 更宽的止损
```

**方法2: 通过资金量间接调整**
```bash
# 小资金模拟低风险
python claude.py --mode trading --capital 500000

# 大资金但少持仓（降低个股权重）
python claude.py --mode trading --capital 2000000 --max-positions 15
```

#### Q: 如何处理停牌股票？
```python
# 程序自动过滤停牌股票，默认设置：
max_suspend_days_60d = 10     # 60日内停牌不超过10天

# 如需更严格过滤，修改策略类中的参数：
strategy.max_suspend_days_60d = 5  # 更严格：仅5天
strategy.min_adv_20d = 50_000_000   # 更高流动性要求
```

#### Q: 如何确保结果可复现？
```bash
# 核心：使用固定的交易日期参数
python claude.py --mode trading --trade-date 20250812

# 完整的可复现配置
python claude.py --mode trading \
  --trade-date 20250812 \
  --capital 1000000 \
  --max-positions 5 \
  --max-stocks 200 \
  --pool-mode auto
```

### 💼 实盘操作问题

#### Q: 如何导入券商交易系统？
**导入方式1: 直接CSV导入**
```bash
# 生成的CSV文件可直接导入主流券商：
# - 招商证券智远APP
# - 华泰证券涨乐财富通
# - 平安证券
# - 中信建投

# 文件位置：daily_trading_plan_YYYYMMDD.csv
```

**导入方式2: 手动执行**
```bash
# 打开CSV文件，按行执行：
# 1. 核对股票代码和价格
# 2. 按plan_shares数量下单
# 3. 参考entry_hint执行时机
# 4. 设置stop_loss止损价
```

#### Q: 如何更新持仓文件？
```bash
# 1. 首次使用：创建空的holdings.json
echo '{}' > holdings.json

# 2. 手动更新（成交后）
# 编辑holdings.json，格式：
{
  "000001": 1000,    # 股票代码: 持仓数量
  "600000": 1500,
  "300750": 800
}

# 3. 下次使用持仓文件
python claude.py --mode trading --current-holdings holdings.json
```

#### Q: 交易计划太多/太少怎么办？
```bash
# 计划太多：减少持仓数量
python claude.py --mode trading --max-positions 3

# 计划太少：增加股票池或持仓数量
python claude.py --mode trading --max-positions 8 --max-stocks 300

# 过滤质量：使用指数成分股
python claude.py --mode trading --pool-mode index --index-code 000300
```

### ⚠️ 性能与资源问题

#### Q: 程序运行太慢怎么办？
```bash
# 方案1：减少数据量
python claude.py --mode analysis --max-stocks 100 --start-date 20241101

# 方案2：增加并发
python claude.py --mode analysis --max-workers 8

# 方案3：禁用可视化
python claude.py --mode analysis --no-dashboard
```

#### Q: 内存不足怎么办？
```bash
# 减少股票数量
python claude.py --mode analysis --max-stocks 50

# 缩短时间范围
python claude.py --mode analysis --start-date 20241201

# 禁用并发（节省内存）
python claude.py --mode analysis --no-concurrent
```

## 📚 技术架构

### 核心类
- **`RiskSensitiveTrendStrategy`**: 主策略类
- **`DailyTradingPlan`**: 交易计划生成器

### 关键方法
- `run_strategy()`: 运行完整策略分析
- `generate_complete_daily_plan()`: 生成每日交易计划
- `calculate_precise_position_size()`: 精确仓位计算
- `check_price_limit_risk()`: 涨跌停风险检查

### 数据来源
- **价格数据**: Qlib金融数据库
- **ST股票**: AkShare API实时获取
- **指数成分**: AkShare指数接口

## 📝 更新日志

### v2.0.0 - 实盘交易引擎
- ✅ 新增每日交易计划生成功能
- ✅ 实现固定随机种子机制
- ✅ 添加精确风险法仓位计算
- ✅ 集成ADV流动性约束
- ✅ 完善涨跌停风险标记
- ✅ 标准化CSV导出格式

### v1.0.0 - 策略分析版本
- ✅ A股趋势跟踪策略
- ✅ 风险调整相对强度
- ✅ 回测分析框架
- ✅ 风险管理仪表板

## 📞 支持与贡献

这是一个基于Qlib和AkShare的开源A股量化交易系统，设计目标是提供生产级的实盘交易支持。

**注意**: 本系统仅供学习和研究使用，不构成投资建议。实盘交易请充分测试并谨慎使用。

## 📄 许可证

MIT License - 详见LICENSE文件