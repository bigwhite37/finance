# 回测引擎API文档

## BacktestEngine 类

`BacktestEngine`负责执行强化学习策略的历史回测。它驱动智能体（Agent）和交易环境（Environment）进行交互，记录整个过程，并最终生成全面的性能和风险分析报告。

### 类签名
```python
class BacktestEngine:
    def __init__(self, config: dict)
```

### 初始化参数
-   **config** (dict): 回测引擎的配置字典，包含交易成本、滑点等设置。

### 核心方法

#### run_backtest
执行一次完整的策略回测。

```python
def run_backtest(self, agent, env, start_date: str, end_date: str, benchmark_data: Optional[pd.DataFrame] = None, safety_shield=None) -> Dict
```

**参数:**
-   **agent**: 训练好的强化学习智能体实例。
-   **env**: 交易环境实例。
-   **start_date** (str): 回测开始日期。
-   **end_date** (str): 回测结束日期。
-   **benchmark_data** (Optional[pd.DataFrame]): 用于对比的基准数据（如沪深300指数）。
-   **safety_shield** (Optional): 安全保护层实例（如`RiskController`），用于在回测中应用风险控制。

**返回值:**
-   **Dict**: 一个包含所有回测结果的字典，包括性能指标、舒适度指标、交易统计、风险指标以及详细的每日历史数据。

### 结果分析

`run_backtest`方法内部调用了两个核心分析器来处理回测结果：

-   **PerformanceAnalyzer**: 计算传统金融领域的绩效指标，如年化收益率、夏普比率、最大回撤等。
-   **ComfortabilityMetrics**: 计算为本系统专门设计的投资心理舒适度指标，如月度最大回撤、连续亏损天数等。

### 其他主要方法

#### generate_backtest_report
将回测结果字典格式化为人类可读的文本报告。

```python
def generate_backtest_report(self, results: Dict) -> str
```

#### save_results / load_results
使用`joblib`将回测结果字典保存到文件或从文件加载，方便后续分析。

```python
def save_results(self, results: Dict, filepath: str)
def load_results(self, filepath: str) -> Dict
```

### 配置参数

```yaml
backtest_config:
  initial_capital: 1000000 # 初始资金
  transaction_cost: 0.001 # 交易成本（双边）
  slippage: 0.0001 # 滑点
  commission: 0.0005 # 佣金
```

### 回测结果字典结构

`run_backtest`返回的结果字典包含以下关键字段：

-   `backtest_summary`: 回测的基本信息（周期、最终净值等）。
-   `performance_metrics`: 核心绩效指标。
-   `comfort_metrics`: 心理舒适度指标。
-   `trading_stats`: 交易统计（换手率、杠杆等）。
-   `risk_metrics`: 详细的风险指标（VaR, CVaR, 偏度等）。
-   `portfolio_history`: `pd.DataFrame`格式的每日组合净值历史。
-   `returns_series`: `pd.Series`格式的每日收益率序列。
