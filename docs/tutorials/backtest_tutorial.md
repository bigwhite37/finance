# 回测分析教程

本教程将指导您如何运行策略回测并深入分析回测结果。回测是检验和评估您策略表现的最重要环节。

## 1. 运行回测

系统的回测功能由`BacktestEngine`驱动。您通常会通过主脚本`main.py`或在您自己的脚本中调用它来运行回测。

**通过主脚本运行:**
```bash
# 运行一个完整的训练和回测流程
python main.py --mode full --config my_config.yaml

# 仅使用已保存的模型进行回测
python main.py --mode backtest --model ./models/my_best_agent.pth
```

**在自定义脚本中运行:**
```python
from backtest.backtest_engine import BacktestEngine

# 假设 agent, env, config, benchmark_data 已准备好
backtest_config = config.get_config('backtest')
engine = BacktestEngine(backtest_config)

results = engine.run_backtest(
    agent=agent,
    env=env,
    start_date='2021-01-01',
    end_date='2023-12-31',
    benchmark_data=benchmark_data # 例如沪深300指数的收益率
)

# 保存结果以备后续分析
engine.save_results(results, 'my_backtest_results.pkl')
```

## 2. 理解回测结果

`run_backtest`方法返回一个包含所有回测信息的字典。您可以使用`generate_backtest_report`快速生成一份文本摘要。

```python
# 加载已保存的结果
results = engine.load_results('my_backtest_results.pkl')

# 生成并打印报告
report_text = engine.generate_backtest_report(results)
print(report_text)
```

**报告示例:**
```
=== A股强化学习量化交易回测报告 ===

【回测概要】
回测期间: 2021-01-04 00:00:00 至 2023-12-29 00:00:00
最终净值: 1.2543

【收益指标】
总收益率: 25.43%
年化收益率: 7.85%
年化波动率: 11.50%
夏普比率: 0.68
最大回撤: -9.87%

【心理舒适度指标】
月度最大回撤: -4.50%
连续亏损天数: 6天
下跌日占比: 45.1%
...
```

## 3. 深入分析与可视化

文本报告提供了概览，但更深入的分析需要直接操作结果字典中的数据，特别是利用`utils/visualization.py`中的工具进行可视化。

回测结果字典中最重要的两个键是：
-   `portfolio_history`: 一个DataFrame，记录了每日的净值、回撤等信息。
-   `returns_series`: 一个Series，记录了每日的收益率。

### 3.1 绘制净值曲线

```python
from utils import visualization

portfolio_history = results['portfolio_history']
benchmark_returns = results['performance_metrics']['基准日收益率']

visualization.plot_net_value_curve(
    portfolio_history,
    benchmark_returns=benchmark_returns
)
```
这将生成策略净值与基准（如沪深300）的对比图，让您直观地看到策略的超额收益。

### 3.2 分析回撤

```python
visualization.plot_drawdown(portfolio_history)
```
这将绘制出策略净值的回撤序列图，帮助您识别策略在哪些时间段内经历了较大的亏损。

### 3.3 查看年度/月度收益

```python
returns_series = results['returns_series']

visualization.plot_annual_returns(returns_series)
visualization.plot_monthly_returns_heatmap(returns_series)
```
这些图表可以帮助您分析策略收益的周期性，例如策略在哪几个月份表现通常较好或较差。

## 4. 关键指标解读

在分析报告时，请重点关注以下几类指标：

-   **风险调整后收益**: **夏普比率 (Sharpe Ratio)** 是最重要的指标。它衡量的是每承担一单位风险，能获得多少超额回报。夏普比率越高越好。
-   **风险指标**: **最大回撤 (Max Drawdown)** 是我们最关心的风险指标。它告诉你在最坏的情况下，策略可能面临多大的亏损。我们的目标是将其控制在10%以内。
-   **心理舒适度指标**: **月度最大回撤** 和 **连续亏损天数** 反映了持有该策略的“体验”。这些指标越小，说明策略运行越平稳，投资者持有体验越好。
-   **交易统计**: **平均换手率 (Avg Turnover)** 反映了策略交易的频率。过高的换手率意味着更高的交易成本和冲击成本。

通过结合这些指标和可视化图表，您可以全面地评估您的策略，并找到下一步的优化方向。
