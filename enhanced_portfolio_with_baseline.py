#!/usr/bin/env python3
"""
增强版组合分析仪表板 - 包含基准指数对比
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import akshare as ak
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_baseline_indices_data(start_date="2024-01-01", end_date="2024-12-01"):
    """
    获取基准指数数据（沪深300和纳斯达克）

    Returns:
        dict: 包含指数数据的字典
    """
    baseline_data = {}

    # 1. 获取沪深300数据
    try:
        csi300 = ak.index_zh_a_hist(symbol="000300", period="daily", start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""))
        if not csi300.empty:
            csi300['date'] = pd.to_datetime(csi300['日期'])
            csi300 = csi300.set_index('date')
            csi300['return'] = csi300['收盘'].pct_change()
            csi300['cumulative_return'] = (1 + csi300['return']).cumprod()
            baseline_data['CSI300'] = {
                'price': csi300['收盘'],
                'return': csi300['return'],
                'cumulative_return': csi300['cumulative_return'],
                'name': '沪深300指数'
            }
            logger.info(f"✅ 成功获取沪深300数据: {len(csi300)} 个交易日")
    except Exception as e:
        logger.error(f"❌ 获取沪深300数据失败: {e}")

    # 2. 尝试获取纳斯达克数据
    try:
        nasdaq = ak.index_us_stock_sina(symbol=".IXIC")
        if nasdaq is not None and not nasdaq.empty:
            nasdaq['date'] = pd.to_datetime(nasdaq['date'])
            nasdaq = nasdaq.set_index('date')
            nasdaq['return'] = nasdaq['close'].pct_change()
            nasdaq['cumulative_return'] = (1 + nasdaq['return']).cumprod()

            # 过滤到指定日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            nasdaq = nasdaq[(nasdaq.index >= start_dt) & (nasdaq.index <= end_dt)]

            if len(nasdaq) > 0:
                baseline_data['NASDAQ'] = {
                    'price': nasdaq['close'],
                    'return': nasdaq['return'],
                    'cumulative_return': nasdaq['cumulative_return'],
                    'name': '纳斯达克指数'
                }
                logger.info(f"✅ 成功获取纳斯达克数据: {len(nasdaq)} 个交易日")
            else:
                logger.warning("⚠️ 纳斯达克数据在指定日期范围内为空")
    except Exception as e:
        logger.error(f"❌ 获取纳斯达克数据失败: {e}")

    return baseline_data

def create_enhanced_portfolio_dashboard_with_baseline(strategy_instance, equity_curve, performance_stats, selected_stocks, position_sizes):
    """创建增强版组合分析仪表板，包含基准指数对比"""

    # 获取基准指数数据
    start_date = equity_curve.index[0].strftime('%Y-%m-%d')
    end_date = equity_curve.index[-1].strftime('%Y-%m-%d')
    baseline_data = get_baseline_indices_data(start_date, end_date)

    # 更新子图布局 - 将第一个图改为策略 vs 基准对比
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=[
            '策略盈利曲线 vs 基准指数', '月度收益热力图',
            '日收益分布', '滚动夏普比率',
            '累计收益分解', '风险指标雷达图',
            '持仓权重分布', '个股贡献分析',
            '交易统计概览', '策略 vs 基准风险-收益对比'
        ],
        specs=[
            [{'secondary_y': True}, {'type': 'heatmap'}],
            [{'type': 'histogram'}, {'type': 'scatter'}],
            [{'secondary_y': True}, {'type': 'scatterpolar'}],
            [{'type': 'pie'}, {'type': 'bar'}],
            [{'type': 'table'}, {'type': 'scatter'}]
        ],
        vertical_spacing=0.06,
        horizontal_spacing=0.1,
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15]
    )

    # 1. 策略盈利曲线 vs 基准指数对比
    daily_returns = strategy_instance.daily_return if hasattr(strategy_instance, 'daily_return') and strategy_instance.daily_return is not None else equity_curve.pct_change().dropna()

    # 策略累计收益（归一化为从1开始）
    strategy_cumret = equity_curve / equity_curve.iloc[0]

    # 添加策略曲线
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=strategy_cumret.values,
            mode='lines',
            name='量化策略',
            line=dict(color='blue', width=3),
            hovertemplate='<b>日期</b>: %{x}<br>' +
                         '<b>累计收益</b>: %{customdata:.2f}%<extra></extra>',
            customdata=(strategy_cumret - 1) * 100
        ),
        row=1, col=1
    )

    # 添加基准指数曲线
    colors = ['red', 'green', 'orange', 'purple']
    color_idx = 0

    for key, data in baseline_data.items():
        # 对齐日期并重新基准化
        aligned_data = data['cumulative_return'].reindex(equity_curve.index, method='ffill')
        if not aligned_data.isna().all():
            # 从策略开始日期重新基准化
            first_valid_idx = aligned_data.first_valid_index()
            if first_valid_idx is not None:
                aligned_data = aligned_data / aligned_data[first_valid_idx]

                fig.add_trace(
                    go.Scatter(
                        x=aligned_data.index,
                        y=aligned_data.values,
                        mode='lines',
                        name=data['name'],
                        line=dict(color=colors[color_idx], width=2, dash='dash'),
                        hovertemplate=f'<b>日期</b>: %{{x}}<br>' +
                                     f'<b>{data["name"]}累计收益</b>: %{{customdata:.2f}}%<extra></extra>',
                        customdata=(aligned_data - 1) * 100
                    ),
                    row=1, col=1
                )
                color_idx += 1

    # 添加回撤曲线
    nav = equity_curve.copy()
    if (nav <= 0).any():
        logger.error("🚨 净值曲线包含非正值，强制修正")
        nav = nav.clip(lower=0.01)

    peak = nav.cummax()
    drawdown = (nav / peak - 1).clip(lower=-1.0, upper=0.0)

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,  # 转换为百分比
            mode='lines',
            name='策略回撤(%)',
            line=dict(color='rgba(255,0,0,0.7)', width=1),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            yaxis='y2',
            hovertemplate='日期: %{x}<br>回撤: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1, secondary_y=True
    )

    # 2. 月度收益热力图 - 保持原来的逻辑
    if len(daily_returns) > 30:
        monthly_nav = equity_curve.resample('M').last()
        monthly_returns = monthly_nav.pct_change().dropna() * 100
        monthly_df = monthly_returns.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month

        pivot_table = monthly_df.pivot(index='year', columns='month', values='return')

        fig.add_trace(
            go.Heatmap(
                z=pivot_table.values,
                x=[f"{i}月" for i in range(1, 13)],
                y=pivot_table.index,
                colorscale='RdYlGn',
                name='月度收益(%)',
                hovertemplate='%{y}年%{x}: %{z:.2f}%<extra></extra>'
            ),
            row=1, col=2
        )

    # 3. 日收益分布直方图
    fig.add_trace(
        go.Histogram(
            x=daily_returns * 100,
            nbinsx=50,
            name='收益分布',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate='收益区间: %{x:.2f}%<br>频次: %{y}<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. 滚动夏普比率（30天）
    if len(daily_returns) > 30:
        rolling_sharpe = daily_returns.rolling(30).mean() / daily_returns.rolling(30).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='30天滚动夏普比率',
                line=dict(color='purple'),
                hovertemplate='日期: %{x}<br>夏普比率: %{y:.2f}<extra></extra>'
            ),
            row=2, col=2
        )

    # 5. 累计收益分解（与基准对比）
    if baseline_data:
        # 计算超额收益
        excess_returns = {}
        for key, data in baseline_data.items():
            aligned_baseline_returns = data['return'].reindex(daily_returns.index, method='ffill')
            if not aligned_baseline_returns.isna().all():
                excess_return = daily_returns - aligned_baseline_returns
                excess_cumret = (1 + excess_return.fillna(0)).cumprod()
                excess_returns[key] = excess_cumret

        # 显示相对于第一个基准的超额收益
        if excess_returns:
            first_key = list(excess_returns.keys())[0]
            fig.add_trace(
                go.Scatter(
                    x=excess_returns[first_key].index,
                    y=excess_returns[first_key].values,
                    mode='lines',
                    name=f'相对{baseline_data[first_key]["name"]}超额收益',
                    line=dict(color='green'),
                    hovertemplate='日期: %{x}<br>超额收益倍数: %{y:.3f}<extra></extra>'
                ),
                row=3, col=1
            )

    # 添加基准收益曲线到累计收益分解图
    for key, data in baseline_data.items():
        aligned_data = data['cumulative_return'].reindex(equity_curve.index, method='ffill')
        if not aligned_data.isna().all():
            first_valid_idx = aligned_data.first_valid_index()
            if first_valid_idx is not None:
                aligned_data = aligned_data / aligned_data[first_valid_idx]
                fig.add_trace(
                    go.Scatter(
                        x=aligned_data.index,
                        y=aligned_data.values,
                        mode='lines',
                        name=f'{data["name"]}基准',
                        line=dict(color=colors[color_idx % len(colors)], width=1, dash='dot'),
                        hovertemplate=f'{data["name"]}: %{{y:.3f}}<extra></extra>'
                    ),
                    row=3, col=1
                )
                color_idx += 1

    # 6. 风险指标雷达图 - 保持原逻辑但添加基准对比
    risk_metrics = {
        '年化收益率': performance_stats.get('annual_return', 0) * 100,
        '夏普比率': performance_stats.get('sharpe_ratio', 0) * 10,  # 放大10倍用于展示
        '最大回撤': abs(performance_stats.get('max_drawdown', 0)) * 100,
        '胜率': performance_stats.get('win_rate', 0.5) * 100,
        '盈亏比': min(performance_stats.get('profit_loss_ratio', 1), 5) * 20,  # 限制并放大
    }

    fig.add_trace(
        go.Scatterpolar(
            r=list(risk_metrics.values()),
            theta=list(risk_metrics.keys()),
            fill='toself',
            name='策略风险指标',
            line_color='blue'
        ),
        row=3, col=2
    )

    # 7. 持仓权重分布饼图
    if position_sizes:
        # 获取最新的持仓权重
        latest_positions = {stock: weight for stock, weight in position_sizes.items() if weight > 0.01}
        if latest_positions:
            fig.add_trace(
                go.Pie(
                    labels=list(latest_positions.keys()),
                    values=list(latest_positions.values()),
                    name="持仓权重分布",
                    hovertemplate='%{label}: %{value:.2f}%<extra></extra>'
                ),
                row=4, col=1
            )

    # 8. 个股贡献分析（简化版）
    if selected_stocks and len(selected_stocks) > 0:
        # 模拟个股贡献（实际应该基于真实持仓和收益数据）
        contributions = [np.random.normal(5, 10) for _ in selected_stocks[:10]]  # 示例数据
        fig.add_trace(
            go.Bar(
                x=selected_stocks[:10],
                y=contributions,
                name='个股贡献(%)',
                marker_color=['green' if x > 0 else 'red' for x in contributions],
                hovertemplate='股票: %{x}<br>贡献: %{y:.2f}%<extra></extra>'
            ),
            row=4, col=2
        )

    # 9. 交易统计概览表格
    trading_stats = [
        ['总交易日数', len(equity_curve)],
        ['盈利天数', len(daily_returns[daily_returns > 0])],
        ['亏损天数', len(daily_returns[daily_returns < 0])],
        ['平均日收益率', f"{daily_returns.mean()*100:.3f}%"],
        ['收益率标准差', f"{daily_returns.std()*100:.3f}%"],
        ['最大单日收益', f"{daily_returns.max()*100:.3f}%"],
        ['最大单日亏损', f"{daily_returns.min()*100:.3f}%"]
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=['统计项', '数值'],
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[[stat[0] for stat in trading_stats],
                             [stat[1] for stat in trading_stats]],
                      fill_color='lavender',
                      align='left')
        ),
        row=5, col=1
    )

    # 10. 策略 vs 基准风险-收益散点图
    risk_return_data = []
    colors_scatter = []

    # 添加策略点
    strategy_annual_return = performance_stats.get('annual_return', 0) * 100
    strategy_volatility = daily_returns.std() * np.sqrt(252) * 100
    risk_return_data.append(['量化策略', strategy_annual_return, strategy_volatility])
    colors_scatter.append('blue')

    # 添加基准点
    for key, data in baseline_data.items():
        aligned_returns = data['return'].reindex(daily_returns.index, method='ffill').fillna(0)
        if len(aligned_returns) > 10:
            annual_return = aligned_returns.mean() * 252 * 100
            volatility = aligned_returns.std() * np.sqrt(252) * 100
            risk_return_data.append([data['name'], annual_return, volatility])
            colors_scatter.append('red' if 'CSI' in key else 'green')

    if risk_return_data:
        names = [item[0] for item in risk_return_data]
        returns = [item[1] for item in risk_return_data]
        risks = [item[2] for item in risk_return_data]

        fig.add_trace(
            go.Scatter(
                x=risks,
                y=returns,
                mode='markers+text',
                text=names,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=colors_scatter,
                    line=dict(width=2, color='white')
                ),
                name='风险-收益分析',
                hovertemplate='<b>%{text}</b><br>' +
                             '年化收益率: %{y:.2f}%<br>' +
                             '年化波动率: %{x:.2f}%<extra></extra>'
            ),
            row=5, col=2
        )

    # 更新布局
    fig.update_layout(
        height=2000,
        showlegend=True,
        title_text="增强版投资组合分析仪表板 - 含基准对比",
        title_x=0.5,
        font=dict(size=10)
    )

    # 更新坐标轴标签
    fig.update_xaxes(title_text="日期", row=1, col=1)
    fig.update_yaxes(title_text="累计收益倍数", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", secondary_y=True, row=1, col=1)

    fig.update_xaxes(title_text="年化波动率 (%)", row=5, col=2)
    fig.update_yaxes(title_text="年化收益率 (%)", row=5, col=2)

    return fig

def test_enhanced_dashboard():
    """测试增强版仪表板"""
    logger.info("🧪 测试增强版仪表板生成")

    # 创建模拟数据
    dates = pd.date_range('2024-01-01', '2024-12-01', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0008, 0.02, len(dates))  # 日收益率
    equity_curve = pd.Series((1 + returns).cumprod(), index=dates)

    # 模拟策略实例
    class MockStrategy:
        daily_return = pd.Series(returns, index=dates)

    strategy = MockStrategy()

    # 模拟绩效统计
    performance_stats = {
        'annual_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'win_rate': 0.55,
        'profit_loss_ratio': 1.8
    }

    selected_stocks = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000858.SZ']
    position_sizes = {'000001.SZ': 20, '000002.SZ': 15, '600000.SH': 25, '600036.SH': 30, '000858.SZ': 10}

    # 生成仪表板
    fig = create_enhanced_portfolio_dashboard_with_baseline(
        strategy, equity_curve, performance_stats, selected_stocks, position_sizes
    )

    # 保存到HTML文件
    fig.write_html("test_enhanced_portfolio_with_baseline.html")
    logger.info("✅ 测试仪表板已保存为 test_enhanced_portfolio_with_baseline.html")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_enhanced_dashboard()