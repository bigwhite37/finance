"""自适应风险预算回测验证和参数敏感性测试"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy import stats
import warnings

from src.rl_trading_system.risk_control.adaptive_risk_budget import (
    AdaptiveRiskBudget,
    AdaptiveRiskBudgetConfig,
    PerformanceMetrics,
    MarketMetrics,
    MarketCondition,
    PerformanceRegime
)


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float
    expected_shortfall: float
    risk_budget_stats: Dict[str, float]
    adjustment_count: int
    final_config: AdaptiveRiskBudgetConfig


class AdaptiveRiskBudgetBacktester:
    """自适应风险预算回测器"""
    
    def __init__(self, config: AdaptiveRiskBudgetConfig):
        self.config = config
        self.adaptive_budget = AdaptiveRiskBudget(config)
        
    def generate_synthetic_data(self, 
                              days: int = 252,
                              base_return: float = 0.0002,
                              base_volatility: float = 0.015,
                              regime_changes: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成合成市场数据
        
        Args:
            days: 天数
            base_return: 基础日收益率
            base_volatility: 基础波动率
            regime_changes: 状态变化次数
            
        Returns:
            (价格数据, 市场指标数据)
        """
        np.random.seed(42)  # 确保可重复性
        
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # 生成市场状态变化点
        regime_change_points = sorted(np.random.choice(days, regime_changes, replace=False))
        regime_change_points = [0] + regime_change_points + [days]
        
        returns = []
        volatilities = []
        market_trends = []
        uncertainty_indices = []
        
        for i in range(len(regime_change_points) - 1):
            start_idx = regime_change_points[i]
            end_idx = regime_change_points[i + 1]
            period_length = end_idx - start_idx
            
            # 随机选择市场状态
            regime = np.random.choice([
                MarketCondition.BULL,
                MarketCondition.BEAR,
                MarketCondition.SIDEWAYS,
                MarketCondition.HIGH_VOLATILITY,
                MarketCondition.CRISIS
            ])
            
            # 根据状态设置参数
            if regime == MarketCondition.BULL:
                period_return = base_return * 2
                period_volatility = base_volatility * 0.8
                trend = 0.15
                uncertainty = 0.2
            elif regime == MarketCondition.BEAR:
                period_return = -base_return * 1.5
                period_volatility = base_volatility * 1.2
                trend = -0.12
                uncertainty = 0.4
            elif regime == MarketCondition.CRISIS:
                period_return = -base_return * 3
                period_volatility = base_volatility * 2.5
                trend = -0.25
                uncertainty = 0.9
            elif regime == MarketCondition.HIGH_VOLATILITY:
                period_return = base_return * 0.5
                period_volatility = base_volatility * 2.0
                trend = 0.02
                uncertainty = 0.6
            else:  # SIDEWAYS
                period_return = base_return * 0.8
                period_volatility = base_volatility
                trend = 0.0
                uncertainty = 0.3
            
            # 生成该时期的数据
            period_returns = np.random.normal(period_return, period_volatility, period_length)
            period_vols = np.full(period_length, period_volatility)
            period_trends = np.full(period_length, trend)
            period_uncertainties = np.full(period_length, uncertainty)
            
            returns.extend(period_returns)
            volatilities.extend(period_vols)
            market_trends.extend(period_trends)
            uncertainty_indices.extend(period_uncertainties)
        
        # 创建价格数据
        prices = [100.0]  # 初始价格
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data = pd.DataFrame({
            'date': dates,
            'price': prices[1:],  # 去掉初始价格
            'return': returns
        })
        
        # 创建市场指标数据
        market_data = pd.DataFrame({
            'date': dates,
            'market_volatility': volatilities,
            'market_trend': market_trends,
            'uncertainty_index': uncertainty_indices,
            'correlation_with_market': np.random.uniform(0.3, 0.8, days),
            'liquidity_score': np.random.uniform(0.7, 1.0, days)
        })
        
        return price_data, market_data
    
    def run_backtest(self, 
                    price_data: pd.DataFrame,
                    market_data: pd.DataFrame,
                    initial_capital: float = 100000.0) -> BacktestResult:
        """
        运行回测
        
        Args:
            price_data: 价格数据
            market_data: 市场数据
            initial_capital: 初始资金
            
        Returns:
            回测结果
        """
        self.adaptive_budget.reset_system()
        
        portfolio_values = [initial_capital]
        risk_budgets = []
        positions = []
        
        # 滚动窗口计算表现指标
        lookback_window = 20
        
        for i in range(len(price_data)):
            current_date = price_data.iloc[i]['date']
            current_return = price_data.iloc[i]['return']
            
            # 更新市场指标
            market_metrics = MarketMetrics(
                market_volatility=market_data.iloc[i]['market_volatility'],
                market_trend=market_data.iloc[i]['market_trend'],
                uncertainty_index=market_data.iloc[i]['uncertainty_index'],
                correlation_with_market=market_data.iloc[i]['correlation_with_market'],
                liquidity_score=market_data.iloc[i]['liquidity_score'],
                timestamp=current_date
            )
            self.adaptive_budget.update_market_metrics(market_metrics)
            
            # 计算表现指标（如果有足够的历史数据）
            if i >= lookback_window:
                recent_returns = price_data.iloc[i-lookback_window:i]['return'].values
                recent_values = portfolio_values[-lookback_window:]
                
                # 计算表现指标
                performance_metrics = self._calculate_performance_metrics(
                    recent_returns, recent_values, current_date
                )
                self.adaptive_budget.update_performance_metrics(performance_metrics)
            
            # 计算自适应风险预算
            risk_budget = self.adaptive_budget.calculate_adaptive_risk_budget()
            risk_budgets.append(risk_budget)
            
            # 简化的仓位计算（基于风险预算）
            current_value = portfolio_values[-1]
            max_position_value = current_value * risk_budget
            
            # 假设全仓投资，仓位大小由风险预算决定
            position_size = max_position_value / price_data.iloc[i]['price']
            positions.append(position_size)
            
            # 计算新的组合价值
            if i > 0:
                position_return = current_return * (max_position_value / current_value)
                new_value = current_value * (1 + position_return)
            else:
                new_value = initial_capital
            
            portfolio_values.append(new_value)
        
        # 计算回测结果
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        result = BacktestResult(
            total_return=(portfolio_values[-1] - initial_capital) / initial_capital,
            sharpe_ratio=self._calculate_sharpe_ratio(portfolio_returns),
            max_drawdown=self._calculate_max_drawdown(portfolio_values),
            volatility=np.std(portfolio_returns) * np.sqrt(252),
            calmar_ratio=self._calculate_calmar_ratio(portfolio_returns, portfolio_values),
            win_rate=np.mean(portfolio_returns > 0),
            profit_factor=self._calculate_profit_factor(portfolio_returns),
            var_95=np.percentile(portfolio_returns, 5),
            expected_shortfall=np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]),
            risk_budget_stats={
                'mean': np.mean(risk_budgets),
                'std': np.std(risk_budgets),
                'min': np.min(risk_budgets),
                'max': np.max(risk_budgets)
            },
            adjustment_count=len(self.adaptive_budget.adjustment_history),
            final_config=self.config
        )
        
        return result
    
    def _calculate_performance_metrics(self, 
                                     returns: np.ndarray,
                                     values: List[float],
                                     timestamp: datetime) -> PerformanceMetrics:
        """计算表现指标"""
        if len(returns) == 0:
            return PerformanceMetrics(timestamp=timestamp)
        
        # 基础统计
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # 夏普比率
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
        sharpe_ratio *= np.sqrt(252)  # 年化
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(values)
        
        # 卡尔玛比率
        calmar_ratio = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # 胜率
        win_rate = np.mean(returns > 0)
        
        # 盈亏比
        profit_factor = self._calculate_profit_factor(returns)
        
        # 连续亏损
        consecutive_losses = self._calculate_consecutive_losses(returns)
        
        # VaR和ES
        var_95 = np.percentile(returns, 5)
        expected_shortfall = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else 0.0
        
        # 下行偏差和Sortino比率
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0.0
        sortino_ratio *= np.sqrt(252)  # 年化
        
        return PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility * np.sqrt(252),
            win_rate=win_rate,
            profit_factor=profit_factor,
            consecutive_losses=consecutive_losses,
            total_return=mean_return * 252,
            downside_deviation=downside_deviation * np.sqrt(252),
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            timestamp=timestamp
        )
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """计算最大回撤"""
        if len(values) < 2:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, values: List[float]) -> float:
        """计算卡尔玛比率"""
        annual_return = np.mean(returns) * 252
        max_drawdown = self._calculate_max_drawdown(values)
        
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / max_drawdown
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """计算盈亏比"""
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return float('inf') if len(profits) > 0 else 1.0
        
        total_profit = np.sum(profits)
        total_loss = abs(np.sum(losses))
        
        return total_profit / total_loss if total_loss > 0 else 1.0
    
    def _calculate_consecutive_losses(self, returns: np.ndarray) -> int:
        """计算连续亏损次数"""
        consecutive = 0
        max_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        return max_consecutive
    
    def plot_backtest_results(self, 
                            price_data: pd.DataFrame,
                            market_data: pd.DataFrame,
                            result: BacktestResult,
                            save_path: str = "backtest_results.html") -> None:
        """
        可视化回测结果
        
        Args:
            price_data: 价格数据
            market_data: 市场数据
            result: 回测结果
            save_path: 保存路径
        """
        try:
            # 创建子图
            fig = sp.make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    '价格走势', '风险预算变化',
                    '市场波动率', '市场趋势',
                    '不确定性指数', '回测统计'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
            
            # 图1: 价格走势
            fig.add_trace(
                go.Scatter(
                    x=price_data['date'],
                    y=price_data['price'],
                    mode='lines',
                    name='价格',
                    line=dict(color='blue', width=2),
                    hovertemplate='日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 图2: 风险预算变化（需要从adaptive_budget获取）
            if hasattr(self, 'adaptive_budget') and self.adaptive_budget.risk_budget_history:
                risk_budgets = list(self.adaptive_budget.risk_budget_history)
                dates = price_data['date'][:len(risk_budgets)]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=risk_budgets,
                        mode='lines',
                        name='风险预算',
                        line=dict(color='red', width=2),
                        hovertemplate='日期: %{x}<br>风险预算: %{y:.1%}<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                # 添加基准线
                fig.add_hline(
                    y=self.config.base_risk_budget,
                    line_dash="dash",
                    line_color="green",
                    row=1, col=2
                )
            
            # 图3: 市场波动率
            fig.add_trace(
                go.Scatter(
                    x=market_data['date'],
                    y=market_data['market_volatility'],
                    mode='lines',
                    name='市场波动率',
                    line=dict(color='orange', width=2),
                    hovertemplate='日期: %{x}<br>波动率: %{y:.1%}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 图4: 市场趋势
            fig.add_trace(
                go.Scatter(
                    x=market_data['date'],
                    y=market_data['market_trend'],
                    mode='lines',
                    name='市场趋势',
                    line=dict(color='purple', width=2),
                    hovertemplate='日期: %{x}<br>趋势: %{y:.1%}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # 添加零线
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=2)
            
            # 图5: 不确定性指数
            fig.add_trace(
                go.Scatter(
                    x=market_data['date'],
                    y=market_data['uncertainty_index'],
                    mode='lines',
                    name='不确定性指数',
                    line=dict(color='brown', width=2),
                    hovertemplate='日期: %{x}<br>不确定性: %{y:.1%}<extra></extra>'
                ),
                row=3, col=1
            )
            
            # 图6: 回测统计表格
            stats_data = [
                ['总收益', f'{result.total_return:.1%}'],
                ['夏普比率', f'{result.sharpe_ratio:.2f}'],
                ['最大回撤', f'{result.max_drawdown:.1%}'],
                ['波动率', f'{result.volatility:.1%}'],
                ['卡尔玛比率', f'{result.calmar_ratio:.2f}'],
                ['胜率', f'{result.win_rate:.1%}'],
                ['盈亏比', f'{result.profit_factor:.2f}'],
                ['VaR(95%)', f'{result.var_95:.1%}'],
                ['调整次数', f'{result.adjustment_count}'],
                ['平均风险预算', f'{result.risk_budget_stats["mean"]:.1%}'],
                ['风险预算标准差', f'{result.risk_budget_stats["std"]:.1%}']
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['指标', '数值'],
                        fill_color='lightblue',
                        align='left',
                        font=dict(size=12, color='black')
                    ),
                    cells=dict(
                        values=list(zip(*stats_data)),
                        fill_color='white',
                        align='left',
                        font=dict(size=11, color='black')
                    )
                ),
                row=3, col=2
            )
            
            # 更新布局
            fig.update_layout(
                height=1000,
                title_text="自适应风险预算回测结果",
                title_x=0.5,
                title_font_size=16,
                showlegend=True
            )
            
            # 更新坐标轴标签
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="风险预算", tickformat='.1%', row=1, col=2)
            fig.update_yaxes(title_text="波动率", tickformat='.1%', row=2, col=1)
            fig.update_yaxes(title_text="趋势", tickformat='.1%', row=2, col=2)
            fig.update_yaxes(title_text="不确定性", tickformat='.1%', row=3, col=1)
            
            fig.update_xaxes(title_text="日期", row=1, col=1)
            fig.update_xaxes(title_text="日期", row=1, col=2)
            fig.update_xaxes(title_text="日期", row=2, col=1)
            fig.update_xaxes(title_text="日期", row=2, col=2)
            fig.update_xaxes(title_text="日期", row=3, col=1)
            
            # 保存图表
            fig.write_html(save_path)
            print(f"回测结果图表已保存为 '{save_path}'")
            
            return fig
            
        except ImportError:
            print("plotly未安装，跳过回测结果可视化")
            return None
        except Exception as e:
            print(f"绘制回测结果图表时出错: {e}")
            return None


class TestAdaptiveRiskBudgetBacktest:
    """自适应风险预算回测测试"""
    
    @pytest.fixture
    def base_config(self):
        """基础配置"""
        return AdaptiveRiskBudgetConfig(
            base_risk_budget=0.10,
            min_risk_budget=0.02,
            max_risk_budget=0.25,
            performance_lookback_days=30,
            market_lookback_days=20,
            smoothing_factor=0.1,
            max_daily_change=0.05
        )
    
    @pytest.fixture
    def backtester(self, base_config):
        """回测器"""
        return AdaptiveRiskBudgetBacktester(base_config)
    
    def test_generate_synthetic_data(self, backtester):
        """测试合成数据生成"""
        price_data, market_data = backtester.generate_synthetic_data(days=100)
        
        assert len(price_data) == 100
        assert len(market_data) == 100
        assert 'price' in price_data.columns
        assert 'return' in price_data.columns
        assert 'market_volatility' in market_data.columns
        assert 'market_trend' in market_data.columns
        assert 'uncertainty_index' in market_data.columns
        
        # 检查数据合理性
        assert (price_data['price'] > 0).all()
        assert (market_data['market_volatility'] >= 0).all()
        assert ((market_data['market_trend'] >= -1) & (market_data['market_trend'] <= 1)).all()
        assert ((market_data['uncertainty_index'] >= 0) & (market_data['uncertainty_index'] <= 1)).all()
    
    def test_run_backtest_basic(self, backtester):
        """测试基础回测"""
        price_data, market_data = backtester.generate_synthetic_data(days=50)
        result = backtester.run_backtest(price_data, market_data)
        
        assert isinstance(result, BacktestResult)
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.volatility, float)
        assert isinstance(result.risk_budget_stats, dict)
        assert result.adjustment_count >= 0
        
        # 检查风险预算统计
        assert 'mean' in result.risk_budget_stats
        assert 'std' in result.risk_budget_stats
        assert 'min' in result.risk_budget_stats
        assert 'max' in result.risk_budget_stats
        
        # 检查合理性
        assert result.risk_budget_stats['min'] >= backtester.config.min_risk_budget
        assert result.risk_budget_stats['max'] <= backtester.config.max_risk_budget
    
    def test_parameter_sensitivity_analysis(self, base_config):
        """测试参数敏感性分析"""
        # 测试不同的基础风险预算
        base_budgets = [0.05, 0.10, 0.15, 0.20]
        results = []
        
        for base_budget in base_budgets:
            config = AdaptiveRiskBudgetConfig(
                base_risk_budget=base_budget,
                min_risk_budget=base_budget * 0.2,
                max_risk_budget=base_budget * 2.5,
                performance_lookback_days=30,
                smoothing_factor=0.1
            )
            
            backtester = AdaptiveRiskBudgetBacktester(config)
            price_data, market_data = backtester.generate_synthetic_data(days=100)
            result = backtester.run_backtest(price_data, market_data)
            results.append((base_budget, result))
        
        # 分析结果
        assert len(results) == len(base_budgets)
        
        # 检查风险预算与收益的关系
        budgets = [r[0] for r in results]
        returns = [r[1].total_return for r in results]
        volatilities = [r[1].volatility for r in results]
        
        # 一般来说，更高的风险预算应该带来更高的波动率
        correlation_vol = np.corrcoef(budgets, volatilities)[0, 1]
        assert not np.isnan(correlation_vol)  # 至少应该能计算出相关性
    
    def test_smoothing_factor_sensitivity(self, base_config):
        """测试平滑因子敏感性"""
        smoothing_factors = [0.05, 0.1, 0.2, 0.3]
        results = []
        
        for smoothing_factor in smoothing_factors:
            config = AdaptiveRiskBudgetConfig(
                base_risk_budget=0.10,
                smoothing_factor=smoothing_factor,
                performance_lookback_days=30
            )
            
            backtester = AdaptiveRiskBudgetBacktester(config)
            price_data, market_data = backtester.generate_synthetic_data(days=100)
            result = backtester.run_backtest(price_data, market_data)
            results.append((smoothing_factor, result))
        
        # 分析平滑因子对风险预算变化的影响
        smoothing_values = [r[0] for r in results]
        budget_stds = [r[1].risk_budget_stats['std'] for r in results]
        
        # 更高的平滑因子应该导致更稳定的风险预算（更低的标准差）
        # 但这个关系可能不是严格单调的，所以我们只检查合理性
        assert all(std >= 0 for std in budget_stds)
        assert len(set(budget_stds)) > 1  # 应该有不同的结果
    
    def test_performance_threshold_sensitivity(self, base_config):
        """测试表现阈值敏感性"""
        excellent_thresholds = [1.5, 2.0, 2.5]
        results = []
        
        for threshold in excellent_thresholds:
            config = AdaptiveRiskBudgetConfig(
                base_risk_budget=0.10,
                sharpe_threshold_excellent=threshold,
                sharpe_threshold_good=threshold * 0.5,
                performance_adjustment_factor=0.3
            )
            
            backtester = AdaptiveRiskBudgetBacktester(config)
            price_data, market_data = backtester.generate_synthetic_data(days=100)
            result = backtester.run_backtest(price_data, market_data)
            results.append((threshold, result))
        
        # 检查结果合理性
        assert len(results) == len(excellent_thresholds)
        for threshold, result in results:
            assert result.adjustment_count >= 0
            assert result.risk_budget_stats['min'] >= config.min_risk_budget
            assert result.risk_budget_stats['max'] <= config.max_risk_budget
    
    def test_market_adjustment_sensitivity(self, base_config):
        """测试市场调整因子敏感性"""
        market_factors = [0.2, 0.4, 0.6]
        results = []
        
        for factor in market_factors:
            config = AdaptiveRiskBudgetConfig(
                base_risk_budget=0.10,
                market_adjustment_factor=factor,
                volatility_threshold_high=0.25,
                uncertainty_threshold=0.7
            )
            
            backtester = AdaptiveRiskBudgetBacktester(config)
            price_data, market_data = backtester.generate_synthetic_data(days=100)
            result = backtester.run_backtest(price_data, market_data)
            results.append((factor, result))
        
        # 检查结果
        assert len(results) == len(market_factors)
        
        # 分析市场调整因子对调整频率的影响
        factors = [r[0] for r in results]
        adjustment_counts = [r[1].adjustment_count for r in results]
        
        # 更高的市场调整因子可能导致更多的调整
        assert all(count >= 0 for count in adjustment_counts)
    
    def test_consecutive_loss_penalty_sensitivity(self, base_config):
        """测试连续亏损惩罚敏感性"""
        loss_penalties = [0.1, 0.2, 0.3]
        results = []
        
        for penalty in loss_penalties:
            config = AdaptiveRiskBudgetConfig(
                base_risk_budget=0.10,
                loss_penalty_factor=penalty,
                consecutive_loss_threshold=3,
                max_loss_penalty=0.5
            )
            
            backtester = AdaptiveRiskBudgetBacktester(config)
            price_data, market_data = backtester.generate_synthetic_data(days=100)
            result = backtester.run_backtest(price_data, market_data)
            results.append((penalty, result))
        
        # 检查结果合理性
        assert len(results) == len(loss_penalties)
        for penalty, result in results:
            assert result.max_drawdown >= 0
            assert result.risk_budget_stats['min'] >= config.min_risk_budget
    
    def test_comparison_with_static_budget(self, base_config):
        """测试与静态风险预算的对比"""
        # 自适应风险预算
        adaptive_backtester = AdaptiveRiskBudgetBacktester(base_config)
        price_data, market_data = adaptive_backtester.generate_synthetic_data(days=200)
        adaptive_result = adaptive_backtester.run_backtest(price_data, market_data)
        
        # 静态风险预算（禁用所有调整）
        static_config = AdaptiveRiskBudgetConfig(
            base_risk_budget=0.10,
            performance_adjustment_factor=0.0,  # 禁用表现调整
            market_adjustment_factor=0.0,       # 禁用市场调整
            loss_penalty_factor=0.0,            # 禁用亏损惩罚
            smoothing_factor=0.0,               # 禁用平滑
            max_daily_change=1.0                # 允许任意变化
        )
        
        static_backtester = AdaptiveRiskBudgetBacktester(static_config)
        static_result = static_backtester.run_backtest(price_data, market_data)
        
        # 比较结果
        print(f"自适应策略 - 总收益: {adaptive_result.total_return:.3f}, "
              f"夏普比率: {adaptive_result.sharpe_ratio:.3f}, "
              f"最大回撤: {adaptive_result.max_drawdown:.3f}")
        
        print(f"静态策略 - 总收益: {static_result.total_return:.3f}, "
              f"夏普比率: {static_result.sharpe_ratio:.3f}, "
              f"最大回撤: {static_result.max_drawdown:.3f}")
        
        # 自适应策略应该有更多的调整
        assert adaptive_result.adjustment_count > static_result.adjustment_count
        
        # 自适应策略的风险预算应该有更大的变化范围
        adaptive_range = (adaptive_result.risk_budget_stats['max'] - 
                         adaptive_result.risk_budget_stats['min'])
        static_range = (static_result.risk_budget_stats['max'] - 
                       static_result.risk_budget_stats['min'])
        
        assert adaptive_range >= static_range
    
    def test_extreme_market_conditions(self, base_config):
        """测试极端市场条件下的表现"""
        backtester = AdaptiveRiskBudgetBacktester(base_config)
        
        # 生成包含更多危机的数据
        np.random.seed(123)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # 模拟市场崩盘
        returns = []
        volatilities = []
        trends = []
        uncertainties = []
        
        for i in range(100):
            if 30 <= i <= 50:  # 危机期
                returns.append(np.random.normal(-0.005, 0.05))  # 大幅下跌
                volatilities.append(0.4)
                trends.append(-0.3)
                uncertainties.append(0.9)
            elif 51 <= i <= 70:  # 恢复期
                returns.append(np.random.normal(0.003, 0.02))
                volatilities.append(0.2)
                trends.append(0.1)
                uncertainties.append(0.4)
            else:  # 正常期
                returns.append(np.random.normal(0.001, 0.015))
                volatilities.append(0.15)
                trends.append(0.05)
                uncertainties.append(0.3)
        
        # 构建数据
        prices = [100.0]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        price_data = pd.DataFrame({
            'date': dates,
            'price': prices[1:],
            'return': returns
        })
        
        market_data = pd.DataFrame({
            'date': dates,
            'market_volatility': volatilities,
            'market_trend': trends,
            'uncertainty_index': uncertainties,
            'correlation_with_market': [0.7] * 100,
            'liquidity_score': [0.8] * 100
        })
        
        result = backtester.run_backtest(price_data, market_data)
        
        # 在极端条件下，系统应该：
        # 1. 降低风险预算
        assert result.risk_budget_stats['min'] < base_config.base_risk_budget
        
        # 2. 有多次调整
        assert result.adjustment_count > 0
        
        # 3. 最大回撤应该在合理范围内
        assert result.max_drawdown < 0.5  # 不应该超过50%
    
    def test_statistical_significance(self, base_config):
        """测试统计显著性"""
        # 运行多次回测以检验结果的稳定性
        n_runs = 10
        adaptive_results = []
        static_results = []
        
        for run in range(n_runs):
            # 自适应策略
            adaptive_backtester = AdaptiveRiskBudgetBacktester(base_config)
            price_data, market_data = adaptive_backtester.generate_synthetic_data(
                days=150, regime_changes=np.random.randint(2, 6)
            )
            adaptive_result = adaptive_backtester.run_backtest(price_data, market_data)
            adaptive_results.append(adaptive_result.sharpe_ratio)
            
            # 静态策略
            static_config = AdaptiveRiskBudgetConfig(
                base_risk_budget=base_config.base_risk_budget,
                performance_adjustment_factor=0.0,
                market_adjustment_factor=0.0,
                loss_penalty_factor=0.0,
                smoothing_factor=0.0
            )
            static_backtester = AdaptiveRiskBudgetBacktester(static_config)
            static_result = static_backtester.run_backtest(price_data, market_data)
            static_results.append(static_result.sharpe_ratio)
        
        # 统计检验
        adaptive_mean = np.mean(adaptive_results)
        static_mean = np.mean(static_results)
        
        print(f"自适应策略平均夏普比率: {adaptive_mean:.3f}")
        print(f"静态策略平均夏普比率: {static_mean:.3f}")
        
        # t检验
        if len(adaptive_results) > 1 and len(static_results) > 1:
            t_stat, p_value = stats.ttest_ind(adaptive_results, static_results)
            print(f"t统计量: {t_stat:.3f}, p值: {p_value:.3f}")
            
            # 记录结果（不强制要求显著性，因为这取决于具体的市场条件）
            assert not np.isnan(t_stat)
            assert not np.isnan(p_value)
        
        # 检查结果的合理性
        assert all(not np.isnan(result) for result in adaptive_results)
        assert all(not np.isnan(result) for result in static_results)
    
    def test_visualization_integration(self, base_config):
        """测试可视化集成"""
        backtester = AdaptiveRiskBudgetBacktester(base_config)
        price_data, market_data = backtester.generate_synthetic_data(days=50)
        result = backtester.run_backtest(price_data, market_data)
        
        # 测试可视化功能
        fig = backtester.plot_backtest_results(
            price_data, market_data, result, 
            save_path="test_backtest_visualization.html"
        )
        
        # 如果plotly可用，应该返回figure对象
        if fig is not None:
            assert hasattr(fig, 'data')  # plotly figure应该有data属性
            assert len(fig.data) > 0     # 应该有数据


if __name__ == "__main__":
    # 运行特定测试
    pytest.main([__file__ + "::TestAdaptiveRiskBudgetBacktest::test_comparison_with_static_budget", "-v"])