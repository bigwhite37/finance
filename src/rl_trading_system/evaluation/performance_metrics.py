"""
绩效指标计算模块
实现收益指标、风险指标、风险调整指标、交易指标的计算
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
from decimal import Decimal
from abc import ABC, abstractmethod

from ..backtest.multi_frequency_backtest import Trade, OrderType


class ReturnMetrics:
    """收益率指标计算类"""
    
    def __init__(self, returns: pd.Series):
        """
        初始化收益率指标计算器
        
        Args:
            returns: 收益率时间序列
        """
        if returns.empty:
            raise ValueError("收益率序列不能为空")
        
        if returns.isna().any() or np.isinf(returns).any():
            raise ValueError("收益率序列包含无效值（NaN或无穷大）")
            
        self.returns = returns
    
    def calculate_total_return(self) -> float:
        """计算总收益率"""
        return float((1 + self.returns).prod() - 1)
    
    def calculate_annualized_return(self, periods_per_year: int = 252) -> float:
        """计算年化收益率"""
        total_return = self.calculate_total_return()
        total_periods = len(self.returns)
        return float((1 + total_return) ** (periods_per_year / total_periods) - 1)
    
    def calculate_monthly_returns(self) -> pd.DataFrame:
        """计算月度收益率"""
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            raise ValueError("收益率序列必须有日期索引")
            
        # 按月分组计算收益率
        monthly_returns = self.returns.groupby(self.returns.index.to_period('M')).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # 转换为DataFrame格式
        return pd.DataFrame({
            'monthly_return': monthly_returns,
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month
        })
    
    def calculate_cumulative_returns(self) -> pd.Series:
        """计算累积收益率"""
        return (1 + self.returns).cumprod() - 1


class RiskMetrics:
    """风险指标计算类"""
    
    def __init__(self, returns: pd.Series):
        """
        初始化风险指标计算器
        
        Args:
            returns: 收益率时间序列
        """
        if returns.empty:
            raise ValueError("收益率序列不能为空")
            
        self.returns = returns
    
    def calculate_volatility(self, annualized: bool = False, periods_per_year: int = 252) -> float:
        """计算波动率"""
        volatility = float(self.returns.std())
        
        if annualized:
            volatility *= np.sqrt(periods_per_year)
            
        return volatility
    
    def calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """计算最大回撤"""
        if portfolio_values.empty:
            raise ValueError("组合价值序列不能为空")
            
        # 计算历史最高点
        peak = portfolio_values.expanding().max()
        
        # 计算回撤
        drawdown = (portfolio_values - peak) / peak
        
        # 返回最大回撤的绝对值
        return float(abs(drawdown.min()))
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """计算风险价值(VaR)"""
        if not 0 < confidence_level < 1:
            raise ValueError("置信水平必须在0和1之间")
            
        # VaR是损失的绝对值
        return float(abs(self.returns.quantile(1 - confidence_level)))
    
    def calculate_cvar(self, confidence_level: float = 0.95) -> float:
        """计算条件风险价值(CVaR)"""
        if not 0 < confidence_level < 1:
            raise ValueError("置信水平必须在0和1之间")
            
        var_value = self.calculate_var(confidence_level)
        
        # 获取超过VaR的损失
        tail_losses = self.returns[self.returns <= -var_value]
        
        if len(tail_losses) > 0:
            return float(abs(tail_losses.mean()))
        else:
            return var_value
    
    def calculate_downside_deviation(self, target_return: float = 0.0, 
                                   annualized: bool = False, 
                                   periods_per_year: int = 252) -> float:
        """计算下行偏差"""
        # 计算低于目标收益率的收益
        downside_returns = self.returns[self.returns < target_return]
        
        if len(downside_returns) == 0:
            return 0.0
            
        # 计算下行偏差
        downside_deviation = float(np.sqrt(((downside_returns - target_return) ** 2).mean()))
        
        if annualized:
            downside_deviation *= np.sqrt(periods_per_year)
            
        return downside_deviation
    
    def calculate_skewness(self) -> float:
        """计算偏度"""
        mean = self.returns.mean()
        std = self.returns.std()
        
        if std == 0:
            return 0.0
            
        skewness = ((self.returns - mean) ** 3).mean() / (std ** 3)
        return float(skewness)
    
    def calculate_kurtosis(self) -> float:
        """计算峰度（超额峰度）"""
        mean = self.returns.mean()
        std = self.returns.std()
        
        if std == 0:
            return 0.0
            
        kurtosis = ((self.returns - mean) ** 4).mean() / (std ** 4) - 3
        return float(kurtosis)


class RiskAdjustedMetrics:
    """风险调整指标计算类"""
    
    def __init__(self, returns: pd.Series):
        """
        初始化风险调整指标计算器
        
        Args:
            returns: 收益率时间序列
        """
        self.returns = returns
        self.risk_metrics = RiskMetrics(returns)
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.03, 
                             periods_per_year: int = 252) -> float:
        """计算夏普比率"""
        # 计算超额收益
        daily_risk_free_rate = risk_free_rate / periods_per_year
        excess_returns = self.returns - daily_risk_free_rate
        
        # 计算夏普比率
        if excess_returns.std() == 0:
            return 0.0
            
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
        return float(sharpe_ratio)
    
    def calculate_sortino_ratio(self, target_return: float = 0.03, 
                              periods_per_year: int = 252) -> float:
        """计算索提诺比率"""
        # 计算超额收益
        daily_target_return = target_return / periods_per_year
        excess_returns = self.returns - daily_target_return
        
        # 计算下行偏差
        downside_deviation = self.risk_metrics.calculate_downside_deviation(
            target_return=daily_target_return
        )
        
        if downside_deviation == 0:
            return 0.0
            
        sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year)
        return float(sortino_ratio)
    
    def calculate_calmar_ratio(self, portfolio_values: pd.Series, 
                             periods_per_year: int = 252) -> float:
        """计算卡玛比率"""
        # 计算年化收益率
        total_return = (1 + self.returns).prod() - 1
        annualized_return = (1 + total_return) ** (periods_per_year / len(self.returns)) - 1
        
        # 计算最大回撤
        max_drawdown = self.risk_metrics.calculate_max_drawdown(portfolio_values)
        
        if max_drawdown == 0:
            return 0.0
            
        calmar_ratio = annualized_return / max_drawdown
        return float(calmar_ratio)
    
    def calculate_information_ratio(self, benchmark_returns: pd.Series, 
                                  periods_per_year: int = 252) -> float:
        """计算信息比率"""
        if len(benchmark_returns) != len(self.returns):
            raise ValueError("基准收益率序列长度与投资组合收益率不匹配")
            
        # 计算主动收益
        active_returns = self.returns - benchmark_returns
        
        # 计算跟踪误差
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
        
        if tracking_error == 0:
            return 0.0
            
        # 计算信息比率
        information_ratio = active_returns.mean() * periods_per_year / tracking_error
        return float(information_ratio)
    
    def calculate_treynor_ratio(self, beta: float, risk_free_rate: float = 0.03, 
                              periods_per_year: int = 252) -> float:
        """计算特雷诺比率"""
        if beta == 0:
            return 0.0
            
        # 计算超额收益
        daily_risk_free_rate = risk_free_rate / periods_per_year
        excess_returns = self.returns - daily_risk_free_rate
        
        # 计算特雷诺比率
        treynor_ratio = excess_returns.mean() * periods_per_year / beta
        return float(treynor_ratio)


class TradingMetrics:
    """交易指标计算类"""
    
    def __init__(self, trades: List[Trade], portfolio_values: pd.Series):
        """
        初始化交易指标计算器
        
        Args:
            trades: 交易记录列表
            portfolio_values: 组合价值时间序列
        """
        self.trades = trades
        self.portfolio_values = portfolio_values
    
    def calculate_turnover_rate(self, period: str = 'annual') -> Union[float, pd.Series]:
        """计算换手率"""
        if not self.trades:
            return 0.0 if period == 'annual' else pd.Series(dtype=float)
            
        # 按时间排序交易
        sorted_trades = sorted(self.trades, key=lambda x: x.timestamp)
        
        # 计算每日交易金额
        trade_amounts = {}
        for trade in sorted_trades:
            trade_date = trade.timestamp.date()
            trade_amount = float(trade.quantity * trade.price)
            
            if trade_date not in trade_amounts:
                trade_amounts[trade_date] = 0
            trade_amounts[trade_date] += trade_amount
        
        # 转换为时间序列
        trade_series = pd.Series(trade_amounts).sort_index()
        
        if period == 'annual':
            # 年化换手率
            total_trade_amount = trade_series.sum()
            average_portfolio_value = self.portfolio_values.mean()
            return float(total_trade_amount / average_portfolio_value)
        
        elif period == 'monthly':
            # 月度换手率
            if isinstance(trade_series.index, pd.DatetimeIndex):
                monthly_turnover = trade_series.groupby(
                    trade_series.index.to_period('M')
                ).sum()
            else:
                # 如果索引不是DatetimeIndex，转换为DatetimeIndex
                trade_series.index = pd.to_datetime(trade_series.index)
                monthly_turnover = trade_series.groupby(
                    trade_series.index.to_period('M')
                ).sum()
            
            # 计算月度平均组合价值
            if isinstance(self.portfolio_values.index, pd.DatetimeIndex):
                monthly_avg_value = self.portfolio_values.groupby(
                    self.portfolio_values.index.to_period('M')
                ).mean()
            else:
                monthly_avg_value = pd.Series([self.portfolio_values.mean()] * len(monthly_turnover),
                                            index=monthly_turnover.index)
            
            return monthly_turnover / monthly_avg_value
        
        else:
            raise ValueError(f"不支持的周期类型: {period}")
    
    def calculate_transaction_cost_analysis(self) -> Dict[str, float]:
        """计算交易成本分析"""
        if not self.trades:
            return {
                'total_commission': 0.0,
                'commission_rate': 0.0,
                'cost_per_trade': 0.0,
                'cost_ratio_to_portfolio': 0.0
            }
        
        # 计算总佣金
        total_commission = sum(float(trade.commission) for trade in self.trades)
        
        # 计算总交易金额
        total_trade_amount = sum(float(trade.quantity * trade.price) for trade in self.trades)
        
        # 计算佣金率
        commission_rate = total_commission / total_trade_amount if total_trade_amount > 0 else 0.0
        
        # 计算平均每笔交易成本
        cost_per_trade = total_commission / len(self.trades)
        
        # 计算成本占组合比例
        average_portfolio_value = self.portfolio_values.mean()
        cost_ratio = total_commission / average_portfolio_value
        
        return {
            'total_commission': total_commission,
            'commission_rate': commission_rate,
            'cost_per_trade': cost_per_trade,
            'cost_ratio_to_portfolio': cost_ratio
        }
    
    def calculate_holding_period_analysis(self) -> Dict[str, float]:
        """计算持仓周期分析"""
        if not self.trades:
            return {
                'average_holding_days': 0.0,
                'median_holding_days': 0.0,
                'max_holding_days': 0.0,
                'min_holding_days': 0.0
            }
        
        # 按股票分组，计算持仓周期
        holding_periods = []
        positions = {}  # symbol -> [buy_trades]
        
        for trade in sorted(self.trades, key=lambda x: x.timestamp):
            symbol = trade.symbol
            
            if symbol not in positions:
                positions[symbol] = []
            
            if trade.trade_type == OrderType.BUY:
                positions[symbol].append(trade)
            elif trade.trade_type == OrderType.SELL and positions[symbol]:
                # 使用FIFO计算持仓周期
                buy_trade = positions[symbol].pop(0)
                holding_days = (trade.timestamp - buy_trade.timestamp).days
                holding_periods.append(holding_days)
        
        if not holding_periods:
            return {
                'average_holding_days': 0.0,
                'median_holding_days': 0.0,
                'max_holding_days': 0.0,
                'min_holding_days': 0.0
            }
        
        return {
            'average_holding_days': float(np.mean(holding_periods)),
            'median_holding_days': float(np.median(holding_periods)),
            'max_holding_days': float(np.max(holding_periods)),
            'min_holding_days': float(np.min(holding_periods))
        }
    
    def calculate_win_loss_analysis(self) -> Dict[str, float]:
        """计算盈亏分析"""
        if not self.trades:
            return {
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'total_trades': 0.0
            }
        
        # 计算每笔交易的盈亏
        trade_pnls = []
        positions = {}  # symbol -> [(quantity, price)]
        
        for trade in sorted(self.trades, key=lambda x: x.timestamp):
            symbol = trade.symbol
            
            if symbol not in positions:
                positions[symbol] = []
            
            if trade.trade_type == OrderType.BUY:
                positions[symbol].append((trade.quantity, float(trade.price)))
            elif trade.trade_type == OrderType.SELL and positions[symbol]:
                # 使用FIFO计算盈亏
                remaining_quantity = trade.quantity
                trade_pnl = 0.0
                
                while remaining_quantity > 0 and positions[symbol]:
                    buy_quantity, buy_price = positions[symbol][0]
                    
                    if buy_quantity <= remaining_quantity:
                        # 完全卖出这个买入记录
                        pnl = buy_quantity * (float(trade.price) - buy_price)
                        trade_pnl += pnl
                        remaining_quantity -= buy_quantity
                        positions[symbol].pop(0)
                    else:
                        # 部分卖出
                        pnl = remaining_quantity * (float(trade.price) - buy_price)
                        trade_pnl += pnl
                        positions[symbol][0] = (buy_quantity - remaining_quantity, buy_price)
                        remaining_quantity = 0
                
                if trade_pnl != 0:  # 只记录有盈亏的交易
                    trade_pnls.append(trade_pnl)
        
        if not trade_pnls:
            return {
                'win_rate': 0.0,
                'profit_loss_ratio': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'total_trades': 0.0
            }
        
        # 分离盈利和亏损交易
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        # 计算指标
        win_rate = len(winning_trades) / len(trade_pnls)
        average_win = np.mean(winning_trades) if winning_trades else 0.0
        average_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
        profit_loss_ratio = average_win / average_loss if average_loss > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'average_win': average_win,
            'average_loss': average_loss,
            'total_trades': float(len(trade_pnls))
        }
    
    def calculate_position_concentration(self) -> Dict[str, float]:
        """计算持仓集中度分析"""
        if not self.trades:
            return {
                'herfindahl_index': 0.0,
                'max_position_weight': 0.0,
                'top_5_concentration': 0.0,
                'effective_positions': 1.0
            }
        
        # 计算最终持仓
        final_positions = {}  # symbol -> quantity
        
        for trade in self.trades:
            symbol = trade.symbol
            
            if symbol not in final_positions:
                final_positions[symbol] = 0
            
            if trade.trade_type == OrderType.BUY:
                final_positions[symbol] += trade.quantity
            elif trade.trade_type == OrderType.SELL:
                final_positions[symbol] -= trade.quantity
        
        # 过滤掉零持仓
        final_positions = {k: v for k, v in final_positions.items() if v > 0}
        
        if not final_positions:
            return {
                'herfindahl_index': 0.0,
                'max_position_weight': 0.0,
                'top_5_concentration': 0.0,
                'effective_positions': 0.0
            }
        
        # 计算持仓权重（这里假设所有股票价格相等，实际应该使用市值权重）
        total_quantity = sum(final_positions.values())
        position_weights = {k: v / total_quantity for k, v in final_positions.items()}
        
        # 计算赫芬达尔指数
        weights = list(position_weights.values())
        herfindahl_index = sum(w ** 2 for w in weights)
        
        # 最大持仓权重
        max_weight = max(weights)
        
        # Top5集中度
        sorted_weights = sorted(weights, reverse=True)
        top5_weights = sorted_weights[:5]
        top5_concentration = sum(top5_weights)
        
        # 有效持仓数
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        return {
            'herfindahl_index': herfindahl_index,
            'max_position_weight': max_weight,
            'top_5_concentration': top5_concentration,
            'effective_positions': effective_positions
        }


class PortfolioMetrics:
    """综合组合指标计算类"""
    
    def __init__(self, returns: pd.Series, portfolio_values: pd.Series, trades: List[Trade]):
        """
        初始化综合组合指标计算器
        
        Args:
            returns: 收益率时间序列
            portfolio_values: 组合价值时间序列
            trades: 交易记录列表
        """
        if len(returns) != len(portfolio_values):
            raise ValueError("收益率序列和组合价值序列长度不匹配")
            
        self.returns = returns
        self.portfolio_values = portfolio_values
        self.trades = trades
        
        # 初始化各个指标计算器
        self.return_metrics = ReturnMetrics(returns)
        self.risk_metrics = RiskMetrics(returns)
        self.risk_adjusted_metrics = RiskAdjustedMetrics(returns)
        self.trading_metrics = TradingMetrics(trades, portfolio_values)
    
    def calculate_comprehensive_metrics(self) -> Dict[str, Dict[str, float]]:
        """计算综合指标"""
        return {
            'return_metrics': {
                'total_return': self.return_metrics.calculate_total_return(),
                'annualized_return': self.return_metrics.calculate_annualized_return(),
            },
            'risk_metrics': {
                'volatility': self.risk_metrics.calculate_volatility(annualized=True),
                'max_drawdown': self.risk_metrics.calculate_max_drawdown(self.portfolio_values),
                'var_95': self.risk_metrics.calculate_var(0.95),
                'cvar_95': self.risk_metrics.calculate_cvar(0.95),
                'skewness': self.risk_metrics.calculate_skewness(),
                'kurtosis': self.risk_metrics.calculate_kurtosis(),
            },
            'risk_adjusted_metrics': {
                'sharpe_ratio': self.risk_adjusted_metrics.calculate_sharpe_ratio(),
                'sortino_ratio': self.risk_adjusted_metrics.calculate_sortino_ratio(),
                'calmar_ratio': self.risk_adjusted_metrics.calculate_calmar_ratio(self.portfolio_values),
            },
            'trading_metrics': {
                **self.trading_metrics.calculate_transaction_cost_analysis(),
                **self.trading_metrics.calculate_win_loss_analysis(),
                **self.trading_metrics.calculate_position_concentration(),
                'annual_turnover': self.trading_metrics.calculate_turnover_rate('annual')
            }
        }
    
    def compare_with_benchmark(self, benchmark_returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """与基准比较"""
        # 计算组合指标
        portfolio_metrics = self.calculate_comprehensive_metrics()
        
        # 计算基准指标
        benchmark_return_metrics = ReturnMetrics(benchmark_returns)
        benchmark_risk_metrics = RiskMetrics(benchmark_returns)
        benchmark_risk_adjusted = RiskAdjustedMetrics(benchmark_returns)
        
        # 基准组合价值（假设初始值与组合相同）
        initial_value = self.portfolio_values.iloc[0]
        benchmark_values = initial_value * (1 + benchmark_return_metrics.calculate_cumulative_returns())
        
        benchmark_metrics = {
            'return_metrics': {
                'total_return': benchmark_return_metrics.calculate_total_return(),
                'annualized_return': benchmark_return_metrics.calculate_annualized_return(),
            },
            'risk_metrics': {
                'volatility': benchmark_risk_metrics.calculate_volatility(annualized=True),
                'max_drawdown': benchmark_risk_metrics.calculate_max_drawdown(benchmark_values),
                'var_95': benchmark_risk_metrics.calculate_var(0.95),
            },
            'risk_adjusted_metrics': {
                'sharpe_ratio': benchmark_risk_adjusted.calculate_sharpe_ratio(),
                'sortino_ratio': benchmark_risk_adjusted.calculate_sortino_ratio(),
                'calmar_ratio': benchmark_risk_adjusted.calculate_calmar_ratio(benchmark_values),
            }
        }
        
        # 计算相对指标
        relative_metrics = {
            'excess_return': (portfolio_metrics['return_metrics']['annualized_return'] - 
                            benchmark_metrics['return_metrics']['annualized_return']),
            'information_ratio': self.risk_adjusted_metrics.calculate_information_ratio(benchmark_returns),
            'tracking_error': (self.returns - benchmark_returns).std() * np.sqrt(252),
        }
        
        return {
            'portfolio_metrics': portfolio_metrics,
            'benchmark_metrics': benchmark_metrics,
            'relative_metrics': relative_metrics
        }
    
    def calculate_rolling_metrics(self, window: int, metric: str) -> pd.Series:
        """计算滚动指标"""
        if metric == 'sharpe_ratio':
            return self.returns.rolling(window=window).apply(
                lambda x: RiskAdjustedMetrics(x).calculate_sharpe_ratio() if len(x) == window else np.nan
            )
        elif metric == 'volatility':
            return self.returns.rolling(window=window).std() * np.sqrt(252)
        elif metric == 'max_drawdown':
            return self.portfolio_values.rolling(window=window).apply(
                lambda x: RiskMetrics(x.pct_change().dropna()).calculate_max_drawdown(x) if len(x) == window else np.nan
            )
        else:
            raise ValueError(f"不支持的滚动指标: {metric}")
    
    def calculate_sector_analysis(self, sector_mapping: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """计算行业分析"""
        sector_analysis = {}
        
        # 按行业分组交易
        sector_trades = {}
        for trade in self.trades:
            sector = sector_mapping.get(trade.symbol, '其他')
            if sector not in sector_trades:
                sector_trades[sector] = []
            sector_trades[sector].append(trade)
        
        # 计算总交易金额
        total_trade_amount = sum(float(trade.quantity * trade.price) for trade in self.trades)
        
        # 计算每个行业的指标
        for sector, trades in sector_trades.items():
            sector_trade_amount = sum(float(trade.quantity * trade.price) for trade in trades)
            weight = sector_trade_amount / total_trade_amount if total_trade_amount > 0 else 0
            
            # 计算收益贡献（这里简化处理）
            sector_returns = 0.0  # 需要更复杂的计算
            
            sector_analysis[sector] = {
                'weight': weight,
                'return_contribution': sector_returns,
                'trade_count': len(trades)
            }
        
        return sector_analysis