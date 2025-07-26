#!/usr/bin/env python3
"""
动态低波筛选器回测验证测试

验证筛选效果是否达到预期指标：
- 年化收益≥6%
- 年化波动≤12%
- 最大回撤≤10%
- 夏普比率≥0.6
- 尾部亏损频次相比基准降低40%+
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from unittest.mock import Mock
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BacktestValidator:
    """回测验证器"""
    
    def __init__(self):
        self.config = self._create_backtest_config()
        self.data_manager = self._create_backtest_data_manager()
        
        # 预期指标阈值
        self.expected_metrics = {
            'annual_return_min': 0.06,      # 年化收益≥6%
            'annual_volatility_max': 0.12,  # 年化波动≤12%
            'max_drawdown_max': 0.10,       # 最大回撤≤10%
            'sharpe_ratio_min': 0.6,        # 夏普比率≥0.6
            'tail_loss_reduction_min': 0.40  # 尾部亏损频次降低≥40%
        }
        
    def _create_backtest_config(self) -> Dict:
        """创建回测配置"""
        return {
            'rolling_windows': [20, 60],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            'garch_window': 250,
            'forecast_horizon': 5,
            'enable_ml_predictor': False,
            'ivol_bad_threshold': 0.3,
            'ivol_good_threshold': 0.6,
            'regime_detection_window': 60,
            'regime_model_type': "HMM",
            'enable_caching': True,
            'cache_expiry_days': 1,
            'parallel_processing': False
        }
    
    def _create_backtest_data_manager(self):
        """创建回测数据管理器"""
        mock_manager = Mock()
        
        # 生成3年回测数据
        dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
        stocks = [f'stock_{i:03d}' for i in range(300)]
        
        np.random.seed(42)
        
        # 生成具有不同风险特征的股票收益率
        stock_returns = self._generate_realistic_stock_returns(dates, stocks)
        
        # 价格数据
        prices = (1 + stock_returns).cumprod() * 100
        
        # 成交量数据
        volumes = pd.DataFrame(
            np.random.lognormal(10, 1, (len(dates), len(stocks))),
            index=dates, columns=stocks
        )
        
        # 因子数据
        factors = ['market', 'size', 'value', 'profitability', 'investment']
        factor_data = pd.DataFrame(
            np.random.normal(0, 0.01, (len(dates), len(factors))),
            index=dates, columns=factors
        )
        
        # 市场数据 - 包含不同波动状态
        market_data = self._generate_market_regime_data(dates)
        
        mock_manager.get_price_data.return_value = prices
        mock_manager.get_volume_data.return_value = volumes
        mock_manager.get_factor_data.return_value = factor_data
        mock_manager.get_market_data.return_value = market_data
        
        return mock_manager
    
    def _generate_realistic_stock_returns(self, dates: pd.DatetimeIndex, stocks: List[str]) -> pd.DataFrame:
        """生成真实的股票收益率数据"""
        n_dates = len(dates)
        n_stocks = len(stocks)
        
        # 为每只股票分配不同的风险特征
        stock_characteristics = {}
        for i, stock in enumerate(stocks):
            stock_characteristics[stock] = {
                'base_vol': np.random.uniform(0.15, 0.50),  # 基础年化波动率15%-50%
                'mean_return': np.random.normal(0.08, 0.05),  # 年化收益率8%±5%
                'beta': np.random.uniform(0.5, 1.5),        # 市场贝塔
                'regime_sensitivity': np.random.uniform(0.5, 2.0)  # 状态敏感性
            }
        
        # 生成市场因子
        market_factor = np.random.normal(0, 0.015, n_dates)  # 市场日收益率
        
        # 生成股票收益率
        returns_data = np.zeros((n_dates, n_stocks))
        
        for i, stock in enumerate(stocks):
            char = stock_characteristics[stock]
            
            # 特异性收益率
            idiosyncratic_returns = np.random.normal(
                char['mean_return'] / 252,  # 日化收益率
                char['base_vol'] / np.sqrt(252),  # 日化波动率
                n_dates
            )
            
            # 市场敞口
            market_exposure = char['beta'] * market_factor
            
            # 总收益率
            total_returns = market_exposure + idiosyncratic_returns
            
            # 添加状态相关的波动率变化
            regime_multiplier = self._get_regime_volatility_multiplier(dates, char['regime_sensitivity'])
            total_returns = total_returns * regime_multiplier
            
            returns_data[:, i] = total_returns
        
        return pd.DataFrame(returns_data, index=dates, columns=stocks)
    
    def _get_regime_volatility_multiplier(self, dates: pd.DatetimeIndex, sensitivity: float) -> np.ndarray:
        """获取状态相关的波动率乘数"""
        n_dates = len(dates)
        multipliers = np.ones(n_dates)
        
        # 定义不同时期的波动率状态
        regime_periods = [
            ('2021-01-01', '2021-06-30', 0.8),   # 低波动期
            ('2021-07-01', '2021-12-31', 1.0),   # 正常波动期
            ('2022-01-01', '2022-06-30', 1.5),   # 高波动期
            ('2022-07-01', '2022-12-31', 1.2),   # 中等波动期
            ('2023-01-01', '2023-06-30', 0.9),   # 低波动期
            ('2023-07-01', '2023-12-31', 1.1),   # 正常波动期
        ]
        
        for start_date, end_date, regime_multiplier in regime_periods:
            mask = (dates >= start_date) & (dates <= end_date)
            # 应用敏感性调整
            adjusted_multiplier = 1 + (regime_multiplier - 1) * sensitivity
            multipliers[mask] = adjusted_multiplier
        
        return multipliers
    
    def _generate_market_regime_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """生成市场状态数据"""
        n_dates = len(dates)
        
        # 基于时间段定义市场状态
        market_returns = np.zeros(n_dates)
        market_volatility = np.zeros(n_dates)
        
        for i, date in enumerate(dates):
            if date < pd.Timestamp('2021-07-01'):
                # 低波动期
                market_returns[i] = np.random.normal(0.0008, 0.012)
                market_volatility[i] = 0.12
            elif date < pd.Timestamp('2022-01-01'):
                # 正常波动期
                market_returns[i] = np.random.normal(0.0005, 0.018)
                market_volatility[i] = 0.18
            elif date < pd.Timestamp('2022-07-01'):
                # 高波动期
                market_returns[i] = np.random.normal(-0.0002, 0.030)
                market_volatility[i] = 0.30
            elif date < pd.Timestamp('2023-01-01'):
                # 中等波动期
                market_returns[i] = np.random.normal(0.0003, 0.022)
                market_volatility[i] = 0.22
            elif date < pd.Timestamp('2023-07-01'):
                # 低波动期
                market_returns[i] = np.random.normal(0.0006, 0.014)
                market_volatility[i] = 0.14
            else:
                # 正常波动期
                market_returns[i] = np.random.normal(0.0004, 0.016)
                market_volatility[i] = 0.16
        
        return pd.DataFrame({
            'returns': market_returns,
            'volatility': market_volatility
        }, index=dates)
    
    def run_backtest_validation(self) -> Dict:
        """运行回测验证"""
        logger.info("开始回测验证测试...")
        
        filter_instance = DynamicLowVolFilter(self.config, self.data_manager)
        
        # 获取数据
        price_data = self.data_manager.get_price_data()
        returns_data = price_data.pct_change().dropna()
        
        # 回测期间
        backtest_start = pd.Timestamp('2022-01-01')
        backtest_end = pd.Timestamp('2023-12-31')
        backtest_dates = pd.date_range(backtest_start, backtest_end, freq='D')
        backtest_dates = backtest_dates.intersection(returns_data.index)
        
        # 运行筛选策略回测
        filtered_portfolio_returns = self._run_filtered_strategy_backtest(
            filter_instance, returns_data, backtest_dates
        )
        
        # 运行基准策略回测（无筛选）
        benchmark_portfolio_returns = self._run_benchmark_strategy_backtest(
            returns_data, backtest_dates
        )
        
        # 计算策略指标
        filtered_metrics = self._calculate_portfolio_metrics(filtered_portfolio_returns)
        benchmark_metrics = self._calculate_portfolio_metrics(benchmark_portfolio_returns)
        
        # 验证指标
        validation_results = self._validate_metrics(filtered_metrics, benchmark_metrics)
        
        # 汇总结果
        results = {
            'backtest_period': f"{backtest_start.strftime('%Y-%m-%d')} 到 {backtest_end.strftime('%Y-%m-%d')}",
            'trading_days': len(backtest_dates),
            'filtered_strategy': filtered_metrics,
            'benchmark_strategy': benchmark_metrics,
            'validation_results': validation_results,
            'overall_passed': validation_results['all_metrics_passed']
        }
        
        self._log_backtest_results(results)
        
        return results
    
    def _run_filtered_strategy_backtest(self, filter_instance: DynamicLowVolFilter, 
                                      returns_data: pd.DataFrame, 
                                      backtest_dates: pd.DatetimeIndex) -> np.ndarray:
        """运行筛选策略回测"""
        logger.info("运行筛选策略回测...")
        
        portfolio_returns = []
        
        for date in backtest_dates:
            if date not in returns_data.index:
                continue
            
            try:
                # 获取筛选掩码
                mask = filter_instance.update_tradable_mask(date)
                
                # 获取当日收益率
                daily_returns = returns_data.loc[date]
                
                # 筛选后的股票
                if mask.sum() > 0:
                    selected_returns = daily_returns[mask]
                    # 等权重投资
                    portfolio_return = selected_returns.mean()
                    portfolio_returns.append(portfolio_return)
                else:
                    # 无可投资股票时收益率为0
                    portfolio_returns.append(0.0)
                    
            except Exception as e:
                logger.warning(f"筛选策略在日期 {date} 失败: {e}")
                portfolio_returns.append(0.0)
        
        return np.array(portfolio_returns)
    
    def _run_benchmark_strategy_backtest(self, returns_data: pd.DataFrame, 
                                       backtest_dates: pd.DatetimeIndex) -> np.ndarray:
        """运行基准策略回测（等权重全市场）"""
        logger.info("运行基准策略回测...")
        
        portfolio_returns = []
        
        for date in backtest_dates:
            if date not in returns_data.index:
                continue
            
            # 获取当日收益率
            daily_returns = returns_data.loc[date]
            
            # 等权重投资所有股票
            portfolio_return = daily_returns.mean()
            portfolio_returns.append(portfolio_return)
        
        return np.array(portfolio_returns)
    
    def _calculate_portfolio_metrics(self, returns: np.ndarray) -> Dict:
        """计算投资组合指标"""
        if len(returns) == 0:
            return {}
        
        # 基础统计
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # 尾部风险指标
        tail_threshold = np.percentile(returns, 5)  # 5%分位数
        tail_losses = returns[returns <= tail_threshold]
        tail_loss_frequency = len(tail_losses) / len(returns)
        
        # VaR和CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # 胜率
        win_rate = np.sum(returns > 0) / len(returns)
        
        # 收益分布统计
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'tail_loss_frequency': tail_loss_frequency,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'total_return': (1 + returns).prod() - 1,
            'volatility_of_volatility': np.std(pd.Series(returns).rolling(20).std().dropna()),
            'trading_days': len(returns)
        }
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """计算偏度"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return np.mean(((returns - mean_return) / std_return) ** 3)
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """计算峰度"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0
        return np.mean(((returns - mean_return) / std_return) ** 4) - 3
    
    def _validate_metrics(self, filtered_metrics: Dict, benchmark_metrics: Dict) -> Dict:
        """验证指标是否达到预期"""
        validation_results = {}
        
        # 1. 年化收益率验证
        annual_return_passed = filtered_metrics['annual_return'] >= self.expected_metrics['annual_return_min']
        validation_results['annual_return'] = {
            'actual': filtered_metrics['annual_return'],
            'expected_min': self.expected_metrics['annual_return_min'],
            'passed': annual_return_passed,
            'vs_benchmark': filtered_metrics['annual_return'] - benchmark_metrics['annual_return']
        }
        
        # 2. 年化波动率验证
        annual_volatility_passed = filtered_metrics['annual_volatility'] <= self.expected_metrics['annual_volatility_max']
        validation_results['annual_volatility'] = {
            'actual': filtered_metrics['annual_volatility'],
            'expected_max': self.expected_metrics['annual_volatility_max'],
            'passed': annual_volatility_passed,
            'vs_benchmark': filtered_metrics['annual_volatility'] - benchmark_metrics['annual_volatility']
        }
        
        # 3. 最大回撤验证
        max_drawdown_passed = filtered_metrics['max_drawdown'] <= self.expected_metrics['max_drawdown_max']
        validation_results['max_drawdown'] = {
            'actual': filtered_metrics['max_drawdown'],
            'expected_max': self.expected_metrics['max_drawdown_max'],
            'passed': max_drawdown_passed,
            'vs_benchmark': filtered_metrics['max_drawdown'] - benchmark_metrics['max_drawdown']
        }
        
        # 4. 夏普比率验证
        sharpe_ratio_passed = filtered_metrics['sharpe_ratio'] >= self.expected_metrics['sharpe_ratio_min']
        validation_results['sharpe_ratio'] = {
            'actual': filtered_metrics['sharpe_ratio'],
            'expected_min': self.expected_metrics['sharpe_ratio_min'],
            'passed': sharpe_ratio_passed,
            'vs_benchmark': filtered_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']
        }
        
        # 5. 尾部亏损频次验证
        tail_loss_reduction = (benchmark_metrics['tail_loss_frequency'] - filtered_metrics['tail_loss_frequency']) / benchmark_metrics['tail_loss_frequency']
        tail_loss_passed = tail_loss_reduction >= self.expected_metrics['tail_loss_reduction_min']
        validation_results['tail_loss_reduction'] = {
            'actual_reduction': tail_loss_reduction,
            'expected_min_reduction': self.expected_metrics['tail_loss_reduction_min'],
            'passed': tail_loss_passed,
            'filtered_frequency': filtered_metrics['tail_loss_frequency'],
            'benchmark_frequency': benchmark_metrics['tail_loss_frequency']
        }
        
        # 综合验证结果
        all_passed = all([
            annual_return_passed,
            annual_volatility_passed,
            max_drawdown_passed,
            sharpe_ratio_passed,
            tail_loss_passed
        ])
        
        validation_results['all_metrics_passed'] = all_passed
        validation_results['passed_count'] = sum([
            annual_return_passed,
            annual_volatility_passed,
            max_drawdown_passed,
            sharpe_ratio_passed,
            tail_loss_passed
        ])
        validation_results['total_count'] = 5
        
        return validation_results
    
    def _log_backtest_results(self, results: Dict):
        """记录回测结果"""
        logger.info("回测验证结果:")
        logger.info(f"  回测期间: {results['backtest_period']}")
        logger.info(f"  交易天数: {results['trading_days']}")
        
        logger.info("筛选策略指标:")
        filtered = results['filtered_strategy']
        logger.info(f"  年化收益率: {filtered['annual_return']:.2%}")
        logger.info(f"  年化波动率: {filtered['annual_volatility']:.2%}")
        logger.info(f"  夏普比率: {filtered['sharpe_ratio']:.3f}")
        logger.info(f"  最大回撤: {filtered['max_drawdown']:.2%}")
        logger.info(f"  尾部亏损频次: {filtered['tail_loss_frequency']:.2%}")
        
        logger.info("基准策略指标:")
        benchmark = results['benchmark_strategy']
        logger.info(f"  年化收益率: {benchmark['annual_return']:.2%}")
        logger.info(f"  年化波动率: {benchmark['annual_volatility']:.2%}")
        logger.info(f"  夏普比率: {benchmark['sharpe_ratio']:.3f}")
        logger.info(f"  最大回撤: {benchmark['max_drawdown']:.2%}")
        logger.info(f"  尾部亏损频次: {benchmark['tail_loss_frequency']:.2%}")
        
        logger.info("验证结果:")
        validation = results['validation_results']
        logger.info(f"  年化收益率: {'通过' if validation['annual_return']['passed'] else '失败'} "
                   f"({validation['annual_return']['actual']:.2%} vs ≥{validation['annual_return']['expected_min']:.2%})")
        logger.info(f"  年化波动率: {'通过' if validation['annual_volatility']['passed'] else '失败'} "
                   f"({validation['annual_volatility']['actual']:.2%} vs ≤{validation['annual_volatility']['expected_max']:.2%})")
        logger.info(f"  最大回撤: {'通过' if validation['max_drawdown']['passed'] else '失败'} "
                   f"({validation['max_drawdown']['actual']:.2%} vs ≤{validation['max_drawdown']['expected_max']:.2%})")
        logger.info(f"  夏普比率: {'通过' if validation['sharpe_ratio']['passed'] else '失败'} "
                   f"({validation['sharpe_ratio']['actual']:.3f} vs ≥{validation['sharpe_ratio']['expected_min']:.3f})")
        logger.info(f"  尾部亏损降低: {'通过' if validation['tail_loss_reduction']['passed'] else '失败'} "
                   f"({validation['tail_loss_reduction']['actual_reduction']:.1%} vs ≥{validation['tail_loss_reduction']['expected_min_reduction']:.1%})")
        
        logger.info(f"总体结果: {'通过' if results['overall_passed'] else '失败'} "
                   f"({validation['passed_count']}/{validation['total_count']})")
    
    def generate_backtest_report(self, results: Dict) -> str:
        """生成回测报告"""
        report_content = f"""# 动态低波筛选器回测验证报告

## 回测概要
- 回测期间: {results['backtest_period']}
- 交易天数: {results['trading_days']}
- 验证结果: {'通过' if results['overall_passed'] else '失败'} ({results['validation_results']['passed_count']}/{results['validation_results']['total_count']})

## 策略表现对比

### 筛选策略指标
"""
        
        filtered = results['filtered_strategy']
        report_content += f"""- 年化收益率: {filtered['annual_return']:.2%}
- 年化波动率: {filtered['annual_volatility']:.2%}
- 夏普比率: {filtered['sharpe_ratio']:.3f}
- 最大回撤: {filtered['max_drawdown']:.2%}
- 尾部亏损频次: {filtered['tail_loss_frequency']:.2%}
- 胜率: {filtered['win_rate']:.2%}
- VaR(95%): {filtered['var_95']:.2%}
- CVaR(95%): {filtered['cvar_95']:.2%}
- 收益偏度: {filtered['skewness']:.3f}
- 收益峰度: {filtered['kurtosis']:.3f}

### 基准策略指标
"""
        
        benchmark = results['benchmark_strategy']
        report_content += f"""- 年化收益率: {benchmark['annual_return']:.2%}
- 年化波动率: {benchmark['annual_volatility']:.2%}
- 夏普比率: {benchmark['sharpe_ratio']:.3f}
- 最大回撤: {benchmark['max_drawdown']:.2%}
- 尾部亏损频次: {benchmark['tail_loss_frequency']:.2%}
- 胜率: {benchmark['win_rate']:.2%}
- VaR(95%): {benchmark['var_95']:.2%}
- CVaR(95%): {benchmark['cvar_95']:.2%}
- 收益偏度: {benchmark['skewness']:.3f}
- 收益峰度: {benchmark['kurtosis']:.3f}

## 指标验证结果

"""
        
        validation = results['validation_results']
        
        # 年化收益率
        report_content += f"""### 年化收益率
- 实际值: {validation['annual_return']['actual']:.2%}
- 要求: ≥ {validation['annual_return']['expected_min']:.2%}
- 结果: {'✅ 通过' if validation['annual_return']['passed'] else '❌ 失败'}
- 相对基准: {validation['annual_return']['vs_benchmark']:+.2%}

"""
        
        # 年化波动率
        report_content += f"""### 年化波动率
- 实际值: {validation['annual_volatility']['actual']:.2%}
- 要求: ≤ {validation['annual_volatility']['expected_max']:.2%}
- 结果: {'✅ 通过' if validation['annual_volatility']['passed'] else '❌ 失败'}
- 相对基准: {validation['annual_volatility']['vs_benchmark']:+.2%}

"""
        
        # 最大回撤
        report_content += f"""### 最大回撤
- 实际值: {validation['max_drawdown']['actual']:.2%}
- 要求: ≤ {validation['max_drawdown']['expected_max']:.2%}
- 结果: {'✅ 通过' if validation['max_drawdown']['passed'] else '❌ 失败'}
- 相对基准: {validation['max_drawdown']['vs_benchmark']:+.2%}

"""
        
        # 夏普比率
        report_content += f"""### 夏普比率
- 实际值: {validation['sharpe_ratio']['actual']:.3f}
- 要求: ≥ {validation['sharpe_ratio']['expected_min']:.3f}
- 结果: {'✅ 通过' if validation['sharpe_ratio']['passed'] else '❌ 失败'}
- 相对基准: {validation['sharpe_ratio']['vs_benchmark']:+.3f}

"""
        
        # 尾部亏损降低
        report_content += f"""### 尾部亏损频次降低
- 实际降低: {validation['tail_loss_reduction']['actual_reduction']:.1%}
- 要求: ≥ {validation['tail_loss_reduction']['expected_min_reduction']:.1%}
- 结果: {'✅ 通过' if validation['tail_loss_reduction']['passed'] else '❌ 失败'}
- 筛选策略频次: {validation['tail_loss_reduction']['filtered_frequency']:.2%}
- 基准策略频次: {validation['tail_loss_reduction']['benchmark_frequency']:.2%}

## 结论

"""
        
        if results['overall_passed']:
            report_content += """✅ **所有指标验证通过**

动态低波筛选器成功达到了预期的风险收益指标：
- 在保持较高收益的同时显著降低了波动率
- 有效控制了最大回撤风险
- 大幅减少了尾部亏损事件
- 整体风险调整后收益表现优异

"""
        else:
            failed_metrics = []
            if not validation['annual_return']['passed']:
                failed_metrics.append("年化收益率")
            if not validation['annual_volatility']['passed']:
                failed_metrics.append("年化波动率")
            if not validation['max_drawdown']['passed']:
                failed_metrics.append("最大回撤")
            if not validation['sharpe_ratio']['passed']:
                failed_metrics.append("夏普比率")
            if not validation['tail_loss_reduction']['passed']:
                failed_metrics.append("尾部亏损降低")
            
            report_content += f"""❌ **部分指标未达到预期**

未通过的指标: {', '.join(failed_metrics)}

建议优化方向：
- 调整筛选阈值参数
- 优化市场状态检测算法
- 改进GARCH波动率预测模型
- 增强IVOL约束筛选逻辑

"""
        
        report_content += """## 建议

1. **参数优化**: 根据回测结果调整关键参数
2. **模型改进**: 考虑引入更先进的预测模型
3. **风险管理**: 进一步完善风险控制机制
4. **定期评估**: 建立定期回测评估机制

"""
        
        return report_content


def main():
    """主函数"""
    logger.info("开始动态低波筛选器回测验证...")
    
    # 创建回测验证器
    validator = BacktestValidator()
    
    # 运行回测验证
    results = validator.run_backtest_validation()
    
    # 生成回测报告
    report_content = validator.generate_backtest_report(results)
    
    # 保存报告
    report_path = 'reports/backtest_validation_report.md'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"回测验证报告已保存到: {report_path}")
    
    # 返回测试结果
    if results['overall_passed']:
        logger.info("回测验证通过！")
        return 0
    else:
        logger.error("回测验证失败！")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)