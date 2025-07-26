#!/usr/bin/env python3
"""
回测测试工具模块

提供回测相关的通用工具和数据生成功能。
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List
from unittest.mock import Mock

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BacktestDataGenerator:
    """回测数据生成器"""
    
    @staticmethod
    def generate_realistic_stock_returns(dates: pd.DatetimeIndex, stocks: List[str]) -> pd.DataFrame:
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
            regime_multiplier = BacktestDataGenerator._get_regime_volatility_multiplier(dates, char['regime_sensitivity'])
            total_returns = total_returns * regime_multiplier
            
            returns_data[:, i] = total_returns
        
        return pd.DataFrame(returns_data, index=dates, columns=stocks)
    
    @staticmethod
    def _get_regime_volatility_multiplier(dates: pd.DatetimeIndex, sensitivity: float) -> np.ndarray:
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
    
    @staticmethod
    def generate_market_regime_data(dates: pd.DatetimeIndex) -> pd.DataFrame:
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
            'market_return': market_returns,
            'market_volatility': market_volatility
        }, index=dates)


class BacktestMetricsCalculator:
    """回测指标计算器"""
    
    @staticmethod
    def calculate_portfolio_metrics(returns: np.ndarray) -> Dict:
        """计算组合性能指标"""
        if len(returns) == 0:
            return {
                'annual_return': 0.0,
                'annual_volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'calmar_ratio': 0.0,
                'sortino_ratio': 0.0,
                'total_return': 0.0,
                'win_rate': 0.0,
                'tail_loss_frequency': 0.0
            }
        
        # 基础指标
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # 年化指标
        annual_return = mean_return * 252
        annual_volatility = volatility * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0.0
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.abs(np.min(drawdowns))
        
        # 偏度和峰度
        skewness = BacktestMetricsCalculator._calculate_skewness(returns)
        kurtosis = BacktestMetricsCalculator._calculate_kurtosis(returns)
        
        # VaR和CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        
        # Calmar比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Sortino比率
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = annual_return / downside_volatility if downside_volatility > 0 else 0.0
        
        # 总收益率
        total_return = np.prod(1 + returns) - 1
        
        # 胜率
        win_rate = np.sum(returns > 0) / len(returns)
        
        # 尾部亏损频次（下跌超过2%的频次）
        tail_loss_frequency = np.sum(returns < -0.02) / len(returns)
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'total_return': total_return,
            'win_rate': win_rate,
            'tail_loss_frequency': tail_loss_frequency
        }
    
    @staticmethod
    def _calculate_skewness(returns: np.ndarray) -> float:
        """计算偏度"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return np.mean(((returns - mean_return) / std_return) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(returns: np.ndarray) -> float:
        """计算峰度"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return np.mean(((returns - mean_return) / std_return) ** 4) - 3.0


class BacktestConfigFactory:
    """回测配置工厂"""
    
    @staticmethod
    def create_backtest_config() -> Dict:
        """创建回测配置"""
        return {
            # 滚动分位筛选配置
            'rolling_windows': [20, 60],
            'percentile_thresholds': {"低": 0.4, "中": 0.3, "高": 0.2},
            
            # GARCH预测配置
            'garch_window': 250,
            'forecast_horizon': 5,
            'enable_ml_predictor': False,
            
            # IVOL约束配置
            'ivol_bad_threshold': 0.3,
            'ivol_good_threshold': 0.6,
            
            # 市场状态检测配置
            'regime_detection_window': 60,
            'regime_model_type': "HMM",
            
            # 性能优化配置
            'enable_caching': False,  # 回测时禁用缓存
            'cache_expiry_days': 1,
            'parallel_processing': False  # 回测时单线程
        }
    
    @staticmethod
    def create_benchmark_data_manager():
        """创建基准数据管理器Mock"""
        mock_manager = Mock()
        
        # 生成回测期间数据
        dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
        dates = dates[dates.weekday < 5]  # 只保留工作日
        
        stocks = [f'stock_{i:03d}' for i in range(100)]  # 100只股票
        
        # 生成价格数据
        returns_data = BacktestDataGenerator.generate_realistic_stock_returns(dates, stocks)
        price_data = (1 + returns_data).cumprod() * 100  # 初始价格100
        
        # 生成成交量数据
        volume_data = pd.DataFrame(
            np.random.lognormal(10, 1, price_data.shape),
            index=dates, columns=stocks
        )
        
        # 生成市场数据
        market_data = BacktestDataGenerator.generate_market_regime_data(dates)
        
        # 配置Mock返回值
        mock_manager.get_price_data.return_value = price_data
        mock_manager.get_volume_data.return_value = volume_data
        mock_manager.get_market_data.return_value = market_data
        
        return mock_manager


class BacktestExpectedMetrics:
    """回测预期指标"""
    
    EXPECTED_METRICS = {
        'annual_return_min': 0.06,      # 年化收益≥6%
        'annual_volatility_max': 0.12,  # 年化波动≤12%
        'max_drawdown_max': 0.10,       # 最大回撤≤10%
        'sharpe_ratio_min': 0.6,        # 夏普比率≥0.6
        'tail_loss_reduction_min': 0.40  # 尾部亏损频次降低≥40%
    }