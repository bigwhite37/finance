"""
回测引擎测试
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import BacktestEngine, PerformanceAnalyzer, ComfortabilityMetrics


class TestBacktestEngine(unittest.TestCase):
    """回测引擎测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'initial_capital': 1000000,
            'transaction_cost': 0.001,
            'slippage': 0.0001,
            'commission': 0.0005,
            'risk_free_rate': 0.03
        }
        
        self.backtest_engine = BacktestEngine(self.config)
        
        # 创建模拟收益率数据
        np.random.seed(42)
        self.returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),  # 模拟一年的日收益率
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.backtest_engine.initial_capital, 1000000)
        self.assertEqual(self.backtest_engine.transaction_cost, 0.001)
        self.assertEqual(self.backtest_engine.slippage, 0.0001)
        self.assertEqual(self.backtest_engine.commission, 0.0005)
        
        # 验证分析器初始化
        self.assertIsInstance(self.backtest_engine.performance_analyzer, PerformanceAnalyzer)
        self.assertIsInstance(self.backtest_engine.comfort_metrics, ComfortabilityMetrics)
    
    def test_initialize_backtest(self):
        """测试回测初始化"""
        # 添加一些历史数据
        self.backtest_engine.portfolio_history.append({'test': 'data'})
        self.backtest_engine.trade_history.append({'test': 'trade'})
        
        # 初始化回测
        self.backtest_engine._initialize_backtest()
        
        # 验证历史数据被清空
        self.assertEqual(len(self.backtest_engine.portfolio_history), 0)
        self.assertEqual(len(self.backtest_engine.trade_history), 0)
        self.assertEqual(len(self.backtest_engine.daily_returns), 0)
        self.assertEqual(len(self.backtest_engine.positions_history), 0)
    
    def test_record_step(self):
        """测试步骤记录"""
        # 模拟动作和信息
        action = np.array([0.2, 0.3, 0.5])
        state = np.array([1.0, 2.0, 3.0])
        info = {
            'portfolio_value': 1.05,
            'total_return': 0.05,
            'max_drawdown': 0.02,
            'volatility': 0.12,
            'sharpe_ratio': 0.8,
            'cash': 0.0
        }
        step = 10
        
        # 记录步骤
        self.backtest_engine._record_step(action, state, info, step)
        
        # 验证记录
        self.assertEqual(len(self.backtest_engine.portfolio_history), 1)
        self.assertEqual(len(self.backtest_engine.positions_history), 1)
        
        # 验证记录内容
        portfolio_record = self.backtest_engine.portfolio_history[0]
        self.assertEqual(portfolio_record['step'], step)
        self.assertEqual(portfolio_record['portfolio_value'], 1.05)
        
        positions_record = self.backtest_engine.positions_history[0]
        self.assertEqual(positions_record['step'], step)
        np.testing.assert_array_equal(positions_record['weights'], action)
    
    def test_calculate_trading_stats(self):
        """测试交易统计计算"""
        # 添加一些位置历史数据
        for i in range(5):
            self.backtest_engine.positions_history.append({
                'step': i,
                'weights': np.random.uniform(-0.1, 0.1, 3),
                'total_leverage': 0.3,
                'num_positions': 3
            })
        
        # 计算交易统计
        stats = self.backtest_engine._calculate_trading_stats()
        
        # 验证统计结果
        expected_keys = [
            'avg_leverage', 'max_leverage', 'avg_positions', 
            'avg_turnover', 'total_trades'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], (int, float))
        
        self.assertEqual(stats['avg_leverage'], 0.3)
        self.assertEqual(stats['avg_positions'], 3)
    
    def test_calculate_risk_metrics(self):
        """测试风险指标计算"""
        risk_metrics = self.backtest_engine._calculate_risk_metrics(self.returns)
        
        # 验证指标存在
        expected_keys = [
            'volatility', 'downside_volatility', 'var_95', 'var_99',
            'cvar_95', 'skewness', 'kurtosis', 'max_consecutive_losses',
            'loss_days_ratio'
        ]
        
        for key in expected_keys:
            self.assertIn(key, risk_metrics)
            self.assertIsInstance(risk_metrics[key], (int, float))
        
        # 验证波动率计算
        expected_vol = self.returns.std() * np.sqrt(252)
        self.assertAlmostEqual(risk_metrics['volatility'], expected_vol, places=6)
    
    def test_max_consecutive_negative(self):
        """测试最大连续亏损计算"""
        # 创建有连续亏损的收益序列
        test_returns = pd.Series([0.01, -0.01, -0.02, -0.01, 0.01, -0.005, 0.02])
        
        max_consecutive = self.backtest_engine._max_consecutive_negative(test_returns)
        
        # 应该是3个连续负收益
        self.assertEqual(max_consecutive, 3)
    
    def test_generate_backtest_report(self):
        """测试回测报告生成"""
        # 创建模拟结果数据
        results = {
            'backtest_summary': {
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'total_steps': 252,
                'final_portfolio_value': 1.08
            },
            'performance_metrics': {
                '总收益率': 0.08,
                '年化收益率': 0.08,
                '年化波动率': 0.12,
                '夏普比率': 0.67,
                '最大回撤': -0.05
            },
            'comfort_metrics': {
                '月度最大回撤': -0.03,
                '连续亏损天数': 3,
                '下跌日占比': 0.45,
                '日VaR_95%': -0.018
            },
            'trading_stats': {
                'avg_leverage': 1.0,
                'max_leverage': 1.2,
                'avg_positions': 5,
                'avg_turnover': 0.15
            },
            'risk_metrics': {
                'downside_volatility': 0.08,
                'var_99': -0.025,
                'cvar_95': -0.022,
                'skewness': -0.1,
                'kurtosis': 0.2
            }
        }
        
        # 生成报告
        report = self.backtest_engine.generate_backtest_report(results)
        
        # 验证报告包含关键信息
        self.assertIn('回测概要', report)
        self.assertIn('收益指标', report)
        self.assertIn('心理舒适度指标', report)
        self.assertIn('交易统计', report)
        self.assertIn('风险指标', report)
        
        # 验证具体数值出现在报告中
        self.assertIn('8.00%', report)  # 年化收益率
        self.assertIn('0.67', report)   # 夏普比率


class TestPerformanceAnalyzer(unittest.TestCase):
    """绩效分析器测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {'risk_free_rate': 0.03}
        self.analyzer = PerformanceAnalyzer(self.config)
        
        # 创建测试收益率数据
        np.random.seed(42)
        self.returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
    
    def test_calculate_return_metrics(self):
        """测试收益指标计算"""
        metrics = self.analyzer._calculate_return_metrics(self.returns)
        
        expected_keys = ['累计收益率', '年化收益率', '年化波动率', '总交易天数']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
        
        # 验证交易天数
        self.assertEqual(metrics['总交易天数'], 252)
        
        # 验证年化波动率计算
        expected_vol = self.returns.std() * np.sqrt(252)
        self.assertAlmostEqual(metrics['年化波动率'], expected_vol, places=6)
    
    def test_calculate_risk_adjusted_metrics(self):
        """测试风险调整指标计算"""
        metrics = self.analyzer._calculate_risk_adjusted_metrics(self.returns)
        
        expected_keys = ['夏普比率', '最大回撤', '卡玛比率']
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
    
    def test_calculate_max_drawdown(self):
        """测试最大回撤计算"""
        # 创建有明显回撤的收益序列
        test_returns = pd.Series([0.05, 0.03, -0.08, -0.05, 0.02, 0.04])
        
        max_dd = self.analyzer.calculate_max_drawdown(test_returns)
        
        # 验证回撤为负值
        self.assertLess(max_dd, 0)
        self.assertIsInstance(max_dd, float)
    
    def test_calculate_alpha_beta(self):
        """测试Alpha和Beta计算"""
        # 创建基准收益率
        benchmark = pd.Series(
            np.random.normal(0.0003, 0.012, 252),
            index=self.returns.index
        )
        
        alpha, beta = self.analyzer.calculate_alpha_beta(self.returns, benchmark)
        
        # 验证返回值
        self.assertIsInstance(alpha, float)
        self.assertIsInstance(beta, float)
        
        # Beta通常应该在合理范围内
        self.assertGreater(beta, -5)
        self.assertLess(beta, 5)
    
    def test_calculate_win_rate(self):
        """测试胜率计算"""
        win_metrics = self.analyzer.calculate_win_rate(self.returns)
        
        expected_keys = ['胜率', '平均盈利', '平均亏损', '盈亏比']
        for key in expected_keys:
            self.assertIn(key, win_metrics)
            self.assertIsInstance(win_metrics[key], (int, float))
        
        # 胜率应该在0-1之间
        self.assertGreaterEqual(win_metrics['胜率'], 0)
        self.assertLessEqual(win_metrics['胜率'], 1)


class TestComfortabilityMetrics(unittest.TestCase):
    """心理舒适度测试"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'monthly_dd_threshold': 0.05,
            'max_consecutive_losses': 5,
            'max_loss_ratio': 0.4,
            'var_95_threshold': 0.01
        }
        
        self.comfort_metrics = ComfortabilityMetrics(self.config)
        
        # 创建测试收益率数据
        np.random.seed(42)
        self.returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=pd.date_range('2023-01-01', periods=252, freq='D')
        )
    
    def test_max_consecutive_losses(self):
        """测试最大连续亏损计算"""
        # 创建有连续亏损的序列
        test_returns = pd.Series([0.01, -0.01, -0.02, -0.01, 0.01, -0.005, -0.01, 0.02])
        
        max_losses = self.comfort_metrics.max_consecutive_losses(test_returns)
        
        # 应该是3个连续负收益
        self.assertEqual(max_losses, 3)
    
    def test_monthly_max_drawdown(self):
        """测试月度最大回撤"""
        monthly_dd = self.comfort_metrics.monthly_max_drawdown(self.returns, window=21)
        
        # 应该返回负值（回撤）
        self.assertLessEqual(monthly_dd, 0)
        self.assertIsInstance(monthly_dd, float)
    
    def test_rolling_sharpe(self):
        """测试滚动夏普比率"""
        rolling_sharpe = self.comfort_metrics.rolling_sharpe(self.returns, window=60)
        
        # 验证返回序列
        self.assertIsInstance(rolling_sharpe, pd.Series)
        self.assertLess(len(rolling_sharpe), len(self.returns))  # 滚动计算会减少数据点
    
    def test_calculate_composite_score(self):
        """测试综合舒适度得分计算"""
        # 创建模拟指标
        metrics = {
            '月度最大回撤': -0.03,
            '连续亏损天数': 3,
            '下跌日占比': 0.45,
            '日VaR_95%': -0.008,
            '波动率稳定性': 0.005,
            '12月滚动夏普': 0.6
        }
        
        score = self.comfort_metrics._calculate_composite_score(metrics)
        
        # 得分应该在0-100之间
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIsInstance(score, float)
    
    def test_calculate_metrics(self):
        """测试完整指标计算"""
        metrics = self.comfort_metrics.calculate_metrics(self.returns)
        
        # 验证关键指标存在
        expected_keys = [
            '月度最大回撤', '连续亏损天数', '下跌日占比', 
            '日VaR_95%', '综合舒适度得分'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
        
        # 验证综合得分在合理范围内
        self.assertGreaterEqual(metrics['综合舒适度得分'], 0)
        self.assertLessEqual(metrics['综合舒适度得分'], 100)


if __name__ == '__main__':
    unittest.main()