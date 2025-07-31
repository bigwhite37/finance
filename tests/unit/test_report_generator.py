"""
回测报告生成模块的单元测试
测试HTML报告生成和可视化图表，收益曲线、持仓分析和风险分解报告，报告的完整性和可读性
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional
from decimal import Decimal
import tempfile
import os
from pathlib import Path

from src.rl_trading_system.evaluation.report_generator import (
    ReportGenerator,
    HTMLReportGenerator,
    ChartGenerator,
    ReportData
)
from src.rl_trading_system.backtest.multi_frequency_backtest import Trade, OrderType
from src.rl_trading_system.evaluation.performance_metrics import PortfolioMetrics


class TestReportData:
    """报告数据测试类"""

    @pytest.fixture
    def sample_report_data(self):
        """创建样本报告数据"""
        # 创建样本数据
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)
        
        # 组合价值
        portfolio_values = [1000000]
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        portfolio_values = pd.Series(portfolio_values[1:], index=dates)
        
        # 基准收益率
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=dates
        )
        
        # 交易记录
        trades = [
            Trade("000001.SZ", OrderType.BUY, 1000, Decimal("10.00"), datetime(2023, 1, 2), Decimal("10.00")),
            Trade("000001.SZ", OrderType.SELL, 500, Decimal("11.00"), datetime(2023, 6, 15), Decimal("5.50")),
            Trade("000002.SZ", OrderType.BUY, 2000, Decimal("5.00"), datetime(2023, 3, 20), Decimal("10.00")),
            Trade("000001.SZ", OrderType.SELL, 500, Decimal("10.50"), datetime(2023, 9, 1), Decimal("5.25")),
        ]
        
        # 持仓数据
        positions_data = {
            '000001.SZ': {'quantity': 0, 'market_value': 0.0, 'weight': 0.0},
            '000002.SZ': {'quantity': 2000, 'market_value': 11000.0, 'weight': 0.01}
        }
        
        return ReportData(
            returns=pd.Series(returns, index=dates),
            portfolio_values=portfolio_values,
            benchmark_returns=benchmark_returns,
            trades=trades,
            positions=positions_data,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 9, 9),
            initial_capital=1000000.0
        )

    def test_report_data_creation(self, sample_report_data):
        """测试报告数据创建"""
        assert isinstance(sample_report_data.returns, pd.Series)
        assert isinstance(sample_report_data.portfolio_values, pd.Series)
        assert isinstance(sample_report_data.benchmark_returns, pd.Series)
        assert isinstance(sample_report_data.trades, list)
        assert isinstance(sample_report_data.positions, dict)
        assert sample_report_data.initial_capital == 1000000.0

    def test_report_data_validation(self):
        """测试报告数据验证"""
        # 测试空收益率序列错误
        with pytest.raises(ValueError, match="收益率序列不能为空"):
            ReportData(
                returns=pd.Series([]),
                portfolio_values=pd.Series([1000000]),
                benchmark_returns=pd.Series([0.01]),
                trades=[],
                positions={},
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                initial_capital=1000000.0
            )

        # 测试长度不匹配错误
        with pytest.raises(ValueError, match="收益率序列和组合价值序列长度不匹配"):
            ReportData(
                returns=pd.Series([0.01, 0.02]),
                portfolio_values=pd.Series([1000000]),  # 长度不匹配
                benchmark_returns=pd.Series([0.01, 0.02]),
                trades=[],
                positions={},
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                initial_capital=1000000.0
            )

    def test_report_data_metrics_calculation(self, sample_report_data):
        """测试报告数据指标计算"""
        metrics = sample_report_data.calculate_metrics()
        
        # 验证指标结构
        assert 'return_metrics' in metrics
        assert 'risk_metrics' in metrics
        assert 'risk_adjusted_metrics' in metrics
        assert 'trading_metrics' in metrics
        
        # 验证具体指标
        assert 'total_return' in metrics['return_metrics']
        assert 'annualized_return' in metrics['return_metrics']
        assert 'volatility' in metrics['risk_metrics']
        assert 'max_drawdown' in metrics['risk_metrics']
        assert 'sharpe_ratio' in metrics['risk_adjusted_metrics']


class TestChartGenerator:
    """图表生成器测试类"""

    @pytest.fixture
    def chart_generator(self):
        """创建图表生成器"""
        return ChartGenerator()

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
        portfolio_values = pd.Series(np.random.uniform(900000, 1100000, 100), index=dates)
        benchmark_values = pd.Series(np.random.uniform(950000, 1050000, 100), index=dates)
        
        return {
            'returns': returns,
            'portfolio_values': portfolio_values,
            'benchmark_values': benchmark_values,
            'dates': dates
        }

    def test_chart_generator_initialization(self, chart_generator):
        """测试图表生成器初始化"""
        assert chart_generator.figure_size == (12, 8)
        assert chart_generator.style == 'seaborn-v0_8'

    def test_returns_chart_generation(self, chart_generator, sample_data):
        """测试收益率图表生成"""
        chart_html = chart_generator.generate_returns_chart(
            portfolio_values=sample_data['portfolio_values'],
            benchmark_values=sample_data['benchmark_values']
        )
        
        # 验证返回HTML字符串
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        assert '<div>' in chart_html  # Plotly生成的HTML应包含div标签

    def test_drawdown_chart_generation(self, chart_generator, sample_data):
        """测试回撤图表生成"""
        chart_html = chart_generator.generate_drawdown_chart(
            portfolio_values=sample_data['portfolio_values']
        )
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        assert '<div>' in chart_html

    def test_rolling_metrics_chart_generation(self, chart_generator, sample_data):
        """测试滚动指标图表生成"""
        # 创建滚动夏普比率数据
        rolling_sharpe = pd.Series(
            np.random.normal(1.5, 0.5, 70),  # 30天窗口，所以数据点较少
            index=sample_data['dates'][30:]
        )
        
        chart_html = chart_generator.generate_rolling_metrics_chart(
            rolling_sharpe, metric_name='夏普比率'
        )
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        assert '<div>' in chart_html

    def test_position_analysis_chart_generation(self, chart_generator):
        """测试持仓分析图表生成"""
        positions_data = {
            '000001.SZ': {'weight': 0.4, 'market_value': 400000},
            '000002.SZ': {'weight': 0.3, 'market_value': 300000},
            '000003.SZ': {'weight': 0.2, 'market_value': 200000},
            '现金': {'weight': 0.1, 'market_value': 100000}
        }
        
        chart_html = chart_generator.generate_position_analysis_chart(positions_data)
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        assert '<div>' in chart_html

    def test_monthly_returns_heatmap_generation(self, chart_generator, sample_data):
        """测试月度收益率热力图生成"""
        # 创建月度收益率数据
        monthly_returns = sample_data['returns'].groupby(
            sample_data['returns'].index.to_period('M')
        ).apply(lambda x: (1 + x).prod() - 1)
        
        # 转换为DataFrame格式
        monthly_df = pd.DataFrame({
            'return': monthly_returns.values,
            'year': [p.year for p in monthly_returns.index],
            'month': [p.month for p in monthly_returns.index]
        })
        
        chart_html = chart_generator.generate_monthly_returns_heatmap(monthly_df)
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        assert '<div>' in chart_html

    def test_risk_metrics_radar_chart_generation(self, chart_generator):
        """测试风险指标雷达图生成"""
        risk_metrics = {
            'volatility': 0.15,
            'max_drawdown': 0.08,
            'var_95': 0.03,
            'skewness': -0.1,
            'kurtosis': 0.2
        }
        
        chart_html = chart_generator.generate_risk_metrics_radar_chart(risk_metrics)
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        assert '<div>' in chart_html

    def test_trading_analysis_chart_generation(self, chart_generator):
        """测试交易分析图表生成"""
        trading_metrics = {
            'win_rate': 0.65,
            'profit_loss_ratio': 1.8,
            'average_win': 0.025,
            'average_loss': 0.015,
            'total_trades': 20
        }
        
        chart_html = chart_generator.generate_trading_analysis_chart(trading_metrics)
        
        assert isinstance(chart_html, str)
        assert len(chart_html) > 0
        assert '<div>' in chart_html

    def test_invalid_data_handling(self, chart_generator):
        """测试无效数据处理"""
        # 测试空数据
        with pytest.raises(ValueError, match="数据不能为空"):
            chart_generator.generate_returns_chart(pd.Series([]), pd.Series([]))

        # 测试长度不匹配的数据
        with pytest.raises(ValueError, match="组合价值和基准价值长度不匹配"):
            chart_generator.generate_returns_chart(
                pd.Series([1000000, 1010000]),
                pd.Series([1000000])  # 长度不匹配
            )


class TestHTMLReportGenerator:
    """HTML报告生成器测试类"""

    @pytest.fixture
    def html_generator(self):
        """创建HTML报告生成器"""
        return HTMLReportGenerator()

    @pytest.fixture
    def sample_report_data(self):
        """创建样本报告数据"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        
        portfolio_values = [1000000]
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        portfolio_values = pd.Series(portfolio_values[1:], index=dates)
        
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 100),
            index=dates
        )
        
        trades = [
            Trade("000001.SZ", OrderType.BUY, 1000, Decimal("10.00"), datetime(2023, 1, 2), Decimal("10.00")),
            Trade("000002.SZ", OrderType.BUY, 2000, Decimal("5.00"), datetime(2023, 2, 15), Decimal("10.00")),
        ]
        
        positions = {
            '000001.SZ': {'quantity': 1000, 'market_value': 11000, 'weight': 0.4},
            '000002.SZ': {'quantity': 2000, 'market_value': 10000, 'weight': 0.35},
            '现金': {'quantity': 0, 'market_value': 7000, 'weight': 0.25}
        }
        
        return ReportData(
            returns=pd.Series(returns, index=dates),
            portfolio_values=portfolio_values,
            benchmark_returns=benchmark_returns,
            trades=trades,
            positions=positions,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 4, 10),
            initial_capital=1000000.0
        )

    def test_html_generator_initialization(self, html_generator):
        """测试HTML生成器初始化"""
        assert html_generator.template_dir is not None
        assert isinstance(html_generator.chart_generator, ChartGenerator)

    def test_summary_section_generation(self, html_generator, sample_report_data):
        """测试摘要部分生成"""
        metrics = sample_report_data.calculate_metrics()
        summary_html = html_generator._generate_summary_section(
            sample_report_data, metrics
        )
        
        assert isinstance(summary_html, str)
        assert len(summary_html) > 0
        assert '总收益率' in summary_html
        assert '年化收益率' in summary_html
        assert '最大回撤' in summary_html
        assert '夏普比率' in summary_html

    def test_performance_section_generation(self, html_generator, sample_report_data):
        """测试绩效分析部分生成"""
        metrics = sample_report_data.calculate_metrics()
        performance_html = html_generator._generate_performance_section(
            sample_report_data, metrics
        )
        
        assert isinstance(performance_html, str)
        assert len(performance_html) > 0
        assert '绩效分析' in performance_html or 'Performance Analysis' in performance_html

    def test_risk_section_generation(self, html_generator, sample_report_data):
        """测试风险分析部分生成"""
        metrics = sample_report_data.calculate_metrics()
        risk_html = html_generator._generate_risk_section(metrics)
        
        assert isinstance(risk_html, str)
        assert len(risk_html) > 0
        assert '风险分析' in risk_html or 'Risk Analysis' in risk_html

    def test_trading_section_generation(self, html_generator, sample_report_data):
        """测试交易分析部分生成"""
        metrics = sample_report_data.calculate_metrics()
        trading_html = html_generator._generate_trading_section(
            sample_report_data, metrics
        )
        
        assert isinstance(trading_html, str)
        assert len(trading_html) > 0
        assert '交易分析' in trading_html or 'Trading Analysis' in trading_html

    def test_positions_section_generation(self, html_generator, sample_report_data):
        """测试持仓分析部分生成"""
        positions_html = html_generator._generate_positions_section(sample_report_data)
        
        assert isinstance(positions_html, str)
        assert len(positions_html) > 0
        assert '持仓分析' in positions_html or 'Positions Analysis' in positions_html

    def test_complete_report_generation(self, html_generator, sample_report_data):
        """测试完整报告生成"""
        # 使用临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        # 生成报告
        html_generator.generate_report(sample_report_data, temp_path)
        
        # 验证文件存在
        assert os.path.exists(temp_path)
        
        # 验证文件内容
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert len(content) > 0
        assert '<html' in content
        assert '</html>' in content
        assert '投资组合分析报告' in content or 'Portfolio Analysis Report' in content
        
        # 验证包含主要部分
        assert '绩效摘要' in content or 'Performance Summary' in content
        assert '风险分析' in content or 'Risk Analysis' in content
        assert '交易分析' in content or 'Trading Analysis' in content
        assert '持仓分析' in content or 'Positions Analysis' in content
        
        # 清理临时文件
        os.unlink(temp_path)

    def test_report_with_benchmark_comparison(self, html_generator, sample_report_data):
        """测试包含基准比较的报告生成"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        # 生成包含基准比较的报告
        html_generator.generate_report(
            sample_report_data, 
            temp_path,
            include_benchmark=True
        )
        
        # 验证文件内容
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '基准比较' in content or 'Benchmark Comparison' in content
        assert '超额收益' in content or 'Excess Return' in content
        
        # 清理临时文件
        os.unlink(temp_path)

    def test_custom_template_usage(self, sample_report_data):
        """测试自定义模板使用"""
        # 创建临时模板目录
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "custom_template.html"
            
            # 创建简单的自定义模板
            custom_template = """
            <!DOCTYPE html>
            <html>
            <head><title>Custom Report</title></head>
            <body>
                <h1>自定义报告</h1>
                <p>总收益率: {{total_return}}</p>
                <p>夏普比率: {{sharpe_ratio}}</p>
            </body>
            </html>
            """
            
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(custom_template)
            
            # 使用自定义模板生成器
            custom_generator = HTMLReportGenerator(template_dir=temp_dir)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                report_path = f.name
            
            custom_generator.generate_report(
                sample_report_data, 
                report_path,
                template_name="custom_template.html"
            )
            
            # 验证自定义模板生效
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert '自定义报告' in content
            assert 'Custom Report' in content
            
            # 清理
            os.unlink(report_path)

    def test_invalid_template_error(self, html_generator, sample_report_data):
        """测试无效模板错误"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        # 测试不存在的模板
        with pytest.raises(FileNotFoundError, match="模板目录不存在|模板文件不存在"):
            html_generator.generate_report(
                sample_report_data,
                temp_path,
                template_name="nonexistent_template.html"
            )
        
        # 清理
        os.unlink(temp_path)

    def test_output_directory_creation(self, html_generator, sample_report_data):
        """测试输出目录创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 使用不存在的子目录
            output_path = Path(temp_dir) / "reports" / "new_report.html"
            
            html_generator.generate_report(sample_report_data, str(output_path))
            
            # 验证目录被创建
            assert output_path.parent.exists()
            assert output_path.exists()


class TestReportGenerator:
    """报告生成器测试类"""

    @pytest.fixture
    def report_generator(self):
        """创建报告生成器"""
        return ReportGenerator()

    @pytest.fixture
    def sample_report_data(self):
        """创建样本报告数据"""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 50)
        
        portfolio_values = [1000000]
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        portfolio_values = pd.Series(portfolio_values[1:], index=dates)
        
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 50),
            index=dates
        )
        
        trades = [
            Trade("000001.SZ", OrderType.BUY, 1000, Decimal("10.00"), datetime(2023, 1, 2), Decimal("10.00")),
        ]
        
        positions = {
            '000001.SZ': {'quantity': 1000, 'market_value': 11000, 'weight': 1.0}
        }
        
        return ReportData(
            returns=pd.Series(returns, index=dates),
            portfolio_values=portfolio_values,
            benchmark_returns=benchmark_returns,
            trades=trades,
            positions=positions,
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 19),
            initial_capital=1000000.0
        )

    def test_report_generator_initialization(self, report_generator):
        """测试报告生成器初始化"""
        assert isinstance(report_generator.html_generator, HTMLReportGenerator)

    def test_generate_comprehensive_report(self, report_generator, sample_report_data):
        """测试生成综合报告"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comprehensive_report.html"
            
            report_generator.generate_comprehensive_report(
                sample_report_data,
                str(output_path)
            )
            
            # 验证报告文件存在
            assert output_path.exists()
            
            # 验证报告内容
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert len(content) > 1000  # 确保报告有实质内容
            assert '<html' in content
            assert '</html>' in content

    def test_batch_report_generation(self, report_generator):
        """测试批量报告生成"""
        # 创建多个报告数据
        report_data_list = []
        for i in range(3):
            dates = pd.date_range(f'2023-0{i+1}-01', periods=30, freq='D')
            np.random.seed(42 + i)
            returns = np.random.normal(0.001, 0.02, 30)
            
            portfolio_values = [1000000]
            for ret in returns:
                portfolio_values.append(portfolio_values[-1] * (1 + ret))
            portfolio_values = pd.Series(portfolio_values[1:], index=dates)
            
            report_data = ReportData(
                returns=pd.Series(returns, index=dates),
                portfolio_values=portfolio_values,
                benchmark_returns=pd.Series(np.random.normal(0.0005, 0.015, 30), index=dates),
                trades=[],
                positions={},
                start_date=dates[0].date(),
                end_date=dates[-1].date(),
                initial_capital=1000000.0
            )
            report_data_list.append((f"report_{i+1}", report_data))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_generator.generate_batch_reports(
                report_data_list,
                temp_dir
            )
            
            # 验证所有报告文件都被创建
            report_files = list(Path(temp_dir).glob("*.html"))
            assert len(report_files) == 3
            
            # 验证文件名正确
            expected_files = {"report_1.html", "report_2.html", "report_3.html"}
            actual_files = {f.name for f in report_files}
            assert actual_files == expected_files

    def test_report_comparison(self, report_generator):
        """测试报告对比功能"""
        # 创建两个报告数据进行对比
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        # 策略A
        np.random.seed(42)
        returns_a = np.random.normal(0.002, 0.02, 50)  # 更高收益
        portfolio_values_a = [1000000]
        for ret in returns_a:
            portfolio_values_a.append(portfolio_values_a[-1] * (1 + ret))
        portfolio_values_a = pd.Series(portfolio_values_a[1:], index=dates)
        
        report_data_a = ReportData(
            returns=pd.Series(returns_a, index=dates),
            portfolio_values=portfolio_values_a,
            benchmark_returns=pd.Series(np.random.normal(0.0005, 0.015, 50), index=dates),
            trades=[],
            positions={},
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 19),
            initial_capital=1000000.0
        )
        
        # 策略B
        np.random.seed(43)
        returns_b = np.random.normal(0.001, 0.015, 50)  # 更低风险
        portfolio_values_b = [1000000]
        for ret in returns_b:
            portfolio_values_b.append(portfolio_values_b[-1] * (1 + ret))
        portfolio_values_b = pd.Series(portfolio_values_b[1:], index=dates)
        
        report_data_b = ReportData(
            returns=pd.Series(returns_b, index=dates),
            portfolio_values=portfolio_values_b,
            benchmark_returns=pd.Series(np.random.normal(0.0005, 0.015, 50), index=dates),
            trades=[],
            positions={},
            start_date=date(2023, 1, 1),
            end_date=date(2023, 2, 19),
            initial_capital=1000000.0
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comparison_report.html"
            
            report_generator.generate_comparison_report(
                {"策略A": report_data_a, "策略B": report_data_b},
                str(output_path)
            )
            
            # 验证对比报告
            assert output_path.exists()
            
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert '策略A' in content
            assert '策略B' in content
            assert '对比分析' in content or 'Comparison Analysis' in content