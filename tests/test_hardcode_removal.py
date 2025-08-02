#!/usr/bin/env python3
"""
硬编码移除的TDD测试
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestHardcodeRemoval:
    """测试硬编码移除"""
    
    def test_backtest_script_should_not_hardcode_trading_days_per_year(self):
        """测试回测脚本不应硬编码每年交易日数"""
        # Green: 当前代码应该使用配置而不是硬编码252
        from scripts.backtest import calculate_performance_metrics
        import inspect
        
        # 读取函数源码检查是否包含硬编码的252
        source = inspect.getsource(calculate_performance_metrics)
        assert '252' not in source, "calculate_performance_metrics函数不应硬编码交易日数252"
    
    def test_backtest_script_should_not_hardcode_risk_free_rate(self):
        """测试回测脚本不应硬编码无风险利率"""
        # Red: 当前代码硬编码了3%的无风险利率
        from scripts.backtest import calculate_performance_metrics
        
        # 读取函数源码检查是否包含硬编码的0.03
        import inspect
        source = inspect.getsource(calculate_performance_metrics)
        assert '0.03' not in source, "calculate_performance_metrics函数不应硬编码无风险利率0.03"
    
    def test_backtest_script_should_not_hardcode_benchmark_symbols(self):
        """测试回测脚本不应硬编码基准指数代码"""
        # Red: 当前代码硬编码了['000300.SH', '000905.SH', '000852.SH']
        from scripts.backtest import run_backtest
        
        import inspect
        source = inspect.getsource(run_backtest)
        
        # 不应直接在函数中硬编码基准指数
        assert "'000300.SH'" not in source, "run_backtest函数不应硬编码基准指数代码"
        assert "'000905.SH'" not in source, "run_backtest函数不应硬编码基准指数代码"
        assert "'000852.SH'" not in source, "run_backtest函数不应硬编码基准指数代码"
    
    def test_backtest_script_should_not_hardcode_default_stock_pool(self):
        """测试回测脚本不应硬编码默认股票池"""
        # Red: 当前代码硬编码了['600519.SH', '600036.SH', '601318.SH']
        from scripts.backtest import run_backtest
        
        import inspect
        source = inspect.getsource(run_backtest)
        
        # 不应硬编码默认股票池
        assert "'600519.SH'" not in source, "run_backtest函数不应硬编码默认股票池"
    
    def test_backtest_script_should_not_hardcode_progress_log_interval(self):
        """测试回测脚本不应硬编码进度日志间隔"""
        # Red: 当前代码硬编码了50步的日志间隔
        from scripts.backtest import run_backtest
        
        import inspect
        source = inspect.getsource(run_backtest)
        
        # 不应硬编码日志间隔
        assert 'step % 50' not in source, "run_backtest函数不应硬编码进度日志间隔"
    
    def test_backtest_script_should_use_configurable_parameters(self):
        """测试回测脚本应该使用可配置的参数"""
        # Green: 修复后应该能传入配置参数
        from scripts.backtest import calculate_performance_metrics
        
        # 创建测试数据
        portfolio_returns = pd.Series([0.01, 0.02, -0.01], 
                                    index=pd.date_range('2020-01-01', periods=3))
        benchmark_returns = pd.Series([0.005, 0.015, 0.002], 
                                     index=pd.date_range('2020-01-01', periods=3))
        
        # 自定义配置
        config = {
            'trading_days_per_year': 250,  # 自定义交易日数
            'risk_free_rate': 0.025,       # 自定义无风险利率
        }
        
        # 应该能接受配置参数
        metrics = calculate_performance_metrics(
            portfolio_returns, benchmark_returns, config
        )
        
        # 验证使用了自定义配置
        assert metrics is not None, "应该能使用自定义配置计算指标"
        
        # 验证使用了自定义交易日数（通过年化收益率计算验证）
        # 自定义250个交易日应该与默认252个交易日产生不同的年化收益率
        default_metrics = calculate_performance_metrics(portfolio_returns, benchmark_returns)
        assert abs(metrics['annual_return'] - default_metrics['annual_return']) > 1e-10, \
            "自定义配置应该产生不同的年化收益率"
    
    def test_backtest_script_should_not_hardcode_visualization_parameters(self):
        """测试回测脚本不应硬编码可视化参数"""
        from scripts.backtest import create_performance_visualization
        import inspect
        
        source = inspect.getsource(create_performance_visualization)
        
        # 不应硬编码图表高度
        assert 'height=800' not in source, "create_performance_visualization函数不应硬编码图表高度800"
        
        # 不应硬编码颜色列表
        assert "colors = ['red', 'green', 'orange', 'purple']" not in source, "create_performance_visualization函数不应硬编码颜色列表"
        
        # 不应硬编码基准名称映射
        assert "'000300.SH': '沪深300'" not in source, "create_performance_visualization函数不应硬编码基准名称映射"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])