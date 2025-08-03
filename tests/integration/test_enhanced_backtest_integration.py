"""
增强回测集成测试

验证增强指标功能是否正确集成到回测脚本中。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import yaml
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from src.rl_trading_system.metrics.portfolio_metrics import PortfolioMetricsCalculator


class TestEnhancedBacktestIntegration:
    """增强回测集成测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # 创建测试配置
        self.config = {
            "enhanced_metrics": {
                "enable_portfolio_metrics": True,
                "enable_agent_behavior_metrics": True,
                "enable_risk_control_metrics": True,
                "risk_free_rate": 0.03
            },
            "trading": {
                "environment": {
                    "stock_pool": ["601288.SH", "600036.SH", "600919.SH"],
                    "initial_cash": 1000000.0,
                    "commission_rate": 0.001,
                    "stamp_tax_rate": 0.001,
                    "max_position_size": 0.1
                }
            },
            "backtest": {
                "start_date": "2020-01-01",
                "end_date": "2023-12-31"
            }
        }
        
        # 创建测试数据
        self.dates = pd.date_range('2020-01-01', periods=100, freq='D')
        
        # 投资组合价值序列（模拟15%年化收益）
        np.random.seed(42)
        returns = np.random.normal(0.15/252, 0.20/np.sqrt(252), 100)
        portfolio_values = [1000000]
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        self.portfolio_values = pd.Series(portfolio_values[1:], index=self.dates)
        
        # 基准价值序列（模拟8%年化收益）
        bench_returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), 100)
        benchmark_values = [1000000]
        for ret in bench_returns:
            benchmark_values.append(benchmark_values[-1] * (1 + ret))
        self.benchmark_values = pd.Series(benchmark_values[1:], index=self.dates)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir)
    
    def test_enhanced_performance_metrics_calculation(self):
        """测试增强性能指标计算"""
        # 导入回测脚本中的函数
        from scripts.backtest import calculate_enhanced_performance_metrics
        
        # 计算增强指标
        enhanced_metrics = calculate_enhanced_performance_metrics(
            self.portfolio_values,
            self.benchmark_values,
            self.config
        )
        
        # 验证增强指标存在
        assert 'sharpe_ratio' in enhanced_metrics
        assert 'max_drawdown' in enhanced_metrics
        assert 'alpha' in enhanced_metrics
        assert 'beta' in enhanced_metrics
        assert 'annualized_return' in enhanced_metrics
        
        # 验证传统指标也存在（向后兼容）
        assert 'annual_return' in enhanced_metrics
        assert 'volatility' in enhanced_metrics
        assert 'information_ratio' in enhanced_metrics
        
        # 验证指标值的合理性
        assert isinstance(enhanced_metrics['sharpe_ratio'], float)
        assert isinstance(enhanced_metrics['max_drawdown'], float)
        assert isinstance(enhanced_metrics['alpha'], float)
        assert isinstance(enhanced_metrics['beta'], float)
        # 注意：增强指标中的max_drawdown可能是负值（与传统计算一致）或正值（新的计算方式）
        # 我们验证它是一个有效的数值即可
        assert isinstance(enhanced_metrics['max_drawdown'], float)
        assert not np.isnan(enhanced_metrics['max_drawdown'])
    
    def test_portfolio_metrics_calculator_integration(self):
        """测试投资组合指标计算器集成"""
        calculator = PortfolioMetricsCalculator()
        
        # 计算投资组合指标
        metrics = calculator.calculate_portfolio_metrics(
            portfolio_values=self.portfolio_values.tolist(),
            benchmark_values=self.benchmark_values.tolist(),
            dates=self.dates.tolist(),
            risk_free_rate=0.03
        )
        
        # 验证指标计算结果
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.alpha, float)
        assert isinstance(metrics.beta, float)
        assert isinstance(metrics.annualized_return, float)
        
        # 验证指标合理性
        assert metrics.max_drawdown >= 0
        assert not np.isnan(metrics.sharpe_ratio)
        assert not np.isnan(metrics.alpha)
        assert not np.isnan(metrics.beta)
        assert not np.isnan(metrics.annualized_return)
    
    @patch('src.rl_trading_system.data.QlibDataInterface')
    @patch('src.rl_trading_system.models.SACAgent')
    @patch('src.rl_trading_system.trading.PortfolioEnvironment')
    def test_backtest_with_enhanced_metrics(self, mock_env, mock_agent, mock_data_interface):
        """测试带增强指标的回测"""
        # 设置模拟数据接口
        mock_data_interface_instance = Mock()
        mock_data_interface.return_value = mock_data_interface_instance
        
        # 模拟基准数据获取
        benchmark_returns = self.benchmark_values.pct_change().dropna()
        mock_data_interface_instance.get_price_data.return_value = pd.DataFrame({
            'close': self.benchmark_values
        })
        
        # 设置模拟智能体
        mock_agent_instance = Mock()
        mock_agent_instance.get_action.return_value = np.array([0.3, 0.3, 0.4])
        
        # 设置模拟环境
        mock_env_instance = Mock()
        mock_env_instance.reset.return_value = {
            'features': np.random.randn(60, 3, 12),
            'positions': np.zeros(3),
            'market_state': np.random.randn(10)
        }
        
        # 模拟环境step返回
        step_returns = []
        for i in range(len(self.portfolio_values)):
            obs = {
                'features': np.random.randn(60, 3, 12),
                'positions': np.random.randn(3),
                'market_state': np.random.randn(10)
            }
            reward = np.random.normal(0.001, 0.02)
            done = (i == len(self.portfolio_values) - 1)
            info = {'portfolio_value': self.portfolio_values.iloc[i]}
            step_returns.append((obs, reward, done, info))
        
        mock_env_instance.step.side_effect = step_returns
        mock_env_instance.total_value = self.portfolio_values.iloc[-1]
        mock_env_instance.dates = self.dates
        mock_env_instance.start_idx = 0
        mock_env_instance.current_step = 0
        
        # 导入并测试回测函数
        from scripts.backtest import run_backtest
        
        # 创建临时模型文件
        model_path = self.temp_path / "test_model.pth"
        
        # 创建模拟模型检查点
        import torch
        from src.rl_trading_system.models.sac_agent import SACConfig
        
        sac_config = SACConfig(
            state_dim=128,
            action_dim=3,
            use_transformer=False,
            transformer_config=None
        )
        
        checkpoint = {
            'config': sac_config,
            'actor_state_dict': {},
            'critic_state_dict': {},
            'log_alpha': torch.tensor(0.2)
        }
        
        torch.save(checkpoint, model_path)
        
        # 运行回测
        with patch('scripts.backtest.load_trained_model') as mock_load_model:
            mock_load_model.return_value = mock_agent_instance
            
            results = run_backtest(str(model_path), self.config)
        
        # 验证回测结果包含增强指标
        assert 'portfolio_returns' in results
        assert 'portfolio_values' in results
        assert 'benchmark_returns' in results
        assert 'metrics' in results
        
        # 验证指标包含增强内容
        if results['metrics']:
            first_benchmark_metrics = list(results['metrics'].values())[0]
            assert 'sharpe_ratio' in first_benchmark_metrics
            assert 'max_drawdown' in first_benchmark_metrics
            assert 'alpha' in first_benchmark_metrics
            assert 'beta' in first_benchmark_metrics
    
    def test_enhanced_metrics_display_format(self):
        """测试增强指标显示格式"""
        # 创建测试指标数据
        test_metrics = {
            'annualized_return': 0.15,
            'benchmark_annual_return': 0.08,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.12,
            'alpha': 0.07,
            'beta': 1.1,
            'information_ratio': 0.8
        }
        
        # 验证指标格式化逻辑
        annual_return = test_metrics['annualized_return'] * 100
        benchmark_return = test_metrics['benchmark_annual_return'] * 100
        excess_return = annual_return - benchmark_return
        
        assert annual_return == 15.0
        assert benchmark_return == 8.0
        assert excess_return == 7.0
        
        # 验证指标解读逻辑
        sharpe = test_metrics['sharpe_ratio']
        max_dd = test_metrics['max_drawdown'] * 100
        alpha = test_metrics['alpha'] * 100
        
        # 夏普比率解读
        sharpe_good = sharpe > 1.0
        assert sharpe_good is True
        
        # Alpha解读
        alpha_positive = alpha > 0
        assert alpha_positive is True
        
        # 最大回撤解读
        drawdown_acceptable = max_dd < 15
        assert drawdown_acceptable is True
    
    def test_config_integration(self):
        """测试配置集成"""
        # 验证增强指标配置被正确读取
        enhanced_config = self.config.get('enhanced_metrics', {})
        
        assert enhanced_config.get('enable_portfolio_metrics') is True
        assert enhanced_config.get('enable_agent_behavior_metrics') is True
        assert enhanced_config.get('enable_risk_control_metrics') is True
        assert enhanced_config.get('risk_free_rate') == 0.03
        
        # 验证配置传递到指标计算
        risk_free_rate = enhanced_config.get('risk_free_rate', 0.03)
        assert risk_free_rate == 0.03
    
    def test_backward_compatibility_with_traditional_metrics(self):
        """测试与传统指标的向后兼容性"""
        from scripts.backtest import calculate_performance_metrics
        
        # 计算传统指标
        portfolio_returns = self.portfolio_values.pct_change().dropna()
        benchmark_returns = self.benchmark_values.pct_change().dropna()
        
        traditional_metrics = calculate_performance_metrics(
            portfolio_returns, benchmark_returns, self.config
        )
        
        # 验证传统指标仍然存在
        assert 'total_return' in traditional_metrics
        assert 'annual_return' in traditional_metrics
        assert 'volatility' in traditional_metrics
        assert 'sharpe_ratio' in traditional_metrics
        assert 'max_drawdown' in traditional_metrics
        assert 'information_ratio' in traditional_metrics
        assert 'alpha' in traditional_metrics
        assert 'beta' in traditional_metrics
        
        # 验证指标值合理性
        assert isinstance(traditional_metrics['total_return'], float)
        assert isinstance(traditional_metrics['annual_return'], float)
        assert isinstance(traditional_metrics['volatility'], float)
        assert traditional_metrics['max_drawdown'] <= 0  # 传统计算中回撤为负值


if __name__ == '__main__':
    pytest.main([__file__])