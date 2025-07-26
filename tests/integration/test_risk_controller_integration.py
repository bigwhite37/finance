"""
风险控制器集成测试 - 测试与动态低波筛选器的集成
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from risk_control.risk_controller import RiskController
from risk_control.dynamic_lowvol_filter import DynamicLowVolFilter, DynamicLowVolConfig
from data.data_manager import DataManager


class TestRiskControllerIntegration:
    """风险控制器集成测试类"""
    
    @pytest.fixture
    def mock_data_manager(self):
        """创建模拟数据管理器"""
        data_manager = Mock()  # 移除 spec 限制
        
        # 模拟价格数据
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        stocks = [f'stock_{i:03d}' for i in range(50)]
        
        price_data = pd.DataFrame(
            np.random.randn(len(dates), len(stocks)).cumsum(axis=0) + 100,
            index=dates,
            columns=stocks
        )
        
        data_manager.get_stock_data.return_value = price_data
        data_manager.get_price_data.return_value = price_data  # 添加 get_price_data 方法
        data_manager.get_market_data.return_value = pd.DataFrame({
            'market_index': np.random.randn(len(dates)).cumsum() + 3000
        }, index=dates)
        
        return data_manager
    
    @pytest.fixture
    def basic_config(self):
        """基础配置"""
        return {
            'max_position': 0.1,
            'max_leverage': 1.2,
            'target_volatility': 0.12,
            'max_drawdown_threshold': 0.10,
            'enable_risk_parity': False,
            'enable_dynamic_lowvol': False
        }
    
    @pytest.fixture
    def lowvol_enabled_config(self):
        """启用动态低波筛选器的配置"""
        return {
            'max_position': 0.1,
            'max_leverage': 1.2,
            'target_volatility': 0.12,
            'max_drawdown_threshold': 0.10,
            'enable_risk_parity': False,
            'enable_dynamic_lowvol': True,
            'dynamic_lowvol': {
                'rolling_windows': [20, 60],
                'percentile_thresholds': {'低': 0.4, '中': 0.3, '高': 0.2},
                'garch_window': 100,  # 减少窗口长度以适应测试数据
                'forecast_horizon': 5,
                'enable_ml_predictor': False,
                'ivol_bad_threshold': 0.3,
                'ivol_good_threshold': 0.6,
                'regime_detection_window': 30,  # 减少窗口长度
                'regime_model_type': 'HMM',
                'enable_caching': True,
                'cache_expiry_days': 1,
                'parallel_processing': False  # 测试时禁用并行处理
            }
        }
    
    @pytest.fixture
    def sample_price_data(self):
        """样本价格数据"""
        dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
        stocks = [f'stock_{i:03d}' for i in range(10)]
        
        # 创建具有不同波动率特征的价格数据
        np.random.seed(42)
        price_data = pd.DataFrame(index=dates, columns=stocks)
        
        for i, stock in enumerate(stocks):
            # 不同股票具有不同的波动率
            volatility = 0.15 + (i % 3) * 0.05  # 0.15, 0.20, 0.25
            returns = np.random.normal(0.0005, volatility/np.sqrt(252), len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            price_data[stock] = prices
        
        return price_data
    
    def test_risk_controller_init_without_lowvol_filter(self, basic_config):
        """测试不启用动态低波筛选器的初始化"""
        risk_controller = RiskController(basic_config)
        
        assert risk_controller.lowvol_filter is None
        assert risk_controller.config == basic_config
        assert risk_controller.target_volatility == 0.12
    
    def test_risk_controller_init_with_lowvol_filter_no_data_manager(self, lowvol_enabled_config):
        """测试启用动态低波筛选器但没有数据管理器的初始化"""
        risk_controller = RiskController(lowvol_enabled_config, data_manager=None)
        
        # 没有数据管理器时，筛选器应该为None
        assert risk_controller.lowvol_filter is None
    
    def test_risk_controller_init_with_lowvol_filter_success(self, lowvol_enabled_config, mock_data_manager):
        """测试成功初始化动态低波筛选器"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            assert risk_controller.lowvol_filter is not None
            mock_filter_class.assert_called_once_with(
                lowvol_enabled_config['dynamic_lowvol'], 
                mock_data_manager
            )
            mock_filter_class.assert_called_once_with(
                lowvol_enabled_config['dynamic_lowvol'], 
                mock_data_manager
            )
    
    def test_risk_controller_init_with_lowvol_filter_failure(self, lowvol_enabled_config, mock_data_manager):
        """测试动态低波筛选器初始化失败的处理"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter_class.side_effect = Exception("初始化失败")
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            # 初始化失败时，筛选器应该为None
            assert risk_controller.lowvol_filter is None
    
    def test_get_adaptive_target_volatility_without_filter(self, basic_config):
        """测试没有筛选器时获取自适应目标波动率"""
        risk_controller = RiskController(basic_config)
        
        adaptive_vol = risk_controller._get_adaptive_target_volatility()
        
        assert adaptive_vol == basic_config['target_volatility']
    
    def test_get_adaptive_target_volatility_with_filter(self, lowvol_enabled_config, mock_data_manager):
        """测试有筛选器时获取自适应目标波动率"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_adaptive_target_volatility.return_value = 0.35
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            adaptive_vol = risk_controller._get_adaptive_target_volatility()
            
            assert adaptive_vol == 0.35  # 应该返回筛选器的自适应目标波动率
            mock_filter.get_adaptive_target_volatility.assert_called_once()
    
    def test_get_adaptive_target_volatility_filter_error(self, lowvol_enabled_config, mock_data_manager):
        """测试筛选器获取自适应目标波动率失败时的处理"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_adaptive_target_volatility.side_effect = Exception("获取失败")
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            adaptive_vol = risk_controller._get_adaptive_target_volatility()
            
            # 应该回退到默认目标波动率
            assert adaptive_vol == lowvol_enabled_config['target_volatility']
    
    def test_apply_lowvol_filter_success(self, lowvol_enabled_config, mock_data_manager, sample_price_data):
        """测试成功应用动态低波筛选掩码"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            # 模拟筛选掩码：前5只股票可交易，后5只不可交易
            tradable_mask = np.array([True] * 5 + [False] * 5)
            mock_filter.update_tradable_mask.return_value = tradable_mask
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            # 创建测试权重
            original_weights = np.array([0.1] * 10)  # 每只股票10%权重
            state = {'current_date': pd.Timestamp('2023-06-30')}
            
            filtered_weights = risk_controller._apply_lowvol_filter(
                original_weights, sample_price_data, state
            )
            
            # 验证筛选效果
            assert np.all(filtered_weights[:5] > 0)  # 前5只股票有权重
            assert np.all(filtered_weights[5:] < 1e-10)  # 后5只股票权重为0
            
            # 验证总杠杆保持不变
            original_leverage = np.sum(np.abs(original_weights))
            filtered_leverage = np.sum(np.abs(filtered_weights))
            assert abs(original_leverage - filtered_leverage) < 1e-10
    
    def test_apply_lowvol_filter_no_current_date(self, lowvol_enabled_config, mock_data_manager, sample_price_data):
        """测试状态中没有当前日期时的处理"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            tradable_mask = np.array([True] * 10)
            mock_filter.update_tradable_mask.return_value = tradable_mask
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            original_weights = np.array([0.1] * 10)
            state = {}  # 没有current_date
            
            filtered_weights = risk_controller._apply_lowvol_filter(
                original_weights, sample_price_data, state
            )
            
            # 应该使用价格数据的最后日期
            expected_date = sample_price_data.index[-1]
            mock_filter.update_tradable_mask.assert_called_with(expected_date)
    
    def test_apply_lowvol_filter_error_handling(self, lowvol_enabled_config, mock_data_manager, sample_price_data):
        """测试筛选器应用失败时的错误处理"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.update_tradable_mask.side_effect = Exception("筛选失败")
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            original_weights = np.array([0.1] * 10)
            state = {'current_date': pd.Timestamp('2023-06-30')}
            
            filtered_weights = risk_controller._apply_lowvol_filter(
                original_weights, sample_price_data, state
            )
            
            # 筛选失败时应该返回原始权重
            assert np.array_equal(filtered_weights, original_weights)
    
    def test_process_weights_with_lowvol_filter(self, lowvol_enabled_config, mock_data_manager, sample_price_data):
        """测试完整的权重处理流程（包含动态低波筛选器）"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_adaptive_target_volatility.return_value = 0.35
            tradable_mask = np.array([True] * 7 + [False] * 3)  # 70%股票可交易
            mock_filter.update_tradable_mask.return_value = tradable_mask
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            # 模拟止损管理器不触发止损
            risk_controller.stop_loss_manager.check_stop_loss = Mock(return_value=False)
            
            raw_weights = np.array([0.12] * 10)  # 略超过单股票限制
            current_nav = 1.05
            state = {
                'current_date': pd.Timestamp('2023-06-30'),
                'max_drawdown': 0.05,
                'portfolio_volatility': 0.15
            }
            
            processed_weights = risk_controller.process_weights(
                raw_weights, sample_price_data, current_nav, state
            )
            
            # 验证基础约束被应用（单股票仓位限制）
            assert np.all(np.abs(processed_weights) <= risk_controller.max_position + 1e-10)
            
            # 验证筛选掩码被应用
            assert np.all(processed_weights[7:] == 0)  # 后3只股票权重为0
            
            # 验证自适应目标波动率被使用
            mock_filter.get_adaptive_target_volatility.assert_called()
    
    def test_get_lowvol_filter_info_not_enabled(self, basic_config):
        """测试获取未启用筛选器的信息"""
        risk_controller = RiskController(basic_config)
        
        info = risk_controller.get_lowvol_filter_info()
        
        assert info['enabled'] is False
        assert info['status'] == 'not_initialized'
        assert '未启用' in info['message']
    
    def test_get_lowvol_filter_info_active(self, lowvol_enabled_config, mock_data_manager):
        """测试获取活跃筛选器的信息"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_filter_statistics.return_value = {'total_updates': 10}
            mock_filter.get_current_regime.return_value = '中'
            mock_filter.get_adaptive_target_volatility.return_value = 0.35
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            info = risk_controller.get_lowvol_filter_info()
            
            assert info['enabled'] is not False
            assert info['status'] == 'active'
            assert info['current_regime'] == '中'
            assert info['adaptive_target_volatility'] == 0.35
            assert info['filter_statistics'] == {'total_updates': 10}
    
    def test_get_lowvol_filter_info_error(self, lowvol_enabled_config, mock_data_manager):
        """测试获取筛选器信息时发生错误的处理"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_filter_statistics.side_effect = Exception("获取统计信息失败")
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            info = risk_controller.get_lowvol_filter_info()
            
            assert info['enabled'] is not False
            assert info['status'] == 'error'
            assert 'error_message' in info
    
    def test_get_risk_report_with_lowvol_filter(self, lowvol_enabled_config, mock_data_manager):
        """测试包含动态低波筛选器信息的风险报告"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_filter_statistics.return_value = {'total_updates': 5}
            mock_filter.get_current_regime.return_value = '高'
            mock_filter.get_adaptive_target_volatility.return_value = 0.30
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            # 添加一些历史记录
            risk_controller.portfolio_history = [
                {
                    'nav': 1.0,
                    'drawdown': 0.0,
                    'volatility': 0.12,
                    'weights': np.array([0.1] * 10),
                    'timestamp': pd.Timestamp.now()
                },
                {
                    'nav': 1.05,
                    'drawdown': 0.02,
                    'volatility': 0.15,
                    'weights': np.array([0.1] * 10),
                    'timestamp': pd.Timestamp.now()
                }
            ]
            
            report = risk_controller.get_risk_report()
            
            # 验证基础报告信息
            assert 'current_nav' in report
            assert 'total_return' in report
            
            # 验证动态低波筛选器信息
            assert 'dynamic_lowvol_filter' in report
            lowvol_info = report['dynamic_lowvol_filter']
            assert lowvol_info['enabled'] is True
            assert lowvol_info['current_regime'] == '高'
            assert lowvol_info['adaptive_target_volatility'] == 0.30
            
            # 验证目标波动率信息
            assert 'target_volatility' in report
            vol_info = report['target_volatility']
            assert vol_info['configured_target'] == lowvol_enabled_config['target_volatility']
            assert vol_info['adaptive_target'] == 0.30
            assert vol_info['using_adaptive'] is True
    
    def test_coordination_with_target_volatility_controller(self, lowvol_enabled_config, mock_data_manager, sample_price_data):
        """测试与目标波动率控制器的协调逻辑"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            mock_filter.get_adaptive_target_volatility.return_value = 0.35
            tradable_mask = np.array([True] * 10)
            mock_filter.update_tradable_mask.return_value = tradable_mask
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            # 模拟目标波动率控制器
            with patch.object(risk_controller.target_vol_controller, 'adjust_leverage') as mock_adjust:
                mock_adjust.return_value = np.array([0.08] * 10)
                
                # 模拟止损管理器不触发止损
                risk_controller.stop_loss_manager.check_stop_loss = Mock(return_value=False)
                
                raw_weights = np.array([0.1] * 10)
                current_nav = 1.0
                state = {'current_date': pd.Timestamp('2023-06-30')}
                
                processed_weights = risk_controller.process_weights(
                    raw_weights, sample_price_data, current_nav, state
                )
                
                # 验证目标波动率控制器被调用，并使用了自适应目标波动率
                mock_adjust.assert_called_once()
                call_args = mock_adjust.call_args
                assert call_args[0][2] == 0.35  # 第三个位置参数是 target_vol
    
    def test_integration_error_resilience(self, lowvol_enabled_config, mock_data_manager, sample_price_data):
        """测试集成过程中的错误恢复能力"""
        with patch('risk_control.risk_controller.DynamicLowVolFilter') as mock_filter_class:
            mock_filter = Mock()
            # 模拟各种错误情况
            mock_filter.get_adaptive_target_volatility.side_effect = Exception("获取自适应波动率失败")
            mock_filter.update_tradable_mask.side_effect = Exception("更新掩码失败")
            mock_filter_class.return_value = mock_filter
            
            risk_controller = RiskController(lowvol_enabled_config, mock_data_manager)
            
            # 模拟止损管理器不触发止损
            risk_controller.stop_loss_manager.check_stop_loss = Mock(return_value=False)
            
            raw_weights = np.array([0.1] * 10)
            current_nav = 1.0
            state = {'current_date': pd.Timestamp('2023-06-30')}
            
            # 即使筛选器出错，权重处理也应该能够完成
            processed_weights = risk_controller.process_weights(
                raw_weights, sample_price_data, current_nav, state
            )
            
            # 验证权重处理完成且结果合理
            assert len(processed_weights) == len(raw_weights)
            assert np.all(np.isfinite(processed_weights))
            assert np.all(np.abs(processed_weights) <= risk_controller.max_position + 1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])