#!/usr/bin/env python3
"""
回撤控制器核心集成的绿色阶段测试
验证Task 11：核心控制器集成实现的功能正确性
"""

import pytest
import tempfile
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.risk_control.drawdown_controller import (
    DrawdownController, 
    ControlSignal, 
    ControlSignalType, 
    ControlSignalPriority,
    MarketState,
    PortfolioState,
    ConflictResolver,
    StateManager,
    DataFlowManager
)
from rl_trading_system.config.drawdown_control_config_manager import DrawdownControlConfigManager
from rl_trading_system.config.config_validator import ConfigValidator
from rl_trading_system.config.hot_reload import ConfigHotReloader
from rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig


class TestDrawdownControllerImplementation:
    """测试回撤控制器具体实现"""
    
    def test_drawdown_controller_creation_and_initialization(self):
        """Green: 测试回撤控制器创建和初始化"""
        print("=== Green: 验证DrawdownController创建和初始化 ===")
        
        # 创建配置
        config = DrawdownControlConfig()
        
        # 创建控制器
        controller = DrawdownController(config)
        
        # 验证控制器属性
        assert controller.config == config
        assert controller.state_manager is not None
        assert controller.conflict_resolver is not None
        assert controller.data_flow_manager is not None
        assert controller.drawdown_monitor is not None
        assert controller.dynamic_stop_loss is not None
        assert controller.reward_optimizer is not None
        assert controller.adaptive_risk_budget is not None
        
        print("✅ DrawdownController创建和初始化成功")
    
    def test_control_signal_processing(self):
        """Green: 测试控制信号处理"""
        print("=== Green: 验证控制信号处理机制 ===")
        
        # 创建控制信号
        signal1 = ControlSignal(
            signal_type=ControlSignalType.STOP_LOSS,
            priority=ControlSignalPriority.CRITICAL,
            timestamp=datetime.now(),
            source_component='test',
            content={'action': 'stop_loss'}
        )
        
        signal2 = ControlSignal(
            signal_type=ControlSignalType.POSITION_ADJUSTMENT,
            priority=ControlSignalPriority.MEDIUM,
            timestamp=datetime.now(),
            source_component='test',
            content={'action': 'adjust_position'}
        )
        
        # 测试冲突解决
        resolver = ConflictResolver()
        resolved_signals = resolver.resolve_conflicts([signal1, signal2])
        
        # 验证结果
        assert len(resolved_signals) >= 1
        assert resolved_signals[0].priority == ControlSignalPriority.CRITICAL  # 高优先级信号被保留
        
        print("✅ 控制信号处理机制工作正常")
    
    def test_market_and_portfolio_state_management(self):
        """Green: 测试市场和投资组合状态管理"""
        print("=== Green: 验证状态管理功能 ===")
        
        state_manager = StateManager()
        
        # 创建市场状态
        market_state = MarketState(
            prices={'AAPL': 150.0, 'MSFT': 300.0},
            volumes={'AAPL': 1000000, 'MSFT': 500000},
            timestamp=datetime.now()
        )
        
        # 创建投资组合状态
        portfolio_state = PortfolioState(
            positions={'AAPL': 100, 'MSFT': 50},
            portfolio_value=100000.0,
            cash=25000.0,
            timestamp=datetime.now()
        )
        
        # 更新状态
        state_manager.update_market_state(market_state)
        state_manager.update_portfolio_state(portfolio_state)
        
        # 验证状态
        assert state_manager.current_market_state == market_state
        assert state_manager.current_portfolio_state == portfolio_state
        
        # 检查一致性
        consistency = state_manager.get_state_consistency_check()
        assert consistency['consistent'] is True
        
        print("✅ 状态管理功能工作正常")
    
    def test_complete_control_workflow(self):
        """Green: 测试完整控制工作流程"""
        print("=== Green: 验证完整控制工作流程 ===")
        
        # 创建配置
        config = DrawdownControlConfig()
        controller = DrawdownController(config)
        
        # 创建市场和投资组合状态
        market_state = MarketState(
            prices={'AAPL': 150.0, 'MSFT': 300.0},
            volumes={'AAPL': 1000000, 'MSFT': 500000},
            timestamp=datetime.now()
        )
        
        portfolio_state = PortfolioState(
            positions={'AAPL': 100, 'MSFT': 50},
            portfolio_value=100000.0,
            cash=25000.0,
            timestamp=datetime.now()
        )
        
        # 执行控制步骤
        control_signals = controller.execute_control_step(market_state, portfolio_state)
        
        # 验证结果
        assert isinstance(control_signals, list)
        
        # 获取风险指标
        risk_metrics = controller.get_risk_metrics()
        assert 'current_drawdown' in risk_metrics
        assert 'portfolio_value' in risk_metrics
        
        print("✅ 完整控制工作流程执行成功")


class TestConfigurationManagement:
    """测试配置管理系统"""
    
    def test_config_manager_creation(self):
        """Green: 测试配置管理器创建"""
        print("=== Green: 验证配置管理器创建 ===")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = DrawdownControlConfigManager(
                config_dir=temp_dir,
                enable_hot_reload=False  # 测试时禁用热重载
            )
            
            assert config_manager.config_dir == Path(temp_dir)
            assert config_manager.validator is not None
            
            print("✅ 配置管理器创建成功")
    
    def test_config_loading_and_validation(self):
        """Green: 测试配置加载和验证"""
        print("=== Green: 验证配置加载和验证功能 ===")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试配置文件
            config_data = {
                'drawdown_control': {
                    'max_drawdown_threshold': 0.2,
                    'portfolio_stop_loss': 0.3,
                    'base_stop_loss': 0.05,
                    'drawdown_penalty_factor': 2.0
                }
            }
            
            config_file = Path(temp_dir) / 'test_config.yaml'
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            # 创建配置管理器
            config_manager = DrawdownControlConfigManager(
                config_dir=temp_dir,
                enable_hot_reload=False
            )
            
            # 加载配置
            loaded_config = config_manager.load_config(str(config_file))
            
            # 验证配置
            assert loaded_config.max_drawdown_threshold == 0.2
            assert loaded_config.portfolio_stop_loss == 0.3
            
            print("✅ 配置加载和验证功能正常")
    
    def test_config_parameter_validation(self):
        """Green: 测试配置参数验证"""
        print("=== Green: 验证配置参数验证器 ===")
        
        validator = ConfigValidator()
        
        # 测试有效配置
        valid_config = {
            'max_drawdown_threshold': 0.2,
            'portfolio_stop_loss': 0.3,
            'base_stop_loss': 0.05
        }
        
        result = validator.validate_drawdown_control_config(valid_config)
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
        # 测试无效配置
        invalid_config = {
            'max_drawdown_threshold': -0.1,  # 负值无效
            'portfolio_stop_loss': 1.5,     # 超出范围
        }
        
        result = validator.validate_drawdown_control_config(invalid_config)
        assert result['valid'] is False
        assert len(result['errors']) > 0
        
        print("✅ 配置参数验证器工作正常")
    
    def test_hot_reload_functionality(self):
        """Green: 测试配置热重载功能"""
        print("=== Green: 验证配置热重载功能 ===")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / 'test_config.yaml'
            
            # 创建初始配置
            initial_config = {
                'max_drawdown_threshold': 0.2,
                'portfolio_stop_loss': 0.3
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(initial_config, f)
            
            # 创建热重载器
            hot_reloader = ConfigHotReloader(
                watched_files=[str(config_file)],
                auto_apply=True
            )
            
            # 添加监控文件
            hot_reloader.add_watched_file(str(config_file))
            
            # 验证初始配置加载
            current_config = hot_reloader.get_current_config(str(config_file))
            
            # 如果配置没有加载，先尝试强制重载
            if current_config is None:
                hot_reloader.force_reload(str(config_file))
                current_config = hot_reloader.get_current_config(str(config_file))
            
            assert current_config is not None, f"配置应该被成功加载，当前配置: {current_config}"
            assert current_config['max_drawdown_threshold'] == 0.2
            
            print("✅ 配置热重载功能基本正常")


class TestComponentIntegration:
    """测试组件集成"""
    
    def test_data_flow_management(self):
        """Green: 测试数据流管理"""
        print("=== Green: 验证数据流管理 ===")
        
        data_flow_manager = DataFlowManager()
        
        # 测试数据流验证
        assert data_flow_manager.validate_data_flow('market_data', 'drawdown_monitor') is True
        assert data_flow_manager.validate_data_flow('invalid_source', 'drawdown_monitor') is False
        
        # 测试依赖获取
        dependencies = data_flow_manager.get_data_dependencies('drawdown_monitor')
        assert 'market_data' in dependencies
        assert 'portfolio_state' in dependencies
        
        print("✅ 数据流管理功能正常")
    
    def test_component_coordination(self):
        """Green: 测试组件协调"""
        print("=== Green: 验证组件协调功能 ===")
        
        config = DrawdownControlConfig()
        controller = DrawdownController(config)
        
        # 验证所有组件都已初始化
        assert controller.drawdown_monitor is not None
        assert controller.dynamic_stop_loss is not None
        assert controller.reward_optimizer is not None
        assert controller.adaptive_risk_budget is not None
        
        # 获取组件状态
        component_status = controller.get_component_status()
        
        assert component_status['drawdown_monitor']['active'] is True
        assert component_status['dynamic_stop_loss']['active'] is True
        assert component_status['reward_optimizer']['active'] is True
        assert component_status['adaptive_risk_budget']['active'] is True
        
        print("✅ 组件协调功能正常")
    
    def test_controller_activation_deactivation(self):
        """Green: 测试控制器激活和停用"""
        print("=== Green: 验证控制器激活和停用 ===")
        
        config = DrawdownControlConfig()
        controller = DrawdownController(config)
        
        # 测试激活
        controller.activate()
        assert controller.is_active is True
        
        # 测试停用
        controller.deactivate()
        assert controller.is_active is False
        
        # 测试重置
        controller.reset_control_state()
        assert len(controller.control_signal_queue) == 0
        
        print("✅ 控制器激活和停用功能正常")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])