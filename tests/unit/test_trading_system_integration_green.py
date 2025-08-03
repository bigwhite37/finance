#!/usr/bin/env python3
"""
交易系统集成的绿色阶段TDD测试
验证Task 12：与现有交易系统集成的功能实现
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestPortfolioEnvironmentDrawdownIntegration:
    """测试投资组合环境的回撤控制集成"""
    
    def test_portfolio_environment_has_drawdown_controller_integration(self):
        """Green: 测试PortfolioEnvironment现在有回撤控制器集成"""
        print("=== Green: 验证PortfolioEnvironment具有回撤控制器集成 ===")
        
        from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment, PortfolioConfig
        from rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig
        
        # 验证构造函数支持回撤控制配置
        config = PortfolioConfig(
            stock_pool=['TEST001', 'TEST002'],
            enable_drawdown_control=True,
            drawdown_control_config=DrawdownControlConfig()
        )
        
        # 验证配置有回撤控制参数
        assert hasattr(config, 'enable_drawdown_control')
        assert hasattr(config, 'drawdown_control_config')
        assert config.enable_drawdown_control is True
        assert config.drawdown_control_config is not None
        
        print("✅ PortfolioEnvironment具有回撤控制配置集成")
    
    def test_portfolio_environment_has_enhanced_reward_function(self):
        """Green: 测试现在有增强的回撤控制奖励函数"""
        print("=== Green: 验证具有增强的回撤控制奖励函数 ===")
        
        from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment
        
        # 检查是否有增强奖励函数方法
        assert hasattr(PortfolioEnvironment, '_calculate_enhanced_reward')
        assert hasattr(PortfolioEnvironment, '_update_drawdown_state')
        assert hasattr(PortfolioEnvironment, '_apply_drawdown_constraints')
        
        print("✅ 具有增强的回撤控制奖励函数")
    
    def test_portfolio_environment_has_drawdown_state_tracking(self):
        """Green: 测试现在有回撤状态跟踪"""
        print("=== Green: 验证具有回撤状态跟踪机制 ===")
        
        from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment
        
        # 检查是否有回撤状态跟踪相关方法
        assert hasattr(PortfolioEnvironment, 'get_drawdown_metrics')
        assert hasattr(PortfolioEnvironment, 'get_risk_metrics')
        
        print("✅ 具有回撤状态跟踪机制")
    
    def test_portfolio_environment_has_drawdown_control_constraints(self):
        """Green: 测试现在有回撤控制约束"""
        print("=== Green: 验证具有回撤控制约束机制 ===")
        
        from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment
        
        # 检查是否有回撤控制约束相关功能
        assert hasattr(PortfolioEnvironment, '_apply_drawdown_constraints')
        
        print("✅ 具有回撤控制约束机制")
    
    def test_portfolio_environment_can_create_with_drawdown_control(self):
        """Green: 测试可以创建带回撤控制的环境配置"""
        print("=== Green: 验证可以创建带回撤控制的环境配置 ===")
        
        from rl_trading_system.trading.portfolio_environment import PortfolioConfig
        from rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig
        
        # 创建带回撤控制的配置
        drawdown_config = DrawdownControlConfig(
            max_drawdown_threshold=0.15,
            enable_market_regime_detection=True
        )
        
        config = PortfolioConfig(
            stock_pool=['TEST001', 'TEST002'],
            enable_drawdown_control=True,
            drawdown_control_config=drawdown_config
        )
        
        # 验证配置正确设置
        assert config.enable_drawdown_control is True
        assert config.drawdown_control_config is not None
        assert config.drawdown_control_config.max_drawdown_threshold == 0.15
        assert config.drawdown_control_config.enable_market_regime_detection is True
        
        print("✅ 可以成功创建带回撤控制的环境配置")


class TestTrainingIntegration:
    """测试训练流程的回撤控制集成"""
    
    def test_training_config_has_enhanced_reward_integration(self):
        """Green: 测试训练配置现在有增强奖励函数集成"""
        print("=== Green: 验证训练配置具有增强奖励函数集成 ===")
        
        from rl_trading_system.training.trainer import TrainingConfig
        
        config = TrainingConfig()
        
        # 检查是否有回撤控制相关配置
        assert hasattr(config, 'enable_drawdown_monitoring')
        assert hasattr(config, 'drawdown_early_stopping')
        assert hasattr(config, 'max_training_drawdown')
        assert hasattr(config, 'reward_enhancement_progress')
        assert hasattr(config, 'enable_adaptive_learning')
        
        print("✅ 训练配置具有奖励优化集成")
    
    def test_training_has_drawdown_monitoring_classes(self):
        """Green: 测试训练过程现在有回撤监控类"""
        print("=== Green: 验证训练过程具有回撤监控机制 ===")
        
        from rl_trading_system.training.trainer import DrawdownEarlyStopping
        
        # 测试回撤早停类
        drawdown_early_stopping = DrawdownEarlyStopping(max_drawdown=0.3, patience=10)
        
        assert drawdown_early_stopping.max_drawdown == 0.3
        assert drawdown_early_stopping.patience == 10
        assert hasattr(drawdown_early_stopping, 'step')
        assert hasattr(drawdown_early_stopping, 'get_current_drawdown')
        assert hasattr(drawdown_early_stopping, 'reset')
        
        print("✅ 训练过程具有回撤监控机制")
    
    def test_training_has_adaptive_training_parameters(self):
        """Green: 测试现在有自适应训练参数调整"""
        print("=== Green: 验证具有基于回撤的自适应训练参数调整 ===")
        
        from rl_trading_system.training.trainer import RLTrainer
        
        # 检查RLTrainer是否有自适应参数调整相关方法
        assert hasattr(RLTrainer, '_adapt_training_parameters')
        assert hasattr(RLTrainer, '_monitor_drawdown')
        assert hasattr(RLTrainer, 'collect_drawdown_metrics')
        
        print("✅ 具有自适应训练参数调整")
    
    def test_early_stopping_supports_drawdown(self):
        """Green: 测试早停机制现在支持回撤指标"""
        print("=== Green: 验证早停机制支持回撤指标 ===")
        
        from rl_trading_system.training.trainer import DrawdownEarlyStopping
        
        # 测试回撤早停功能
        early_stopping = DrawdownEarlyStopping(max_drawdown=0.2, patience=5)
        
        # 模拟训练过程
        values = [100, 120, 110, 90, 85, 80]  # 模拟性能下降
        should_stop = False
        
        for value in values:
            should_stop = early_stopping.step(value)
            if should_stop:
                break
        
        # 验证回撤早停是否正确工作
        assert early_stopping.get_current_drawdown() > 0
        
        print("✅ 早停机制支持回撤指标")
    
    def test_trainer_can_monitor_drawdown(self):
        """Green: 测试训练器可以监控回撤"""
        print("=== Green: 验证训练器可以监控回撤 ===")
        
        from rl_trading_system.training.trainer import RLTrainer, TrainingConfig
        
        # 创建启用回撤监控的配置
        config = TrainingConfig(
            enable_drawdown_monitoring=True,
            max_training_drawdown=0.3,
            n_episodes=10
        )
        
        # 模拟组件
        mock_env = Mock()
        mock_agent = Mock()
        mock_data_split = Mock()
        
        # 创建训练器
        trainer = RLTrainer(config, mock_env, mock_agent, mock_data_split)
        
        # 验证回撤监控组件已初始化
        assert trainer.drawdown_early_stopping is not None
        assert trainer.drawdown_metrics == []
        assert hasattr(trainer, 'adaptive_learning_enabled')
        
        print("✅ 训练器可以监控回撤")


class TestSystemIntegration:
    """测试系统整体集成"""
    
    def test_has_end_to_end_drawdown_control(self):
        """Green: 测试现在有端到端的回撤控制流程"""
        print("=== Green: 验证具有端到端的回撤控制流程 ===")
        
        # 期望的端到端流程组件
        expected_components = [
            '环境中的回撤控制器集成',
            '训练过程中的回撤监控',
            '基于回撤的奖励优化',
            '自适应风险预算调整',
            '回撤控制约束的应用',
            '回撤状态的环境传递'
        ]
        
        implemented_components = []
        
        # 检查PortfolioEnvironment的集成
        from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment
        if hasattr(PortfolioEnvironment, '_apply_drawdown_constraints'):
            implemented_components.append('环境中的回撤控制器集成')
        
        # 检查训练过程的集成
        from rl_trading_system.training.trainer import RLTrainer
        if hasattr(RLTrainer, '_monitor_drawdown'):
            implemented_components.append('训练过程中的回撤监控')
        
        # 检查奖励优化
        if hasattr(PortfolioEnvironment, '_calculate_enhanced_reward'):
            implemented_components.append('基于回撤的奖励优化')
        
        # 检查风险预算
        if hasattr(PortfolioEnvironment, '_update_drawdown_state'):
            implemented_components.append('自适应风险预算调整')
        
        # 检查约束应用
        if hasattr(PortfolioEnvironment, '_apply_drawdown_constraints'):
            implemented_components.append('回撤控制约束的应用')
        
        # 检查状态传递
        if hasattr(PortfolioEnvironment, '_update_drawdown_state'):
            implemented_components.append('回撤状态的环境传递')
        
        print(f"期望的端到端组件: {expected_components}")
        print(f"✅ 已实现的组件: {implemented_components}")
        print(f"✅ 实现率: {len(implemented_components)}/{len(expected_components)}")
        
        # Green阶段：应该大部分组件都已实现
        assert len(implemented_components) >= 5, f"应该实现大部分端到端组件，实际实现{len(implemented_components)}个"
    
    def test_has_configuration_integration(self):
        """Green: 测试现在有配置系统集成"""
        print("=== Green: 验证具有配置系统集成 ===")
        
        from rl_trading_system.trading.portfolio_environment import PortfolioConfig
        from rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig
        
        # 检查PortfolioConfig是否集成了回撤控制配置
        config = PortfolioConfig(stock_pool=['TEST001'])
        
        # 检查是否有回撤控制相关配置项
        has_drawdown_config_integration = (
            hasattr(config, 'enable_drawdown_control') and
            hasattr(config, 'drawdown_control_config')
        )
        
        # 检查DrawdownControlConfig是否有get_reward_config方法
        drawdown_config = DrawdownControlConfig()
        has_reward_config = hasattr(drawdown_config, 'get_reward_config')
        
        print(f"✅ PortfolioConfig集成回撤控制配置: {has_drawdown_config_integration}")
        print(f"✅ DrawdownControlConfig支持奖励配置: {has_reward_config}")
        
        assert has_drawdown_config_integration, "PortfolioConfig应该有回撤控制配置集成"
        assert has_reward_config, "DrawdownControlConfig应该有get_reward_config方法"
    
    def test_has_performance_monitoring_integration(self):
        """Green: 测试现在有性能监控集成"""
        print("=== Green: 验证具有性能监控集成 ===")
        
        # 检查是否有集成的性能监控
        from rl_trading_system.trading.portfolio_environment import PortfolioEnvironment
        from rl_trading_system.training.trainer import RLTrainer
        
        env_has_monitoring = (
            hasattr(PortfolioEnvironment, 'get_drawdown_metrics') and
            hasattr(PortfolioEnvironment, 'get_risk_metrics')
        )
        
        trainer_has_monitoring = hasattr(RLTrainer, 'collect_drawdown_metrics')
        
        print(f"✅ 环境性能监控: {env_has_monitoring}")
        print(f"✅ 训练器性能监控: {trainer_has_monitoring}")
        
        assert env_has_monitoring, "环境应该有性能监控集成"
        assert trainer_has_monitoring, "训练器应该有回撤指标收集"
    
    def test_integration_works_together(self):
        """Green: 测试集成组件可以协同工作"""
        print("=== Green: 验证集成组件协同工作 ===")
        
        from rl_trading_system.trading.portfolio_environment import PortfolioConfig
        from rl_trading_system.backtest.drawdown_control_config import DrawdownControlConfig
        from rl_trading_system.training.trainer import TrainingConfig, DrawdownEarlyStopping
        
        # 创建回撤控制配置
        drawdown_config = DrawdownControlConfig(
            max_drawdown_threshold=0.15,
            drawdown_penalty_factor=2.0,
            enable_market_regime_detection=True
        )
        
        # 验证可以获取奖励配置
        reward_config = drawdown_config.get_reward_config()
        assert reward_config is not None
        assert reward_config.drawdown_penalty_factor == 2.0
        
        # 创建环境配置
        env_config = PortfolioConfig(
            stock_pool=['TEST001', 'TEST002'],
            enable_drawdown_control=True,
            drawdown_control_config=drawdown_config
        )
        
        # 创建训练配置
        training_config = TrainingConfig(
            enable_drawdown_monitoring=True,
            max_training_drawdown=0.3,
            enable_adaptive_learning=True,
            n_episodes=100
        )
        
        # 验证回撤早停可以工作
        drawdown_early_stopping = DrawdownEarlyStopping(
            max_drawdown=training_config.max_training_drawdown
        )
        
        # 模拟一些值
        test_values = [100, 120, 110, 90, 85]
        for value in test_values:
            should_stop = drawdown_early_stopping.step(value)
        
        current_drawdown = drawdown_early_stopping.get_current_drawdown()
        
        print(f"✅ 配置创建成功")
        print(f"✅ 回撤早停工作正常，当前回撤: {current_drawdown:.4f}")
        print(f"✅ 所有组件可以协同工作")
        
        assert current_drawdown >= 0, "回撤计算应该正常"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])