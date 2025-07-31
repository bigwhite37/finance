"""
基本集成测试

测试系统的基本集成功能
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from unittest.mock import Mock

from src.rl_trading_system.system_integration import (
    SystemConfig, TradingSystem, system_manager
)


class TestBasicIntegration:
    """基本集成测试"""
    
    def test_system_creation_and_initialization(self):
        """测试系统创建和初始化"""
        # 创建简单配置
        config = SystemConfig(
            data_source="qlib",
            stock_pool=['000001.SZ', '000002.SZ'],
            lookback_window=30,
            initial_cash=100000.0,
            transformer_config={
                'd_model': 64,
                'n_heads': 4,
                'n_layers': 2
            },
            sac_config={
                'hidden_dim': 128
            },
            enable_monitoring=False,
            enable_audit=False,
            enable_risk_control=False,
            log_level="WARNING"
        )
        
        # 创建系统
        success = system_manager.create_system("test_basic", config)
        assert success, "系统创建应该成功"
        
        # 检查系统状态
        status = system_manager.get_system_status("test_basic")
        assert status is not None, "应该能获取系统状态"
        assert status['state'] == 'stopped', "初始状态应该是stopped"
        assert status['portfolio_value'] == 100000.0, "初始组合价值应该正确"
        
        # 清理
        system_manager.remove_system("test_basic")
    
    def test_system_lifecycle(self):
        """测试系统生命周期"""
        # 创建系统
        config = SystemConfig(
            stock_pool=['000001.SZ'],
            initial_cash=50000.0,
            enable_monitoring=False,
            enable_audit=False,
            enable_risk_control=False,
            log_level="ERROR"  # 减少日志输出
        )
        
        system_manager.create_system("test_lifecycle", config)
        
        # 启动系统
        success = system_manager.start_system("test_lifecycle")
        assert success, "系统启动应该成功"
        
        # 检查运行状态
        status = system_manager.get_system_status("test_lifecycle")
        assert status['state'] == 'running', "系统应该在运行"
        
        # 停止系统
        success = system_manager.stop_system("test_lifecycle")
        assert success, "系统停止应该成功"
        
        # 检查停止状态
        status = system_manager.get_system_status("test_lifecycle")
        assert status['state'] == 'stopped', "系统应该已停止"
        
        # 清理
        system_manager.remove_system("test_lifecycle")
    
    def test_multiple_systems(self):
        """测试多系统管理"""
        systems = []
        
        # 创建多个系统
        for i in range(3):
            config = SystemConfig(
                stock_pool=['000001.SZ'],
                initial_cash=10000.0 * (i + 1),
                enable_monitoring=False,
                enable_audit=False,
                enable_risk_control=False,
                log_level="ERROR"
            )
            
            name = f"test_multi_{i}"
            success = system_manager.create_system(name, config)
            assert success, f"系统{name}创建应该成功"
            systems.append(name)
        
        # 检查系统列表
        all_systems = system_manager.list_systems()
        for name in systems:
            assert name in all_systems, f"系统{name}应该在列表中"
        
        # 清理所有系统
        for name in systems:
            system_manager.remove_system(name)
    
    def test_system_configuration(self):
        """测试系统配置"""
        # 测试不同配置
        configs = [
            {
                'stock_pool': ['000001.SZ'],
                'initial_cash': 100000.0,
                'update_frequency': '1D'
            },
            {
                'stock_pool': ['000001.SZ', '000002.SZ'],
                'initial_cash': 200000.0,
                'update_frequency': '1H'
            }
        ]
        
        for i, config_dict in enumerate(configs):
            config = SystemConfig(
                **config_dict,
                enable_monitoring=False,
                enable_audit=False,
                enable_risk_control=False,
                log_level="ERROR"
            )
            
            name = f"test_config_{i}"
            success = system_manager.create_system(name, config)
            assert success, f"配置{i}的系统创建应该成功"
            
            # 验证配置
            status = system_manager.get_system_status(name)
            assert status['config']['initial_cash'] == config_dict['initial_cash']
            assert len(status['config']['stock_pool']) == len(config_dict['stock_pool'])
            
            # 清理
            system_manager.remove_system(name)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试启动不存在的系统
        success = system_manager.start_system("nonexistent")
        assert not success, "启动不存在的系统应该失败"
        
        # 测试获取不存在系统的状态
        status = system_manager.get_system_status("nonexistent")
        assert status is None, "不存在的系统状态应该为None"
        
        # 测试重复创建系统
        config = SystemConfig(
            stock_pool=['000001.SZ'],
            enable_monitoring=False,
            enable_audit=False,
            enable_risk_control=False,
            log_level="ERROR"
        )
        
        success1 = system_manager.create_system("test_duplicate", config)
        assert success1, "第一次创建应该成功"
        
        success2 = system_manager.create_system("test_duplicate", config)
        assert not success2, "重复创建应该失败"
        
        # 清理
        system_manager.remove_system("test_duplicate")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])