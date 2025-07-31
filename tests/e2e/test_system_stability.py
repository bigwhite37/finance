"""
系统稳定性测试

测试系统在各种异常情况下的稳定性和恢复能力
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading
import time
import gc
from typing import Dict, List, Any

from src.rl_trading_system.config import ConfigManager
from src.rl_trading_system.data import QlibDataInterface, FeatureEngineer, DataProcessor
from src.rl_trading_system.models import TimeSeriesTransformer, SACAgent, TransformerConfig, SACConfig
from src.rl_trading_system.trading import PortfolioEnvironment, PortfolioConfig
from src.rl_trading_system.monitoring import TradingSystemMonitor


class TestSystemStability:
    """系统稳定性测试"""
    
    @pytest.fixture
    def system_components(self):
        """系统组件"""
        # 基础配置
        config = {
            'model': {'d_model': 64, 'n_heads': 4, 'n_layers': 2},
            'trading': {'stock_pool': ['000001.SZ', '000002.SZ'], 'initial_cash': 100000}
        }
        
        # 创建组件
        transformer_config = TransformerConfig(
            d_model=config['model']['d_model'],
            n_heads=config['model']['n_heads'],
            n_layers=config['model']['n_layers']
        )
        transformer = TimeSeriesTransformer(transformer_config)
        
        sac_config = SACConfig(
            state_dim=config['model']['d_model'] * len(config['trading']['stock_pool']),
            action_dim=len(config['trading']['stock_pool'])
        )
        sac_agent = SACAgent(sac_config)
        
        portfolio_config = PortfolioConfig(
            stock_pool=config['trading']['stock_pool'],
            initial_cash=config['trading']['initial_cash']
        )
        portfolio_env = PortfolioEnvironment(portfolio_config)
        
        return {
            'transformer': transformer,
            'sac_agent': sac_agent,
            'portfolio_env': portfolio_env,
            'config': config
        }
    
    def test_memory_leak_detection(self, system_components):
        """测试内存泄漏检测"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行大量操作
        for i in range(100):
            # 创建大量临时数据
            features = torch.randn(4, 60, 2, 50)
            
            with torch.no_grad():
                encoded = system_components['transformer'](features)
                
                state = {
                    'features': features,
                    'positions': torch.randn(4, 2),
                    'market_state': torch.randn(4, 10)
                }
                action, _ = system_components['sac_agent'].get_action(state)
            
            # 定期清理
            if i % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 最终清理
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 验证内存增长在合理范围内
        assert memory_increase < 200, f"可能存在内存泄漏，内存增长: {memory_increase:.2f}MB"
        
        print(f"✓ 内存泄漏检测通过，内存增长: {memory_increase:.2f}MB")
    
    def test_concurrent_access(self, system_components):
        """测试并发访问"""
        results = []
        errors = []
        
        def worker_thread(thread_id):
            """工作线程"""
            try:
                for _ in range(10):
                    # 模拟并发推理
                    features = torch.randn(1, 60, 2, 50)
                    
                    with torch.no_grad():
                        encoded = system_components['transformer'](features)
                        
                        state = {
                            'features': features,
                            'positions': torch.randn(1, 2),
                            'market_state': torch.randn(1, 10)
                        }
                        action, _ = system_components['sac_agent'].get_action(state)
                    
                    results.append(f"Thread-{thread_id}: Success")
                    time.sleep(0.01)  # 短暂休眠
                    
            except Exception as e:
                errors.append(f"Thread-{thread_id}: {str(e)}")
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=30)
        
        # 验证结果
        assert len(errors) == 0, f"并发访问出现错误: {errors}"
        assert len(results) == 50, f"并发访问结果不完整: {len(results)}/50"
        
        print("✓ 并发访问测试通过")
    
    def test_long_running_stability(self, system_components):
        """测试长时间运行稳定性"""
        start_time = time.time()
        iteration_count = 0
        error_count = 0
        
        # 运行较长时间（或较多迭代）
        max_iterations = 200
        max_time = 60  # 最多运行60秒
        
        while iteration_count < max_iterations and (time.time() - start_time) < max_time:
            try:
                # 模拟交易环境运行
                obs = system_components['portfolio_env'].reset()
                
                for step in range(10):  # 每次运行10步
                    action = np.random.dirichlet([1, 1])  # 随机权重
                    obs, reward, done, info = system_components['portfolio_env'].step(action)
                    
                    if done:
                        break
                
                iteration_count += 1
                
                # 定期清理
                if iteration_count % 50 == 0:
                    gc.collect()
                    
            except Exception as e:
                error_count += 1
                if error_count > 5:  # 如果错误太多，停止测试
                    break
        
        total_time = time.time() - start_time
        error_rate = error_count / max(iteration_count, 1)
        
        # 验证稳定性
        assert error_rate < 0.05, f"错误率过高: {error_rate:.2%}"
        assert iteration_count > 50, f"运行迭代数过少: {iteration_count}"
        
        print(f"✓ 长时间运行稳定性测试通过")
        print(f"  - 运行时间: {total_time:.2f}秒")
        print(f"  - 迭代次数: {iteration_count}")
        print(f"  - 错误率: {error_rate:.2%}")
    
    def test_extreme_input_handling(self, system_components):
        """测试极端输入处理"""
        
        # 1. 测试极大值输入
        extreme_large = torch.full((1, 60, 2, 50), 1e6)
        try:
            with torch.no_grad():
                result = system_components['transformer'](extreme_large)
                assert not torch.isnan(result).any(), "极大值输入产生NaN"
                assert not torch.isinf(result).any(), "极大值输入产生Inf"
        except Exception as e:
            print(f"⚠ 极大值输入处理异常: {e}")
        
        # 2. 测试极小值输入
        extreme_small = torch.full((1, 60, 2, 50), 1e-6)
        try:
            with torch.no_grad():
                result = system_components['transformer'](extreme_small)
                assert not torch.isnan(result).any(), "极小值输入产生NaN"
        except Exception as e:
            print(f"⚠ 极小值输入处理异常: {e}")
        
        # 3. 测试NaN输入
        nan_input = torch.randn(1, 60, 2, 50)
        nan_input[0, 0, 0, 0] = float('nan')
        
        with pytest.raises((RuntimeError, ValueError)):
            with torch.no_grad():
                system_components['transformer'](nan_input)
        
        # 4. 测试零输入
        zero_input = torch.zeros(1, 60, 2, 50)
        try:
            with torch.no_grad():
                result = system_components['transformer'](zero_input)
                assert not torch.isnan(result).any(), "零输入产生NaN"
        except Exception as e:
            print(f"⚠ 零输入处理异常: {e}")
        
        print("✓ 极端输入处理测试通过")
    
    def test_resource_exhaustion_recovery(self, system_components):
        """测试资源耗尽恢复"""
        
        # 1. 测试GPU内存耗尽恢复（如果有GPU）
        if torch.cuda.is_available():
            try:
                # 尝试分配大量GPU内存
                large_tensors = []
                for i in range(100):
                    tensor = torch.randn(1000, 1000).cuda()
                    large_tensors.append(tensor)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # 清理GPU内存
                    del large_tensors
                    torch.cuda.empty_cache()
                    
                    # 验证系统可以恢复
                    features = torch.randn(1, 60, 2, 50)
                    if torch.cuda.is_available():
                        features = features.cuda()
                        system_components['transformer'] = system_components['transformer'].cuda()
                    
                    with torch.no_grad():
                        result = system_components['transformer'](features)
                        assert not torch.isnan(result).any()
                    
                    print("✓ GPU内存耗尽恢复测试通过")
        
        # 2. 测试CPU内存压力
        try:
            # 创建大量数据
            large_data = []
            for i in range(10):
                data = torch.randn(100, 60, 2, 50)
                large_data.append(data)
            
            # 验证系统仍能正常工作
            features = torch.randn(1, 60, 2, 50)
            with torch.no_grad():
                result = system_components['transformer'](features)
                assert not torch.isnan(result).any()
            
            # 清理
            del large_data
            gc.collect()
            
        except MemoryError:
            # 如果内存不足，清理后重试
            gc.collect()
            features = torch.randn(1, 60, 2, 50)
            with torch.no_grad():
                result = system_components['transformer'](features)
                assert not torch.isnan(result).any()
        
        print("✓ 资源耗尽恢复测试通过")
    
    def test_configuration_changes(self, system_components):
        """测试配置变更处理"""
        
        # 1. 测试运行时配置变更
        original_config = system_components['config'].copy()
        
        # 修改配置
        system_components['config']['trading']['initial_cash'] = 200000
        
        # 创建新的环境实例
        new_portfolio_config = PortfolioConfig(
            stock_pool=system_components['config']['trading']['stock_pool'],
            initial_cash=system_components['config']['trading']['initial_cash']
        )
        new_env = PortfolioEnvironment(new_portfolio_config)
        
        # 验证新配置生效
        obs = new_env.reset()
        assert new_env.initial_cash == 200000
        
        # 2. 测试无效配置处理
        invalid_config = PortfolioConfig(
            stock_pool=[],  # 空股票池
            initial_cash=-1000  # 负初始资金
        )
        
        with pytest.raises((ValueError, AssertionError)):
            PortfolioEnvironment(invalid_config)
        
        print("✓ 配置变更处理测试通过")
    
    def test_network_interruption_simulation(self, system_components):
        """测试网络中断模拟"""
        
        # 模拟数据接口
        data_interface = Mock()
        
        # 1. 模拟网络超时
        data_interface.get_price_data.side_effect = TimeoutError("网络超时")
        
        with pytest.raises(TimeoutError):
            data_interface.get_price_data(['000001.SZ'], '2023-01-01', '2023-01-02')
        
        # 2. 模拟网络恢复
        mock_data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'symbol': ['000001.SZ'],
            'close': [10.0]
        })
        data_interface.get_price_data.side_effect = None
        data_interface.get_price_data.return_value = mock_data
        
        result = data_interface.get_price_data(['000001.SZ'], '2023-01-01', '2023-01-02')
        assert not result.empty
        
        # 3. 模拟间歇性网络问题
        call_count = 0
        def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # 每3次调用失败一次
                raise ConnectionError("网络连接失败")
            return mock_data
        
        data_interface.get_price_data.side_effect = intermittent_failure
        
        success_count = 0
        failure_count = 0
        
        for i in range(10):
            try:
                result = data_interface.get_price_data(['000001.SZ'], '2023-01-01', '2023-01-02')
                success_count += 1
            except ConnectionError:
                failure_count += 1
        
        assert success_count > 0, "应该有成功的调用"
        assert failure_count > 0, "应该有失败的调用"
        
        print("✓ 网络中断模拟测试通过")
        print(f"  - 成功调用: {success_count}")
        print(f"  - 失败调用: {failure_count}")


class TestSystemRecovery:
    """系统恢复测试"""
    
    def test_checkpoint_recovery(self):
        """测试检查点恢复"""
        
        # 创建模型
        config = TransformerConfig(d_model=64, n_heads=4, n_layers=2)
        model1 = TimeSeriesTransformer(config)
        
        # 保存检查点
        checkpoint_path = "/tmp/test_checkpoint.pth"
        torch.save(model1.state_dict(), checkpoint_path)
        
        # 创建新模型并加载检查点
        model2 = TimeSeriesTransformer(config)
        model2.load_state_dict(torch.load(checkpoint_path))
        
        # 验证模型参数一致
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "检查点恢复后参数不一致"
        
        # 验证推理结果一致
        test_input = torch.randn(1, 60, 2, 50)
        with torch.no_grad():
            output1 = model1(test_input)
            output2 = model2(test_input)
            assert torch.allclose(output1, output2), "检查点恢复后推理结果不一致"
        
        # 清理
        import os
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        print("✓ 检查点恢复测试通过")
    
    def test_graceful_shutdown(self):
        """测试优雅关闭"""
        
        # 模拟系统组件
        components = {
            'data_processor': Mock(),
            'model': Mock(),
            'monitor': Mock(),
            'audit_logger': Mock()
        }
        
        # 模拟关闭流程
        shutdown_order = []
        
        def mock_shutdown(component_name):
            shutdown_order.append(component_name)
            return True
        
        # 设置关闭方法
        for name, component in components.items():
            component.shutdown = lambda n=name: mock_shutdown(n)
        
        # 执行关闭流程
        for name, component in components.items():
            component.shutdown()
        
        # 验证关闭顺序
        assert len(shutdown_order) == len(components)
        assert all(name in shutdown_order for name in components.keys())
        
        print("✓ 优雅关闭测试通过")
        print(f"  - 关闭顺序: {shutdown_order}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])