"""
配置管理器测试
"""

import unittest
import tempfile
import os
import yaml
import json
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ConfigManager, get_default_config


class TestConfigManager(unittest.TestCase):
    """配置管理器测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config_manager = ConfigManager()
        
        # 创建测试配置数据
        self.test_config = {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'universe': 'csi300'
            },
            'agent': {
                'learning_rate': 3e-4,
                'batch_size': 64
            }
        }
    
    def test_initialization_with_default_config(self):
        """测试默认配置初始化"""
        # 验证默认配置加载
        self.assertIsNotNone(self.config_manager.config)
        self.assertIn('data', self.config_manager.config)
        self.assertIn('agent', self.config_manager.config)
        self.assertIn('risk_control', self.config_manager.config)
    
    def test_get_config(self):
        """测试配置获取"""
        # 获取完整配置
        full_config = self.config_manager.get_config()
        self.assertIsInstance(full_config, dict)
        
        # 获取特定节配置
        data_config = self.config_manager.get_config('data')
        self.assertIsInstance(data_config, dict)
        self.assertIn('provider', data_config)
        
        # 获取不存在的节
        empty_config = self.config_manager.get_config('nonexistent')
        self.assertEqual(empty_config, {})
    
    def test_update_config(self):
        """测试配置更新"""
        # 更新根级配置
        updates = {'new_key': 'new_value'}
        self.config_manager.update_config(updates)
        
        self.assertEqual(self.config_manager.config['new_key'], 'new_value')
        
        # 更新特定节配置
        agent_updates = {'new_param': 0.123}
        self.config_manager.update_config(agent_updates, 'agent')
        
        self.assertEqual(self.config_manager.config['agent']['new_param'], 0.123)
    
    def test_get_value(self):
        """测试通过路径获取配置值"""
        # 获取存在的值
        learning_rate = self.config_manager.get_value('agent.learning_rate')
        self.assertIsNotNone(learning_rate)
        
        # 获取不存在的值
        nonexistent = self.config_manager.get_value('nonexistent.key', 'default')
        self.assertEqual(nonexistent, 'default')
    
    def test_set_value(self):
        """测试通过路径设置配置值"""
        # 设置新值
        self.config_manager.set_value('test.new_param', 123)
        
        # 验证设置成功
        self.assertEqual(self.config_manager.get_value('test.new_param'), 123)
        
        # 设置嵌套值
        self.config_manager.set_value('deep.nested.value', 'test')
        self.assertEqual(self.config_manager.get_value('deep.nested.value'), 'test')
    
    def test_save_load_yaml(self):
        """测试YAML格式保存和加载"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # 保存配置
            self.config_manager.save_config(tmp_path)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(tmp_path))
            
            # 创建新的配置管理器并加载
            new_manager = ConfigManager()
            original_config = new_manager.config.copy()
            
            new_manager.load_config(tmp_path)
            
            # 验证配置相同
            self.assertEqual(new_manager.config, self.config_manager.config)
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_load_json(self):
        """测试JSON格式保存和加载"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # 保存配置
            self.config_manager.save_config(tmp_path)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(tmp_path))
            
            # 直接读取JSON验证格式
            with open(tmp_path, 'r', encoding='utf-8') as f:
                loaded_json = json.load(f)
            
            # 验证JSON格式正确
            self.assertIsInstance(loaded_json, dict)
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_deep_merge(self):
        """测试深度合并功能"""
        base = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3
            }
        }
        
        update = {
            'b': {
                'd': 4,
                'e': 5
            },
            'f': 6
        }
        
        result = self.config_manager._deep_merge(base, update)
        
        # 验证合并结果
        self.assertEqual(result['a'], 1)
        self.assertEqual(result['b']['c'], 2)
        self.assertEqual(result['b']['d'], 4)  # 更新值
        self.assertEqual(result['b']['e'], 5)  # 新增值
        self.assertEqual(result['f'], 6)
    
    def test_validate_config(self):
        """测试配置验证"""
        # 默认配置应该通过验证
        self.assertTrue(self.config_manager.validate_config())
        
        # 删除必需的节
        del self.config_manager.config['data']
        self.assertFalse(self.config_manager.validate_config())
    
    def test_create_run_config(self):
        """测试运行配置创建"""
        # 测试训练模式
        train_config = self.config_manager.create_run_config('training')
        self.assertIn('2022-12-31', train_config['data']['end_date'])
        
        # 测试回测模式
        backtest_config = self.config_manager.create_run_config('backtest')
        self.assertIn('2022-01-01', backtest_config['data']['start_date'])
        
        # 测试默认模式
        default_config = self.config_manager.create_run_config('default')
        self.assertIsInstance(default_config, dict)
    
    def test_specialized_config_getters(self):
        """测试专门的配置获取器"""
        # 测试各种获取器
        data_config = self.config_manager.get_data_config()
        self.assertIn('provider', data_config)
        
        agent_config = self.config_manager.get_agent_config()
        self.assertIn('learning_rate', agent_config)
        
        env_config = self.config_manager.get_environment_config()
        self.assertIn('max_position', env_config)
        
        risk_config = self.config_manager.get_risk_control_config()
        self.assertIn('target_volatility', risk_config)
        
        backtest_config = self.config_manager.get_backtest_config()
        self.assertIn('initial_capital', backtest_config)
        
        training_config = self.config_manager.get_training_config()
        self.assertIn('total_episodes', training_config)
        
        comfort_config = self.config_manager.get_comfort_config()
        self.assertIn('monthly_dd_threshold', comfort_config)


class TestDefaultConfig(unittest.TestCase):
    """默认配置测试"""
    
    def test_get_default_config(self):
        """测试默认配置获取"""
        config = get_default_config()
        
        # 验证主要配置节存在
        required_sections = [
            'data', 'factors', 'environment', 'agent',
            'risk_control', 'backtest', 'training'
        ]
        
        for section in required_sections:
            self.assertIn(section, config)
            self.assertIsInstance(config[section], dict)
    
    def test_config_values_reasonable(self):
        """测试配置值合理性"""
        config = get_default_config()
        
        # 验证学习率在合理范围内
        lr = config['agent']['learning_rate']
        self.assertGreater(lr, 0)
        self.assertLess(lr, 1)
        
        # 验证目标波动率合理
        target_vol = config['risk_control']['target_volatility']
        self.assertGreater(target_vol, 0)
        self.assertLess(target_vol, 1)
        
        # 验证最大仓位限制
        max_pos = config['environment']['max_position']
        self.assertGreater(max_pos, 0)
        self.assertLessEqual(max_pos, 1)
        
        # 验证交易成本合理
        transaction_cost = config['environment']['transaction_cost']
        self.assertGreater(transaction_cost, 0)
        self.assertLess(transaction_cost, 0.1)
    
    def test_config_consistency(self):
        """测试配置一致性"""
        config = get_default_config()
        
        # 环境和回测的交易成本应该相近
        env_cost = config['environment']['transaction_cost']
        backtest_cost = config['backtest']['transaction_cost']
        
        # 允许一定差异但不应该相差太大
        self.assertLess(abs(env_cost - backtest_cost), 0.01)
        
        # 风险控制参数应该一致
        env_max_pos = config['environment']['max_position']
        risk_max_pos = config['risk_control']['max_position']
        
        self.assertEqual(env_max_pos, risk_max_pos)


if __name__ == '__main__':
    unittest.main()