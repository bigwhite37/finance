"""
CVaR-PPO智能体测试
"""

import unittest
import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent import CVaRPPOAgent, ActorCriticNetwork


class TestCVaRPPOAgent(unittest.TestCase):
    """CVaR-PPO智能体测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.state_dim = 10
        self.action_dim = 5
        self.config = {
            'hidden_dim': 64,
            'learning_rate': 3e-4,
            'clip_epsilon': 0.2,
            'ppo_epochs': 5,
            'batch_size': 32,
            'gamma': 0.99,
            'lambda_gae': 0.95,
            'cvar_alpha': 0.05,
            'cvar_lambda': 1.0,
            'cvar_threshold': -0.02
        }
        
        self.agent = CVaRPPOAgent(self.state_dim, self.action_dim, self.config)
    
    def test_network_initialization(self):
        """测试网络初始化"""
        network = ActorCriticNetwork(self.state_dim, self.action_dim, 64)
        
        # 测试前向传播
        state = torch.randn(1, self.state_dim)
        action_mean, action_std, value, cvar_estimate = network(state)
        
        # 验证输出维度
        self.assertEqual(action_mean.shape, (1, self.action_dim))
        self.assertEqual(action_std.shape, (self.action_dim,))
        self.assertEqual(value.shape, (1, 1))
        self.assertEqual(cvar_estimate.shape, (1, 1))
        
        # 验证标准差为正
        self.assertTrue(torch.all(action_std > 0))
    
    def test_agent_initialization(self):
        """测试智能体初始化"""
        # 验证配置设置
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertEqual(self.agent.hidden_dim, 64)
        self.assertEqual(self.agent.lr, 3e-4)
        
        # 验证网络和优化器
        self.assertIsInstance(self.agent.network, ActorCriticNetwork)
        self.assertIsInstance(self.agent.optimizer, torch.optim.Adam)
    
    def test_get_action(self):
        """测试动作获取"""
        state = np.random.randn(self.state_dim)
        
        # 测试随机动作
        action, log_prob, value, cvar_estimate = self.agent.get_action(state, deterministic=False)
        
        # 验证输出维度和类型
        self.assertEqual(len(action), self.action_dim)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)
        self.assertIsInstance(cvar_estimate, float)
        
        # 测试确定性动作
        det_action, det_log_prob, det_value, det_cvar = self.agent.get_action(state, deterministic=True)
        
        # 确定性动作的log_prob应该为0
        self.assertEqual(det_log_prob, 0.0)
        self.assertEqual(len(det_action), self.action_dim)
    
    def test_store_transition(self):
        """测试经验存储"""
        state = np.random.randn(self.state_dim)
        action = np.random.randn(self.action_dim)
        
        # 存储经验
        self.agent.store_transition(
            state=state,
            action=action,
            reward=0.1,
            value=0.5,
            log_prob=-1.2,
            done=False,
            cvar_estimate=-0.01
        )
        
        # 验证存储
        self.assertEqual(len(self.agent.memory['states']), 1)
        self.assertEqual(len(self.agent.memory['actions']), 1)
        self.assertEqual(len(self.agent.memory['rewards']), 1)
        
        # 验证存储内容
        np.testing.assert_array_equal(self.agent.memory['states'][0], state)
        np.testing.assert_array_equal(self.agent.memory['actions'][0], action)
        self.assertEqual(self.agent.memory['rewards'][0], 0.1)
    
    def test_compute_gae(self):
        """测试GAE计算"""
        # 创建测试数据
        rewards = torch.tensor([0.1, -0.05, 0.15, 0.0, 0.08])
        values = torch.tensor([0.2, 0.15, 0.25, 0.1, 0.18])
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        
        # 计算GAE
        advantages, returns = self.agent._compute_gae(rewards, values, dones)
        
        # 验证输出维度
        self.assertEqual(advantages.shape, rewards.shape)
        self.assertEqual(returns.shape, rewards.shape)
        
        # 验证advantages标准化
        self.assertAlmostEqual(advantages.mean().item(), 0.0, places=6)
        self.assertAlmostEqual(advantages.std().item(), 1.0, places=6)
    
    def test_compute_cvar_target(self):
        """测试CVaR目标计算"""
        rewards = torch.tensor([-0.02, 0.01, -0.05, 0.03, -0.01, 0.02, -0.08, 0.00])
        
        cvar_target = self.agent._compute_cvar_target(rewards)
        
        # 验证输出维度
        self.assertEqual(cvar_target.shape, rewards.shape)
        
        # 验证CVaR值（应该是尾部的平均值）
        var_quantile = torch.quantile(rewards, self.agent.cvar_alpha)
        expected_cvar = rewards[rewards <= var_quantile].mean()
        
        # 所有目标值应该相等并等于CVaR
        self.assertTrue(torch.allclose(cvar_target, expected_cvar))
    
    def test_update_with_empty_memory(self):
        """测试空记忆更新"""
        # 清空记忆
        self.agent._clear_memory()
        
        # 尝试更新
        result = self.agent.update()
        
        # 应该返回空字典
        self.assertEqual(result, {})
    
    def test_update_with_data(self):
        """测试带数据更新"""
        # 添加一些经验数据
        for _ in range(10):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            
            self.agent.store_transition(
                state=state,
                action=action,
                reward=np.random.normal(0, 0.1),
                value=np.random.normal(0, 0.2),
                log_prob=np.random.normal(-1, 0.5),
                done=False,
                cvar_estimate=np.random.normal(-0.01, 0.01)
            )
        
        # 执行更新
        result = self.agent.update()
        
        # 验证返回结果
        self.assertIn('total_loss', result)
        self.assertIn('avg_cvar_estimate', result)
        self.assertIsInstance(result['total_loss'], float)
        
        # 验证记忆被清空
        self.assertEqual(len(self.agent.memory['states']), 0)
    
    def test_clear_memory(self):
        """测试记忆清空"""
        # 添加一些数据
        state = np.random.randn(self.state_dim)
        action = np.random.randn(self.action_dim)
        
        self.agent.store_transition(
            state, action, 0.1, 0.5, -1.2, False, -0.01
        )
        
        # 验证数据存在
        self.assertEqual(len(self.agent.memory['states']), 1)
        
        # 清空记忆
        self.agent._clear_memory()
        
        # 验证记忆被清空
        for key in self.agent.memory:
            self.assertEqual(len(self.agent.memory[key]), 0)
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        import tempfile
        import os
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # 保存模型
        self.agent.save_model(tmp_path)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(tmp_path))
        
        # 创建新智能体
        new_agent = CVaRPPOAgent(self.state_dim, self.action_dim, self.config)
        
        # 加载模型
        new_agent.load_model(tmp_path)
        
        # 验证参数相同
        original_params = self.agent.network.state_dict()
        loaded_params = new_agent.network.state_dict()
        
        for key in original_params:
            self.assertTrue(torch.allclose(original_params[key], loaded_params[key]))
        
        # 清理临时文件
        os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()