"""
SAC智能体集成测试
"""
import pytest
import torch
import numpy as np

from src.rl_trading_system.models import (
    SACAgent, SACConfig, Experience
)


class TestSACIntegration:
    """SAC智能体集成测试"""
    
    def test_complete_sac_workflow(self):
        """测试完整的SAC工作流程"""
        # 创建配置
        config = SACConfig(
            state_dim=32,
            action_dim=5,
            hidden_dim=64,
            batch_size=16,
            buffer_capacity=500,
            learning_starts=20
        )
        
        # 创建智能体
        agent = SACAgent(config)
        
        # 模拟环境交互
        for episode in range(10):
            state = torch.randn(config.state_dim)
            
            for step in range(20):
                # 获取动作
                action = agent.get_action(state, deterministic=False)
                
                # 模拟环境反馈
                reward = np.random.uniform(-1, 1)
                next_state = torch.randn(config.state_dim)
                done = step == 19
                
                # 添加经验
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                agent.add_experience(experience)
                
                # 更新网络
                if agent.can_update():
                    losses = agent.update()
                    
                    # 验证损失值
                    if losses:
                        assert 'critic_loss' in losses
                        assert 'actor_loss' in losses
                        assert isinstance(losses['critic_loss'], float)
                        assert isinstance(losses['actor_loss'], float)
                
                state = next_state
        
        # 验证智能体状态
        assert agent.training_step > 0
        assert agent.total_env_steps > 0
        assert agent.replay_buffer.size > 0
        
        # 测试评估模式
        agent.eval()
        test_state = torch.randn(config.state_dim)
        action1 = agent.get_action(test_state, deterministic=True)
        action2 = agent.get_action(test_state, deterministic=True)
        
        # 确定性动作应该相同
        assert torch.allclose(action1, action2, atol=1e-6)
        
        print("✅ SAC智能体集成测试通过")


if __name__ == "__main__":
    test = TestSACIntegration()
    test.test_complete_sac_workflow()