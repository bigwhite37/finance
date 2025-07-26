# 增强架构集成指南

## 集成步骤

### 第一步：更新CVaR-PPO Agent

将现有的`cvar_ppo_agent.py`中的网络替换为增强架构：

```python
# 在 CVaRPPOAgent.__init__ 中替换网络初始化
from .enhanced_architecture import EnhancedActorCriticNetwork

# 替换原有的
# self.network = ActorCriticNetwork(state_dim, action_dim, self.hidden_dim).to(self.device)

# 为新的增强网络
self.network = EnhancedActorCriticNetwork(
    state_dim=state_dim, 
    action_dim=action_dim,
    hidden_dim=self.hidden_dim,
    num_attention_heads=config.get('num_attention_heads', 8),
    num_residual_blocks=config.get('num_residual_blocks', 3)
).to(self.device)
```

### 第二步：更新前向传播处理

修改`get_action`方法以处理新的输出格式：

```python
def get_action(self, state: np.ndarray, deterministic: bool = False):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    with torch.no_grad():
        outputs = self.network(state_tensor)  # 新的字典输出格式
        action_mean = outputs['action_mean']
        action_std = outputs['action_std']
        value = outputs['value']
        cvar_estimate = outputs['cvar_estimate']
        
        # 可选：利用市场状态信息
        market_state_probs = outputs['market_state_probs']
        
        # 其余逻辑保持不变...
```

### 第三步：集成增强训练策略（可选）

如果要使用高级训练策略，可以替换整个更新逻辑：

```python
from .enhanced_training_strategy import AdvancedPPOTrainer

class CVaRPPOAgent:
    def __init__(self, state_dim, action_dim, config):
        # 如果启用高级训练
        if config.get('use_advanced_training', False):
            self.advanced_trainer = AdvancedPPOTrainer(state_dim, action_dim, config)
            self.network = self.advanced_trainer.network
        else:
            # 使用增强架构但保持原有训练策略
            self.network = EnhancedActorCriticNetwork(...)
```

### 第四步：更新配置文件

在`config/default_config.py`中添加新的配置项：

```python
# 在agent配置中添加
'agent': {
    # 现有配置...
    
    # 增强架构配置
    'use_enhanced_architecture': True,
    'num_attention_heads': 8,
    'num_residual_blocks': 3,
    'num_factor_groups': 4,
    
    # 高级训练配置（可选）
    'use_advanced_training': False,  # 第一阶段设为False
    'use_per': False,
    'use_curriculum': True,
    'use_ensemble': False,
    
    # 损失权重
    'value_coef': 0.5,
    'cvar_coef': 0.1,
    'entropy_coef': 0.01,
    
    # 优化参数
    'weight_decay': 1e-5,
    'max_grad_norm': 1.0,
}
```

## 渐进式集成策略

### 阶段1：基础架构升级（推荐先实施）
- ✅ 只集成EnhancedActorCriticNetwork
- ✅ 保持现有训练流程不变
- ✅ 验证系统稳定性和性能改进

### 阶段2：训练策略优化（可选进阶）
- 在阶段1验证成功后再考虑
- 集成AdvancedPPOTrainer
- 启用课程学习和自适应学习率

### 阶段3：完整功能启用（高级用户）
- 启用优先经验回放
- 启用多时间框架训练
- 启用集成训练（需要更多计算资源）

## 最小变更集成方案

如果希望最小化代码变更，可以只做以下修改：

1. **仅替换网络架构**：
```python
# 在cvar_ppo_agent.py中
from .enhanced_architecture import EnhancedActorCriticNetwork

# 在__init__中替换
self.network = EnhancedActorCriticNetwork(state_dim, action_dim, self.hidden_dim)

# 更新forward调用
def get_action(self, state, deterministic=False):
    # ...
    outputs = self.network(state_tensor)
    action_mean = outputs['action_mean']
    action_std = outputs['action_std'] 
    value = outputs['value']
    cvar_estimate = outputs['cvar_estimate']
    # ...
```

2. **更新配置**：
```python
# 在config/default_config.py中添加
'num_attention_heads': 8,
'num_residual_blocks': 3,
```

这样就可以获得大部分架构改进的好处，同时保持代码稳定性。

## 测试验证

在集成后，建议运行以下测试：

### 功能测试
```bash
# 运行增强因子测试（已通过）
python test_enhanced_factors.py

# 运行小规模训练测试
python main.py --config config_train.yaml --mode train --episodes 10

# 运行简单回测
python main.py --config config_backtest.yaml --mode backtest
```

### 性能基准测试
- 对比增强前后的收益指标
- 监控训练收敛速度
- 验证内存使用情况

## 回滚方案

如果遇到问题，可以快速回滚：

1. **备份原始文件**：
   - 备份`cvar_ppo_agent.py`
   - 备份`config/default_config.py`

2. **回滚命令**：
```bash
# 恢复原始文件
cp cvar_ppo_agent.py.backup cvar_ppo_agent.py
cp config/default_config.py.backup config/default_config.py
```

## 性能监控

集成后关注以下指标：

### 训练指标
- 损失收敛曲线
- 梯度范数变化
- 学习率变化轨迹

### 性能指标  
- 年化收益率变化
- 最大回撤控制
- 夏普比率改善

### 系统指标
- 内存使用量
- 训练时间
- GPU利用率

---

通过这个渐进式集成方案，可以安全地升级到增强架构，最大化收益改进的同时最小化集成风险。