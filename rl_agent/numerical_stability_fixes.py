"""
CVaR-PPO数值稳定性修复模块
解决NaN梯度问题，提升训练稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_stable_ratio(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算数值稳定的重要性采样比率
    
    Args:
        new_log_probs: 新策略的对数概率
        old_log_probs: 旧策略的对数概率
        eps: 数值稳定项
    
    Returns:
        稳定的比率tensor
    """
    # 计算对数概率差，并限制范围防止exp溢出
    log_ratio = new_log_probs - old_log_probs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)  # 防止exp溢出
    
    # 计算比率
    ratio = torch.exp(log_ratio)
    
    # 额外的数值稳定性检查
    ratio = torch.clamp(ratio, min=eps, max=1.0/eps)
    
    return ratio


def stable_mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    数值稳定的MSE损失计算
    
    Args:
        pred: 预测值
        target: 目标值
        reduction: 聚合方式
    
    Returns:
        稳定的MSE损失
    """
    # 确保输入没有NaN或无穷值
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 计算误差并限制范围
    error = pred - target
    error = torch.clamp(error, min=-100.0, max=100.0)
    
    # 计算平方误差
    squared_error = error.pow(2)
    
    if reduction == 'mean':
        return squared_error.mean()
    elif reduction == 'sum':
        return squared_error.sum()
    else:
        return squared_error


def compute_stable_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 0.99, 
                            gae_lambda: float = 0.95) -> torch.Tensor:
    """
    计算数值稳定的优势函数
    
    Args:
        rewards: 奖励序列
        values: 价值函数估计
        gamma: 折扣因子
        gae_lambda: GAE参数
    
    Returns:
        稳定的优势估计
    """
    advantages = []
    advantage = 0.0
    
    # 反向计算GAE
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            delta = rewards[i] - values[i]
        else:
            delta = rewards[i] + gamma * values[i + 1] - values[i]
        
        # 限制delta范围防止累积误差
        delta = torch.clamp(delta, min=-10.0, max=10.0)
        
        advantage = delta + gamma * gae_lambda * advantage
        advantage = torch.clamp(advantage, min=-100.0, max=100.0)
        
        advantages.insert(0, advantage)
    
    advantages = torch.stack(advantages)
    
    # 标准化优势（可选，但有助于稳定性）
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages


def apply_gradient_clipping(model: nn.Module, max_norm: float = 1.0) -> float:
    """
    应用渐进式梯度裁剪
    
    Args:
        model: 神经网络模型
        max_norm: 最大梯度范数
    
    Returns:
        梯度范数
    """
    # 计算当前梯度范数
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            # 检查并修复NaN梯度
            if torch.isnan(param.grad).any():
                param.grad = torch.zeros_like(param.grad)
            
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (1.0 / 2)
    
    # 应用梯度裁剪
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-8)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    
    return total_norm


def initialize_network_weights(network: nn.Module, gain: float = 1.0):
    """
    改进的网络权重初始化
    
    Args:
        network: 神经网络
        gain: 初始化增益
    """
    for module in network.modules():
        if isinstance(module, nn.Linear):
            # 使用Xavier初始化，更稳定
            nn.init.xavier_uniform_(module.weight, gain=gain)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


def create_stable_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 1e-5) -> torch.optim.Optimizer:
    """
    创建数值稳定的优化器
    
    Args:
        model: 神经网络模型
        lr: 学习率
        weight_decay: 权重衰减
    
    Returns:
        配置好的优化器
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=True  # 使用AMSGrad变体提高稳定性
    )


def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    检查tensor的数值健康状况
    
    Args:
        tensor: 要检查的tensor
        name: tensor名称（用于日志）
    
    Returns:
        是否健康
    """
    if torch.isnan(tensor).any():
        print(f"警告: {name} 包含NaN值")
        return False
    
    if torch.isinf(tensor).any():
        print(f"警告: {name} 包含无穷值")
        return False
    
    if tensor.abs().max() > 1e6:
        print(f"警告: {name} 包含过大的值: {tensor.abs().max().item()}")
        return False
    
    return True


class StableLossFunction:
    """稳定的损失函数计算器"""
    
    def __init__(self, clip_epsilon: float = 0.2, cvar_lambda: float = 0.1, cvar_threshold: float = -0.02):
        self.clip_epsilon = clip_epsilon
        self.cvar_lambda = cvar_lambda
        self.cvar_threshold = cvar_threshold
    
    def compute_ppo_loss(self, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                        advantages: torch.Tensor) -> torch.Tensor:
        """计算稳定的PPO策略损失"""
        # 使用稳定的比率计算
        ratio = compute_stable_ratio(new_log_probs, old_log_probs)
        
        # 确保advantages数值稳定
        advantages = torch.clamp(advantages, min=-10.0, max=10.0)
        
        # 计算替代损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    def compute_value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """计算稳定的价值函数损失"""
        return stable_mse_loss(values.squeeze(), returns)
    
    def compute_cvar_loss(self, cvar_pred: torch.Tensor, cvar_target: torch.Tensor) -> torch.Tensor:
        """计算稳定的CVaR损失"""
        cvar_mse = stable_mse_loss(cvar_pred.squeeze(), cvar_target)
        
        # CVaR惩罚项
        cvar_penalty = F.relu(cvar_pred.squeeze().mean() - self.cvar_threshold)
        cvar_penalty = torch.clamp(cvar_penalty, max=10.0)  # 限制惩罚范围
        
        return self.cvar_lambda * cvar_mse + 2.0 * cvar_penalty
    
    def compute_total_loss(self, new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                          advantages: torch.Tensor, values: torch.Tensor, returns: torch.Tensor,
                          cvar_pred: torch.Tensor, cvar_target: torch.Tensor) -> torch.Tensor:
        """计算稳定的总损失"""
        policy_loss = self.compute_ppo_loss(new_log_probs, old_log_probs, advantages)
        value_loss = self.compute_value_loss(values, returns)
        cvar_loss = self.compute_cvar_loss(cvar_pred, cvar_target)
        
        total_loss = policy_loss + 0.5 * value_loss + cvar_loss
        
        # 最终的数值稳定性检查
        total_loss = torch.clamp(total_loss, min=-100.0, max=100.0)
        
        return total_loss


# 导出的主要修复函数
__all__ = [
    'compute_stable_ratio',
    'stable_mse_loss', 
    'compute_stable_advantages',
    'apply_gradient_clipping',
    'initialize_network_weights',
    'create_stable_optimizer',
    'check_tensor_health',
    'StableLossFunction'
]