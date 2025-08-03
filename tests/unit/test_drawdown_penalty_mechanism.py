"""
回撤惩罚机制单元测试
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.rl_trading_system.risk_control.reward_optimizer import (
    RewardOptimizer,
    RewardConfig
)


class TestDrawdownPenaltyMechanism:
    """回撤惩罚机制测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.config = RewardConfig(
            drawdown_penalty_factor=2.0,
            drawdown_threshold=0.02,
            dynamic_penalty_enabled=True,
            penalty_escalation_factor=1.5,
            consecutive_loss_threshold=3,
            time_decay_enabled=True,
            penalty_decay_rate=0.1,
            phase_penalty_multipliers={
                'NORMAL': 0.0,
                'DRAWDOWN_START': 1.0,
                'DRAWDOWN_CONTINUE': 1.5,
                'RECOVERY': 0.5
            }
        )
        self.optimizer = RewardOptimizer(self.config)
    
    def test_basic_drawdown_penalty(self):
        """测试基础回撤惩罚"""
        # 小于阈值的回撤
        small_penalty = self.optimizer._calculate_drawdown_penalty(-0.01)
        assert small_penalty == 0.0
        
        # 超过阈值的回撤
        large_penalty = self.optimizer._calculate_drawdown_penalty(-0.05)
        assert large_penalty > 0
        
        # 更大的回撤应该有更大的惩罚
        very_large_penalty = self.optimizer._calculate_drawdown_penalty(-0.10)
        assert very_large_penalty > large_penalty
    
    def test_enhanced_drawdown_penalty_phases(self):
        """测试不同阶段的回撤惩罚"""
        drawdown = -0.05
        
        # 正常阶段
        penalty_normal = self.optimizer._calculate_enhanced_drawdown_penalty(drawdown, 'NORMAL')
        
        # 回撤开始阶段
        penalty_start = self.optimizer._calculate_enhanced_drawdown_penalty(drawdown, 'DRAWDOWN_START')
        
        # 回撤持续阶段
        penalty_continue = self.optimizer._calculate_enhanced_drawdown_penalty(drawdown, 'DRAWDOWN_CONTINUE')
        
        # 恢复阶段
        penalty_recovery = self.optimizer._calculate_enhanced_drawdown_penalty(drawdown, 'RECOVERY')
        
        # 验证阶段差异化
        assert penalty_normal == 0.0  # 正常阶段无惩罚
        assert penalty_continue > penalty_start  # 持续阶段惩罚更重
        assert penalty_recovery < penalty_start  # 恢复阶段惩罚较轻
    
    def test_dynamic_penalty_consecutive_losses(self):
        """测试连续亏损的动态惩罚"""
        positions = {'AAPL': 1.0}
        drawdown = -0.05
        
        # 模拟连续亏损
        for i in range(5):
            returns = -0.01  # 连续亏损
            self.optimizer.calculate_risk_adjusted_reward(
                returns=returns,
                drawdown=drawdown,
                positions=positions,
                drawdown_phase='DRAWDOWN_CONTINUE'
            )
        
        # 连续亏损应该增加惩罚倍数
        assert self.optimizer.consecutive_losses == 5
        
        # 计算动态惩罚倍数
        multiplier = self.optimizer._calculate_dynamic_penalty_multiplier(drawdown)
        assert multiplier > 1.0  # 应该有额外的惩罚倍数
    
    def test_dynamic_penalty_drawdown_deterioration(self):
        """测试回撤恶化的动态惩罚"""
        positions = {'AAPL': 1.0}
        
        # 第一次较小回撤
        self.optimizer.calculate_risk_adjusted_reward(
            returns=0.01,
            drawdown=-0.03,
            positions=positions,
            drawdown_phase='DRAWDOWN_START'
        )
        
        # 第二次回撤恶化
        self.optimizer.calculate_risk_adjusted_reward(
            returns=-0.01,
            drawdown=-0.06,  # 回撤恶化
            positions=positions,
            drawdown_phase='DRAWDOWN_CONTINUE'
        )
        
        # 计算动态惩罚倍数
        multiplier = self.optimizer._calculate_dynamic_penalty_multiplier(-0.06)
        assert multiplier > 1.0  # 回撤恶化应该增加惩罚
    
    def test_time_decay_multiplier(self):
        """测试时间衰减倍数"""
        # 添加一些历史惩罚数据
        self.optimizer.penalty_history = [1.0, 2.0, 1.5, 0.5, 3.0]
        
        # 计算时间衰减倍数
        decay_multiplier = self.optimizer._calculate_time_decay_multiplier()
        
        assert isinstance(decay_multiplier, float)
        assert 0.5 <= decay_multiplier <= 1.0  # 应该在合理范围内
    
    def test_time_decay_disabled(self):
        """测试禁用时间衰减"""
        config = RewardConfig(time_decay_enabled=False)
        optimizer = RewardOptimizer(config)
        
        # 添加历史数据
        optimizer.penalty_history = [1.0, 2.0, 1.5]
        
        # 时间衰减应该返回1.0
        decay_multiplier = optimizer._calculate_time_decay_multiplier()
        assert decay_multiplier == 1.0
    
    def test_phase_penalty_multiplier(self):
        """测试阶段惩罚倍数"""
        # 测试各个阶段的倍数
        assert self.optimizer._calculate_phase_penalty_multiplier('NORMAL') == 0.0
        assert self.optimizer._calculate_phase_penalty_multiplier('DRAWDOWN_START') == 1.0
        assert self.optimizer._calculate_phase_penalty_multiplier('DRAWDOWN_CONTINUE') == 1.5
        assert self.optimizer._calculate_phase_penalty_multiplier('RECOVERY') == 0.5
        
        # 测试未知阶段
        assert self.optimizer._calculate_phase_penalty_multiplier('UNKNOWN') == 1.0
    
    def test_penalty_analysis(self):
        """测试惩罚分析报告"""
        positions = {'AAPL': 0.5, 'GOOGL': 0.5}
        
        # 添加一些测试数据
        for i in range(10):
            returns = np.random.normal(0.001, 0.02)
            drawdown = np.random.uniform(-0.08, -0.01)
            phase = 'DRAWDOWN_CONTINUE' if i % 2 == 0 else 'RECOVERY'
            
            self.optimizer.calculate_risk_adjusted_reward(
                returns=returns,
                drawdown=drawdown,
                positions=positions,
                drawdown_phase=phase
            )
        
        # 获取惩罚分析
        analysis = self.optimizer.get_penalty_analysis()
        
        assert 'total_penalties' in analysis
        assert 'avg_penalty' in analysis
        assert 'max_penalty' in analysis
        assert 'penalty_trend' in analysis
        assert 'phase_penalty_contribution' in analysis
        assert analysis['total_penalties'] == 10
    
    def test_penalty_analysis_no_data(self):
        """测试无数据时的惩罚分析"""
        analysis = self.optimizer.get_penalty_analysis()
        assert analysis['status'] == '无惩罚历史数据'
    
    def test_penalty_trend_calculation(self):
        """测试惩罚趋势计算"""
        # 上升趋势
        self.optimizer.penalty_history = [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2]
        trend = self.optimizer._calculate_penalty_trend()
        assert trend == '上升'
        
        # 下降趋势
        self.optimizer.penalty_history = [3.0, 2.8, 2.5, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8]
        trend = self.optimizer._calculate_penalty_trend()
        assert trend == '下降'
        
        # 稳定趋势
        self.optimizer.penalty_history = [2.0, 2.1, 1.9, 2.0, 2.1, 1.9, 2.0, 2.1, 1.9, 2.0]
        trend = self.optimizer._calculate_penalty_trend()
        assert trend == '稳定'
    
    def test_penalty_trend_insufficient_data(self):
        """测试数据不足时的趋势计算"""
        self.optimizer.penalty_history = [1.0, 2.0]
        trend = self.optimizer._calculate_penalty_trend()
        assert trend == '数据不足'
    
    def test_comprehensive_penalty_mechanism(self):
        """测试综合惩罚机制"""
        positions = {'AAPL': 0.6, 'GOOGL': 0.4}
        
        # 模拟一个完整的回撤周期
        scenarios = [
            (0.01, -0.01, 'NORMAL'),           # 正常状态
            (-0.01, -0.03, 'DRAWDOWN_START'),  # 回撤开始
            (-0.02, -0.05, 'DRAWDOWN_CONTINUE'), # 回撤持续
            (-0.01, -0.06, 'DRAWDOWN_CONTINUE'), # 回撤恶化
            (0.005, -0.04, 'RECOVERY'),        # 开始恢复
            (0.01, -0.02, 'RECOVERY'),         # 继续恢复
            (0.015, -0.01, 'NORMAL')           # 回到正常
        ]
        
        rewards = []
        penalties = []
        
        for returns, drawdown, phase in scenarios:
            reward = self.optimizer.calculate_risk_adjusted_reward(
                returns=returns,
                drawdown=drawdown,
                positions=positions,
                drawdown_phase=phase
            )
            rewards.append(reward)
            penalties.append(self.optimizer.last_drawdown_penalty)
        
        # 验证惩罚模式
        assert penalties[0] == 0.0  # 正常状态无惩罚
        assert penalties[1] > 0.0   # 回撤开始有惩罚
        assert penalties[2] > penalties[1]  # 回撤持续惩罚更重
        assert penalties[3] > penalties[2]  # 回撤恶化惩罚最重
        assert penalties[4] < penalties[3]  # 恢复期惩罚减轻
        assert penalties[5] < penalties[4]  # 继续恢复惩罚更轻
        assert penalties[6] == 0.0  # 回到正常无惩罚
    
    def test_max_penalty_limit(self):
        """测试最大惩罚限制"""
        # 设置较小的最大惩罚限制
        config = RewardConfig(max_drawdown_penalty=5.0)
        optimizer = RewardOptimizer(config)
        
        # 极大的回撤
        extreme_drawdown = -0.5
        penalty = optimizer._calculate_enhanced_drawdown_penalty(
            extreme_drawdown, 'DRAWDOWN_CONTINUE'
        )
        
        # 惩罚不应超过最大限制
        assert penalty <= config.max_drawdown_penalty
    
    def test_dynamic_penalty_disabled(self):
        """测试禁用动态惩罚"""
        config = RewardConfig(dynamic_penalty_enabled=False)
        optimizer = RewardOptimizer(config)
        
        # 添加连续亏损
        optimizer.return_history = [-0.01, -0.02, -0.01, -0.015]
        optimizer.consecutive_losses = 4
        
        # 动态惩罚倍数应该为1.0
        multiplier = optimizer._calculate_dynamic_penalty_multiplier(-0.05)
        assert multiplier == 1.0
    
    def test_penalty_with_different_configurations(self):
        """测试不同配置下的惩罚计算"""
        # 高惩罚配置
        high_penalty_config = RewardConfig(
            drawdown_penalty_factor=5.0,
            drawdown_nonlinearity=2.0,
            drawdown_threshold=0.02
        )
        high_penalty_optimizer = RewardOptimizer(high_penalty_config)
        
        # 低惩罚配置
        low_penalty_config = RewardConfig(
            drawdown_penalty_factor=1.0,
            drawdown_nonlinearity=2.0,
            drawdown_threshold=0.02
        )
        low_penalty_optimizer = RewardOptimizer(low_penalty_config)
        
        drawdown = -0.05
        
        # 测试基础惩罚计算
        high_basic_penalty = high_penalty_optimizer._calculate_drawdown_penalty(drawdown)
        low_basic_penalty = low_penalty_optimizer._calculate_drawdown_penalty(drawdown)
        
        # 高惩罚配置应该产生更大的基础惩罚
        assert high_basic_penalty > low_basic_penalty
        
        # 验证具体数值
        expected_high = 5.0 * (0.03 ** 2.0)  # 5.0 * (0.05-0.02)^2
        expected_low = 1.0 * (0.03 ** 2.0)   # 1.0 * (0.05-0.02)^2
        
        assert abs(high_basic_penalty - expected_high) < 1e-10
        assert abs(low_basic_penalty - expected_low) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__])