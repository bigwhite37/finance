"""
奖励函数优化器单元测试
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.rl_trading_system.risk_control.reward_optimizer import (
    RewardOptimizer,
    RewardConfig,
    RiskAdjustedMetrics
)


class TestRewardOptimizer:
    """奖励函数优化器测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.config = RewardConfig(
            base_return_weight=1.0,
            risk_aversion_coefficient=0.5,
            drawdown_penalty_factor=2.0,
            drawdown_threshold=0.02,
            diversification_bonus=0.1,
            concentration_penalty=0.5,
            max_single_position=0.2
        )
        self.optimizer = RewardOptimizer(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        assert self.optimizer.config == self.config
        assert len(self.optimizer.return_history) == 0
        assert len(self.optimizer.drawdown_history) == 0
        assert len(self.optimizer.position_history) == 0
        assert self.optimizer.performance_stats['total_episodes'] == 0
    
    def test_calculate_risk_adjusted_reward_basic(self):
        """测试基础风险调整奖励计算"""
        returns = 0.01  # 1%收益
        drawdown = -0.01  # 1%回撤（小于阈值）
        positions = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4}
        
        reward = self.optimizer.calculate_risk_adjusted_reward(
            returns=returns,
            drawdown=drawdown,
            positions=positions
        )
        
        # 基础收益奖励应该是主要部分
        assert reward > 0
        assert len(self.optimizer.return_history) == 1
        assert len(self.optimizer.drawdown_history) == 1
        assert len(self.optimizer.position_history) == 1
    
    def test_calculate_risk_adjusted_reward_with_penalty(self):
        """测试带回撤惩罚的奖励计算"""
        # 直接测试基础惩罚计算
        small_penalty = self.optimizer._calculate_drawdown_penalty(-0.01)
        large_penalty = self.optimizer._calculate_drawdown_penalty(-0.05)
        
        # 大回撤应该有更大的惩罚
        assert large_penalty > small_penalty
        assert small_penalty == 0.0  # 小于阈值应该无惩罚
        assert large_penalty > 0.0   # 大于阈值应该有惩罚
    
    def test_calculate_risk_adjusted_reward_concentration_penalty(self):
        """测试集中度惩罚"""
        returns = 0.01
        drawdown = -0.01
        concentrated_positions = {'AAPL': 0.8, 'GOOGL': 0.2}  # 高度集中
        diversified_positions = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
        
        reward_concentrated = self.optimizer.calculate_risk_adjusted_reward(
            returns=returns,
            drawdown=drawdown,
            positions=concentrated_positions
        )
        
        # 重置优化器以进行对比
        self.optimizer.reset()
        
        reward_diversified = self.optimizer.calculate_risk_adjusted_reward(
            returns=returns,
            drawdown=drawdown,
            positions=diversified_positions
        )
        
        # 分散持仓应该获得更高奖励
        assert reward_diversified > reward_concentrated
    
    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        # 正常情况
        returns = [0.01, 0.02, -0.01, 0.015, 0.005]
        sharpe = self.optimizer.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
        
        # 零波动率情况
        constant_returns = [0.01] * 5
        sharpe_zero_vol = self.optimizer.calculate_sharpe_ratio(constant_returns)
        assert sharpe_zero_vol == 0.0
        
        # 数据不足情况
        insufficient_data = [0.01]
        sharpe_insufficient = self.optimizer.calculate_sharpe_ratio(insufficient_data)
        assert sharpe_insufficient == 0.0
    
    def test_calculate_calmar_ratio(self):
        """测试卡尔玛比率计算"""
        returns = [0.01, 0.02, -0.03, 0.015, 0.005]
        max_drawdown = -0.03
        
        calmar = self.optimizer.calculate_calmar_ratio(returns, max_drawdown)
        assert isinstance(calmar, float)
        assert calmar > 0  # 正收益和负回撤应该产生正的卡尔玛比率
        
        # 零回撤情况
        calmar_zero_dd = self.optimizer.calculate_calmar_ratio(returns, 0.0)
        assert calmar_zero_dd == 0.0
    
    def test_calculate_sortino_ratio(self):
        """测试索提诺比率计算"""
        returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        
        sortino = self.optimizer.calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)
        
        # 无负收益情况
        positive_returns = [0.01, 0.02, 0.015, 0.005]
        sortino_positive = self.optimizer.calculate_sortino_ratio(positive_returns)
        assert sortino_positive == float('inf') or sortino_positive > 10  # 应该很大
    
    def test_calculate_risk_metrics(self):
        """测试风险指标计算"""
        returns = [0.01, 0.02, -0.01, 0.015, -0.005, 0.008, -0.002, 0.012]
        
        metrics = self.optimizer.calculate_risk_metrics(returns)
        
        assert isinstance(metrics, RiskAdjustedMetrics)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.calmar_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert metrics.max_drawdown <= 0  # 最大回撤应该是负数或零
        assert metrics.volatility >= 0  # 波动率应该非负
        assert metrics.var_95 <= 0  # 95% VaR应该是负数或零
        assert metrics.cvar_95 <= metrics.var_95  # CVaR应该小于等于VaR
    
    def test_drawdown_penalty_calculation(self):
        """测试回撤惩罚计算"""
        # 小于阈值的回撤
        small_drawdown = -0.01
        penalty_small = self.optimizer._calculate_drawdown_penalty(small_drawdown)
        assert penalty_small == 0.0
        
        # 超过阈值的回撤
        large_drawdown = -0.05
        penalty_large = self.optimizer._calculate_drawdown_penalty(large_drawdown)
        assert penalty_large > 0
        
        # 更大的回撤应该有更大的惩罚
        very_large_drawdown = -0.10
        penalty_very_large = self.optimizer._calculate_drawdown_penalty(very_large_drawdown)
        assert penalty_very_large > penalty_large
    
    def test_diversification_reward_calculation(self):
        """测试多样化奖励计算"""
        # 高度集中的持仓
        concentrated_positions = {'AAPL': 0.9, 'GOOGL': 0.1}
        reward_concentrated = self.optimizer._calculate_diversification_reward(concentrated_positions)
        
        # 分散的持仓
        diversified_positions = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
        reward_diversified = self.optimizer._calculate_diversification_reward(diversified_positions)
        
        # 分散持仓应该获得更高奖励
        assert reward_diversified > reward_concentrated
        
        # 空持仓
        empty_positions = {}
        reward_empty = self.optimizer._calculate_diversification_reward(empty_positions)
        assert reward_empty == 0.0
    
    def test_volatility_adjustment_calculation(self):
        """测试波动率调整计算"""
        returns = 0.01
        
        # 正常波动率
        normal_volatility = 0.15
        adjustment_normal = self.optimizer._calculate_volatility_adjustment(normal_volatility, returns)
        
        # 高波动率
        high_volatility = 0.35
        adjustment_high = self.optimizer._calculate_volatility_adjustment(high_volatility, returns)
        
        # 高波动率应该有负调整（惩罚）
        assert adjustment_high < adjustment_normal
        
        # 无波动率数据
        adjustment_none = self.optimizer._calculate_volatility_adjustment(None, returns)
        assert isinstance(adjustment_none, float)
    
    def test_optimize_reward_parameters(self):
        """测试奖励参数优化"""
        # 先添加一些历史数据
        for i in range(60):
            returns = np.random.normal(0.001, 0.02)
            drawdown = np.random.uniform(-0.05, 0)
            positions = {'AAPL': 0.5, 'GOOGL': 0.5}
            self.optimizer.calculate_risk_adjusted_reward(returns, drawdown, positions)
        
        target_metrics = {
            'target_sharpe': 2.0,
            'target_calmar': 1.5,
            'max_drawdown': 0.08,
            'target_volatility': 0.15
        }
        
        new_config = self.optimizer.optimize_reward_parameters(target_metrics)
        
        assert isinstance(new_config, RewardConfig)
        assert new_config.sharpe_target == 2.0
        assert new_config.calmar_target == 1.5
        assert new_config.max_volatility == 0.15
    
    def test_optimize_reward_parameters_insufficient_data(self):
        """测试数据不足时的参数优化"""
        # 只添加少量数据
        for i in range(10):
            returns = 0.001
            drawdown = -0.01
            positions = {'AAPL': 0.5, 'GOOGL': 0.5}
            self.optimizer.calculate_risk_adjusted_reward(returns, drawdown, positions)
        
        target_metrics = {'target_sharpe': 2.0}
        new_config = self.optimizer.optimize_reward_parameters(target_metrics)
        
        # 应该返回原配置
        assert new_config.drawdown_penalty_factor == self.config.drawdown_penalty_factor
    
    def test_performance_summary(self):
        """测试性能摘要"""
        # 添加一些历史数据
        for i in range(10):
            returns = np.random.normal(0.001, 0.01)
            drawdown = np.random.uniform(-0.03, 0)
            positions = {'AAPL': 0.4, 'GOOGL': 0.6}
            self.optimizer.calculate_risk_adjusted_reward(returns, drawdown, positions)
        
        summary = self.optimizer.get_performance_summary()
        
        assert 'episodes' in summary
        assert 'avg_return' in summary
        assert 'total_return' in summary
        assert 'risk_metrics' in summary
        assert 'reward_stats' in summary
        assert 'config' in summary
        assert summary['episodes'] == 10
    
    def test_performance_summary_no_data(self):
        """测试无数据时的性能摘要"""
        summary = self.optimizer.get_performance_summary()
        assert summary['status'] == '无历史数据'
    
    def test_reset(self):
        """测试重置功能"""
        # 添加一些数据
        self.optimizer.calculate_risk_adjusted_reward(0.01, -0.02, {'AAPL': 1.0})
        
        assert len(self.optimizer.return_history) > 0
        assert len(self.optimizer.reward_history) > 0
        
        # 重置
        self.optimizer.reset()
        
        assert len(self.optimizer.return_history) == 0
        assert len(self.optimizer.drawdown_history) == 0
        assert len(self.optimizer.position_history) == 0
        assert len(self.optimizer.reward_history) == 0
        assert self.optimizer.performance_stats['total_episodes'] == 0
    
    def test_history_length_limit(self):
        """测试历史数据长度限制"""
        # 设置较小的回看窗口
        config = RewardConfig(lookback_window=5)
        optimizer = RewardOptimizer(config)
        
        # 添加超过窗口大小的数据
        for i in range(10):
            optimizer.calculate_risk_adjusted_reward(0.01, -0.01, {'AAPL': 1.0})
        
        # 历史数据应该被限制在窗口大小内
        assert len(optimizer.return_history) == 5
        assert len(optimizer.drawdown_history) == 5
        assert len(optimizer.position_history) == 5
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 零收益
        reward_zero = self.optimizer.calculate_risk_adjusted_reward(0.0, 0.0, {})
        assert isinstance(reward_zero, float)
        
        # 极大收益但小回撤
        reward_large = self.optimizer.calculate_risk_adjusted_reward(0.1, -0.01, {'AAPL': 0.5, 'GOOGL': 0.5})
        assert isinstance(reward_large, float)  # 只检查类型，不检查符号
        
        # 极大回撤
        reward_large_dd = self.optimizer.calculate_risk_adjusted_reward(0.01, -0.2, {'AAPL': 1.0})
        assert reward_large_dd < 0  # 应该是负奖励
    
    def test_config_to_dict(self):
        """测试配置转换为字典"""
        config_dict = self.config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'base_return_weight' in config_dict
        assert 'drawdown_penalty_factor' in config_dict
        assert 'diversification_bonus' in config_dict
        assert config_dict['base_return_weight'] == 1.0
    
    def test_risk_metrics_to_dict(self):
        """测试风险指标转换为字典"""
        returns = [0.01, 0.02, -0.01, 0.015, -0.005]
        metrics = self.optimizer.calculate_risk_metrics(returns)
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'sharpe_ratio' in metrics_dict
        assert 'calmar_ratio' in metrics_dict
        assert 'max_drawdown' in metrics_dict
        assert 'volatility' in metrics_dict


if __name__ == '__main__':
    pytest.main([__file__])