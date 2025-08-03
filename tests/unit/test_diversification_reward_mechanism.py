"""
多样化奖励机制单元测试
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.rl_trading_system.risk_control.reward_optimizer import (
    RewardOptimizer,
    RewardConfig
)


class TestDiversificationRewardMechanism:
    """多样化奖励机制测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.config = RewardConfig(
            diversification_bonus=0.1,
            concentration_penalty=0.5,
            max_single_position=0.2
        )
        self.optimizer = RewardOptimizer(self.config)
    
    def test_basic_diversification_reward(self):
        """测试基础多样化奖励"""
        # 高度集中的持仓
        concentrated_positions = {'AAPL': 0.9, 'GOOGL': 0.1}
        basic_reward_concentrated = self.optimizer._calculate_basic_diversification_reward(concentrated_positions)
        
        # 分散的持仓
        diversified_positions = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
        basic_reward_diversified = self.optimizer._calculate_basic_diversification_reward(diversified_positions)
        
        # 分散持仓应该获得更高的基础奖励
        assert basic_reward_diversified > basic_reward_concentrated
        
        # 空持仓
        empty_positions = {}
        basic_reward_empty = self.optimizer._calculate_basic_diversification_reward(empty_positions)
        assert basic_reward_empty == 0.0
    
    def test_dynamic_diversification_reward(self):
        """测试动态多样化奖励"""
        # 没有历史数据时
        dynamic_reward = self.optimizer._calculate_dynamic_diversification_reward({'AAPL': 1.0})
        assert dynamic_reward == 0.0
        
        # 添加一些历史数据（高集中度）
        for i in range(5):
            concentrated_pos = {'AAPL': 0.8, 'GOOGL': 0.2}
            self.optimizer.position_history.append(concentrated_pos)
        
        # 当前持仓更分散
        current_positions = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.3}
        dynamic_reward = self.optimizer._calculate_dynamic_diversification_reward(current_positions)
        
        # 应该获得动态奖励（因为比历史更分散）
        assert dynamic_reward > 0.0
    
    def test_enhanced_concentration_penalty(self):
        """测试增强的集中度惩罚"""
        # 单一持仓过大
        over_concentrated = {'AAPL': 0.8, 'GOOGL': 0.2}
        penalty_over = self.optimizer._calculate_enhanced_concentration_penalty(over_concentrated)
        assert penalty_over > 0.0
        
        # 正常分散持仓
        normal_positions = {'AAPL': 0.15, 'GOOGL': 0.15, 'MSFT': 0.15, 'TSLA': 0.15, 'NVDA': 0.4}
        penalty_normal = self.optimizer._calculate_enhanced_concentration_penalty(normal_positions)
        
        # 过度集中的惩罚应该更大
        assert penalty_over > penalty_normal
        
        # 测试前3大持仓占比过高的情况
        top3_concentrated = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.2, 'TSLA': 0.1}
        penalty_top3 = self.optimizer._calculate_enhanced_concentration_penalty(top3_concentrated)
        assert penalty_top3 > 0.0
    
    def test_correlation_adjustment_reward(self):
        """测试相关性调整奖励"""
        # 单一资产
        single_asset = {'AAPL': 1.0}
        correlation_reward_single = self.optimizer._calculate_correlation_adjustment_reward(single_asset)
        assert correlation_reward_single == 0.0
        
        # 多资产
        multi_assets = {'AAPL': 0.2, 'GOOGL': 0.2, 'MSFT': 0.2, 'TSLA': 0.2, 'NVDA': 0.2}
        correlation_reward_multi = self.optimizer._calculate_correlation_adjustment_reward(multi_assets)
        assert correlation_reward_multi > 0.0
        
        # 更多资产应该获得更高的相关性调整奖励
        more_assets = {f'STOCK_{i}': 0.1 for i in range(10)}
        correlation_reward_more = self.optimizer._calculate_correlation_adjustment_reward(more_assets)
        assert correlation_reward_more > correlation_reward_multi
    
    def test_concentration_score_calculation(self):
        """测试集中度评分计算"""
        # 完全集中
        fully_concentrated = {'AAPL': 1.0}
        score_concentrated = self.optimizer._calculate_concentration_score(fully_concentrated)
        assert score_concentrated == 1.0
        
        # 完全分散（4个等权重资产）
        equally_weighted = {'AAPL': 0.25, 'GOOGL': 0.25, 'MSFT': 0.25, 'TSLA': 0.25}
        score_diversified = self.optimizer._calculate_concentration_score(equally_weighted)
        assert score_diversified == 0.25  # 1/4 = 0.25
        
        # 空持仓
        empty_positions = {}
        score_empty = self.optimizer._calculate_concentration_score(empty_positions)
        assert score_empty == 0.0
    
    def test_portfolio_diversification_metrics(self):
        """测试投资组合多样化指标计算"""
        positions = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.2, 'TSLA': 0.1}
        
        metrics = self.optimizer.calculate_portfolio_diversification_metrics(positions)
        
        # 检查所有指标都存在
        expected_keys = [
            'herfindahl_index', 'effective_number_of_assets', 'max_weight',
            'weight_entropy', 'diversification_ratio', 'concentration_score'
        ]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
        
        # 检查指标的合理性
        assert 0 <= metrics['herfindahl_index'] <= 1
        assert metrics['effective_number_of_assets'] > 0
        assert abs(metrics['max_weight'] - 0.4) < 1e-10  # 最大权重
        assert metrics['weight_entropy'] > 0
        assert 0 <= metrics['diversification_ratio'] <= 1
        assert metrics['concentration_score'] == metrics['herfindahl_index']
    
    def test_portfolio_diversification_metrics_empty(self):
        """测试空持仓的多样化指标"""
        empty_positions = {}
        metrics = self.optimizer.calculate_portfolio_diversification_metrics(empty_positions)
        
        # 所有指标都应该为0
        for value in metrics.values():
            assert value == 0.0
    
    def test_optimize_diversification_parameters(self):
        """测试多样化参数优化"""
        # 无历史数据
        suggestions = self.optimizer.optimize_diversification_parameters()
        assert suggestions['status'] == '无历史数据'
        
        # 添加历史数据（低多样化）
        for i in range(20):
            concentrated_pos = {'AAPL': 0.7, 'GOOGL': 0.3}
            self.optimizer.position_history.append(concentrated_pos)
        
        # 目标多样化水平
        target_diversification = 0.8
        suggestions = self.optimizer.optimize_diversification_parameters(target_diversification)
        
        # 检查建议内容
        assert 'current_diversification' in suggestions
        assert 'target_diversification' in suggestions
        assert 'diversification_gap' in suggestions
        assert 'diversification_bonus_adjustment' in suggestions
        assert 'concentration_penalty_adjustment' in suggestions
        assert 'max_position_adjustment' in suggestions
        
        # 检查多样化差距
        assert suggestions['diversification_gap'] != 0
        
        # 如果多样化不足，应该有相应的调整建议
        if suggestions['diversification_gap'] > 0:
            assert suggestions['diversification_bonus_adjustment'] >= 1.0
            assert suggestions['concentration_penalty_adjustment'] >= 1.0
            assert suggestions['max_position_adjustment'] <= self.config.max_single_position
    
    def test_optimize_diversification_parameters_sufficient(self):
        """测试多样化已足够时的参数优化"""
        # 添加历史数据（高多样化）
        for i in range(20):
            diversified_pos = {'AAPL': 0.2, 'GOOGL': 0.2, 'MSFT': 0.2, 'TSLA': 0.2, 'NVDA': 0.2}
            self.optimizer.position_history.append(diversified_pos)
        
        target_diversification = 0.6  # 较低的目标
        suggestions = self.optimizer.optimize_diversification_parameters(target_diversification)
        
        # 多样化已足够，不需要调整
        assert suggestions['diversification_bonus_adjustment'] == 1.0
        assert suggestions['concentration_penalty_adjustment'] == 1.0
        assert suggestions['max_position_adjustment'] == self.config.max_single_position
    
    def test_comprehensive_diversification_reward(self):
        """测试综合多样化奖励机制"""
        # 添加一些历史数据
        for i in range(10):
            hist_pos = {'AAPL': 0.6, 'GOOGL': 0.4}
            self.optimizer.position_history.append(hist_pos)
        
        # 测试不同类型的持仓
        test_cases = [
            ({'AAPL': 1.0}, '完全集中'),
            ({'AAPL': 0.5, 'GOOGL': 0.5}, '适度分散'),
            ({'AAPL': 0.2, 'GOOGL': 0.2, 'MSFT': 0.2, 'TSLA': 0.2, 'NVDA': 0.2}, '高度分散'),
            ({'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.2, 'TSLA': 0.15, 'NVDA': 0.1}, '渐进分散')
        ]
        
        rewards = []
        for positions, description in test_cases:
            reward = self.optimizer._calculate_diversification_reward(positions)
            rewards.append((reward, description))
        
        # 验证奖励趋势：更分散的持仓应该获得更高奖励
        assert rewards[2][0] > rewards[1][0]  # 高度分散 > 适度分散
        assert rewards[1][0] > rewards[0][0]  # 适度分散 > 完全集中
    
    def test_diversification_reward_with_negative_positions(self):
        """测试包含空头持仓的多样化奖励"""
        # 包含空头持仓
        positions_with_shorts = {'AAPL': 0.5, 'GOOGL': -0.2, 'MSFT': 0.3, 'TSLA': 0.4}
        
        # 计算各种奖励组件
        basic_reward = self.optimizer._calculate_basic_diversification_reward(positions_with_shorts)
        concentration_penalty = self.optimizer._calculate_enhanced_concentration_penalty(positions_with_shorts)
        correlation_adjustment = self.optimizer._calculate_correlation_adjustment_reward(positions_with_shorts)
        
        # 所有组件都应该正常工作
        assert isinstance(basic_reward, float)
        assert isinstance(concentration_penalty, float)
        assert isinstance(correlation_adjustment, float)
        assert correlation_adjustment > 0  # 多资产应该有正的相关性调整
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 零权重持仓
        zero_positions = {'AAPL': 0.0, 'GOOGL': 0.0}
        reward_zero = self.optimizer._calculate_diversification_reward(zero_positions)
        assert reward_zero == 0.0
        
        # 极小权重
        tiny_positions = {'AAPL': 1e-10, 'GOOGL': 1e-10}
        reward_tiny = self.optimizer._calculate_diversification_reward(tiny_positions)
        assert isinstance(reward_tiny, float)
        
        # 单一资产但权重小于阈值
        small_single = {'AAPL': 0.1}
        reward_small_single = self.optimizer._calculate_diversification_reward(small_single)
        assert isinstance(reward_small_single, float)
    
    def test_diversification_metrics_consistency(self):
        """测试多样化指标的一致性"""
        positions = {'AAPL': 0.4, 'GOOGL': 0.3, 'MSFT': 0.2, 'TSLA': 0.1}
        
        # 通过不同方法计算集中度
        concentration_score1 = self.optimizer._calculate_concentration_score(positions)
        metrics = self.optimizer.calculate_portfolio_diversification_metrics(positions)
        concentration_score2 = metrics['concentration_score']
        
        # 两种方法应该得到相同结果
        assert abs(concentration_score1 - concentration_score2) < 1e-10
        
        # 有效资产数量应该与赫芬达尔指数一致
        expected_effective_assets = 1.0 / metrics['herfindahl_index']
        assert abs(metrics['effective_number_of_assets'] - expected_effective_assets) < 1e-10
    
    def test_parameter_sensitivity(self):
        """测试参数敏感性"""
        positions = {'AAPL': 0.3, 'GOOGL': 0.3, 'MSFT': 0.4}
        
        # 测试不同的多样化奖励系数
        configs = [
            RewardConfig(diversification_bonus=0.05),
            RewardConfig(diversification_bonus=0.1),
            RewardConfig(diversification_bonus=0.2)
        ]
        
        rewards = []
        for config in configs:
            optimizer = RewardOptimizer(config)
            reward = optimizer._calculate_basic_diversification_reward(positions)
            rewards.append(reward)
        
        # 更高的奖励系数应该产生更高的奖励
        assert rewards[2] > rewards[1] > rewards[0]


if __name__ == '__main__':
    pytest.main([__file__])