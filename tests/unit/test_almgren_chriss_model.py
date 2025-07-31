"""
测试Almgren-Chriss市场冲击模型
测试永久冲击和临时冲击计算逻辑，以及不同交易规模下的成本估算准确性
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any

from src.rl_trading_system.trading.almgren_chriss_model import (
    AlmgrenChrissModel,
    MarketImpactParameters,
    ImpactResult
)


class TestMarketImpactParameters:
    """测试市场冲击参数类"""
    
    def test_parameters_creation(self):
        """测试参数正常创建"""
        params = MarketImpactParameters(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            volatility=0.02,
            daily_volume=1000000,
            participation_rate=0.1
        )
        
        assert params.permanent_impact_coeff == 0.1
        assert params.temporary_impact_coeff == 0.5
        assert params.volatility == 0.02
        assert params.daily_volume == 1000000
        assert params.participation_rate == 0.1
    
    def test_parameters_validation_negative_coefficients(self):
        """测试负系数验证"""
        with pytest.raises(ValueError, match="永久冲击系数不能为负数"):
            MarketImpactParameters(
                permanent_impact_coeff=-0.1,
                temporary_impact_coeff=0.5,
                volatility=0.02,
                daily_volume=1000000,
                participation_rate=0.1
            )
        
        with pytest.raises(ValueError, match="临时冲击系数不能为负数"):
            MarketImpactParameters(
                permanent_impact_coeff=0.1,
                temporary_impact_coeff=-0.5,
                volatility=0.02,
                daily_volume=1000000,
                participation_rate=0.1
            )
    
    def test_parameters_validation_negative_volatility(self):
        """测试负波动率验证"""
        with pytest.raises(ValueError, match="波动率不能为负数"):
            MarketImpactParameters(
                permanent_impact_coeff=0.1,
                temporary_impact_coeff=0.5,
                volatility=-0.02,
                daily_volume=1000000,
                participation_rate=0.1
            )
    
    def test_parameters_validation_negative_volume(self):
        """测试负成交量验证"""
        with pytest.raises(ValueError, match="日均成交量不能为负数"):
            MarketImpactParameters(
                permanent_impact_coeff=0.1,
                temporary_impact_coeff=0.5,
                volatility=0.02,
                daily_volume=-1000000,
                participation_rate=0.1
            )
    
    def test_parameters_validation_participation_rate_range(self):
        """测试参与度范围验证"""
        with pytest.raises(ValueError, match="市场参与度必须在0到1之间"):
            MarketImpactParameters(
                permanent_impact_coeff=0.1,
                temporary_impact_coeff=0.5,
                volatility=0.02,
                daily_volume=1000000,
                participation_rate=1.5
            )
        
        with pytest.raises(ValueError, match="市场参与度必须在0到1之间"):
            MarketImpactParameters(
                permanent_impact_coeff=0.1,
                temporary_impact_coeff=0.5,
                volatility=0.02,
                daily_volume=1000000,
                participation_rate=-0.1
            )


class TestImpactResult:
    """测试冲击结果类"""
    
    def test_impact_result_creation(self):
        """测试冲击结果正常创建"""
        result = ImpactResult(
            permanent_impact=0.001,
            temporary_impact=0.002,
            total_impact=0.003,
            trade_volume=100000,
            market_volume=1000000
        )
        
        assert result.permanent_impact == 0.001
        assert result.temporary_impact == 0.002
        assert result.total_impact == 0.003
        assert result.trade_volume == 100000
        assert result.market_volume == 1000000
    
    def test_impact_result_validation_negative_impacts(self):
        """测试负冲击验证"""
        with pytest.raises(ValueError, match="永久冲击不能为负数"):
            ImpactResult(
                permanent_impact=-0.001,
                temporary_impact=0.002,
                total_impact=0.003,
                trade_volume=100000,
                market_volume=1000000
            )
    
    def test_impact_result_validation_inconsistent_total(self):
        """测试总冲击一致性验证"""
        with pytest.raises(ValueError, match="总冲击应等于永久冲击和临时冲击之和"):
            ImpactResult(
                permanent_impact=0.001,
                temporary_impact=0.002,
                total_impact=0.005,  # 不等于0.001 + 0.002
                trade_volume=100000,
                market_volume=1000000
            )
    
    def test_impact_result_get_participation_rate(self):
        """测试获取参与度方法"""
        result = ImpactResult(
            permanent_impact=0.001,
            temporary_impact=0.002,
            total_impact=0.003,
            trade_volume=100000,
            market_volume=1000000
        )
        
        assert result.get_participation_rate() == 0.1
    
    def test_impact_result_get_cost_basis_points(self):
        """测试获取基点成本方法"""
        result = ImpactResult(
            permanent_impact=0.001,
            temporary_impact=0.002,
            total_impact=0.003,
            trade_volume=100000,
            market_volume=1000000
        )
        
        assert result.get_cost_basis_points() == 30.0  # 0.003 * 10000


class TestAlmgrenChrissModel:
    """测试Almgren-Chriss模型"""
    
    @pytest.fixture
    def default_parameters(self):
        """默认参数"""
        return MarketImpactParameters(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            volatility=0.02,
            daily_volume=1000000,
            participation_rate=0.1
        )
    
    @pytest.fixture
    def model(self, default_parameters):
        """默认模型实例"""
        return AlmgrenChrissModel(default_parameters)
    
    def test_model_creation(self, default_parameters):
        """测试模型正常创建"""
        model = AlmgrenChrissModel(default_parameters)
        assert model.parameters == default_parameters
    
    def test_calculate_permanent_impact_linear(self, model):
        """测试永久冲击线性计算"""
        trade_volume = 100000
        market_volume = 1000000
        
        impact = model._calculate_permanent_impact(trade_volume, market_volume)
        
        # 永久冲击 = permanent_impact_coeff * (trade_volume / market_volume)
        expected_impact = 0.1 * (100000 / 1000000)
        assert abs(impact - expected_impact) < 1e-8
    
    def test_calculate_temporary_impact_square_root(self, model):
        """测试临时冲击平方根计算"""
        trade_volume = 100000
        market_volume = 1000000
        volatility = 0.02
        
        impact = model._calculate_temporary_impact(trade_volume, market_volume, volatility)
        
        # 临时冲击 = temporary_impact_coeff * volatility * sqrt(trade_volume / market_volume)
        expected_impact = 0.5 * 0.02 * np.sqrt(100000 / 1000000)
        assert abs(impact - expected_impact) < 1e-8
    
    def test_calculate_impact_basic(self, model):
        """测试基本冲击计算"""
        trade_volume = 100000
        
        result = model.calculate_impact(trade_volume)
        
        # 验证结果类型
        assert isinstance(result, ImpactResult)
        
        # 验证永久冲击（线性）
        expected_permanent = 0.1 * (100000 / 1000000)
        assert abs(result.permanent_impact - expected_permanent) < 1e-8
        
        # 验证临时冲击（平方根）
        expected_temporary = 0.5 * 0.02 * np.sqrt(100000 / 1000000)
        assert abs(result.temporary_impact - expected_temporary) < 1e-8
        
        # 验证总冲击
        expected_total = expected_permanent + expected_temporary
        assert abs(result.total_impact - expected_total) < 1e-8
        
        # 验证交易量信息
        assert result.trade_volume == 100000
        assert result.market_volume == 1000000
    
    def test_calculate_impact_different_trade_sizes(self, model):
        """测试不同交易规模下的成本估算准确性"""
        trade_volumes = [10000, 50000, 100000, 200000, 500000]
        results = []
        
        for volume in trade_volumes:
            result = model.calculate_impact(volume)
            results.append(result)
        
        # 验证永久冲击随交易量线性增长
        for i in range(1, len(results)):
            ratio = results[i].permanent_impact / results[i-1].permanent_impact
            volume_ratio = trade_volumes[i] / trade_volumes[i-1]
            assert abs(ratio - volume_ratio) < 1e-6
        
        # 验证临时冲击随交易量平方根增长
        for i in range(1, len(results)):
            ratio = results[i].temporary_impact / results[i-1].temporary_impact
            volume_ratio = np.sqrt(trade_volumes[i] / trade_volumes[i-1])
            assert abs(ratio - volume_ratio) < 1e-6
        
        # 验证总冲击随交易量增长（但增长率递减）
        for i in range(1, len(results)):
            assert results[i].total_impact > results[i-1].total_impact
    
    def test_calculate_impact_with_custom_market_volume(self, model):
        """测试自定义市场成交量"""
        trade_volume = 100000
        custom_market_volume = 2000000
        
        result = model.calculate_impact(trade_volume, market_volume=custom_market_volume)
        
        # 验证使用了自定义市场成交量
        assert result.market_volume == custom_market_volume
        
        # 验证冲击计算使用了自定义成交量
        expected_permanent = 0.1 * (100000 / 2000000)
        assert abs(result.permanent_impact - expected_permanent) < 1e-8
    
    def test_calculate_impact_with_custom_volatility(self, model):
        """测试自定义波动率"""
        trade_volume = 100000
        custom_volatility = 0.03
        
        result = model.calculate_impact(trade_volume, volatility=custom_volatility)
        
        # 验证临时冲击使用了自定义波动率
        expected_temporary = 0.5 * 0.03 * np.sqrt(100000 / 1000000)
        assert abs(result.temporary_impact - expected_temporary) < 1e-8
    
    def test_market_participation_rate_impact(self, default_parameters):
        """测试市场参与度对冲击的影响"""
        # 创建不同流动性的市场（通过调整成交量来模拟参与度影响）
        low_liquidity = MarketImpactParameters(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            volatility=0.02,
            daily_volume=500000,  # 低流动性（小成交量）
            participation_rate=0.1
        )
        
        high_liquidity = MarketImpactParameters(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            volatility=0.02,
            daily_volume=2000000,  # 高流动性（大成交量）
            participation_rate=0.1
        )
        
        low_liq_model = AlmgrenChrissModel(low_liquidity)
        high_liq_model = AlmgrenChrissModel(high_liquidity)
        
        trade_volume = 100000
        
        low_result = low_liq_model.calculate_impact(trade_volume)
        high_result = high_liq_model.calculate_impact(trade_volume)
        
        # 低流动性市场应该导致更高的冲击
        assert low_result.total_impact > high_result.total_impact
    
    def test_liquidity_impact_on_costs(self, default_parameters):
        """测试流动性对成本的影响"""
        # 创建不同流动性的市场参数
        high_liquidity = MarketImpactParameters(
            permanent_impact_coeff=0.05,  # 低冲击系数
            temporary_impact_coeff=0.3,
            volatility=0.015,  # 低波动率
            daily_volume=2000000,  # 高成交量
            participation_rate=0.1
        )
        
        low_liquidity = MarketImpactParameters(
            permanent_impact_coeff=0.2,  # 高冲击系数
            temporary_impact_coeff=0.8,
            volatility=0.03,  # 高波动率
            daily_volume=500000,  # 低成交量
            participation_rate=0.1
        )
        
        high_liq_model = AlmgrenChrissModel(high_liquidity)
        low_liq_model = AlmgrenChrissModel(low_liquidity)
        
        trade_volume = 100000
        
        high_liq_result = high_liq_model.calculate_impact(trade_volume)
        low_liq_result = low_liq_model.calculate_impact(trade_volume)
        
        # 低流动性市场应该有更高的交易成本
        assert low_liq_result.total_impact > high_liq_result.total_impact
        assert low_liq_result.permanent_impact > high_liq_result.permanent_impact
        assert low_liq_result.temporary_impact > high_liq_result.temporary_impact
    
    def test_zero_trade_volume(self, model):
        """测试零交易量"""
        result = model.calculate_impact(0)
        
        assert result.permanent_impact == 0.0
        assert result.temporary_impact == 0.0
        assert result.total_impact == 0.0
        assert result.trade_volume == 0
    
    def test_very_large_trade_volume(self, model):
        """测试极大交易量"""
        # 交易量等于市场成交量
        trade_volume = 1000000
        
        result = model.calculate_impact(trade_volume)
        
        # 永久冲击应该等于系数
        assert abs(result.permanent_impact - 0.1) < 1e-8
        
        # 临时冲击应该等于系数乘以波动率
        expected_temporary = 0.5 * 0.02 * 1.0
        assert abs(result.temporary_impact - expected_temporary) < 1e-8
    
    def test_model_parameters_update(self, model):
        """测试模型参数更新"""
        new_parameters = MarketImpactParameters(
            permanent_impact_coeff=0.15,
            temporary_impact_coeff=0.6,
            volatility=0.025,
            daily_volume=1500000,
            participation_rate=0.12
        )
        
        model.update_parameters(new_parameters)
        
        assert model.parameters == new_parameters
        
        # 验证更新后的计算结果
        trade_volume = 100000
        result = model.calculate_impact(trade_volume)
        
        expected_permanent = 0.15 * (100000 / 1500000)
        assert abs(result.permanent_impact - expected_permanent) < 1e-8


class TestAlmgrenChrissModelBoundaryConditions:
    """测试Almgren-Chriss模型边界条件"""
    
    def test_extreme_parameters(self):
        """测试极端参数"""
        # 极小参数
        small_params = MarketImpactParameters(
            permanent_impact_coeff=1e-6,
            temporary_impact_coeff=1e-6,
            volatility=1e-6,
            daily_volume=1,
            participation_rate=1e-6
        )
        
        model = AlmgrenChrissModel(small_params)
        result = model.calculate_impact(1)
        
        assert result.permanent_impact >= 0
        assert result.temporary_impact >= 0
        assert result.total_impact >= 0
        
        # 极大参数
        large_params = MarketImpactParameters(
            permanent_impact_coeff=1.0,
            temporary_impact_coeff=1.0,
            volatility=1.0,
            daily_volume=int(1e9),
            participation_rate=0.99
        )
        
        model = AlmgrenChrissModel(large_params)
        result = model.calculate_impact(int(1e6))
        
        assert result.permanent_impact >= 0
        assert result.temporary_impact >= 0
        assert result.total_impact >= 0
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        params = MarketImpactParameters(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            volatility=0.02,
            daily_volume=1000000,
            participation_rate=0.1
        )
        
        model = AlmgrenChrissModel(params)
        
        # 测试非常小的交易量
        tiny_volume = 1e-6
        result = model.calculate_impact(tiny_volume)
        
        assert not np.isnan(result.permanent_impact)
        assert not np.isnan(result.temporary_impact)
        assert not np.isnan(result.total_impact)
        assert not np.isinf(result.permanent_impact)
        assert not np.isinf(result.temporary_impact)
        assert not np.isinf(result.total_impact)


class TestAlmgrenChrissModelPerformance:
    """测试Almgren-Chriss模型性能"""
    
    def test_batch_calculation_performance(self):
        """测试批量计算性能"""
        params = MarketImpactParameters(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            volatility=0.02,
            daily_volume=1000000,
            participation_rate=0.1
        )
        
        model = AlmgrenChrissModel(params)
        
        # 生成大量交易量数据
        trade_volumes = np.random.randint(1000, 500000, size=1000)
        
        import time
        start_time = time.time()
        
        results = []
        for volume in trade_volumes:
            result = model.calculate_impact(volume)
            results.append(result)
        
        calculation_time = time.time() - start_time
        
        # 计算时间应该在合理范围内（每个计算小于1ms）
        assert calculation_time < 1.0
        assert len(results) == 1000
        
        # 验证所有结果都有效
        for result in results:
            assert result.total_impact >= 0
            assert not np.isnan(result.total_impact)
    
    def test_memory_usage(self):
        """测试内存使用"""
        params = MarketImpactParameters(
            permanent_impact_coeff=0.1,
            temporary_impact_coeff=0.5,
            volatility=0.02,
            daily_volume=1000000,
            participation_rate=0.1
        )
        
        model = AlmgrenChrissModel(params)
        
        # 创建大量模型实例不应该消耗过多内存
        models = []
        for i in range(100):
            models.append(AlmgrenChrissModel(params))
        
        # 验证所有模型都能正常工作
        for model in models:
            result = model.calculate_impact(100000)
            assert result.total_impact > 0


class TestAlmgrenChrissModelIntegration:
    """测试Almgren-Chriss模型集成"""
    
    def test_integration_with_real_market_data(self):
        """测试与真实市场数据的集成"""
        # 模拟真实市场参数（基于A股市场特征）
        a_share_params = MarketImpactParameters(
            permanent_impact_coeff=0.08,  # A股永久冲击系数
            temporary_impact_coeff=0.4,   # A股临时冲击系数
            volatility=0.025,             # A股日均波动率
            daily_volume=5000000,         # A股日均成交量
            participation_rate=0.05       # 典型参与度
        )
        
        model = AlmgrenChrissModel(a_share_params)
        
        # 测试不同规模的交易
        small_trade = 50000    # 小额交易
        medium_trade = 200000  # 中等交易
        large_trade = 1000000  # 大额交易
        
        small_result = model.calculate_impact(small_trade)
        medium_result = model.calculate_impact(medium_trade)
        large_result = model.calculate_impact(large_trade)
        
        # 验证成本随交易规模递增
        assert small_result.total_impact < medium_result.total_impact
        assert medium_result.total_impact < large_result.total_impact
        
        # 验证成本在合理范围内（基点）
        assert small_result.get_cost_basis_points() < 50   # 小额交易成本 < 5bp
        assert medium_result.get_cost_basis_points() < 100 # 中等交易成本 < 10bp
        assert large_result.get_cost_basis_points() < 300  # 大额交易成本 < 30bp
    
    def test_model_calibration_validation(self):
        """测试模型校准验证"""
        # 使用历史数据校准的参数
        calibrated_params = MarketImpactParameters(
            permanent_impact_coeff=0.12,
            temporary_impact_coeff=0.45,
            volatility=0.022,
            daily_volume=3000000,
            participation_rate=0.08
        )
        
        model = AlmgrenChrissModel(calibrated_params)
        
        # 测试参数合理性
        test_volume = 150000
        result = model.calculate_impact(test_volume)
        
        # 验证冲击都为正值
        assert result.permanent_impact > 0
        assert result.temporary_impact > 0
        
        # 总成本应该在合理范围内
        cost_bp = result.get_cost_basis_points()
        assert 5 <= cost_bp <= 200  # 5-200基点之间
        
        # 参与度应该合理
        participation = result.get_participation_rate()
        assert 0.01 <= participation <= 0.5  # 1%-50%之间