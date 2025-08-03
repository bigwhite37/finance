"""
测试数据自动生成和模拟工具

该模块提供各种测试数据生成功能，用于支持回撤控制系统的全面测试。
包括：
- 投资组合净值数据生成
- 市场数据模拟
- 回撤场景生成
- 压力测试数据构造
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import random
from enum import Enum
import sys
import os

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from rl_trading_system.risk_control.adaptive_risk_budget import PerformanceMetrics, MarketMetrics
except ImportError:
    # 如果导入失败，定义简化版本
    @dataclass
    class PerformanceMetrics:
        sharpe_ratio: float = 0.0
        return_rate: float = 0.0
        volatility: float = 0.0
        max_drawdown: float = 0.0
        consecutive_losses: int = 0
    
    @dataclass  
    class MarketMetrics:
        market_volatility: float = 0.0
        market_trend: float = 0.0
        uncertainty_index: float = 0.0
        correlation_breakdown: bool = False


class MarketScenario(Enum):
    """市场场景类型"""
    BULL_MARKET = "牛市"
    BEAR_MARKET = "熊市"
    SIDEWAYS = "震荡市"
    HIGH_VOLATILITY = "高波动"
    CRASH = "崩盘"
    RECOVERY = "恢复"
    BUBBLE = "泡沫"


@dataclass
class DataGenerationConfig:
    """数据生成配置"""
    # 基础参数
    start_date: datetime = datetime(2020, 1, 1)
    end_date: datetime = datetime(2023, 12, 31)
    frequency: str = 'D'  # 'D'=日频, 'H'=小时频
    
    # 随机种子
    random_seed: Optional[int] = 42
    
    # 投资组合参数
    initial_value: float = 100000.0
    num_assets: int = 10
    
    # 市场参数
    base_return: float = 0.0008  # 日均收益率
    base_volatility: float = 0.015  # 日波动率
    
    # 特殊场景参数
    crash_probability: float = 0.05  # 崩盘概率
    bubble_probability: float = 0.03  # 泡沫概率


class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self, config: DataGenerationConfig = None):
        """
        初始化数据生成器
        
        Args:
            config: 数据生成配置
        """
        self.config = config or DataGenerationConfig()
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            random.seed(self.config.random_seed)
    
    def generate_portfolio_values(self, 
                                scenario: MarketScenario = MarketScenario.BULL_MARKET,
                                length: int = 252) -> np.ndarray:
        """
        生成投资组合净值序列
        
        Args:
            scenario: 市场场景
            length: 数据长度
            
        Returns:
            投资组合净值序列
        """
        returns = self._generate_returns(scenario, length)
        values = self.config.initial_value * np.cumprod(1 + returns)
        return values
    
    def generate_market_data(self, 
                           scenario: MarketScenario = MarketScenario.BULL_MARKET,
                           length: int = 252) -> pd.DataFrame:
        """
        生成完整的市场数据
        
        Args:
            scenario: 市场场景
            length: 数据长度
            
        Returns:
            包含价格、收益率、成交量等的DataFrame
        """
        dates = pd.date_range(
            start=self.config.start_date,
            periods=length,
            freq=self.config.frequency
        )
        
        # 生成收益率
        returns = self._generate_returns(scenario, length)
        
        # 生成价格
        prices = self.config.initial_value * np.cumprod(1 + returns)
        
        # 生成成交量（对数正态分布）
        base_volume = 1000000
        volume_returns = np.random.lognormal(
            mean=np.log(base_volume),
            sigma=0.3,
            size=length
        )
        
        # 生成波动率
        volatility = self._generate_volatility(scenario, length)
        
        return pd.DataFrame({
            'date': dates,
            'price': prices,
            'returns': returns,
            'volume': volume_returns,
            'volatility': volatility,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, length))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, length))),
            'open': np.roll(prices, 1),  # 简化的开盘价
            'close': prices
        })
    
    def generate_drawdown_scenarios(self) -> Dict[str, np.ndarray]:
        """
        生成各种回撤场景的数据
        
        Returns:
            包含不同回撤场景的字典
        """
        scenarios = {}
        
        # 1. 轻微回撤场景
        scenarios['light_drawdown'] = self._generate_specific_drawdown(
            max_drawdown=0.05, recovery_speed=0.8
        )
        
        # 2. 中等回撤场景
        scenarios['moderate_drawdown'] = self._generate_specific_drawdown(
            max_drawdown=0.15, recovery_speed=0.5
        )
        
        # 3. 严重回撤场景
        scenarios['severe_drawdown'] = self._generate_specific_drawdown(
            max_drawdown=0.35, recovery_speed=0.3
        )
        
        # 4. 长期回撤场景
        scenarios['prolonged_drawdown'] = self._generate_prolonged_drawdown(
            drawdown_duration=100, max_drawdown=0.25
        )
        
        # 5. 快速恢复场景
        scenarios['quick_recovery'] = self._generate_specific_drawdown(
            max_drawdown=0.20, recovery_speed=1.2
        )
        
        # 6. 多次回撤场景
        scenarios['multiple_drawdowns'] = self._generate_multiple_drawdowns(
            num_drawdowns=3, avg_drawdown=0.12
        )
        
        return scenarios
    
    def generate_stress_test_data(self) -> Dict[str, Any]:
        """
        生成压力测试数据
        
        Returns:
            压力测试场景数据
        """
        stress_scenarios = {}
        
        # 1. 极端波动场景
        stress_scenarios['extreme_volatility'] = {
            'returns': np.random.normal(0, 0.05, 252),  # 5%日波动率
            'description': '极端波动场景'
        }
        
        # 2. 连续下跌场景
        consecutive_down = -np.abs(np.random.normal(0.02, 0.01, 20))  # 连续20天下跌
        normal_returns = np.random.normal(0.001, 0.015, 232)
        stress_scenarios['consecutive_decline'] = {
            'returns': np.concatenate([consecutive_down, normal_returns]),
            'description': '连续下跌场景'
        }
        
        # 3. 跳空缺口场景
        gap_returns = np.random.normal(0.001, 0.015, 252)
        gap_returns[50] = -0.15  # 15%跳空下跌
        gap_returns[150] = 0.10   # 10%跳空上涨
        stress_scenarios['price_gaps'] = {
            'returns': gap_returns,
            'description': '跳空缺口场景'
        }
        
        # 4. 流动性危机场景
        liquidity_crisis = self._generate_liquidity_crisis_scenario()
        stress_scenarios['liquidity_crisis'] = liquidity_crisis
        
        # 5. 市场崩盘恢复场景
        crash_recovery = self._generate_crash_recovery_scenario()
        stress_scenarios['crash_recovery'] = crash_recovery
        
        return stress_scenarios
    
    def generate_performance_metrics(self, 
                                   scenario: MarketScenario = MarketScenario.BULL_MARKET,
                                   count: int = 10) -> List[PerformanceMetrics]:
        """
        生成性能指标数据
        
        Args:
            scenario: 市场场景
            count: 生成数量
            
        Returns:
            性能指标列表
        """
        metrics_list = []
        
        for _ in range(count):
            if scenario == MarketScenario.BULL_MARKET:
                sharpe_ratio = np.random.normal(1.5, 0.3)
                return_rate = np.random.normal(0.12, 0.05)
                max_drawdown = -np.random.uniform(0.02, 0.08)
                consecutive_losses = np.random.randint(0, 3)
            
            elif scenario == MarketScenario.BEAR_MARKET:
                sharpe_ratio = np.random.normal(-0.5, 0.5)
                return_rate = np.random.normal(-0.08, 0.06)
                max_drawdown = -np.random.uniform(0.15, 0.40)
                consecutive_losses = np.random.randint(3, 10)
            
            elif scenario == MarketScenario.HIGH_VOLATILITY:
                sharpe_ratio = np.random.normal(0.2, 0.8)
                return_rate = np.random.normal(0.05, 0.15)
                max_drawdown = -np.random.uniform(0.08, 0.25)
                consecutive_losses = np.random.randint(1, 6)
            
            else:  # SIDEWAYS
                sharpe_ratio = np.random.normal(0.3, 0.4)
                return_rate = np.random.normal(0.02, 0.03)
                max_drawdown = -np.random.uniform(0.03, 0.12)
                consecutive_losses = np.random.randint(0, 4)
            
            metrics = PerformanceMetrics(
                sharpe_ratio=sharpe_ratio,
                return_rate=return_rate,
                volatility=abs(return_rate) / max(abs(sharpe_ratio), 0.1),
                max_drawdown=max_drawdown,
                consecutive_losses=consecutive_losses
            )
            metrics_list.append(metrics)
        
        return metrics_list
    
    def generate_market_metrics(self, 
                              scenario: MarketScenario = MarketScenario.BULL_MARKET,
                              count: int = 10) -> List[MarketMetrics]:
        """
        生成市场指标数据
        
        Args:
            scenario: 市场场景
            count: 生成数量
            
        Returns:
            市场指标列表
        """
        metrics_list = []
        
        for _ in range(count):
            if scenario == MarketScenario.BULL_MARKET:
                market_volatility = np.random.uniform(0.08, 0.15)
                market_trend = np.random.uniform(0.05, 0.12)
                uncertainty_index = np.random.uniform(0.1, 0.4)
            
            elif scenario == MarketScenario.BEAR_MARKET:
                market_volatility = np.random.uniform(0.20, 0.40)
                market_trend = np.random.uniform(-0.15, -0.05)
                uncertainty_index = np.random.uniform(0.6, 0.9)
            
            elif scenario == MarketScenario.HIGH_VOLATILITY:
                market_volatility = np.random.uniform(0.25, 0.50)
                market_trend = np.random.uniform(-0.05, 0.05)
                uncertainty_index = np.random.uniform(0.7, 1.0)
            
            elif scenario == MarketScenario.CRASH:
                market_volatility = np.random.uniform(0.40, 0.80)
                market_trend = np.random.uniform(-0.30, -0.15)
                uncertainty_index = np.random.uniform(0.9, 1.0)
            
            else:  # SIDEWAYS
                market_volatility = np.random.uniform(0.10, 0.20)
                market_trend = np.random.uniform(-0.02, 0.02)
                uncertainty_index = np.random.uniform(0.3, 0.6)
            
            metrics = MarketMetrics(
                market_volatility=market_volatility,
                market_trend=market_trend,
                uncertainty_index=uncertainty_index,
                correlation_breakdown=np.random.choice([True, False], p=[0.2, 0.8])
            )
            metrics_list.append(metrics)
        
        return metrics_list
    
    def _generate_returns(self, scenario: MarketScenario, length: int) -> np.ndarray:
        """生成收益率序列"""
        if scenario == MarketScenario.BULL_MARKET:
            returns = np.random.normal(
                loc=self.config.base_return * 2,
                scale=self.config.base_volatility,
                size=length
            )
        
        elif scenario == MarketScenario.BEAR_MARKET:
            returns = np.random.normal(
                loc=-self.config.base_return * 1.5,
                scale=self.config.base_volatility * 1.3,
                size=length
            )
        
        elif scenario == MarketScenario.HIGH_VOLATILITY:
            returns = np.random.normal(
                loc=self.config.base_return,
                scale=self.config.base_volatility * 2.5,
                size=length
            )
        
        elif scenario == MarketScenario.CRASH:
            # 前期正常，然后崩盘
            normal_period = int(length * 0.7)
            crash_period = length - normal_period
            
            normal_returns = np.random.normal(
                loc=self.config.base_return,
                scale=self.config.base_volatility,
                size=normal_period
            )
            
            crash_returns = np.random.normal(
                loc=-0.05,  # 日均-5%收益
                scale=0.08,  # 高波动
                size=crash_period
            )
            
            returns = np.concatenate([normal_returns, crash_returns])
        
        elif scenario == MarketScenario.RECOVERY:
            # V型反转
            decline_period = int(length * 0.3)
            recovery_period = length - decline_period
            
            decline_returns = np.random.normal(
                loc=-0.02,
                scale=0.03,
                size=decline_period
            )
            
            recovery_returns = np.random.normal(
                loc=0.03,
                scale=0.02,
                size=recovery_period
            )
            
            returns = np.concatenate([decline_returns, recovery_returns])
        
        else:  # SIDEWAYS
            returns = np.random.normal(
                loc=0,
                scale=self.config.base_volatility * 0.8,
                size=length
            )
        
        return returns
    
    def _generate_volatility(self, scenario: MarketScenario, length: int) -> np.ndarray:
        """生成波动率序列"""
        base_vol = self.config.base_volatility
        
        if scenario == MarketScenario.HIGH_VOLATILITY:
            volatility = np.random.gamma(shape=2, scale=base_vol * 1.5, size=length)
        elif scenario == MarketScenario.CRASH:
            volatility = np.concatenate([
                np.random.gamma(shape=1, scale=base_vol, size=int(length * 0.7)),
                np.random.gamma(shape=3, scale=base_vol * 2, size=length - int(length * 0.7))
            ])
        else:
            volatility = np.random.gamma(shape=1.5, scale=base_vol, size=length)
        
        return volatility
    
    def _generate_specific_drawdown(self, max_drawdown: float, recovery_speed: float) -> np.ndarray:
        """生成特定回撤模式的数据"""
        length = 252
        values = np.zeros(length)
        values[0] = self.config.initial_value
        
        # 第一阶段：正常上涨
        growth_phase = int(length * 0.3)
        for i in range(1, growth_phase):
            daily_return = np.random.normal(0.001, 0.01)
            values[i] = values[i-1] * (1 + daily_return)
        
        peak_value = values[growth_phase - 1]
        
        # 第二阶段：回撤
        drawdown_phase = int(length * 0.3)
        target_value = peak_value * (1 - max_drawdown)
        
        for i in range(growth_phase, growth_phase + drawdown_phase):
            # 渐进式回撤
            progress = (i - growth_phase) / drawdown_phase
            current_target = peak_value * (1 - max_drawdown * progress)
            noise = np.random.normal(0, 0.005)
            values[i] = current_target * (1 + noise)
        
        # 第三阶段：恢复
        recovery_phase = length - growth_phase - drawdown_phase
        for i in range(growth_phase + drawdown_phase, length):
            progress = (i - growth_phase - drawdown_phase) / recovery_phase
            recovery_factor = min(1.0, progress * recovery_speed)
            
            recovery_value = target_value + (peak_value - target_value) * recovery_factor
            noise = np.random.normal(0, 0.008)
            values[i] = recovery_value * (1 + noise)
        
        return values
    
    def _generate_prolonged_drawdown(self, drawdown_duration: int, max_drawdown: float) -> np.ndarray:
        """生成长期回撤场景"""
        length = 300  # 更长的序列
        values = np.zeros(length)
        values[0] = self.config.initial_value
        
        # 初期上涨
        for i in range(1, 50):
            values[i] = values[i-1] * (1 + np.random.normal(0.001, 0.01))
        
        peak_value = values[49]
        bottom_value = peak_value * (1 - max_drawdown)
        
        # 长期回撤期
        for i in range(50, 50 + drawdown_duration):
            # 在谷底附近震荡
            oscillation = np.random.normal(0, 0.02)
            values[i] = bottom_value * (1 + oscillation)
        
        # 最终恢复
        for i in range(50 + drawdown_duration, length):
            recovery_rate = np.random.normal(0.002, 0.015)
            values[i] = values[i-1] * (1 + recovery_rate)
        
        return values
    
    def _generate_multiple_drawdowns(self, num_drawdowns: int, avg_drawdown: float) -> np.ndarray:
        """生成多次回撤场景"""
        length = 400
        values = np.zeros(length)
        values[0] = self.config.initial_value
        
        segment_length = length // (num_drawdowns + 1)
        
        for segment in range(num_drawdowns + 1):
            start_idx = segment * segment_length
            end_idx = min((segment + 1) * segment_length, length)
            
            if segment < num_drawdowns:
                # 回撤段
                drawdown = np.random.normal(avg_drawdown, avg_drawdown * 0.3)
                drawdown = max(0.02, min(0.5, drawdown))  # 限制在合理范围
                segment_values = self._generate_specific_drawdown(drawdown, 0.6)
                
                # 缩放到当前段
                segment_scaled = segment_values[:end_idx - start_idx]
                if start_idx > 0:
                    scale_factor = values[start_idx - 1] / segment_scaled[0]
                    segment_scaled *= scale_factor
                
                values[start_idx:end_idx] = segment_scaled
            else:
                # 最后的恢复段
                for i in range(start_idx, end_idx):
                    if i > 0:
                        values[i] = values[i-1] * (1 + np.random.normal(0.002, 0.015))
        
        return values[:length]
    
    def _generate_liquidity_crisis_scenario(self) -> Dict[str, Any]:
        """生成流动性危机场景"""
        returns = np.random.normal(0.001, 0.015, 252)
        
        # 在特定时期增加极端事件
        crisis_periods = [80, 120, 200]  # 三个危机时期
        
        for period in crisis_periods:
            # 流动性枯竭，价格剧烈波动但成交量萎缩
            for i in range(period, min(period + 10, 252)):
                returns[i] = np.random.normal(0, 0.08)  # 高波动，无明确方向
        
        # 生成对应的成交量数据（危机期间大幅下降）
        base_volume = np.random.lognormal(np.log(1000000), 0.3, 252)
        
        for period in crisis_periods:
            for i in range(period, min(period + 10, 252)):
                base_volume[i] *= 0.3  # 成交量降至30%
        
        return {
            'returns': returns,
            'volume': base_volume,
            'description': '流动性危机场景'
        }
    
    def _generate_crash_recovery_scenario(self) -> Dict[str, Any]:
        """生成崩盘恢复场景"""
        returns = np.random.normal(0.001, 0.015, 300)
        
        # 崩盘期（第100-110天）
        crash_start = 100
        crash_duration = 10
        
        for i in range(crash_start, crash_start + crash_duration):
            returns[i] = np.random.normal(-0.08, 0.05)  # 平均每日-8%
        
        # 恢复期（第111-200天）
        recovery_start = crash_start + crash_duration
        recovery_duration = 90
        
        for i in range(recovery_start, recovery_start + recovery_duration):
            # 逐渐减弱的正收益
            recovery_strength = 1 - (i - recovery_start) / recovery_duration
            daily_return = np.random.normal(0.02 * recovery_strength, 0.025)
            returns[i] = daily_return
        
        return {
            'returns': returns,
            'description': '市场崩盘后V型恢复场景'
        }


class ScenarioTester:
    """场景测试器"""
    
    def __init__(self, generator: TestDataGenerator):
        """
        初始化场景测试器
        
        Args:
            generator: 数据生成器
        """
        self.generator = generator
    
    def run_scenario_tests(self, 
                          test_function: callable,
                          scenarios: List[MarketScenario] = None) -> Dict[str, Any]:
        """
        运行多场景测试
        
        Args:
            test_function: 要测试的函数
            scenarios: 测试场景列表
            
        Returns:
            各场景的测试结果
        """
        if scenarios is None:
            scenarios = list(MarketScenario)
        
        results = {}
        
        for scenario in scenarios:
            try:
                # 生成测试数据
                test_data = self.generator.generate_portfolio_values(scenario)
                
                # 执行测试
                result = test_function(test_data)
                
                results[scenario.value] = {
                    'success': True,
                    'result': result,
                    'data_length': len(test_data),
                    'scenario': scenario.value
                }
            
            except Exception as e:
                results[scenario.value] = {
                    'success': False,
                    'error': str(e),
                    'scenario': scenario.value
                }
        
        return results
    
    def validate_generated_data(self, data: np.ndarray) -> Dict[str, Any]:
        """
        验证生成数据的质量
        
        Args:
            data: 生成的数据
            
        Returns:
            数据质量报告
        """
        validation_report = {
            'length': len(data),
            'min_value': np.min(data),
            'max_value': np.max(data),
            'mean_value': np.mean(data),
            'std_value': np.std(data),
            'has_nan': np.any(np.isnan(data)),
            'has_inf': np.any(np.isinf(data)),
            'monotonic_increasing': np.all(np.diff(data) >= 0),
            'monotonic_decreasing': np.all(np.diff(data) <= 0),
            'data_quality': 'good'
        }
        
        # 评估数据质量
        if validation_report['has_nan'] or validation_report['has_inf']:
            validation_report['data_quality'] = 'poor'
        elif validation_report['std_value'] == 0:
            validation_report['data_quality'] = 'constant'
        elif validation_report['std_value'] > validation_report['mean_value']:
            validation_report['data_quality'] = 'high_volatility'
        
        return validation_report


# 便利函数
def create_test_portfolio_data(scenario: str = "normal", length: int = 252) -> np.ndarray:
    """
    快速创建测试投资组合数据
    
    Args:
        scenario: 场景类型 ("normal", "crash", "bull", "bear", "volatile")
        length: 数据长度
        
    Returns:
        投资组合净值数据
    """
    generator = TestDataGenerator()
    
    scenario_map = {
        "normal": MarketScenario.SIDEWAYS,
        "crash": MarketScenario.CRASH,
        "bull": MarketScenario.BULL_MARKET,
        "bear": MarketScenario.BEAR_MARKET,
        "volatile": MarketScenario.HIGH_VOLATILITY,
        "recovery": MarketScenario.RECOVERY
    }
    
    market_scenario = scenario_map.get(scenario, MarketScenario.SIDEWAYS)
    return generator.generate_portfolio_values(market_scenario, length)


def create_drawdown_test_data() -> Dict[str, np.ndarray]:
    """
    创建回撤测试数据集
    
    Returns:
        包含各种回撤场景的数据字典
    """
    generator = TestDataGenerator()
    return generator.generate_drawdown_scenarios()


if __name__ == "__main__":
    # 示例用法
    generator = TestDataGenerator()
    
    # 生成牛市数据
    bull_data = generator.generate_portfolio_values(MarketScenario.BULL_MARKET)
    print(f"牛市数据: 长度={len(bull_data)}, 最终值={bull_data[-1]:.2f}")
    
    # 生成回撤场景
    drawdown_scenarios = generator.generate_drawdown_scenarios()
    for name, data in drawdown_scenarios.items():
        max_dd = (np.min(data) - np.max(data[:len(data)//2])) / np.max(data[:len(data)//2])
        print(f"{name}: 最大回撤={max_dd:.2%}")
    
    # 生成压力测试数据
    stress_data = generator.generate_stress_test_data()
    print(f"生成了{len(stress_data)}种压力测试场景")