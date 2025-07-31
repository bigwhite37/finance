"""
测试核心数据模型
测试MarketData、FeatureVector、TradingState等数据类的功能
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any
import json
import pickle

from src.rl_trading_system.data.data_models import (
    MarketData,
    FeatureVector,
    TradingState,
    TradingAction,
    TransactionRecord
)


class TestMarketData:
    """测试MarketData数据类"""
    
    def test_market_data_creation(self):
        """测试MarketData正常创建"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        market_data = MarketData(
            timestamp=timestamp,
            symbol="000001.SZ",
            open_price=10.0,
            high_price=10.5,
            low_price=9.8,
            close_price=10.2,
            volume=1000000,
            amount=10200000.0
        )
        
        assert market_data.timestamp == timestamp
        assert market_data.symbol == "000001.SZ"
        assert market_data.open_price == 10.0
        assert market_data.high_price == 10.5
        assert market_data.low_price == 9.8
        assert market_data.close_price == 10.2
        assert market_data.volume == 1000000
        assert market_data.amount == 10200000.0
    
    def test_market_data_validation_price_order(self):
        """测试价格顺序验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        # 测试high < low的情况
        with pytest.raises(ValueError, match="最高价不能低于最低价"):
            MarketData(
                timestamp=timestamp,
                symbol="000001.SZ",
                open_price=10.0,
                high_price=9.5,  # high < low
                low_price=9.8,
                close_price=10.2,
                volume=1000000,
                amount=10200000.0
            )
    
    def test_market_data_validation_negative_volume(self):
        """测试负成交量验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="成交量不能为负数"):
            MarketData(
                timestamp=timestamp,
                symbol="000001.SZ",
                open_price=10.0,
                high_price=10.5,
                low_price=9.8,
                close_price=10.2,
                volume=-1000,  # 负成交量
                amount=10200000.0
            )
    
    def test_market_data_validation_negative_amount(self):
        """测试负成交额验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="成交额不能为负数"):
            MarketData(
                timestamp=timestamp,
                symbol="000001.SZ",
                open_price=10.0,
                high_price=10.5,
                low_price=9.8,
                close_price=10.2,
                volume=1000000,
                amount=-10200000.0  # 负成交额
            )
    
    def test_market_data_serialization(self):
        """测试MarketData序列化"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        market_data = MarketData(
            timestamp=timestamp,
            symbol="000001.SZ",
            open_price=10.0,
            high_price=10.5,
            low_price=9.8,
            close_price=10.2,
            volume=1000000,
            amount=10200000.0
        )
        
        # 测试to_dict
        data_dict = market_data.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == "000001.SZ"
        assert data_dict['open_price'] == 10.0
        
        # 测试from_dict
        restored_data = MarketData.from_dict(data_dict)
        assert restored_data.symbol == market_data.symbol
        assert restored_data.open_price == market_data.open_price
        assert restored_data.timestamp == market_data.timestamp
    
    def test_market_data_json_serialization(self):
        """测试JSON序列化"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        market_data = MarketData(
            timestamp=timestamp,
            symbol="000001.SZ",
            open_price=10.0,
            high_price=10.5,
            low_price=9.8,
            close_price=10.2,
            volume=1000000,
            amount=10200000.0
        )
        
        # 测试JSON序列化
        json_str = market_data.to_json()
        assert isinstance(json_str, str)
        
        # 测试JSON反序列化
        restored_data = MarketData.from_json(json_str)
        assert restored_data.symbol == market_data.symbol
        assert restored_data.open_price == market_data.open_price


class TestFeatureVector:
    """测试FeatureVector数据类"""
    
    def test_feature_vector_creation(self):
        """测试FeatureVector正常创建"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        feature_vector = FeatureVector(
            timestamp=timestamp,
            symbol="000001.SZ",
            technical_indicators={"rsi": 65.5, "macd": 0.12},
            fundamental_factors={"pe_ratio": 15.2, "pb_ratio": 1.8},
            market_microstructure={"bid_ask_spread": 0.01, "order_imbalance": 0.05}
        )
        
        assert feature_vector.timestamp == timestamp
        assert feature_vector.symbol == "000001.SZ"
        assert feature_vector.technical_indicators["rsi"] == 65.5
        assert feature_vector.fundamental_factors["pe_ratio"] == 15.2
        assert feature_vector.market_microstructure["bid_ask_spread"] == 0.01
    
    def test_feature_vector_validation_empty_features(self):
        """测试空特征验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="技术指标不能为空"):
            FeatureVector(
                timestamp=timestamp,
                symbol="000001.SZ",
                technical_indicators={},  # 空字典
                fundamental_factors={"pe_ratio": 15.2},
                market_microstructure={"bid_ask_spread": 0.01}
            )
    
    def test_feature_vector_validation_nan_values(self):
        """测试NaN值验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="特征值不能包含NaN"):
            FeatureVector(
                timestamp=timestamp,
                symbol="000001.SZ",
                technical_indicators={"rsi": float('nan')},  # NaN值
                fundamental_factors={"pe_ratio": 15.2},
                market_microstructure={"bid_ask_spread": 0.01}
            )
    
    def test_feature_vector_serialization(self):
        """测试FeatureVector序列化"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        feature_vector = FeatureVector(
            timestamp=timestamp,
            symbol="000001.SZ",
            technical_indicators={"rsi": 65.5, "macd": 0.12},
            fundamental_factors={"pe_ratio": 15.2, "pb_ratio": 1.8},
            market_microstructure={"bid_ask_spread": 0.01, "order_imbalance": 0.05}
        )
        
        # 测试to_dict
        data_dict = feature_vector.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['technical_indicators']['rsi'] == 65.5
        
        # 测试from_dict
        restored_vector = FeatureVector.from_dict(data_dict)
        assert restored_vector.symbol == feature_vector.symbol
        assert restored_vector.technical_indicators['rsi'] == feature_vector.technical_indicators['rsi']


class TestTradingState:
    """测试TradingState数据类"""
    
    def test_trading_state_creation(self):
        """测试TradingState正常创建"""
        features = np.random.randn(60, 100, 50)  # lookback_window, n_stocks, n_features
        positions = np.random.rand(100)
        positions = positions / positions.sum()  # 标准化权重
        market_state = np.random.randn(10)
        
        trading_state = TradingState(
            features=features,
            positions=positions,
            market_state=market_state,
            cash=100000.0,
            total_value=1000000.0
        )
        
        assert trading_state.features.shape == (60, 100, 50)
        assert trading_state.positions.shape == (100,)
        assert trading_state.market_state.shape == (10,)
        assert trading_state.cash == 100000.0
        assert trading_state.total_value == 1000000.0
    
    def test_trading_state_validation_features_shape(self):
        """测试特征维度验证"""
        features = np.random.randn(60, 50)  # 错误的维度
        positions = np.random.rand(100)
        market_state = np.random.randn(10)
        
        with pytest.raises(ValueError, match="特征数组必须是3维"):
            TradingState(
                features=features,
                positions=positions,
                market_state=market_state,
                cash=100000.0,
                total_value=1000000.0
            )
    
    def test_trading_state_validation_positions_sum(self):
        """测试持仓权重和验证"""
        features = np.random.randn(60, 100, 50)
        positions = np.array([0.5, 0.6])  # 权重和不为1
        market_state = np.random.randn(10)
        
        with pytest.raises(ValueError, match="持仓权重和必须接近1"):
            TradingState(
                features=features,
                positions=positions,
                market_state=market_state,
                cash=100000.0,
                total_value=1000000.0
            )
    
    def test_trading_state_validation_negative_cash(self):
        """测试负现金验证"""
        features = np.random.randn(60, 100, 50)
        positions = np.random.rand(100)
        positions = positions / positions.sum()
        market_state = np.random.randn(10)
        
        with pytest.raises(ValueError, match="现金不能为负数"):
            TradingState(
                features=features,
                positions=positions,
                market_state=market_state,
                cash=-1000.0,  # 负现金
                total_value=1000000.0
            )
    
    def test_trading_state_serialization(self):
        """测试TradingState序列化"""
        features = np.random.randn(60, 100, 50)
        positions = np.random.rand(100)
        positions = positions / positions.sum()
        market_state = np.random.randn(10)
        
        trading_state = TradingState(
            features=features,
            positions=positions,
            market_state=market_state,
            cash=100000.0,
            total_value=1000000.0
        )
        
        # 测试to_dict
        data_dict = trading_state.to_dict()
        assert isinstance(data_dict, dict)
        assert 'features' in data_dict
        assert 'positions' in data_dict
        
        # 测试from_dict
        restored_state = TradingState.from_dict(data_dict)
        assert np.allclose(restored_state.features, trading_state.features)
        assert np.allclose(restored_state.positions, trading_state.positions)
        assert restored_state.cash == trading_state.cash


class TestTradingAction:
    """测试TradingAction数据类"""
    
    def test_trading_action_creation(self):
        """测试TradingAction正常创建"""
        target_weights = np.random.rand(100)
        target_weights = target_weights / target_weights.sum()
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        trading_action = TradingAction(
            target_weights=target_weights,
            confidence=0.85,
            timestamp=timestamp
        )
        
        assert trading_action.target_weights.shape == (100,)
        assert abs(trading_action.target_weights.sum() - 1.0) < 1e-6
        assert trading_action.confidence == 0.85
        assert trading_action.timestamp == timestamp
    
    def test_trading_action_validation_weights_sum(self):
        """测试权重和验证"""
        target_weights = np.array([0.5, 0.6])  # 权重和不为1
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="目标权重和必须接近1"):
            TradingAction(
                target_weights=target_weights,
                confidence=0.85,
                timestamp=timestamp
            )
    
    def test_trading_action_validation_confidence_range(self):
        """测试置信度范围验证"""
        target_weights = np.array([0.5, 0.5])
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="置信度必须在0到1之间"):
            TradingAction(
                target_weights=target_weights,
                confidence=1.5,  # 超出范围
                timestamp=timestamp
            )
    
    def test_trading_action_serialization(self):
        """测试TradingAction序列化"""
        target_weights = np.array([0.6, 0.4])
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        trading_action = TradingAction(
            target_weights=target_weights,
            confidence=0.85,
            timestamp=timestamp
        )
        
        # 测试to_dict
        data_dict = trading_action.to_dict()
        assert isinstance(data_dict, dict)
        assert 'target_weights' in data_dict
        assert 'confidence' in data_dict
        
        # 测试from_dict
        restored_action = TradingAction.from_dict(data_dict)
        assert np.allclose(restored_action.target_weights, trading_action.target_weights)
        assert restored_action.confidence == trading_action.confidence


class TestTransactionRecord:
    """测试TransactionRecord数据类"""
    
    def test_transaction_record_creation(self):
        """测试TransactionRecord正常创建"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        record = TransactionRecord(
            timestamp=timestamp,
            symbol="000001.SZ",
            action_type="buy",
            quantity=1000,
            price=10.5,
            commission=10.5,
            stamp_tax=0.0,
            slippage=2.1,
            total_cost=12.6
        )
        
        assert record.timestamp == timestamp
        assert record.symbol == "000001.SZ"
        assert record.action_type == "buy"
        assert record.quantity == 1000
        assert record.price == 10.5
        assert record.commission == 10.5
        assert record.stamp_tax == 0.0
        assert record.slippage == 2.1
        assert record.total_cost == 12.6
    
    def test_transaction_record_validation_action_type(self):
        """测试交易类型验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="交易类型必须是'buy'或'sell'"):
            TransactionRecord(
                timestamp=timestamp,
                symbol="000001.SZ",
                action_type="invalid",  # 无效类型
                quantity=1000,
                price=10.5,
                commission=10.5,
                stamp_tax=0.0,
                slippage=2.1,
                total_cost=12.6
            )
    
    def test_transaction_record_validation_negative_quantity(self):
        """测试负数量验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="交易数量不能为负数"):
            TransactionRecord(
                timestamp=timestamp,
                symbol="000001.SZ",
                action_type="buy",
                quantity=-1000,  # 负数量
                price=10.5,
                commission=10.5,
                stamp_tax=0.0,
                slippage=2.1,
                total_cost=12.6
            )
    
    def test_transaction_record_validation_negative_price(self):
        """测试负价格验证"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        with pytest.raises(ValueError, match="价格不能为负数"):
            TransactionRecord(
                timestamp=timestamp,
                symbol="000001.SZ",
                action_type="buy",
                quantity=1000,
                price=-10.5,  # 负价格
                commission=10.5,
                stamp_tax=0.0,
                slippage=2.1,
                total_cost=12.6
            )
    
    def test_transaction_record_serialization(self):
        """测试TransactionRecord序列化"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        record = TransactionRecord(
            timestamp=timestamp,
            symbol="000001.SZ",
            action_type="buy",
            quantity=1000,
            price=10.5,
            commission=10.5,
            stamp_tax=0.0,
            slippage=2.1,
            total_cost=12.6
        )
        
        # 测试to_dict
        data_dict = record.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == "000001.SZ"
        assert data_dict['action_type'] == "buy"
        
        # 测试from_dict
        restored_record = TransactionRecord.from_dict(data_dict)
        assert restored_record.symbol == record.symbol
        assert restored_record.action_type == record.action_type
        assert restored_record.quantity == record.quantity


class TestDataModelsBoundaryConditions:
    """测试数据模型边界条件"""
    
    def test_zero_values(self):
        """测试零值处理"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        # MarketData允许零价格（停牌情况）
        market_data = MarketData(
            timestamp=timestamp,
            symbol="000001.SZ",
            open_price=0.0,
            high_price=0.0,
            low_price=0.0,
            close_price=0.0,
            volume=0,
            amount=0.0
        )
        assert market_data.open_price == 0.0
    
    def test_extreme_values(self):
        """测试极值处理"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        # 测试极大值
        market_data = MarketData(
            timestamp=timestamp,
            symbol="000001.SZ",
            open_price=1e6,
            high_price=1e6,
            low_price=1e6,
            close_price=1e6,
            volume=int(1e9),
            amount=1e15
        )
        assert market_data.open_price == 1e6
    
    def test_unicode_symbols(self):
        """测试Unicode符号处理"""
        timestamp = datetime(2023, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
        
        market_data = MarketData(
            timestamp=timestamp,
            symbol="平安银行.SZ",  # 中文符号
            open_price=10.0,
            high_price=10.5,
            low_price=9.8,
            close_price=10.2,
            volume=1000000,
            amount=10200000.0
        )
        assert market_data.symbol == "平安银行.SZ"


class TestDataModelsPerformance:
    """测试数据模型性能"""
    
    def test_large_array_serialization(self):
        """测试大数组序列化性能"""
        # 创建大型特征数组
        features = np.random.randn(252, 1000, 100)  # 一年数据，1000只股票，100个特征
        positions = np.random.rand(1000)
        positions = positions / positions.sum()
        market_state = np.random.randn(50)
        
        trading_state = TradingState(
            features=features,
            positions=positions,
            market_state=market_state,
            cash=1000000.0,
            total_value=10000000.0
        )
        
        # 测试序列化时间（应该在合理范围内）
        import time
        start_time = time.time()
        data_dict = trading_state.to_dict()
        serialization_time = time.time() - start_time
        
        # 序列化时间应该小于1秒
        assert serialization_time < 1.0
        
        # 测试反序列化
        start_time = time.time()
        restored_state = TradingState.from_dict(data_dict)
        deserialization_time = time.time() - start_time
        
        # 反序列化时间应该小于1秒
        assert deserialization_time < 1.0
        
        # 验证数据完整性
        assert np.allclose(restored_state.features, trading_state.features)