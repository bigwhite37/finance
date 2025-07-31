"""
核心数据模型
定义系统中使用的核心数据结构，包括市场数据、特征向量、交易状态等
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import json
import math


@dataclass
class MarketData:
    """市场数据结构"""
    timestamp: datetime
    symbol: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    amount: float
    
    def __post_init__(self):
        """数据验证"""
        self._validate()
    
    def _validate(self):
        """验证数据有效性"""
        # 验证价格顺序
        if self.high_price < self.low_price:
            raise ValueError("最高价不能低于最低价")
        
        # 验证成交量和成交额非负
        if self.volume < 0:
            raise ValueError("成交量不能为负数")
        
        if self.amount < 0:
            raise ValueError("成交额不能为负数")
        
        # 验证价格非负
        if any(price < 0 for price in [self.open_price, self.high_price, 
                                      self.low_price, self.close_price]):
            raise ValueError("价格不能为负数")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'amount': self.amount
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """从字典创建对象"""
        data_copy = data.copy()
        data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        return cls(**data_copy)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MarketData':
        """从JSON字符串创建对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class FeatureVector:
    """特征向量结构"""
    timestamp: datetime
    symbol: str
    technical_indicators: Dict[str, float]
    fundamental_factors: Dict[str, float]
    market_microstructure: Dict[str, float]
    
    def __post_init__(self):
        """数据验证"""
        self._validate()
    
    def _validate(self):
        """验证数据有效性"""
        # 验证特征字典非空
        if not self.technical_indicators:
            raise ValueError("技术指标不能为空")
        
        if not self.fundamental_factors:
            raise ValueError("基本面因子不能为空")
        
        if not self.market_microstructure:
            raise ValueError("市场微观结构特征不能为空")
        
        # 验证特征值不包含NaN
        all_features = {**self.technical_indicators, 
                       **self.fundamental_factors, 
                       **self.market_microstructure}
        
        for name, value in all_features.items():
            if math.isnan(value):
                raise ValueError(f"特征值不能包含NaN: {name}")
            if math.isinf(value):
                raise ValueError(f"特征值不能包含无穷大: {name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'technical_indicators': self.technical_indicators,
            'fundamental_factors': self.fundamental_factors,
            'market_microstructure': self.market_microstructure
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureVector':
        """从字典创建对象"""
        data_copy = data.copy()
        data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        return cls(**data_copy)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FeatureVector':
        """从JSON字符串创建对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_all_features(self) -> Dict[str, float]:
        """获取所有特征的合并字典"""
        return {**self.technical_indicators, 
                **self.fundamental_factors, 
                **self.market_microstructure}


@dataclass
class TradingState:
    """交易状态结构"""
    features: np.ndarray  # [lookback_window, n_stocks, n_features]
    positions: np.ndarray  # [n_stocks]
    market_state: np.ndarray  # [market_features]
    cash: float
    total_value: float
    
    def __post_init__(self):
        """数据验证"""
        self._validate()
    
    def _validate(self):
        """验证数据有效性"""
        # 验证特征数组维度
        if self.features.ndim != 3:
            raise ValueError("特征数组必须是3维")
        
        # 验证持仓权重和
        if abs(self.positions.sum() - 1.0) > 1e-6:
            raise ValueError("持仓权重和必须接近1")
        
        # 验证持仓权重非负
        if (self.positions < 0).any():
            raise ValueError("持仓权重不能为负数")
        
        # 验证现金非负
        if self.cash < 0:
            raise ValueError("现金不能为负数")
        
        # 验证总价值为正
        if self.total_value <= 0:
            raise ValueError("总价值必须为正数")
        
        # 验证数组不包含NaN或无穷大
        if np.isnan(self.features).any():
            raise ValueError("特征数组不能包含NaN")
        
        if np.isinf(self.features).any():
            raise ValueError("特征数组不能包含无穷大")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'features': self.features.tolist(),
            'positions': self.positions.tolist(),
            'market_state': self.market_state.tolist(),
            'cash': self.cash,
            'total_value': self.total_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingState':
        """从字典创建对象"""
        return cls(
            features=np.array(data['features']),
            positions=np.array(data['positions']),
            market_state=np.array(data['market_state']),
            cash=data['cash'],
            total_value=data['total_value']
        )
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TradingState':
        """从JSON字符串创建对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_portfolio_value(self) -> float:
        """获取投资组合价值（不包括现金）"""
        return self.total_value - self.cash
    
    def get_leverage(self) -> float:
        """获取杠杆率"""
        portfolio_value = self.get_portfolio_value()
        if portfolio_value == 0:
            return 0.0
        return self.total_value / portfolio_value


@dataclass
class TradingAction:
    """交易动作结构"""
    target_weights: np.ndarray  # [n_stocks]
    confidence: float
    timestamp: datetime
    
    def __post_init__(self):
        """数据验证"""
        self._validate()
    
    def _validate(self):
        """验证数据有效性"""
        # 验证权重和
        if abs(self.target_weights.sum() - 1.0) > 1e-6:
            raise ValueError("目标权重和必须接近1")
        
        # 验证权重非负
        if (self.target_weights < 0).any():
            raise ValueError("目标权重不能为负数")
        
        # 验证置信度范围
        if not (0 <= self.confidence <= 1):
            raise ValueError("置信度必须在0到1之间")
        
        # 验证数组不包含NaN或无穷大
        if np.isnan(self.target_weights).any():
            raise ValueError("目标权重不能包含NaN")
        
        if np.isinf(self.target_weights).any():
            raise ValueError("目标权重不能包含无穷大")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'target_weights': self.target_weights.tolist(),
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingAction':
        """从字典创建对象"""
        return cls(
            target_weights=np.array(data['target_weights']),
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TradingAction':
        """从JSON字符串创建对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_concentration(self) -> float:
        """获取权重集中度（Herfindahl指数）"""
        return np.sum(self.target_weights ** 2)
    
    def get_active_positions(self, threshold: float = 1e-6) -> int:
        """获取活跃持仓数量"""
        return np.sum(self.target_weights > threshold)


@dataclass
class TransactionRecord:
    """交易记录结构"""
    timestamp: datetime
    symbol: str
    action_type: str  # 'buy' or 'sell'
    quantity: int
    price: float
    commission: float
    stamp_tax: float
    slippage: float
    total_cost: float
    
    def __post_init__(self):
        """数据验证"""
        self._validate()
    
    def _validate(self):
        """验证数据有效性"""
        # 验证交易类型
        if self.action_type not in ['buy', 'sell']:
            raise ValueError("交易类型必须是'buy'或'sell'")
        
        # 验证数量为正
        if self.quantity < 0:
            raise ValueError("交易数量不能为负数")
        
        # 验证价格为正
        if self.price < 0:
            raise ValueError("价格不能为负数")
        
        # 验证成本项非负
        if any(cost < 0 for cost in [self.commission, self.stamp_tax, 
                                    self.slippage, self.total_cost]):
            raise ValueError("成本项不能为负数")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'action_type': self.action_type,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'stamp_tax': self.stamp_tax,
            'slippage': self.slippage,
            'total_cost': self.total_cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionRecord':
        """从字典创建对象"""
        data_copy = data.copy()
        data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        return cls(**data_copy)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TransactionRecord':
        """从JSON字符串创建对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_transaction_value(self) -> float:
        """获取交易价值"""
        return self.quantity * self.price
    
    def get_cost_ratio(self) -> float:
        """获取成本比率"""
        transaction_value = self.get_transaction_value()
        if transaction_value == 0:
            return 0.0
        return self.total_cost / transaction_value


# 工具函数
def validate_array_shape(array: np.ndarray, expected_shape: tuple, name: str):
    """验证数组形状"""
    if array.shape != expected_shape:
        raise ValueError(f"{name}的形状应为{expected_shape}，实际为{array.shape}")


def validate_weights(weights: np.ndarray, name: str = "权重"):
    """验证权重数组"""
    if (weights < 0).any():
        raise ValueError(f"{name}不能为负数")
    
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError(f"{name}和必须接近1")
    
    if np.isnan(weights).any():
        raise ValueError(f"{name}不能包含NaN")
    
    if np.isinf(weights).any():
        raise ValueError(f"{name}不能包含无穷大")


def validate_price_data(prices: Dict[str, float]):
    """验证价格数据"""
    required_fields = ['open', 'high', 'low', 'close']
    
    for field in required_fields:
        if field not in prices:
            raise ValueError(f"缺少必要的价格字段: {field}")
        
        if prices[field] < 0:
            raise ValueError(f"价格字段{field}不能为负数")
    
    if prices['high'] < prices['low']:
        raise ValueError("最高价不能低于最低价")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """标准化权重，确保和为1"""
    total = weights.sum()
    if total == 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def calculate_portfolio_metrics(weights: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
    """计算投资组合指标"""
    portfolio_return = np.dot(weights, returns)
    concentration = np.sum(weights ** 2)  # Herfindahl指数
    active_positions = np.sum(weights > 1e-6)
    
    return {
        'portfolio_return': portfolio_return,
        'concentration': concentration,
        'active_positions': active_positions,
        'max_weight': weights.max(),
        'min_weight': weights.min()
    }