"""
目标波动率控制器
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TargetVolatilityController:
    """目标波动率管理"""

    def __init__(self, config: Dict):
        """
        初始化目标波动率控制器

        Args:
            config: 配置参数
        """
        self.config = config
        self.target_vol = config.get('target_volatility', 0.12)
        self.lookback_window = config.get('vol_lookback', 20)
        self.max_leverage_multiplier = config.get('max_leverage_multiplier', 2.0)
        self.min_leverage_multiplier = config.get('min_leverage_multiplier', 0.5)

        # 调整历史记录
        self.adjustment_history = []

    def adjust_leverage(self,
                       portfolio_weights: np.ndarray,
                       price_data: pd.DataFrame,
                       target_vol: Optional[float] = None) -> np.ndarray:
        """
        动态调整杠杆以维持目标波动率

        Args:
            portfolio_weights: 组合权重
            price_data: 价格数据
            target_vol: 目标波动率（可选）

        Returns:
            调整后的组合权重
        """
        if target_vol is None:
            target_vol = self.target_vol

        if len(price_data) < self.lookback_window:
            return portfolio_weights

        # 计算历史收益率
        returns = price_data.pct_change().dropna()

        if len(returns) < self.lookback_window:
            return portfolio_weights

        # 计算组合历史收益率
        recent_returns = returns.iloc[-self.lookback_window:]
        portfolio_returns = (recent_returns * portfolio_weights).sum(axis=1)

        # 计算已实现波动率
        realized_vol = portfolio_returns.std() * np.sqrt(252)

        if realized_vol <= 0:
            return portfolio_weights

        # 计算杠杆调整倍数
        leverage_multiplier = target_vol / realized_vol

        # 限制调整范围
        leverage_multiplier = np.clip(
            leverage_multiplier,
            self.min_leverage_multiplier,
            self.max_leverage_multiplier
        )

        # 调整权重
        adjusted_weights = portfolio_weights * leverage_multiplier

        # 记录调整
        self._record_adjustment(realized_vol, target_vol, leverage_multiplier)

        return adjusted_weights

    def calculate_target_leverage(self,
                                 returns: pd.Series,
                                 target_vol: Optional[float] = None) -> float:
        """
        计算目标杠杆倍数

        Args:
            returns: 收益率序列
            target_vol: 目标波动率

        Returns:
            目标杠杆倍数
        """
        if target_vol is None:
            target_vol = self.target_vol

        realized_vol = returns.std() * np.sqrt(252)

        if realized_vol <= 0:
            return 1.0

        leverage = target_vol / realized_vol
        return np.clip(leverage, self.min_leverage_multiplier, self.max_leverage_multiplier)

    def estimate_portfolio_volatility(self,
                                    weights: np.ndarray,
                                    price_data: pd.DataFrame) -> float:
        """
        估计组合波动率

        Args:
            weights: 组合权重
            price_data: 价格数据

        Returns:
            预期年化波动率
        """
        if len(price_data) < self.lookback_window:
            return 0.0

        returns = price_data.pct_change().dropna()

        if len(returns) < self.lookback_window:
            return 0.0

        # 计算协方差矩阵
        recent_returns = returns.iloc[-self.lookback_window:]
        cov_matrix = recent_returns.cov()

        # 计算组合方差
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))

        # 年化波动率
        portfolio_vol = np.sqrt(portfolio_variance * 252)

        return portfolio_vol

    def should_adjust(self,
                     current_vol: float,
                     target_vol: Optional[float] = None,
                     tolerance: float = 0.02) -> bool:
        """
        判断是否需要调整

        Args:
            current_vol: 当前波动率
            target_vol: 目标波动率
            tolerance: 容忍度

        Returns:
            是否需要调整
        """
        if target_vol is None:
            target_vol = self.target_vol

        return abs(current_vol - target_vol) > tolerance

    def get_volatility_regime(self,
                            price_data: pd.DataFrame) -> str:
        """
        判断波动率状态

        Args:
            price_data: 价格数据

        Returns:
            波动率状态 ('低', '中', '高')
        """
        if len(price_data) < 60:
            return '中'

        returns = price_data.pct_change().dropna()

        # 短期波动率 (20日)
        short_vol = returns.iloc[-20:].std() * np.sqrt(252)

        # 长期波动率 (60日)
        long_vol = returns.iloc[-60:].std() * np.sqrt(252)

        # 判断状态
        if (short_vol < long_vol * 0.8).any():
            return '低'
        elif (short_vol > long_vol * 1.2).any():
            return '高'
        else:
            return '中'

    def adaptive_target_volatility(self,
                                 price_data: pd.DataFrame,
                                 base_target: Optional[float] = None) -> float:
        """
        自适应目标波动率

        Args:
            price_data: 价格数据
            base_target: 基础目标波动率

        Returns:
            调整后的目标波动率
        """
        if base_target is None:
            base_target = self.target_vol

        regime = self.get_volatility_regime(price_data)

        # 根据市场环境调整目标波动率
        if regime == '低':
            return base_target * 0.9  # 低波环境略微降低目标
        elif regime == '高':
            return base_target * 1.1  # 高波环境略微提高目标
        else:
            return base_target

    def _record_adjustment(self, realized_vol: float, target_vol: float, multiplier: float):
        """记录调整历史"""
        self.adjustment_history.append({
            'timestamp': pd.Timestamp.now(),
            'realized_vol': realized_vol,
            'target_vol': target_vol,
            'leverage_multiplier': multiplier,
            'adjustment_magnitude': abs(multiplier - 1.0)
        })

        # 保持历史长度
        if len(self.adjustment_history) > 252:
            self.adjustment_history = self.adjustment_history[-252:]

    def get_adjustment_stats(self) -> Dict:
        """获取调整统计信息"""
        if not self.adjustment_history:
            return {}

        history = self.adjustment_history

        return {
            'total_adjustments': len(history),
            'avg_realized_vol': np.mean([h['realized_vol'] for h in history]),
            'avg_target_vol': np.mean([h['target_vol'] for h in history]),
            'avg_leverage_multiplier': np.mean([h['leverage_multiplier'] for h in history]),
            'max_adjustment': max([h['adjustment_magnitude'] for h in history]),
            'adjustment_frequency': len([h for h in history if h['adjustment_magnitude'] > 0.1])
        }

    def get_adjustment_count(self) -> int:
        """获取调整次数"""
        return len(self.adjustment_history)

    def reset(self):
        """重置控制器"""
        self.adjustment_history.clear()