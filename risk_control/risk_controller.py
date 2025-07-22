"""
风险控制器 - 统一的风险管理接口
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from .target_volatility import TargetVolatilityController
from .risk_parity import RiskParityOptimizer
from .stop_loss import DynamicStopLoss

logger = logging.getLogger(__name__)


class RiskController:
    """统一风险控制器"""

    def __init__(self, config: Dict):
        """
        初始化风险控制器

        Args:
            config: 风险控制配置
        """
        self.config = config

        # 初始化子模块
        self.target_vol_controller = TargetVolatilityController(config)
        self.risk_parity_optimizer = RiskParityOptimizer(config)
        self.stop_loss_manager = DynamicStopLoss(config)

        # 风险监控参数
        self.max_position = config.get('max_position', 0.1)
        self.max_leverage = config.get('max_leverage', 1.2)
        self.target_volatility = config.get('target_volatility', 0.12)
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.10)

        # 风险指标缓存
        self.risk_metrics = {}
        self.portfolio_history = []

    def process_weights(self,
                       raw_weights: np.ndarray,
                       price_data: pd.DataFrame,
                       current_nav: float,
                       state: Dict) -> np.ndarray:
        """
        处理原始权重，应用风险控制

        Args:
            raw_weights: RL输出的原始权重
            price_data: 价格数据
            current_nav: 当前净值
            state: 环境状态信息

        Returns:
            风险调整后的权重
        """
        # 1. 基础约束
        weights = self._apply_basic_constraints(raw_weights)

        # 2. 检查止损条件
        if self.stop_loss_manager.check_stop_loss(current_nav):
            logger.warning("触发动态止损，减仓50%")
            weights = weights * 0.5

        # 3. 目标波动率调整
        if len(price_data) > 20:
            weights = self.target_vol_controller.adjust_leverage(
                weights, price_data, self.target_volatility
            )

        # 4. 风险平价优化（可选）
        if self.config.get('enable_risk_parity', False) and len(price_data) > 60:
            weights = self._apply_risk_parity(weights, price_data)

        # 5. 最终安全检查
        weights = self._final_safety_check(weights, state)

        # 6. 更新历史记录
        self._update_history(weights, current_nav, state)

        return weights

    def _apply_basic_constraints(self, weights: np.ndarray) -> np.ndarray:
        """应用基础约束"""
        # 单股票仓位限制
        weights = np.clip(weights, -self.max_position, self.max_position)

        # 总杠杆限制
        total_leverage = np.sum(np.abs(weights))
        if total_leverage > self.max_leverage:
            weights = weights * (self.max_leverage / total_leverage)

        return weights

    def _apply_risk_parity(self, weights: np.ndarray, price_data: pd.DataFrame) -> np.ndarray:
        """应用风险平价优化"""
        returns = price_data.pct_change().dropna()

        if len(returns) < 60:
            return weights

        # 计算协方差矩阵
        cov_matrix = returns.iloc[-60:].cov()

        # 风险平价权重
        rp_weights = self.risk_parity_optimizer.optimize_weights(
            expected_returns=returns.mean(),
            cov_matrix=cov_matrix
        )

        # 组合原始权重和风险平价权重
        alpha_weight = self.config.get('alpha_weight', 0.7)
        final_weights = alpha_weight * weights + (1 - alpha_weight) * rp_weights.values

        return final_weights

    def _final_safety_check(self, weights: np.ndarray, state: Dict) -> np.ndarray:
        """最终安全检查"""
        current_drawdown = state.get('max_drawdown', 0.0)

        # 极端回撤保护
        if current_drawdown > self.max_drawdown_threshold * 0.8:
            protection_factor = 1 - (current_drawdown / self.max_drawdown_threshold) * 0.5
            weights = weights * max(protection_factor, 0.3)  # 最少保留30%仓位

        return weights

    def calculate_portfolio_risk(self,
                               weights: np.ndarray,
                               price_data: pd.DataFrame) -> Dict:
        """
        计算组合风险指标

        Args:
            weights: 组合权重
            price_data: 价格数据

        Returns:
            风险指标字典
        """
        if len(price_data) < 20:
            return {}

        returns = price_data.pct_change().dropna()

        # 组合收益率序列
        portfolio_returns = (returns * weights).sum(axis=1)

        # 计算风险指标
        risk_metrics = {
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'sharpe_ratio': (portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
            'total_leverage': np.sum(np.abs(weights)),
            'max_position': np.max(np.abs(weights)),
            'num_positions': np.sum(np.abs(weights) > 0.001)
        }

        return risk_metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()

    def check_risk_limits(self,
                         weights: np.ndarray,
                         price_data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        检查风险限制

        Args:
            weights: 组合权重
            price_data: 价格数据

        Returns:
            (is_within_limits, violation_messages)
        """
        violations = []

        # 基础约束检查
        if np.max(np.abs(weights)) > self.max_position:
            violations.append(f"单股票仓位超限: {np.max(np.abs(weights)):.3f} > {self.max_position}")

        if np.sum(np.abs(weights)) > self.max_leverage:
            violations.append(f"总杠杆超限: {np.sum(np.abs(weights)):.3f} > {self.max_leverage}")

        # 风险指标检查
        if len(price_data) > 20:
            risk_metrics = self.calculate_portfolio_risk(weights, price_data)

            if risk_metrics.get('volatility', 0) > self.target_volatility * 1.5:
                violations.append(f"波动率过高: {risk_metrics['volatility']:.3f} > {self.target_volatility * 1.5:.3f}")

            if abs(risk_metrics.get('var_95', 0)) > 0.03:
                violations.append(f"VaR过高: {abs(risk_metrics['var_95']):.3f} > 0.03")

        return len(violations) == 0, violations

    def _update_history(self, weights: np.ndarray, nav: float, state: Dict):
        """更新历史记录"""
        self.portfolio_history.append({
            'weights': weights.copy(),
            'nav': nav,
            'timestamp': pd.Timestamp.now(),
            'drawdown': state.get('max_drawdown', 0.0),
            'volatility': state.get('portfolio_volatility', 0.0)
        })

        # 保持历史长度
        if len(self.portfolio_history) > 252:  # 保留1年历史
            self.portfolio_history = self.portfolio_history[-252:]

    def get_risk_report(self) -> Dict:
        """生成风险报告"""
        if not self.portfolio_history:
            return {}

        recent_history = self.portfolio_history[-30:]  # 最近30期

        navs = [h['nav'] for h in recent_history]
        drawdowns = [h['drawdown'] for h in recent_history]
        volatilities = [h['volatility'] for h in recent_history]

        return {
            'current_nav': navs[-1] if navs else 1.0,
            'total_return': (navs[-1] / navs[0] - 1) if len(navs) > 1 else 0.0,
            'max_drawdown': max(drawdowns) if drawdowns else 0.0,
            'avg_volatility': np.mean(volatilities) if volatilities else 0.0,
            'total_positions': len(self.portfolio_history),
            'stop_loss_triggers': self.stop_loss_manager.get_trigger_count(),
            'risk_adjustments': self.target_vol_controller.get_adjustment_count()
        }

    def reset(self):
        """重置控制器状态"""
        self.portfolio_history.clear()
        self.risk_metrics.clear()
        self.stop_loss_manager.reset()
        self.target_vol_controller.reset()