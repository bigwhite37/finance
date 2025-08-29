#!/usr/bin/env python3
"""
动态MAR (Minimum Acceptable Return) 计算辅助函数
支持基于无风险利率、基准收益率的动态调整
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

def calculate_dynamic_mar(
    eval_end: int,
    df: pd.DataFrame,
    mar_mode: str = 'dynamic_benchmark',
    risk_free_rate: float = 0.025,
    benchmark_return: float = 0.08,
    lookback_days: int = 180,
    min_periods: int = 60,
    benchmark_data: Optional[pd.DataFrame] = None,
    config_dict: Optional[dict] = None
) -> float:
    """
    计算动态MAR (最低可接受收益率)

    Parameters:
    -----------
    eval_end : int
        评估截止位置
    df : pd.DataFrame
        股票价格数据
    mar_mode : str
        MAR模式：'fixed', 'risk_free', 'dynamic_benchmark'
    risk_free_rate : float
        年化无风险利率
    benchmark_return : float
        年化基准收益率（固定模式下使用）
    lookbook_days : int
        动态计算的回看天数
    min_periods : int
        最小计算周期
    benchmark_data : pd.DataFrame, optional
        基准指数数据（用于动态基准模式）
    config_dict : dict, optional
        简化的配置字典，用于并行计算场景

    Returns:
    --------
    float
        年化MAR
    """

    # 如果提供了配置字典，优先使用其中的参数
    if config_dict:
        mar_mode = config_dict.get('mar_mode', mar_mode)
        risk_free_rate = config_dict.get('risk_free_rate', risk_free_rate)
        benchmark_return = config_dict.get('benchmark_return', benchmark_return)
        lookback_days = config_dict.get('sortino_lookback', lookback_days)
        min_periods = config_dict.get('sortino_min_periods', min_periods)

    if mar_mode == 'fixed':
        # 固定MAR：使用配置的基准收益率
        return float(benchmark_return)

    elif mar_mode == 'risk_free':
        # 无风险利率MAR：使用配置的无风险利率
        return float(risk_free_rate)

    elif mar_mode == 'dynamic_benchmark':
        # 动态基准MAR：基于历史基准收益+无风险利率的组合
        try:
            if benchmark_data is not None and len(benchmark_data) > 0:
                # 使用实际基准数据计算历史收益
                start_pos = max(0, eval_end - lookback_days)
                if start_pos >= eval_end - min_periods:
                    # 数据不足，回退到风险自由利率
                    logger.debug(f"基准数据不足，回退到无风险利率: {risk_free_rate}")
                    return float(risk_free_rate)

                benchmark_window = benchmark_data.iloc[start_pos:eval_end]
                if 'close' in benchmark_window.columns:
                    prices = benchmark_window['close'].dropna()
                    if len(prices) >= min_periods:
                        # 计算历史年化收益率
                        returns = prices.pct_change().dropna()
                        if len(returns) > 0:
                            historical_return = returns.mean() * 252  # 年化
                            # MAR = 70% 历史基准收益 + 30% 无风险利率（保守）
                            dynamic_mar = 0.7 * float(historical_return) + 0.3 * float(risk_free_rate)
                            logger.debug(f"动态MAR计算: 历史收益={historical_return:.4f}, MAR={dynamic_mar:.4f}")
                            return float(dynamic_mar)

            # 基准数据不可用或计算失败，回退到组合方式
            logger.debug(f"基准数据不可用，使用配置基准: {benchmark_return}")
            # MAR = 80% 配置基准 + 20% 无风险利率
            combined_mar = 0.8 * float(benchmark_return) + 0.2 * float(risk_free_rate)
            return float(combined_mar)

        except Exception as e:
            logger.warning(f"动态MAR计算失败: {e}, 回退到无风险利率")
            return float(risk_free_rate)

    else:
        # 未知模式，回退到无风险利率
        logger.warning(f"未知MAR模式: {mar_mode}, 使用无风险利率")
        return float(risk_free_rate)

def calculate_rolling_volatility_weights(
    eval_end: int,
    df: pd.DataFrame,
    total_weight_base: float = 0.6,
    downside_weight_base: float = 0.4,
    rolling_window: int = 252,
    min_periods: int = 60,
    adjustment_factor: float = 0.2
) -> tuple[float, float]:
    """
    计算时间滚动的波动率权重组合

    根据历史波动率特征动态调整总波动率vs下行波动率的权重

    Parameters:
    -----------
    eval_end : int
        评估截止位置
    df : pd.DataFrame
        股票价格数据
    total_weight_base : float
        总波动率基础权重
    downside_weight_base : float
        下行波动率基础权重
    rolling_window : int
        滚动调整窗口
    min_periods : int
        最小计算周期
    adjustment_factor : float
        调整幅度因子

    Returns:
    --------
    tuple[float, float]
        (调整后的总波动率权重, 调整后的下行波动率权重)
    """

    try:
        start_pos = max(0, eval_end - rolling_window)
        if start_pos >= eval_end - min_periods:
            # 数据不足，使用基础权重
            return float(total_weight_base), float(downside_weight_base)

        # 计算历史收益
        price_window = df['close'].iloc[start_pos:eval_end].dropna()
        if len(price_window) < min_periods:
            return float(total_weight_base), float(downside_weight_base)

        returns = price_window.pct_change().dropna()
        if len(returns) < min_periods:
            return float(total_weight_base), float(downside_weight_base)

        # 计算波动率特征
        total_vol = returns.std() * np.sqrt(252)
        negative_returns = returns[returns < 0]

        if len(negative_returns) == 0:
            # 没有负收益，降低下行波动权重
            total_weight = min(1.0, total_weight_base + adjustment_factor)
            downside_weight = max(0.0, downside_weight_base - adjustment_factor)
        else:
            downside_vol = negative_returns.std() * np.sqrt(252)
            downside_ratio = downside_vol / total_vol if total_vol > 0 else 0.5

            # 下行波动占比高时，提高下行波动权重
            if downside_ratio > 0.8:  # 下行占主导
                total_weight = max(0.0, total_weight_base - adjustment_factor)
                downside_weight = min(1.0, downside_weight_base + adjustment_factor)
            elif downside_ratio < 0.4:  # 波动相对均匀
                total_weight = min(1.0, total_weight_base + adjustment_factor * 0.5)
                downside_weight = max(0.0, downside_weight_base - adjustment_factor * 0.5)
            else:
                # 适中情况，使用基础权重
                total_weight = float(total_weight_base)
                downside_weight = float(downside_weight_base)

        # 确保权重和为1.0
        total_sum = total_weight + downside_weight
        if total_sum > 0:
            total_weight = total_weight / total_sum
            downside_weight = downside_weight / total_sum
        else:
            # 异常情况，回退到基础权重
            total_weight = total_weight_base
            downside_weight = downside_weight_base

        logger.debug(f"滚动权重调整: 总波动={total_weight:.3f}, 下行波动={downside_weight:.3f}")
        return float(total_weight), float(downside_weight)

    except Exception as e:
        logger.warning(f"滚动权重计算失败: {e}, 使用基础权重")
        return float(total_weight_base), float(downside_weight_base)