"""
心理舒适度评估指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ComfortabilityMetrics:
    """心理舒适度量化指标"""
    
    def __init__(self, config: Dict):
        """
        初始化心理舒适度评估器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.comfort_thresholds = {
            'monthly_drawdown': config.get('monthly_dd_threshold', 0.05),
            'consecutive_losses': config.get('max_consecutive_losses', 5),
            'loss_ratio': config.get('max_loss_ratio', 0.4),
            'var_95': config.get('var_95_threshold', 0.01)
        }
        
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """
        计算心理舒适度相关指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            心理舒适度指标字典
        """
        if len(returns) < 2:
            return {}
        
        metrics = {}
        
        # 核心舒适度指标
        metrics.update(self._calculate_drawdown_comfort(returns))
        metrics.update(self._calculate_loss_comfort(returns))
        metrics.update(self._calculate_volatility_comfort(returns))
        metrics.update(self._calculate_consistency_comfort(returns))
        
        # 综合舒适度得分
        metrics['综合舒适度得分'] = self._calculate_composite_score(metrics)
        
        return metrics
    
    def _calculate_drawdown_comfort(self, returns: pd.Series) -> Dict:
        """计算回撤舒适度指标"""
        # 月度最大回撤
        monthly_max_drawdown = self.monthly_max_drawdown(returns)
        
        # 回撤持续时间
        drawdown_duration = self._calculate_drawdown_duration(returns)
        
        # 回撤频率
        drawdown_frequency = self._calculate_drawdown_frequency(returns)
        
        return {
            '月度最大回撤': monthly_max_drawdown,
            '平均回撤持续天数': drawdown_duration,
            '回撤频率': drawdown_frequency
        }
    
    def _calculate_loss_comfort(self, returns: pd.Series) -> Dict:
        """计算亏损舒适度指标"""
        # 连续亏损天数
        max_consecutive_losses = self.max_consecutive_losses(returns)
        
        # 下跌日占比
        loss_ratio = (returns < 0).mean()
        
        # 极端亏损频率
        extreme_loss_freq = (returns < -0.02).mean()
        
        # 大额亏损频率 (超过1%)
        large_loss_freq = (returns < -0.01).mean()
        
        return {
            '连续亏损天数': max_consecutive_losses,
            '下跌日占比': loss_ratio,
            '极端亏损频率': extreme_loss_freq,
            '大额亏损频率': large_loss_freq
        }
    
    def _calculate_volatility_comfort(self, returns: pd.Series) -> Dict:
        """计算波动率舒适度指标"""
        # VaR指标
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # 滚动波动率稳定性
        rolling_vol = returns.rolling(window=20).std()
        vol_stability = rolling_vol.std() if len(rolling_vol.dropna()) > 1 else 0
        
        # 异常波动天数 (超过日均波动2倍)
        daily_vol = returns.std()
        extreme_vol_days = (abs(returns) > 2 * daily_vol).mean()
        
        return {
            '日VaR_95%': var_95,
            '日VaR_99%': var_99,
            '波动率稳定性': vol_stability,
            '异常波动频率': extreme_vol_days
        }
    
    def _calculate_consistency_comfort(self, returns: pd.Series) -> Dict:
        """计算一致性舒适度指标"""
        # 12个月滚动夏普比率
        rolling_sharpe = self.rolling_sharpe(returns, 252)
        sharpe_stability = rolling_sharpe.std() if len(rolling_sharpe.dropna()) > 1 else 0
        
        # 收益序列的预测性 (自相关性)
        if len(returns) > 20:
            autocorr = returns.autocorr(lag=1)
        else:
            autocorr = 0
        
        # 正收益一致性
        positive_streaks = self._calculate_positive_streaks(returns)
        
        return {
            '12月滚动夏普': rolling_sharpe.iloc[-1] if not rolling_sharpe.empty else 0,
            '夏普比率稳定性': sharpe_stability,
            '收益自相关性': autocorr,
            '最长连续盈利天数': max(positive_streaks) if positive_streaks else 0
        }
    
    def monthly_max_drawdown(self, returns: pd.Series, window: int = 21) -> float:
        """计算月度（21个交易日）最大回撤"""
        if len(returns) < window:
            return 0.0
            
        monthly_drawdowns = []
        
        for i in range(window, len(returns) + 1):
            period_returns = returns.iloc[i-window:i]
            cumulative = (1 + period_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            monthly_drawdowns.append(drawdowns.min())
        
        return min(monthly_drawdowns) if monthly_drawdowns else 0.0
    
    def max_consecutive_losses(self, returns: pd.Series) -> int:
        """计算最大连续亏损天数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def rolling_sharpe(self, returns: pd.Series, window: int) -> pd.Series:
        """计算滚动夏普比率"""
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std
        return rolling_sharpe.dropna()
    
    def _calculate_drawdown_duration(self, returns: pd.Series) -> float:
        """计算平均回撤持续时间"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        
        in_drawdown = cumulative < running_max
        
        if not in_drawdown.any():
            return 0.0
        
        # 找到回撤开始和结束点
        drawdown_periods = []
        start_drawdown = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_drawdown is None:
                start_drawdown = i
            elif not is_dd and start_drawdown is not None:
                drawdown_periods.append(i - start_drawdown)
                start_drawdown = None
        
        # 处理最后一个回撤期
        if start_drawdown is not None:
            drawdown_periods.append(len(returns) - start_drawdown)
        
        return np.mean(drawdown_periods) if drawdown_periods else 0.0
    
    def _calculate_drawdown_frequency(self, returns: pd.Series) -> float:
        """计算回撤频率（每年回撤次数）"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        
        in_drawdown = cumulative < running_max
        
        # 计算回撤开始次数
        drawdown_starts = 0
        prev_in_dd = False
        
        for is_dd in in_drawdown:
            if is_dd and not prev_in_dd:
                drawdown_starts += 1
            prev_in_dd = is_dd
        
        # 年化频率
        years = len(returns) / 252
        return drawdown_starts / years if years > 0 else 0
    
    def _calculate_positive_streaks(self, returns: pd.Series) -> List[int]:
        """计算正收益连续天数序列"""
        streaks = []
        current_streak = 0
        
        for ret in returns:
            if ret > 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        # 处理最后一个连续期
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """计算综合舒适度得分 (0-100分)"""
        score = 100  # 满分开始
        
        # 回撤惩罚
        monthly_dd = abs(metrics.get('月度最大回撤', 0))
        if monthly_dd > self.comfort_thresholds['monthly_drawdown']:
            score -= (monthly_dd / self.comfort_thresholds['monthly_drawdown'] - 1) * 20
        
        # 连续亏损惩罚
        consecutive_losses = metrics.get('连续亏损天数', 0)
        if consecutive_losses > self.comfort_thresholds['consecutive_losses']:
            score -= (consecutive_losses / self.comfort_thresholds['consecutive_losses'] - 1) * 15
        
        # 下跌日占比惩罚
        loss_ratio = metrics.get('下跌日占比', 0)
        if loss_ratio > self.comfort_thresholds['loss_ratio']:
            score -= (loss_ratio / self.comfort_thresholds['loss_ratio'] - 1) * 10
        
        # VaR惩罚
        var_95 = abs(metrics.get('日VaR_95%', 0))
        if var_95 > self.comfort_thresholds['var_95']:
            score -= (var_95 / self.comfort_thresholds['var_95'] - 1) * 15
        
        # 波动率稳定性奖励
        vol_stability = metrics.get('波动率稳定性', 0)
        if vol_stability < 0.01:  # 低波动率稳定性为好
            score += 5
        
        # 夏普比率奖励
        sharpe = metrics.get('12月滚动夏普', 0)
        if sharpe > 0.5:
            score += min((sharpe - 0.5) * 10, 10)
        
        return float(max(0, min(100, score)))  # 限制在0-100范围内并转为float