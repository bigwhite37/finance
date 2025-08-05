"""
自定义PortfolioEnv环境，支持多资产组合管理、回撤控制和奖励函数
基于Gymnasium接口，兼容Stable-Baselines3
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, List
import logging
from collections import deque
import matplotlib.pyplot as plt

# 尝试设置中文字体，如果失败就使用默认字体
try:
    from font_config import setup_chinese_font
    setup_chinese_font()
except ImportError:
    # 如果font_config不存在，使用基本的中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    """
    组合投资环境

    观察空间：历史价格、技术指标、持仓权重等
    动作空间：各资产的目标权重分配
    奖励函数：考虑收益率和回撤的复合奖励
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 data: pd.DataFrame,
                 initial_cash: float = 1000000,
                 lookback_window: int = 30,
                 transaction_cost: float = 0.003,
                 max_drawdown_threshold: float = 0.15,
                 reward_penalty: float = 2.0,
                 features: List[str] = None,
                 rebalance_freq: str = "daily",
                 max_steps: int = None):
        """
        初始化组合环境

        Args:
            data: 多股票价格数据，MultiIndex (datetime, instrument)
            initial_cash: 初始资金
            lookback_window: 历史观察窗口长度
            transaction_cost: 交易成本率
            max_drawdown_threshold: 最大回撤阈值
            reward_penalty: 回撤惩罚系数
            features: 特征列名列表
            rebalance_freq: 调仓频率
            max_steps: 最大步数（None则使用全部数据长度）
        """
        super().__init__()

        self.data = data.copy()
        self.initial_cash = initial_cash
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost
        self.max_drawdown_threshold = max_drawdown_threshold
        self.reward_penalty = reward_penalty
        self.rebalance_freq = rebalance_freq

        # 提取股票列表和时间序列
        # Qlib数据索引是(instrument, datetime)
        self.stock_list = list(data.index.get_level_values(0).unique())
        self.time_index = sorted(data.index.get_level_values(1).unique())
        self.num_stocks = len(self.stock_list)
        self.num_periods = len(self.time_index)

        # 设置最大步数（确保有足够长的episode）
        if max_steps is None:
            # 默认使用全部可用步数
            self.max_steps = self.num_periods - self.lookback_window - 1
        else:
            # 用户指定的步数，但不能超过数据长度
            self.max_steps = min(max_steps, self.num_periods - self.lookback_window - 1)

        # 确保最小episode长度
        self.max_steps = max(self.max_steps, 60)  # 至少60步（约3个月日频数据）

        # 特征列
        if features is None:
            self.features = ["$close", "$open", "$high", "$low", "$volume"]
        else:
            self.features = features
        self.num_features = len(self.features)

        # 验证数据完整性
        self._validate_data()

        # 定义动作和观察空间
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_stocks,),
            dtype=np.float32
        )

        # 观察空间：历史价格特征 + 当前持仓权重 + 市场状态
        obs_dim = (self.lookback_window * self.num_features * self.num_stocks +  # 历史特征
                  self.num_stocks +  # 当前权重
                  3)  # 市场状态：总价值、回撤、波动率

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 环境状态变量
        self.reset()

    def _validate_data(self):
        """验证数据完整性"""
        if self.data.empty:
            raise ValueError("数据不能为空")

        missing_features = set(self.features) - set(self.data.columns)
        if missing_features:
            raise ValueError(f"数据中缺少特征列: {missing_features}")

        # 详细检查数据缺失情况
        self._check_and_handle_missing_data()

        logger.info(f"数据验证完成：{self.num_stocks}只股票，{self.num_periods}个时间点")

    def _check_and_handle_missing_data(self):
        """检查并处理数据缺失值"""
        # 统计缺失值情况
        missing_stats = self.data.isnull().sum()
        total_missing = missing_stats.sum()

        if total_missing == 0:
            logger.info("数据完整，无缺失值")
            return

        # 详细报告缺失情况
        total_points = len(self.data)
        missing_ratio = total_missing / (total_points * len(self.features))

        logger.warning(f"数据缺失值统计:")
        logger.warning(f"总缺失点数: {total_missing}")
        logger.warning(f"缺失比例: {missing_ratio:.2%}")

        # 按特征统计缺失情况
        for feature in self.features:
            if feature in missing_stats and missing_stats[feature] > 0:
                feature_missing_ratio = missing_stats[feature] / total_points
                logger.warning(f"特征 {feature} 缺失: {missing_stats[feature]} 个点 ({feature_missing_ratio:.2%})")

        # 缺失值处理策略
        if missing_ratio > 0.1:  # 缺失超过10%
            raise RuntimeError(
                f"数据缺失过多 ({missing_ratio:.2%})，可能影响模型质量。\n"
                f"建议检查数据源或调整时间范围。缺失统计: {dict(missing_stats[missing_stats > 0])}"
            )
        elif missing_ratio > 0.05:  # 缺失5%-10%
            logger.error(
                f"数据缺失较多 ({missing_ratio:.2%})，将使用智能填充处理，但建议检查数据质量。\n"
                f"缺失统计: {dict(missing_stats[missing_stats > 0])}"
            )
        else:  # 缺失少于5%
            logger.info(
                f"数据存在少量缺失 ({missing_ratio:.2%})，属于正常现象（停牌/新股），将使用智能填充处理。\n"
                f"缺失统计: {dict(missing_stats[missing_stats > 0])}"
            )

        # 使用智能缺失值填充策略
        original_data = self.data.copy()

        # 1. 对于价格相关特征，使用前向填充（保持最后已知价格）
        price_features = [f for f in self.features if any(p in f for p in ['close', 'open', 'high', 'low', 'price'])]
        for feature in price_features:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].ffill()

        # 2. 对于成交量相关特征，使用0填充（停牌期间成交量为0）
        volume_features = [f for f in self.features if 'volume' in f]
        for feature in volume_features:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].fillna(0)

        # 3. 对于涨跌相关特征，使用0填充（停牌期间涨跌为0）
        change_features = [f for f in self.features if 'change' in f]
        for feature in change_features:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].fillna(0)

        # 4. 对于factor类特征，使用前向填充
        factor_features = [f for f in self.features if 'factor' in f]
        for feature in factor_features:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].ffill()

        # 5. 处理剩余缺失值
        remaining_missing = self.data.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"智能填充后仍有 {remaining_missing} 个缺失值，使用后向填充和均值填充")
            self.data = self.data.bfill()  # 后向填充

            # 最后使用均值填充
            for feature in self.features:
                if self.data[feature].isnull().any():
                    feature_mean = original_data[feature].mean()
                    if pd.isna(feature_mean):
                        # 使用合理的默认值
                        if any(p in feature for p in ['close', 'open', 'high', 'low', 'price']):
                            default_value = 100.0  # 价格类特征默认值
                        elif 'volume' in feature:
                            default_value = 0.0  # 成交量默认值（停牌时为0）
                        elif 'change' in feature:
                            default_value = 0.0  # 涨跌默认值
                        else:
                            default_value = 1.0  # 其他特征默认值
                        logger.warning(f"特征 {feature} 均值为NaN，使用默认值 {default_value}")
                        self.data[feature].fillna(default_value, inplace=True)
                    else:
                        self.data[feature].fillna(feature_mean, inplace=True)

        # 最终验证
        final_missing = self.data.isnull().sum().sum()
        if final_missing > 0:
            raise RuntimeError(f"数据处理后仍有 {final_missing} 个缺失值，无法继续处理")

        logger.info("数据缺失值处理完成")

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境状态"""
        super().reset(seed=seed)

        # 重置时间指针
        self.current_step = self.lookback_window

        # 重置投资组合状态
        self.total_value = self.initial_cash
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.num_stocks)  # 持有股数
        self.weights = np.zeros(self.num_stocks)   # 权重

        # 重置性能追踪
        self.value_history = deque([self.initial_cash], maxlen=252)  # 保留一年的历史
        self.return_history = deque(maxlen=252)
        self.drawdown_history = deque(maxlen=252)
        
        # 重置episode奖励历史（用于动态归一化）
        self.episode_rewards = []

        # 滚动峰值回撤计算
        self.peak_window = deque([self.initial_cash], maxlen=252)  # 一年峰值窗口
        self.rolling_peak = self.initial_cash
        self.current_drawdown = 0.0
        self.max_drawdown_so_far = 0.0

        # CVaR风险指标
        self.cvar_alpha = 0.05  # 5%分位
        self.cvar_value = 0.0

        self.steps_taken = 0

        # 重置交易记录
        self.trade_history = []
        self.transaction_costs = 0.0

        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步交易决策"""
        if self.current_step >= self.num_periods - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # 归一化动作（确保权重和为1）
        action = np.clip(action, 0, 1)
        if action.sum() > 0:
            action = action / action.sum()
        else:
            action = np.ones(self.num_stocks) / self.num_stocks

        # 获取当前价格
        current_prices = self._get_current_prices()

        # 执行调仓
        old_weights = self.weights.copy()
        trade_cost = self._rebalance_portfolio(action, current_prices)

        # 先更新组合价值用当前价格（调仓后的即时价值）
        self._update_portfolio_value(current_prices)

        # 再前进一步
        self.current_step += 1
        self.steps_taken += 1

        # 计算下一期价格变动对组合的影响
        if self.current_step < self.num_periods:
            next_prices = self._get_current_prices()
            self._update_portfolio_value(next_prices)

        # 计算奖励
        reward = self._calculate_reward(old_weights, action, trade_cost)

        # 检查终止条件 - 添加最小步数要求
        min_steps = min(120, self.max_steps // 2)  # 最少120步或一半max_steps

        # 只有达到最小步数后才允许因回撤提前终止
        early_termination_allowed = self.steps_taken >= min_steps

        terminated = (self.current_step >= min(self.num_periods - 1, self.lookback_window + self.max_steps) or
                     (early_termination_allowed and self.current_drawdown >= self.max_drawdown_threshold))

        # 记录交易历史（使用调仓时的价格）
        self._record_trade(action, current_prices, trade_cost)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_current_prices(self) -> np.ndarray:
        """获取当前时间点的股票价格"""
        current_time = self.time_index[self.current_step]
        prices = []

        for stock in self.stock_list:
            try:
                price = self.data.loc[(stock, current_time), "$close"]
                prices.append(price)
            except KeyError:
                # 如果数据缺失，使用前一天的价格
                if len(prices) > 0:
                    prices.append(prices[-1])
                else:
                    prices.append(100.0)  # 默认价格

        return np.array(prices)

    def _rebalance_portfolio(self, target_weights: np.ndarray, prices: np.ndarray) -> float:
        """调仓到目标权重，应用动态头寸缩放"""
        # 计算当前持仓价值
        current_holdings_value = np.sum(self.holdings * prices)
        total_value = self.cash + current_holdings_value

        # Drawdown-Modulated Position Sizing - 根据回撤动态缩放头寸
        drawdown_scale = self._calculate_position_scale()
        scaled_target_weights = target_weights * drawdown_scale

        # 重新归一化权重确保和为1
        weight_sum = np.sum(scaled_target_weights)
        if weight_sum > 0:
            scaled_target_weights = scaled_target_weights / weight_sum
        else:
            # 如果所有权重都被缩放为0，保持现金
            scaled_target_weights = np.zeros_like(target_weights)

        # 计算目标持仓
        target_values = scaled_target_weights * total_value
        target_shares = target_values / (prices + 1e-10)  # 避免除零

        # 计算交易量和成本
        trade_volumes = np.abs(target_shares - self.holdings)
        trade_values = trade_volumes * prices
        trade_cost = np.sum(trade_values) * self.transaction_cost

        # 更新持仓
        self.holdings = target_shares
        self.weights = scaled_target_weights  # 记录实际使用的权重

        # 确保现金不为负数
        new_cash = total_value - np.sum(target_values) - trade_cost
        if new_cash < 0:
            # 如果现金不足，按比例缩减持仓
            available_cash = total_value - trade_cost
            if available_cash > 0:
                scale_factor = available_cash / np.sum(target_values)
                target_values *= scale_factor
                self.holdings = target_values / (prices + 1e-10)
                new_cash = 0.0
            else:
                # 完全没有资金，清空持仓
                self.holdings = np.zeros_like(self.holdings)
                new_cash = total_value - trade_cost

        self.cash = max(0.0, new_cash)
        self.transaction_costs += trade_cost

        return trade_cost

    def _calculate_position_scale(self) -> float:
        """
        计算基于回撤的头寸缩放因子
        根据fix.md建议：scale = max(0.0, 1 - current_drawdown / max_drawdown_threshold)
        """
        if self.max_drawdown_threshold <= 0:
            return 1.0

        # 基础缩放：回撤越接近阈值，头寸越小
        base_scale = max(0.0, 1.0 - self.current_drawdown / self.max_drawdown_threshold)

        # CVaR调整：如果CVaR风险高，进一步缩放
        cvar_adjustment = 1.0
        if self.cvar_value > 0.02:  # CVaR超过2%时开始缩放
            cvar_adjustment = max(0.5, 1.0 - (self.cvar_value - 0.02) * 10)

        # 波动率调整：高波动时更保守
        volatility = self._get_current_volatility()
        vol_adjustment = 1.0
        if volatility > 0.25:  # 年化波动率超过25%时缩放
            vol_adjustment = max(0.7, 1.0 - (volatility - 0.25) * 2)

        # 综合缩放因子
        final_scale = base_scale * cvar_adjustment * vol_adjustment

        return max(0.1, final_scale)  # 最小保持10%的头寸

    def _update_portfolio_value(self, prices: np.ndarray):
        """更新组合价值"""
        # 验证价格数据
        if np.any(prices <= 0):
            # 只在真正的非正价格时警告（不是数值精度问题）
            significant_negative = prices[prices < -1e-6]  # 小于-0.000001的才警告
            if len(significant_negative) > 0:
                logger.warning(f"发现的确的非正价格: {significant_negative}")
            raise ValueError("价格数据包含非正价格")

        # 验证持仓数据（忽略浮点数精度误差）
        significant_negative_holdings = self.holdings[self.holdings < -1e-6]  # 小于-0.000001的才警告
        if len(significant_negative_holdings) > 0:
            logger.warning(f"发现的确的负持仓: {significant_negative_holdings}")
            raise ValueError("持仓数据包含负持仓")

        # 修复浮点数精度问题
        self.holdings = np.where(np.abs(self.holdings) < 1e-10, 0.0, self.holdings)  # 极小值设为0
        self.holdings = np.maximum(self.holdings, 0.0)  # 确保非负

        holdings_value = np.sum(self.holdings * prices)

        # 验证现金
        if self.cash < -1e-6:  # 只有的确的负现金才警告
            logger.warning(f"发现负现金: {self.cash}")
            raise ValueError("现金数据包含负现金")
        self.cash = max(0.0, self.cash)

        self.total_value = self.cash + holdings_value

        # 验证总价值合理性
        if self.total_value <= 0:
            logger.error(f"总价值非正: cash={self.cash}, holdings_value={holdings_value}")
            raise ValueError("总价值包含非正价值")

        # 更新历史记录
        self.value_history.append(self.total_value)

        # 计算收益率
        if len(self.value_history) > 1:
            return_rate = (self.total_value / self.value_history[-2]) - 1
            self.return_history.append(return_rate)

        # 滚动峰值回撤计算 - 避免早期随机高点影响
        self.peak_window.append(self.total_value)
        self.rolling_peak = max(self.peak_window)

        # 使用滚动峰值计算回撤
        if self.rolling_peak > 0:
            self.current_drawdown = max(0, (self.rolling_peak - self.total_value) / self.rolling_peak)
        else:
            self.current_drawdown = 0.0

        self.drawdown_history.append(self.current_drawdown)

        # 更新全局最大回撤
        if self.current_drawdown > self.max_drawdown_so_far:
            self.max_drawdown_so_far = self.current_drawdown

        # 计算CVaR风险指标
        self._update_cvar()

    def _update_cvar(self):
        """更新CVaR风险指标"""
        if len(self.return_history) >= 50:  # 至少50个样本才计算CVaR
            returns = np.array(self.return_history)
            # 计算5%分位的平均损失
            sorted_returns = np.sort(returns)
            n_tail = max(1, int(len(returns) * self.cvar_alpha))
            tail_returns = sorted_returns[:n_tail]
            self.cvar_value = -np.mean(tail_returns) if len(tail_returns) > 0 else 0.0
        else:
            self.cvar_value = 0.0

    def _calculate_reward(self, old_weights: np.ndarray, new_weights: np.ndarray, trade_cost: float) -> float:
        """
        修复的奖励函数：解决训练停滞问题
        主要修复：去除有问题的归一化、修正动态权重、添加基础奖励
        """
        # 1. 多尺度收益信息（降低放大系数避免过度波动）
        current_return = self.return_history[-1] if len(self.return_history) > 0 else 0.0
        
        # 短期收益（5日）
        if len(self.return_history) >= 5:
            short_term_return = np.mean(list(self.return_history)[-5:])
        else:
            short_term_return = current_return
            
        # 中期收益（20日）
        if len(self.return_history) >= 20:
            medium_term_return = np.mean(list(self.return_history)[-20:])
        else:
            medium_term_return = short_term_return
            
        # 复合收益信号（降低放大系数从100到50）
        composite_return = (0.5 * current_return + 0.3 * short_term_return + 0.2 * medium_term_return) * 50
        
        # 2. 计算Sharpe比率奖励（风险调整收益）
        sharpe_reward = 0.0
        if len(self.return_history) >= 10:
            recent_returns = np.array(list(self.return_history)[-20:]) if len(self.return_history) >= 20 else np.array(list(self.return_history))
            if len(recent_returns) > 1 and np.std(recent_returns) > 0:
                # 年化Sharpe比率（假设无风险利率3%）
                excess_return = np.mean(recent_returns) - 0.03/252
                sharpe_ratio = excess_return / np.std(recent_returns) * np.sqrt(252)
                sharpe_reward = np.clip(sharpe_ratio * 0.3, -1.0, 1.0)  # 减少Sharpe影响
        
        # 3. 修正的动态风险惩罚权重（压力期降低惩罚）
        volatility = self._get_current_volatility()
        market_stress = min(1.0, volatility / 0.3)  # 市场压力因子
        
        # 修正：压力期实际降低惩罚
        lambda_dd = 0.03 * (1 - market_stress * 0.3)  # 压力大时降低回撤惩罚
        lambda_cvar = 0.02 * (1 - market_stress * 0.2)  # 压力大时降低CVaR惩罚
        
        # 4. 风险惩罚项
        drawdown_penalty = lambda_dd * (self.current_drawdown * 100)
        cvar_penalty = lambda_cvar * (self.cvar_value * 100)
        
        # 5. 交易成本惩罚（减少惩罚强度）
        cost_bps = (trade_cost / self.total_value * 10000) if self.total_value > 0 else 0
        cost_penalty = cost_bps * 0.005  # 降低成本惩罚
        
        # 6. 波动率奖励/惩罚（降低惩罚强度）
        target_volatility = 0.15
        vol_deviation = abs(volatility - target_volatility)
        vol_penalty = 0.2 * max(0, vol_deviation - 0.08)  # 提高容忍度并降低惩罚
        
        # 7. 组合分散度奖励
        diversity_reward = 0.0
        if len(new_weights) > 1:
            hhi = np.sum(new_weights ** 2)
            optimal_hhi = 1.0 / len(new_weights)
            diversity_score = 1.0 - (hhi - optimal_hhi) / (1.0 - optimal_hhi)
            diversity_reward = diversity_score * 0.1  # 降低分散度奖励影响
        
        # 8. 添加基础奖励以避免持续负值
        base_reward = 0.5  # 基础正奖励
        
        # 9. 综合奖励计算
        raw_reward = (base_reward + composite_return + sharpe_reward + diversity_reward
                     - drawdown_penalty - cvar_penalty - cost_penalty - vol_penalty)
        
        # 10. 简单裁剪，移除有问题的归一化和tanh压缩
        final_reward = np.clip(raw_reward, -5.0, 5.0)
        
        # 11. 保存原始奖励用于分析（可选）
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        self.episode_rewards.append(final_reward)
        
        return final_reward

    def _get_current_volatility(self) -> float:
        """计算当前30日年化波动率"""
        if len(self.return_history) >= 30:
            recent_returns = np.array(list(self.return_history)[-30:])
            daily_vol = np.std(recent_returns)
            annualized_vol = daily_vol * np.sqrt(252)  # 年化
            return annualized_vol
        elif len(self.return_history) >= 10:
            recent_returns = np.array(list(self.return_history)[-10:])
            daily_vol = np.std(recent_returns)
            annualized_vol = daily_vol * np.sqrt(252)
            return annualized_vol
        else:
            return 0.0

    def _get_observation(self) -> np.ndarray:
        """获取当前观察状态"""
        observation = []

        # 历史价格特征
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step

        for i in range(start_idx, end_idx):
            if i < len(self.time_index):
                time_point = self.time_index[i]
                for stock in self.stock_list:
                    for feature in self.features:
                        try:
                            value = self.data.loc[(stock, time_point), feature]
                            observation.append(value)
                        except KeyError:
                            observation.append(0.0)
            else:
                # 填充缺失的历史数据
                observation.extend([0.0] * (self.num_stocks * self.num_features))

        # 补充到固定长度
        expected_hist_len = self.lookback_window * self.num_stocks * self.num_features
        while len(observation) < expected_hist_len:
            observation.append(0.0)

        # 当前权重
        observation.extend(self.weights.tolist())

        # 市场状态
        observation.extend([
            self.total_value / self.initial_cash,  # 相对价值
            self.current_drawdown,                 # 当前回撤
            np.std(list(self.return_history)) if len(self.return_history) > 1 else 0.0  # 波动率
        ])

        return np.array(observation, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            "total_value": self.total_value,
            "cash": self.cash,
            "holdings_value": np.sum(self.holdings * self._get_current_prices()) if self.current_step < self.num_periods else 0,
            "weights": self.weights.copy(),
            "current_drawdown": self.current_drawdown,
            "max_drawdown": max(self.drawdown_history) if self.drawdown_history else 0.0,
            "transaction_costs": self.transaction_costs,
            "total_return": (self.total_value / self.initial_cash) - 1,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "current_step": self.current_step,
            "num_trades": len(self.trade_history)
        }

    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        if len(self.return_history) < 2:
            return 0.0

        returns = np.array(list(self.return_history))
        if np.std(returns) == 0:
            return 0.0

        # 假设无风险利率为年化3%
        risk_free_rate = 0.03 / 252  # 日化无风险利率
        excess_returns = returns - risk_free_rate

        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

    def _record_trade(self, action: np.ndarray, prices: np.ndarray, cost: float):
        """记录交易历史"""
        trade_record = {
            "step": self.current_step,
            "timestamp": self.time_index[self.current_step] if self.current_step < len(self.time_index) else None,
            "action": action.copy(),
            "prices": prices.copy(),
            "cost": cost,
            "total_value": self.total_value,
            "drawdown": self.current_drawdown
        }
        self.trade_history.append(trade_record)

    def render(self, mode: str = "human"):
        """渲染环境状态"""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Total Value: {self.total_value:.2f}")
            print(f"Current Drawdown: {self.current_drawdown:.4f}")
            print(f"Weights: {self.weights}")
            print("-" * 50)

    def get_portfolio_performance(self) -> Dict[str, Any]:
        """获取组合性能统计"""
        if len(self.value_history) < 2:
            return {"error": "历史数据不足"}

        values = np.array(list(self.value_history))
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] / values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(values)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        max_drawdown = max(self.drawdown_history) if self.drawdown_history else 0.0

        # 计算Calmar比率
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.inf

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "transaction_costs": self.transaction_costs,
            "num_trades": len(self.trade_history)
        }

    def plot_performance(self, save_path: str = None):
        """绘制组合表现图"""
        if len(self.value_history) < 2:
            print("历史数据不足，无法绘图")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 资金曲线
        axes[0, 0].plot(list(self.value_history))
        axes[0, 0].set_title("组合价值曲线")
        axes[0, 0].set_ylabel("价值")

        # 回撤曲线
        axes[0, 1].plot(list(self.drawdown_history))
        axes[0, 1].fill_between(range(len(self.drawdown_history)), list(self.drawdown_history), alpha=0.3)
        axes[0, 1].set_title("回撤曲线")
        axes[0, 1].set_ylabel("回撤")

        # 收益率分布
        if len(self.return_history) > 0:
            axes[1, 0].hist(list(self.return_history), bins=50, alpha=0.7)
            axes[1, 0].set_title("收益率分布")
            axes[1, 0].set_xlabel("日收益率")

        # 权重变化
        if len(self.trade_history) > 0:
            weights_history = [trade["action"] for trade in self.trade_history]
            weights_array = np.array(weights_history).T

            for i, stock in enumerate(self.stock_list[:5]):  # 只显示前5只股票
                axes[1, 1].plot(weights_array[i], label=stock)
            axes[1, 1].set_title("持仓权重变化")
            axes[1, 1].set_ylabel("权重")
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
