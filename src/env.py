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
from font_config import setup_chinese_font

# 设置中文字体
setup_chinese_font()

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
                 rebalance_freq: str = "daily"):
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

        # 检查数据是否有缺失值
        if self.data.isnull().any().any():
            logger.warning("数据中存在缺失值，将进行前向填充")
            self.data = self.data.ffill().bfill()

        logger.info(f"数据验证完成：{self.num_stocks}只股票，{self.num_periods}个时间点")

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
        self.max_value = self.initial_cash
        self.current_drawdown = 0.0

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

        # 前进一步
        self.current_step += 1

        # 计算新的组合价值
        if self.current_step < self.num_periods:
            new_prices = self._get_current_prices()
            self._update_portfolio_value(new_prices)

        # 计算奖励
        reward = self._calculate_reward(old_weights, action, trade_cost)

        # 检查终止条件
        terminated = (self.current_step >= self.num_periods - 1 or
                     self.current_drawdown >= self.max_drawdown_threshold)

        # 记录交易历史
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
        """调仓到目标权重"""
        # 计算当前持仓价值
        current_holdings_value = np.sum(self.holdings * prices)
        total_value = self.cash + current_holdings_value

        # 计算目标持仓
        target_values = target_weights * total_value
        target_shares = target_values / prices

        # 计算交易量和成本
        trade_volumes = np.abs(target_shares - self.holdings)
        trade_values = trade_volumes * prices
        trade_cost = np.sum(trade_values) * self.transaction_cost

        # 更新持仓
        self.holdings = target_shares
        self.weights = target_weights
        self.cash = total_value - np.sum(target_values) - trade_cost
        self.transaction_costs += trade_cost

        return trade_cost

    def _update_portfolio_value(self, prices: np.ndarray):
        """更新组合价值"""
        holdings_value = np.sum(self.holdings * prices)
        self.total_value = self.cash + holdings_value

        # 更新历史记录
        self.value_history.append(self.total_value)

        # 计算收益率
        if len(self.value_history) > 1:
            return_rate = (self.total_value / self.value_history[-2]) - 1
            self.return_history.append(return_rate)

        # 更新最大价值和回撤
        if self.total_value > self.max_value:
            self.max_value = self.total_value

        self.current_drawdown = (self.max_value - self.total_value) / self.max_value
        self.drawdown_history.append(self.current_drawdown)

    def _calculate_reward(self, old_weights: np.ndarray, new_weights: np.ndarray, trade_cost: float) -> float:
        """
        计算奖励函数
        考虑收益率、回撤惩罚、交易成本
        """
        # 基础收益率奖励
        if len(self.return_history) > 0:
            portfolio_return = self.return_history[-1]
        else:
            portfolio_return = 0.0

        # 回撤惩罚
        drawdown_penalty = 0.0
        if self.current_drawdown > 0.05:  # 5%以上回撤开始惩罚
            drawdown_penalty = self.reward_penalty * (self.current_drawdown - 0.05)

        # 交易成本惩罚
        cost_penalty = trade_cost / self.total_value

        # 波动率惩罚（可选）
        volatility_penalty = 0.0
        if len(self.return_history) >= 10:
            recent_returns = list(self.return_history)[-10:]
            volatility = np.std(recent_returns)
            volatility_penalty = 0.1 * volatility  # 适度惩罚高波动

        # 综合奖励
        reward = portfolio_return - drawdown_penalty - cost_penalty - volatility_penalty

        return reward

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


class DrawdownEarlyStoppingCallback:
    """回撤早停回调"""

    def __init__(self, max_drawdown: float = 0.15, patience: int = 10):
        self.max_drawdown = max_drawdown
        self.patience = patience
        self.violation_count = 0

    def __call__(self, env: PortfolioEnv) -> bool:
        """检查是否需要早停"""
        if env.current_drawdown > self.max_drawdown:
            self.violation_count += 1
            if self.violation_count >= self.patience:
                logger.warning(f"回撤超过阈值{self.max_drawdown:.2%}，连续{self.patience}步，触发早停")
                return True
        else:
            self.violation_count = 0

        return False


if __name__ == "__main__":
    # 测试环境
    print("测试PortfolioEnv环境...")

    # 创建模拟数据
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    stocks = ["000001.SZ", "000002.SZ", "600000.SH"]

    data_list = []
    for date in dates:
        for stock in stocks:
            price = 100 + np.random.randn() * 10
            data_list.append({
                "datetime": date,
                "instrument": stock,
                "$close": price,
                "$open": price * (1 + np.random.randn() * 0.01),
                "$high": price * (1 + abs(np.random.randn()) * 0.02),
                "$low": price * (1 - abs(np.random.randn()) * 0.02),
                "$volume": np.random.randint(1000, 10000)
            })

    df = pd.DataFrame(data_list)
    df = df.set_index(["datetime", "instrument"])

    # 创建环境
    env = PortfolioEnv(df, initial_cash=100000, lookback_window=10)

    # 测试环境
    obs, info = env.reset()
    print(f"观察空间维度: {obs.shape}")
    print(f"动作空间: {env.action_space}")

    # 运行几步
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.4f}, total_value={info['total_value']:.2f}")

        if terminated:
            break

    # 获取性能统计
    performance = env.get_portfolio_performance()
    print("\n组合性能:")
    for key, value in performance.items():
        print(f"{key}: {value}")