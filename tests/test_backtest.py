#!/usr/bin/env python3
"""
回测功能测试脚本
验证训练好的模型在测试数据上的表现
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from data_loader import QlibDataLoader, split_data
from env import PortfolioEnv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_model_and_data(model_path: str, config_path: str = None):
    """加载模型和数据"""
    print(f"加载模型: {model_path}")

    # 加载模型
    model = PPO.load(model_path)
    print("模型加载成功")

    # 加载数据
    loader = QlibDataLoader()
    loader.initialize_qlib()

    # 使用与训练时相同的股票数量
    stocks = loader.get_stock_list(market='csi300', limit=50)
    data = loader.load_data(
        instruments=stocks,
        start_time='2020-01-01',  # 使用与训练时相同的时间范围
        end_time='2023-12-31',
        freq='day'
    )

    # 分割数据（使用与训练时相同的分割）
    train_data, valid_data, test_data = split_data(
        data,
        '2020-01-01', '2022-12-31',  # 训练集
        '2023-01-01', '2023-06-30',  # 验证集
        '2023-07-01', '2023-12-31'   # 测试集
    )

    print(f"数据加载完成 - 测试集: {test_data.shape}")

    return model, test_data

def run_backtest(model, test_data):
    """运行回测"""
    print("开始回测...")

    # 创建测试环境（参数必须与训练时一致）
    env = PortfolioEnv(
        data=test_data,
        initial_cash=1000000,
        lookback_window=30,  # 与训练时保持一致
        transaction_cost=0.003,
        max_drawdown_threshold=0.15,
        reward_penalty=2.0,
        features=['$close', '$open', '$high', '$low', '$volume', '$change', '$factor']  # 与训练时保持一致
    )

    # 运行回测
    obs, info = env.reset(seed=42)
    episode_length = 0
    total_reward = 0

    # 记录历史数据
    value_history = [env.total_value]
    weight_history = []
    action_history = []

    print("开始模拟交易...")

    while True:
        # 模型预测
        action, _ = model.predict(obs, deterministic=True)
        action_history.append(action.copy())
        weight_history.append(env.weights.copy())

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_length += 1
        value_history.append(env.total_value)

        if episode_length % 5 == 0:
            print(f"第{episode_length}步: 总价值={env.total_value:.2f}, 回撤={env.current_drawdown:.2%}")

        if terminated or truncated:
            break

    print(f"回测完成: {episode_length}步")

    # 获取最终表现
    performance = env.get_portfolio_performance()

    return {
        'performance': performance,
        'value_history': value_history,
        'weight_history': weight_history,
        'action_history': action_history,
        'total_reward': total_reward,
        'episode_length': episode_length,
        'env': env
    }

def analyze_results(results):
    """分析回测结果"""
    print("\n" + "="*60)
    print("回测结果分析")
    print("="*60)

    perf = results['performance']

    print(f"总收益率: {perf.get('total_return', 0):.2%}")
    print(f"年化收益率: {perf.get('annualized_return', 0):.2%}")
    print(f"波动率: {perf.get('volatility', 0):.2%}")
    print(f"夏普比率: {perf.get('sharpe_ratio', 0):.3f}")
    print(f"最大回撤: {perf.get('max_drawdown', 0):.2%}")
    print(f"Calmar比率: {perf.get('calmar_ratio', 0):.3f}")
    print(f"交易成本: {perf.get('transaction_costs', 0):.2f}")
    print(f"交易次数: {perf.get('num_trades', 0)}")

    # 绘制结果
    plot_results(results)

def plot_results(results):
    """绘制回测结果"""
    # 设置中文字体（如果可用）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    value_history = results['value_history']
    weight_history = np.array(results['weight_history'])
    env = results['env']

    # 资金曲线
    axes[0, 0].plot(value_history)
    axes[0, 0].set_title('Portfolio Value')
    axes[0, 0].set_ylabel('Value (CNY)')
    axes[0, 0].grid(True)

    # 回撤曲线
    if len(env.drawdown_history) > 0:
        axes[0, 1].plot(list(env.drawdown_history))
        axes[0, 1].fill_between(range(len(env.drawdown_history)),
                               list(env.drawdown_history), alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True)

    # 收益率分布
    if len(env.return_history) > 0:
        axes[1, 0].hist(list(env.return_history), bins=30, alpha=0.7)
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].grid(True)

    # 权重变化（显示前5只股票）
    if len(weight_history) > 0:
        for i in range(min(5, weight_history.shape[1])):
            axes[1, 1].plot(weight_history[:, i], label=f'Stock{i+1}')
        axes[1, 1].set_title('Portfolio Weights')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

    plt.tight_layout()

    # 保存图像
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/backtest_results.png', dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: results/backtest_results.png")

    plt.show()

def main():
    """主函数"""
    model_path = "models/final_model_20250805_083702.zip"

    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    try:
        # 加载模型和数据
        model, test_data = load_model_and_data(model_path)

        # 运行回测
        results = run_backtest(model, test_data)

        # 分析结果
        analyze_results(results)

        print("\n回测完成！")

    except Exception as e:
        print(f"回测过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()