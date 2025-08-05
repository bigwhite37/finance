#!/usr/bin/env python3
"""
系统集成测试脚本
验证所有组件能正常工作
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import QlibDataLoader, split_data
from env import PortfolioEnv
import pandas as pd
import numpy as np

def test_data_loading():
    """测试数据加载功能"""
    print("测试数据加载...")
    
    loader = QlibDataLoader()
    loader.initialize_qlib()
    
    # 获取少量股票
    stocks = loader.get_stock_list(market='csi300', limit=3)
    print(f"获取股票: {stocks}")
    
    # 加载数据
    data = loader.load_data(
        instruments=stocks,
        start_time='2023-01-01',
        end_time='2023-03-31',
        freq='day'
    )
    print(f"加载数据: {data.shape}")
    
    # 测试数据分割
    train_data, valid_data, test_data = split_data(
        data, 
        '2023-01-01', '2023-02-15',
        '2023-02-16', '2023-02-28', 
        '2023-03-01', '2023-03-31'
    )
    
    print(f"数据分割: 训练={train_data.shape}, 验证={valid_data.shape}, 测试={test_data.shape}")
    
    return train_data, valid_data, test_data

def test_environment(data):
    """测试环境功能"""
    print("测试环境...")
    
    env = PortfolioEnv(
        data=data,
        initial_cash=1000000,
        lookback_window=5,
        transaction_cost=0.003,
        features=['$close', '$open', '$high', '$low', '$volume']
    )
    
    obs, info = env.reset(seed=42)
    print(f"环境重置成功，观察空间: {obs.shape}")
    
    # 测试几步
    for i in range(3):
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"步骤 {i+1}: 奖励={reward:.4f}, 结束={terminated}")
        
        if terminated or truncated:
            break
    
    print("环境测试完成")

def main():
    """主测试函数"""
    print("=" * 60)
    print("强化学习投资系统集成测试")
    print("=" * 60)
    
    try:
        # 测试数据加载
        train_data, valid_data, test_data = test_data_loading()
        
        # 测试环境
        test_environment(train_data)
        
        print("\n" + "=" * 60)
        print("所有测试通过！系统运行正常。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        raise

if __name__ == "__main__":
    main()