"""
Adaptive hyperparameter tuning example for O2O RL training.
Demonstrates how to use the adaptive tuning system to optimize training performance.
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Any
import logging
import matplotlib.pyplot as plt
from pathlib import Path

from trainer.adaptive_hyperparameter_tuner import (
    AdaptiveHyperparameterTuner, PerformanceMetrics, HyperparameterSearcher
)
from trainer.o2o_coordinator import O2OTrainingCoordinator, O2OCoordinatorConfig
from rl_agent.cvar_ppo_agent import CVaRPPOAgent
from rl_agent.trading_environment import TradingEnvironment
from data.offline_dataset import OfflineDataset
from config.config_manager import ConfigManager

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    
    # 生成模拟市场数据
    n_days = 1000
    n_assets = 50
    
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # 因子数据
    factor_data = pd.DataFrame({
        'date': np.repeat(dates, n_assets),
        'asset': np.tile([f'asset_{i:03d}' for i in range(n_assets)], n_days),
        'factor_1': np.random.randn(n_days * n_assets),
        'factor_2': np.random.randn(n_days * n_assets),
        'factor_3': np.random.randn(n_days * n_assets),
        'factor_4': np.random.randn(n_days * n_assets),
        'factor_5': np.random.randn(n_days * n_assets),
    })
    
    # 价格数据
    price_data = pd.DataFrame({
        'date': np.repeat(dates, n_assets),
        'asset': np.tile([f'asset_{i:03d}' for i in range(n_assets)], n_days),
        'open': 100 + np.random.randn(n_days * n_assets) * 10,
        'high': 105 + np.random.randn(n_days * n_assets) * 10,
        'low': 95 + np.random.randn(n_days * n_assets) * 10,
        'close': 100 + np.random.randn(n_days * n_assets) * 10,
        'volume': np.random.randint(1000, 10000, n_days * n_assets)
    })
    
    return factor_data, price_data


def demonstrate_adaptive_learning_rate():
    """演示自适应学习率调整"""
    print("\n=== 自适应学习率调整演示 ===")
    
    from trainer.adaptive_hyperparameter_tuner import AdaptiveLearningRateScheduler
    
    # 创建自适应学习率调度器
    scheduler = AdaptiveLearningRateScheduler(
        initial_lr=3e-4,
        min_lr=1e-6,
        max_lr=1e-2,
        patience=10,
        factor=0.8
    )
    
    # 模拟训练过程
    performance_history = []
    lr_history = []
    
    for step in range(200):
        # 模拟性能变化
        if step < 50:
            # 初期快速提升
            performance = 0.1 + step * 0.01 + np.random.normal(0, 0.005)
        elif step < 100:
            # 中期平稳
            performance = 0.6 + np.random.normal(0, 0.01)
        elif step < 150:
            # 后期缓慢提升
            performance = 0.6 + (step - 100) * 0.002 + np.random.normal(0, 0.01)
        else:
            # 最后阶段性能下降
            performance = 0.7 - (step - 150) * 0.005 + np.random.normal(0, 0.01)
            
        # 更新学习率
        new_lr = scheduler.step(performance)
        
        performance_history.append(performance)
        lr_history.append(new_lr)
        
        if step % 20 == 0:
            print(f"Step {step}: Performance={performance:.4f}, LR={new_lr:.2e}")
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(performance_history, label='Performance', color='blue')
    ax1.set_ylabel('Performance')
    ax1.set_title('Performance and Learning Rate Evolution')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(lr_history, label='Learning Rate', color='red')
    ax2.set_ylabel('Learning Rate')
    ax2.set_xlabel('Training Step')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('adaptive_learning_rate_demo.png', dpi=300, bbox_inches='tight')
    print("学习率调整可视化已保存: adaptive_learning_rate_demo.png")


def demonstrate_adaptive_sampling_ratio():
    """演示自适应采样比例调整"""
    print("\n=== 自适应采样比例调整演示 ===")
    
    from trainer.adaptive_hyperparameter_tuner import AdaptiveSamplingRatioController
    
    # 创建自适应采样比例控制器
    controller = AdaptiveSamplingRatioController(
        initial_rho=0.2,
        min_rho=0.1,
        max_rho=0.9,
        adaptation_rate=0.01
    )
    
    # 模拟训练过程
    rho_history = []
    performance_history = []
    kl_history = []
    
    for step in range(500):
        # 模拟性能和KL散度变化
        if step < 100:
            # 初期：性能提升，KL散度较低
            performance = 0.1 + step * 0.005 + np.random.normal(0, 0.01)
            kl_div = 0.005 + np.random.normal(0, 0.002)
        elif step < 200:
            # 中期：性能平稳，KL散度增加（分布漂移）
            performance = 0.6 + np.random.normal(0, 0.02)
            kl_div = 0.05 + (step - 100) * 0.001 + np.random.normal(0, 0.01)
        elif step < 350:
            # 后期：性能下降，KL散度很高
            performance = 0.6 - (step - 200) * 0.002 + np.random.normal(0, 0.02)
            kl_div = 0.15 + np.random.normal(0, 0.02)
        else:
            # 恢复期：性能恢复，KL散度降低
            performance = 0.3 + (step - 350) * 0.003 + np.random.normal(0, 0.01)
            kl_div = max(0.15 - (step - 350) * 0.001, 0.01) + np.random.normal(0, 0.005)
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            loss=1.0 - performance,
            reward=performance,
            cvar_estimate=-0.02,
            kl_divergence=kl_div
        )
        
        # 更新采样比例
        new_rho = controller.update(metrics, kl_div)
        
        rho_history.append(new_rho)
        performance_history.append(performance)
        kl_history.append(kl_div)
        
        if step % 50 == 0:
            print(f"Step {step}: Performance={performance:.4f}, KL={kl_div:.4f}, ρ={new_rho:.3f}")
    
    # 可视化结果
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    ax1.plot(performance_history, label='Performance', color='blue')
    ax1.set_ylabel('Performance')
    ax1.set_title('Adaptive Sampling Ratio Control')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(kl_history, label='KL Divergence', color='orange')
    ax2.axhline(y=0.1, color='red', linestyle='--', label='High KL Threshold')
    ax2.set_ylabel('KL Divergence')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(rho_history, label='Sampling Ratio (ρ)', color='green')
    ax3.set_ylabel('Sampling Ratio')
    ax3.set_xlabel('Training Step')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('adaptive_sampling_ratio_demo.png', dpi=300, bbox_inches='tight')
    print("采样比例调整可视化已保存: adaptive_sampling_ratio_demo.png")


def demonstrate_hyperparameter_search():
    """演示超参数搜索"""
    print("\n=== 超参数搜索演示 ===")
    
    # 定义搜索空间
    search_space = {
        'learning_rate': (1e-5, 1e-2),
        'cvar_lambda': (0.1, 5.0),
        'clip_epsilon': (0.1, 0.3),
        'gamma': (0.95, 0.999)
    }
    
    # 创建搜索器
    searcher = HyperparameterSearcher(
        search_space=search_space,
        n_trials=30,
        n_random_trials=10
    )
    
    # 模拟超参数搜索过程
    def objective_function(config: Dict[str, float]) -> float:
        """模拟目标函数（实际中这会是完整的训练过程）"""
        # 简化的性能函数，基于超参数组合
        lr = config['learning_rate']
        cvar_lambda = config['cvar_lambda']
        clip_epsilon = config['clip_epsilon']
        gamma = config['gamma']
        
        # 模拟性能计算（实际中会运行完整训练）
        performance = (
            -np.log(lr) * 0.1 +  # 学习率不宜过大或过小
            np.exp(-cvar_lambda) * 0.2 +  # CVaR权重适中最好
            (0.2 - clip_epsilon) ** 2 * (-2) +  # clip_epsilon接近0.2最好
            gamma * 0.5 +  # gamma越大越好
            np.random.normal(0, 0.1)  # 添加噪声
        )
        
        return performance
    
    # 运行搜索
    search_results = []
    for trial_idx in range(30):
        # 获取建议配置
        config = searcher.suggest_config(trial_idx)
        
        # 评估配置
        performance = objective_function(config)
        
        # 报告结果
        searcher.report_result(config, performance)
        
        search_results.append({
            'trial': trial_idx,
            'config': config,
            'performance': performance
        })
        
        if trial_idx % 5 == 0:
            print(f"Trial {trial_idx}: Performance={performance:.4f}")
            print(f"  Config: {config}")
    
    # 获取最佳配置
    best_config = searcher.get_best_config()
    print(f"\n最佳配置: {best_config}")
    print(f"最佳性能: {searcher.best_performance:.4f}")
    
    # 可视化搜索过程
    performances = [r['performance'] for r in search_results]
    best_so_far = []
    current_best = float('-inf')
    
    for perf in performances:
        if perf > current_best:
            current_best = perf
        best_so_far.append(current_best)
    
    plt.figure(figsize=(10, 6))
    plt.plot(performances, 'o-', alpha=0.7, label='Trial Performance')
    plt.plot(best_so_far, 'r-', linewidth=2, label='Best So Far')
    plt.xlabel('Trial')
    plt.ylabel('Performance')
    plt.title('Hyperparameter Search Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparameter_search_demo.png', dpi=300, bbox_inches='tight')
    print("超参数搜索可视化已保存: hyperparameter_search_demo.png")


def demonstrate_full_adaptive_tuning():
    """演示完整的自适应调优系统"""
    print("\n=== 完整自适应调优系统演示 ===")
    
    # 创建自适应调优器配置
    tuning_config = {
        'initial_lr': 3e-4,
        'min_lr': 1e-6,
        'max_lr': 1e-2,
        'initial_rho': 0.2,
        'min_rho': 0.1,
        'max_rho': 0.9,
        'initial_beta': 1.0,
        'target_kl': 0.01,
        'enable_hyperparameter_search': False
    }
    
    # 创建调优器
    tuner = AdaptiveHyperparameterTuner(tuning_config)
    
    # 模拟训练过程
    training_history = {
        'step': [],
        'performance': [],
        'learning_rate': [],
        'sampling_ratio': [],
        'trust_region_beta': [],
        'kl_divergence': []
    }
    
    for step in range(1000):
        # 模拟训练指标
        if step < 200:
            # 初期：快速学习
            performance = 0.1 + step * 0.002 + np.random.normal(0, 0.01)
            kl_div = 0.005 + np.random.normal(0, 0.002)
        elif step < 500:
            # 中期：平稳期
            performance = 0.5 + np.random.normal(0, 0.02)
            kl_div = 0.01 + np.random.normal(0, 0.005)
        elif step < 700:
            # 分布漂移期
            performance = 0.5 - (step - 500) * 0.001 + np.random.normal(0, 0.03)
            kl_div = 0.01 + (step - 500) * 0.0002 + np.random.normal(0, 0.01)
        else:
            # 恢复期
            performance = 0.3 + (step - 700) * 0.001 + np.random.normal(0, 0.02)
            kl_div = max(0.05 - (step - 700) * 0.0001, 0.005) + np.random.normal(0, 0.005)
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            loss=1.0 - performance,
            reward=performance,
            cvar_estimate=-0.02,
            kl_divergence=kl_div,
            memory_usage=0.7,
            training_time=1.0
        )
        
        # 更新超参数
        updated_params = tuner.update(metrics, kl_div)
        
        # 记录历史
        training_history['step'].append(step)
        training_history['performance'].append(performance)
        training_history['learning_rate'].append(updated_params['learning_rate'])
        training_history['sampling_ratio'].append(updated_params['sampling_ratio'])
        training_history['trust_region_beta'].append(updated_params['trust_region_beta'])
        training_history['kl_divergence'].append(kl_div)
        
        if step % 100 == 0:
            print(f"Step {step}: Performance={performance:.4f}, "
                  f"LR={updated_params['learning_rate']:.2e}, "
                  f"ρ={updated_params['sampling_ratio']:.3f}, "
                  f"β={updated_params['trust_region_beta']:.3f}")
    
    # 可视化完整训练过程
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 性能和KL散度
    ax1 = axes[0, 0]
    ax1.plot(training_history['step'], training_history['performance'], 
             label='Performance', color='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(training_history['step'], training_history['kl_divergence'], 
                  label='KL Divergence', color='orange', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Performance', color='blue')
    ax1_twin.set_ylabel('KL Divergence', color='orange')
    ax1.set_title('Performance and KL Divergence')
    ax1.grid(True)
    
    # 学习率
    ax2 = axes[0, 1]
    ax2.plot(training_history['step'], training_history['learning_rate'], 
             label='Learning Rate', color='red')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_yscale('log')
    ax2.set_title('Adaptive Learning Rate')
    ax2.grid(True)
    
    # 采样比例
    ax3 = axes[1, 0]
    ax3.plot(training_history['step'], training_history['sampling_ratio'], 
             label='Sampling Ratio (ρ)', color='green')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Sampling Ratio')
    ax3.set_title('Adaptive Sampling Ratio')
    ax3.grid(True)
    
    # 信任域参数
    ax4 = axes[1, 1]
    ax4.plot(training_history['step'], training_history['trust_region_beta'], 
             label='Trust Region β', color='purple')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Trust Region β')
    ax4.set_title('Adaptive Trust Region Parameter')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('full_adaptive_tuning_demo.png', dpi=300, bbox_inches='tight')
    print("完整自适应调优可视化已保存: full_adaptive_tuning_demo.png")
    
    # 打印统计信息
    stats = tuner.get_statistics()
    print(f"\n调优统计信息:")
    print(f"  总步数: {stats['step_count']}")
    print(f"  当前学习率: {stats['learning_rate_stats']['current']:.2e}")
    print(f"  最佳性能: {stats['learning_rate_stats']['best_performance']:.4f}")
    print(f"  采样比例统计: {stats['sampling_ratio_stats']}")
    print(f"  信任域统计: {stats['trust_region_stats']}")


def main():
    """主函数"""
    print("自适应超参数调优演示")
    print("=" * 50)
    
    # 创建输出目录
    Path("adaptive_tuning_results").mkdir(exist_ok=True)
    
    try:
        # 演示各个组件
        demonstrate_adaptive_learning_rate()
        demonstrate_adaptive_sampling_ratio()
        demonstrate_hyperparameter_search()
        demonstrate_full_adaptive_tuning()
        
        print("\n" + "=" * 50)
        print("所有演示完成！")
        print("生成的可视化文件:")
        print("- adaptive_learning_rate_demo.png")
        print("- adaptive_sampling_ratio_demo.png")
        print("- hyperparameter_search_demo.png")
        print("- full_adaptive_tuning_demo.png")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()