#!/usr/bin/env python3
"""
O2O强化学习训练示例脚本

本脚本展示了如何使用O2O (Offline-to-Online) 强化学习框架进行量化交易策略训练。
包含完整的三阶段训练流程：离线预训练 -> 热身微调 -> 在线学习

使用方法:
    python examples/o2o_training_example.py --config config/o2o_example_config.yaml
    
作者: O2O RL Team
日期: 2024
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from config.config_manager import ConfigManager
from data.data_manager import DataManager
from rl_agent.cvar_ppo_agent import EnhancedCVaRPPOAgent
from rl_agent.trading_environment import EnhancedTradingEnvironment
from trainer.o2o_coordinator import O2OTrainingCoordinator
from monitoring.value_drift_monitor import ValueDriftMonitor
from utils.logger import setup_logger
from utils.metrics import PerformanceMetrics
from utils.visualization import O2OVisualizer


class O2OTrainingExample:
    """O2O训练示例类"""
    
    def __init__(self, config_path: str):
        """
        初始化O2O训练示例
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # 初始化组件
        self.data_manager = None
        self.agent = None
        self.env = None
        self.coordinator = None
        self.drift_monitor = None
        self.visualizer = None
        
        # 训练状态
        self.training_results = {}
        self.performance_metrics = PerformanceMetrics()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"配置文件未找到: {self.config_path}")
            print("使用默认配置...")
            return self._get_default_config()
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'data': {
                'start_date': '2020-01-01',
                'end_date': '2023-12-31',
                'universe': 'csi300',
                'factors': ['technical', 'fundamental'],
                'apply_cleaning': True,
                'outlier_threshold': 3.0
            },
            'o2o': {
                # 离线预训练参数
                'offline_epochs': 100,
                'behavior_cloning_weight': 0.5,
                'td_learning_weight': 0.5,
                'offline_batch_size': 256,
                'offline_lr': 3e-4,
                
                # 热身微调参数
                'warmup_days': 60,
                'warmup_epochs': 20,
                'critic_only_updates': True,
                'warmup_lr': 1e-4,
                'convergence_threshold': 0.01,
                
                # 在线学习参数
                'initial_rho': 0.2,
                'rho_increment': 0.01,
                'trust_region_beta': 1.0,
                'beta_decay': 0.99,
                'online_buffer_size': 10000,
                'priority_alpha': 0.6,
                'priority_beta': 0.4,
                
                # 漂移检测参数
                'kl_threshold': 0.1,
                'sharpe_drop_threshold': 0.2,
                'cvar_breach_threshold': -0.02,
                'drift_window': 30
            },
            'agent': {
                'hidden_dims': [256, 128, 64],
                'activation': 'relu',
                'cvar_alpha': 0.05,
                'risk_lambda': 1.0,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5
            },
            'environment': {
                'initial_capital': 1000000,
                'transaction_cost': 0.001,
                'max_position': 0.1,
                'rebalance_freq': 'daily'
            },
            'training': {
                'max_episodes': 1000,
                'eval_freq': 50,
                'save_freq': 100,
                'early_stopping_patience': 200
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'log_dir': 'logs',
                'tensorboard': True
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_dir = log_config.get('log_dir', 'logs')
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'o2o_training_{timestamp}.log')
        
        return setup_logger(
            name='o2o_training',
            level=log_level,
            log_file=log_file if log_config.get('save_logs', True) else None
        )
    
    def prepare_data(self):
        """准备训练数据"""
        self.logger.info("开始准备训练数据...")
        
        data_config = self.config['data']
        self.data_manager = DataManager()
        
        try:
            # 加载因子数据
            self.logger.info("加载因子数据...")
            factor_data = self.data_manager.load_factor_data(
                start_date=data_config['start_date'],
                end_date=data_config['end_date'],
                universe=data_config.get('universe', 'csi300')
            )
            
            # 加载价格数据
            self.logger.info("加载价格数据...")
            price_data = self.data_manager.load_price_data(
                start_date=data_config['start_date'],
                end_date=data_config['end_date'],
                universe=data_config.get('universe', 'csi300')
            )
            
            # 数据清洗
            if data_config.get('apply_cleaning', True):
                self.logger.info("执行数据清洗...")
                factor_data = self._clean_data(
                    factor_data, 
                    threshold=data_config.get('outlier_threshold', 3.0)
                )
            
            self.factor_data = factor_data
            self.price_data = price_data
            
            self.logger.info(f"数据准备完成 - 因子数据形状: {factor_data.shape}, "
                           f"价格数据形状: {price_data.shape}")
            
        except Exception as e:
            self.logger.error(f"数据准备失败: {e}")
            raise
    
    def _clean_data(self, data: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """数据清洗"""
        # 移除异常值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = data[col].mean()
            std = data[col].std()
            data[col] = data[col].clip(
                lower=mean - threshold * std,
                upper=mean + threshold * std
            )
        
        # 填充缺失值
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def initialize_components(self):
        """初始化训练组件"""
        self.logger.info("初始化训练组件...")
        
        try:
            # 获取数据维度
            state_dim = self.factor_data.shape[1]
            action_dim = len(self.price_data.columns)
            
            # 初始化智能体
            self.logger.info("初始化CVaR-PPO智能体...")
            self.agent = EnhancedCVaRPPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                config=self.config
            )
            
            # 初始化环境
            self.logger.info("初始化交易环境...")
            self.env = EnhancedTradingEnvironment(
                factor_data=self.factor_data,
                price_data=self.price_data,
                config=self.config
            )
            
            # 初始化漂移监控器
            self.logger.info("初始化分布漂移监控器...")
            self.drift_monitor = ValueDriftMonitor(self.config['o2o'])
            
            # 初始化训练协调器
            self.logger.info("初始化O2O训练协调器...")
            self.coordinator = O2OTrainingCoordinator(
                agent=self.agent,
                env=self.env,
                config=self.config,
                drift_monitor=self.drift_monitor
            )
            
            # 初始化可视化工具
            self.visualizer = O2OVisualizer(self.config)
            
            self.logger.info("组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def run_offline_pretraining(self):
        """执行离线预训练"""
        self.logger.info("=" * 50)
        self.logger.info("开始离线预训练阶段")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # 执行离线预训练
            offline_results = self.coordinator.run_offline_pretraining()
            
            # 记录结果
            self.training_results['offline'] = offline_results
            
            # 计算训练时间
            training_time = time.time() - start_time
            self.logger.info(f"离线预训练完成，耗时: {training_time:.2f}秒")
            
            # 评估预训练效果
            self._evaluate_offline_performance(offline_results)
            
        except Exception as e:
            self.logger.error(f"离线预训练失败: {e}")
            raise
    
    def run_warmup_finetuning(self):
        """执行热身微调"""
        self.logger.info("=" * 50)
        self.logger.info("开始热身微调阶段")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # 执行热身微调
            warmup_results = self.coordinator.run_warmup_finetuning()
            
            # 记录结果
            self.training_results['warmup'] = warmup_results
            
            # 计算训练时间
            training_time = time.time() - start_time
            self.logger.info(f"热身微调完成，耗时: {training_time:.2f}秒")
            
            # 评估微调效果
            self._evaluate_warmup_performance(warmup_results)
            
        except Exception as e:
            self.logger.error(f"热身微调失败: {e}")
            raise
    
    def run_online_learning(self):
        """执行在线学习"""
        self.logger.info("=" * 50)
        self.logger.info("开始在线学习阶段")
        self.logger.info("=" * 50)
        
        start_time = time.time()
        
        try:
            # 执行在线学习
            online_results = self.coordinator.start_online_learning()
            
            # 记录结果
            self.training_results['online'] = online_results
            
            # 计算训练时间
            training_time = time.time() - start_time
            self.logger.info(f"在线学习完成，耗时: {training_time:.2f}秒")
            
            # 评估在线学习效果
            self._evaluate_online_performance(online_results)
            
        except Exception as e:
            self.logger.error(f"在线学习失败: {e}")
            raise
    
    def run_full_training(self):
        """执行完整的O2O训练流程"""
        self.logger.info("开始完整的O2O训练流程")
        
        total_start_time = time.time()
        
        try:
            # 阶段1: 离线预训练
            self.run_offline_pretraining()
            
            # 阶段2: 热身微调
            self.run_warmup_finetuning()
            
            # 阶段3: 在线学习
            self.run_online_learning()
            
            # 计算总训练时间
            total_time = time.time() - total_start_time
            self.logger.info(f"完整训练流程完成，总耗时: {total_time:.2f}秒")
            
            # 生成最终报告
            self._generate_final_report()
            
        except Exception as e:
            self.logger.error(f"训练流程失败: {e}")
            raise
    
    def _evaluate_offline_performance(self, results: Dict):
        """评估离线预训练性能"""
        self.logger.info("评估离线预训练性能...")
        
        # 提取关键指标
        final_loss = results.get('final_loss', 0)
        convergence_epoch = results.get('convergence_epoch', 0)
        behavior_cloning_loss = results.get('behavior_cloning_loss', 0)
        td_loss = results.get('td_loss', 0)
        
        self.logger.info(f"最终损失: {final_loss:.6f}")
        self.logger.info(f"收敛轮数: {convergence_epoch}")
        self.logger.info(f"行为克隆损失: {behavior_cloning_loss:.6f}")
        self.logger.info(f"TD学习损失: {td_loss:.6f}")
        
        # 可视化训练过程
        if hasattr(self, 'visualizer'):
            self.visualizer.plot_offline_training_curves(results)
    
    def _evaluate_warmup_performance(self, results: Dict):
        """评估热身微调性能"""
        self.logger.info("评估热身微调性能...")
        
        # 提取关键指标
        critic_improvement = results.get('critic_improvement', 0)
        value_baseline_change = results.get('value_baseline_change', 0)
        convergence_achieved = results.get('convergence_achieved', False)
        
        self.logger.info(f"Critic改进: {critic_improvement:.6f}")
        self.logger.info(f"价值基线变化: {value_baseline_change:.6f}")
        self.logger.info(f"是否收敛: {convergence_achieved}")
        
        # 可视化微调过程
        if hasattr(self, 'visualizer'):
            self.visualizer.plot_warmup_progress(results)
    
    def _evaluate_online_performance(self, results: Dict):
        """评估在线学习性能"""
        self.logger.info("评估在线学习性能...")
        
        # 提取关键指标
        final_return = results.get('final_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0)
        cvar_risk = results.get('cvar_risk', 0)
        
        self.logger.info(f"最终收益率: {final_return:.4f}")
        self.logger.info(f"夏普比率: {sharpe_ratio:.4f}")
        self.logger.info(f"最大回撤: {max_drawdown:.4f}")
        self.logger.info(f"CVaR风险: {cvar_risk:.4f}")
        
        # 可视化在线学习过程
        if hasattr(self, 'visualizer'):
            self.visualizer.plot_online_performance(results)
    
    def _generate_final_report(self):
        """生成最终训练报告"""
        self.logger.info("生成最终训练报告...")
        
        # 创建报告目录
        report_dir = 'reports'
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成报告文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f'o2o_training_report_{timestamp}.md')
        
        # 生成报告内容
        report_content = self._create_report_content()
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"训练报告已保存: {report_file}")
        
        # 生成可视化报告
        if hasattr(self, 'visualizer'):
            viz_report = os.path.join(report_dir, f'o2o_visualization_{timestamp}.html')
            self.visualizer.generate_comprehensive_report(
                self.training_results, viz_report
            )
            self.logger.info(f"可视化报告已保存: {viz_report}")
    
    def _create_report_content(self) -> str:
        """创建报告内容"""
        report = f"""# O2O强化学习训练报告

## 训练概览

- **训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **配置文件**: {self.config_path}
- **数据范围**: {self.config['data']['start_date']} 至 {self.config['data']['end_date']}

## 训练结果

### 离线预训练阶段
"""
        
        if 'offline' in self.training_results:
            offline = self.training_results['offline']
            report += f"""
- 训练轮数: {offline.get('total_epochs', 'N/A')}
- 最终损失: {offline.get('final_loss', 'N/A'):.6f}
- 收敛轮数: {offline.get('convergence_epoch', 'N/A')}
- 行为克隆损失: {offline.get('behavior_cloning_loss', 'N/A'):.6f}
- TD学习损失: {offline.get('td_loss', 'N/A'):.6f}
"""
        
        report += "\n### 热身微调阶段\n"
        
        if 'warmup' in self.training_results:
            warmup = self.training_results['warmup']
            report += f"""
- 微调天数: {warmup.get('warmup_days', 'N/A')}
- Critic改进: {warmup.get('critic_improvement', 'N/A'):.6f}
- 价值基线变化: {warmup.get('value_baseline_change', 'N/A'):.6f}
- 收敛状态: {warmup.get('convergence_achieved', 'N/A')}
"""
        
        report += "\n### 在线学习阶段\n"
        
        if 'online' in self.training_results:
            online = self.training_results['online']
            report += f"""
- 最终收益率: {online.get('final_return', 'N/A'):.4f}
- 夏普比率: {online.get('sharpe_ratio', 'N/A'):.4f}
- 最大回撤: {online.get('max_drawdown', 'N/A'):.4f}
- CVaR风险: {online.get('cvar_risk', 'N/A'):.4f}
- 漂移检测次数: {online.get('drift_detections', 'N/A')}
"""
        
        report += f"""
## 配置参数

### O2O参数
```yaml
{yaml.dump(self.config.get('o2o', {}), default_flow_style=False)}
```

### 智能体参数
```yaml
{yaml.dump(self.config.get('agent', {}), default_flow_style=False)}
```

## 总结

本次O2O训练{'成功' if self.training_results else '未完成'}完成了三阶段训练流程。
详细的性能分析和可视化结果请参考相应的图表文件。

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def save_model(self, save_path: str = None):
        """保存训练好的模型"""
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f'models/o2o_model_{timestamp}.pth'
        
        # 创建模型目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型
        self.coordinator.save_model(save_path)
        self.logger.info(f"模型已保存: {save_path}")
    
    def load_model(self, model_path: str):
        """加载预训练模型"""
        self.coordinator.load_model(model_path)
        self.logger.info(f"模型已加载: {model_path}")


def create_example_config():
    """创建示例配置文件"""
    config_dir = 'config'
    os.makedirs(config_dir, exist_ok=True)
    
    config_file = os.path.join(config_dir, 'o2o_example_config.yaml')
    
    example = O2OTrainingExample.__new__(O2OTrainingExample)
    default_config = example._get_default_config()
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"示例配置文件已创建: {config_file}")
    return config_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='O2O强化学习训练示例')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/o2o_example_config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--create-config', 
        action='store_true',
        help='创建示例配置文件'
    )
    parser.add_argument(
        '--stage', 
        type=str, 
        choices=['offline', 'warmup', 'online', 'full'],
        default='full',
        help='训练阶段选择'
    )
    parser.add_argument(
        '--load-model', 
        type=str,
        help='加载预训练模型路径'
    )
    parser.add_argument(
        '--save-model', 
        type=str,
        help='保存模型路径'
    )
    
    args = parser.parse_args()
    
    # 创建示例配置文件
    if args.create_config:
        create_example_config()
        return
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        print("使用 --create-config 创建示例配置文件")
        return
    
    try:
        # 初始化训练示例
        trainer = O2OTrainingExample(args.config)
        
        # 准备数据
        trainer.prepare_data()
        
        # 初始化组件
        trainer.initialize_components()
        
        # 加载预训练模型（如果指定）
        if args.load_model:
            trainer.load_model(args.load_model)
        
        # 根据指定阶段执行训练
        if args.stage == 'offline':
            trainer.run_offline_pretraining()
        elif args.stage == 'warmup':
            trainer.run_warmup_finetuning()
        elif args.stage == 'online':
            trainer.run_online_learning()
        else:  # full
            trainer.run_full_training()
        
        # 保存模型（如果指定）
        if args.save_model:
            trainer.save_model(args.save_model)
        
        print("训练完成！")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()