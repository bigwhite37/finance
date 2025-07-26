#!/usr/bin/env python3
"""
O2O训练结果分析工具

分析O2O训练结果，生成详细的性能报告和可视化图表，包括：
- 训练过程分析
- 性能指标对比
- 风险收益分析
- 分布漂移分析
- 策略稳定性评估

使用方法:
    python examples/o2o_results_analyzer.py --results-dir results --output-dir analysis
    
作者: O2O RL Team
日期: 2024
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.metrics import PerformanceMetrics
from utils.visualization import O2OVisualizer


class O2OResultsAnalyzer:
    """O2O训练结果分析器"""
    
    def __init__(self, results_dir: str, output_dir: str = 'analysis'):
        """
        初始化结果分析器
        
        Args:
            results_dir: 结果目录
            output_dir: 输出目录
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.logger = setup_logger('o2o_analyzer', level='INFO')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 分析数据
        self.training_results = {}
        self.performance_metrics = PerformanceMetrics()
        self.analysis_results = {}
        
        # 可视化工具
        self.visualizer = O2OVisualizer({})
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_results(self):
        """加载训练结果"""
        self.logger.info("加载训练结果...")
        
        try:
            # 查找结果文件
            result_files = self._find_result_files()
            
            if not result_files:
                raise FileNotFoundError(f"在 {self.results_dir} 中未找到结果文件")
            
            # 加载每个结果文件
            for result_file in result_files:
                self._load_single_result(result_file)
            
            self.logger.info(f"成功加载 {len(self.training_results)} 个训练结果")
            
        except Exception as e:
            self.logger.error(f"加载结果失败: {e}")
            raise
    
    def _find_result_files(self) -> List[str]:
        """查找结果文件"""
        result_files = []
        
        if not os.path.exists(self.results_dir):
            return result_files
        
        for file in os.listdir(self.results_dir):
            if file.endswith('.json') or file.endswith('.pkl'):
                result_files.append(os.path.join(self.results_dir, file))
        
        return sorted(result_files)
    
    def _load_single_result(self, result_file: str):
        """加载单个结果文件"""
        try:
            if result_file.endswith('.json'):
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif result_file.endswith('.pkl'):
                import pickle
                with open(result_file, 'rb') as f:
                    data = pickle.load(f)
            else:
                return
            
            # 提取文件名作为键
            key = os.path.splitext(os.path.basename(result_file))[0]
            self.training_results[key] = data
            
        except Exception as e:
            self.logger.warning(f"加载文件 {result_file} 失败: {e}")
    
    def analyze_training_process(self):
        """分析训练过程"""
        self.logger.info("分析训练过程...")
        
        process_analysis = {}
        
        for exp_name, results in self.training_results.items():
            analysis = {
                'stages': {},
                'convergence': {},
                'stability': {}
            }
            
            # 分析各阶段
            if 'offline' in results:
                analysis['stages']['offline'] = self._analyze_offline_stage(
                    results['offline']
                )
            
            if 'warmup' in results:
                analysis['stages']['warmup'] = self._analyze_warmup_stage(
                    results['warmup']
                )
            
            if 'online' in results:
                analysis['stages']['online'] = self._analyze_online_stage(
                    results['online']
                )
            
            # 分析收敛性
            analysis['convergence'] = self._analyze_convergence(results)
            
            # 分析稳定性
            analysis['stability'] = self._analyze_stability(results)
            
            process_analysis[exp_name] = analysis
        
        self.analysis_results['training_process'] = process_analysis
        return process_analysis
    
    def _analyze_offline_stage(self, offline_results: Dict) -> Dict:
        """分析离线预训练阶段"""
        analysis = {}
        
        # 训练效率
        total_epochs = offline_results.get('total_epochs', 0)
        convergence_epoch = offline_results.get('convergence_epoch', total_epochs)
        analysis['training_efficiency'] = convergence_epoch / total_epochs if total_epochs > 0 else 0
        
        # 损失下降
        loss_history = offline_results.get('loss_history', [])
        if loss_history:
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            analysis['loss_reduction'] = (initial_loss - final_loss) / initial_loss
            analysis['loss_stability'] = np.std(loss_history[-10:]) if len(loss_history) >= 10 else 0
        
        # 行为克隆vs TD学习
        bc_loss = offline_results.get('behavior_cloning_loss', 0)
        td_loss = offline_results.get('td_loss', 0)
        analysis['bc_td_ratio'] = bc_loss / td_loss if td_loss > 0 else float('inf')
        
        return analysis
    
    def _analyze_warmup_stage(self, warmup_results: Dict) -> Dict:
        """分析热身微调阶段"""
        analysis = {}
        
        # 微调效果
        analysis['critic_improvement'] = warmup_results.get('critic_improvement', 0)
        analysis['value_baseline_change'] = warmup_results.get('value_baseline_change', 0)
        analysis['convergence_achieved'] = warmup_results.get('convergence_achieved', False)
        
        # 适应速度
        warmup_epochs = warmup_results.get('warmup_epochs', 0)
        convergence_epoch = warmup_results.get('convergence_epoch', warmup_epochs)
        analysis['adaptation_speed'] = convergence_epoch / warmup_epochs if warmup_epochs > 0 else 0
        
        return analysis
    
    def _analyze_online_stage(self, online_results: Dict) -> Dict:
        """分析在线学习阶段"""
        analysis = {}
        
        # 性能指标
        analysis['final_return'] = online_results.get('final_return', 0)
        analysis['sharpe_ratio'] = online_results.get('sharpe_ratio', 0)
        analysis['max_drawdown'] = online_results.get('max_drawdown', 0)
        analysis['cvar_risk'] = online_results.get('cvar_risk', 0)
        
        # 适应性指标
        analysis['drift_detections'] = online_results.get('drift_detections', 0)
        analysis['rho_evolution'] = online_results.get('rho_evolution', [])
        analysis['kl_divergence_history'] = online_results.get('kl_divergence_history', [])
        
        # 稳定性指标
        returns_history = online_results.get('returns_history', [])
        if returns_history:
            analysis['return_volatility'] = np.std(returns_history)
            analysis['return_skewness'] = stats.skew(returns_history)
            analysis['return_kurtosis'] = stats.kurtosis(returns_history)
        
        return analysis
    
    def _analyze_convergence(self, results: Dict) -> Dict:
        """分析收敛性"""
        convergence_analysis = {}
        
        # 各阶段收敛情况
        stages = ['offline', 'warmup', 'online']
        for stage in stages:
            if stage in results:
                stage_results = results[stage]
                convergence_analysis[f'{stage}_converged'] = stage_results.get('convergence_achieved', False)
                convergence_analysis[f'{stage}_convergence_epoch'] = stage_results.get('convergence_epoch', 0)
        
        # 整体收敛评分
        converged_stages = sum([convergence_analysis.get(f'{stage}_converged', False) for stage in stages])
        convergence_analysis['overall_convergence_score'] = converged_stages / len(stages)
        
        return convergence_analysis
    
    def _analyze_stability(self, results: Dict) -> Dict:
        """分析稳定性"""
        stability_analysis = {}
        
        # 训练稳定性
        if 'online' in results:
            online_results = results['online']
            
            # 收益稳定性
            returns_history = online_results.get('returns_history', [])
            if returns_history:
                stability_analysis['return_stability'] = 1 / (1 + np.std(returns_history))
            
            # 策略稳定性
            policy_entropy_history = online_results.get('policy_entropy_history', [])
            if policy_entropy_history:
                stability_analysis['policy_stability'] = 1 / (1 + np.std(policy_entropy_history))
            
            # 风险稳定性
            cvar_history = online_results.get('cvar_history', [])
            if cvar_history:
                stability_analysis['risk_stability'] = 1 / (1 + np.std(cvar_history))
        
        return stability_analysis
    
    def compare_experiments(self):
        """对比不同实验"""
        self.logger.info("对比不同实验...")
        
        if len(self.training_results) < 2:
            self.logger.warning("实验数量不足，无法进行对比")
            return {}
        
        comparison = {
            'performance_comparison': {},
            'efficiency_comparison': {},
            'stability_comparison': {}
        }
        
        # 性能对比
        performance_metrics = ['final_return', 'sharpe_ratio', 'max_drawdown', 'cvar_risk']
        for metric in performance_metrics:
            comparison['performance_comparison'][metric] = {}
            for exp_name, results in self.training_results.items():
                if 'online' in results:
                    comparison['performance_comparison'][metric][exp_name] = \
                        results['online'].get(metric, 0)
        
        # 效率对比
        efficiency_metrics = ['total_training_time', 'convergence_speed', 'sample_efficiency']
        for metric in efficiency_metrics:
            comparison['efficiency_comparison'][metric] = {}
            for exp_name, results in self.training_results.items():
                # 计算效率指标
                if metric == 'total_training_time':
                    total_time = 0
                    for stage in ['offline', 'warmup', 'online']:
                        if stage in results:
                            total_time += results[stage].get('training_time', 0)
                    comparison['efficiency_comparison'][metric][exp_name] = total_time
                
                elif metric == 'convergence_speed':
                    # 计算平均收敛速度
                    convergence_epochs = []
                    for stage in ['offline', 'warmup']:
                        if stage in results:
                            convergence_epochs.append(
                                results[stage].get('convergence_epoch', 0)
                            )
                    comparison['efficiency_comparison'][metric][exp_name] = \
                        np.mean(convergence_epochs) if convergence_epochs else 0
        
        self.analysis_results['experiment_comparison'] = comparison
        return comparison
    
    def analyze_risk_return_profile(self):
        """分析风险收益特征"""
        self.logger.info("分析风险收益特征...")
        
        risk_return_analysis = {}
        
        for exp_name, results in self.training_results.items():
            if 'online' not in results:
                continue
            
            online_results = results['online']
            analysis = {}
            
            # 基本风险收益指标
            returns = online_results.get('returns_history', [])
            if returns:
                analysis['mean_return'] = np.mean(returns)
                analysis['return_volatility'] = np.std(returns)
                analysis['sharpe_ratio'] = analysis['mean_return'] / analysis['return_volatility'] \
                    if analysis['return_volatility'] > 0 else 0
                
                # 风险指标
                analysis['var_95'] = np.percentile(returns, 5)
                analysis['cvar_95'] = np.mean([r for r in returns if r <= analysis['var_95']])
                analysis['max_drawdown'] = self._calculate_max_drawdown(returns)
                
                # 分布特征
                analysis['skewness'] = stats.skew(returns)
                analysis['kurtosis'] = stats.kurtosis(returns)
                analysis['jarque_bera_pvalue'] = stats.jarque_bera(returns)[1]
                
                # 尾部风险
                analysis['tail_ratio'] = len([r for r in returns if r < -2*analysis['return_volatility']]) / len(returns)
                
            risk_return_analysis[exp_name] = analysis
        
        self.analysis_results['risk_return_profile'] = risk_return_analysis
        return risk_return_analysis
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """计算最大回撤"""
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def analyze_distribution_drift(self):
        """分析分布漂移"""
        self.logger.info("分析分布漂移...")
        
        drift_analysis = {}
        
        for exp_name, results in self.training_results.items():
            if 'online' not in results:
                continue
            
            online_results = results['online']
            analysis = {}
            
            # KL散度分析
            kl_history = online_results.get('kl_divergence_history', [])
            if kl_history:
                analysis['mean_kl_divergence'] = np.mean(kl_history)
                analysis['max_kl_divergence'] = np.max(kl_history)
                analysis['kl_trend'] = self._calculate_trend(kl_history)
                analysis['drift_frequency'] = len([kl for kl in kl_history if kl > 0.1]) / len(kl_history)
            
            # 采样比例演化
            rho_history = online_results.get('rho_evolution', [])
            if rho_history:
                analysis['final_rho'] = rho_history[-1]
                analysis['rho_growth_rate'] = (rho_history[-1] - rho_history[0]) / len(rho_history) \
                    if len(rho_history) > 1 else 0
                analysis['rho_stability'] = 1 / (1 + np.std(rho_history))
            
            # 漂移事件分析
            analysis['total_drift_events'] = online_results.get('drift_detections', 0)
            analysis['drift_response_effectiveness'] = self._analyze_drift_response(online_results)
            
            drift_analysis[exp_name] = analysis
        
        self.analysis_results['distribution_drift'] = drift_analysis
        return drift_analysis
    
    def _calculate_trend(self, data: List[float]) -> float:
        """计算趋势（斜率）"""
        if len(data) < 2:
            return 0
        
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        return slope
    
    def _analyze_drift_response(self, online_results: Dict) -> float:
        """分析漂移响应效果"""
        # 简化的漂移响应效果评估
        drift_events = online_results.get('drift_detections', 0)
        if drift_events == 0:
            return 1.0  # 无漂移，响应效果完美
        
        # 基于漂移后的性能恢复评估
        returns_history = online_results.get('returns_history', [])
        if not returns_history:
            return 0.5  # 默认中等效果
        
        # 简单评估：漂移后收益的恢复程度
        post_drift_returns = returns_history[-min(30, len(returns_history)):]  # 最近30期
        pre_drift_returns = returns_history[:min(30, len(returns_history))]   # 前30期
        
        if len(post_drift_returns) > 0 and len(pre_drift_returns) > 0:
            post_performance = np.mean(post_drift_returns)
            pre_performance = np.mean(pre_drift_returns)
            
            if pre_performance != 0:
                recovery_ratio = post_performance / pre_performance
                return min(1.0, max(0.0, recovery_ratio))
        
        return 0.5
    
    def generate_visualizations(self):
        """生成可视化图表"""
        self.logger.info("生成可视化图表...")
        
        # 创建图表目录
        charts_dir = os.path.join(self.output_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. 训练过程可视化
        self._plot_training_process(charts_dir)
        
        # 2. 性能对比图
        self._plot_performance_comparison(charts_dir)
        
        # 3. 风险收益散点图
        self._plot_risk_return_scatter(charts_dir)
        
        # 4. 分布漂移分析图
        self._plot_distribution_drift(charts_dir)
        
        # 5. 交互式仪表板
        self._create_interactive_dashboard(charts_dir)
        
        self.logger.info(f"可视化图表已保存到: {charts_dir}")
    
    def _plot_training_process(self, output_dir: str):
        """绘制训练过程图"""
        for exp_name, results in self.training_results.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'训练过程分析 - {exp_name}', fontsize=16)
            
            # 离线训练损失
            if 'offline' in results:
                ax = axes[0, 0]
                loss_history = results['offline'].get('loss_history', [])
                if loss_history:
                    ax.plot(loss_history, color='blue', linewidth=2)
                    ax.set_title('离线预训练损失')
                    ax.set_xlabel('轮数')
                    ax.set_ylabel('损失值')
                    ax.grid(True, alpha=0.3)
            
            # 热身微调过程
            if 'warmup' in results:
                ax = axes[0, 1]
                warmup_loss = results['warmup'].get('loss_history', [])
                if warmup_loss:
                    ax.plot(warmup_loss, color='orange', linewidth=2)
                    ax.set_title('热身微调损失')
                    ax.set_xlabel('轮数')
                    ax.set_ylabel('损失值')
                    ax.grid(True, alpha=0.3)
            
            # 在线学习收益
            if 'online' in results:
                ax = axes[1, 0]
                returns = results['online'].get('returns_history', [])
                if returns:
                    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
                    ax.plot(cumulative_returns, color='green', linewidth=2)
                    ax.set_title('累积收益率')
                    ax.set_xlabel('时间步')
                    ax.set_ylabel('累积收益率')
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2%}'))
                    ax.grid(True, alpha=0.3)
                
                # 采样比例演化
                ax = axes[1, 1]
                rho_history = results['online'].get('rho_evolution', [])
                if rho_history:
                    ax.plot(rho_history, color='purple', linewidth=2)
                    ax.set_title('采样比例演化')
                    ax.set_xlabel('时间步')
                    ax.set_ylabel('在线采样比例 ρ')
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'training_process_{exp_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance_comparison(self, output_dir: str):
        """绘制性能对比图"""
        if 'experiment_comparison' not in self.analysis_results:
            return
        
        comparison = self.analysis_results['experiment_comparison']
        performance_data = comparison.get('performance_comparison', {})
        
        if not performance_data:
            return
        
        # 准备数据
        metrics = list(performance_data.keys())
        experiments = list(self.training_results.keys())
        
        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('实验性能对比', fontsize=16)
        
        for i, metric in enumerate(metrics[:4]):  # 最多显示4个指标
            ax = axes[i//2, i%2]
            
            values = []
            labels = []
            for exp_name in experiments:
                if exp_name in performance_data[metric]:
                    values.append(performance_data[metric][exp_name])
                    labels.append(exp_name)
            
            if values:
                bars = ax.bar(labels, values, alpha=0.7)
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('值')
                
                # 添加数值标签
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.4f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_return_scatter(self, output_dir: str):
        """绘制风险收益散点图"""
        if 'risk_return_profile' not in self.analysis_results:
            return
        
        risk_return_data = self.analysis_results['risk_return_profile']
        
        # 准备数据
        returns = []
        risks = []
        labels = []
        
        for exp_name, analysis in risk_return_data.items():
            if 'mean_return' in analysis and 'return_volatility' in analysis:
                returns.append(analysis['mean_return'])
                risks.append(analysis['return_volatility'])
                labels.append(exp_name)
        
        if not returns:
            return
        
        # 创建散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(risks, returns, s=100, alpha=0.7, c=range(len(labels)), cmap='viridis')
        
        # 添加标签
        for i, label in enumerate(labels):
            plt.annotate(label, (risks[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('风险 (收益率标准差)')
        plt.ylabel('收益 (平均收益率)')
        plt.title('风险-收益散点图')
        plt.grid(True, alpha=0.3)
        
        # 添加有效前沿参考线
        if len(returns) > 1:
            # 简单的有效前沿近似
            sorted_indices = np.argsort(risks)
            sorted_risks = np.array(risks)[sorted_indices]
            sorted_returns = np.array(returns)[sorted_indices]
            plt.plot(sorted_risks, sorted_returns, '--', alpha=0.5, label='参考线')
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'risk_return_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_drift(self, output_dir: str):
        """绘制分布漂移分析图"""
        if 'distribution_drift' not in self.analysis_results:
            return
        
        drift_data = self.analysis_results['distribution_drift']
        
        for exp_name, analysis in drift_data.items():
            if exp_name not in self.training_results or 'online' not in self.training_results[exp_name]:
                continue
            
            online_results = self.training_results[exp_name]['online']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'分布漂移分析 - {exp_name}', fontsize=16)
            
            # KL散度历史
            ax = axes[0, 0]
            kl_history = online_results.get('kl_divergence_history', [])
            if kl_history:
                ax.plot(kl_history, color='red', linewidth=2)
                ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='漂移阈值')
                ax.set_title('KL散度演化')
                ax.set_xlabel('时间步')
                ax.set_ylabel('KL散度')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 采样比例演化
            ax = axes[0, 1]
            rho_history = online_results.get('rho_evolution', [])
            if rho_history:
                ax.plot(rho_history, color='blue', linewidth=2)
                ax.set_title('采样比例演化')
                ax.set_xlabel('时间步')
                ax.set_ylabel('在线采样比例 ρ')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
            
            # 收益率分布
            ax = axes[1, 0]
            returns = online_results.get('returns_history', [])
            if returns:
                ax.hist(returns, bins=30, alpha=0.7, color='green', density=True)
                ax.axvline(np.mean(returns), color='red', linestyle='--', label=f'均值: {np.mean(returns):.4f}')
                ax.set_title('收益率分布')
                ax.set_xlabel('收益率')
                ax.set_ylabel('密度')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 漂移事件时间线
            ax = axes[1, 1]
            drift_events = online_results.get('drift_event_times', [])
            if drift_events:
                ax.scatter(drift_events, [1]*len(drift_events), color='red', s=50, alpha=0.7)
                ax.set_title('漂移事件时间线')
                ax.set_xlabel('时间步')
                ax.set_ylabel('漂移事件')
                ax.set_ylim(0.5, 1.5)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'distribution_drift_{exp_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_interactive_dashboard(self, output_dir: str):
        """创建交互式仪表板"""
        # 创建Plotly仪表板
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('训练损失', '累积收益', '风险指标', '采样比例', 'KL散度', '性能对比'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (exp_name, results) in enumerate(self.training_results.items()):
            color = colors[i % len(colors)]
            
            # 训练损失
            if 'offline' in results:
                loss_history = results['offline'].get('loss_history', [])
                if loss_history:
                    fig.add_trace(
                        go.Scatter(x=list(range(len(loss_history))), y=loss_history,
                                 name=f'{exp_name} - 离线损失', line=dict(color=color)),
                        row=1, col=1
                    )
            
            # 累积收益
            if 'online' in results:
                returns = results['online'].get('returns_history', [])
                if returns:
                    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
                    fig.add_trace(
                        go.Scatter(x=list(range(len(cumulative_returns))), y=cumulative_returns,
                                 name=f'{exp_name} - 累积收益', line=dict(color=color)),
                        row=1, col=2
                    )
                
                # 采样比例
                rho_history = results['online'].get('rho_evolution', [])
                if rho_history:
                    fig.add_trace(
                        go.Scatter(x=list(range(len(rho_history))), y=rho_history,
                                 name=f'{exp_name} - 采样比例', line=dict(color=color)),
                        row=2, col=2
                    )
                
                # KL散度
                kl_history = results['online'].get('kl_divergence_history', [])
                if kl_history:
                    fig.add_trace(
                        go.Scatter(x=list(range(len(kl_history))), y=kl_history,
                                 name=f'{exp_name} - KL散度', line=dict(color=color)),
                        row=3, col=1
                    )
        
        # 性能对比柱状图
        if 'experiment_comparison' in self.analysis_results:
            performance_data = self.analysis_results['experiment_comparison'].get('performance_comparison', {})
            if 'final_return' in performance_data:
                exp_names = list(performance_data['final_return'].keys())
                returns = list(performance_data['final_return'].values())
                
                fig.add_trace(
                    go.Bar(x=exp_names, y=returns, name='最终收益率'),
                    row=3, col=2
                )
        
        # 更新布局
        fig.update_layout(
            height=900,
            title_text="O2O训练结果交互式仪表板",
            showlegend=True
        )
        
        # 保存交互式图表
        dashboard_file = os.path.join(output_dir, 'interactive_dashboard.html')
        pyo.plot(fig, filename=dashboard_file, auto_open=False)
        
        self.logger.info(f"交互式仪表板已保存: {dashboard_file}")
    
    def generate_report(self):
        """生成分析报告"""
        self.logger.info("生成分析报告...")
        
        # 创建报告内容
        report_content = self._create_analysis_report()
        
        # 保存报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.output_dir, f'o2o_analysis_report_{timestamp}.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"分析报告已保存: {report_file}")
        return report_file
    
    def _create_analysis_report(self) -> str:
        """创建分析报告内容"""
        report = f"""# O2O强化学习训练结果分析报告

## 报告概览

- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **结果目录**: {self.results_dir}
- **分析实验数量**: {len(self.training_results)}

## 实验列表

"""
        
        for i, exp_name in enumerate(self.training_results.keys(), 1):
            report += f"{i}. {exp_name}\n"
        
        report += "\n## 训练过程分析\n\n"
        
        if 'training_process' in self.analysis_results:
            process_analysis = self.analysis_results['training_process']
            
            for exp_name, analysis in process_analysis.items():
                report += f"### {exp_name}\n\n"
                
                # 离线预训练分析
                if 'offline' in analysis['stages']:
                    offline = analysis['stages']['offline']
                    report += f"**离线预训练阶段**:\n"
                    report += f"- 训练效率: {offline.get('training_efficiency', 0):.3f}\n"
                    report += f"- 损失下降: {offline.get('loss_reduction', 0):.3f}\n"
                    report += f"- 损失稳定性: {offline.get('loss_stability', 0):.6f}\n\n"
                
                # 热身微调分析
                if 'warmup' in analysis['stages']:
                    warmup = analysis['stages']['warmup']
                    report += f"**热身微调阶段**:\n"
                    report += f"- Critic改进: {warmup.get('critic_improvement', 0):.6f}\n"
                    report += f"- 适应速度: {warmup.get('adaptation_speed', 0):.3f}\n"
                    report += f"- 收敛状态: {warmup.get('convergence_achieved', False)}\n\n"
                
                # 在线学习分析
                if 'online' in analysis['stages']:
                    online = analysis['stages']['online']
                    report += f"**在线学习阶段**:\n"
                    report += f"- 最终收益率: {online.get('final_return', 0):.4f}\n"
                    report += f"- 夏普比率: {online.get('sharpe_ratio', 0):.4f}\n"
                    report += f"- 最大回撤: {online.get('max_drawdown', 0):.4f}\n"
                    report += f"- CVaR风险: {online.get('cvar_risk', 0):.4f}\n"
                    report += f"- 漂移检测次数: {online.get('drift_detections', 0)}\n\n"
        
        # 风险收益分析
        if 'risk_return_profile' in self.analysis_results:
            report += "## 风险收益分析\n\n"
            risk_return_data = self.analysis_results['risk_return_profile']
            
            report += "| 实验名称 | 平均收益 | 收益波动 | 夏普比率 | VaR(95%) | CVaR(95%) | 偏度 | 峰度 |\n"
            report += "|---------|---------|---------|---------|----------|-----------|------|------|\n"
            
            for exp_name, analysis in risk_return_data.items():
                report += f"| {exp_name} | "
                report += f"{analysis.get('mean_return', 0):.4f} | "
                report += f"{analysis.get('return_volatility', 0):.4f} | "
                report += f"{analysis.get('sharpe_ratio', 0):.4f} | "
                report += f"{analysis.get('var_95', 0):.4f} | "
                report += f"{analysis.get('cvar_95', 0):.4f} | "
                report += f"{analysis.get('skewness', 0):.4f} | "
                report += f"{analysis.get('kurtosis', 0):.4f} |\n"
            
            report += "\n"
        
        # 分布漂移分析
        if 'distribution_drift' in self.analysis_results:
            report += "## 分布漂移分析\n\n"
            drift_data = self.analysis_results['distribution_drift']
            
            for exp_name, analysis in drift_data.items():
                report += f"### {exp_name}\n\n"
                report += f"- 平均KL散度: {analysis.get('mean_kl_divergence', 0):.6f}\n"
                report += f"- 最大KL散度: {analysis.get('max_kl_divergence', 0):.6f}\n"
                report += f"- 漂移频率: {analysis.get('drift_frequency', 0):.3f}\n"
                report += f"- 最终采样比例: {analysis.get('final_rho', 0):.3f}\n"
                report += f"- 漂移响应效果: {analysis.get('drift_response_effectiveness', 0):.3f}\n\n"
        
        # 实验对比
        if 'experiment_comparison' in self.analysis_results:
            report += "## 实验对比\n\n"
            comparison = self.analysis_results['experiment_comparison']
            
            if 'performance_comparison' in comparison:
                report += "### 性能对比\n\n"
                performance_data = comparison['performance_comparison']
                
                for metric, values in performance_data.items():
                    if values:
                        best_exp = max(values.items(), key=lambda x: x[1])
                        worst_exp = min(values.items(), key=lambda x: x[1])
                        report += f"**{metric.replace('_', ' ').title()}**:\n"
                        report += f"- 最佳: {best_exp[0]} ({best_exp[1]:.4f})\n"
                        report += f"- 最差: {worst_exp[0]} ({worst_exp[1]:.4f})\n\n"
        
        report += "## 总结与建议\n\n"
        report += self._generate_recommendations()
        
        report += f"\n---\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report
    
    def _generate_recommendations(self) -> str:
        """生成建议"""
        recommendations = []
        
        # 基于分析结果生成建议
        if 'training_process' in self.analysis_results:
            process_analysis = self.analysis_results['training_process']
            
            # 检查训练效率
            avg_efficiency = np.mean([
                analysis['stages'].get('offline', {}).get('training_efficiency', 0)
                for analysis in process_analysis.values()
            ])
            
            if avg_efficiency < 0.7:
                recommendations.append("- 建议优化离线预训练参数，提高训练效率")
            
            # 检查收敛性
            convergence_rates = [
                analysis['convergence'].get('overall_convergence_score', 0)
                for analysis in process_analysis.values()
            ]
            
            if np.mean(convergence_rates) < 0.8:
                recommendations.append("- 建议调整学习率和收敛阈值，改善收敛性")
        
        # 基于风险收益分析生成建议
        if 'risk_return_profile' in self.analysis_results:
            risk_return_data = self.analysis_results['risk_return_profile']
            
            sharpe_ratios = [
                analysis.get('sharpe_ratio', 0)
                for analysis in risk_return_data.values()
            ]
            
            if np.mean(sharpe_ratios) < 1.0:
                recommendations.append("- 建议优化风险控制参数，提高风险调整收益")
        
        # 基于漂移分析生成建议
        if 'distribution_drift' in self.analysis_results:
            drift_data = self.analysis_results['distribution_drift']
            
            drift_frequencies = [
                analysis.get('drift_frequency', 0)
                for analysis in drift_data.values()
            ]
            
            if np.mean(drift_frequencies) > 0.3:
                recommendations.append("- 建议调整漂移检测阈值，减少误报")
            elif np.mean(drift_frequencies) < 0.1:
                recommendations.append("- 建议降低漂移检测阈值，提高敏感性")
        
        if not recommendations:
            recommendations.append("- 当前配置表现良好，建议继续使用")
        
        return "\n".join(recommendations)
    
    def run_full_analysis(self):
        """运行完整分析流程"""
        self.logger.info("开始完整分析流程...")
        
        try:
            # 1. 加载结果
            self.load_results()
            
            # 2. 分析训练过程
            self.analyze_training_process()
            
            # 3. 对比实验
            self.compare_experiments()
            
            # 4. 分析风险收益
            self.analyze_risk_return_profile()
            
            # 5. 分析分布漂移
            self.analyze_distribution_drift()
            
            # 6. 生成可视化
            self.generate_visualizations()
            
            # 7. 生成报告
            report_file = self.generate_report()
            
            self.logger.info("完整分析流程完成")
            return report_file
            
        except Exception as e:
            self.logger.error(f"分析流程失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='O2O训练结果分析工具')
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results',
        help='结果目录路径'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='analysis',
        help='输出目录路径'
    )
    parser.add_argument(
        '--analysis-type', 
        type=str, 
        choices=['full', 'process', 'comparison', 'risk', 'drift'],
        default='full',
        help='分析类型'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建分析器
        analyzer = O2OResultsAnalyzer(
            results_dir=args.results_dir,
            output_dir=args.output_dir
        )
        
        # 根据分析类型执行相应分析
        if args.analysis_type == 'full':
            analyzer.run_full_analysis()
        elif args.analysis_type == 'process':
            analyzer.load_results()
            analyzer.analyze_training_process()
            analyzer.generate_report()
        elif args.analysis_type == 'comparison':
            analyzer.load_results()
            analyzer.compare_experiments()
            analyzer.generate_report()
        elif args.analysis_type == 'risk':
            analyzer.load_results()
            analyzer.analyze_risk_return_profile()
            analyzer.generate_report()
        elif args.analysis_type == 'drift':
            analyzer.load_results()
            analyzer.analyze_distribution_drift()
            analyzer.generate_report()
        
        print("分析完成！")
        
    except Exception as e:
        print(f"分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()