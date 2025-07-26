#!/usr/bin/env python3
"""
O2O训练性能监控脚本

实时监控O2O训练过程中的关键指标，包括：
- 训练进度和损失变化
- 采样比例演化
- 分布漂移检测
- 风险指标监控
- 资源使用情况

使用方法:
    python examples/o2o_performance_monitor.py --log-dir logs --update-interval 10
    
作者: O2O RL Team
日期: 2024
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.metrics import PerformanceMetrics


class O2OPerformanceMonitor:
    """O2O性能监控器"""
    
    def __init__(self, log_dir: str = 'logs', update_interval: int = 10):
        """
        初始化性能监控器
        
        Args:
            log_dir: 日志目录
            update_interval: 更新间隔（秒）
        """
        self.log_dir = log_dir
        self.update_interval = update_interval
        self.logger = setup_logger('o2o_monitor', level='INFO')
        
        # 监控数据
        self.training_metrics = {
            'timestamps': [],
            'offline_loss': [],
            'warmup_loss': [],
            'online_return': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'cvar_risk': [],
            'rho_values': [],
            'kl_divergence': [],
            'drift_events': [],
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_usage': []
        }
        
        # 当前训练状态
        self.current_stage = 'offline'  # offline, warmup, online
        self.training_start_time = None
        self.last_update_time = None
        
        # GUI组件
        self.root = None
        self.fig = None
        self.axes = None
        self.canvas = None
        
    def start_monitoring(self, gui_mode: bool = True):
        """开始监控"""
        self.logger.info("开始O2O训练性能监控...")
        self.training_start_time = datetime.now()
        
        if gui_mode:
            self._start_gui_monitoring()
        else:
            self._start_console_monitoring()
    
    def _start_gui_monitoring(self):
        """启动GUI监控界面"""
        self.root = tk.Tk()
        self.root.title("O2O训练性能监控")
        self.root.geometry("1200x800")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建状态信息框架
        status_frame = ttk.LabelFrame(main_frame, text="训练状态", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 状态标签
        self.status_labels = {}
        status_info = [
            ('当前阶段', 'current_stage'),
            ('训练时间', 'training_time'),
            ('最后更新', 'last_update'),
            ('内存使用', 'memory_usage'),
            ('CPU使用', 'cpu_usage')
        ]
        
        for i, (label_text, key) in enumerate(status_info):
            ttk.Label(status_frame, text=f"{label_text}:").grid(
                row=i//3, column=(i%3)*2, sticky=tk.W, padx=(0, 5)
            )
            self.status_labels[key] = ttk.Label(status_frame, text="N/A")
            self.status_labels[key].grid(
                row=i//3, column=(i%3)*2+1, sticky=tk.W, padx=(0, 20)
            )
        
        # 创建图表框架
        chart_frame = ttk.LabelFrame(main_frame, text="性能图表", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图表
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 8))
        self.fig.tight_layout(pad=3.0)
        
        # 设置子图标题
        subplot_titles = [
            '训练损失', '收益率曲线', '风险指标',
            '采样比例演化', 'KL散度', '资源使用'
        ]
        
        for ax, title in zip(self.axes.flat, subplot_titles):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        # 嵌入matplotlib到tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 启动定时更新
        self._schedule_update()
        
        # 启动GUI主循环
        self.root.mainloop()
    
    def _start_console_monitoring(self):
        """启动控制台监控"""
        try:
            while True:
                self._update_metrics()
                self._print_console_status()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            self.logger.info("监控被用户中断")
    
    def _schedule_update(self):
        """调度GUI更新"""
        self._update_metrics()
        self._update_gui()
        self.root.after(self.update_interval * 1000, self._schedule_update)
    
    def _update_metrics(self):
        """更新监控指标"""
        current_time = datetime.now()
        
        # 读取训练日志
        training_data = self._read_training_logs()
        
        # 更新训练指标
        if training_data:
            self._process_training_data(training_data, current_time)
        
        # 更新系统资源指标
        self._update_system_metrics(current_time)
        
        self.last_update_time = current_time
    
    def _read_training_logs(self) -> Optional[Dict]:
        """读取训练日志文件"""
        try:
            # 查找最新的日志文件
            log_files = []
            if os.path.exists(self.log_dir):
                for file in os.listdir(self.log_dir):
                    if file.startswith('o2o_training_') and file.endswith('.log'):
                        log_path = os.path.join(self.log_dir, file)
                        log_files.append((log_path, os.path.getmtime(log_path)))
            
            if not log_files:
                return None
            
            # 选择最新的日志文件
            latest_log = max(log_files, key=lambda x: x[1])[0]
            
            # 解析日志文件
            return self._parse_log_file(latest_log)
            
        except Exception as e:
            self.logger.warning(f"读取训练日志失败: {e}")
            return None
    
    def _parse_log_file(self, log_file: str) -> Dict:
        """解析日志文件"""
        training_data = {
            'stage': 'offline',
            'metrics': {}
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 解析日志行
            for line in lines[-100:]:  # 只处理最近100行
                line = line.strip()
                
                # 检测训练阶段
                if '开始离线预训练阶段' in line:
                    training_data['stage'] = 'offline'
                elif '开始热身微调阶段' in line:
                    training_data['stage'] = 'warmup'
                elif '开始在线学习阶段' in line:
                    training_data['stage'] = 'online'
                
                # 提取指标
                self._extract_metrics_from_line(line, training_data['metrics'])
            
            return training_data
            
        except Exception as e:
            self.logger.warning(f"解析日志文件失败: {e}")
            return {}
    
    def _extract_metrics_from_line(self, line: str, metrics: Dict):
        """从日志行提取指标"""
        # 定义指标模式
        patterns = {
            'loss': r'损失[：:]\s*([\d\.-]+)',
            'return': r'收益率[：:]\s*([\d\.-]+)',
            'sharpe': r'夏普[比率]*[：:]\s*([\d\.-]+)',
            'drawdown': r'回撤[：:]\s*([\d\.-]+)',
            'cvar': r'CVaR[：:]\s*([\d\.-]+)',
            'rho': r'采样比例[：:]\s*([\d\.-]+)',
            'kl': r'KL散度[：:]\s*([\d\.-]+)'
        }
        
        import re
        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                try:
                    metrics[key] = float(match.group(1))
                except ValueError:
                    pass
    
    def _process_training_data(self, training_data: Dict, timestamp: datetime):
        """处理训练数据"""
        self.current_stage = training_data.get('stage', 'offline')
        metrics = training_data.get('metrics', {})
        
        # 更新时间序列数据
        self.training_metrics['timestamps'].append(timestamp)
        
        # 更新各项指标
        self.training_metrics['offline_loss'].append(
            metrics.get('loss', np.nan) if self.current_stage == 'offline' else np.nan
        )
        self.training_metrics['warmup_loss'].append(
            metrics.get('loss', np.nan) if self.current_stage == 'warmup' else np.nan
        )
        self.training_metrics['online_return'].append(
            metrics.get('return', np.nan) if self.current_stage == 'online' else np.nan
        )
        self.training_metrics['sharpe_ratio'].append(metrics.get('sharpe', np.nan))
        self.training_metrics['max_drawdown'].append(metrics.get('drawdown', np.nan))
        self.training_metrics['cvar_risk'].append(metrics.get('cvar', np.nan))
        self.training_metrics['rho_values'].append(metrics.get('rho', np.nan))
        self.training_metrics['kl_divergence'].append(metrics.get('kl', np.nan))
        
        # 检测漂移事件
        if metrics.get('kl', 0) > 0.1:  # KL散度阈值
            self.training_metrics['drift_events'].append(timestamp)
        
        # 限制数据长度
        max_points = 1000
        for key in self.training_metrics:
            if isinstance(self.training_metrics[key], list) and \
               len(self.training_metrics[key]) > max_points:
                self.training_metrics[key] = self.training_metrics[key][-max_points:]
    
    def _update_system_metrics(self, timestamp: datetime):
        """更新系统资源指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.training_metrics['cpu_usage'].append(cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.training_metrics['memory_usage'].append(memory_percent)
        
        # GPU使用率（如果可用）
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
                self.training_metrics['gpu_usage'].append(gpu_percent)
            else:
                self.training_metrics['gpu_usage'].append(0)
        except ImportError:
            self.training_metrics['gpu_usage'].append(0)
    
    def _update_gui(self):
        """更新GUI界面"""
        if not self.root:
            return
        
        # 更新状态标签
        self._update_status_labels()
        
        # 更新图表
        self._update_charts()
        
        # 刷新画布
        self.canvas.draw()
    
    def _update_status_labels(self):
        """更新状态标签"""
        if not self.status_labels:
            return
        
        # 当前阶段
        stage_names = {
            'offline': '离线预训练',
            'warmup': '热身微调',
            'online': '在线学习'
        }
        self.status_labels['current_stage'].config(
            text=stage_names.get(self.current_stage, '未知')
        )
        
        # 训练时间
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            hours, remainder = divmod(elapsed.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            self.status_labels['training_time'].config(text=time_str)
        
        # 最后更新时间
        if self.last_update_time:
            update_str = self.last_update_time.strftime('%H:%M:%S')
            self.status_labels['last_update'].config(text=update_str)
        
        # 资源使用
        if self.training_metrics['memory_usage']:
            memory_str = f"{self.training_metrics['memory_usage'][-1]:.1f}%"
            self.status_labels['memory_usage'].config(text=memory_str)
        
        if self.training_metrics['cpu_usage']:
            cpu_str = f"{self.training_metrics['cpu_usage'][-1]:.1f}%"
            self.status_labels['cpu_usage'].config(text=cpu_str)
    
    def _update_charts(self):
        """更新图表"""
        if not self.axes:
            return
        
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()
            ax.grid(True, alpha=0.3)
        
        timestamps = self.training_metrics['timestamps']
        if not timestamps:
            return
        
        # 转换时间戳为相对时间（分钟）
        if self.training_start_time:
            time_minutes = [(t - self.training_start_time).total_seconds() / 60 
                           for t in timestamps]
        else:
            time_minutes = list(range(len(timestamps)))
        
        # 1. 训练损失
        ax = self.axes[0, 0]
        ax.set_title('训练损失')
        
        offline_loss = [x for x in self.training_metrics['offline_loss'] if not np.isnan(x)]
        warmup_loss = [x for x in self.training_metrics['warmup_loss'] if not np.isnan(x)]
        
        if offline_loss:
            ax.plot(time_minutes[:len(offline_loss)], offline_loss, 
                   label='离线预训练', color='blue', alpha=0.7)
        if warmup_loss:
            warmup_start = len(offline_loss)
            ax.plot(time_minutes[warmup_start:warmup_start+len(warmup_loss)], 
                   warmup_loss, label='热身微调', color='orange', alpha=0.7)
        
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('损失值')
        ax.legend()
        
        # 2. 收益率曲线
        ax = self.axes[0, 1]
        ax.set_title('收益率曲线')
        
        returns = [x for x in self.training_metrics['online_return'] if not np.isnan(x)]
        if returns:
            # 计算累积收益
            cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
            online_start = len(timestamps) - len(returns)
            ax.plot(time_minutes[online_start:], cumulative_returns, 
                   color='green', linewidth=2)
        
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('累积收益率')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2%}'))
        
        # 3. 风险指标
        ax = self.axes[0, 2]
        ax.set_title('风险指标')
        
        sharpe = [x for x in self.training_metrics['sharpe_ratio'] if not np.isnan(x)]
        drawdown = [x for x in self.training_metrics['max_drawdown'] if not np.isnan(x)]
        
        if sharpe:
            ax.plot(time_minutes[-len(sharpe):], sharpe, 
                   label='夏普比率', color='purple', alpha=0.7)
        if drawdown:
            ax2 = ax.twinx()
            ax2.plot(time_minutes[-len(drawdown):], drawdown, 
                    label='最大回撤', color='red', alpha=0.7)
            ax2.set_ylabel('最大回撤')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2%}'))
        
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('夏普比率')
        ax.legend(loc='upper left')
        
        # 4. 采样比例演化
        ax = self.axes[1, 0]
        ax.set_title('采样比例演化')
        
        rho_values = [x for x in self.training_metrics['rho_values'] if not np.isnan(x)]
        if rho_values:
            ax.plot(time_minutes[-len(rho_values):], rho_values, 
                   color='brown', linewidth=2)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50%基准')
        
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('在线采样比例 ρ')
        ax.set_ylim(0, 1)
        ax.legend()
        
        # 5. KL散度
        ax = self.axes[1, 1]
        ax.set_title('KL散度')
        
        kl_values = [x for x in self.training_metrics['kl_divergence'] if not np.isnan(x)]
        if kl_values:
            ax.plot(time_minutes[-len(kl_values):], kl_values, 
                   color='orange', linewidth=2)
            ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='漂移阈值')
        
        # 标记漂移事件
        for drift_time in self.training_metrics['drift_events']:
            if self.training_start_time:
                drift_minutes = (drift_time - self.training_start_time).total_seconds() / 60
                ax.axvline(x=drift_minutes, color='red', alpha=0.3)
        
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('KL散度')
        ax.legend()
        
        # 6. 资源使用
        ax = self.axes[1, 2]
        ax.set_title('资源使用')
        
        if self.training_metrics['cpu_usage']:
            ax.plot(time_minutes, self.training_metrics['cpu_usage'], 
                   label='CPU', color='blue', alpha=0.7)
        if self.training_metrics['memory_usage']:
            ax.plot(time_minutes, self.training_metrics['memory_usage'], 
                   label='内存', color='green', alpha=0.7)
        if self.training_metrics['gpu_usage'] and any(self.training_metrics['gpu_usage']):
            ax.plot(time_minutes, self.training_metrics['gpu_usage'], 
                   label='GPU', color='red', alpha=0.7)
        
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('使用率 (%)')
        ax.set_ylim(0, 100)
        ax.legend()
    
    def _print_console_status(self):
        """打印控制台状态"""
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 60)
        print("O2O训练性能监控")
        print("=" * 60)
        
        # 基本信息
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            print(f"训练时间: {elapsed}")
        
        print(f"当前阶段: {self.current_stage}")
        
        if self.last_update_time:
            print(f"最后更新: {self.last_update_time.strftime('%H:%M:%S')}")
        
        print("-" * 60)
        
        # 训练指标
        if self.training_metrics['timestamps']:
            latest_idx = -1
            
            if self.current_stage == 'offline' and self.training_metrics['offline_loss']:
                loss = self.training_metrics['offline_loss'][latest_idx]
                if not np.isnan(loss):
                    print(f"离线训练损失: {loss:.6f}")
            
            elif self.current_stage == 'warmup' and self.training_metrics['warmup_loss']:
                loss = self.training_metrics['warmup_loss'][latest_idx]
                if not np.isnan(loss):
                    print(f"热身微调损失: {loss:.6f}")
            
            elif self.current_stage == 'online':
                if self.training_metrics['online_return']:
                    ret = self.training_metrics['online_return'][latest_idx]
                    if not np.isnan(ret):
                        print(f"在线收益率: {ret:.4f}")
                
                if self.training_metrics['sharpe_ratio']:
                    sharpe = self.training_metrics['sharpe_ratio'][latest_idx]
                    if not np.isnan(sharpe):
                        print(f"夏普比率: {sharpe:.4f}")
                
                if self.training_metrics['rho_values']:
                    rho = self.training_metrics['rho_values'][latest_idx]
                    if not np.isnan(rho):
                        print(f"采样比例: {rho:.3f}")
        
        print("-" * 60)
        
        # 系统资源
        if self.training_metrics['cpu_usage']:
            cpu = self.training_metrics['cpu_usage'][-1]
            print(f"CPU使用率: {cpu:.1f}%")
        
        if self.training_metrics['memory_usage']:
            memory = self.training_metrics['memory_usage'][-1]
            print(f"内存使用率: {memory:.1f}%")
        
        if self.training_metrics['gpu_usage'] and self.training_metrics['gpu_usage'][-1] > 0:
            gpu = self.training_metrics['gpu_usage'][-1]
            print(f"GPU使用率: {gpu:.1f}%")
        
        print("=" * 60)
        print("按 Ctrl+C 停止监控")
    
    def save_monitoring_data(self, filepath: str):
        """保存监控数据"""
        monitoring_data = {
            'training_start_time': self.training_start_time.isoformat() if self.training_start_time else None,
            'current_stage': self.current_stage,
            'metrics': {}
        }
        
        # 转换时间戳为字符串
        for key, values in self.training_metrics.items():
            if key == 'timestamps':
                monitoring_data['metrics'][key] = [t.isoformat() for t in values]
            elif key == 'drift_events':
                monitoring_data['metrics'][key] = [t.isoformat() for t in values]
            else:
                monitoring_data['metrics'][key] = values
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(monitoring_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"监控数据已保存: {filepath}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='O2O训练性能监控')
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='logs',
        help='日志目录路径'
    )
    parser.add_argument(
        '--update-interval', 
        type=int, 
        default=10,
        help='更新间隔（秒）'
    )
    parser.add_argument(
        '--console-mode', 
        action='store_true',
        help='使用控制台模式（无GUI）'
    )
    parser.add_argument(
        '--save-data', 
        type=str,
        help='保存监控数据到文件'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建监控器
        monitor = O2OPerformanceMonitor(
            log_dir=args.log_dir,
            update_interval=args.update_interval
        )
        
        # 开始监控
        monitor.start_monitoring(gui_mode=not args.console_mode)
        
        # 保存监控数据（如果指定）
        if args.save_data:
            monitor.save_monitoring_data(args.save_data)
        
    except KeyboardInterrupt:
        print("\n监控被用户中断")
    except Exception as e:
        print(f"监控失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()