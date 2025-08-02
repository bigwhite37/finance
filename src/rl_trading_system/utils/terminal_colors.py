#!/usr/bin/env python3
"""
终端着色工具
为训练和回测脚本提供彩色日志输出
"""

import sys
from typing import Optional


class TerminalColors:
    """终端颜色代码"""
    
    # 基础颜色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 亮色
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # 背景色
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # 样式
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'
    
    # 重置
    RESET = '\033[0m'
    
    @classmethod
    def is_terminal_color_supported(cls) -> bool:
        """检查终端是否支持颜色"""
        return (
            hasattr(sys.stdout, 'isatty') and 
            sys.stdout.isatty() and 
            'TERM' in sys.__dict__.get('environ', {})
        )


class ColorFormatter:
    """彩色格式化器"""
    
    def __init__(self, enable_color: Optional[bool] = None):
        """
        初始化颜色格式化器
        
        Args:
            enable_color: 是否启用颜色。None表示自动检测
        """
        if enable_color is None:
            self.enable_color = TerminalColors.is_terminal_color_supported()
        else:
            self.enable_color = enable_color
    
    def colorize(self, text: str, color: str, style: str = '') -> str:
        """
        给文本着色
        
        Args:
            text: 要着色的文本
            color: 颜色代码
            style: 样式代码
            
        Returns:
            着色后的文本
        """
        if not self.enable_color:
            return text
            
        return f"{style}{color}{text}{TerminalColors.RESET}"
    
    def success(self, text: str) -> str:
        """成功信息 - 绿色"""
        return self.colorize(text, TerminalColors.BRIGHT_GREEN, TerminalColors.BOLD)
    
    def warning(self, text: str) -> str:
        """警告信息 - 黄色"""
        return self.colorize(text, TerminalColors.BRIGHT_YELLOW, TerminalColors.BOLD)
    
    def error(self, text: str) -> str:
        """错误信息 - 红色"""
        return self.colorize(text, TerminalColors.BRIGHT_RED, TerminalColors.BOLD)
    
    def info(self, text: str) -> str:
        """信息 - 蓝色"""
        return self.colorize(text, TerminalColors.BRIGHT_BLUE)
    
    def highlight(self, text: str) -> str:
        """高亮 - 青色加粗"""
        return self.colorize(text, TerminalColors.BRIGHT_CYAN, TerminalColors.BOLD)
    
    def dim(self, text: str) -> str:
        """暗淡 - 灰色"""
        return self.colorize(text, TerminalColors.BRIGHT_BLACK)
    
    def path(self, text: str) -> str:
        """路径 - 洋红色"""
        return self.colorize(text, TerminalColors.BRIGHT_MAGENTA)
    
    def number(self, text: str) -> str:
        """数字 - 青色"""
        return self.colorize(text, TerminalColors.CYAN)
    
    def status_box(self, text: str, status: str = 'info') -> str:
        """状态框"""
        if status == 'success':
            return self.colorize(f" ✅ {text} ", TerminalColors.WHITE, TerminalColors.BG_GREEN)
        elif status == 'warning':
            return self.colorize(f" ⚠️  {text} ", TerminalColors.BLACK, TerminalColors.BG_YELLOW)
        elif status == 'error':
            return self.colorize(f" ❌ {text} ", TerminalColors.WHITE, TerminalColors.BG_RED)
        else:
            return self.colorize(f" ℹ️  {text} ", TerminalColors.WHITE, TerminalColors.BG_BLUE)


def print_banner(title: str, subtitle: str = '', formatter: Optional[ColorFormatter] = None):
    """打印标题横幅"""
    if formatter is None:
        formatter = ColorFormatter()
    
    # 计算横幅宽度
    width = max(len(title), len(subtitle)) + 10
    width = max(width, 60)
    
    print()
    print(formatter.highlight("=" * width))
    print(formatter.highlight(f"  {title.center(width-4)}  "))
    if subtitle:
        print(formatter.info(f"  {subtitle.center(width-4)}  "))
    print(formatter.highlight("=" * width))
    print()


def print_section(title: str, formatter: Optional[ColorFormatter] = None):
    """打印章节标题"""
    if formatter is None:
        formatter = ColorFormatter()
    
    print()
    print(formatter.info(f"{'─' * 50}"))
    print(formatter.highlight(f"📋 {title}"))
    print(formatter.info(f"{'─' * 50}"))


def print_model_recommendation(model_paths: dict, formatter: Optional[ColorFormatter] = None):
    """打印模型推荐信息"""
    if formatter is None:
        formatter = ColorFormatter()
    
    print()
    print(formatter.status_box("模型文件保存完成", "success"))
    print()
    
    print(formatter.highlight("🎯 推荐使用的模型："))
    print()
    
    # 推荐最佳模型
    if 'best_model' in model_paths and model_paths['best_model']:
        print(f"  {formatter.success('🥇 最佳性能模型')} (推荐用于生产环境)")
        print(f"     {formatter.path(model_paths['best_model'])}")
        print()
    
    # 最终模型
    if 'final_model' in model_paths and model_paths['final_model']:
        status = "🥈 最终训练模型" if 'best_model' in model_paths else "🎯 推荐模型"
        print(f"  {formatter.info(status)} (用于测试和评估)")
        print(f"     {formatter.path(model_paths['final_model'])}")
        print()
    
    print(formatter.highlight("📝 使用方法："))
    print(f"  {formatter.info('回测命令：')}")
    
    # 生成回测命令
    recommended_model = model_paths.get('best_model') or model_paths.get('final_model')
    if recommended_model:
        print(f"    {formatter.dim('python scripts/backtest.py --model-path')} {formatter.path(recommended_model)}")
    print()


def print_training_progress(episode: int, total_episodes: int, reward: float, 
                          length: int, formatter: Optional[ColorFormatter] = None):
    """打印训练进度"""
    if formatter is None:
        formatter = ColorFormatter()
    
    progress = episode / total_episodes * 100
    
    # 进度条
    bar_length = 30
    filled = int(bar_length * progress / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    # 格式化奖励值颜色
    if reward > 0:
        reward_str = formatter.success(f"{reward:8.2f}")
    elif reward < -50:
        reward_str = formatter.error(f"{reward:8.2f}")
    else:
        reward_str = formatter.warning(f"{reward:8.2f}")
    
    print(f"  Episode {formatter.number(f'{episode:4d}/{total_episodes}')} "
          f"[{formatter.info(bar)}] {formatter.number(f'{progress:5.1f}%')} "
          f"| Reward: {reward_str} | Length: {formatter.number(str(length))}")


def print_training_stats(stats: dict, formatter: Optional[ColorFormatter] = None):
    """打印训练统计信息"""
    if formatter is None:
        formatter = ColorFormatter()
    
    print_section("训练统计", formatter)
    
    # 核心指标
    core_metrics = ['mean_reward', 'std_reward', 'max_reward', 'mean_length']
    
    for key in core_metrics:
        if key in stats:
            value = stats[key]
            if 'reward' in key:
                if value > 0:
                    value_str = formatter.success(f"{value:8.2f}")
                elif value < -50:
                    value_str = formatter.error(f"{value:8.2f}")
                else:
                    value_str = formatter.warning(f"{value:8.2f}")
            else:
                value_str = formatter.number(f"{value:8.2f}")
            
            print(f"  {formatter.info(key.replace('_', ' ').title().ljust(15))}: {value_str}")
    
    # 其他指标
    other_metrics = [k for k in stats.keys() if k not in core_metrics]
    if other_metrics:
        print()
        print(formatter.dim("  其他指标:"))
        for key in other_metrics:
            value = stats[key]
            value_str = formatter.number(f"{value:8.2f}" if isinstance(value, (int, float)) else str(value))
            print(f"    {formatter.dim(key.replace('_', ' ').title().ljust(15))}: {value_str}")


def print_evaluation_results(stats: dict, formatter: Optional[ColorFormatter] = None):
    """打印评估结果"""
    if formatter is None:
        formatter = ColorFormatter()
    
    print_section("最终评估结果", formatter)
    
    # 评估指标
    eval_metrics = ['mean_reward', 'std_reward', 'min_reward', 'max_reward', 'mean_length']
    
    for key in eval_metrics:
        if key in stats:
            value = stats[key]
            if 'reward' in key:
                if value > 100:
                    value_str = formatter.success(f"{value:8.2f}")
                elif value < -100:
                    value_str = formatter.error(f"{value:8.2f}")
                else:
                    value_str = formatter.info(f"{value:8.2f}")
            else:
                value_str = formatter.number(f"{value:8.2f}")
            
            print(f"  {formatter.info(key.replace('_', ' ').title().ljust(15))}: {value_str}")


# 全局格式化器实例
_global_formatter = ColorFormatter()

def get_formatter() -> ColorFormatter:
    """获取全局格式化器"""
    return _global_formatter

def set_color_enabled(enabled: bool):
    """设置全局颜色启用状态"""
    global _global_formatter
    _global_formatter = ColorFormatter(enabled)