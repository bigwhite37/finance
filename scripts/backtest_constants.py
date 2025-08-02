#!/usr/bin/env python3
"""
回测脚本的配置常量
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class BacktestConstants:
    """回测脚本的配置常量"""
    
    # 金融计算常量
    DEFAULT_TRADING_DAYS_PER_YEAR: int = 252
    DEFAULT_RISK_FREE_RATE: float = 0.03
    
    # 默认基准指数（如果配置中未指定）
    DEFAULT_BENCHMARK_SYMBOLS: List[str] = None
    
    # 默认股票池（如果配置中未指定）
    DEFAULT_STOCK_POOL: List[str] = None
    
    # 默认交易成本
    DEFAULT_COMMISSION_RATE: float = 0.001
    DEFAULT_STAMP_TAX_RATE: float = 0.001
    DEFAULT_MAX_POSITION_SIZE: float = 0.1
    
    # 日志和进度相关
    DEFAULT_PROGRESS_LOG_INTERVAL: int = 50
    
    # 可视化配置
    CHART_HEIGHT: int = 800
    CHART_COLORS: List[str] = None
    
    # 基准名称映射
    BENCHMARK_NAME_MAP: Dict[str, str] = None
    
    # 输出文件名
    RESULTS_JSON_FILENAME: str = "backtest_results.json"
    CHART_HTML_FILENAME: str = "performance_chart.html"
    
    def __post_init__(self):
        """后处理默认值"""
        if self.DEFAULT_BENCHMARK_SYMBOLS is None:
            self.DEFAULT_BENCHMARK_SYMBOLS = ['000300.SH', '000905.SH', '000852.SH']
        
        if self.DEFAULT_STOCK_POOL is None:
            self.DEFAULT_STOCK_POOL = ['600519.SH', '600036.SH', '601318.SH']
        
        if self.CHART_COLORS is None:
            self.CHART_COLORS = ['red', 'green', 'orange', 'purple']
        
        if self.BENCHMARK_NAME_MAP is None:
            self.BENCHMARK_NAME_MAP = {
                '000300.SH': '沪深300',
                '000905.SH': '中证500',
                '000852.SH': '中证1000'
            }


# 全局配置实例
BACKTEST_CONFIG = BacktestConstants()


def get_config_value(config_dict: Dict[str, Any], key: str, default_value: Any) -> Any:
    """
    安全地从配置字典中获取值，如果不存在则使用默认值
    
    Args:
        config_dict: 配置字典
        key: 配置键（支持嵌套键，如 'backtest.risk_free_rate'）
        default_value: 默认值
        
    Returns:
        配置值或默认值
    """
    keys = key.split('.')
    current_dict = config_dict
    
    try:
        for k in keys:
            current_dict = current_dict[k]
        return current_dict
    except (KeyError, TypeError):
        return default_value