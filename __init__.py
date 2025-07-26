"""
A股强化学习量化交易系统
基于qlib的CVaR-PPO交易智能体
"""

__version__ = "1.0.0"
__author__ = "RL Trading System"

from .data import DataManager
from .factors import FactorEngine
from .rl_agent import TradingEnvironment, CVaRPPOAgent
from .risk_control import RiskController
from .backtest import BacktestEngine

__all__ = [
    'DataManager',
    'FactorEngine', 
    'TradingEnvironment',
    'CVaRPPOAgent',
    'RiskController',
    'BacktestEngine'
]