"""
配置管理模块
"""

from .config_manager import ConfigManager
from .default_config import get_default_config

__all__ = ['ConfigManager', 'get_default_config']