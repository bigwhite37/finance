"""
日志配置模块

使用loguru进行结构化日志记录
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "1 day",
    retention: str = "30 days",
    compression: str = "gz",
    format_string: Optional[str] = None,
) -> None:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        rotation: 日志轮转策略
        retention: 日志保留时间
        compression: 压缩格式
        format_string: 自定义格式字符串
    """
    # 移除默认处理器
    logger.remove()
    
    # 默认格式
    if format_string is None:
        format_string = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        level=log_level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # 添加文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_path,
            level=log_level,
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=True,
            diagnose=True,
        )
    
    logger.info(f"日志系统初始化完成，级别: {log_level}")


def get_logger(name: str):
    """获取指定名称的logger"""
    return logger.bind(name=name)