"""
数据接口使用示例
演示如何使用QlibDataInterface和AkshareDataInterface获取数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import datetime, timedelta
from src.rl_trading_system.data.qlib_interface import QlibDataInterface
from src.rl_trading_system.data.akshare_interface import AkshareDataInterface
from src.rl_trading_system.data.data_cache import get_global_cache
from src.rl_trading_system.data.data_quality import get_global_quality_checker

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_qlib_interface():
    """演示Qlib数据接口使用"""
    logger.info("=== Qlib数据接口演示 ===")
    
    try:
        # 创建Qlib接口实例
        qlib_interface = QlibDataInterface()
        
        # 获取股票列表（模拟，实际需要qlib环境）
        logger.info("获取股票列表...")
        # stock_list = qlib_interface.get_stock_list('A')
        # logger.info(f"获取到{len(stock_list)}只股票")
        
        # 获取价格数据（模拟）
        symbols = ['000001.SZ', '000002.SZ']
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        
        logger.info(f"获取价格数据: {symbols}")
        # price_data = qlib_interface.get_price_data(symbols, start_date, end_date)
        # logger.info(f"获取到{len(price_data)}条价格数据")
        
        logger.info("Qlib接口演示完成（需要qlib环境才能实际运行）")
        
    except ImportError as e:
        logger.warning(f"Qlib未安装: {e}")
    except Exception as e:
        logger.error(f"Qlib接口演示失败: {e}")


def demo_akshare_interface():
    """演示Akshare数据接口使用"""
    logger.info("=== Akshare数据接口演示 ===")
    
    try:
        # 创建Akshare接口实例
        akshare_interface = AkshareDataInterface(rate_limit=0.5)
        
        # 获取股票列表（模拟，实际需要akshare环境）
        logger.info("获取股票列表...")
        # stock_list = akshare_interface.get_stock_list('A')
        # logger.info(f"获取到{len(stock_list)}只股票")
        
        # 获取价格数据（模拟）
        symbols = ['000001', '000002']
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        
        logger.info(f"获取价格数据: {symbols}")
        # price_data = akshare_interface.get_price_data(symbols, start_date, end_date)
        # logger.info(f"获取到{len(price_data)}条价格数据")
        
        logger.info("Akshare接口演示完成（需要akshare环境才能实际运行）")
        
    except ImportError as e:
        logger.warning(f"Akshare未安装: {e}")
    except Exception as e:
        logger.error(f"Akshare接口演示失败: {e}")


def demo_cache_usage():
    """演示缓存使用"""
    logger.info("=== 缓存使用演示 ===")
    
    # 获取全局缓存实例
    cache = get_global_cache()
    
    # 设置缓存
    test_data = "这是测试数据"
    cache.set("test_key", test_data, ttl=60)  # 缓存60秒
    logger.info("数据已缓存")
    
    # 获取缓存
    cached_data = cache.get("test_key")
    logger.info(f"从缓存获取数据: {cached_data}")
    
    # 获取缓存信息
    cache_info = cache.get_cache_info()
    logger.info(f"缓存信息: {cache_info}")
    
    # 清理缓存
    cache.clear()
    logger.info("缓存已清理")


def demo_quality_checker():
    """演示数据质量检查"""
    logger.info("=== 数据质量检查演示 ===")
    
    import pandas as pd
    import numpy as np
    
    # 获取全局质量检查器
    checker = get_global_quality_checker()
    
    # 创建测试价格数据
    price_data = pd.DataFrame({
        'open': [10.0, 11.0, 12.0, 13.0, 14.0],
        'high': [11.0, 12.0, 13.0, 14.0, 15.0],
        'low': [9.0, 10.0, 11.0, 12.0, 13.0],
        'close': [10.5, 11.5, 12.5, 13.5, 14.5],
        'volume': [1000, 1200, 1100, 1300, 1150],
        'amount': [10500, 13800, 13750, 17550, 16675]
    })
    
    # 检查数据质量
    quality_report = checker.check_data_quality(price_data, 'price')
    logger.info(f"数据质量状态: {quality_report['status']}")
    logger.info(f"数据质量分数: {quality_report['score']:.2f}")
    logger.info(f"问题数量: {len(quality_report['issues'])}")
    logger.info(f"警告数量: {len(quality_report['warnings'])}")
    
    # 创建有问题的数据
    bad_data = pd.DataFrame({
        'open': [10.0, -1.0, 12.0],  # 包含负值
        'high': [11.0, 12.0, 11.0],  # 最高价低于最低价
        'low': [9.0, 10.0, 13.0],
        'close': [10.5, np.nan, 12.5],  # 包含缺失值
        'volume': [1000, 1200, 1100],
        'amount': [10500, 13800, 13750]
    })
    
    bad_quality_report = checker.check_data_quality(bad_data, 'price')
    logger.info(f"问题数据质量状态: {bad_quality_report['status']}")
    logger.info(f"问题数据质量分数: {bad_quality_report['score']:.2f}")
    logger.info(f"发现的问题: {bad_quality_report['issues']}")
    
    # 清洗数据
    cleaned_data = checker.clean_data(bad_data, 'price', 'conservative')
    logger.info(f"清洗前数据行数: {len(bad_data)}")
    logger.info(f"清洗后数据行数: {len(cleaned_data)}")


def main():
    """主函数"""
    logger.info("数据接口使用示例开始")
    
    # 演示各个功能
    demo_qlib_interface()
    print()
    
    demo_akshare_interface()
    print()
    
    demo_cache_usage()
    print()
    
    demo_quality_checker()
    
    logger.info("数据接口使用示例结束")


if __name__ == "__main__":
    main()