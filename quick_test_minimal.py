#!/usr/bin/env python3
"""
最小化测试版本 - 快速验证改进系统能否运行并达到8%目标
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置简单日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """简化版主函数"""
    logger.info("开始最小化测试")
    
    try:
        # 1. 运行现有的quick_start系统看看基线性能
        logger.info("步骤1: 运行quick_start.py获取基线性能")
        
        # 使用最简单的运行方式
        import subprocess
        result = subprocess.run([
            'python', 'quick_start.py', 
            '--mode', 'train',
            '--train_config', 'config_csi300_2020_2022_train.yaml'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("quick_start.py 训练成功完成")
            logger.info(f"输出: {result.stdout[-500:]}")  # 显示最后500字符
        else:
            logger.error(f"quick_start.py 训练失败: {result.stderr}")
            
        # 2. 尝试运行回测
        logger.info("步骤2: 运行基线回测")
        result = subprocess.run([
            'python', 'quick_start.py',
            '--mode', 'backtest',
            '--backtest_config', 'config_csi300_2023_backtest.yaml'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("quick_start.py 回测成功完成")
            # 解析年化收益率
            output = result.stdout
            if "年化收益率:" in output:
                # 提取年化收益率
                lines = output.split('\n')
                for line in lines:
                    if "年化收益率:" in line:
                        logger.info(f"发现结果: {line}")
                        
        else:
            logger.error(f"quick_start.py 回测失败: {result.stderr}")

    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
    
    logger.info("最小化测试完成")

if __name__ == "__main__":
    main()