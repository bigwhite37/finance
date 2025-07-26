#!/usr/bin/env python3
"""
测试增强因子库的计算能力和质量
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager
from data import DataManager  
from factors import FactorEngine

def test_enhanced_factors():
    """测试增强因子库"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 初始化配置
    config_manager = ConfigManager()
    
    # 数据管理器
    data_config = config_manager.get_data_config()
    data_manager = DataManager(data_config)
    
    # 因子引擎
    factor_config = config_manager.get_config('factors')
    factor_engine = FactorEngine(factor_config)
    
    logger.info("开始测试增强因子库...")
    
    # 获取少量测试数据
    logger.info("获取测试数据...")
    test_data_config = data_config.copy()
    test_data_config['start_date'] = '2022-01-01'
    test_data_config['end_date'] = '2022-03-31'
    
    stock_data = data_manager.get_stock_data(
        start_time=test_data_config['start_date'],
        end_time=test_data_config['end_date']
    )
    
    if stock_data.empty:
        logger.error("无法获取测试数据")
        return False
        
    # 准备价格和成交量数据
    price_data = stock_data['$close'].unstack().T
    volume_data = stock_data['$volume'].unstack().T
    
    # 选择前50只股票进行测试
    test_stocks = price_data.columns[:50]
    price_data = price_data[test_stocks]
    volume_data = volume_data[test_stocks]
    
    logger.info(f"测试数据形状 - 价格: {price_data.shape}, 成交量: {volume_data.shape}")
    
    # 测试每个因子的计算
    logger.info("开始测试各因子计算...")
    
    # 获取增强因子列表
    enhanced_factors = factor_engine.default_factors
    logger.info(f"增强因子列表 ({len(enhanced_factors)}个): {enhanced_factors}")
    
    successful_factors = []
    failed_factors = []
    
    for factor_name in enhanced_factors:
        try:
            logger.info(f"测试因子: {factor_name}")
            
            # 尝试计算单个因子
            test_factors = [factor_name]
            factor_data = factor_engine.calculate_all_factors(
                price_data, volume_data, test_factors
            )
            
            if not factor_data.empty:
                # 分析因子质量
                factor_stats = {
                    'name': factor_name,
                    'shape': factor_data.shape,
                    'null_ratio': factor_data.isnull().sum().sum() / factor_data.size,
                    'inf_ratio': np.isinf(factor_data.values).sum() / factor_data.size,
                    'mean': factor_data.mean().mean(),
                    'std': factor_data.std().mean(),
                    'min': factor_data.min().min(),
                    'max': factor_data.max().max()
                }
                
                logger.info(f"  ✓ {factor_name}: 形状{factor_stats['shape']}, "
                           f"缺失率{factor_stats['null_ratio']:.3f}, "
                           f"均值{factor_stats['mean']:.4f}, "
                           f"标准差{factor_stats['std']:.4f}")
                
                successful_factors.append(factor_stats)
            else:
                logger.warning(f"  ✗ {factor_name}: 计算结果为空")
                failed_factors.append(factor_name)
                
        except Exception as e:
            logger.error(f"  ✗ {factor_name}: 计算失败 - {str(e)}")
            failed_factors.append(factor_name)
    
    # 测试完整因子库计算
    logger.info("\n" + "="*50)
    logger.info("测试完整因子库计算...")
    
    try:
        full_factor_data = factor_engine.calculate_all_factors(price_data, volume_data)
        logger.info(f"✓ 完整因子库计算成功! 数据形状: {full_factor_data.shape}")
        
        # 分析完整因子库质量
        total_nulls = full_factor_data.isnull().sum().sum()
        total_size = full_factor_data.size
        null_ratio = total_nulls / total_size
        
        logger.info(f"  总体缺失率: {null_ratio:.3f}")
        logger.info(f"  因子数量: {full_factor_data.shape[1] if len(full_factor_data.shape) > 1 else 'Unknown'}")
        logger.info(f"  时间序列长度: {len(full_factor_data.index.get_level_values('datetime').unique())}")
        logger.info(f"  股票数量: {len(full_factor_data.index.get_level_values('instrument').unique())}")
        
    except Exception as e:
        logger.error(f"✗ 完整因子库计算失败: {str(e)}")
        return False
    
    # 汇总结果
    logger.info("\n" + "="*50)
    logger.info("测试结果汇总:")
    logger.info(f"  成功因子: {len(successful_factors)}/{len(enhanced_factors)}")
    logger.info(f"  失败因子: {len(failed_factors)}/{len(enhanced_factors)}")
    
    if failed_factors:
        logger.warning(f"  失败因子列表: {failed_factors}")
    
    # 因子质量评估
    if successful_factors:
        logger.info("\n因子质量分析:")
        high_quality_factors = []
        for factor in successful_factors:
            # 质量标准：缺失率<10%，无无限值，标准差>0
            if (factor['null_ratio'] < 0.1 and 
                factor['inf_ratio'] == 0 and 
                factor['std'] > 0):
                high_quality_factors.append(factor['name'])
        
        logger.info(f"  高质量因子 ({len(high_quality_factors)}个): {high_quality_factors}")
        
        success_rate = len(successful_factors) / len(enhanced_factors)
        quality_rate = len(high_quality_factors) / len(enhanced_factors)
        
        logger.info(f"  因子计算成功率: {success_rate:.1%}")
        logger.info(f"  因子高质量率: {quality_rate:.1%}")
        
        return success_rate > 0.8 and quality_rate > 0.6
    
    return False

if __name__ == "__main__":
    success = test_enhanced_factors()
    print(f"\n增强因子库测试 {'成功' if success else '失败'}")
    sys.exit(0 if success else 1)