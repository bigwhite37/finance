#!/usr/bin/env python3
"""
O2O配置管理示例

此脚本演示如何使用O2O配置管理功能，包括：
1. 加载和验证O2O配置
2. 获取优化建议
3. 配置迁移
4. 生成验证报告
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_manager import ConfigManager
from config.o2o_config_validator import O2OConfigValidator
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_o2o_config():
    """演示基本的O2O配置使用"""
    print("=== 基本O2O配置演示 ===")
    
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 检查是否有O2O配置
    if 'o2o' not in config_manager.config:
        print("未找到O2O配置，创建默认配置...")
        o2o_template = config_manager.create_o2o_template()
        config_manager.update_config(o2o_template)
    
    # 获取O2O配置
    o2o_config = config_manager.get_o2o_config()
    print(f"O2O配置节数: {len(o2o_config)}")
    
    # 验证配置
    is_valid = config_manager.validate_config()
    print(f"配置验证结果: {'通过' if is_valid else '失败'}")
    
    print()


def demo_o2o_validation():
    """演示O2O配置验证功能"""
    print("=== O2O配置验证演示 ===")
    
    config_manager = ConfigManager()
    
    # 确保有O2O配置
    if 'o2o' not in config_manager.config:
        o2o_template = config_manager.create_o2o_template()
        config_manager.update_config(o2o_template)
    
    # 获取详细验证报告
    report = config_manager.get_o2o_validation_report()
    print(report)
    
    print()


def demo_optimization_suggestions():
    """演示优化建议功能"""
    print("=== 优化建议演示 ===")
    
    config_manager = ConfigManager()
    
    # 确保有O2O配置
    if 'o2o' not in config_manager.config:
        o2o_template = config_manager.create_o2o_template()
        config_manager.update_config(o2o_template)
    
    # 获取优化建议
    suggestions = config_manager.get_o2o_optimization_suggestions()
    
    if suggestions:
        for category, items in suggestions.items():
            if items:
                print(f"{category.upper()}建议:")
                for item in items:
                    print(f"  - {item}")
                print()
    else:
        print("当前配置无需优化建议")
    
    print()


def demo_config_migration():
    """演示配置迁移功能"""
    print("=== 配置迁移演示 ===")
    
    # 创建一个没有O2O配置的配置管理器
    config_manager = ConfigManager()
    
    # 移除O2O配置（如果存在）
    if 'o2o' in config_manager.config:
        del config_manager.config['o2o']
    
    print("原配置中O2O节存在:", 'o2o' in config_manager.config)
    
    # 执行迁移
    success = config_manager.migrate_config_to_o2o(backup=False)
    print(f"迁移结果: {'成功' if success else '失败'}")
    print("迁移后O2O节存在:", 'o2o' in config_manager.config)
    
    if success:
        # 验证迁移后的配置
        is_valid = config_manager.validate_config()
        print(f"迁移后配置验证: {'通过' if is_valid else '失败'}")
    
    print()


def demo_stage_specific_config():
    """演示获取特定阶段配置"""
    print("=== 阶段特定配置演示 ===")
    
    config_manager = ConfigManager()
    
    # 确保有O2O配置
    if 'o2o' not in config_manager.config:
        o2o_template = config_manager.create_o2o_template()
        config_manager.update_config(o2o_template)
    
    # 获取各阶段配置
    stages = ['offline_pretraining', 'warmup_finetuning', 'online_learning']
    
    for stage in stages:
        stage_config = config_manager.get_o2o_stage_config(stage)
        print(f"{stage}配置:")
        for key, value in stage_config.items():
            print(f"  {key}: {value}")
        print()


def demo_hot_update():
    """演示热更新功能"""
    print("=== 热更新功能演示 ===")
    
    config_manager = ConfigManager()
    
    # 确保有O2O配置
    if 'o2o' not in config_manager.config:
        o2o_template = config_manager.create_o2o_template()
        config_manager.update_config(o2o_template)
    
    # 启用热更新
    success = config_manager.enable_o2o_hot_update()
    print(f"热更新启用结果: {'成功' if success else '失败'}")
    
    if success:
        hot_update_config = config_manager.get_o2o_config().get('hot_update', {})
        print("热更新配置:")
        for key, value in hot_update_config.items():
            print(f"  {key}: {value}")
    
    print()


def demo_custom_validation():
    """演示自定义验证场景"""
    print("=== 自定义验证场景演示 ===")
    
    # 创建一个有问题的配置
    problematic_config = {
        'offline_pretraining': {
            'epochs': -10,  # 错误：负数
            'behavior_cloning_weight': 1.5,  # 错误：超出范围
            'td_learning_weight': -0.5,  # 错误：负数
            'learning_rate': 1.0,  # 错误：学习率过大
            'batch_size': 0  # 错误：批次大小为0
        },
        'warmup_finetuning': {
            'days': 0,  # 错误：天数为0
            'epochs': 1000,  # 警告：epochs过大
            'learning_rate': 1e-8  # 警告：学习率过小
        },
        'online_learning': {
            'initial_rho': 2.0,  # 错误：超出范围
            'rho_increment': -0.01,  # 错误：负增量
            'trust_region_beta': 0  # 错误：beta为0
        },
        'drift_detection': {
            'kl_threshold': -0.1,  # 错误：负阈值
            'sharpe_drop_threshold': 2.0,  # 错误：超出范围
            'cvar_breach_threshold': 0.1  # 错误：正阈值
        },
        'buffer_config': {
            'online_buffer_size': -1000,  # 错误：负大小
            'priority_alpha': 2.0,  # 错误：超出范围
            'priority_beta': 0  # 错误：beta为0
        },
        'risk_constraints': {
            'base_cvar_lambda': -1.0,  # 错误：负权重
            'lambda_scaling_factor': 0  # 错误：缩放因子为0
        },
        'training_flow': {
            'enable_offline_pretraining': 'yes',  # 错误：非布尔值
            'enable_warmup_finetuning': 1,  # 错误：非布尔值
            'enable_online_learning': None  # 错误：None值
        },
        'monitoring': {
            'report_frequency': -50  # 错误：负频率
        }
    }
    
    # 验证有问题的配置
    validator = O2OConfigValidator()
    is_valid, errors = validator.validate_o2o_config(problematic_config)
    
    print(f"有问题配置的验证结果: {'通过' if is_valid else '失败'}")
    print(f"发现错误数量: {len(errors)}")
    
    if errors:
        print("错误详情:")
        for i, error in enumerate(errors[:10], 1):  # 只显示前10个错误
            print(f"  {i}. {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")
    
    print()


def main():
    """主函数"""
    print("O2O配置管理系统演示")
    print("=" * 50)
    
    try:
        # 运行各种演示
        demo_basic_o2o_config()
        demo_o2o_validation()
        demo_optimization_suggestions()
        demo_config_migration()
        demo_stage_specific_config()
        demo_hot_update()
        demo_custom_validation()
        
        print("所有演示完成！")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()