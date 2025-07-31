#!/usr/bin/env python3
"""
配置管理器使用示例

演示如何使用ConfigManager加载、验证和管理配置文件
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl_trading_system.config import (
    ConfigManager, 
    MODEL_CONFIG_SCHEMA, 
    TRADING_CONFIG_SCHEMA,
    ConfigLoadError,
    ConfigValidationError
)


def main():
    """主函数"""
    print("=== 配置管理器使用示例 ===\n")
    
    # 创建配置管理器实例
    config_manager = ConfigManager()
    
    # 示例1：基本配置加载
    print("1. 基本配置加载")
    try:
        model_config = config_manager.load_config("config/model_config.yaml")
        print(f"   模型维度: {model_config['model']['transformer']['d_model']}")
        print(f"   学习率: {model_config['model']['sac']['lr_actor']}")
    except ConfigLoadError as e:
        print(f"   配置加载失败: {e}")
    
    print()
    
    # 示例2：环境变量覆盖
    print("2. 环境变量覆盖")
    
    # 设置环境变量
    os.environ['MODEL_TRANSFORMER_D_MODEL'] = '512'
    os.environ['MODEL_SAC_LR_ACTOR'] = '1e-3'
    
    try:
        model_config_with_env = config_manager.load_config(
            "config/model_config.yaml", 
            enable_env_override=True
        )
        print(f"   原始模型维度: 256")
        print(f"   环境变量覆盖后: {model_config_with_env['model']['transformer']['d_model']}")
        print(f"   原始学习率: 3e-4")
        print(f"   环境变量覆盖后: {model_config_with_env['model']['sac']['lr_actor']}")
    except ConfigLoadError as e:
        print(f"   配置加载失败: {e}")
    
    # 清理环境变量
    del os.environ['MODEL_TRANSFORMER_D_MODEL']
    del os.environ['MODEL_SAC_LR_ACTOR']
    
    print()
    
    # 示例3：配置验证
    print("3. 配置验证")
    try:
        trading_config = config_manager.load_config("config/trading_config.yaml")
        config_manager.validate_config(trading_config, TRADING_CONFIG_SCHEMA)
        print("   ✓ 交易配置验证通过")
    except ConfigValidationError as e:
        print(f"   ✗ 配置验证失败: {e}")
    except ConfigLoadError as e:
        print(f"   ✗ 配置加载失败: {e}")
    
    print()
    
    # 示例4：应用默认值
    print("4. 应用默认值")
    try:
        # 创建一个不完整的配置
        incomplete_config = {
            'model': {
                'transformer': {
                    'd_model': 128  # 只设置部分值
                }
            }
        }
        
        # 应用默认值
        complete_config = config_manager.apply_defaults(incomplete_config, MODEL_CONFIG_SCHEMA)
        
        print(f"   设置的值: d_model = {complete_config['model']['transformer']['d_model']}")
        print(f"   默认值: n_heads = {complete_config['model']['transformer']['n_heads']}")
        print(f"   默认值: dropout = {complete_config['model']['transformer']['dropout']}")
        
    except Exception as e:
        print(f"   应用默认值失败: {e}")
    
    print()
    
    # 示例5：完整工作流程
    print("5. 完整工作流程（加载 + 验证 + 默认值）")
    try:
        complete_config = config_manager.load_and_validate_config(
            "config/model_config.yaml",
            MODEL_CONFIG_SCHEMA,
            enable_env_override=True
        )
        print("   ✓ 完整工作流程执行成功")
        print(f"   最终配置包含 {len(complete_config)} 个顶级配置项")
        
    except (ConfigLoadError, ConfigValidationError) as e:
        print(f"   ✗ 完整工作流程失败: {e}")
    
    print()
    
    # 示例6：多配置文件合并
    print("6. 多配置文件合并")
    try:
        config_files = [
            "config/model_config.yaml",
            "config/trading_config.yaml"
        ]
        
        merged_config = config_manager.load_configs(config_files)
        
        print(f"   合并后包含配置项: {list(merged_config.keys())}")
        print("   ✓ 多配置文件合并成功")
        
    except ConfigLoadError as e:
        print(f"   ✗ 多配置文件合并失败: {e}")
    
    print()
    
    # 示例7：配置缓存
    print("7. 配置缓存")
    try:
        # 首次加载（从文件）
        config1 = config_manager.load_config("config/model_config.yaml", use_cache=True)
        
        # 再次加载（从缓存）
        config2 = config_manager.load_config("config/model_config.yaml", use_cache=True)
        
        print("   ✓ 配置缓存功能正常")
        print(f"   两次加载的配置内容相同: {config1 == config2}")
        
    except ConfigLoadError as e:
        print(f"   ✗ 配置缓存测试失败: {e}")
    
    print("\n=== 示例完成 ===")


if __name__ == "__main__":
    main()