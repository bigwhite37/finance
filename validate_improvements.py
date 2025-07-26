#!/usr/bin/env python3
"""
改进实施验证脚本
快速验证所有改进是否正确实施
"""

import sys
import os
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_improvements():
    """验证所有改进是否正确实施"""
    
    print("=" * 60)
    print("         改进实施验证报告")
    print("=" * 60)
    
    validation_results = {}
    
    # 1. 验证奖励函数优化
    print("\n🔍 验证奖励函数优化...")
    try:
        from rl_agent.trading_environment import TradingEnvironment
        import inspect
        
        # 创建临时环境实例
        test_env = TradingEnvironment(None, None, {})
        
        # 检查_calculate_reward方法源码
        reward_code = inspect.getsource(test_env._calculate_reward)
        
        # 检查关键改进特征
        has_amplification = "10.0" in reward_code or "放大收益信号" in reward_code
        has_momentum_bonus = "momentum_bonus" in reward_code or "持续盈利" in reward_code
        
        if has_amplification and has_momentum_bonus:
            validation_results['reward_function'] = True
            print("  ✅ 奖励函数已优化 - 包含收益放大和动量奖励")
        else:
            validation_results['reward_function'] = False
            print("  ❌ 奖励函数未优化")
        
    except Exception as e:
        validation_results['reward_function'] = False
        print(f"  ❌ 奖励函数验证失败: {e}")
    
    # 2. 验证因子库增强
    print("\n🔍 验证因子库增强...")
    try:
        from factors import FactorEngine
        
        factor_engine = FactorEngine({})
        enhanced_factors = factor_engine.default_factors
        
        expected_factors = [
            'momentum_20d', 'momentum_60d', 'price_reversal',
            'ma_ratio_20d', 'ma_ratio_60d', 'bollinger_position',
            'williams_r', 'rsi_14d', 'volume_ratio', 'turnover_rate',
            'volume_price_trend', 'volatility_20d', 'volatility_60d',
            'price_volume_correlation', 'mean_reversion_5d', 
            'trend_strength', 'volume_momentum'
        ]
        
        # 检查因子数量和关键因子
        has_enough_factors = len(enhanced_factors) >= 15
        key_factors_present = all(factor in enhanced_factors for factor in expected_factors[:10])
        
        if has_enough_factors and key_factors_present:
            validation_results['factor_enhancement'] = True
            print(f"  ✅ 因子库已增强 - 包含 {len(enhanced_factors)} 个因子")
            print(f"     关键因子: {', '.join(enhanced_factors[:8])}...")
        else:
            validation_results['factor_enhancement'] = False
            print(f"  ❌ 因子库未充分增强 - 仅有 {len(enhanced_factors)} 个因子")
        
    except Exception as e:
        validation_results['factor_enhancement'] = False
        print(f"  ❌ 因子库验证失败: {e}")
    
    # 3. 验证风险控制优化
    print("\n🔍 验证风险控制优化...")
    try:
        from config import ConfigManager
        
        config_manager = ConfigManager()
        shield_config = config_manager.get_config('safety_shield')
        env_config = config_manager.get_config('environment')
        
        # 检查关键参数是否已优化
        optimizations = {
            '单股票仓位限制': shield_config.get('max_position', 0) >= 0.16,
            '杠杆倍数': shield_config.get('max_leverage', 1.0) >= 1.3,
            'VaR阈值': shield_config.get('var_threshold', 0) >= 0.03,
            '回撤阈值': shield_config.get('max_drawdown_threshold', 0) >= 0.1,
            '环境杠杆': env_config.get('max_leverage', 1.0) >= 1.3,
            '回撤惩罚降低': env_config.get('lambda1', 2.0) <= 1.6
        }
        
        optimized_count = sum(optimizations.values())
        total_params = len(optimizations)
        
        if optimized_count >= total_params * 0.8:  # 80%的参数已优化
            validation_results['risk_optimization'] = True
            print(f"  ✅ 风险控制已优化 - {optimized_count}/{total_params} 项参数已改进")
        else:
            validation_results['risk_optimization'] = False
            print(f"  ❌ 风险控制优化不足 - 仅 {optimized_count}/{total_params} 项参数已改进")
        
        # 显示具体参数
        for param_name, is_optimized in optimizations.items():
            status = "✓" if is_optimized else "✗"
            print(f"     {status} {param_name}")
        
    except Exception as e:
        validation_results['risk_optimization'] = False
        print(f"  ❌ 风险控制验证失败: {e}")
    
    # 4. 验证网络架构升级
    print("\n🔍 验证网络架构升级...")
    try:
        enhanced_arch_exists = os.path.exists('rl_agent/enhanced_architecture.py')
        training_strategy_exists = os.path.exists('rl_agent/enhanced_training_strategy.py')
        
        if enhanced_arch_exists and training_strategy_exists:
            validation_results['architecture_upgrade'] = True
            print("  ✅ 网络架构升级文件已创建")
            print("     - enhanced_architecture.py: 增强网络架构")
            print("     - enhanced_training_strategy.py: 高级训练策略")
        else:
            validation_results['architecture_upgrade'] = False
            print("  ❌ 网络架构升级文件缺失")
            if not enhanced_arch_exists:
                print("     - 缺少 enhanced_architecture.py")
            if not training_strategy_exists:
                print("     - 缺少 enhanced_training_strategy.py")
        
    except Exception as e:
        validation_results['architecture_upgrade'] = False
        print(f"  ❌ 网络架构验证失败: {e}")
    
    # 5. 验证配置文件更新
    print("\n🔍 验证配置文件更新...")
    try:
        config_files_check = {
            'config/default_config.py': os.path.exists('config/default_config.py'),
            'config_train.yaml': os.path.exists('config_train.yaml'),
            'config_backtest.yaml': os.path.exists('config_backtest.yaml')
        }
        
        all_configs_exist = all(config_files_check.values())
        
        if all_configs_exist:
            print("  ✅ 配置文件完整")
            for config_file, exists in config_files_check.items():
                print(f"     ✓ {config_file}")
        else:
            print("  ⚠️  部分配置文件缺失")
            for config_file, exists in config_files_check.items():
                status = "✓" if exists else "✗"
                print(f"     {status} {config_file}")
        
    except Exception as e:
        print(f"  ❌ 配置文件验证失败: {e}")
    
    # 6. 总结报告
    print("\n" + "=" * 60)
    print("         验证结果总结")
    print("=" * 60)
    
    implemented_count = sum(validation_results.values())
    total_improvements = len(validation_results)
    
    improvement_names = {
        'reward_function': '奖励函数优化',
        'factor_enhancement': '因子库增强',
        'risk_optimization': '风险控制优化',
        'architecture_upgrade': '网络架构升级'
    }
    
    print(f"\n📊 实施状态: {implemented_count}/{total_improvements} 项改进已完成")
    
    for key, implemented in validation_results.items():
        name = improvement_names.get(key, key)
        status = "✅ 已实施" if implemented else "❌ 未实施"
        print(f"  {status} {name}")
    
    # 综合评估
    implementation_rate = implemented_count / total_improvements
    
    if implementation_rate >= 0.8:
        print("\n🎉 综合评估: 优秀 - 改进实施完成度高")
        next_step = "建议运行完整训练测试验证性能改进"
    elif implementation_rate >= 0.6:
        print("\n👍 综合评估: 良好 - 主要改进已实施")
        next_step = "建议完成剩余改进后进行训练测试"
    elif implementation_rate >= 0.4:
        print("\n📈 综合评估: 部分完成 - 需要完善更多改进")
        next_step = "建议优先完成核心改进（奖励函数、因子库）"
    else:
        print("\n⚠️  综合评估: 需要更多工作")
        next_step = "建议从基础改进开始逐步实施"
    
    print(f"\n🚀 下一步建议: {next_step}")
    print("\n" + "=" * 60)
    
    return validation_results, implementation_rate

def main():
    """主函数"""
    try:
        validation_results, implementation_rate = validate_improvements()
        
        # 返回适当的退出码
        if implementation_rate >= 0.8:
            print("✅ 验证完成 - 改进实施状态良好")
            sys.exit(0)
        else:
            print("⚠️  验证完成 - 需要完善更多改进")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 验证过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()