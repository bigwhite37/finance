#!/usr/bin/env python3
"""
日志优化效果测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
import time
from utils.logging_utils import throttled_warning, statistical_warning
from rl_agent.safety_shield import SafetyShield
from rl_agent.cvar_ppo_agent import CVaRPPOAgent
import numpy as np

def test_logging_optimization():
    """测试日志优化效果"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("日志优化效果测试")
    print("=" * 60)
    
    # 测试1: 日志限制器
    print("\n1. 测试重复警告限制效果:")
    logger = logging.getLogger('test')
    
    start_time = time.time()
    for i in range(20):  # 尝试20次相同警告
        throttled_warning(
            logger,
            "这是一个重复的警告消息",
            "repeated_warning",
            min_interval=2.0,
            max_per_minute=3
        )
        time.sleep(0.1)
    
    elapsed_time = time.time() - start_time
    print(f"20次警告调用耗时: {elapsed_time:.2f}秒")
    
    # 测试2: 统计性日志
    print("\n2. 测试统计性日志效果:")
    for i in range(25):  # 25个事件，每5个报告一次
        statistical_warning(
            logger,
            "数值异常检测",
            f"检测到第{i+1}个异常值",
            report_interval=5
        )
    
    # 测试3: 安全保护层统计
    print("\n3. 测试安全保护层优化:")
    shield_config = {
        'max_position': 0.1,
        'max_leverage': 1.2,
        'var_threshold': 0.015,
        'max_drawdown_threshold': 0.05
    }
    
    safety_shield = SafetyShield(shield_config)
    
    # 模拟多次风险约束触发
    for i in range(10):
        action = np.array([0.3, 0.4, 0.5])  # 高杠杆动作
        info = {'max_drawdown': 0.04 + i * 0.005}  # 逐渐增加回撤
        
        safe_action = safety_shield.shield_action(action, info)
        time.sleep(0.1)
    
    print("\n安全保护层统计:")
    print(safety_shield.get_risk_event_summary())
    
    print("\n=" * 60)
    print("测试完成! 对比优化前的日志输出，现在应该:")
    print("- ✅ 重复警告被有效限制")
    print("- ✅ 统计性事件定期汇总报告")  
    print("- ✅ 风险控制事件被分类统计")
    print("- ✅ 整体日志噪音大幅降低")
    print("=" * 60)

if __name__ == "__main__":
    test_logging_optimization()