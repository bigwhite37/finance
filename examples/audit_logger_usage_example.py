#!/usr/bin/env python3
"""
审计日志系统使用示例
演示如何使用审计日志系统记录交易决策、查询历史记录和生成合规报告
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from src.rl_trading_system.audit.audit_logger import (
    AuditLogger, AuditQueryInterface, DataRetentionManager, ComplianceReportGenerator
)
from src.rl_trading_system.data.data_models import (
    TradingState, TradingAction, TransactionRecord
)


async def main():
    """主函数演示审计日志系统的使用"""
    
    # 配置审计日志系统
    config = {
        'influxdb': {
            'url': 'http://localhost:8086',
            'token': 'your_influxdb_token',
            'org': 'trading',
            'bucket': 'audit'
        },
        'relational_db_url': 'postgresql://user:password@localhost:5432/audit',
        'retention_days': 1825,  # 5年
        'batch_size': 100,
        'flush_interval': 60,  # 60秒
        'cleanup_interval_hours': 24
    }
    
    # 初始化审计日志器
    audit_logger = AuditLogger(config)
    
    try:
        # 启动审计日志器
        await audit_logger.start()
        print("审计日志器启动成功")
        
        # 示例1: 记录交易决策
        await demo_trading_decision_logging(audit_logger)
        
        # 示例2: 记录交易执行
        await demo_transaction_execution_logging(audit_logger)
        
        # 示例3: 记录风险违规
        await demo_risk_violation_logging(audit_logger)
        
        # 示例4: 查询审计记录
        await demo_audit_query(config)
        
        # 示例5: 数据保留管理
        await demo_data_retention(config)
        
        # 示例6: 生成合规报告
        await demo_compliance_report(config)
        
    finally:
        # 停止审计日志器
        await audit_logger.stop()
        print("审计日志器已停止")


async def demo_trading_decision_logging(audit_logger: AuditLogger):
    """演示交易决策记录"""
    print("\n=== 交易决策记录演示 ===")
    
    # 创建模拟交易状态
    state = TradingState(
        features=np.random.randn(60, 4, 50),  # 60天历史，4只股票，50个特征
        positions=np.array([0.25, 0.25, 0.25, 0.25]),  # 当前持仓
        market_state=np.random.randn(10),  # 市场状态特征
        cash=10000.0,
        total_value=100000.0
    )
    
    # 创建模拟交易动作
    action = TradingAction(
        target_weights=np.array([0.3, 0.2, 0.3, 0.2]),  # 目标权重
        confidence=0.85,
        timestamp=datetime.now()
    )
    
    # 模型输出
    model_outputs = {
        "q_values": [0.1, 0.2, 0.3, 0.4],
        "actor_loss": 0.05,
        "critic_loss": 0.03,
        "entropy": 0.8
    }
    
    # 特征重要性
    feature_importance = {
        "rsi": 0.3,
        "macd": 0.2,
        "volume": 0.15,
        "price_momentum": 0.25,
        "volatility": 0.1
    }
    
    # 记录交易决策
    await audit_logger.log_trading_decision(
        session_id="demo_session_001",
        model_version="v1.2.3",
        input_state=state,
        output_action=action,
        model_outputs=model_outputs,
        feature_importance=feature_importance,
        execution_time_ms=15.5
    )
    
    print("交易决策记录完成")


async def demo_transaction_execution_logging(audit_logger: AuditLogger):
    """演示交易执行记录"""
    print("\n=== 交易执行记录演示 ===")
    
    # 创建交易记录
    transaction = TransactionRecord(
        timestamp=datetime.now(),
        symbol="000001.SZ",
        action_type="buy",
        quantity=1000,
        price=10.5,
        commission=10.5,
        stamp_tax=0.0,
        slippage=5.25,
        total_cost=15.75
    )
    
    # 执行详情
    execution_details = {
        "order_id": "ORD_20240101_001",
        "fill_ratio": 1.0,
        "execution_venue": "SZSE",
        "market_impact": 0.002,
        "timing_cost": 0.001
    }
    
    # 记录交易执行
    await audit_logger.log_transaction_execution(
        session_id="demo_session_001",
        transaction=transaction,
        execution_details=execution_details
    )
    
    print("交易执行记录完成")


async def demo_risk_violation_logging(audit_logger: AuditLogger):
    """演示风险违规记录"""
    print("\n=== 风险违规记录演示 ===")
    
    violation_details = {
        "violation_type": "concentration_limit",
        "threshold": 0.3,
        "actual_value": 0.45,
        "affected_symbols": ["000001.SZ", "000002.SZ"],
        "severity": "high",
        "recommended_action": "reduce_position"
    }
    
    # 记录风险违规
    await audit_logger.log_risk_violation(
        session_id="demo_session_001",
        model_version="v1.2.3",
        violation_type="concentration_limit",
        violation_details=violation_details
    )
    
    print("风险违规记录完成")


async def demo_audit_query(config: Dict[str, Any]):
    """演示审计记录查询"""
    print("\n=== 审计记录查询演示 ===")
    
    # 初始化查询接口
    query_interface = AuditQueryInterface(config)
    
    try:
        await query_interface.connect()
        
        # 按时间范围查询
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        records = await query_interface.query_by_time_range(
            start_time=start_time,
            end_time=end_time,
            event_type="trading_decision",
            limit=10
        )
        
        print(f"查询到 {len(records)} 条交易决策记录")
        
        # 按会话查询
        session_records = await query_interface.query_by_session(
            session_id="demo_session_001",
            limit=10
        )
        
        print(f"会话 demo_session_001 有 {len(session_records)} 条记录")
        
        # 按模型版本查询
        model_records = await query_interface.query_by_model_version(
            model_version="v1.2.3",
            limit=10
        )
        
        print(f"模型版本 v1.2.3 有 {len(model_records)} 条记录")
        
    finally:
        await query_interface.disconnect()


async def demo_data_retention(config: Dict[str, Any]):
    """演示数据保留管理"""
    print("\n=== 数据保留管理演示 ===")
    
    # 初始化数据保留管理器
    retention_manager = DataRetentionManager(config)
    
    try:
        await retention_manager.start()
        
        # 获取数据统计
        stats = await retention_manager.get_data_statistics()
        print("数据统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 执行数据清理（在实际环境中会自动定期执行）
        print("执行数据清理...")
        await retention_manager.cleanup_expired_data()
        print("数据清理完成")
        
    finally:
        await retention_manager.stop()


async def demo_compliance_report(config: Dict[str, Any]):
    """演示合规报告生成"""
    print("\n=== 合规报告生成演示 ===")
    
    # 初始化合规报告生成器
    report_generator = ComplianceReportGenerator(config)
    
    try:
        await report_generator.start()
        
        # 生成过去30天的合规报告
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        report = await report_generator.generate_compliance_report(
            period_start=start_time,
            period_end=end_time
        )
        
        print("合规报告生成完成:")
        print(f"  报告ID: {report.report_id}")
        print(f"  报告期间: {report.period_start} 到 {report.period_end}")
        print(f"  总决策数: {report.total_decisions}")
        print(f"  风险违规数: {len(report.risk_violations)}")
        print(f"  合规分数: {report.compliance_score:.2f}")
        
        if report.concentration_analysis:
            print("  集中度分析:")
            for key, value in report.concentration_analysis.items():
                print(f"    {key}: {value:.4f}")
        
        if report.model_performance:
            print("  模型性能:")
            for key, value in report.model_performance.items():
                print(f"    {key}: {value:.4f}")
        
    finally:
        await report_generator.stop()


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())