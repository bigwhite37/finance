"""
审计日志系统
实现交易决策记录、存储机制、查询接口和数据完整性管理
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import logging
from abc import ABC, abstractmethod

import numpy as np
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import asyncpg

from ..data.data_models import TradingState, TradingAction, TransactionRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AuditRecord:
    """审计记录基础结构"""
    record_id: str
    timestamp: datetime
    event_type: str  # trading_decision, transaction_execution, model_update, etc.
    user_id: str
    session_id: str
    model_version: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 有效的事件类型
    VALID_EVENT_TYPES = {
        'trading_decision', 'transaction_execution', 'model_update',
        'risk_violation', 'system_error', 'compliance_check'
    }
    
    def __post_init__(self):
        """数据验证"""
        self._validate()
    
    def _validate(self):
        """验证数据有效性"""
        if self.event_type not in self.VALID_EVENT_TYPES:
            raise ValueError(f"无效的事件类型: {self.event_type}")
        
        if not self.record_id:
            raise ValueError("记录ID不能为空")
        
        if not self.session_id:
            raise ValueError("会话ID不能为空")
        
        if not self.model_version:
            raise ValueError("模型版本不能为空")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'record_id': self.record_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'model_version': self.model_version,
            'data': json.dumps(self.data),
            'metadata': json.dumps(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditRecord':
        """从字典创建对象"""
        return cls(
            record_id=data['record_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=data['event_type'],
            user_id=data['user_id'],
            session_id=data['session_id'],
            model_version=data['model_version'],
            data=json.loads(data['data']) if isinstance(data['data'], str) else data['data'],
            metadata=json.loads(data['metadata']) if isinstance(data['metadata'], str) else data['metadata']
        )
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AuditRecord':
        """从JSON字符串创建对象"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class DecisionRecord:
    """交易决策记录"""
    decision_id: str
    timestamp: datetime
    model_version: str
    input_state: TradingState
    output_action: TradingAction
    model_outputs: Dict[str, Any]
    feature_importance: Dict[str, float]
    risk_metrics: Dict[str, float]
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'decision_id': self.decision_id,
            'timestamp': self.timestamp.isoformat(),
            'model_version': self.model_version,
            'input_state': self.input_state.to_dict(),
            'output_action': self.output_action.to_dict(),
            'model_outputs': json.dumps(self.model_outputs),
            'feature_importance': json.dumps(self.feature_importance),
            'risk_metrics': json.dumps(self.risk_metrics),
            'execution_time_ms': self.execution_time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionRecord':
        """从字典创建对象"""
        return cls(
            decision_id=data['decision_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            model_version=data['model_version'],
            input_state=TradingState.from_dict(data['input_state']),
            output_action=TradingAction.from_dict(data['output_action']),
            model_outputs=json.loads(data['model_outputs']) if isinstance(data['model_outputs'], str) else data['model_outputs'],
            feature_importance=json.loads(data['feature_importance']) if isinstance(data['feature_importance'], str) else data['feature_importance'],
            risk_metrics=json.loads(data['risk_metrics']) if isinstance(data['risk_metrics'], str) else data['risk_metrics'],
            execution_time_ms=data.get('execution_time_ms')
        )


@dataclass
class ComplianceReport:
    """合规报告"""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_decisions: int
    risk_violations: List[Dict[str, Any]]
    concentration_analysis: Dict[str, float]
    model_performance: Dict[str, float]
    compliance_score: float
    
    def __post_init__(self):
        """数据验证"""
        if not (0 <= self.compliance_score <= 1):
            raise ValueError("合规分数必须在0到1之间")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'report_id': self.report_id,
            'generated_at': self.generated_at.isoformat(),
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_decisions': self.total_decisions,
            'risk_violations': self.risk_violations,
            'concentration_analysis': self.concentration_analysis,
            'model_performance': self.model_performance,
            'compliance_score': self.compliance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceReport':
        """从字典创建对象"""
        return cls(
            report_id=data['report_id'],
            generated_at=datetime.fromisoformat(data['generated_at']),
            period_start=datetime.fromisoformat(data['period_start']),
            period_end=datetime.fromisoformat(data['period_end']),
            total_decisions=data['total_decisions'],
            risk_violations=data['risk_violations'],
            concentration_analysis=data['concentration_analysis'],
            model_performance=data['model_performance'],
            compliance_score=data['compliance_score']
        )


class DatabaseInterface(ABC):
    """数据库接口抽象类"""
    
    @abstractmethod
    async def connect(self):
        """连接数据库"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开数据库连接"""
        pass
    
    @abstractmethod
    async def write_records(self, records: List[AuditRecord]):
        """写入记录"""
        pass
    
    @abstractmethod
    async def query_records(self, **kwargs) -> List[AuditRecord]:
        """查询记录"""
        pass


class InfluxDBInterface(DatabaseInterface):
    """InfluxDB时序数据库接口"""
    
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.url = url
        self.token = token
        self.org = org
        self.bucket = bucket
        self.client = None
        self.write_api = None
    
    async def connect(self):
        """连接InfluxDB"""
        try:
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            logger.info("InfluxDB连接成功")
        except Exception as e:
            logger.error(f"InfluxDB连接失败: {e}")
            raise
    
    async def disconnect(self):
        """断开InfluxDB连接"""
        if self.client:
            self.client.close()
            logger.info("InfluxDB连接已断开")
    
    async def write_records(self, records: List[AuditRecord]):
        """写入记录到InfluxDB"""
        try:
            points = []
            for record in records:
                point = Point("audit_record") \
                    .tag("event_type", record.event_type) \
                    .tag("user_id", record.user_id) \
                    .tag("session_id", record.session_id) \
                    .tag("model_version", record.model_version) \
                    .field("record_id", record.record_id) \
                    .field("data", json.dumps(record.data)) \
                    .field("metadata", json.dumps(record.metadata)) \
                    .time(record.timestamp)
                points.append(point)
            
            self.write_api.write(bucket=self.bucket, record=points)
            logger.debug(f"写入{len(records)}条记录到InfluxDB")
            
        except Exception as e:
            logger.error(f"写入InfluxDB失败: {e}")
            raise
    
    async def query_records(self, **kwargs) -> List[AuditRecord]:
        """从InfluxDB查询记录"""
        # InfluxDB主要用于时序数据存储，复杂查询通过PostgreSQL进行
        pass


class PostgreSQLInterface(DatabaseInterface):
    """PostgreSQL关系数据库接口"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    async def connect(self):
        """连接PostgreSQL"""
        try:
            self.pool = await asyncpg.create_pool(self.connection_string)
            
            # 创建审计表
            await self._create_tables()
            logger.info("PostgreSQL连接成功")
            
        except Exception as e:
            logger.error(f"PostgreSQL连接失败: {e}")
            raise
    
    async def disconnect(self):
        """断开PostgreSQL连接"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL连接已断开")
    
    async def _create_tables(self):
        """创建审计相关表"""
        async with self.pool.acquire() as conn:
            # 审计记录表
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_records (
                    record_id VARCHAR(255) PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    user_id VARCHAR(100) NOT NULL,
                    session_id VARCHAR(255) NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    data JSONB NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            ''')
            
            # 决策记录表
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS decision_records (
                    decision_id VARCHAR(255) PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    input_state JSONB NOT NULL,
                    output_action JSONB NOT NULL,
                    model_outputs JSONB DEFAULT '{}',
                    feature_importance JSONB DEFAULT '{}',
                    risk_metrics JSONB DEFAULT '{}',
                    execution_time_ms FLOAT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            ''')
            
            # 合规报告表
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id VARCHAR(255) PRIMARY KEY,
                    generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                    total_decisions INTEGER NOT NULL,
                    risk_violations JSONB DEFAULT '[]',
                    concentration_analysis JSONB DEFAULT '{}',
                    model_performance JSONB DEFAULT '{}',
                    compliance_score FLOAT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            ''')
            
            # 创建索引
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_records(timestamp);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_records(event_type);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_records(session_id);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_model_version ON audit_records(model_version);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_decision_timestamp ON decision_records(timestamp);')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_decision_model_version ON decision_records(model_version);')
    
    async def write_records(self, records: List[AuditRecord]):
        """写入记录到PostgreSQL"""
        try:
            async with self.pool.acquire() as conn:
                # 批量插入审计记录
                values = [
                    (r.record_id, r.timestamp, r.event_type, r.user_id, 
                     r.session_id, r.model_version, json.dumps(r.data), 
                     json.dumps(r.metadata))
                    for r in records
                ]
                
                await conn.executemany('''
                    INSERT INTO audit_records 
                    (record_id, timestamp, event_type, user_id, session_id, 
                     model_version, data, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (record_id) DO NOTHING
                ''', values)
                
                logger.debug(f"写入{len(records)}条记录到PostgreSQL")
                
        except Exception as e:
            logger.error(f"写入PostgreSQL失败: {e}")
            raise
    
    async def query_records(self, **kwargs) -> List[AuditRecord]:
        """从PostgreSQL查询记录"""
        try:
            async with self.pool.acquire() as conn:
                query = "SELECT * FROM audit_records WHERE 1=1"
                params = []
                param_count = 0
                
                # 构建查询条件
                if 'start_time' in kwargs and 'end_time' in kwargs:
                    param_count += 2
                    query += f" AND timestamp BETWEEN ${param_count-1} AND ${param_count}"
                    params.extend([kwargs['start_time'], kwargs['end_time']])
                
                if 'event_type' in kwargs:
                    param_count += 1
                    query += f" AND event_type = ${param_count}"
                    params.append(kwargs['event_type'])
                
                if 'session_id' in kwargs:
                    param_count += 1
                    query += f" AND session_id = ${param_count}"
                    params.append(kwargs['session_id'])
                
                if 'model_version' in kwargs:
                    param_count += 1
                    query += f" AND model_version = ${param_count}"
                    params.append(kwargs['model_version'])
                
                query += " ORDER BY timestamp DESC"
                
                if 'limit' in kwargs:
                    param_count += 1
                    query += f" LIMIT ${param_count}"
                    params.append(kwargs['limit'])
                
                rows = await conn.fetch(query, *params)
                
                # 转换为AuditRecord对象
                records = []
                for row in rows:
                    record = AuditRecord(
                        record_id=row['record_id'],
                        timestamp=row['timestamp'],
                        event_type=row['event_type'],
                        user_id=row['user_id'],
                        session_id=row['session_id'],
                        model_version=row['model_version'],
                        data=json.loads(row['data']) if isinstance(row['data'], str) else row['data'],
                        metadata=json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                    )
                    records.append(record)
                
                return records
                
        except Exception as e:
            logger.error(f"查询PostgreSQL失败: {e}")
            raise
    
    async def write_decision_record(self, record: DecisionRecord):
        """写入决策记录"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO decision_records 
                    (decision_id, timestamp, model_version, input_state, output_action,
                     model_outputs, feature_importance, risk_metrics, execution_time_ms)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (decision_id) DO NOTHING
                ''', 
                    record.decision_id, record.timestamp, record.model_version,
                    json.dumps(record.input_state.to_dict()),
                    json.dumps(record.output_action.to_dict()),
                    json.dumps(record.model_outputs),
                    json.dumps(record.feature_importance),
                    json.dumps(record.risk_metrics),
                    record.execution_time_ms
                )
                
        except Exception as e:
            logger.error(f"写入决策记录失败: {e}")
            raise
    
    async def get_decision_record(self, decision_id: str) -> Optional[DecisionRecord]:
        """获取决策记录"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM decision_records WHERE decision_id = $1",
                    decision_id
                )
                
                if row:
                    return DecisionRecord(
                        decision_id=row['decision_id'],
                        timestamp=row['timestamp'],
                        model_version=row['model_version'],
                        input_state=TradingState.from_dict(json.loads(row['input_state'])),
                        output_action=TradingAction.from_dict(json.loads(row['output_action'])),
                        model_outputs=json.loads(row['model_outputs']),
                        feature_importance=json.loads(row['feature_importance']),
                        risk_metrics=json.loads(row['risk_metrics']),
                        execution_time_ms=row['execution_time_ms']
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"获取决策记录失败: {e}")
            raise


class AuditLogger:
    """审计日志器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_records: List[AuditRecord] = []
        self.batch_lock = asyncio.Lock()
        self.is_running = False
        self.flush_task = None
        
        # 初始化数据库接口
        self._init_databases()
    
    def _init_databases(self):
        """初始化数据库接口"""
        # InfluxDB配置
        influx_config = self.config.get('influxdb', {})
        if influx_config:
            self.timeseries_db = InfluxDBInterface(
                url=influx_config.get('url', 'http://localhost:8086'),
                token=influx_config.get('token', ''),
                org=influx_config.get('org', 'trading'),
                bucket=influx_config.get('bucket', 'audit')
            )
        else:
            self.timeseries_db = None
        
        # PostgreSQL配置
        postgres_url = self.config.get('relational_db_url', 'postgresql://localhost:5432/audit')
        self.relational_db = PostgreSQLInterface(postgres_url)
    
    async def start(self):
        """启动审计日志器"""
        if self.is_running:
            return
        
        try:
            # 连接数据库
            if self.timeseries_db:
                await self.timeseries_db.connect()
            await self.relational_db.connect()
            
            self.is_running = True
            
            # 启动定期刷新任务
            flush_interval = self.config.get('flush_interval', 60)
            self.flush_task = asyncio.create_task(self._periodic_flush(flush_interval))
            
            logger.info("审计日志器启动成功")
            
        except Exception as e:
            logger.error(f"审计日志器启动失败: {e}")
            raise
    
    async def stop(self):
        """停止审计日志器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 停止定期刷新任务
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # 刷新剩余记录
        await self._flush_batch()
        
        # 断开数据库连接
        if self.timeseries_db:
            await self.timeseries_db.disconnect()
        await self.relational_db.disconnect()
        
        logger.info("审计日志器已停止")
    
    async def log_trading_decision(self, session_id: str, model_version: str,
                                 input_state: TradingState, output_action: TradingAction,
                                 model_outputs: Dict[str, Any],
                                 feature_importance: Dict[str, float],
                                 execution_time_ms: Optional[float] = None):
        """记录交易决策"""
        try:
            # 生成决策ID
            decision_id = self._generate_record_id()
            
            # 创建决策记录
            decision_record = DecisionRecord(
                decision_id=decision_id,
                timestamp=datetime.now(),
                model_version=model_version,
                input_state=input_state,
                output_action=output_action,
                model_outputs=model_outputs,
                feature_importance=feature_importance,
                risk_metrics=self._calculate_risk_metrics(output_action),
                execution_time_ms=execution_time_ms
            )
            
            # 写入决策记录表
            await self.relational_db.write_decision_record(decision_record)
            
            # 创建审计记录
            audit_record = AuditRecord(
                record_id=self._generate_record_id(),
                timestamp=datetime.now(),
                event_type="trading_decision",
                user_id="system",
                session_id=session_id,
                model_version=model_version,
                data={
                    "decision_id": decision_id,
                    "target_weights": output_action.target_weights.tolist(),
                    "confidence": output_action.confidence,
                    "portfolio_value": input_state.total_value,
                    "cash": input_state.cash
                },
                metadata={
                    "execution_time_ms": execution_time_ms,
                    "feature_count": len(feature_importance),
                    "model_output_keys": list(model_outputs.keys())
                }
            )
            
            # 添加到批次
            async with self.batch_lock:
                self.batch_records.append(audit_record)
                
                # 如果批次满了，立即刷新
                if len(self.batch_records) >= self.config.get('batch_size', 100):
                    await self._flush_batch()
            
            logger.debug(f"记录交易决策: {decision_id}")
            
        except Exception as e:
            logger.error(f"记录交易决策失败: {e}")
            raise
    
    async def log_transaction_execution(self, session_id: str, transaction: TransactionRecord,
                                      execution_details: Dict[str, Any]):
        """记录交易执行"""
        try:
            audit_record = AuditRecord(
                record_id=self._generate_record_id(),
                timestamp=datetime.now(),
                event_type="transaction_execution",
                user_id="system",
                session_id=session_id,
                model_version="execution",  # 交易执行使用特殊标识
                data={
                    "symbol": transaction.symbol,
                    "action_type": transaction.action_type,
                    "quantity": transaction.quantity,
                    "price": transaction.price,
                    "commission": transaction.commission,
                    "stamp_tax": transaction.stamp_tax,
                    "slippage": transaction.slippage,
                    "total_cost": transaction.total_cost,
                    **execution_details
                },
                metadata={
                    "transaction_value": transaction.get_transaction_value(),
                    "cost_ratio": transaction.get_cost_ratio()
                }
            )
            
            async with self.batch_lock:
                self.batch_records.append(audit_record)
            
            logger.debug(f"记录交易执行: {transaction.symbol} {transaction.action_type}")
            
        except Exception as e:
            logger.error(f"记录交易执行失败: {e}")
            raise
    
    async def log_risk_violation(self, session_id: str, model_version: str,
                               violation_type: str, violation_details: Dict[str, Any]):
        """记录风险违规"""
        try:
            audit_record = AuditRecord(
                record_id=self._generate_record_id(),
                timestamp=datetime.now(),
                event_type="risk_violation",
                user_id="system",
                session_id=session_id,
                model_version=model_version,
                data={
                    "violation_type": violation_type,
                    **violation_details
                },
                metadata={
                    "severity": violation_details.get("severity", "medium")
                }
            )
            
            async with self.batch_lock:
                self.batch_records.append(audit_record)
            
            logger.warning(f"记录风险违规: {violation_type}")
            
        except Exception as e:
            logger.error(f"记录风险违规失败: {e}")
            raise
    
    async def _flush_batch(self):
        """刷新批次记录"""
        if not self.batch_records:
            return
        
        try:
            records_to_flush = self.batch_records.copy()
            self.batch_records.clear()
            
            # 写入时序数据库
            if self.timeseries_db:
                await self.timeseries_db.write_records(records_to_flush)
            
            # 写入关系数据库
            await self.relational_db.write_records(records_to_flush)
            
            logger.debug(f"刷新{len(records_to_flush)}条审计记录")
            
        except Exception as e:
            logger.error(f"刷新批次记录失败: {e}")
            # 将记录重新加入批次
            async with self.batch_lock:
                self.batch_records.extend(records_to_flush)
            raise
    
    async def _periodic_flush(self, interval: int):
        """定期刷新"""
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                async with self.batch_lock:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期刷新失败: {e}")
    
    def _generate_record_id(self) -> str:
        """生成记录ID"""
        return str(uuid.uuid4())
    
    def _calculate_risk_metrics(self, action: TradingAction) -> Dict[str, float]:
        """计算风险指标"""
        return {
            "concentration": action.get_concentration(),
            "active_positions": float(action.get_active_positions()),
            "max_weight": float(action.target_weights.max()),
            "min_weight": float(action.target_weights.min())
        }


class AuditQueryInterface:
    """审计查询接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        postgres_url = config.get('relational_db_url', 'postgresql://localhost:5432/audit')
        self.relational_db = PostgreSQLInterface(postgres_url)
    
    async def connect(self):
        """连接数据库"""
        await self.relational_db.connect()
    
    async def disconnect(self):
        """断开数据库连接"""
        await self.relational_db.disconnect()
    
    async def query_by_time_range(self, start_time: datetime, end_time: datetime,
                                event_type: Optional[str] = None,
                                limit: Optional[int] = None) -> List[AuditRecord]:
        """按时间范围查询"""
        kwargs = {
            'start_time': start_time,
            'end_time': end_time
        }
        
        if event_type:
            kwargs['event_type'] = event_type
        
        if limit:
            kwargs['limit'] = limit
        
        return await self.relational_db.query_records(**kwargs)
    
    async def query_by_session(self, session_id: str,
                             limit: Optional[int] = None) -> List[AuditRecord]:
        """按会话查询"""
        kwargs = {'session_id': session_id}
        if limit:
            kwargs['limit'] = limit
        
        return await self.relational_db.query_records(**kwargs)
    
    async def query_by_model_version(self, model_version: str,
                                   limit: Optional[int] = None) -> List[AuditRecord]:
        """按模型版本查询"""
        kwargs = {'model_version': model_version}
        if limit:
            kwargs['limit'] = limit
        
        return await self.relational_db.query_records(**kwargs)
    
    async def get_decision_details(self, decision_id: str) -> Optional[DecisionRecord]:
        """获取决策详情"""
        return await self.relational_db.get_decision_record(decision_id)


class DataRetentionManager:
    """数据保留管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retention_days = config.get('retention_days', 1825)  # 默认5年
        postgres_url = config.get('relational_db_url', 'postgresql://localhost:5432/audit')
        self.relational_db = PostgreSQLInterface(postgres_url)
        
        # 初始化查询接口
        self.query_interface = AuditQueryInterface(config)
    
    async def start(self):
        """启动数据保留管理器"""
        await self.relational_db.connect()
        await self.query_interface.connect()
        
        # 启动定期清理任务
        cleanup_interval = self.config.get('cleanup_interval_hours', 24) * 3600
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup(cleanup_interval))
        
        logger.info("数据保留管理器启动成功")
    
    async def stop(self):
        """停止数据保留管理器"""
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.query_interface.disconnect()
        await self.relational_db.disconnect()
        
        logger.info("数据保留管理器已停止")
    
    async def cleanup_expired_data(self):
        """清理过期数据"""
        try:
            retention_date = self._calculate_retention_date()
            
            async with self.relational_db.pool.acquire() as conn:
                # 清理过期的审计记录
                audit_deleted = await conn.execute(
                    "DELETE FROM audit_records WHERE timestamp < $1",
                    retention_date
                )
                
                # 清理过期的决策记录
                decision_deleted = await conn.execute(
                    "DELETE FROM decision_records WHERE timestamp < $1",
                    retention_date
                )
                
                # 清理过期的合规报告
                report_deleted = await conn.execute(
                    "DELETE FROM compliance_reports WHERE period_end < $1",
                    retention_date
                )
                
                logger.info(f"清理过期数据完成: 审计记录{audit_deleted}条, "
                           f"决策记录{decision_deleted}条, 合规报告{report_deleted}条")
                
        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")
            raise
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        try:
            async with self.relational_db.pool.acquire() as conn:
                # 审计记录统计
                audit_stats = await conn.fetchrow('''
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(timestamp) as oldest_record,
                        MAX(timestamp) as newest_record
                    FROM audit_records
                ''')
                
                # 决策记录统计
                decision_stats = await conn.fetchrow('''
                    SELECT COUNT(*) as total_decisions
                    FROM decision_records
                ''')
                
                # 合规报告统计
                report_stats = await conn.fetchrow('''
                    SELECT COUNT(*) as total_reports
                    FROM compliance_reports
                ''')
                
                # 数据库大小统计（PostgreSQL特定）
                size_stats = await conn.fetchrow('''
                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                ''')
                
                return {
                    'total_records': audit_stats['total_records'],
                    'total_decisions': decision_stats['total_decisions'],
                    'total_reports': report_stats['total_reports'],
                    'oldest_record': audit_stats['oldest_record'].isoformat() if audit_stats['oldest_record'] else None,
                    'newest_record': audit_stats['newest_record'].isoformat() if audit_stats['newest_record'] else None,
                    'database_size': size_stats['db_size'],
                    'retention_days': self.retention_days
                }
                
        except Exception as e:
            logger.error(f"获取数据统计失败: {e}")
            raise
    
    async def _periodic_cleanup(self, interval: int):
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(interval)
                await self.cleanup_expired_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期清理任务失败: {e}")
    
    def _calculate_retention_date(self) -> datetime:
        """计算数据保留截止日期"""
        return datetime.now() - timedelta(days=self.retention_days)


class ComplianceReportGenerator:
    """合规报告生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_interface = AuditQueryInterface(config)
    
    async def start(self):
        """启动合规报告生成器"""
        await self.query_interface.connect()
        logger.info("合规报告生成器启动成功")
    
    async def stop(self):
        """停止合规报告生成器"""
        await self.query_interface.disconnect()
        logger.info("合规报告生成器已停止")
    
    async def generate_compliance_report(self, period_start: datetime, 
                                       period_end: datetime) -> ComplianceReport:
        """生成合规报告"""
        try:
            # 查询期间内的所有决策记录
            decisions = await self.query_interface.query_by_time_range(
                period_start, period_end, event_type="trading_decision"
            )
            
            # 查询期间内的风险违规记录
            violations = await self.query_interface.query_by_time_range(
                period_start, period_end, event_type="risk_violation"
            )
            
            # 分析集中度
            concentration_analysis = await self._analyze_concentration(decisions)
            
            # 分析模型性能
            model_performance = await self._calculate_model_performance(decisions)
            
            # 计算合规分数
            compliance_score = self._calculate_compliance_score(
                len(decisions), len(violations), concentration_analysis
            )
            
            # 创建合规报告
            report = ComplianceReport(
                report_id=str(uuid.uuid4()),
                generated_at=datetime.now(),
                period_start=period_start,
                period_end=period_end,
                total_decisions=len(decisions),
                risk_violations=[v.data for v in violations],
                concentration_analysis=concentration_analysis,
                model_performance=model_performance,
                compliance_score=compliance_score
            )
            
            # 保存报告
            await self._save_report(report)
            
            logger.info(f"生成合规报告: {report.report_id}")
            return report
            
        except Exception as e:
            logger.error(f"生成合规报告失败: {e}")
            raise
    
    async def _analyze_concentration(self, decisions: List[AuditRecord]) -> Dict[str, float]:
        """分析持仓集中度"""
        if not decisions:
            return {}
        
        concentrations = []
        max_weights = []
        
        for decision in decisions:
            target_weights = decision.data.get('target_weights', [])
            if target_weights:
                weights = np.array(target_weights)
                concentration = np.sum(weights ** 2)  # Herfindahl指数
                concentrations.append(concentration)
                max_weights.append(weights.max())
        
        if concentrations:
            return {
                'avg_concentration': np.mean(concentrations),
                'max_concentration': np.max(concentrations),
                'min_concentration': np.min(concentrations),
                'avg_max_weight': np.mean(max_weights),
                'max_single_weight': np.max(max_weights)
            }
        
        return {}
    
    async def _calculate_model_performance(self, decisions: List[AuditRecord]) -> Dict[str, float]:
        """计算模型性能"""
        if not decisions:
            return {}
        
        confidences = []
        execution_times = []
        
        for decision in decisions:
            confidence = decision.data.get('confidence', 0)
            confidences.append(confidence)
            
            execution_time = decision.metadata.get('execution_time_ms', 0)
            if execution_time:
                execution_times.append(execution_time)
        
        performance = {
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        if execution_times:
            performance.update({
                'avg_execution_time_ms': np.mean(execution_times),
                'max_execution_time_ms': np.max(execution_times),
                'min_execution_time_ms': np.min(execution_times)
            })
        
        return performance
    
    def _calculate_compliance_score(self, total_decisions: int, violation_count: int,
                                  concentration_analysis: Dict[str, float]) -> float:
        """计算合规分数"""
        if total_decisions == 0:
            return 1.0
        
        # 基础分数
        base_score = 1.0
        
        # 违规惩罚
        violation_penalty = min(0.5, violation_count / total_decisions)
        base_score -= violation_penalty
        
        # 集中度惩罚
        if concentration_analysis:
            max_concentration = concentration_analysis.get('max_concentration', 0)
            if max_concentration > 0.5:  # 集中度过高
                concentration_penalty = min(0.3, (max_concentration - 0.5) * 0.6)
                base_score -= concentration_penalty
        
        return max(0.0, base_score)
    
    async def _save_report(self, report: ComplianceReport):
        """保存合规报告"""
        try:
            async with self.query_interface.relational_db.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO compliance_reports 
                    (report_id, generated_at, period_start, period_end, total_decisions,
                     risk_violations, concentration_analysis, model_performance, compliance_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''',
                    report.report_id, report.generated_at, report.period_start,
                    report.period_end, report.total_decisions,
                    json.dumps(report.risk_violations),
                    json.dumps(report.concentration_analysis),
                    json.dumps(report.model_performance),
                    report.compliance_score
                )
                
        except Exception as e:
            logger.error(f"保存合规报告失败: {e}")
            raiseal_db = PostgreSQLInterface(postgres_url)
    
    async def connect(self):
        """连接数据库"""
        await self.relational_db.connect()
    
    async def disconnect(self):
        """断开数据库连接"""
        await self.relational_db.disconnect()
    
    async def cleanup_expired_data(self):
        """清理过期数据"""
        try:
            retention_date = self._calculate_retention_date()
            
            async with self.relational_db.pool.acquire() as conn:
                # 删除过期的审计记录
                deleted_audit = await conn.execute(
                    "DELETE FROM audit_records WHERE timestamp < $1",
                    retention_date
                )
                
                # 删除过期的决策记录
                deleted_decisions = await conn.execute(
                    "DELETE FROM decision_records WHERE timestamp < $1",
                    retention_date
                )
                
                logger.info(f"清理过期数据完成: 审计记录{deleted_audit}, 决策记录{deleted_decisions}")
                
        except Exception as e:
            logger.error(f"清理过期数据失败: {e}")
            raise
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计"""
        try:
            async with self.relational_db.pool.acquire() as conn:
                # 审计记录统计
                audit_stats = await conn.fetchrow('''
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(timestamp) as oldest_record,
                        MAX(timestamp) as newest_record
                    FROM audit_records
                ''')
                
                # 决策记录统计
                decision_stats = await conn.fetchrow('''
                    SELECT COUNT(*) as total_decisions
                    FROM decision_records
                ''')
                
                # 数据库大小统计
                size_stats = await conn.fetchrow('''
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('audit_records')) as audit_size,
                        pg_size_pretty(pg_total_relation_size('decision_records')) as decision_size
                ''')
                
                return {
                    'total_records': audit_stats['total_records'],
                    'total_decisions': decision_stats['total_decisions'],
                    'oldest_record': audit_stats['oldest_record'].isoformat() if audit_stats['oldest_record'] else None,
                    'newest_record': audit_stats['newest_record'].isoformat() if audit_stats['newest_record'] else None,
                    'audit_table_size': size_stats['audit_size'],
                    'decision_table_size': size_stats['decision_size']
                }
                
        except Exception as e:
            logger.error(f"获取数据统计失败: {e}")
            raise
    
    def _calculate_retention_date(self) -> datetime:
        """计算保留日期"""
        return datetime.now() - timedelta(days=self.retention_days)


class ComplianceReportGenerator:
    """合规报告生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_interface = AuditQueryInterface(config)
    
    async def generate_compliance_report(self, period_start: datetime, 
                                       period_end: datetime) -> ComplianceReport:
        """生成合规报告"""
        try:
            await self.query_interface.connect()
            
            # 查询期间内的所有记录
            records = await self.query_interface.query_by_time_range(
                period_start, period_end
            )
            
            # 统计决策数量
            decision_records = [r for r in records if r.event_type == 'trading_decision']
            total_decisions = len(decision_records)
            
            # 分析风险违规
            risk_violations = [r for r in records if r.event_type == 'risk_violation']
            violation_analysis = self._analyze_violations(risk_violations)
            
            # 分析持仓集中度
            concentration_analysis = await self._analyze_concentration(decision_records)
            
            # 计算模型性能
            model_performance = await self._calculate_model_performance(decision_records)
            
            # 计算合规分数
            compliance_score = self._calculate_compliance_score(
                total_decisions, len(risk_violations), concentration_analysis
            )
            
            report = ComplianceReport(
                report_id=str(uuid.uuid4()),
                generated_at=datetime.now(),
                period_start=period_start,
                period_end=period_end,
                total_decisions=total_decisions,
                risk_violations=violation_analysis,
                concentration_analysis=concentration_analysis,
                model_performance=model_performance,
                compliance_score=compliance_score
            )
            
            # 保存报告
            await self._save_report(report)
            
            return report
            
        finally:
            await self.query_interface.disconnect()
    
    def _analyze_violations(self, violations: List[AuditRecord]) -> List[Dict[str, Any]]:
        """分析违规情况"""
        violation_summary = {}
        
        for violation in violations:
            violation_type = violation.data.get('violation_type', 'unknown')
            if violation_type not in violation_summary:
                violation_summary[violation_type] = {
                    'count': 0,
                    'severity_counts': {'low': 0, 'medium': 0, 'high': 0}
                }
            
            violation_summary[violation_type]['count'] += 1
            severity = violation.metadata.get('severity', 'medium')
            violation_summary[violation_type]['severity_counts'][severity] += 1
        
        return [
            {
                'type': vtype,
                'count': vdata['count'],
                'severity_distribution': vdata['severity_counts']
            }
            for vtype, vdata in violation_summary.items()
        ]
    
    async def _analyze_concentration(self, decisions: List[AuditRecord]) -> Dict[str, float]:
        """分析持仓集中度"""
        if not decisions:
            return {}
        
        concentrations = []
        max_weights = []
        
        for decision in decisions:
            # 从决策记录中提取集中度信息
            decision_id = decision.data.get('decision_id')
            if decision_id:
                decision_detail = await self.query_interface.get_decision_details(decision_id)
                if decision_detail:
                    concentration = decision_detail.risk_metrics.get('concentration', 0)
                    max_weight = decision_detail.risk_metrics.get('max_weight', 0)
                    concentrations.append(concentration)
                    max_weights.append(max_weight)
        
        if concentrations:
            return {
                'avg_concentration': np.mean(concentrations),
                'max_concentration': np.max(concentrations),
                'min_concentration': np.min(concentrations),
                'avg_max_weight': np.mean(max_weights),
                'max_single_weight': np.max(max_weights)
            }
        
        return {}
    
    async def _calculate_model_performance(self, decisions: List[AuditRecord]) -> Dict[str, float]:
        """计算模型性能"""
        if not decisions:
            return {}
        
        confidences = []
        execution_times = []
        
        for decision in decisions:
            confidence = decision.data.get('confidence', 0)
            confidences.append(confidence)
            
            execution_time = decision.metadata.get('execution_time_ms', 0)
            if execution_time:
                execution_times.append(execution_time)
        
        performance = {
            'avg_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
        if execution_times:
            performance.update({
                'avg_execution_time_ms': np.mean(execution_times),
                'max_execution_time_ms': np.max(execution_times),
                'min_execution_time_ms': np.min(execution_times)
            })
        
        return performance
    
    def _calculate_compliance_score(self, total_decisions: int, violation_count: int,
                                  concentration_analysis: Dict[str, float]) -> float:
        """计算合规分数"""
        if total_decisions == 0:
            return 1.0
        
        # 基础分数
        base_score = 1.0
        
        # 违规惩罚
        violation_penalty = min(0.5, violation_count / total_decisions)
        base_score -= violation_penalty
        
        # 集中度惩罚
        if concentration_analysis:
            max_concentration = concentration_analysis.get('max_concentration', 0)
            if max_concentration > 0.5:  # 集中度过高
                concentration_penalty = min(0.3, (max_concentration - 0.5) * 0.6)
                base_score -= concentration_penalty
        
        return max(0.0, base_score)
    
    async def _save_report(self, report: ComplianceReport):
        """保存合规报告"""
        try:
            async with self.query_interface.relational_db.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO compliance_reports 
                    (report_id, generated_at, period_start, period_end, total_decisions,
                     risk_violations, concentration_analysis, model_performance, compliance_score)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''',
                    report.report_id, report.generated_at, report.period_start,
                    report.period_end, report.total_decisions,
                    json.dumps(report.risk_violations),
                    json.dumps(report.concentration_analysis),
                    json.dumps(report.model_performance),
                    report.compliance_score
                )
                
        except Exception as e:
            logger.error(f"保存合规报告失败: {e}")
            raise