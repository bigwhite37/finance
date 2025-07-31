-- 交易系统数据库初始化脚本

-- 创建审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    system_name VARCHAR(100),
    user_id VARCHAR(100),
    event_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建交易记录表
CREATE TABLE IF NOT EXISTS trading_records (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    system_name VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    action_type VARCHAR(10) NOT NULL, -- 'buy', 'sell'
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 4) NOT NULL,
    commission DECIMAL(10, 4) DEFAULT 0,
    stamp_tax DECIMAL(10, 4) DEFAULT 0,
    slippage DECIMAL(10, 4) DEFAULT 0,
    total_cost DECIMAL(10, 4) NOT NULL,
    portfolio_value DECIMAL(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建模型版本表
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL UNIQUE,
    model_type VARCHAR(50) NOT NULL,
    model_path VARCHAR(255) NOT NULL,
    performance_metrics JSONB,
    status VARCHAR(20) DEFAULT 'inactive', -- 'active', 'inactive', 'testing', 'deprecated'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建系统配置表
CREATE TABLE IF NOT EXISTS system_configs (
    id SERIAL PRIMARY KEY,
    system_name VARCHAR(100) NOT NULL,
    config_key VARCHAR(100) NOT NULL,
    config_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(system_name, config_key)
);

-- 创建告警记录表
CREATE TABLE IF NOT EXISTS alert_records (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    system_name VARCHAR(100) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    alert_level VARCHAR(20) NOT NULL, -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message TEXT NOT NULL,
    metric_name VARCHAR(100),
    metric_value DECIMAL(15, 6),
    threshold_value DECIMAL(15, 6),
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'acknowledged', 'resolved'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- 创建性能指标表
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    system_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15, 6) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_system_name ON audit_logs(system_name);

CREATE INDEX IF NOT EXISTS idx_trading_records_timestamp ON trading_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_trading_records_system_name ON trading_records(system_name);
CREATE INDEX IF NOT EXISTS idx_trading_records_symbol ON trading_records(symbol);

CREATE INDEX IF NOT EXISTS idx_model_versions_status ON model_versions(status);
CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON model_versions(created_at);

CREATE INDEX IF NOT EXISTS idx_alert_records_timestamp ON alert_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_alert_records_system_name ON alert_records(system_name);
CREATE INDEX IF NOT EXISTS idx_alert_records_status ON alert_records(status);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_system_name ON performance_metrics(system_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_metric_name ON performance_metrics(metric_name);

-- 插入初始数据
INSERT INTO model_versions (version, model_type, model_path, status) 
VALUES ('v1.0.0', 'sac_transformer', '/app/checkpoints/best_model.pth', 'active')
ON CONFLICT (version) DO NOTHING;

-- 创建视图
CREATE OR REPLACE VIEW trading_summary AS
SELECT 
    system_name,
    DATE(timestamp) as trading_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN action_type = 'buy' THEN quantity ELSE 0 END) as total_buy_quantity,
    SUM(CASE WHEN action_type = 'sell' THEN quantity ELSE 0 END) as total_sell_quantity,
    SUM(total_cost) as total_transaction_cost,
    AVG(portfolio_value) as avg_portfolio_value
FROM trading_records
GROUP BY system_name, DATE(timestamp)
ORDER BY system_name, trading_date DESC;

-- 创建函数：清理旧数据
CREATE OR REPLACE FUNCTION cleanup_old_data(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- 清理旧的审计日志
    DELETE FROM audit_logs 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * retention_days;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- 清理旧的性能指标
    DELETE FROM performance_metrics 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * retention_days;
    
    -- 清理已解决的旧告警
    DELETE FROM alert_records 
    WHERE status = 'resolved' 
    AND resolved_at < CURRENT_TIMESTAMP - INTERVAL '1 day' * (retention_days / 2);
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 创建定时清理任务（需要pg_cron扩展）
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data(90);');

COMMIT;