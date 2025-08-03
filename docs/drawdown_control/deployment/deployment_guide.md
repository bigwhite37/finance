# 部署指南

## 概述

本文档提供回撤控制系统在生产环境中的完整部署指南，包括环境准备、系统安装、配置调优、服务启动和监控设置等步骤。

## 系统要求

### 硬件要求

#### 最低配置
- **CPU**: 4核心 Intel/AMD x64 处理器
- **内存**: 8GB RAM
- **存储**: 50GB 可用磁盘空间 (SSD推荐)
- **网络**: 100Mbps 网络连接

#### 推荐配置
- **CPU**: 8核心以上 Intel/AMD x64 处理器
- **内存**: 16GB RAM 以上
- **存储**: 200GB SSD
- **网络**: 1Gbps 网络连接
- **GPU**: NVIDIA GPU (可选，用于加速计算)

#### 大规模生产环境
- **CPU**: 16核心以上或多节点集群
- **内存**: 32GB RAM 以上
- **存储**: 500GB+ SSD，支持RAID
- **网络**: 10Gbps 网络连接
- **负载均衡**: 支持高可用性配置

### 软件要求

#### 操作系统
- **Linux**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+ (推荐)
- **Windows**: Windows Server 2019+ (支持，但不推荐生产环境)
- **macOS**: macOS 11+ (仅用于开发和测试)

#### 运行时环境
```bash
# Python 环境
Python 3.8+
pip 21.0+

# 系统依赖
gcc/g++ 9.0+
cmake 3.16+
git 2.25+

# 可选依赖
Docker 20.10+
Kubernetes 1.20+
Redis 6.0+
PostgreSQL 12+
```

## 安装步骤

### 1. 环境准备

#### 创建用户和目录
```bash
# 创建系统用户
sudo useradd -m -s /bin/bash drawdown-control
sudo usermod -aG sudo drawdown-control

# 创建应用目录
sudo mkdir -p /opt/drawdown-control
sudo mkdir -p /var/log/drawdown-control
sudo mkdir -p /etc/drawdown-control
sudo mkdir -p /var/lib/drawdown-control

# 设置目录权限
sudo chown -R drawdown-control:drawdown-control /opt/drawdown-control
sudo chown -R drawdown-control:drawdown-control /var/log/drawdown-control
sudo chown -R drawdown-control:drawdown-control /var/lib/drawdown-control
```

#### 安装系统依赖
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.8 python3.8-dev python3-pip
sudo apt install -y build-essential cmake git
sudo apt install -y postgresql-client redis-tools
sudo apt install -y supervisor nginx

# CentOS/RHEL
sudo yum update -y
sudo yum install -y python38 python38-devel python38-pip
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git
sudo yum install -y postgresql redis
sudo yum install -y supervisor nginx
```

### 2. 应用安装

#### 下载和安装应用
```bash
# 切换到应用用户
sudo su - drawdown-control

# 下载应用代码
cd /opt/drawdown-control
git clone https://github.com/your-org/drawdown-control-system.git .

# 创建Python虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 安装Python依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### 配置文件设置
```bash
# 复制配置模板
cp config/production.yaml.template /etc/drawdown-control/config.yaml

# 编辑配置文件
sudo nano /etc/drawdown-control/config.yaml
```

### 3. 数据库设置

#### PostgreSQL 配置
```bash
# 安装 PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# 创建数据库和用户
sudo -u postgres psql << EOF
CREATE DATABASE drawdown_control;
CREATE USER drawdown_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE drawdown_control TO drawdown_user;
\q
EOF

# 初始化数据库表
cd /opt/drawdown-control
source venv/bin/activate
python -m rl_trading_system.database.init_db
```

#### Redis 配置
```bash
# 安装和启动 Redis
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# 配置 Redis
sudo nano /etc/redis/redis.conf
# 设置以下参数：
# maxmemory 2gb
# maxmemory-policy allkeys-lru
# save 900 1
# save 300 10

sudo systemctl restart redis-server
```

### 4. 服务配置

#### Supervisor 配置
创建 Supervisor 配置文件：

```ini
# /etc/supervisor/conf.d/drawdown-control.conf
[program:drawdown-control-api]
command=/opt/drawdown-control/venv/bin/python -m rl_trading_system.api.server
directory=/opt/drawdown-control
user=drawdown-control
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/drawdown-control/api.log
environment=PYTHONPATH="/opt/drawdown-control"

[program:drawdown-control-monitor]
command=/opt/drawdown-control/venv/bin/python -m rl_trading_system.monitoring.service
directory=/opt/drawdown-control
user=drawdown-control
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/drawdown-control/monitor.log
environment=PYTHONPATH="/opt/drawdown-control"

[program:drawdown-control-worker]
command=/opt/drawdown-control/venv/bin/python -m rl_trading_system.workers.main
directory=/opt/drawdown-control
user=drawdown-control
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/drawdown-control/worker.log
environment=PYTHONPATH="/opt/drawdown-control"
numprocs=4
process_name=%(program_name)s_%(process_num)02d
```

#### Nginx 配置
创建 Nginx 配置文件：

```nginx
# /etc/nginx/sites-available/drawdown-control
server {
    listen 80;
    server_name your-domain.com;
    
    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL 配置
    ssl_certificate /etc/ssl/certs/drawdown-control.crt;
    ssl_certificate_key /etc/ssl/private/drawdown-control.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # API 代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
    
    # WebSocket 支持
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # 静态文件
    location /static/ {
        alias /opt/drawdown-control/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # 健康检查
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
```

### 5. 启动服务

#### 启动和启用服务
```bash
# 重新加载 Supervisor 配置
sudo supervisorctl reread
sudo supervisorctl update

# 启动所有服务
sudo supervisorctl start drawdown-control-api
sudo supervisorctl start drawdown-control-monitor
sudo supervisorctl start drawdown-control-worker:*

# 启用 Nginx 站点
sudo ln -s /etc/nginx/sites-available/drawdown-control /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# 设置服务自启动
sudo systemctl enable supervisor
sudo systemctl enable nginx
sudo systemctl enable postgresql
sudo systemctl enable redis-server
```

#### 验证部署
```bash
# 检查服务状态
sudo supervisorctl status

# 检查日志
tail -f /var/log/drawdown-control/api.log
tail -f /var/log/drawdown-control/monitor.log

# 测试 API 接口
curl -X GET http://localhost/api/health
curl -X GET https://your-domain.com/api/health
```

## 高可用性部署

### 1. 负载均衡配置

#### HAProxy 配置
```haproxy
# /etc/haproxy/haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend drawdown_control_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/drawdown-control.pem
    redirect scheme https if !{ ssl_fc }
    default_backend drawdown_control_backend

backend drawdown_control_backend
    balance roundrobin
    option httpchk GET /api/health
    server app1 10.0.1.10:8000 check
    server app2 10.0.1.11:8000 check
    server app3 10.0.1.12:8000 check
```

### 2. 数据库集群

#### PostgreSQL 主从配置
```bash
# 主服务器配置
# /etc/postgresql/12/main/postgresql.conf
wal_level = replica
max_wal_senders = 3
wal_keep_segments = 64
archive_mode = on
archive_command = 'test ! -f /var/lib/postgresql/12/main/archive/%f && cp %p /var/lib/postgresql/12/main/archive/%f'

# /etc/postgresql/12/main/pg_hba.conf
host replication replicator 10.0.1.0/24 md5

# 从服务器配置
# recovery.conf
standby_mode = on
primary_conninfo = 'host=10.0.1.10 port=5432 user=replicator'
```

### 3. Redis 集群

#### Redis Sentinel 配置
```redis
# /etc/redis/sentinel.conf
port 26379
sentinel monitor mymaster 10.0.1.10 6379 2
sentinel down-after-milliseconds mymaster 30000
sentinel parallel-syncs mymaster 1
sentinel failover-timeout mymaster 180000
```

## 监控和告警

### 1. Prometheus 配置

#### 应用指标配置
```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'drawdown-control'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

### 2. Grafana 仪表板

#### 系统监控仪表板
```json
{
  "dashboard": {
    "id": null,
    "title": "回撤控制系统监控",
    "panels": [
      {
        "title": "当前回撤水平",
        "type": "stat",
        "targets": [
          {
            "expr": "drawdown_current_level",
            "legendFormat": "当前回撤"
          }
        ]
      },
      {
        "title": "API 响应时间",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
            "legendFormat": "平均响应时间"
          }
        ]
      }
    ]
  }
}
```

### 3. 告警规则

#### Prometheus 告警规则
```yaml
# /etc/prometheus/rules/drawdown-control.yml
groups:
  - name: drawdown-control
    rules:
      - alert: HighDrawdown
        expr: drawdown_current_level > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "回撤水平过高"
          description: "当前回撤水平为 {{ $value }}"
          
      - alert: CriticalDrawdown
        expr: drawdown_current_level > 0.15
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "严重回撤警告"
          description: "当前回撤水平为 {{ $value }}，已触发严重警告"
          
      - alert: APIHighLatency
        expr: rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m]) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "API 响应时间过长"
          description: "API 平均响应时间为 {{ $value }} 秒"
```

## 安全配置

### 1. 防火墙设置

```bash
# UFW 防火墙配置
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.1.0/24 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.1.0/24 to any port 6379  # Redis
sudo ufw enable
```

### 2. SSL/TLS 配置

```bash
# 生成 SSL 证书 (Let's Encrypt)
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# 或使用自签名证书
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/drawdown-control.key \
    -out /etc/ssl/certs/drawdown-control.crt
```

### 3. 访问控制

```python
# 应用层访问控制配置
SECURITY_CONFIG = {
    "authentication": {
        "enabled": True,
        "method": "jwt",
        "secret_key": os.environ["JWT_SECRET_KEY"],
        "expiry_hours": 24
    },
    "authorization": {
        "enabled": True,
        "roles": ["admin", "operator", "viewer"],
        "permissions": {
            "admin": ["read", "write", "delete", "admin"],
            "operator": ["read", "write"],
            "viewer": ["read"]
        }
    },
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 1000,
        "burst_size": 100
    }
}
```

## 备份和恢复

### 1. 数据备份策略

```bash
#!/bin/bash
# /opt/drawdown-control/scripts/backup.sh

# 数据库备份
pg_dump -h localhost -U drawdown_user drawdown_control > \
    /var/backups/drawdown-control/db_$(date +%Y%m%d_%H%M%S).sql

# 配置文件备份
tar -czf /var/backups/drawdown-control/config_$(date +%Y%m%d_%H%M%S).tar.gz \
    /etc/drawdown-control/

# 日志文件备份
tar -czf /var/backups/drawdown-control/logs_$(date +%Y%m%d_%H%M%S).tar.gz \
    /var/log/drawdown-control/

# 清理旧备份 (保留30天)
find /var/backups/drawdown-control/ -name "*.sql" -mtime +30 -delete
find /var/backups/drawdown-control/ -name "*.tar.gz" -mtime +30 -delete
```

### 2. 自动化备份

```bash
# 添加到 crontab
sudo crontab -e

# 每天凌晨2点执行备份
0 2 * * * /opt/drawdown-control/scripts/backup.sh

# 每周日凌晨3点执行完整备份
0 3 * * 0 /opt/drawdown-control/scripts/full_backup.sh
```

## 故障排除

### 常见问题和解决方案

#### 1. 服务启动失败
```bash
# 检查服务状态
sudo supervisorctl status
sudo systemctl status nginx

# 查看错误日志
sudo tail -f /var/log/drawdown-control/api.log
sudo tail -f /var/log/nginx/error.log

# 检查配置文件语法
sudo nginx -t
python -m rl_trading_system.config.validate
```

#### 2. 数据库连接问题
```bash
# 检查数据库连接
psql -h localhost -U drawdown_user -d drawdown_control

# 检查数据库状态
sudo systemctl status postgresql
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"
```

#### 3. 性能问题
```bash
# 检查系统资源使用
htop
iostat -x 1
free -h

# 检查应用性能指标
curl http://localhost:8000/metrics
```

## 维护计划

### 定期维护任务

#### 每日任务
- 检查系统日志
- 验证备份完成
- 监控系统资源使用
- 检查告警状态

#### 每周任务
- 更新系统安全补丁
- 清理临时文件和日志
- 检查数据库性能
- 验证监控系统正常

#### 每月任务
- 完整系统备份
- 性能基准测试
- 安全审计
- 容量规划评估

#### 每季度任务
- 系统版本升级
- 配置优化审查
- 灾难恢复演练
- 文档更新

## 相关文档

- [监控告警配置](./monitoring_setup.md) - 详细监控配置
- [故障排除手册](./troubleshooting.md) - 常见问题解决
- [维护手册](./maintenance.md) - 日常维护指南
- [安全配置指南](../security/security_architecture.md) - 安全最佳实践