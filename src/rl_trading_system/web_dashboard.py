"""
Web仪表板

提供Web界面来监控和管理交易系统
"""

import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
import plotly.graph_objs as go
import plotly.utils

from .system_integration import system_manager, SystemConfig, SystemState


class WebDashboard:
    """Web仪表板"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.secret_key = "trading_system_dashboard"
        
        # 设置路由
        self._setup_routes()
        
        # 历史数据存储
        self.history_data: Dict[str, List[Dict]] = {}
        self.max_history_points = 1000
        
        # 启动数据收集线程
        self.data_collection_thread = threading.Thread(target=self._collect_data, daemon=True)
        self.data_collection_thread.start()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            """主页"""
            systems = system_manager.list_systems()
            system_statuses = {}
            
            for name in systems:
                status = system_manager.get_system_status(name)
                if status:
                    system_statuses[name] = status
            
            return render_template_string(INDEX_TEMPLATE, 
                                        systems=system_statuses,
                                        current_time=datetime.now())
        
        @self.app.route('/system/<name>')
        def system_detail(name):
            """系统详情页"""
            status = system_manager.get_system_status(name)
            if not status:
                return "系统不存在", 404
            
            # 获取历史数据
            history = self.history_data.get(name, [])
            
            # 生成图表
            charts = self._generate_charts(name, history)
            
            return render_template_string(SYSTEM_DETAIL_TEMPLATE,
                                        name=name,
                                        status=status,
                                        charts=charts,
                                        current_time=datetime.now())
        
        @self.app.route('/api/systems')
        def api_systems():
            """API: 获取所有系统状态"""
            systems = system_manager.list_systems()
            result = {}
            
            for name in systems:
                status = system_manager.get_system_status(name)
                if status:
                    result[name] = status
            
            return jsonify(result)
        
        @self.app.route('/api/system/<name>')
        def api_system_status(name):
            """API: 获取特定系统状态"""
            status = system_manager.get_system_status(name)
            if not status:
                return jsonify({'error': '系统不存在'}), 404
            
            return jsonify(status)
        
        @self.app.route('/api/system/<name>/start', methods=['POST'])
        def api_start_system(name):
            """API: 启动系统"""
            success = system_manager.start_system(name)
            return jsonify({'success': success})
        
        @self.app.route('/api/system/<name>/stop', methods=['POST'])
        def api_stop_system(name):
            """API: 停止系统"""
            success = system_manager.stop_system(name)
            return jsonify({'success': success})
        
        @self.app.route('/api/system/<name>/history')
        def api_system_history(name):
            """API: 获取系统历史数据"""
            history = self.history_data.get(name, [])
            return jsonify(history)
        
        @self.app.route('/create_system', methods=['GET', 'POST'])
        def create_system():
            """创建系统页面"""
            if request.method == 'POST':
                try:
                    # 获取表单数据
                    name = request.form['name']
                    stock_pool = request.form['stock_pool'].split(',')
                    initial_cash = float(request.form['initial_cash'])
                    
                    # 创建配置
                    config = SystemConfig(
                        stock_pool=[s.strip() for s in stock_pool],
                        initial_cash=initial_cash,
                        update_frequency=request.form.get('update_frequency', '1D'),
                        enable_monitoring=True,
                        enable_audit=True,
                        enable_risk_control=True
                    )
                    
                    # 创建系统
                    success = system_manager.create_system(name, config)
                    
                    if success:
                        return redirect(url_for('index'))
                    else:
                        return render_template_string(CREATE_SYSTEM_TEMPLATE, 
                                                    error="系统创建失败")
                
                except Exception as e:
                    return render_template_string(CREATE_SYSTEM_TEMPLATE, 
                                                error=f"创建失败: {str(e)}")
            
            return render_template_string(CREATE_SYSTEM_TEMPLATE)
    
    def _collect_data(self):
        """收集历史数据"""
        while True:
            try:
                systems = system_manager.list_systems()
                current_time = datetime.now()
                
                for name in systems:
                    status = system_manager.get_system_status(name)
                    if status and status['state'] == SystemState.RUNNING.value:
                        # 记录历史数据点
                        data_point = {
                            'timestamp': current_time.isoformat(),
                            'portfolio_value': status['portfolio_value'],
                            'total_return': status['stats']['total_return'],
                            'positions': status['current_positions']
                        }
                        
                        if name not in self.history_data:
                            self.history_data[name] = []
                        
                        self.history_data[name].append(data_point)
                        
                        # 限制历史数据点数量
                        if len(self.history_data[name]) > self.max_history_points:
                            self.history_data[name] = self.history_data[name][-self.max_history_points:]
                
                # 每30秒收集一次数据
                threading.Event().wait(30)
                
            except Exception as e:
                print(f"数据收集异常: {e}")
                threading.Event().wait(60)  # 出错时等待更长时间
    
    def _generate_charts(self, system_name: str, history: List[Dict]) -> Dict[str, str]:
        """生成图表"""
        charts = {}
        
        if not history:
            return charts
        
        # 提取时间序列数据
        timestamps = [datetime.fromisoformat(point['timestamp']) for point in history]
        portfolio_values = [point['portfolio_value'] for point in history]
        total_returns = [point['total_return'] for point in history]
        
        # 1. 组合价值图表
        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(go.Scatter(
            x=timestamps,
            y=portfolio_values,
            mode='lines',
            name='组合价值',
            line=dict(color='blue', width=2)
        ))
        portfolio_fig.update_layout(
            title='组合价值变化',
            xaxis_title='时间',
            yaxis_title='价值 (元)',
            height=400
        )
        charts['portfolio_value'] = json.dumps(portfolio_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 2. 收益率图表
        return_fig = go.Figure()
        return_fig.add_trace(go.Scatter(
            x=timestamps,
            y=[r * 100 for r in total_returns],  # 转换为百分比
            mode='lines',
            name='总收益率',
            line=dict(color='green', width=2)
        ))
        return_fig.update_layout(
            title='收益率变化',
            xaxis_title='时间',
            yaxis_title='收益率 (%)',
            height=400
        )
        charts['total_return'] = json.dumps(return_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 3. 持仓分布图表（使用最新数据）
        if history:
            latest_positions = history[-1]['positions']
            # 假设我们知道股票名称
            stock_names = [f'股票{i+1}' for i in range(len(latest_positions))]
            
            position_fig = go.Figure(data=[go.Pie(
                labels=stock_names,
                values=latest_positions,
                hole=0.3
            )])
            position_fig.update_layout(
                title='当前持仓分布',
                height=400
            )
            charts['positions'] = json.dumps(position_fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return charts
    
    def run(self, debug: bool = False):
        """运行Web服务器"""
        print(f"启动Web仪表板: http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


# HTML模板
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>交易系统仪表板</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <style>
        .system-card { margin-bottom: 20px; }
        .status-running { color: #28a745; }
        .status-stopped { color: #dc3545; }
        .status-paused { color: #ffc107; }
        .status-error { color: #dc3545; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">交易系统仪表板</span>
            <span class="navbar-text">{{ current_time.strftime('%Y-%m-%d %H:%M:%S') }}</span>
        </div>
    </nav>
    
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2>系统概览</h2>
                    <a href="/create_system" class="btn btn-primary">创建新系统</a>
                </div>
                
                {% if not systems %}
                <div class="alert alert-info">
                    <h4>没有运行的系统</h4>
                    <p>点击"创建新系统"开始使用交易系统。</p>
                </div>
                {% else %}
                <div class="row">
                    {% for name, status in systems.items() %}
                    <div class="col-md-6 col-lg-4">
                        <div class="card system-card">
                            <div class="card-header">
                                <h5 class="card-title">{{ name }}</h5>
                                <span class="badge bg-{% if status.state == 'running' %}success{% elif status.state == 'stopped' %}secondary{% elif status.state == 'paused' %}warning{% else %}danger{% endif %}">
                                    {{ status.state }}
                                </span>
                            </div>
                            <div class="card-body">
                                <p><strong>组合价值:</strong> ¥{{ "%.2f"|format(status.portfolio_value) }}</p>
                                <p><strong>总收益:</strong> {{ "%.2f%%"|format(status.stats.total_return * 100) }}</p>
                                <p><strong>交易次数:</strong> {{ status.stats.total_trades }}</p>
                                <div class="btn-group" role="group">
                                    <a href="/system/{{ name }}" class="btn btn-info btn-sm">详情</a>
                                    {% if status.state == 'stopped' %}
                                    <button class="btn btn-success btn-sm" onclick="startSystem('{{ name }}')">启动</button>
                                    {% elif status.state == 'running' %}
                                    <button class="btn btn-danger btn-sm" onclick="stopSystem('{{ name }}')">停止</button>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <script>
        function startSystem(name) {
            fetch(`/api/system/${name}/start`, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    } else {
                        alert('启动失败');
                    }
                });
        }
        
        function stopSystem(name) {
            if (confirm('确定要停止系统吗？')) {
                fetch(`/api/system/${name}/stop`, {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            location.reload();
                        } else {
                            alert('停止失败');
                        }
                    });
            }
        }
        
        // 自动刷新页面
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""

SYSTEM_DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ name }} - 系统详情</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">交易系统仪表板</a>
            <span class="navbar-text">{{ current_time.strftime('%Y-%m-%d %H:%M:%S') }}</span>
        </div>
    </nav>
    
    <div class="container mt-4">
        <div class="row">
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h2>{{ name }}</h2>
                    <div class="btn-group">
                        {% if status.state == 'stopped' %}
                        <button class="btn btn-success" onclick="startSystem()">启动</button>
                        {% elif status.state == 'running' %}
                        <button class="btn btn-danger" onclick="stopSystem()">停止</button>
                        {% endif %}
                        <a href="/" class="btn btn-secondary">返回</a>
                    </div>
                </div>
                
                <!-- 系统状态 -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5>系统状态</h5>
                                <span class="badge bg-{% if status.state == 'running' %}success{% elif status.state == 'stopped' %}secondary{% elif status.state == 'paused' %}warning{% else %}danger{% endif %} fs-6">
                                    {{ status.state }}
                                </span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5>组合价值</h5>
                                <h4>¥{{ "%.2f"|format(status.portfolio_value) }}</h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5>总收益</h5>
                                <h4 class="{% if status.stats.total_return >= 0 %}text-success{% else %}text-danger{% endif %}">
                                    {{ "%.2f%%"|format(status.stats.total_return * 100) }}
                                </h4>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card">
                            <div class="card-body text-center">
                                <h5>交易次数</h5>
                                <h4>{{ status.stats.total_trades }}</h4>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- 图表 -->
                {% if charts %}
                <div class="row">
                    {% if 'portfolio_value' in charts %}
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <div id="portfolio-chart"></div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                    
                    {% if 'total_return' in charts %}
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <div id="return-chart"></div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                    
                    {% if 'positions' in charts %}
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <div id="position-chart"></div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>
                {% endif %}
                
                <!-- 详细信息 -->
                <div class="row mt-4">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>当前持仓</h5>
                            </div>
                            <div class="card-body">
                                {% for i, position in enumerate(status.current_positions) %}
                                <div class="d-flex justify-content-between">
                                    <span>股票{{ i+1 }}:</span>
                                    <span>{{ "%.2f%%"|format(position * 100) }}</span>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                    
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header">
                                <h5>统计信息</h5>
                            </div>
                            <div class="card-body">
                                {% for key, value in status.stats.items() %}
                                <div class="d-flex justify-content-between">
                                    <span>{{ key }}:</span>
                                    <span>{{ value }}</span>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 渲染图表
        {% if charts %}
        {% if 'portfolio_value' in charts %}
        Plotly.newPlot('portfolio-chart', {{ charts.portfolio_value|safe }});
        {% endif %}
        
        {% if 'total_return' in charts %}
        Plotly.newPlot('return-chart', {{ charts.total_return|safe }});
        {% endif %}
        
        {% if 'positions' in charts %}
        Plotly.newPlot('position-chart', {{ charts.positions|safe }});
        {% endif %}
        {% endif %}
        
        function startSystem() {
            fetch(`/api/system/{{ name }}/start`, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    } else {
                        alert('启动失败');
                    }
                });
        }
        
        function stopSystem() {
            if (confirm('确定要停止系统吗？')) {
                fetch(`/api/system/{{ name }}/stop`, {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            location.reload();
                        } else {
                            alert('停止失败');
                        }
                    });
            }
        }
        
        // 自动刷新页面
        setTimeout(() => location.reload(), 60000);
    </script>
</body>
</html>
"""

CREATE_SYSTEM_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>创建交易系统</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">交易系统仪表板</a>
        </div>
    </nav>
    
    <div class="container mt-4">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h4>创建新的交易系统</h4>
                    </div>
                    <div class="card-body">
                        {% if error %}
                        <div class="alert alert-danger">{{ error }}</div>
                        {% endif %}
                        
                        <form method="POST">
                            <div class="mb-3">
                                <label for="name" class="form-label">系统名称</label>
                                <input type="text" class="form-control" id="name" name="name" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="stock_pool" class="form-label">股票池</label>
                                <input type="text" class="form-control" id="stock_pool" name="stock_pool" 
                                       value="000001.SZ,000002.SZ,600000.SH" required>
                                <div class="form-text">用逗号分隔股票代码</div>
                            </div>
                            
                            <div class="mb-3">
                                <label for="initial_cash" class="form-label">初始资金</label>
                                <input type="number" class="form-control" id="initial_cash" name="initial_cash" 
                                       value="1000000" step="1000" required>
                            </div>
                            
                            <div class="mb-3">
                                <label for="update_frequency" class="form-label">更新频率</label>
                                <select class="form-select" id="update_frequency" name="update_frequency">
                                    <option value="1D" selected>每日</option>
                                    <option value="1H">每小时</option>
                                    <option value="5min">每5分钟</option>
                                </select>
                            </div>
                            
                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary">创建系统</button>
                                <a href="/" class="btn btn-secondary">取消</a>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


def create_dashboard(host: str = "127.0.0.1", port: int = 5000) -> WebDashboard:
    """创建Web仪表板实例"""
    return WebDashboard(host, port)


if __name__ == "__main__":
    dashboard = create_dashboard()
    dashboard.run(debug=True)