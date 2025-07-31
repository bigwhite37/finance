"""
API服务器实现
实现标准化的REST API和数据格式，API路由、请求处理和响应格式化，标准化错误码和错误信息返回
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import time
import threading
from abc import ABC, abstractmethod
import hashlib
import secrets
import logging


class HTTPMethod(Enum):
    """HTTP方法枚举"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class APIStatus(Enum):
    """API状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"


@dataclass
class APIEndpoint:
    """API端点定义"""
    path: str
    methods: List[HTTPMethod]
    handler: Callable
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    auth_required: bool = True
    rate_limit: Optional[Dict[str, int]] = None
    version: str = "v1"
    status: APIStatus = APIStatus.ACTIVE
    tags: List[str] = field(default_factory=list)


@dataclass
class APIConfig:
    """API配置"""
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    max_request_size: int = 1024 * 1024  # 1MB
    timeout: int = 30
    cors_enabled: bool = True
    api_version: str = "v1"
    documentation_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    max_concurrent_requests: int = 50
    authentication_required: bool = True


@dataclass
class APIError:
    """API错误定义"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    status_code: int = 400
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class APIResponse:
    """API响应定义"""
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""
    success: bool = True
    data: Optional[Any] = None
    error: Optional[APIError] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0

    def __post_init__(self):
        if self.headers.get('Content-Type') is None:
            self.headers['Content-Type'] = 'application/json'


class RequestValidator:
    """请求验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_trading_request(self, request_data: Dict[str, Any]) -> 'ValidationResult':
        """验证交易请求"""
        errors = []
        
        # 验证必需字段
        required_fields = ['symbol', 'action', 'quantity', 'price']
        for field in required_fields:
            if not request_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # 验证符号
        symbol = request_data.get('symbol', '')
        if not symbol or len(symbol) < 1:
            errors.append("Symbol cannot be empty")
        
        # 验证动作
        action = request_data.get('action', '')
        if action not in ['BUY', 'SELL']:
            errors.append("Invalid action: must be 'BUY' or 'SELL'")
        
        # 验证数量
        quantity = request_data.get('quantity', 0)
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            errors.append("Invalid quantity: must be a positive number")
        
        # 验证价格
        price = request_data.get('price', 0)
        if not isinstance(price, (int, float)) or price <= 0:
            errors.append("Invalid price: must be a positive number")
        
        # 验证订单类型
        order_type = request_data.get('order_type')
        if order_type and order_type not in ['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']:
            errors.append("Invalid order type")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def validate_date_range(self, start_date, end_date) -> 'ValidationResult':
        """验证日期范围"""
        errors = []
        
        if start_date > end_date:
            errors.append("Start date must be before end date")
        
        # 检查日期范围是否合理（不超过5年）
        max_range = timedelta(days=365 * 5)
        if end_date - start_date > max_range:
            errors.append("Date range cannot exceed 5 years")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
    
    def validate_pagination_params(self, page: int, limit: int) -> 'ValidationResult':
        """验证分页参数"""
        errors = []
        
        if page < 1:
            errors.append("Page must be greater than 0")
        
        if limit < 1 or limit > 1000:
            errors.append("Limit must be between 1 and 1000")
        
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)


class ResponseFormatter:
    """响应格式化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_success_response(self, data: Any, status_code: int = 200) -> APIResponse:
        """格式化成功响应"""
        response_body = {
            'success': True,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'version': 'v1'
        }
        
        return APIResponse(
            status_code=status_code,
            body=json.dumps(response_body),
            success=True,
            data=data,
            timestamp=datetime.now()
        )
    
    def format_error_response(self, error: APIError, status_code: int = 400) -> APIResponse:
        """格式化错误响应"""
        response_body = {
            'success': False,
            'error': {
                'code': error.code,
                'message': error.message,
                'details': error.details or {},
                'timestamp': error.timestamp.isoformat()
            },
            'version': 'v1'
        }
        
        return APIResponse(
            status_code=status_code,
            body=json.dumps(response_body),
            success=False,
            error=error,
            timestamp=datetime.now()
        )
    
    def format_paginated_response(self, items: List[Any], total_count: int, 
                                page: int, limit: int) -> APIResponse:
        """格式化分页响应"""
        total_pages = (total_count + limit - 1) // limit
        
        data = {
            'items': items,
            'pagination': {
                'page': page,
                'limit': limit,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            }
        }
        
        return self.format_success_response(data)


class BaseAPIHandler(ABC):
    """API处理器基类"""
    
    def __init__(self, system: Any):
        self.system = system
        self.logger = logging.getLogger(__name__)
        self.validator = RequestValidator()
        self.formatter = ResponseFormatter()
    
    @abstractmethod
    def get_endpoints(self) -> List[APIEndpoint]:
        """获取处理器的端点定义"""
        pass


class TradingAPIHandler(BaseAPIHandler):
    """交易API处理器"""
    
    def __init__(self, trading_system: Any):
        super().__init__(trading_system)
        self.trading_system = trading_system
    
    def get_endpoints(self) -> List[APIEndpoint]:
        """获取交易相关端点"""
        return [
            APIEndpoint(
                path="/api/v1/trading/execute",
                methods=[HTTPMethod.POST],
                handler=self.execute_trade,
                description="Execute a trade order",
                parameters={
                    'symbol': {'type': 'string', 'required': True},
                    'action': {'type': 'string', 'enum': ['BUY', 'SELL'], 'required': True},
                    'quantity': {'type': 'number', 'required': True},
                    'price': {'type': 'number', 'required': True}
                }
            ),
            APIEndpoint(
                path="/api/v1/trading/orders/{order_id}",
                methods=[HTTPMethod.GET],
                handler=self.get_order_status,
                description="Get order status"
            ),
            APIEndpoint(
                path="/api/v1/trading/orders/{order_id}",
                methods=[HTTPMethod.DELETE],
                handler=self.cancel_order,
                description="Cancel an order"
            ),
            APIEndpoint(
                path="/api/v1/trading/history",
                methods=[HTTPMethod.GET],
                handler=self.get_trade_history,
                description="Get trade history"
            ),
            APIEndpoint(
                path="/api/v1/trading/performance",
                methods=[HTTPMethod.GET],
                handler=self.get_trading_performance,
                description="Get trading performance metrics"
            )
        ]
    
    def execute_trade(self, request_data: Dict[str, Any]) -> APIResponse:
        """执行交易"""
        # 验证请求
        validation_result = self.validator.validate_trading_request(request_data)
        if not validation_result.is_valid:
            error = APIError(
                code='VALIDATION_ERROR',
                message='Invalid trading request',
                details={'validation_errors': validation_result.errors}
            )
            return self.formatter.format_error_response(error, 400)
        
        # 执行交易
        trade_result = self.trading_system.execute_trade(request_data)
        return self.formatter.format_success_response(trade_result)
    
    def get_order_status(self, order_id: str) -> APIResponse:
        """获取订单状态"""
        order_status = self.trading_system.get_order_status(order_id)
        return self.formatter.format_success_response(order_status)
    
    def cancel_order(self, order_id: str) -> APIResponse:
        """取消订单"""
        cancel_result = self.trading_system.cancel_order(order_id)
        return self.formatter.format_success_response(cancel_result)
    
    def get_trade_history(self, start_date=None, end_date=None, symbol=None, limit=100) -> APIResponse:
        """获取交易历史"""
        trade_history = self.trading_system.get_trade_history(
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
            limit=limit
        )
        
        return self.formatter.format_success_response({
            'trades': trade_history,
            'total_count': len(trade_history)
        })
    
    def get_trading_performance(self, period_days: int = 30) -> APIResponse:
        """获取交易性能指标"""
        performance_data = self.trading_system.get_trading_performance(period_days)
        return self.formatter.format_success_response(performance_data)


class PortfolioAPIHandler(BaseAPIHandler):
    """投资组合API处理器"""
    
    def __init__(self, trading_system: Any):
        super().__init__(trading_system)
        self.trading_system = trading_system
    
    def get_endpoints(self) -> List[APIEndpoint]:
        """获取投资组合相关端点"""
        return [
            APIEndpoint(
                path="/api/v1/portfolio/positions",
                methods=[HTTPMethod.GET],
                handler=self.get_positions,
                description="Get portfolio positions"
            ),
            APIEndpoint(
                path="/api/v1/portfolio/performance",
                methods=[HTTPMethod.GET],
                handler=self.get_performance,
                description="Get portfolio performance"
            ),
            APIEndpoint(
                path="/api/v1/portfolio/positions/{symbol}",
                methods=[HTTPMethod.GET],
                handler=self.get_position_detail,
                description="Get position detail"
            ),
            APIEndpoint(
                path="/api/v1/portfolio/allocation",
                methods=[HTTPMethod.GET],
                handler=self.get_allocation_analysis,
                description="Get portfolio allocation analysis"
            )
        ]
    
    def get_positions(self) -> APIResponse:
        """获取投资组合持仓"""
        portfolio_data = self.trading_system.get_portfolio()
        return self.formatter.format_success_response(portfolio_data)
    
    def get_performance(self, period_days: int = 365) -> APIResponse:
        """获取投资组合绩效"""
        performance_data = self.trading_system.get_portfolio_performance(period_days)
        return self.formatter.format_success_response(performance_data)
    
    def get_position_detail(self, symbol: str) -> APIResponse:
        """获取持仓详情"""
        position_detail = self.trading_system.get_position_detail(symbol)
        return self.formatter.format_success_response(position_detail)
    
    def get_allocation_analysis(self) -> APIResponse:
        """获取投资组合配置分析"""
        allocation_data = self.trading_system.analyze_portfolio_allocation()
        return self.formatter.format_success_response(allocation_data)


class RiskAPIHandler(BaseAPIHandler):
    """风险API处理器"""
    
    def __init__(self, risk_system: Any):
        super().__init__(risk_system)
        self.risk_system = risk_system
    
    def get_endpoints(self) -> List[APIEndpoint]:
        """获取风险相关端点"""
        return [
            APIEndpoint(
                path="/api/v1/risk/assessment",
                methods=[HTTPMethod.GET],
                handler=self.get_risk_assessment,
                description="Get risk assessment"
            ),
            APIEndpoint(
                path="/api/v1/risk/check",
                methods=[HTTPMethod.POST],
                handler=self.check_trade_risk,
                description="Check trade risk"
            ),
            APIEndpoint(
                path="/api/v1/risk/limits",
                methods=[HTTPMethod.GET],
                handler=self.get_risk_limits,
                description="Get risk limits"
            ),
            APIEndpoint(
                path="/api/v1/risk/stress-test",
                methods=[HTTPMethod.POST],
                handler=self.stress_test_portfolio,
                description="Run portfolio stress test"
            )
        ]
    
    def get_risk_assessment(self) -> APIResponse:
        """获取风险评估"""
        risk_assessment = self.risk_system.assess_portfolio_risk()
        return self.formatter.format_success_response(risk_assessment)
    
    def check_trade_risk(self, trade_request: Dict[str, Any]) -> APIResponse:
        """检查交易风险"""
        risk_check_result = self.risk_system.check_trade_risk(trade_request)
        return self.formatter.format_success_response(risk_check_result)
    
    def get_risk_limits(self) -> APIResponse:
        """获取风险限制"""
        risk_limits = self.risk_system.get_risk_limits()
        return self.formatter.format_success_response(risk_limits)
    
    def stress_test_portfolio(self, stress_scenarios: List[Dict[str, Any]]) -> APIResponse:
        """投资组合压力测试"""
        stress_test_result = self.risk_system.run_stress_test(stress_scenarios)
        return self.formatter.format_success_response(stress_test_result)


class ModelAPIHandler(BaseAPIHandler):
    """模型API处理器"""
    
    def __init__(self, model_system: Any):
        super().__init__(model_system)
        self.model_system = model_system
    
    def get_endpoints(self) -> List[APIEndpoint]:
        """获取模型相关端点"""
        return [
            APIEndpoint(
                path="/api/v1/model/predict",
                methods=[HTTPMethod.POST],
                handler=self.predict,
                description="Make model prediction"
            ),
            APIEndpoint(
                path="/api/v1/model/status",
                methods=[HTTPMethod.GET],
                handler=self.get_model_status,
                description="Get model status"
            ),
            APIEndpoint(
                path="/api/v1/model/performance",
                methods=[HTTPMethod.GET],
                handler=self.get_model_performance,
                description="Get model performance metrics"
            )
        ]
    
    def predict(self, request_data: Dict[str, Any]) -> APIResponse:
        """模型预测"""
        prediction_result = self.model_system.predict(request_data)
        return self.formatter.format_success_response(prediction_result)
    
    def get_model_status(self) -> APIResponse:
        """获取模型状态"""
        model_status = self.model_system.get_status()
        return self.formatter.format_success_response(model_status)
    
    def get_model_performance(self) -> APIResponse:
        """获取模型性能指标"""
        performance_data = self.model_system.get_performance_metrics()
        return self.formatter.format_success_response(performance_data)


class DataAPIHandler(BaseAPIHandler):
    """数据API处理器"""
    
    def __init__(self, data_system: Any):
        super().__init__(data_system)
        self.data_system = data_system
    
    def get_endpoints(self) -> List[APIEndpoint]:
        """获取数据相关端点"""
        return [
            APIEndpoint(
                path="/api/v1/data/market",
                methods=[HTTPMethod.GET],
                handler=self.get_market_data,
                description="Get market data"
            ),
            APIEndpoint(
                path="/api/v1/data/historical",
                methods=[HTTPMethod.GET],
                handler=self.get_historical_data,
                description="Get historical data"
            ),
            APIEndpoint(
                path="/api/v1/data/indicators",
                methods=[HTTPMethod.GET],
                handler=self.get_technical_indicators,
                description="Get technical indicators"
            ),
            APIEndpoint(
                path="/api/v1/data/fundamental",
                methods=[HTTPMethod.GET],
                handler=self.get_fundamental_data,
                description="Get fundamental data"
            )
        ]
    
    def get_market_data(self, symbols: List[str]) -> APIResponse:
        """获取市场数据"""
        market_data = self.data_system.get_market_data(symbols)
        return self.formatter.format_success_response(market_data)
    
    def get_historical_data(self, symbol: str, start_date, end_date, interval: str = '1d') -> APIResponse:
        """获取历史数据"""
        historical_data = self.data_system.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        return self.formatter.format_success_response(historical_data)
    
    def get_technical_indicators(self, symbol: str, indicators: List[str]) -> APIResponse:
        """获取技术指标"""
        indicator_data = self.data_system.get_technical_indicators(symbol, indicators)
        return self.formatter.format_success_response(indicator_data)
    
    def get_fundamental_data(self, symbol: str) -> APIResponse:
        """获取基本面数据"""
        fundamental_data = self.data_system.get_fundamental_data(symbol)
        return self.formatter.format_success_response(fundamental_data)


class APIServer:
    """API服务器"""
    
    def __init__(self, config: APIConfig, trading_system: Any):
        self.config = config
        self.trading_system = trading_system
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.request_validator = RequestValidator()
        self.response_formatter = ResponseFormatter()
        
        # 初始化处理器
        self.handlers = {}
        self._register_handlers()
        
        # 初始化认证和限流（模拟）
        self.authenticator = self._create_authenticator()
        self.rate_limiter = self._create_rate_limiter()
        
        # 性能统计
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'requests_per_second': 0.0,
            'start_time': datetime.now()
        }
        self._stats_lock = threading.Lock()
    
    def _register_handlers(self):
        """注册API处理器"""
        self.handlers['trading'] = TradingAPIHandler(self.trading_system)
        self.handlers['portfolio'] = PortfolioAPIHandler(self.trading_system)
        
        # 模拟其他系统
        mock_risk_system = type('MockRiskSystem', (), {
            'assess_portfolio_risk': lambda: {},
            'check_trade_risk': lambda x: {},
            'get_risk_limits': lambda: {},
            'run_stress_test': lambda x: {}
        })()
        self.handlers['risk'] = RiskAPIHandler(mock_risk_system)
        
        mock_model_system = type('MockModelSystem', (), {
            'predict': lambda x: {},
            'get_status': lambda: {},
            'get_performance_metrics': lambda: {}
        })()
        self.handlers['model'] = ModelAPIHandler(mock_model_system)
        
        mock_data_system = type('MockDataSystem', (), {
            'get_market_data': lambda x: {},
            'get_historical_data': lambda **kw: {},
            'get_technical_indicators': lambda x, y: {},
            'get_fundamental_data': lambda x: {}
        })()
        self.handlers['data'] = DataAPIHandler(mock_data_system)
    
    def _create_authenticator(self):
        """创建认证器（模拟）"""
        return type('MockAuthenticator', (), {
            'authenticate': lambda token: type('AuthResult', (), {
                'authenticated': True,
                'user_id': 'test_user',
                'permissions': []
            })()
        })()
    
    def _create_rate_limiter(self):
        """创建限流器（模拟）"""
        return type('MockRateLimiter', (), {
            'allow_request': lambda req: type('RateLimitResult', (), {
                'allowed': True,
                'remaining_requests': 99,
                'reset_time': datetime.now() + timedelta(minutes=1)
            })()
        })()
    
    def get_registered_endpoints(self) -> List[APIEndpoint]:
        """获取所有注册的端点"""
        endpoints = []
        for handler in self.handlers.values():
            endpoints.extend(handler.get_endpoints())
        
        # 添加系统端点
        endpoints.extend([
            APIEndpoint(
                path="/api/v1/health",
                methods=[HTTPMethod.GET],
                handler=self._health_check,
                description="Health check endpoint",
                auth_required=False
            ),
            APIEndpoint(
                path="/api/v1/docs",
                methods=[HTTPMethod.GET],
                handler=self._generate_documentation,
                description="API documentation",
                auth_required=False
            )
        ])
        
        return endpoints
    
    def process_request(self, request: Dict[str, Any]) -> APIResponse:
        """处理请求"""
        start_time = time.time()
        
        # 更新统计
        with self._stats_lock:
            self.performance_stats['total_requests'] += 1
        
        # 解析请求
        method = request.get('method', 'GET')
        path = request.get('path', '/')
        headers = request.get('headers', {})
        body = request.get('body', '')
        
        # CORS处理
        if method == 'OPTIONS':
            return self._handle_cors_preflight(headers)
        
        # 路由请求
        endpoint, handler = self._route_request(method, path)
        if not endpoint:
            error = APIError(
                code='NOT_FOUND',
                message=f'Endpoint not found: {method} {path}',
                status_code=404
            )
            return self.response_formatter.format_error_response(error, 404)
        
        # 认证检查
        if endpoint.auth_required:
            auth_token = headers.get('Authorization', '')
            auth_result = self.authenticator.authenticate(auth_token)
            if not auth_result.authenticated:
                error = APIError(
                    code='UNAUTHORIZED',
                    message='Authentication required',
                    status_code=401
                )
                return self.response_formatter.format_error_response(error, 401)
        
        # 限流检查
        rate_limit_result = self.rate_limiter.allow_request(request)
        if not rate_limit_result.allowed:
            error = APIError(
                code='RATE_LIMIT_EXCEEDED',
                message='Rate limit exceeded',
                status_code=429
            )
            response = self.response_formatter.format_error_response(error, 429)
            response.headers['Retry-After'] = str(int(rate_limit_result.reset_time.timestamp()))
            return response
        
        # 处理请求
        try:
            # 解析请求数据
            request_data = {}
            if body:
                request_data = json.loads(body)
            
            # 调用处理器
            if hasattr(handler, '__call__'):
                if request_data:
                    response = handler(request_data)
                else:
                    response = handler()
            else:
                response = self.response_formatter.format_success_response({})
            
            # 添加CORS头
            if self.config.cors_enabled:
                response.headers.update(self._get_cors_headers())
            
            # 更新统计
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            
            with self._stats_lock:
                self.performance_stats['successful_requests'] += 1
                self._update_response_time_stats(processing_time)
            
            return response
            
        except json.JSONDecodeError:
            error = APIError(
                code='INVALID_JSON',
                message='Invalid JSON in request body',
                status_code=400
            )
            return self.response_formatter.format_error_response(error, 400)
        
        except Exception as e:
            # 按照要求，不捕获异常，让其暴露
            raise e
    
    def _route_request(self, method: str, path: str) -> tuple:
        """路由请求到处理器"""
        # 简化的路由逻辑
        endpoints = self.get_registered_endpoints()
        
        for endpoint in endpoints:
            if endpoint.path == path and HTTPMethod(method) in endpoint.methods:
                return endpoint, endpoint.handler
        
        return None, None
    
    def _handle_cors_preflight(self, headers: Dict[str, str]) -> APIResponse:
        """处理CORS预检请求"""
        response = APIResponse(
            status_code=200,
            headers=self._get_cors_headers()
        )
        
        # 添加请求的方法和头部
        if 'Access-Control-Request-Method' in headers:
            response.headers['Access-Control-Allow-Methods'] = headers['Access-Control-Request-Method']
        
        if 'Access-Control-Request-Headers' in headers:
            response.headers['Access-Control-Allow-Headers'] = headers['Access-Control-Request-Headers']
        
        return response
    
    def _get_cors_headers(self) -> Dict[str, str]:
        """获取CORS头部"""
        return {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '86400'
        }
    
    def _health_check(self) -> APIResponse:
        """健康检查"""
        uptime = datetime.now() - self.performance_stats['start_time']
        
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': self.config.api_version,
            'uptime': str(uptime)
        }
        
        return self.response_formatter.format_success_response(health_data)
    
    def _generate_documentation(self) -> APIResponse:
        """生成API文档"""
        doc_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Documentation</title>
        </head>
        <body>
            <h1>API Documentation</h1>
            <h2>Trading API</h2>
            <p>POST /api/v1/trading/execute - Execute a trade</p>
            <h2>Portfolio API</h2>
            <p>GET /api/v1/portfolio/positions - Get portfolio positions</p>
        </body>
        </html>
        """
        
        return APIResponse(
            status_code=200,
            headers={'Content-Type': 'text/html'},
            body=doc_html
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._stats_lock:
            uptime = datetime.now() - self.performance_stats['start_time']
            uptime_seconds = uptime.total_seconds()
            
            if uptime_seconds > 0:
                self.performance_stats['requests_per_second'] = (
                    self.performance_stats['total_requests'] / uptime_seconds
                )
            
            return self.performance_stats.copy()
    
    def _update_response_time_stats(self, processing_time: float):
        """更新响应时间统计"""
        current_avg = self.performance_stats['average_response_time']
        total_requests = self.performance_stats['total_requests']
        
        # 计算新的平均响应时间
        new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
        self.performance_stats['average_response_time'] = new_avg