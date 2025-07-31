"""
API接口系统的单元测试
测试API接口的功能完整性和性能，标准化数据格式和错误处理，API文档和使用示例
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.rl_trading_system.api.api_server import (
    APIServer,
    TradingAPIHandler,
    PortfolioAPIHandler,
    RiskAPIHandler,
    ModelAPIHandler,
    DataAPIHandler,
    APIResponse,
    APIError,
    RequestValidator,
    ResponseFormatter,
    APIConfig,
    APIEndpoint,
    HTTPMethod,
    APIStatus
)

from src.rl_trading_system.api.authentication import (
    APIKeyAuthenticator,
    JWTAuthenticator,
    AuthenticationManager,
    APIKey,
    JWTToken,
    AuthResult,
    PermissionLevel
)

from src.rl_trading_system.api.rate_limiter import (
    RateLimiter,
    TokenBucket,
    SlidingWindow,
    RateLimitConfig,
    RateLimitResult,
    ConcurrencyLimiter
)


class TestAPIServer:
    """API服务器测试类"""

    @pytest.fixture
    def api_config(self):
        """创建API配置"""
        return APIConfig(
            host="127.0.0.1",
            port=8080,
            debug=False,
            max_request_size=1024*1024,  # 1MB
            timeout=30,
            cors_enabled=True,
            api_version="v1",
            documentation_enabled=True,
            rate_limit_requests_per_minute=100,
            max_concurrent_requests=50,
            authentication_required=True
        )

    @pytest.fixture
    def mock_trading_system(self):
        """创建模拟交易系统"""
        system = Mock()
        system.get_portfolio.return_value = {
            'positions': [],
            'cash': 100000.0,
            'total_value': 100000.0,
            'timestamp': datetime.now().isoformat()
        }
        system.execute_trade.return_value = {
            'trade_id': 'TXN_001',
            'status': 'executed',
            'execution_time': datetime.now().isoformat()
        }
        return system

    @pytest.fixture
    def api_server(self, api_config, mock_trading_system):
        """创建API服务器"""
        return APIServer(config=api_config, trading_system=mock_trading_system)

    def test_api_server_initialization(self, api_server, api_config):
        """测试API服务器初始化"""
        assert api_server.config == api_config
        assert api_server.trading_system is not None
        assert api_server.handlers is not None
        assert api_server.authenticator is not None
        assert api_server.rate_limiter is not None
        assert api_server.request_validator is not None
        assert api_server.response_formatter is not None
        
        # 验证处理器已注册
        assert len(api_server.handlers) > 0
        handler_types = [type(handler).__name__ for handler in api_server.handlers.values()]
        assert 'TradingAPIHandler' in handler_types
        assert 'PortfolioAPIHandler' in handler_types
        assert 'RiskAPIHandler' in handler_types

    def test_api_endpoint_registration(self, api_server):
        """测试API端点注册"""
        endpoints = api_server.get_registered_endpoints()
        
        # 验证核心端点已注册
        endpoint_paths = [ep.path for ep in endpoints]
        assert '/api/v1/trading/execute' in endpoint_paths
        assert '/api/v1/portfolio/positions' in endpoint_paths
        assert '/api/v1/risk/assessment' in endpoint_paths
        assert '/api/v1/model/predict' in endpoint_paths
        assert '/api/v1/data/market' in endpoint_paths
        
        # 验证HTTP方法
        trading_endpoint = next(ep for ep in endpoints if ep.path == '/api/v1/trading/execute')
        assert HTTPMethod.POST in trading_endpoint.methods
        
        portfolio_endpoint = next(ep for ep in endpoints if ep.path == '/api/v1/portfolio/positions')
        assert HTTPMethod.GET in portfolio_endpoint.methods

    def test_request_processing_pipeline(self, api_server):
        """测试请求处理管道"""
        mock_request = {
            'method': 'POST',
            'path': '/api/v1/trading/execute',
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer test_token'
            },
            'body': json.dumps({
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'price': 150.0
            })
        }
        
        # 模拟请求处理
        with patch.object(api_server.authenticator, 'authenticate') as mock_auth:
            mock_auth.return_value = AuthResult(
                authenticated=True,
                user_id='test_user',
                permissions=[PermissionLevel.TRADING]
            )
            
            with patch.object(api_server.rate_limiter, 'allow_request') as mock_rate_limit:
                mock_rate_limit.return_value = RateLimitResult(
                    allowed=True,
                    remaining_requests=99,
                    reset_time=datetime.now() + timedelta(minutes=1)
                )
                
                response = api_server.process_request(mock_request)
                
                assert response.status_code == 200
                assert response.headers['Content-Type'] == 'application/json'
                assert 'trade_id' in json.loads(response.body)

    def test_api_error_handling(self, api_server):
        """测试API错误处理"""
        # 测试无效请求
        invalid_request = {
            'method': 'POST',
            'path': '/api/v1/trading/execute',
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'invalid': 'data'})
        }
        
        response = api_server.process_request(invalid_request)
        assert response.status_code == 400
        
        error_data = json.loads(response.body)
        assert error_data['error']['code'] == 'VALIDATION_ERROR'
        assert error_data['error']['message'] is not None
        assert 'details' in error_data['error']

    def test_api_performance_monitoring(self, api_server):
        """测试API性能监控"""
        # 模拟多个并发请求
        requests = []
        for i in range(10):
            request = {
                'method': 'GET',
                'path': '/api/v1/portfolio/positions',
                'headers': {'Authorization': 'Bearer test_token'},
                'body': ''
            }
            requests.append(request)
        
        start_time = time.time()
        
        # 并发处理请求
        with ThreadPoolExecutor(max_workers=5) as executor:
            responses = list(executor.map(api_server.process_request, requests))
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证性能指标
        assert processing_time < 2.0  # 应该在2秒内完成
        assert all(r.status_code in [200, 401, 429] for r in responses)  # 所有响应都应该有效
        
        # 检查性能统计
        stats = api_server.get_performance_stats()
        assert stats['total_requests'] >= 10
        assert stats['average_response_time'] > 0
        assert stats['requests_per_second'] > 0

    def test_api_health_check(self, api_server):
        """测试API健康检查"""
        health_request = {
            'method': 'GET',
            'path': '/api/v1/health',
            'headers': {},
            'body': ''
        }
        
        response = api_server.process_request(health_request)
        assert response.status_code == 200
        
        health_data = json.loads(response.body)
        assert health_data['status'] == 'healthy'
        assert health_data['timestamp'] is not None
        assert health_data['version'] is not None
        assert health_data['uptime'] >= 0

    def test_api_documentation_generation(self, api_server):
        """测试API文档生成"""
        doc_request = {
            'method': 'GET',
            'path': '/api/v1/docs',
            'headers': {},
            'body': ''
        }
        
        response = api_server.process_request(doc_request)
        assert response.status_code == 200
        assert 'text/html' in response.headers['Content-Type']
        
        # 验证文档内容包含关键信息
        doc_content = response.body
        assert 'API Documentation' in doc_content
        assert '/api/v1/trading/execute' in doc_content
        assert 'POST' in doc_content

    def test_api_versioning(self, api_server):
        """测试API版本控制"""
        # 测试v1版本
        v1_request = {
            'method': 'GET',
            'path': '/api/v1/portfolio/positions',
            'headers': {'Authorization': 'Bearer test_token'},
            'body': ''
        }
        
        response_v1 = api_server.process_request(v1_request)
        assert response_v1.status_code in [200, 401]
        
        # 测试不支持的版本
        v2_request = {
            'method': 'GET',
            'path': '/api/v2/portfolio/positions',
            'headers': {'Authorization': 'Bearer test_token'},
            'body': ''
        }
        
        response_v2 = api_server.process_request(v2_request)
        assert response_v2.status_code == 404

    def test_cors_handling(self, api_server):
        """测试CORS处理"""
        options_request = {
            'method': 'OPTIONS',
            'path': '/api/v1/trading/execute',
            'headers': {
                'Origin': 'http://localhost:3000',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type,Authorization'
            },
            'body': ''
        }
        
        response = api_server.process_request(options_request)
        assert response.status_code == 200
        assert 'Access-Control-Allow-Origin' in response.headers
        assert 'Access-Control-Allow-Methods' in response.headers
        assert 'Access-Control-Allow-Headers' in response.headers


class TestTradingAPIHandler:
    """交易API处理器测试类"""

    @pytest.fixture
    def trading_handler(self):
        """创建交易API处理器"""
        mock_trading_system = Mock()
        return TradingAPIHandler(trading_system=mock_trading_system)

    @pytest.fixture
    def valid_trade_request(self):
        """创建有效的交易请求"""
        return {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'order_type': 'LIMIT',
            'time_in_force': 'DAY'
        }

    def test_execute_trade_success(self, trading_handler, valid_trade_request):
        """测试交易执行成功"""
        # 模拟交易系统响应
        trading_handler.trading_system.execute_trade.return_value = {
            'trade_id': 'TXN_001',
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'executed_price': 149.95,
            'status': 'FILLED',
            'execution_time': datetime.now().isoformat(),
            'commission': 1.0
        }
        
        response = trading_handler.execute_trade(valid_trade_request)
        
        assert response.status_code == 200
        assert response.data['trade_id'] == 'TXN_001'
        assert response.data['status'] == 'FILLED'
        assert response.data['executed_price'] == 149.95

    def test_execute_trade_validation_error(self, trading_handler):
        """测试交易请求验证错误"""
        invalid_request = {
            'symbol': '',  # 空符号
            'action': 'INVALID_ACTION',  # 无效动作
            'quantity': -100,  # 负数量
            'price': 0  # 无效价格
        }
        
        response = trading_handler.execute_trade(invalid_request)
        
        assert response.status_code == 400
        assert response.error.code == 'VALIDATION_ERROR'
        assert 'symbol' in response.error.details
        assert 'action' in response.error.details
        assert 'quantity' in response.error.details

    def test_get_order_status(self, trading_handler):
        """测试获取订单状态"""
        order_id = 'ORD_001'
        
        trading_handler.trading_system.get_order_status.return_value = {
            'order_id': order_id,
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'filled_quantity': 50,
            'remaining_quantity': 50,
            'status': 'PARTIALLY_FILLED',
            'average_price': 149.50,
            'created_time': datetime.now().isoformat(),
            'updated_time': datetime.now().isoformat()
        }
        
        response = trading_handler.get_order_status(order_id)
        
        assert response.status_code == 200
        assert response.data['order_id'] == order_id
        assert response.data['status'] == 'PARTIALLY_FILLED'
        assert response.data['filled_quantity'] == 50

    def test_cancel_order(self, trading_handler):
        """测试取消订单"""
        order_id = 'ORD_001'
        
        trading_handler.trading_system.cancel_order.return_value = {
            'order_id': order_id,
            'status': 'CANCELLED',
            'cancel_time': datetime.now().isoformat(),
            'remaining_quantity': 50
        }
        
        response = trading_handler.cancel_order(order_id)
        
        assert response.status_code == 200
        assert response.data['order_id'] == order_id
        assert response.data['status'] == 'CANCELLED'

    def test_get_trade_history(self, trading_handler):
        """测试获取交易历史"""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        trading_handler.trading_system.get_trade_history.return_value = [
            {
                'trade_id': 'TXN_001',
                'symbol': 'AAPL',
                'action': 'BUY',
                'quantity': 100,
                'price': 149.95,
                'execution_time': datetime.now().isoformat(),
                'commission': 1.0
            },
            {
                'trade_id': 'TXN_002',
                'symbol': 'MSFT',
                'action': 'SELL',
                'quantity': 50,
                'price': 280.10,
                'execution_time': datetime.now().isoformat(),
                'commission': 1.0
            }
        ]
        
        response = trading_handler.get_trade_history(
            start_date=start_date.date(),
            end_date=end_date.date(),
            symbol=None,
            limit=100
        )
        
        assert response.status_code == 200
        assert len(response.data['trades']) == 2
        assert response.data['trades'][0]['trade_id'] == 'TXN_001'
        assert response.data['total_count'] == 2

    def test_trading_performance_metrics(self, trading_handler):
        """测试交易性能指标"""
        trading_handler.trading_system.get_trading_performance.return_value = {
            'total_trades': 150,
            'successful_trades': 145,
            'failed_trades': 5,
            'success_rate': 0.967,
            'total_volume': 15000000.0,
            'total_commission': 150.0,
            'average_execution_time_ms': 120.5,
            'period_start': (datetime.now() - timedelta(days=30)).isoformat(),
            'period_end': datetime.now().isoformat()
        }
        
        response = trading_handler.get_trading_performance(period_days=30)
        
        assert response.status_code == 200
        assert response.data['total_trades'] == 150
        assert response.data['success_rate'] == 0.967
        assert response.data['average_execution_time_ms'] == 120.5


class TestPortfolioAPIHandler:
    """投资组合API处理器测试类"""

    @pytest.fixture
    def portfolio_handler(self):
        """创建投资组合API处理器"""
        mock_trading_system = Mock()
        return PortfolioAPIHandler(trading_system=mock_trading_system)

    def test_get_portfolio_positions(self, portfolio_handler):
        """测试获取投资组合持仓"""
        portfolio_handler.trading_system.get_portfolio.return_value = {
            'positions': [
                {
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'average_price': 145.0,
                    'current_price': 150.0,
                    'market_value': 15000.0,
                    'unrealized_pnl': 500.0,
                    'weight': 0.15
                },
                {
                    'symbol': 'MSFT',
                    'quantity': 50,
                    'average_price': 275.0,
                    'current_price': 280.0,
                    'market_value': 14000.0,
                    'unrealized_pnl': 250.0,
                    'weight': 0.14
                }
            ],
            'cash': 71000.0,
            'total_value': 100000.0,
            'total_pnl': 750.0,
            'timestamp': datetime.now().isoformat()
        }
        
        response = portfolio_handler.get_positions()
        
        assert response.status_code == 200
        assert len(response.data['positions']) == 2
        assert response.data['cash'] == 71000.0
        assert response.data['total_value'] == 100000.0
        assert response.data['total_pnl'] == 750.0

    def test_get_portfolio_performance(self, portfolio_handler):
        """测试获取投资组合绩效"""
        portfolio_handler.trading_system.get_portfolio_performance.return_value = {
            'total_return': 0.075,
            'annualized_return': 0.12,
            'sharpe_ratio': 1.35,
            'max_drawdown': -0.08,
            'volatility': 0.15,
            'var_95': -0.025,
            'cvar_95': -0.045,
            'calmar_ratio': 1.5,
            'sortino_ratio': 1.8,
            'period_start': (datetime.now() - timedelta(days=365)).isoformat(),
            'period_end': datetime.now().isoformat()
        }
        
        response = portfolio_handler.get_performance(period_days=365)
        
        assert response.status_code == 200
        assert response.data['total_return'] == 0.075
        assert response.data['sharpe_ratio'] == 1.35
        assert response.data['max_drawdown'] == -0.08

    def test_get_position_detail(self, portfolio_handler):
        """测试获取持仓详情"""
        symbol = 'AAPL'
        
        portfolio_handler.trading_system.get_position_detail.return_value = {
            'symbol': symbol,
            'quantity': 100,
            'average_price': 145.0,
            'current_price': 150.0,
            'market_value': 15000.0,
            'unrealized_pnl': 500.0,
            'realized_pnl': 200.0,
            'total_pnl': 700.0,
            'weight': 0.15,
            'sector': 'Technology',
            'first_buy_date': (datetime.now() - timedelta(days=60)).isoformat(),
            'last_trade_date': (datetime.now() - timedelta(days=5)).isoformat(),
            'holding_period_days': 60,
            'trade_count': 5
        }
        
        response = portfolio_handler.get_position_detail(symbol)
        
        assert response.status_code == 200
        assert response.data['symbol'] == symbol
        assert response.data['quantity'] == 100
        assert response.data['total_pnl'] == 700.0
        assert response.data['sector'] == 'Technology'

    def test_portfolio_allocation_analysis(self, portfolio_handler):
        """测试投资组合配置分析"""
        portfolio_handler.trading_system.analyze_portfolio_allocation.return_value = {
            'sector_allocation': {
                'Technology': 0.45,
                'Financial': 0.25,
                'Healthcare': 0.20,
                'Energy': 0.10
            },
            'top_holdings': [
                {'symbol': 'AAPL', 'weight': 0.15},
                {'symbol': 'MSFT', 'weight': 0.14},
                {'symbol': 'GOOGL', 'weight': 0.12}
            ],
            'concentration_metrics': {
                'herfindahl_index': 0.15,
                'top_10_concentration': 0.75,
                'effective_positions': 12
            },
            'diversification_score': 0.85,
            'rebalancing_recommendations': [
                {
                    'action': 'REDUCE',
                    'symbol': 'AAPL',
                    'current_weight': 0.15,
                    'target_weight': 0.12,
                    'reason': 'Over-concentration'
                }
            ]
        }
        
        response = portfolio_handler.get_allocation_analysis()
        
        assert response.status_code == 200
        assert 'sector_allocation' in response.data
        assert response.data['concentration_metrics']['herfindahl_index'] == 0.15
        assert len(response.data['rebalancing_recommendations']) == 1


class TestRiskAPIHandler:
    """风险API处理器测试类"""

    @pytest.fixture
    def risk_handler(self):
        """创建风险API处理器"""
        mock_risk_system = Mock()
        return RiskAPIHandler(risk_system=mock_risk_system)

    def test_get_risk_assessment(self, risk_handler):
        """测试获取风险评估"""
        risk_handler.risk_system.assess_portfolio_risk.return_value = {
            'overall_risk_score': 6.5,
            'risk_level': 'MEDIUM',
            'concentration_risk': 4.2,
            'sector_risk': 3.8,
            'volatility_risk': 7.1,
            'liquidity_risk': 2.5,
            'var_1d_95': -0.025,
            'cvar_1d_95': -0.045,
            'beta': 1.15,
            'risk_factors': [
                {
                    'factor': 'Technology Concentration',
                    'score': 7.5,
                    'impact': 'HIGH',
                    'description': '科技股集中度过高'
                },
                {
                    'factor': 'Market Volatility',
                    'score': 6.0,
                    'impact': 'MEDIUM',
                    'description': '市场波动率上升'
                }
            ],
            'recommendations': [
                '减少科技股持仓比例',
                '增加防御性资产配置',
                '考虑设置更严格的止损水平'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        response = risk_handler.get_risk_assessment()
        
        assert response.status_code == 200
        assert response.data['overall_risk_score'] == 6.5
        assert response.data['risk_level'] == 'MEDIUM'
        assert len(response.data['risk_factors']) == 2
        assert len(response.data['recommendations']) == 3

    def test_check_trade_risk(self, risk_handler):
        """测试检查交易风险"""
        trade_request = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 500,
            'price': 150.0
        }
        
        risk_handler.risk_system.check_trade_risk.return_value = {
            'approved': False,
            'risk_score': 8.5,
            'violations': [
                {
                    'type': 'POSITION_CONCENTRATION',
                    'severity': 'HIGH',
                    'message': '持仓集中度超过限制',
                    'current_value': 0.18,
                    'limit': 0.15
                },
                {
                    'type': 'SECTOR_EXPOSURE',
                    'severity': 'MEDIUM',
                    'message': '科技行业暴露过高',
                    'current_value': 0.52,
                    'limit': 0.50
                }
            ],
            'recommendations': [
                '建议减少交易规模至300股',
                '考虑在其他行业增加配置以平衡风险'
            ],
            'estimated_new_risk_score': 7.2,
            'timestamp': datetime.now().isoformat()
        }
        
        response = risk_handler.check_trade_risk(trade_request)
        
        assert response.status_code == 200
        assert response.data['approved'] == False
        assert response.data['risk_score'] == 8.5
        assert len(response.data['violations']) == 2
        assert response.data['violations'][0]['type'] == 'POSITION_CONCENTRATION'

    def test_get_risk_limits(self, risk_handler):
        """测试获取风险限制"""
        risk_handler.risk_system.get_risk_limits.return_value = {
            'position_limits': {
                'max_single_position_weight': 0.15,
                'max_sector_exposure': 0.50,
                'max_portfolio_concentration': 0.60
            },
            'loss_limits': {
                'stop_loss_threshold': 0.05,
                'max_daily_loss': 0.02,
                'max_drawdown': 0.10
            },
            'volatility_limits': {
                'max_portfolio_volatility': 0.20,
                'max_beta': 1.50,
                'var_limit_1d_95': 0.030
            },
            'liquidity_limits': {
                'min_cash_ratio': 0.05,
                'min_liquidity_coverage': 0.10
            },
            'leverage_limits': {
                'max_leverage': 2.0,
                'margin_requirement': 0.25
            },
            'last_updated': datetime.now().isoformat()
        }
        
        response = risk_handler.get_risk_limits()
        
        assert response.status_code == 200
        assert 'position_limits' in response.data
        assert response.data['position_limits']['max_single_position_weight'] == 0.15
        assert response.data['loss_limits']['stop_loss_threshold'] == 0.05

    def test_stress_test_portfolio(self, risk_handler):
        """测试投资组合压力测试"""
        stress_scenarios = [
            {
                'name': 'market_crash',
                'market_shock': -0.20,
                'volatility_increase': 2.0
            },
            {
                'name': 'sector_rotation',
                'tech_shock': -0.15,
                'finance_boost': 0.10
            }
        ]
        
        risk_handler.risk_system.run_stress_test.return_value = {
            'scenarios': [
                {
                    'name': 'market_crash',
                    'portfolio_impact': -0.18,
                    'worst_position_impact': -0.25,
                    'var_impact': -0.045,
                    'liquidity_impact': 0.15,
                    'recovery_time_days': 45
                },
                {
                    'name': 'sector_rotation', 
                    'portfolio_impact': -0.08,
                    'worst_position_impact': -0.15,
                    'var_impact': -0.020,
                    'liquidity_impact': 0.05,
                    'recovery_time_days': 20
                }
            ],
            'summary': {
                'worst_case_loss': -0.18,
                'average_loss': -0.13,
                'positions_at_risk': 8,
                'estimated_margin_call_level': -0.22
            },
            'recommendations': [
                '增加现金头寸作为缓冲',
                '考虑购买看跌期权进行对冲',
                '重新平衡行业配置'
            ],
            'test_timestamp': datetime.now().isoformat()
        }
        
        response = risk_handler.stress_test_portfolio(stress_scenarios)
        
        assert response.status_code == 200
        assert len(response.data['scenarios']) == 2
        assert response.data['summary']['worst_case_loss'] == -0.18
        assert len(response.data['recommendations']) == 3


class TestDataAPIHandler:
    """数据API处理器测试类"""

    @pytest.fixture
    def data_handler(self):
        """创建数据API处理器"""
        mock_data_system = Mock()
        return DataAPIHandler(data_system=mock_data_system)

    def test_get_market_data(self, data_handler):
        """测试获取市场数据"""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        data_handler.data_system.get_market_data.return_value = {
            'AAPL': {
                'symbol': 'AAPL',
                'price': 150.25,
                'change': 2.15,
                'change_percent': 0.0145,
                'volume': 45678900,
                'open': 148.10,
                'high': 151.20,
                'low': 147.85,
                'previous_close': 148.10,
                'market_cap': 2456789012345,
                'timestamp': datetime.now().isoformat()
            },
            'MSFT': {
                'symbol': 'MSFT',
                'price': 280.50,
                'change': -1.25,
                'change_percent': -0.0044,
                'volume': 23456789,
                'open': 281.75,
                'high': 282.90,
                'low': 279.50,
                'previous_close': 281.75,
                'market_cap': 2087654321098,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        response = data_handler.get_market_data(symbols)
        
        assert response.status_code == 200
        assert len(response.data) == 2  # GOOGL might not be returned if not available
        assert response.data['AAPL']['price'] == 150.25
        assert response.data['MSFT']['change_percent'] == -0.0044

    def test_get_historical_data(self, data_handler):
        """测试获取历史数据"""
        symbol = 'AAPL'
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        data_handler.data_system.get_historical_data.return_value = {
            'symbol': symbol,
            'data': [
                {
                    'date': (datetime.now() - timedelta(days=i)).date().isoformat(),
                    'open': 148.0 + i * 0.5,
                    'high': 152.0 + i * 0.5,
                    'low': 146.0 + i * 0.5,
                    'close': 150.0 + i * 0.5,
                    'volume': 40000000 + i * 100000,
                    'adjusted_close': 150.0 + i * 0.5
                }
                for i in range(30)
            ],
            'period': '30d',
            'interval': '1d'
        }
        
        response = data_handler.get_historical_data(
            symbol=symbol,
            start_date=start_date.date(),
            end_date=end_date.date(),
            interval='1d'
        )
        
        assert response.status_code == 200
        assert response.data['symbol'] == symbol
        assert len(response.data['data']) == 30
        assert response.data['period'] == '30d'

    def test_get_technical_indicators(self, data_handler):
        """测试获取技术指标"""
        symbol = 'AAPL'
        indicators = ['SMA_20', 'EMA_12', 'RSI_14', 'MACD', 'BOLLINGER_BANDS']
        
        data_handler.data_system.get_technical_indicators.return_value = {
            'symbol': symbol,
            'indicators': {
                'SMA_20': 148.75,
                'EMA_12': 149.25,
                'RSI_14': 62.5,
                'MACD': {
                    'macd': 1.25,
                    'signal': 1.15,
                    'histogram': 0.10
                },
                'BOLLINGER_BANDS': {
                    'upper': 155.0,
                    'middle': 150.0,
                    'lower': 145.0
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        response = data_handler.get_technical_indicators(symbol, indicators)
        
        assert response.status_code == 200
        assert response.data['symbol'] == symbol
        assert response.data['indicators']['SMA_20'] == 148.75
        assert response.data['indicators']['RSI_14'] == 62.5
        assert 'macd' in response.data['indicators']['MACD']

    def test_get_fundamental_data(self, data_handler):
        """测试获取基本面数据"""
        symbol = 'AAPL'
        
        data_handler.data_system.get_fundamental_data.return_value = {
            'symbol': symbol,
            'company_info': {
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap': 2456789012345,
                'employees': 154000
            },
            'financial_metrics': {
                'pe_ratio': 25.6,
                'pb_ratio': 8.2,
                'dividend_yield': 0.0065,
                'roe': 0.175,
                'roa': 0.095,
                'debt_to_equity': 1.73,
                'current_ratio': 1.05,
                'eps_ttm': 5.89,
                'revenue_ttm': 387540000000,
                'gross_margin': 0.38
            },
            'valuation': {
                'enterprise_value': 2567890123456,
                'ev_revenue': 6.6,
                'ev_ebitda': 18.9,
                'price_to_sales': 6.3,
                'price_to_book': 8.2
            },
            'last_updated': datetime.now().isoformat()
        }
        
        response = data_handler.get_fundamental_data(symbol)
        
        assert response.status_code == 200
        assert response.data['symbol'] == symbol
        assert response.data['company_info']['name'] == 'Apple Inc.'
        assert response.data['financial_metrics']['pe_ratio'] == 25.6
        assert response.data['valuation']['ev_revenue'] == 6.6


class TestRequestValidator:
    """请求验证器测试类"""

    @pytest.fixture
    def validator(self):
        """创建请求验证器"""
        return RequestValidator()

    def test_validate_trading_request(self, validator):
        """测试交易请求验证"""
        valid_request = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'price': 150.0,
            'order_type': 'LIMIT'
        }
        
        result = validator.validate_trading_request(valid_request)
        assert result.is_valid == True
        assert len(result.errors) == 0

    def test_validate_trading_request_errors(self, validator):
        """测试交易请求验证错误"""
        invalid_request = {
            'symbol': '',  # 空符号
            'action': 'INVALID',  # 无效动作
            'quantity': -100,  # 负数量
            'price': 0,  # 无效价格
            'order_type': 'UNKNOWN'  # 未知订单类型
        }
        
        result = validator.validate_trading_request(invalid_request)
        assert result.is_valid == False
        assert len(result.errors) > 0
        assert any('symbol' in error for error in result.errors)
        assert any('action' in error for error in result.errors)
        assert any('quantity' in error for error in result.errors)

    def test_validate_date_range(self, validator):
        """测试日期范围验证"""
        # 有效日期范围
        start_date = datetime(2023, 1, 1).date()
        end_date = datetime(2023, 12, 31).date()
        
        result = validator.validate_date_range(start_date, end_date)
        assert result.is_valid == True
        
        # 无效日期范围（开始日期晚于结束日期）
        invalid_start = datetime(2023, 12, 31).date()
        invalid_end = datetime(2023, 1, 1).date()
        
        result = validator.validate_date_range(invalid_start, invalid_end)
        assert result.is_valid == False
        assert len(result.errors) > 0

    def test_validate_pagination_params(self, validator):
        """测试分页参数验证"""
        # 有效分页参数
        result = validator.validate_pagination_params(page=1, limit=50)
        assert result.is_valid == True
        
        # 无效分页参数
        result = validator.validate_pagination_params(page=0, limit=1001)
        assert result.is_valid == False
        assert len(result.errors) > 0


class TestResponseFormatter:
    """响应格式化器测试类"""

    @pytest.fixture
    def formatter(self):
        """创建响应格式化器"""
        return ResponseFormatter()

    def test_format_success_response(self, formatter):
        """测试成功响应格式化"""
        data = {'test': 'data', 'value': 123}
        
        response = formatter.format_success_response(data, status_code=200)
        
        assert response.status_code == 200
        assert response.data == data
        assert response.success == True
        assert response.timestamp is not None

    def test_format_error_response(self, formatter):
        """测试错误响应格式化"""
        error = APIError(
            code='VALIDATION_ERROR',
            message='Invalid request parameters',
            details={'field': 'symbol', 'reason': 'required'}
        )
        
        response = formatter.format_error_response(error, status_code=400)
        
        assert response.status_code == 400
        assert response.success == False
        assert response.error.code == 'VALIDATION_ERROR'
        assert response.error.message == 'Invalid request parameters'

    def test_format_paginated_response(self, formatter):
        """测试分页响应格式化"""
        items = [{'id': i, 'name': f'item_{i}'} for i in range(10)]
        
        response = formatter.format_paginated_response(
            items=items,
            total_count=100,
            page=1,
            limit=10
        )
        
        assert response.status_code == 200
        assert len(response.data['items']) == 10
        assert response.data['pagination']['total_count'] == 100
        assert response.data['pagination']['page'] == 1
        assert response.data['pagination']['limit'] == 10
        assert response.data['pagination']['total_pages'] == 10


class TestAPIPerformance:
    """API性能测试类"""

    @pytest.fixture
    def api_server(self):
        """创建API服务器用于性能测试"""
        config = APIConfig(
            host="127.0.0.1",
            port=8080,
            max_concurrent_requests=100
        )
        mock_trading_system = Mock()
        return APIServer(config=config, trading_system=mock_trading_system)

    def test_concurrent_request_handling(self, api_server):
        """测试并发请求处理"""
        requests = []
        for i in range(50):
            request = {
                'method': 'GET',
                'path': '/api/v1/portfolio/positions',
                'headers': {'Authorization': 'Bearer test_token'},
                'body': ''
            }
            requests.append(request)
        
        start_time = time.time()
        
        # 并发处理请求
        with ThreadPoolExecutor(max_workers=10) as executor:
            responses = list(executor.map(api_server.process_request, requests))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 验证性能
        assert total_time < 5.0  # 应该在5秒内完成50个请求
        assert len(responses) == 50
        
        # 验证响应时间分布
        response_times = [r.processing_time for r in responses if hasattr(r, 'processing_time')]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            assert avg_response_time < 0.1  # 平均响应时间应该小于100ms

    def test_memory_usage_monitoring(self, api_server):
        """测试内存使用监控"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 处理大量请求
        for i in range(100):
            request = {
                'method': 'GET',
                'path': '/api/v1/health',
                'headers': {},
                'body': ''
            }
            api_server.process_request(request)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 内存增长应该在合理范围内（小于50MB）
        assert memory_increase < 50 * 1024 * 1024

    def test_response_time_under_load(self, api_server):
        """测试负载下的响应时间"""
        response_times = []
        
        for i in range(20):
            request = {
                'method': 'GET',
                'path': '/api/v1/health',
                'headers': {},
                'body': ''
            }
            
            start_time = time.time()
            response = api_server.process_request(request)
            end_time = time.time()
            
            response_times.append(end_time - start_time)
        
        # 验证响应时间稳定性
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        assert avg_time < 0.05  # 平均响应时间小于50ms
        assert max_time < 0.2   # 最大响应时间小于200ms
        
        # 99%的请求应该在100ms内完成
        response_times.sort()
        p99_time = response_times[int(len(response_times) * 0.99)]
        assert p99_time < 0.1