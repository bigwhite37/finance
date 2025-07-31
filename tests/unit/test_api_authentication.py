"""
API认证系统的单元测试
测试API密钥和JWT令牌认证机制，认证的安全性和访问控制
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""
import pytest
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

from src.rl_trading_system.api.authentication import (
    APIKeyAuthenticator,
    JWTAuthenticator,
    AuthenticationManager,
    APIKey,
    JWTToken,
    AuthResult,
    PermissionLevel,
    UserRole,
    AuthConfig,
    TokenBlacklist,
    SecurityAuditLog
)


class TestAPIKeyAuthenticator:
    """API密钥认证器测试类"""

    @pytest.fixture
    def api_key_config(self):
        """创建API密钥配置"""
        return {
            'hash_algorithm': 'sha256',
            'key_length': 64,
            'expiry_days': 365,
            'max_keys_per_user': 5
        }

    @pytest.fixture
    def api_key_authenticator(self, api_key_config):
        """创建API密钥认证器"""
        return APIKeyAuthenticator(config=api_key_config)

    @pytest.fixture
    def sample_api_key(self):
        """创建样本API密钥"""
        return APIKey(
            key_id='key_123456',
            user_id='user_001',
            key_hash='abc123def456',
            permissions=[PermissionLevel.TRADING, PermissionLevel.PORTFOLIO_READ],
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365),
            is_active=True,
            name='Main Trading Key',
            last_used=None
        )

    def test_generate_api_key(self, api_key_authenticator):
        """测试生成API密钥"""
        user_id = 'user_001'
        permissions = [PermissionLevel.TRADING, PermissionLevel.PORTFOLIO_READ]
        name = 'Test Key'
        
        api_key, raw_key = api_key_authenticator.generate_api_key(
            user_id=user_id,
            permissions=permissions,
            name=name,
            expires_in_days=30
        )
        
        assert api_key.user_id == user_id
        assert api_key.permissions == permissions
        assert api_key.name == name
        assert api_key.is_active == True
        assert len(raw_key) == 64  # 密钥长度
        assert api_key.key_hash != raw_key  # 哈希值不等于原始密钥
        assert api_key.expires_at > datetime.now()

    def test_authenticate_valid_key(self, api_key_authenticator, sample_api_key):
        """测试验证有效API密钥"""
        # 模拟存储的API密钥
        raw_key = 'test_api_key_12345'
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        sample_api_key.key_hash = key_hash
        
        # 模拟从数据库获取密钥
        with patch.object(api_key_authenticator, '_get_api_key_by_prefix') as mock_get:
            mock_get.return_value = sample_api_key
            
            auth_result = api_key_authenticator.authenticate(raw_key)
            
            assert auth_result.authenticated == True
            assert auth_result.user_id == sample_api_key.user_id
            assert auth_result.permissions == sample_api_key.permissions
            assert auth_result.auth_method == 'API_KEY'

    def test_authenticate_invalid_key(self, api_key_authenticator):
        """测试验证无效API密钥"""
        invalid_key = 'invalid_api_key'
        
        with patch.object(api_key_authenticator, '_get_api_key_by_prefix') as mock_get:
            mock_get.return_value = None
            
            auth_result = api_key_authenticator.authenticate(invalid_key)
            
            assert auth_result.authenticated == False
            assert auth_result.user_id is None
            assert auth_result.error_code == 'INVALID_API_KEY'

    def test_authenticate_expired_key(self, api_key_authenticator, sample_api_key):
        """测试验证过期API密钥"""
        # 设置密钥为过期
        sample_api_key.expires_at = datetime.now() - timedelta(days=1)
        
        raw_key = 'test_api_key_12345'
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        sample_api_key.key_hash = key_hash
        
        with patch.object(api_key_authenticator, '_get_api_key_by_prefix') as mock_get:
            mock_get.return_value = sample_api_key
            
            auth_result = api_key_authenticator.authenticate(raw_key)
            
            assert auth_result.authenticated == False
            assert auth_result.error_code == 'API_KEY_EXPIRED'

    def test_authenticate_inactive_key(self, api_key_authenticator, sample_api_key):
        """测试验证非活跃API密钥"""
        # 设置密钥为非活跃
        sample_api_key.is_active = False
        
        raw_key = 'test_api_key_12345'
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        sample_api_key.key_hash = key_hash
        
        with patch.object(api_key_authenticator, '_get_api_key_by_prefix') as mock_get:
            mock_get.return_value = sample_api_key
            
            auth_result = api_key_authenticator.authenticate(raw_key)
            
            assert auth_result.authenticated == False
            assert auth_result.error_code == 'API_KEY_INACTIVE'

    def test_revoke_api_key(self, api_key_authenticator, sample_api_key):
        """测试撤销API密钥"""
        key_id = sample_api_key.key_id
        
        with patch.object(api_key_authenticator, '_update_api_key') as mock_update:
            result = api_key_authenticator.revoke_api_key(key_id)
            
            assert result == True
            mock_update.assert_called_once()
            
            # 验证撤销操作的参数
            call_args = mock_update.call_args[0]
            assert call_args[0] == key_id
            assert call_args[1]['is_active'] == False

    def test_list_user_api_keys(self, api_key_authenticator):
        """测试列出用户API密钥"""
        user_id = 'user_001'
        
        mock_keys = [
            APIKey(
                key_id=f'key_{i}',
                user_id=user_id,
                key_hash=f'hash_{i}',
                permissions=[PermissionLevel.PORTFOLIO_READ],
                created_at=datetime.now() - timedelta(days=i),
                expires_at=datetime.now() + timedelta(days=365-i),
                is_active=True,
                name=f'Key {i}',
                last_used=datetime.now() - timedelta(hours=i)
            )
            for i in range(3)
        ]
        
        with patch.object(api_key_authenticator, '_get_user_api_keys') as mock_get:
            mock_get.return_value = mock_keys
            
            keys = api_key_authenticator.list_user_api_keys(user_id)
            
            assert len(keys) == 3
            assert all(key.user_id == user_id for key in keys)
            assert keys[0].name == 'Key 0'

    def test_api_key_usage_tracking(self, api_key_authenticator, sample_api_key):
        """测试API密钥使用追踪"""
        raw_key = 'test_api_key_12345'
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        sample_api_key.key_hash = key_hash
        
        with patch.object(api_key_authenticator, '_get_api_key_by_prefix') as mock_get:
            with patch.object(api_key_authenticator, '_update_last_used') as mock_update:
                mock_get.return_value = sample_api_key
                
                auth_result = api_key_authenticator.authenticate(raw_key)
                
                assert auth_result.authenticated == True
                mock_update.assert_called_once_with(sample_api_key.key_id)


class TestJWTAuthenticator:
    """JWT认证器测试类"""

    @pytest.fixture
    def jwt_config(self):
        """创建JWT配置"""
        return {
            'secret_key': 'test_secret_key_12345',
            'algorithm': 'HS256',
            'access_token_expire_minutes': 30,
            'refresh_token_expire_days': 7,
            'issuer': 'trading-system',
            'audience': 'api-clients'
        }

    @pytest.fixture
    def jwt_authenticator(self, jwt_config):
        """创建JWT认证器"""
        return JWTAuthenticator(config=jwt_config)

    @pytest.fixture
    def sample_user_data(self):
        """创建样本用户数据"""
        return {
            'user_id': 'user_001',
            'username': 'test_user',
            'email': 'test@example.com',
            'role': UserRole.TRADER,
            'permissions': [PermissionLevel.TRADING, PermissionLevel.PORTFOLIO_READ]
        }

    def test_generate_jwt_token(self, jwt_authenticator, sample_user_data):
        """测试生成JWT令牌"""
        access_token, refresh_token = jwt_authenticator.generate_tokens(sample_user_data)
        
        assert access_token is not None
        assert refresh_token is not None
        assert len(access_token) > 50  # JWT令牌应该有一定长度
        assert len(refresh_token) > 50
        
        # 验证令牌结构
        assert access_token.count('.') == 2  # JWT应该有3个部分
        assert refresh_token.count('.') == 2

    def test_decode_valid_jwt_token(self, jwt_authenticator, sample_user_data, jwt_config):
        """测试解码有效JWT令牌"""
        access_token, _ = jwt_authenticator.generate_tokens(sample_user_data)
        
        decoded_data = jwt_authenticator.decode_token(access_token)
        
        assert decoded_data['user_id'] == sample_user_data['user_id']
        assert decoded_data['username'] == sample_user_data['username']
        assert decoded_data['role'] == sample_user_data['role'].value
        assert decoded_data['iss'] == jwt_config['issuer']
        assert decoded_data['aud'] == jwt_config['audience']

    def test_decode_expired_jwt_token(self, jwt_authenticator, sample_user_data):
        """测试解码过期JWT令牌"""
        # 创建一个立即过期的令牌
        expired_payload = {
            **sample_user_data,
            'exp': datetime.utcnow() - timedelta(minutes=1),
            'iat': datetime.utcnow() - timedelta(minutes=2)
        }
        
        expired_token = jwt.encode(
            expired_payload,
            jwt_authenticator.config['secret_key'],
            algorithm=jwt_authenticator.config['algorithm']
        )
        
        # 解码应该引发异常而不是捕获
        with pytest.raises(jwt.ExpiredSignatureError):
            jwt_authenticator.decode_token(expired_token)

    def test_decode_invalid_jwt_token(self, jwt_authenticator):
        """测试解码无效JWT令牌"""
        invalid_token = 'invalid.jwt.token'
        
        # 解码应该引发异常而不是捕获
        with pytest.raises(jwt.DecodeError):
            jwt_authenticator.decode_token(invalid_token)

    def test_authenticate_with_valid_jwt(self, jwt_authenticator, sample_user_data):
        """测试使用有效JWT进行认证"""
        access_token, _ = jwt_authenticator.generate_tokens(sample_user_data)
        
        auth_result = jwt_authenticator.authenticate(f"Bearer {access_token}")
        
        assert auth_result.authenticated == True
        assert auth_result.user_id == sample_user_data['user_id']
        assert auth_result.permissions == sample_user_data['permissions']
        assert auth_result.auth_method == 'JWT'

    def test_authenticate_with_invalid_jwt(self, jwt_authenticator):
        """测试使用无效JWT进行认证"""
        invalid_token = 'Bearer invalid.jwt.token'
        
        auth_result = jwt_authenticator.authenticate(invalid_token)
        
        assert auth_result.authenticated == False
        assert auth_result.error_code == 'INVALID_JWT_TOKEN'

    def test_refresh_jwt_token(self, jwt_authenticator, sample_user_data):
        """测试刷新JWT令牌"""
        _, refresh_token = jwt_authenticator.generate_tokens(sample_user_data)
        
        new_access_token = jwt_authenticator.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        assert len(new_access_token) > 50
        
        # 验证新令牌有效
        decoded_data = jwt_authenticator.decode_token(new_access_token)
        assert decoded_data['user_id'] == sample_user_data['user_id']

    def test_jwt_token_blacklist(self, jwt_authenticator, sample_user_data):
        """测试JWT令牌黑名单"""
        access_token, _ = jwt_authenticator.generate_tokens(sample_user_data)
        
        # 将令牌加入黑名单
        jwt_authenticator.blacklist_token(access_token)
        
        # 验证黑名单中的令牌无法通过认证
        auth_result = jwt_authenticator.authenticate(f"Bearer {access_token}")
        
        assert auth_result.authenticated == False
        assert auth_result.error_code == 'TOKEN_BLACKLISTED'


class TestAuthenticationManager:
    """认证管理器测试类"""

    @pytest.fixture
    def auth_config(self):
        """创建认证配置"""
        return AuthConfig(
            api_key_enabled=True,
            jwt_enabled=True,
            session_timeout_minutes=30,
            max_failed_attempts=5,
            lockout_duration_minutes=15,
            require_2fa=False,
            password_policy={
                'min_length': 8,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_symbols': True
            }
        )

    @pytest.fixture
    def auth_manager(self, auth_config):
        """创建认证管理器"""
        return AuthenticationManager(config=auth_config)

    def test_authenticate_with_api_key(self, auth_manager):
        """测试使用API密钥认证"""
        api_key = 'test_api_key_12345'
        
        # 模拟API密钥认证器
        with patch.object(auth_manager.api_key_authenticator, 'authenticate') as mock_auth:
            mock_auth.return_value = AuthResult(
                authenticated=True,
                user_id='user_001',
                permissions=[PermissionLevel.TRADING],
                auth_method='API_KEY'
            )
            
            auth_result = auth_manager.authenticate(api_key=api_key)
            
            assert auth_result.authenticated == True
            assert auth_result.auth_method == 'API_KEY'
            mock_auth.assert_called_once_with(api_key)

    def test_authenticate_with_jwt_token(self, auth_manager):
        """测试使用JWT令牌认证"""
        jwt_token = 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...'
        
        # 模拟JWT认证器
        with patch.object(auth_manager.jwt_authenticator, 'authenticate') as mock_auth:
            mock_auth.return_value = AuthResult(
                authenticated=True,
                user_id='user_001',
                permissions=[PermissionLevel.PORTFOLIO_READ],
                auth_method='JWT'
            )
            
            auth_result = auth_manager.authenticate(jwt_token=jwt_token)
            
            assert auth_result.authenticated == True
            assert auth_result.auth_method == 'JWT'
            mock_auth.assert_called_once_with(jwt_token)

    def test_login_with_credentials(self, auth_manager):
        """测试使用用户名密码登录"""
        username = 'test_user'
        password = 'SecurePassword123!'
        
        # 模拟用户验证
        with patch.object(auth_manager, '_verify_user_credentials') as mock_verify:
            mock_verify.return_value = {
                'user_id': 'user_001',
                'username': username,
                'role': UserRole.TRADER,
                'permissions': [PermissionLevel.TRADING, PermissionLevel.PORTFOLIO_READ]
            }
            
            with patch.object(auth_manager.jwt_authenticator, 'generate_tokens') as mock_generate:
                mock_generate.return_value = ('access_token', 'refresh_token')
                
                auth_result = auth_manager.login(username, password)
                
                assert auth_result.authenticated == True
                assert auth_result.user_id == 'user_001'
                assert 'access_token' in auth_result.tokens
                assert 'refresh_token' in auth_result.tokens

    def test_login_with_invalid_credentials(self, auth_manager):
        """测试使用无效凭据登录"""
        username = 'test_user'
        password = 'wrong_password'
        
        with patch.object(auth_manager, '_verify_user_credentials') as mock_verify:
            mock_verify.return_value = None
            
            auth_result = auth_manager.login(username, password)
            
            assert auth_result.authenticated == False
            assert auth_result.error_code == 'INVALID_CREDENTIALS'

    def test_failed_login_attempts_tracking(self, auth_manager):
        """测试失败登录尝试追踪"""
        username = 'test_user'
        password = 'wrong_password'
        
        with patch.object(auth_manager, '_verify_user_credentials') as mock_verify:
            mock_verify.return_value = None
            
            # 多次失败登录
            for i in range(6):
                auth_result = auth_manager.login(username, password)
                
                if i < 5:
                    assert auth_result.error_code == 'INVALID_CREDENTIALS'
                else:
                    # 第6次应该被锁定
                    assert auth_result.error_code == 'ACCOUNT_LOCKED'

    def test_logout_functionality(self, auth_manager):
        """测试登出功能"""
        access_token = 'valid_access_token'
        refresh_token = 'valid_refresh_token'
        
        with patch.object(auth_manager.jwt_authenticator, 'blacklist_token') as mock_blacklist:
            result = auth_manager.logout(access_token, refresh_token)
            
            assert result == True
            assert mock_blacklist.call_count == 2  # 两个令牌都应该被加入黑名单

    def test_permission_checking(self, auth_manager):
        """测试权限检查"""
        auth_result = AuthResult(
            authenticated=True,
            user_id='user_001',
            permissions=[PermissionLevel.TRADING, PermissionLevel.PORTFOLIO_READ],
            auth_method='JWT'
        )
        
        # 测试有权限的操作
        assert auth_manager.has_permission(auth_result, PermissionLevel.TRADING) == True
        assert auth_manager.has_permission(auth_result, PermissionLevel.PORTFOLIO_READ) == True
        
        # 测试无权限的操作
        assert auth_manager.has_permission(auth_result, PermissionLevel.ADMIN) == False

    def test_session_management(self, auth_manager):
        """测试会话管理"""
        user_id = 'user_001'
        session_data = {
            'user_id': user_id,
            'login_time': datetime.now(),
            'last_activity': datetime.now(),
            'ip_address': '192.168.1.100',
            'user_agent': 'TestClient/1.0'
        }
        
        # 创建会话
        session_id = auth_manager.create_session(session_data)
        assert session_id is not None
        assert len(session_id) > 20
        
        # 获取会话
        retrieved_session = auth_manager.get_session(session_id)
        assert retrieved_session['user_id'] == user_id
        
        # 更新会话活动时间
        auth_manager.update_session_activity(session_id)
        updated_session = auth_manager.get_session(session_id)
        assert updated_session['last_activity'] > session_data['last_activity']
        
        # 销毁会话
        auth_manager.destroy_session(session_id)
        destroyed_session = auth_manager.get_session(session_id)
        assert destroyed_session is None


class TestSecurityAuditLog:
    """安全审计日志测试类"""

    @pytest.fixture
    def audit_log(self):
        """创建安全审计日志"""
        return SecurityAuditLog()

    def test_log_authentication_success(self, audit_log):
        """测试记录认证成功日志"""
        event_data = {
            'user_id': 'user_001',
            'auth_method': 'JWT',
            'ip_address': '192.168.1.100',
            'user_agent': 'TestClient/1.0',
            'timestamp': datetime.now()
        }
        
        log_entry = audit_log.log_authentication_success(event_data)
        
        assert log_entry['event_type'] == 'AUTHENTICATION_SUCCESS'
        assert log_entry['user_id'] == 'user_001'
        assert log_entry['severity'] == 'INFO'
        assert log_entry['timestamp'] is not None

    def test_log_authentication_failure(self, audit_log):
        """测试记录认证失败日志"""
        event_data = {
            'username': 'test_user',
            'reason': 'INVALID_CREDENTIALS',
            'ip_address': '192.168.1.100',
            'user_agent': 'TestClient/1.0',
            'timestamp': datetime.now()
        }
        
        log_entry = audit_log.log_authentication_failure(event_data)
        
        assert log_entry['event_type'] == 'AUTHENTICATION_FAILURE'
        assert log_entry['username'] == 'test_user'
        assert log_entry['reason'] == 'INVALID_CREDENTIALS'
        assert log_entry['severity'] == 'WARNING'

    def test_log_suspicious_activity(self, audit_log):
        """测试记录可疑活动日志"""
        event_data = {
            'user_id': 'user_001',
            'activity_type': 'MULTIPLE_FAILED_LOGINS',
            'details': {'failed_attempts': 5, 'time_window_minutes': 5},
            'ip_address': '192.168.1.100',
            'timestamp': datetime.now()
        }
        
        log_entry = audit_log.log_suspicious_activity(event_data)
        
        assert log_entry['event_type'] == 'SUSPICIOUS_ACTIVITY'
        assert log_entry['activity_type'] == 'MULTIPLE_FAILED_LOGINS'
        assert log_entry['severity'] == 'HIGH'
        assert log_entry['details']['failed_attempts'] == 5

    def test_query_audit_logs(self, audit_log):
        """测试查询审计日志"""
        # 添加一些测试日志
        for i in range(10):
            audit_log.log_authentication_success({
                'user_id': f'user_{i:03d}',
                'auth_method': 'API_KEY',
                'ip_address': f'192.168.1.{100+i}',
                'timestamp': datetime.now() - timedelta(hours=i)
            })
        
        # 按用户ID查询
        user_logs = audit_log.query_logs(user_id='user_005')
        assert len(user_logs) == 1
        assert user_logs[0]['user_id'] == 'user_005'
        
        # 按时间范围查询
        start_time = datetime.now() - timedelta(hours=5)
        end_time = datetime.now()
        recent_logs = audit_log.query_logs(start_time=start_time, end_time=end_time)
        assert len(recent_logs) >= 5
        
        # 按事件类型查询
        auth_logs = audit_log.query_logs(event_type='AUTHENTICATION_SUCCESS')
        assert len(auth_logs) == 10
        assert all(log['event_type'] == 'AUTHENTICATION_SUCCESS' for log in auth_logs)

    def test_security_metrics_generation(self, audit_log):
        """测试安全指标生成"""
        # 添加各种类型的日志
        for i in range(20):
            if i % 4 == 0:
                audit_log.log_authentication_failure({
                    'username': f'user_{i}',
                    'reason': 'INVALID_CREDENTIALS',
                    'ip_address': '192.168.1.100',
                    'timestamp': datetime.now() - timedelta(minutes=i)
                })
            else:
                audit_log.log_authentication_success({
                    'user_id': f'user_{i}',
                    'auth_method': 'JWT',
                    'ip_address': '192.168.1.100',
                    'timestamp': datetime.now() - timedelta(minutes=i)
                })
        
        metrics = audit_log.generate_security_metrics(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        assert 'total_authentication_attempts' in metrics
        assert 'successful_authentications' in metrics
        assert 'failed_authentications' in metrics
        assert 'success_rate' in metrics
        assert metrics['total_authentication_attempts'] == 20
        assert metrics['successful_authentications'] == 15
        assert metrics['failed_authentications'] == 5
        assert metrics['success_rate'] == 0.75


class TestTokenBlacklist:
    """令牌黑名单测试类"""

    @pytest.fixture
    def token_blacklist(self):
        """创建令牌黑名单"""
        return TokenBlacklist()

    def test_add_token_to_blacklist(self, token_blacklist):
        """测试添加令牌到黑名单"""
        token = 'test_token_12345'
        expiry = datetime.now() + timedelta(hours=1)
        
        result = token_blacklist.add_token(token, expiry)
        
        assert result == True
        assert token_blacklist.is_blacklisted(token) == True

    def test_remove_expired_tokens(self, token_blacklist):
        """测试移除过期令牌"""
        # 添加已过期的令牌
        expired_token = 'expired_token'
        expired_expiry = datetime.now() - timedelta(hours=1)
        token_blacklist.add_token(expired_token, expired_expiry)
        
        # 添加未过期的令牌
        valid_token = 'valid_token'
        valid_expiry = datetime.now() + timedelta(hours=1)
        token_blacklist.add_token(valid_token, valid_expiry)
        
        # 清理过期令牌
        removed_count = token_blacklist.cleanup_expired_tokens()
        
        assert removed_count == 1
        assert token_blacklist.is_blacklisted(expired_token) == False
        assert token_blacklist.is_blacklisted(valid_token) == True

    def test_blacklist_size_limit(self, token_blacklist):
        """测试黑名单大小限制"""
        # 假设黑名单最大容量为1000
        max_size = 1000
        
        # 添加超过限制的令牌
        for i in range(max_size + 10):
            token = f'token_{i}'
            expiry = datetime.now() + timedelta(hours=1)
            token_blacklist.add_token(token, expiry)
        
        # 验证黑名单大小不超过限制
        current_size = token_blacklist.get_size()
        assert current_size <= max_size

    def test_concurrent_blacklist_operations(self, token_blacklist):
        """测试并发黑名单操作"""
        import threading
        import time
        
        tokens_added = []
        
        def add_tokens(start_index, count):
            for i in range(start_index, start_index + count):
                token = f'concurrent_token_{i}'
                expiry = datetime.now() + timedelta(hours=1)
                token_blacklist.add_token(token, expiry)
                tokens_added.append(token)
                time.sleep(0.001)  # 小延迟模拟实际操作
        
        # 创建多个线程并发添加令牌
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_tokens, args=(i * 10, 10))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有令牌都被正确添加
        assert len(tokens_added) == 50
        for token in tokens_added:
            assert token_blacklist.is_blacklisted(token) == True