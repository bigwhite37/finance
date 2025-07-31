"""
API认证系统实现
实现API密钥和JWT令牌认证系统，访问控制和权限管理
严格遵循TDD开发，不允许捕获异常，让异常暴露以尽早发现错误
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets
import jwt
import logging
import threading
from abc import ABC, abstractmethod


class PermissionLevel(Enum):
    """权限级别枚举"""
    READ_ONLY = "read_only"
    PORTFOLIO_READ = "portfolio_read"
    TRADING = "trading"
    RISK_MANAGEMENT = "risk_management"
    ADMIN = "admin"


class UserRole(Enum):
    """用户角色枚举"""
    VIEWER = "viewer"
    TRADER = "trader"
    MANAGER = "manager"
    ADMIN = "admin"


@dataclass
class APIKey:
    """API密钥数据类"""
    key_id: str
    user_id: str
    key_hash: str
    permissions: List[PermissionLevel]
    created_at: datetime
    expires_at: datetime
    is_active: bool
    name: str
    last_used: Optional[datetime] = None
    usage_count: int = 0


@dataclass
class JWTToken:
    """JWT令牌数据类"""
    token: str
    user_id: str
    expires_at: datetime
    token_type: str = "access"  # access or refresh
    issued_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuthResult:
    """认证结果数据类"""
    authenticated: bool
    user_id: Optional[str] = None
    permissions: List[PermissionLevel] = field(default_factory=list)
    auth_method: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    tokens: Optional[Dict[str, str]] = None


@dataclass
class AuthConfig:
    """认证配置"""
    api_key_enabled: bool = True
    jwt_enabled: bool = True
    session_timeout_minutes: int = 30
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 15
    require_2fa: bool = False
    password_policy: Dict[str, Any] = field(default_factory=dict)


class APIKeyAuthenticator:
    """API密钥认证器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._api_keys_cache = {}
        self._cache_lock = threading.Lock()
    
    def generate_api_key(self, user_id: str, permissions: List[PermissionLevel], 
                        name: str, expires_in_days: int = 365) -> tuple:
        """生成API密钥"""
        # 生成密钥ID和原始密钥
        key_id = f"key_{secrets.token_hex(8)}"
        # secrets.token_urlsafe(n) generates approximately 4n/3 characters, so we use 48 for ~64 chars
        key_length = self.config.get('key_length', 64)
        token_bytes = max(1, key_length * 3 // 4)  # Convert to approximate byte count
        raw_key = secrets.token_urlsafe(token_bytes)
        
        # 计算哈希
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # 创建API密钥对象
        api_key = APIKey(
            key_id=key_id,
            user_id=user_id,
            key_hash=key_hash,
            permissions=permissions,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=expires_in_days),
            is_active=True,
            name=name
        )
        
        # 存储到缓存（实际应该存储到数据库）
        with self._cache_lock:
            self._api_keys_cache[key_id] = api_key
        
        return api_key, raw_key
    
    def authenticate(self, api_key: str) -> AuthResult:
        """认证API密钥"""
        if not api_key:
            return AuthResult(
                authenticated=False,
                error_code='MISSING_API_KEY',
                error_message='API key is required'
            )
        
        # 从密钥中提取前缀来查找对应的密钥记录
        stored_key = self._get_api_key_by_prefix(api_key)
        if not stored_key:
            return AuthResult(
                authenticated=False,
                error_code='INVALID_API_KEY',
                error_message='Invalid API key'
            )
        
        # 验证密钥哈希
        provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if provided_hash != stored_key.key_hash:
            return AuthResult(
                authenticated=False,
                error_code='INVALID_API_KEY',
                error_message='Invalid API key'
            )
        
        # 检查密钥是否过期
        if datetime.now() > stored_key.expires_at:
            return AuthResult(
                authenticated=False,
                error_code='API_KEY_EXPIRED',
                error_message='API key has expired'
            )
        
        # 检查密钥是否活跃
        if not stored_key.is_active:
            return AuthResult(
                authenticated=False,
                error_code='API_KEY_INACTIVE',
                error_message='API key is inactive'
            )
        
        # 更新使用记录
        self._update_last_used(stored_key.key_id)
        
        return AuthResult(
            authenticated=True,
            user_id=stored_key.user_id,
            permissions=stored_key.permissions,
            auth_method='API_KEY'
        )
    
    def revoke_api_key(self, key_id: str) -> bool:
        """撤销API密钥"""
        with self._cache_lock:
            if key_id in self._api_keys_cache:
                self._api_keys_cache[key_id].is_active = False
                return True
        
        # 实际应该更新数据库
        return self._update_api_key(key_id, {'is_active': False})
    
    def list_user_api_keys(self, user_id: str) -> List[APIKey]:
        """列出用户的API密钥"""
        user_keys = []
        with self._cache_lock:
            for api_key in self._api_keys_cache.values():
                if api_key.user_id == user_id:
                    user_keys.append(api_key)
        
        return sorted(user_keys, key=lambda k: k.created_at, reverse=True)
    
    def _get_api_key_by_prefix(self, api_key: str) -> Optional[APIKey]:
        """通过密钥前缀获取API密钥（模拟实现）"""
        # 实际实现中应该通过数据库查询
        with self._cache_lock:
            for stored_key in self._api_keys_cache.values():
                if hashlib.sha256(api_key.encode()).hexdigest() == stored_key.key_hash:
                    return stored_key
        return None
    
    def _update_api_key(self, key_id: str, updates: Dict[str, Any]) -> bool:
        """更新API密钥（模拟实现）"""
        # 实际实现中应该更新数据库
        return True
    
    def _update_last_used(self, key_id: str):
        """更新最后使用时间"""
        with self._cache_lock:
            if key_id in self._api_keys_cache:
                self._api_keys_cache[key_id].last_used = datetime.now()
                self._api_keys_cache[key_id].usage_count += 1


class JWTAuthenticator:
    """JWT认证器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.blacklist = TokenBlacklist()
    
    def generate_tokens(self, user_data: Dict[str, Any]) -> tuple:
        """生成JWT令牌对"""
        # 当前时间
        now = datetime.utcnow()
        
        # 访问令牌载荷
        access_payload = {
            'user_id': user_data['user_id'],
            'username': user_data.get('username'),
            'email': user_data.get('email'),
            'role': user_data.get('role').value if hasattr(user_data.get('role'), 'value') else user_data.get('role'),
            'permissions': [p.value if hasattr(p, 'value') else p for p in user_data.get('permissions', [])],
            'iat': now,
            'exp': now + timedelta(minutes=self.config.get('access_token_expire_minutes', 30)),
            'iss': self.config.get('issuer', 'trading-system'),
            'aud': self.config.get('audience', 'api-clients'),
            'type': 'access'
        }
        
        # 刷新令牌载荷
        refresh_payload = {
            'user_id': user_data['user_id'],
            'iat': now,
            'exp': now + timedelta(days=self.config.get('refresh_token_expire_days', 7)),
            'iss': self.config.get('issuer', 'trading-system'),
            'aud': self.config.get('audience', 'api-clients'),
            'type': 'refresh'
        }
        
        # 生成令牌
        access_token = jwt.encode(
            access_payload,
            self.config['secret_key'],
            algorithm=self.config.get('algorithm', 'HS256')
        )
        
        refresh_token = jwt.encode(
            refresh_payload,
            self.config['secret_key'],
            algorithm=self.config.get('algorithm', 'HS256')
        )
        
        return access_token, refresh_token
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """解码JWT令牌"""
        # 这里不捕获异常，让它们暴露出来
        decoded_data = jwt.decode(
            token,
            self.config['secret_key'],
            algorithms=[self.config.get('algorithm', 'HS256')],
            audience=self.config.get('audience'),
            issuer=self.config.get('issuer')
        )
        
        return decoded_data
    
    def authenticate(self, auth_header: str) -> AuthResult:
        """使用JWT进行认证"""
        if not auth_header or not auth_header.startswith('Bearer '):
            return AuthResult(
                authenticated=False,
                error_code='MISSING_JWT_TOKEN',
                error_message='JWT token is required'
            )
        
        token = auth_header[7:]  # 移除 "Bearer " 前缀
        
        # 检查令牌是否在黑名单中
        if self.blacklist.is_blacklisted(token):
            return AuthResult(
                authenticated=False,
                error_code='TOKEN_BLACKLISTED',
                error_message='Token has been revoked'
            )
        
        # 解码令牌，如果有任何错误都会抛出异常
        decoded_data = self.decode_token(token)
        
        # 验证令牌类型
        if decoded_data.get('type') != 'access':
            return AuthResult(
                authenticated=False,
                error_code='INVALID_TOKEN_TYPE',
                error_message='Invalid token type for authentication'
            )
        
        # 转换权限
        permissions = []
        for perm in decoded_data.get('permissions', []):
            if isinstance(perm, str):
                permissions.append(PermissionLevel(perm))
            else:
                permissions.append(perm)
        
        return AuthResult(
            authenticated=True,
            user_id=decoded_data['user_id'],
            permissions=permissions,
            auth_method='JWT'
        )
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """刷新访问令牌"""
        # 解码刷新令牌
        decoded_data = self.decode_token(refresh_token)
        
        # 验证令牌类型
        if decoded_data.get('type') != 'refresh':
            raise ValueError("Invalid token type for refresh")
        
        # 生成新的访问令牌
        user_data = {
            'user_id': decoded_data['user_id'],
            'permissions': decoded_data.get('permissions', [])
        }
        
        access_token, _ = self.generate_tokens(user_data)
        return access_token
    
    def blacklist_token(self, token: str):
        """将令牌加入黑名单"""
        # 解码令牌获取过期时间
        decoded_data = self.decode_token(token)
        exp_timestamp = decoded_data.get('exp')
        if exp_timestamp:
            expiry = datetime.utcfromtimestamp(exp_timestamp)
            self.blacklist.add_token(token, expiry)


class AuthenticationManager:
    """认证管理器"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化认证器
        if config.api_key_enabled:
            self.api_key_authenticator = APIKeyAuthenticator({
                'hash_algorithm': 'sha256',
                'key_length': 64
            })
        
        if config.jwt_enabled:
            self.jwt_authenticator = JWTAuthenticator({
                'secret_key': 'default_secret_key',  # 应该从环境变量获取
                'algorithm': 'HS256',
                'access_token_expire_minutes': 30,
                'refresh_token_expire_days': 7,
                'issuer': 'trading-system',
                'audience': 'api-clients'
            })
        
        # 会话管理
        self.sessions = {}
        self.session_lock = threading.Lock()
        
        # 失败尝试跟踪
        self.failed_attempts = {}
        self.failed_attempts_lock = threading.Lock()
    
    def authenticate(self, api_key: str = None, jwt_token: str = None) -> AuthResult:
        """统一认证入口"""
        if api_key and self.config.api_key_enabled:
            return self.api_key_authenticator.authenticate(api_key)
        
        if jwt_token and self.config.jwt_enabled:
            return self.jwt_authenticator.authenticate(jwt_token)
        
        return AuthResult(
            authenticated=False,
            error_code='NO_CREDENTIALS',
            error_message='No valid credentials provided'
        )
    
    def login(self, username: str, password: str) -> AuthResult:
        """用户登录"""
        # 检查账户是否被锁定
        if self._is_account_locked(username):
            return AuthResult(
                authenticated=False,
                error_code='ACCOUNT_LOCKED',
                error_message='Account is temporarily locked due to too many failed attempts'
            )
        
        # 验证用户凭据
        user_data = self._verify_user_credentials(username, password)
        if not user_data:
            self._record_failed_attempt(username)
            return AuthResult(
                authenticated=False,
                error_code='INVALID_CREDENTIALS',
                error_message='Invalid username or password'
            )
        
        # 重置失败尝试计数
        self._reset_failed_attempts(username)
        
        # 生成JWT令牌
        if self.config.jwt_enabled:
            access_token, refresh_token = self.jwt_authenticator.generate_tokens(user_data)
            
            return AuthResult(
                authenticated=True,
                user_id=user_data['user_id'],
                permissions=user_data.get('permissions', []),
                auth_method='PASSWORD',
                tokens={
                    'access_token': access_token,
                    'refresh_token': refresh_token
                }
            )
        
        return AuthResult(authenticated=True, user_id=user_data['user_id'])
    
    def logout(self, access_token: str = None, refresh_token: str = None) -> bool:
        """用户登出"""
        success = True
        
        if access_token and self.config.jwt_enabled:
            self.jwt_authenticator.blacklist_token(access_token)
        
        if refresh_token and self.config.jwt_enabled:
            self.jwt_authenticator.blacklist_token(refresh_token)
        
        return success
    
    def has_permission(self, auth_result: AuthResult, required_permission: PermissionLevel) -> bool:
        """检查权限"""
        if not auth_result.authenticated:
            return False
        
        return required_permission in auth_result.permissions
    
    def create_session(self, session_data: Dict[str, Any]) -> str:
        """创建用户会话"""
        session_id = secrets.token_hex(32)
        
        with self.session_lock:
            self.sessions[session_id] = {
                **session_data,
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取用户会话"""
        with self.session_lock:
            return self.sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """更新会话活动时间"""
        with self.session_lock:
            if session_id in self.sessions:
                self.sessions[session_id]['last_activity'] = datetime.now()
    
    def destroy_session(self, session_id: str):
        """销毁用户会话"""
        with self.session_lock:
            self.sessions.pop(session_id, None)
    
    def _verify_user_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """验证用户凭据（模拟实现）"""
        # 实际实现中应该查询数据库并验证密码哈希
        if username == "test_user" and password == "correct_password":
            return {
                'user_id': 'user_001',
                'username': username,
                'role': UserRole.TRADER,
                'permissions': [PermissionLevel.TRADING, PermissionLevel.PORTFOLIO_READ]
            }
        return None
    
    def _record_failed_attempt(self, username: str):
        """记录失败尝试"""
        with self.failed_attempts_lock:
            if username not in self.failed_attempts:
                self.failed_attempts[username] = {
                    'count': 0,
                    'first_attempt': datetime.now()
                }
            
            self.failed_attempts[username]['count'] += 1
            self.failed_attempts[username]['last_attempt'] = datetime.now()
    
    def _reset_failed_attempts(self, username: str):
        """重置失败尝试"""
        with self.failed_attempts_lock:
            self.failed_attempts.pop(username, None)
    
    def _is_account_locked(self, username: str) -> bool:
        """检查账户是否被锁定"""
        with self.failed_attempts_lock:
            if username not in self.failed_attempts:
                return False
            
            attempt_data = self.failed_attempts[username]
            if attempt_data['count'] >= self.config.max_failed_attempts:
                # 检查锁定时间是否已过
                lockout_duration = timedelta(minutes=self.config.lockout_duration_minutes)
                if datetime.now() - attempt_data['last_attempt'] < lockout_duration:
                    return True
                else:
                    # 锁定时间已过，重置计数
                    self.failed_attempts.pop(username)
            
            return False


class SecurityAuditLog:
    """安全审计日志"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.log_entries = []
        self.log_lock = threading.Lock()
    
    def log_authentication_success(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录认证成功事件"""
        log_entry = {
            'event_type': 'AUTHENTICATION_SUCCESS',
            'severity': 'INFO',
            'timestamp': event_data.get('timestamp', datetime.now()),
            'user_id': event_data.get('user_id'),
            'auth_method': event_data.get('auth_method'),
            'ip_address': event_data.get('ip_address'),
            'user_agent': event_data.get('user_agent')
        }
        
        with self.log_lock:
            self.log_entries.append(log_entry)
        
        return log_entry
    
    def log_authentication_failure(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录认证失败事件"""
        log_entry = {
            'event_type': 'AUTHENTICATION_FAILURE',
            'severity': 'WARNING',
            'timestamp': event_data.get('timestamp', datetime.now()),
            'username': event_data.get('username'),
            'reason': event_data.get('reason'),
            'ip_address': event_data.get('ip_address'),
            'user_agent': event_data.get('user_agent')
        }
        
        with self.log_lock:
            self.log_entries.append(log_entry)
        
        return log_entry
    
    def log_suspicious_activity(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """记录可疑活动事件"""
        log_entry = {
            'event_type': 'SUSPICIOUS_ACTIVITY',
            'severity': 'HIGH',
            'timestamp': event_data.get('timestamp', datetime.now()),
            'user_id': event_data.get('user_id'),
            'activity_type': event_data.get('activity_type'),
            'details': event_data.get('details', {}),
            'ip_address': event_data.get('ip_address')
        }
        
        with self.log_lock:
            self.log_entries.append(log_entry)
        
        return log_entry
    
    def query_logs(self, user_id: str = None, event_type: str = None,
                  start_time: datetime = None, end_time: datetime = None) -> List[Dict[str, Any]]:
        """查询审计日志"""
        with self.log_lock:
            filtered_logs = self.log_entries.copy()
        
        # 应用过滤条件
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.get('user_id') == user_id]
        
        if event_type:
            filtered_logs = [log for log in filtered_logs if log.get('event_type') == event_type]
        
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.get('timestamp') >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.get('timestamp') <= end_time]
        
        # 按时间倒序排列
        filtered_logs.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        
        return filtered_logs
    
    def generate_security_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """生成安全指标"""
        logs_in_period = self.query_logs(start_time=start_time, end_time=end_time)
        
        total_attempts = len(logs_in_period)
        successful_auths = len([log for log in logs_in_period 
                               if log.get('event_type') == 'AUTHENTICATION_SUCCESS'])
        failed_auths = len([log for log in logs_in_period 
                           if log.get('event_type') == 'AUTHENTICATION_FAILURE'])
        
        success_rate = successful_auths / total_attempts if total_attempts > 0 else 0
        
        return {
            'total_authentication_attempts': total_attempts,
            'successful_authentications': successful_auths,
            'failed_authentications': failed_auths,
            'success_rate': success_rate,
            'suspicious_activities': len([log for log in logs_in_period 
                                        if log.get('event_type') == 'SUSPICIOUS_ACTIVITY']),
            'period_start': start_time.isoformat(),
            'period_end': end_time.isoformat()
        }


class TokenBlacklist:
    """令牌黑名单"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.blacklisted_tokens = {}  # token -> expiry_time
        self.blacklist_lock = threading.Lock()
    
    def add_token(self, token: str, expiry: datetime) -> bool:
        """添加令牌到黑名单"""
        with self.blacklist_lock:
            # 如果黑名单已满，清理过期令牌
            if len(self.blacklisted_tokens) >= self.max_size:
                self.cleanup_expired_tokens()
            
            # 如果仍然满了，移除最旧的令牌
            if len(self.blacklisted_tokens) >= self.max_size:
                oldest_token = min(self.blacklisted_tokens.keys(), 
                                 key=lambda k: self.blacklisted_tokens[k])
                del self.blacklisted_tokens[oldest_token]
            
            self.blacklisted_tokens[token] = expiry
            return True
    
    def is_blacklisted(self, token: str) -> bool:
        """检查令牌是否在黑名单中"""
        with self.blacklist_lock:
            if token in self.blacklisted_tokens:
                # 检查是否过期
                if datetime.now() > self.blacklisted_tokens[token]:
                    del self.blacklisted_tokens[token]
                    return False
                return True
            return False
    
    def cleanup_expired_tokens(self) -> int:
        """清理过期令牌"""
        with self.blacklist_lock:
            current_time = datetime.now()
            expired_tokens = [token for token, expiry in self.blacklisted_tokens.items() 
                            if current_time > expiry]
            
            for token in expired_tokens:
                del self.blacklisted_tokens[token]
            
            return len(expired_tokens)
    
    def get_size(self) -> int:
        """获取黑名单大小"""
        with self.blacklist_lock:
            return len(self.blacklisted_tokens)