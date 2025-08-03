"""
回撤控制配置管理器

提供回撤控制系统的配置管理，包括动态加载、热更新、参数验证和默认值处理。
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..backtest.drawdown_control_config import DrawdownControlConfig
from .config_validator import ConfigValidator

logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件监控处理器"""
    
    def __init__(self, config_manager: 'DrawdownControlConfigManager'):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__ + '.ConfigFileHandler')
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        if event.src_path in self.config_manager.watched_files:
            self.logger.info(f"检测到配置文件变更: {event.src_path}")
            # 延迟重新加载，避免文件写入过程中的读取
            threading.Timer(1.0, self.config_manager._reload_config_file, 
                          args=[event.src_path]).start()


@dataclass
class ConfigVersion:
    """配置版本信息"""
    version: str
    timestamp: datetime
    source_file: str
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'source_file': self.source_file,
            'checksum': self.checksum
        }


class DrawdownControlConfigManager:
    """
    回撤控制配置管理器
    
    提供完整的配置管理功能：
    - 配置文件加载和保存
    - 动态热重载
    - 参数验证和默认值处理
    - 配置版本管理
    """
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 enable_hot_reload: bool = True,
                 auto_backup: bool = True):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
            enable_hot_reload: 是否启用热重载
            auto_backup: 是否自动备份配置
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.enable_hot_reload = enable_hot_reload
        self.auto_backup = auto_backup
        self.logger = logging.getLogger(__name__)
        
        # 配置状态
        self.current_config: Optional[DrawdownControlConfig] = None
        self.config_history: List[ConfigVersion] = []
        self.config_callbacks: List[Callable[[DrawdownControlConfig], None]] = []
        
        # 文件监控
        self.watched_files: List[str] = []
        self.file_observer: Optional[Observer] = None
        self.config_lock = threading.Lock()
        
        # 验证器
        self.validator = ConfigValidator()
        
        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"初始化回撤控制配置管理器，配置目录: {self.config_dir}")
    
    def load_config(self, 
                   config_file: Optional[str] = None,
                   config_dict: Optional[Dict[str, Any]] = None) -> DrawdownControlConfig:
        """
        加载配置
        
        Args:
            config_file: 配置文件路径
            config_dict: 配置字典（优先级高于文件）
            
        Returns:
            回撤控制配置对象
        """
        if config_dict:
            return self._load_from_dict(config_dict)
        
        if config_file:
            return self._load_from_file(config_file)
        
        # 尝试加载默认配置文件
        default_files = [
            self.config_dir / "drawdown_control.yaml",
            self.config_dir / "drawdown_control.yml", 
            self.config_dir / "drawdown_control.json"
        ]
        
        for file_path in default_files:
            if file_path.exists():
                return self._load_from_file(str(file_path))
        
        # 如果没有找到配置文件，返回默认配置
        self.logger.warning("未找到配置文件，使用默认配置")
        return self._create_default_config()
    
    def _load_from_file(self, config_file: str) -> DrawdownControlConfig:
        """从文件加载配置"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 提取回撤控制配置部分
            drawdown_config_data = config_data.get('drawdown_control', config_data)
            
            # 验证配置
            validation_result = self.validator.validate_drawdown_control_config(drawdown_config_data)
            if not validation_result['valid']:
                raise ValueError(f"配置验证失败: {validation_result['errors']}")
            
            # 创建配置对象
            config = DrawdownControlConfig.from_dict(drawdown_config_data)
            
            # 记录配置版本
            self._record_config_version(config, str(config_path))
            
            # 设置文件监控
            if self.enable_hot_reload:
                self._add_file_watch(str(config_path))
            
            # 更新当前配置
            with self.config_lock:
                self.current_config = config
            
            self.logger.info(f"成功加载配置文件: {config_file}")
            
            # 触发配置更新回调
            self._trigger_config_callbacks(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"加载配置文件失败 {config_file}: {e}")
            raise RuntimeError(f"加载配置文件失败 {config_file}: {e}")
    
    def _load_from_dict(self, config_dict: Dict[str, Any]) -> DrawdownControlConfig:
        """从字典加载配置"""
        try:
            # 验证配置
            validation_result = self.validator.validate_drawdown_control_config(config_dict)
            if not validation_result['valid']:
                raise ValueError(f"配置验证失败: {validation_result['errors']}")
            
            # 创建配置对象
            config = DrawdownControlConfig.from_dict(config_dict)
            
            # 更新当前配置
            with self.config_lock:
                self.current_config = config
            
            self.logger.info("成功从字典加载配置")
            
            # 触发配置更新回调
            self._trigger_config_callbacks(config)
            
            return config
            
        except Exception as e:
            self.logger.error(f"从字典加载配置失败: {e}")
            raise RuntimeError(f"从字典加载配置失败: {e}")
    
    def _create_default_config(self) -> DrawdownControlConfig:
        """创建默认配置"""
        config = DrawdownControlConfig()
        
        # 记录默认配置版本
        self._record_config_version(config, "default")
        
        # 更新当前配置
        with self.config_lock:
            self.current_config = config
        
        self.logger.info("创建默认配置")
        
        # 触发配置更新回调
        self._trigger_config_callbacks(config)
        
        return config
    
    def save_config(self, 
                   config: DrawdownControlConfig,
                   output_file: Optional[str] = None,
                   format: str = 'yaml') -> str:
        """
        保存配置到文件
        
        Args:
            config: 配置对象
            output_file: 输出文件路径
            format: 文件格式 ('yaml' 或 'json')
            
        Returns:
            保存的文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format == 'yaml':
                output_file = self.config_dir / f"drawdown_control_{timestamp}.yaml"
            else:
                output_file = self.config_dir / f"drawdown_control_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = {'drawdown_control': config.to_dict()}
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if format == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"配置已保存到: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"保存配置失败 {output_path}: {e}")
            raise RuntimeError(f"保存配置失败 {output_path}: {e}")
    
    def reload_config(self, config_file: Optional[str] = None) -> DrawdownControlConfig:
        """重新加载配置"""
        self.logger.info("重新加载配置")
        
        if self.auto_backup and self.current_config:
            self._backup_current_config()
        
        return self.load_config(config_file)
    
    def _reload_config_file(self, file_path: str):
        """重新加载指定配置文件"""
        try:
            self.logger.info(f"热重载配置文件: {file_path}")
            self._load_from_file(file_path)
        except Exception as e:
            self.logger.error(f"热重载配置文件失败 {file_path}: {e}")
    
    def get_current_config(self) -> Optional[DrawdownControlConfig]:
        """获取当前配置"""
        with self.config_lock:
            return self.current_config
    
    def update_config_parameter(self, parameter_path: str, value: Any) -> bool:
        """
        更新配置参数
        
        Args:
            parameter_path: 参数路径，如 'max_drawdown_threshold'
            value: 新值
            
        Returns:
            是否更新成功
        """
        if not self.current_config:
            self.logger.error("当前没有加载的配置")
            return False
        
        try:
            # 创建配置副本
            config_dict = self.current_config.to_dict()
            
            # 更新参数
            if parameter_path in config_dict:
                old_value = config_dict[parameter_path]
                config_dict[parameter_path] = value
                
                # 验证更新后的配置
                validation_result = self.validator.validate_drawdown_control_config(config_dict)
                if not validation_result['valid']:
                    self.logger.error(f"参数更新验证失败: {validation_result['errors']}")
                    return False
                
                # 创建新配置对象
                new_config = DrawdownControlConfig.from_dict(config_dict)
                
                # 更新当前配置
                with self.config_lock:
                    self.current_config = new_config
                
                self.logger.info(f"参数更新成功: {parameter_path} = {old_value} -> {value}")
                
                # 触发配置更新回调
                self._trigger_config_callbacks(new_config)
                
                return True
            else:
                self.logger.error(f"未知的配置参数: {parameter_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"更新配置参数失败 {parameter_path}: {e}")
            return False
    
    def register_config_callback(self, callback: Callable[[DrawdownControlConfig], None]):
        """注册配置更新回调"""
        self.config_callbacks.append(callback)
        self.logger.info(f"注册配置更新回调: {callback.__name__}")
    
    def _trigger_config_callbacks(self, config: DrawdownControlConfig):
        """触发配置更新回调"""
        for callback in self.config_callbacks:
            try:
                callback(config)
            except Exception as e:
                self.logger.error(f"配置回调执行失败 {callback.__name__}: {e}")
    
    def _record_config_version(self, config: DrawdownControlConfig, source: str):
        """记录配置版本"""
        import hashlib
        
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        checksum = hashlib.md5(config_str.encode()).hexdigest()
        
        version = ConfigVersion(
            version=f"v{len(self.config_history) + 1}",
            timestamp=datetime.now(),
            source_file=source,
            checksum=checksum
        )
        
        self.config_history.append(version)
        
        # 保持历史记录数量限制
        if len(self.config_history) > 100:
            self.config_history = self.config_history[-100:]
    
    def _add_file_watch(self, file_path: str):
        """添加文件监控"""
        if file_path in self.watched_files:
            return
        
        self.watched_files.append(file_path)
        
        if self.file_observer is None:
            self.file_observer = Observer()
            event_handler = ConfigFileHandler(self)
            
            # 监控配置文件目录
            watch_dir = Path(file_path).parent
            self.file_observer.schedule(event_handler, str(watch_dir), recursive=False)
            self.file_observer.start()
            
            self.logger.info(f"启动配置文件监控: {watch_dir}")
    
    def _backup_current_config(self):
        """备份当前配置"""
        if not self.current_config:
            return
        
        backup_dir = self.config_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"drawdown_control_backup_{timestamp}.yaml"
        
        try:
            self.save_config(self.current_config, str(backup_file), 'yaml')
            self.logger.info(f"配置已备份到: {backup_file}")
        except Exception as e:
            self.logger.error(f"配置备份失败: {e}")
    
    def get_config_history(self) -> List[Dict[str, Any]]:
        """获取配置历史"""
        return [version.to_dict() for version in self.config_history]
    
    def validate_current_config(self) -> Dict[str, Any]:
        """验证当前配置"""
        if not self.current_config:
            return {'valid': False, 'errors': ['当前没有加载的配置']}
        
        return self.validator.validate_drawdown_control_config(self.current_config.to_dict())
    
    def stop(self):
        """停止配置管理器"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.file_observer = None
        
        self.watched_files.clear()
        self.config_callbacks.clear()
        
        self.logger.info("配置管理器已停止")