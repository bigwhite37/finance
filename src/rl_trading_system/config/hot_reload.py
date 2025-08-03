"""
配置热重载器

实现配置文件的实时监控和热重载功能，支持配置变更的自动检测和应用。
"""

import yaml
import json
import os
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from .config_validator import ConfigValidator

logger = logging.getLogger(__name__)


@dataclass
class ConfigChangeEvent:
    """配置变更事件"""
    file_path: str
    change_type: str  # 'modified', 'created', 'deleted'
    timestamp: datetime
    old_config: Optional[Dict[str, Any]] = None
    new_config: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'file_path': self.file_path,
            'change_type': self.change_type,
            'timestamp': self.timestamp.isoformat(),
            'old_config': self.old_config,
            'new_config': self.new_config,
            'validation_result': self.validation_result
        }


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监控处理器"""
    
    def __init__(self, hot_reloader: 'ConfigHotReloader'):
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(__name__ + '.ConfigFileWatcher')
        self._last_modified = {}
        self._debounce_delay = 1.0  # 防抖延迟（秒）
    
    def on_modified(self, event):
        """文件修改事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # 只监控目标配置文件
        if not self.hot_reloader.is_watched_file(file_path):
            return
        
        # 防抖处理，避免重复触发
        current_time = time.time()
        last_time = self._last_modified.get(file_path, 0)
        
        if current_time - last_time < self._debounce_delay:
            return
        
        self._last_modified[file_path] = current_time
        
        self.logger.info(f"检测到配置文件变更: {file_path}")
        
        # 延迟处理，确保文件写入完成
        threading.Timer(
            self._debounce_delay,
            self.hot_reloader._handle_file_change,
            args=[file_path, 'modified']
        ).start()
    
    def on_created(self, event):
        """文件创建事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        if self.hot_reloader.is_watched_file(file_path):
            self.logger.info(f"检测到配置文件创建: {file_path}")
            self.hot_reloader._handle_file_change(file_path, 'created')
    
    def on_deleted(self, event):
        """文件删除事件处理"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        if self.hot_reloader.is_watched_file(file_path):
            self.logger.info(f"检测到配置文件删除: {file_path}")
            self.hot_reloader._handle_file_change(file_path, 'deleted')


class ConfigHotReloader:
    """
    配置热重载器
    
    提供配置文件的实时监控和热重载功能：
    - 文件系统监控
    - 配置变更检测
    - 自动验证和应用
    - 变更回调通知
    - 回滚机制
    """
    
    def __init__(self,
                 watched_files: Optional[List[str]] = None,
                 auto_apply: bool = True,
                 backup_enabled: bool = True,
                 max_backup_count: int = 10):
        """
        初始化热重载器
        
        Args:
            watched_files: 监控的文件列表
            auto_apply: 是否自动应用配置变更
            backup_enabled: 是否启用备份
            max_backup_count: 最大备份数量
        """
        self.watched_files: List[str] = watched_files or []
        self.auto_apply = auto_apply
        self.backup_enabled = backup_enabled
        self.max_backup_count = max_backup_count
        self.logger = logging.getLogger(__name__)
        
        # 文件监控
        self.observer: Optional[Observer] = None
        self.file_watcher = ConfigFileWatcher(self)
        self.watched_directories: set = set()
        
        # 配置状态
        self.current_configs: Dict[str, Dict[str, Any]] = {}
        self.config_checksums: Dict[str, str] = {}
        self.change_callbacks: List[Callable[[ConfigChangeEvent], None]] = []
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 验证器
        self.validator = ConfigValidator()
        
        # 变更历史
        self.change_history: List[ConfigChangeEvent] = []
        
        self.logger.info("配置热重载器初始化完成")
    
    def add_watched_file(self, file_path: str):
        """添加监控文件"""
        file_path = str(Path(file_path).resolve())
        
        if file_path not in self.watched_files:
            self.watched_files.append(file_path)
            
            # 先加载初始配置，再设置监控
            self._load_initial_config(file_path)
            self._setup_file_watch(file_path)
            
            self.logger.info(f"添加监控文件: {file_path}")
    
    def remove_watched_file(self, file_path: str):
        """移除监控文件"""
        file_path = str(Path(file_path).resolve())
        
        if file_path in self.watched_files:
            self.watched_files.remove(file_path)
            
            with self.lock:
                self.current_configs.pop(file_path, None)
                self.config_checksums.pop(file_path, None)
            
            self.logger.info(f"移除监控文件: {file_path}")
    
    def is_watched_file(self, file_path: str) -> bool:
        """检查是否为监控文件"""
        file_path = str(Path(file_path).resolve())
        return file_path in self.watched_files
    
    def start_watching(self):
        """开始监控"""
        if self.observer is not None:
            self.logger.warning("文件监控已启动")
            return
        
        if not self.watched_files:
            self.logger.warning("没有监控文件，跳过监控启动")
            return
        
        self.observer = Observer()
        
        # 为每个监控目录添加监控
        for file_path in self.watched_files:
            self._setup_file_watch(file_path)
        
        self.observer.start()
        self.logger.info("配置文件监控已启动")
    
    def stop_watching(self):
        """停止监控"""
        if self.observer is None:
            return
        
        self.observer.stop()
        self.observer.join()
        self.observer = None
        self.watched_directories.clear()
        
        self.logger.info("配置文件监控已停止")
    
    def _setup_file_watch(self, file_path: str):
        """设置文件监控"""
        if self.observer is None:
            return
        
        file_path = Path(file_path)
        watch_dir = file_path.parent
        
        if str(watch_dir) not in self.watched_directories:
            self.observer.schedule(self.file_watcher, str(watch_dir), recursive=False)
            self.watched_directories.add(str(watch_dir))
    
    def _load_initial_config(self, file_path: str):
        """加载初始配置"""
        try:
            config_data = self._load_config_file(file_path)
            if config_data:
                # 验证配置（不强制要求通过验证）
                validation_result = self._validate_config(config_data, file_path)
                if not validation_result['valid']:
                    self.logger.warning(f"配置验证失败但仍加载: {validation_result['errors']}")
                
                checksum = self._calculate_checksum(config_data)
                
                with self.lock:
                    self.current_configs[file_path] = config_data
                    self.config_checksums[file_path] = checksum
                
                self.logger.info(f"加载初始配置: {file_path}")
        except Exception as e:
            self.logger.error(f"加载初始配置失败 {file_path}: {e}")
    
    def _load_config_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """加载配置文件"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    self.logger.warning(f"不支持的配置文件格式: {file_path.suffix}")
                    return None
        except Exception as e:
            self.logger.error(f"读取配置文件失败 {file_path}: {e}")
            return None
    
    def _calculate_checksum(self, config_data: Dict[str, Any]) -> str:
        """计算配置校验和"""
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _handle_file_change(self, file_path: str, change_type: str):
        """处理文件变更"""
        try:
            old_config = self.current_configs.get(file_path)
            old_checksum = self.config_checksums.get(file_path)
            
            if change_type == 'deleted':
                new_config = None
                validation_result = {'valid': True, 'errors': [], 'warnings': []}
            else:
                new_config = self._load_config_file(file_path)
                
                if new_config is None:
                    self.logger.error(f"无法加载配置文件: {file_path}")
                    return
                
                new_checksum = self._calculate_checksum(new_config)
                
                # 检查是否真正发生变更
                if old_checksum == new_checksum:
                    self.logger.debug(f"配置文件内容未变更: {file_path}")
                    return
                
                # 验证新配置
                validation_result = self._validate_config(new_config, file_path)
                
                if not validation_result['valid'] and not self.auto_apply:
                    self.logger.error(f"配置验证失败: {validation_result['errors']}")
                    return
            
            # 创建变更事件
            change_event = ConfigChangeEvent(
                file_path=file_path,
                change_type=change_type,
                timestamp=datetime.now(),
                old_config=old_config,
                new_config=new_config,
                validation_result=validation_result
            )
            
            # 记录变更历史
            self._record_change_event(change_event)
            
            # 应用配置变更
            if self.auto_apply and validation_result['valid']:
                self._apply_config_change(file_path, new_config)
            
            # 触发回调
            self._trigger_change_callbacks(change_event)
            
            self.logger.info(f"处理配置变更完成: {file_path} ({change_type})")
            
        except Exception as e:
            self.logger.error(f"处理配置变更失败 {file_path}: {e}")
    
    def _validate_config(self, config_data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """验证配置"""
        try:
            # 根据文件路径判断配置类型
            if 'drawdown_control' in file_path.lower() or 'drawdown_control' in config_data:
                drawdown_config = config_data.get('drawdown_control', config_data)
                return self.validator.validate_drawdown_control_config(drawdown_config)
            else:
                # 默认验证
                return {'valid': True, 'errors': [], 'warnings': []}
        except Exception as e:
            return {'valid': False, 'errors': [str(e)], 'warnings': []}
    
    def _apply_config_change(self, file_path: str, new_config: Optional[Dict[str, Any]]):
        """应用配置变更"""
        if self.backup_enabled and file_path in self.current_configs:
            self._backup_config(file_path, self.current_configs[file_path])
        
        with self.lock:
            if new_config is None:
                self.current_configs.pop(file_path, None)
                self.config_checksums.pop(file_path, None)
            else:
                self.current_configs[file_path] = new_config
                self.config_checksums[file_path] = self._calculate_checksum(new_config)
    
    def _backup_config(self, file_path: str, config_data: Dict[str, Any]):
        """备份配置"""
        try:
            backup_dir = Path(file_path).parent / 'backups'
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = Path(file_path).stem
            backup_file = backup_dir / f"{file_name}_backup_{timestamp}.yaml"
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            # 清理过期备份
            self._cleanup_old_backups(backup_dir, file_name)
            
            self.logger.info(f"配置已备份: {backup_file}")
            
        except Exception as e:
            self.logger.error(f"配置备份失败: {e}")
    
    def _cleanup_old_backups(self, backup_dir: Path, file_prefix: str):
        """清理过期备份"""
        try:
            backup_files = list(backup_dir.glob(f"{file_prefix}_backup_*.yaml"))
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # 保留最新的备份文件
            for old_backup in backup_files[self.max_backup_count:]:
                old_backup.unlink()
                self.logger.debug(f"删除过期备份: {old_backup}")
                
        except Exception as e:
            self.logger.error(f"清理备份文件失败: {e}")
    
    def _record_change_event(self, event: ConfigChangeEvent):
        """记录变更事件"""
        self.change_history.append(event)
        
        # 限制历史记录数量
        if len(self.change_history) > 1000:
            self.change_history = self.change_history[-1000:]
    
    def _trigger_change_callbacks(self, event: ConfigChangeEvent):
        """触发变更回调"""
        for callback in self.change_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"配置变更回调执行失败: {e}")
    
    def register_change_callback(self, callback: Callable[[ConfigChangeEvent], None]):
        """注册变更回调"""
        self.change_callbacks.append(callback)
        self.logger.info(f"注册配置变更回调: {callback.__name__}")
    
    def get_current_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """获取当前配置"""
        with self.lock:
            return self.current_configs.get(file_path)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有当前配置"""
        with self.lock:
            return self.current_configs.copy()
    
    def get_change_history(self, file_path: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取变更历史"""
        history = self.change_history
        
        if file_path:
            history = [event for event in history if event.file_path == file_path]
        
        # 返回最近的记录
        recent_history = history[-limit:] if len(history) > limit else history
        
        return [event.to_dict() for event in recent_history]
    
    def force_reload(self, file_path: Optional[str] = None):
        """强制重载配置"""
        if file_path:
            if file_path in self.watched_files:
                self._handle_file_change(file_path, 'modified')
        else:
            for watched_file in self.watched_files:
                self._handle_file_change(watched_file, 'modified')
        
        self.logger.info(f"强制重载配置: {file_path or '所有文件'}")
    
    def rollback_config(self, file_path: str, target_timestamp: Optional[datetime] = None) -> bool:
        """
        回滚配置到指定时间点
        
        Args:
            file_path: 文件路径
            target_timestamp: 目标时间戳，None表示回滚到上一个版本
            
        Returns:
            是否回滚成功
        """
        try:
            # 查找目标配置
            target_event = None
            
            file_history = [event for event in self.change_history if event.file_path == file_path]
            file_history.sort(key=lambda e: e.timestamp, reverse=True)
            
            if target_timestamp is None:
                # 回滚到上一个版本
                if len(file_history) >= 2:
                    target_event = file_history[1]  # 跳过最新版本
            else:
                # 查找指定时间点的版本
                for event in file_history:
                    if event.timestamp <= target_timestamp:
                        target_event = event
                        break
            
            if target_event is None or target_event.old_config is None:
                self.logger.error(f"未找到可回滚的配置版本: {file_path}")
                return False
            
            # 写入回滚配置
            file_path_obj = Path(file_path)
            with open(file_path_obj, 'w', encoding='utf-8') as f:
                if file_path_obj.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(target_event.old_config, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(target_event.old_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"配置回滚成功: {file_path} -> {target_event.timestamp}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置回滚失败 {file_path}: {e}")
            return False
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_watching()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_watching()