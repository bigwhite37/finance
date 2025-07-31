"""
系统集成模块

集成所有组件，实现完整的交易决策流程和数据流管道
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from .config import ConfigManager
from .data import (
    QlibDataInterface, AkshareDataInterface, FeatureEngineer, 
    DataProcessor, MarketData, TradingState, TradingAction
)
from .models import TimeSeriesTransformer, SACAgent, TransformerConfig, SACConfig
from .trading import PortfolioEnvironment, PortfolioConfig, TransactionCostModel
from .training import RLTrainer, TrainingConfig
from .evaluation import PortfolioMetrics
from .monitoring import TradingSystemMonitor
from .audit import AuditLogger
from .deployment import ModelVersionManager
from .risk_control import RiskController


class SystemState(Enum):
    """系统状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SystemConfig:
    """系统配置"""
    # 数据配置
    data_source: str = "qlib"  # qlib, akshare, or both
    stock_pool: List[str] = None
    lookback_window: int = 60
    update_frequency: str = "1D"  # 1D, 1H, 5min
    
    # 模型配置
    model_path: Optional[str] = None
    transformer_config: Optional[Dict] = None
    sac_config: Optional[Dict] = None
    
    # 交易配置
    initial_cash: float = 1000000.0
    max_position_size: float = 0.1
    commission_rate: float = 0.001
    stamp_tax_rate: float = 0.001
    
    # 系统配置
    enable_monitoring: bool = True
    enable_audit: bool = True
    enable_risk_control: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.stock_pool is None:
            self.stock_pool = ['000001.SZ', '000002.SZ', '600000.SH']


class TradingSystem:
    """完整交易系统"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.state = SystemState.STOPPED
        self.logger = self._setup_logging()
        
        # 系统组件
        self.config_manager: Optional[ConfigManager] = None
        self.data_interface = None
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.data_processor: Optional[DataProcessor] = None
        self.transformer: Optional[TimeSeriesTransformer] = None
        self.sac_agent: Optional[SACAgent] = None
        self.portfolio_env: Optional[PortfolioEnvironment] = None
        self.cost_model: Optional[TransactionCostModel] = None
        self.performance_metrics: Optional[PortfolioMetrics] = None
        self.monitor: Optional[TradingSystemMonitor] = None
        self.audit_logger: Optional[AuditLogger] = None
        self.risk_controller: Optional[RiskController] = None
        self.version_manager: Optional[ModelVersionManager] = None
        
        # 运行时状态
        self.current_positions = np.zeros(len(config.stock_pool))
        self.portfolio_value = config.initial_cash
        self.last_update_time: Optional[datetime] = None
        self.trading_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 性能统计
        self.stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'start_time': None,
            'last_trade_time': None
        }
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger(f"TradingSystem")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize(self) -> bool:
        """初始化系统组件"""
        try:
            self.state = SystemState.STARTING
            self.logger.info("开始初始化交易系统...")
            
            # 1. 配置管理器
            self.config_manager = ConfigManager()
            self.logger.info("✓ 配置管理器初始化完成")
            
            # 2. 数据接口
            self._initialize_data_components()
            self.logger.info("✓ 数据组件初始化完成")
            
            # 3. 模型组件
            self._initialize_model_components()
            self.logger.info("✓ 模型组件初始化完成")
            
            # 4. 交易组件
            self._initialize_trading_components()
            self.logger.info("✓ 交易组件初始化完成")
            
            # 5. 监控和审计组件
            if self.config.enable_monitoring:
                self._initialize_monitoring_components()
                self.logger.info("✓ 监控组件初始化完成")
            
            if self.config.enable_audit:
                self._initialize_audit_components()
                self.logger.info("✓ 审计组件初始化完成")
            
            # 6. 风险控制组件
            if self.config.enable_risk_control:
                self._initialize_risk_components()
                self.logger.info("✓ 风险控制组件初始化完成")
            
            # 7. 版本管理组件
            self._initialize_deployment_components()
            self.logger.info("✓ 部署组件初始化完成")
            
            self.state = SystemState.STOPPED
            self.logger.info("交易系统初始化完成")
            return True
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    def _initialize_data_components(self):
        """初始化数据组件"""
        # 数据接口
        if self.config.data_source == "qlib":
            self.data_interface = QlibDataInterface()
        elif self.config.data_source == "akshare":
            self.data_interface = AkshareDataInterface()
        else:
            # 使用Qlib作为主要数据源，Akshare作为备用
            self.data_interface = QlibDataInterface()
        
        # 特征工程
        self.feature_engineer = FeatureEngineer()
        
        # 数据处理器
        self.data_processor = DataProcessor(self.feature_engineer)
    
    def _initialize_model_components(self):
        """初始化模型组件"""
        # Transformer配置
        transformer_config = TransformerConfig(
            **(self.config.transformer_config or {})
        )
        self.transformer = TimeSeriesTransformer(transformer_config)
        
        # SAC配置
        sac_config = SACConfig(
            state_dim=transformer_config.d_model * len(self.config.stock_pool),
            action_dim=len(self.config.stock_pool),
            **(self.config.sac_config or {})
        )
        self.sac_agent = SACAgent(sac_config)
        
        # 加载预训练模型
        if self.config.model_path:
            self.load_model(self.config.model_path)
    
    def _initialize_trading_components(self):
        """初始化交易组件"""
        # 投资组合环境配置
        portfolio_config = PortfolioConfig(
            stock_pool=self.config.stock_pool,
            initial_cash=self.config.initial_cash,
            lookback_window=self.config.lookback_window,
            commission_rate=self.config.commission_rate,
            stamp_tax_rate=self.config.stamp_tax_rate
        )
        
        # 投资组合环境
        self.portfolio_env = PortfolioEnvironment(portfolio_config, self.data_interface)
        
        # 交易成本模型
        from .trading.transaction_cost_model import CostParameters
        cost_params = CostParameters(
            commission_rate=self.config.commission_rate,
            stamp_tax_rate=self.config.stamp_tax_rate,
            min_commission=5.0,
            transfer_fee_rate=0.00002
        )
        self.cost_model = TransactionCostModel(cost_params)
        
        # 性能评估 (will be initialized when needed)
        self.performance_metrics = None
    
    def _initialize_monitoring_components(self):
        """初始化监控组件"""
        self.monitor = TradingSystemMonitor()
    
    def _initialize_audit_components(self):
        """初始化审计组件"""
        audit_config = {
            'influxdb': {
                'url': 'http://localhost:8086',
                'token': 'trading_token',
                'org': 'trading',
                'bucket': 'audit'
            },
            'postgresql': {
                'host': 'localhost',
                'port': 5432,
                'database': 'trading_db',
                'user': 'trading_user',
                'password': 'trading_pass'
            }
        }
        self.audit_logger = AuditLogger(audit_config)
    
    def _initialize_risk_components(self):
        """初始化风险控制组件"""
        risk_config = {
            'max_position_size': self.config.max_position_size,
            'max_drawdown_limit': 0.15,
            'stop_loss_threshold': 0.05,
            'position_concentration_limit': 0.3,
            'var_limit': 0.03,
            'tracking_error_limit': 0.05
        }
        self.risk_controller = RiskController(risk_config)
    
    def _initialize_deployment_components(self):
        """初始化部署组件"""
        self.version_manager = ModelVersionManager()
    
    def start(self) -> bool:
        """启动交易系统"""
        if self.state != SystemState.STOPPED:
            self.logger.warning(f"系统当前状态为 {self.state.value}，无法启动")
            return False
        
        try:
            self.state = SystemState.STARTING
            self.logger.info("启动交易系统...")
            
            # 重置停止事件
            self.stop_event.clear()
            
            # 初始化统计信息
            self.stats['start_time'] = datetime.now()
            
            # 启动交易线程
            self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.trading_thread.start()
            
            self.state = SystemState.RUNNING
            self.logger.info("交易系统启动成功")
            
            # 记录审计日志
            if self.audit_logger:
                try:
                    # 简化审计日志记录（避免异步调用）
                    pass  # 暂时跳过审计日志
                except Exception as e:
                    self.logger.warning(f"审计日志记录失败: {e}")
            
            return True
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"系统启动失败: {e}")
            return False
    
    def stop(self) -> bool:
        """停止交易系统"""
        if self.state not in [SystemState.RUNNING, SystemState.PAUSED]:
            self.logger.warning(f"系统当前状态为 {self.state.value}，无法停止")
            return False
        
        try:
            self.state = SystemState.STOPPING
            self.logger.info("停止交易系统...")
            
            # 设置停止事件
            self.stop_event.set()
            
            # 等待交易线程结束
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=30)
                if self.trading_thread.is_alive():
                    self.logger.warning("交易线程未能在30秒内正常结束")
            
            self.state = SystemState.STOPPED
            self.logger.info("交易系统已停止")
            
            # 记录审计日志
            if self.audit_logger:
                try:
                    # 简化审计日志记录（避免异步调用）
                    pass  # 暂时跳过审计日志
                except Exception as e:
                    self.logger.warning(f"审计日志记录失败: {e}")
            
            return True
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"系统停止失败: {e}")
            return False
    
    def pause(self) -> bool:
        """暂停交易系统"""
        if self.state != SystemState.RUNNING:
            self.logger.warning(f"系统当前状态为 {self.state.value}，无法暂停")
            return False
        
        self.state = SystemState.PAUSED
        self.logger.info("交易系统已暂停")
        return True
    
    def resume(self) -> bool:
        """恢复交易系统"""
        if self.state != SystemState.PAUSED:
            self.logger.warning(f"系统当前状态为 {self.state.value}，无法恢复")
            return False
        
        self.state = SystemState.RUNNING
        self.logger.info("交易系统已恢复")
        return True
    
    def _trading_loop(self):
        """主交易循环"""
        self.logger.info("交易循环开始")
        
        while not self.stop_event.is_set():
            try:
                if self.state == SystemState.RUNNING:
                    # 执行一次交易决策
                    self._execute_trading_cycle()
                
                # 等待下一个周期
                self._wait_for_next_cycle()
                
            except Exception as e:
                self.logger.error(f"交易循环异常: {e}")
                if self.monitor:
                    self.monitor.log_metric('system_error', 1, datetime.now())
                
                # 短暂休眠后继续
                time.sleep(5)
        
        self.logger.info("交易循环结束")
    
    def _execute_trading_cycle(self):
        """执行一次完整的交易周期"""
        try:
            # 1. 获取市场数据
            market_data = self._get_market_data()
            if market_data is None or market_data.empty:
                self.logger.warning("未获取到市场数据，跳过本次交易")
                return
            
            # 2. 特征工程和数据预处理
            processed_data = self._process_market_data(market_data)
            if processed_data is None:
                self.logger.warning("数据处理失败，跳过本次交易")
                return
            
            # 3. 构建交易状态
            trading_state = self._build_trading_state(processed_data)
            if trading_state is None:
                self.logger.warning("交易状态构建失败，跳过本次交易")
                return
            
            # 4. 模型推理
            trading_action = self._generate_trading_action(trading_state)
            if trading_action is None:
                self.logger.warning("交易动作生成失败，跳过本次交易")
                return
            
            # 5. 风险控制
            if self.risk_controller:
                trading_action = self.risk_controller.validate_action(
                    trading_action, trading_state
                )
                if trading_action is None:
                    self.logger.warning("交易动作被风险控制拒绝")
                    return
            
            # 6. 执行交易
            execution_result = self._execute_trade(trading_action)
            
            # 7. 更新系统状态
            self._update_system_state(execution_result)
            
            # 8. 记录和监控
            self._log_trading_cycle(trading_action, execution_result)
            
            self.last_update_time = datetime.now()
            
        except Exception as e:
            self.logger.error(f"交易周期执行异常: {e}")
            raise
    
    def _get_market_data(self) -> Optional[pd.DataFrame]:
        """获取市场数据"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=self.config.lookback_window + 10)).strftime('%Y-%m-%d')
            
            data = self.data_interface.get_price_data(
                symbols=self.config.stock_pool,
                start_date=start_date,
                end_date=end_date
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            return None
    
    def _process_market_data(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """处理市场数据"""
        try:
            # 特征工程
            features = self.feature_engineer.calculate_technical_indicators(market_data)
            
            # 数据预处理
            processed_data = self.data_processor.process_data(features)
            
            # 转换为模型输入格式
            # [lookback_window, n_stocks, n_features]
            feature_array = self._convert_to_model_input(processed_data)
            
            return feature_array
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {e}")
            return None
    
    def _convert_to_model_input(self, processed_data: pd.DataFrame) -> np.ndarray:
        """转换数据为模型输入格式"""
        # 这里需要根据实际的数据格式进行转换
        # 假设processed_data已经是按时间和股票组织的特征数据
        
        # 获取最近的lookback_window天数据
        recent_data = processed_data.tail(self.config.lookback_window)
        
        # 转换为 [lookback_window, n_stocks, n_features] 格式
        n_stocks = len(self.config.stock_pool)
        n_features = len([col for col in recent_data.columns if col not in ['datetime', 'instrument']])
        
        feature_array = np.zeros((self.config.lookback_window, n_stocks, n_features))
        
        # 填充数据（这里需要根据实际数据结构调整）
        for i, stock in enumerate(self.config.stock_pool):
            stock_data = recent_data[recent_data.get('instrument', recent_data.index) == stock]
            if not stock_data.empty:
                feature_cols = [col for col in stock_data.columns if col not in ['datetime', 'instrument']]
                feature_array[:len(stock_data), i, :] = stock_data[feature_cols].values
        
        return feature_array
    
    def _build_trading_state(self, processed_data: np.ndarray) -> Optional[TradingState]:
        """构建交易状态"""
        try:
            # 市场状态特征（简化版本）
            market_state = np.array([
                self.portfolio_value / self.config.initial_cash,  # 相对收益
                np.sum(self.current_positions > 0.01),  # 持仓股票数
                np.max(self.current_positions),  # 最大持仓比例
                np.std(self.current_positions),  # 持仓分散度
                time.time() % (24 * 3600) / (24 * 3600),  # 时间特征
                0.0, 0.0, 0.0, 0.0, 0.0  # 预留特征
            ])
            
            trading_state = TradingState(
                features=processed_data,
                positions=self.current_positions.copy(),
                market_state=market_state,
                cash=self.config.initial_cash * (1 - np.sum(self.current_positions)),
                total_value=self.portfolio_value
            )
            
            return trading_state
            
        except Exception as e:
            self.logger.error(f"交易状态构建失败: {e}")
            return None
    
    def _generate_trading_action(self, trading_state: TradingState) -> Optional[TradingAction]:
        """生成交易动作"""
        try:
            # 准备模型输入
            features_tensor = torch.FloatTensor(trading_state.features).unsqueeze(0)
            positions_tensor = torch.FloatTensor(trading_state.positions).unsqueeze(0)
            market_state_tensor = torch.FloatTensor(trading_state.market_state).unsqueeze(0)
            
            state = {
                'features': features_tensor,
                'positions': positions_tensor,
                'market_state': market_state_tensor
            }
            
            # 模型推理
            with torch.no_grad():
                action_tensor, log_prob = self.sac_agent.get_action(state, deterministic=True)
                target_weights = action_tensor.squeeze(0).numpy()
                confidence = torch.exp(log_prob).item()
            
            trading_action = TradingAction(
                target_weights=target_weights,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            return trading_action
            
        except Exception as e:
            self.logger.error(f"交易动作生成失败: {e}")
            return None
    
    def _execute_trade(self, trading_action: TradingAction) -> Dict[str, Any]:
        """执行交易"""
        try:
            # 计算交易成本
            cost_breakdown = self.cost_model.calculate_total_cost(
                current_positions=self.current_positions,
                target_positions=trading_action.target_weights,
                prices=np.ones(len(self.config.stock_pool)) * 10.0,  # 简化价格
                volumes=np.ones(len(self.config.stock_pool)) * 1000000  # 简化成交量
            )
            
            # 更新持仓
            old_positions = self.current_positions.copy()
            self.current_positions = trading_action.target_weights.copy()
            
            # 更新统计
            self.stats['total_trades'] += 1
            self.stats['last_trade_time'] = datetime.now()
            
            # 模拟收益计算（实际应该基于真实价格变化）
            portfolio_return = np.random.normal(0.001, 0.02)  # 模拟日收益
            self.portfolio_value *= (1 + portfolio_return)
            
            execution_result = {
                'success': True,
                'old_positions': old_positions,
                'new_positions': self.current_positions,
                'transaction_cost': cost_breakdown.total_cost,
                'portfolio_return': portfolio_return,
                'portfolio_value': self.portfolio_value,
                'timestamp': datetime.now()
            }
            
            self.stats['successful_trades'] += 1
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"交易执行失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _update_system_state(self, execution_result: Dict[str, Any]):
        """更新系统状态"""
        if execution_result['success']:
            # 更新收益统计
            total_return = (self.portfolio_value - self.config.initial_cash) / self.config.initial_cash
            self.stats['total_return'] = total_return
            
            # 更新最大回撤（简化计算）
            if hasattr(self, 'peak_value'):
                self.peak_value = max(self.peak_value, self.portfolio_value)
                drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
                self.stats['max_drawdown'] = max(self.stats['max_drawdown'], drawdown)
            else:
                self.peak_value = self.portfolio_value
    
    def _log_trading_cycle(self, trading_action: TradingAction, execution_result: Dict[str, Any]):
        """记录交易周期"""
        # 监控指标
        if self.monitor:
            self.monitor.log_metric('portfolio_value', self.portfolio_value, datetime.now())
            self.monitor.log_metric('total_return', self.stats['total_return'], datetime.now())
            if execution_result['success']:
                self.monitor.log_metric('transaction_cost', execution_result['transaction_cost'], datetime.now())
        
        # 审计日志
        if self.audit_logger:
            try:
                # 简化审计日志记录（避免异步调用）
                pass  # 暂时跳过审计日志
            except Exception as e:
                self.logger.warning(f"审计日志记录失败: {e}")
    
    def _wait_for_next_cycle(self):
        """等待下一个交易周期"""
        if self.config.update_frequency == "1D":
            sleep_time = 24 * 3600  # 1天
        elif self.config.update_frequency == "1H":
            sleep_time = 3600  # 1小时
        elif self.config.update_frequency == "5min":
            sleep_time = 300  # 5分钟
        else:
            sleep_time = 60  # 默认1分钟
        
        # 分段休眠，以便及时响应停止信号
        for _ in range(int(sleep_time)):
            if self.stop_event.is_set():
                break
            time.sleep(1)
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'state': self.state.value,
            'current_positions': self.current_positions.tolist(),
            'portfolio_value': self.portfolio_value,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'stats': self.stats.copy(),
            'config': {
                'stock_pool': self.config.stock_pool,
                'initial_cash': self.config.initial_cash,
                'update_frequency': self.config.update_frequency
            }
        }
    
    def load_model(self, model_path: str) -> bool:
        """加载模型"""
        try:
            model_dir = Path(model_path)
            
            # 加载Transformer
            transformer_path = model_dir / "transformer.pth"
            if transformer_path.exists():
                self.transformer.load_state_dict(torch.load(transformer_path))
                self.logger.info(f"Transformer模型加载成功: {transformer_path}")
            
            # 加载SAC Agent
            sac_path = model_dir / "sac_agent.pth"
            if sac_path.exists():
                self.sac_agent.load_state_dict(torch.load(sac_path))
                self.logger.info(f"SAC Agent模型加载成功: {sac_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """保存模型"""
        try:
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存Transformer
            transformer_path = model_dir / "transformer.pth"
            torch.save(self.transformer.state_dict(), transformer_path)
            
            # 保存SAC Agent
            sac_path = model_dir / "sac_agent.pth"
            torch.save(self.sac_agent.state_dict(), sac_path)
            
            # 保存配置
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                import json
                json.dump(self.config.__dict__, f, indent=2, default=str)
            
            self.logger.info(f"模型保存成功: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"模型保存失败: {e}")
            return False


class SystemManager:
    """系统管理器"""
    
    def __init__(self):
        self.systems: Dict[str, TradingSystem] = {}
        self.logger = logging.getLogger("SystemManager")
    
    def create_system(self, name: str, config: SystemConfig) -> bool:
        """创建交易系统"""
        if name in self.systems:
            self.logger.warning(f"系统 {name} 已存在")
            return False
        
        try:
            system = TradingSystem(config)
            if system.initialize():
                self.systems[name] = system
                self.logger.info(f"系统 {name} 创建成功")
                return True
            else:
                self.logger.error(f"系统 {name} 初始化失败")
                return False
                
        except Exception as e:
            self.logger.error(f"创建系统 {name} 失败: {e}")
            return False
    
    def start_system(self, name: str) -> bool:
        """启动交易系统"""
        if name not in self.systems:
            self.logger.error(f"系统 {name} 不存在")
            return False
        
        return self.systems[name].start()
    
    def stop_system(self, name: str) -> bool:
        """停止交易系统"""
        if name not in self.systems:
            self.logger.error(f"系统 {name} 不存在")
            return False
        
        return self.systems[name].stop()
    
    def get_system_status(self, name: str) -> Optional[Dict[str, Any]]:
        """获取系统状态"""
        if name not in self.systems:
            return None
        
        return self.systems[name].get_status()
    
    def list_systems(self) -> List[str]:
        """列出所有系统"""
        return list(self.systems.keys())
    
    def remove_system(self, name: str) -> bool:
        """移除系统"""
        if name not in self.systems:
            self.logger.error(f"系统 {name} 不存在")
            return False
        
        # 先停止系统
        if self.systems[name].state == SystemState.RUNNING:
            self.systems[name].stop()
        
        # 移除系统
        del self.systems[name]
        self.logger.info(f"系统 {name} 已移除")
        return True


# 全局系统管理器实例
system_manager = SystemManager()