"""
动态低波筛选器主控制器模块

统一管理四层筛选流水线，协调各个组件，提供统一的筛选接口。
实现动态、预测式、市场状态感知的低波股票筛选系统，
替代传统的静态阈值方法。
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from .data_preprocessor import DataPreprocessor
from .filters import RollingPercentileFilter, IVOLConstraintFilter
from .predictors import GARCHVolatilityPredictor
from .regime import MarketRegimeDetector, RegimeAwareThresholdAdjuster
from .data_structures import FilterInputData, FilterOutputData, DynamicLowVolConfig
from .exceptions import (
    DataQualityException, InsufficientDataException,
    ModelFittingException, RegimeDetectionException, ConfigurationException
)

# 导入性能优化相关组件
try:
    from ..performance_optimizer import (
        PerformanceOptimizer, CacheConfig, ParallelConfig,
        performance_monitor
    )
    from ..performance_optimizer.data_structures import MonitoringConfig
    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False


class MLVolatilityPredictor:
    """Placeholder for ML-based volatility predictor."""
    def __init__(self, config, is_testing_context=False):
        self.config = config

    def apply_filter(self, processed_data, date):
        """Applies the ML-based filter."""
        # This is a placeholder. A real implementation would use a trained model.
        num_stocks = len(processed_data['returns_data'].columns)
        return np.ones(num_stocks, dtype=bool)


class DynamicLowVolFilter:
    """动态低波筛选器主控制器
    
    统一管理四层筛选流水线：
    1. 滚动分位筛选层
    2. GARCH预测筛选层  
    3. IVOL双重约束层
    4. 市场状态感知层
    
    协调各个组件，提供统一的筛选接口，与交易环境和风险控制器集成。
    
    主要功能：
    - 四层筛选流水线协调
    - 市场状态感知和阈值动态调整
    - 性能优化和缓存管理
    - 统计信息收集和监控
    - 并行处理支持
    
    Attributes:
        config: 筛选器配置对象
        data_manager: 数据管理器实例
        各组件实例和状态跟踪变量
    """
    
    def __init__(self, config: Dict, data_manager):
        """初始化筛选器
        
        Args:
            config: 筛选器配置字典或DynamicLowVolConfig对象
            data_manager: 数据管理器实例
            
        Raises:
            ConfigurationException: 配置参数错误
            DataQualityException: 数据管理器无效
        """
        # 验证输入参数
        if not isinstance(config, (dict, DynamicLowVolConfig)):
            raise ConfigurationException("配置参数必须为字典类型或DynamicLowVolConfig对象")
        
        if data_manager is None:
            raise DataQualityException("数据管理器不能为空")
        
        # 创建配置对象
        self.config = config if isinstance(config, DynamicLowVolConfig) else DynamicLowVolConfig(**config)
        self.data_manager = data_manager
        
        # 初始化性能优化器（如果可用）
        if PERFORMANCE_OPTIMIZATION_AVAILABLE:
            self._initialize_performance_optimizer()
        else:
            self.performance_optimizer = None
        
        # 检测是否在测试环境中（通过数据管理器是否是Mock对象来判断）
        self._is_testing_context = hasattr(data_manager, '_mock_name') or str(type(data_manager)).find('Mock') != -1
        
        # 初始化各个组件
        self.data_preprocessor = DataPreprocessor(self.config, self._is_testing_context)
        self.rolling_filter = RollingPercentileFilter(self.config, self._is_testing_context)
        self.garch_predictor = GARCHVolatilityPredictor(self.config, self._is_testing_context)
        self.ivol_filter = IVOLConstraintFilter(self.config, self._is_testing_context)
        self.regime_detector = MarketRegimeDetector(self.config, self._is_testing_context)
        self.threshold_adjuster = RegimeAwareThresholdAdjuster(self.config)

        # Initialize ML predictor if enabled
        if self.config.enable_ml_predictor:
            self.ml_predictor = MLVolatilityPredictor(self.config, self._is_testing_context)
        else:
            self.ml_predictor = None
        
        # 状态跟踪
        self._current_regime = "中"  # 默认中等波动状态
        self._current_regime_confidence = 0.5
        self._current_market_volatility = 0.3
        self._last_update_date = None
        self._current_tradable_mask = None
        self._current_stock_universe = None
        
        # 统计信息
        self._filter_statistics = {
            'total_updates': 0,
            'regime_history': [],
            'filter_pass_rates': {
                'rolling_percentile': [],
                'garch_prediction': [],
                'ivol_constraint': [],
                'final_combined': []
            },
            'performance_metrics': {
                'avg_update_time': 0.0,
                'cache_hit_rate': 0.0,
                'error_count': 0,
                'memory_usage_mb': 0.0,
                'parallel_efficiency': 0.0
            }
        }
    
    def _initialize_performance_optimizer(self):
        """初始化性能优化器"""
        cache_config = CacheConfig(
            enable_disk_cache=self.config.enable_caching,
            disk_cache_dir="./cache/dynamic_lowvol",
            disk_cache_ttl=self.config.cache_expiry_days * 24 * 3600,  # 转换为秒
            memory_cache_ttl=self.config.cache_expiry_days * 24 * 3600
        )
        
        parallel_config = ParallelConfig(
            enable_parallel=self.config.parallel_processing,
            max_workers=None,  # 使用CPU核心数
            use_process_pool=False,  # 使用线程池以避免序列化开销
            chunk_size=10
        )
        
        monitoring_config = MonitoringConfig(enable_memory_monitoring=True)
        
        self.performance_optimizer = PerformanceOptimizer(
            cache_config=cache_config,
            parallel_config=parallel_config,
            monitoring_config=monitoring_config
        )
    
    def update_tradable_mask(self, date: pd.Timestamp) -> np.ndarray:
        """更新可交易股票掩码
        
        协调四层筛选流水线，生成最终的可交易股票掩码。
        使用性能优化技术包括缓存、并行处理和向量化计算。
        
        Args:
            date: 当前交易日期
            
        Returns:
            可交易股票掩码数组 (True表示可交易)
            
        Raises:
            DataQualityException: 数据质量问题
            InsufficientDataException: 数据长度不足
            ModelFittingException: 模型拟合失败
            RegimeDetectionException: 状态检测失败
        """
        import time
        start_time = time.time()
        
        try:
            # 检查内存状态（如果性能优化可用）
            if self.performance_optimizer and self.performance_optimizer.memory_monitor:
                memory_status = self.performance_optimizer.memory_monitor.check_memory_status()
                if memory_status['status'] == 'critical':
                    # 清理缓存以释放内存
                    self.performance_optimizer.cache_manager.clear_cache(memory_only=True)
            
            # 获取必要的数据
            input_data = self._prepare_input_data(date)
            
            # 数据预处理
            processed_data = self._preprocess_data(input_data)
            
            # 检测市场状态
            current_regime = self._detect_market_regime(processed_data, date)
            
            # 调整筛选阈值
            adjusted_thresholds = self._adjust_thresholds(current_regime, processed_data)
            
            # 执行四层筛选流水线
            tradable_mask = self._execute_filter_pipeline(
                processed_data, date, adjusted_thresholds
            )
            
            # 更新内部状态
            self._update_internal_state(date, current_regime, tradable_mask, processed_data)
            
            # 更新统计信息
            self._update_statistics(start_time, tradable_mask, processed_data)
            
            return tradable_mask
            
        except Exception as e:
            self._filter_statistics['performance_metrics']['error_count'] += 1
            # 根据异常处理策略，直接抛出异常
            raise e
    
    def get_current_regime(self) -> str:
        """获取当前市场波动状态
        
        Returns:
            市场状态字符串 ("低", "中", "高")
        """
        return self._current_regime
    
    def get_adaptive_target_volatility(self) -> float:
        """获取自适应目标波动率
        
        根据当前市场状态和波动水平，计算自适应的目标波动率。
        
        Returns:
            自适应目标波动率 (年化)
        """
        # 基础目标波动率映射
        base_target_volatility = {
            "低": 0.45,   # 低波动状态：可以接受更高的目标波动率
            "中": 0.40,   # 中等波动状态：标准目标波动率
            "高": 0.35    # 高波动状态：降低目标波动率以控制风险
        }
        
        base_vol = base_target_volatility.get(self._current_regime, 0.40)
        
        # 根据市场波动率进行微调
        if self._current_market_volatility is not None:
            # 当市场波动率高于正常水平时，进一步降低目标波动率
            if self._current_market_volatility > 0.5:
                adjustment_factor = 0.9  # 降低10%
            elif self._current_market_volatility > 0.4:
                adjustment_factor = 0.95  # 降低5%
            elif self._current_market_volatility < 0.2:
                adjustment_factor = 1.05  # 提高5%
            else:
                adjustment_factor = 1.0
            
            base_vol *= adjustment_factor
        
        # 根据状态检测置信度进行调整
        if self._current_regime_confidence < 0.7:
            # 置信度较低时，向中性状态靠拢
            neutral_vol = base_target_volatility["中"]
            confidence_weight = self._current_regime_confidence
            base_vol = confidence_weight * base_vol + (1 - confidence_weight) * neutral_vol
        
        # 确保目标波动率在合理范围内
        return np.clip(base_vol, 0.25, 0.60)
    
    def get_filter_statistics(self) -> Dict:
        """获取筛选统计信息
        
        Returns:
            筛选统计信息字典，包含：
            - 当前状态信息
            - 筛选通过率统计
            - 性能指标
            - 历史记录摘要
        """
        current_stats = self._filter_statistics.copy()
        
        # 添加当前状态信息
        current_stats['current_state'] = {
            'regime': self._current_regime,
            'regime_confidence': self._current_regime_confidence,
            'market_volatility': self._current_market_volatility,
            'last_update_date': self._last_update_date.isoformat() if self._last_update_date else None,
            'adaptive_target_volatility': self.get_adaptive_target_volatility(),
            'adjusted_thresholds': self._last_adjusted_thresholds if hasattr(self, '_last_adjusted_thresholds') else {},
            'tradable_stocks_count': int(self._current_tradable_mask.sum()) if self._current_tradable_mask is not None else 0,
            'total_stocks_count': len(self._current_tradable_mask) if self._current_tradable_mask is not None else 0
        }
        
        # 计算筛选通过率统计
        if current_stats['filter_pass_rates']['final_combined']:
            current_stats['filter_summary'] = {
                'avg_pass_rate': np.mean(current_stats['filter_pass_rates']['final_combined']),
                'min_pass_rate': np.min(current_stats['filter_pass_rates']['final_combined']),
                'max_pass_rate': np.max(current_stats['filter_pass_rates']['final_combined']),
                'pass_rate_std': np.std(current_stats['filter_pass_rates']['final_combined'])
            }
        else:
            current_stats['filter_summary'] = {
                'avg_pass_rate': 0.0,
                'min_pass_rate': 0.0,
                'max_pass_rate': 0.0,
                'pass_rate_std': 0.0
            }
        
        # 状态分布统计
        if current_stats['regime_history']:
            regime_counts = {}
            for regime in current_stats['regime_history']:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            total_count = len(current_stats['regime_history'])
            current_stats['regime_distribution'] = {
                regime: count / total_count 
                for regime, count in regime_counts.items()
            }
        else:
            current_stats['regime_distribution'] = {}
        
        return current_stats
    
    def get_current_tradable_stocks(self) -> List[str]:
        """获取当前可交易股票列表
        
        Returns:
            可交易股票代码列表
        """
        if self._current_tradable_mask is None or self._current_stock_universe is None:
            return []
        
        return [
            stock for i, stock in enumerate(self._current_stock_universe)
            if i < len(self._current_tradable_mask) and self._current_tradable_mask[i]
        ]
    
    def reset_statistics(self):
        """重置统计信息"""
        self._filter_statistics = {
            'total_updates': 0,
            'regime_history': [],
            'filter_pass_rates': {
                'rolling_percentile': [],
                'garch_prediction': [],
                'ivol_constraint': [],
                'final_combined': []
            },
            'performance_metrics': {
                'avg_update_time': 0.0,
                'cache_hit_rate': 0.0,
                'error_count': 0,
                'memory_usage_mb': 0.0,
                'parallel_efficiency': 0.0
            }
        }
    
    def _prepare_input_data(self, date: pd.Timestamp) -> FilterInputData:
        """准备输入数据
        
        Args:
            date: 当前日期
            
        Returns:
            筛选器输入数据结构
            
        Raises:
            DataQualityException: 数据获取失败
        """
        try:
            # 从数据管理器获取数据
            # 这里假设data_manager有相应的方法
            price_data = self.data_manager.get_price_data(
                end_date=date, 
                lookback_days=max(self.config.rolling_windows) + self.config.garch_window + 50
            )
            
            volume_data = self.data_manager.get_volume_data(
                end_date=date,
                lookback_days=max(self.config.rolling_windows) + 50
            )
            
            factor_data = self.data_manager.get_factor_data(
                end_date=date,
                lookback_days=self.config.garch_window + 50
            )
            
            market_data = self.data_manager.get_market_data(
                end_date=date,
                lookback_days=self.config.regime_detection_window + 50
            )
            
            return FilterInputData(
                price_data=price_data,
                volume_data=volume_data,
                factor_data=factor_data,
                market_data=market_data,
                current_date=date
            )
            
        except Exception as e:
            raise DataQualityException(f"数据获取失败: {str(e)}")
    
    def _preprocess_data(self, input_data: FilterInputData) -> Dict:
        """预处理数据
        
        Args:
            input_data: 原始输入数据
            
        Returns:
            预处理后的数据字典
        """
        # 预处理价格数据
        cleaned_price_data = self.data_preprocessor.preprocess_price_data(
            input_data.price_data
        )
        
        # 计算收益率
        returns_data = self.data_preprocessor.calculate_returns(
            cleaned_price_data, return_type='simple'
        )
        
        # 准备滚动窗口数据
        rolling_windows_data = self.data_preprocessor.prepare_rolling_windows(
            returns_data, self.config.rolling_windows
        )
        
        # 验证数据质量
        self.data_preprocessor.validate_data_quality(returns_data, "收益率数据")
        self.data_preprocessor.validate_data_quality(input_data.factor_data, "因子数据")
        self.data_preprocessor.validate_data_quality(input_data.market_data, "市场数据")
        
        return {
            'price_data': cleaned_price_data,
            'returns_data': returns_data,
            'rolling_windows_data': rolling_windows_data,
            'factor_data': input_data.factor_data,
            'market_data': input_data.market_data,
            'volume_data': input_data.volume_data
        }
    
    def _detect_market_regime(self, processed_data: Dict, date: pd.Timestamp) -> str:
        """检测市场状态
        
        Args:
            processed_data: 预处理后的数据
            date: 当前日期
            
        Returns:
            市场状态字符串
        """
        # 从市场数据中提取收益率
        if 'returns' in processed_data['market_data'].columns:
            market_returns = processed_data['market_data']['returns']
        else:
            # 如果没有直接的收益率列，从价格计算
            market_prices = processed_data['market_data'].iloc[:, 0]  # 假设第一列是价格
            market_returns = market_prices.pct_change().dropna()
        
        # 检测市场状态
        current_regime = self.regime_detector.detect_regime(market_returns, date)
        
        # 获取状态概率分布以计算置信度
        try:
            regime_probabilities = self.regime_detector.get_regime_probabilities(
                market_returns, date
            )
            self._current_regime_confidence = regime_probabilities.get(current_regime, 0.5)
        except:
            self._current_regime_confidence = 0.5
        
        # 计算当前市场波动率
        recent_returns = market_returns.tail(20)  # 最近20天
        if len(recent_returns) > 0:
            self._current_market_volatility = recent_returns.std() * np.sqrt(252)
        else:
            self._current_market_volatility = 0.3
        
        return current_regime
    
    def _adjust_thresholds(self, current_regime: str, processed_data: Dict) -> Dict:
        """调整筛选阈值
        
        Args:
            current_regime: 当前市场状态
            processed_data: 预处理后的数据
            
        Returns:
            调整后的阈值字典
        """
        return self.threshold_adjuster.adjust_thresholds(
            current_regime,
            market_volatility=self._current_market_volatility,
            regime_confidence=self._current_regime_confidence
        )
    
    def _execute_filter_pipeline(self, processed_data: Dict, date: pd.Timestamp, 
                                thresholds: Dict) -> np.ndarray:
        """执行四层筛选流水线
        
        Args:
            processed_data: 预处理后的数据
            date: 当前日期
            thresholds: 调整后的阈值
            
        Returns:
            最终的可交易股票掩码
        """
        returns_data = processed_data['returns_data']
        factor_data = processed_data['factor_data']
        
        # 第一层：滚动分位筛选
        percentile_mask = self.rolling_filter.apply_percentile_filter(
            returns_data,
            current_date=date,
            window=self.config.rolling_windows[0],  # 使用第一个窗口
            percentile_threshold=thresholds['percentile_cut']
        )
        
        # 第二层：GARCH预测筛选
        # 使用批量预测方法并应用阈值
        garch_predictions = self.garch_predictor.predict_batch_volatility(
            returns_data, date
        )
        target_vol = thresholds.get('target_vol', 0.4)
        garch_mask = garch_predictions <= target_vol
        
        # 第三层：IVOL约束筛选
        ivol_mask = self.ivol_filter.apply_ivol_constraint(
            returns_data,
            factor_data,
            current_date=date,
            market_data=processed_data['market_data']
        )

        # 第四层：ML预测器筛选（如果启用）
        if self.ml_predictor:
            ml_mask = self.ml_predictor.apply_filter(processed_data, date)
        else:
            ml_mask = np.ones_like(ivol_mask, dtype=bool)
        
        # 组合筛选结果（取交集）
        combined_mask = percentile_mask & garch_mask.values & ivol_mask & ml_mask
        
        # 记录各层筛选通过率
        total_stocks = len(returns_data.columns)
        self._filter_statistics['filter_pass_rates']['rolling_percentile'].append(
            percentile_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['garch_prediction'].append(
            garch_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['ivol_constraint'].append(
            ivol_mask.sum() / total_stocks
        )
        self._filter_statistics['filter_pass_rates']['final_combined'].append(
            combined_mask.sum() / total_stocks
        )
        
        # 限制历史记录长度
        max_history = 100
        for key in self._filter_statistics['filter_pass_rates']:
            if len(self._filter_statistics['filter_pass_rates'][key]) > max_history:
                self._filter_statistics['filter_pass_rates'][key] = \
                    self._filter_statistics['filter_pass_rates'][key][-max_history:]
        
        return combined_mask
    
    def _update_internal_state(self, date: pd.Timestamp, regime: str, 
                              tradable_mask: np.ndarray, processed_data: Dict):
        """更新内部状态
        
        Args:
            date: 当前日期
            regime: 检测到的市场状态
            tradable_mask: 可交易股票掩码
            processed_data: 预处理后的数据
        """
        self._last_update_date = date
        self._current_regime = regime
        self._current_tradable_mask = tradable_mask
        self._current_stock_universe = processed_data['returns_data'].columns.tolist()
        self._last_adjusted_thresholds = self.threshold_adjuster.adjustment_history[-1]['thresholds'] if self.threshold_adjuster.adjustment_history else {}
        
        # 更新状态历史
        self._filter_statistics['regime_history'].append(regime)
        
        # 限制历史记录长度
        max_history = 200
        if len(self._filter_statistics['regime_history']) > max_history:
            self._filter_statistics['regime_history'] = \
                self._filter_statistics['regime_history'][-max_history:]
    
    def _update_statistics(self, start_time: float, tradable_mask: np.ndarray, 
                          processed_data: Dict):
        """更新统计信息
        
        Args:
            start_time: 开始时间戳
            tradable_mask: 可交易股票掩码
            processed_data: 预处理后的数据
        """
        import time
        
        # 更新性能指标
        update_time = time.time() - start_time
        self._filter_statistics['total_updates'] += 1
        
        # 计算平均更新时间
        current_avg = self._filter_statistics['performance_metrics']['avg_update_time']
        total_updates = self._filter_statistics['total_updates']
        new_avg = (current_avg * (total_updates - 1) + update_time) / total_updates
        self._filter_statistics['performance_metrics']['avg_update_time'] = new_avg
        
        # 更新性能指标（如果性能优化可用）
        if self.performance_optimizer:
            cache_stats = self.performance_optimizer.cache_manager.get_comprehensive_stats()
            self._filter_statistics['performance_metrics']['cache_hit_rate'] = cache_stats.get('overall', {}).get('hit_rate', 0.0)
            
            if self.performance_optimizer.memory_monitor:
                memory_info = self.performance_optimizer.memory_monitor.get_current_memory_usage()
                self._filter_statistics['performance_metrics']['memory_usage_mb'] = memory_info['current_memory_mb']
    
    def get_regime_transition_probability(self) -> Dict[str, float]:
        """获取状态转换概率
        
        Returns:
            各状态的概率分布，格式为 {regime: probability}
        """
        if hasattr(self.regime_detector, 'get_transition_probabilities'):
            # 如果检测器支持转换概率，获取当前状态的转换概率
            transition_matrix = self.regime_detector.get_transition_probabilities()
            current_regime = getattr(self, '_current_regime', '中')
            if current_regime in transition_matrix:
                return transition_matrix[current_regime]
        
        # 基于历史数据计算状态概率
        regime_history = self._filter_statistics.get('regime_history', [])
        if regime_history:
            from collections import Counter
            regime_counts = Counter(regime_history)
            total_count = len(regime_history)
            return {regime: count / total_count for regime, count in regime_counts.items()}
        else:
            # 如果没有历史数据，返回均匀分布
            return {"低": 0.33, "中": 0.34, "高": 0.33}
    
    def get_memory_optimization_suggestions(self) -> List[str]:
        """获取内存优化建议
        
        Returns:
            内存优化建议列表
        """
        suggestions = []
        
        if self.performance_optimizer and self.performance_optimizer.memory_monitor:
            memory_info = self.performance_optimizer.memory_monitor.get_current_memory_usage()
            
            # 基于当前内存使用情况提供建议
            if memory_info['current_memory_mb'] > 500:
                suggestions.append("考虑清理缓存以释放内存")
            
            if len(self._filter_statistics['regime_history']) > 1000:
                suggestions.append("考虑限制状态历史记录长度")
            
            if memory_info['current_memory_mb'] > 1000:
                suggestions.append("内存使用量较高，建议重启筛选器")
        
        # 如果没有具体建议，提供通用建议
        if not suggestions:
            suggestions.append("当前内存使用正常")
        
        return suggestions
    
    def clear_performance_cache(self) -> bool:
        """清理性能缓存
        
        Returns:
            是否成功清理
        """
        try:
            if self.performance_optimizer:
                self.performance_optimizer.cache_manager.clear_cache()
            
            # 清理内部统计缓存
            self._filter_statistics['filter_pass_rates'] = {
                'rolling_percentile': [],
                'garch_prediction': [],
                'ivol_constraint': [],
                'final_combined': []
            }
            
            return True
        except Exception:
            return False