"""
特征工程模块
实现技术指标计算、基本面因子提取和市场微观结构特征计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, VarianceThreshold
from sklearn.impute import SimpleImputer
import logging

from .data_models import FeatureVector

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.scalers = {}
        self.feature_names = {}
        # 特征缓存机制
        self._feature_cache = {}
        self._cache_enabled = False
        
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 价格数据，包含open, high, low, close, volume, amount列
            
        Returns:
            包含技术指标的DataFrame
        """
        if data.empty:
            raise ValueError("输入数据不能为空")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"缺少必要列: {missing_columns}")
        
        result = pd.DataFrame(index=data.index)
        
        # 计算各种技术指标
        sma_result = self.calculate_sma(data)
        ema_result = self.calculate_ema(data)
        rsi_result = self.calculate_rsi(data)
        macd_result = self.calculate_macd(data)
        bb_result = self.calculate_bollinger_bands(data)
        stoch_result = self.calculate_stochastic(data)
        atr_result = self.calculate_atr(data)
        volume_result = self.calculate_volume_indicators(data)
        
        # 合并所有指标
        for df in [sma_result, ema_result, rsi_result, macd_result, 
                  bb_result, stoch_result, atr_result, volume_result]:
            result = result.join(df, how='outer')
        
        # 添加基础价格特征以匹配已训练模型的37个特征期望
        result['high_low_ratio'] = data['high'] / data['low']
        result['close_open_ratio'] = data['close'] / data['open']
        
        return result
    
    def calculate_sma(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """计算简单移动平均线"""
        result = pd.DataFrame(index=data.index)
        
        # 只使用与已训练模型一致的SMA窗口
        windows = [5, 10]
        for w in windows:
            if len(data) >= w:
                result[f'sma_{w}'] = data['close'].rolling(window=w).mean()
        
        return result
    
    def calculate_ema(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算指数移动平均线"""
        result = pd.DataFrame(index=data.index)
        
        # 计算EMA12和EMA26
        result['ema_12'] = data['close'].ewm(span=12).mean()
        result['ema_26'] = data['close'].ewm(span=26).mean()
        
        # 添加额外EMA以匹配已训练模型的37个特征期望
        result['ema_50'] = data['close'].ewm(span=50).mean()
        result['ema_100'] = data['close'].ewm(span=100).mean()
        result['ema_200'] = data['close'].ewm(span=200).mean()
        
        return result
    
    def calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """计算相对强弱指数"""
        result = pd.DataFrame(index=data.index)
        
        # 计算多个窗口的RSI以匹配已训练模型的37个特征期望
        windows = [7, 14, 21, 30]
        
        for w in windows:
            # 计算价格变化
            delta = data['close'].diff()
            
            # 分离上涨和下跌
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # 计算平均收益和损失
            avg_gain = gain.rolling(window=w).mean()
            avg_loss = loss.rolling(window=w).mean()
            
            # 计算RSI
            rs = avg_gain / avg_loss
            result[f'rsi_{w}'] = 100 - (100 / (1 + rs))
        
        return result
    
    def calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        result = pd.DataFrame(index=data.index)
        
        # 计算EMA
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        
        # 计算MACD线
        result['macd'] = ema_12 - ema_26
        
        # 计算信号线
        result['macd_signal'] = result['macd'].ewm(span=9).mean()
        
        # 计算MACD直方图
        result['macd_histogram'] = result['macd'] - result['macd_signal']
        
        return result
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """计算布林带"""
        result = pd.DataFrame(index=data.index)
        
        # 计算移动平均和标准差
        sma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        
        # 计算布林带
        result['bb_upper'] = sma + (std * std_dev)
        result['bb_middle'] = sma
        result['bb_lower'] = sma - (std * std_dev)
        
        # 计算布林带宽度和位置
        result['bb_width'] = result['bb_upper'] - result['bb_lower']
        result['bb_position'] = (data['close'] - result['bb_lower']) / result['bb_width']
        
        return result
    
    def calculate_stochastic(self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        """计算随机指标"""
        result = pd.DataFrame(index=data.index)
        
        # 计算最高价和最低价
        high_max = data['high'].rolling(window=k_window).max()
        low_min = data['low'].rolling(window=k_window).min()
        
        # 计算%K
        result['stoch_k'] = 100 * (data['close'] - low_min) / (high_max - low_min)
        
        # 计算%D
        result['stoch_d'] = result['stoch_k'].rolling(window=d_window).mean()
        
        return result
    
    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """计算平均真实波幅"""
        result = pd.DataFrame(index=data.index)
        
        # 计算真实波幅
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # 计算ATR
        result['atr_14'] = true_range.rolling(window=window).mean()
        
        return result
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        result = pd.DataFrame(index=data.index)
        
        # 成交量移动平均
        result['volume_sma'] = data['volume'].rolling(window=20).mean()
        
        # 成交量比率
        result['volume_ratio'] = data['volume'] / result['volume_sma']
        
        # OBV (On Balance Volume)
        price_change = data['close'].diff()
        volume_direction = np.where(price_change > 0, data['volume'], 
                                  np.where(price_change < 0, -data['volume'], 0))
        result['obv'] = volume_direction.cumsum()
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        result['vwap'] = (typical_price * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
        
        return result
    
    def calculate_fundamental_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算基本面因子
        
        Args:
            data: 基本面数据
            
        Returns:
            包含基本面因子的DataFrame
        """
        if data.empty:
            raise ValueError("输入数据不能为空")
        
        result = pd.DataFrame(index=data.index)
        
        # 计算各类基本面因子
        valuation_result = self.calculate_valuation_factors(data)
        profitability_result = self.calculate_profitability_factors(data)
        growth_result = self.calculate_growth_factors(data)
        leverage_result = self.calculate_leverage_factors(data)
        
        # 合并所有因子
        for df in [valuation_result, profitability_result, growth_result, leverage_result]:
            result = result.join(df, how='outer')
        
        return result
    
    def calculate_valuation_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算估值因子"""
        result = pd.DataFrame(index=data.index)
        
        # 直接使用已有的估值指标
        if 'pe_ratio' in data.columns:
            result['pe_ratio'] = data['pe_ratio']
        if 'pb_ratio' in data.columns:
            result['pb_ratio'] = data['pb_ratio']
        
        # 计算其他估值指标（如果有相关数据）
        if 'market_cap' in data.columns and 'revenue' in data.columns:
            result['ps_ratio'] = data['market_cap'] / data['revenue']
        if 'market_cap' in data.columns and 'cash_flow' in data.columns:
            result['pcf_ratio'] = data['market_cap'] / data['cash_flow']
        
        return result
    
    def calculate_profitability_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算盈利能力因子"""
        result = pd.DataFrame(index=data.index)
        
        # 直接使用已有的盈利指标
        if 'roe' in data.columns:
            result['roe'] = data['roe']
        if 'roa' in data.columns:
            result['roa'] = data['roa']
        
        # 计算其他盈利指标（如果有相关数据）
        if 'gross_profit' in data.columns and 'revenue' in data.columns:
            result['gross_margin'] = data['gross_profit'] / data['revenue']
        if 'net_profit' in data.columns and 'revenue' in data.columns:
            result['net_margin'] = data['net_profit'] / data['revenue']
        
        return result
    
    def calculate_growth_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算成长性因子"""
        result = pd.DataFrame(index=data.index)
        
        # 直接使用已有的成长指标
        if 'revenue_growth' in data.columns:
            result['revenue_growth'] = data['revenue_growth']
        if 'profit_growth' in data.columns:
            result['profit_growth'] = data['profit_growth']
        
        # 计算EPS增长率（如果有相关数据）
        if 'eps' in data.columns:
            result['eps_growth'] = data['eps'].pct_change(periods=4, fill_method=None)  # 年度增长率
        
        return result
    
    def calculate_leverage_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算杠杆因子"""
        result = pd.DataFrame(index=data.index)
        
        # 直接使用已有的杠杆指标
        if 'debt_ratio' in data.columns:
            result['debt_ratio'] = data['debt_ratio']
        if 'current_ratio' in data.columns:
            result['current_ratio'] = data['current_ratio']
        
        # 计算其他杠杆指标（如果有相关数据）
        if 'total_debt' in data.columns and 'total_equity' in data.columns:
            result['debt_to_equity'] = data['total_debt'] / data['total_equity']
        if 'quick_assets' in data.columns and 'current_liabilities' in data.columns:
            result['quick_ratio'] = data['quick_assets'] / data['current_liabilities']
        
        return result
    
    def calculate_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场微观结构特征
        
        Args:
            data: 价格数据
            
        Returns:
            包含微观结构特征的DataFrame
        """
        if data.empty:
            raise ValueError("输入数据不能为空")
        
        result = pd.DataFrame(index=data.index)
        
        # 计算各类微观结构特征
        liquidity_result = self.calculate_liquidity_features(data)
        volatility_result = self.calculate_volatility_features(data)
        momentum_result = self.calculate_momentum_features(data)
        
        # 合并所有特征
        for df in [liquidity_result, volatility_result, momentum_result]:
            result = result.join(df, how='outer')
        
        return result
    
    def calculate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算流动性特征"""
        result = pd.DataFrame(index=data.index)
        
        # 换手率（如果有流通股本数据）
        if 'float_shares' in data.columns:
            result['turnover_rate'] = data['volume'] / data['float_shares']
        else:
            # 使用成交量相对指标
            result['turnover_rate'] = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Amihud非流动性指标
        returns = data['close'].pct_change(fill_method=None).abs()
        dollar_volume = data['amount']
        # 安全处理除零情况：当成交量为0或NaN时，设置流动性指标为无穷大（表示完全不流动）
        with np.errstate(divide='ignore', invalid='ignore'):
            amihud = returns / dollar_volume
            # 将无穷大和NaN替换为一个很大的数值，表示极低的流动性
            result['amihud_illiquidity'] = np.where(
                np.isfinite(amihud), amihud, 1e6
            )
        
        # 买卖价差（简化计算）
        result['bid_ask_spread'] = (data['high'] - data['low']) / data['close']
        
        return result
    
    def calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率特征"""
        result = pd.DataFrame(index=data.index)
        
        # 已实现波动率
        returns = data['close'].pct_change(fill_method=None)
        result['realized_volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Garman-Klass波动率估计（优化：使用向量化操作）
        high_low_ratio = data['high'] / data['low']
        close_open_ratio = data['close'] / data['open']
        
        # 优化：使用numpy向量化操作代替apply lambda
        log_hl = np.where(high_low_ratio > 1, np.log(high_low_ratio), 0)
        log_co = np.where((close_open_ratio > 0) & (close_open_ratio != 1), np.log(close_open_ratio), 0)
        
        gk_vol = log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        result['garman_klass_volatility'] = pd.Series(gk_vol, index=data.index).rolling(window=20).mean() * 252
        
        # Parkinson波动率估计（优化：使用向量化操作）
        log_hl_parkinson = np.where(high_low_ratio > 1, np.log(high_low_ratio), 0)
        parkinson_vol = log_hl_parkinson ** 2 / (4 * np.log(2))
        result['parkinson_volatility'] = pd.Series(parkinson_vol, index=data.index).rolling(window=20).mean() * 252
        
        return result
    
    def calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算动量特征"""
        result = pd.DataFrame(index=data.index)
        
        # 价格动量
        result['price_momentum_1m'] = data['close'].pct_change(periods=20, fill_method=None)  # 1个月
        result['price_momentum_3m'] = data['close'].pct_change(periods=60, fill_method=None)  # 3个月
        
        # 成交量动量
        result['volume_momentum'] = data['volume'].pct_change(periods=20, fill_method=None)
        
        return result
    
    def normalize_features(self, features: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        标准化特征
        
        Args:
            features: 特征数据
            method: 标准化方法，'zscore', 'minmax', 'robust'
            
        Returns:
            标准化后的特征数据
        """
        if features.empty:
            return features
        
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        # 只对数值列进行标准化
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return features
        
        result = features.copy()
        result[numeric_columns] = scaler.fit_transform(features[numeric_columns])
        
        # 保存scaler以便后续使用
        self.scalers[method] = scaler
        
        return result
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            data: 输入数据
            method: 处理方法，'ffill', 'bfill', 'mean', 'median', 'drop', 'ffill_bfill_zero'
            
        Returns:
            处理后的数据
        """
        if data.empty:
            return data
        
        result = data.copy()
        
        if method == 'ffill':
            result = result.ffill()
        elif method == 'bfill':
            result = result.bfill()
        elif method == 'ffill_bfill_zero':
            # 三步处理：先前向填充，再后向填充，最后用0填充剩余的NaN
            result = result.ffill().bfill().fillna(0)
        elif method == 'mean':
            imputer = SimpleImputer(strategy='mean')
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                result[numeric_columns] = imputer.fit_transform(result[numeric_columns])
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
            numeric_columns = result.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                result[numeric_columns] = imputer.fit_transform(result[numeric_columns])
        elif method == 'drop':
            result = result.dropna()
        else:
            raise ValueError(f"不支持的缺失值处理方法: {method}")
        
        return result
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        检测异常值
        
        Args:
            data: 输入数据
            method: 检测方法，'iqr', 'zscore'
            threshold: 阈值
            
        Returns:
            异常值标记（True表示异常值）
        """
        if data.empty:
            return pd.DataFrame()
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        outliers = pd.DataFrame(False, index=data.index, columns=data.columns)
        
        for col in numeric_columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (data[col] < lower_bound) | (data[col] > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers[col] = z_scores > threshold
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
        
        return outliers
    
    def treat_outliers(self, data: pd.DataFrame, method: str = 'clip', threshold: float = 1.5) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            data: 输入数据
            method: 处理方法，'clip', 'remove'
            threshold: 阈值
            
        Returns:
            处理后的数据
        """
        if data.empty:
            return data
        
        result = data.copy()
        outliers = self.detect_outliers(data, threshold=threshold)
        
        if method == 'clip':
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'remove':
            # 移除任何列有异常值的行
            outlier_rows = outliers.any(axis=1)
            result = result[~outlier_rows]
        else:
            raise ValueError(f"不支持的异常值处理方法: {method}")
        
        return result
    
    def select_features_by_correlation(self, features: pd.DataFrame, target: pd.Series, 
                                     threshold: float = 0.1) -> List[str]:
        """
        基于相关性选择特征
        
        Args:
            features: 特征数据
            target: 目标变量
            threshold: 相关性阈值
            
        Returns:
            选择的特征名列表
        """
        correlations = features.corrwith(target).abs()
        selected_features = correlations[correlations >= threshold].index.tolist()
        return selected_features
    
    def select_features_by_mutual_info(self, features: pd.DataFrame, target: pd.Series, 
                                     k: int = 10) -> List[str]:
        """
        基于互信息选择特征
        
        Args:
            features: 特征数据
            target: 目标变量
            k: 选择的特征数量
            
        Returns:
            选择的特征名列表
        """
        # 处理缺失值
        features_clean = features.fillna(features.mean())
        target_clean = target.fillna(target.mean())
        
        # 确保索引对齐
        common_index = features_clean.index.intersection(target_clean.index)
        features_clean = features_clean.loc[common_index]
        target_clean = target_clean.loc[common_index]
        
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(features_clean, target_clean)
        
        selected_features = features.columns[selector.get_support()].tolist()
        return selected_features
    
    def select_features_by_variance(self, features: pd.DataFrame, threshold: float = 0.1) -> List[str]:
        """
        基于方差阈值选择特征
        
        Args:
            features: 特征数据
            threshold: 方差阈值
            
        Returns:
            选择的特征名列表
        """
        # 处理缺失值
        features_clean = features.fillna(features.mean())
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(features_clean)
        
        selected_features = features.columns[selector.get_support()].tolist()
        return selected_features
    
    def enable_feature_cache(self):
        """启用特征缓存"""
        self._cache_enabled = True
        logger.info("特征缓存已启用")
    
    def disable_feature_cache(self):
        """禁用特征缓存"""
        self._cache_enabled = False
        logger.info("特征缓存已禁用")
    
    def clear_feature_cache(self):
        """清除特征缓存"""
        self._feature_cache.clear()
        logger.info("特征缓存已清除")
    
    def _generate_cache_key(self, data: pd.DataFrame, method_name: str) -> str:
        """生成缓存键"""
        # 使用数据的形状、索引范围和方法名生成唯一键
        if isinstance(data.index, pd.MultiIndex):
            # 对于MultiIndex，使用第一层和第二层的最小最大值
            level0_min = data.index.get_level_values(0).min()
            level0_max = data.index.get_level_values(0).max()
            level1_min = data.index.get_level_values(1).min()
            level1_max = data.index.get_level_values(1).max()
            key_parts = [method_name, str(data.shape), str(level0_min), str(level0_max), 
                        str(level1_min), str(level1_max)]
        else:
            index_min = data.index.min()
            index_max = data.index.max()
            key_parts = [method_name, str(data.shape), str(index_min), str(index_max)]
        
        return "_".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[pd.DataFrame]:
        """获取缓存结果"""
        if not self._cache_enabled:
            return None
        return self._feature_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: pd.DataFrame):
        """缓存结果"""
        if self._cache_enabled:
            self._feature_cache[cache_key] = result.copy()
    
    def calculate_volatility_features_optimized(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率特征（优化版本，支持缓存）"""
        cache_key = self._generate_cache_key(data, "volatility_features")
        
        # 尝试从缓存获取结果
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            logger.debug(f"从缓存获取波动率特征，键: {cache_key}")
            return cached_result
        
        # 计算新结果
        result = self.calculate_volatility_features(data)
        
        # 缓存结果
        self._cache_result(cache_key, result)
        logger.debug(f"缓存波动率特征，键: {cache_key}")
        
        return result
    
    def combine_features(self, feature_dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        合并多个特征DataFrame
        
        Args:
            feature_dfs: 特征DataFrame列表
            
        Returns:
            合并后的特征DataFrame
        """
        if not feature_dfs:
            return pd.DataFrame()
        
        result = feature_dfs[0].copy()
        
        for df in feature_dfs[1:]:
            result = result.join(df, how='outer', rsuffix='_dup')
            
            # 移除重复列
            dup_columns = [col for col in result.columns if col.endswith('_dup')]
            result = result.drop(columns=dup_columns)
        
        return result
    
    def create_feature_vector(self, timestamp: datetime, symbol: str, 
                            normalized_features: pd.Series) -> FeatureVector:
        """
        创建特征向量对象
        
        Args:
            timestamp: 时间戳
            symbol: 股票代码
            normalized_features: 标准化后的特征
            
        Returns:
            FeatureVector对象
        """
        # 分类特征
        technical_indicators = {}
        fundamental_factors = {}
        market_microstructure = {}
        
        # 技术指标特征
        tech_keywords = ['sma', 'ema', 'rsi', 'macd', 'bb', 'stoch', 'atr', 'volume', 'obv', 'vwap']
        
        # 基本面因子特征
        fundamental_keywords = ['pe', 'pb', 'ps', 'pcf', 'roe', 'roa', 'margin', 'growth', 'debt', 'ratio']
        
        # 市场微观结构特征
        micro_keywords = ['turnover', 'illiquidity', 'spread', 'volatility', 'momentum']
        
        for feature_name, value in normalized_features.items():
            if pd.isna(value):
                value = 0.0
            
            # 根据特征名称分类
            if any(keyword in feature_name.lower() for keyword in tech_keywords):
                technical_indicators[feature_name] = float(value)
            elif any(keyword in feature_name.lower() for keyword in fundamental_keywords):
                fundamental_factors[feature_name] = float(value)
            elif any(keyword in feature_name.lower() for keyword in micro_keywords):
                market_microstructure[feature_name] = float(value)
            else:
                # 默认归类为技术指标
                technical_indicators[feature_name] = float(value)
        
        # 确保每个类别至少有一个特征
        if not technical_indicators:
            technical_indicators['default_tech'] = 0.0
        if not fundamental_factors:
            fundamental_factors['default_fundamental'] = 0.0
        if not market_microstructure:
            market_microstructure['default_micro'] = 0.0
        
        return FeatureVector(
            timestamp=timestamp,
            symbol=symbol,
            technical_indicators=technical_indicators,
            fundamental_factors=fundamental_factors,
            market_microstructure=market_microstructure
        )
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        为投资组合环境计算特征的主要方法
        
        Args:
            data: 价格数据DataFrame，包含open, high, low, close, volume, amount列
            
        Returns:
            包含所有特征的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，返回空DataFrame")
            return pd.DataFrame()
        
        try:
            # 计算各种特征
            logger.info("计算技术指标...")
            technical_features = self.calculate_technical_indicators(data)
            
            logger.debug("计算微观结构特征...")
            microstructure_features = self.calculate_microstructure_features(data)
            
            logger.debug("计算波动率特征...")
            volatility_features = self.calculate_volatility_features(data)
            
            logger.debug("计算动量特征...")
            momentum_features = self.calculate_momentum_features(data)
            
            # 合并所有特征（不包含原始价格数据）
            logger.debug("合并特征...")
            all_features = self.combine_features([
                technical_features,
                microstructure_features,
                volatility_features,
                momentum_features
            ])
            
            # 处理缺失值
            logger.debug("处理缺失值...")
            clean_features = self.handle_missing_values(all_features, method='ffill_bfill_zero')
            
            # 标准化特征
            logger.debug("标准化特征...")
            normalized_features = self.normalize_features(clean_features)
            
            logger.info(f"特征计算完成，共生成 {len(normalized_features.columns)} 个特征")
            return normalized_features
            
        except Exception as e:
            logger.error(f"特征计算失败: {e}")
            raise