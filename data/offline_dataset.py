"""
离线数据集管理器

实现历史数据加载和PyTorch数据集转换，支持Qlib数据接口，
提供数据清洗和标准化功能，创建行为克隆数据集生成功能。
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from torch.utils.data import TensorDataset, Dataset
from dataclasses import dataclass
import logging
from .data_manager import DataManager

logger = logging.getLogger(__name__)


@dataclass
class OfflineDataConfig:
    """离线数据集配置"""
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    lookback_window: int = 20
    prediction_horizon: int = 1
    min_samples_per_stock: int = 100
    normalize_features: bool = True
    include_market_features: bool = True
    behavior_cloning_threshold: float = 0.02


class OfflineDataset(Dataset):
    """离线历史数据集管理器
    
    支持Qlib数据转换为PyTorch格式，实现数据清洗和标准化，
    创建行为克隆数据集生成功能。
    """
    
    def __init__(self, data_manager: DataManager, config: OfflineDataConfig):
        """
        初始化离线数据集
        
        Args:
            data_manager: 数据管理器实例
            config: 离线数据集配置
        """
        self.data_manager = data_manager
        self.config = config
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        self.actions = None
        self.rewards = None
        self.metadata = {}
        
        logger.info(f"初始化离线数据集: {config.start_date} 到 {config.end_date}")
        
    def load_historical_data(self, 
                           instruments: Optional[List[str]] = None,
                           fields: Optional[List[str]] = None) -> 'OfflineDataset':
        """
        加载并转换历史数据为PyTorch数据集
        
        Args:
            instruments: 股票代码列表
            fields: 数据字段列表
            
        Returns:
            self，支持链式调用
        """
        if fields is None:
            fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
            
        # 加载股票数据
        logger.info("开始加载历史股票数据...")
        stock_data = self.data_manager.get_stock_data(
            instruments=instruments,
            start_time=self.config.start_date,
            end_time=self.config.end_date,
            fields=fields
        )
        
        # 加载市场数据
        if self.config.include_market_features:
            logger.info("加载市场指数数据...")
            market_data = self.data_manager.get_market_data(
                start_time=self.config.start_date,
                end_time=self.config.end_date
            )
            self.metadata['market_data'] = market_data
        
        self.raw_data = stock_data
        logger.info(f"加载原始数据完成: {stock_data.shape}")
        
        # 应用数据清洗
        self._apply_data_cleaning()
        
        # 创建特征和目标
        self._create_features_and_targets()
        
        return self
        
    def _apply_data_cleaning(self):
        """应用数据清洗和标准化功能"""
        if self.raw_data is None:
            raise ValueError("必须先加载数据")
            
        logger.info("开始数据清洗和预处理...")
        
        # 复制原始数据
        cleaned_data = self.raw_data.copy()
        
        # 1. 去除异常值
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        
        # 2. 价格数据合理性检查
        price_columns = ['$open', '$high', '$low', '$close', '$vwap']
        for col in cleaned_data.columns:
            col_name = str(col)
            if any(price_col in col_name for price_col in price_columns):
                # 价格必须为正数
                cleaned_data[col] = cleaned_data[col].where(cleaned_data[col] > 0.001)
                
        # 3. 成交量数据检查
        volume_columns = ['$volume']
        for col in cleaned_data.columns:
            col_name = str(col)
            if any(vol_col in col_name for vol_col in volume_columns):
                # 成交量必须非负
                cleaned_data[col] = cleaned_data[col].where(cleaned_data[col] >= 0)
        
        # 4. 填充缺失值 - 先前向填充，再后向填充
        cleaned_data = cleaned_data.ffill().bfill()
        
        # 5. 最终缺失值处理 - 对于仍存在的NaN，用合理的默认值填充
        for col in cleaned_data.columns:
            col_name = str(col)
            if any(price_col in col_name for price_col in price_columns):
                # 价格数据：先尝试用列均值填充，如果均值也是NaN，用固定值
                col_mean = cleaned_data[col].mean()
                fill_value = col_mean if not pd.isna(col_mean) else 50.0  # 默认价格
                cleaned_data[col] = cleaned_data[col].fillna(fill_value)
            elif any(vol_col in col_name for vol_col in volume_columns):
                # 成交量用0填充
                cleaned_data[col] = cleaned_data[col].fillna(0)
            else:
                # 其他数据用0填充
                cleaned_data[col] = cleaned_data[col].fillna(0)
        
        # 最终检查确保没有NaN - 用0填充所有剩余的NaN
        cleaned_data = cleaned_data.fillna(0)
        
        # 额外的安全检查：如果标准化后产生了NaN，也要处理
        if self.config.normalize_features:
            # 在标准化前再次检查
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 6. 数据标准化（如果启用）
        if self.config.normalize_features:
            cleaned_data = self._normalize_features(cleaned_data)
            # 标准化后再次检查和清理NaN
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan).fillna(0)
            
        self.processed_data = cleaned_data
        logger.info(f"数据清洗完成: {cleaned_data.shape}")
        
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """特征标准化"""
        if data.empty or not hasattr(data.index, 'nlevels') or data.index.nlevels < 2:
            return data
            
        normalized_data = data.copy()
        
        # 对每只股票分别标准化
        for instrument in data.index.get_level_values(1).unique():
            # 获取该股票的所有数据
            mask = data.index.get_level_values(1) == instrument
            instrument_data = data[mask]
            
            # 计算滚动标准化参数
            rolling_mean = instrument_data.rolling(window=60, min_periods=20).mean()
            rolling_std = instrument_data.rolling(window=60, min_periods=20).std()
            
            # Z-score标准化
            normalized_instrument = (instrument_data - rolling_mean) / (rolling_std + 1e-8)
            
            # 更新标准化数据
            normalized_data[mask] = normalized_instrument
                
        return normalized_data
        
    def _create_features_and_targets(self):
        """创建特征和目标变量"""
        if self.processed_data is None:
            raise ValueError("必须先进行数据清洗")
            
        # 处理空数据的情况
        if self.processed_data.empty or not hasattr(self.processed_data.index, 'nlevels') or self.processed_data.index.nlevels < 2:
            self.features = np.array([], dtype=np.float32).reshape(0, 0)
            self.targets = np.array([], dtype=np.float32)
            self.metadata['samples'] = []
            logger.warning("数据为空或格式不正确，未创建任何特征样本")
            return
            
        logger.info("创建特征和目标变量...")
        
        features_list = []
        targets_list = []
        metadata_list = []
        
        # 按股票分组处理
        for instrument in self.processed_data.index.get_level_values(1).unique():
            instrument_data = self.processed_data.xs(instrument, level=1)
            
            if len(instrument_data) < self.config.min_samples_per_stock:
                continue
                
            # 创建滑动窗口特征
            for i in range(self.config.lookback_window, 
                          len(instrument_data) - self.config.prediction_horizon):
                
                # 特征：过去lookback_window天的数据
                feature_window = instrument_data.iloc[
                    i - self.config.lookback_window:i
                ].values.flatten()
                
                # 目标：未来prediction_horizon天的收益率
                current_price = instrument_data.iloc[i]['$close']
                future_price = instrument_data.iloc[i + self.config.prediction_horizon]['$close']
                
                # 避免除零和极端值
                if current_price <= 0 or future_price <= 0:
                    continue
                    
                target_return = (future_price - current_price) / current_price
                
                # 过滤极端收益率（超过±50%的日收益率被认为是异常值）
                if abs(target_return) > 0.5:
                    continue
                
                features_list.append(feature_window)
                targets_list.append(target_return)
                
                # 元数据
                metadata_list.append({
                    'instrument': instrument,
                    'date': instrument_data.index[i],
                    'price': current_price
                })
        
        # 转换为numpy数组
        if features_list:
            self.features = np.array(features_list, dtype=np.float32)
            self.targets = np.array(targets_list, dtype=np.float32)
            self.metadata['samples'] = metadata_list
            logger.info(f"创建特征完成: {self.features.shape[0]} 个样本，{self.features.shape[1]} 个特征")
        else:
            self.features = np.array([], dtype=np.float32).reshape(0, 0)
            self.targets = np.array([], dtype=np.float32)
            self.metadata['samples'] = []
            logger.warning("未创建任何有效特征样本")
        
    def create_behavior_dataset(self, 
                              policy_actions: Optional[pd.DataFrame] = None) -> TensorDataset:
        """
        创建行为克隆数据集
        
        Args:
            policy_actions: 专家策略的动作数据
            
        Returns:
            行为克隆数据集
        """
        if self.features is None or self.targets is None:
            raise ValueError("必须先创建特征和目标变量")
            
        logger.info("创建行为克隆数据集...")
        
        if policy_actions is None:
            # 如果没有提供专家动作，基于收益率创建简单的动作标签
            actions = self._generate_simple_actions()
        else:
            # 使用提供的专家动作
            actions = self._align_expert_actions(policy_actions)
            
        # 过滤有效样本
        valid_indices = ~np.isnan(actions)
        valid_features = self.features[valid_indices]
        valid_actions = actions[valid_indices]
        
        # 转换为PyTorch张量
        feature_tensor = torch.FloatTensor(valid_features)
        action_tensor = torch.FloatTensor(valid_actions)
        
        # 创建数据集
        behavior_dataset = TensorDataset(feature_tensor, action_tensor)
        
        logger.info(f"行为克隆数据集创建完成: {len(behavior_dataset)} 个样本")
        return behavior_dataset
        
    def _generate_simple_actions(self) -> np.ndarray:
        """基于收益率生成简单的动作标签"""
        actions = np.zeros(len(self.targets))
        
        # 基于收益率阈值生成动作
        # 动作：[-1, 0, 1] 分别表示卖出、持有、买入
        threshold = self.config.behavior_cloning_threshold
        
        actions[self.targets > threshold] = 1.0   # 买入
        actions[self.targets < -threshold] = -1.0  # 卖出
        # 其余保持0（持有）
        
        return actions
        
    def _align_expert_actions(self, policy_actions: pd.DataFrame) -> np.ndarray:
        """对齐专家动作数据"""
        actions = np.full(len(self.targets), np.nan)
        
        for i, metadata in enumerate(self.metadata['samples']):
            instrument = metadata['instrument']
            date = metadata['date']
            
            # 查找对应的专家动作
            if instrument in policy_actions.columns:
                if date in policy_actions.index:
                    actions[i] = policy_actions.loc[date, instrument]
                    
        return actions
        
    def create_tensor_dataset(self) -> TensorDataset:
        """创建标准的PyTorch张量数据集"""
        if self.features is None or self.targets is None:
            raise ValueError("必须先创建特征和目标变量")
            
        # 转换为PyTorch张量
        feature_tensor = torch.FloatTensor(self.features)
        target_tensor = torch.FloatTensor(self.targets)
        
        return TensorDataset(feature_tensor, target_tensor)
        
    def apply_data_augmentation(self, dataset: TensorDataset) -> TensorDataset:
        """
        应用数据增强技术
        
        Args:
            dataset: 原始数据集
            
        Returns:
            增强后的数据集
        """
        logger.info("应用数据增强...")
        
        features, targets = dataset.tensors
        augmented_features = []
        augmented_targets = []
        
        # 原始数据
        augmented_features.append(features)
        augmented_targets.append(targets)
        
        # 1. 添加噪声
        noise_std = 0.01
        noise = torch.randn_like(features) * noise_std
        augmented_features.append(features + noise)
        augmented_targets.append(targets)
        
        # 2. 时间序列平移（如果特征结构允许）
        if len(features.shape) > 1 and features.shape[1] > 10:
            # 随机时间平移
            shift_size = min(5, features.shape[1] // 10)
            shifted_features = torch.roll(features, shifts=shift_size, dims=1)
            augmented_features.append(shifted_features)
            augmented_targets.append(targets)
        
        # 合并所有增强数据
        final_features = torch.cat(augmented_features, dim=0)
        final_targets = torch.cat(augmented_targets, dim=0)
        
        logger.info(f"数据增强完成: {len(final_features)} 个样本")
        return TensorDataset(final_features, final_targets)
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self) if self.features is not None else 0,
            'feature_dim': self.features.shape[1] if self.features is not None and len(self.features.shape) > 1 else 0,
            'date_range': (self.config.start_date, self.config.end_date),
            'lookback_window': self.config.lookback_window,
            'prediction_horizon': self.config.prediction_horizon
        }
        
        if self.targets is not None and len(self.targets) > 0:
            stats.update({
                'target_mean': float(np.mean(self.targets)),
                'target_std': float(np.std(self.targets)),
                'target_min': float(np.min(self.targets)),
                'target_max': float(np.max(self.targets))
            })
            
        return stats
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.features) if self.features is not None else 0
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        if self.features is None or self.targets is None:
            raise ValueError("数据集未初始化")
            
        if len(self.features) == 0:
            raise IndexError("数据集为空")
            
        feature = torch.FloatTensor(self.features[idx])
        target = torch.FloatTensor([self.targets[idx]])
        
        return feature, target