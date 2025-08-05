"""
数据初始化和加载模块
支持Qlib多频率数据初始化，包含分钟线和日线数据
"""
import os
import qlib
from qlib.config import REG_CN
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.data import LocalExpressionProvider
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class QlibDataLoader:
    """Qlib数据加载器，支持多频率数据初始化"""

    def __init__(self, data_root: str = None):
        self.data_root = data_root or os.path.expanduser("~/.qlib/qlib_data")
        self.is_initialized = False

    def initialize_qlib(self, provider_uri: Dict[str, str] = None) -> None:
        """
        初始化Qlib，支持多频率数据源

        Args:
            provider_uri: 数据源配置字典，格式如 {"1min": "path/to/1min_data", "day": "path/to/day_data"}
        """
        if provider_uri is None:
            provider_uri = {
                "1min": f"{self.data_root}/cn_data_1min",
                "day": f"{self.data_root}/cn_data"
            }

        # 检查数据目录是否存在
        for freq, path in provider_uri.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据目录不存在: {path}")

            # 检查必要目录结构（官方格式）
            required_dirs = ["calendars", "instruments", "features"]
            for dir_name in required_dirs:
                dir_path = os.path.join(path, dir_name)
                if not os.path.exists(dir_path):
                    logger.warning(f"可能缺少必要目录: {dir_path}")
            
            # 检查关键文件（根据数据频率）
            if freq == "day":
                calendar_file = os.path.join(path, "calendars", "day.txt")
            elif freq == "1min":
                calendar_file = os.path.join(path, "calendars", "1min.txt")
            else:
                calendar_file = os.path.join(path, "calendars", f"{freq}.txt")
                
            instruments_file = os.path.join(path, "instruments", "all.txt")
            
            for file_path in [calendar_file, instruments_file]:
                if not os.path.exists(file_path):
                    logger.warning(f"关键文件不存在: {file_path}")
                else:
                    logger.debug(f"发现数据文件: {file_path}")

        try:
            qlib.init(
                provider_uri=provider_uri,
                region=REG_CN,
                express_cache_dir=os.path.join(self.data_root, "cache"),
                clear_mem_cache=False
            )
            self.is_initialized = True
            logger.info(f"Qlib初始化成功，数据源: {provider_uri}")
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            raise RuntimeError(f"Qlib初始化失败: {e}")

    def get_stock_list(self, market: str = "all", limit: int = None) -> list:
        """
        获取股票列表

        Args:
            market: 市场类型，"all", "csi300", "csi500"等
            limit: 限制返回数量

        Returns:
            股票代码列表
        """
        if not self.is_initialized:
            raise RuntimeError("请先初始化Qlib")

        try:
            if market == "all":
                instruments_file = os.path.join(self.data_root, "cn_data", "instruments", "all.txt")
            else:
                instruments_file = os.path.join(self.data_root, "cn_data", "instruments", f"{market}.txt")
            
            if not os.path.exists(instruments_file):
                raise FileNotFoundError(f"股票池文件不存在: {instruments_file}")
            
            stock_list = []
            with open(instruments_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # 格式: SH600000	2005-04-08	2023-12-31
                        parts = line.split('\t')
                        if len(parts) >= 1:
                            # 保持原始格式: SH600000, SZ000001 (Qlib内部格式)
                            symbol = parts[0].lower()  # 转为小写
                            stock_list.append(symbol)
            if limit:
                stock_list = stock_list[:limit]

            logger.info(f"获取到{len(stock_list)}只股票")
            return stock_list

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise RuntimeError(f"获取股票列表失败: {e}")

    def load_data(self,
                  instruments: list,
                  start_time: str,
                  end_time: str,
                  freq: str = "day",
                  fields: list = None) -> pd.DataFrame:
        """
        加载股票数据

        Args:
            instruments: 股票代码列表
            start_time: 开始时间，格式"2020-01-01"
            end_time: 结束时间，格式"2023-12-31"
            freq: 数据频率，"day"或"1min"
            fields: 字段列表，如["$open", "$high", "$low", "$close", "$volume"]

        Returns:
            多索引DataFrame，索引为(datetime, instrument)
        """
        if not self.is_initialized:
            raise RuntimeError("请先初始化Qlib")

        if fields is None:
            fields = ["$open", "$high", "$low", "$close", "$volume", "$change", "$factor"]

        try:
            data = D.features(
                instruments=instruments,
                fields=fields,
                start_time=start_time,
                end_time=end_time,
                freq=freq
            )

            logger.info(f"加载数据成功: {data.shape}, 时间范围: {start_time} - {end_time}")
            
            # 进行数据质量检查
            self._validate_loaded_data(data, instruments, fields)
            
            return data

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise RuntimeError(f"加载数据失败: {e}")

    def create_dataset(self,
                      instruments: list,
                      start_time: str,
                      end_time: str,
                      freq: str = "day",
                      fields: list = None,
                      feature_config: dict = None) -> DatasetH:
        """
        创建Qlib Dataset用于训练

        Args:
            instruments: 股票代码列表
            start_time: 开始时间
            end_time: 结束时间
            freq: 数据频率
            fields: 基础字段
            feature_config: 特征工程配置

        Returns:
            DatasetH对象
        """
        if not self.is_initialized:
            raise RuntimeError("请先初始化Qlib")

        if fields is None:
            fields = ["$open", "$high", "$low", "$close", "$volume"]

        # 默认特征工程配置
        if feature_config is None:
            feature_config = {
                "kbar": {},
                "price": {
                    "windows": [5, 10, 20, 30, 60],
                    "feature": ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
                },
                "volume": {
                    "windows": [5, 10, 20, 30, 60],
                    "feature": ["VSTD", "VWAP", "VEMA", "VMA"]
                }
            }

        try:
            # 构建特征表达式
            feature_list = self._build_features(fields, feature_config)

            handler = {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start_time,
                    "end_time": end_time,
                    "fit_start_time": start_time,
                    "fit_end_time": end_time,
                    "instruments": instruments,
                    "infer_processors": [
                        {
                            "class": "RobustZScoreNorm",
                            "kwargs": {"fields_group": "feature", "clip_outlier": True}
                        },
                        {
                            "class": "Fillna",
                            "kwargs": {"fields_group": "feature"}
                        }
                    ],
                    "learn_processors": [
                        {
                            "class": "DropnaLabel"
                        },
                        {
                            "class": "CSRankNorm",
                            "kwargs": {"fields_group": "label"}
                        }
                    ],
                    "label": ["Ref($close, -2) / Ref($close, -1) - 1"]
                }
            }

            dataset = DatasetH(handler)
            logger.info("Dataset创建成功")
            return dataset

        except Exception as e:
            logger.error(f"创建Dataset失败: {e}")
            raise RuntimeError(f"创建Dataset失败: {e}")

    def _build_features(self, fields: list, config: dict) -> list:
        """构建特征表达式列表"""
        features = []

        # 基础K线特征
        for field in fields:
            features.append(field)

        # 价格特征
        if "price" in config:
            price_config = config["price"]
            windows = price_config.get("windows", [5, 10, 20])
            price_features = price_config.get("feature", ["KMID", "KLEN"])

            for window in windows:
                for feat in price_features:
                    if feat == "KMID":
                        features.append(f"($close - $open) / $open")
                    elif feat == "KLEN":
                        features.append(f"($high - $low) / $open")
                    elif feat == "ROC":
                        features.append(f"Ref($close, -{window}) / $close - 1")
                    elif feat == "MA":
                        features.append(f"Mean($close, {window})")

        # 成交量特征
        if "volume" in config:
            volume_config = config["volume"]
            windows = volume_config.get("windows", [5, 10, 20])
            volume_features = volume_config.get("feature", ["VSTD", "VWAP"])

            for window in windows:
                for feat in volume_features:
                    if feat == "VSTD":
                        features.append(f"Std($volume, {window})")
                    elif feat == "VWAP":
                        features.append(f"Sum($volume * $close, {window}) / Sum($volume, {window})")
                    elif feat == "VMA":
                        features.append(f"Mean($volume, {window})")

        return features

    def _validate_loaded_data(self, data: pd.DataFrame, instruments: list, fields: list):
        """验证加载的数据质量"""
        if data.empty:
            raise RuntimeError("加载的数据为空")
        
        # 检查数据形状
        expected_columns = len(fields)
        actual_columns = len(data.columns)
        if actual_columns != expected_columns:
            logger.warning(f"数据列数不匹配: 期望 {expected_columns}, 实际 {actual_columns}")
        
        # 检查缺失值
        missing_stats = data.isnull().sum()
        total_missing = missing_stats.sum()
        
        if total_missing > 0:
            total_points = len(data)
            missing_ratio = total_missing / (total_points * len(data.columns))
            
            logger.warning(f"数据加载时发现缺失值:")
            logger.warning(f"总缺失点数: {total_missing} / {total_points * len(data.columns)} ({missing_ratio:.2%})")
            
            # 按列统计缺失情况
            for col in data.columns:
                if missing_stats[col] > 0:
                    col_missing_ratio = missing_stats[col] / total_points
                    logger.warning(f"列 {col} 缺失: {missing_stats[col]} ({col_missing_ratio:.2%})")
            
            # 根据缺失比例决定处理策略
            if missing_ratio > 0.2:  # 缺失超过20%
                raise RuntimeError(
                    f"数据缺失过多 ({missing_ratio:.2%})，无法继续处理。\n"
                    f"建议检查：\n"
                    f"1. 数据源是否正常\n"
                    f"2. 股票代码是否正确\n"
                    f"3. 时间范围是否合理\n"
                    f"缺失统计: {dict(missing_stats[missing_stats > 0])}"
                )
            elif missing_ratio > 0.1:  # 缺失10%-20%
                logger.error(
                    f"数据缺失较多 ({missing_ratio:.2%})，将在环境中进行处理，但建议检查数据质量。"
                )
            else:
                logger.info(f"数据存在少量缺失 ({missing_ratio:.2%})，将在环境中进行处理")
        else:
            logger.info("数据加载完整，无缺失值")
        
        # 检查数据范围合理性
        for col in data.columns:
            if '$' in col and ('price' in col or 'close' in col or 'open' in col or 'high' in col or 'low' in col):
                # 价格类数据应该大于0
                negative_prices = (data[col] <= 0).sum()
                if negative_prices > 0:
                    logger.warning(f"发现 {negative_prices} 个非正价格在列 {col}")
            elif '$volume' in col:
                # 成交量应该非负
                negative_volume = (data[col] < 0).sum()
                if negative_volume > 0:
                    logger.warning(f"发现 {negative_volume} 个负成交量在列 {col}")
        
        logger.info("数据质量检查完成")

    def get_calendar(self, start_time: str, end_time: str, freq: str = "day") -> list:
        """
        获取交易日历

        Args:
            start_time: 开始时间
            end_time: 结束时间
            freq: 频率

        Returns:
            交易日列表
        """
        if not self.is_initialized:
            raise RuntimeError("请先初始化Qlib")

        try:
            calendar = D.calendar(start_time=start_time, end_time=end_time, freq=freq)
            return calendar.tolist()
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            raise RuntimeError(f"获取交易日历失败: {e}")


def split_data(data: pd.DataFrame,
               train_start: str, train_end: str,
               valid_start: str, valid_end: str,
               test_start: str, test_end: str,
               apply_scaling: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分割数据为训练集、验证集、测试集

    Args:
        data: 原始数据，多索引(datetime, instrument)
        train_start: 训练集开始时间
        train_end: 训练集结束时间
        valid_start: 验证集开始时间
        valid_end: 验证集结束时间
        test_start: 测试集开始时间
        test_end: 测试集结束时间
        apply_scaling: 是否应用标准化（防止数据泄漏）

    Returns:
        训练集、验证集、测试集
    """
    try:
        # 处理多索引DataFrame的时间切片
        if isinstance(data.index, pd.MultiIndex):
            # Qlib返回的索引是(instrument, datetime)，需要用第二级索引过滤时间
            datetime_level = data.index.get_level_values(1)  # 第二级是datetime
            
            train_data = data.loc[(datetime_level >= train_start) & (datetime_level <= train_end)]
            valid_data = data.loc[(datetime_level >= valid_start) & (datetime_level <= valid_end)]
            test_data = data.loc[(datetime_level >= test_start) & (datetime_level <= test_end)]
        else:
            # 单索引情况
            train_data = data.loc[train_start:train_end]
            valid_data = data.loc[valid_start:valid_end]
            test_data = data.loc[test_start:test_end]

        logger.info(f"数据分割完成 - 训练集: {train_data.shape}, 验证集: {valid_data.shape}, 测试集: {test_data.shape}")
        
        # 检查数据是否为空
        if train_data.empty:
            logger.warning(f"训练集为空，时间范围: {train_start} - {train_end}")
        if valid_data.empty:
            logger.warning(f"验证集为空，时间范围: {valid_start} - {valid_end}")
        if test_data.empty:
            logger.warning(f"测试集为空，时间范围: {test_start} - {test_end}")
        
        # 应用标准化处理（防止数据泄漏）
        if apply_scaling and not train_data.empty:
            logger.info("应用标准化处理以防止数据泄漏...")
            
            # 为每个数据集单独fit scaler
            scalers = {}
            scaled_datasets = {}
            
            for name, dataset in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
                if dataset.empty:
                    scaled_datasets[name] = dataset
                    continue
                
                # 为当前数据集创建scaler
                scaler = StandardScaler()
                
                # 只对数值型列进行标准化
                numeric_columns = dataset.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    logger.warning(f"{name}数据集没有数值型列，跳过标准化")
                    scaled_datasets[name] = dataset
                    continue
                
                # 复制数据避免修改原始数据
                scaled_data = dataset.copy()
                
                # fit并transform当前数据集
                scaled_values = scaler.fit_transform(dataset[numeric_columns])
                scaled_data[numeric_columns] = scaled_values
                
                scalers[name] = scaler
                scaled_datasets[name] = scaled_data
                
                logger.info(f"{name}数据集标准化完成 - 特征数: {len(numeric_columns)}")
            
            train_data = scaled_datasets['train']
            valid_data = scaled_datasets['valid'] 
            test_data = scaled_datasets['test']
            
            logger.info("所有数据集标准化处理完成")
            
        return train_data, valid_data, test_data

    except Exception as e:
        logger.error(f"数据分割失败: {e}")
        raise RuntimeError(f"数据分割失败: {e}")


if __name__ == "__main__":
    # 示例用法
    loader = QlibDataLoader()

    try:
        # 初始化Qlib
        loader.initialize_qlib()

        # 获取股票列表
        stocks = loader.get_stock_list(market="csi300", limit=50)
        print(f"获取到{len(stocks)}只股票")

        # 加载数据
        data = loader.load_data(
            instruments=stocks[:10],
            start_time="2020-01-01",
            end_time="2023-12-31",
            freq="day"
        )
        print(f"数据形状: {data.shape}")

        # 获取交易日历
        calendar = loader.get_calendar("2023-01-01", "2023-12-31")
        print(f"2023年交易日数量: {len(calendar)}")

    except Exception as e:
        print(f"示例运行失败: {e}")