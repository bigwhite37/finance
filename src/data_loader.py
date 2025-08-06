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
        except (ImportError, FileNotFoundError, PermissionError, ValueError) as e:
            logger.error(f"Qlib初始化失败: {e}")
            raise RuntimeError(f"Qlib初始化失败: {e}")

    def get_stock_list(self, market: str = "all", limit: int = None, custom_pool: list = None) -> list:
        """
        获取股票列表

        Args:
            market: 市场类型，"all", "csi300", "csi500"等
            limit: 限制返回数量
            custom_pool: 自定义股票池列表，如果提供则使用此列表

        Returns:
            股票代码列表
        """
        if not self.is_initialized:
            raise RuntimeError("请先初始化Qlib")

        try:
            # 如果提供了自定义股票池，直接使用
            if custom_pool:
                stock_list = [stock.lower() if not stock.islower() else stock for stock in custom_pool]
                if limit:
                    stock_list = stock_list[:limit]
                logger.info(f"使用自定义股票池，获取到{len(stock_list)}只股票")
                return stock_list

            # 从qlib数据文件中读取股票池
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

            logger.info(f"从{market}市场获取到{len(stock_list)}只股票")
            return stock_list

        except (FileNotFoundError, PermissionError, UnicodeDecodeError, ValueError) as e:
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

        except (ValueError, KeyError, pd.errors.EmptyDataError, ImportError) as e:
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

        except (ValueError, KeyError, ImportError, TypeError) as e:
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

        # 按股票检查缺失值（正确的检测方式）
        total_missing = 0
        missing_instruments = []

        if isinstance(data.index, pd.MultiIndex) and data.index.names[0] == 'instrument':
            # qlib的多索引格式：(instrument, datetime)
            for instrument in instruments:
                try:
                    instrument_data = data.xs(instrument, level=0)
                    missing_stats = instrument_data.isnull().sum()
                    instrument_missing = missing_stats.sum()

                    if instrument_missing > 0:
                        total_missing += instrument_missing
                        missing_instruments.append(instrument)

                        # 找出缺失的日期范围
                        missing_mask = instrument_data.isnull().any(axis=1)
                        missing_dates = instrument_data[missing_mask].index

                        if len(missing_dates) > 0:
                            date_ranges = self._get_date_ranges(missing_dates)
                            logger.warning(f"股票 {instrument} 数据缺失:")
                            logger.warning(f"  缺失天数: {len(missing_dates)} 天")
                            logger.warning(f"  缺失日期: {date_ranges}")
                            logger.warning(f"  缺失特征: {list(missing_stats[missing_stats > 0].index)}")
                    else:
                        logger.debug(f"股票 {instrument} 数据完整")

                except KeyError:
                    logger.error(f"股票 {instrument} 在数据中不存在")
                    raise RuntimeError(f"股票 {instrument} 数据缺失")
        else:
            # 简单索引格式的缺失值检测
            missing_stats = data.isnull().sum()
            total_missing = missing_stats.sum()

            if total_missing > 0:
                logger.warning(f"数据存在缺失值: {dict(missing_stats[missing_stats > 0])}")

        # 总体评估
        if total_missing > 0:
            total_expected_points = len(instruments) * len(data.index.get_level_values(1).unique()) * len(fields)
            missing_ratio = total_missing / total_expected_points

            logger.info(f"数据质量评估:")
            logger.info(f"  总股票数: {len(instruments)}")
            logger.info(f"  缺失股票数: {len(missing_instruments)}")
            logger.info(f"  总缺失率: {missing_ratio:.2%}")

            # 根据缺失比例决定处理策略
            if missing_ratio > 0.1:  # 缺失超过10%
                raise RuntimeError(
                    f"数据缺失过多 ({missing_ratio:.2%})，可能影响模型质量。\n"
                    f"缺失股票: {missing_instruments}\n"
                    f"建议检查数据源或调整股票池。"
                )
            elif missing_ratio > 0.05:  # 缺失5%-10%
                logger.warning(f"数据缺失较多 ({missing_ratio:.2%})，将自动处理，但建议检查数据质量")
            else:
                logger.info(f"数据存在少量缺失 ({missing_ratio:.2%})，属于正常现象（停牌等），将自动处理")
        else:
            logger.info("所有股票数据完整，无缺失值")

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

    def _get_date_ranges(self, dates):
        """将日期列表转换为范围描述"""
        if len(dates) == 0:
            return "无"
        elif len(dates) <= 3:
            return [d.strftime('%Y-%m-%d') for d in dates]
        else:
            # 找连续日期段
            sorted_dates = sorted(dates)
            ranges = []
            start_date = sorted_dates[0]
            end_date = sorted_dates[0]

            for i in range(1, len(sorted_dates)):
                if (sorted_dates[i] - end_date).days <= 2:  # 允许周末间隔
                    end_date = sorted_dates[i]
                else:
                    if start_date == end_date:
                        ranges.append(start_date.strftime('%Y-%m-%d'))
                    else:
                        ranges.append(f"{start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')}")
                    start_date = sorted_dates[i]
                    end_date = sorted_dates[i]

            # 添加最后一个范围
            if start_date == end_date:
                ranges.append(start_date.strftime('%Y-%m-%d'))
            else:
                ranges.append(f"{start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')}")

            return ranges

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
        except (ValueError, KeyError, ImportError) as e:
            logger.error(f"获取交易日历失败: {e}")
            raise RuntimeError(f"获取交易日历失败: {e}")


def split_data(data: pd.DataFrame,
               train_start: str, train_end: str,
               valid_start: str, valid_end: str,
               test_start: str, test_end: str,
               apply_scaling: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            raise ValueError("训练集为空")
        if valid_data.empty:
            logger.warning(f"验证集为空，时间范围: {valid_start} - {valid_end}")
            raise ValueError("验证集为空")
        if test_data.empty:
            logger.warning(f"测试集为空，时间范围: {test_start} - {test_end}")
            raise ValueError("测试集为空")

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

                # 只对特定数值型列进行标准化，排除价格和成交量等不应标准化的列
                numeric_columns = dataset.select_dtypes(include=[np.number]).columns

                # 排除价格相关列（不应该标准化为负数）
                exclude_patterns = ['$close', '$open', '$high', '$low', '$volume', '$factor']
                scalable_columns = [col for col in numeric_columns
                                  if not any(pattern in col for pattern in exclude_patterns)]

                if len(scalable_columns) == 0:
                    logger.info(f"{name}数据集没有需要标准化的列（价格/成交量列已排除），跳过标准化")
                    scaled_datasets[name] = dataset
                    continue

                # 复制数据避免修改原始数据
                scaled_data = dataset.copy()

                # 只对可标准化的列进行fit并transform
                if len(scalable_columns) > 0:
                    scaled_values = scaler.fit_transform(dataset[scalable_columns])
                    scaled_data[scalable_columns] = scaled_values
                    logger.info(f"{name}数据集标准化列: {scalable_columns}")
                else:
                    logger.info(f"{name}数据集无需标准化的列")

                scalers[name] = scaler
                scaled_datasets[name] = scaled_data

                logger.info(f"{name}数据集标准化完成 - 标准化特征数: {len(scalable_columns)}, 保留原值特征数: {len(numeric_columns) - len(scalable_columns)}")

            train_data = scaled_datasets['train']
            valid_data = scaled_datasets['valid']
            test_data = scaled_datasets['test']

            logger.info("所有数据集标准化处理完成")

        return train_data, valid_data, test_data

    except (ValueError, KeyError, pd.errors.EmptyDataError, IndexError) as e:
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

    except (RuntimeError, ValueError, ImportError) as e:
        print(f"示例运行失败: {e}")