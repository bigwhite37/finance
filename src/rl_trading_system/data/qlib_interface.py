"""
Qlib数据接口实现
"""

from typing import List, Optional, Tuple
import pandas as pd
import logging
from .interfaces import DataInterface
from .data_cache import get_global_cache
from .data_quality import get_global_quality_checker

logger = logging.getLogger(__name__)


class QlibDataInterface(DataInterface):
    """Qlib数据接口实现"""
    
    def __init__(self, provider_uri: str = None):
        """
        初始化Qlib数据接口
        
        Args:
            provider_uri: Qlib数据提供者URI
        """
        super().__init__()
        self.provider_uri = provider_uri
        self._initialized = False
    
    def _initialize_qlib(self):
        """初始化Qlib"""
        if self._initialized:
            return
        
        try:
            import qlib
            from qlib.config import REG_CN
            
            # 初始化qlib
            if self.provider_uri:
                qlib.init(provider_uri=self.provider_uri, region=REG_CN)
            else:
                # 使用默认配置
                qlib.init(region=REG_CN)
            
            self._initialized = True
            logger.info("Qlib初始化成功")
            
        except ImportError:
            logger.error("Qlib未安装，请先安装qlib: pip install pyqlib")
            raise ImportError("Qlib未安装")
        except Exception as e:
            logger.error(f"Qlib初始化失败: {e}")
            raise
    
    def _convert_to_qlib_format(self, symbols: List[str]) -> List[str]:
        """
        将标准股票代码格式转换为Qlib内部格式
        
        Args:
            symbols: 标准格式股票代码列表，如 ['000001.SZ', '600000.SH']
            
        Returns:
            Qlib格式股票代码列表，如 ['sz000001', 'sh600000']
        """
        qlib_symbols = []
        for symbol in symbols:
            if '.' in symbol:
                code, exchange = symbol.split('.')
                if exchange.upper() == 'SZ':
                    qlib_symbols.append(f'sz{code.lower()}')
                elif exchange.upper() == 'SH':
                    qlib_symbols.append(f'sh{code.lower()}')
                else:
                    logger.warning(f"未知交易所代码: {exchange}，使用原始格式")
                    qlib_symbols.append(symbol.lower())
            else:
                # 如果没有交易所后缀，假设是深交所
                qlib_symbols.append(f'sz{symbol.lower()}')
        
        logger.info(f"股票代码格式转换: {symbols} -> {qlib_symbols}")
        return qlib_symbols
    
    def _convert_index_to_original_format(self, data: pd.DataFrame, 
                                        original_symbols: List[str], 
                                        qlib_symbols: List[str]) -> pd.DataFrame:
        """
        将数据框索引中的Qlib格式符号转换回原始格式
        
        Args:
            data: 包含Qlib格式符号的数据框
            original_symbols: 原始符号列表
            qlib_symbols: Qlib格式符号列表
            
        Returns:
            索引已转换为原始格式的数据框
        """
        if data.empty:
            return data
            
        # 创建符号映射字典 qlib -> original
        symbol_mapping = dict(zip(qlib_symbols, original_symbols))
        
        # 如果数据有多层索引且包含instrument层级
        if isinstance(data.index, pd.MultiIndex) and 'instrument' in data.index.names:
            # 重置索引，转换instrument列，然后重新设置索引
            data_reset = data.reset_index()
            data_reset['instrument'] = data_reset['instrument'].map(symbol_mapping).fillna(data_reset['instrument'])
            data = data_reset.set_index(['instrument', 'datetime'])
        elif 'instrument' in data.columns:
            # 如果instrument是列
            data['instrument'] = data['instrument'].map(symbol_mapping).fillna(data['instrument'])
        
        return data
    
    def get_stock_list(self, market: str = 'A') -> List[str]:
        """
        获取股票列表
        
        Args:
            market: 市场代码，'A'表示A股
            
        Returns:
            股票代码列表
        """
        cache_key = self._get_cache_key('get_stock_list', market=market)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result.tolist() if hasattr(cached_result, 'tolist') else cached_result
        
        try:
            self._initialize_qlib()
            
            import qlib
            from qlib.data import D
            
            # 获取股票列表
            instruments = D.instruments(market=market)
            
            # 缓存结果
            self._set_cache(cache_key, pd.Series(instruments))
            
            logger.info(f"成功获取{len(instruments)}只{market}股票")
            return instruments
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise
    
    def get_price_data(self, symbols: List[str], 
                      start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取价格数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            价格数据DataFrame
        """
        # 参数验证
        if not self.validate_symbols(symbols):
            raise ValueError("股票代码列表无效")
        
        if not self.validate_date_range(start_date, end_date):
            raise ValueError("日期范围无效")
        
        # 转换为Qlib内部格式：000001.SZ -> sz000001, 600000.SH -> sh600000
        formatted_symbols = self._convert_to_qlib_format(symbols)

        cache_key = self._get_cache_key('get_price_data', 
                                       symbols=tuple(formatted_symbols),
                                       start_date=start_date, 
                                       end_date=end_date)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            self._initialize_qlib()
            
            import qlib
            from qlib.data import D
            
            # 定义需要获取的字段（只获取有数据的字段）
            fields = ['$open', '$high', '$low', '$close', '$volume']
            
            # 获取数据
            data = D.features(formatted_symbols, fields, 
                            start_time=start_date, 
                            end_time=end_date)
            
            if data.empty:
                logger.error(f"无法获取股票数据: symbols={symbols}, "
                           f"start_date={start_date}, end_date={end_date}")
                raise RuntimeError(f"无法获取必要的股票数据: symbols={symbols}, "
                                 f"start_date={start_date}, end_date={end_date}")
            
            # 重命名列以符合标准格式
            column_mapping = {
                '$open': 'open',
                '$high': 'high', 
                '$low': 'low',
                '$close': 'close',
                '$volume': 'volume'
            }
            
            # 如果amount字段存在且有数据，则添加到映射中
            if '$amount' in data.columns and not data['$amount'].isnull().all():
                column_mapping['$amount'] = 'amount'
            else:
                # 如果amount字段不存在或全为空，则计算amount = close * volume
                data['$amount'] = data['$close'] * data['$volume']
                column_mapping['$amount'] = 'amount'
            data = data.rename(columns=column_mapping)
            
            # 将索引中的Qlib格式符号转换回原始格式
            data = self._convert_index_to_original_format(data, symbols, formatted_symbols)
            
            # 标准化数据格式
            data = self.standardize_dataframe(data, 'price')
            
            # 数据质量检查
            quality_report = self.check_data_quality(data, 'price')
            if quality_report['status'] == 'warning':
                logger.warning(f"数据质量问题: {quality_report['issues']}")
            
            # 缓存结果
            self._set_cache(cache_key, data)
            
            logger.info(f"成功获取价格数据: {len(data)}条记录")
            return data
            
        except Exception as e:
            logger.error(f"获取价格数据失败: {e}")
            raise
    
    def get_fundamental_data(self, symbols: List[str], 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取基本面数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            基本面数据DataFrame
        """
        # 参数验证
        if not self.validate_symbols(symbols):
            raise ValueError("股票代码列表无效")
        
        if not self.validate_date_range(start_date, end_date):
            raise ValueError("日期范围无效")
        
        # 转换为Qlib内部格式
        formatted_symbols = self._convert_to_qlib_format(symbols)

        cache_key = self._get_cache_key('get_fundamental_data',
                                       symbols=tuple(formatted_symbols),
                                       start_date=start_date,
                                       end_date=end_date)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            self._initialize_qlib()
            
            import qlib
            from qlib.data import D
            
            # 定义基本面字段
            fundamental_fields = [
                'PE',  # 市盈率
                'PB',  # 市净率
                'PS',  # 市销率
                'PCF', # 市现率
                'TOTAL_MV',  # 总市值
                'CIRC_MV',   # 流通市值
                'ROE',       # 净资产收益率
                'ROA',       # 总资产收益率
                'GROSS_PROFIT_MARGIN',  # 毛利率
                'NET_PROFIT_MARGIN'     # 净利率
            ]
            
            # 获取数据
            data = D.features(formatted_symbols, fundamental_fields,
                            start_time=start_date,
                            end_time=end_date)
            
            if data.empty:
                logger.warning(f"未获取到基本面数据: symbols={symbols}, "
                             f"start_date={start_date}, end_date={end_date}")
                return pd.DataFrame()
            
            # 将索引中的Qlib格式符号转换回原始格式
            data = self._convert_index_to_original_format(data, symbols, formatted_symbols)
            
            # 标准化数据格式
            data = self.standardize_dataframe(data, 'fundamental')
            
            # 数据质量检查
            quality_report = self.check_data_quality(data, 'fundamental')
            if quality_report['status'] == 'warning':
                logger.warning(f"基本面数据质量问题: {quality_report['issues']}")
            
            # 缓存结果
            self._set_cache(cache_key, data)
            
            logger.info(f"成功获取基本面数据: {len(data)}条记录")
            return data
            
        except Exception as e:
            logger.error(f"获取基本面数据失败: {e}")
            raise
    
    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取实时数据（如果支持）
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            实时数据DataFrame
        """
        logger.warning("Qlib接口暂不支持实时数据获取")
        return pd.DataFrame()
    
    def get_calendar(self, market: str = 'A') -> List[str]:
        """
        获取交易日历
        
        Args:
            market: 市场代码
            
        Returns:
            交易日期列表
        """
        try:
            self._initialize_qlib()
            
            import qlib
            from qlib.data import D
            
            calendar = D.calendar(market=market)
            return [date.strftime('%Y-%m-%d') for date in calendar]
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            raise
    
    def get_available_date_range(self, symbols: List[str]) -> Tuple[str, str]:
        """
        获取指定股票的可用数据范围
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            Tuple[str, str]: (开始日期, 结束日期)
        """
        try:
            self._initialize_qlib()
            
            import qlib
            from qlib.data import D
            
            # 参数验证
            if not self.validate_symbols(symbols):
                raise ValueError("股票代码列表无效")
            
            # 转换为Qlib内部格式
            formatted_symbols = self._convert_to_qlib_format(symbols)
            
            if not formatted_symbols:
                raise RuntimeError("没有有效的股票代码")
            
            # 获取可用的交易日历
            calendar = D.calendar(freq='day')
            if len(calendar) == 0:
                raise RuntimeError("无法获取交易日历")
            
            # 从交易日历获取大致范围
            earliest_possible = calendar[0].strftime('%Y-%m-%d')
            latest_possible = calendar[-1].strftime('%Y-%m-%d')
            
            # 使用第一个股票来探测实际可用范围
            test_symbol = formatted_symbols[0]
            fields = ['$close']
            
            # 尝试获取最近一年的数据来验证数据可用性
            from datetime import datetime, timedelta
            end_dt = datetime.strptime(latest_possible, '%Y-%m-%d')
            start_dt = end_dt - timedelta(days=365)
            start_test = start_dt.strftime('%Y-%m-%d')
            
            try:
                test_data = D.features([test_symbol], fields, 
                                     start_time=start_test, 
                                     end_time=latest_possible)
                if not test_data.empty:
                    # 有最近数据，使用保守的范围估计
                    # 通常A股数据从2005年左右开始比较完整
                    conservative_start = "2005-01-01"
                    return conservative_start, latest_possible
            except (ValueError, KeyError, AttributeError) as e:
                # 数据访问相关的预期异常，继续尝试其他方法
                logger.debug(f"获取最近数据失败: {e}")
            except Exception as e:
                # 其他未预期异常，记录并继续
                logger.warning(f"获取最近数据时发生未预期错误: {e}")
                # 不抛出异常，继续尝试历史数据
            
            # 如果最近一年没有数据，尝试历史数据
            # 测试过去几年是否有数据
            for year_offset in range(1, 10):
                test_year = int(latest_possible[:4]) - year_offset
                test_start = f"{test_year}-01-01"
                test_end = f"{test_year}-12-31"
                
                try:
                    test_data = D.features([test_symbol], fields,
                                         start_time=test_start,
                                         end_time=test_end)
                    if not test_data.empty:
                        # 找到有数据的年份
                        return test_start, test_end
                except (ValueError, KeyError, AttributeError) as e:
                    # 数据访问相关的预期异常，继续尝试下一年
                    logger.debug(f"测试年份 {test_year} 数据失败: {e}")
                    continue
                except Exception as e:
                    # 其他未预期异常，记录但继续尝试
                    logger.warning(f"测试年份 {test_year} 时发生未预期错误: {e}")
                    continue
            
            # 如果都没有找到数据，抛出异常
            raise RuntimeError(f"无法找到股票 {symbols} 的任何可用数据范围")
            
        except Exception as e:
            logger.error(f"获取可用数据范围失败: {e}")
            raise RuntimeError(f"无法获取股票可用数据范围: {e}") from e