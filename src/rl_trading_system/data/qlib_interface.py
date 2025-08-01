"""
Qlib数据接口实现
"""

from typing import List, Optional
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
            
            # 定义需要获取的字段
            fields = ['$open', '$high', '$low', '$close', '$volume', '$amount']
            
            # 获取数据
            data = D.features(formatted_symbols, fields, 
                            start_time=start_date, 
                            end_time=end_date)
            
            if data.empty:
                logger.warning(f"未获取到数据: symbols={symbols}, "
                             f"start_date={start_date}, end_date={end_date}")
                return pd.DataFrame()
            
            # 重命名列以符合标准格式
            column_mapping = {
                '$open': 'open',
                '$high': 'high', 
                '$low': 'low',
                '$close': 'close',
                '$volume': 'volume',
                '$amount': 'amount'
            }
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