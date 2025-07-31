"""
Akshare数据接口实现
"""

from typing import List, Optional
import pandas as pd
import logging
import time
from .interfaces import DataInterface
from .data_cache import get_global_cache
from .data_quality import get_global_quality_checker

logger = logging.getLogger(__name__)


class AkshareDataInterface(DataInterface):
    """Akshare数据接口实现"""
    
    def __init__(self, rate_limit: float = 0.1):
        """
        初始化Akshare数据接口
        
        Args:
            rate_limit: API调用间隔（秒），用于避免频率限制
        """
        super().__init__()
        self.rate_limit = rate_limit
        self._last_call_time = 0
    
    def _rate_limit_wait(self):
        """等待以避免API频率限制"""
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        if time_since_last_call < self.rate_limit:
            wait_time = self.rate_limit - time_since_last_call
            time.sleep(wait_time)
        
        self._last_call_time = time.time()
    
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
            import akshare as ak
            
            self._rate_limit_wait()
            
            if market == 'A':
                # 获取A股股票列表
                stock_info = ak.stock_info_a_code_name()
                
                # 转换为标准格式（添加交易所后缀）
                stock_list = []
                for _, row in stock_info.iterrows():
                    code = row['code']
                    # 根据代码判断交易所
                    if code.startswith('6'):
                        stock_list.append(f"{code}.SH")
                    else:
                        stock_list.append(f"{code}.SZ")
                
                # 缓存结果
                self._set_cache(cache_key, pd.Series(stock_list))
                
                logger.info(f"成功获取{len(stock_list)}只A股股票")
                return stock_list
            
            else:
                logger.warning(f"暂不支持市场: {market}")
                return []
                
        except ImportError:
            logger.error("Akshare未安装，请先安装akshare: pip install akshare")
            raise ImportError("Akshare未安装")
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
        
        cache_key = self._get_cache_key('get_price_data',
                                       symbols=tuple(symbols),
                                       start_date=start_date,
                                       end_date=end_date)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            import akshare as ak
            
            all_data = []
            
            for symbol in symbols:
                self._rate_limit_wait()
                
                # 转换股票代码格式（去掉交易所后缀）
                clean_symbol = symbol.split('.')[0]
                
                try:
                    # 获取历史数据
                    data = ak.stock_zh_a_hist(
                        symbol=clean_symbol,
                        period="daily",
                        start_date=start_date.replace('-', ''),
                        end_date=end_date.replace('-', ''),
                        adjust=""
                    )
                    
                    if data.empty:
                        logger.warning(f"股票{symbol}无数据")
                        continue
                    
                    # 标准化列名
                    column_mapping = {
                        '日期': 'datetime',
                        '开盘': 'open',
                        '最高': 'high',
                        '最低': 'low',
                        '收盘': 'close',
                        '成交量': 'volume',
                        '成交额': 'amount'
                    }
                    
                    # 重命名列
                    data = data.rename(columns=column_mapping)
                    
                    # 添加股票代码列
                    data['instrument'] = symbol
                    
                    # 转换日期格式
                    data['datetime'] = pd.to_datetime(data['datetime'])
                    
                    # 设置索引
                    data = data.set_index(['datetime', 'instrument'])
                    
                    all_data.append(data)
                    
                except Exception as e:
                    logger.error(f"获取股票{symbol}数据失败: {e}")
                    continue
            
            if not all_data:
                logger.warning("未获取到任何数据")
                return pd.DataFrame()
            
            # 合并所有数据
            combined_data = pd.concat(all_data)
            
            # 标准化数据格式
            combined_data = self.standardize_dataframe(combined_data, 'price')
            
            # 数据质量检查
            quality_report = self.check_data_quality(combined_data, 'price')
            if quality_report['status'] == 'warning':
                logger.warning(f"数据质量问题: {quality_report['issues']}")
            
            # 缓存结果
            self._set_cache(cache_key, combined_data)
            
            logger.info(f"成功获取价格数据: {len(combined_data)}条记录")
            return combined_data
            
        except ImportError:
            logger.error("Akshare未安装，请先安装akshare: pip install akshare")
            raise ImportError("Akshare未安装")
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
        
        cache_key = self._get_cache_key('get_fundamental_data',
                                       symbols=tuple(symbols),
                                       start_date=start_date,
                                       end_date=end_date)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            return cached_result
        
        try:
            import akshare as ak
            
            all_data = []
            
            for symbol in symbols:
                self._rate_limit_wait()
                
                # 转换股票代码格式
                clean_symbol = symbol.split('.')[0]
                
                try:
                    # 获取基本面数据（财务指标）
                    data = ak.stock_financial_analysis_indicator(symbol=clean_symbol)
                    
                    if data.empty:
                        logger.warning(f"股票{symbol}无基本面数据")
                        continue
                    
                    # 添加股票代码列
                    data['instrument'] = symbol
                    
                    # 转换日期格式
                    if '日期' in data.columns:
                        data['datetime'] = pd.to_datetime(data['日期'])
                        data = data.set_index(['datetime', 'instrument'])
                    
                    all_data.append(data)
                    
                except Exception as e:
                    logger.error(f"获取股票{symbol}基本面数据失败: {e}")
                    continue
            
            if not all_data:
                logger.warning("未获取到任何基本面数据")
                return pd.DataFrame()
            
            # 合并所有数据
            combined_data = pd.concat(all_data)
            
            # 标准化数据格式
            combined_data = self.standardize_dataframe(combined_data, 'fundamental')
            
            # 数据质量检查
            quality_report = self.check_data_quality(combined_data, 'fundamental')
            if quality_report['status'] == 'warning':
                logger.warning(f"基本面数据质量问题: {quality_report['issues']}")
            
            # 缓存结果
            self._set_cache(cache_key, combined_data)
            
            logger.info(f"成功获取基本面数据: {len(combined_data)}条记录")
            return combined_data
            
        except ImportError:
            logger.error("Akshare未安装，请先安装akshare: pip install akshare")
            raise ImportError("Akshare未安装")
        except Exception as e:
            logger.error(f"获取基本面数据失败: {e}")
            raise
    
    def get_realtime_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取实时数据
        
        Args:
            symbols: 股票代码列表
            
        Returns:
            实时数据DataFrame
        """
        try:
            import akshare as ak
            
            all_data = []
            
            for symbol in symbols:
                self._rate_limit_wait()
                
                clean_symbol = symbol.split('.')[0]
                
                try:
                    # 获取实时数据
                    data = ak.stock_zh_a_spot_em()
                    
                    # 筛选指定股票
                    stock_data = data[data['代码'] == clean_symbol]
                    
                    if stock_data.empty:
                        continue
                    
                    # 标准化格式
                    stock_data['instrument'] = symbol
                    stock_data['datetime'] = pd.Timestamp.now()
                    
                    all_data.append(stock_data)
                    
                except Exception as e:
                    logger.error(f"获取股票{symbol}实时数据失败: {e}")
                    continue
            
            if not all_data:
                return pd.DataFrame()
            
            combined_data = pd.concat(all_data)
            logger.info(f"成功获取实时数据: {len(combined_data)}条记录")
            return combined_data
            
        except ImportError:
            logger.error("Akshare未安装，请先安装akshare: pip install akshare")
            raise ImportError("Akshare未安装")
        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            raise
    
    def get_index_data(self, index_code: str, 
                      start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数数据
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            指数数据DataFrame
        """
        try:
            import akshare as ak
            
            self._rate_limit_wait()
            
            # 获取指数数据
            data = ak.stock_zh_index_daily(
                symbol=index_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )
            
            if data.empty:
                return pd.DataFrame()
            
            # 标准化格式
            data['datetime'] = pd.to_datetime(data['date'])
            data['instrument'] = index_code
            data = data.set_index(['datetime', 'instrument'])
            
            logger.info(f"成功获取指数{index_code}数据: {len(data)}条记录")
            return data
            
        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            raise