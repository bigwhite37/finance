"""
数据管理器 - 统一数据接口和管理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import qlib
from qlib.data import D
import logging

logger = logging.getLogger(__name__)


class DataManager:
    """基于qlib的数据管理系统"""
    
    def __init__(self, config: Dict):
        """
        初始化数据管理器
        
        Args:
            config: 数据配置参数
        """
        self.config = config
        self.provider = config.get('provider', 'yahoo')
        self.region = config.get('region', 'cn')
        self.universe = config.get('universe', 'csi300')
        
        # 初始化qlib
        self._init_qlib()
        
        # 数据缓存
        self._cache = {}
        
    def _init_qlib(self):
        """初始化qlib数据源"""
        qlib.init(
            provider_uri=self.config.get('provider_uri', '~/.qlib/qlib_data/cn_data'),
            region=self.region
        )
        logger.info(f"Qlib初始化成功: {self.provider}")
    
    def get_stock_data(self, 
                      instruments: Optional[List[str]] = None,
                      start_time: str = "2020-01-01",
                      end_time: str = "2023-12-31",
                      fields: List[str] = None) -> pd.DataFrame:
        """
        获取股票基础数据
        
        Args:
            instruments: 股票代码列表，None则使用默认股票池
            start_time: 开始时间
            end_time: 结束时间  
            fields: 数据字段
            
        Returns:
            股票数据DataFrame
        """
        if fields is None:
            fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
            
        if instruments is None:
            instruments = self._get_universe_stocks(start_time, end_time)
            
        cache_key = f"stock_data_{hash(str(instruments))}_{start_time}_{end_time}"
        
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
            
        data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_time,
            end_time=end_time
        )
        
        # 数据清洗
        data = self._clean_data(data)
        self._cache[cache_key] = data
        
        logger.info(f"获取股票数据成功: {data.shape}")
        return data.copy()
    
    def get_market_data(self, 
                       start_time: str = "2020-01-01",
                       end_time: str = "2023-12-31") -> pd.DataFrame:
        """
        获取市场数据(指数、期货等)
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            市场数据DataFrame
        """
        market_instruments = [
            "SH000300",  # 沪深300
            "SH000905",  # 中证500  
            "SH000852"   # 中证1000
        ]
        
        market_data = D.features(
            instruments=market_instruments,
            fields=["$close", "$volume"],
            start_time=start_time,
            end_time=end_time
        )
        
        return self._clean_data(market_data)
    
    def _get_universe_stocks(self, start_time: str, end_time: str) -> List[str]:
        """获取股票池"""
        if self.universe == "csi300":
            instruments = D.instruments(market="csi300")
        elif self.universe == "csi500": 
            instruments = D.instruments(market="csi500")
        else:
            instruments = D.instruments(market="all")
            
        return list(instruments) if instruments else []
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        if data.empty:
            return data
            
        # 去除异常值
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # 填充缺失值 - 使用前值填充
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def get_trading_calendar(self, 
                           start_time: str = "2020-01-01",
                           end_time: str = "2023-12-31") -> List[str]:
        """
        获取交易日历
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            交易日列表
        """
        calendar = D.calendar(start_time=start_time, end_time=end_time)
        return [str(date) for date in calendar] if calendar is not None else []
    
    def update_cache_data(self, end_date: str = None):
        """更新缓存数据"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # 清空缓存，强制重新获取数据
        self._cache.clear()
        logger.info(f"数据缓存已清空，将在下次访问时重新加载至 {end_date}")
    
    def get_data_info(self) -> Dict:
        """获取数据信息统计"""
        return {
            "provider": self.provider,
            "region": self.region, 
            "universe": self.universe,
            "cache_size": len(self._cache),
            "cache_keys": list(self._cache.keys())
        }