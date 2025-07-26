"""
Qlib数据提供器 - 扩展qlib数据接口
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import qlib
from qlib.data import D
import logging

logger = logging.getLogger(__name__)


class QlibDataProvider:
    """Qlib数据提供器扩展"""
    
    def __init__(self, region: str = "cn"):
        self.region = region
        
    def get_factor_data(self, 
                       instruments: List[str],
                       factors: List[str],
                       start_time: str,
                       end_time: str) -> pd.DataFrame:
        """
        获取因子数据
        
        Args:
            instruments: 股票代码
            factors: 因子表达式列表
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            因子数据
        """
        try:
            factor_data = D.features(
                instruments=instruments,
                fields=factors,
                start_time=start_time,
                end_time=end_time
            )
            
            return factor_data.dropna(how='all')
            
        except Exception as e:
            logger.error(f"获取因子数据失败: {e}")
            raise
    
    def get_fundamental_data(self, 
                           instruments: List[str],
                           start_time: str,
                           end_time: str) -> pd.DataFrame:
        """
        获取基本面数据
        
        Args:
            instruments: 股票代码
            start_time: 开始时间  
            end_time: 结束时间
            
        Returns:
            基本面数据
        """
        fundamental_fields = [
            "$close",
            "$volume", 
            "$market_cap",
            "$pe_ratio",
            "$pb_ratio"
        ]
        
        try:
            return D.features(
                instruments=instruments,
                fields=fundamental_fields,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            logger.error(f"获取基本面数据失败: {e}")
            raise
    
    def calculate_returns(self, price_data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        计算收益率
        
        Args:
            price_data: 价格数据
            periods: 计算周期
            
        Returns:
            收益率数据
        """
        return price_data.pct_change(periods=periods)
    
    def calculate_volatility(self, 
                           returns: pd.DataFrame, 
                           window: int = 20) -> pd.DataFrame:
        """
        计算滚动波动率
        
        Args:
            returns: 收益率数据
            window: 滚动窗口
            
        Returns:
            波动率数据
        """
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def filter_low_volatility_stocks(self,
                                   price_data: pd.DataFrame,
                                   threshold: float = 0.2,
                                   window: int = 60) -> List[str]:
        """
        筛选低波动股票
        
        Args:
            price_data: 价格数据
            threshold: 波动率阈值
            window: 计算窗口
            
        Returns:
            低波动股票列表
        """
        returns = self.calculate_returns(price_data)
        volatility = self.calculate_volatility(returns, window)
        
        # 计算平均波动率
        avg_vol = volatility.mean()
        
        # 筛选低波动股票
        low_vol_stocks = avg_vol[avg_vol < threshold].index.tolist()
        
        return low_vol_stocks
    
    def get_industry_data(self, 
                         instruments: List[str],
                         start_time: str,
                         end_time: str) -> pd.DataFrame:
        """
        获取行业数据
        
        Args:
            instruments: 股票代码
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            行业数据
        """
        try:
            # 尝试获取行业分类数据
            industry_fields = ["$industry", "$sector"]
            return D.features(
                instruments=instruments,
                fields=industry_fields,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as e:
            logger.error(f"获取行业数据失败: {e}")
            raise
    
    def validate_instruments(self, instruments: List[str]) -> List[str]:
        """
        验证股票代码有效性
        
        Args:
            instruments: 股票代码列表
            
        Returns:
            有效的股票代码列表
        """
        valid_instruments = []
        
        for instrument in instruments:
            try:
                # 尝试获取少量数据验证代码有效性
                test_data = D.features(
                    instruments=[instrument],
                    fields=["$close"],
                    start_time="2023-01-01",
                    end_time="2023-01-10"
                )
                
                if not test_data.empty:
                    valid_instruments.append(instrument)
                    
            except Exception:
                logger.warning(f"股票代码无效: {instrument}")
                continue
                
        return valid_instruments
    
    def get_benchmark_data(self, 
                          benchmark: str = "SH000300",
                          start_time: str = "2020-01-01", 
                          end_time: str = "2023-12-31") -> pd.DataFrame:
        """
        获取基准数据
        
        Args:
            benchmark: 基准指数代码
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            基准数据
        """
        try:
            benchmark_data = D.features(
                instruments=[benchmark],
                fields=["$close"],
                start_time=start_time,
                end_time=end_time
            )
            
            return benchmark_data.dropna()
            
        except Exception as e:
            logger.error(f"获取基准数据失败: {e}")
            raise