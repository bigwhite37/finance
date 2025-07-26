"""
因子引擎 - 统一的因子计算和管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from .alpha_factors import AlphaFactors
from .risk_factors import RiskFactors

logger = logging.getLogger(__name__)


class FactorEngine:
    """因子工程引擎"""

    def __init__(self, config: Dict):
        """
        初始化因子引擎

        Args:
            config: 因子配置参数
        """
        self.config = config
        self.alpha_factors_calculator = AlphaFactors(config)
        self.risk_factors_calculator = RiskFactors(config)

        # 因子缓存
        self._factor_cache = {}

        # 默认因子列表
        self.default_factors = [
            "price_reversal", "bollinger_position", "williams_r", # 反转和震荡因子
            "volume_ratio", 
            "volatility_60d", "rsi_14d",
            "ma_ratio_20d", "turnover_rate"
        ]

    def calculate_all_factors(self,
                            price_data: pd.DataFrame,
                            volume_data: Optional[pd.DataFrame] = None,
                            factors: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            price_data: 价格数据
            volume_data: 成交量数据
            factors: 指定计算的因子列表

        Returns:
            因子数据DataFrame
        """
        if factors is None:
            factors = self.default_factors

        # 计算Alpha因子
        alpha_factor_names = [f for f in factors if hasattr(self.alpha_factors_calculator, f"calculate_{f}")]
        alpha_data = self.alpha_factors_calculator.calculate_factors(price_data, volume_data, alpha_factor_names)

        # 计算风险因子
        risk_factor_names = [f for f in factors if hasattr(self.risk_factors_calculator, f"calculate_{f}")]
        risk_data = self.risk_factors_calculator.calculate_factors(price_data, volume_data, risk_factor_names)

        # 合并因子数据
        all_factors = pd.concat([alpha_data, risk_data], axis=1)
        
        # Reshape to (datetime, instrument) MultiIndex
        all_factors_stacked = all_factors.stack()
        all_factors_stacked.index.names = ['datetime', 'instrument']

        return self._post_process_factors(all_factors_stacked)

    def _post_process_factors(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """因子后处理"""
        # 去除无效值
        factor_data = factor_data.replace([np.inf, -np.inf], np.nan)

        # 按截面填充缺失值
        factor_data = factor_data.groupby(level='datetime').transform(lambda x: x.fillna(x.mean()))
        factor_data = factor_data.fillna(0) # Fill any remaining NaNs

        # 按截面进行标准化和去极值
        
        
        return factor_data.fillna(0)

    

    def filter_low_volatility_universe(self,
                                     price_data: pd.DataFrame,
                                     threshold: float = 0.2,
                                     window: int = 60) -> List[str]:
        """
        筛选低波动股票池

        Args:
            price_data: 价格数据
            threshold: 波动率阈值
            window: 计算窗口

        Returns:
            低波动股票列表
        """
        if len(price_data) < window:
            logger.warning(f"数据长度 {len(price_data)} 小于窗口 {window}，使用前100只股票")
            return price_data.columns[:100].tolist()

        returns = price_data.pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        avg_volatility = volatility.mean().dropna()

        low_vol_stocks = avg_volatility[avg_volatility < threshold].index.tolist()

        if not low_vol_stocks:
            logger.warning(f"阈值 {threshold} 太严格，筛选出0只股票，使用波动率最小的前100只")
            low_vol_stocks = avg_volatility.nsmallest(100).index.tolist()

        logger.info(f"筛选出 {len(low_vol_stocks)} 只低波动股票")
        return low_vol_stocks
