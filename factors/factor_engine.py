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

        # 增强因子列表 - 添加更多alpha因子
        self.default_factors = [
            "roe_factor",
            "volatility_20d",
            "volatility_60d",
        ]

    def calculate_all_factors(self,
                            price_data: pd.DataFrame,
                            volume_data: Optional[pd.DataFrame] = None,
                            roe_data: Optional[pd.DataFrame] = None,
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
        if alpha_factor_names:
            alpha_data = self.alpha_factors_calculator.calculate_factors(price_data, volume_data, roe_data, alpha_factor_names)
        else:
            alpha_data = pd.DataFrame()

        # 计算风险因子
        risk_factor_names = [f for f in factors if hasattr(self.risk_factors_calculator, f"calculate_{f}")]
        if risk_factor_names:
            risk_data = self.risk_factors_calculator.calculate_factors(price_data, volume_data, risk_factor_names)
        else:
            risk_data = pd.DataFrame()

        # 合并因子数据
        if not alpha_data.empty and not risk_data.empty:
            all_factors = pd.concat([alpha_data, risk_data], axis=1)
        elif not alpha_data.empty:
            all_factors = alpha_data
        elif not risk_data.empty:
            all_factors = risk_data
        else:
            logger.warning("未计算出任何因子数据")
            # 创建空的因子数据框
            all_factors = pd.DataFrame(
                index=price_data.index,
                columns=pd.MultiIndex.from_tuples([], names=['factor', 'instrument'])
            )
        
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

        # 按截面进行去极值和标准化
        def robust_scaler(x):
            # 1. Winsorization (去极值)
            p5 = np.percentile(x, 5)
            p95 = np.percentile(x, 95)
            x = np.clip(x, p5, p95)
            
            # 2. Standardization (标准化)
            mean = np.mean(x)
            std = np.std(x)
            if std > 1e-8:
                return (x - mean) / std
            return x - mean # 如果标准差为0，则只进行中心化

        processed_data = factor_data.groupby(level='datetime').transform(robust_scaler)
        
        return processed_data.fillna(0)

    

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
