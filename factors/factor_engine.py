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
        self.alpha_factors = AlphaFactors(config)
        self.risk_factors = RiskFactors(config)
        
        # 因子缓存
        self._factor_cache = {}
        
        # 默认因子列表
        self.default_factors = [
            "return_20d", "return_60d", 
            "volume_ratio", "price_momentum",
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
            
        factor_results = []
        
        # 计算Alpha因子
        alpha_factors = [f for f in factors if self._is_alpha_factor(f)]
        if alpha_factors:
            alpha_data = self.alpha_factors.calculate_factors(
                price_data, volume_data, alpha_factors
            )
            factor_results.append(alpha_data)
            
        # 计算风险因子
        risk_factors = [f for f in factors if self._is_risk_factor(f)]
        if risk_factors:
            risk_data = self.risk_factors.calculate_factors(
                price_data, volume_data, risk_factors
            )
            factor_results.append(risk_data)
            
        # 合并因子数据
        if factor_results:
            all_factors = pd.concat(factor_results, axis=1)
            return self._post_process_factors(all_factors)
        else:
            return pd.DataFrame()
    
    def calculate_factor_exposure(self, 
                                 factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子暴露度
        
        Args:
            factor_data: 因子数据
            
        Returns:
            标准化后的因子暴露度
        """
        # Z-score标准化
        exposure = factor_data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        
        # 极值处理 - Winsorize
        exposure = exposure.clip(lower=exposure.quantile(0.01), 
                               upper=exposure.quantile(0.99), 
                               axis=0)
        
        return exposure.fillna(0)
    
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
        
        # 计算收益率
        returns = price_data.pct_change(fill_method=None)
        
        # 计算滚动波动率
        volatility = returns.rolling(window=window).std() * np.sqrt(252)
        
        # 计算平均波动率
        avg_volatility = volatility.mean().dropna()
        
        # 筛选低波动股票
        low_vol_stocks = avg_volatility[avg_volatility < threshold].index.tolist()
        
        # 如果筛选结果为空，使用波动率最小的前100只
        if len(low_vol_stocks) == 0:
            logger.warning(f"阈值 {threshold} 太严格，筛选出0只股票，使用波动率最小的前100只")
            low_vol_stocks = avg_volatility.nsmallest(100).index.tolist()
        
        logger.info(f"筛选出 {len(low_vol_stocks)} 只低波动股票")
        return low_vol_stocks
    
    def calculate_garch_volatility(self, 
                                 returns: pd.DataFrame,
                                 forecast_horizon: int = 5) -> pd.DataFrame:
        """
        计算GARCH预测波动率
        
        Args:
            returns: 收益率数据
            forecast_horizon: 预测天数
            
        Returns:
            GARCH波动率预测
        """
        from arch import arch_model
        garch_predictions = {}
        
        for symbol in returns.columns:
            symbol_returns = returns[symbol].dropna() * 100
            
            # GARCH(1,1)模型
            model = arch_model(
                symbol_returns, 
                vol='GARCH', 
                p=1, q=1,
                rescale=False
            )
            fitted_model = model.fit(disp='off')
            
            # 预测波动率
            forecast = fitted_model.forecast(horizon=forecast_horizon)
            garch_vol = np.sqrt(forecast.variance.iloc[-1, :].mean()) / 100
            
            garch_predictions[symbol] = garch_vol
        
        # 转换为DataFrame
        garch_df = pd.DataFrame([garch_predictions], 
                              index=[returns.index[-1]])
        return garch_df.reindex(columns=returns.columns).fillna(method='ffill')
    
    def create_composite_factor(self, 
                              factor_data: pd.DataFrame,
                              weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        创建复合因子
        
        Args:
            factor_data: 因子数据
            weights: 因子权重
            
        Returns:
            复合因子序列
        """
        if weights is None:
            weights = {col: 1/len(factor_data.columns) 
                      for col in factor_data.columns}
        
        # 标准化因子
        normalized_factors = self.calculate_factor_exposure(factor_data)
        
        # 加权合成
        composite_factor = sum(
            normalized_factors[factor] * weight 
            for factor, weight in weights.items() 
            if factor in normalized_factors.columns
        )
        
        return composite_factor
    
    def _is_alpha_factor(self, factor_name: str) -> bool:
        """判断是否为Alpha因子"""
        alpha_keywords = [
            'return', 'momentum', 'rsi', 'ma_ratio', 
            'price', 'trend', 'reversal'
        ]
        return any(keyword in factor_name.lower() for keyword in alpha_keywords)
    
    def _is_risk_factor(self, factor_name: str) -> bool:
        """判断是否为风险因子"""
        risk_keywords = [
            'volatility', 'volume', 'turnover', 'beta',
            'var', 'drawdown', 'skew'
        ]
        return any(keyword in factor_name.lower() for keyword in risk_keywords)
    
    def _post_process_factors(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """因子后处理"""
        # 去除无效值
        factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
        
        # 填充缺失值
        factor_data = factor_data.ffill().bfill().fillna(0)
        
        # 异常值处理 - 更严格的限制
        for col in factor_data.columns:
            # 使用IQR方法检测异常值
            Q1 = factor_data[col].quantile(0.25)
            Q3 = factor_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            factor_data[col] = factor_data[col].clip(lower=lower_bound, upper=upper_bound)
            
            # 最终NaN检查
            if factor_data[col].isna().any():
                factor_data[col] = factor_data[col].fillna(factor_data[col].median())
        
        # 最终数值稳定性验证
        assert not factor_data.isna().any().any(), "因子数据仍包含NaN值"
        assert not np.isinf(factor_data.values).any(), "因子数据仍包含无穷值"
        
        return factor_data
    
    def get_factor_info(self) -> Dict:
        """获取因子信息统计"""
        return {
            "total_factors": len(self.default_factors),
            "alpha_factors": len([f for f in self.default_factors if self._is_alpha_factor(f)]),
            "risk_factors": len([f for f in self.default_factors if self._is_risk_factor(f)]),
            "cache_size": len(self._factor_cache)
        }