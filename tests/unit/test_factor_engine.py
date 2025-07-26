"""
因子引擎测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factors import FactorEngine


class TestFactorEngine(unittest.TestCase):
    """因子引擎测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config = {
            'default_factors': ['return_20d', 'volatility_60d', 'volume_ratio']
        }
        self.factor_engine = FactorEngine(self.config)
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        self.price_data = pd.DataFrame(
            index=dates,
            columns=['stock_001', 'stock_002', 'stock_003'],
            data=100 + np.cumsum(np.random.normal(0, 0.02, (len(dates), 3)), axis=0)
        )
        
        self.volume_data = pd.DataFrame(
            index=dates,
            columns=['stock_001', 'stock_002', 'stock_003'],
            data=np.random.exponential(1000000, (len(dates), 3))
        )
    
    def test_is_alpha_factor(self):
        """测试Alpha因子识别"""
        self.assertTrue(self.factor_engine._is_alpha_factor('return_20d'))
        self.assertTrue(self.factor_engine._is_alpha_factor('price_momentum'))
        self.assertFalse(self.factor_engine._is_alpha_factor('volatility_60d'))
    
    def test_is_risk_factor(self):
        """测试风险因子识别"""
        self.assertTrue(self.factor_engine._is_risk_factor('volatility_60d'))
        self.assertTrue(self.factor_engine._is_risk_factor('volume_ratio'))
        self.assertFalse(self.factor_engine._is_risk_factor('return_20d'))
    
    def test_filter_low_volatility_universe(self):
        """测试低波动股票筛选"""
        low_vol_stocks = self.factor_engine.filter_low_volatility_universe(
            self.price_data, threshold=0.3, window=30
        )
        
        # 验证返回股票列表
        self.assertIsInstance(low_vol_stocks, list)
        self.assertTrue(len(low_vol_stocks) <= len(self.price_data.columns))
        
        # 验证股票代码格式
        for stock in low_vol_stocks:
            self.assertIn(stock, self.price_data.columns)
    
    def test_calculate_factor_exposure(self):
        """测试因子暴露度计算"""
        # 设置随机种子以确保可重复性
        np.random.seed(42)
        
        # 创建测试因子数据
        factor_data = pd.DataFrame({
            'factor_1': np.random.normal(0, 1, 100),
            'factor_2': np.random.normal(0, 2, 100),
            'factor_3': np.random.normal(1, 0.5, 100)
        })
        
        # 计算因子暴露度
        exposure = self.factor_engine.calculate_factor_exposure(factor_data)
        
        # 验证标准化结果
        for col in exposure.columns:
            self.assertAlmostEqual(exposure[col].mean(), 0, places=1)
            # 由于Winsorize处理，标准差可能略小于1，放宽标准
            self.assertGreater(exposure[col].std(), 0.5)
            self.assertLess(exposure[col].std(), 1.5)
    
    def test_create_composite_factor(self):
        """测试复合因子创建"""
        # 创建测试因子数据
        factor_data = pd.DataFrame({
            'factor_1': np.random.normal(0, 1, 100),
            'factor_2': np.random.normal(0, 1, 100)
        })
        
        # 创建复合因子
        weights = {'factor_1': 0.6, 'factor_2': 0.4}
        composite = self.factor_engine.create_composite_factor(factor_data, weights)
        
        # 验证结果
        self.assertEqual(len(composite), len(factor_data))
        self.assertIsInstance(composite, pd.Series)
    
    def test_post_process_factors(self):
        """测试因子后处理"""
        # 创建包含异常值的数据
        data = pd.DataFrame({
            'factor_1': [1, 2, np.inf, 4, 5],
            'factor_2': [-np.inf, 2, 3, 4, np.nan],
            'factor_3': [1, 2, 100, 4, 5]  # 包含极端值
        })
        
        # 后处理
        processed = self.factor_engine._post_process_factors(data)
        
        # 验证无穷值被处理
        self.assertFalse(np.isinf(processed.values).any())
        
        # 验证缺失值被填充
        self.assertFalse(processed.isnull().any().any())
    
    def test_get_factor_info(self):
        """测试因子信息获取"""
        info = self.factor_engine.get_factor_info()
        
        # 验证信息内容
        self.assertIn('total_factors', info)
        self.assertIn('alpha_factors', info)
        self.assertIn('risk_factors', info)
        self.assertIn('cache_size', info)
        
        # 验证数量逻辑
        self.assertEqual(
            info['total_factors'], 
            info['alpha_factors'] + info['risk_factors']
        )


if __name__ == '__main__':
    unittest.main()