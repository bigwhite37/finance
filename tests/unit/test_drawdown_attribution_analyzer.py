"""
回撤归因分析器单元测试
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

from src.rl_trading_system.risk_control.drawdown_attribution_analyzer import (
    DrawdownAttributionAnalyzer,
    AttributionResult,
    StockInfo
)


class TestDrawdownAttributionAnalyzer:
    """回撤归因分析器测试类"""
    
    @pytest.fixture
    def sample_stock_info(self):
        """样本股票信息"""
        return {
            'AAPL': StockInfo('AAPL', '苹果公司', '科技', '消费电子', 2.5e12),
            'GOOGL': StockInfo('GOOGL', '谷歌', '科技', '互联网', 1.8e12),
            'JPM': StockInfo('JPM', '摩根大通', '金融', '银行', 4.5e11),
            'JNJ': StockInfo('JNJ', '强生', '医疗', '制药', 4.2e11),
            'XOM': StockInfo('XOM', '埃克森美孚', '能源', '石油', 3.8e11)
        }
    
    @pytest.fixture
    def sample_factor_loadings(self):
        """样本因子载荷"""
        return {
            'AAPL': {'size_factor': 0.8, 'value_factor': -0.3, 'growth_factor': 0.6, 'momentum_factor': 0.4},
            'GOOGL': {'size_factor': 0.9, 'value_factor': -0.2, 'growth_factor': 0.8, 'momentum_factor': 0.3},
            'JPM': {'size_factor': 0.6, 'value_factor': 0.7, 'growth_factor': 0.2, 'momentum_factor': -0.1},
            'JNJ': {'size_factor': 0.7, 'value_factor': 0.4, 'growth_factor': 0.3, 'momentum_factor': 0.1},
            'XOM': {'size_factor': 0.5, 'value_factor': 0.8, 'growth_factor': -0.2, 'momentum_factor': -0.3}
        }
    
    @pytest.fixture
    def analyzer(self, sample_stock_info, sample_factor_loadings):
        """创建分析器实例"""
        return DrawdownAttributionAnalyzer(
            stock_info=sample_stock_info,
            factor_loadings=sample_factor_loadings,
            lookback_window=20,
            min_contribution_threshold=0.005
        )
    
    @pytest.fixture
    def sample_positions(self):
        """样本持仓"""
        return {
            'AAPL': 0.25,
            'GOOGL': 0.20,
            'JPM': 0.20,
            'JNJ': 0.20,
            'XOM': 0.15
        }
    
    @pytest.fixture
    def sample_negative_returns(self):
        """样本负收益（回撤情况）"""
        return {
            'AAPL': -0.08,  # -8%
            'GOOGL': -0.05, # -5%
            'JPM': -0.12,   # -12%
            'JNJ': -0.03,   # -3%
            'XOM': -0.15    # -15%
        }
    
    @pytest.fixture
    def sample_factor_returns(self):
        """样本因子收益"""
        return {
            'size_factor': -0.02,
            'value_factor': 0.01,
            'growth_factor': -0.03,
            'momentum_factor': -0.01
        }
    
    def test_analyzer_initialization(self, sample_stock_info, sample_factor_loadings):
        """测试分析器初始化"""
        analyzer = DrawdownAttributionAnalyzer(
            stock_info=sample_stock_info,
            factor_loadings=sample_factor_loadings
        )
        
        assert analyzer.stock_info == sample_stock_info
        assert analyzer.factor_loadings == sample_factor_loadings
        assert analyzer.lookback_window == 20
        assert analyzer.min_contribution_threshold == 0.01
        assert len(analyzer.attribution_history) == 0
    
    def test_calculate_portfolio_drawdown(self, analyzer, sample_positions, sample_negative_returns):
        """测试投资组合回撤计算"""
        drawdown = analyzer._calculate_portfolio_drawdown(sample_positions, sample_negative_returns)
        
        # 计算预期回撤
        expected_drawdown = (0.25 * -0.08 + 0.20 * -0.05 + 0.20 * -0.12 + 
                           0.20 * -0.03 + 0.15 * -0.15)
        
        assert abs(drawdown - expected_drawdown) < 1e-6
        assert drawdown < 0  # 回撤应该为负值
    
    def test_analyze_stock_contributions(self, analyzer, sample_positions, sample_negative_returns):
        """测试个股贡献分析"""
        contributions = analyzer._analyze_stock_contributions(
            sample_positions, sample_negative_returns
        )
        
        # 验证贡献计算
        expected_aapl_contrib = 0.25 * -0.08
        expected_jpm_contrib = 0.20 * -0.12
        
        assert 'AAPL' in contributions
        assert 'JPM' in contributions
        assert abs(contributions['AAPL'] - expected_aapl_contrib) < 1e-6
        assert abs(contributions['JPM'] - expected_jpm_contrib) < 1e-6
        
        # 验证所有贡献都是负值（回撤）
        for contrib in contributions.values():
            assert contrib <= 0
    
    def test_analyze_sector_contributions(self, analyzer, sample_positions, sample_negative_returns):
        """测试行业贡献分析"""
        stock_contributions = analyzer._analyze_stock_contributions(
            sample_positions, sample_negative_returns
        )
        sector_contributions = analyzer._analyze_sector_contributions(
            sample_positions, sample_negative_returns, stock_contributions
        )
        
        # 验证行业分组
        assert '科技' in sector_contributions
        assert '金融' in sector_contributions
        assert '医疗' in sector_contributions
        assert '能源' in sector_contributions
        
        # 验证科技行业贡献（AAPL + GOOGL）
        expected_tech_contrib = stock_contributions.get('AAPL', 0) + stock_contributions.get('GOOGL', 0)
        if abs(expected_tech_contrib) >= analyzer.min_contribution_threshold:
            assert abs(sector_contributions['科技'] - expected_tech_contrib) < 1e-6
    
    def test_analyze_factor_contributions(self, analyzer, sample_positions, 
                                        sample_negative_returns, sample_factor_returns):
        """测试因子贡献分析"""
        factor_contributions = analyzer._analyze_factor_contributions(
            sample_positions, sample_negative_returns, sample_factor_returns
        )
        
        # 验证因子贡献存在
        assert len(factor_contributions) > 0
        
        # 验证因子贡献计算逻辑
        # 计算投资组合在size_factor上的暴露
        size_exposure = (0.25 * 0.8 + 0.20 * 0.9 + 0.20 * 0.6 + 
                        0.20 * 0.7 + 0.15 * 0.5)
        expected_size_contrib = size_exposure * sample_factor_returns['size_factor']
        
        # 如果size因子贡献为负且超过阈值，应该包含在结果中
        if expected_size_contrib < 0 and abs(expected_size_contrib) >= analyzer.min_contribution_threshold:
            # 检查是否在某个因子类别中
            found_in_category = any('size' in key.lower() or '规模' in key 
                                  for key in factor_contributions.keys())
            assert found_in_category
    
    def test_analyze_interaction_effects(self, analyzer):
        """测试交互效应分析"""
        stock_contributions = {'AAPL': -0.02, 'GOOGL': -0.01}
        sector_contributions = {'科技': -0.03}
        factor_contributions = {'size因子': -0.005}
        
        interaction_effects = analyzer._analyze_interaction_effects(
            stock_contributions, sector_contributions, factor_contributions
        )
        
        # 验证交互效应存在
        assert isinstance(interaction_effects, dict)
        
        # 如果有交互效应，验证其合理性
        for effect_name, effect_value in interaction_effects.items():
            assert isinstance(effect_value, float)
            assert '交互' in effect_name
    
    def test_calculate_attribution_confidence(self, analyzer):
        """测试归因置信度计算"""
        stock_contributions = {'AAPL': -0.02, 'GOOGL': -0.01, 'JPM': -0.015}
        sector_contributions = {'科技': -0.03, '金融': -0.015}
        factor_contributions = {'size因子': -0.005}
        unexplained_portion = -0.005
        total_drawdown = -0.05
        
        confidence = analyzer._calculate_attribution_confidence(
            stock_contributions, sector_contributions, factor_contributions,
            unexplained_portion, total_drawdown
        )
        
        assert 0 <= confidence <= 1
        assert isinstance(confidence, float)
    
    def test_full_attribution_analysis(self, analyzer, sample_positions, 
                                     sample_negative_returns, sample_factor_returns):
        """测试完整归因分析流程"""
        result = analyzer.analyze_drawdown_attribution(
            current_positions=sample_positions,
            position_returns=sample_negative_returns,
            factor_returns=sample_factor_returns
        )
        
        # 验证结果类型
        assert isinstance(result, AttributionResult)
        
        # 验证基本属性
        assert result.total_drawdown < 0  # 应该是负值（回撤）
        assert 0 <= result.confidence_score <= 1
        assert isinstance(result.stock_contributions, dict)
        assert isinstance(result.sector_contributions, dict)
        assert isinstance(result.factor_contributions, dict)
        
        # 验证历史记录更新
        assert len(analyzer.attribution_history) == 1
        assert len(analyzer.portfolio_history) == 1
        assert len(analyzer.returns_history) == 1
    
    def test_attribution_result_to_dict(self, analyzer, sample_positions, sample_negative_returns):
        """测试归因结果转换为字典"""
        result = analyzer.analyze_drawdown_attribution(
            current_positions=sample_positions,
            position_returns=sample_negative_returns
        )
        
        result_dict = result.to_dict()
        
        # 验证字典结构
        required_keys = [
            'timestamp', 'total_drawdown', 'stock_contributions',
            'sector_contributions', 'factor_contributions',
            'interaction_effects', 'unexplained_portion', 'confidence_score'
        ]
        
        for key in required_keys:
            assert key in result_dict
        
        # 验证时间戳格式
        assert isinstance(result_dict['timestamp'], str)
        datetime.fromisoformat(result_dict['timestamp'])  # 应该能够解析
    
    def test_stock_info_to_dict(self, sample_stock_info):
        """测试股票信息转换为字典"""
        stock_info = sample_stock_info['AAPL']
        info_dict = stock_info.to_dict()
        
        expected_keys = ['symbol', 'name', 'sector', 'industry', 'market_cap']
        for key in expected_keys:
            assert key in info_dict
        
        assert info_dict['symbol'] == 'AAPL'
        assert info_dict['name'] == '苹果公司'
        assert info_dict['sector'] == '科技'
    
    def test_multiple_attribution_analyses(self, analyzer, sample_positions):
        """测试多次归因分析"""
        # 第一次分析
        returns1 = {'AAPL': -0.05, 'GOOGL': -0.03, 'JPM': -0.08, 'JNJ': -0.02, 'XOM': -0.10}
        result1 = analyzer.analyze_drawdown_attribution(sample_positions, returns1)
        
        # 第二次分析
        returns2 = {'AAPL': -0.03, 'GOOGL': -0.06, 'JPM': -0.04, 'JNJ': -0.01, 'XOM': -0.12}
        result2 = analyzer.analyze_drawdown_attribution(sample_positions, returns2)
        
        # 验证历史记录
        assert len(analyzer.attribution_history) == 2
        assert analyzer.attribution_history[0] == result1
        assert analyzer.attribution_history[1] == result2
        
        # 验证结果不同
        assert result1.total_drawdown != result2.total_drawdown
    
    def test_empty_inputs(self, analyzer):
        """测试空输入处理"""
        # 空持仓
        result = analyzer.analyze_drawdown_attribution({}, {})
        assert result.total_drawdown == 0.0
        assert len(result.stock_contributions) == 0
        
        # 不匹配的持仓和收益
        positions = {'AAPL': 0.5, 'GOOGL': 0.5}
        returns = {'MSFT': -0.05, 'TSLA': -0.03}
        result = analyzer.analyze_drawdown_attribution(positions, returns)
        assert result.total_drawdown == 0.0
    
    def test_positive_returns_handling(self, analyzer, sample_positions):
        """测试正收益处理（无回撤情况）"""
        positive_returns = {
            'AAPL': 0.05,
            'GOOGL': 0.03,
            'JPM': 0.02,
            'JNJ': 0.04,
            'XOM': 0.01
        }
        
        result = analyzer.analyze_drawdown_attribution(sample_positions, positive_returns)
        
        # 正收益情况下，总回撤应该为0（因为_calculate_portfolio_drawdown返回min(0, return)）
        assert result.total_drawdown == 0.0
        # 个股贡献应该为空（只考虑负贡献）
        assert len(result.stock_contributions) == 0
    
    def test_mixed_returns_handling(self, analyzer, sample_positions):
        """测试混合收益处理"""
        mixed_returns = {
            'AAPL': 0.02,   # 正收益
            'GOOGL': -0.05, # 负收益
            'JPM': -0.03,   # 负收益
            'JNJ': 0.01,    # 正收益
            'XOM': -0.08    # 负收益
        }
        
        result = analyzer.analyze_drawdown_attribution(sample_positions, mixed_returns)
        
        # 应该只包含负贡献的股票
        negative_contributors = ['GOOGL', 'JPM', 'XOM']
        for stock in negative_contributors:
            if abs(sample_positions[stock] * mixed_returns[stock]) >= analyzer.min_contribution_threshold:
                assert stock in result.stock_contributions
        
        # 正收益股票不应该在贡献中
        positive_contributors = ['AAPL', 'JNJ']
        for stock in positive_contributors:
            assert stock not in result.stock_contributions
    
    def test_min_contribution_threshold(self, analyzer):
        """测试最小贡献阈值过滤"""
        positions = {'AAPL': 0.01, 'GOOGL': 0.99}  # AAPL权重很小
        returns = {'AAPL': -0.10, 'GOOGL': -0.02}   # AAPL收益率大但权重小
        
        result = analyzer.analyze_drawdown_attribution(positions, returns)
        
        # AAPL的贡献 = 0.01 * -0.10 = -0.001，小于默认阈值0.005
        # 应该被归入"其他"或被过滤掉
        aapl_contrib = abs(positions['AAPL'] * returns['AAPL'])
        if aapl_contrib < analyzer.min_contribution_threshold:
            assert 'AAPL' not in result.stock_contributions or '其他' in result.stock_contributions
    
    def test_historical_data_limit(self, analyzer, sample_positions):
        """测试历史数据长度限制"""
        # 添加超过限制的历史数据
        max_history = analyzer.lookback_window * 2
        
        for i in range(max_history + 5):
            returns = {stock: -0.01 * (i + 1) for stock in sample_positions.keys()}
            analyzer.analyze_drawdown_attribution(sample_positions, returns)
        
        # 验证历史数据长度不超过限制
        assert len(analyzer.attribution_history) <= max_history
        assert len(analyzer.portfolio_history) <= max_history
        assert len(analyzer.returns_history) <= max_history
    
    @patch('plotly.graph_objects.Figure')
    def test_generate_attribution_visualization(self, mock_figure, analyzer, 
                                              sample_positions, sample_negative_returns):
        """测试归因可视化生成"""
        result = analyzer.analyze_drawdown_attribution(sample_positions, sample_negative_returns)
        
        # 模拟图表对象
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        
        figures = analyzer.generate_attribution_visualization(result)
        
        # 验证返回的图表类型
        expected_chart_types = ['waterfall', 'stock_pie', 'sector_bar', 'factor_radar', 'confidence']
        for chart_type in expected_chart_types:
            assert chart_type in figures
    
    def test_generate_attribution_report(self, analyzer, sample_positions, sample_negative_returns):
        """测试归因报告生成"""
        result = analyzer.analyze_drawdown_attribution(sample_positions, sample_negative_returns)
        
        report = analyzer.generate_attribution_report(result)
        
        # 验证报告结构
        required_sections = [
            'report_info', 'executive_summary', 'detailed_analysis',
            'risk_insights', 'recommendations'
        ]
        
        for section in required_sections:
            assert section in report
        
        # 验证执行摘要内容
        summary = report['executive_summary']
        assert 'total_drawdown_pct' in summary
        assert 'confidence_level' in summary
        assert 'main_contributors' in summary
        assert 'key_findings' in summary
    
    def test_save_report_functionality(self, analyzer, sample_positions, sample_negative_returns):
        """测试报告保存功能"""
        result = analyzer.analyze_drawdown_attribution(sample_positions, sample_negative_returns)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "test_report"
            
            report = analyzer.generate_attribution_report(result, str(report_path))
            
            # 验证文件生成
            json_file = report_path.with_suffix('.json')
            md_file = report_path.with_suffix('.md')
            
            assert json_file.exists()
            assert md_file.exists()
            
            # 验证JSON文件内容
            with open(json_file, 'r', encoding='utf-8') as f:
                saved_report = json.load(f)
            
            assert saved_report == report
            
            # 验证Markdown文件不为空
            assert md_file.stat().st_size > 0
    
    def test_get_historical_attribution_summary(self, analyzer, sample_positions):
        """测试历史归因摘要"""
        # 添加多个历史记录
        for i in range(5):
            returns = {stock: -0.02 * (i + 1) for stock in sample_positions.keys()}
            analyzer.analyze_drawdown_attribution(sample_positions, returns)
        
        summary = analyzer.get_historical_attribution_summary()
        
        # 验证摘要结构
        assert 'analysis_period' in summary
        assert 'drawdown_statistics' in summary
        assert 'confidence_statistics' in summary
        assert 'frequent_contributors' in summary
        
        # 验证统计数据
        assert summary['analysis_period']['total_analyses'] == 5
        assert 'average_drawdown' in summary['drawdown_statistics']
        assert 'average_confidence' in summary['confidence_statistics']
    
    def test_edge_cases(self, analyzer):
        """测试边界情况"""
        # 测试单一持仓
        single_position = {'AAPL': 1.0}
        single_return = {'AAPL': -0.05}
        
        result = analyzer.analyze_drawdown_attribution(single_position, single_return)
        assert result.total_drawdown == -0.05
        assert 'AAPL' in result.stock_contributions
        
        # 测试零权重
        zero_weight_positions = {'AAPL': 0.0, 'GOOGL': 1.0}
        returns = {'AAPL': -0.10, 'GOOGL': -0.02}
        
        result = analyzer.analyze_drawdown_attribution(zero_weight_positions, returns)
        assert 'AAPL' not in result.stock_contributions  # 零权重不应有贡献
        assert result.total_drawdown == -0.02  # 只有GOOGL的贡献
    
    def test_confidence_level_descriptions(self, analyzer):
        """测试置信度等级描述"""
        test_cases = [
            (0.95, "非常高"),
            (0.85, "高"),
            (0.65, "中等"),
            (0.45, "低"),
            (0.25, "非常低")
        ]
        
        for confidence, expected_level in test_cases:
            level = analyzer._get_confidence_level(confidence)
            assert level == expected_level
    
    def test_concentration_analysis(self, analyzer):
        """测试集中度分析"""
        # 高集中度情况
        high_concentration = {'AAPL': -0.08, 'GOOGL': -0.01, 'JPM': -0.01}
        concentration_analysis = analyzer._analyze_concentration(high_concentration)
        
        assert 'herfindahl_index' in concentration_analysis
        assert 'concentration_level' in concentration_analysis
        assert concentration_analysis['herfindahl_index'] > 0.5  # 高集中度
        
        # 低集中度情况
        low_concentration = {'A': -0.02, 'B': -0.02, 'C': -0.02, 'D': -0.02, 'E': -0.02}
        concentration_analysis = analyzer._analyze_concentration(low_concentration)
        
        assert concentration_analysis['herfindahl_index'] < 0.25  # 低集中度


if __name__ == '__main__':
    pytest.main([__file__])