"""
回撤归因分析器

该模块实现了回撤归因分析功能，包括：
- 个股贡献分析
- 行业贡献分析  
- 因子暴露贡献分析
- 归因结果可视化和报告生成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """归因分析结果数据类"""
    timestamp: datetime
    total_drawdown: float                      # 总回撤
    stock_contributions: Dict[str, float]      # 个股贡献
    sector_contributions: Dict[str, float]     # 行业贡献
    factor_contributions: Dict[str, float]     # 因子贡献
    interaction_effects: Dict[str, float]      # 交互效应
    unexplained_portion: float                 # 未解释部分
    confidence_score: float                    # 归因置信度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_drawdown': self.total_drawdown,
            'stock_contributions': self.stock_contributions,
            'sector_contributions': self.sector_contributions,
            'factor_contributions': self.factor_contributions,
            'interaction_effects': self.interaction_effects,
            'unexplained_portion': self.unexplained_portion,
            'confidence_score': self.confidence_score
        }


@dataclass
class StockInfo:
    """股票信息数据类"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap
        }


class DrawdownAttributionAnalyzer:
    """
    回撤归因分析器
    
    负责分析回撤的来源，包括：
    - 个股层面的贡献分析
    - 行业层面的贡献分析
    - 因子暴露的贡献分析
    - 归因结果的可视化和报告生成
    """
    
    def __init__(self, 
                 stock_info: Dict[str, StockInfo] = None,
                 factor_loadings: Dict[str, Dict[str, float]] = None,
                 lookback_window: int = 20,
                 min_contribution_threshold: float = 0.01):
        """
        初始化回撤归因分析器
        
        Args:
            stock_info: 股票基本信息字典 {symbol: StockInfo}
            factor_loadings: 因子载荷字典 {symbol: {factor_name: loading}}
            lookback_window: 回看窗口期（交易日）
            min_contribution_threshold: 最小贡献阈值，低于此值的贡献将被归入其他
        """
        self.stock_info = stock_info or {}
        self.factor_loadings = factor_loadings or {}
        self.lookback_window = lookback_window
        self.min_contribution_threshold = min_contribution_threshold
        
        # 历史数据存储
        self.attribution_history: List[AttributionResult] = []
        self.portfolio_history: List[Dict[str, float]] = []
        self.returns_history: List[Dict[str, float]] = []
        
        # 预定义的因子类别
        self.factor_categories = {
            'size': ['market_cap', 'log_market_cap', 'size_factor'],
            'value': ['pe_ratio', 'pb_ratio', 'ps_ratio', 'value_factor'],
            'growth': ['roe', 'roa', 'revenue_growth', 'growth_factor'],
            'momentum': ['momentum_1m', 'momentum_3m', 'momentum_6m', 'momentum_factor'],
            'quality': ['debt_ratio', 'current_ratio', 'quality_factor'],
            'volatility': ['volatility_20d', 'volatility_60d', 'vol_factor']
        }
        
        logger.info("回撤归因分析器初始化完成")
    
    def analyze_drawdown_attribution(self, 
                                   current_positions: Dict[str, float],
                                   position_returns: Dict[str, float],
                                   benchmark_returns: Dict[str, float] = None,
                                   factor_returns: Dict[str, float] = None) -> AttributionResult:
        """
        执行回撤归因分析
        
        Args:
            current_positions: 当前持仓权重 {symbol: weight}
            position_returns: 各资产收益率 {symbol: return}
            benchmark_returns: 基准收益率 {symbol: return}，可选
            factor_returns: 因子收益率 {factor: return}，可选
            
        Returns:
            AttributionResult: 归因分析结果
        """
        timestamp = datetime.now()
        
        # 计算总回撤
        total_drawdown = self._calculate_portfolio_drawdown(current_positions, position_returns)
        
        # 个股贡献分析
        stock_contributions = self._analyze_stock_contributions(
            current_positions, position_returns, benchmark_returns
        )
        
        # 行业贡献分析
        sector_contributions = self._analyze_sector_contributions(
            current_positions, position_returns, stock_contributions
        )
        
        # 因子贡献分析
        factor_contributions = self._analyze_factor_contributions(
            current_positions, position_returns, factor_returns
        )
        
        # 交互效应分析
        interaction_effects = self._analyze_interaction_effects(
            stock_contributions, sector_contributions, factor_contributions
        )
        
        # 计算未解释部分
        explained_portion = (sum(stock_contributions.values()) + 
                           sum(interaction_effects.values()))
        unexplained_portion = total_drawdown - explained_portion
        
        # 计算归因置信度
        confidence_score = self._calculate_attribution_confidence(
            stock_contributions, sector_contributions, factor_contributions, 
            unexplained_portion, total_drawdown
        )
        
        # 创建归因结果
        result = AttributionResult(
            timestamp=timestamp,
            total_drawdown=total_drawdown,
            stock_contributions=stock_contributions,
            sector_contributions=sector_contributions,
            factor_contributions=factor_contributions,
            interaction_effects=interaction_effects,
            unexplained_portion=unexplained_portion,
            confidence_score=confidence_score
        )
        
        # 保存历史记录
        self.attribution_history.append(result)
        self.portfolio_history.append(current_positions.copy())
        self.returns_history.append(position_returns.copy())
        
        # 限制历史数据长度
        if len(self.attribution_history) > self.lookback_window * 2:
            self.attribution_history.pop(0)
            self.portfolio_history.pop(0)
            self.returns_history.pop(0)
        
        logger.info(f"完成回撤归因分析，总回撤: {total_drawdown:.4f}, 置信度: {confidence_score:.3f}")
        
        return result
    
    def _calculate_portfolio_drawdown(self, positions: Dict[str, float], 
                                    returns: Dict[str, float]) -> float:
        """计算投资组合回撤"""
        portfolio_return = 0.0
        
        for symbol, weight in positions.items():
            if symbol in returns:
                portfolio_return += weight * returns[symbol]
        
        # 如果是负收益，则为回撤
        return min(0.0, portfolio_return)
    
    def _analyze_stock_contributions(self, 
                                   positions: Dict[str, float],
                                   returns: Dict[str, float],
                                   benchmark_returns: Dict[str, float] = None) -> Dict[str, float]:
        """分析个股对回撤的贡献"""
        contributions = {}
        
        for symbol, weight in positions.items():
            if symbol in returns:
                stock_return = returns[symbol]
                
                # 如果有基准，计算超额收益
                if benchmark_returns and symbol in benchmark_returns:
                    excess_return = stock_return - benchmark_returns[symbol]
                    contribution = weight * excess_return
                else:
                    contribution = weight * stock_return
                
                # 只考虑负贡献（回撤）
                if contribution < 0:
                    contributions[symbol] = contribution
        
        # 过滤小贡献
        filtered_contributions = {}
        other_contribution = 0.0
        
        for symbol, contrib in contributions.items():
            if abs(contrib) >= self.min_contribution_threshold:
                filtered_contributions[symbol] = contrib
            else:
                other_contribution += contrib
        
        if abs(other_contribution) >= self.min_contribution_threshold:
            filtered_contributions['其他'] = other_contribution
        
        return filtered_contributions
    
    def _analyze_sector_contributions(self, 
                                    positions: Dict[str, float],
                                    returns: Dict[str, float],
                                    stock_contributions: Dict[str, float]) -> Dict[str, float]:
        """分析行业对回撤的贡献"""
        sector_contributions = {}
        
        # 按行业汇总贡献
        for symbol, contribution in stock_contributions.items():
            if symbol == '其他':
                continue
                
            if symbol in self.stock_info:
                sector = self.stock_info[symbol].sector
            else:
                sector = '未知行业'
            
            if sector not in sector_contributions:
                sector_contributions[sector] = 0.0
            
            sector_contributions[sector] += contribution
        
        # 过滤小贡献
        filtered_contributions = {}
        other_contribution = 0.0
        
        for sector, contrib in sector_contributions.items():
            if abs(contrib) >= self.min_contribution_threshold:
                filtered_contributions[sector] = contrib
            else:
                other_contribution += contrib
        
        if abs(other_contribution) >= self.min_contribution_threshold:
            filtered_contributions['其他行业'] = other_contribution
        
        return filtered_contributions
    
    def _analyze_factor_contributions(self, 
                                    positions: Dict[str, float],
                                    returns: Dict[str, float],
                                    factor_returns: Dict[str, float] = None) -> Dict[str, float]:
        """分析因子暴露对回撤的贡献"""
        if not self.factor_loadings or not factor_returns:
            return {}
        
        factor_contributions = {}
        
        # 计算投资组合在各因子上的暴露
        portfolio_factor_exposure = {}
        
        for factor_name in factor_returns.keys():
            exposure = 0.0
            for symbol, weight in positions.items():
                if symbol in self.factor_loadings and factor_name in self.factor_loadings[symbol]:
                    exposure += weight * self.factor_loadings[symbol][factor_name]
            portfolio_factor_exposure[factor_name] = exposure
        
        # 计算因子贡献
        for factor_name, factor_return in factor_returns.items():
            if factor_name in portfolio_factor_exposure:
                contribution = portfolio_factor_exposure[factor_name] * factor_return
                
                # 只考虑负贡献
                if contribution < 0:
                    factor_contributions[factor_name] = contribution
        
        # 按因子类别汇总
        category_contributions = {}
        for category, factors in self.factor_categories.items():
            category_contrib = 0.0
            for factor in factors:
                if factor in factor_contributions:
                    category_contrib += factor_contributions[factor]
            
            if abs(category_contrib) >= self.min_contribution_threshold:
                category_contributions[f'{category}因子'] = category_contrib
        
        return category_contributions
    
    def _analyze_interaction_effects(self, 
                                   stock_contributions: Dict[str, float],
                                   sector_contributions: Dict[str, float],
                                   factor_contributions: Dict[str, float]) -> Dict[str, float]:
        """分析交互效应"""
        # 简化的交互效应分析
        # 实际应用中可以使用更复杂的模型
        
        interaction_effects = {}
        
        # 股票-行业交互效应（简化计算）
        stock_sector_interaction = 0.0
        for contrib in stock_contributions.values():
            stock_sector_interaction += contrib * 0.1  # 假设10%的交互效应
        
        if abs(stock_sector_interaction) >= self.min_contribution_threshold:
            interaction_effects['股票-行业交互'] = stock_sector_interaction
        
        # 行业-因子交互效应
        if sector_contributions and factor_contributions:
            sector_factor_interaction = 0.0
            for sector_contrib in sector_contributions.values():
                for factor_contrib in factor_contributions.values():
                    sector_factor_interaction += sector_contrib * factor_contrib * 0.05
            
            if abs(sector_factor_interaction) >= self.min_contribution_threshold:
                interaction_effects['行业-因子交互'] = sector_factor_interaction
        
        return interaction_effects
    
    def _calculate_attribution_confidence(self, 
                                        stock_contributions: Dict[str, float],
                                        sector_contributions: Dict[str, float],
                                        factor_contributions: Dict[str, float],
                                        unexplained_portion: float,
                                        total_drawdown: float) -> float:
        """计算归因置信度"""
        if abs(total_drawdown) < 1e-6:
            return 1.0
        
        # 基于解释比例计算置信度
        explained_ratio = 1.0 - abs(unexplained_portion / total_drawdown)
        explained_ratio = max(0.0, min(1.0, explained_ratio))
        
        # 基于数据完整性调整置信度
        data_completeness = 1.0
        
        # 检查股票信息完整性
        stock_info_ratio = len([s for s in stock_contributions.keys() 
                               if s in self.stock_info]) / max(1, len(stock_contributions))
        data_completeness *= stock_info_ratio
        
        # 检查因子载荷完整性
        if self.factor_loadings:
            factor_info_ratio = len([s for s in stock_contributions.keys() 
                                   if s in self.factor_loadings]) / max(1, len(stock_contributions))
            data_completeness *= factor_info_ratio
        else:
            data_completeness *= 0.8  # 缺少因子信息降低置信度
        
        # 综合置信度
        confidence = explained_ratio * data_completeness
        
        return confidence   
 
    def generate_attribution_visualization(self, 
                                         result: AttributionResult,
                                         save_path: str = None) -> Dict[str, go.Figure]:
        """
        生成归因分析可视化图表
        
        Args:
            result: 归因分析结果
            save_path: 保存路径，如果提供则保存图表
            
        Returns:
            Dict[str, go.Figure]: 包含各种图表的字典
        """
        figures = {}
        
        # 1. 回撤贡献瀑布图
        figures['waterfall'] = self._create_waterfall_chart(result)
        
        # 2. 个股贡献饼图
        figures['stock_pie'] = self._create_stock_contribution_pie(result)
        
        # 3. 行业贡献条形图
        figures['sector_bar'] = self._create_sector_contribution_bar(result)
        
        # 4. 因子贡献雷达图
        figures['factor_radar'] = self._create_factor_contribution_radar(result)
        
        # 5. 历史归因趋势图
        if len(self.attribution_history) > 1:
            figures['trend'] = self._create_attribution_trend_chart()
        
        # 6. 归因置信度仪表盘
        figures['confidence'] = self._create_confidence_gauge(result)
        
        # 保存图表
        if save_path:
            self._save_figures(figures, save_path)
        
        logger.info(f"生成了 {len(figures)} 个可视化图表")
        
        return figures
    
    def _create_waterfall_chart(self, result: AttributionResult) -> go.Figure:
        """创建回撤贡献瀑布图"""
        # 准备数据
        categories = ['总回撤']
        values = [result.total_drawdown]
        
        # 添加个股贡献（取前10个最大贡献）
        sorted_stocks = sorted(result.stock_contributions.items(), 
                              key=lambda x: x[1])[:10]
        for stock, contrib in sorted_stocks:
            categories.append(f'个股: {stock}')
            values.append(contrib)
        
        # 添加行业贡献
        for sector, contrib in result.sector_contributions.items():
            categories.append(f'行业: {sector}')
            values.append(contrib)
        
        # 添加因子贡献
        for factor, contrib in result.factor_contributions.items():
            categories.append(f'因子: {factor}')
            values.append(contrib)
        
        # 添加未解释部分
        if abs(result.unexplained_portion) > 1e-6:
            categories.append('未解释部分')
            values.append(result.unexplained_portion)
        
        # 创建瀑布图
        fig = go.Figure(go.Waterfall(
            name="回撤归因",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(categories) - 1),
            x=categories,
            textposition="outside",
            text=[f"{v:.2%}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "red"}},
            increasing={"marker": {"color": "green"}},
            totals={"marker": {"color": "blue"}}
        ))
        
        fig.update_layout(
            title="回撤归因瀑布图",
            xaxis_title="贡献来源",
            yaxis_title="贡献度",
            yaxis_tickformat=".2%",
            showlegend=False,
            height=600
        )
        
        return fig
    
    def _create_stock_contribution_pie(self, result: AttributionResult) -> go.Figure:
        """创建个股贡献饼图"""
        if not result.stock_contributions:
            return go.Figure().add_annotation(
                text="无个股贡献数据", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # 取绝对值最大的前10个
        sorted_contributions = sorted(result.stock_contributions.items(),
                                    key=lambda x: abs(x[1]), reverse=True)[:10]
        
        labels = [item[0] for item in sorted_contributions]
        values = [abs(item[1]) for item in sorted_contributions]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='<b>%{label}</b><br>贡献度: %{value:.4f}<br>占比: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="个股回撤贡献分布",
            height=500
        )
        
        return fig
    
    def _create_sector_contribution_bar(self, result: AttributionResult) -> go.Figure:
        """创建行业贡献条形图"""
        if not result.sector_contributions:
            return go.Figure().add_annotation(
                text="无行业贡献数据", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        sectors = list(result.sector_contributions.keys())
        contributions = list(result.sector_contributions.values())
        
        # 按贡献大小排序
        sorted_data = sorted(zip(sectors, contributions), key=lambda x: x[1])
        sectors, contributions = zip(*sorted_data)
        
        colors = ['red' if c < 0 else 'green' for c in contributions]
        
        fig = go.Figure(data=[go.Bar(
            x=contributions,
            y=sectors,
            orientation='h',
            marker_color=colors,
            text=[f'{c:.2%}' for c in contributions],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>贡献度: %{x:.4f}<extra></extra>'
        )])
        
        fig.update_layout(
            title="行业回撤贡献分析",
            xaxis_title="贡献度",
            yaxis_title="行业",
            xaxis_tickformat=".2%",
            height=max(400, len(sectors) * 30)
        )
        
        return fig
    
    def _create_factor_contribution_radar(self, result: AttributionResult) -> go.Figure:
        """创建因子贡献雷达图"""
        if not result.factor_contributions:
            return go.Figure().add_annotation(
                text="无因子贡献数据", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        factors = list(result.factor_contributions.keys())
        contributions = [abs(c) for c in result.factor_contributions.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=contributions,
            theta=factors,
            fill='toself',
            name='因子贡献',
            hovertemplate='<b>%{theta}</b><br>贡献度: %{r:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(contributions) * 1.1] if contributions else [0, 1]
                )),
            title="因子回撤贡献雷达图",
            height=500
        )
        
        return fig
    
    def _create_attribution_trend_chart(self) -> go.Figure:
        """创建归因趋势图"""
        if len(self.attribution_history) < 2:
            return go.Figure().add_annotation(
                text="历史数据不足", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        timestamps = [r.timestamp for r in self.attribution_history]
        total_drawdowns = [r.total_drawdown for r in self.attribution_history]
        confidence_scores = [r.confidence_score for r in self.attribution_history]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('回撤趋势', '归因置信度趋势'),
            vertical_spacing=0.1
        )
        
        # 回撤趋势
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=total_drawdowns,
                mode='lines+markers',
                name='总回撤',
                line=dict(color='red'),
                hovertemplate='时间: %{x}<br>回撤: %{y:.2%}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 置信度趋势
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidence_scores,
                mode='lines+markers',
                name='归因置信度',
                line=dict(color='blue'),
                hovertemplate='时间: %{x}<br>置信度: %{y:.2%}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="时间", row=2, col=1)
        fig.update_yaxes(title_text="回撤", tickformat=".2%", row=1, col=1)
        fig.update_yaxes(title_text="置信度", tickformat=".2%", row=2, col=1)
        
        fig.update_layout(
            title="回撤归因历史趋势",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _create_confidence_gauge(self, result: AttributionResult) -> go.Figure:
        """创建归因置信度仪表盘"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result.confidence_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "归因置信度"},
            delta={'reference': 0.8, 'position': "top"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'size': 16}
        )
        
        return fig
    
    def _save_figures(self, figures: Dict[str, go.Figure], save_path: str):
        """保存图表到文件"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            # 保存为HTML
            html_path = save_dir / f"{name}.html"
            fig.write_html(str(html_path))
            
            # 保存为PNG（需要安装kaleido）
            try:
                png_path = save_dir / f"{name}.png"
                fig.write_image(str(png_path), width=1200, height=800)
            except Exception as e:
                logger.warning(f"无法保存PNG图片 {name}: {e}")
        
        logger.info(f"图表已保存到 {save_path}")
    
    def generate_attribution_report(self, 
                                  result: AttributionResult,
                                  save_path: str = None) -> Dict[str, Any]:
        """
        生成归因分析报告
        
        Args:
            result: 归因分析结果
            save_path: 报告保存路径
            
        Returns:
            Dict[str, Any]: 报告内容
        """
        report = {
            'report_info': {
                'title': '回撤归因分析报告',
                'timestamp': result.timestamp.isoformat(),
                'total_drawdown': result.total_drawdown,
                'confidence_score': result.confidence_score
            },
            'executive_summary': self._generate_executive_summary(result),
            'detailed_analysis': {
                'stock_analysis': self._analyze_stock_contributions_detail(result),
                'sector_analysis': self._analyze_sector_contributions_detail(result),
                'factor_analysis': self._analyze_factor_contributions_detail(result)
            },
            'risk_insights': self._generate_risk_insights(result),
            'recommendations': self._generate_recommendations(result)
        }
        
        # 保存报告
        if save_path:
            self._save_report(report, save_path)
        
        logger.info("归因分析报告生成完成")
        
        return report
    
    def _generate_executive_summary(self, result: AttributionResult) -> Dict[str, Any]:
        """生成执行摘要"""
        # 找出最大贡献者
        max_stock_contrib = max(result.stock_contributions.items(), 
                               key=lambda x: abs(x[1])) if result.stock_contributions else None
        max_sector_contrib = max(result.sector_contributions.items(), 
                                key=lambda x: abs(x[1])) if result.sector_contributions else None
        max_factor_contrib = max(result.factor_contributions.items(), 
                                key=lambda x: abs(x[1])) if result.factor_contributions else None
        
        summary = {
            'total_drawdown_pct': f"{result.total_drawdown:.2%}",
            'confidence_level': self._get_confidence_level(result.confidence_score),
            'main_contributors': {
                'top_stock': {
                    'name': max_stock_contrib[0] if max_stock_contrib else '无',
                    'contribution': f"{max_stock_contrib[1]:.2%}" if max_stock_contrib else '0.00%'
                },
                'top_sector': {
                    'name': max_sector_contrib[0] if max_sector_contrib else '无',
                    'contribution': f"{max_sector_contrib[1]:.2%}" if max_sector_contrib else '0.00%'
                },
                'top_factor': {
                    'name': max_factor_contrib[0] if max_factor_contrib else '无',
                    'contribution': f"{max_factor_contrib[1]:.2%}" if max_factor_contrib else '0.00%'
                }
            },
            'unexplained_portion_pct': f"{result.unexplained_portion:.2%}",
            'key_findings': self._generate_key_findings(result)
        }
        
        return summary
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """获取置信度等级描述"""
        if confidence_score >= 0.9:
            return "非常高"
        elif confidence_score >= 0.7:
            return "高"
        elif confidence_score >= 0.5:
            return "中等"
        elif confidence_score >= 0.3:
            return "低"
        else:
            return "非常低"
    
    def _generate_key_findings(self, result: AttributionResult) -> List[str]:
        """生成关键发现"""
        findings = []
        
        # 分析总回撤水平
        if abs(result.total_drawdown) > 0.05:
            findings.append(f"投资组合出现显著回撤 {result.total_drawdown:.2%}")
        
        # 分析个股集中度
        if result.stock_contributions:
            max_stock_contrib = max(result.stock_contributions.values(), key=abs)
            if abs(max_stock_contrib) > abs(result.total_drawdown) * 0.3:
                findings.append("个股集中度风险较高，单一股票贡献超过总回撤的30%")
        
        # 分析行业集中度
        if result.sector_contributions:
            max_sector_contrib = max(result.sector_contributions.values(), key=abs)
            if abs(max_sector_contrib) > abs(result.total_drawdown) * 0.5:
                findings.append("行业集中度风险较高，单一行业贡献超过总回撤的50%")
        
        # 分析未解释部分
        if abs(result.unexplained_portion) > abs(result.total_drawdown) * 0.3:
            findings.append("存在较大未解释部分，可能存在模型风险或数据缺失")
        
        # 分析置信度
        if result.confidence_score < 0.5:
            findings.append("归因分析置信度较低，建议完善数据或模型")
        
        return findings
    
    def _analyze_stock_contributions_detail(self, result: AttributionResult) -> Dict[str, Any]:
        """详细分析个股贡献"""
        if not result.stock_contributions:
            return {'message': '无个股贡献数据'}
        
        sorted_contribs = sorted(result.stock_contributions.items(), 
                                key=lambda x: x[1])
        
        analysis = {
            'total_stocks': len(result.stock_contributions),
            'worst_performer': {
                'symbol': sorted_contribs[0][0],
                'contribution': f"{sorted_contribs[0][1]:.2%}"
            },
            'best_performer': {
                'symbol': sorted_contribs[-1][0],
                'contribution': f"{sorted_contribs[-1][1]:.2%}"
            },
            'concentration_analysis': self._analyze_concentration(result.stock_contributions),
            'detailed_contributions': [
                {
                    'symbol': symbol,
                    'contribution_pct': f"{contrib:.2%}",
                    'stock_info': self.stock_info.get(symbol, {}).to_dict() if symbol in self.stock_info else {}
                }
                for symbol, contrib in sorted_contribs
            ]
        }
        
        return analysis
    
    def _analyze_sector_contributions_detail(self, result: AttributionResult) -> Dict[str, Any]:
        """详细分析行业贡献"""
        if not result.sector_contributions:
            return {'message': '无行业贡献数据'}
        
        sorted_contribs = sorted(result.sector_contributions.items(), 
                                key=lambda x: x[1])
        
        analysis = {
            'total_sectors': len(result.sector_contributions),
            'worst_sector': {
                'name': sorted_contribs[0][0],
                'contribution': f"{sorted_contribs[0][1]:.2%}"
            },
            'best_sector': {
                'name': sorted_contribs[-1][0],
                'contribution': f"{sorted_contribs[-1][1]:.2%}"
            },
            'sector_diversification': self._calculate_sector_diversification(result.sector_contributions),
            'detailed_contributions': [
                {
                    'sector': sector,
                    'contribution_pct': f"{contrib:.2%}"
                }
                for sector, contrib in sorted_contribs
            ]
        }
        
        return analysis
    
    def _analyze_factor_contributions_detail(self, result: AttributionResult) -> Dict[str, Any]:
        """详细分析因子贡献"""
        if not result.factor_contributions:
            return {'message': '无因子贡献数据'}
        
        sorted_contribs = sorted(result.factor_contributions.items(), 
                                key=lambda x: x[1])
        
        analysis = {
            'total_factors': len(result.factor_contributions),
            'worst_factor': {
                'name': sorted_contribs[0][0],
                'contribution': f"{sorted_contribs[0][1]:.2%}"
            },
            'best_factor': {
                'name': sorted_contribs[-1][0],
                'contribution': f"{sorted_contribs[-1][1]:.2%}"
            },
            'factor_balance': self._analyze_factor_balance(result.factor_contributions),
            'detailed_contributions': [
                {
                    'factor': factor,
                    'contribution_pct': f"{contrib:.2%}"
                }
                for factor, contrib in sorted_contribs
            ]
        }
        
        return analysis
    
    def _analyze_concentration(self, contributions: Dict[str, float]) -> Dict[str, Any]:
        """分析集中度"""
        values = list(contributions.values())
        abs_values = [abs(v) for v in values]
        
        # 计算Herfindahl指数
        total_abs = sum(abs_values)
        if total_abs > 0:
            normalized_values = [v / total_abs for v in abs_values]
            herfindahl_index = sum(v ** 2 for v in normalized_values)
        else:
            herfindahl_index = 0
        
        # 计算前N大贡献占比
        sorted_abs_values = sorted(abs_values, reverse=True)
        top_3_ratio = sum(sorted_abs_values[:3]) / total_abs if total_abs > 0 else 0
        top_5_ratio = sum(sorted_abs_values[:5]) / total_abs if total_abs > 0 else 0
        
        return {
            'herfindahl_index': herfindahl_index,
            'top_3_contribution_ratio': f"{top_3_ratio:.2%}",
            'top_5_contribution_ratio': f"{top_5_ratio:.2%}",
            'concentration_level': self._get_concentration_level(herfindahl_index)
        }
    
    def _get_concentration_level(self, herfindahl_index: float) -> str:
        """获取集中度等级"""
        if herfindahl_index > 0.5:
            return "高度集中"
        elif herfindahl_index > 0.25:
            return "中度集中"
        elif herfindahl_index > 0.1:
            return "轻度集中"
        else:
            return "高度分散"
    
    def _calculate_sector_diversification(self, sector_contributions: Dict[str, float]) -> Dict[str, Any]:
        """计算行业多样化程度"""
        num_sectors = len(sector_contributions)
        
        # 理想情况下的均匀分布
        ideal_weight = 1.0 / num_sectors if num_sectors > 0 else 0
        
        # 计算与均匀分布的偏离度
        total_abs = sum(abs(v) for v in sector_contributions.values())
        if total_abs > 0:
            actual_weights = [abs(v) / total_abs for v in sector_contributions.values()]
            deviation = sum(abs(w - ideal_weight) for w in actual_weights)
            diversification_score = 1 - (deviation / 2)  # 标准化到0-1
        else:
            diversification_score = 0
        
        return {
            'num_sectors': num_sectors,
            'diversification_score': diversification_score,
            'diversification_level': self._get_diversification_level(diversification_score)
        }
    
    def _get_diversification_level(self, score: float) -> str:
        """获取多样化等级"""
        if score > 0.8:
            return "高度多样化"
        elif score > 0.6:
            return "中度多样化"
        elif score > 0.4:
            return "轻度多样化"
        else:
            return "集中化"
    
    def _analyze_factor_balance(self, factor_contributions: Dict[str, float]) -> Dict[str, Any]:
        """分析因子平衡性"""
        if not factor_contributions:
            return {}
        
        # 按因子类别分组
        category_contribs = {}
        for factor, contrib in factor_contributions.items():
            # 简化的因子分类
            if 'size' in factor.lower() or '规模' in factor:
                category = '规模因子'
            elif 'value' in factor.lower() or '价值' in factor:
                category = '价值因子'
            elif 'growth' in factor.lower() or '成长' in factor:
                category = '成长因子'
            elif 'momentum' in factor.lower() or '动量' in factor:
                category = '动量因子'
            else:
                category = '其他因子'
            
            if category not in category_contribs:
                category_contribs[category] = 0
            category_contribs[category] += contrib
        
        return {
            'factor_categories': len(category_contribs),
            'category_contributions': {
                cat: f"{contrib:.2%}" 
                for cat, contrib in category_contribs.items()
            },
            'balance_score': self._calculate_balance_score(category_contribs)
        }
    
    def _calculate_balance_score(self, category_contribs: Dict[str, float]) -> float:
        """计算因子平衡得分"""
        if not category_contribs:
            return 0
        
        values = list(category_contribs.values())
        abs_values = [abs(v) for v in values]
        
        if sum(abs_values) == 0:
            return 1.0
        
        # 计算标准差，标准差越小越平衡
        mean_abs = sum(abs_values) / len(abs_values)
        variance = sum((v - mean_abs) ** 2 for v in abs_values) / len(abs_values)
        std_dev = variance ** 0.5
        
        # 标准化到0-1，1表示完全平衡
        balance_score = max(0, 1 - std_dev / mean_abs) if mean_abs > 0 else 1
        
        return balance_score
    
    def _generate_risk_insights(self, result: AttributionResult) -> List[str]:
        """生成风险洞察"""
        insights = []
        
        # 基于个股贡献的洞察
        if result.stock_contributions:
            max_stock_contrib = max(result.stock_contributions.items(), key=lambda x: abs(x[1]))
            if abs(max_stock_contrib[1]) > 0.02:  # 2%以上贡献
                insights.append(f"个股 {max_stock_contrib[0]} 对回撤贡献最大，建议关注其基本面变化")
        
        # 基于行业贡献的洞察
        if result.sector_contributions:
            max_sector_contrib = max(result.sector_contributions.items(), key=lambda x: abs(x[1]))
            if abs(max_sector_contrib[1]) > 0.03:  # 3%以上贡献
                insights.append(f"行业 {max_sector_contrib[0]} 风险较高，建议降低该行业配置")
        
        # 基于因子贡献的洞察
        if result.factor_contributions:
            negative_factors = [f for f, c in result.factor_contributions.items() if c < -0.01]
            if negative_factors:
                insights.append(f"因子 {', '.join(negative_factors)} 表现不佳，建议调整因子暴露")
        
        # 基于未解释部分的洞察
        if abs(result.unexplained_portion) > 0.02:
            insights.append("存在较大未解释回撤，可能存在模型外风险或黑天鹅事件")
        
        return insights
    
    def _generate_recommendations(self, result: AttributionResult) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于集中度的建议
        if result.stock_contributions:
            concentration = self._analyze_concentration(result.stock_contributions)
            if concentration['herfindahl_index'] > 0.3:
                recommendations.append("建议降低个股集中度，增加持仓分散化")
        
        # 基于行业分布的建议
        if result.sector_contributions:
            sector_analysis = self._calculate_sector_diversification(result.sector_contributions)
            if sector_analysis['diversification_score'] < 0.5:
                recommendations.append("建议增加行业多样化，避免行业集中风险")
        
        # 基于因子暴露的建议
        if result.factor_contributions:
            negative_contribs = [c for c in result.factor_contributions.values() if c < 0]
            if len(negative_contribs) > len(result.factor_contributions) * 0.7:
                recommendations.append("多数因子贡献为负，建议重新评估因子选择和权重")
        
        # 基于置信度的建议
        if result.confidence_score < 0.6:
            recommendations.append("归因分析置信度较低，建议完善股票信息和因子数据")
        
        # 基于历史趋势的建议
        if len(self.attribution_history) > 5:
            recent_drawdowns = [r.total_drawdown for r in self.attribution_history[-5:]]
            if all(d < -0.01 for d in recent_drawdowns):  # 连续回撤
                recommendations.append("连续出现回撤，建议暂时降低风险暴露")
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any], save_path: str):
        """保存报告到文件"""
        save_path = Path(save_path)
        
        # 保存JSON格式
        json_path = save_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存Markdown格式
        md_path = save_path.with_suffix('.md')
        self._save_markdown_report(report, md_path)
        
        logger.info(f"报告已保存到 {json_path} 和 {md_path}")
    
    def _save_markdown_report(self, report: Dict[str, Any], save_path: Path):
        """保存Markdown格式报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            # 标题
            f.write(f"# {report['report_info']['title']}\n\n")
            f.write(f"**生成时间**: {report['report_info']['timestamp']}\n")
            f.write(f"**总回撤**: {report['report_info']['total_drawdown']:.2%}\n")
            f.write(f"**置信度**: {report['report_info']['confidence_score']:.2%}\n\n")
            
            # 执行摘要
            f.write("## 执行摘要\n\n")
            summary = report['executive_summary']
            f.write(f"- **总回撤**: {summary['total_drawdown_pct']}\n")
            f.write(f"- **置信度等级**: {summary['confidence_level']}\n")
            f.write(f"- **未解释部分**: {summary['unexplained_portion_pct']}\n\n")
            
            f.write("### 主要贡献者\n")
            f.write(f"- **最大个股贡献**: {summary['main_contributors']['top_stock']['name']} ({summary['main_contributors']['top_stock']['contribution']})\n")
            f.write(f"- **最大行业贡献**: {summary['main_contributors']['top_sector']['name']} ({summary['main_contributors']['top_sector']['contribution']})\n")
            f.write(f"- **最大因子贡献**: {summary['main_contributors']['top_factor']['name']} ({summary['main_contributors']['top_factor']['contribution']})\n\n")
            
            f.write("### 关键发现\n")
            for finding in summary['key_findings']:
                f.write(f"- {finding}\n")
            f.write("\n")
            
            # 风险洞察
            f.write("## 风险洞察\n\n")
            for insight in report['risk_insights']:
                f.write(f"- {insight}\n")
            f.write("\n")
            
            # 改进建议
            f.write("## 改进建议\n\n")
            for recommendation in report['recommendations']:
                f.write(f"- {recommendation}\n")
            f.write("\n")
    
    def get_historical_attribution_summary(self) -> Dict[str, Any]:
        """获取历史归因分析摘要"""
        if not self.attribution_history:
            return {'message': '无历史数据'}
        
        # 计算统计指标
        total_drawdowns = [r.total_drawdown for r in self.attribution_history]
        confidence_scores = [r.confidence_score for r in self.attribution_history]
        
        # 统计最频繁的贡献者
        all_stock_contribs = {}
        all_sector_contribs = {}
        
        for result in self.attribution_history:
            for stock, contrib in result.stock_contributions.items():
                if stock not in all_stock_contribs:
                    all_stock_contribs[stock] = []
                all_stock_contribs[stock].append(contrib)
            
            for sector, contrib in result.sector_contributions.items():
                if sector not in all_sector_contribs:
                    all_sector_contribs[sector] = []
                all_sector_contribs[sector].append(contrib)
        
        summary = {
            'analysis_period': {
                'start_date': self.attribution_history[0].timestamp.isoformat(),
                'end_date': self.attribution_history[-1].timestamp.isoformat(),
                'total_analyses': len(self.attribution_history)
            },
            'drawdown_statistics': {
                'average_drawdown': f"{np.mean(total_drawdowns):.2%}",
                'max_drawdown': f"{min(total_drawdowns):.2%}",
                'min_drawdown': f"{max(total_drawdowns):.2%}",
                'drawdown_volatility': f"{np.std(total_drawdowns):.2%}"
            },
            'confidence_statistics': {
                'average_confidence': f"{np.mean(confidence_scores):.2%}",
                'min_confidence': f"{min(confidence_scores):.2%}",
                'max_confidence': f"{max(confidence_scores):.2%}"
            },
            'frequent_contributors': {
                'stocks': self._get_frequent_contributors(all_stock_contribs),
                'sectors': self._get_frequent_contributors(all_sector_contribs)
            }
        }
        
        return summary
    
    def _get_frequent_contributors(self, contributions: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """获取频繁贡献者"""
        contributor_stats = []
        
        for name, contribs in contributions.items():
            if len(contribs) >= 3:  # 至少出现3次
                contributor_stats.append({
                    'name': name,
                    'frequency': len(contribs),
                    'average_contribution': f"{np.mean(contribs):.2%}",
                    'total_contribution': f"{sum(contribs):.2%}"
                })
        
        # 按频率排序
        contributor_stats.sort(key=lambda x: x['frequency'], reverse=True)
        
        return contributor_stats[:10]  # 返回前10个