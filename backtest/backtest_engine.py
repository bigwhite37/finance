"""
回测引擎 - 基于qlib的强化学习策略回测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from .performance_analyzer import PerformanceAnalyzer
from .comfort_metrics import ComfortabilityMetrics

logger = logging.getLogger(__name__)


class BacktestEngine:
    """强化学习策略回测系统"""
    
    def __init__(self, config: Dict):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置参数
        """
        self.config = config
        
        # 回测参数
        self.initial_capital = config.get('initial_capital', 1000000)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.slippage = config.get('slippage', 0.0001)
        self.commission = config.get('commission', 0.0005)
        
        # 分析器
        self.performance_analyzer = PerformanceAnalyzer(config)
        self.comfort_metrics = ComfortabilityMetrics(config)
        
        # 回测结果
        self.portfolio_history = []
        self.trade_history = []
        self.daily_returns = []
        self.positions_history = []
        
    def run_backtest(self, 
                    agent,
                    env,
                    start_date: str,
                    end_date: str,
                    benchmark_data: Optional[pd.DataFrame] = None,
                    safety_shield=None) -> Dict:
        """
        执行策略回测
        
        Args:
            agent: 训练好的智能体
            env: 交易环境
            start_date: 开始日期
            end_date: 结束日期
            benchmark_data: 基准数据
            
        Returns:
            回测结果字典
        """
        logger.info(f"开始回测: {start_date} 至 {end_date}")
        
        # 初始化
        self._initialize_backtest()
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # 重置环境
        state, info = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # 获取智能体动作
            action, log_prob, value, cvar_estimate = agent.get_action(state, deterministic=True)
            
            # 应用安全保护层
            safe_action = action
            if safety_shield is not None:
                safe_action = safety_shield.shield_action(action, info)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(safe_action)
            
            # 记录交易信息（使用受保护的动作）
            self._record_step(safe_action, state, info, step_count)
            
            # 更新状态
            state = next_state
            done = terminated or truncated
            step_count += 1
            
            if step_count % 50 == 0:
                logger.info(f"回测进度: {step_count} 步, 当前净值: {info.get('portfolio_value', 1.0):.4f}")
        
        # 分析结果
        results = self._analyze_results(benchmark_data)
        
        logger.info("回测完成")
        return results
    
    def _initialize_backtest(self):
        """初始化回测状态"""
        self.portfolio_history.clear()
        self.trade_history.clear()
        self.daily_returns.clear()
        self.positions_history.clear()
    
    def _record_step(self, action: np.ndarray, state: np.ndarray, info: Dict, step: int):
        """记录每步的交易信息"""
        # 计算当前日期（历史回测日期）
        current_date = self.start_date + pd.Timedelta(days=step)
        
        # 记录组合信息
        portfolio_info = {
            'step': step,
            'date': current_date,
            'portfolio_value': info.get('portfolio_value', 1.0),
            'total_return': info.get('total_return', 0.0),
            'max_drawdown': info.get('max_drawdown', 0.0),
            'volatility': info.get('volatility', 0.0),
            'sharpe_ratio': info.get('sharpe_ratio', 0.0),
            'cash': info.get('cash', 0.0)
        }
        self.portfolio_history.append(portfolio_info)
        
        # 记录仓位信息
        positions = {
            'step': step,
            'weights': action.copy(),
            'total_leverage': np.sum(np.abs(action)),
            'num_positions': np.sum(np.abs(action) > 0.001)
        }
        self.positions_history.append(positions)
        
        # 记录收益率
        if len(self.portfolio_history) > 1:
            prev_value = self.portfolio_history[-2]['portfolio_value']
            current_value = portfolio_info['portfolio_value']
            daily_return = (current_value / prev_value) - 1
            self.daily_returns.append(daily_return)
    
    def _analyze_results(self, benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
        """分析回测结果"""
        if not self.portfolio_history:
            return {}
        
        # 转换为DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        returns_series = pd.Series(self.daily_returns)
        
        # 基础绩效分析
        performance_metrics = self.performance_analyzer.generate_report(
            returns_series, benchmark_data
        )
        
        # 心理舒适度分析
        comfort_metrics = self.comfort_metrics.calculate_metrics(returns_series)
        
        # 交易分析
        trading_stats = self._calculate_trading_stats()
        
        # 风险分析
        risk_metrics = self._calculate_risk_metrics(returns_series)
        
        # 汇总结果
        results = {
            'backtest_summary': {
                'start_date': portfolio_df['date'].iloc[0] if not portfolio_df.empty else None,
                'end_date': portfolio_df['date'].iloc[-1] if not portfolio_df.empty else None,
                'total_steps': len(self.portfolio_history),
                'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1] if not portfolio_df.empty else 1.0
            },
            'performance_metrics': performance_metrics,
            'comfort_metrics': comfort_metrics,
            'trading_stats': trading_stats,
            'risk_metrics': risk_metrics,
            'portfolio_history': portfolio_df,
            'returns_series': returns_series
        }
        
        return results
    
    def _calculate_trading_stats(self) -> Dict:
        """计算交易统计"""
        if not self.positions_history:
            return {}
        
        positions_df = pd.DataFrame(self.positions_history)
        
        # 计算交易统计
        avg_leverage = positions_df['total_leverage'].mean()
        max_leverage = positions_df['total_leverage'].max()
        avg_positions = positions_df['num_positions'].mean()
        
        # 计算换手率
        turnover_rates = []
        for i in range(1, len(self.positions_history)):
            prev_weights = self.positions_history[i-1]['weights']
            curr_weights = self.positions_history[i]['weights']
            turnover = np.sum(np.abs(curr_weights - prev_weights)) / 2
            turnover_rates.append(turnover)
        
        return {
            'avg_leverage': avg_leverage,
            'max_leverage': max_leverage,
            'avg_positions': avg_positions,
            'avg_turnover': np.mean(turnover_rates) if turnover_rates else 0.0,
            'total_trades': len(turnover_rates)
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """计算风险指标"""
        if len(returns) < 2:
            return {}
        
        return {
            'volatility': returns.std() * np.sqrt(252),
            'downside_volatility': returns[returns < 0].std() * np.sqrt(252),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'max_consecutive_losses': self._max_consecutive_negative(returns),
            'loss_days_ratio': (returns < 0).mean()
        }
    
    def _max_consecutive_negative(self, returns: pd.Series) -> int:
        """计算最大连续亏损天数"""
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
    
    def generate_backtest_report(self, results: Dict) -> str:
        """生成回测报告"""
        if not results:
            return "无回测数据"
        
        summary = results.get('backtest_summary', {})
        performance = results.get('performance_metrics', {})
        comfort = results.get('comfort_metrics', {})
        trading = results.get('trading_stats', {})
        risk = results.get('risk_metrics', {})
        
        report = f"""
=== A股强化学习量化交易回测报告 ===

【回测概要】
回测期间: {summary.get('start_date', 'N/A')} 至 {summary.get('end_date', 'N/A')}
回测天数: {summary.get('total_steps', 0)}
最终净值: {summary.get('final_portfolio_value', 1.0):.4f}

【收益指标】
总收益率: {performance.get('总收益率', 0.0):.2%}
年化收益率: {performance.get('年化收益率', 0.0):.2%}
年化波动率: {performance.get('年化波动率', 0.0):.2%}
夏普比率: {performance.get('夏普比率', 0.0):.2f}
最大回撤: {performance.get('最大回撤', 0.0):.2%}

【心理舒适度指标】
月度最大回撤: {comfort.get('月度最大回撤', 0.0):.2%}
连续亏损天数: {comfort.get('连续亏损天数', 0):.0f}天
下跌日占比: {comfort.get('下跌日占比', 0.0):.1%}
95% VaR: {comfort.get('日VaR_95%', 0.0):.2%}

【交易统计】
平均杠杆: {trading.get('avg_leverage', 0.0):.2f}
最大杠杆: {trading.get('max_leverage', 0.0):.2f}
平均持仓数: {trading.get('avg_positions', 0.0):.0f}
平均换手率: {trading.get('avg_turnover', 0.0):.2%}

【风险指标】
下行波动率: {risk.get('downside_volatility', 0.0):.2%}
99% VaR: {risk.get('var_99', 0.0):.2%}
95% CVaR: {risk.get('cvar_95', 0.0):.2%}
收益偏度: {risk.get('skewness', 0.0):.2f}
收益峰度: {risk.get('kurtosis', 0.0):.2f}
"""
        
        return report
    
    def save_results(self, results: Dict, filepath: str):
        """保存回测结果"""
        import joblib
        joblib.dump(results, filepath)
        logger.info(f"回测结果已保存至: {filepath}")
    
    def load_results(self, filepath: str) -> Dict:
        """加载回测结果"""
        import joblib
        results = joblib.load(filepath)
        logger.info(f"回测结果已加载: {filepath}")
        return results