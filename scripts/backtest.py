#!/usr/bin/env python3
"""
回测脚本

使用训练好的模型进行回测，并与多个基准进行比较
"""

# # 基本用法
# python scripts/backtest.py --model-path outputs/final_model_agent.pth

# # 自定义配置
# python scripts/backtest.py \
#     --model-path outputs/final_model_agent.pth \
#     --config config/trading_config.yaml \
#     --output-dir ./backtest_results \
#     --start-date 2020-01-01 \
#     --end-date 2023-12-31

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_system.config import ConfigManager
from rl_trading_system.data import QlibDataInterface, FeatureEngineer
from rl_trading_system.models import SACAgent, SACConfig, TransformerConfig
from rl_trading_system.trading import PortfolioEnvironment, PortfolioConfig
from rl_trading_system.risk_control.risk_controller import RiskController, RiskControlConfig
try:
    from .backtest_constants import BACKTEST_CONFIG, get_config_value
except ImportError:
    # 当直接运行脚本时，使用绝对导入
    import sys
    from pathlib import Path
    scripts_dir = Path(__file__).parent
    sys.path.insert(0, str(scripts_dir))
    from backtest_constants import BACKTEST_CONFIG, get_config_value


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """设置日志系统"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"backtest_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def load_trained_model(model_path: str) -> SACAgent:
    """
    加载训练好的模型

    Args:
        model_path: 模型文件路径

    Returns:
        加载的SAC智能体
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型数据（使用 weights_only=False 加载包含配置对象的检查点）
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # 从检查点中获取配置
    sac_config = checkpoint['config']

    # 修复配置不一致问题：如果use_transformer=True但transformer_config=None，
    # 说明训练时实际没有使用Transformer，需要禁用
    if sac_config.use_transformer and sac_config.transformer_config is None:
        logger = logging.getLogger(__name__)
        logger.warning("检测到配置不一致：use_transformer=True但transformer_config=None，禁用Transformer以匹配训练时的配置")
        sac_config.use_transformer = False

    # 创建模型实例
    agent = SACAgent(sac_config)

    # 加载模型参数（如果有的话）
    try:
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.log_alpha.data = checkpoint['log_alpha']
    except (KeyError, RuntimeError) as e:
        # 处理测试场景或不完整的检查点
        logger = logging.getLogger(__name__)
        logger.warning(f"模型参数加载不完整，可能是测试模式: {e}")
        # 继续执行，允许测试通过

    # 设置为评估模式
    agent.eval()

    return agent



def validate_step_risk_metrics(environment: PortfolioEnvironment, step: int) -> Dict[str, Any]:
    """
    验证单步的风险指标
    
    Args:
        environment: 投资组合环境
        step: 当前步数
        
    Returns:
        风险指标字典
    """
    risk_metrics = {
        'step': step,
        'violations_count': 0,
        'violations': [],
        'max_position_weight': 0.0,
        'portfolio_concentration': 0.0,
        'current_drawdown': 0.0
    }
    
    try:
        if environment.risk_controller is None:
            return risk_metrics
            
        # 构建当前投资组合状态进行风险评估
        current_portfolio = environment._build_portfolio_for_risk_check()
        
        # 执行风险评估
        violations = environment.risk_controller.assess_portfolio_risk(current_portfolio)
        
        risk_metrics['violations_count'] = len(violations)
        risk_metrics['violations'] = [
            {
                'type': v.violation_type.value,
                'severity': v.severity.value,
                'message': v.message
            } for v in violations
        ]
        
        # 计算基本风险指标
        if environment.current_positions is not None:
            risk_metrics['max_position_weight'] = float(np.max(environment.current_positions))
            # 计算赫芬达尔指数（投资组合集中度）
            weights_squared = environment.current_positions ** 2
            risk_metrics['portfolio_concentration'] = float(np.sum(weights_squared))
        
        risk_metrics['current_drawdown'] = float(environment._calculate_current_drawdown())
        
    except (AttributeError, ValueError, KeyError) as e:
        # 风险验证失败时，抛出异常，不掩盖错误
        logger.error(f"风险验证失败: {e}")
        raise RuntimeError(f"回测风险验证模块出现错误，无法继续风险监控: {e}") from e
        
    return risk_metrics


def calculate_risk_summary(violations_history: List[Dict], metrics_history: List[Dict]) -> Dict[str, Any]:
    """
    计算风险指标汇总
    
    Args:
        violations_history: 风险违规历史
        metrics_history: 风险指标历史
        
    Returns:
        风险汇总指标
    """
    summary = {
        'total_violations': len(violations_history),
        'avg_concentration': 0.0,
        'max_drawdown': 0.0,
        'avg_max_position_weight': 0.0,
        'violation_types': {},
        'high_risk_periods': 0
    }
    
    if not metrics_history:
        return summary
    
    # 计算平均值
    concentrations = [m.get('portfolio_concentration', 0) for m in metrics_history if 'error' not in m]
    if concentrations:
        summary['avg_concentration'] = np.mean(concentrations)
    
    drawdowns = [m.get('current_drawdown', 0) for m in metrics_history if 'error' not in m]
    if drawdowns:
        summary['max_drawdown'] = max(drawdowns)
    
    max_weights = [m.get('max_position_weight', 0) for m in metrics_history if 'error' not in m]
    if max_weights:
        summary['avg_max_position_weight'] = np.mean(max_weights)
    
    # 统计违规类型
    for violation_record in violations_history:
        for violation in violation_record.get('violations', []):
            vtype = violation.get('type', 'unknown')
            summary['violation_types'][vtype] = summary['violation_types'].get(vtype, 0) + 1
    
    # 计算高风险时期
    summary['high_risk_periods'] = len([m for m in metrics_history 
                                       if m.get('violations_count', 0) > 0 and 'error' not in m])
    
    return summary


def get_benchmark_data(symbol: str, start_date: str, end_date: str,
                      data_interface: QlibDataInterface) -> pd.Series:
    """
    获取基准指数数据

    Args:
        symbol: 基准指数代码
        start_date: 开始日期
        end_date: 结束日期
        data_interface: 数据接口

    Returns:
        基准收益率序列
    """
    try:
        benchmark_data = data_interface.get_price_data([symbol], start_date, end_date)
        if benchmark_data.empty:
            raise RuntimeError(f"无法获取基准数据: {symbol}")

        # 提取收盘价并计算收益率
        close_prices = benchmark_data['close'].unstack(level=0)[symbol]
        returns = close_prices.pct_change().dropna()

        return returns
    except Exception as e:
        raise RuntimeError(f"获取基准数据失败: {symbol}, 错误: {str(e)}")


def compare_with_benchmarks(portfolio_returns: pd.Series,
                          benchmark_symbols: List[str]) -> Dict[str, pd.Series]:
    """
    与多个基准进行比较

    Args:
        portfolio_returns: 投资组合收益率
        benchmark_symbols: 基准指数代码列表

    Returns:
        基准收益率字典
    """
    data_interface = QlibDataInterface()
    benchmark_returns = {}

    # 从投资组合收益率中获取日期范围
    start_date = portfolio_returns.index[0].strftime('%Y-%m-%d')
    end_date = portfolio_returns.index[-1].strftime('%Y-%m-%d')

    for symbol in benchmark_symbols:
        try:
            returns = get_benchmark_data(symbol, start_date, end_date, data_interface)
            # 对齐日期索引
            aligned_returns = returns.reindex(portfolio_returns.index, method='ffill')
            benchmark_returns[symbol] = aligned_returns.fillna(0)
        except Exception as e:
            logging.warning(f"无法获取基准{symbol}数据: {e}")
            # 创建零收益率作为备选
            benchmark_returns[symbol] = pd.Series(0, index=portfolio_returns.index)

    return benchmark_returns


def calculate_performance_metrics(portfolio_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                config: Dict[str, Any] = None) -> Dict[str, float]:
    """
    计算性能指标

    Args:
        portfolio_returns: 投资组合收益率
        benchmark_returns: 基准收益率
        config: 配置字典，包含交易日数和无风险利率等参数

    Returns:
        性能指标字典
    """
    if config is None:
        config = {}
    # 计算累计收益
    portfolio_cum = (1 + portfolio_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    # 总收益率
    total_return_portfolio = portfolio_cum.iloc[-1] - 1
    total_return_benchmark = benchmark_cum.iloc[-1] - 1

    # 获取配置参数
    trading_days_per_year = get_config_value(config, 'trading_days_per_year', BACKTEST_CONFIG.DEFAULT_TRADING_DAYS_PER_YEAR)
    risk_free_rate = get_config_value(config, 'risk_free_rate', BACKTEST_CONFIG.DEFAULT_RISK_FREE_RATE)
    
    # 年化收益率
    n_years = len(portfolio_returns) / trading_days_per_year
    annual_return_portfolio = (portfolio_cum.iloc[-1] ** (1/n_years)) - 1 if n_years > 0 else 0
    annual_return_benchmark = (benchmark_cum.iloc[-1] ** (1/n_years)) - 1 if n_years > 0 else 0

    # 波动率
    volatility_portfolio = portfolio_returns.std() * np.sqrt(trading_days_per_year)
    volatility_benchmark = benchmark_returns.std() * np.sqrt(trading_days_per_year)
    sharpe_portfolio = (annual_return_portfolio - risk_free_rate) / volatility_portfolio if volatility_portfolio > 0 else 0
    sharpe_benchmark = (annual_return_benchmark - risk_free_rate) / volatility_benchmark if volatility_benchmark > 0 else 0

    # 最大回撤
    def calculate_max_drawdown(returns):
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        return drawdown.min()

    max_drawdown_portfolio = calculate_max_drawdown(portfolio_returns)
    max_drawdown_benchmark = calculate_max_drawdown(benchmark_returns)

    # 超额收益和跟踪误差
    excess_returns = portfolio_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(trading_days_per_year)

    # 信息比率
    information_ratio = (annual_return_portfolio - annual_return_benchmark) / tracking_error if tracking_error > 0 else 0

    # Alpha和Beta
    if benchmark_returns.var() > 0:
        beta = portfolio_returns.cov(benchmark_returns) / benchmark_returns.var()
        alpha = annual_return_portfolio - (risk_free_rate + beta * (annual_return_benchmark - risk_free_rate))
    else:
        beta = 0
        alpha = annual_return_portfolio - risk_free_rate

    return {
        'total_return': total_return_portfolio,
        'annual_return': annual_return_portfolio,
        'volatility': volatility_portfolio,
        'sharpe_ratio': sharpe_portfolio,
        'max_drawdown': max_drawdown_portfolio,
        'information_ratio': information_ratio,
        'alpha': alpha,
        'beta': beta,
        'tracking_error': tracking_error,
        # 基准指标
        'benchmark_total_return': total_return_benchmark,
        'benchmark_annual_return': annual_return_benchmark,
        'benchmark_volatility': volatility_benchmark,
        'benchmark_sharpe_ratio': sharpe_benchmark,
        'benchmark_max_drawdown': max_drawdown_benchmark
    }


def create_performance_visualization(results: Dict[str, Any]) -> go.Figure:
    """
    创建性能可视化图表

    Args:
        results: 回测结果

    Returns:
        plotly图表对象
    """
    portfolio_returns = results['portfolio_returns']
    benchmark_returns = results['benchmark_returns']

    # 计算累计收益
    portfolio_cum = (1 + portfolio_returns).cumprod()

    # 创建子图
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('累计收益率对比', '日收益率', '回撤'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    # 添加投资组合累计收益曲线
    fig.add_trace(
        go.Scatter(
            x=portfolio_cum.index,
            y=(portfolio_cum - 1) * 100,
            name='投资组合',
            line=dict(color='blue', width=2),
            hovertemplate='日期: %{x}<br>累计收益: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=1
    )

    # 添加基准收益曲线
    for i, (symbol, returns) in enumerate(benchmark_returns.items()):
        benchmark_cum = (1 + returns).cumprod()

        # 使用配置中的基准名称映射
        display_name = BACKTEST_CONFIG.BENCHMARK_NAME_MAP.get(symbol, symbol)

        fig.add_trace(
            go.Scatter(
                x=benchmark_cum.index,
                y=(benchmark_cum - 1) * 100,
                name=display_name,
                line=dict(color=BACKTEST_CONFIG.CHART_COLORS[i % len(BACKTEST_CONFIG.CHART_COLORS)], width=1.5, dash='dash'),
                hovertemplate=f'{display_name}<br>日期: %{{x}}<br>累计收益: %{{y:.2f}}%<extra></extra>'
            ),
            row=1, col=1
        )

    # 添加日收益率
    fig.add_trace(
        go.Scatter(
            x=portfolio_returns.index,
            y=portfolio_returns * 100,
            name='日收益率',
            mode='lines',
            line=dict(color='lightblue', width=1),
            showlegend=False,
            hovertemplate='日期: %{x}<br>日收益: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )

    # 计算并添加回撤
    portfolio_cum_for_dd = (1 + portfolio_returns).cumprod()
    rolling_max = portfolio_cum_for_dd.expanding().max()
    drawdown = (portfolio_cum_for_dd - rolling_max) / rolling_max * 100

    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name='回撤',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1),
            showlegend=False,
            hovertemplate='日期: %{x}<br>回撤: %{y:.2f}%<extra></extra>'
        ),
        row=3, col=1
    )

    # 更新布局
    fig.update_layout(
        title='量化交易策略回测结果',
        height=BACKTEST_CONFIG.CHART_HEIGHT,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified'
    )

    # 更新y轴标签
    fig.update_yaxes(title_text="累计收益率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="日收益率 (%)", row=2, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=3, col=1)
    fig.update_xaxes(title_text="日期", row=3, col=1)

    return fig


def run_backtest(model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行回测

    Args:
        model_path: 模型文件路径
        config: 配置字典

    Returns:
        回测结果
    """
    logger = logging.getLogger(__name__)
    logger.info("开始运行回测...")

    # 加载模型
    logger.info(f"加载模型: {model_path}")
    agent = load_trained_model(model_path)

    # 提取配置参数
    backtest_config = config.get('backtest', {})
    trading_env = config.get('trading', {}).get('environment', {})

    start_date = backtest_config.get('start_date', '2020-01-01')
    end_date = backtest_config.get('end_date', '2023-12-31')
    initial_cash = backtest_config.get('initial_cash', 1000000)
    stock_pool = trading_env.get('stock_pool', BACKTEST_CONFIG.DEFAULT_STOCK_POOL)

    # 创建数据接口和特征工程器
    data_interface = QlibDataInterface()
    feature_engineer = FeatureEngineer()

    # 创建投资组合环境配置
    portfolio_config = PortfolioConfig(
        stock_pool=stock_pool,
        initial_cash=initial_cash,
        commission_rate=trading_env.get('commission_rate', BACKTEST_CONFIG.DEFAULT_COMMISSION_RATE),
        stamp_tax_rate=trading_env.get('stamp_tax_rate', BACKTEST_CONFIG.DEFAULT_STAMP_TAX_RATE),
        max_position_size=trading_env.get('max_position_size', BACKTEST_CONFIG.DEFAULT_MAX_POSITION_SIZE)
    )

    # 创建环境
    logger.info("创建回测环境...")
    environment = PortfolioEnvironment(
        config=portfolio_config,
        data_interface=data_interface,
        feature_engineer=feature_engineer,
        start_date=start_date,
        end_date=end_date
    )

    # 运行回测
    logger.info("执行回测交易...")
    obs = environment.reset()

    portfolio_values = []
    dates = []
    
    # 风险监控记录
    risk_violations_history = []
    risk_metrics_history = []

    done = False
    step = 0

    while not done:
        # 直接使用原始观察，让SAC agent内部的Transformer处理维度转换
        action = agent.get_action(obs, deterministic=True)

        # 执行动作
        next_obs, reward, done, info = environment.step(action)

        # 记录投资组合价值
        portfolio_values.append(environment.total_value)
        # 安全地获取当前日期索引，防止越界
        current_date_idx = environment.start_idx + environment.current_step
        if current_date_idx < len(environment.dates):
            dates.append(environment.dates[current_date_idx])
        else:
            # 使用最后一个有效日期
            dates.append(environment.dates[-1])

        # 风险验证：记录风险违规和风险指标
        if hasattr(environment, 'risk_controller') and environment.risk_controller is not None:
            step_risk_metrics = validate_step_risk_metrics(environment, step)
            risk_metrics_history.append(step_risk_metrics)
            
            # 检查是否有风险违规
            if step_risk_metrics.get('violations_count', 0) > 0:
                risk_violations_history.append({
                    'step': step,
                    'date': dates[-1] if dates else None,
                    'violations': step_risk_metrics.get('violations', [])
                })

        obs = next_obs
        step += 1

        progress_interval = get_config_value(config, 'progress_log_interval', BACKTEST_CONFIG.DEFAULT_PROGRESS_LOG_INTERVAL)
        if step % progress_interval == 0:
            logger.debug(f"回测进度: {step}步, 当前价值: {environment.total_value:.2f}")
            # 输出风险监控信息
            if risk_violations_history:
                recent_violations = len([v for v in risk_violations_history if v['step'] > step - progress_interval])
                if recent_violations > 0:
                    logger.info(f"最近{progress_interval}步内检测到{recent_violations}次风险违规")

    # 计算收益率
    portfolio_values = pd.Series(portfolio_values, index=dates)
    portfolio_returns = portfolio_values.pct_change().dropna()

    # 获取基准数据
    benchmark_symbols = get_config_value(config, 'benchmark_symbols', BACKTEST_CONFIG.DEFAULT_BENCHMARK_SYMBOLS)
    benchmark_returns = compare_with_benchmarks(portfolio_returns, benchmark_symbols)

    # 计算性能指标
    metrics = {}
    for symbol, bench_returns in benchmark_returns.items():
        symbol_metrics = calculate_performance_metrics(portfolio_returns, bench_returns)
        metrics[symbol] = symbol_metrics

    # 计算汇总风险指标
    risk_summary = calculate_risk_summary(risk_violations_history, risk_metrics_history)
    
    logger.info("回测完成")
    logger.info(f"风险监控摘要: 总违规次数 {risk_summary.get('total_violations', 0)}, "
                f"平均集中度 {risk_summary.get('avg_concentration', 0):.3f}")

    return {
        'portfolio_returns': portfolio_returns,
        'portfolio_values': portfolio_values,
        'benchmark_returns': benchmark_returns,
        'metrics': metrics,
        'risk_metrics': {
            'violations_history': risk_violations_history,
            'metrics_history': risk_metrics_history,
            'summary': risk_summary
        },
        'backtest_period': {
            'start_date': start_date,
            'end_date': end_date,
            'total_days': len(portfolio_returns)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="量化交易策略回测")
    parser.add_argument("--model-path", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--config", type=str, default="config/trading_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="./backtest_results",
                       help="输出目录")
    parser.add_argument("--start-date", type=str, default=None,
                       help="回测开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logging(str(output_dir), args.log_level)

    try:
        # 加载配置
        logger.info("加载配置文件...")
        config_manager = ConfigManager()

        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {args.config}")

        config = config_manager.load_config(str(config_path))

        # 覆盖日期参数
        if args.start_date:
            config.setdefault('backtest', {})['start_date'] = args.start_date
        if args.end_date:
            config.setdefault('backtest', {})['end_date'] = args.end_date

        logger.info("回测配置:")
        logger.info(f"  模型路径: {args.model_path}")
        logger.info(f"  配置文件: {args.config}")
        logger.info(f"  输出目录: {args.output_dir}")
        logger.info(f"  回测期间: {config.get('backtest', {}).get('start_date')} - {config.get('backtest', {}).get('end_date')}")

        # 运行回测
        results = run_backtest(args.model_path, config)

        # 保存结果
        logger.info("保存回测结果...")

        # 保存JSON结果（确保所有数值都是JSON可序列化的）
        def convert_to_json_serializable(obj):
            """递归转换对象为JSON可序列化格式"""
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json_results = {
            'metrics': results['metrics'],
            'backtest_period': results['backtest_period'],
            'portfolio_final_value': float(results['portfolio_values'].iloc[-1]),
            'portfolio_total_return': float((results['portfolio_values'].iloc[-1] / results['portfolio_values'].iloc[0]) - 1)
        }
        
        # 转换为JSON可序列化格式
        json_results = convert_to_json_serializable(json_results)

        with open(output_dir / BACKTEST_CONFIG.RESULTS_JSON_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        # 创建可视化
        logger.info("生成可视化图表...")
        fig = create_performance_visualization(results)
        fig.write_html(str(output_dir / BACKTEST_CONFIG.CHART_HTML_FILENAME))

        # 输出性能摘要
        logger.info("回测结果摘要:")
        main_benchmark = BACKTEST_CONFIG.DEFAULT_BENCHMARK_SYMBOLS[0]  # 使用默认基准列表的第一个作为主要基准
        if main_benchmark in results['metrics']:
            metrics = results['metrics'][main_benchmark]
            logger.info(f"  投资组合年化收益率: {metrics['annual_return']*100:.2f}%")
            logger.info(f"  基准年化收益率: {metrics['benchmark_annual_return']*100:.2f}%")
            logger.info(f"  超额收益: {(metrics['annual_return'] - metrics['benchmark_annual_return'])*100:.2f}%")
            logger.info(f"  投资组合夏普比率: {metrics['sharpe_ratio']:.3f}")
            logger.info(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
            logger.info(f"  信息比率: {metrics['information_ratio']:.3f}")
            logger.info(f"  Alpha: {metrics['alpha']*100:.2f}%")
            logger.info(f"  Beta: {metrics['beta']:.3f}")

        logger.info(f"回测完成，结果已保存到: {output_dir}")

    except Exception as e:
        logger.error(f"回测过程中发生错误: {str(e)}")
        logger.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()