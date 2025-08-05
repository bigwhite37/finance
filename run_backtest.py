#!/usr/bin/env python3
"""
专业回测脚本
使用训练好的RL模型进行回测分析，集成Qlib可视化功能
支持命令行参数指定模型路径和配置
"""
import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入依赖模块
from data_loader import QlibDataLoader
from backtest import BacktestAnalyzer
from qlib_visualization import QlibVisualizer, create_qlib_report
from stable_baselines3 import PPO, SAC


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志配置"""
    # 创建logs目录
    os.makedirs("logs", exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("回测日志系统初始化完成")
    return logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="强化学习投资策略回测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="训练好的模型文件路径（.zip格式）"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_daily.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="结果输出目录"
    )

    parser.add_argument(
        "--test-start",
        type=str,
        help="测试开始时间 (YYYY-MM-DD)，覆盖配置文件"
    )

    parser.add_argument(
        "--test-end",
        type=str,
        help="测试结束时间 (YYYY-MM-DD)，覆盖配置文件"
    )

    parser.add_argument(
        "--stocks",
        type=int,
        help="股票数量限制，覆盖配置文件"
    )

    parser.add_argument(
        "--initial-cash",
        type=float,
        default=1000000,
        help="初始资金"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="跳过绘图，仅生成报告"
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="跳过Excel导出"
    )

    parser.add_argument(
        "--enable-qlib-viz",
        action="store_true",
        help="启用Qlib风格的专业可视化分析"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )

    parser.add_argument(
        "--qlib-uri",
        type=str,
        help="Qlib数据URI，覆盖配置文件"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def load_trained_model(model_path: str, algorithm: str, logger: logging.Logger):
    """
    加载训练好的模型

    Args:
        model_path: 模型文件路径
        algorithm: 算法类型 (PPO/SAC)
        logger: 日志记录器

    Returns:
        加载的模型对象
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    logger.info(f"加载{algorithm}模型: {model_path}")

    try:
        if algorithm.upper() == "PPO":
            model = PPO.load(model_path)
        elif algorithm.upper() == "SAC":
            model = SAC.load(model_path)
        else:
            raise ValueError(f"不支持的算法类型: {algorithm}")

        logger.info(f"模型加载成功: {algorithm}")
        return model

    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise RuntimeError(f"模型加载失败: {e}")


def prepare_test_data(config: dict, args, logger: logging.Logger):
    """
    准备测试数据

    Args:
        config: 配置字典
        args: 命令行参数
        logger: 日志记录器

    Returns:
        测试数据DataFrame
    """
    logger.info("准备测试数据...")

    # 使用命令行参数覆盖配置
    data_config = config['data'].copy()

    if args.test_start:
        data_config['test_start'] = args.test_start
    if args.test_end:
        data_config['test_end'] = args.test_end
    if args.stocks:
        data_config['stock_limit'] = args.stocks
    if args.qlib_uri:
        data_config['provider_uri']['day'] = args.qlib_uri

    logger.info(f"测试期间: {data_config['test_start']} - {data_config['test_end']}")
    logger.info(f"股票数量: {data_config['stock_limit']}")

    # 初始化数据加载器
    data_loader = QlibDataLoader(data_config)

    # 加载测试数据
    _, _, test_data = data_loader.load_data()

    logger.info(f"测试数据形状: {test_data.shape}")
    logger.info(f"数据时间范围: {test_data.index.get_level_values(1).min()} - {test_data.index.get_level_values(1).max()}")

    return test_data


def run_backtest_analysis(model, test_data, args, config, logger: logging.Logger):
    """
    运行回测分析

    Args:
        model: 训练好的模型
        test_data: 测试数据
        args: 命令行参数
        config: 配置字典
        logger: 日志记录器

    Returns:
        回测结果字典
    """
    logger.info("开始回测分析...")

    # 创建回测分析器
    analyzer = BacktestAnalyzer()

    # 从配置文件获取环境参数
    env_config = config.get('environment', {})

    # 运行回测（传递环境配置）
    results = analyzer.run_backtest(
        model=model,
        test_data=test_data,
        initial_cash=args.initial_cash,
        env_config=env_config
    )

    logger.info("回测分析完成")

    # 输出关键指标摘要
    rl_metrics = results['performance_metrics']['rl_strategy']
    bench_metrics = results['performance_metrics']['benchmark']

    logger.info("=" * 60)
    logger.info("回测结果摘要:")
    logger.info("-" * 30)
    logger.info(f"RL策略总收益率: {rl_metrics['total_return']:.2%}")
    logger.info(f"RL策略年化收益率: {rl_metrics['annualized_return']:.2%}")
    logger.info(f"RL策略最大回撤: {rl_metrics['max_drawdown']:.2%}")
    logger.info(f"RL策略夏普比率: {rl_metrics['sharpe_ratio']:.3f}")
    logger.info(f"RL策略Calmar比率: {rl_metrics['calmar_ratio']:.3f}")
    logger.info("-" * 30)
    logger.info(f"基准总收益率: {bench_metrics['total_return']:.2%}")
    logger.info(f"基准年化收益率: {bench_metrics['annualized_return']:.2%}")
    logger.info(f"基准最大回撤: {bench_metrics['max_drawdown']:.2%}")
    logger.info(f"基准夏普比率: {bench_metrics['sharpe_ratio']:.3f}")
    logger.info("-" * 30)

    if 'relative' in results['performance_metrics']:
        rel_metrics = results['performance_metrics']['relative']
        logger.info(f"超额收益: {rel_metrics['excess_return']:.2%}")
        logger.info(f"信息比率: {rel_metrics['information_ratio']:.3f}")

    logger.info("=" * 60)

    return analyzer, results


def generate_outputs(analyzer, results, args, logger: logging.Logger):
    """
    生成输出文件

    Args:
        analyzer: 回测分析器
        results: 回测结果
        args: 命令行参数
        logger: 日志记录器
    """
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 生成文本报告
    report_path = output_dir / f"backtest_report_{timestamp}.txt"
    logger.info(f"生成回测报告: {report_path}")
    analyzer.create_performance_report(str(report_path))

    # 生成标准可视化图表
    if not args.no_plot:
        plot_path = output_dir / f"backtest_plots_{timestamp}.png"
        logger.info(f"生成标准可视化图表: {plot_path}")
        analyzer.plot_performance(str(plot_path))

    # 生成Qlib风格专业可视化分析
    if args.enable_qlib_viz:
        logger.info("生成Qlib风格专业分析报告...")
        qlib_chart_path = create_qlib_report(
            results=results,
            output_dir=str(output_dir),
            timestamp=timestamp
        )
        logger.info(f"Qlib专业分析图表已生成: {qlib_chart_path}")

    # 导出Excel
    if not args.no_export:
        excel_path = output_dir / f"backtest_results_{timestamp}.xlsx"
        logger.info(f"导出Excel结果: {excel_path}")
        analyzer.export_results(str(excel_path))

    logger.info(f"所有结果已保存到目录: {output_dir}")


def validate_inputs(args, logger: logging.Logger):
    """验证输入参数"""
    logger.info("验证输入参数...")

    # 验证模型文件
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    if not args.model_path.endswith('.zip'):
        raise ValueError("模型文件必须是.zip格式")

    # 验证配置文件
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")

    # 验证时间格式
    if args.test_start:
        try:
            datetime.strptime(args.test_start, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"测试开始时间格式错误: {args.test_start}")

    if args.test_end:
        try:
            datetime.strptime(args.test_end, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"测试结束时间格式错误: {args.test_end}")

    # 验证数值参数
    if args.stocks and args.stocks <= 0:
        raise ValueError(f"股票数量必须大于0: {args.stocks}")

    if args.initial_cash <= 0:
        raise ValueError(f"初始资金必须大于0: {args.initial_cash}")

    logger.info("输入参数验证通过")


def main():
    """主函数"""
    args = parse_arguments()
    logger = setup_logging(args.log_level)

    try:
        logger.info("=" * 80)
        logger.info("强化学习投资策略回测系统")
        logger.info("=" * 80)
        logger.info(f"模型路径: {args.model_path}")
        logger.info(f"配置文件: {args.config}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"初始资金: {args.initial_cash:,.0f}")

        # 验证输入
        validate_inputs(args, logger)

        # 加载配置
        logger.info("加载配置文件...")
        config = load_config(args.config)

        # 获取算法类型
        algorithm = config.get('model', {}).get('algorithm', 'PPO')
        logger.info(f"算法类型: {algorithm}")

        # 加载模型
        model = load_trained_model(args.model_path, algorithm, logger)

        # 准备测试数据
        test_data = prepare_test_data(config, args, logger)

        # 运行回测分析
        analyzer, results = run_backtest_analysis(model, test_data, args, config, logger)

        # 生成输出文件
        generate_outputs(analyzer, results, args, logger)

        logger.info("回测完成！")
        return 0

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")
        return 0

    except Exception as e:
        logger.error(f"回测执行失败: {e}", exc_info=True)
        return 1

    finally:
        logger.info("清理资源...")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)