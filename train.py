"""
主训练脚本
用于启动完整的强化学习训练流水线
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from trainer import RLTrainer, load_config


def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    # 创建logs目录
    os.makedirs("logs", exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("日志系统初始化完成")
    return logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="强化学习投资策略训练脚本")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_daily.yaml",
        help="配置文件路径"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "full"],
        default="full",
        help="运行模式：train=仅训练, evaluate=仅评估, full=训练+评估"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="评估模式下的模型路径"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="强制使用GPU（如果可用）"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="随机种子，覆盖配置文件中的设置"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式，只检查配置不实际训练"
    )

    return parser.parse_args()


def validate_config(config: dict, logger: logging.Logger) -> bool:
    """验证配置文件"""
    logger.info("验证配置文件...")

    required_sections = ["data", "environment", "model", "training"]
    for section in required_sections:
        if section not in config:
            logger.error(f"配置文件缺少必要部分: {section}")
            return False

    # 验证数据配置
    data_config = config["data"]
    required_data_fields = ["start_time", "end_time", "train_start", "train_end",
                           "valid_start", "valid_end", "test_start", "test_end"]
    for field in required_data_fields:
        if field not in data_config:
            logger.error(f"数据配置缺少字段: {field}")
            return False

    # 验证时间顺序
    try:
        from datetime import datetime
        times = [datetime.strptime(data_config[field], "%Y-%m-%d") for field in required_data_fields]
        if not all(times[i] <= times[i+1] for i in range(0, len(times), 2)):
            logger.error("时间配置顺序错误")
            return False
    except ValueError as e:
        logger.error(f"时间格式错误: {e}")
        return False

    # 验证模型配置
    model_config = config["model"]
    if model_config.get("algorithm") not in ["SAC", "PPO"]:
        logger.error(f"不支持的算法: {model_config.get('algorithm')}")
        return False

    logger.info("配置文件验证通过")
    return True


def check_dependencies(logger: logging.Logger) -> bool:
    """检查依赖项"""
    logger.info("检查依赖项...")

    required_packages = [
        "qlib", "stable_baselines3", "torch", "gymnasium",
        "numpy", "pandas", "matplotlib", "yaml"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"缺少依赖包: {missing_packages}")
        logger.error("请运行: pip install -r requirements.txt")
        return False

    # 检查GPU可用性
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"检测到GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("未检测到GPU，将使用CPU")
    except Exception as e:
        logger.warning(f"GPU检查失败: {e}")

    logger.info("依赖项检查完成")
    return True


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 设置日志
    logger = setup_logging(args.log_level)

    try:
        logger.info("=" * 80)
        logger.info("强化学习投资策略训练系统启动")
        logger.info("=" * 80)
        logger.info(f"配置文件: {args.config}")
        logger.info(f"运行模式: {args.mode}")
        logger.info(f"日志级别: {args.log_level}")

        # 检查依赖
        if not check_dependencies(logger):
            logger.error("依赖项检查失败，退出")
            return 1

        # 加载配置
        logger.info("加载配置文件...")
        if not os.path.exists(args.config):
            logger.error(f"配置文件不存在: {args.config}")
            return 1

        config = load_config(args.config)

        # 命令行参数覆盖配置
        if args.seed:
            config.setdefault("training", {})["seed"] = args.seed
            logger.info(f"使用命令行种子: {args.seed}")

        if args.gpu:
            config.setdefault("model", {})["device"] = "cuda"
            logger.info("强制使用GPU")

        # 验证配置
        if not validate_config(config, logger):
            logger.error("配置验证失败，退出")
            return 1

        # 干运行模式
        if args.dry_run:
            logger.info("干运行模式，配置检查完成，退出")
            return 0

        # 创建训练器
        logger.info("创建训练器...")
        trainer = RLTrainer(config)

        # 根据模式运行
        if args.mode == "train":
            logger.info("开始训练模式...")
            # 初始化数据和环境
            train_data, valid_data, test_data = trainer.initialize_data()
            trainer.create_environments(train_data, valid_data)
            trainer.create_model()
            # 仅训练
            trainer.train()

        elif args.mode == "evaluate":
            logger.info("开始评估模式...")
            if not args.model_path:
                logger.error("评估模式需要指定模型路径")
                return 1

            if not os.path.exists(args.model_path):
                logger.error(f"模型文件不存在: {args.model_path}")
                return 1

            # 初始化数据用于评估
            train_data, valid_data, test_data = trainer.initialize_data()
            trainer.create_environments(train_data, valid_data)

            # 评估模型
            results = trainer.evaluate(model_path=args.model_path, test_data=test_data)
            logger.info("评估完成")

            # 输出关键指标
            perf = results.get("portfolio_performance", {})
            logger.info("=" * 50)
            logger.info("评估结果摘要:")
            logger.info(f"总收益率: {perf.get('total_return', 0):.2%}")
            logger.info(f"年化收益率: {perf.get('annualized_return', 0):.2%}")
            logger.info(f"最大回撤: {perf.get('max_drawdown', 0):.2%}")
            logger.info(f"夏普比率: {perf.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Calmar比率: {perf.get('calmar_ratio', 0):.3f}")
            logger.info("=" * 50)

        elif args.mode == "full":
            logger.info("开始完整流水线...")
            results = trainer.run_full_pipeline()

            # 输出训练结果摘要
            evaluation_results = results.get("evaluation_results", {})
            perf = evaluation_results.get("portfolio_performance", {})

            logger.info("=" * 80)
            logger.info("训练完成！结果摘要:")
            logger.info("-" * 40)
            logger.info(f"训练时间: {results.get('training_time', 'N/A')}")
            logger.info(f"平均回合奖励: {evaluation_results.get('mean_reward', 0):.4f}")
            logger.info(f"总收益率: {perf.get('total_return', 0):.2%}")
            logger.info(f"年化收益率: {perf.get('annualized_return', 0):.2%}")
            logger.info(f"最大回撤: {perf.get('max_drawdown', 0):.2%}")
            logger.info(f"夏普比率: {perf.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Calmar比率: {perf.get('calmar_ratio', 0):.3f}")
            logger.info("=" * 80)

        logger.info("程序执行完成")
        return 0

    except KeyboardInterrupt:
        logger.info("收到中断信号，正在退出...")
        return 0

    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)
        return 1

    finally:
        logger.info("清理资源...")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)