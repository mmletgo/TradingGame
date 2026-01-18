#!/usr/bin/env python3
"""联盟训练启动脚本

基于 AlphaStar 联盟训练思路的 NEAT 进化训练。

用法:
    python scripts/train_league.py --rounds 200
    python scripts/train_league.py --resume checkpoints/league_training/gen_100.pkl
    python scripts/train_league.py --no-league-exploiter
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.create_config import create_default_config
from src.core.log_engine import get_logger, setup_logging
from src.training.arena import MultiArenaConfig
from src.training.league import LeagueTrainingConfig
from src.training.league.league_trainer import LeagueTrainer


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="联盟训练 - 基于 AlphaStar 联盟训练思路",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 基本参数
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="训练轮数，不指定则无限训练",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练",
    )

    # 竞技场配置
    parser.add_argument(
        "--num-arenas",
        type=int,
        default=27,
        help="竞技场数量（默认 27）",
    )
    parser.add_argument(
        "--episodes-per-arena",
        type=int,
        default=2,
        help="每个竞技场的 episode 数（默认 2）",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="每个 episode 的 tick 数（默认 1000）",
    )

    # 联盟训练配置
    parser.add_argument(
        "--no-league-exploiter",
        action="store_true",
        help="禁用 League Exploiter",
    )
    parser.add_argument(
        "--no-main-exploiter",
        action="store_true",
        help="禁用 Main Exploiter",
    )
    parser.add_argument(
        "--milestone-interval",
        type=int,
        default=50,
        help="里程碑保存间隔（默认每 50 代）",
    )
    parser.add_argument(
        "--exploiter-ratio",
        type=float,
        default=0.1,
        help="Exploiter 种群占 Main 的比例（默认 0.1）",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["uniform", "pfsp", "diverse"],
        default="pfsp",
        help="对手采样策略（默认 pfsp）",
    )

    # 检查点配置
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/league_training",
        help="检查点目录（默认 checkpoints/league_training）",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="检查点保存间隔（默认每 10 代）",
    )

    # 其他
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别（默认 INFO）",
    )

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    # 设置日志
    setup_logging(level=args.log_level)
    logger = get_logger("train_league")

    logger.info("=" * 60)
    logger.info("联盟训练 - 基于 AlphaStar 联盟训练思路")
    logger.info("=" * 60)

    # 创建配置
    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=args.checkpoint_interval,
    )

    # 多竞技场配置
    multi_config = MultiArenaConfig(
        num_arenas=args.num_arenas,
        episodes_per_arena=args.episodes_per_arena,
    )

    # 联盟训练配置
    league_config = LeagueTrainingConfig(
        pool_dir=f"{args.checkpoint_dir}/opponent_pools",
        milestone_interval=args.milestone_interval,
        enable_league_exploiter=not args.no_league_exploiter,
        enable_main_exploiter=not args.no_main_exploiter,
        exploiter_population_ratio=args.exploiter_ratio,
        sampling_strategy=args.sampling_strategy,
        num_arenas=args.num_arenas,
        episodes_per_arena=args.episodes_per_arena,
    )

    # 验证配置
    league_config.validate()

    # 打印配置
    logger.info(f"竞技场数量: {args.num_arenas}")
    logger.info(f"每竞技场 episode 数: {args.episodes_per_arena}")
    logger.info(f"Episode 长度: {args.episode_length} ticks")
    logger.info(f"League Exploiter: {'启用' if not args.no_league_exploiter else '禁用'}")
    logger.info(f"Main Exploiter: {'启用' if not args.no_main_exploiter else '禁用'}")
    logger.info(f"采样策略: {args.sampling_strategy}")
    logger.info(f"检查点目录: {args.checkpoint_dir}")

    # 确保检查点目录存在
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建训练器
    trainer = LeagueTrainer(config, multi_config, league_config)

    try:
        # 初始化
        trainer.setup()

        # 恢复检查点
        if args.resume:
            logger.info(f"从检查点恢复: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # 定义回调
        def checkpoint_callback(generation: int) -> None:
            checkpoint_path = checkpoint_dir / f"gen_{generation:05d}.pkl.gz"
            trainer.save_checkpoint(str(checkpoint_path))
            logger.info(f"保存检查点: {checkpoint_path}")

        def progress_callback(stats: dict) -> None:
            generation = stats.get('generation', 0)
            pool_sizes = stats.get('pool_sizes', {})
            logger.info(
                f"[Gen {generation}] "
                f"对手池: {sum(pool_sizes.values())} 条目"
            )

        # 开始训练
        trainer.train(
            num_rounds=args.rounds,
            checkpoint_callback=checkpoint_callback,
            progress_callback=progress_callback,
        )

    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.exception(f"训练出错: {e}")
        raise
    finally:
        trainer.stop()
        logger.info("训练器已停止")


if __name__ == "__main__":
    main()
