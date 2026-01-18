#!/usr/bin/env python3
"""联盟训练启动脚本

基于 AlphaStar 联盟训练思路的 NEAT 进化训练。

用法:
    python scripts/train_league.py                    # 无限训练，自动加载最新检查点
    python scripts/train_league.py --rounds 200      # 指定训练轮数
    python scripts/train_league.py --fresh           # 从头开始，不加载检查点
    python scripts/train_league.py --no-league-exploiter
"""
import argparse
import re
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
from src.config.config import AgentType


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
        help="从指定检查点恢复训练",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="从头开始训练，不自动加载检查点",
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


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """查找最新的检查点文件

    Args:
        checkpoint_dir: 检查点目录

    Returns:
        最新检查点路径，如果没有则返回 None
    """
    if not checkpoint_dir.exists():
        return None

    # 支持的检查点格式：gen_00100.pkl.gz, gen_00100.pkl
    checkpoint_files: list[tuple[int, Path]] = []

    for pattern in ["gen_*.pkl.gz", "gen_*.pkl"]:
        for f in checkpoint_dir.glob(pattern):
            # 从文件名提取代数
            match = re.search(r"gen_(\d+)", f.name)
            if match:
                gen = int(match.group(1))
                checkpoint_files.append((gen, f))

    if not checkpoint_files:
        return None

    # 按代数排序，返回最新的
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    return checkpoint_files[0][1]


def episode_callback(stats: dict) -> None:
    """Episode 完成回调函数"""
    from datetime import datetime

    episode = stats.get("episode", 0)
    arena_high_prices = stats.get("arena_high_prices", [])
    arena_low_prices = stats.get("arena_low_prices", [])
    arena_end_reasons = stats.get("arena_end_reasons", [])
    arena_end_ticks = stats.get("arena_end_ticks", [])

    current_time = datetime.now().strftime("%H:%M:%S")

    # 格式化各竞技场的 High/Low 和结束原因
    arena_price_strs: list[str] = []
    for i, (high, low) in enumerate(zip(arena_high_prices, arena_low_prices)):
        end_reason = arena_end_reasons[i] if i < len(arena_end_reasons) else None
        end_tick = arena_end_ticks[i] if i < len(arena_end_ticks) else 0
        if end_reason is None:
            reason_str = f"ok@{end_tick}"
        else:
            # 简化结束原因显示
            reason_abbr = {
                "population_depleted:RETAIL": "pop:R",
                "population_depleted:RETAIL_PRO": "pop:RP",
                "population_depleted:WHALE": "pop:W",
                "population_depleted:MARKET_MAKER": "pop:MM",
                "one_sided_orderbook": "ob",
                "catfish": "cat",
            }
            reason_str = reason_abbr.get(end_reason, end_reason[:8])
            reason_str = f"{reason_str}@{end_tick}"
        arena_price_strs.append(f"A{i}:({high:.2f},{low:.2f})[{reason_str}]")

    prices_str = " ".join(arena_price_strs)
    print(f"  Episode {episode:4d} | {current_time} | {prices_str}")


def progress_callback(stats: dict) -> None:
    """训练进度回调函数"""
    from datetime import datetime

    import numpy as np

    generation = stats.get("generation", 0)
    total_episodes = stats.get("total_episodes", 0)
    round_time = stats.get("total_time", 0.0)
    species_fitness_stats = stats.get("species_fitness_stats", {})
    pool_sizes = stats.get("pool_sizes", {})

    current_time = datetime.now().strftime("%H:%M:%S")

    # 第一行：基础信息
    total_pool = sum(pool_sizes.values()) if pool_sizes else 0
    print(
        f"Gen {generation:4d} | {current_time} | "
        f"Episodes={total_episodes:6d} | "
        f"Time={round_time:.1f}s | "
        f"Pool={total_pool}"
    )

    # 后续行：各物种的 species 适应度分布
    type_order = [
        AgentType.RETAIL,
        AgentType.RETAIL_PRO,
        AgentType.WHALE,
        AgentType.MARKET_MAKER,
    ]
    for agent_type in type_order:
        type_stats = species_fitness_stats.get(agent_type, {})
        type_name = agent_type.value
        species_count = type_stats.get("species_count", 0)
        species_fitnesses = type_stats.get("species_avg_fitnesses", [])

        if species_fitnesses:
            arr = np.array(species_fitnesses)
            mean_val = float(arr.mean())
            std_val = float(arr.std())
            print(
                f"  {type_name}: species={species_count}, fitness={mean_val:.4f}±{std_val:.4f}"
            )
        else:
            print(f"  {type_name}: species={species_count}, fitness=N/A")


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
        resume_path: str | None = None

        if args.resume:
            # 用户指定了检查点
            resume_path = args.resume
        elif not args.fresh:
            # 自动查找最新检查点
            latest = find_latest_checkpoint(checkpoint_dir)
            if latest:
                resume_path = str(latest)
                logger.info(f"自动检测到最新检查点: {latest.name}")

        if resume_path:
            logger.info(f"从检查点恢复: {resume_path}")
            trainer.load_checkpoint(resume_path)
        else:
            logger.info("从头开始训练")

        # 定义检查点回调
        def checkpoint_callback(generation: int) -> None:
            checkpoint_path = checkpoint_dir / f"gen_{generation:05d}.pkl.gz"
            trainer.save_checkpoint(str(checkpoint_path))
            logger.info(f"保存检查点: {checkpoint_path}")

        # 开始训练
        trainer.train(
            num_rounds=args.rounds,
            checkpoint_callback=checkpoint_callback,
            progress_callback=progress_callback,
            episode_callback=episode_callback,
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
