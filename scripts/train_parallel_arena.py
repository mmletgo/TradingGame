#!/usr/bin/env python3
"""多竞技场并行推理训练脚本

使用 ParallelArenaTrainer 进行训练，多个竞技场的神经网络推理并行执行，
只有交易配对和账户更新串行执行。

与 SingleArenaTrainer 的区别：
- SingleArenaTrainer: 串行遍历竞技场，每个竞技场内使用 OpenMP 并行推理
- ParallelArenaTrainer: 合并所有竞技场的推理请求为一个大批量，OpenMP 并行执行

使用方法:
    python scripts/train_parallel_arena.py [选项]

示例:
    # 默认训练（10个竞技场，每个10个episode，共100轮）
    python scripts/train_parallel_arena.py --rounds 100

    # 自定义竞技场数量和episode数
    python scripts/train_parallel_arena.py --num-arenas 8 --episodes-per-arena 5 --rounds 200

    # 从检查点恢复
    python scripts/train_parallel_arena.py --resume checkpoints/parallel_arena_gen_50.pkl
"""

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

# 清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.log_engine.logger import setup_logging
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

from create_config import create_default_config


def progress_callback(stats: dict[str, Any]) -> None:
    """训练进度回调函数"""
    from datetime import datetime

    generation = stats.get("generation", 0)
    total_episodes = stats.get("total_episodes", 0)
    round_time = stats.get("total_time", 0.0)
    avg_fitnesses = stats.get("avg_fitnesses", {})

    current_time = datetime.now().strftime("%H:%M:%S")

    # 格式化适应度信息
    fitness_info: list[str] = []
    for (agent_type, _), fitness in avg_fitnesses.items():
        avg = float(fitness.mean()) if hasattr(fitness, "mean") else float(fitness)
        fitness_info.append(f"{agent_type.value}={avg:.4f}")

    fitness_str = ", ".join(fitness_info[:4])  # 只显示前4个

    print(
        f"Gen {generation:4d} | {current_time} | "
        f"Episodes={total_episodes:6d} | "
        f"Time={round_time:.1f}s | "
        f"{fitness_str}"
    )


def episode_callback(stats: dict[str, Any]) -> None:
    """Episode 完成回调函数"""
    from datetime import datetime

    episode = stats.get("episode", 0)
    high_price = stats.get("high_price", 0.0)
    low_price = stats.get("low_price", 0.0)
    num_arenas = stats.get("num_arenas", 1)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(
        f"  Episode {episode:4d} | {current_time} | "
        f"Arenas={num_arenas} | "
        f"high={high_price:.2f}, low={low_price:.2f}"
    )


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NEAT AI 交易模拟竞技场 - 多竞技场并行推理训练模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="训练轮数（每轮 = 所有竞技场完成 + 一次进化）。默认无限",
    )
    parser.add_argument(
        "--num-arenas",
        type=int,
        default=25,
        help="竞技场数量（默认: 25）",
    )
    parser.add_argument(
        "--episodes-per-arena",
        type=int,
        default=2,
        help="每个竞技场运行的 episode 数（默认: 2）",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100,
        help="每个 episode 的 tick 数量（默认: 1000）",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="检查点保存间隔（轮数，默认: 10，0 表示不保存）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从指定检查点恢复训练",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="配置文件目录（默认: config）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="日志目录（默认: logs）",
    )
    parser.add_argument(
        "--catfish",
        action="store_const",
        const=True,
        default=None,
        help="启用鲶鱼机制。不指定时使用默认值",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    # 计算总 episode 数
    episodes_per_round = args.num_arenas * args.episodes_per_arena

    print("=" * 70)
    print("NEAT AI 交易模拟竞技场 - 多竞技场并行推理训练模式")
    print("=" * 70)
    print(f"竞技场数量: {args.num_arenas}")
    print(f"每竞技场 Episode 数: {args.episodes_per_arena}")
    print(f"每轮总 Episode 数: {episodes_per_round}")
    if args.rounds:
        print(f"训练轮数: {args.rounds}")
        print(f"总 Episode 数: {args.rounds * episodes_per_round}")
    else:
        print("训练轮数: 无限模式（Ctrl+C 停止）")
    print(f"Episode 长度: {args.episode_length} ticks")
    print(f"检查点间隔: {args.checkpoint_interval} 轮")
    if args.resume:
        print(f"恢复检查点: {args.resume}")
    print("=" * 70)

    # 创建配置
    config_kwargs: dict[str, Any] = {
        "episode_length": args.episode_length,
        "config_dir": args.config_dir,
    }
    if args.catfish is not None:
        config_kwargs["catfish_enabled"] = args.catfish

    config = create_default_config(**config_kwargs)

    # 设置多竞技场配置到 config
    config.training.num_arenas = args.num_arenas
    config.training.episodes_per_arena = args.episodes_per_arena

    # 打印 Catfish 状态
    if config.catfish is not None and config.catfish.enabled:
        print("Catfish: enabled")
    else:
        print("Catfish: disabled")
    print("=" * 70)

    # 创建多竞技场配置
    multi_config = MultiArenaConfig(
        num_arenas=args.num_arenas,
        episodes_per_arena=args.episodes_per_arena,
    )

    # 创建训练器
    trainer = ParallelArenaTrainer(config, multi_config)

    # 初始化
    print("\n初始化训练环境...")
    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 恢复检查点
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"正在从检查点恢复: {args.resume}")
            trainer.load_checkpoint(args.resume)
            print(f"已恢复到 Generation {trainer.generation}")
        else:
            print(f"错误: 检查点文件不存在: {args.resume}")
            sys.exit(1)

    # 定义检查点回调
    def checkpoint_callback(generation: int) -> None:
        if args.checkpoint_interval > 0 and generation % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/parallel_arena_gen_{generation}.pkl"
            trainer.save_checkpoint(checkpoint_path)
            print(f"  [检查点已保存: {checkpoint_path}]")

    # 开始训练
    print("\n开始训练...")
    print("-" * 70)

    train_start = time.time()
    try:
        trainer.train(
            num_rounds=args.rounds,
            checkpoint_callback=checkpoint_callback,
            progress_callback=progress_callback,
            episode_callback=episode_callback,
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        # 保存紧急检查点
        emergency_path = (
            f"checkpoints/emergency_parallel_arena_gen_{trainer.generation}.pkl"
        )
        trainer.save_checkpoint(emergency_path)
        print(f"紧急检查点已保存: {emergency_path}")
    finally:
        trainer.stop()

    train_time = time.time() - train_start
    total_episodes = trainer.total_episodes

    print("-" * 70)
    print("训练完成！")
    print(f"总 Generation: {trainer.generation}")
    print(f"总 Episode: {total_episodes}")
    print(f"总耗时: {train_time:.2f}s")
    if trainer.generation > 0:
        print(f"平均每轮: {train_time / trainer.generation:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
