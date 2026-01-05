#!/usr/bin/env python3
"""多竞技场并行训练脚本

用于运行多个独立竞技场的并行训练，通过多进程架构绕过 GIL 限制。
支持定期迁移 Agent 基因组，促进跨竞技场的策略交流。

默认配置: 16 个竞技场并行

使用方法:
    python scripts/train_multi_arena.py [选项]

示例:
    # 默认 16 个竞技场训练 100 个 episode
    python scripts/train_multi_arena.py --episodes 100

    # 自定义竞技场数量
    python scripts/train_multi_arena.py --num-arenas 10 --episodes 100

    # 自定义迁移参数
    python scripts/train_multi_arena.py --migration-interval 20 --migration-count 10

    # 从检查点恢复
    python scripts/train_multi_arena.py --resume checkpoints/multi_arena_ep_50.pkl
"""

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

# 关键：在导入任何项目模块之前，先清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.log_engine.logger import setup_logging
from src.training.arena import ArenaManager, MultiArenaConfig, MigrationStrategy

from create_config import create_default_config


def progress_callback(state: dict[str, Any]) -> None:
    """训练进度回调函数

    Args:
        state: 训练状态字典
    """
    episode = state.get("episode", 0)
    total_episodes = state.get("total_episodes", 0)
    num_arenas = state.get("total_arenas", 0)
    avg_volatility = state.get("avg_volatility", 0.0)
    avg_volume = state.get("avg_volume", 0.0)
    avg_tick_count = state.get("avg_tick_count", 0.0)

    info_parts = [f"Episode {episode}/{total_episodes}"]
    info_parts.append(f"Arenas: {num_arenas}")
    info_parts.append(f"Volatility: {avg_volatility:.4f}")
    info_parts.append(f"Volume: {avg_volume:.0f}")
    info_parts.append(f"Ticks: {avg_tick_count:.0f}")

    print(" | ".join(info_parts))


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NEAT AI 交易模拟竞技场 - 多竞技场并行训练模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 基本参数
    parser.add_argument(
        "--episodes",
        type=int,
        default=4000,
        help="训练的 episode 数量（默认: 4000）",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100,
        help="每个 episode 的 tick 数量（默认: 100）",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="检查点保存间隔（episode 数，默认: 50，0 表示不保存）",
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

    # 鲶鱼参数
    parser.add_argument(
        "--catfish",
        action="store_const",
        const=True,
        default=None,
        help="启用鲶鱼机制",
    )
    parser.add_argument(
        "--catfish-fund-multiplier",
        type=float,
        default=3.0,
        help="鲶鱼资金倍数（默认: 3.0）",
    )

    # 多竞技场参数
    parser.add_argument(
        "--num-arenas",
        type=int,
        default=16,
        help="竞技场数量（默认: 16）",
    )
    parser.add_argument(
        "--migration-interval",
        type=int,
        default=10,
        help="迁移间隔（episode 数，默认: 10）",
    )
    parser.add_argument(
        "--migration-count",
        type=int,
        default=5,
        help="每次迁移的 Agent 数量（每种群，默认: 5）",
    )
    parser.add_argument(
        "--migration-best-ratio",
        type=float,
        default=0.5,
        help="迁移最好个体的比例（默认: 0.5，即一半最好一半最差）",
    )
    parser.add_argument(
        "--migration-strategy",
        type=str,
        default="ring",
        choices=["ring", "random", "best_to_worst"],
        help="迁移策略（默认: ring）",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    # 打印配置
    print("=" * 70)
    print("NEAT AI 交易模拟竞技场 - 多竞技场并行训练模式")
    print("=" * 70)
    print(f"竞技场数量: {args.num_arenas}")
    print(f"Episodes: {args.episodes}")
    print(f"Episode Length: {args.episode_length} ticks")
    print(f"Checkpoint Interval: {args.checkpoint_interval}")
    print(f"迁移间隔: 每 {args.migration_interval} 个 episode")
    print(f"迁移数量: {args.migration_count} 个 Agent/种群")
    print(f"迁移策略: {args.migration_strategy}")
    if args.resume:
        print(f"Resume From: {args.resume}")

    # 创建基础配置
    config_kwargs = {
        "episode_length": args.episode_length,
        "checkpoint_interval": args.checkpoint_interval,
        "config_dir": args.config_dir,
        "catfish_fund_multiplier": args.catfish_fund_multiplier,
    }
    if args.catfish is not None:
        config_kwargs["catfish_enabled"] = args.catfish

    config = create_default_config(**config_kwargs)

    # 打印 Catfish 状态
    if config.catfish is not None and config.catfish.enabled:
        print(f"Catfish: enabled, multiplier={config.catfish.fund_multiplier}x")
    else:
        print("Catfish: disabled")
    print("=" * 70)

    # 解析迁移策略
    strategy_map = {
        "ring": MigrationStrategy.RING,
        "random": MigrationStrategy.RANDOM,
        "best_to_worst": MigrationStrategy.BEST_TO_WORST,
    }
    migration_strategy = strategy_map[args.migration_strategy]

    # 创建多竞技场配置
    multi_config = MultiArenaConfig(
        num_arenas=args.num_arenas,
        base_config=config,
        migration_interval=args.migration_interval,
        migration_count=args.migration_count,
        migration_best_ratio=args.migration_best_ratio,
        migration_strategy=migration_strategy,
        checkpoint_interval=args.checkpoint_interval,
    )

    # 创建竞技场管理器
    manager = ArenaManager(multi_config)

    # 初始化
    print("创建竞技场进程...")
    start_time = time.time()
    manager.setup()
    setup_time = time.time() - start_time
    print(f"竞技场进程创建完成（耗时: {setup_time:.2f}s）")

    # 启动竞技场进程
    print("启动竞技场进程...")
    start_time = time.time()
    manager.start()
    start_time_elapsed = time.time() - start_time
    print(f"所有竞技场启动完成（耗时: {start_time_elapsed:.2f}s）")

    # 恢复检查点
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"正在从检查点恢复: {args.resume}")
            manager.load_checkpoint(args.resume)
            print("检查点已恢复")
        else:
            print(f"错误: 检查点文件不存在: {args.resume}")
            manager.stop()
            sys.exit(1)

    # 开始训练
    print("\n开始多竞技场训练...")
    print("-" * 70)

    train_start = time.time()
    try:
        manager.train(
            episodes=args.episodes,
            progress_callback=progress_callback,
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        # 保存紧急检查点
        emergency_path = "checkpoints/multi_arena_emergency.pkl"
        manager.save_checkpoint(emergency_path)
        print(f"紧急检查点已保存: {emergency_path}")
    finally:
        # 确保停止所有进程
        manager.stop()

    train_time = time.time() - train_start

    # 打印统计
    print("-" * 70)
    print("多竞技场训练完成！")
    print(f"竞技场数量: {args.num_arenas}")
    print(f"总 Episode: {args.episodes}")
    print(f"总耗时: {train_time:.2f}s")
    print(f"平均每 Episode: {train_time / max(args.episodes, 1):.2f}s")
    print(f"平均每竞技场每 Episode: {train_time / max(args.episodes * args.num_arenas, 1):.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
