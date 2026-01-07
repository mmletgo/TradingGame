#!/usr/bin/env python3
"""多竞技场并行训练脚本

用于运行多个独立竞技场的并行训练，通过多进程架构绕过 GIL 限制。
支持定期迁移 Agent 基因组，促进跨竞技场的策略交流。

新架构特点：
- 每个竞技场独立进化，不等待其他竞技场
- 主进程只负责监控和收集状态
- 迁移通过共享检查点实现，各竞技场自主读写
- 默认自动从最新检查点恢复训练

默认配置: 12 个竞技场并行，自动恢复启用

使用方法:
    python scripts/train_multi_arena.py [选项]

示例:
    # 默认配置训练（自动恢复）
    python scripts/train_multi_arena.py --episodes 100

    # 禁用自动恢复，从头开始训练
    python scripts/train_multi_arena.py --no-resume --episodes 100

    # 自定义竞技场数量
    python scripts/train_multi_arena.py --num-arenas 10 --episodes 100

    # 自定义迁移参数
    python scripts/train_multi_arena.py --migration-interval 20 --checkpoint-interval 50
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
    from datetime import datetime

    avg_volatility = state.get("avg_volatility", 0.0)

    # 获取各竞技场的 episode 进度
    arena_episodes = state.get("arena_episodes", {})
    if arena_episodes:
        min_ep = min(arena_episodes.values())
        max_ep = max(arena_episodes.values())
        ep_range = f"{min_ep}-{max_ep}"
    else:
        ep_range = "0"

    # 获取当前时间（时分秒）
    current_time = datetime.now().strftime("%H:%M:%S")

    info_parts = [f"[{current_time}]", f"Episodes: {ep_range}"]
    info_parts.append(f"Volatility: {avg_volatility:.4f}")

    # 获取各物种最精英 species 的平均适应度
    elite_fitness = state.get("elite_species_fitness", {})
    if elite_fitness:
        # 使用缩写：R=retail, P=retail_pro, W=whale, M=market_maker
        abbr_map = {
            "retail": "R",
            "retail_pro": "P",
            "whale": "W",
            "market_maker": "M",
        }
        fitness_parts = []
        for type_key, fitness in elite_fitness.items():
            # 规范化 key（处理大小写）
            normalized_key = type_key.lower() if isinstance(type_key, str) else str(type_key).lower()
            abbr = abbr_map.get(normalized_key, normalized_key[:1].upper())
            fitness_parts.append(f"{abbr}={fitness:.2f}")
        if fitness_parts:
            info_parts.append(f"Elite: {' '.join(fitness_parts)}")

    print(" | ".join(info_parts))


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NEAT AI 交易模拟竞技场 - 多竞技场并行训练模式（异步监控）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 基本参数
    parser.add_argument(
        "--episodes",
        type=int,
        default=4000,
        help="每个竞技场的最大 episode 数量（默认: 4000）",
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
        default=10,
        help="检查点保存间隔（episode 数，默认: 10，0 表示不保存）",
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
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/multi_arena",
        help="检查点目录（各竞技场独立文件模式，默认: checkpoints/multi_arena）",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=1.0,
        help="监控轮询间隔（秒，默认: 1.0）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用自动从检查点恢复（默认: 自动恢复）",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    # 打印配置
    print("=" * 70)
    print("NEAT AI 交易模拟竞技场 - 多竞技场并行训练模式（异步监控）")
    print("=" * 70)
    print(f"竞技场数量: {args.num_arenas}")
    print(f"每个竞技场最大 Episodes: {args.episodes}")
    print(f"Episode Length: {args.episode_length} ticks")
    print(f"Checkpoint Interval: {args.checkpoint_interval}")
    print(f"迁移间隔: 每 {args.migration_interval} 个 episode")
    print(f"迁移数量: {args.migration_count} 个 Agent/种群")
    print(f"迁移策略: {args.migration_strategy}")
    print(f"检查点目录: {args.checkpoint_dir}")
    print(f"监控间隔: {args.monitor_interval}s")
    print(f"自动恢复: {'禁用' if args.no_resume else '启用'}")

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
        max_episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
    )

    # 创建竞技场管理器
    manager = ArenaManager(multi_config, auto_resume=not args.no_resume)

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

    # 开始训练
    print("\n开始多竞技场训练（异步监控模式）...")
    print("每个竞技场独立进化，通过共享检查点进行迁移")
    print("-" * 70)

    train_start = time.time()
    try:
        manager.monitor(
            progress_callback=progress_callback,
            check_interval=args.monitor_interval,
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    finally:
        # 确保停止所有进程
        manager.stop()

    train_time = time.time() - train_start

    # 获取最终统计
    summary = manager.get_summary()
    arena_episodes = summary.get("arena_episodes", {})
    total_episodes_run = sum(arena_episodes.values()) if arena_episodes else 0

    # 打印统计
    print("-" * 70)
    print("多竞技场训练完成！")
    print(f"竞技场数量: {args.num_arenas}")
    print(f"各竞技场完成的 Episode: {arena_episodes}")
    print(f"总 Episode 数: {total_episodes_run}")
    print(f"总耗时: {train_time:.2f}s")
    if total_episodes_run > 0:
        print(f"平均每 Episode: {train_time / total_episodes_run:.3f}s")
    print(f"检查点目录: {args.checkpoint_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
