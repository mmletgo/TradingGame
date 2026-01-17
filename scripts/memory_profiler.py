#!/usr/bin/env python3
"""内存泄漏分析脚本

用于定位多竞技场模式的内存泄漏问题。
在关键位置打印内存使用情况，帮助确定泄漏源。

使用方法:
    python scripts/memory_profiler.py --rounds 20
"""

import argparse
import gc
import os
import sys
import tracemalloc
from pathlib import Path
from typing import Any

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_memory_mb() -> float:
    """获取当前进程的 RSS 内存使用量 (MB)"""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # KB -> MB
    except Exception:
        pass
    return 0.0


def get_gc_stats() -> dict[str, int]:
    """获取 GC 统计信息"""
    stats = gc.get_stats()
    return {
        "gen0_collections": stats[0]["collections"],
        "gen1_collections": stats[1]["collections"],
        "gen2_collections": stats[2]["collections"],
        "gen0_uncollectable": stats[0]["uncollectable"],
        "gen1_uncollectable": stats[1]["uncollectable"],
        "gen2_uncollectable": stats[2]["uncollectable"],
    }


def print_memory_snapshot(label: str, prev_mem: float = 0.0) -> float:
    """打印内存快照"""
    current_mem = get_memory_mb()
    delta = current_mem - prev_mem if prev_mem > 0 else 0
    delta_str = f" ({delta:+.1f} MB)" if prev_mem > 0 else ""
    print(f"[MEM] {label}: {current_mem:.1f} MB{delta_str}")
    return current_mem


def print_object_counts(label: str = "") -> None:
    """打印主要对象类型的计数"""
    # 强制执行完整 GC
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    counts: dict[str, int] = {}
    for obj in gc.get_objects():
        type_name = type(obj).__name__
        counts[type_name] = counts.get(type_name, 0) + 1

    # 打印可能泄漏的对象类型
    suspect_types = [
        "DefaultGenome",
        "NodeGene",
        "ConnectionGene",
        "Species",
        "dict",
        "list",
        "ndarray",
        "AgentAccountState",
        "ArenaState",
    ]

    print(f"[OBJ] Object counts {label}:")
    for t in suspect_types:
        if t in counts:
            print(f"      {t}: {counts[t]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="内存泄漏分析工具")
    parser.add_argument("--rounds", type=int, default=20, help="训练轮数")
    parser.add_argument("--num-arenas", type=int, default=10, help="竞技场数量")
    parser.add_argument("--episodes-per-arena", type=int, default=2, help="每竞技场 episode 数")
    parser.add_argument("--episode-length", type=int, default=50, help="Episode 长度")
    parser.add_argument("--trace", action="store_true", help="启用 tracemalloc 追踪")
    parser.add_argument("--checkpoint-test", action="store_true", help="测试 checkpoint 内存")
    args = parser.parse_args()

    # 启用 tracemalloc（可选，会有性能开销）
    if args.trace:
        tracemalloc.start(25)
        print("[TRACE] tracemalloc 已启用")

    print("=" * 70)
    print("内存泄漏分析工具")
    print("=" * 70)
    print(f"配置: {args.num_arenas} 竞技场 x {args.episodes_per_arena} episodes x {args.episode_length} ticks")
    print(f"训练轮数: {args.rounds}")
    print("=" * 70)

    from src.core.log_engine.logger import setup_logging
    from src.training.arena import MultiArenaConfig, ParallelArenaTrainer
    from create_config import create_default_config

    setup_logging("logs")

    # 初始内存
    mem_start = print_memory_snapshot("初始状态")

    # 创建配置
    config = create_default_config(
        episode_length=args.episode_length,
        config_dir="config",
        catfish_enabled=False,  # 禁用鲶鱼简化测试
    )
    config.training.num_arenas = args.num_arenas
    config.training.episodes_per_arena = args.episodes_per_arena

    multi_config = MultiArenaConfig(
        num_arenas=args.num_arenas,
        episodes_per_arena=args.episodes_per_arena,
    )

    mem_after_config = print_memory_snapshot("配置创建后", mem_start)

    # 创建训练器
    trainer = ParallelArenaTrainer(config, multi_config)
    mem_after_trainer = print_memory_snapshot("训练器创建后", mem_after_config)

    # 初始化
    trainer.setup()
    mem_after_setup = print_memory_snapshot("初始化后", mem_after_trainer)
    print_object_counts("setup 后")

    # 基准内存
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)

    try:
        # 导入 malloc_trim
        from src.training.population import malloc_trim
    except ImportError:
        malloc_trim = lambda: None

    malloc_trim()
    mem_baseline = print_memory_snapshot("GC 后基准", mem_after_setup)

    print("\n" + "-" * 70)
    print("开始训练循环...")
    print("-" * 70)

    mem_history: list[tuple[int, float, float]] = []

    try:
        for round_idx in range(args.rounds):
            mem_before_round = get_memory_mb()

            # 运行一轮
            stats = trainer.run_round()

            mem_after_round = get_memory_mb()
            mem_delta = mem_after_round - mem_before_round

            # 每轮后的内存
            print(f"[MEM] Round {round_idx + 1}: {mem_after_round:.1f} MB (delta: {mem_delta:+.1f} MB)")
            mem_history.append((round_idx + 1, mem_after_round, mem_delta))

            # 测试 checkpoint 保存的内存影响
            if args.checkpoint_test and (round_idx + 1) % 5 == 0:
                mem_before_save = get_memory_mb()
                print(f"[MEM] === 开始保存 checkpoint (round {round_idx + 1}) ===")

                # 保存 checkpoint
                checkpoint_path = f"/tmp/test_checkpoint_gen_{round_idx + 1}.pkl"
                trainer.save_checkpoint(checkpoint_path)

                mem_after_save = get_memory_mb()
                print(f"[MEM] Checkpoint 保存后: {mem_after_save:.1f} MB (delta: {mem_after_save - mem_before_save:+.1f} MB)")

                # 清理测试文件
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)

                # GC 并检查
                gc.collect(0)
                gc.collect(1)
                gc.collect(2)
                malloc_trim()
                mem_after_gc = get_memory_mb()
                print(f"[MEM] GC 后: {mem_after_gc:.1f} MB (delta from save: {mem_after_gc - mem_before_save:+.1f} MB)")
                print_object_counts(f"round {round_idx + 1}")

            # 每 5 轮打印一次详细对象统计
            if (round_idx + 1) % 5 == 0 and not args.checkpoint_test:
                gc.collect(0)
                gc.collect(1)
                gc.collect(2)
                malloc_trim()
                print_object_counts(f"round {round_idx + 1}")

    except KeyboardInterrupt:
        print("\n训练被中断")
    finally:
        trainer.stop()

    print("\n" + "=" * 70)
    print("内存分析结果")
    print("=" * 70)

    if mem_history:
        first_mem = mem_history[0][1]
        last_mem = mem_history[-1][1]
        total_growth = last_mem - mem_baseline
        avg_growth_per_round = total_growth / len(mem_history) if mem_history else 0

        print(f"基准内存: {mem_baseline:.1f} MB")
        print(f"第 1 轮后: {first_mem:.1f} MB")
        print(f"最后一轮后: {last_mem:.1f} MB")
        print(f"总增长: {total_growth:.1f} MB")
        print(f"平均每轮增长: {avg_growth_per_round:.1f} MB")

        # 检查是否有明显泄漏
        if avg_growth_per_round > 50:  # 每轮增长超过 50MB
            print("\n[警告] 检测到明显的内存泄漏！")
            print("建议检查以下区域:")
            print("  1. NEAT 种群历史数据清理 (_cleanup_neat_history)")
            print("  2. Worker 进程内存清理 (_cleanup_worker_neat_history)")
            print("  3. genome 对象引用链")
        elif avg_growth_per_round > 10:  # 每轮增长超过 10MB
            print("\n[注意] 存在轻微的内存增长")
        else:
            print("\n[正常] 内存使用稳定")

    # tracemalloc 报告
    if args.trace:
        print("\n" + "-" * 70)
        print("tracemalloc Top 20 内存分配")
        print("-" * 70)
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("lineno")
        for stat in stats[:20]:
            print(stat)

    print("\n分析完成")


if __name__ == "__main__":
    main()
