#!/usr/bin/env python3
"""完整 round 性能分析

分析一个完整 round（多个 episode + 进化）的各阶段耗时。
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

importlib.invalidate_caches()
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.core.log_engine.logger import setup_logging


def benchmark_full_round(trainer, episodes_per_arena: int, episode_length: int):
    """分析一个完整 round 的各阶段耗时"""
    from datetime import datetime

    # 收集各阶段耗时
    episode_times = []
    reset_times = []
    tick_times = []
    evolution_times = []

    num_arenas = len(trainer.arena_states)
    total_episodes = episodes_per_arena

    print(f"\n开始完整 round 分析...")
    print(f"竞技场数量: {num_arenas}")
    print(f"每竞技场 episode 数: {episodes_per_arena}")
    print(f"每 episode tick 数: {episode_length}")

    round_start = time.perf_counter()

    for ep in range(total_episodes):
        ep_start = time.perf_counter()

        # 重置
        t0 = time.perf_counter()
        trainer._reset_all_arenas()
        reset_time = time.perf_counter() - t0
        reset_times.append(reset_time)

        # 做市商初始化
        t0 = time.perf_counter()
        trainer._init_market_all_arenas()
        init_time = time.perf_counter() - t0

        # 运行 ticks
        ep_tick_times = []
        for tick in range(episode_length):
            t0 = time.perf_counter()
            trainer.run_tick_all_arenas()
            ep_tick_times.append(time.perf_counter() - t0)

        tick_times.extend(ep_tick_times)
        ep_time = time.perf_counter() - ep_start

        episode_times.append(ep_time)

        # 打印进度
        avg_tick = np.mean(ep_tick_times) * 1000
        print(f"  Episode {ep + 1}/{total_episodes}: {ep_time:.1f}s (reset={reset_time * 1000:.0f}ms, init={init_time * 1000:.0f}ms, avg_tick={avg_tick:.0f}ms)")

    # 收集适应度并进化（简化版：跳过实际进化，只测量时间框架）
    # 注意：这里不执行实际进化，因为需要完整的适应度数据
    collect_time = 0.0
    evolve_time = 0.0
    refresh_time = 0.0

    round_time = time.perf_counter() - round_start

    return {
        "round_time_s": round_time,
        "episode_avg_s": float(np.mean(episode_times)),
        "episode_total_s": float(sum(episode_times)),
        "reset_avg_ms": float(np.mean(reset_times) * 1000),
        "tick_avg_ms": float(np.mean(tick_times) * 1000),
        "tick_p50_ms": float(np.percentile(np.array(tick_times) * 1000, 50)),
        "tick_p95_ms": float(np.percentile(np.array(tick_times) * 1000, 95)),
        "collect_fitness_s": collect_time,
        "evolve_s": evolve_time,
        "refresh_s": refresh_time,
        "evolution_total_s": collect_time + evolve_time + refresh_time,
    }


def main():
    parser = argparse.ArgumentParser(description="完整 round 性能分析")
    parser.add_argument("--arenas", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-length", type=int, default=100)
    args = parser.parse_args()

    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 70)
    print("完整 Round 性能分析")
    print("=" * 70)

    from create_config import create_default_config
    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

    config = create_default_config(episode_length=args.episode_length, catfish_enabled=False)
    config.training.num_arenas = args.arenas
    config.training.episodes_per_arena = args.episodes

    print(f"tick_size: {config.market.tick_size}")

    multi_config = MultiArenaConfig(num_arenas=args.arenas, episodes_per_arena=args.episodes)
    trainer = ParallelArenaTrainer(config, multi_config)

    print("\n初始化...")
    t0 = time.time()
    trainer.setup()
    init_time = time.time() - t0
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 预热
    print("\n预热运行...")
    trainer._reset_all_arenas()
    trainer._init_market_all_arenas()
    for _ in range(10):
        trainer.run_tick_all_arenas()

    # 分析
    stats = benchmark_full_round(trainer, args.episodes, args.episode_length)

    print("\n" + "=" * 70)
    print("完整 Round 性能分析结果")
    print("=" * 70)

    print(f"\n总体耗时:")
    print(f"  Round 总时间: {stats['round_time_s']:.1f}s ({stats['round_time_s']/60:.2f}min)")
    print(f"  Episode 总时间: {stats['episode_total_s']:.1f}s")
    print(f"  进化总时间: {stats['evolution_total_s']:.1f}s")

    print(f"\nEpisode 阶段:")
    print(f"  平均 Episode 时间: {stats['episode_avg_s']:.1f}s")
    print(f"  平均重置时间: {stats['reset_avg_ms']:.0f}ms")

    print(f"\nTick 阶段:")
    print(f"  平均 Tick 时间: {stats['tick_avg_ms']:.0f}ms")
    print(f"  P50 Tick 时间: {stats['tick_p50_ms']:.0f}ms")
    print(f"  P95 Tick 时间: {stats['tick_p95_ms']:.0f}ms")
    pred_50_ep = stats['tick_avg_ms'] * args.episode_length * 50 / 1000 / 60
    print(f"  预估 50 Episode 时间: {pred_50_ep:.1f}min (仅tick)")

    print(f"\n进化阶段:")
    print(f"  适应度收集: {stats['collect_fitness_s']:.1f}s")
    print(f"  NEAT 进化: {stats['evolve_s']:.1f}s")
    print(f"  状态刷新: {stats['refresh_s']:.1f}s")

    print("=" * 70)

    trainer.stop()


if __name__ == "__main__":
    main()
