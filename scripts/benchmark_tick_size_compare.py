#!/usr/bin/env python3
"""对比不同 tick_size 的性能差异"""

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


def benchmark_tick_size(tick_size: float, num_arenas: int = 25, num_ticks: int = 10):
    """测试指定 tick_size 的性能"""
    from create_config import create_default_config
    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

    config = create_default_config(episode_length=100, catfish_enabled=False)
    config.market.tick_size = tick_size
    config.training.num_arenas = num_arenas

    multi_config = MultiArenaConfig(num_arenas=num_arenas, episodes_per_arena=1)
    trainer = ParallelArenaTrainer(config, multi_config)

    print(f"\n初始化 (tick_size={tick_size})...")
    t0 = time.time()
    trainer.setup()
    init_time = time.time() - t0
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    print("初始化做市商...")
    trainer._init_market_all_arenas()

    print(f"预热 (5 ticks)...")
    for _ in range(5):
        trainer.run_tick_all_arenas()

    print(f"测试 ({num_ticks} ticks)...")
    times = []
    for i in range(num_ticks):
        t0 = time.perf_counter()
        trainer.run_tick_all_arenas()
        times.append(time.perf_counter() - t0)

    trainer.stop()

    return {
        "tick_size": tick_size,
        "mean_ms": float(np.mean(times) * 1000),
        "std_ms": float(np.std(times) * 1000),
        "p50_ms": float(np.percentile(np.array(times) * 1000, 50)),
        "p95_ms": float(np.percentile(np.array(times) * 1000, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description="对比 tick_size 性能")
    parser.add_argument("--arenas", type=int, default=25)
    parser.add_argument("--ticks", type=int, default=10)
    args = parser.parse_args()

    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 70)
    print("tick_size 性能对比测试")
    print(f"竞技场数量: {args.arenas}")
    print(f"测试 ticks: {args.ticks}")
    print("=" * 70)

    # 测试不同的 tick_size
    results = []
    for tick_size in [0.1, 0.01]:
        result = benchmark_tick_size(tick_size, args.arenas, args.ticks)
        results.append(result)

    print("\n" + "=" * 70)
    print("对比结果 (单位: 毫秒)")
    print("=" * 70)
    print(f"{'tick_size':<12} {'平均':>10} {'标准差':>10} {'P50':>10} {'P95':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['tick_size']:<12} {r['mean_ms']:>10.2f} {r['std_ms']:>10.2f} {r['p50_ms']:>10.2f} {r['p95_ms']:>10.2f}")

    if len(results) >= 2:
        ratio = results[1]['mean_ms'] / results[0]['mean_ms']
        print(f"\n性能比: tick_size=0.01 / tick_size=0.1 = {ratio:.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    main()
