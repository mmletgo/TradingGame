#!/usr/bin/env python3
"""单 tick 性能基准测试脚本

测量单竞技场模式下 1 tick 的耗时。

使用方法:
    python scripts/benchmark_tick.py [--ticks N] [--warmup N]
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

import numpy as np

from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer

from create_config import create_default_config


def run_benchmark(trainer: Trainer, num_ticks: int = 50, warmup_ticks: int = 10) -> dict[str, dict[str, float]]:
    """运行性能基准测试

    Args:
        trainer: Trainer 实例
        num_ticks: 测试的 tick 数量
        warmup_ticks: 预热的 tick 数量

    Returns:
        各阶段的统计数据
    """
    print(f"\n运行预热 ({warmup_ticks} ticks)...")

    # 预热
    for _ in range(warmup_ticks):
        trainer.run_tick()

    print(f"运行基准测试 ({num_ticks} ticks)...")

    # 正式测试
    tick_times: list[float] = []

    for i in range(num_ticks):
        t0 = time.perf_counter()
        trainer.run_tick()
        elapsed = time.perf_counter() - t0
        tick_times.append(elapsed)

        if (i + 1) % 10 == 0:
            avg_total = np.mean(tick_times) * 1000
            print(f"  已完成 {i + 1}/{num_ticks} ticks, 平均耗时: {avg_total:.2f}ms")

    # 计算统计数据
    arr = np.array(tick_times) * 1000  # 转换为毫秒
    stats = {
        "total": {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    }

    return stats


def print_stats(stats: dict[str, dict[str, float]]) -> None:
    """打印统计数据"""
    print("\n" + "=" * 80)
    print("性能基准测试结果 (单位: 毫秒)")
    print("=" * 80)

    if "total" in stats:
        s = stats["total"]
        print(f"\n{'阶段':<20} {'平均':>10} {'标准差':>10} {'P50':>10} {'P95':>10} {'P99':>10}")
        print("-" * 70)
        print(f"{'total':<20} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p50']:>10.2f} {s['p95']:>10.2f} {s['p99']:>10.2f}")

    print("=" * 80)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="单 tick 性能基准测试")
    parser.add_argument("--ticks", type=int, default=50, help="测试的 tick 数量（默认: 50）")
    parser.add_argument("--warmup", type=int, default=10, help="预热的 tick 数量（默认: 10）")
    parser.add_argument("--episode-length", type=int, default=1000, help="Episode 长度（默认: 1000）")

    args = parser.parse_args()

    # 设置日志（静默）
    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 60)
    print("单竞技场模式 - 1 Tick 性能基准测试")
    print("=" * 60)

    # 创建配置
    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=0,
        catfish_enabled=False,
    )

    # 打印配置信息
    total_agents = sum(cfg.count for cfg in config.agents.values())
    print(f"总 Agent 数量: {total_agents}")
    for agent_type, cfg in config.agents.items():
        print(f"  - {agent_type.value}: {cfg.count}")

    # 创建训练器
    trainer = Trainer(config)

    # 初始化
    print("\n初始化训练环境...")
    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 运行基准测试
    stats = run_benchmark(trainer, num_ticks=args.ticks, warmup_ticks=args.warmup)

    # 打印结果
    print_stats(stats)

    # 清理
    trainer.stop()


if __name__ == "__main__":
    main()
