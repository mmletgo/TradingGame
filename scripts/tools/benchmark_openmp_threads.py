#!/usr/bin/env python3
"""OpenMP 线程数基准测试脚本

测试单竞技场模式下不同 openmp_threads 参数对每 tick 耗时的影响。

使用方法:
    python scripts/benchmark_openmp_threads.py
    python scripts/benchmark_openmp_threads.py --warmup-ticks 50 --test-ticks 200
"""

import argparse
import importlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.create_config import create_default_config
from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer


@dataclass
class BenchmarkResult:
    """基准测试结果"""

    num_threads: int
    warmup_ticks: int
    test_ticks: int
    total_time_ms: float
    avg_tick_time_ms: float
    min_tick_time_ms: float
    max_tick_time_ms: float
    p50_tick_time_ms: float
    p95_tick_time_ms: float
    p99_tick_time_ms: float


def reset_episode_for_benchmark(trainer: Trainer) -> None:
    """为基准测试重置 episode 状态（不执行进化）

    Args:
        trainer: Trainer 实例
    """
    # 重置所有种群的 Agent 账户
    for population in trainer.populations.values():
        population.reset_agents()

    # 重置市场状态
    trainer._reset_market()

    # 重置 tick 计数和淘汰计数
    trainer.tick = 0
    trainer._pop_liquidated_counts.clear()
    trainer._eliminating_agents.clear()

    # 初始化 episode 价格统计
    initial_price = trainer.config.market.initial_price
    trainer._episode_high_price = initial_price
    trainer._episode_low_price = initial_price


def reinit_network_caches(trainer: Trainer, num_threads: int) -> None:
    """重新初始化网络缓存，以便使用新的线程数

    Args:
        trainer: Trainer 实例
        num_threads: 新的 OpenMP 线程数
    """
    from src.training._cython.batch_decide_openmp import BatchNetworkCache
    from src.bio.agents.base import AgentType

    # 缓存类型常量
    CACHE_TYPE_FULL = 1
    CACHE_TYPE_MARKET_MAKER = 2

    trainer._network_caches = {}

    for agent_type, population in trainer.populations.items():
        if not population.agents:
            continue

        # 确定缓存类型
        if agent_type == AgentType.MARKET_MAKER:
            cache_type = CACHE_TYPE_MARKET_MAKER
        else:  # RETAIL_PRO
            cache_type = CACHE_TYPE_FULL

        # 创建新缓存
        num_networks = len(population.agents)
        cache = BatchNetworkCache(num_networks, cache_type, num_threads)

        # 提取网络数据
        networks = [agent.brain.network for agent in population.agents]
        cache.update_networks(networks)

        trainer._network_caches[agent_type] = cache

    trainer._cache_initialized = True


def run_benchmark(
    trainer: Trainer,
    num_threads: int,
    warmup_ticks: int,
    test_ticks: int,
) -> BenchmarkResult:
    """运行单次基准测试

    Args:
        trainer: 已初始化的 Trainer
        num_threads: OpenMP 线程数
        warmup_ticks: 预热 tick 数（不计入统计）
        test_ticks: 测试 tick 数

    Returns:
        基准测试结果
    """
    # 动态设置 OpenMP 全局线程数
    from src.training._cython.batch_decide_openmp import set_num_threads

    set_num_threads(num_threads)

    # 重新初始化缓存以使用新的线程数
    reinit_network_caches(trainer, num_threads)

    # 重置 episode 状态
    reset_episode_for_benchmark(trainer)

    # 预热阶段
    for _ in range(warmup_ticks):
        trainer.run_tick()

    # 测试阶段
    tick_times: list[float] = []

    for _ in range(test_ticks):
        start = time.perf_counter()
        trainer.run_tick()
        elapsed = (time.perf_counter() - start) * 1000  # 转换为毫秒
        tick_times.append(elapsed)

    # 计算统计数据
    tick_times.sort()
    total_time = sum(tick_times)
    avg_time = total_time / len(tick_times)
    min_time = tick_times[0]
    max_time = tick_times[-1]

    # 计算百分位数
    def percentile(data: list[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    p50 = percentile(tick_times, 50)
    p95 = percentile(tick_times, 95)
    p99 = percentile(tick_times, 99)

    return BenchmarkResult(
        num_threads=num_threads,
        warmup_ticks=warmup_ticks,
        test_ticks=test_ticks,
        total_time_ms=total_time,
        avg_tick_time_ms=avg_time,
        min_tick_time_ms=min_time,
        max_tick_time_ms=max_time,
        p50_tick_time_ms=p50,
        p95_tick_time_ms=p95,
        p99_tick_time_ms=p99,
    )


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="OpenMP 线程数基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--warmup-ticks",
        type=int,
        default=100,
        help="预热 tick 数（不计入统计，默认: 100）",
    )
    parser.add_argument(
        "--test-ticks",
        type=int,
        default=300,
        help="测试 tick 数（默认: 300）",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="1,2,4,8,12,16,20,24,28,32",
        help="要测试的线程数，逗号分隔（默认: 1,2,4,8,12,16,20,24,28,32）",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="每个线程数重复测试次数（默认: 3）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="日志目录（默认: logs）",
    )

    args = parser.parse_args()

    # 解析线程数列表
    thread_counts = [int(x.strip()) for x in args.threads.split(",")]

    # 设置日志（静默模式）
    setup_logging(args.log_dir)

    print("=" * 80)
    print("OpenMP 线程数基准测试")
    print("=" * 80)
    print(f"预热 Ticks: {args.warmup_ticks}")
    print(f"测试 Ticks: {args.test_ticks}")
    print(f"测试线程数: {thread_counts}")
    print(f"重复次数: {args.repeat}")
    print("=" * 80)

    # 创建配置
    config = create_default_config(
        episode_length=10000,  # 足够长以完成所有测试
        checkpoint_interval=0,  # 不保存检查点
    )

    # 创建训练器并初始化
    print("\n正在初始化训练环境...")
    init_start = time.time()
    trainer = Trainer(config)
    trainer.setup()
    init_time = time.time() - init_start
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 显示 Agent 配置
    print("\n当前 Agent 配置:")
    for agent_type, agent_config in config.agents.items():
        print(f"  {agent_type.value}: {agent_config.count} 个")

    # 存储所有结果
    all_results: dict[int, list[BenchmarkResult]] = {t: [] for t in thread_counts}

    # 运行测试
    print("\n" + "-" * 80)
    print("开始基准测试...")
    print("-" * 80)

    for repeat_idx in range(args.repeat):
        print(f"\n第 {repeat_idx + 1}/{args.repeat} 轮测试:")

        for num_threads in thread_counts:
            result = run_benchmark(
                trainer=trainer,
                num_threads=num_threads,
                warmup_ticks=args.warmup_ticks,
                test_ticks=args.test_ticks,
            )
            all_results[num_threads].append(result)

            print(
                f"  threads={num_threads:2d}: "
                f"avg={result.avg_tick_time_ms:7.3f}ms, "
                f"p50={result.p50_tick_time_ms:7.3f}ms, "
                f"p95={result.p95_tick_time_ms:7.3f}ms, "
                f"min={result.min_tick_time_ms:7.3f}ms, "
                f"max={result.max_tick_time_ms:7.3f}ms"
            )

    # 汇总结果
    print("\n" + "=" * 80)
    print("汇总结果（多次测试平均值）")
    print("=" * 80)
    print(
        f"{'线程数':>8} | {'平均(ms)':>10} | {'P50(ms)':>10} | {'P95(ms)':>10} | {'P99(ms)':>10} | {'最小(ms)':>10} | {'最大(ms)':>10}"
    )
    print("-" * 80)

    summary_results: list[tuple[int, float, float, float, float, float, float]] = []

    for num_threads in thread_counts:
        results = all_results[num_threads]
        avg_avg = sum(r.avg_tick_time_ms for r in results) / len(results)
        avg_p50 = sum(r.p50_tick_time_ms for r in results) / len(results)
        avg_p95 = sum(r.p95_tick_time_ms for r in results) / len(results)
        avg_p99 = sum(r.p99_tick_time_ms for r in results) / len(results)
        avg_min = sum(r.min_tick_time_ms for r in results) / len(results)
        avg_max = sum(r.max_tick_time_ms for r in results) / len(results)

        summary_results.append(
            (num_threads, avg_avg, avg_p50, avg_p95, avg_p99, avg_min, avg_max)
        )

        print(
            f"{num_threads:>8} | "
            f"{avg_avg:>10.3f} | "
            f"{avg_p50:>10.3f} | "
            f"{avg_p95:>10.3f} | "
            f"{avg_p99:>10.3f} | "
            f"{avg_min:>10.3f} | "
            f"{avg_max:>10.3f}"
        )

    # 找出最优线程数
    best_by_avg = min(summary_results, key=lambda x: x[1])
    best_by_p50 = min(summary_results, key=lambda x: x[2])
    best_by_p95 = min(summary_results, key=lambda x: x[3])

    print("\n" + "=" * 80)
    print("最优线程数分析")
    print("=" * 80)
    print(f"按平均耗时: {best_by_avg[0]} 线程 ({best_by_avg[1]:.3f}ms)")
    print(f"按 P50 耗时: {best_by_p50[0]} 线程 ({best_by_p50[2]:.3f}ms)")
    print(f"按 P95 耗时: {best_by_p95[0]} 线程 ({best_by_p95[3]:.3f}ms)")

    # 计算相对于单线程的加速比
    baseline_avg = summary_results[0][1]  # 线程数=1 的结果
    print("\n相对于单线程的加速比:")
    for num_threads, avg_avg, _, _, _, _, _ in summary_results:
        speedup = baseline_avg / avg_avg
        print(f"  {num_threads:2d} 线程: {speedup:.2f}x")

    print("\n" + "=" * 80)
    print(f"建议: 将 openmp_threads 设置为 {best_by_avg[0]}")
    print("=" * 80)


if __name__ == "__main__":
    main()
