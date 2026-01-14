#!/usr/bin/env python3
"""多竞技场模式 tick 性能基准测试脚本

测量多竞技场模式下 1 tick 各阶段的耗时。

使用方法:
    python scripts/benchmark_parallel_arena_tick.py [--ticks N] [--warmup N] [--arenas N]
"""

import argparse
import importlib
import sys
import time
from pathlib import Path

# 清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from src.core.log_engine.logger import setup_logging

from create_config import create_default_config


def run_benchmark_with_timing(
    trainer: "ParallelArenaTrainer",
    num_ticks: int = 50,
    warmup_ticks: int = 10,
) -> dict[str, dict[str, float]]:
    """运行性能基准测试，记录各阶段耗时

    Args:
        trainer: ParallelArenaTrainer 实例
        num_ticks: 测试的 tick 数量
        warmup_ticks: 预热的 tick 数量

    Returns:
        各阶段的统计数据
    """
    import random
    from src.training.arena.arena_state import AgentAccountState

    print(f"\n运行预热 ({warmup_ticks} ticks)...")

    # 预热
    for _ in range(warmup_ticks):
        trainer.run_tick_all_arenas()

    print(f"运行基准测试 ({num_ticks} ticks)...")

    # 正式测试 - 记录各阶段耗时
    times_total: list[float] = []
    times_prepare: list[float] = []
    times_liquidation: list[float] = []
    times_catfish: list[float] = []
    times_market_state: list[float] = []
    times_collect_agents: list[float] = []
    times_inference: list[float] = []
    times_execute: list[float] = []

    for tick_idx in range(num_ticks):
        t_total_start = time.perf_counter()

        # ========== 阶段1: 准备 ==========
        t_prepare_start = time.perf_counter()
        arena_market_states = []
        arena_active_agents = []

        t_liq_total = 0.0
        t_catfish_total = 0.0
        t_market_total = 0.0
        t_collect_total = 0.0

        for arena in trainer.arena_states:
            arena.tick += 1

            # Tick 1: 只记录做市商初始挂单后的状态
            if arena.tick == 1:
                current_price = arena.smooth_mid_price
                arena.price_history.append(current_price)
                arena.tick_history_prices.append(current_price)
                arena.tick_history_volumes.append(0.0)
                arena.tick_history_amounts.append(0.0)
                arena.update_price_stats(current_price)
                t_ms = time.perf_counter()
                arena_market_states.append(trainer._compute_market_state_for_arena(arena))
                t_market_total += time.perf_counter() - t_ms
                arena_active_agents.append([])
                continue

            # 获取当前价格
            current_price = (
                arena.smooth_mid_price
                if arena.smooth_mid_price > 0
                else arena.matching_engine._orderbook.last_price
            )

            # 强平检查
            t_liq = time.perf_counter()
            trainer._handle_liquidations_for_arena(arena, current_price)
            t_liq_total += time.perf_counter() - t_liq

            # 鲶鱼行动
            t_cat = time.perf_counter()
            trainer._catfish_action_for_arena(arena)
            t_catfish_total += time.perf_counter() - t_cat

            # 计算市场状态
            t_ms = time.perf_counter()
            market_state = trainer._compute_market_state_for_arena(arena)
            t_market_total += time.perf_counter() - t_ms
            arena_market_states.append(market_state)

            # 收集活跃的 Agent 状态
            t_col = time.perf_counter()
            active_states: list[AgentAccountState] = [
                state for state in arena.agent_states.values()
                if not state.is_liquidated
            ]
            random.shuffle(active_states)
            t_collect_total += time.perf_counter() - t_col
            arena_active_agents.append(active_states)

        t_prepare_end = time.perf_counter()
        times_prepare.append(t_prepare_end - t_prepare_start)
        times_liquidation.append(t_liq_total)
        times_catfish.append(t_catfish_total)
        times_market_state.append(t_market_total)
        times_collect_agents.append(t_collect_total)

        # ========== 阶段2: 批量推理 ==========
        t_inference_start = time.perf_counter()
        all_decisions = trainer._batch_inference_all_arenas_direct(
            arena_market_states, arena_active_agents
        )
        t_inference_end = time.perf_counter()
        times_inference.append(t_inference_end - t_inference_start)

        # ========== 阶段3: 执行 ==========
        t_execute_start = time.perf_counter()
        for arena_idx, arena in enumerate(trainer.arena_states):
            if arena.tick == 1:
                continue

            decisions = all_decisions.get(arena_idx, [])
            tick_trades = []

            for state, action, params in decisions:
                from src.bio.agents.base import AgentType
                if state.agent_type == AgentType.MARKET_MAKER:
                    trades = trainer._execute_mm_action_in_arena(arena, state, params)
                else:
                    trades = trainer._execute_non_mm_action_in_arena(
                        arena, state, action, params
                    )
                tick_trades.extend(trades)

            # 记录价格历史
            current_price = arena.matching_engine._orderbook.last_price
            arena.price_history.append(current_price)
            arena.update_price_stats(current_price)

            # 记录 tick 历史数据
            arena.tick_history_prices.append(current_price)
            volume, amount = trainer._aggregate_tick_trades(tick_trades)
            arena.tick_history_volumes.append(volume)
            arena.tick_history_amounts.append(amount)

            # 鲶鱼强平检查
            trainer._check_catfish_liquidation_for_arena(arena, current_price)

        t_execute_end = time.perf_counter()
        times_execute.append(t_execute_end - t_execute_start)

        t_total_end = time.perf_counter()
        times_total.append(t_total_end - t_total_start)

        if (tick_idx + 1) % 10 == 0:
            avg_total = np.mean(times_total) * 1000
            print(f"  已完成 {tick_idx + 1}/{num_ticks} ticks, 平均耗时: {avg_total:.2f}ms")

    # 计算统计数据
    def calc_stats(arr: list[float]) -> dict[str, float]:
        arr_ms = np.array(arr) * 1000
        return {
            "mean": float(np.mean(arr_ms)),
            "std": float(np.std(arr_ms)),
            "min": float(np.min(arr_ms)),
            "max": float(np.max(arr_ms)),
            "p50": float(np.percentile(arr_ms, 50)),
            "p95": float(np.percentile(arr_ms, 95)),
            "p99": float(np.percentile(arr_ms, 99)),
        }

    stats = {
        "total": calc_stats(times_total),
        "prepare": calc_stats(times_prepare),
        "  liquidation": calc_stats(times_liquidation),
        "  catfish": calc_stats(times_catfish),
        "  market_state": calc_stats(times_market_state),
        "  collect_agents": calc_stats(times_collect_agents),
        "inference": calc_stats(times_inference),
        "execute": calc_stats(times_execute),
    }

    return stats


def print_stats(stats: dict[str, dict[str, float]]) -> None:
    """打印统计数据"""
    print("\n" + "=" * 90)
    print("多竞技场模式 - 性能基准测试结果 (单位: 毫秒)")
    print("=" * 90)

    print(f"\n{'阶段':<20} {'平均':>10} {'标准差':>10} {'P50':>10} {'P95':>10} {'P99':>10} {'占比':>8}")
    print("-" * 90)

    total_mean = stats["total"]["mean"]
    for name, s in stats.items():
        pct = (s["mean"] / total_mean * 100) if total_mean > 0 else 0
        print(f"{name:<20} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p50']:>10.2f} {s['p95']:>10.2f} {s['p99']:>10.2f} {pct:>7.1f}%")

    print("=" * 90)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="多竞技场模式 tick 性能基准测试")
    parser.add_argument("--ticks", type=int, default=50, help="测试的 tick 数量（默认: 50）")
    parser.add_argument("--warmup", type=int, default=10, help="预热的 tick 数量（默认: 10）")
    parser.add_argument("--arenas", type=int, default=2, help="竞技场数量（默认: 2）")

    args = parser.parse_args()

    # 设置日志（静默）
    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 70)
    print("多竞技场模式 - 1 Tick 性能基准测试")
    print("=" * 70)

    # 创建配置
    config = create_default_config(
        episode_length=1000,
        checkpoint_interval=0,
        catfish_enabled=False,
    )

    # 打印配置信息
    total_agents = sum(cfg.count for cfg in config.agents.values())
    print(f"竞技场数量: {args.arenas}")
    print(f"总 Agent 数量: {total_agents}")
    for agent_type, cfg in config.agents.items():
        print(f"  - {agent_type.value}: {cfg.count}")

    # 创建多竞技场训练器
    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

    multi_config = MultiArenaConfig(
        num_arenas=args.arenas,
        episodes_per_arena=50,
    )
    trainer = ParallelArenaTrainer(config, multi_config)

    # 初始化
    print("\n初始化训练环境...")
    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 运行做市商初始化
    print("运行做市商初始化...")
    trainer._init_market_all_arenas()

    # 运行基准测试
    stats = run_benchmark_with_timing(trainer, num_ticks=args.ticks, warmup_ticks=args.warmup)

    # 打印结果
    print_stats(stats)

    # 清理
    trainer.stop()


if __name__ == "__main__":
    main()
