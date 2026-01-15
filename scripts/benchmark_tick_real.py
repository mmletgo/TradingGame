#!/usr/bin/env python3
"""多竞技场真实 tick 性能基准测试脚本

直接调用 run_tick_all_arenas() 测量真实的 tick 耗时（包含 Worker 池）。
支持详细的阶段耗时分析。
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
from create_config import create_default_config


def run_benchmark_detailed(
    trainer, num_ticks: int = 50, warmup_ticks: int = 10
) -> dict[str, dict[str, float]]:
    """运行详细的 tick 基准测试，分阶段计时"""
    import random
    from typing import Any

    print(f"\n运行预热 ({warmup_ticks} ticks)...")
    for _ in range(warmup_ticks):
        trainer.run_tick_all_arenas()

    print(f"运行详细基准测试 ({num_ticks} ticks)...")

    times_total: list[float] = []
    times_phase1: list[float] = []
    times_phase1_liq: list[float] = []
    times_phase1_collect: list[float] = []
    times_phase1_market: list[float] = []
    times_phase2: list[float] = []
    times_phase3: list[float] = []

    for tick_idx in range(num_ticks):
        t_total_start = time.perf_counter()

        # ========== 阶段1: 准备 ==========
        t_phase1_start = time.perf_counter()
        arena_market_states = []
        arena_active_agents = []
        arena_catfish_trades = [[] for _ in trainer.arena_states]

        t_liq_total = 0.0
        t_collect_total = 0.0
        t_market_total = 0.0

        for arena_idx, arena in enumerate(trainer.arena_states):
            arena.tick += 1

            if arena.tick == 1:
                current_price = arena.smooth_mid_price
                arena.price_history.append(current_price)
                arena.tick_history_prices.append(current_price)
                arena.tick_history_volumes.append(0.0)
                arena.tick_history_amounts.append(0.0)
                if arena_idx in trainer._worker_depth_cache:
                    _, _, last_price, mid_price = trainer._worker_depth_cache[arena_idx]
                    actual_price = last_price if last_price > 0 else mid_price if mid_price > 0 else current_price
                else:
                    actual_price = current_price
                if actual_price > arena.episode_high_price:
                    arena.episode_high_price = actual_price
                if actual_price < arena.episode_low_price:
                    arena.episode_low_price = actual_price
                t_ms = time.perf_counter()
                arena_market_states.append(trainer._compute_market_state_for_arena(arena))
                t_market_total += time.perf_counter() - t_ms
                arena_active_agents.append([])
                continue

            current_price = (
                arena.smooth_mid_price
                if arena.smooth_mid_price > 0
                else arena.matching_engine._orderbook.last_price
            )

            # 强平
            t_liq = time.perf_counter()
            trainer._handle_liquidations_for_arena(arena, current_price)
            t_liq_total += time.perf_counter() - t_liq

            # 鲶鱼
            arena_catfish_trades[arena_idx] = trainer._catfish_action_for_arena(arena)

            # 市场状态
            t_ms = time.perf_counter()
            market_state = trainer._compute_market_state_for_arena(arena)
            t_market_total += time.perf_counter() - t_ms
            arena_market_states.append(market_state)

            # 收集 agent
            t_col = time.perf_counter()
            active_states = [
                state for state in arena.agent_states.values()
                if not state.is_liquidated
            ]
            random.shuffle(active_states)
            t_collect_total += time.perf_counter() - t_col
            arena_active_agents.append(active_states)

        t_phase1_end = time.perf_counter()
        times_phase1.append(t_phase1_end - t_phase1_start)
        times_phase1_liq.append(t_liq_total)
        times_phase1_collect.append(t_collect_total)
        times_phase1_market.append(t_market_total)

        # ========== 阶段2: 批量推理 ==========
        t_phase2_start = time.perf_counter()
        all_decisions = trainer._batch_inference_all_arenas_direct(
            arena_market_states, arena_active_agents
        )
        t_phase2_end = time.perf_counter()
        times_phase2.append(t_phase2_end - t_phase2_start)

        # ========== 阶段3: 执行 ==========
        t_phase3_start = time.perf_counter()

        if trainer._execute_worker_pool is not None:
            # Worker 池执行
            filtered_decisions = {
                arena_idx: decisions
                for arena_idx, decisions in all_decisions.items()
                if trainer.arena_states[arena_idx].tick > 1
            }
            if filtered_decisions:
                results = trainer._execute_with_worker_pool(filtered_decisions)
                arena_tick_trades = trainer._process_worker_results(results)

                # 后处理
                for arena_idx in filtered_decisions.keys():
                    arena = trainer.arena_states[arena_idx]
                    tick_trades = arena_tick_trades.get(arena_idx, [])
                    catfish_trades = arena_catfish_trades[arena_idx]
                    if catfish_trades:
                        tick_trades = catfish_trades + tick_trades

                    actual_price = arena.smooth_mid_price
                    if arena_idx in trainer._worker_depth_cache:
                        _, _, last_price, mid_price = trainer._worker_depth_cache[arena_idx]
                        if last_price > 0:
                            actual_price = last_price
                        elif mid_price > 0:
                            actual_price = mid_price

                    current_price = arena.smooth_mid_price
                    arena.price_history.append(current_price)
                    trainer._update_episode_price_stats_from_trades(
                        arena, tick_trades, fallback_price=actual_price
                    )
                    arena.tick_history_prices.append(current_price)
                    volume, amount = trainer._aggregate_tick_trades(tick_trades)
                    arena.tick_history_volumes.append(volume)
                    arena.tick_history_amounts.append(amount)
                    trainer._check_catfish_liquidation_for_arena(arena, current_price)
        else:
            # 串行执行（简化）
            from src.bio.agents.base import AgentType
            for arena_idx, arena in enumerate(trainer.arena_states):
                if arena.tick == 1:
                    continue
                decisions = all_decisions.get(arena_idx, [])
                tick_trades = []
                for state, action, params in decisions:
                    if state.agent_type == AgentType.MARKET_MAKER:
                        trades = trainer._execute_mm_action_in_arena(arena, state, params)
                    else:
                        trades = trainer._execute_non_mm_action_in_arena(arena, state, action, params)
                    tick_trades.extend(trades)
                current_price = arena.matching_engine._orderbook.last_price
                arena.price_history.append(current_price)
                trainer._update_episode_price_stats_from_trades(arena, tick_trades, fallback_price=current_price)
                arena.tick_history_prices.append(current_price)
                volume, amount = trainer._aggregate_tick_trades(tick_trades)
                arena.tick_history_volumes.append(volume)
                arena.tick_history_amounts.append(amount)
                trainer._check_catfish_liquidation_for_arena(arena, current_price)

        t_phase3_end = time.perf_counter()
        times_phase3.append(t_phase3_end - t_phase3_start)

        t_total_end = time.perf_counter()
        times_total.append(t_total_end - t_total_start)

        if (tick_idx + 1) % 10 == 0:
            avg_total = np.mean(times_total) * 1000
            print(f"  已完成 {tick_idx + 1}/{num_ticks} ticks, 平均耗时: {avg_total:.2f}ms")

    def calc_stats(arr: list[float]) -> dict[str, float]:
        arr_ms = np.array(arr) * 1000
        return {
            "mean": float(np.mean(arr_ms)),
            "std": float(np.std(arr_ms)),
            "p50": float(np.percentile(arr_ms, 50)),
            "p95": float(np.percentile(arr_ms, 95)),
            "p99": float(np.percentile(arr_ms, 99)),
        }

    return {
        "total": calc_stats(times_total),
        "phase1_prepare": calc_stats(times_phase1),
        "  liquidation": calc_stats(times_phase1_liq),
        "  collect_agents": calc_stats(times_phase1_collect),
        "  market_state": calc_stats(times_phase1_market),
        "phase2_inference": calc_stats(times_phase2),
        "phase3_execute": calc_stats(times_phase3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="多竞技场真实 tick 性能基准测试")
    parser.add_argument("--ticks", type=int, default=50, help="测试的 tick 数量")
    parser.add_argument("--warmup", type=int, default=10, help="预热的 tick 数量")
    parser.add_argument("--arenas", type=int, default=25, help="竞技场数量")
    parser.add_argument("--episode-length", type=int, default=100, help="Episode 长度")

    args = parser.parse_args()

    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 70)
    print("多竞技场真实 tick 性能基准测试（含详细阶段分析）")
    print("=" * 70)

    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=0,
        catfish_enabled=False,
    )
    config.training.num_arenas = args.arenas
    config.training.episodes_per_arena = 1

    total_agents = sum(cfg.count for cfg in config.agents.values())
    print(f"竞技场数量: {args.arenas}")
    print(f"总 Agent 数量: {total_agents}")
    for agent_type, cfg in config.agents.items():
        print(f"  - {agent_type.value}: {cfg.count}")

    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

    multi_config = MultiArenaConfig(
        num_arenas=args.arenas,
        episodes_per_arena=1,
    )
    trainer = ParallelArenaTrainer(config, multi_config)

    print("\n初始化训练环境...")
    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    if trainer._execute_worker_pool is not None:
        print(f"Execute Worker 池: 已启用")
    else:
        print(f"Execute Worker 池: 未启用（将使用串行执行）")

    print("初始化做市商...")
    trainer._init_market_all_arenas()

    stats = run_benchmark_detailed(trainer, num_ticks=args.ticks, warmup_ticks=args.warmup)

    print("\n" + "=" * 90)
    print("真实 tick 性能测试结果（含阶段分析）(单位: 毫秒)")
    print("=" * 90)

    total_mean = stats["total"]["mean"]
    print(f"\n{'阶段':<20} {'平均':>10} {'标准差':>10} {'P50':>10} {'P95':>10} {'占比':>8}")
    print("-" * 90)
    for name, s in stats.items():
        pct = (s["mean"] / total_mean * 100) if total_mean > 0 else 0
        print(f"{name:<20} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p50']:>10.2f} {s['p95']:>10.2f} {pct:>7.1f}%")

    estimated_episode = stats["total"]["mean"] * args.episode_length / 1000
    print(f"\n预估 {args.episode_length} tick Episode 耗时: {estimated_episode:.1f}s ({estimated_episode/60:.1f}min)")
    print("=" * 90)

    trainer.stop()


if __name__ == "__main__":
    main()
