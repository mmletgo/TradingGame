#!/usr/bin/env python3
"""推理阶段内部耗时详细分析"""

import argparse
import importlib
import sys
import time
from pathlib import Path

importlib.invalidate_caches()
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import random

from src.core.log_engine.logger import setup_logging
from create_config import create_default_config


def benchmark_inference_detailed(trainer, num_ticks: int = 20) -> dict:
    """详细测量推理阶段各子步骤耗时"""
    from src.bio.agents.base import AgentType

    print(f"\n运行预热 (5 ticks)...")
    for _ in range(5):
        trainer.run_tick_all_arenas()

    print(f"运行详细推理分析 ({num_ticks} ticks)...")

    times_data_collect = []  # 数据收集（提取 AgentAccountState）
    times_cython_call = []   # Cython decide_multi_arena_direct 调用
    times_result_parse = []  # 结果解析（包括做市商解析）
    times_mm_parse = []      # 做市商解析单独计时
    times_total_inference = []

    mm_count_per_tick = []
    non_mm_count_per_tick = []

    for tick_idx in range(num_ticks):
        # 准备阶段（简化）
        arena_market_states = []
        arena_active_agents = []

        for arena in trainer.arena_states:
            arena.tick += 1
            if arena.tick == 1:
                arena_market_states.append(trainer._compute_market_state_for_arena(arena))
                arena_active_agents.append([])
                continue

            current_price = arena.smooth_mid_price if arena.smooth_mid_price > 0 else arena.matching_engine._orderbook.last_price
            trainer._handle_liquidations_for_arena(arena, current_price)
            trainer._catfish_action_for_arena(arena)
            market_state = trainer._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)

            active_states = [s for s in arena.agent_states.values() if not s.is_liquidated]
            random.shuffle(active_states)
            arena_active_agents.append(active_states)

        # ========== 推理阶段详细计时 ==========
        t_inference_start = time.perf_counter()

        results = {}
        for arena_idx in range(len(arena_active_agents)):
            results[arena_idx] = []

        trainer._last_inference_arrays.clear()

        if trainer.network_caches is None:
            continue

        # 步骤1: 数据收集
        t_collect_start = time.perf_counter()

        type_arena_data = {}
        for arena_idx, states in enumerate(arena_active_agents):
            for state in states:
                agent_type = state.agent_type
                network_idx = trainer._get_network_index(agent_type, state.agent_id)
                if network_idx < 0:
                    continue
                type_arena_data.setdefault(agent_type, {}).setdefault(arena_idx, []).append((state, network_idx))

        t_collect_end = time.perf_counter()
        times_data_collect.append((t_collect_end - t_collect_start) * 1000)

        # 步骤2&3: Cython 调用 + 结果解析
        t_cython_total = 0.0
        t_parse_total = 0.0
        t_mm_parse_total = 0.0
        tick_mm_count = 0
        tick_non_mm_count = 0

        for agent_type, arena_data in type_arena_data.items():
            cache = trainer.network_caches.get(agent_type)
            if cache is None or not cache.is_valid():
                continue

            sorted_arena_indices = sorted(arena_data.keys())
            states_per_arena = []
            market_states = []
            network_indices_per_arena = []
            state_mapping = []

            for arena_idx in sorted_arena_indices:
                arena_entries = arena_data[arena_idx]
                arena_states_list = []
                arena_network_indices = []
                for state, network_idx in arena_entries:
                    arena_states_list.append(state)
                    arena_network_indices.append(network_idx)
                states_per_arena.append(arena_states_list)
                market_states.append(arena_market_states[arena_idx])
                network_indices_per_arena.append(arena_network_indices)
                state_mapping.append(arena_states_list)

            is_market_maker = agent_type == AgentType.MARKET_MAKER

            # Cython 调用
            t_cython_start = time.perf_counter()
            try:
                raw_results = cache.decide_multi_arena_direct(
                    states_per_arena,
                    market_states,
                    network_indices_per_arena,
                    return_array=not is_market_maker,
                )
            except Exception:
                continue
            t_cython_end = time.perf_counter()
            t_cython_total += (t_cython_end - t_cython_start) * 1000

            if not is_market_maker:
                trainer._last_inference_arrays[agent_type] = {}

            # 结果解析
            t_parse_start = time.perf_counter()

            for result_idx, arena_idx in enumerate(sorted_arena_indices):
                arena_results = raw_results.get(result_idx, None)
                arena_states_list = state_mapping[result_idx]
                market_state = arena_market_states[arena_idx]
                mid_price = market_state.mid_price
                tick_size = market_state.tick_size if market_state.tick_size > 0 else 0.01

                if is_market_maker:
                    if arena_results is None:
                        arena_results = []
                    t_mm_start = time.perf_counter()
                    for i, raw_result in enumerate(arena_results):
                        if i >= len(arena_states_list):
                            break
                        state = arena_states_list[i]
                        try:
                            nn_output, _, _ = raw_result
                            action, params = trainer._parse_market_maker_output(
                                state, nn_output, mid_price, tick_size
                            )
                            results[arena_idx].append((state, action, params))
                            tick_mm_count += 1
                        except Exception:
                            pass
                    t_mm_end = time.perf_counter()
                    t_mm_parse_total += (t_mm_end - t_mm_start) * 1000
                else:
                    if arena_results is None or len(arena_results) == 0:
                        continue

                    decisions_array = arena_results
                    num_agents = min(len(decisions_array), len(arena_states_list))
                    non_hold_mask = decisions_array[:num_agents, 0] != 0
                    non_hold_indices = np.where(non_hold_mask)[0]

                    agent_ids = np.array(
                        [arena_states_list[i].agent_id for i in non_hold_indices],
                        dtype=np.float64,
                    )
                    filtered_decisions = decisions_array[:num_agents][non_hold_mask]
                    trainer._last_inference_arrays[agent_type][arena_idx] = (
                        agent_ids,
                        filtered_decisions.copy(),
                    )

                    for i in non_hold_indices:
                        state = arena_states_list[i]
                        try:
                            action_type_int = int(decisions_array[i, 0])
                            side_int = int(decisions_array[i, 1])
                            price = float(decisions_array[i, 2])
                            quantity_ratio = float(decisions_array[i, 3])
                            action, params = trainer._convert_retail_result(
                                state, action_type_int, side_int, price, quantity_ratio, mid_price
                            )
                            results[arena_idx].append((state, action, params))
                            tick_non_mm_count += 1
                        except Exception:
                            pass

            t_parse_end = time.perf_counter()
            t_parse_total += (t_parse_end - t_parse_start) * 1000

        t_inference_end = time.perf_counter()

        times_cython_call.append(t_cython_total)
        times_result_parse.append(t_parse_total)
        times_mm_parse.append(t_mm_parse_total)
        times_total_inference.append((t_inference_end - t_inference_start) * 1000)
        mm_count_per_tick.append(tick_mm_count)
        non_mm_count_per_tick.append(tick_non_mm_count)

        if (tick_idx + 1) % 5 == 0:
            print(f"  已完成 {tick_idx + 1}/{num_ticks} ticks")

    def stats(arr):
        a = np.array(arr)
        return {"mean": a.mean(), "std": a.std(), "p50": np.percentile(a, 50), "p95": np.percentile(a, 95)}

    return {
        "total_inference": stats(times_total_inference),
        "data_collect": stats(times_data_collect),
        "cython_call": stats(times_cython_call),
        "result_parse": stats(times_result_parse),
        "mm_parse": stats(times_mm_parse),
        "mm_count": np.mean(mm_count_per_tick),
        "non_mm_count": np.mean(non_mm_count_per_tick),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arenas", type=int, default=25)
    parser.add_argument("--ticks", type=int, default=20)
    args = parser.parse_args()

    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 70)
    print("推理阶段内部耗时详细分析")
    print(f"竞技场数量: {args.arenas}")
    print("=" * 70)

    config = create_default_config(episode_length=100, catfish_enabled=False)
    config.training.num_arenas = args.arenas

    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
    multi_config = MultiArenaConfig(num_arenas=args.arenas, episodes_per_arena=1)
    trainer = ParallelArenaTrainer(config, multi_config)

    print("\n初始化...")
    trainer.setup()
    trainer._init_market_all_arenas()

    stats = benchmark_inference_detailed(trainer, num_ticks=args.ticks)

    print("\n" + "=" * 80)
    print("推理阶段内部耗时分布 (单位: 毫秒)")
    print("=" * 80)

    total = stats["total_inference"]["mean"]
    for name in ["total_inference", "data_collect", "cython_call", "result_parse", "mm_parse"]:
        s = stats[name]
        pct = s["mean"] / total * 100 if total > 0 else 0
        print(f"{name:<20}: mean={s['mean']:8.1f}, std={s['std']:6.1f}, p50={s['p50']:8.1f}, p95={s['p95']:8.1f} ({pct:5.1f}%)")

    print(f"\n平均做市商数/tick: {stats['mm_count']:.0f}")
    print(f"平均非做市商数/tick: {stats['non_mm_count']:.0f}")
    print("=" * 80)

    trainer.stop()


if __name__ == "__main__":
    main()
