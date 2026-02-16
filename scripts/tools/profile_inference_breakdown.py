#!/usr/bin/env python3
"""详细分解推理阶段内部各步骤的耗时"""

import importlib
import sys
import time
from pathlib import Path
import statistics

# 清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.log_engine.logger import setup_logging


def main() -> None:
    """主函数"""
    import argparse
    import random

    import numpy as np

    parser = argparse.ArgumentParser(description="推理阶段内部分解性能分析")
    parser.add_argument("--num-arenas", type=int, default=25, help="竞技场数量")
    parser.add_argument("--num-ticks", type=int, default=5, help="测试的 tick 数量")
    parser.add_argument("--episode-length", type=int, default=100, help="Episode 长度")
    args = parser.parse_args()

    # 设置日志
    setup_logging("logs")

    print("=" * 70)
    print("推理阶段内部分解性能分析")
    print("=" * 70)
    print(f"竞技场数量: {args.num_arenas}")
    print(f"测试 tick 数: {args.num_ticks}")
    print("=" * 70)

    # 导入必要模块
    from scripts.create_config import create_default_config
    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
    from src.training.arena.arena_state import (
        AgentAccountState,
        calculate_order_quantity_from_state,
    )
    from src.market.market_state import NormalizedMarketState
    from src.bio.agents.base import AgentType, ActionType

    # 创建配置
    config = create_default_config(
        episode_length=args.episode_length,
        config_dir="config",
    )
    config.training.num_arenas = args.num_arenas
    config.training.episodes_per_arena = 1

    multi_config = MultiArenaConfig(
        num_arenas=args.num_arenas,
        episodes_per_arena=1,
    )

    # 创建训练器
    print("\n初始化训练环境...")
    trainer = ParallelArenaTrainer(config, multi_config)

    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 重置所有竞技场
    print("\n开始性能测试...")
    trainer._reset_all_arenas()
    trainer._init_market_all_arenas()

    # 收集计时数据
    timing_data: dict[str, list[float]] = {
        "total_inference": [],
        "data_collection": [],
        "cython_retail": [],
        "cython_mm": [],
        "result_parsing_retail": [],
        "result_parsing_mm": [],
        "convert_calls": [],
    }

    # 预先让竞技场跳过 tick 1
    for arena in trainer.arena_states:
        arena.tick = 1
        arena.price_history.append(arena.smooth_mid_price)
        arena.tick_history_prices.append(arena.smooth_mid_price)
        arena.tick_history_volumes.append(0.0)
        arena.tick_history_amounts.append(0.0)

    # 运行测试
    for tick_num in range(args.num_ticks):
        for arena in trainer.arena_states:
            arena.tick += 1

        # 收集市场状态和活跃 Agent
        arena_market_states: list[NormalizedMarketState] = []
        arena_active_agents: list[list[AgentAccountState]] = []

        for arena_idx, arena in enumerate(trainer.arena_states):
            market_state = trainer._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)

            active_states = [
                state
                for state in arena.agent_states.values()
                if not state.is_liquidated
            ]
            random.shuffle(active_states)
            arena_active_agents.append(active_states)

        # ========== 手动实现推理阶段并测量各步骤 ==========
        inference_start = time.perf_counter()

        # 1. 数据收集和分组
        collection_start = time.perf_counter()

        type_arena_data: dict[
            AgentType, dict[int, list[tuple[AgentAccountState, int]]]
        ] = {}

        for arena_idx, states in enumerate(arena_active_agents):
            for state in states:
                agent_type = state.agent_type
                network_idx = trainer._get_network_index(agent_type, state.agent_id)
                if network_idx < 0:
                    continue
                type_arena_data.setdefault(agent_type, {}).setdefault(
                    arena_idx, []
                ).append((state, network_idx))

        collection_end = time.perf_counter()
        timing_data["data_collection"].append(collection_end - collection_start)

        # 2. 按类型调用 Cython 推理
        results: dict[int, list] = {i: [] for i in range(len(arena_active_agents))}
        trainer._last_inference_arrays.clear()

        total_retail_cython = 0.0
        total_mm_cython = 0.0
        total_retail_parsing = 0.0
        total_mm_parsing = 0.0
        total_convert_time = 0.0

        for agent_type, arena_data in type_arena_data.items():
            cache = trainer.network_caches.get(agent_type)
            if cache is None or not cache.is_valid():
                continue

            # 准备参数
            sorted_arena_indices = sorted(arena_data.keys())

            states_per_arena: list[list[AgentAccountState]] = []
            market_states_list: list[NormalizedMarketState] = []
            network_indices_per_arena: list[list[int]] = []
            state_mapping: list[list[AgentAccountState]] = []

            for arena_idx in sorted_arena_indices:
                arena_entries = arena_data[arena_idx]
                arena_states_list = [s for s, _ in arena_entries]
                arena_network_indices = [idx for _, idx in arena_entries]

                states_per_arena.append(arena_states_list)
                market_states_list.append(arena_market_states[arena_idx])
                network_indices_per_arena.append(arena_network_indices)
                state_mapping.append(arena_states_list)

            is_market_maker = agent_type == AgentType.MARKET_MAKER

            # Cython 调用
            cython_start = time.perf_counter()
            try:
                raw_results = cache.decide_multi_arena_direct(
                    states_per_arena,
                    market_states_list,
                    network_indices_per_arena,
                    return_array=not is_market_maker,
                )
            except Exception as e:
                print(f"推理失败 {agent_type.value}: {e}")
                raw_results = {}
            cython_end = time.perf_counter()

            if is_market_maker:
                total_mm_cython += cython_end - cython_start
            else:
                total_retail_cython += cython_end - cython_start

            # 结果解析
            parsing_start = time.perf_counter()

            for result_idx, arena_idx in enumerate(sorted_arena_indices):
                arena_results = raw_results.get(result_idx, None)
                arena_states_list = state_mapping[result_idx]
                market_state = arena_market_states[arena_idx]
                mid_price = market_state.mid_price
                tick_size = (
                    market_state.tick_size if market_state.tick_size > 0 else 0.01
                )

                if is_market_maker:
                    # 做市商解析
                    if arena_results is None:
                        arena_results = []
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
                        except Exception:
                            pass
                else:
                    # 非做市商解析
                    if arena_results is None or len(arena_results) == 0:
                        continue

                    decisions_array: np.ndarray = arena_results
                    num_agents = min(len(decisions_array), len(arena_states_list))

                    # NumPy 过滤
                    non_hold_mask = decisions_array[:num_agents, 0] != 0
                    non_hold_indices = np.where(non_hold_mask)[0]

                    # 缓存数组
                    agent_ids = np.array(
                        [arena_states_list[i].agent_id for i in non_hold_indices],
                        dtype=np.float64,
                    )
                    filtered_decisions = decisions_array[:num_agents][non_hold_mask]
                    trainer._last_inference_arrays.setdefault(agent_type, {})[
                        arena_idx
                    ] = (
                        agent_ids,
                        filtered_decisions.copy(),
                    )

                    # 逐个转换（这是瓶颈！）
                    convert_start = time.perf_counter()
                    for i in non_hold_indices:
                        state = arena_states_list[i]
                        try:
                            action_type_int = int(decisions_array[i, 0])
                            side_int = int(decisions_array[i, 1])
                            price = float(decisions_array[i, 2])
                            quantity_ratio = float(decisions_array[i, 3])

                            action, params = trainer._convert_retail_result(
                                state,
                                action_type_int,
                                side_int,
                                price,
                                quantity_ratio,
                                mid_price,
                            )
                            results[arena_idx].append((state, action, params))
                        except Exception:
                            pass
                    convert_end = time.perf_counter()
                    total_convert_time += convert_end - convert_start

            parsing_end = time.perf_counter()

            if is_market_maker:
                total_mm_parsing += parsing_end - parsing_start
            else:
                total_retail_parsing += parsing_end - parsing_start

        timing_data["cython_retail"].append(total_retail_cython)
        timing_data["cython_mm"].append(total_mm_cython)
        timing_data["result_parsing_retail"].append(total_retail_parsing)
        timing_data["result_parsing_mm"].append(total_mm_parsing)
        timing_data["convert_calls"].append(total_convert_time)

        inference_end = time.perf_counter()
        timing_data["total_inference"].append(inference_end - inference_start)

        print(
            f"  Tick {tick_num + 1}/{args.num_ticks} 完成, 推理耗时: {timing_data['total_inference'][-1]*1000:.1f}ms"
        )

    # 打印统计结果
    print("\n" + "=" * 70)
    print("推理阶段内部分解性能分析结果")
    print("=" * 70)

    def stats_str(times: list[float], name: str) -> str:
        if not times or all(t == 0 for t in times):
            return f"{name}: 无数据"
        mean = statistics.mean(times) * 1000
        if len(times) > 1:
            std = statistics.stdev(times) * 1000
            return f"{name}: {mean:.1f}ms ± {std:.1f}ms"
        return f"{name}: {mean:.1f}ms"

    print("\n--- 推理阶段总耗时分解 ---")
    print(stats_str(timing_data["total_inference"], "推理总耗时"))
    print(stats_str(timing_data["data_collection"], "数据收集/分组"))

    print("\n--- Cython 调用耗时 ---")
    print(stats_str(timing_data["cython_retail"], "非做市商 Cython"))
    print(stats_str(timing_data["cython_mm"], "做市商 Cython"))

    print("\n--- 结果解析耗时 ---")
    print(stats_str(timing_data["result_parsing_retail"], "非做市商解析（含转换）"))
    print(stats_str(timing_data["result_parsing_mm"], "做市商解析"))
    print(stats_str(timing_data["convert_calls"], "★ _convert_retail_result 调用"))

    # 计算各阶段占比
    if timing_data["total_inference"]:
        total_mean = statistics.mean(timing_data["total_inference"])
        print("\n--- 各阶段占比 ---")
        for name, label in [
            ("data_collection", "数据收集"),
            ("cython_retail", "Cython(非MM)"),
            ("cython_mm", "Cython(MM)"),
            ("convert_calls", "★ convert_retail"),
            ("result_parsing_mm", "MM解析"),
        ]:
            if timing_data[name]:
                pct = statistics.mean(timing_data[name]) / total_mean * 100
                print(f"{label}: {pct:.1f}%")

    # 计算各类型决策数量
    print("\n--- 决策数量 ---")
    total_decisions = sum(len(r) for r in results.values())
    print(f"总决策数: {total_decisions}")

    # 清理
    trainer.stop()
    print("\n测试完成！")


if __name__ == "__main__":
    main()
