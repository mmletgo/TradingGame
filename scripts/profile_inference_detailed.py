#!/usr/bin/env python3
"""详细分析推理阶段的性能瓶颈

分解 _batch_inference_all_arenas_direct 方法的各个步骤：
1. 数据收集和分组
2. 准备推理参数
3. Cython 批量推理调用
4. 结果解析
"""

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

    parser = argparse.ArgumentParser(description="推理阶段详细性能分析")
    parser.add_argument("--num-arenas", type=int, default=25, help="竞技场数量")
    parser.add_argument("--num-ticks", type=int, default=10, help="测试的 tick 数量")
    parser.add_argument("--episode-length", type=int, default=100, help="Episode 长度")
    args = parser.parse_args()

    # 设置日志
    setup_logging("logs")

    print("=" * 70)
    print("推理阶段详细性能分析")
    print("=" * 70)
    print(f"竞技场数量: {args.num_arenas}")
    print(f"测试 tick 数: {args.num_ticks}")
    print("=" * 70)

    # 导入必要模块
    from create_config import create_default_config
    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
    from src.training.arena.arena_state import AgentAccountState
    from src.market.market_state import NormalizedMarketState
    from src.bio.agents.base import AgentType, ActionType

    # 创建配置
    config = create_default_config(
        episode_length=args.episode_length,
        config_dir="config",
        catfish_enabled=False,
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
        "cython_call": [],
        "result_parsing": [],
        # 按类型分解
        "retail_cython": [],
        "retail_pro_cython": [],
        "whale_cython": [],
        "mm_cython": [],
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
        # 准备阶段（简化）
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

        # ========== 测量推理阶段详细耗时 ==========
        inference_start = time.perf_counter()

        # 1. 数据收集和分组
        collection_start = time.perf_counter()

        type_arena_data: dict[AgentType, dict[int, list[tuple[AgentAccountState, int]]]] = {}

        for arena_idx, states in enumerate(arena_active_agents):
            for state in states:
                agent_type = state.agent_type
                network_idx = trainer._get_network_index(agent_type, state.agent_id)
                if network_idx < 0:
                    continue
                type_arena_data.setdefault(agent_type, {}).setdefault(arena_idx, []).append((state, network_idx))

        collection_end = time.perf_counter()
        timing_data["data_collection"].append(collection_end - collection_start)

        # 2. 按类型调用 Cython 推理
        results: dict[int, list] = {i: [] for i in range(len(arena_active_agents))}
        trainer._last_inference_arrays.clear()

        total_cython_time = 0.0
        type_times = {
            AgentType.RETAIL: 0.0,
            AgentType.RETAIL_PRO: 0.0,
            AgentType.WHALE: 0.0,
            AgentType.MARKET_MAKER: 0.0,
        }

        for agent_type, arena_data in type_arena_data.items():
            cache = trainer.network_caches.get(agent_type)
            if cache is None or not cache.is_valid():
                continue

            # 准备参数
            sorted_arena_indices = sorted(arena_data.keys())

            states_per_arena: list[list[AgentAccountState]] = []
            market_states_list: list[NormalizedMarketState] = []
            network_indices_per_arena: list[list[int]] = []

            for arena_idx in sorted_arena_indices:
                arena_entries = arena_data[arena_idx]
                arena_states_list = [s for s, _ in arena_entries]
                arena_network_indices = [idx for _, idx in arena_entries]

                states_per_arena.append(arena_states_list)
                market_states_list.append(arena_market_states[arena_idx])
                network_indices_per_arena.append(arena_network_indices)

            is_market_maker = agent_type == AgentType.MARKET_MAKER

            # 调用 Cython
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

            cython_time = cython_end - cython_start
            total_cython_time += cython_time
            type_times[agent_type] = cython_time

        timing_data["cython_call"].append(total_cython_time)
        timing_data["retail_cython"].append(type_times[AgentType.RETAIL])
        timing_data["retail_pro_cython"].append(type_times[AgentType.RETAIL_PRO])
        timing_data["whale_cython"].append(type_times[AgentType.WHALE])
        timing_data["mm_cython"].append(type_times[AgentType.MARKET_MAKER])

        # 3. 结果解析（跳过详细解析，只测量时间）
        parsing_start = time.perf_counter()
        # 这里省略结果解析逻辑，因为我们主要关心 Cython 调用
        parsing_end = time.perf_counter()
        timing_data["result_parsing"].append(parsing_end - parsing_start)

        inference_end = time.perf_counter()
        timing_data["total_inference"].append(inference_end - inference_start)

        if (tick_num + 1) % 5 == 0:
            print(f"  Tick {tick_num + 1}/{args.num_ticks} 完成, 推理耗时: {timing_data['total_inference'][-1]*1000:.1f}ms")

    # 打印统计结果
    print("\n" + "=" * 70)
    print("推理阶段详细性能分析结果")
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
    print(stats_str(timing_data["cython_call"], "Cython调用"))
    print(stats_str(timing_data["result_parsing"], "结果解析"))

    print("\n--- 按 Agent 类型分解 Cython 调用 ---")
    print(stats_str(timing_data["retail_cython"], "RETAIL"))
    print(stats_str(timing_data["retail_pro_cython"], "RETAIL_PRO"))
    print(stats_str(timing_data["whale_cython"], "WHALE"))
    print(stats_str(timing_data["mm_cython"], "MARKET_MAKER"))

    # 计算各类型 Agent 数量
    print("\n--- Agent 数量分布 ---")
    type_counts: dict[AgentType, int] = {}
    for arena in trainer.arena_states:
        for state in arena.agent_states.values():
            type_counts[state.agent_type] = type_counts.get(state.agent_type, 0) + 1

    for agent_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{agent_type.value}: {count}")

    # 计算各阶段占比
    if timing_data["total_inference"]:
        total_mean = statistics.mean(timing_data["total_inference"])
        print("\n--- 各阶段占比 ---")
        for name in ["data_collection", "cython_call", "result_parsing"]:
            if timing_data[name]:
                pct = statistics.mean(timing_data[name]) / total_mean * 100
                print(f"{name}: {pct:.1f}%")

    # 清理
    trainer.stop()
    print("\n测试完成！")


if __name__ == "__main__":
    main()
