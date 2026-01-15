#!/usr/bin/env python3
"""详细分析执行阶段（Worker 池）的性能瓶颈"""

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

    parser = argparse.ArgumentParser(description="执行阶段详细性能分析")
    parser.add_argument("--num-arenas", type=int, default=25, help="竞技场数量")
    parser.add_argument("--num-ticks", type=int, default=10, help="测试的 tick 数量")
    parser.add_argument("--episode-length", type=int, default=100, help="Episode 长度")
    args = parser.parse_args()

    # 设置日志
    setup_logging("logs")

    print("=" * 70)
    print("执行阶段详细性能分析")
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
    from src.training.arena.execute_worker import ArenaExecuteData

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

    # 检查 Worker 池状态
    if trainer._execute_worker_pool is None:
        print("警告: Worker 池未创建！")
        trainer.stop()
        return

    print(f"Worker 池类型: {type(trainer._execute_worker_pool).__name__}")

    # 重置所有竞技场
    print("\n开始性能测试...")
    trainer._reset_all_arenas()
    trainer._init_market_all_arenas()

    # 收集计时数据
    timing_data: dict[str, list[float]] = {
        "total_tick": [],
        "prepare": [],
        "inference": [],
        "build_decisions": [],
        "worker_execute": [],
        "process_results": [],
        "postprocess": [],
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
        tick_start = time.perf_counter()

        # ========== 准备阶段 ==========
        prepare_start = time.perf_counter()

        for arena in trainer.arena_states:
            arena.tick += 1

        arena_market_states: list[NormalizedMarketState] = []
        arena_active_agents: list[list[AgentAccountState]] = []

        for arena_idx, arena in enumerate(trainer.arena_states):
            current_price = (
                arena.smooth_mid_price
                if arena.smooth_mid_price > 0
                else arena.matching_engine._orderbook.last_price
            )

            # 强平检查
            trainer._handle_liquidations_for_arena(arena, current_price)

            # 计算市场状态
            market_state = trainer._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)

            # 收集活跃 Agent
            active_states = [
                state
                for state in arena.agent_states.values()
                if not state.is_liquidated
            ]
            random.shuffle(active_states)
            arena_active_agents.append(active_states)

        prepare_end = time.perf_counter()
        timing_data["prepare"].append(prepare_end - prepare_start)

        # ========== 推理阶段 ==========
        inference_start = time.perf_counter()
        all_decisions = trainer._batch_inference_all_arenas_direct(
            arena_market_states, arena_active_agents
        )
        inference_end = time.perf_counter()
        timing_data["inference"].append(inference_end - inference_start)

        # ========== 构建决策数据 ==========
        build_start = time.perf_counter()

        filtered_decisions = {
            arena_idx: decisions
            for arena_idx, decisions in all_decisions.items()
            if trainer.arena_states[arena_idx].tick > 1
        }

        # 模拟 _execute_with_worker_pool 中的数据准备
        arena_commands = {}
        for arena_idx, decisions in filtered_decisions.items():
            arena = trainer.arena_states[arena_idx]

            # 收集强平 Agent
            liquidated_agents = []

            # 收集非做市商决策
            non_mm_decisions = []
            mm_decisions_list = []

            for state, action, params in decisions:
                if state.agent_type == AgentType.MARKET_MAKER:
                    mm_decisions_list.append((
                        state.agent_id,
                        params.get("bid_orders", []),
                        params.get("ask_orders", []),
                    ))
                else:
                    if action == ActionType.HOLD:
                        continue
                    # 转换为元组格式
                    action_int = action.value if hasattr(action, 'value') else int(action)
                    side_int = params.get("side", 1)
                    price = params.get("price", 0.0)
                    quantity = params.get("quantity", 0)
                    non_mm_decisions.append((state.agent_id, action_int, side_int, price, quantity))

            arena_commands[arena_idx] = ArenaExecuteData(
                liquidated_agents=liquidated_agents,
                decisions=non_mm_decisions,
                mm_decisions=mm_decisions_list,
            )

        build_end = time.perf_counter()
        timing_data["build_decisions"].append(build_end - build_start)

        # ========== Worker 执行 ==========
        worker_start = time.perf_counter()
        results = trainer._execute_worker_pool.execute_all(arena_commands)
        worker_end = time.perf_counter()
        timing_data["worker_execute"].append(worker_end - worker_start)

        # ========== 处理结果 ==========
        process_start = time.perf_counter()
        arena_tick_trades = trainer._process_worker_results(results)
        process_end = time.perf_counter()
        timing_data["process_results"].append(process_end - process_start)

        # ========== 后处理 ==========
        postprocess_start = time.perf_counter()

        for arena_idx in filtered_decisions.keys():
            arena = trainer.arena_states[arena_idx]
            tick_trades = arena_tick_trades.get(arena_idx, [])

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

        postprocess_end = time.perf_counter()
        timing_data["postprocess"].append(postprocess_end - postprocess_start)

        tick_end = time.perf_counter()
        timing_data["total_tick"].append(tick_end - tick_start)

        if (tick_num + 1) % 5 == 0:
            print(f"  Tick {tick_num + 1}/{args.num_ticks} 完成, 总耗时: {timing_data['total_tick'][-1]*1000:.1f}ms")

    # 打印统计结果
    print("\n" + "=" * 70)
    print("执行阶段详细性能分析结果")
    print("=" * 70)

    def stats_str(times: list[float], name: str) -> str:
        if not times or all(t == 0 for t in times):
            return f"{name}: 无数据"
        mean = statistics.mean(times) * 1000
        if len(times) > 1:
            std = statistics.stdev(times) * 1000
            return f"{name}: {mean:.1f}ms ± {std:.1f}ms"
        return f"{name}: {mean:.1f}ms"

    print("\n--- Tick 各阶段耗时 ---")
    print(stats_str(timing_data["total_tick"], "总耗时/tick"))
    print(stats_str(timing_data["prepare"], "准备阶段"))
    print(stats_str(timing_data["inference"], "推理阶段"))
    print(stats_str(timing_data["build_decisions"], "构建决策数据"))
    print(stats_str(timing_data["worker_execute"], "Worker执行"))
    print(stats_str(timing_data["process_results"], "处理结果"))
    print(stats_str(timing_data["postprocess"], "后处理"))

    # 计算各阶段占比
    if timing_data["total_tick"]:
        total_mean = statistics.mean(timing_data["total_tick"])
        print("\n--- 各阶段占比 ---")
        for name in ["prepare", "inference", "build_decisions", "worker_execute", "process_results", "postprocess"]:
            if timing_data[name]:
                pct = statistics.mean(timing_data[name]) / total_mean * 100
                print(f"{name}: {pct:.1f}%")

    # 决策数量统计
    print("\n--- 决策数量统计 ---")
    total_non_mm = sum(len(cmd.decisions) for cmd in arena_commands.values())
    total_mm = sum(len(cmd.mm_decisions) for cmd in arena_commands.values())
    print(f"非做市商决策数: {total_non_mm}")
    print(f"做市商决策数: {total_mm}")

    # 清理
    trainer.stop()
    print("\n测试完成！")


if __name__ == "__main__":
    main()
