#!/usr/bin/env python3
"""详细分析多竞技场模式每个 tick 的性能瓶颈

直接在训练器中添加计时代码进行更精确的测量
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

    parser = argparse.ArgumentParser(description="详细性能分析工具")
    parser.add_argument("--num-arenas", type=int, default=25, help="竞技场数量")
    parser.add_argument("--num-ticks", type=int, default=10, help="测试的 tick 数量")
    parser.add_argument("--episode-length", type=int, default=100, help="Episode 长度")
    args = parser.parse_args()

    # 设置日志
    setup_logging("logs")

    print("=" * 70)
    print("多竞技场详细性能分析")
    print("=" * 70)
    print(f"竞技场数量: {args.num_arenas}")
    print(f"测试 tick 数: {args.num_ticks}")
    print("=" * 70)

    # 导入必要模块
    from scripts.create_config import create_default_config
    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
    from src.training.arena.arena_state import AgentAccountState
    from src.market.market_state import NormalizedMarketState
    from src.market.matching.trade import Trade

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
        "total": [],
        "prepare": [],
        "liquidation": [],
        "catfish": [],
        "market_state": [],
        "collect_active": [],
        "shuffle": [],
        "inference": [],
        "execute": [],
        "worker_call": [],
        "worker_process": [],
        "postprocess": [],
    }

    # 手动执行 tick 并测量各阶段
    for tick_num in range(args.num_ticks):
        tick_start = time.perf_counter()

        # ========== 阶段1: 准备（串行）==========
        prepare_start = time.perf_counter()

        arena_market_states: list[NormalizedMarketState] = []
        arena_active_agents: list[list[AgentAccountState]] = []
        arena_catfish_trades: list[list[Trade]] = [[] for _ in trainer.arena_states]

        total_liquidation = 0.0
        total_catfish = 0.0
        total_market_state = 0.0
        total_collect = 0.0
        total_shuffle = 0.0

        for arena_idx, arena in enumerate(trainer.arena_states):
            arena.tick += 1

            if arena.tick == 1:
                actual_price = arena.smooth_mid_price
                if arena.arena_id in trainer._worker_depth_cache:
                    _, _, last_price, mid_price = trainer._worker_depth_cache[
                        arena.arena_id
                    ]
                    if last_price > 0:
                        actual_price = last_price
                    elif mid_price > 0:
                        actual_price = mid_price

                current_price = arena.smooth_mid_price
                arena.price_history.append(current_price)
                arena.tick_history_prices.append(current_price)
                arena.tick_history_volumes.append(0.0)
                arena.tick_history_amounts.append(0.0)
                if actual_price > arena.episode_high_price:
                    arena.episode_high_price = actual_price
                if actual_price < arena.episode_low_price:
                    arena.episode_low_price = actual_price

                ms_start = time.perf_counter()
                arena_market_states.append(
                    trainer._compute_market_state_for_arena(arena)
                )
                total_market_state += time.perf_counter() - ms_start

                arena_active_agents.append([])
                continue

            current_price = (
                arena.smooth_mid_price
                if arena.smooth_mid_price > 0
                else arena.matching_engine._orderbook.last_price
            )

            # 强平检查
            liq_start = time.perf_counter()
            trainer._handle_liquidations_for_arena(arena, current_price)
            total_liquidation += time.perf_counter() - liq_start

            # 鲶鱼行动
            cat_start = time.perf_counter()
            arena_catfish_trades[arena_idx] = trainer._catfish_action_for_arena(arena)
            total_catfish += time.perf_counter() - cat_start

            # 计算市场状态
            ms_start = time.perf_counter()
            market_state = trainer._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)
            total_market_state += time.perf_counter() - ms_start

            # 收集活跃 Agent
            collect_start = time.perf_counter()
            active_states: list[AgentAccountState] = [
                state
                for state in arena.agent_states.values()
                if not state.is_liquidated
            ]
            total_collect += time.perf_counter() - collect_start

            # 随机打乱
            shuffle_start = time.perf_counter()
            random.shuffle(active_states)
            total_shuffle += time.perf_counter() - shuffle_start

            arena_active_agents.append(active_states)

        prepare_end = time.perf_counter()
        timing_data["prepare"].append(prepare_end - prepare_start)
        timing_data["liquidation"].append(total_liquidation)
        timing_data["catfish"].append(total_catfish)
        timing_data["market_state"].append(total_market_state)
        timing_data["collect_active"].append(total_collect)
        timing_data["shuffle"].append(total_shuffle)

        # ========== 阶段2: 批量推理 ==========
        inference_start = time.perf_counter()
        all_decisions = trainer._batch_inference_all_arenas_direct(
            arena_market_states, arena_active_agents
        )
        inference_end = time.perf_counter()
        timing_data["inference"].append(inference_end - inference_start)

        # ========== 阶段3: 执行 ==========
        execute_start = time.perf_counter()

        if trainer._execute_worker_pool is not None:
            # Worker 池执行
            filtered_decisions = {
                arena_idx: decisions
                for arena_idx, decisions in all_decisions.items()
                if trainer.arena_states[arena_idx].tick > 1
            }

            if filtered_decisions:
                # Worker 池调用
                worker_call_start = time.perf_counter()
                results = trainer._execute_with_worker_pool(filtered_decisions)
                worker_call_end = time.perf_counter()
                timing_data["worker_call"].append(worker_call_end - worker_call_start)

                # 处理结果
                process_start = time.perf_counter()
                arena_tick_trades = trainer._process_worker_results(results)
                process_end = time.perf_counter()
                timing_data["worker_process"].append(process_end - process_start)

        execute_end = time.perf_counter()
        timing_data["execute"].append(execute_end - execute_start)

        # ========== 阶段4: 后处理 ==========
        postprocess_start = time.perf_counter()

        if trainer._execute_worker_pool is not None and filtered_decisions:
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
                trainer._should_end_episode_early_for_arena(arena)

        postprocess_end = time.perf_counter()
        timing_data["postprocess"].append(postprocess_end - postprocess_start)

        tick_end = time.perf_counter()
        timing_data["total"].append(tick_end - tick_start)

        if (tick_num + 1) % 5 == 0:
            print(
                f"  Tick {tick_num + 1}/{args.num_ticks} 完成, 耗时: {timing_data['total'][-1]*1000:.1f}ms"
            )

    # 打印统计结果
    print("\n" + "=" * 70)
    print("性能分析结果")
    print("=" * 70)

    def stats_str(times: list[float], name: str) -> str:
        if not times:
            return f"{name}: 无数据"
        mean = statistics.mean(times) * 1000
        if len(times) > 1:
            std = statistics.stdev(times) * 1000
            return f"{name}: {mean:.1f}ms ± {std:.1f}ms"
        return f"{name}: {mean:.1f}ms"

    print("\n--- 主要阶段耗时 ---")
    print(stats_str(timing_data["total"], "总耗时/tick"))
    print(stats_str(timing_data["prepare"], "准备阶段"))
    print(stats_str(timing_data["inference"], "推理阶段"))
    print(stats_str(timing_data["execute"], "执行阶段"))
    print(stats_str(timing_data["postprocess"], "后处理阶段"))

    print("\n--- 准备阶段详细分解 ---")
    print(stats_str(timing_data["liquidation"], "强平检查"))
    print(stats_str(timing_data["catfish"], "鲶鱼行动"))
    print(stats_str(timing_data["market_state"], "市场状态计算"))
    print(stats_str(timing_data["collect_active"], "收集活跃Agent"))
    print(stats_str(timing_data["shuffle"], "随机打乱"))

    print("\n--- 执行阶段详细分解 ---")
    print(stats_str(timing_data["worker_call"], "Worker池调用"))
    print(stats_str(timing_data["worker_process"], "结果处理"))

    # 计算各阶段占比
    if timing_data["total"]:
        total_mean = statistics.mean(timing_data["total"])
        print("\n--- 各阶段占比 ---")
        for name in ["prepare", "inference", "execute", "postprocess"]:
            if timing_data[name]:
                pct = statistics.mean(timing_data[name]) / total_mean * 100
                print(f"{name}: {pct:.1f}%")

    # 额外信息
    print("\n--- 额外信息 ---")
    total_agents = sum(len(arena.agent_states) for arena in trainer.arena_states)
    print(f"总 Agent 数量: {total_agents}")
    print(f"每竞技场 Agent 数: {total_agents // args.num_arenas}")

    if trainer._execute_worker_pool is not None:
        print("执行模式: Worker 池并行")
    else:
        print("执行模式: 串行")

    # 清理
    trainer.stop()
    print("\n测试完成！")


if __name__ == "__main__":
    main()
