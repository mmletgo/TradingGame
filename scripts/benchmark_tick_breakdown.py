#!/usr/bin/env python3
"""单tick性能分析脚本

分析多竞技场模式下单个tick的各阶段耗时，定位性能瓶颈。
"""

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
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
from create_config import create_default_config


def benchmark_tick_detailed(
    trainer: ParallelArenaTrainer,
    num_ticks: int = 10,
) -> dict[str, list[float]]:
    """详细分析单tick各阶段耗时

    Args:
        trainer: 训练器实例
        num_ticks: 测试的tick数量

    Returns:
        各阶段耗时列表
    """
    import random
    from src.bio.agents.base import AgentType
    from src.market.orderbook.order import OrderType, OrderSide, Order
    from src.training.arena.arena_state import AgentAccountState

    timings: dict[str, list[float]] = {
        "phase1_total": [],          # 阶段1总时间
        "phase1_liquidation": [],     # 强平检查
        "phase1_catfish": [],         # 鲶鱼行动
        "phase1_market_state": [],    # 计算市场状态
        "phase1_collect_agents": [],  # 收集活跃agent
        "phase2_inference": [],       # 阶段2推理
        "phase3_total": [],           # 阶段3总时间
        "phase3_worker_execute": [],  # Worker执行
        "phase3_process_results": [], # 处理结果
        "total_tick": [],             # 总tick时间
    }

    # 跳过tick 1（初始化tick）
    trainer.run_tick_all_arenas()

    for tick_idx in range(num_ticks):
        tick_start = time.perf_counter()

        # ========== 阶段1: 准备 ==========
        phase1_start = time.perf_counter()

        from src.training.arena.parallel_arena_trainer import NormalizedMarketState

        arena_market_states: list[NormalizedMarketState] = []
        arena_active_agents: list[list[AgentAccountState]] = []
        arena_catfish_trades: list[list] = [[] for _ in trainer.arena_states]

        liq_times = []
        catfish_times = []
        market_state_times = []
        collect_times = []

        for arena_idx, arena in enumerate(trainer.arena_states):
            arena.tick += 1

            if arena.tick == 1:
                arena_market_states.append(trainer._compute_market_state_for_arena(arena))
                arena_active_agents.append([])
                continue

            # 获取当前价格
            current_price = (
                arena.smooth_mid_price
                if arena.smooth_mid_price > 0
                else arena.matching_engine._orderbook.last_price
            )

            # 强平检查
            t0 = time.perf_counter()
            trainer._handle_liquidations_for_arena(arena, current_price)
            liq_times.append(time.perf_counter() - t0)

            # 鲶鱼行动
            t0 = time.perf_counter()
            arena_catfish_trades[arena_idx] = trainer._catfish_action_for_arena(arena)
            catfish_times.append(time.perf_counter() - t0)

            # 计算市场状态
            t0 = time.perf_counter()
            market_state = trainer._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)
            market_state_times.append(time.perf_counter() - t0)

            # 收集活跃agent
            t0 = time.perf_counter()
            active_states: list[AgentAccountState] = [
                state
                for state in arena.agent_states.values()
                if not state.is_liquidated
            ]
            random.shuffle(active_states)
            arena_active_agents.append(active_states)
            collect_times.append(time.perf_counter() - t0)

        phase1_end = time.perf_counter()
        timings["phase1_total"].append(phase1_end - phase1_start)
        timings["phase1_liquidation"].append(sum(liq_times))
        timings["phase1_catfish"].append(sum(catfish_times))
        timings["phase1_market_state"].append(sum(market_state_times))
        timings["phase1_collect_agents"].append(sum(collect_times))

        # ========== 阶段2: 批量推理 ==========
        phase2_start = time.perf_counter()
        all_decisions = trainer._batch_inference_all_arenas_direct(
            arena_market_states, arena_active_agents
        )
        phase2_end = time.perf_counter()
        timings["phase2_inference"].append(phase2_end - phase2_start)

        # ========== 阶段3: 执行 ==========
        phase3_start = time.perf_counter()

        if trainer._execute_worker_pool is not None:
            # 过滤掉 tick=1 的竞技场
            filtered_decisions = {
                arena_idx: decisions
                for arena_idx, decisions in all_decisions.items()
                if trainer.arena_states[arena_idx].tick > 1
            }

            if filtered_decisions:
                # Worker执行
                t0 = time.perf_counter()
                results = trainer._execute_with_worker_pool(filtered_decisions)
                timings["phase3_worker_execute"].append(time.perf_counter() - t0)

                # 处理结果
                t0 = time.perf_counter()
                arena_tick_trades = trainer._process_worker_results(results)
                timings["phase3_process_results"].append(time.perf_counter() - t0)

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
        else:
            timings["phase3_worker_execute"].append(0)
            timings["phase3_process_results"].append(0)

        phase3_end = time.perf_counter()
        timings["phase3_total"].append(phase3_end - phase3_start)

        tick_end = time.perf_counter()
        timings["total_tick"].append(tick_end - tick_start)

        # 进度显示
        print(f"  Tick {tick_idx + 1}/{num_ticks}: {(tick_end - tick_start)*1000:.1f}ms")

    return timings


def print_timing_stats(timings: dict[str, list[float]], title: str = "Timing Analysis") -> None:
    """打印耗时统计"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    for name, values in timings.items():
        if not values:
            continue
        arr = np.array(values) * 1000  # 转换为毫秒
        print(f"{name:30s}: mean={arr.mean():8.2f}ms, std={arr.std():8.2f}ms, "
              f"min={arr.min():8.2f}ms, max={arr.max():8.2f}ms")

    print(f"{'='*70}")


def main() -> None:
    parser_args = {
        "num_arenas": 25,
        "episodes_per_arena": 1,
        "episode_length": 100,
        "num_ticks": 20,
    }

    import argparse
    parser = argparse.ArgumentParser(description="单tick性能分析")
    parser.add_argument("--num-arenas", type=int, default=25, help="竞技场数量")
    parser.add_argument("--num-ticks", type=int, default=20, help="测试tick数量")
    args = parser.parse_args()

    print("=" * 70)
    print("单tick性能分析")
    print("=" * 70)
    print(f"竞技场数量: {args.num_arenas}")
    print(f"测试tick数量: {args.num_ticks}")

    # 设置日志
    setup_logging("logs")

    # 创建配置
    config = create_default_config(
        episode_length=100,
        config_dir="config",
        catfish_enabled=False,
    )
    config.training.num_arenas = args.num_arenas
    config.training.episodes_per_arena = 1

    # 创建多竞技场配置
    multi_config = MultiArenaConfig(
        num_arenas=args.num_arenas,
        episodes_per_arena=1,
    )

    # 创建训练器
    trainer = ParallelArenaTrainer(config, multi_config)

    print("\n初始化训练环境...")
    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 统计Agent数量
    total_agents = sum(len(arena.agent_states) for arena in trainer.arena_states)
    print(f"总Agent数量: {total_agents} ({total_agents // args.num_arenas} per arena)")

    # 重置所有竞技场准备测试
    print("\n重置竞技场...")
    trainer._reset_all_arenas()
    trainer._init_market_all_arenas()

    # 运行性能测试
    print(f"\n开始性能测试 ({args.num_ticks} ticks)...")
    timings = benchmark_tick_detailed(trainer, args.num_ticks)

    # 打印结果
    print_timing_stats(timings, "Tick Breakdown Analysis")

    # 计算各阶段占比
    total_mean = np.mean(timings["total_tick"]) * 1000
    phase1_mean = np.mean(timings["phase1_total"]) * 1000
    phase2_mean = np.mean(timings["phase2_inference"]) * 1000
    phase3_mean = np.mean(timings["phase3_total"]) * 1000

    print(f"\n阶段耗时占比:")
    print(f"  阶段1 (准备):    {phase1_mean:8.2f}ms ({phase1_mean/total_mean*100:5.1f}%)")
    print(f"  阶段2 (推理):    {phase2_mean:8.2f}ms ({phase2_mean/total_mean*100:5.1f}%)")
    print(f"  阶段3 (执行):    {phase3_mean:8.2f}ms ({phase3_mean/total_mean*100:5.1f}%)")
    print(f"  总计:            {total_mean:8.2f}ms")

    # 计算1个episode的预估时间
    episode_time_estimate = total_mean * 100 / 1000  # 100 ticks
    print(f"\n预估 Episode 耗时: {episode_time_estimate:.1f}s (100 ticks)")
    print(f"预估 50 Episodes 耗时: {episode_time_estimate * 50 / 60:.1f} min")

    # 清理
    trainer.stop()
    print("\n测试完成")


if __name__ == "__main__":
    main()
