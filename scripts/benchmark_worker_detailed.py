#!/usr/bin/env python3
"""Worker 执行阶段内部耗时分析

分析 Worker 池执行时各部分的耗时分布。
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
import random

from src.core.log_engine.logger import setup_logging


def count_orderbook_operations(trainer):
    """统计一个 tick 中的订单簿操作数量"""
    from src.bio.agents.base import AgentType

    # 准备阶段
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

    # 推理
    all_decisions = trainer._batch_inference_all_arenas_direct(arena_market_states, arena_active_agents)

    # 统计各类操作
    total_mm_cancel = 0  # 做市商撤单
    total_mm_new = 0  # 做市商新挂单
    total_non_mm = 0  # 非做市商操作
    mm_agents = 0  # 活跃做市商数量

    for arena_idx, decisions in all_decisions.items():
        if trainer.arena_states[arena_idx].tick <= 1:
            continue
        for state, action, params in decisions:
            if state.agent_type == AgentType.MARKET_MAKER:
                mm_agents += 1
                # 每个做市商会撤销上次的挂单，然后新挂单
                bid_orders = params.get("bid_orders", [])
                ask_orders = params.get("ask_orders", [])
                # 假设上次也挂了差不多数量的单
                total_mm_cancel += len(state.bid_order_ids) + len(state.ask_order_ids)
                total_mm_new += len(bid_orders) + len(ask_orders)
            else:
                from src.bio.agents.base import ActionType
                if action != ActionType.HOLD:
                    total_non_mm += 1

    return {
        "mm_agents": mm_agents,
        "mm_cancel_orders": total_mm_cancel,
        "mm_new_orders": total_mm_new,
        "non_mm_operations": total_non_mm,
        "total_operations": total_mm_cancel + total_mm_new + total_non_mm,
    }


def analyze_orderbook_depth(trainer):
    """分析订单簿深度"""
    depths = []
    for arena in trainer.arena_states:
        ob = arena.matching_engine._orderbook
        bid_depth = len(ob.bids)
        ask_depth = len(ob.asks)
        depths.append((bid_depth, ask_depth))

    avg_bid = np.mean([d[0] for d in depths])
    avg_ask = np.mean([d[1] for d in depths])
    max_bid = max(d[0] for d in depths)
    max_ask = max(d[1] for d in depths)

    return {
        "avg_bid_levels": avg_bid,
        "avg_ask_levels": avg_ask,
        "max_bid_levels": max_bid,
        "max_ask_levels": max_ask,
    }


def main():
    parser = argparse.ArgumentParser(description="Worker 执行阶段分析")
    parser.add_argument("--arenas", type=int, default=25)
    args = parser.parse_args()

    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 70)
    print("Worker 执行阶段内部分析")
    print(f"竞技场数量: {args.arenas}")
    print("=" * 70)

    from create_config import create_default_config
    config = create_default_config(episode_length=100, catfish_enabled=False)
    config.training.num_arenas = args.arenas

    print(f"tick_size: {config.market.tick_size}")
    print(f"做市商数量: {config.agents.get(config.agents.__class__.__mro__[0].__class__.__name__, None)}")

    from src.bio.agents.base import AgentType
    mm_count = config.agents[AgentType.MARKET_MAKER].count
    print(f"做市商数量: {mm_count}")
    print(f"预期每tick做市商订单操作: {args.arenas} × {mm_count} × 20(撤) × 20(新) = {args.arenas * mm_count * 40}")

    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
    multi_config = MultiArenaConfig(num_arenas=args.arenas, episodes_per_arena=1)
    trainer = ParallelArenaTrainer(config, multi_config)

    print("\n初始化...")
    trainer.setup()
    print("初始化做市商...")
    trainer._init_market_all_arenas()

    # 运行几个 tick
    print("\n运行 10 个 tick 热身...")
    for _ in range(10):
        trainer.run_tick_all_arenas()

    # 分析订单簿深度
    print("\n分析订单簿深度...")
    depth_stats = analyze_orderbook_depth(trainer)
    print(f"平均买盘档位数: {depth_stats['avg_bid_levels']:.0f}")
    print(f"平均卖盘档位数: {depth_stats['avg_ask_levels']:.0f}")
    print(f"最大买盘档位数: {depth_stats['max_bid_levels']}")
    print(f"最大卖盘档位数: {depth_stats['max_ask_levels']}")

    # 统计操作数量
    print("\n统计一个 tick 的操作数量...")
    op_stats = count_orderbook_operations(trainer)
    print(f"活跃做市商: {op_stats['mm_agents']}")
    print(f"做市商撤单数: {op_stats['mm_cancel_orders']}")
    print(f"做市商新挂单数: {op_stats['mm_new_orders']}")
    print(f"非做市商操作数: {op_stats['non_mm_operations']}")
    print(f"总订单簿操作数: {op_stats['total_operations']}")

    # 估算每个操作的平均耗时
    print("\n运行 5 个 tick 统计耗时...")
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        trainer.run_tick_all_arenas()
        times.append(time.perf_counter() - t0)

    avg_time = np.mean(times) * 1000
    ops_per_tick = op_stats['total_operations']
    if ops_per_tick > 0:
        time_per_op = avg_time / ops_per_tick
        print(f"平均 tick 耗时: {avg_time:.1f}ms")
        print(f"每个订单簿操作平均耗时: {time_per_op:.4f}ms = {time_per_op * 1000:.2f}μs")

    print("=" * 70)
    trainer.stop()


if __name__ == "__main__":
    main()
