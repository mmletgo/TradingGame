#!/usr/bin/env python3
"""详细的性能分析脚本

分析 tick_size 对各阶段性能的影响。
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


def analyze_execute_phase(trainer, num_ticks: int = 20) -> dict:
    """详细分析执行阶段的耗时"""
    from src.bio.agents.base import AgentType

    print(f"\n运行预热 (5 ticks)...")
    for _ in range(5):
        trainer.run_tick_all_arenas()

    print(f"运行执行阶段详细分析 ({num_ticks} ticks)...")

    times_liq = []  # 强平处理
    times_prepare = []  # 准备数据
    times_worker_call = []  # Worker 调用
    times_process_result = []  # 处理结果

    mm_order_counts = []  # 做市商订单数量
    non_mm_decision_counts = []  # 非做市商决策数量

    for tick_idx in range(num_ticks):
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

            t0 = time.perf_counter()
            trainer._handle_liquidations_for_arena(arena, current_price)
            times_liq.append(time.perf_counter() - t0)

            trainer._catfish_action_for_arena(arena)
            market_state = trainer._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)

            active_states = [s for s in arena.agent_states.values() if not s.is_liquidated]
            random.shuffle(active_states)
            arena_active_agents.append(active_states)

        # 推理
        all_decisions = trainer._batch_inference_all_arenas_direct(arena_market_states, arena_active_agents)

        # 执行阶段分析
        if trainer._execute_worker_pool is not None:
            filtered_decisions = {
                arena_idx: decisions
                for arena_idx, decisions in all_decisions.items()
                if trainer.arena_states[arena_idx].tick > 1
            }

            if filtered_decisions:
                # 统计决策数量
                tick_mm_orders = 0
                tick_non_mm = 0
                for arena_idx, decisions in filtered_decisions.items():
                    for state, action, params in decisions:
                        if state.agent_type == AgentType.MARKET_MAKER:
                            tick_mm_orders += len(params.get("bid_orders", [])) + len(params.get("ask_orders", []))
                        else:
                            tick_non_mm += 1
                mm_order_counts.append(tick_mm_orders)
                non_mm_decision_counts.append(tick_non_mm)

                # 准备数据
                t0 = time.perf_counter()
                from src.training.arena.execute_worker import ArenaExecuteData
                arena_commands = {}
                for arena_idx, decisions in filtered_decisions.items():
                    arena = trainer.arena_states[arena_idx]
                    liquidated_agents = []
                    mm_decisions = []
                    non_mm_decisions = []

                    for state in arena.agent_states.values():
                        if state.is_liquidated and state.agent_id in arena.eliminating_agents:
                            is_mm = state.agent_type == AgentType.MARKET_MAKER
                            liquidated_agents.append((state.agent_id, state.position_quantity, is_mm))

                    for state, action, params in decisions:
                        if state.agent_type == AgentType.MARKET_MAKER:
                            mm_decisions.append((state.agent_id, params.get("bid_orders", []), params.get("ask_orders", [])))
                        else:
                            from src.bio.agents.base import ActionType
                            action_int = {
                                ActionType.HOLD: 0,
                                ActionType.PLACE_BID: 1,
                                ActionType.PLACE_ASK: 2,
                                ActionType.CANCEL: 3,
                                ActionType.MARKET_BUY: 4,
                                ActionType.MARKET_SELL: 5,
                            }.get(action, 0)
                            if action_int == 0:
                                continue
                            side_int = 1 if action in (ActionType.PLACE_BID, ActionType.MARKET_BUY) else 2
                            non_mm_decisions.append((state.agent_id, action_int, side_int, params.get("price", 0.0), int(params.get("quantity", 0))))

                    decisions_array = trainer._build_decisions_array_from_cache(arena_idx)
                    arena_commands[arena_idx] = ArenaExecuteData(
                        liquidated_agents=liquidated_agents,
                        decisions=non_mm_decisions,
                        mm_decisions=mm_decisions,
                        decisions_array=decisions_array,
                    )
                times_prepare.append(time.perf_counter() - t0)

                # Worker 调用
                t0 = time.perf_counter()
                results = trainer._execute_worker_pool.execute_all(arena_commands)
                times_worker_call.append(time.perf_counter() - t0)

                # 处理结果
                t0 = time.perf_counter()
                arena_tick_trades = trainer._process_worker_results(results)

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
                    trainer._update_episode_price_stats_from_trades(arena, tick_trades, fallback_price=actual_price)
                    arena.tick_history_prices.append(current_price)
                    volume, amount = trainer._aggregate_tick_trades(tick_trades)
                    arena.tick_history_volumes.append(volume)
                    arena.tick_history_amounts.append(amount)
                times_process_result.append(time.perf_counter() - t0)

        if (tick_idx + 1) % 5 == 0:
            print(f"  已完成 {tick_idx + 1}/{num_ticks} ticks")

    def stats(arr):
        if len(arr) == 0:
            return {"mean": 0, "std": 0, "p50": 0, "p95": 0}
        a = np.array(arr) * 1000
        return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}

    return {
        "liquidation": stats(times_liq),
        "prepare_data": stats(times_prepare),
        "worker_call": stats(times_worker_call),
        "process_result": stats(times_process_result),
        "mm_orders_per_tick": float(np.mean(mm_order_counts)) if mm_order_counts else 0,
        "non_mm_decisions_per_tick": float(np.mean(non_mm_decision_counts)) if non_mm_decision_counts else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="详细性能分析")
    parser.add_argument("--arenas", type=int, default=25)
    parser.add_argument("--ticks", type=int, default=20)
    args = parser.parse_args()

    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 70)
    print("详细性能分析")
    print(f"竞技场数量: {args.arenas}")
    print("=" * 70)

    from create_config import create_default_config
    config = create_default_config(episode_length=100, catfish_enabled=False)
    config.training.num_arenas = args.arenas

    print(f"tick_size: {config.market.tick_size}")

    total_agents = sum(cfg.count for cfg in config.agents.values())
    print(f"总 Agent 数量: {total_agents}")
    for agent_type, cfg in config.agents.items():
        print(f"  - {agent_type.value}: {cfg.count}")

    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
    multi_config = MultiArenaConfig(num_arenas=args.arenas, episodes_per_arena=1)
    trainer = ParallelArenaTrainer(config, multi_config)

    print("\n初始化...")
    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    print("初始化做市商...")
    trainer._init_market_all_arenas()

    stats = analyze_execute_phase(trainer, num_ticks=args.ticks)

    print("\n" + "=" * 80)
    print("执行阶段详细耗时分布 (单位: 毫秒)")
    print("=" * 80)

    for name in ["liquidation", "prepare_data", "worker_call", "process_result"]:
        s = stats[name]
        print(f"{name:<20}: mean={s['mean']:8.2f}, std={s['std']:6.2f}, p50={s['p50']:8.2f}, p95={s['p95']:8.2f}")

    print(f"\n平均做市商订单数/tick: {stats['mm_orders_per_tick']:.0f}")
    print(f"平均非做市商决策数/tick: {stats['non_mm_decisions_per_tick']:.0f}")
    print("=" * 80)

    trainer.stop()


if __name__ == "__main__":
    main()
