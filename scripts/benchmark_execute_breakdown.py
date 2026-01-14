#!/usr/bin/env python3
"""执行阶段详细耗时分析脚本

分析多竞技场模式下执行阶段内部的耗时分布。
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


def run_detailed_benchmark(
    trainer: "ParallelArenaTrainer",
    num_ticks: int = 30,
    warmup_ticks: int = 5,
) -> dict[str, dict[str, float]]:
    """运行详细的执行阶段耗时分析"""
    import random
    from src.training.arena.arena_state import AgentAccountState
    from src.bio.agents.base import AgentType

    print(f"\n运行预热 ({warmup_ticks} ticks)...")
    for _ in range(warmup_ticks):
        trainer.run_tick_all_arenas()

    print(f"运行详细分析 ({num_ticks} ticks)...")

    times_mm_total: list[float] = []
    times_mm_cancel: list[float] = []
    times_mm_bid_orders: list[float] = []
    times_mm_ask_orders: list[float] = []
    times_non_mm_total: list[float] = []
    times_history_update: list[float] = []

    mm_order_counts: list[int] = []
    non_mm_action_counts: list[int] = []

    for tick_idx in range(num_ticks):
        # 准备阶段（简化版）
        arena_market_states = []
        arena_active_agents = []

        for arena in trainer.arena_states:
            arena.tick += 1
            if arena.tick == 1:
                current_price = arena.smooth_mid_price
                arena.price_history.append(current_price)
                arena.tick_history_prices.append(current_price)
                arena.tick_history_volumes.append(0.0)
                arena.tick_history_amounts.append(0.0)
                arena.update_price_stats(current_price)
                arena_market_states.append(trainer._compute_market_state_for_arena(arena))
                arena_active_agents.append([])
                continue

            current_price = (
                arena.smooth_mid_price
                if arena.smooth_mid_price > 0
                else arena.matching_engine._orderbook.last_price
            )
            trainer._handle_liquidations_for_arena(arena, current_price)
            trainer._catfish_action_for_arena(arena)
            market_state = trainer._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)
            active_states = [
                state for state in arena.agent_states.values()
                if not state.is_liquidated
            ]
            random.shuffle(active_states)
            arena_active_agents.append(active_states)

        # 推理阶段
        all_decisions = trainer._batch_inference_all_arenas_direct(
            arena_market_states, arena_active_agents
        )

        # ========== 执行阶段详细计时 ==========
        tick_mm_cancel = 0.0
        tick_mm_bid = 0.0
        tick_mm_ask = 0.0
        tick_non_mm = 0.0
        tick_history = 0.0
        tick_mm_orders = 0
        tick_non_mm_actions = 0

        for arena_idx, arena in enumerate(trainer.arena_states):
            if arena.tick == 1:
                continue

            decisions = all_decisions.get(arena_idx, [])
            tick_trades = []
            matching_engine = arena.matching_engine

            for state, action, params in decisions:
                if state.agent_type == AgentType.MARKET_MAKER:
                    # 做市商：分别计时撤单、买单、卖单
                    t0 = time.perf_counter()
                    cancel_order = matching_engine.cancel_order
                    for order_id in state.bid_order_ids:
                        cancel_order(order_id)
                    for order_id in state.ask_order_ids:
                        cancel_order(order_id)
                    state.bid_order_ids.clear()
                    state.ask_order_ids.clear()
                    tick_mm_cancel += time.perf_counter() - t0

                    # 买单
                    t0 = time.perf_counter()
                    from src.market.orderbook.order import Order, OrderSide, OrderType
                    for order_spec in params.get("bid_orders", []):
                        order_id = state.generate_order_id(arena.arena_id)
                        order = Order(
                            order_id=order_id,
                            agent_id=state.agent_id,
                            side=OrderSide.BUY,
                            order_type=OrderType.LIMIT,
                            price=order_spec["price"],
                            quantity=int(order_spec["quantity"]),
                        )
                        trades = matching_engine.process_order(order)
                        trainer._update_trade_accounts(arena, state, trades)
                        tick_trades.extend(trades)
                        if matching_engine._orderbook.order_map.get(order_id):
                            state.bid_order_ids.append(order_id)
                        tick_mm_orders += 1
                    tick_mm_bid += time.perf_counter() - t0

                    # 卖单
                    t0 = time.perf_counter()
                    for order_spec in params.get("ask_orders", []):
                        order_id = state.generate_order_id(arena.arena_id)
                        order = Order(
                            order_id=order_id,
                            agent_id=state.agent_id,
                            side=OrderSide.SELL,
                            order_type=OrderType.LIMIT,
                            price=order_spec["price"],
                            quantity=int(order_spec["quantity"]),
                        )
                        trades = matching_engine.process_order(order)
                        trainer._update_trade_accounts(arena, state, trades)
                        tick_trades.extend(trades)
                        if matching_engine._orderbook.order_map.get(order_id):
                            state.ask_order_ids.append(order_id)
                        tick_mm_orders += 1
                    tick_mm_ask += time.perf_counter() - t0

                else:
                    # 非做市商
                    t0 = time.perf_counter()
                    trades = trainer._execute_non_mm_action_in_arena(
                        arena, state, action, params
                    )
                    tick_non_mm += time.perf_counter() - t0
                    tick_trades.extend(trades)
                    tick_non_mm_actions += 1

            # 历史记录更新
            t0 = time.perf_counter()
            current_price = arena.matching_engine._orderbook.last_price
            arena.price_history.append(current_price)
            arena.update_price_stats(current_price)
            arena.tick_history_prices.append(current_price)
            volume, amount = trainer._aggregate_tick_trades(tick_trades)
            arena.tick_history_volumes.append(volume)
            arena.tick_history_amounts.append(amount)
            trainer._check_catfish_liquidation_for_arena(arena, current_price)
            tick_history += time.perf_counter() - t0

        times_mm_cancel.append(tick_mm_cancel)
        times_mm_bid_orders.append(tick_mm_bid)
        times_mm_ask_orders.append(tick_mm_ask)
        times_mm_total.append(tick_mm_cancel + tick_mm_bid + tick_mm_ask)
        times_non_mm_total.append(tick_non_mm)
        times_history_update.append(tick_history)
        mm_order_counts.append(tick_mm_orders)
        non_mm_action_counts.append(tick_non_mm_actions)

        if (tick_idx + 1) % 10 == 0:
            print(f"  已完成 {tick_idx + 1}/{num_ticks} ticks")

    def calc_stats(arr: list[float]) -> dict[str, float]:
        arr_ms = np.array(arr) * 1000
        return {
            "mean": float(np.mean(arr_ms)),
            "std": float(np.std(arr_ms)),
            "p50": float(np.percentile(arr_ms, 50)),
            "p95": float(np.percentile(arr_ms, 95)),
        }

    stats = {
        "mm_total": calc_stats(times_mm_total),
        "  mm_cancel": calc_stats(times_mm_cancel),
        "  mm_bid": calc_stats(times_mm_bid_orders),
        "  mm_ask": calc_stats(times_mm_ask_orders),
        "non_mm_total": calc_stats(times_non_mm_total),
        "history_update": calc_stats(times_history_update),
    }

    print(f"\n平均做市商订单数/tick: {np.mean(mm_order_counts):.0f}")
    print(f"平均非做市商动作数/tick: {np.mean(non_mm_action_counts):.0f}")

    return stats


def print_stats(stats: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 80)
    print("执行阶段详细耗时分布 (单位: 毫秒)")
    print("=" * 80)

    total = sum(s["mean"] for name, s in stats.items() if not name.startswith("  "))
    print(f"\n{'阶段':<20} {'平均':>10} {'标准差':>10} {'P50':>10} {'P95':>10} {'占比':>8}")
    print("-" * 80)

    for name, s in stats.items():
        pct = (s["mean"] / total * 100) if total > 0 else 0
        print(f"{name:<20} {s['mean']:>10.2f} {s['std']:>10.2f} {s['p50']:>10.2f} {s['p95']:>10.2f} {pct:>7.1f}%")

    print("-" * 80)
    print(f"{'execute_total':<20} {total:>10.2f}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="执行阶段详细耗时分析")
    parser.add_argument("--ticks", type=int, default=30, help="测试的 tick 数量")
    parser.add_argument("--warmup", type=int, default=5, help="预热的 tick 数量")
    parser.add_argument("--arenas", type=int, default=2, help="竞技场数量")
    args = parser.parse_args()

    import logging
    setup_logging("logs", console_level=logging.WARNING)

    print("=" * 70)
    print("执行阶段详细耗时分析")
    print("=" * 70)

    config = create_default_config(episode_length=1000, checkpoint_interval=0, catfish_enabled=False)

    from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
    multi_config = MultiArenaConfig(num_arenas=args.arenas, episodes_per_arena=50)
    trainer = ParallelArenaTrainer(config, multi_config)

    print("\n初始化训练环境...")
    start_time = time.time()
    trainer.setup()
    print(f"初始化完成（耗时: {time.time() - start_time:.2f}s）")

    trainer._init_market_all_arenas()

    stats = run_detailed_benchmark(trainer, num_ticks=args.ticks, warmup_ticks=args.warmup)
    print_stats(stats)

    trainer.stop()


if __name__ == "__main__":
    main()
