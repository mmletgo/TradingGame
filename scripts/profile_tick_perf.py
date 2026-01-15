#!/usr/bin/env python3
"""分析多竞技场模式每个 tick 的性能瓶颈

测量各阶段耗时：
1. 准备阶段（强平检查、市场状态计算）
2. 批量推理阶段（神经网络推理）
3. 执行阶段（Worker 池执行或串行执行）
4. 后处理阶段（价格历史记录等）
"""

import importlib
import sys
import time
from pathlib import Path
from typing import Any
import statistics

# 清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.log_engine.logger import setup_logging
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig
from create_config import create_default_config


class TimingStats:
    """记录各阶段耗时"""

    def __init__(self) -> None:
        self.prepare_times: list[float] = []
        self.inference_times: list[float] = []
        self.execute_times: list[float] = []
        self.postprocess_times: list[float] = []
        self.total_times: list[float] = []

        # 详细分解
        self.liquidation_times: list[float] = []
        self.catfish_times: list[float] = []
        self.market_state_times: list[float] = []
        self.collect_active_times: list[float] = []

    def print_summary(self) -> None:
        """打印耗时统计摘要"""
        print("\n" + "=" * 70)
        print("性能分析结果")
        print("=" * 70)

        def stats_str(times: list[float], name: str) -> str:
            if not times:
                return f"{name}: 无数据"
            mean = statistics.mean(times) * 1000  # 转为毫秒
            if len(times) > 1:
                std = statistics.stdev(times) * 1000
                return f"{name}: {mean:.1f}ms ± {std:.1f}ms"
            return f"{name}: {mean:.1f}ms"

        print("\n--- 主要阶段耗时 ---")
        print(stats_str(self.total_times, "总耗时/tick"))
        print(stats_str(self.prepare_times, "准备阶段"))
        print(stats_str(self.inference_times, "推理阶段"))
        print(stats_str(self.execute_times, "执行阶段"))
        print(stats_str(self.postprocess_times, "后处理阶段"))

        print("\n--- 准备阶段详细分解 ---")
        print(stats_str(self.liquidation_times, "强平检查"))
        print(stats_str(self.catfish_times, "鲶鱼行动"))
        print(stats_str(self.market_state_times, "市场状态计算"))
        print(stats_str(self.collect_active_times, "收集活跃Agent"))

        # 计算各阶段占比
        if self.total_times and statistics.mean(self.total_times) > 0:
            total_mean = statistics.mean(self.total_times)
            print("\n--- 各阶段占比 ---")
            if self.prepare_times:
                print(f"准备阶段: {statistics.mean(self.prepare_times)/total_mean*100:.1f}%")
            if self.inference_times:
                print(f"推理阶段: {statistics.mean(self.inference_times)/total_mean*100:.1f}%")
            if self.execute_times:
                print(f"执行阶段: {statistics.mean(self.execute_times)/total_mean*100:.1f}%")
            if self.postprocess_times:
                print(f"后处理阶段: {statistics.mean(self.postprocess_times)/total_mean*100:.1f}%")


def patch_trainer_for_timing(trainer: ParallelArenaTrainer, stats: TimingStats) -> None:
    """给 trainer 打补丁以测量各阶段耗时"""
    import random
    import numpy as np

    from src.bio.agents.base import ActionType, AgentType
    from src.market.market_state import NormalizedMarketState
    from src.market.matching.trade import Trade
    from src.market.orderbook.order import Order, OrderSide, OrderType
    from src.training.arena.arena_state import AgentAccountState

    original_run_tick = trainer.run_tick_all_arenas

    def timed_run_tick() -> bool:
        """带计时的 run_tick_all_arenas"""
        tick_start = time.perf_counter()

        # 阶段1: 准备（串行）
        prepare_start = time.perf_counter()

        arena_market_states: list[NormalizedMarketState] = []
        arena_active_agents: list[list[AgentAccountState]] = []
        arena_catfish_trades: list[list[Trade]] = [[] for _ in trainer.arena_states]

        total_liquidation_time = 0.0
        total_catfish_time = 0.0
        total_market_state_time = 0.0
        total_collect_active_time = 0.0

        for arena_idx, arena in enumerate(trainer.arena_states):
            arena.tick += 1

            # Tick 1: 只记录做市商初始挂单后的状态
            if arena.tick == 1:
                actual_price = arena.smooth_mid_price
                if arena.arena_id in trainer._worker_depth_cache:
                    _, _, last_price, mid_price = trainer._worker_depth_cache[arena.arena_id]
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
                arena_market_states.append(trainer._compute_market_state_for_arena(arena))
                total_market_state_time += time.perf_counter() - ms_start

                arena_active_agents.append([])
                continue

            # 获取当前价格
            current_price = (
                arena.smooth_mid_price
                if arena.smooth_mid_price > 0
                else arena.matching_engine._orderbook.last_price
            )

            # 强平检查
            liq_start = time.perf_counter()
            trainer._handle_liquidations_for_arena(arena, current_price)
            total_liquidation_time += time.perf_counter() - liq_start

            # 鲶鱼行动
            cat_start = time.perf_counter()
            arena_catfish_trades[arena_idx] = trainer._catfish_action_for_arena(arena)
            total_catfish_time += time.perf_counter() - cat_start

            # 计算市场状态
            ms_start = time.perf_counter()
            market_state = trainer._compute_market_state_for_arena(arena)
            arena_market_states.append(market_state)
            total_market_state_time += time.perf_counter() - ms_start

            # 收集活跃的 Agent 状态
            collect_start = time.perf_counter()
            active_states: list[AgentAccountState] = [
                state
                for state in arena.agent_states.values()
                if not state.is_liquidated
            ]
            random.shuffle(active_states)
            arena_active_agents.append(active_states)
            total_collect_active_time += time.perf_counter() - collect_start

        prepare_end = time.perf_counter()
        stats.prepare_times.append(prepare_end - prepare_start)
        stats.liquidation_times.append(total_liquidation_time)
        stats.catfish_times.append(total_catfish_time)
        stats.market_state_times.append(total_market_state_time)
        stats.collect_active_times.append(total_collect_active_time)

        # 阶段2: 批量推理（并行）
        inference_start = time.perf_counter()
        all_decisions = trainer._batch_inference_all_arenas_direct(
            arena_market_states, arena_active_agents
        )
        inference_end = time.perf_counter()
        stats.inference_times.append(inference_end - inference_start)

        # 阶段3: 执行
        execute_start = time.perf_counter()
        all_continue = True

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
        else:
            # 串行执行
            for arena_idx, arena in enumerate(trainer.arena_states):
                if arena.tick == 1:
                    continue

                decisions = all_decisions.get(arena_idx, [])
                tick_trades: list[Trade] = []

                matching_engine = arena.matching_engine
                orderbook = matching_engine._orderbook
                process_order = matching_engine.process_order
                cancel_order = matching_engine.cancel_order
                order_map_get = orderbook.order_map.get
                agent_states_get = arena.agent_states.get
                recent_trades = arena.recent_trades
                recent_trades_append = recent_trades.append

                for state, action, params in decisions:
                    if state.agent_type == AgentType.MARKET_MAKER:
                        # 做市商执行
                        for order_id in state.bid_order_ids:
                            cancel_order(order_id)
                        for order_id in state.ask_order_ids:
                            cancel_order(order_id)
                        state.bid_order_ids.clear()
                        state.ask_order_ids.clear()

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
                            trades = process_order(order)
                            for trade in trades:
                                is_buyer = trade.is_buyer_taker
                                fee = trade.buyer_fee if is_buyer else trade.seller_fee
                                state.on_trade(trade.price, trade.quantity, is_buyer, fee, is_maker=False)
                                recent_trades_append(trade)
                                tick_trades.append(trade)

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
                            trades = process_order(order)
                            for trade in trades:
                                is_buyer = not trade.is_buyer_taker
                                fee = trade.buyer_fee if is_buyer else trade.seller_fee
                                state.on_trade(trade.price, trade.quantity, is_buyer, fee, is_maker=False)
                                recent_trades_append(trade)
                                tick_trades.append(trade)
                    else:
                        # 非做市商执行（简化）
                        pass

        execute_end = time.perf_counter()
        stats.execute_times.append(execute_end - execute_start)

        # 阶段4: 后处理
        postprocess_start = time.perf_counter()
        # 简化的后处理
        postprocess_end = time.perf_counter()
        stats.postprocess_times.append(postprocess_end - postprocess_start)

        tick_end = time.perf_counter()
        stats.total_times.append(tick_end - tick_start)

        return all_continue

    # 替换方法
    trainer.run_tick_all_arenas = timed_run_tick  # type: ignore


def main() -> None:
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="性能分析工具")
    parser.add_argument("--num-arenas", type=int, default=25, help="竞技场数量")
    parser.add_argument("--num-ticks", type=int, default=10, help="测试的 tick 数量")
    parser.add_argument("--episode-length", type=int, default=100, help="Episode 长度")
    args = parser.parse_args()

    # 设置日志
    setup_logging("logs")

    print("=" * 70)
    print("多竞技场性能分析")
    print("=" * 70)
    print(f"竞技场数量: {args.num_arenas}")
    print(f"测试 tick 数: {args.num_ticks}")
    print("=" * 70)

    # 创建配置
    config = create_default_config(
        episode_length=args.episode_length,
        config_dir="config",
        catfish_enabled=False,  # 禁用鲶鱼以使用 Worker 池
    )
    config.training.num_arenas = args.num_arenas
    config.training.episodes_per_arena = 1

    # 创建多竞技场配置
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

    # 创建计时统计
    stats = TimingStats()

    # 重置所有竞技场
    print("\n开始性能测试...")
    trainer._reset_all_arenas()
    trainer._init_market_all_arenas()

    # 运行指定数量的 tick
    for tick in range(args.num_ticks):
        tick_start = time.perf_counter()

        # 直接调用原始方法，手动计时
        trainer.run_tick_all_arenas()

        tick_time = time.perf_counter() - tick_start
        stats.total_times.append(tick_time)

        if (tick + 1) % 5 == 0:
            print(f"  Tick {tick + 1}/{args.num_ticks} 完成, 耗时: {tick_time*1000:.1f}ms")

    # 打印统计
    stats.print_summary()

    # 额外打印一些信息
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
