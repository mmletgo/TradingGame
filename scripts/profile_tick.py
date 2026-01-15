#!/usr/bin/env python3
"""分析单个 tick 的耗时分布"""

import sys
import time
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from create_config import create_default_config
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig


def main() -> None:
    """主函数"""
    # 创建配置（禁用鲶鱼以启用 Worker 池）
    config = create_default_config(
        episode_length=10,  # 只运行10个tick
        catfish_enabled=False,  # 禁用鲶鱼
    )

    num_arenas = 25
    config.training.num_arenas = num_arenas
    config.training.episodes_per_arena = 1

    multi_config = MultiArenaConfig(
        num_arenas=num_arenas,
        episodes_per_arena=1,
    )

    print("=" * 70)
    print("Tick 性能分析")
    print("=" * 70)
    print(f"竞技场数量: {num_arenas}")
    print(f"Agent 数量: ~10,300 × {num_arenas} = ~{10300 * num_arenas:,}")
    print("=" * 70)

    # 创建训练器
    trainer = ParallelArenaTrainer(config, multi_config)

    # 初始化
    print("\n正在初始化...")
    init_start = time.perf_counter()
    trainer.setup()
    init_time = time.perf_counter() - init_start
    print(f"初始化耗时: {init_time:.2f}s")

    # 重置竞技场
    print("\n正在重置竞技场...")
    reset_start = time.perf_counter()
    trainer._reset_all_arenas()
    reset_time = time.perf_counter() - reset_start
    print(f"重置耗时: {reset_time:.2f}s")

    # 做市商初始化
    print("\n正在初始化做市商...")
    mm_start = time.perf_counter()
    trainer._init_market_all_arenas()
    mm_time = time.perf_counter() - mm_start
    print(f"做市商初始化耗时: {mm_time:.2f}s")

    # 手动执行 Tick 1
    trainer._is_running = True
    trainer.run_tick_all_arenas()  # Tick 1，只是展示

    # 分析 Tick 2+ 的详细耗时
    print("\n" + "=" * 70)
    print("Tick 2+ 详细分析（实际执行的 tick）")
    print("=" * 70)

    import random
    from src.market.market_state import NormalizedMarketState
    from src.training.arena.arena_state import AgentAccountState

    # ===== 阶段1: 准备阶段 =====
    print("\n[阶段1: 准备阶段]")

    prep_start = time.perf_counter()
    arena_market_states: list[NormalizedMarketState] = []
    arena_active_agents: list[list[AgentAccountState]] = []

    liquidation_total = 0.0
    catfish_total = 0.0
    market_state_total = 0.0
    collect_agents_total = 0.0
    shuffle_total = 0.0

    for arena_idx, arena in enumerate(trainer.arena_states):
        arena.tick += 1

        # 获取当前价格
        current_price = (
            arena.smooth_mid_price
            if arena.smooth_mid_price > 0
            else arena.matching_engine._orderbook.last_price
        )

        # 强平检查
        t0 = time.perf_counter()
        trainer._handle_liquidations_for_arena(arena, current_price)
        liquidation_total += time.perf_counter() - t0

        # 鲶鱼行动
        t0 = time.perf_counter()
        trainer._catfish_action_for_arena(arena)
        catfish_total += time.perf_counter() - t0

        # 计算市场状态
        t0 = time.perf_counter()
        market_state = trainer._compute_market_state_for_arena(arena)
        market_state_total += time.perf_counter() - t0
        arena_market_states.append(market_state)

        # 收集活跃 Agent 状态
        t0 = time.perf_counter()
        active_states: list[AgentAccountState] = [
            state
            for state in arena.agent_states.values()
            if not state.is_liquidated
        ]
        collect_agents_total += time.perf_counter() - t0

        # 随机打乱
        t0 = time.perf_counter()
        random.shuffle(active_states)
        shuffle_total += time.perf_counter() - t0

        arena_active_agents.append(active_states)

    prep_time = time.perf_counter() - prep_start

    print(f"  总耗时: {prep_time*1000:.1f}ms")
    print(f"  - 强平检查: {liquidation_total*1000:.1f}ms")
    print(f"  - 鲶鱼行动: {catfish_total*1000:.1f}ms")
    print(f"  - 市场状态计算: {market_state_total*1000:.1f}ms")
    print(f"  - 收集活跃Agent: {collect_agents_total*1000:.1f}ms")
    print(f"  - 随机打乱: {shuffle_total*1000:.1f}ms")

    # ===== 阶段2: 批量推理 =====
    print("\n[阶段2: 批量推理]")

    inference_start = time.perf_counter()
    all_decisions = trainer._batch_inference_all_arenas_direct(
        arena_market_states, arena_active_agents
    )
    inference_time = time.perf_counter() - inference_start

    total_decisions = sum(len(d) for d in all_decisions.values())
    print(f"  总耗时: {inference_time*1000:.1f}ms")
    print(f"  决策数量: {total_decisions:,}")

    # ===== 阶段3: 执行 =====
    print("\n[阶段3: 执行]")

    if trainer._execute_worker_pool is not None:
        print("  使用 Worker 池执行")

        exec_start = time.perf_counter()
        results = trainer._execute_with_worker_pool(all_decisions)
        exec_call_time = time.perf_counter() - exec_start

        process_start = time.perf_counter()
        trainer._process_worker_results(results)
        process_time = time.perf_counter() - process_start

        print(f"  执行调用耗时: {exec_call_time*1000:.1f}ms")
        print(f"  处理结果耗时: {process_time*1000:.1f}ms")
        print(f"  总执行耗时: {(exec_call_time + process_time)*1000:.1f}ms")
    else:
        print("  串行执行（Worker 池未启用）")

    # ===== 汇总 =====
    total_tick_time = prep_time + inference_time
    print("\n" + "=" * 70)
    print("Tick 耗时汇总（不含执行）")
    print("=" * 70)
    print(f"阶段1 (准备): {prep_time*1000:.1f}ms ({prep_time/total_tick_time*100:.1f}%)")
    print(f"阶段2 (推理): {inference_time*1000:.1f}ms ({inference_time/total_tick_time*100:.1f}%)")
    print(f"总计: {total_tick_time*1000:.1f}ms")
    print()
    print(f"Worker 池状态: {'已启用' if trainer._execute_worker_pool is not None else '未启用'}")

    # 运行几个完整的 tick 来验证
    print("\n" + "=" * 70)
    print("运行完整 tick 验证（Tick 3+）")
    print("=" * 70)

    tick_times = []
    for i in range(5):
        tick_start = time.perf_counter()
        trainer.run_tick_all_arenas()
        tick_time = time.perf_counter() - tick_start
        tick_times.append(tick_time)
        print(f"  Tick {i+3}: {tick_time*1000:.1f}ms")

    avg_tick = sum(tick_times) / len(tick_times)
    print(f"\n平均每tick: {avg_tick*1000:.1f}ms")
    print(f"100 tick 预计耗时: {avg_tick * 100:.1f}s")

    # 清理
    trainer.stop()
    print("\n分析完成")


if __name__ == "__main__":
    main()
