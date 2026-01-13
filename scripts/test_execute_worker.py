#!/usr/bin/env python3
"""测试 ArenaExecuteWorkerPool 的基本功能"""

import argparse
import importlib
import sys
import time
from pathlib import Path

importlib.invalidate_caches()
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.core.log_engine.logger import setup_logging
from create_config import create_default_config


def test_worker_pool_basic():
    """测试 Worker 池的基本功能"""
    from src.training.arena import ArenaExecuteWorkerPool, ArenaExecuteData

    config = create_default_config(
        episode_length=100, checkpoint_interval=0, catfish_enabled=False
    )

    print("=" * 70)
    print("测试 ArenaExecuteWorkerPool 基本功能")
    print("=" * 70)

    num_arenas = 4
    num_workers = 2
    arena_ids = list(range(num_arenas))

    print(f"\n创建 Worker 池: {num_workers} workers, {num_arenas} arenas")

    # 准备 fee_rates
    fee_rates: dict[int, tuple[float, float]] = {}
    # 添加一些测试用的 agent fee rates
    for i in range(100):
        fee_rates[i] = (0.0002, 0.0005)  # maker_rate, taker_rate

    try:
        pool = ArenaExecuteWorkerPool(
            num_workers=num_workers,
            arena_ids=arena_ids,
            config=config,
        )

        print("启动 Worker 池...")
        start_time = time.time()
        pool.start()
        print(f"启动完成 ({time.time() - start_time:.2f}s)")

        print("\n重置所有竞技场...")
        start_time = time.time()
        pool.reset_all(
            initial_price=config.market.initial_price,
            fee_rates=fee_rates,
        )
        print(f"重置完成 ({time.time() - start_time:.2f}s)")

        print("\n获取订单簿深度...")
        start_time = time.time()
        depths = pool.get_all_depths()
        print(f"获取完成 ({time.time() - start_time:.2f}s)")
        print(f"获取到 {len(depths)} 个竞技场的深度")

        for arena_id, (bid_depth, ask_depth, last_price, mid_price) in depths.items():
            print(f"  Arena {arena_id}: last_price={last_price}, mid_price={mid_price}")

        print("\n测试执行（空决策）...")
        arena_commands = {
            arena_id: ArenaExecuteData(
                liquidated_agents=[],
                decisions=[],
                mm_decisions=[],
            )
            for arena_id in arena_ids
        }

        start_time = time.time()
        results = pool.execute_all(arena_commands)
        print(f"执行完成 ({time.time() - start_time:.2f}s)")
        print(f"获取到 {len(results)} 个竞技场的结果")

        for arena_id, result in results.items():
            if result.error:
                print(f"  Arena {arena_id}: ERROR - {result.error}")
            else:
                print(f"  Arena {arena_id}: {len(result.trades)} trades")

        print("\n关闭 Worker 池...")
        pool.shutdown()
        print("测试完成!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="测试 ArenaExecuteWorkerPool")
    args = parser.parse_args()

    setup_logging("logs", console_level=logging.WARNING)

    success = test_worker_pool_basic()
    if success:
        print("\n✓ 所有测试通过!")
    else:
        print("\n✗ 测试失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
