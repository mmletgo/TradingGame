#!/usr/bin/env python3
"""多竞技场数量基准测试脚本

测试不同竞技场数量配置下完成100个episode所需的时间。

使用方法:
    python scripts/benchmark_arena_count.py
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

from src.core.log_engine.logger import setup_logging
from src.training.arena import ParallelArenaTrainer, MultiArenaConfig

from create_config import create_default_config


def run_benchmark(num_arenas: int, episodes_per_arena: int, episode_length: int = 1000) -> dict[str, Any]:
    """运行单个基准测试配置

    Args:
        num_arenas: 竞技场数量
        episodes_per_arena: 每个竞技场的episode数量
        episode_length: 每个episode的tick数量

    Returns:
        包含测试结果的字典
    """
    total_episodes = num_arenas * episodes_per_arena
    print(f"\n{'='*60}", flush=True)
    print(f"测试配置: {num_arenas} 竞技场 × {episodes_per_arena} episodes = {total_episodes} episodes", flush=True)
    print(f"{'='*60}", flush=True)

    # 创建配置
    config = create_default_config(
        episode_length=episode_length,
        config_dir="config",
    )
    config.training.num_arenas = num_arenas
    config.training.episodes_per_arena = episodes_per_arena

    # 创建多竞技场配置
    multi_config = MultiArenaConfig(
        num_arenas=num_arenas,
        episodes_per_arena=episodes_per_arena,
    )

    # 创建训练器
    trainer = ParallelArenaTrainer(config, multi_config)

    # 初始化
    print("初始化中...")
    init_start = time.time()
    trainer.setup()
    init_time = time.time() - init_start
    print(f"初始化耗时: {init_time:.2f}s")

    # 运行一轮训练
    print("运行训练轮次...")
    round_start = time.time()
    try:
        _ = trainer.run_round()
        round_time = time.time() - round_start

        result = {
            "num_arenas": num_arenas,
            "episodes_per_arena": episodes_per_arena,
            "total_episodes": total_episodes,
            "init_time": init_time,
            "round_time": round_time,
            "total_time": init_time + round_time,
            "success": True,
            "error": None,
        }

        print(f"训练耗时: {round_time:.2f}s")
        print(f"总耗时: {result['total_time']:.2f}s")

    except Exception as e:
        round_time = time.time() - round_start
        result = {
            "num_arenas": num_arenas,
            "episodes_per_arena": episodes_per_arena,
            "total_episodes": total_episodes,
            "init_time": init_time,
            "round_time": round_time,
            "total_time": init_time + round_time,
            "success": False,
            "error": str(e),
        }
        print(f"测试失败: {e}")
    finally:
        trainer.stop()

    return result


def main() -> None:
    """主函数"""
    import logging
    # 设置日志（静默模式）
    setup_logging("logs", console_level=logging.WARNING)

    # 测试配置：竞技场数量 × 每竞技场episode数 = 100
    test_configs = [
        (2, 50),    # 2 × 50 = 100
        (4, 25),    # 4 × 25 = 100
        (5, 20),    # 5 × 20 = 100
        (10, 10),   # 10 × 10 = 100
        (20, 5),    # 20 × 5 = 100
        (25, 4),    # 25 × 4 = 100
        (50, 2),    # 50 × 2 = 100
    ]

    # episode 长度（使用较短的长度加快测试）
    # 使用 100 ticks 以加快测试速度（原始 1000 ticks 测试时间太长）
    episode_length = 100

    print("=" * 70)
    print("多竞技场数量基准测试")
    print("=" * 70)
    print(f"目标: 找到最快完成 100 个 episode 的竞技场配置")
    print(f"Episode 长度: {episode_length} ticks")
    print(f"测试配置数量: {len(test_configs)}")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    for num_arenas, episodes_per_arena in test_configs:
        result = run_benchmark(num_arenas, episodes_per_arena, episode_length)
        results.append(result)

    # 输出汇总结果
    print("\n" + "=" * 70)
    print("基准测试结果汇总")
    print("=" * 70)
    print(f"{'配置':<20} {'初始化':>10} {'训练':>10} {'总耗时':>10} {'状态':>8}")
    print("-" * 70)

    successful_results = []
    for r in results:
        config_str = f"{r['num_arenas']}×{r['episodes_per_arena']}"
        status = "成功" if r["success"] else "失败"
        print(f"{config_str:<20} {r['init_time']:>10.2f}s {r['round_time']:>10.2f}s {r['total_time']:>10.2f}s {status:>8}")
        if r["success"]:
            successful_results.append(r)

    if successful_results:
        # 按总耗时排序
        successful_results.sort(key=lambda x: x["total_time"])

        print("\n" + "=" * 70)
        print("排名（按总耗时从快到慢）")
        print("=" * 70)
        for i, r in enumerate(successful_results, 1):
            config_str = f"{r['num_arenas']} 竞技场 × {r['episodes_per_arena']} episodes"
            print(f"{i}. {config_str}: {r['total_time']:.2f}s")

        best = successful_results[0]
        print("\n" + "=" * 70)
        print(f"最佳配置: {best['num_arenas']} 竞技场 × {best['episodes_per_arena']} episodes/竞技场")
        print(f"完成 100 episodes 总耗时: {best['total_time']:.2f}s")
        print(f"  - 初始化: {best['init_time']:.2f}s")
        print(f"  - 训练: {best['round_time']:.2f}s")
        print("=" * 70)
    else:
        print("\n所有测试都失败了！")


if __name__ == "__main__":
    main()
