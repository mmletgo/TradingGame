#!/usr/bin/env python3
"""性能分析脚本

使用 cProfile 分析训练过程的性能瓶颈。
"""

import cProfile
import importlib
import pstats
import sys
import time
from pathlib import Path
from io import StringIO

# 关键：在导入任何项目模块之前，先清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer

from create_config import create_default_config


def run_training() -> tuple[float, float]:
    """运行训练（被 profile 的函数）

    Returns:
        tuple[float, float]: (初始化耗时, tick+进化耗时)
    """
    # 使用与训练相同的默认配置，但减少 episode_length 以加快分析
    config = create_default_config(
        episode_length=10,  # 减少到 10 ticks
        checkpoint_interval=0,  # 不保存检查点
    )
    trainer = Trainer(config)

    # 计时初始化阶段
    init_start = time.perf_counter()
    trainer.setup()
    init_end = time.perf_counter()
    init_time = init_end - init_start

    # 计时 tick + 进化阶段
    train_start = time.perf_counter()
    trainer.train(episodes=1)  # 只运行 1 个 episode
    train_end = time.perf_counter()
    train_time = train_end - train_start

    return init_time, train_time


def main() -> None:
    """主函数"""
    setup_logging("logs")

    print("=" * 60)
    print("性能分析 - NEAT AI 交易模拟")
    print("=" * 60)
    print("配置: 1000 散户, 100 高级散户, 100 庄家, 100 做市商")
    print("运行: 1 episode x 10 ticks")
    print("=" * 60)
    print()

    # 使用 cProfile 进行性能分析
    profiler = cProfile.Profile()
    profiler.enable()

    init_time, train_time = run_training()

    profiler.disable()

    # 打印耗时统计
    print("\n" + "=" * 60)
    print("耗时统计")
    print("=" * 60)
    print(f"初始化耗时 (setup):     {init_time:.3f} 秒")
    print(f"tick+进化耗时 (train):  {train_time:.3f} 秒")
    print(f"总耗时:                 {init_time + train_time:.3f} 秒")
    print("=" * 60)

    # 输出统计结果
    print("\n" + "=" * 60)
    print("性能分析结果 (按累计时间排序 - 前 50)")
    print("=" * 60)

    # 按累计时间排序
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(50)

    print("\n" + "=" * 60)
    print("性能分析结果 (按自身时间排序 - 前 50)")
    print("=" * 60)

    # 按自身时间排序（不含子调用）
    stats.sort_stats("tottime")
    stats.print_stats(50)

    # 保存完整结果到文件
    stats.dump_stats("profile_results.prof")
    print(f"\n完整结果已保存到: profile_results.prof")
    print("使用 snakeviz profile_results.prof 可视化查看")


if __name__ == "__main__":
    main()
