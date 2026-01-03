#!/usr/bin/env python3
"""性能分析脚本

使用 cProfile 分析训练过程的性能瓶颈。
"""

import cProfile
import pstats
import sys
import time
from pathlib import Path
from io import StringIO

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bio.agents.base import AgentType
from src.config.config import (
    AgentConfig,
    Config,
    DemoConfig,
    MarketConfig,
    TrainingConfig,
)
from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer


def create_profile_config() -> Config:
    """创建用于性能分析的配置（较小规模以加快分析）"""
    market = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )

    # 使用较小的 agent 数量以加快分析
    agents = {
        AgentType.RETAIL: AgentConfig(
            count=10000,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        ),
        AgentType.RETAIL_PRO: AgentConfig(
            count=100,
            initial_balance=10000.0,
            leverage=100.0,
            maintenance_margin_rate=0.005,
            maker_fee_rate=0.0002,
            taker_fee_rate=0.0005,
        ),
        AgentType.WHALE: AgentConfig(
            count=100,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,
        ),
        AgentType.MARKET_MAKER: AgentConfig(
            count=1000,
            initial_balance=10000000.0,
            leverage=10.0,
            maintenance_margin_rate=0.05,
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,
        ),
    }

    training = TrainingConfig(
        episode_length=100,  # 减少到 100 ticks
        checkpoint_interval=0,  # 不保存检查点
        neat_config_path="config",
    )

    demo = DemoConfig(
        host="localhost",
        port=8000,
        tick_interval=100,
    )

    return Config(market=market, agents=agents, training=training, demo=demo)


def run_training() -> tuple[float, float]:
    """运行训练（被 profile 的函数）

    Returns:
        tuple[float, float]: (初始化耗时, tick+进化耗时)
    """
    config = create_profile_config()
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
    print("配置: 10000 散户, 100 高级散户, 10 庄家, 100 做市商")
    print("运行: 1 episode x 100 ticks")
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
