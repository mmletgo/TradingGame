#!/usr/bin/env python3
"""加载单个竞技场的 checkpoint 并使用 UI 观察现象

用于调试多竞技场训练中单个竞技场的问题。

使用方法:
    # 加载 arena_14 的 checkpoint
    python scripts/inspect_arena.py --arena-id 14

    # 加载后继续训练几个 episode
    python scripts/inspect_arena.py --arena-id 14 --episodes 5
"""

import argparse
import importlib
import pickle
import sys
from pathlib import Path

# 关键：在导入任何项目模块之前，先清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer
from src.ui.training_app import TrainingUIApp

from create_config import create_default_config


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="加载单个竞技场的 checkpoint 并使用 UI 观察",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arena-id",
        type=int,
        required=True,
        help="要加载的竞技场 ID（如 14）",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/multi_arena",
        help="多竞技场检查点目录（默认: checkpoints/multi_arena）",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="加载后继续训练的 episode 数量（默认: 0，仅观察）",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100,
        help="每个 episode 的 tick 数量（默认: 100）",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="配置文件目录（默认: config）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="日志目录（默认: logs）",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    print("=" * 60)
    print(f"加载竞技场 {args.arena_id} 的 checkpoint")
    print("=" * 60)

    # 构建检查点路径
    checkpoint_path = Path(args.checkpoint_dir) / f"arena_{args.arena_id}" / "checkpoint.pkl"

    if not checkpoint_path.exists():
        print(f"错误: 检查点文件不存在: {checkpoint_path}")
        sys.exit(1)

    print(f"检查点文件: {checkpoint_path}")
    print(f"文件大小: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 加载检查点
    print("正在加载检查点...")
    with open(checkpoint_path, "rb") as f:
        arena_checkpoint = pickle.load(f)

    print(f"检查点加载完成")
    print(f"  Arena ID: {arena_checkpoint.arena_id}")
    print(f"  Episode: {arena_checkpoint.episode}")

    # 获取 trainer checkpoint 数据（在 populations 字段中）
    trainer_checkpoint = arena_checkpoint.populations
    print(f"  Tick: {trainer_checkpoint.get('tick', '未知')}")

    # 创建配置（从检查点中获取 episode_length）
    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=10,
        config_dir=args.config_dir,
    )

    # 创建训练器
    trainer = Trainer(config, arena_id=args.arena_id)

    # 初始化训练器（不创建种群，直接从检查点加载）
    print("正在初始化训练器...")
    trainer.setup(checkpoint=trainer_checkpoint)
    print("训练器初始化完成")

    # 设置 is_running 标志
    trainer.is_running = True

    # 打印当前状态信息
    print("\n当前状态:")
    print(f"  Episode: {trainer.episode}")
    print(f"  Tick: {trainer.tick}")
    print(f"  种群数量: {len(trainer.populations)}")

    for agent_type, pop in trainer.populations.items():
        print(f"  {agent_type.value}: {len(pop.agents)} 个 Agent")

    # 打印价格信息
    if hasattr(trainer, 'matching_engine') and trainer.matching_engine:
        orderbook = trainer.matching_engine.orderbook
        if orderbook:
            best_bid = orderbook.get_best_bid()
            best_ask = orderbook.get_best_ask()
            print(f"\n订单簿状态:")
            print(f"  最佳买价: {best_bid if best_bid is not None else '无'}")
            print(f"  最佳卖价: {best_ask if best_ask is not None else '无'}")
            if best_bid is not None and best_ask is not None:
                print(f"  价差: {best_ask - best_bid}")

    # 创建并运行UI应用
    app = TrainingUIApp(trainer, episodes=args.episodes)
    print("\n启动训练UI...")
    print("按 [开始] 按钮开始训练（或仅观察当前状态）")
    print("=" * 60)
    app.run()


if __name__ == "__main__":
    main()
