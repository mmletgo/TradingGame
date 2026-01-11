#!/usr/bin/env python3
"""单竞技场训练脚本

使用 SingleArenaTrainer 进行训练，命令行参数与 train_multi_arena.py 兼容。

使用方法:
    python scripts/train_single_arena.py [选项]

示例:
    # 默认训练（10个竞技场，每个10个episode，共100轮）
    python scripts/train_single_arena.py --rounds 100

    # 自定义竞技场数量和episode数
    python scripts/train_single_arena.py --num-arenas 8 --episodes-per-arena 5 --rounds 200

    # 从检查点恢复
    python scripts/train_single_arena.py --resume checkpoints/ep_50.pkl
"""

import argparse
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
from src.training.arena import SingleArenaTrainer

from create_config import create_default_config


def progress_callback(stats: dict[str, Any]) -> None:
    """训练进度回调函数"""
    from datetime import datetime

    episode = stats.get("episode", 0)
    tick_count = stats.get("tick_count", 0)
    high_price = stats.get("high_price", 0.0)
    low_price = stats.get("low_price", 0.0)
    final_price = stats.get("final_price", 0.0)

    current_time = datetime.now().strftime("%H:%M:%S")

    # 获取种群统计
    populations = stats.get("populations", {})
    pop_info: list[str] = []
    for agent_type, pop_data in populations.items():
        if isinstance(pop_data, dict):
            count = pop_data.get("count", 0)
            gen = pop_data.get("generation", 0)
            pop_info.append(f"{agent_type}(g{gen})")

    pop_str = ", ".join(pop_info[:4])  # 只显示前4个

    print(
        f"Ep {episode:4d} | {current_time} | "
        f"Ticks={tick_count:4d} | "
        f"Price=[{low_price:.2f}-{high_price:.2f}]={final_price:.2f} | "
        f"{pop_str}"
    )


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NEAT AI 交易模拟竞技场 - 单竞技场训练模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="训练轮数（每轮 = num-arenas * episodes-per-arena 个 episode）。默认无限",
    )
    parser.add_argument(
        "--num-arenas",
        type=int,
        default=10,
        help="模拟竞技场数量（用于计算每轮episode数，默认: 10）",
    )
    parser.add_argument(
        "--episodes-per-arena",
        type=int,
        default=10,
        help="每个竞技场运行的 episode 数（用于计算每轮episode数，默认: 10）",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="每个 episode 的 tick 数量（默认: 1000）",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="检查点保存间隔（轮数，默认: 10，0 表示不保存）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从指定检查点恢复训练",
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
    parser.add_argument(
        "--catfish",
        action="store_const",
        const=True,
        default=None,
        help="启用鲶鱼机制。不指定时使用默认值",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    # 计算每轮 episode 数
    episodes_per_round = args.num_arenas * args.episodes_per_arena

    print("=" * 70)
    print("NEAT AI 交易模拟竞技场 - 单竞技场训练模式")
    print("=" * 70)
    print(f"模拟竞技场数量: {args.num_arenas}")
    print(f"每竞技场 Episode 数: {args.episodes_per_arena}")
    print(f"每轮总 Episode 数: {episodes_per_round}")
    if args.rounds:
        total_episodes = args.rounds * episodes_per_round
        print(f"训练轮数: {args.rounds}")
        print(f"总 Episode 数: {total_episodes}")
    else:
        print("训练轮数: 无限模式（Ctrl+C 停止）")
        total_episodes = None
    print(f"Episode 长度: {args.episode_length} ticks")
    print(f"检查点间隔: {args.checkpoint_interval} 轮")
    if args.resume:
        print(f"恢复检查点: {args.resume}")
    print("=" * 70)

    # 创建配置
    config_kwargs: dict[str, Any] = {
        "episode_length": args.episode_length,
        "config_dir": args.config_dir,
    }
    if args.catfish is not None:
        config_kwargs["catfish_enabled"] = args.catfish

    # 设置检查点间隔（按轮数计算）
    # checkpoint_interval 是按 episode 数计算的
    checkpoint_interval_episodes = args.checkpoint_interval * episodes_per_round
    config_kwargs["checkpoint_interval"] = checkpoint_interval_episodes

    config = create_default_config(**config_kwargs)

    # 打印 Catfish 状态
    if config.catfish is not None and config.catfish.enabled:
        print("Catfish: enabled")
    else:
        print("Catfish: disabled")
    print("=" * 70)

    # 创建训练器
    trainer = SingleArenaTrainer(config)

    # 初始化
    print("\n初始化训练环境...")
    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 恢复检查点
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"正在从检查点恢复: {args.resume}")
            trainer.load_checkpoint(args.resume)
            print(f"已恢复到 Episode {trainer.episode}")
        else:
            print(f"错误: 检查点文件不存在: {args.resume}")
            sys.exit(1)

    # 定义进度回调包装器
    def state_callback(state: dict[str, Any]) -> None:
        """状态回调包装器，添加价格统计信息"""
        # 获取价格统计
        price_stats = trainer.get_price_stats()
        state.update(price_stats)
        progress_callback(state)

    # 开始训练
    print("\n开始训练...")
    print("-" * 70)

    train_start = time.time()
    try:
        trainer.train(
            episodes=total_episodes,
            state_callback=state_callback,
        )
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        # 保存紧急检查点
        emergency_path = f"checkpoints/emergency_ep_{trainer.episode}.pkl"
        trainer.save_checkpoint(emergency_path)
        print(f"紧急检查点已保存: {emergency_path}")
    finally:
        trainer.stop()

    train_time = time.time() - train_start

    print("-" * 70)
    print("训练完成！")
    print(f"总 Episode: {trainer.episode}")
    print(f"总耗时: {train_time:.2f}s")
    if trainer.episode > 0:
        print(f"平均每 Episode: {train_time / trainer.episode:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
