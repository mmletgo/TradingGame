#!/usr/bin/env python3
"""无 UI 训练脚本

用于高性能后台训练，不启动 WebUI。
支持命令行参数配置和检查点恢复。

使用方法:
    python scripts/train_noui.py [选项]

示例:
    # 训练 100 个 episode
    python scripts/train_noui.py --episodes 100

    # 从检查点恢复训练
    python scripts/train_noui.py --resume checkpoints/ep_50.pkl --episodes 100

    # 自定义参数
    python scripts/train_noui.py --episodes 500 --checkpoint-interval 50
"""

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

# 关键：在导入任何项目模块之前，先清除 importlib 缓存
# 这可以解决修改代码后由于缓存导致的运行时问题
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.log_engine.logger import setup_logging
from src.training.generation_saver import GenerationSaver
from src.training.trainer import Trainer


from create_config import create_default_config


def progress_callback(state: dict[str, Any]) -> None:
    """训练进度回调函数

    Args:
        state: 训练状态字典
    """
    from datetime import datetime

    episode = state.get("episode", 0)
    high_price = state.get("high_price", 0.0)
    low_price = state.get("low_price", 0.0)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(
        f"Episode {episode} | {current_time} | high={high_price:.2f}, low={low_price:.2f}"
    )


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NEAT AI 交易模拟竞技场 - 无 UI 训练模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=4000,
        help="训练的 episode 数量（默认: 4000）。与 --infinite 互斥",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="无限训练模式，直到手动中断（Ctrl+C）。与 --episodes 互斥",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100,
        help="每个 episode 的 tick 数量（默认: 100）",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="检查点保存间隔（episode 数，默认: 10，0 表示不保存）",
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
        help="启用鲶鱼机制（三种行为模式同时运行）。不指定时使用 create_config.py 中的默认值。",
    )
    parser.add_argument(
        "--catfish-fund-multiplier",
        type=float,
        default=3.0,
        help="鲶鱼资金倍数（相对于做市商，默认: 3.0）",
    )

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    # 确定训练模式
    infinite_mode = args.infinite
    episodes_to_train: int | None = None if infinite_mode else args.episodes

    print("=" * 60)
    print("NEAT AI 交易模拟竞技场 - 无 UI 训练模式")
    print("=" * 60)
    if infinite_mode:
        print("Episodes: 无限模式（Ctrl+C 停止）")
    else:
        print(f"Episodes: {args.episodes}")
    print(f"Episode Length: {args.episode_length} ticks")
    print(f"Checkpoint Interval: {args.checkpoint_interval}")
    if args.resume:
        print(f"Resume From: {args.resume}")

    # 创建配置（必须在打印 catfish 状态之前，以获取最终配置）
    config_kwargs = {
        "episode_length": args.episode_length,
        "checkpoint_interval": args.checkpoint_interval,
        "config_dir": args.config_dir,
        "catfish_fund_multiplier": args.catfish_fund_multiplier,
    }
    # 只有在用户明确指定 --catfish 时才覆盖默认值
    if args.catfish is not None:
        config_kwargs["catfish_enabled"] = args.catfish

    config = create_default_config(**config_kwargs)

    # 打印 Catfish 状态（使用最终配置值）
    if config.catfish is not None and config.catfish.enabled:
        print(
            f"Catfish: enabled (三种模式同时运行), multiplier={config.catfish.fund_multiplier}x"
        )
    else:
        print("Catfish: disabled")
    print("=" * 60)

    # 创建训练器
    trainer = Trainer(config)

    # 设置每代保存器
    generation_saver = GenerationSaver(output_dir="checkpoints/generations")
    trainer.set_generation_saver(generation_saver)

    # 初始化
    print("初始化训练环境...")
    start_time = time.time()
    trainer.setup()
    init_time = time.time() - start_time
    print(f"初始化完成（耗时: {init_time:.2f}s）")

    # 恢复检查点（自动查找最新或使用指定的）
    resume_path_str = args.resume
    if resume_path_str is None:
        # 自动查找最新的检查点
        resume_path_str = Trainer.find_latest_checkpoint()
        if resume_path_str:
            print(f"自动发现最新检查点: {resume_path_str}")

    if resume_path_str:
        resume_path = Path(resume_path_str)
        if resume_path.exists():
            print(f"正在从检查点恢复: {resume_path_str}")
            trainer.load_checkpoint(resume_path_str)
            print(f"已恢复到 Episode {trainer.episode}")
        else:
            print(f"警告: 检查点文件不存在: {resume_path_str}")
            sys.exit(1)

    # 开始训练
    print("\n开始训练...")
    print("-" * 60)

    train_start = time.time()
    try:
        trainer.train(episodes=episodes_to_train, state_callback=progress_callback)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        # 保存紧急检查点
        emergency_path = f"checkpoints/emergency_ep_{trainer.episode}.pkl"
        trainer.save_checkpoint(emergency_path)
        print(f"紧急检查点已保存: {emergency_path}")

    train_time = time.time() - train_start
    trained_episodes = episodes_to_train if episodes_to_train else trainer.episode

    print("-" * 60)
    print("训练完成！")
    print(f"总 Episode: {trainer.episode}")
    print(f"总耗时: {train_time:.2f}s")
    print(f"平均每 Episode: {train_time / max(trained_episodes, 1):.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
