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


def create_default_config(
    episode_length: int = 1000,
    checkpoint_interval: int = 10,
    config_dir: str = "config",
) -> Config:
    """创建默认配置

    Args:
        episode_length: 每个 episode 的 tick 数量
        checkpoint_interval: 检查点间隔（episode 数）
        config_dir: 配置文件目录（Population 会在此目录下查找对应的 NEAT 配置）

    Returns:
        默认配置对象
    """
    market = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
    )

    agents = {
        AgentType.RETAIL: AgentConfig(
            count=10000,
            initial_balance=10000.0,  # 1万
            leverage=100.0,
            maintenance_margin_rate=0.005,  # 0.5%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.RETAIL_PRO: AgentConfig(
            count=100,
            initial_balance=10000.0,  # 1万
            leverage=100.0,
            maintenance_margin_rate=0.005,  # 0.5%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.WHALE: AgentConfig(
            count=100,
            initial_balance=10000000.0,  # 1000万
            leverage=10.0,
            maintenance_margin_rate=0.05,  # 5%
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
        AgentType.MARKET_MAKER: AgentConfig(
            count=1000,
            initial_balance=10000000.0,  # 1000万
            leverage=10.0,
            maintenance_margin_rate=0.05,  # 5%
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
    }

    training = TrainingConfig(
        episode_length=episode_length,
        checkpoint_interval=checkpoint_interval,
        neat_config_path=config_dir,  # 配置目录，Population 会自动选择对应的配置文件
    )

    demo = DemoConfig(
        host="localhost",
        port=8000,
        tick_interval=100,
    )

    return Config(market=market, agents=agents, training=training, demo=demo)


def progress_callback(state: dict[str, Any]) -> None:
    """训练进度回调函数

    Args:
        state: 训练状态字典
    """
    episode = state.get("episode", 0)
    populations = state.get("populations", {})

    info_parts = [f"Episode {episode}"]
    for agent_type, pop_info in populations.items():
        gen = pop_info.get("generation", 0)
        count = pop_info.get("count", 0)
        info_parts.append(f"{agent_type}: gen={gen}, count={count}")

    print(" | ".join(info_parts))


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NEAT AI 交易模拟竞技场 - 无 UI 训练模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="训练的 episode 数量（默认: 100）",
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

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_dir)

    print("=" * 60)
    print("NEAT AI 交易模拟竞技场 - 无 UI 训练模式")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Episode Length: {args.episode_length} ticks")
    print(f"Checkpoint Interval: {args.checkpoint_interval}")
    if args.resume:
        print(f"Resume From: {args.resume}")
    print("=" * 60)

    # 创建配置
    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=args.checkpoint_interval,
        config_dir=args.config_dir,
    )

    # 创建训练器
    trainer = Trainer(config)

    # 初始化
    print("初始化训练环境...")
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
            print(f"警告: 检查点文件不存在: {args.resume}")
            sys.exit(1)

    # 开始训练
    print("\n开始训练...")
    print("-" * 60)

    train_start = time.time()
    try:
        trainer.train(episodes=args.episodes, state_callback=progress_callback)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        # 保存紧急检查点
        emergency_path = f"checkpoints/emergency_ep_{trainer.episode}.pkl"
        trainer.save_checkpoint(emergency_path)
        print(f"紧急检查点已保存: {emergency_path}")

    train_time = time.time() - train_start

    print("-" * 60)
    print("训练完成！")
    print(f"总 Episode: {trainer.episode}")
    print(f"总耗时: {train_time:.2f}s")
    print(f"平均每 Episode: {train_time / max(args.episodes, 1):.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
