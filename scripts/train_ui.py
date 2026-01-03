#!/usr/bin/env python3
"""带UI界面的训练脚本

提供可视化界面进行NEAT训练，支持实时监控订单簿、价格走势、
成交记录和种群统计信息。

使用方法:
    python scripts/train_ui.py [选项]

示例:
    # 训练 100 个 episode
    python scripts/train_ui.py --episodes 100

    # 从检查点恢复训练
    python scripts/train_ui.py --resume checkpoints/ep_50.pkl --episodes 100

    # 自定义参数
    python scripts/train_ui.py --episodes 500 --episode-length 2000
"""

import argparse
import importlib
import sys
from pathlib import Path

# 关键：在导入任何项目模块之前，先清除 importlib 缓存
# 这可以解决修改代码后由于缓存导致的运行时问题
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.config import (
    AgentConfig,
    AgentType,
    Config,
    DemoConfig,
    MarketConfig,
    TrainingConfig,
)
from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer
from src.ui.training_app import TrainingUIApp


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
            initial_balance=100000.0,  # 10万
            leverage=1.0,
            maintenance_margin_rate=0.5,  # 50%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.RETAIL_PRO: AgentConfig(
            count=100,
            initial_balance=100000.0,  # 10万
            leverage=1.0,
            maintenance_margin_rate=0.5,  # 0%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.WHALE: AgentConfig(
            count=100,
            initial_balance=10000000.0,  # 1000万
            leverage=1.0,
            maintenance_margin_rate=0.5,  # 50%
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
        AgentType.MARKET_MAKER: AgentConfig(
            count=100,
            initial_balance=10000000.0,  # 1000万
            leverage=1.0,
            maintenance_margin_rate=0.5,  # 50%
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


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NEAT AI 交易模拟竞技场 - 带UI训练模式",
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

    print("=" * 60, flush=True)
    print("NEAT AI 交易模拟竞技场 - 带UI训练模式", flush=True)
    print("=" * 60, flush=True)
    print(f"Episodes: {args.episodes}", flush=True)
    print(f"Episode Length: {args.episode_length} ticks", flush=True)
    print(f"Checkpoint Interval: {args.checkpoint_interval}", flush=True)
    if args.resume:
        print(f"Resume From: {args.resume}", flush=True)
    print("=" * 60, flush=True)

    # 创建配置
    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=args.checkpoint_interval,
        config_dir=args.config_dir,
    )

    # 创建训练器
    trainer = Trainer(config)

    # 初始化
    print("初始化训练环境...", flush=True)
    trainer.setup()
    print("训练器初始化完成", flush=True)

    # 预热：重置一次市场和Agent，确保所有懒加载完成
    print("预热中（重置市场和Agent）...", flush=True)
    for population in trainer.populations.values():
        population.reset_agents()
    trainer._reset_market()

    # 运行一个测试tick确保一切就绪
    print("运行测试tick...", flush=True)
    trainer.run_tick()
    trainer.tick = 0  # 重置tick计数

    # 再次重置，准备正式训练
    for population in trainer.populations.values():
        population.reset_agents()
    trainer._reset_market()
    print("预热完成，所有初始化已就绪", flush=True)

    # 加载检查点
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"正在从检查点恢复: {args.resume}", flush=True)
            trainer.load_checkpoint(args.resume)
            print(f"已恢复到 Episode {trainer.episode}", flush=True)
        else:
            print(f"警告: 检查点文件不存在: {args.resume}", flush=True)
            sys.exit(1)

    # 创建并运行UI应用
    app = TrainingUIApp(trainer, episodes=args.episodes)
    print("启动训练UI...", flush=True)
    print("按 [开始] 按钮开始训练", flush=True)
    app.run()


if __name__ == "__main__":
    main()
