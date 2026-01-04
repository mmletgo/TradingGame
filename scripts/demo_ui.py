#!/usr/bin/env python3
"""演示模式脚本

从检查点加载训练好的模型，仅运行展示，不进化。
支持速度控制，适合展示训练效果。

使用方法:
    python scripts/demo_ui.py [选项]

示例:
    # 使用随机初始化的Agent运行演示
    python scripts/demo_ui.py

    # 从检查点加载训练好的模型
    python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl

    # 自定义参数
    python scripts/demo_ui.py --checkpoint checkpoints/ep_100.pkl --episode-length 2000
"""

import argparse
import sys
from pathlib import Path

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
from src.ui.demo_app import DemoUIApp


def create_demo_config(
    episode_length: int = 1000,
    config_dir: str = "config",
) -> Config:
    """创建演示配置

    与训练配置相同，用于演示模式。

    Args:
        episode_length: 每个 episode 的 tick 数量
        config_dir: 配置文件目录

    Returns:
        演示配置对象
    """
    market = MarketConfig(
        initial_price=100.0,
        tick_size=0.1,
        lot_size=1.0,
        depth=100,
        ema_alpha=0.2,
    )

    agents = {
        AgentType.RETAIL: AgentConfig(
            count=10000,
            initial_balance=200000.0,  # 20万
            leverage=1.0,
            maintenance_margin_rate=0.1,  # 10%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.RETAIL_PRO: AgentConfig(
            count=100,
            initial_balance=200000.0,  # 20万
            leverage=1.0,
            maintenance_margin_rate=0.1,  # 10%
            maker_fee_rate=0.0002,  # 万2
            taker_fee_rate=0.0005,  # 万5
        ),
        AgentType.BULL_WHALE: AgentConfig(
            count=100,  # 多头庄家
            initial_balance=10000000.0,  # 1000万
            leverage=1.0,
            maintenance_margin_rate=0.1,  # 10%
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
        AgentType.BEAR_WHALE: AgentConfig(
            count=100,  # 空头庄家
            initial_balance=10000000.0,  # 1000万
            leverage=1.0,
            maintenance_margin_rate=0.1,  # 10%
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
        AgentType.MARKET_MAKER: AgentConfig(
            count=100,
            initial_balance=20000000.0,  # 2000万
            leverage=1.0,
            maintenance_margin_rate=0.1,  # 10%
            maker_fee_rate=-0.0001,  # 负万1 (maker rebate)
            taker_fee_rate=0.0001,  # 万1
        ),
    }

    training = TrainingConfig(
        episode_length=episode_length,
        checkpoint_interval=0,  # 演示模式不需要保存检查点
        neat_config_path=config_dir,
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
        description="NEAT AI 交易模拟竞技场 - 演示模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="检查点路径（可选，不指定则使用随机初始化的Agent）",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="每个 episode 的 tick 数量（默认: 1000）",
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
    print("NEAT AI 交易模拟竞技场 - 演示模式")
    print("=" * 60)
    print(f"Episode Length: {args.episode_length} ticks")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    else:
        print("Checkpoint: 未指定（使用随机初始化的Agent）")
    print("=" * 60)

    # 验证检查点文件存在
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"错误: 检查点文件不存在: {args.checkpoint}")
            sys.exit(1)

    # 创建配置
    config = create_demo_config(
        episode_length=args.episode_length,
        config_dir=args.config_dir,
    )

    # 创建训练器
    trainer = Trainer(config)

    # 初始化
    print("初始化演示环境...")
    trainer.setup()
    print("训练器初始化完成")

    # 预热：重置一次市场和Agent，确保所有懒加载完成
    print("预热中（重置市场和Agent）...")
    for population in trainer.populations.values():
        population.reset_agents()
    trainer._reset_market()

    # 运行一个测试tick确保一切就绪
    print("运行测试tick...")
    trainer.run_tick()
    trainer.tick = 0  # 重置tick计数

    # 再次重置，准备正式演示
    for population in trainer.populations.values():
        population.reset_agents()
    trainer._reset_market()
    print("预热完成，所有初始化已就绪")

    # 创建并运行演示UI
    app = DemoUIApp(trainer, checkpoint_path=args.checkpoint)
    print("启动演示UI...")
    print("按 [开始] 按钮开始演示")
    print("使用速度滑块调整演示速度")
    app.run()


if __name__ == "__main__":
    main()
