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
import importlib
import sys
from pathlib import Path

# 关键：在导入任何项目模块之前，先清除 importlib 缓存
importlib.invalidate_caches()

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer
from src.ui.demo_app import DemoUIApp

from create_config import create_default_config


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

    # 创建配置（使用与训练相同的默认配置，但不保存检查点）
    config = create_default_config(
        episode_length=args.episode_length,
        checkpoint_interval=0,  # 演示模式不保存检查点
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
