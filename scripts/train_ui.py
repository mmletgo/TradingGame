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

from src.core.log_engine.logger import setup_logging
from src.training.trainer import Trainer
from src.ui.training_app import TrainingUIApp

from create_config import create_default_config


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="NEAT AI 交易模拟竞技场 - 带UI训练模式",
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
        default=1000,
        help="每个 episode 的 tick 数量（默认: 1000）",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="检查点保存间隔（episode 数，默认: 100，0 表示不保存）",
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
        "--evolution-interval",
        type=int,
        default=10,
        help="每多少个 episode 进化一次（默认: 10）",
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

    # 创建配置（必须在打印 catfish 状态之前，以获取最终配置）
    config_kwargs = {
        "episode_length": args.episode_length,
        "checkpoint_interval": args.checkpoint_interval,
        "config_dir": args.config_dir,
    }
    # 只有在用户明确指定 --catfish 时才覆盖默认值
    if args.catfish is not None:
        config_kwargs["catfish_enabled"] = args.catfish

    config = create_default_config(**config_kwargs)

    # 打印 Catfish 状态（使用最终配置值）
    if config.catfish is not None and config.catfish.enabled:
        print(
            f"Catfish: enabled (三种模式同时运行), action_probability={config.catfish.action_probability:.1%}",
            flush=True,
        )
    else:
        print("Catfish: disabled", flush=True)
    print("=" * 60, flush=True)

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

    # 加载检查点（自动查找最新或使用指定的）
    resume_path_str = args.resume
    if resume_path_str is None:
        # 自动查找最新的检查点
        resume_path_str = Trainer.find_latest_checkpoint()
        if resume_path_str:
            print(f"自动发现最新检查点: {resume_path_str}", flush=True)

    if resume_path_str:
        resume_path = Path(resume_path_str)
        if resume_path.exists():
            print(f"正在从检查点恢复: {resume_path_str}", flush=True)
            trainer.load_checkpoint(resume_path_str)
            print(f"已恢复到 Episode {trainer.episode}", flush=True)
        else:
            print(f"警告: 检查点文件不存在: {resume_path_str}", flush=True)
            sys.exit(1)

    # 创建并运行UI应用
    app = TrainingUIApp(trainer, episodes=args.episodes)
    print("启动训练UI...", flush=True)
    print("按 [开始] 按钮开始训练", flush=True)
    app.run()


if __name__ == "__main__":
    main()
