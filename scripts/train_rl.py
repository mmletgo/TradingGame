"""RL 微调训练脚本

将 NEAT 进化出的交易策略转换为 PyTorch 网络，
在真实市场数据回放环境中使用 PPO 算法进行微调。

用法:
    python scripts/train_rl.py \\
        --checkpoint checkpoints/ep_100.pkl \\
        --agent-type RETAIL_PRO \\
        --data-dir /path/to/HFtrade/monitor/data \\
        --exchange binance_usdt_swap \\
        --pair btc_usdt \\
        --date-start 2026-03-01 \\
        --date-end 2026-03-15 \\
        --tick-size 0.01 \\
        --total-timesteps 1000000
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

# 清除缓存并设置项目根目录
importlib.invalidate_caches()
project_root: Path = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.replay.config import ReplayConfig
from src.replay.replay_env import ReplayEnv
from src.rl.config import RLConfig
from src.rl.converter import NEATtoPyTorchConverter
from src.rl.policy import ActorCriticPolicy
from src.rl.trainer import PPOTrainer


def main() -> None:
    """训练入口"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="RL 微调训练"
    )

    # NEAT checkpoint
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="NEAT checkpoint 路径"
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default="RETAIL_PRO",
        choices=["RETAIL_PRO", "MARKET_MAKER"],
    )
    parser.add_argument(
        "--genome-rank",
        type=int,
        default=0,
        help="使用第几名的基因组（0=最优）",
    )

    # 数据
    parser.add_argument(
        "--data-dir", type=str, required=True, help="HFtrade monitor/data 路径"
    )
    parser.add_argument(
        "--exchange", type=str, default="binance_usdt_swap"
    )
    parser.add_argument("--pair", type=str, default="btc_usdt")
    parser.add_argument("--date-start", type=str, required=True)
    parser.add_argument("--date-end", type=str, required=True)
    parser.add_argument("--tick-size", type=float, default=0.01)

    # 回放环境
    parser.add_argument(
        "--initial-balance", type=float, default=20_000.0, help="初始资金"
    )
    parser.add_argument("--leverage", type=float, default=10.0, help="杠杆倍数")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=10_000,
        help="每 episode 最大步数",
    )

    # 训练超参数
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--freeze-actor-steps", type=int, default=50_000)
    parser.add_argument("--actor-lr-scale", type=float, default=0.1)
    parser.add_argument("--initial-log-std", type=float, default=-1.0)
    parser.add_argument("--output-dir", type=str, default="rl_checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50_000)

    # 恢复训练
    parser.add_argument(
        "--resume", type=str, default="", help="恢复训练的 RL checkpoint 路径"
    )

    args: argparse.Namespace = parser.parse_args()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger: logging.Logger = logging.getLogger("train_rl")

    # 1. 转换 NEAT -> PyTorch
    logger.info(f"加载 NEAT checkpoint: {args.checkpoint}")
    results = NEATtoPyTorchConverter.convert_from_checkpoint(
        args.checkpoint, args.agent_type
    )
    if not results:
        logger.error("未找到可转换的基因组")
        sys.exit(1)

    if args.genome_rank >= len(results):
        logger.error(
            f"genome_rank={args.genome_rank} 超出范围 "
            f"(共 {len(results)} 个基因组)"
        )
        sys.exit(1)

    genome_id: int
    fitness: float
    genome_id, actor, fitness = results[args.genome_rank]
    logger.info(
        f"使用基因组 #{genome_id} "
        f"(rank={args.genome_rank}, fitness={fitness:.4f})"
    )

    # 2. 确定维度
    obs_dim: int
    act_dim: int
    if args.agent_type == "RETAIL_PRO":
        obs_dim, act_dim = 527, 3
    else:
        obs_dim, act_dim = 592, 43

    # 3. 创建回放环境
    replay_config: ReplayConfig = ReplayConfig(
        hftrade_data_dir=args.data_dir,
        exchange=args.exchange,
        pair=args.pair,
        date_start=args.date_start,
        date_end=args.date_end,
        tick_size=args.tick_size,
        agent_type=args.agent_type,
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        episode_length=args.episode_length,
    )
    logger.info("加载市场数据...")
    env: ReplayEnv = ReplayEnv(replay_config)
    logger.info(f"数据加载完成: {env._num_snapshots} 个订单簿快照")

    # 4. 创建策略和训练配置
    rl_config: RLConfig = RLConfig(
        checkpoint_path=args.checkpoint,
        agent_type=args.agent_type,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        initial_log_std=args.initial_log_std,
        freeze_actor_steps=args.freeze_actor_steps,
        actor_lr_scale=args.actor_lr_scale,
        output_dir=args.output_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    policy: ActorCriticPolicy = ActorCriticPolicy(
        actor=actor,
        obs_dim=obs_dim,
        act_dim=act_dim,
        initial_log_std=rl_config.initial_log_std,
        critic_hidden_sizes=rl_config.critic_hidden_sizes,
        freeze_actor=rl_config.freeze_actor_steps > 0,
    )

    # 5. 创建训练器
    trainer: PPOTrainer = PPOTrainer(rl_config, env, policy)

    # 恢复训练
    if args.resume:
        logger.info(f"恢复训练: {args.resume}")
        trainer.load(args.resume)

    # 6. 开始训练
    logger.info("=" * 60)
    logger.info("开始 RL 训练")
    logger.info(f"  Agent 类型: {args.agent_type}")
    logger.info(f"  观测维度: {obs_dim}")
    logger.info(f"  动作维度: {act_dim}")
    logger.info(f"  总步数: {args.total_timesteps:,}")
    logger.info(f"  每轮 rollout 步数: {args.n_steps}")
    logger.info(f"  Mini-batch 大小: {args.batch_size}")
    logger.info(f"  PPO epoch 数: {args.n_epochs}")
    logger.info(f"  学习率: {args.learning_rate}")
    logger.info(f"  Actor 冻结步数: {args.freeze_actor_steps:,}")
    logger.info(f"  Actor LR 缩放: {args.actor_lr_scale}")
    logger.info(f"  输出目录: {args.output_dir}")
    logger.info("=" * 60)

    metrics: dict[str, list[float]] = trainer.train()

    # 7. 打印最终指标
    if metrics["episode_reward"]:
        final_rewards: list[float] = metrics["episode_reward"][-100:]
        avg_reward: float = sum(final_rewards) / len(final_rewards)
        logger.info(
            f"最终 {len(final_rewards)} episode 平均 reward: {avg_reward:.4f}"
        )
        logger.info(
            f"最终 policy_loss: {metrics['policy_loss'][-1]:.4f}, "
            f"value_loss: {metrics['value_loss'][-1]:.4f}, "
            f"entropy: {metrics['entropy'][-1]:.4f}"
        )
    else:
        logger.warning("训练过程中未完成任何 episode")


if __name__ == "__main__":
    main()
