"""RL 训练配置"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.replay.config import ReplayConfig


@dataclass
class RLConfig:
    """RL 训练配置

    Attributes:
        checkpoint_path: NEAT checkpoint 文件路径
        agent_type: "RETAIL_PRO" 或 "MARKET_MAKER"
        total_timesteps: 总训练步数
        learning_rate: 学习率
        gamma: 折扣因子
        gae_lambda: GAE lambda
        clip_range: PPO clip 范围
        n_steps: 每次 rollout 的步数
        batch_size: mini-batch 大小
        n_epochs: 每次更新的 epoch 数
        ent_coef: 熵正则化系数
        vf_coef: value loss 系数
        max_grad_norm: 梯度裁剪
        initial_log_std: 初始动作标准差（对数）
        critic_hidden_sizes: critic MLP 隐藏层尺寸
        freeze_actor_steps: 冻结 actor 的步数（0=不冻结）
        actor_lr_scale: 解冻 actor 后的学习率缩放
        output_dir: 输出目录
        log_interval: 日志间隔（update 次数）
        save_interval: checkpoint 保存间隔（步数）
    """

    checkpoint_path: str = ""
    agent_type: str = "RETAIL_PRO"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    initial_log_std: float = -1.0
    critic_hidden_sizes: tuple[int, ...] = (256, 256)
    freeze_actor_steps: int = 50_000
    actor_lr_scale: float = 0.1
    output_dir: str = "rl_checkpoints"
    log_interval: int = 10
    save_interval: int = 50_000
