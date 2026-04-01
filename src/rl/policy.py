"""Actor-Critic 策略模块

将转换后的 NEAT 网络包装为 RL 兼容的策略：
- Actor: NEAT 网络输出动作均值
- log_std: 可学习的动作标准差参数
- Critic: 独立的 MLP 估计状态价值
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.rl.converter import NEATNetwork


class ActorCriticPolicy(nn.Module):
    """Actor-Critic 策略

    Actor（NEAT 网络）输出确定性动作均值。
    可学习的 log_std 参数创建随机策略。
    独立的 MLP Critic 估计状态价值。

    支持 actor 冻结/解冻调度：先只训练 critic 和 log_std，
    后期解冻 actor 用更小的学习率微调。

    Attributes:
        actor: 转换后的 NEAT 网络
        log_std: 动作标准差参数（对数空间）
        critic: 状态价值网络
    """

    def __init__(
        self,
        actor: NEATNetwork,
        obs_dim: int,
        act_dim: int,
        initial_log_std: float = -1.0,
        critic_hidden_sizes: tuple[int, ...] = (256, 256),
        freeze_actor: bool = True,
    ) -> None:
        """
        Args:
            actor: 转换后的 NEAT 网络
            obs_dim: 观测空间维度（67 散户，132 做市商）
            act_dim: 动作空间维度（3 散户，43 做市商）
            initial_log_std: 初始动作标准差（对数，-1.0 ≈ std=0.37，保守探索）
            critic_hidden_sizes: critic MLP 隐藏层尺寸
            freeze_actor: 是否冻结 actor 参数
        """
        super().__init__()

        self.actor = actor
        self.log_std = nn.Parameter(torch.full((act_dim,), initial_log_std))

        # 构建 critic MLP
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h_size in critic_hidden_sizes:
            layers.append(nn.Linear(in_dim, h_size))
            layers.append(nn.Tanh())
            in_dim = h_size
        layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*layers)

        # 冻结 actor
        if freeze_actor:
            self.freeze_actor()

    def freeze_actor(self) -> None:
        """冻结 actor 参数（不参与梯度更新）"""
        for param in self.actor.parameters():
            param.requires_grad = False

    def unfreeze_actor(self) -> None:
        """解冻 actor 参数"""
        for param in self.actor.parameters():
            param.requires_grad = True

    @property
    def actor_frozen(self) -> bool:
        """actor 是否被冻结"""
        return not next(self.actor.parameters()).requires_grad

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取动作、log_prob、entropy 和 value

        Args:
            obs: 观测向量 shape (batch, obs_dim)
            action: 如果提供，计算该动作的 log_prob（PPO 更新时使用）
            deterministic: 是否使用确定性动作（评估时使用）

        Returns:
            (action, log_prob, entropy, value)
            - action: shape (batch, act_dim)
            - log_prob: shape (batch,)
            - entropy: shape (batch,)
            - value: shape (batch, 1)
        """
        # Actor 前向传播
        action_mean = self.actor(obs)  # (batch, act_dim)
        std = self.log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, std)

        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.rsample()  # 重参数化采样

        # 计算 log_prob 和 entropy
        log_prob = dist.log_prob(action).sum(dim=-1)  # (batch,)
        entropy = dist.entropy().sum(dim=-1)           # (batch,)

        # Critic 前向传播
        value = self.critic(obs)  # (batch, 1)

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """仅计算状态价值（GAE 计算时使用）

        Args:
            obs: shape (batch, obs_dim)

        Returns:
            shape (batch, 1)
        """
        return self.critic(obs)

    def get_action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """获取确定性动作均值（部署时使用）

        Args:
            obs: shape (batch, obs_dim) 或 (obs_dim,)

        Returns:
            shape (batch, act_dim) 或 (act_dim,)
        """
        return self.actor(obs)

    def get_param_groups(self, actor_lr_scale: float = 0.1, base_lr: float = 3e-4) -> list[dict]:
        """获取分组学习率的参数组

        Actor 使用更小的学习率（base_lr * actor_lr_scale），
        Critic 和 log_std 使用正常学习率。

        Args:
            actor_lr_scale: actor 学习率缩放因子
            base_lr: 基础学习率

        Returns:
            optimizer param_groups 列表
        """
        return [
            {"params": self.actor.parameters(), "lr": base_lr * actor_lr_scale},
            {"params": self.critic.parameters(), "lr": base_lr},
            {"params": [self.log_std], "lr": base_lr},
        ]
