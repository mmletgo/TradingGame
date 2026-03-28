"""PPO 训练器（CleanRL 风格）

自定义 PPO-Clip 实现，用于对 NEAT 进化出的交易策略进行 RL 微调。

特性:
- GAE 优势估计
- PPO-Clip 目标函数
- Value function clipping
- 梯度裁剪
- Actor 冻结/解冻调度（先训练 critic + log_std，后期解冻 actor 用更小学习率微调）
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.replay.replay_env import ReplayEnv
from src.rl.config import RLConfig
from src.rl.policy import ActorCriticPolicy

logger = logging.getLogger(__name__)


class PPOTrainer:
    """PPO-Clip 训练器

    训练流程:
    1. 收集 rollout（n_steps 步交互数据）
    2. 计算 GAE 优势和 returns
    3. Mini-batch PPO 更新（n_epochs 个 epoch）
    4. 定期保存 checkpoint 和输出日志
    """

    def __init__(
        self,
        config: RLConfig,
        env: ReplayEnv,
        policy: ActorCriticPolicy,
        device: torch.device | None = None,
    ) -> None:
        """初始化训练器

        Args:
            config: RL 训练配置
            env: Gymnasium 回放环境
            policy: Actor-Critic 策略网络
            device: 计算设备（默认自动选择）
        """
        self._config: RLConfig = config
        self._env: ReplayEnv = env
        self._policy: ActorCriticPolicy = policy
        self._device: torch.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._policy.to(self._device)

        # 优化器（分组学习率）
        self._optimizer: Adam = Adam(
            self._policy.get_param_groups(
                actor_lr_scale=config.actor_lr_scale,
                base_lr=config.learning_rate,
            ),
            eps=1e-5,
        )

        # 训练状态
        self._global_step: int = 0
        self._num_updates: int = 0
        self._actor_unfrozen: bool = False

    def train(self) -> dict[str, list[float]]:
        """主训练循环

        按 CleanRL 风格执行:
        - 外循环: 每轮收集 n_steps 步数据
        - 内循环: n_epochs 个 epoch 的 mini-batch PPO 更新

        Returns:
            训练指标字典，包含 episode_reward, episode_length,
            policy_loss, value_loss, entropy, approx_kl
        """
        config: RLConfig = self._config
        n_steps: int = config.n_steps
        batch_size: int = config.batch_size
        n_epochs: int = config.n_epochs
        total_timesteps: int = config.total_timesteps

        # 计算总更新次数
        num_updates: int = total_timesteps // n_steps

        # Rollout 缓冲区维度
        obs_dim: int = self._env.observation_space.shape[0]
        act_dim: int = self._env.action_space.shape[0]

        # 指标记录
        metrics: dict[str, list[float]] = {
            "episode_reward": [],
            "episode_length": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
        }

        obs_np, _ = self._env.reset()
        obs_tensor: torch.Tensor = torch.tensor(
            obs_np, dtype=torch.float32, device=self._device
        )
        episode_reward: float = 0.0
        episode_length: int = 0

        start_time: float = time.time()

        for update in range(1, num_updates + 1):
            # === 收集 rollout ===
            batch_obs: torch.Tensor = torch.zeros(
                (n_steps, obs_dim), device=self._device
            )
            batch_actions: torch.Tensor = torch.zeros(
                (n_steps, act_dim), device=self._device
            )
            batch_log_probs: torch.Tensor = torch.zeros(
                n_steps, device=self._device
            )
            batch_rewards: torch.Tensor = torch.zeros(
                n_steps, device=self._device
            )
            batch_dones: torch.Tensor = torch.zeros(
                n_steps, device=self._device
            )
            batch_values: torch.Tensor = torch.zeros(
                n_steps, device=self._device
            )

            for step in range(n_steps):
                batch_obs[step] = obs_tensor

                with torch.no_grad():
                    action, log_prob, _, value = self._policy.get_action_and_value(
                        obs_tensor.unsqueeze(0)
                    )
                    action = action.squeeze(0)
                    log_prob = log_prob.squeeze(0)
                    value = value.squeeze()

                batch_actions[step] = action
                batch_log_probs[step] = log_prob
                batch_values[step] = value

                # 环境 step
                action_np: np.ndarray = action.cpu().numpy().clip(-1, 1)
                next_obs_np, reward, terminated, truncated, info = self._env.step(
                    action_np
                )
                done: bool = terminated or truncated

                batch_rewards[step] = reward
                batch_dones[step] = float(done)

                episode_reward += reward
                episode_length += 1
                self._global_step += 1

                if done:
                    metrics["episode_reward"].append(episode_reward)
                    metrics["episode_length"].append(float(episode_length))
                    episode_reward = 0.0
                    episode_length = 0
                    next_obs_np, _ = self._env.reset()

                obs_tensor = torch.tensor(
                    next_obs_np, dtype=torch.float32, device=self._device
                )

            # === 计算 GAE ===
            with torch.no_grad():
                next_value: torch.Tensor = self._policy.get_value(
                    obs_tensor.unsqueeze(0)
                ).squeeze()

            advantages, returns = self._compute_gae(
                batch_rewards, batch_values, batch_dones, next_value
            )

            # === PPO 更新 ===
            b_obs: torch.Tensor = batch_obs
            b_actions: torch.Tensor = batch_actions
            b_log_probs: torch.Tensor = batch_log_probs
            b_advantages: torch.Tensor = advantages
            b_returns: torch.Tensor = returns
            b_values: torch.Tensor = batch_values

            # 归一化优势
            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

            # Mini-batch 更新
            indices: np.ndarray = np.arange(n_steps)
            policy_loss_val: float = 0.0
            value_loss_val: float = 0.0
            entropy_val: float = 0.0
            approx_kl_val: float = 0.0

            for epoch in range(n_epochs):
                np.random.shuffle(indices)
                for start in range(0, n_steps, batch_size):
                    end: int = start + batch_size
                    mb_idx: np.ndarray = indices[start:end]

                    mb_obs: torch.Tensor = b_obs[mb_idx]
                    mb_actions: torch.Tensor = b_actions[mb_idx]
                    mb_old_log_probs: torch.Tensor = b_log_probs[mb_idx]
                    mb_advantages: torch.Tensor = b_advantages[mb_idx]
                    mb_returns: torch.Tensor = b_returns[mb_idx]
                    mb_old_values: torch.Tensor = b_values[mb_idx]

                    _, new_log_prob, entropy, new_value = (
                        self._policy.get_action_and_value(mb_obs, mb_actions)
                    )
                    new_value = new_value.squeeze(-1)

                    # PPO-Clip 策略损失
                    ratio: torch.Tensor = (new_log_prob - mb_old_log_probs).exp()
                    surr1: torch.Tensor = ratio * mb_advantages
                    surr2: torch.Tensor = (
                        torch.clamp(
                            ratio,
                            1.0 - config.clip_range,
                            1.0 + config.clip_range,
                        )
                        * mb_advantages
                    )
                    policy_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

                    # Value loss (clipped)
                    v_clipped: torch.Tensor = mb_old_values + torch.clamp(
                        new_value - mb_old_values,
                        -config.clip_range,
                        config.clip_range,
                    )
                    v_loss1: torch.Tensor = (new_value - mb_returns).pow(2)
                    v_loss2: torch.Tensor = (v_clipped - mb_returns).pow(2)
                    value_loss: torch.Tensor = (
                        torch.max(v_loss1, v_loss2).mean() * 0.5
                    )

                    # Entropy bonus
                    entropy_loss: torch.Tensor = -entropy.mean()

                    # Total loss
                    loss: torch.Tensor = (
                        policy_loss
                        + config.vf_coef * value_loss
                        + config.ent_coef * entropy_loss
                    )

                    self._optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self._policy.parameters(), config.max_grad_norm
                    )
                    self._optimizer.step()

                    # 记录最后一个 mini-batch 的指标
                    with torch.no_grad():
                        approx_kl_val = (
                            ((ratio - 1) - ratio.log()).mean().item()
                        )

                    policy_loss_val = policy_loss.item()
                    value_loss_val = value_loss.item()
                    entropy_val = -entropy_loss.item()

            self._num_updates += 1

            # 记录指标
            metrics["policy_loss"].append(policy_loss_val)
            metrics["value_loss"].append(value_loss_val)
            metrics["entropy"].append(entropy_val)
            metrics["approx_kl"].append(approx_kl_val)

            # Actor 解冻调度
            if (
                not self._actor_unfrozen
                and config.freeze_actor_steps > 0
                and self._global_step >= config.freeze_actor_steps
            ):
                self._unfreeze_actor()

            # 日志
            if update % config.log_interval == 0:
                avg_reward: float = (
                    float(np.mean(metrics["episode_reward"][-10:]))
                    if metrics["episode_reward"]
                    else 0.0
                )
                elapsed: float = time.time() - start_time
                sps: float = self._global_step / elapsed if elapsed > 0 else 0.0
                logger.info(
                    f"Update {update}/{num_updates} | "
                    f"Step {self._global_step}/{total_timesteps} | "
                    f"Reward(10ep): {avg_reward:.4f} | "
                    f"PL: {policy_loss_val:.4f} | "
                    f"VL: {value_loss_val:.4f} | "
                    f"Ent: {entropy_val:.4f} | "
                    f"KL: {approx_kl_val:.4f} | "
                    f"SPS: {sps:.0f}"
                )

            # 保存 checkpoint
            if (
                config.save_interval > 0
                and self._global_step % config.save_interval < n_steps
            ):
                self.save(
                    str(
                        Path(config.output_dir)
                        / f"step_{self._global_step}.pt"
                    )
                )

        # 最终保存
        self.save(str(Path(config.output_dir) / "final.pt"))

        total_time: float = time.time() - start_time
        logger.info(
            f"训练完成: {self._global_step} 步, "
            f"{self._num_updates} 次更新, "
            f"耗时 {total_time:.1f}s"
        )

        return metrics

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 GAE 优势和 returns

        使用广义优势估计（Generalized Advantage Estimation）:
        A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

        Args:
            rewards: shape (n_steps,) 每步 reward
            values: shape (n_steps,) 每步 value 估计
            dones: shape (n_steps,) 每步终止标志
            next_value: 最后一步之后的 value 估计

        Returns:
            (advantages, returns) 各 shape (n_steps,)
        """
        gamma: float = self._config.gamma
        gae_lambda: float = self._config.gae_lambda
        n_steps: int = len(rewards)

        advantages: torch.Tensor = torch.zeros(n_steps, device=self._device)
        last_gae: float = 0.0

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val: torch.Tensor = next_value
            else:
                next_val = values[t + 1]

            next_non_terminal: float = 1.0 - float(dones[t])
            delta: torch.Tensor = (
                rewards[t] + gamma * next_val * next_non_terminal - values[t]
            )
            advantages[t] = last_gae = float(delta) + gamma * gae_lambda * next_non_terminal * last_gae

        returns: torch.Tensor = advantages + values
        return advantages, returns

    def _unfreeze_actor(self) -> None:
        """解冻 actor 并重建优化器

        解冻后使用更小的学习率（base_lr * actor_lr_scale）微调 actor，
        同时保持 critic 和 log_std 使用正常学习率。
        """
        self._policy.unfreeze_actor()
        self._optimizer = Adam(
            self._policy.get_param_groups(
                actor_lr_scale=self._config.actor_lr_scale,
                base_lr=self._config.learning_rate,
            ),
            eps=1e-5,
        )
        self._actor_unfrozen = True
        logger.info(f"Actor 已解冻 (step={self._global_step})")

    def save(self, path: str) -> None:
        """保存 checkpoint

        保存策略网络、优化器状态和训练进度。

        Args:
            path: 保存路径
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self._policy.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "global_step": self._global_step,
                "num_updates": self._num_updates,
                "actor_unfrozen": self._actor_unfrozen,
                "config": {
                    "agent_type": self._config.agent_type,
                    "total_timesteps": self._config.total_timesteps,
                    "learning_rate": self._config.learning_rate,
                    "freeze_actor_steps": self._config.freeze_actor_steps,
                },
            },
            path,
        )
        logger.info(f"Checkpoint 已保存: {path}")

    def load(self, path: str) -> None:
        """加载 checkpoint

        恢复策略网络、优化器状态和训练进度。
        如果 checkpoint 中 actor 已解冻，则同步解冻当前策略。

        Args:
            path: checkpoint 文件路径
        """
        ckpt: dict[str, Any] = torch.load(
            path, map_location=self._device, weights_only=False
        )
        self._policy.load_state_dict(ckpt["policy_state_dict"])
        self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._global_step = ckpt["global_step"]
        self._num_updates = ckpt["num_updates"]
        if ckpt.get("actor_unfrozen") and not self._actor_unfrozen:
            self._unfreeze_actor()
        logger.info(f"Checkpoint 已加载: {path} (step={self._global_step})")
