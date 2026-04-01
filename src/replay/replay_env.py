"""Gymnasium 环境封装

将 ReplayEngine 封装为标准 Gymnasium 环境，供 RL 训练器使用。
每个 step 对应一个订单簿快照。Agent 观察市场状态后决策，
引擎模拟成交并返回 reward。

Observation: NormalizedMarketState 展平为 float32 数组
- RETAIL_PRO: shape (67,)
- MARKET_MAKER: shape (132,)

Action: 连续动作空间 [-1, 1]
- RETAIL_PRO: shape (3,) -- 动作选择 + 价格偏移 + 数量比例
- MARKET_MAKER: shape (43,) -- 双边挂单参数
"""
from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from src.replay.config import ReplayConfig
from src.replay.data_loader import DataLoader
from src.replay.replay_engine import ReplayEngine


class ReplayEnv(gymnasium.Env):
    """基于真实市场数据回放的 Gymnasium 环境

    数据流:
        __init__() 加载数据 -> reset() 随机起始点 -> step() 逐步回放

    Observation 向量结构与原始 Agent.observe() 完全一致:
    - 散户 67 维: 降维后的观测向量
    - 做市商 132 维: 降维后的观测向量
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, config: ReplayConfig) -> None:
        """初始化环境

        加载市场数据，创建回放引擎，定义观测/动作空间。

        Args:
            config: 回放环境配置
        """
        super().__init__()
        self._config: ReplayConfig = config
        self._engine: ReplayEngine = ReplayEngine(config)

        # 加载数据
        loader: DataLoader = DataLoader(config)
        ob_snapshots, trades = loader.load()
        self._engine.load_data(ob_snapshots, trades)
        self._num_snapshots: int = len(ob_snapshots)

        # 根据 agent 类型确定维度
        obs_dim: int
        act_dim: int
        if config.agent_type == "RETAIL_PRO":
            obs_dim = 67
            act_dim = 3
        else:
            obs_dim = 132
            act_dim = 43

        self.observation_space: spaces.Box = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space: spaces.Box = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        self._obs_dim: int = obs_dim

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境

        随机选择一个起始点（留出 episode_length + warmup 的余量），
        重置引擎状态并返回初始观测。

        Args:
            seed: 随机种子
            options: 额外选项（未使用）

        Returns:
            (observation, info) 元组
        """
        super().reset(seed=seed)

        # 随机起始点（留出 episode_length + warmup 的余量）
        warmup: int = 200
        max_start: int = max(0, self._num_snapshots - self._config.episode_length - warmup)
        start_idx: int = int(self.np_random.integers(0, max(1, max_start)))

        market_state = self._engine.reset(start_idx=start_idx)
        obs: np.ndarray = self._state_to_obs(market_state)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行一步

        将动作传递给引擎，返回 Gymnasium 标准 5 元组。

        Args:
            action: 神经网络输出的动作向量

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        result = self._engine.step(action)
        obs: np.ndarray = self._state_to_obs(result.market_state)
        return obs, result.reward, result.done, False, result.info

    def _state_to_obs(self, state: Any) -> np.ndarray:
        """将 NormalizedMarketState 转为 observation 数组

        按照与原始 Agent.observe() 完全一致的向量布局拼接:
        - bid_data + ask_data + trade_prices + trade_quantities
        - position_info (从 engine 的 account 获取)
        - pending_info (回放环境中简化)
        - tick_history_prices + tick_history_volumes + tick_history_amounts
        - as_features (做市商额外 8 值)

        Args:
            state: NormalizedMarketState 对象

        Returns:
            shape (obs_dim,) 的 float32 数组
        """
        parts: list[np.ndarray] = [
            state.bid_data.astype(np.float32),
            state.ask_data.astype(np.float32),
            state.trade_prices.astype(np.float32),
            state.trade_quantities.astype(np.float32),
        ]

        # 持仓信息（4 值）：从 engine 的 account 状态获取
        account = self._engine._account
        assert account is not None, "engine 必须已初始化 account"
        mid_price: float = state.mid_price
        equity: float = account.get_equity(mid_price)

        # [0] 持仓价值归一化: position_value / (equity * leverage)
        pos_value: float = abs(account.position.quantity) * mid_price
        pos_norm: float = (
            pos_value / (equity * account.leverage)
            if equity > 0 and account.leverage > 0
            else 0.0
        )

        # [1] 持仓均价归一化: (avg_price - mid_price) / mid_price
        avg_price_norm: float = (
            (account.position.avg_price - mid_price) / mid_price
            if account.position.quantity != 0 and mid_price > 0
            else 0.0
        )

        # [2] 余额归一化: balance / initial_balance
        balance_norm: float = (
            account.balance / account.initial_balance
            if account.initial_balance > 0
            else 0.0
        )

        # [3] 净值归一化: equity / initial_balance
        equity_norm: float = (
            equity / account.initial_balance
            if account.initial_balance > 0
            else 0.0
        )

        position_info: np.ndarray = np.array(
            [pos_norm, avg_price_norm, balance_norm, equity_norm],
            dtype=np.float32,
        )
        parts.append(position_info)

        # 挂单信息
        if self._config.agent_type == "RETAIL_PRO":
            # 散户挂单信息（3 值）: 价格归一化 + 数量归一化 + 方向
            # 回放环境中挂单信息从 engine 的 pending_orders 获取
            pending_info: np.ndarray = self._build_retail_pending_info(mid_price)
            parts.append(pending_info)
        else:
            # 做市商挂单信息（60 值）: 10 买单 x 3 + 10 卖单 x 3
            pending_info = self._build_mm_pending_info(mid_price)
            parts.append(pending_info)

        # tick 历史
        parts.extend([
            state.tick_history_prices.astype(np.float32),
            state.tick_history_volumes.astype(np.float32),
            state.tick_history_amounts.astype(np.float32),
        ])

        # 做市商额外 AS 特征（8 值）
        if self._config.agent_type == "MARKET_MAKER":
            # 简化处理：从 market_state 提取可用的 AS 参数
            as_features: np.ndarray = np.array([
                0.0,                # reservation_offset (简化为 0)
                0.0,                # optimal_half_spread / mid_price
                state.as_sigma,     # sigma (已实现波动率)
                state.as_tau,       # tau (剩余时间比例)
                0.0,                # kappa (归一化后)
                0.0,                # inventory_risk
                0.0,                # gamma / max_gamma
                0.0,                # spread / sigma
            ], dtype=np.float32)
            parts.append(as_features)

        obs: np.ndarray = np.concatenate(parts)
        assert obs.shape == (self._obs_dim,), (
            f"obs shape {obs.shape} != ({self._obs_dim},)"
        )
        return obs

    def _build_retail_pending_info(self, mid_price: float) -> np.ndarray:
        """构建散户挂单信息（3 值）

        与 Agent._get_pending_order_inputs() 一致:
        - [0]: 价格归一化 (order_price - mid_price) / mid_price
        - [1]: 数量归一化 log10(quantity + 1) / 10
        - [2]: 方向 (买单 1.0, 卖单 -1.0, 无挂单 0.0)

        Args:
            mid_price: 当前中间价

        Returns:
            shape (3,) 的 float32 数组
        """
        from math import log10

        result: np.ndarray = np.zeros(3, dtype=np.float32)
        pending = self._engine._pending_orders
        if not pending or mid_price <= 0:
            return result

        # 散户只有一个挂单，取第一个
        order = pending[0]
        result[0] = np.float32((order.price - mid_price) / mid_price)
        result[1] = np.float32(log10(order.quantity + 1.0) / 10.0)
        result[2] = np.float32(1.0 if order.side == 1 else -1.0)
        return result

    def _build_mm_pending_info(self, mid_price: float) -> np.ndarray:
        """构建做市商挂单信息（60 值）

        与 MarketMakerAgent._get_pending_order_inputs() 一致:
        - 10 买单 x 3 (价格归一化, 数量归一化, 有效标志)
        - 10 卖单 x 3 (价格归一化, 数量归一化, 有效标志)

        Args:
            mid_price: 当前中间价

        Returns:
            shape (60,) 的 float32 数组
        """
        from math import log10

        result: np.ndarray = np.zeros(60, dtype=np.float32)
        if mid_price <= 0:
            return result

        pending = self._engine._pending_orders
        bid_idx: int = 0
        ask_idx: int = 0

        for order in pending:
            if order.side == 1 and bid_idx < 10:
                # 买单
                offset: int = bid_idx * 3
                result[offset] = np.float32((order.price - mid_price) / mid_price)
                result[offset + 1] = np.float32(log10(order.quantity + 1.0) / 10.0)
                result[offset + 2] = np.float32(1.0)  # 有效标志
                bid_idx += 1
            elif order.side == -1 and ask_idx < 10:
                # 卖单
                offset = 30 + ask_idx * 3
                result[offset] = np.float32((order.price - mid_price) / mid_price)
                result[offset + 1] = np.float32(log10(order.quantity + 1.0) / 10.0)
                result[offset + 2] = np.float32(1.0)  # 有效标志
                ask_idx += 1

        return result
