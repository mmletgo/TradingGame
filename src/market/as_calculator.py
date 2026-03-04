"""Avellaneda-Stoikov 最优做市模型计算器

基于 AS 模型计算 reservation price（保留价格）和 optimal spread（最优价差），
用于做市商的报价中心偏移和最小价差控制。

核心公式：
- reservation_price = mid × (1 - q_norm × γ × σ² × τ)
- optimal_spread = γ × σ² × τ + (2/γ) × ln(1 + γ/κ)
"""

from dataclasses import dataclass
from math import log

import numpy as np
from numpy.typing import NDArray

from src.config.config import ASConfig


@dataclass
class ASQuoteParams:
    """AS 模型计算结果

    Attributes:
        reservation_price: 保留价格 r = s × (1 - q_norm × γ × σ² × τ)
        reservation_offset: 归一化偏移 (r - s) / s
        optimal_spread: 最优价差 δ* = γσ²τ + (2/γ)ln(1 + γ/κ)
        optimal_half_spread: δ*/2
        sigma: 已实现波动率
        tau: 剩余时间比例
        kappa: 订单到达强度
        inventory_risk: q_norm × γ × σ² × τ
    """

    reservation_price: float
    reservation_offset: float
    optimal_spread: float
    optimal_half_spread: float
    sigma: float
    tau: float
    kappa: float
    inventory_risk: float


class ASCalculator:
    """Avellaneda-Stoikov 最优做市模型计算器

    Args:
        config: AS 模型配置
    """

    def __init__(self, config: ASConfig) -> None:
        self._config: ASConfig = config

    @staticmethod
    def compute_volatility(
        raw_tick_prices: NDArray[np.float64],
        vol_window: int,
        min_sigma: float,
        max_sigma: float,
    ) -> float:
        """计算已实现波动率（log-return 标准差）

        Args:
            raw_tick_prices: 原始 tick 价格序列
            vol_window: 回看窗口大小
            min_sigma: 波动率下限
            max_sigma: 波动率上限

        Returns:
            波动率 σ，clamped 到 [min_sigma, max_sigma]
        """
        if raw_tick_prices is None or len(raw_tick_prices) < 2:
            return min_sigma

        # 取最近 vol_window 个价格
        prices = raw_tick_prices[-vol_window:]
        if len(prices) < 2:
            return min_sigma

        # 过滤掉零值和负值
        valid = prices[prices > 0]
        if len(valid) < 2:
            return min_sigma

        # 计算 log returns
        log_returns = np.diff(np.log(valid))
        if len(log_returns) == 0:
            return min_sigma

        sigma = float(np.std(log_returns))
        return max(min_sigma, min(max_sigma, sigma))

    @staticmethod
    def compute_kappa(
        tick_history_volumes: NDArray[np.float64] | None,
        kappa_base: float,
    ) -> float:
        """计算订单到达强度（基于历史成交量的代理值）

        Args:
            tick_history_volumes: tick 历史成交量序列
            kappa_base: 基础到达率

        Returns:
            订单到达率 κ
        """
        if tick_history_volumes is None or len(tick_history_volumes) == 0:
            return kappa_base

        avg_volume = float(np.mean(np.abs(tick_history_volumes)))
        # 使用 log 变换平滑：κ = kappa_base × (1 + log(1 + avg_vol) / 10)
        return kappa_base * (1.0 + log(1.0 + avg_volume) / 10.0)

    def compute(
        self,
        mid_price: float,
        inventory_qty: int,
        equity: float,
        leverage: float,
        sigma: float,
        tau: float,
        kappa: float,
    ) -> ASQuoteParams:
        """计算 AS 模型的 reservation price 和 optimal spread

        Args:
            mid_price: 当前中间价
            inventory_qty: 持仓数量（带符号）
            equity: 账户净值
            leverage: 杠杆倍数
            sigma: 波动率
            tau: 剩余时间比例 (0, 1]
            kappa: 订单到达率

        Returns:
            ASQuoteParams 计算结果
        """
        gamma = self._config.gamma

        # 防止 tau 为零
        tau = max(0.001, tau)

        # 归一化库存：q_norm = (pos_qty × mid) / (equity × leverage)
        max_pos_value = equity * leverage
        if max_pos_value > 0 and mid_price > 0:
            q_norm = (inventory_qty * mid_price) / max_pos_value
            q_norm = max(-1.0, min(1.0, q_norm))
        else:
            q_norm = 0.0

        sigma_sq = sigma * sigma

        # inventory_risk = q_norm × γ × σ² × τ
        inventory_risk = q_norm * gamma * sigma_sq * tau

        # reservation_price = mid × (1 - inventory_risk)
        reservation_offset = -inventory_risk
        reservation_offset = max(
            -self._config.max_reservation_offset,
            min(self._config.max_reservation_offset, reservation_offset),
        )
        reservation_price = mid_price * (1.0 + reservation_offset)

        # optimal_spread = γσ²τ + (2/γ) × ln(1 + γ/κ)
        kappa_safe = max(1e-6, kappa)
        optimal_spread = gamma * sigma_sq * tau + (2.0 / gamma) * log(
            1.0 + gamma / kappa_safe
        )
        optimal_half_spread = optimal_spread * 0.5

        return ASQuoteParams(
            reservation_price=reservation_price,
            reservation_offset=reservation_offset,
            optimal_spread=optimal_spread,
            optimal_half_spread=optimal_half_spread,
            sigma=sigma,
            tau=tau,
            kappa=kappa,
            inventory_risk=inventory_risk,
        )
