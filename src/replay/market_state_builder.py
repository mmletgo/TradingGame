"""将真实市场数据转换为 NormalizedMarketState

使用与 TradingGame_origin 完全相同的归一化公式构建
NormalizedMarketState 对象，供回放环境中的 agent 使用。

归一化公式:
- 订单簿价格: (price - mid_price) / mid_price
- 订单簿数量: log10(quantity + 1) / 10
- 成交价格: (price - mid_price) / mid_price
- 成交数量(带方向): sign(qty) * log10(|qty| + 1) / 10
- tick 历史价格: (price - base_price) / base_price
- tick 历史成交量: sign(vol) * log10(|vol| + 1) / 10
- tick 历史成交额: sign(amt) * log10(|amt| + 1) / 12
"""
from __future__ import annotations

from collections import deque
from math import copysign, log10

import numpy as np
from numpy.typing import NDArray

from src.market.market_state import NormalizedMarketState
from src.replay.config import ReplayConfig
from src.replay.data_loader import MarketTrade, OrderbookSnapshot


class MarketStateBuilder:
    """从真实订单簿快照和成交数据构建 NormalizedMarketState

    内部维护历史缓冲区，每次调用 build() 返回最新的市场状态。
    典型调用顺序: add_trades() -> update_tick() -> build()
    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config: ReplayConfig = config
        self._trade_history_len: int = config.trade_history_len  # 100
        self._tick_history_len: int = config.tick_history_len    # 100
        self._depth: int = config.ob_depth                       # 5
        self.reset()

    def reset(self) -> None:
        """重置所有内部缓冲区"""
        # 成交历史 (归一化后的值)
        self._trade_prices: deque[float] = deque(maxlen=self._trade_history_len)
        self._trade_quantities: deque[float] = deque(maxlen=self._trade_history_len)

        # tick 级别历史 (原始 mid_price，用于后续归一化)
        self._tick_prices: deque[float] = deque(maxlen=self._tick_history_len)
        # tick 级别历史 (带方向的成交量和成交额，原始值)
        self._tick_volumes: deque[float] = deque(maxlen=self._tick_history_len)
        self._tick_amounts: deque[float] = deque(maxlen=self._tick_history_len)

        # AS 模型使用的原始价格序列
        self._raw_tick_prices: deque[float] = deque(maxlen=1000)

        # tick 历史价格归一化基准 (第一个 tick 的 mid_price)
        self._base_price: float = 0.0

        # 当前 tick 编号
        self._current_tick: int = 0

        # episode 长度
        self._episode_length: int = self._config.episode_length

    def add_trades(self, trades: list[MarketTrade], mid_price: float) -> None:
        """处理成交事件，更新成交历史缓冲区

        必须在 update_tick() 之前调用。每笔成交归一化后加入缓冲区。

        Args:
            trades: 两个快照之间发生的成交列表
            mid_price: 当前中间价，用于归一化
        """
        if mid_price <= 0:
            return

        for trade in trades:
            # 价格归一化: (price - mid_price) / mid_price
            norm_price: float = (trade.price - mid_price) / mid_price
            self._trade_prices.append(norm_price)

            # 数量归一化(带方向): sign(qty) * log10(|qty| + 1) / 10
            # taker 方向: buy -> 正, sell -> 负
            qty: float = trade.amount
            if trade.side == "sell":
                qty = -qty
            norm_qty: float = _signed_log_norm(qty, 10.0)
            self._trade_quantities.append(norm_qty)

    def update_tick(
        self, ob: OrderbookSnapshot, tick_volume: float, tick_amount: float
    ) -> None:
        """更新 tick 级别的历史数据

        记录该 tick 的 mid_price、成交量和成交额到历史缓冲区。

        Args:
            ob: 当前订单簿快照
            tick_volume: 该 tick 的带方向成交量（正=taker买入为主）
            tick_amount: 该 tick 的带方向成交额（正=taker买入为主）
        """
        mid_price: float = ob.mid_price

        # 设置 base_price (第一个 tick 的价格)
        if self._base_price == 0.0 and mid_price > 0:
            self._base_price = mid_price

        # 记录 mid_price 到 tick 历史
        self._tick_prices.append(mid_price)

        # 记录原始 mid_price 到 AS 模型用的历史
        self._raw_tick_prices.append(mid_price)

        # 记录成交量/额到 tick 历史（原始值，build 时归一化）
        self._tick_volumes.append(tick_volume)
        self._tick_amounts.append(tick_amount)

    def build(self, ob: OrderbookSnapshot) -> NormalizedMarketState:
        """从当前状态构建 NormalizedMarketState

        Args:
            ob: 当前订单簿快照

        Returns:
            归一化后的市场状态对象
        """
        mid_price: float = ob.mid_price

        # 1. 构建 bid_data / ask_data: shape (depth*2,)
        bid_data: NDArray[np.float32] = self._normalize_orderbook_side(
            ob.bid_prices, ob.bid_amounts, mid_price
        )
        ask_data: NDArray[np.float32] = self._normalize_orderbook_side(
            ob.ask_prices, ob.ask_amounts, mid_price
        )

        # 2. 构建 trade_prices / trade_quantities: shape (trade_history_len,)
        trade_prices: NDArray[np.float32] = _pad_deque(
            self._trade_prices, self._trade_history_len
        )
        trade_quantities: NDArray[np.float32] = _pad_deque(
            self._trade_quantities, self._trade_history_len
        )

        # 3. 构建 tick 历史: shape (tick_history_len,) each
        tick_history_prices: NDArray[np.float32] = self._build_tick_history_prices()
        tick_history_volumes: NDArray[np.float32] = self._build_tick_history_values(
            self._tick_volumes, scale=10.0
        )
        tick_history_amounts: NDArray[np.float32] = self._build_tick_history_values(
            self._tick_amounts, scale=12.0
        )

        # 4. 构建 raw_tick_prices (AS 模型使用)
        raw_prices: NDArray[np.float64]
        if len(self._raw_tick_prices) > 0:
            raw_prices = np.array(list(self._raw_tick_prices), dtype=np.float64)
        else:
            raw_prices = np.empty(0, dtype=np.float64)

        self._current_tick += 1

        return NormalizedMarketState(
            mid_price=mid_price,
            tick_size=self._config.tick_size,
            bid_data=bid_data,
            ask_data=ask_data,
            trade_prices=trade_prices,
            trade_quantities=trade_quantities,
            tick_history_prices=tick_history_prices,
            tick_history_volumes=tick_history_volumes,
            tick_history_amounts=tick_history_amounts,
            current_tick=self._current_tick,
            episode_length=self._episode_length,
            raw_tick_prices=raw_prices if len(raw_prices) > 0 else None,
        )

    def _normalize_orderbook_side(
        self,
        prices: NDArray[np.float64],
        amounts: NDArray[np.float64],
        mid_price: float,
    ) -> NDArray[np.float32]:
        """归一化一侧订单簿为 shape (depth*2,) 的 float32 数组

        交错排列: [price_norm_0, qty_norm_0, price_norm_1, qty_norm_1, ...]
        - 价格归一化: (price - mid_price) / mid_price
        - 数量归一化: log10(quantity + 1) / 10

        Args:
            prices: 该侧价格数组, shape (depth,)
            amounts: 该侧数量数组, shape (depth,)
            mid_price: 中间价

        Returns:
            交错排列的归一化数组, shape (depth*2,)
        """
        depth: int = self._depth
        result: NDArray[np.float32] = np.zeros(depth * 2, dtype=np.float32)

        n: int = min(len(prices), depth)
        for i in range(n):
            price: float = float(prices[i])
            qty: float = float(amounts[i])

            # 价格归一化
            if mid_price > 0 and price > 0:
                result[i * 2] = np.float32((price - mid_price) / mid_price)
            else:
                result[i * 2] = np.float32(0.0)

            # 数量归一化: log10(quantity + 1) / 10
            result[i * 2 + 1] = np.float32(log10(qty + 1.0) / 10.0) if qty > 0 else np.float32(0.0)

        return result

    def _build_tick_history_prices(self) -> NDArray[np.float32]:
        """构建 tick 历史价格归一化数组

        公式: (price - base_price) / base_price
        base_price 为第一个 tick 的 mid_price。

        Returns:
            shape (tick_history_len,) 的 float32 数组，不足前面补零
        """
        length: int = self._tick_history_len
        result: NDArray[np.float32] = np.zeros(length, dtype=np.float32)

        n: int = len(self._tick_prices)
        if n == 0 or self._base_price <= 0:
            return result

        # 数据右对齐 (最新在末尾)
        start: int = length - n if n < length else 0
        data_start: int = 0 if n <= length else n - length
        for i in range(min(n, length)):
            price: float = self._tick_prices[data_start + i]
            result[start + i] = np.float32(
                (price - self._base_price) / self._base_price
            )

        return result

    def _build_tick_history_values(
        self, values_deque: deque[float], scale: float
    ) -> NDArray[np.float32]:
        """构建 tick 历史成交量/额归一化数组

        公式: sign(v) * log10(|v| + 1) / scale

        Args:
            values_deque: 原始值的 deque
            scale: 归一化分母 (成交量用 10, 成交额用 12)

        Returns:
            shape (tick_history_len,) 的 float32 数组，不足前面补零
        """
        length: int = self._tick_history_len
        result: NDArray[np.float32] = np.zeros(length, dtype=np.float32)

        n: int = len(values_deque)
        if n == 0:
            return result

        start: int = length - n if n < length else 0
        data_start: int = 0 if n <= length else n - length
        for i in range(min(n, length)):
            v: float = values_deque[data_start + i]
            result[start + i] = np.float32(_signed_log_norm(v, scale))

        return result


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------

def _signed_log_norm(value: float, scale: float) -> float:
    """带方向的对数归一化: sign(v) * log10(|v| + 1) / scale

    Args:
        value: 原始值（可为负）
        scale: 归一化分母

    Returns:
        归一化后的值
    """
    if value == 0.0:
        return 0.0
    abs_val: float = abs(value)
    sign: float = 1.0 if value > 0 else -1.0
    return sign * log10(abs_val + 1.0) / scale


def _pad_deque(d: deque[float], target_len: int) -> NDArray[np.float32]:
    """将 deque 转为固定长度的 float32 数组，不足部分在前面补零

    Args:
        d: 源数据 deque
        target_len: 目标长度

    Returns:
        shape (target_len,) 的 float32 数组
    """
    result: NDArray[np.float32] = np.zeros(target_len, dtype=np.float32)
    n: int = len(d)
    if n == 0:
        return result

    # 数据右对齐 (最新在末尾)
    start: int = target_len - n if n < target_len else 0
    data_start: int = 0 if n <= target_len else n - target_len
    for i in range(min(n, target_len)):
        result[start + i] = np.float32(d[data_start + i])

    return result
