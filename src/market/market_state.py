"""归一化市场状态数据类

用于预计算和缓存每个 tick 的公共市场数据，避免每个 Agent 重复计算。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class NormalizedMarketState:
    """预计算的归一化市场数据

    Attributes:
        mid_price: 中间价（用于归一化计算的参考价格）
        tick_size: 最小价格变动单位
        bid_data: 买盘数据，shape (200,)，100档 × 2（价格归一化 + 数量）
        ask_data: 卖盘数据，shape (200,)，100档 × 2（价格归一化 + 数量）
        trade_prices: 成交价格归一化，shape (100,)
        trade_quantities: 成交数量（带方向），shape (100,)，正数表示taker是买方，负数表示taker是卖方
    """
    mid_price: float
    tick_size: float
    bid_data: NDArray[np.float32]      # shape: (200,)
    ask_data: NDArray[np.float32]      # shape: (200,)
    trade_prices: NDArray[np.float32]  # shape: (100,)
    trade_quantities: NDArray[np.float32]  # shape: (100,)
