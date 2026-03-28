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
        bid_data: 买盘数据，shape (depth*2,)，depth档 × 2（价格归一化 + 数量）
        ask_data: 卖盘数据，shape (depth*2,)，depth档 × 2（价格归一化 + 数量）
        trade_prices: 成交价格归一化，shape (100,)
        trade_quantities: 成交数量（带方向），shape (100,)，正数表示taker是买方，负数表示taker是卖方
        tick_history_prices: Tick 历史价格（以第一个价格为基准归一化），shape (100,)，最新数据在末尾
        tick_history_volumes: Tick 历史成交量（带方向，正=taker买入为主），shape (100,)，最新数据在末尾
        tick_history_amounts: Tick 历史成交额（带方向，正=taker买入为主），shape (100,)，最新数据在末尾
    """
    mid_price: float
    tick_size: float
    bid_data: NDArray[np.float32]      # shape: (depth*2,)
    ask_data: NDArray[np.float32]      # shape: (depth*2,)
    trade_prices: NDArray[np.float32]  # shape: (100,)
    trade_quantities: NDArray[np.float32]  # shape: (100,)
    tick_history_prices: NDArray[np.float32]   # shape: (100,) - tick 历史价格（以第一个价格为基准归一化）
    tick_history_volumes: NDArray[np.float32]  # shape: (100,) - tick 历史成交量（带方向）
    tick_history_amounts: NDArray[np.float32]  # shape: (100,) - tick 历史成交额（带方向）
    current_tick: int = 0                      # 当前 tick 编号
    episode_length: int = 1000                 # Episode 总长度
    raw_tick_prices: NDArray[np.float64] | None = None  # 原始 tick 价格（未归一化）

    # AS 模型预计算参数（每 tick 计算一次，所有做市商共享）
    as_sigma: float = 0.0                      # 已实现波动率
    as_tau: float = 1.0                        # 剩余时间比例
    as_kappa: float = 1.5                      # 订单到达强度
    as_gamma: float = 0.1                      # 基础风险厌恶系数
    as_gamma_adj_min: float = 0.1              # NN gamma 调整下限
    as_gamma_adj_max: float = 10.0             # NN gamma 调整上限
    as_spread_adj_min: float = 0.5             # NN spread 调整下限
    as_spread_adj_max: float = 2.0             # NN spread 调整上限
    as_max_reservation_offset: float = 0.05    # reservation price 最大偏移比例
