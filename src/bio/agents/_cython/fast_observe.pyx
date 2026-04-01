# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""快速 observe 函数 - Cython 实现

提供 Agent observe() 方法的加速实现，用于构建神经网络输入向量。

两种 Agent 类型的输入维度：
- 高级散户（RetailPro）: 67 维
- 做市商（MarketMaker）: 132 维
"""

import numpy as np
cimport numpy as np
from libc.math cimport log10

# 类型定义
ctypedef np.float64_t DTYPE_t
ctypedef np.float32_t DTYPE32_t


cpdef void fast_observe_full(
    double[:] output_buffer,
    float[:] bid_data,
    float[:] ask_data,
    float[:] tick_history_prices,
    float[:] tick_history_volumes,
    double position_value_normalized,
    double position_avg_price_normalized,
    double balance_normalized,
    double equity_normalized,
    double pending_price_normalized,
    double pending_qty_normalized,
    double pending_side,
) noexcept nogil:
    """构建高级散户/庄家的神经网络输入向量（67 维）

    输入布局：
    - 0-9: 买盘5档（每档2个值：价格归一化 + 数量）
    - 10-19: 卖盘5档（每档2个值：价格归一化 + 数量）
    - 20-23: 持仓信息（4个值）
    - 24-26: 挂单信息（3个值）
    - 27-46: tick历史价格（最近20个）
    - 47-66: tick历史成交量（最近20个）

    Args:
        output_buffer: 预分配的输出缓冲区（67维）
        bid_data: 买盘数据（10维）
        ask_data: 卖盘数据（10维）
        tick_history_prices: tick历史价格（最近20个）
        tick_history_volumes: tick历史成交量（最近20个）
        position_value_normalized: 持仓价值归一化
        position_avg_price_normalized: 持仓均价归一化
        balance_normalized: 余额归一化
        equity_normalized: 净值归一化
        pending_price_normalized: 挂单价格归一化
        pending_qty_normalized: 挂单数量归一化
        pending_side: 挂单方向（1.0买/-1.0卖/0.0无）
    """
    cdef int i

    # 买盘5档（10个值）
    for i in range(10):
        output_buffer[i] = bid_data[i]

    # 卖盘5档（10个值）
    for i in range(10):
        output_buffer[10 + i] = ask_data[i]

    # 持仓信息（4个值）
    output_buffer[20] = position_value_normalized
    output_buffer[21] = position_avg_price_normalized
    output_buffer[22] = balance_normalized
    output_buffer[23] = equity_normalized

    # 挂单信息（3个值）
    output_buffer[24] = pending_price_normalized
    output_buffer[25] = pending_qty_normalized
    output_buffer[26] = pending_side

    # tick历史价格（最近20个）
    for i in range(20):
        output_buffer[27 + i] = tick_history_prices[i]

    # tick历史成交量（最近20个）
    for i in range(20):
        output_buffer[47 + i] = tick_history_volumes[i]


cpdef void fast_observe_market_maker(
    double[:] output_buffer,
    float[:] bid_data,
    float[:] ask_data,
    float[:] tick_history_prices,
    float[:] tick_history_volumes,
    double position_value_normalized,
    double position_avg_price_normalized,
    double balance_normalized,
    double equity_normalized,
    double[:] pending_order_inputs,
) noexcept nogil:
    """构建做市商的神经网络输入向量（132 维）

    输入布局：
    - 0-9: 买盘5档（每档2个值：价格归一化 + 数量）
    - 10-19: 卖盘5档（每档2个值：价格归一化 + 数量）
    - 20-23: 持仓信息（4个值）
    - 24-83: 挂单信息（60个值：10买单+10卖单，每单3个值）
    - 84-103: tick历史价格（最近20个）
    - 104-123: tick历史成交量（最近20个）
    - 124-131: AS模型特征（8个值，由Python调用方填充）

    Args:
        output_buffer: 预分配的输出缓冲区（132维）
        bid_data: 买盘数据（10维）
        ask_data: 卖盘数据（10维）
        tick_history_prices: tick历史价格（最近20个）
        tick_history_volumes: tick历史成交量（最近20个）
        position_value_normalized: 持仓价值归一化
        position_avg_price_normalized: 持仓均价归一化
        balance_normalized: 余额归一化
        equity_normalized: 净值归一化
        pending_order_inputs: 做市商挂单信息（60维）
    """
    cdef int i

    # 买盘5档（10个值）
    for i in range(10):
        output_buffer[i] = bid_data[i]

    # 卖盘5档（10个值）
    for i in range(10):
        output_buffer[10 + i] = ask_data[i]

    # 持仓信息（4个值）
    output_buffer[20] = position_value_normalized
    output_buffer[21] = position_avg_price_normalized
    output_buffer[22] = balance_normalized
    output_buffer[23] = equity_normalized

    # 挂单信息（60个值）
    for i in range(60):
        output_buffer[24 + i] = pending_order_inputs[i]

    # tick历史价格（最近20个）
    for i in range(20):
        output_buffer[84 + i] = tick_history_prices[i]

    # tick历史成交量（最近20个）
    for i in range(20):
        output_buffer[104 + i] = tick_history_volumes[i]


cpdef tuple get_position_inputs(
    double equity,
    double leverage,
    int position_quantity,
    double position_avg_price,
    double balance,
    double initial_balance,
    double mid_price,
):
    """计算持仓信息输入（4个值）

    Args:
        equity: 账户净值
        leverage: 杠杆倍数
        position_quantity: 持仓数量（正=多头，负=空头）
        position_avg_price: 持仓均价
        balance: 账户余额
        initial_balance: 初始余额
        mid_price: 中间价

    Returns:
        (position_value_normalized, position_avg_price_normalized,
         balance_normalized, equity_normalized)
    """
    cdef double position_value_normalized
    cdef double position_avg_price_normalized
    cdef double balance_normalized
    cdef double equity_normalized
    cdef double position_value

    # 持仓价值归一化
    position_value = abs(position_quantity) * mid_price
    if equity > 0 and leverage > 0:
        position_value_normalized = position_value / (equity * leverage)
    else:
        position_value_normalized = 0.0

    # 持仓均价归一化
    if position_quantity == 0:
        position_avg_price_normalized = 0.0
    elif mid_price > 0:
        position_avg_price_normalized = (position_avg_price - mid_price) / mid_price
    else:
        position_avg_price_normalized = 0.0

    # 余额归一化
    if initial_balance > 0:
        balance_normalized = balance / initial_balance
    else:
        balance_normalized = 0.0

    # 净值归一化
    if initial_balance > 0:
        equity_normalized = equity / initial_balance
    else:
        equity_normalized = 0.0

    return (
        position_value_normalized,
        position_avg_price_normalized,
        balance_normalized,
        equity_normalized,
    )


cpdef tuple get_pending_order_inputs(
    double order_price,
    int order_quantity,
    int order_side,
    double mid_price,
):
    """计算挂单信息输入（3个值）

    Args:
        order_price: 订单价格（0表示无挂单）
        order_quantity: 订单数量
        order_side: 订单方向（1=买，2=卖，0=无）
        mid_price: 中间价

    Returns:
        (pending_price_normalized, pending_qty_normalized, pending_side)
    """
    cdef double pending_price_normalized
    cdef double pending_qty_normalized
    cdef double pending_side

    if order_price == 0 or order_quantity == 0:
        return (0.0, 0.0, 0.0)

    # 价格归一化
    if mid_price > 0:
        pending_price_normalized = (order_price - mid_price) / mid_price
    else:
        pending_price_normalized = 0.0

    # 数量使用对数归一化：log10(qty + 1) / 10
    pending_qty_normalized = log10(<double>order_quantity + 1.0) / 10.0

    # 方向：买单1.0，卖单-1.0
    if order_side == 1:  # BUY
        pending_side = 1.0
    elif order_side == 2:  # SELL
        pending_side = -1.0
    else:
        pending_side = 0.0

    return (pending_price_normalized, pending_qty_normalized, pending_side)
