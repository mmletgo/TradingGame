"""
快速撮合引擎类型存根文件
"""

from typing import Any


class FastTrade:
    """快速成交记录（Cython）"""
    trade_id: int
    price: float
    quantity: int
    buyer_id: int
    seller_id: int
    buyer_fee: float
    seller_fee: float
    is_buyer_taker: bool
    timestamp: float

    def __init__(
        self,
        trade_id: int,
        price: float,
        quantity: int,
        buyer_id: int,
        seller_id: int,
        buyer_fee: float,
        seller_fee: float,
        is_buyer_taker: bool,
    ) -> None: ...


def fast_match_orders(
    orderbook: Any,
    order: Any,
    fee_rates: dict[int, tuple[float, float]],
    next_trade_id: int,
    is_limit_order: bool,
) -> tuple[list[FastTrade], int, int]:
    """
    快速撮合核心逻辑

    Args:
        orderbook: OrderBook 实例
        order: 待撮合订单
        fee_rates: agent_id -> (maker_rate, taker_rate)
        next_trade_id: 下一个成交 ID
        is_limit_order: 是否为限价单

    Returns:
        (trades, remaining, next_trade_id): 成交列表、剩余数量、更新后的 trade_id
    """
    ...
