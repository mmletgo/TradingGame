"""
成交记录模块

本模块定义了成交记录(Trade)数据类，用于记录订单撮合成功后的交易信息。
"""

import time


class Trade:
    """
    成交记录

    记录订单撮合成功后的交易信息，包括价格、数量、买卖双方和手续费。

    Attributes:
        trade_id: 成交ID
        price: 成交价格
        quantity: 成交数量
        buyer_id: 买方Agent ID
        seller_id: 卖方Agent ID
        buyer_fee: 买方手续费
        seller_fee: 卖方手续费
        is_buyer_taker: 买方是否为taker（主动吃单方）
        timestamp: 成交时间戳
    """

    __slots__ = ('trade_id', 'price', 'quantity', 'buyer_id', 'seller_id',
                 'buyer_fee', 'seller_fee', 'is_buyer_taker', 'timestamp')

    def __init__(
        self,
        trade_id: int,
        price: float,
        quantity: float,
        buyer_id: int,
        seller_id: int,
        buyer_fee: float,
        seller_fee: float,
        is_buyer_taker: bool,
        timestamp: float | None = None,
    ) -> None:
        """
        创建成交记录

        Args:
            trade_id: 成交ID
            price: 成交价格
            quantity: 成交数量
            buyer_id: 买方Agent ID
            seller_id: 卖方Agent ID
            buyer_fee: 买方手续费
            seller_fee: 卖方手续费
            is_buyer_taker: 买方是否为taker（主动吃单方）
            timestamp: 成交时间戳（可选，默认使用当前时间）
        """
        self.trade_id = trade_id
        self.price = price
        self.quantity = quantity
        self.buyer_id = buyer_id
        self.seller_id = seller_id
        self.buyer_fee = buyer_fee
        self.seller_fee = seller_fee
        self.is_buyer_taker = is_buyer_taker
        self.timestamp = timestamp if timestamp is not None else time.time()
