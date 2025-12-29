"""
订单模块

定义订单方向、订单类型和订单数据类
"""

from enum import IntEnum
import time


class OrderSide(IntEnum):
    """订单方向枚举"""

    BUY = 1
    SELL = -1


class OrderType(IntEnum):
    """订单类型枚举"""

    LIMIT = 1
    MARKET = 2


class Order:
    """订单数据类"""

    def __init__(
        self,
        order_id: int,
        agent_id: int,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        quantity: float,
    ) -> None:
        """
        初始化订单

        Args:
            order_id: 订单ID
            agent_id: Agent ID
            side: 买卖方向
            order_type: 订单类型
            price: 价格
            quantity: 数量
        """
        self.order_id: int = order_id
        self.agent_id: int = agent_id
        self.side: OrderSide = side
        self.order_type: OrderType = order_type
        self.price: float = price
        self.quantity: float = quantity
        self.filled_quantity: float = 0.0
        self.timestamp: float = time.time()
