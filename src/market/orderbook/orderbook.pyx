"""
订单簿模块（Cython 加速）

定义价格档位和订单簿
"""

# cython: language_level=3
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.market.orderbook.order import Order


cdef class PriceLevel:
    """价格档位（Cython）

    使用 OrderedDict 存储订单，保证 O(1) 的删除操作，
    同时保持 FIFO 顺序（时间优先原则）。
    """

    cdef public double price
    cdef public object orders  # OrderedDict: order_id -> Order
    cdef public double total_quantity

    def __init__(self, price: float) -> None:
        """
        创建价格档位

        Args:
            price: 档位价格
        """
        self.price = price
        self.orders = OrderedDict()  # order_id -> Order，保持 FIFO 顺序
        self.total_quantity = 0.0  # 总数量

    def add_order(self, order: "Order") -> None:
        """
        向价格档位添加订单

        Args:
            order: 订单对象
        """
        self.orders[order.order_id] = order
        self.total_quantity += order.quantity

    def remove_order(self, order_id: int) -> "Order | None":
        """
        从价格档位移除订单

        Args:
            order_id: 订单 ID

        Returns:
            被移除的订单对象，如果订单不存在则返回 None
        """
        if order_id in self.orders:
            order = self.orders.pop(order_id)  # O(1) 删除
            self.total_quantity -= order.quantity
            return order
        return None

    def get_volume(self) -> float:
        """
        获取该价格档位的总挂单量

        Returns:
            总挂单量
        """
        return self.total_quantity
