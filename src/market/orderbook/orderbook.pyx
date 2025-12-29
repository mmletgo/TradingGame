"""
订单簿模块（Cython 加速）

定义价格档位和订单簿
"""

# cython: language_level=3
from collections import OrderedDict
from typing import TYPE_CHECKING

from src.market.orderbook.order import OrderSide

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


cdef class OrderBook:
    """订单簿（Cython）

    维护买卖盘价格档位，支持 O(1) 订单查找。
    """

    cdef public dict bids  # 买盘: price -> PriceLevel
    cdef public dict asks  # 卖盘: price -> PriceLevel
    cdef public dict order_map  # order_id -> Order
    cdef public double last_price
    cdef public double tick_size

    def __init__(self, tick_size: float = 0.1) -> None:
        """
        创建订单簿

        Args:
            tick_size: 最小变动单位
        """
        self.bids = {}  # 买盘
        self.asks = {}  # 卖盘
        self.order_map = {}  # 订单ID映射
        self.last_price = 0.0  # 最新价
        self.tick_size = tick_size

    def add_order(self, order: "Order") -> None:
        """
        向订单簿添加订单

        根据订单方向添加到买盘或卖盘，如果价格档位不存在则创建。

        Args:
            order: 订单对象
        """
        # 根据订单方向选择盘口
        if order.side == OrderSide.BUY:
            side_book = self.bids
        else:  # OrderSide.SELL
            side_book = self.asks

        # 检查价格档位是否存在，不存在则创建
        if order.price not in side_book:
            side_book[order.price] = PriceLevel(price=order.price)

        # 将订单添加到价格档位
        side_book[order.price].add_order(order)

        # 添加到订单映射表
        self.order_map[order.order_id] = order

    def cancel_order(self, order_id: int) -> "Order | None":
        """
        撤销订单

        根据订单ID查找并移除，如果档位变空则删除档位。

        Args:
            order_id: 订单 ID

        Returns:
            被撤销的订单对象，如果订单不存在则返回 None
        """
        # 1. 从 order_map 查找订单
        if order_id not in self.order_map:
            return None

        order = self.order_map[order_id]

        # 2. 根据订单方向选择盘口
        if order.side == OrderSide.BUY:
            side_book = self.bids
        else:  # OrderSide.SELL
            side_book = self.asks

        # 3. 从价格档位移除订单
        price_level = side_book[order.price]
        removed_order = price_level.remove_order(order_id)

        # 4. 如果档位变空，删除档位
        if len(price_level.orders) == 0:
            del side_book[order.price]

        # 5. 从 order_map 移除
        del self.order_map[order_id]

        return removed_order

    def get_best_bid(self) -> float | None:
        """
        获取最优买价

        返回买盘最高价，无买单返回 None

        Returns:
            买盘最高价格，如果买盘为空则返回 None
        """
        if self.bids:
            return max(self.bids.keys())
        return None

    def get_best_ask(self) -> float | None:
        """
        获取最优卖价

        返回卖盘最低价，无卖单返回 None

        Returns:
            卖盘最低价格，如果卖盘为空则返回 None
        """
        if self.asks:
            return min(self.asks.keys())
        return None

    def get_depth(self, levels: int = 100) -> dict[str, list[tuple[float, float]]]:
        """
        获取盘口深度

        返回买卖各 N 档的价格和数量。买盘从高到低，卖盘从低到高。

        Args:
            levels: 获取的档位数，默认 100

        Returns:
            {"bids": [(price, quantity), ...], "asks": [(price, quantity), ...]}
            买盘按价格降序排列，卖盘按价格升序排列
        """
        # 买盘：按价格从高到低排序，取前 levels 档
        bid_prices = sorted(self.bids.keys(), reverse=True)[:levels]
        bids = [(price, self.bids[price].get_volume()) for price in bid_prices]

        # 卖盘：按价格从低到高排序，取前 levels 档
        ask_prices = sorted(self.asks.keys())[:levels]
        asks = [(price, self.asks[price].get_volume()) for price in ask_prices]

        return {"bids": bids, "asks": asks}
