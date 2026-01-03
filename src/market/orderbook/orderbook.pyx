"""
订单簿模块（Cython 加速）

定义价格档位和订单簿
"""

# cython: language_level=3
from collections import OrderedDict
from typing import TYPE_CHECKING

from sortedcontainers import SortedDict

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
    cdef public long long total_quantity  # 64 位整数，避免大量订单累积时溢出

    def __init__(self, price: float) -> None:
        """
        创建价格档位

        Args:
            price: 档位价格
        """
        self.price = price
        self.orders = OrderedDict()  # order_id -> Order，保持 FIFO 顺序
        self.total_quantity = 0  # 总数量

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
            # 减去剩余数量（未成交部分），而不是原始数量
            # 因为部分成交时 total_quantity 已经减去了成交数量
            remaining = order.quantity - order.filled_quantity
            self.total_quantity -= remaining
            return order
        return None

    def get_volume(self) -> int:
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

    cdef public object bids  # 买盘: SortedDict[price -> PriceLevel]，升序排列
    cdef public object asks  # 卖盘: SortedDict[price -> PriceLevel]，升序排列
    cdef public dict order_map  # order_id -> Order
    cdef public double last_price
    cdef public double tick_size
    cdef public bint _depth_dirty  # 深度缓存是否失效
    cdef public object _cached_depth  # 缓存的深度数据
    cdef public int _cached_levels  # 缓存的档位数

    def __init__(self, tick_size: float = 0.1) -> None:
        """
        创建订单簿

        Args:
            tick_size: 最小变动单位
        """
        self.bids = SortedDict()  # 买盘，升序排列，最大键在末尾
        self.asks = SortedDict()  # 卖盘，升序排列，最小键在开头
        self.order_map = {}  # 订单ID映射
        self.last_price = 0.0  # 最新价
        self.tick_size = tick_size
        self._depth_dirty = True  # 初始时缓存无效
        self._cached_depth = None
        self._cached_levels = 0

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

        # 【修复浮点精度问题】将价格舍入到 tick_size 的整数倍
        # 使用 round() 消除浮点精度误差（如 91.30000000000001 -> 91.3）
        cdef double normalized_price = round(order.price / self.tick_size) * self.tick_size
        # 再次舍入以消除乘法引入的微小误差
        normalized_price = round(normalized_price, 10)
        order.price = normalized_price

        # 检查价格档位是否存在，不存在则创建
        if normalized_price not in side_book:
            side_book[normalized_price] = PriceLevel(price=normalized_price)

        # 将订单添加到价格档位
        side_book[normalized_price].add_order(order)

        # 添加到订单映射表
        self.order_map[order.order_id] = order

        # 标记缓存失效
        self._depth_dirty = True

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

        # 【修复浮点精度问题】使用与 add_order 相同的归一化逻辑
        cdef double normalized_price = round(order.price / self.tick_size) * self.tick_size
        normalized_price = round(normalized_price, 10)

        # 3. 从价格档位移除订单
        if normalized_price not in side_book:
            # 价格档位不存在，可能是数据不一致，仍需从 order_map 移除
            del self.order_map[order_id]
            self._depth_dirty = True
            return order

        price_level = side_book[normalized_price]
        removed_order = price_level.remove_order(order_id)

        # 4. 如果档位变空，删除档位
        if len(price_level.orders) == 0:
            del side_book[normalized_price]

        # 5. 从 order_map 移除
        del self.order_map[order_id]

        # 标记缓存失效
        self._depth_dirty = True

        return removed_order

    def get_best_bid(self) -> float | None:
        """
        获取最优买价

        返回买盘最高价，无买单返回 None。
        使用 SortedDict.peekitem(-1) 获取最大键，时间复杂度 O(1)。

        Returns:
            买盘最高价格，如果买盘为空则返回 None
        """
        if self.bids:
            return self.bids.peekitem(-1)[0]
        return None

    def get_best_ask(self) -> float | None:
        """
        获取最优卖价

        返回卖盘最低价，无卖单返回 None。
        使用 SortedDict.peekitem(0) 获取最小键，时间复杂度 O(1)。

        Returns:
            卖盘最低价格，如果卖盘为空则返回 None
        """
        if self.asks:
            return self.asks.peekitem(0)[0]
        return None

    def get_mid_price(self) -> float | None:
        """
        获取中间价

        返回最优买卖价的平均值，无法计算返回 None

        Returns:
            中间价，如果买盘或卖盘任一为空则返回 None
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()

        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None

    def get_depth(self, levels: int = 100) -> dict[str, list[tuple[float, float]]]:
        """
        获取盘口深度

        返回买卖各 N 档的价格和数量。买盘从高到低，卖盘从低到高。
        利用 SortedDict 已排序特性，避免每次调用都排序，时间复杂度从 O(n log n) 降为 O(levels)。
        使用缓存机制，当订单簿未变化时直接返回缓存结果。

        Args:
            levels: 获取的档位数，默认 100

        Returns:
            {"bids": [(price, quantity), ...], "asks": [(price, quantity), ...]}
            买盘按价格降序排列，卖盘按价格升序排列
        """
        # 如果缓存有效且 levels 匹配，直接返回
        if not self._depth_dirty and self._cached_depth is not None and self._cached_levels == levels:
            return self._cached_depth

        # 买盘：SortedDict 升序排列，取最后 levels 个并反转得到降序
        # 使用切片获取最后 levels 个键，然后反转
        bid_keys = self.bids.keys()
        bid_count = len(bid_keys)
        start_idx = max(0, bid_count - levels)
        bid_prices = list(bid_keys[start_idx:])[::-1]  # 反转为降序
        bids = [(price, self.bids[price].get_volume()) for price in bid_prices]

        # 卖盘：SortedDict 升序排列，直接取前 levels 个
        ask_keys = self.asks.keys()
        ask_prices = list(ask_keys[:levels])
        asks = [(price, self.asks[price].get_volume()) for price in ask_prices]

        result = {"bids": bids, "asks": asks}

        # 缓存结果
        self._cached_depth = result
        self._cached_levels = levels
        self._depth_dirty = False

        return result

    def clear(self, reset_price: float | None = None) -> None:
        """
        清空订单簿

        清空所有买卖盘和订单映射，可选地重置最新价。
        重新初始化 SortedDict 以确保完全清空。

        Args:
            reset_price: 重置后的最新价，如果为 None 则保持当前值
        """
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.order_map.clear()
        if reset_price is not None:
            self.last_price = reset_price
        # 清空缓存
        self._depth_dirty = True
        self._cached_depth = None
