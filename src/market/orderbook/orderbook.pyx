# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c++
"""
订单簿模块（Cython + C++ 加速）

使用 C++ std::map 替代 SortedDict，侵入式链表替代 OrderedDict，
unordered_map 替代 Python dict，实现全 C 级别的订单簿操作。
"""

import numpy as np
cimport numpy as np
from libc.math cimport NAN, isnan
from libc.stdlib cimport malloc, free
from libcpp.map cimport map as cppmap
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from cpython.ref cimport Py_INCREF, Py_DECREF, PyObject
from cython.operator cimport dereference as deref, preincrement as inc, predecrement as dec


cdef class PriceLevel:
    """价格档位（Cython + C++）

    使用侵入式双向链表维护订单 FIFO 顺序，
    使用 C++ unordered_map 提供 O(1) 的订单查找/删除。

    Attributes:
        price: 档位价格
        total_quantity: 该档位总挂单量（未成交部分）
    """
    # 属性在 .pxd 中声明

    def __cinit__(self, double price=0.0):
        self.price = price
        self.total_quantity = 0
        self.head = NULL
        self.tail = NULL

    def __dealloc__(self):
        # 由 OrderBook 统一释放 COrder 内存，这里不做操作
        pass

    cdef void add_order(self, COrder* order):
        """追加订单到链表尾部（FIFO）"""
        order.prev = self.tail
        order.next = NULL
        if self.tail != NULL:
            self.tail.next = order
        else:
            self.head = order
        self.tail = order
        self.order_lookup[order.order_id] = order
        self.total_quantity += (order.quantity - order.filled_quantity)

    cdef COrder* remove_order(self, long long order_id):
        """从链表中移除订单，返回被移除的 COrder*

        注意：调用者不要 free 返回的指针，OrderBook 统一管理。
        """
        cdef unordered_map[long long, COrder*].iterator it = self.order_lookup.find(order_id)
        if it == self.order_lookup.end():
            return NULL
        cdef COrder* order = deref(it).second
        self.order_lookup.erase(it)

        # 从双向链表中摘除
        if order.prev != NULL:
            order.prev.next = order.next
        else:
            self.head = order.next
        if order.next != NULL:
            order.next.prev = order.prev
        else:
            self.tail = order.prev

        order.prev = NULL
        order.next = NULL

        # 更新 total_quantity
        cdef int remaining = order.quantity - order.filled_quantity
        self.total_quantity -= remaining
        if self.total_quantity < 0:
            self.total_quantity = 0

        return order

    cdef bint is_empty(self):
        """判断档位是否为空"""
        return self.head == NULL

    def get_volume(self) -> int:
        """获取该价格档位的总挂单量"""
        return self.total_quantity


cdef class OrderBook:
    """订单簿（Cython + C++）

    使用 C++ std::map 维护买卖盘价格档位的有序性，
    使用 C++ unordered_map 维护全局订单索引。
    PriceLevel 作为 Python cdef class 在 map 中以 PyObject* 存储，
    手动管理引用计数。

    Attributes:
        last_price: 最新成交价
        tick_size: 最小变动单位
    """
    # 属性在 .pxd 中声明

    def __cinit__(self, double tick_size=0.01):
        self.last_price = 0.0
        self.tick_size = tick_size
        self._depth_dirty = True
        self._cached_depth = None
        self._cached_levels = 0

    def __init__(self, tick_size: float = 0.01) -> None:
        """创建订单簿

        Args:
            tick_size: 最小变动单位
        """
        pass  # __cinit__ 已初始化

    def __dealloc__(self):
        """析构函数：释放所有 COrder 内存和 PriceLevel 引用"""
        self._dealloc_all()

    cdef void _dealloc_all(self):
        """释放所有资源"""
        # 释放所有 COrder
        cdef COrder* order
        cdef unordered_map[long long, COrder*].iterator order_it = self.order_map_cpp.begin()
        while order_it != self.order_map_cpp.end():
            free(deref(order_it).second)
            inc(order_it)
        self.order_map_cpp.clear()

        # 释放所有 PriceLevel 的 Python 引用
        cdef cppmap[double, PyObject*].iterator map_it

        map_it = self.bids_map.begin()
        while map_it != self.bids_map.end():
            Py_DECREF(<object>deref(map_it).second)
            inc(map_it)
        self.bids_map.clear()

        map_it = self.asks_map.begin()
        while map_it != self.asks_map.end():
            Py_DECREF(<object>deref(map_it).second)
            inc(map_it)
        self.asks_map.clear()

    cdef void _erase_price_level(self, cppmap[double, PyObject*]* side_map, double price):
        """从 map 中删除价格档位并释放引用"""
        cdef cppmap[double, PyObject*].iterator it = side_map.find(price)
        if it != side_map.end():
            Py_DECREF(<object>deref(it).second)
            side_map.erase(it)

    # ========================================================================
    # 热路径方法（C 级别，无 Python 对象开销）
    # ========================================================================

    cdef COrder* add_order_raw(self, long long order_id, long long agent_id,
                                int side, int order_type, double price, int quantity):
        """创建 COrder（malloc）并加入订单簿

        Args:
            order_id: 订单 ID
            agent_id: Agent ID
            side: 1=BUY, -1=SELL
            order_type: 1=LIMIT, 2=MARKET
            price: 价格
            quantity: 数量

        Returns:
            新创建的 COrder 指针，失败返回 NULL
        """
        # 价格归一化
        cdef double normalized_price = round(price / self.tick_size) * self.tick_size
        normalized_price = round(normalized_price, 10)

        # 分配 COrder
        cdef COrder* order = <COrder*>malloc(sizeof(COrder))
        if order == NULL:
            return NULL
        order.order_id = order_id
        order.agent_id = agent_id
        order.side = side
        order.order_type = order_type
        order.price = normalized_price
        order.quantity = quantity
        order.filled_quantity = 0
        order.prev = NULL
        order.next = NULL

        # 选择买盘或卖盘
        cdef cppmap[double, PyObject*]* side_map
        if side == 1:  # BUY
            side_map = &self.bids_map
        else:          # SELL
            side_map = &self.asks_map

        # 查找或创建 PriceLevel
        cdef cppmap[double, PyObject*].iterator map_it = side_map.find(normalized_price)
        cdef PriceLevel level
        if map_it == side_map.end():
            level = PriceLevel.__new__(PriceLevel)
            level.price = normalized_price
            level.total_quantity = 0
            level.head = NULL
            level.tail = NULL
            Py_INCREF(level)
            deref(side_map)[normalized_price] = <PyObject*>level
        else:
            level = <PriceLevel>deref(map_it).second

        level.add_order(order)
        self.order_map_cpp[order_id] = order
        self._depth_dirty = True

        return order

    cdef bint cancel_order_fast(self, long long order_id):
        """快速撤单（C 级别）

        从 order_map_cpp 查到 COrder，从 PriceLevel 移除，free COrder。

        Args:
            order_id: 订单 ID

        Returns:
            True 表示撤单成功，False 表示订单不存在
        """
        cdef unordered_map[long long, COrder*].iterator it = self.order_map_cpp.find(order_id)
        if it == self.order_map_cpp.end():
            return False
        cdef COrder* order = deref(it).second
        self.order_map_cpp.erase(it)

        # 从对应的 PriceLevel 中移除
        cdef double price = order.price
        cdef int side = order.side
        cdef cppmap[double, PyObject*]* side_map
        if side == 1:
            side_map = &self.bids_map
        else:
            side_map = &self.asks_map

        cdef cppmap[double, PyObject*].iterator map_it = side_map.find(price)
        if map_it != side_map.end():
            (<PriceLevel>deref(map_it).second).remove_order(order_id)

            # 空档位清理
            if (<PriceLevel>deref(map_it).second).is_empty():
                Py_DECREF(<object>deref(map_it).second)
                side_map.erase(map_it)

        free(order)
        self._depth_dirty = True
        return True

    cdef double get_best_bid_price(self):
        """获取最优买价（C 级别）

        Returns:
            最优买价，无买单返回 NAN
        """
        if self.bids_map.empty():
            return NAN
        # std::map 升序排列，最后一个元素是最大键
        cdef cppmap[double, PyObject*].iterator it = self.bids_map.end()
        dec(it)
        return deref(it).first

    cdef double get_best_ask_price(self):
        """获取最优卖价（C 级别）

        Returns:
            最优卖价，无卖单返回 NAN
        """
        if self.asks_map.empty():
            return NAN
        return deref(self.asks_map.begin()).first

    cdef bint has_order(self, long long order_id):
        """检查订单是否存在（C 级别）

        Args:
            order_id: 订单 ID

        Returns:
            True 表示订单存在
        """
        return self.order_map_cpp.count(order_id) > 0

    # ========================================================================
    # Python 兼容方法（供非热路径使用）
    # ========================================================================

    def add_order(self, order) -> None:
        """Python 兼容的添加订单方法

        Args:
            order: 订单对象（Python Order 类实例）
        """
        self.add_order_raw(
            order.order_id, order.agent_id,
            int(order.side), int(order.order_type),
            order.price, order.quantity
        )

    def cancel_order(self, order_id):
        """Python 兼容的撤单方法

        Args:
            order_id: 订单 ID

        Returns:
            True 表示撤单成功，None 表示订单不存在
        """
        if self.cancel_order_fast(order_id):
            return True
        return None

    def get_best_bid(self):
        """获取最优买价

        Returns:
            买盘最高价格，如果买盘为空则返回 None
        """
        cdef double p = self.get_best_bid_price()
        if isnan(p):
            return None
        return p

    def get_best_ask(self):
        """获取最优卖价

        Returns:
            卖盘最低价格，如果卖盘为空则返回 None
        """
        cdef double p = self.get_best_ask_price()
        if isnan(p):
            return None
        return p

    def get_mid_price(self):
        """获取中间价

        Returns:
            中间价，如果买盘或卖盘任一为空则返回 None
        """
        cdef double bid = self.get_best_bid_price()
        cdef double ask = self.get_best_ask_price()
        if isnan(bid) or isnan(ask):
            return None
        return (bid + ask) / 2.0

    def get_depth(self, int levels=100):
        """获取盘口深度

        返回买卖各 N 档的价格和数量。买盘从高到低，卖盘从低到高。
        使用缓存机制，当订单簿未变化时直接返回缓存结果。

        Args:
            levels: 获取的档位数，默认 100

        Returns:
            {"bids": [(price, quantity), ...], "asks": [(price, quantity), ...]}
        """
        if not self._depth_dirty and self._cached_depth is not None and self._cached_levels == levels:
            return self._cached_depth

        cdef list bids = []
        cdef list asks = []
        cdef int i = 0

        # 买盘：从大到小（从 end 向 begin 遍历）
        cdef cppmap[double, PyObject*].iterator bit
        if not self.bids_map.empty():
            bit = self.bids_map.end()
            dec(bit)
            while i < levels:
                bids.append((deref(bit).first, (<PriceLevel>deref(bit).second).total_quantity))
                i += 1
                if bit == self.bids_map.begin():
                    break
                dec(bit)

        # 卖盘：从小到大（从 begin 向 end 遍历）
        i = 0
        cdef cppmap[double, PyObject*].iterator ait = self.asks_map.begin()
        while ait != self.asks_map.end() and i < levels:
            asks.append((deref(ait).first, (<PriceLevel>deref(ait).second).total_quantity))
            inc(ait)
            i += 1

        result = {"bids": bids, "asks": asks}
        self._cached_depth = result
        self._cached_levels = levels
        self._depth_dirty = False
        return result

    def get_depth_numpy(self, int levels=100):
        """获取盘口深度（NumPy 格式）

        直接返回 NumPy 数组，避免在 Python 层转换。
        买盘按价格降序，卖盘按价格升序。

        Args:
            levels: 获取的档位数，默认 100

        Returns:
            tuple[NDArray, NDArray]: (bid_data, ask_data)
            各 shape (levels, 2)，列0=价格，列1=数量。未填充的档位为 0。
        """
        cdef np.ndarray[np.float32_t, ndim=2] bid_data = np.zeros((levels, 2), dtype=np.float32)
        cdef np.ndarray[np.float32_t, ndim=2] ask_data = np.zeros((levels, 2), dtype=np.float32)

        # 买盘：从大到小（从 end 反向遍历）
        cdef int i = 0
        cdef cppmap[double, PyObject*].iterator bit
        if not self.bids_map.empty():
            bit = self.bids_map.end()
            dec(bit)
            while i < levels:
                bid_data[i, 0] = deref(bit).first
                bid_data[i, 1] = (<PriceLevel>deref(bit).second).total_quantity
                i += 1
                if bit == self.bids_map.begin():
                    break
                dec(bit)

        # 卖盘：从小到大（从 begin 正向遍历）
        i = 0
        cdef cppmap[double, PyObject*].iterator ait = self.asks_map.begin()
        while ait != self.asks_map.end() and i < levels:
            ask_data[i, 0] = deref(ait).first
            ask_data[i, 1] = (<PriceLevel>deref(ait).second).total_quantity
            inc(ait)
            i += 1

        return bid_data, ask_data

    def clear(self, reset_price=None) -> None:
        """清空订单簿

        释放所有 COrder 内存和 PriceLevel 引用。

        Args:
            reset_price: 重置后的最新价，如果为 None 则保持当前值
        """
        self._dealloc_all()

        if reset_price is not None:
            self.last_price = reset_price
        self._depth_dirty = True
        self._cached_depth = None

    @property
    def order_map(self):
        """返回 Python dict 快照，供非热路径兼容使用。

        注意：这是一个快照，修改返回的 dict 不会影响订单簿。
        """
        from src.market.orderbook.order import Order, OrderSide, OrderType
        result = {}
        cdef unordered_map[long long, COrder*].iterator it = self.order_map_cpp.begin()
        cdef COrder* order
        while it != self.order_map_cpp.end():
            order = deref(it).second
            py_order = Order(
                order_id=order.order_id,
                agent_id=order.agent_id,
                side=OrderSide(order.side),
                order_type=OrderType(order.order_type),
                price=order.price,
                quantity=order.quantity,
            )
            py_order.filled_quantity = order.filled_quantity
            result[order.order_id] = py_order
            inc(it)
        return result
