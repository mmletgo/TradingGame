# cython: language_level=3
"""
订单簿声明文件（Cython + C++）

定义 C 级别的订单结构体、PriceLevel 和 OrderBook 类型，
供其他 .pyx 文件 cimport 使用。
"""

from libcpp.map cimport map as cppmap
from libcpp.unordered_map cimport unordered_map
from cpython.ref cimport PyObject

import numpy as np
cimport numpy as np


# C 级别的订单结构体（侵入式双向链表节点）
cdef struct COrder:
    long long order_id
    long long agent_id
    int side          # 1=BUY, -1=SELL
    int order_type    # 1=LIMIT, 2=MARKET
    double price
    int quantity
    int filled_quantity
    COrder* prev
    COrder* next


cdef class PriceLevel:
    """价格档位（Cython + C++）

    使用侵入式双向链表维护订单的 FIFO 顺序，
    使用 unordered_map 提供 O(1) 的订单查找/删除。
    """
    cdef public double price
    cdef public long long total_quantity
    cdef COrder* head
    cdef COrder* tail
    cdef unordered_map[long long, COrder*] order_lookup

    cdef void add_order(self, COrder* order)
    cdef COrder* remove_order(self, long long order_id)
    cdef bint is_empty(self)


cdef class OrderBook:
    """订单簿（Cython + C++）

    使用 std::map 维护价格档位的有序性，
    使用 unordered_map 维护订单的 O(1) 查找。

    bids_map: std::map<double, PyObject*>  买盘（升序，最大键=最优买价）
    asks_map: std::map<double, PyObject*>  卖盘（升序，最小键=最优卖价）
    order_map_cpp: unordered_map<long long, COrder*>  全局订单索引
    """
    cdef cppmap[double, PyObject*] bids_map
    cdef cppmap[double, PyObject*] asks_map
    cdef unordered_map[long long, COrder*] order_map_cpp
    cdef public double last_price
    cdef public double tick_size
    cdef public bint _depth_dirty
    cdef public object _cached_depth
    cdef public int _cached_levels

    # 内部方法
    cdef void _dealloc_all(self)
    cdef void _erase_price_level(self, cppmap[double, PyObject*]* side_map, double price)

    # 热路径方法（C 级别）
    cdef COrder* add_order_raw(self, long long order_id, long long agent_id,
                                int side, int order_type, double price, int quantity)
    cdef bint cancel_order_fast(self, long long order_id)
    cdef double get_best_bid_price(self)
    cdef double get_best_ask_price(self)
    cdef bint has_order(self, long long order_id)
