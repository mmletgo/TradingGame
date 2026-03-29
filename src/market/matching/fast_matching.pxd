# cython: language_level=3
"""
快速撮合引擎声明文件（Cython + C++）

声明导出的 cdef class 和 cpdef 函数，供其他 Cython 模块使用。
buyer_id/seller_id 使用 long long 以匹配项目中 agent_id 的 64 位整数类型。
"""


cdef class FastTrade:
    """快速成交记录"""
    cdef public int trade_id
    cdef public double price
    cdef public int quantity
    cdef public long long buyer_id
    cdef public long long seller_id
    cdef public double buyer_fee
    cdef public double seller_fee
    cdef public bint is_buyer_taker
    cdef public double timestamp


cdef class FastMatchingEngine:
    """快速撮合引擎"""
    cdef public object _orderbook
    cdef public int _next_trade_id
    cdef public dict _fee_rates
    cdef public double _tick_size

    cpdef list process_order_raw(self, long long order_id, long long agent_id,
                                 int side, int order_type, double price, int quantity)
    cpdef list process_order(self, object order)
    cpdef bint cancel_order(self, long long order_id)
    cpdef void register_agent(self, long long agent_id, double maker_rate, double taker_rate)
    cpdef double calculate_fee(self, long long agent_id, double amount, bint is_maker)
