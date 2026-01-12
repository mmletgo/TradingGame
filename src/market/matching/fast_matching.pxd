# cython: language_level=3
"""
快速撮合引擎声明文件（Cython）

声明导出的 cdef class 和 cpdef 函数，供其他 Cython 模块使用。
"""


cdef class FastTrade:
    """快速成交记录"""
    cdef public int trade_id
    cdef public double price
    cdef public int quantity
    cdef public int buyer_id
    cdef public int seller_id
    cdef public double buyer_fee
    cdef public double seller_fee
    cdef public bint is_buyer_taker
    cdef public double timestamp


cdef class FastMatchingEngine:
    """快速撮合引擎"""
    cdef public object orderbook
    cdef public int _next_trade_id
    cdef public dict _fee_rates
    cdef public double _tick_size

    cpdef void register_agent(self, int agent_id, double maker_rate, double taker_rate)
    cpdef double calculate_fee(self, int agent_id, double amount, bint is_maker)
    cpdef list match_limit_order(self, object order)
    cpdef list match_market_order(self, object order)
    cpdef list process_order(self, object order)
    cpdef bint cancel_order(self, int order_id)


cpdef tuple fast_match_orders(
    object orderbook,
    object order,
    dict fee_rates,
    int next_trade_id,
    bint is_limit_order
)
