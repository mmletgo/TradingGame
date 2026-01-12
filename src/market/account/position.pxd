# cython: language_level=3
"""Position 类的 Cython 声明文件"""


cdef class Position:
    """持仓类的 cdef 声明"""

    cdef public int quantity
    cdef public double avg_price
    cdef public double realized_pnl

    cpdef double update(self, int side, int quantity, double price)
    cpdef double get_unrealized_pnl(self, double current_price)
    cpdef double get_margin_used(self, double current_price, double leverage)
