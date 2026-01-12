# cython: language_level=3
"""持仓模块（Cython 加速）"""

from src.market.orderbook.order import OrderSide


cdef class Position:
    """持仓类（Cython 加速）

    用于记录 Agent 当前持有的资产数量、平均买入价格以及已实现的盈亏。
    """

    # 属性声明已移至 position.pxd

    def __init__(self):
        """创建持仓对象"""
        self.quantity = 0
        self.avg_price = 0.0
        self.realized_pnl = 0.0

    cpdef double update(self, int side, int quantity, double price):
        """更新持仓，返回已实现盈亏

        Args:
            side: 成交方向（1=BUY, -1=SELL）
            quantity: 成交数量
            price: 成交价格

        Returns:
            本次成交产生的已实现盈亏
        """
        cdef double realized = 0.0
        cdef double total_cost
        cdef int abs_quantity, remaining

        # 空仓：直接开仓
        if self.quantity == 0:
            self.quantity = quantity * side
            self.avg_price = price
            return 0.0

        # 持多头
        if self.quantity > 0:
            if side == 1:  # BUY
                # 加多仓：加权平均计算新均价
                total_cost = self.quantity * self.avg_price + quantity * price
                self.quantity += quantity
                self.avg_price = total_cost / self.quantity
            else:  # SELL
                if quantity < self.quantity:
                    # 减多仓
                    realized = (price - self.avg_price) * quantity
                    self.quantity -= quantity
                elif quantity == self.quantity:
                    # 完全平多
                    realized = (price - self.avg_price) * self.quantity
                    self.quantity = 0
                    self.avg_price = 0.0
                else:
                    # 反向开空
                    realized = (price - self.avg_price) * self.quantity
                    remaining = quantity - self.quantity
                    self.quantity = -remaining
                    self.avg_price = price
        # 持空头
        else:
            if side == -1:  # SELL
                # 加空仓
                total_cost = abs(self.quantity) * self.avg_price + quantity * price
                self.quantity -= quantity
                self.avg_price = total_cost / abs(self.quantity)
            else:  # BUY
                abs_quantity = abs(self.quantity)
                if quantity < abs_quantity:
                    # 减空仓
                    realized = (self.avg_price - price) * quantity
                    self.quantity += quantity
                elif quantity == abs_quantity:
                    # 完全平空
                    realized = (self.avg_price - price) * abs_quantity
                    self.quantity = 0
                    self.avg_price = 0.0
                else:
                    # 反向开多
                    realized = (self.avg_price - price) * abs_quantity
                    remaining = quantity - abs_quantity
                    self.quantity = remaining
                    self.avg_price = price

        self.realized_pnl += realized
        return realized

    cpdef double get_unrealized_pnl(self, double current_price):
        """计算浮动盈亏"""
        return (current_price - self.avg_price) * self.quantity

    cpdef double get_margin_used(self, double current_price, double leverage):
        """计算占用保证金"""
        return abs(self.quantity) * current_price / leverage
