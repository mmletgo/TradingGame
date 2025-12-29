"""持仓模块

本模块定义持仓(Position)类，用于记录 Agent 的持仓状态。
"""

from dataclasses import dataclass, field
from typing import Self

from src.market.orderbook.order import OrderSide


@dataclass
class Position:
    """持仓类

    用于记录 Agent 当前持有的资产数量、平均买入价格以及已实现的盈亏。

    Attributes:
        quantity: 持仓数量（正为多头，负为空头）
        avg_price: 持仓均价
        realized_pnl: 已实现盈亏
    """

    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    def __init__(self) -> None:
        """创建持仓对象

        初始化数量、均价、已实现盈亏为 0。
        """
        self.quantity: float = 0.0
        self.avg_price: float = 0.0
        self.realized_pnl: float = 0.0

    def update(self, side: OrderSide, quantity: float, price: float) -> float:
        """更新持仓，返回已实现盈亏

        根据成交方向和数量更新持仓状态，计算并返回本次成交产生的已实现盈亏。
        支持开仓、加仓、减仓、反向开仓等场景。

        Args:
            side: 成交方向（BUY 做多 或 SELL 做空）
            quantity: 成交数量
            price: 成交价格

        Returns:
            本次成交产生的已实现盈亏（仅在平仓时非零）
        """
        realized: float = 0.0
        eps: float = 1e-9  # 浮点数精度阈值

        # 空仓：直接开仓
        if abs(self.quantity) < eps:
            self.quantity = quantity * side.value
            self.avg_price = price
            return 0.0

        # 持多头
        if self.quantity > 0:
            if side == OrderSide.BUY:
                # 加多仓：加权平均计算新均价
                total_cost = self.quantity * self.avg_price + quantity * price
                self.quantity += quantity
                self.avg_price = total_cost / self.quantity
            else:
                # SELL: 减仓、平仓或反向开空
                if quantity < self.quantity - eps:
                    # 减多仓
                    realized = (price - self.avg_price) * quantity
                    self.quantity -= quantity
                elif abs(quantity - self.quantity) < eps:
                    # 完全平多
                    realized = (price - self.avg_price) * self.quantity
                    self.quantity = 0.0
                    self.avg_price = 0.0
                else:
                    # 反向开空：先平多仓，再开空仓
                    realized = (price - self.avg_price) * self.quantity
                    remaining = quantity - self.quantity
                    self.quantity = -remaining
                    self.avg_price = price

        # 持空头（对称逻辑）
        else:
            if side == OrderSide.SELL:
                # 加空仓：加权平均计算新均价
                total_cost = abs(self.quantity) * self.avg_price + quantity * price
                self.quantity -= quantity  # quantity 是负数，绝对值增加
                self.avg_price = total_cost / abs(self.quantity)
            else:
                # BUY: 减空仓、平空仓或反向开多
                abs_quantity = abs(self.quantity)
                if quantity < abs_quantity - eps:
                    # 减空仓
                    realized = (self.avg_price - price) * quantity
                    self.quantity += quantity  # quantity 是正数，负值减小
                elif abs(quantity - abs_quantity) < eps:
                    # 完全平空
                    realized = (self.avg_price - price) * abs_quantity
                    self.quantity = 0.0
                    self.avg_price = 0.0
                else:
                    # 反向开多：先平空仓，再开多仓
                    realized = (self.avg_price - price) * abs_quantity
                    remaining = quantity - abs_quantity
                    self.quantity = remaining
                    self.avg_price = price

        self.realized_pnl += realized
        return realized

    def get_unrealized_pnl(self, current_price: float) -> float:
        """计算浮动盈亏

        根据当前市场价格计算持仓的浮动盈亏（未实现盈亏）。
        统一公式：(当前价 - 均价) × 持仓数量

        Args:
            current_price: 当前市场价格

        Returns:
            浮动盈亏（正数盈利，负数亏损）
        """
        return (current_price - self.avg_price) * self.quantity

    def get_margin_used(self, current_price: float, leverage: float) -> float:
        """计算占用保证金

        根据当前持仓数量、市场价格和杠杆倍数，计算持仓占用的保证金金额。

        Args:
            current_price: 当前市场价格
            leverage: 杠杆倍数

        Returns:
            占用的保证金金额
        """
        return abs(self.quantity) * current_price / leverage
