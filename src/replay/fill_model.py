"""成交模拟模型

保守模型：agent 排在真实队列之后。
只有当真实成交穿越 agent 的挂单价格时，agent 才能成交。
"""
from __future__ import annotations

from dataclasses import dataclass

from src.replay.config import ReplayConfig
from src.replay.data_loader import MarketTrade, OrderbookSnapshot


@dataclass
class PendingOrder:
    """Agent 的挂单"""

    order_id: int
    side: int  # 1=BUY, -1=SELL
    price: float
    quantity: int
    timestamp_ms: int


class FillModel:
    """成交模拟模型

    保守模型：agent 排在真实队列之后。
    只有当真实成交穿越 agent 的挂单价格时，agent 才能成交。
    """

    def __init__(self, config: ReplayConfig) -> None:
        self._config = config

    def check_passive_fills(
        self,
        pending_orders: list[PendingOrder],
        trade: MarketTrade,
    ) -> list[tuple[PendingOrder, int]]:
        """检查真实成交是否触发 agent 挂单的被动成交

        规则：
        - agent 的买单 (side=1): 当真实卖方成交 (trade.side='sell')
          且成交价 <= agent 挂单价时触发
        - agent 的卖单 (side=-1): 当真实买方成交 (trade.side='buy')
          且成交价 >= agent 挂单价时触发
        - 成交量 = min(挂单剩余量, 成交量转为整数 lots)

        Args:
            pending_orders: agent 当前的挂单列表
            trade: 真实市场成交

        Returns:
            [(order, fill_qty), ...] 被触发的挂单和成交量
        """
        fills: list[tuple[PendingOrder, int]] = []
        trade_qty_lots: int = max(1, int(trade.amount))
        remaining_trade_qty: int = trade_qty_lots

        for order in pending_orders:
            if remaining_trade_qty <= 0:
                break

            triggered: bool = False
            if order.side == 1 and trade.side == "sell" and trade.price <= order.price:
                triggered = True
            elif order.side == -1 and trade.side == "buy" and trade.price >= order.price:
                triggered = True

            if triggered:
                fill_qty: int = min(order.quantity, remaining_trade_qty)
                if fill_qty > 0:
                    fills.append((order, fill_qty))
                    remaining_trade_qty -= fill_qty

        return fills

    def check_active_fill(
        self,
        side: int,
        quantity: int,
        ob: OrderbookSnapshot,
    ) -> tuple[float, int]:
        """模拟 agent 市价单的主动成交

        市价买 -> 按 best ask 成交
        市价卖 -> 按 best bid 成交
        成交量限于对手方最优一档的深度

        Args:
            side: 1=BUY, -1=SELL
            quantity: 委托数量
            ob: 当前订单簿快照

        Returns:
            (fill_price, fill_quantity)。如果对手方为空返回 (0.0, 0)
        """
        if side == 1:  # BUY -> take from asks
            if len(ob.ask_prices) > 0 and ob.ask_prices[0] > 0:
                fill_price: float = float(ob.ask_prices[0])
                available: int = max(1, int(ob.ask_amounts[0]))
                fill_qty: int = min(quantity, available)
                return fill_price, fill_qty
        else:  # SELL -> take from bids
            if len(ob.bid_prices) > 0 and ob.bid_prices[0] > 0:
                fill_price = float(ob.bid_prices[0])
                available = max(1, int(ob.bid_amounts[0]))
                fill_qty = min(quantity, available)
                return fill_price, fill_qty

        return 0.0, 0
