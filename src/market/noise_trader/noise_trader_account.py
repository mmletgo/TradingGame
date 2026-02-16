"""噪声交易者账户模块"""

from src.market.matching.trade import Trade


class NoiseTraderAccount:
    """噪声交易者账户

    无限资金，不会被强平。保留持仓和 PnL 跟踪，可作为 ADL 对手方。
    """

    def __init__(self, trader_id: int) -> None:
        self.trader_id: int = trader_id
        self.initial_balance: float = 1e18
        self.balance: float = self.initial_balance
        self.position_qty: int = 0
        self.position_avg_price: float = 0.0
        self.realized_pnl: float = 0.0

    def on_trade(self, trade: Trade, is_taker: bool) -> None:
        """处理成交（无手续费）"""
        qty = trade.quantity
        price = trade.price

        # Determine direction based on trade
        if is_taker:
            direction = 1 if trade.is_buyer_taker else -1
        else:
            direction = -1 if trade.is_buyer_taker else 1

        signed_qty = qty * direction

        if self.position_qty == 0:
            # Open new position
            self.position_qty = signed_qty
            self.position_avg_price = price
        elif (self.position_qty > 0 and direction > 0) or (self.position_qty < 0 and direction < 0):
            # Add to position
            total_cost = abs(self.position_qty) * self.position_avg_price + qty * price
            self.position_qty += signed_qty
            if self.position_qty != 0:
                self.position_avg_price = total_cost / abs(self.position_qty)
        else:
            # Reduce or reverse position
            close_qty = min(qty, abs(self.position_qty))
            if self.position_qty > 0:
                pnl = close_qty * (price - self.position_avg_price)
            else:
                pnl = close_qty * (self.position_avg_price - price)
            self.realized_pnl += pnl
            self.balance += pnl

            remaining = qty - close_qty
            if self.position_qty > 0:
                self.position_qty -= close_qty
            else:
                self.position_qty += close_qty

            if remaining > 0 and self.position_qty == 0:
                # Reverse position
                self.position_qty = remaining * direction
                self.position_avg_price = price

    def on_adl_trade(self, price: float, quantity: int) -> None:
        """处理 ADL 成交"""
        if self.position_qty == 0:
            return
        close_qty = min(quantity, abs(self.position_qty))
        if self.position_qty > 0:
            pnl = close_qty * (price - self.position_avg_price)
            self.position_qty -= close_qty
        else:
            pnl = close_qty * (self.position_avg_price - price)
            self.position_qty += close_qty
        self.realized_pnl += pnl
        self.balance += pnl

    def get_equity(self, current_price: float) -> float:
        """获取净值"""
        unrealized_pnl = 0.0
        if self.position_qty != 0:
            if self.position_qty > 0:
                unrealized_pnl = self.position_qty * (current_price - self.position_avg_price)
            else:
                unrealized_pnl = abs(self.position_qty) * (self.position_avg_price - current_price)
        return self.balance + unrealized_pnl

    def reset(self) -> None:
        """重置账户"""
        self.balance = self.initial_balance
        self.position_qty = 0
        self.position_avg_price = 0.0
        self.realized_pnl = 0.0
