"""鲶鱼账户模块"""

from src.market.account.position import Position
from src.market.matching.trade import Trade
from src.market.orderbook.order import OrderSide


class CatfishAccount:
    """鲶鱼账户类

    简化版账户，用于管理鲶鱼的余额、持仓、杠杆和保证金。

    Attributes:
        catfish_id: 鲶鱼 ID（负数）
        initial_balance: 初始余额
        balance: 当前余额
        position: 持仓对象
        leverage: 杠杆倍数
        maintenance_margin_rate: 维持保证金率
    """

    def __init__(
        self,
        catfish_id: int,
        initial_balance: float,
        leverage: float,
        maintenance_margin_rate: float,
    ) -> None:
        """创建鲶鱼账户"""
        self.catfish_id: int = catfish_id
        self.initial_balance: float = initial_balance
        self.balance: float = initial_balance
        self.position: Position = Position()
        self.leverage: float = leverage
        self.maintenance_margin_rate: float = maintenance_margin_rate

    def get_equity(self, current_price: float) -> float:
        """计算净值 = 余额 + 浮动盈亏"""
        return self.balance + self.position.get_unrealized_pnl(current_price)

    def get_margin_ratio(self, current_price: float) -> float:
        """计算保证金率 = 净值 / 持仓市值"""
        equity = self.get_equity(current_price)
        position_value = abs(self.position.quantity) * current_price
        if position_value == 0:
            return float("inf")
        return equity / position_value

    def check_liquidation(self, current_price: float) -> bool:
        """检查是否需要强平"""
        margin_ratio = self.get_margin_ratio(current_price)
        return margin_ratio < self.maintenance_margin_rate

    def on_trade(self, trade: Trade, is_buyer: bool) -> None:
        """处理成交（复用 Account 的逻辑，但手续费为0）

        Args:
            trade: 成交记录
            is_buyer: 是否为买方
        """
        # 确定成交方向（鲶鱼手续费为0，不从 trade 读取手续费）
        if is_buyer:
            side = OrderSide.BUY
        else:
            side = OrderSide.SELL

        # 更新持仓，获取已实现盈亏
        realized_pnl = self.position.update(side.value, trade.quantity, trade.price)

        # 将已实现盈亏加到余额（鲶鱼无手续费）
        self.balance += realized_pnl

    def on_adl_trade(self, quantity: int, price: float, is_taker: bool) -> float:
        """处理 ADL 成交

        Args:
            quantity: 成交数量（正数）
            price: 成交价格（破产价格）
            is_taker: 是否为被强平方

        Returns:
            已实现盈亏
        """
        abs_position = abs(self.position.quantity)
        actual_quantity = min(quantity, abs_position)

        if actual_quantity <= 0:
            return 0.0

        if self.position.quantity > 0:
            # 多头平仓/被减仓
            realized_pnl = (price - self.position.avg_price) * actual_quantity
            self.position.quantity -= actual_quantity
        else:
            # 空头平仓/被减仓
            realized_pnl = (self.position.avg_price - price) * actual_quantity
            self.position.quantity += actual_quantity

        # 仓位清零时重置均价
        if self.position.quantity == 0:
            self.position.avg_price = 0.0

        # 更新余额
        self.balance += realized_pnl
        self.position.realized_pnl += realized_pnl

        return realized_pnl

    def reset(self) -> None:
        """重置账户（Episode 开始时调用）"""
        self.balance = self.initial_balance
        self.position = Position()
