"""噪声交易者模块"""

import random

from src.market.noise_trader.noise_trader_account import NoiseTraderAccount
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.matching.trade import Trade

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.config.config import NoiseTraderConfig
    from src.market.matching.matching_engine import MatchingEngine


class NoiseTrader:
    """噪声交易者

    每 tick 以固定概率行动，行动时随机买/卖，下单量从对数正态分布采样。
    100个独立噪声交易者产生近似布朗运动的价格随机游走。
    """

    def __init__(self, trader_id: int, config: "NoiseTraderConfig") -> None:
        if trader_id >= 0:
            raise ValueError(f"噪声交易者ID必须为负数，当前值: {trader_id}")

        self.trader_id: int = trader_id
        self.config: "NoiseTraderConfig" = config
        self.account: NoiseTraderAccount = NoiseTraderAccount(trader_id)
        self._next_order_id: int = trader_id * 1_000_000

    def decide(self, buy_probability: float = 0.5) -> tuple[bool, int, int]:
        """决策是否行动、方向和数量

        Args:
            buy_probability: 买入概率，默认0.5表示等概率买卖

        Returns:
            (should_act, direction, quantity):
            - should_act: 是否行动
            - direction: 1=买, -1=卖
            - quantity: 下单数量
        """
        if random.random() >= self.config.action_probability:
            return False, 0, 0

        direction = 1 if random.random() < buy_probability else -1
        quantity = max(1, int(random.lognormvariate(self.config.quantity_mu, self.config.quantity_sigma)))
        return True, direction, quantity

    def execute(
        self,
        direction: int,
        quantity: int,
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """执行市价单

        Args:
            direction: 1=买, -1=卖
            quantity: 下单数量
            matching_engine: 撮合引擎

        Returns:
            成交记录列表
        """
        self._next_order_id -= 1
        order = Order(
            order_id=self._next_order_id,
            agent_id=self.trader_id,
            side=OrderSide.BUY if direction == 1 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            price=0.0,
            quantity=quantity,
        )

        trades = matching_engine.process_order(order)
        for trade in trades:
            # 噪声交易者提交市价单，始终是 taker
            self.account.on_trade(trade, is_taker=True)
        return trades

    def reset(self) -> None:
        """重置噪声交易者"""
        self.account.reset()
        self._next_order_id = self.trader_id * 1_000_000
