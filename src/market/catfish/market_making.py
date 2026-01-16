# -*- coding: utf-8 -*-
"""
做市鲶鱼模块

实现做市策略的鲶鱼，通过双边挂限价单为市场提供流动性。
与其他吃单鲶鱼不同，做市鲶鱼不消耗订单簿深度，反而增加流动性。
"""

from typing import TYPE_CHECKING

from src.config.config import CatfishConfig
from src.market.catfish.catfish_base import CatfishBase
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.matching.trade import Trade

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine
    from src.market.orderbook.orderbook import OrderBook


class MarketMakingCatfish(CatfishBase):
    """
    做市鲶鱼 - 提供双边流动性

    与其他吃单鲶鱼不同，做市鲶鱼通过挂限价单为市场提供流动性。
    不会消耗订单簿深度，反而会增加流动性。

    策略逻辑：
    - 每个 tick 在 last_price 附近双边挂限价单
    - 挂单前先撤销上一个 tick 的所有挂单
    - 买单在 last_price - 1~3 tick 处，卖单在 last_price + 1~3 tick 处
    - 每档挂单量固定为100

    注意：挂单基于 last_price 而非 best_bid/best_ask，确保在流动性枯竭时
    仍能在中间价附近提供双边流动性。

    Attributes:
        catfish_id: 鲶鱼ID（固定为 -4）
        config: 鲶鱼配置
        _pending_order_ids: 当前挂单ID列表
        _target_depth: 目标深度（在盘口附近挂几档）
        _order_size: 每档挂单量
    """

    def __init__(
        self,
        catfish_id: int,
        config: CatfishConfig,
        initial_balance: float = 0.0,
        leverage: float = 10.0,
        maintenance_margin_rate: float = 0.05,
    ) -> None:
        """
        初始化做市鲶鱼

        Args:
            catfish_id: 鲶鱼ID（应为 -4）
            config: 鲶鱼配置
            initial_balance: 初始余额
            leverage: 杠杆倍数
            maintenance_margin_rate: 维持保证金率
        """
        super().__init__(
            catfish_id, config,
            initial_balance, leverage, maintenance_margin_rate
        )
        # 当前挂单ID列表
        self._pending_order_ids: list[int] = []
        # 目标深度（在盘口附近挂几档）
        self._target_depth: int = 3
        # 每档挂单量
        self._order_size: int = 100

    def decide(
        self,
        orderbook: "OrderBook",
        tick: int,
        price_history: list[float],
    ) -> tuple[bool, int]:
        """
        决策是否行动以及行动方向

        做市鲶鱼总是行动，返回 (True, 0)。
        direction=0 表示双边挂单（不同于其他鲶鱼的 1=买/-1=卖）。

        Args:
            orderbook: 订单簿
            tick: 当前tick
            price_history: 历史价格列表（未使用）

        Returns:
            (should_act, direction): 总是返回 (True, 0)
        """
        return True, 0

    def execute(
        self,
        direction: int,
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """
        执行做市：先撤销旧挂单，再双边挂新单

        与其他鲶鱼不同，做市鲶鱼：
        1. 不提交市价单，只挂限价单
        2. 挂单在盘口外侧，不会立即成交
        3. 返回空列表（无即时成交）

        Args:
            direction: 方向（此处未使用，因为是双边挂单）
            matching_engine: 撮合引擎

        Returns:
            空列表（限价单挂在盘口外侧，不会立即成交）
        """
        # 1. 撤销旧订单
        for order_id in self._pending_order_ids:
            matching_engine.cancel_order(order_id)
        self._pending_order_ids.clear()

        # 2. 获取订单簿信息
        orderbook = matching_engine.orderbook
        tick_size: float = orderbook.tick_size
        last_price: float = orderbook.last_price

        # 3. 注册鲶鱼费率（maker=0, taker=0）
        matching_engine.register_agent(self.catfish_id, 0.0, 0.0)

        # 4. 以 last_price 为基准挂单
        # 买单在 last_price 下方，卖单在 last_price 上方
        # 这样在流动性枯竭时也能在中间价附近提供双边流动性
        for i in range(1, self._target_depth + 1):
            # 买单：last_price - i*tick_size
            bid_price: float = last_price - tick_size * i
            bid_order = Order(
                order_id=self._generate_order_id(),
                agent_id=self.catfish_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=bid_price,
                quantity=self._order_size,
            )
            matching_engine.process_order(bid_order)
            self._pending_order_ids.append(bid_order.order_id)

            # 卖单：last_price + i*tick_size
            ask_price: float = last_price + tick_size * i
            ask_order = Order(
                order_id=self._generate_order_id(),
                agent_id=self.catfish_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=ask_price,
                quantity=self._order_size,
            )
            matching_engine.process_order(ask_order)
            self._pending_order_ids.append(ask_order.order_id)

        # 限价单在 last_price 附近，提供双边流动性
        return []

    def reset(self) -> None:
        """
        重置做市鲶鱼状态

        每个 Episode 开始时调用，重置账户并清空挂单ID列表。
        """
        super().reset()
        # 清空挂单ID列表
        self._pending_order_ids.clear()
