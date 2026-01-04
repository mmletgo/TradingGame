"""
鲶鱼基类模块

定义鲶鱼（Catfish）的抽象基类。鲶鱼是一种特殊的市场参与者，
用于在训练中引入外部扰动，增加市场动态性。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.config.config import CatfishConfig
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.matching.trade import Trade

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine
    from src.market.orderbook.orderbook import OrderBook


class CatfishBase(ABC):
    """
    鲶鱼抽象基类

    鲶鱼是一种特殊的市场参与者，不参与 NEAT 进化，
    而是按照预设的规则进行交易，为市场引入外部扰动。

    Attributes:
        catfish_id: 鲶鱼ID（使用负数避免与Agent冲突）
        config: 鲶鱼配置
        _next_order_id: 下一个订单ID（负数空间）
        _last_action_tick: 上次行动的tick
    """

    def __init__(
        self, catfish_id: int, config: CatfishConfig, phase_offset: int = 0
    ) -> None:
        """
        初始化鲶鱼

        Args:
            catfish_id: 鲶鱼ID（应为负数）
            config: 鲶鱼配置
            phase_offset: 相位偏移（用于错开多个鲶鱼的触发时间）
        """
        if catfish_id >= 0:
            raise ValueError(f"鲶鱼ID必须为负数，当前值: {catfish_id}")

        self.catfish_id: int = catfish_id
        self.config: CatfishConfig = config
        self.phase_offset: int = phase_offset
        self._next_order_id: int = catfish_id * 1_000_000  # 负数空间的订单ID
        self._last_action_tick: int = -1000  # 初始值设为很早，确保首次可以行动

    @abstractmethod
    def decide(
        self,
        orderbook: "OrderBook",
        tick: int,
        price_history: list[float],
    ) -> tuple[bool, int]:
        """
        决策是否行动以及行动方向

        Args:
            orderbook: 订单簿
            tick: 当前tick
            price_history: 历史价格列表

        Returns:
            (should_act, direction): 是否行动和方向（1=买，-1=卖）
        """
        pass

    def execute(
        self,
        direction: int,
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """
        执行市价单

        Args:
            direction: 方向（1=买，-1=卖）
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        orderbook = matching_engine.orderbook
        quantity = self._calculate_quantity(orderbook)

        if quantity <= 0:
            return []

        side = OrderSide.BUY if direction > 0 else OrderSide.SELL

        order = Order(
            order_id=self._generate_order_id(),
            agent_id=self.catfish_id,
            side=side,
            order_type=OrderType.MARKET,
            price=0.0,  # 市价单价格不重要
            quantity=quantity,
        )

        # 注册鲶鱼的费率（使用庄家费率）
        matching_engine.register_agent(self.catfish_id, 0.0, 0.0001)

        trades = matching_engine.match_market_order(order)
        return trades

    def _calculate_quantity(self, orderbook: "OrderBook") -> int:
        """
        计算下单数量

        根据订单簿深度和资金配置计算下单数量。

        Args:
            orderbook: 订单簿

        Returns:
            下单数量
        """
        mid_price = orderbook.get_mid_price()
        if mid_price is None or mid_price <= 0:
            return 0

        # 计算鲶鱼资金
        fund = self.config.fund_multiplier * self.config.market_maker_base_fund

        # 计算可买数量（简单计算，不考虑杠杆）
        quantity = int(fund / mid_price)

        return max(1, quantity)

    def _generate_order_id(self) -> int:
        """
        生成订单ID（负数空间）

        Returns:
            新的订单ID（负数）
        """
        order_id = self._next_order_id
        self._next_order_id -= 1
        return order_id

    def can_act(self, tick: int) -> bool:
        """
        检查是否可以行动（冷却时间检查 + 相位偏移）

        Args:
            tick: 当前tick

        Returns:
            是否可以行动
        """
        # 考虑相位偏移：(tick - phase_offset) 必须是冷却周期的整数倍
        effective_tick = tick - self.phase_offset
        if effective_tick < 0:
            return False
        return tick - self._last_action_tick >= self.config.action_cooldown

    def record_action(self, tick: int) -> None:
        """
        记录行动时间

        Args:
            tick: 当前tick
        """
        self._last_action_tick = tick
