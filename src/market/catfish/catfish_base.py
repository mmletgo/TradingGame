"""
鲶鱼基类模块

定义鲶鱼（Catfish）的抽象基类。鲶鱼是一种特殊的市场参与者，
用于在训练中引入外部扰动，增加市场动态性。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from src.config.config import CatfishConfig
from src.market.catfish.catfish_account import CatfishAccount
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
        account: 鲶鱼账户
        is_liquidated: 是否被强平
    """

    def __init__(
        self,
        catfish_id: int,
        config: CatfishConfig,
        phase_offset: int = 0,
        initial_balance: float = 0.0,
        leverage: float = 10.0,
        maintenance_margin_rate: float = 0.05,
    ) -> None:
        """
        初始化鲶鱼

        Args:
            catfish_id: 鲶鱼ID（应为负数）
            config: 鲶鱼配置
            phase_offset: 相位偏移（用于错开多个鲶鱼的触发时间）
            initial_balance: 初始余额
            leverage: 杠杆倍数
            maintenance_margin_rate: 维持保证金率
        """
        if catfish_id >= 0:
            raise ValueError(f"鲶鱼ID必须为负数，当前值: {catfish_id}")

        self.catfish_id: int = catfish_id
        self.config: CatfishConfig = config
        self.phase_offset: int = phase_offset
        self._next_order_id: int = catfish_id * 1_000_000  # 负数空间的订单ID
        self._last_action_tick: int = -1000  # 初始值设为很早，确保首次可以行动

        # 新增：创建鲶鱼账户
        self.account: CatfishAccount = CatfishAccount(
            catfish_id=catfish_id,
            initial_balance=initial_balance,
            leverage=leverage,
            maintenance_margin_rate=maintenance_margin_rate,
        )

        # 新增：是否被强平
        self.is_liquidated: bool = False

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
        quantity = self._calculate_quantity(orderbook, direction)

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

        # 注册鲶鱼费率（maker=0, taker=0）
        matching_engine.register_agent(self.catfish_id, 0.0, 0.0)

        trades = matching_engine.match_market_order(order)
        return trades

    def _calculate_quantity(self, orderbook: "OrderBook", direction: int) -> int:
        """计算下单数量（让盘口波动至少 3 tick）

        Args:
            orderbook: 订单簿
            direction: 方向（1=买，-1=卖）

        Returns:
            下单数量
        """
        target_ticks = 3

        # 获取盘口深度
        depth = orderbook.get_depth(levels=target_ticks)

        if direction > 0:  # 买入，吃卖盘
            levels = depth["asks"]
        else:  # 卖出，吃买盘
            levels = depth["bids"]

        if len(levels) < target_ticks:
            return 0

        # 累加前 target_ticks 档的数量
        total_qty = 0
        for i in range(min(target_ticks, len(levels))):
            price, qty = levels[i]
            total_qty += int(qty)  # qty 可能是 float

        return total_qty

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

    def reset(self) -> None:
        """重置鲶鱼状态（Episode 开始时调用）"""
        self.account.reset()
        self.is_liquidated = False
        self._last_action_tick = -1000
