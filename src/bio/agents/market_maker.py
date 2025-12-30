"""做市商 Agent 模块

本模块定义做市商 Agent 类，继承自 Agent 基类。
"""

import time
from typing import Any

from src.bio.agents.base import Agent, ActionType
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event, EventType
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.orderbook.orderbook import OrderBook
from src.market.matching.trade import Trade


class MarketMakerAgent(Agent):
    """做市商 Agent

    代表市场流动性的提供者，初始资产 1000万，杠杆 10 倍。

    做市商通过同时维护买卖双边挂单（每边1-10单）为市场提供深度。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
        bid_order_ids: 买单订单ID列表（最多10个）
        ask_order_ids: 卖单订单ID列表（最多10个）
    """

    agent_id: int
    brain: Brain
    bid_order_ids: list[int]
    ask_order_ids: list[int]

    def __init__(
        self, agent_id: int, brain: Brain, config: AgentConfig, event_bus: EventBus
    ) -> None:
        """创建做市商 Agent

        调用父类构造函数，设置类型为 MARKET_MAKER，初始化买卖挂单列表。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
            event_bus: 事件总线
        """
        super().__init__(agent_id, AgentType.MARKET_MAKER, brain, config, event_bus)

        # 做市商每边（买/卖）可以挂1-10个订单
        self.bid_order_ids: list[int] = []
        self.ask_order_ids: list[int] = []

    def get_action_space(self) -> list[ActionType]:
        """获取做市商可用动作

        做市商可以执行双边挂单、清仓两种动作。

        Returns:
            可用动作类型列表 [QUOTE, CLEAR_POSITION]
        """
        return [ActionType.QUOTE, ActionType.CLEAR_POSITION]

    def decide(
        self, orderbook: OrderBook, recent_trades: list[Trade]
    ) -> tuple[ActionType, dict[str, Any]]:
        """决策下一步动作

        做市商通过神经网络决定是双边挂单还是清仓。
        神经网络输出结构（共 22 个值）：
        - 输出[0]: QUOTE 动作得分
        - 输出[1]: CLEAR_POSITION 动作得分
        - 输出[2-6]: 买单1-5的价格偏移（-1到1，相对于mid_price）
        - 输出[7-11]: 买单1-5的数量权重（-1到1，映射到0-1，10单归一化后总和为1.0）
        - 输出[12-16]: 卖单1-5的价格偏移（-1到1，相对于mid_price）
        - 输出[17-21]: 卖单1-5的数量权重（-1到1，映射到0-1，10单归一化后总和为1.0）

        Args:
            orderbook: 订单簿对象
            recent_trades: 最近成交记录列表

        Returns:
            (动作类型, 动作参数字典)
            - QUOTE: {"bid_orders": [{"price": float, "quantity": float}, ...],
                      "ask_orders": [{"price": float, "quantity": float}, ...]}
            - CLEAR_POSITION: {}
        """
        # 1. 观察市场，获取神经网络输入（复用基类方法）
        inputs = self.observe(orderbook, recent_trades)

        # 2. 神经网络前向传播
        outputs = self.brain.forward(inputs)

        # 3. 验证输出维度（需要 22 个值）
        if len(outputs) < 22:
            raise ValueError(f"神经网络输出维度不足，期望 22，实际 {len(outputs)}")

        # 4. 解析动作类型（选择 QUOTE 或 CLEAR_POSITION）
        quote_score = outputs[0]
        clear_score = outputs[1]

        if clear_score > quote_score:
            return ActionType.CLEAR_POSITION, {}

        # 5. QUOTE 动作：解析买卖单参数
        # 获取参考价格
        mid_price = orderbook.get_mid_price()
        if mid_price is None:
            mid_price = orderbook.last_price
        if mid_price == 0:
            mid_price = 100.0

        tick_size = orderbook.tick_size if orderbook.tick_size > 0 else 0.1

        # 首先收集所有 10 个订单的原始数量比例，然后归一化使总和为 1.0
        bid_raw_ratios: list[float] = []
        ask_raw_ratios: list[float] = []

        # 收集买单数量比例（映射到 [0, 1]）
        for i in range(5):
            quantity_ratio_norm = max(-1.0, min(1.0, outputs[7 + i]))
            # 映射 [-1, 1] 到 [0, 1]
            raw_ratio = (quantity_ratio_norm + 1) / 2
            bid_raw_ratios.append(max(0.0, raw_ratio))

        # 收集卖单数量比例（映射到 [0, 1]）
        for i in range(5):
            quantity_ratio_norm = max(-1.0, min(1.0, outputs[17 + i]))
            # 映射 [-1, 1] 到 [0, 1]
            raw_ratio = (quantity_ratio_norm + 1) / 2
            ask_raw_ratios.append(max(0.0, raw_ratio))

        # 计算总和并归一化（确保 10 个订单的总比例 = 1.0）
        total_raw_ratio = sum(bid_raw_ratios) + sum(ask_raw_ratios)
        if total_raw_ratio > 0:
            # 归一化系数，使总和 = 1.0
            normalize_factor = 1.0 / total_raw_ratio
            bid_ratios = [r * normalize_factor for r in bid_raw_ratios]
            ask_ratios = [r * normalize_factor for r in ask_raw_ratios]
        else:
            # 如果所有比例都是 0，则0
            bid_ratios = [0] * 5
            ask_ratios = [0] * 5

        bid_orders: list[dict[str, float]] = []
        ask_orders: list[dict[str, float]] = []

        # 解析 5 个买单（价格和数量）
        for i in range(5):
            price_offset_norm = max(-1.0, min(1.0, outputs[2 + i]))
            quantity_ratio = bid_ratios[i]

            # 计算价格（买单价格严格低于 mid_price）
            # 至少偏移 1 tick，最多偏移 100 ticks
            price_offset_ticks = max(
                1.0, abs(price_offset_norm) * 100
            )  # 1-100 ticks below
            price = mid_price - price_offset_ticks * tick_size

            quantity = self._calculate_order_quantity(price, quantity_ratio)

            # 只添加数量 > 0 的订单
            if quantity > 0:
                bid_orders.append({"price": price, "quantity": quantity})

        # 解析 5 个卖单（价格和数量）
        for i in range(5):
            price_offset_norm = max(-1.0, min(1.0, outputs[12 + i]))
            quantity_ratio = ask_ratios[i]

            # 计算价格（卖单价格严格高于 mid_price）
            # 至少偏移 1 tick，最多偏移 100 ticks
            price_offset_ticks = max(
                1.0, abs(price_offset_norm) * 100
            )  # 1-100 ticks above
            price = mid_price + price_offset_ticks * tick_size

            quantity = self._calculate_order_quantity(price, quantity_ratio)

            # 只添加数量 > 0 的订单
            if quantity > 0:
                ask_orders.append({"price": price, "quantity": quantity})

        return ActionType.QUOTE, {"bid_orders": bid_orders, "ask_orders": ask_orders}

    def execute_action(
        self, action: ActionType, params: dict[str, Any], event_bus: EventBus
    ) -> None:
        """执行动作

        做市商执行双边挂单或清仓动作。
        - QUOTE: 先撤掉所有旧挂单，然后双边各挂 1-5 单（每单价格和数量由神经网络决定）
        - CLEAR_POSITION: 先撤掉所有挂单，再根据持仓方向市价平仓

        Args:
            action: 动作类型
            params: 动作参数字典
                - QUOTE: {"bid_orders": [{"price": float, "quantity": float}, ...],
                          "ask_orders": [{"price": float, "quantity": float}, ...]}
                - CLEAR_POSITION: {}
            event_bus: 事件总线
        """
        timestamp = time.time()

        if action == ActionType.QUOTE:
            # 1. 先撤掉所有旧挂单
            for order_id in self.bid_order_ids:
                cancel_event = Event(
                    EventType.ORDER_CANCELLED,
                    timestamp,
                    {"order_id": order_id, "agent_id": self.agent_id},
                )
                event_bus.publish(cancel_event)

            for order_id in self.ask_order_ids:
                cancel_event = Event(
                    EventType.ORDER_CANCELLED,
                    timestamp,
                    {"order_id": order_id, "agent_id": self.agent_id},
                )
                event_bus.publish(cancel_event)

            # 清空挂单列表
            self.bid_order_ids.clear()
            self.ask_order_ids.clear()

            # 2. 挂新的双边订单
            bid_orders = params.get("bid_orders", [])
            ask_orders = params.get("ask_orders", [])

            # 挂买单
            for order_spec in bid_orders:
                price = order_spec["price"]
                quantity = order_spec["quantity"]
                order_id = self._generate_order_id()
                order = Order(
                    order_id=order_id,
                    agent_id=self.agent_id,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=quantity,
                )
                place_event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
                event_bus.publish(place_event)
                self.bid_order_ids.append(order_id)

            # 挂卖单
            for order_spec in ask_orders:
                price = order_spec["price"]
                quantity = order_spec["quantity"]
                order_id = self._generate_order_id()
                order = Order(
                    order_id=order_id,
                    agent_id=self.agent_id,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=quantity,
                )
                place_event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
                event_bus.publish(place_event)
                self.ask_order_ids.append(order_id)

        elif action == ActionType.CLEAR_POSITION:
            # 1. 先撤掉所有挂单
            for order_id in self.bid_order_ids:
                cancel_event = Event(
                    EventType.ORDER_CANCELLED,
                    timestamp,
                    {"order_id": order_id, "agent_id": self.agent_id},
                )
                event_bus.publish(cancel_event)

            for order_id in self.ask_order_ids:
                cancel_event = Event(
                    EventType.ORDER_CANCELLED,
                    timestamp,
                    {"order_id": order_id, "agent_id": self.agent_id},
                )
                event_bus.publish(cancel_event)

            # 清空挂单列表
            self.bid_order_ids.clear()
            self.ask_order_ids.clear()

            # 2. 根据持仓方向市价平仓
            position_qty = self.account.position.quantity

            if position_qty > 0:
                # 有多仓，市价卖出平仓
                order_id = self._generate_order_id()
                order = Order(
                    order_id=order_id,
                    agent_id=self.agent_id,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    price=0.0,
                    quantity=position_qty,
                )
                event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
                event_bus.publish(event)

            elif position_qty < 0:
                # 有空仓，市价买入平仓
                order_id = self._generate_order_id()
                order = Order(
                    order_id=order_id,
                    agent_id=self.agent_id,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    price=0.0,
                    quantity=abs(position_qty),
                )
                event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
                event_bus.publish(event)

            # 持仓为0时不做任何操作
