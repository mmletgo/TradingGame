"""散户 Agent 模块

本模块定义散户 Agent 类，继承自 Agent 基类。
"""

import time
from typing import Any, Callable

from src.bio.agents.base import ActionType
from src.bio.agents.base import Agent
from src.config.config import AgentConfig, AgentType
from src.bio.brain.brain import Brain
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event, EventType
from src.market.orderbook.order import Order, OrderSide, OrderType


class RetailAgent(Agent):
    """散户 Agent

    代表市场中数量最多的交易参与者，初始资产较少，杠杆倍数最高。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
        _action_handlers: 动作处理器字典，将动作类型映射到处理函数
    """

    agent_id: int
    brain: Brain
    _action_handlers: dict[ActionType, Callable[[dict[str, Any], EventBus], None]]

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig, event_bus: EventBus) -> None:
        """创建散户 Agent

        调用父类构造函数，设置类型为 RETAIL，并初始化动作处理器。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
            event_bus: 事件总线
        """
        super().__init__(agent_id, AgentType.RETAIL, brain, config, event_bus)
        self._init_action_handlers()

    def _init_action_handlers(self) -> None:
        """初始化动作处理器字典

        先调用父类初始化基础动作处理器，再覆盖 PLACE_BID 和 PLACE_ASK 为散户特定实现。
        """
        super()._init_action_handlers()
        # 覆盖 PLACE_BID 和 PLACE_ASK 为散户特定实现（先撤旧单再挂新单）
        self._action_handlers[ActionType.PLACE_BID] = lambda params, event_bus: self._handle_place_order(
            ActionType.PLACE_BID, params, event_bus
        )
        self._action_handlers[ActionType.PLACE_ASK] = lambda params, event_bus: self._handle_place_order(
            ActionType.PLACE_ASK, params, event_bus
        )

    def get_action_space(self) -> list[ActionType]:
        """获取散户可用动作空间

        散户可以执行的动作包括：不动、挂买单、挂卖单、撤单、市价买入、市价卖出。

        Returns:
            可用动作类型列表
        """
        return [
            ActionType.HOLD,
            ActionType.PLACE_BID,
            ActionType.PLACE_ASK,
            ActionType.CANCEL,
            ActionType.MARKET_BUY,
            ActionType.MARKET_SELL,
        ]

    def _handle_place_order(self, action: ActionType, params: dict[str, Any], event_bus: EventBus) -> None:
        """处理挂单操作

        散户同时只能挂一单。如果已有挂单，先撤旧单再挂新单。

        Args:
            action: 动作类型（PLACE_BID 或 PLACE_ASK）
            params: 动作参数字典，包含 {"price": float, "quantity": float}
            event_bus: 事件总线
        """
        # 如果已有挂单，先撤单
        if self.account.pending_order_id is not None:
            cancel_event = Event(
                EventType.ORDER_CANCELLED,
                time.time(),
                {"order_id": self.account.pending_order_id, "agent_id": self.agent_id},
            )
            event_bus.publish(cancel_event)

        # 再挂新单
        timestamp = time.time()
        price = params["price"]
        quantity = params["quantity"]
        order_id = self._generate_order_id()
        side = OrderSide.BUY if action == ActionType.PLACE_BID else OrderSide.SELL
        order = Order(
            order_id=order_id,
            agent_id=self.agent_id,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=quantity,
        )
        place_event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
        event_bus.publish(place_event)

    def execute_action(self, action: ActionType, params: dict[str, Any], event_bus: EventBus) -> None:
        """执行动作

        使用字典分发模式，将动作分发到对应的处理函数。
        散户的 PLACE_BID 和 PLACE_ASK 使用特定实现（先撤旧单再挂新单），
        其他动作继承自父类的分发表。

        Args:
            action: 动作类型
            params: 动作参数字典
                - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
                - MARKET_BUY/MARKET_SELL: {"quantity": float}
                - CANCEL: {"order_id": int} (可选，默认使用账户的 pending_order_id)
                - HOLD: {}
            event_bus: 事件总线
        """
        if self.is_liquidated:
            return
        handler = self._action_handlers.get(action)
        if handler:
            handler(params, event_bus)
