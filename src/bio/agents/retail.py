"""散户 Agent 模块

本模块定义散户 Agent 类，继承自 Agent 基类。
"""

import time
from typing import Any

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
    """

    agent_id: int
    brain: Brain

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig, event_bus: EventBus) -> None:
        """创建散户 Agent

        调用父类构造函数，设置类型为 RETAIL。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
            event_bus: 事件总线
        """
        super().__init__(agent_id, AgentType.RETAIL, brain, config, event_bus)

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

    def execute_action(self, action: ActionType, params: dict[str, Any], event_bus: EventBus) -> None:
        """执行动作

        散户同时只能挂一单。如果是挂单操作且已有挂单，先撤旧单再挂新单。

        Args:
            action: 动作类型
            params: 动作参数字典
                - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
                - MARKET_BUY/MARKET_SELL: {"quantity": float}
                - CANCEL: {"order_id": int} (可选，默认使用账户的 pending_order_id)
                - HOLD: {}
            event_bus: 事件总线
        """
        # 如果已被强平，不执行任何动作
        if self.is_liquidated:
            return

        # 处理挂单操作：先撤旧单（如有），再挂新单
        if action == ActionType.PLACE_BID or action == ActionType.PLACE_ASK:
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

        else:
            # 其他操作直接调用父类执行
            super().execute_action(action, params, event_bus)
