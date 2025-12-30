"""庄家 Agent 模块

本模块定义庄家 Agent 类，继承自 Agent 基类。
"""

import time
from typing import Any

from src.bio.agents.base import ActionType, Agent
from src.bio.brain.brain import Brain
from src.config.config import AgentConfig, AgentType
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event, EventType
from src.market.orderbook.order import Order, OrderSide, OrderType


class WhaleAgent(Agent):
    """庄家 Agent

    代表市场中拥有大量资金的参与者，初始资产 1000万，杠杆 10 倍。

    庄家"绝不不动"，必须持续参与市场，同时只能挂一单。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
    """

    agent_id: int
    brain: Brain

    def __init__(
        self, agent_id: int, brain: Brain, config: AgentConfig, event_bus: EventBus
    ) -> None:
        """创建庄家 Agent

        调用父类构造函数，设置类型为 WHALE。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
            event_bus: 事件总线
        """
        super().__init__(agent_id, AgentType.WHALE, brain, config, event_bus)

    def get_action_space(self) -> list[ActionType]:
        """获取庄家可用动作空间

        庄家"绝不不动"，不能选择 HOLD 动作，也不能单纯撤单。
        可选动作：
        - PLACE_BID: 挂买单
        - PLACE_ASK: 挂卖单
        - MARKET_BUY: 市价买入
        - MARKET_SELL: 市价卖出

        Returns:
            庄家可用的动作类型列表（不包含 HOLD、CANCEL 和 CLEAR_POSITION）
        """
        return [
            ActionType.PLACE_BID,
            ActionType.PLACE_ASK,
            ActionType.MARKET_BUY,
            ActionType.MARKET_SELL,
        ]

    def execute_action(
        self, action: ActionType, params: dict[str, Any], event_bus: EventBus
    ) -> None:
        """执行动作

        庄家同时只能挂一单。下单时（挂单或市价单）如果已有挂单，先撤旧单再执行新动作。

        Args:
            action: 动作类型
            params: 动作参数字典
                - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
                - MARKET_BUY/MARKET_SELL: {"quantity": float}
            event_bus: 事件总线
        """
        # 如果已被强平，不执行任何动作
        if self.is_liquidated:
            return

        # 如果是下单动作，先检查并撤掉旧挂单
        if self.account.pending_order_id is not None:
            cancel_event = Event(
                EventType.ORDER_CANCELLED,
                time.time(),
                {"order_id": self.account.pending_order_id, "agent_id": self.agent_id},
            )
            event_bus.publish(cancel_event)

        # 处理挂单操作
        if action == ActionType.PLACE_BID or action == ActionType.PLACE_ASK:
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

        # 处理市价单操作
        elif action == ActionType.MARKET_BUY or action == ActionType.MARKET_SELL:
            timestamp = time.time()
            quantity = params["quantity"]
            order_id = self._generate_order_id()
            side = OrderSide.BUY if action == ActionType.MARKET_BUY else OrderSide.SELL
            order = Order(
                order_id=order_id,
                agent_id=self.agent_id,
                side=side,
                order_type=OrderType.MARKET,
                price=0.0,  # 市价单价格无意义
                quantity=quantity,
            )
            place_event = Event(EventType.ORDER_PLACED, timestamp, {"order": order})
            event_bus.publish(place_event)
