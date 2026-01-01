"""庄家 Agent 模块

本模块定义庄家 Agent 类，继承自 Agent 基类。
"""

from typing import Any, Callable

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
        _action_handlers: 动作处理函数字典分发表
    """

    agent_id: int
    brain: Brain
    _action_handlers: dict[ActionType, Callable[[dict[str, Any], EventBus], None]]

    def __init__(
        self, agent_id: int, brain: Brain, config: AgentConfig, event_bus: EventBus
    ) -> None:
        """创建庄家 Agent

        调用父类构造函数，设置类型为 WHALE，并初始化动作处理器。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
            event_bus: 事件总线
        """
        super().__init__(agent_id, AgentType.WHALE, brain, config, event_bus)
        self._init_action_handlers()

    def _init_action_handlers(self) -> None:
        """初始化动作处理器字典

        将动作类型映射到对应的处理函数。
        """
        self._action_handlers = {
            ActionType.PLACE_BID: lambda params, event_bus: self._handle_place_order(
                ActionType.PLACE_BID, params, event_bus
            ),
            ActionType.PLACE_ASK: lambda params, event_bus: self._handle_place_order(
                ActionType.PLACE_ASK, params, event_bus
            ),
            ActionType.MARKET_BUY: lambda params, event_bus: self._handle_market_order(
                ActionType.MARKET_BUY, params, event_bus
            ),
            ActionType.MARKET_SELL: lambda params, event_bus: self._handle_market_order(
                ActionType.MARKET_SELL, params, event_bus
            ),
        }

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

    def _cancel_pending_order(self, event_bus: EventBus) -> None:
        """撤销当前挂单

        如果存在挂单，发布撤单事件。

        Args:
            event_bus: 事件总线
        """
        if self.account.pending_order_id is not None:
            cancel_event = Event(
                EventType.ORDER_CANCELLED,
                0.0,
                {"order_id": self.account.pending_order_id, "agent_id": self.agent_id},
            )
            event_bus.publish(cancel_event)

    def _handle_place_order(
        self, action: ActionType, params: dict[str, Any], event_bus: EventBus
    ) -> None:
        """处理限价单

        先撤销旧挂单，然后挂新限价单。

        Args:
            action: 动作类型（PLACE_BID 或 PLACE_ASK）
            params: 动作参数字典，包含 price 和 quantity
            event_bus: 事件总线
        """
        self._cancel_pending_order(event_bus)
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
        place_event = Event(EventType.ORDER_PLACED, 0.0, {"order": order})
        event_bus.publish(place_event)

    def _handle_market_order(
        self, action: ActionType, params: dict[str, Any], event_bus: EventBus
    ) -> None:
        """处理市价单

        先撤销旧挂单，然后下市价单。

        Args:
            action: 动作类型（MARKET_BUY 或 MARKET_SELL）
            params: 动作参数字典，包含 quantity
            event_bus: 事件总线
        """
        self._cancel_pending_order(event_bus)
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
        place_event = Event(EventType.ORDER_PLACED, 0.0, {"order": order})
        event_bus.publish(place_event)

    def execute_action(
        self, action: ActionType, params: dict[str, Any], event_bus: EventBus
    ) -> None:
        """执行动作

        庄家同时只能挂一单。下单时（挂单或市价单）如果已有挂单，先撤旧单再执行新动作。
        使用字典分发模式将动作类型映射到对应的处理函数。

        Args:
            action: 动作类型
            params: 动作参数字典
                - PLACE_BID/PLACE_ASK: {"price": float, "quantity": float}
                - MARKET_BUY/MARKET_SELL: {"quantity": float}
            event_bus: 事件总线
        """
        if self.is_liquidated:
            return
        handler = self._action_handlers.get(action)
        if handler:
            handler(params, event_bus)
