"""散户 Agent 模块

本模块定义散户 Agent 类，继承自 Agent 基类。
"""

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from src.bio.agents.base import ActionType
from src.bio.agents.base import Agent
from src.config.config import AgentConfig, AgentType
from src.bio.brain.brain import Brain
from src.core.event_engine.event_bus import EventBus
from src.core.event_engine.events import Event, EventType
from src.market.market_state import NormalizedMarketState
from src.market.matching.trade import Trade
from src.market.orderbook.order import Order, OrderSide, OrderType
from src.market.orderbook.orderbook import OrderBook

if TYPE_CHECKING:
    from src.market.matching.matching_engine import MatchingEngine


class RetailAgent(Agent):
    """散户 Agent

    代表市场中数量最多的交易参与者，初始资产较少，杠杆倍数最高。
    散户只能看到买卖各10档订单簿和最近10笔成交。

    Attributes:
        agent_id: Agent ID
        brain: NEAT 神经网络
        account: 交易账户
        _action_handlers: 动作处理器字典，将动作类型映射到处理函数
    """

    # 散户可见的订单簿档位数和成交笔数
    ORDERBOOK_DEPTH: int = 10
    TRADE_HISTORY_SIZE: int = 10
    # 输入缓冲区大小: 10档买盘(20) + 10档卖盘(20) + 10笔成交价格(10) + 10笔成交数量(10) + 持仓(4) + 挂单(3) = 67
    INPUT_SIZE: int = 67

    agent_id: int
    brain: Brain
    _action_handlers: dict[ActionType, Callable[[dict[str, Any], EventBus], None]]

    def __init__(self, agent_id: int, brain: Brain, config: AgentConfig, event_bus: EventBus) -> None:
        """创建散户 Agent

        调用父类构造函数，设置类型为 RETAIL，并初始化动作处理器。
        重写输入缓冲区为更小的尺寸（67），因为散户只能看到有限的市场数据。

        Args:
            agent_id: Agent ID
            brain: NEAT 神经网络
            config: Agent 配置
            event_bus: 事件总线
        """
        super().__init__(agent_id, AgentType.RETAIL, brain, config, event_bus)
        # 覆盖父类的输入缓冲区，使用更小的尺寸
        self._input_buffer = np.zeros(self.INPUT_SIZE, dtype=np.float64)
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

    def observe(self, market_state: NormalizedMarketState, orderbook: OrderBook) -> np.ndarray:
        """从预计算的市场状态构建神经网络输入（散户限制版）

        散户只能看到买卖各10档订单簿和最近10笔成交。

        Args:
            market_state: 预计算的归一化市场数据
            orderbook: 订单簿（用于查询挂单信息）

        Returns:
            神经网络输入向量（67维 ndarray）
        """
        depth = self.ORDERBOOK_DEPTH
        trade_size = self.TRADE_HISTORY_SIZE

        # 买盘前10档: 每档2个值（价格归一化 + 数量），取前20个值
        self._input_buffer[:depth * 2] = market_state.bid_data[:depth * 2]

        # 卖盘前10档: 每档2个值，取前20个值
        offset = depth * 2  # 20
        self._input_buffer[offset:offset + depth * 2] = market_state.ask_data[:depth * 2]

        # 最近10笔成交价格
        offset += depth * 2  # 40
        self._input_buffer[offset:offset + trade_size] = market_state.trade_prices[:trade_size]

        # 最近10笔成交数量
        offset += trade_size  # 50
        self._input_buffer[offset:offset + trade_size] = market_state.trade_quantities[:trade_size]

        # 持仓信息（4个值）
        offset += trade_size  # 60
        self._input_buffer[offset:offset + 4] = self._get_position_inputs(market_state.mid_price)

        # 挂单信息（3个值）
        offset += 4  # 64
        self._input_buffer[offset:offset + 3] = self._get_pending_order_inputs(market_state.mid_price, orderbook)

        return self._input_buffer  # 不调用 .tolist()

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
                0.0,
                {"order_id": self.account.pending_order_id, "agent_id": self.agent_id},
            )
            event_bus.publish(cancel_event)

        # 再挂新单
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

    def execute_action_direct(
        self,
        action: ActionType,
        params: dict[str, Any],
        matching_engine: "MatchingEngine",
    ) -> list[Trade]:
        """直接执行动作（训练模式，绕过事件系统）

        散户特定实现：PLACE_BID/PLACE_ASK 会先撤旧单再挂新单。

        Args:
            action: 动作类型
            params: 动作参数字典
            matching_engine: 撮合引擎

        Returns:
            成交列表
        """
        if self.is_liquidated:
            return []

        trades: list[Trade] = []

        if action == ActionType.PLACE_BID or action == ActionType.PLACE_ASK:
            # 散户特定：先撤旧单再挂新单
            if self.account.pending_order_id is not None:
                matching_engine.cancel_order_direct(self.account.pending_order_id)
                self.account.pending_order_id = None  # 清除旧挂单ID
            side = OrderSide.BUY if action == ActionType.PLACE_BID else OrderSide.SELL
            trades = self._place_limit_order_direct(
                side, params["price"], params["quantity"], matching_engine
            )
        else:
            # 其他动作使用父类实现
            trades = super().execute_action_direct(action, params, matching_engine)

        return trades
